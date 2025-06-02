"""
Neural Network Operations for Architecture Search
===============================================

This module provides the basic building blocks for neural architectures
including convolutions, pooling, skip connections, and mixed operations
used in differentiable architecture search.

Key Features:
- Standard operations (conv, pool, skip, zero)
- Mixed operations for DARTS
- Operation factory functions
- FLOPs and parameter counting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
import math

from ..core.base import BaseOperation
from ..core.architecture import OperationSpec


class ConvOperation(BaseOperation):
    """Convolution operation with batch normalization and activation."""
    
    def __init__(self, 
                 channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: Optional[int] = None,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = False,
                 use_bn: bool = True,
                 activation: str = "relu"):
        """
        Initialize convolution operation.
        
        Args:
            channels: Number of input/output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Padding size (auto-computed if None)
            dilation: Dilation rate
            groups: Number of groups for grouped convolution
            bias: Whether to use bias
            use_bn: Whether to use batch normalization
            activation: Activation function name
        """
        super().__init__(channels)
        
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        
        self.conv = nn.Conv2d(
            channels, channels, kernel_size, stride, padding, dilation, groups, bias
        )
        
        self.bn = nn.BatchNorm2d(channels) if use_bn else nn.Identity()
        
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "swish":
            self.activation = nn.SiLU(inplace=True)
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolution operation."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
    
    def compute_flops(self, input_shape: tuple) -> int:
        """Compute FLOPs for convolution."""
        _, C, H, W = input_shape
        kernel_flops = self.conv.kernel_size[0] * self.conv.kernel_size[1] * C
        output_elements = H * W * self.conv.out_channels
        return kernel_flops * output_elements


class DepthwiseConvOperation(BaseOperation):
    """Depthwise separable convolution operation."""
    
    def __init__(self, 
                 channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: Optional[int] = None):
        """Initialize depthwise separable convolution."""
        super().__init__(channels)
        
        if padding is None:
            padding = (kernel_size - 1) // 2
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size, stride, padding, groups=channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through depthwise separable convolution."""
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.activation(x)
        
        return x


class PoolOperation(BaseOperation):
    """Pooling operation (max or average)."""
    
    def __init__(self, 
                 channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 pool_type: str = "max"):
        """
        Initialize pooling operation.
        
        Args:
            channels: Number of channels (unchanged by pooling)
            kernel_size: Pooling kernel size
            stride: Pooling stride
            padding: Padding size
            pool_type: Type of pooling ('max' or 'avg')
        """
        super().__init__(channels)
        
        if pool_type == "max":
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type == "avg":
            self.pool = nn.AvgPool2d(kernel_size, stride, padding)
        else:
            raise ValueError(f"Unknown pool type: {pool_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through pooling operation."""
        return self.pool(x)
    
    def compute_flops(self, input_shape: tuple) -> int:
        """Compute FLOPs for pooling (relatively small)."""
        _, C, H, W = input_shape
        return H * W * C  # Simplified estimate


class SkipConnection(BaseOperation):
    """Identity/skip connection operation."""
    
    def __init__(self, channels: int):
        """Initialize skip connection."""
        super().__init__(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through skip connection."""
        return x
    
    def compute_flops(self, input_shape: tuple) -> int:
        """Skip connection has no computational cost."""
        return 0


class ZeroOperation(BaseOperation):
    """Zero operation (outputs zeros)."""
    
    def __init__(self, channels: int):
        """Initialize zero operation."""
        super().__init__(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through zero operation."""
        return torch.zeros_like(x)
    
    def compute_flops(self, input_shape: tuple) -> int:
        """Zero operation has no computational cost."""
        return 0


class MixedOperation(nn.Module):
    """
    Mixed operation for DARTS that combines multiple operations
    with learnable weights (softmax over operation choices).
    """
    
    def __init__(self, operations: List[OperationSpec], channels: int):
        """
        Initialize mixed operation.
        
        Args:
            operations: List of operation specifications
            channels: Number of channels
        """
        super().__init__()
        
        self.operations = nn.ModuleList()
        for op_spec in operations:
            operation = create_operation(op_spec, channels)
            self.operations.append(operation)
    
    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through mixed operation.
        
        Args:
            x: Input tensor
            weights: Operation weights (softmax probabilities)
            
        Returns:
            Weighted combination of operation outputs
        """
        output = sum(w * op(x) for w, op in zip(weights, self.operations))
        return output


class SeparableConv(BaseOperation):
    """Separable convolution (factorized convolution)."""
    
    def __init__(self, 
                 channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: Optional[int] = None,
                 dilation: int = 1):
        """Initialize separable convolution."""
        super().__init__(channels)
        
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        
        # First convolution
        self.conv1 = nn.Conv2d(
            channels, channels, (1, kernel_size), (1, stride), (0, padding), 
            (1, dilation), bias=False
        )
        self.conv2 = nn.Conv2d(
            channels, channels, (kernel_size, 1), (stride, 1), (padding, 0), 
            (dilation, 1), bias=False
        )
        
        self.bn = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through separable convolution."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class DilatedConv(BaseOperation):
    """Dilated convolution operation."""
    
    def __init__(self, 
                 channels: int,
                 kernel_size: int = 3,
                 dilation: int = 2):
        """Initialize dilated convolution."""
        super().__init__(channels)
        
        padding = (kernel_size - 1) // 2 * dilation
        
        self.conv = nn.Conv2d(
            channels, channels, kernel_size, 1, padding, dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through dilated convolution."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


def create_operation(op_spec: OperationSpec, channels: int) -> BaseOperation:
    """
    Factory function to create operations from specifications.
    
    Args:
        op_spec: Operation specification
        channels: Number of channels
        
    Returns:
        Created operation instance
    """
    op_name = op_spec.name.lower()
    op_type = op_spec.type
    params = op_spec.params
    
    if op_type == "conv":
        kernel_size = params.get("kernel_size", 3)
        stride = params.get("stride", 1)
        
        if "groups" in params and params["groups"] == "input_channels":
            # Depthwise convolution
            return DepthwiseConvOperation(channels, kernel_size, stride)
        else:
            return ConvOperation(channels, kernel_size, stride, **params)
    
    elif op_type == "pool":
        pool_type = params.get("type", "max")
        kernel_size = params.get("kernel_size", 3)
        stride = params.get("stride", 1)
        padding = params.get("padding", 1)
        
        return PoolOperation(channels, kernel_size, stride, padding, pool_type)
    
    elif op_type == "skip":
        return SkipConnection(channels)
    
    elif op_type == "zero":
        return ZeroOperation(channels)
    
    elif op_name == "sep_conv":
        kernel_size = params.get("kernel_size", 3)
        return SeparableConv(channels, kernel_size)
    
    elif op_name == "dil_conv":
        kernel_size = params.get("kernel_size", 3)
        dilation = params.get("dilation", 2)
        return DilatedConv(channels, kernel_size, dilation)
    
    else:
        # Default to regular convolution
        return ConvOperation(channels, 3, 1)


# Operation registry for easy lookup
OPERATION_REGISTRY = {
    "conv3x3": lambda c: ConvOperation(c, 3),
    "conv5x5": lambda c: ConvOperation(c, 5),
    "conv1x1": lambda c: ConvOperation(c, 1),
    "dw_conv3x3": lambda c: DepthwiseConvOperation(c, 3),
    "dw_conv5x5": lambda c: DepthwiseConvOperation(c, 5),
    "sep_conv3x3": lambda c: SeparableConv(c, 3),
    "sep_conv5x5": lambda c: SeparableConv(c, 5),
    "dil_conv3x3": lambda c: DilatedConv(c, 3),
    "dil_conv5x5": lambda c: DilatedConv(c, 5),
    "maxpool3x3": lambda c: PoolOperation(c, 3, 1, 1, "max"),
    "avgpool3x3": lambda c: PoolOperation(c, 3, 1, 1, "avg"),
    "skip": lambda c: SkipConnection(c),
    "zero": lambda c: ZeroOperation(c),
}


def get_operation_by_name(name: str, channels: int) -> BaseOperation:
    """
    Get operation by name from registry.
    
    Args:
        name: Operation name
        channels: Number of channels
        
    Returns:
        Operation instance
    """
    if name in OPERATION_REGISTRY:
        return OPERATION_REGISTRY[name](channels)
    else:
        raise ValueError(f"Unknown operation: {name}") 