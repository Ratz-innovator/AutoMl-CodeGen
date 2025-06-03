"""
Neural Network Operations for Architecture Search
===============================================

This module provides comprehensive building blocks for neural architectures
including modern operations like attention mechanisms, advanced normalization,
and state-of-the-art activations for neural architecture search.

Key Features:
- Standard operations (conv, pool, skip, zero)
- Attention mechanisms (self-attention, channel attention, spatial attention)
- Advanced normalization (LayerNorm, GroupNorm, RMSNorm)
- Modern activations (Swish, GELU, Mish, PReLU)
- Mixed operations for DARTS
- Graph neural network operations
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


# ===========================
# MODERN OPERATIONS
# ===========================

class SelfAttentionOperation(BaseOperation):
    """Self-attention operation for capturing long-range dependencies."""
    
    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.1):
        """Initialize self-attention operation."""
        super().__init__(channels)
        
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through self-attention."""
        B, C, H, W = x.shape
        
        # Reshape to sequence format
        x_flat = x.view(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
        
        # Compute Q, K, V
        q = self.query(x_flat).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x_flat).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x_flat).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention computation
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, H * W, C)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        # Reshape back to spatial format
        output = output.transpose(1, 2).view(B, C, H, W)
        
        # Residual connection
        return x + output
    
    def compute_flops(self, input_shape: tuple) -> int:
        """Compute FLOPs for self-attention."""
        _, C, H, W = input_shape
        seq_len = H * W
        
        # Q, K, V projections
        qkv_flops = 3 * seq_len * C * C
        
        # Attention computation
        attn_flops = seq_len * seq_len * C
        
        # Output projection
        out_flops = seq_len * C * C
        
        return qkv_flops + attn_flops + out_flops


class ChannelAttentionOperation(BaseOperation):
    """Channel attention operation (Squeeze-and-Excitation style)."""
    
    def __init__(self, channels: int, reduction: int = 16):
        """Initialize channel attention operation."""
        super().__init__(channels)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through channel attention."""
        B, C, H, W = x.shape
        
        # Average pooling branch
        avg_out = self.avg_pool(x).view(B, C)
        avg_out = self.fc(avg_out)
        
        # Max pooling branch
        max_out = self.max_pool(x).view(B, C)
        max_out = self.fc(max_out)
        
        # Combine and apply sigmoid
        attention = self.sigmoid(avg_out + max_out).view(B, C, 1, 1)
        
        return x * attention
    
    def compute_flops(self, input_shape: tuple) -> int:
        """Compute FLOPs for channel attention."""
        _, C, H, W = input_shape
        reduction = 16
        
        # FC layers
        fc_flops = C * (C // reduction) + (C // reduction) * C
        
        return fc_flops


class SpatialAttentionOperation(BaseOperation):
    """Spatial attention operation."""
    
    def __init__(self, channels: int, kernel_size: int = 7):
        """Initialize spatial attention operation."""
        super().__init__(channels)
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through spatial attention."""
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and apply convolution
        attention_input = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention_input))
        
        return x * attention
    
    def compute_flops(self, input_shape: tuple) -> int:
        """Compute FLOPs for spatial attention."""
        _, C, H, W = input_shape
        kernel_size = 7
        
        # Convolution FLOPs
        conv_flops = kernel_size * kernel_size * 2 * H * W
        
        return conv_flops


class LayerNormOperation(BaseOperation):
    """Layer normalization operation."""
    
    def __init__(self, channels: int, eps: float = 1e-6):
        """Initialize layer normalization."""
        super().__init__(channels)
        
        self.norm = nn.LayerNorm(channels, eps=eps)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through layer normalization."""
        B, C, H, W = x.shape
        
        # Reshape for layer norm
        x_flat = x.view(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
        x_norm = self.norm(x_flat)
        
        # Reshape back
        return x_norm.transpose(1, 2).view(B, C, H, W)
    
    def compute_flops(self, input_shape: tuple) -> int:
        """Compute FLOPs for layer normalization."""
        _, C, H, W = input_shape
        return H * W * C  # Simplified estimate


class GroupNormOperation(BaseOperation):
    """Group normalization operation."""
    
    def __init__(self, channels: int, num_groups: int = 32, eps: float = 1e-6):
        """Initialize group normalization."""
        super().__init__(channels)
        
        # Adjust num_groups if channels is smaller
        num_groups = min(num_groups, channels)
        if channels % num_groups != 0:
            num_groups = channels  # Fallback to layer norm behavior
        
        self.norm = nn.GroupNorm(num_groups, channels, eps=eps)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through group normalization."""
        return self.norm(x)
    
    def compute_flops(self, input_shape: tuple) -> int:
        """Compute FLOPs for group normalization."""
        _, C, H, W = input_shape
        return H * W * C  # Simplified estimate


class RMSNormOperation(BaseOperation):
    """RMS normalization operation."""
    
    def __init__(self, channels: int, eps: float = 1e-6):
        """Initialize RMS normalization."""
        super().__init__(channels)
        
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(channels))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RMS normalization."""
        B, C, H, W = x.shape
        
        # Reshape for normalization
        x_flat = x.view(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
        
        # RMS normalization
        rms = torch.sqrt(torch.mean(x_flat ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x_flat / rms * self.scale
        
        # Reshape back
        return x_norm.transpose(1, 2).view(B, C, H, W)
    
    def compute_flops(self, input_shape: tuple) -> int:
        """Compute FLOPs for RMS normalization."""
        _, C, H, W = input_shape
        return H * W * C  # Simplified estimate


class SwishActivation(BaseOperation):
    """Swish activation operation."""
    
    def __init__(self, channels: int):
        """Initialize Swish activation."""
        super().__init__(channels)
        self.activation = nn.SiLU()  # SiLU is equivalent to Swish
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Swish activation."""
        return self.activation(x)
    
    def compute_flops(self, input_shape: tuple) -> int:
        """Compute FLOPs for Swish activation."""
        _, C, H, W = input_shape
        return H * W * C  # Simplified estimate


class GELUActivation(BaseOperation):
    """GELU activation operation."""
    
    def __init__(self, channels: int):
        """Initialize GELU activation."""
        super().__init__(channels)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through GELU activation."""
        return self.activation(x)
    
    def compute_flops(self, input_shape: tuple) -> int:
        """Compute FLOPs for GELU activation."""
        _, C, H, W = input_shape
        return H * W * C  # Simplified estimate


class MishActivation(BaseOperation):
    """Mish activation operation."""
    
    def __init__(self, channels: int):
        """Initialize Mish activation."""
        super().__init__(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Mish activation."""
        return x * torch.tanh(F.softplus(x))
    
    def compute_flops(self, input_shape: tuple) -> int:
        """Compute FLOPs for Mish activation."""
        _, C, H, W = input_shape
        return H * W * C * 3  # Approximate cost for tanh and softplus


class PReLUActivation(BaseOperation):
    """PReLU activation operation."""
    
    def __init__(self, channels: int):
        """Initialize PReLU activation."""
        super().__init__(channels)
        self.activation = nn.PReLU(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through PReLU activation."""
        return self.activation(x)
    
    def compute_flops(self, input_shape: tuple) -> int:
        """Compute FLOPs for PReLU activation."""
        _, C, H, W = input_shape
        return H * W * C  # Simplified estimate


class AdaptivePoolOperation(BaseOperation):
    """Adaptive pooling operation."""
    
    def __init__(self, channels: int, output_size: int = 1, pool_type: str = "avg"):
        """Initialize adaptive pooling."""
        super().__init__(channels)
        
        if pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool2d(output_size)
        elif pool_type == "max":
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        else:
            raise ValueError(f"Unknown pool type: {pool_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through adaptive pooling."""
        return self.pool(x)
    
    def compute_flops(self, input_shape: tuple) -> int:
        """Compute FLOPs for adaptive pooling."""
        _, C, H, W = input_shape
        return H * W * C  # Simplified estimate


# Graph Neural Network Operations
class GraphConvOperation(BaseOperation):
    """Graph convolution operation for GNN architectures."""
    
    def __init__(self, channels: int, conv_type: str = "gcn"):
        """Initialize graph convolution."""
        super().__init__(channels)
        
        self.conv_type = conv_type
        self.linear = nn.Linear(channels, channels)
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor, adj_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through graph convolution."""
        # For CNN architectures, treat as regular convolution
        # In actual GNN usage, this would use the adjacency matrix
        x_transformed = self.linear(x.view(x.size(0), x.size(1), -1).transpose(1, 2))
        x_transformed = x_transformed.transpose(1, 2).view_as(x)
        return self.activation(x_transformed)
    
    def compute_flops(self, input_shape: tuple) -> int:
        """Compute FLOPs for graph convolution."""
        _, C, H, W = input_shape
        return H * W * C * C  # Linear transformation


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
        
        if pool_type == "adaptive_avg":
            return AdaptivePoolOperation(channels, 1, "avg")
        elif pool_type == "adaptive_max":
            return AdaptivePoolOperation(channels, 1, "max")
        else:
            kernel_size = params.get("kernel_size", 3)
            stride = params.get("stride", 1)
            padding = params.get("padding", 1)
            return PoolOperation(channels, kernel_size, stride, padding, pool_type)
    
    elif op_type == "attention":
        attn_type = params.get("type", "self")
        
        if attn_type == "self":
            num_heads = params.get("heads", 8)
            return SelfAttentionOperation(channels, num_heads)
        elif attn_type == "channel":
            reduction = params.get("reduction", 16)
            return ChannelAttentionOperation(channels, reduction)
        elif attn_type == "spatial":
            kernel_size = params.get("kernel_size", 7)
            return SpatialAttentionOperation(channels, kernel_size)
        else:
            # Default to self-attention
            return SelfAttentionOperation(channels, 8)
    
    elif op_type == "advanced_norm":
        norm_type = params.get("type", "layer")
        
        if norm_type == "layer":
            return LayerNormOperation(channels)
        elif norm_type == "group":
            num_groups = params.get("num_groups", 32)
            return GroupNormOperation(channels, num_groups)
        elif norm_type == "rms":
            return RMSNormOperation(channels)
        else:
            # Default to layer norm
            return LayerNormOperation(channels)
    
    elif op_type == "norm":
        norm_type = params.get("type", "batch")
        
        if norm_type == "batch":
            # Return a simple batch norm wrapped in BaseOperation
            class BatchNormOperation(BaseOperation):
                def __init__(self, channels):
                    super().__init__(channels)
                    self.norm = nn.BatchNorm2d(channels)
                
                def forward(self, x):
                    return self.norm(x)
                
                def compute_flops(self, input_shape):
                    _, C, H, W = input_shape
                    return H * W * C
            
            return BatchNormOperation(channels)
        else:
            return LayerNormOperation(channels)
    
    elif op_type == "activation":
        act_type = params.get("type", "relu")
        
        if act_type == "swish":
            return SwishActivation(channels)
        elif act_type == "gelu":
            return GELUActivation(channels)
        elif act_type == "mish":
            return MishActivation(channels)
        elif act_type == "prelu":
            return PReLUActivation(channels)
        elif act_type == "relu":
            # Return a simple ReLU wrapped in BaseOperation
            class ReLUOperation(BaseOperation):
                def __init__(self, channels):
                    super().__init__(channels)
                    self.activation = nn.ReLU(inplace=True)
                
                def forward(self, x):
                    return self.activation(x)
                
                def compute_flops(self, input_shape):
                    return 0  # ReLU has minimal cost
            
            return ReLUOperation(channels)
        else:
            return SwishActivation(channels)  # Default to Swish
    
    elif op_type == "graph":
        conv_type = params.get("type", "gcn")
        return GraphConvOperation(channels, conv_type)
    
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