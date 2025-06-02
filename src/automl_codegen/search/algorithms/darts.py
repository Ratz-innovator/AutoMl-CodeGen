"""
DARTS: Differentiable Architecture Search

Implementation of the DARTS algorithm for neural architecture search using
gradient-based optimization instead of evolutionary approaches.

Paper: "DARTS: Differentiable Architecture Search" (Liu et al., 2019)
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple
import copy
import numpy as np

from ..space.search_space import SearchSpace
from ..objectives.multi_objective import MultiObjectiveOptimizer

logger = logging.getLogger(__name__)


class MixedOp(nn.Module):
    """Mixed operation for DARTS search space."""
    
    def __init__(self, primitive_ops: List[str], channels: int, stride: int = 1):
        super().__init__()
        self.primitive_ops = primitive_ops
        self.ops = nn.ModuleList()
        
        for op_name in primitive_ops:
            op = self._get_operation(op_name, channels, stride)
            self.ops.append(op)
    
    def _get_operation(self, op_name: str, channels: int, stride: int) -> nn.Module:
        """Create operation based on name."""
        if op_name == 'conv3x3':
            return nn.Sequential(
                nn.Conv2d(channels, channels, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        elif op_name == 'conv1x1':
            return nn.Sequential(
                nn.Conv2d(channels, channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        elif op_name == 'avgpool':
            return nn.AvgPool2d(3, stride=stride, padding=1)
        elif op_name == 'maxpool':
            return nn.MaxPool2d(3, stride=stride, padding=1)
        elif op_name == 'skip':
            if stride == 1:
                return nn.Identity()
            else:
                return nn.Sequential(
                    nn.Conv2d(channels, channels, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(channels)
                )
        elif op_name == 'zero':
            return ZeroOp(stride)
        else:
            # Default to conv3x3
            return nn.Sequential(
                nn.Conv2d(channels, channels, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Forward pass with architecture weights."""
        outputs = []
        for i, op in enumerate(self.ops):
            outputs.append(weights[i] * op(x))
        return sum(outputs)


class ZeroOp(nn.Module):
    """Zero operation (no connection)."""
    
    def __init__(self, stride: int):
        super().__init__()
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            return x.mul(0.)
        else:
            return x[:, :, ::self.stride, ::self.stride].mul(0.)


class DARTSCell(nn.Module):
    """DARTS search cell."""
    
    def __init__(self, primitive_ops: List[str], num_nodes: int = 4, channels: int = 64):
        super().__init__()
        self.num_nodes = num_nodes
        self.primitive_ops = primitive_ops
        
        # Mixed operations for each edge
        self.mixed_ops = nn.ModuleList()
        self.edges = []
        
        # Create edges: each node connects to all previous nodes
        for i in range(num_nodes):
            for j in range(i + 1):  # Connect to input and all previous nodes
                edge = (j, i + 1)  # (from, to)
                self.edges.append(edge)
                mixed_op = MixedOp(primitive_ops, channels)
                self.mixed_ops.append(mixed_op)
    
    def forward(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Forward pass through cell."""
        # Initialize node states
        states = [x, x]  # Two input nodes
        
        edge_idx = 0
        for i in range(self.num_nodes):
            # Collect inputs from all previous nodes
            node_inputs = []
            for j in range(i + 2):  # +2 for the two input nodes
                if edge_idx < len(self.mixed_ops):
                    edge_weights = F.softmax(alpha[edge_idx], dim=0)
                    output = self.mixed_ops[edge_idx](states[j], edge_weights)
                    node_inputs.append(output)
                    edge_idx += 1
            
            # Combine inputs for this node
            if node_inputs:
                states.append(sum(node_inputs))
            else:
                states.append(states[-1])  # Fallback
        
        # Return concatenation of all intermediate nodes
        return torch.cat(states[2:], dim=1)  # Skip the two input nodes


class DARTSSearch:
    """DARTS differentiable architecture search algorithm."""
    
    def __init__(
        self,
        search_space: SearchSpace,
        objectives: MultiObjectiveOptimizer,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        self.search_space = search_space
        self.objectives = objectives
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        
        # DARTS hyperparameters
        self.primitive_ops = [
            'conv3x3', 'conv1x1', 'avgpool', 'maxpool', 'skip', 'zero'
        ]
        self.num_cells = self.config.get('num_cells', 8)
        self.num_nodes = self.config.get('num_nodes', 4)
        self.channels = self.config.get('channels', 64)
        
        # Learning rates
        self.arch_lr = self.config.get('arch_lr', 3e-4)
        self.weight_lr = self.config.get('weight_lr', 1e-3)
        
        # Architecture parameters (alpha)
        self.num_edges = sum(range(2, self.num_nodes + 2))  # Number of edges per cell
        self.num_ops = len(self.primitive_ops)
        
        # Initialize architecture parameters
        self.alpha = nn.Parameter(
            torch.randn(self.num_edges, self.num_ops) * 1e-3
        )
        
        # Create supernet
        self.supernet = self._create_supernet()
        
        # Optimizers
        self.arch_optimizer = optim.Adam([self.alpha], lr=self.arch_lr)
        self.weight_optimizer = optim.SGD(
            self.supernet.parameters(), 
            lr=self.weight_lr, 
            momentum=0.9, 
            weight_decay=3e-4
        )
        
        self.logger.info(f"Initialized DARTS with {self.num_cells} cells, {self.num_nodes} nodes")
    
    def _create_supernet(self) -> nn.Module:
        """Create the differentiable supernet."""
        class DARTSSupernet(nn.Module):
            def __init__(self, primitive_ops, num_cells, num_nodes, channels, num_classes=10):
                super().__init__()
                self.num_cells = num_cells
                
                # Stem convolution
                self.stem = nn.Sequential(
                    nn.Conv2d(3, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels)
                )
                
                # DARTS cells
                self.cells = nn.ModuleList()
                for i in range(num_cells):
                    cell = DARTSCell(primitive_ops, num_nodes, channels)
                    self.cells.append(cell)
                
                # Classification head
                self.global_pooling = nn.AdaptiveAvgPool2d(1)
                self.classifier = nn.Linear(channels * num_nodes, num_classes)
            
            def forward(self, x, alpha):
                x = self.stem(x)
                
                for cell in self.cells:
                    x = cell(x, alpha)
                
                # Global average pooling
                x = self.global_pooling(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                
                return x
        
        return DARTSSupernet(
            self.primitive_ops, 
            self.num_cells, 
            self.num_nodes, 
            self.channels
        )
    
    def search(
        self,
        train_loader: Any,
        val_loader: Any,
        num_epochs: int = 50,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run DARTS search process.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader  
            num_epochs: Number of search epochs
            
        Returns:
            Best architecture found
        """
        self.logger.info(f"Starting DARTS search for {num_epochs} epochs")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.supernet.to(device)
        self.alpha.to(device)
        
        best_arch = None
        best_score = float('-inf')
        
        for epoch in range(num_epochs):
            # Train weights
            train_loss = self._train_weights(train_loader, device)
            
            # Train architecture
            arch_loss = self._train_architecture(val_loader, device)
            
            # Evaluate current architecture
            val_acc = self._evaluate_architecture(val_loader, device)
            
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"Train Loss: {train_loss:.4f}, "
                f"Arch Loss: {arch_loss:.4f}, "
                f"Val Acc: {val_acc:.4f}"
            )
            
            # Update best architecture
            if val_acc > best_score:
                best_score = val_acc
                best_arch = self._derive_architecture()
                self.logger.info(f"New best architecture found with accuracy: {best_score:.4f}")
        
        # Derive final architecture
        final_arch = self._derive_architecture()
        
        return {
            'architecture': final_arch,
            'score': best_score,
            'search_history': []  # Could store intermediate results
        }
    
    def _train_weights(self, train_loader: Any, device: torch.device) -> float:
        """Train network weights with fixed architecture."""
        self.supernet.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= 50:  # Limit batches for efficiency
                break
                
            data, target = data.to(device), target.to(device)
            
            self.weight_optimizer.zero_grad()
            
            # Forward pass
            output = self.supernet(data, self.alpha)
            loss = F.cross_entropy(output, target)
            
            # Backward pass
            loss.backward()
            self.weight_optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _train_architecture(self, val_loader: Any, device: torch.device) -> float:
        """Train architecture parameters."""
        self.supernet.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(val_loader):
            if batch_idx >= 25:  # Limit batches for efficiency
                break
                
            data, target = data.to(device), target.to(device)
            
            self.arch_optimizer.zero_grad()
            
            # Forward pass
            output = self.supernet(data, self.alpha)
            loss = F.cross_entropy(output, target)
            
            # Backward pass
            loss.backward()
            self.arch_optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _evaluate_architecture(self, val_loader: Any, device: torch.device) -> float:
        """Evaluate current architecture."""
        self.supernet.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                if batch_idx >= 20:  # Limit batches for efficiency
                    break
                    
                data, target = data.to(device), target.to(device)
                output = self.supernet(data, self.alpha)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        return correct / max(total, 1)
    
    def _derive_architecture(self) -> Dict[str, Any]:
        """Derive discrete architecture from continuous parameters."""
        # Get the most likely operations for each edge
        alpha_softmax = F.softmax(self.alpha, dim=1)
        selected_ops = alpha_softmax.argmax(dim=1)
        
        # Convert to architecture specification
        layers = []
        
        for i, op_idx in enumerate(selected_ops):
            op_name = self.primitive_ops[op_idx]
            
            if op_name == 'conv3x3':
                layer = {
                    'type': 'conv2d',
                    'out_channels': self.channels,
                    'kernel_size': 3,
                    'stride': 1,
                    'padding': 1
                }
            elif op_name == 'conv1x1':
                layer = {
                    'type': 'conv2d', 
                    'out_channels': self.channels,
                    'kernel_size': 1,
                    'stride': 1,
                    'padding': 0
                }
            elif op_name == 'avgpool':
                layer = {'type': 'adaptive_pool', 'output_size': (1, 1)}
            elif op_name == 'maxpool':
                layer = {'type': 'maxpool', 'kernel_size': 3, 'stride': 1, 'padding': 1}
            elif op_name == 'skip':
                continue  # Skip connections handled separately
            elif op_name == 'zero':
                continue  # Zero operations are not included
            else:
                continue
            
            layers.append(layer)
            
            # Add activation after convolution
            if layer['type'] == 'conv2d':
                layers.append({'type': 'relu'})
        
        # Add final classification layer
        layers.extend([
            {'type': 'adaptive_pool', 'output_size': (1, 1)},
            {'type': 'flatten'},
            {'type': 'linear', 'out_features': 10}
        ])
        
        architecture = {
            'task': 'image_classification',
            'layers': layers,
            'connections': 'sequential',
            'input_shape': (3, 32, 32),
            'algorithm': 'darts',
            'alpha_weights': alpha_softmax.detach().cpu().numpy().tolist()
        }
        
        return architecture 