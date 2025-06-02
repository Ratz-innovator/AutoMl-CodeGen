"""
Architecture Trainer Module

This module provides training and evaluation capabilities for neural architectures.
"""

import logging
import time
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.optim as optim
from ..utils.config import Config
from ..datasets.loaders import get_dataloader, get_dataset_info

logger = logging.getLogger(__name__)

class ArchitectureTrainer:
    """
    Architecture trainer for neural architecture search.
    
    This class provides training and evaluation capabilities for neural architectures:
    - Dataset loading and preprocessing
    - Model creation from architecture specifications
    - Training loop with optimization
    - Validation and metric calculation
    - Hardware-aware evaluation
    """
    
    def __init__(self, task: str = None, dataset: str = None, config: Optional[Config] = None, logger=None):
        """Initialize trainer with configuration."""
        self.task = task
        self.dataset_name = dataset
        self.config = config or Config()
        self.logger = logger or logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load dataset info
        if dataset:
            self.dataset_info = get_dataset_info(dataset)
        else:
            self.dataset_info = {}
        
        self.logger.info(f"ArchitectureTrainer initialized for {task} on {dataset}")
        self.logger.info(f"Using device: {self.device}")
    
    def train_architecture(
        self,
        architecture: Dict[str, Any],
        dataset: str,
        max_epochs: int = 10,
        batch_size: int = 128,
        **kwargs
    ) -> Dict[str, float]:
        """
        Train and evaluate an architecture.
        
        Args:
            architecture: Architecture specification
            dataset: Dataset name
            max_epochs: Maximum training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Training architecture for {max_epochs} epochs on {dataset}")
        
        try:
            # Create model from architecture
            model = self._create_model_from_architecture(architecture)
            model = model.to(self.device)
            
            # Get data loaders
            train_loader, val_loader, _ = get_dataloader(
                dataset, batch_size=batch_size, num_workers=2
            )
            
            # Setup training
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Training loop
            best_val_acc = 0.0
            train_time = time.time()
            
            for epoch in range(max_epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    if batch_idx >= 5:  # Limit training for NAS (quick evaluation)
                        break
                        
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = output.max(1)
                    train_total += target.size(0)
                    train_correct += predicted.eq(target).sum().item()
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(val_loader):
                        if batch_idx >= 3:  # Limit validation for NAS
                            break
                            
                        data, target = data.to(self.device), target.to(self.device)
                        output = model(data)
                        loss = criterion(output, target)
                        
                        val_loss += loss.item()
                        _, predicted = output.max(1)
                        val_total += target.size(0)
                        val_correct += predicted.eq(target).sum().item()
                
                # Calculate accuracies
                train_acc = train_correct / train_total if train_total > 0 else 0.0
                val_acc = val_correct / val_total if val_total > 0 else 0.0
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                
                logger.debug(f"Epoch {epoch}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            train_time = time.time() - train_time
            
            # Calculate model statistics
            num_params = sum(p.numel() for p in model.parameters())
            
            # Estimate FLOPs (simplified)
            flops = self._estimate_flops(architecture)
            
            metrics = {
                'accuracy': best_val_acc,
                'train_accuracy': train_acc,
                'loss': val_loss / max(len(val_loader), 1),
                'train_time': train_time,
                'parameters': num_params,
                'flops': flops,
                'latency': train_time / max_epochs,  # Rough estimate
                'memory': self._estimate_memory(model)
            }
            
            logger.info(f"Architecture training completed. Best accuracy: {best_val_acc:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            # Return poor metrics on failure
            return {
                'accuracy': 0.1,
                'train_accuracy': 0.1,
                'loss': 10.0,
                'train_time': 0.0,
                'parameters': 1e6,
                'flops': 1e9,
                'latency': 100.0,
                'memory': 1000.0
            }
    
    def _create_model_from_architecture(self, architecture: Dict[str, Any]) -> nn.Module:
        """Create a PyTorch model from architecture specification."""
        layers = architecture.get('layers', [])
        
        if not layers:
            raise ValueError("Architecture must have layers")
        
        # Simple sequential model creation
        model_layers = []
        input_shape = architecture.get('input_shape', (3, 32, 32))
        current_channels = input_shape[0]
        spatial_size = input_shape[1] if len(input_shape) > 1 else None
        
        for i, layer_spec in enumerate(layers):
            layer_type = layer_spec.get('type')
            
            if layer_type == 'conv2d':
                out_channels = layer_spec.get('out_channels', 64)
                kernel_size = layer_spec.get('kernel_size', 3)
                stride = layer_spec.get('stride', 1)
                padding = layer_spec.get('padding', 1)
                
                layer = nn.Conv2d(current_channels, out_channels, kernel_size, 
                                stride=stride, padding=padding)
                model_layers.append(layer)
                current_channels = out_channels
                
                if spatial_size and stride > 1:
                    spatial_size = spatial_size // stride
                    
            elif layer_type == 'batchnorm':
                layer = nn.BatchNorm2d(current_channels)
                model_layers.append(layer)
                
            elif layer_type == 'relu':
                layer = nn.ReLU(inplace=layer_spec.get('inplace', False))
                model_layers.append(layer)
                
            elif layer_type == 'adaptive_pool':
                output_size = layer_spec.get('output_size', (1, 1))
                layer = nn.AdaptiveAvgPool2d(output_size)
                model_layers.append(layer)
                spatial_size = output_size[0] if isinstance(output_size, tuple) else output_size
                
            elif layer_type == 'flatten':
                layer = nn.Flatten()
                model_layers.append(layer)
                if spatial_size:
                    current_channels = current_channels * spatial_size * spatial_size
                spatial_size = None
                
            elif layer_type == 'linear':
                out_features = layer_spec.get('out_features', 10)
                in_features = layer_spec.get('in_features', current_channels)
                
                layer = nn.Linear(in_features, out_features)
                model_layers.append(layer)
                current_channels = out_features
                
            elif layer_type == 'dropout':
                p = layer_spec.get('p', 0.5)
                layer = nn.Dropout(p=p)
                model_layers.append(layer)
        
        return nn.Sequential(*model_layers)
    
    def _estimate_flops(self, architecture: Dict[str, Any]) -> float:
        """Estimate FLOPs for the architecture."""
        flops = 0
        layers = architecture.get('layers', [])
        input_shape = architecture.get('input_shape', (3, 32, 32))
        
        current_shape = input_shape
        
        for layer_spec in layers:
            layer_type = layer_spec.get('type')
            
            if layer_type == 'conv2d' and len(current_shape) >= 3:
                out_channels = layer_spec.get('out_channels', 64)
                kernel_size = layer_spec.get('kernel_size', 3)
                
                # Rough FLOP calculation for conv2d
                output_elements = out_channels * current_shape[1] * current_shape[2]
                kernel_flops = kernel_size * kernel_size * current_shape[0]
                flops += output_elements * kernel_flops
                
                current_shape = (out_channels, current_shape[1], current_shape[2])
                
            elif layer_type == 'linear':
                in_features = layer_spec.get('in_features', current_shape[0] if current_shape else 1024)
                out_features = layer_spec.get('out_features', 10)
                flops += in_features * out_features
                current_shape = (out_features,)
        
        return flops
    
    def _estimate_memory(self, model: nn.Module) -> float:
        """Estimate memory usage in MB."""
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())
        total_memory = param_memory + buffer_memory
        return total_memory / (1024 * 1024)  # Convert to MB
    
    def evaluate_architecture(
        self,
        architecture: Dict[str, Any],
        dataset_config: Dict[str, Any],
        max_epochs: int = 1,
        quick_eval: bool = True,
        **kwargs
    ) -> Dict[str, float]:
        """Evaluate architecture performance."""
        dataset_name = dataset_config.get('name', 'dummy')
        
        if dataset_name == 'dummy':
            # Return simulated metrics for dummy dataset
            import numpy as np
            return {
                'accuracy': np.random.uniform(0.7, 0.9),
                'loss': np.random.uniform(0.1, 0.5),
                'train_time': 0.1,
                'parameters': 10000,
                'flops': 1e6,
                'latency': 5.0,
                'memory': 50.0
            }
        
        return self.train_architecture(architecture, dataset_name, max_epochs=max_epochs, **kwargs) 