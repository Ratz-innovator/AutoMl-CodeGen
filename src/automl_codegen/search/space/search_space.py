"""
Neural Architecture Search Space Definition

This module defines the search space for neural architectures, including
operations, connections, and constraints. It provides methods for sampling
architectures and validating their structure.
"""

import random
import copy
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

@dataclass
class LayerSpec:
    """Specification for a neural network layer."""
    type: str
    params: Dict[str, Any]
    constraints: Optional[Dict[str, Any]] = None
    
class SearchSpace:
    """
    Neural architecture search space definition.
    
    This class defines the space of possible neural network architectures
    that can be explored during search. It includes operations, connection
    patterns, and hardware-aware constraints.
    
    Example:
        >>> space = SearchSpace(task='image_classification')
        >>> architecture = space.sample_architecture()
        >>> valid_ops = space.get_valid_layer_types(layer_idx=3)
    """
    
    def __init__(
        self,
        task: str = 'image_classification',
        custom_space: Optional[Dict[str, Any]] = None,
        hardware_target: str = 'gpu',
        min_layers: int = 3,
        max_layers: int = 20,
        **kwargs
    ):
        """
        Initialize search space.
        
        Args:
            task: Task type (image_classification, object_detection, etc.)
            custom_space: Custom search space definition
            hardware_target: Target hardware platform
            min_layers: Minimum number of layers
            max_layers: Maximum number of layers
        """
        self.task = task
        self.hardware_target = hardware_target
        self.min_layers = min_layers
        self.max_layers = max_layers
        
        # Load default search space based on task
        if custom_space:
            self.search_space = custom_space
        else:
            self.search_space = self._get_default_search_space()
        
        # Available operations
        self.operations = self._get_available_operations()
        
        # Connection patterns
        self.connection_patterns = ['sequential', 'residual', 'dense']
        
        # Hardware constraints
        self.hardware_constraints = self._get_hardware_constraints()
    
    def _get_default_search_space(self) -> Dict[str, Any]:
        """Get default search space based on task."""
        if self.task == 'image_classification':
            return {
                'input_channels': 3,
                'num_classes': 10,  # Will be updated based on dataset
                'operations': [
                    'conv2d', 'depthwise_conv', 'pointwise_conv',
                    'linear', 'batchnorm', 'layernorm',
                    'relu', 'gelu', 'swish', 'attention',
                    'adaptive_pool', 'dropout', 'flatten'
                ],
                'channels': [16, 32, 64, 128, 256, 512, 1024],
                'kernel_sizes': [1, 3, 5, 7],
                'strides': [1, 2],
                'activations': ['relu', 'gelu', 'swish'],
                'pool_sizes': [2, 3, 4],
                'dropout_rates': [0.0, 0.1, 0.2, 0.3, 0.5],
                'attention_heads': [1, 2, 4, 8]
            }
        
        elif self.task == 'object_detection':
            return {
                'input_channels': 3,
                'operations': [
                    'conv2d', 'depthwise_conv', 'pointwise_conv',
                    'linear', 'batchnorm', 'relu', 'gelu',
                    'upsample', 'concat', 'fpn'
                ],
                'channels': [32, 64, 128, 256, 512, 1024],
                'kernel_sizes': [1, 3, 5],
                'strides': [1, 2],
                'num_anchors': [3, 6, 9],
                'fpn_levels': [3, 4, 5]
            }
        
        else:
            # Generic search space
            return {
                'operations': [
                    'conv2d', 'linear', 'batchnorm', 'relu',
                    'dropout', 'adaptive_pool', 'flatten'
                ],
                'channels': [32, 64, 128, 256, 512],
                'kernel_sizes': [1, 3, 5],
                'strides': [1, 2]
            }
    
    def _get_available_operations(self) -> Dict[str, Dict[str, Any]]:
        """Get available operations with their parameter spaces."""
        operations = {
            'conv2d': {
                'params': {
                    'out_channels': self.search_space.get('channels', [64, 128, 256]),
                    'kernel_size': self.search_space.get('kernel_sizes', [1, 3, 5]),
                    'stride': self.search_space.get('strides', [1, 2]),
                    'padding': 'auto',  # Will be calculated based on kernel_size
                    'bias': [True, False]
                },
                'constraints': {
                    'input_dims': 4,  # Batch, Channel, Height, Width
                    'output_dims': 4
                }
            },
            
            'depthwise_conv': {
                'params': {
                    'kernel_size': self.search_space.get('kernel_sizes', [3, 5, 7]),
                    'stride': self.search_space.get('strides', [1, 2]),
                    'padding': 'auto'
                },
                'constraints': {
                    'input_dims': 4,
                    'output_dims': 4
                }
            },
            
            'pointwise_conv': {
                'params': {
                    'out_channels': self.search_space.get('channels', [64, 128, 256])
                },
                'constraints': {
                    'input_dims': 4,
                    'output_dims': 4
                }
            },
            
            'linear': {
                'params': {
                    'out_features': [64, 128, 256, 512, 1024],
                    'bias': [True, False]
                },
                'constraints': {
                    'input_dims': 2,  # Batch, Features
                    'output_dims': 2
                }
            },
            
            'batchnorm': {
                'params': {
                    'eps': [1e-5, 1e-4],
                    'momentum': [0.1, 0.01]
                },
                'constraints': {
                    'follows': ['conv2d', 'linear']
                }
            },
            
            'layernorm': {
                'params': {
                    'eps': [1e-5, 1e-4]
                },
                'constraints': {}
            },
            
            'relu': {
                'params': {
                    'inplace': [True, False]
                },
                'constraints': {}
            },
            
            'gelu': {
                'params': {},
                'constraints': {}
            },
            
            'swish': {
                'params': {},
                'constraints': {}
            },
            
            'attention': {
                'params': {
                    'num_heads': self.search_space.get('attention_heads', [1, 2, 4, 8]),
                    'dropout': self.search_space.get('dropout_rates', [0.0, 0.1])
                },
                'constraints': {
                    'input_dims': 3  # Batch, Sequence, Features
                }
            },
            
            'adaptive_pool': {
                'params': {
                    'output_size': [(1, 1), (2, 2), (4, 4)]
                },
                'constraints': {
                    'input_dims': 4,
                    'output_dims': 4
                }
            },
            
            'dropout': {
                'params': {
                    'p': self.search_space.get('dropout_rates', [0.1, 0.2, 0.3, 0.5])
                },
                'constraints': {}
            },
            
            'flatten': {
                'params': {},
                'constraints': {
                    'input_dims': [3, 4],
                    'output_dims': 2
                }
            }
        }
        
        return operations
    
    def _get_hardware_constraints(self) -> Dict[str, Any]:
        """Get hardware-specific constraints."""
        if self.hardware_target == 'mobile':
            return {
                'max_parameters': 10_000_000,  # 10M parameters
                'max_flops': 500_000_000,      # 500M FLOPs
                'max_memory': 100,             # 100MB
                'preferred_ops': ['depthwise_conv', 'pointwise_conv'],
                'avoid_ops': ['attention'],
                'max_channels': 512
            }
        
        elif self.hardware_target == 'edge':
            return {
                'max_parameters': 5_000_000,   # 5M parameters
                'max_flops': 100_000_000,      # 100M FLOPs
                'max_memory': 50,              # 50MB
                'preferred_ops': ['depthwise_conv', 'pointwise_conv'],
                'avoid_ops': ['attention', 'large_kernels'],
                'max_channels': 256
            }
        
        elif self.hardware_target == 'gpu':
            return {
                'max_parameters': 100_000_000, # 100M parameters
                'max_flops': 10_000_000_000,   # 10B FLOPs
                'max_memory': 8000,            # 8GB
                'preferred_ops': ['conv2d', 'attention'],
                'avoid_ops': [],
                'max_channels': 2048
            }
        
        else:  # cpu or auto
            return {
                'max_parameters': 50_000_000,  # 50M parameters
                'max_flops': 5_000_000_000,    # 5B FLOPs
                'max_memory': 4000,            # 4GB
                'preferred_ops': ['conv2d', 'linear'],
                'avoid_ops': ['attention'],
                'max_channels': 1024
            }
    
    def sample_architecture(self) -> Dict[str, Any]:
        """
        Sample a random architecture from the search space.
        
        Returns:
            Dictionary representing a neural architecture
        """
        # Sample number of layers
        num_layers = random.randint(self.min_layers, self.max_layers)
        
        # Sample connection pattern
        connection_pattern = random.choice(self.connection_patterns)
        
        # Sample layers
        layers = []
        current_channels = self.search_space.get('input_channels', 3)
        spatial_dims = True  # Whether we're still in spatial dimensions
        
        for i in range(num_layers):
            layer = self._sample_layer(i, current_channels, spatial_dims, layers)
            layers.append(layer)
            
            # Update state for next layer
            if layer['type'] in ['conv2d', 'depthwise_conv', 'pointwise_conv']:
                current_channels = layer.get('out_channels', current_channels)
            elif layer['type'] == 'linear':
                current_channels = layer.get('out_features', current_channels)
                spatial_dims = False
            elif layer['type'] == 'flatten':
                spatial_dims = False
        
        # Ensure we have a final classification layer for classification tasks
        if self.task == 'image_classification' and layers[-1]['type'] != 'linear':
            if spatial_dims:
                layers.append({'type': 'adaptive_pool', 'output_size': (1, 1)})
                layers.append({'type': 'flatten'})
            
            num_classes = self.search_space.get('num_classes', 10)
            layers.append({
                'type': 'linear',
                'out_features': num_classes
            })
        
        architecture = {
            'task': self.task,
            'layers': layers,
            'connections': connection_pattern,
            'input_shape': self._get_input_shape(),
            'hardware_target': self.hardware_target
        }
        
        return architecture
    
    def _sample_layer(
        self,
        layer_idx: int,
        input_channels: int,
        spatial_dims: bool,
        previous_layers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Sample a single layer."""
        # Get valid operations for this position
        valid_ops = self._get_valid_operations(layer_idx, spatial_dims, previous_layers)
        
        if not valid_ops:
            # Fallback to basic operations
            valid_ops = ['relu'] if spatial_dims else ['linear']
        
        # Sample operation type
        op_type = random.choice(valid_ops)
        
        # Sample parameters for this operation
        op_spec = self.operations.get(op_type, {})
        layer = {'type': op_type}
        
        # Sample parameters
        for param, values in op_spec.get('params', {}).items():
            if isinstance(values, list):
                layer[param] = random.choice(values)
            elif param == 'padding' and values == 'auto':
                # Auto-calculate padding
                kernel_size = layer.get('kernel_size', 3)
                layer[param] = kernel_size // 2
            elif param == 'out_channels':
                # Ensure channels don't exceed hardware constraints
                max_channels = self.hardware_constraints.get('max_channels', 1024)
                available_channels = [c for c in values if c <= max_channels]
                if available_channels:
                    layer[param] = random.choice(available_channels)
                else:
                    layer[param] = min(values)
        
        # Set input-dependent parameters
        if op_type == 'batchnorm':
            layer['num_features'] = input_channels
        elif op_type == 'linear' and 'in_features' not in layer:
            # For linear layers, we'll set in_features based on previous layer
            layer['in_features'] = input_channels
        
        return layer
    
    def _get_valid_operations(
        self,
        layer_idx: int,
        spatial_dims: bool,
        previous_layers: List[Dict[str, Any]]
    ) -> List[str]:
        """Get valid operations for a given position."""
        all_ops = list(self.operations.keys())
        valid_ops = []
        
        # Filter based on spatial dimensions
        for op in all_ops:
            op_spec = self.operations[op]
            constraints = op_spec.get('constraints', {})
            
            # Check input dimension requirements
            if 'input_dims' in constraints:
                required_dims = constraints['input_dims']
                if isinstance(required_dims, int):
                    required_dims = [required_dims]
                
                current_dims = 4 if spatial_dims else 2
                if current_dims not in required_dims:
                    continue
            
            # Check if operation should follow specific operations
            if 'follows' in constraints and previous_layers:
                last_layer_type = previous_layers[-1]['type']
                if last_layer_type not in constraints['follows']:
                    continue
            
            # Check hardware constraints
            if op in self.hardware_constraints.get('avoid_ops', []):
                continue
            
            valid_ops.append(op)
        
        # Ensure we have some valid operations
        if not valid_ops:
            if spatial_dims:
                valid_ops = ['conv2d', 'relu']
            else:
                valid_ops = ['linear', 'relu']
        
        return valid_ops
    
    def get_valid_layer_types(self, layer_idx: int) -> List[str]:
        """Get valid layer types for a specific position."""
        # This is a simplified version for external use
        return list(self.operations.keys())
    
    def sample_layer(self) -> Dict[str, Any]:
        """Sample a random layer (for mutations)."""
        op_type = random.choice(list(self.operations.keys()))
        return self._sample_layer(0, 64, True, [])
    
    def _get_input_shape(self) -> Tuple[int, ...]:
        """Get input shape based on task."""
        if self.task == 'image_classification':
            return (3, 224, 224)
        elif self.task == 'object_detection':
            return (3, 416, 416)
        elif self.task == 'text_classification':
            return (512,)
        else:
            return (3, 224, 224)
    
    def validate_architecture(self, architecture: Dict[str, Any]) -> bool:
        """
        Validate that an architecture is valid within this search space.
        
        Args:
            architecture: Architecture to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            layers = architecture.get('layers', [])
            
            # Check layer count
            if len(layers) < self.min_layers or len(layers) > self.max_layers:
                return False
            
            # Check each layer
            for i, layer in enumerate(layers):
                layer_type = layer.get('type')
                if layer_type not in self.operations:
                    return False
                
                # Check parameters
                op_spec = self.operations[layer_type]
                for param, value in layer.items():
                    if param == 'type':
                        continue
                    
                    if param in op_spec.get('params', {}):
                        valid_values = op_spec['params'][param]
                        if isinstance(valid_values, list) and value not in valid_values:
                            return False
            
            return True
            
        except Exception:
            return False
    
    def estimate_complexity(self, architecture: Dict[str, Any]) -> Dict[str, float]:
        """
        Estimate the computational complexity of an architecture.
        
        Args:
            architecture: Architecture to analyze
            
        Returns:
            Dictionary with complexity metrics
        """
        layers = architecture.get('layers', [])
        
        total_params = 0
        total_flops = 0
        memory_mb = 0
        
        # Simplified complexity estimation
        current_shape = list(self._get_input_shape())
        
        for layer in layers:
            layer_type = layer.get('type')
            
            if layer_type == 'conv2d':
                out_channels = layer.get('out_channels', 64)
                kernel_size = layer.get('kernel_size', 3)
                
                if len(current_shape) == 3:  # C, H, W
                    in_channels = current_shape[0]
                    h, w = current_shape[1], current_shape[2]
                    
                    # Parameters: out_channels * in_channels * kernel_size^2
                    layer_params = out_channels * in_channels * kernel_size * kernel_size
                    total_params += layer_params
                    
                    # FLOPs: output_height * output_width * layer_params
                    stride = layer.get('stride', 1)
                    out_h, out_w = h // stride, w // stride
                    layer_flops = out_h * out_w * layer_params
                    total_flops += layer_flops
                    
                    # Update shape
                    current_shape = [out_channels, out_h, out_w]
            
            elif layer_type == 'linear':
                out_features = layer.get('out_features', 128)
                
                if len(current_shape) == 1:  # Features
                    in_features = current_shape[0]
                elif len(current_shape) == 3:  # Need to flatten
                    in_features = current_shape[0] * current_shape[1] * current_shape[2]
                    current_shape = [in_features]
                else:
                    in_features = 1024  # Default assumption
                
                layer_params = in_features * out_features
                total_params += layer_params
                total_flops += layer_params
                
                current_shape = [out_features]
            
            elif layer_type == 'flatten':
                if len(current_shape) == 3:
                    flattened_size = current_shape[0] * current_shape[1] * current_shape[2]
                    current_shape = [flattened_size]
        
        # Estimate memory usage (very rough)
        memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per parameter
        
        return {
            'parameters': total_params,
            'flops': total_flops,
            'memory_mb': memory_mb
        }
    
    def mutate_architecture(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a random mutation to an architecture.
        
        Args:
            architecture: Architecture to mutate
            
        Returns:
            Mutated architecture
        """
        mutated = copy.deepcopy(architecture)
        layers = mutated['layers']
        
        if not layers:
            return mutated
        
        mutation_type = random.choice([
            'change_layer_type',
            'change_parameters',
            'add_layer',
            'remove_layer'
        ])
        
        if mutation_type == 'change_layer_type' and layers:
            # Change the type of a random layer
            idx = random.randint(0, len(layers) - 1)
            current_type = layers[idx]['type']
            valid_types = self.get_valid_layer_types(idx)
            
            if len(valid_types) > 1:
                new_types = [t for t in valid_types if t != current_type]
                if new_types:
                    new_type = random.choice(new_types)
                    layers[idx] = self._sample_layer(idx, 64, True, layers[:idx])
                    layers[idx]['type'] = new_type
        
        elif mutation_type == 'change_parameters' and layers:
            # Change parameters of a random layer
            idx = random.randint(0, len(layers) - 1)
            layer = layers[idx]
            layer_type = layer['type']
            
            if layer_type in self.operations:
                op_spec = self.operations[layer_type]
                params = op_spec.get('params', {})
                
                if params:
                    param_name = random.choice(list(params.keys()))
                    if isinstance(params[param_name], list):
                        layer[param_name] = random.choice(params[param_name])
        
        elif mutation_type == 'add_layer' and len(layers) < self.max_layers:
            # Add a new random layer
            idx = random.randint(0, len(layers))
            new_layer = self.sample_layer()
            layers.insert(idx, new_layer)
        
        elif mutation_type == 'remove_layer' and len(layers) > self.min_layers:
            # Remove a random layer
            idx = random.randint(0, len(layers) - 1)
            layers.pop(idx)
        
        return mutated 