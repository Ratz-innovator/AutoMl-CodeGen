"""
PyTorch Code Generator

This module generates PyTorch model code from neural architecture specifications.
"""

import logging
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class PyTorchGenerator:
    """
    PyTorch code generator for neural architectures.
    
    Converts architecture specifications into executable PyTorch model code.
    """
    
    def __init__(self, optimization_level: int = 2, code_style: str = "clean"):
        """
        Initialize PyTorch generator.
        
        Args:
            optimization_level: Code optimization level (0-3)
            code_style: Code style ('clean', 'performance', 'readable')
        """
        self.optimization_level = optimization_level
        self.code_style = code_style
        self.indent = "    "
        logger.info(f"PyTorch generator initialized (optimization: {optimization_level})")
    
    def generate_model(self, context: Dict[str, Any]) -> str:
        """
        Generate PyTorch model code from architecture.
        
        Args:
            context: Generation context containing architecture and settings
            
        Returns:
            Generated PyTorch model code as string
        """
        logger.info("Generating PyTorch model code")
        
        architecture = context['architecture']
        
        # Validate architecture
        if not self._validate_architecture(architecture):
            raise ValueError("Invalid architecture specification")
        
        # Generate model class
        model_code = self._generate_model_class(architecture)
        
        # Generate imports
        imports = self._generate_imports(architecture)
        
        # Generate helper functions if needed
        helpers = self._generate_helpers(architecture)
        
        # Combine all code parts
        full_code = f"{imports}\n\n{helpers}\n\n{model_code}"
        
        # Apply optimizations
        if self.optimization_level > 0:
            full_code = self._optimize_code(full_code)
        
        logger.info("PyTorch code generation completed")
        return full_code
    
    def _validate_architecture(self, architecture: Dict[str, Any]) -> bool:
        """Validate architecture specification."""
        required_keys = ['task', 'layers']
        for key in required_keys:
            if key not in architecture:
                logger.error(f"Architecture missing required key: {key}")
                return False
        
        layers = architecture['layers']
        if not isinstance(layers, list) or len(layers) == 0:
            logger.error("Architecture must have at least one layer")
            return False
        
        return True
    
    def _generate_imports(self, architecture: Dict[str, Any]) -> str:
        """Generate import statements."""
        imports = [
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F"
        ]
        
        # Add task-specific imports
        task = architecture.get('task', 'image_classification')
        if 'attention' in str(architecture.get('layers', [])):
            imports.append("import math")
        
        return "\n".join(imports)
    
    def _generate_helpers(self, architecture: Dict[str, Any]) -> str:
        """Generate helper functions if needed."""
        helpers = []
        
        # Check if we need attention helper
        layers = architecture.get('layers', [])
        has_attention = any(layer.get('type') == 'attention' for layer in layers)
        
        if has_attention:
            helpers.append(self._generate_attention_helper())
        
        return "\n\n".join(helpers)
    
    def _generate_attention_helper(self) -> str:
        """Generate multi-head attention helper."""
        return '''class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.w_o(attn_output)'''
    
    def _generate_model_class(self, architecture: Dict[str, Any]) -> str:
        """Generate the main model class."""
        task = architecture.get('task', 'image_classification')
        class_name = f"GeneratedModel"
        
        # Generate __init__ method
        init_method = self._generate_init_method(architecture)
        
        # Generate forward method
        forward_method = self._generate_forward_method(architecture)
        
        # Combine into class
        model_class = f'''class {class_name}(nn.Module):
    def __init__(self):
        super().__init__()
{init_method}

{forward_method}'''
        
        return model_class
    
    def _generate_init_method(self, architecture: Dict[str, Any]) -> str:
        """Generate model __init__ method."""
        layers = architecture.get('layers', [])
        input_shape = architecture.get('input_shape', (3, 224, 224))
        
        init_lines = []
        layer_names = []
        current_channels = input_shape[0] if len(input_shape) >= 3 else 1
        
        for i, layer in enumerate(layers):
            layer_type = layer.get('type')
            layer_name = f"layer_{i}"
            layer_names.append(layer_name)
            
            if layer_type == 'conv2d':
                out_channels = layer.get('out_channels', 64)
                kernel_size = layer.get('kernel_size', 3)
                stride = layer.get('stride', 1)
                padding = layer.get('padding', kernel_size // 2)
                bias = layer.get('bias', True)
                
                init_lines.append(
                    f"{self.indent}{self.indent}self.{layer_name} = nn.Conv2d("
                    f"{current_channels}, {out_channels}, {kernel_size}, "
                    f"stride={stride}, padding={padding}, bias={bias})"
                )
                current_channels = out_channels
                
            elif layer_type == 'linear':
                out_features = layer.get('out_features', 10)
                in_features = layer.get('in_features', current_channels)
                bias = layer.get('bias', True)
                
                init_lines.append(
                    f"{self.indent}{self.indent}self.{layer_name} = nn.Linear("
                    f"{in_features}, {out_features}, bias={bias})"
                )
                current_channels = out_features
                
            elif layer_type == 'batchnorm':
                num_features = layer.get('num_features', current_channels)
                eps = layer.get('eps', 1e-5)
                momentum = layer.get('momentum', 0.1)
                
                init_lines.append(
                    f"{self.indent}{self.indent}self.{layer_name} = nn.BatchNorm2d("
                    f"{num_features}, eps={eps}, momentum={momentum})"
                )
                
            elif layer_type == 'dropout':
                p = layer.get('p', 0.5)
                init_lines.append(
                    f"{self.indent}{self.indent}self.{layer_name} = nn.Dropout(p={p})"
                )
                
            elif layer_type == 'adaptive_pool':
                output_size = layer.get('output_size', (1, 1))
                init_lines.append(
                    f"{self.indent}{self.indent}self.{layer_name} = nn.AdaptiveAvgPool2d({output_size})"
                )
                
            elif layer_type == 'attention':
                num_heads = layer.get('num_heads', 8)
                dropout = layer.get('dropout', 0.1)
                init_lines.append(
                    f"{self.indent}{self.indent}self.{layer_name} = MultiHeadAttention("
                    f"{current_channels}, {num_heads}, dropout={dropout})"
                )
        
        # Store layer names for forward method
        self._layer_names = layer_names
        
        return "\n".join(init_lines)
    
    def _generate_forward_method(self, architecture: Dict[str, Any]) -> str:
        """Generate model forward method."""
        layers = architecture.get('layers', [])
        
        forward_lines = [
            f"{self.indent}def forward(self, x):"
        ]
        
        for i, layer in enumerate(layers):
            layer_type = layer.get('type')
            layer_name = f"layer_{i}"
            
            if layer_type in ['conv2d', 'linear', 'batchnorm', 'dropout', 'adaptive_pool', 'attention']:
                forward_lines.append(f"{self.indent}{self.indent}x = self.{layer_name}(x)")
                
            elif layer_type == 'relu':
                inplace = layer.get('inplace', False)
                forward_lines.append(f"{self.indent}{self.indent}x = F.relu(x, inplace={inplace})")
                
            elif layer_type == 'gelu':
                forward_lines.append(f"{self.indent}{self.indent}x = F.gelu(x)")
                
            elif layer_type == 'swish':
                forward_lines.append(f"{self.indent}{self.indent}x = x * torch.sigmoid(x)")
                
            elif layer_type == 'flatten':
                forward_lines.append(f"{self.indent}{self.indent}x = torch.flatten(x, 1)")
                
            else:
                # Unknown layer type, add as comment
                forward_lines.append(f"{self.indent}{self.indent}# {layer_type}: {layer}")
        
        forward_lines.append(f"{self.indent}{self.indent}return x")
        
        return "\n".join(forward_lines)
    
    def _optimize_code(self, code: str) -> str:
        """Apply code optimizations based on optimization level."""
        if self.optimization_level >= 1:
            # Level 1: Remove unnecessary whitespace
            lines = code.split('\n')
            optimized_lines = []
            for line in lines:
                if line.strip():  # Keep non-empty lines
                    optimized_lines.append(line)
                elif optimized_lines and optimized_lines[-1].strip():  # Add blank line after non-empty
                    optimized_lines.append('')
            code = '\n'.join(optimized_lines)
        
        if self.optimization_level >= 2:
            # Level 2: Add performance optimizations (without jit decoration on class)
            # Note: torch.jit.script should be applied to instances, not class definitions
            pass
        
        if self.optimization_level >= 3:
            # Level 3: Add memory optimization hints
            code = code.replace('def forward(self, x):', 
                              'def forward(self, x):\n        # torch.cuda.empty_cache()  # Uncomment for GPU memory optimization')
        
        return code
    

    
    def generate_training(self, context: Dict[str, Any]) -> str:
        """Generate training loop code."""
        return '''
# Training code
def train_model(model, train_loader, val_loader, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        print(f'Epoch {epoch}: Val Loss: {val_loss:.4f}, Accuracy: {correct/len(val_loader.dataset):.4f}')
'''

    def generate_inference(self, context: Dict[str, Any]) -> str:
        """Generate inference code."""
        return '''
# Inference code
def predict(model, x):
    model.eval()
    with torch.no_grad():
        output = model(x)
        return F.softmax(output, dim=1)

# Example usage:
# model = GeneratedModel()
# model.load_state_dict(torch.load('model.pth'))
# predictions = predict(model, input_tensor)
'''

    def generate_deployment(self, context: Dict[str, Any]) -> str:
        """Generate deployment code."""
        return '''
# Deployment code
import torch.jit

def optimize_for_deployment(model):
    """Optimize model for deployment."""
    model.eval()
    
    # Convert to TorchScript
    traced_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
    
    # Optimize
    optimized_model = torch.jit.optimize_for_inference(traced_model)
    
    return optimized_model

# Save optimized model
# optimized = optimize_for_deployment(model)
# torch.jit.save(optimized, 'optimized_model.pt')
'''

    def get_requirements(self, context: Dict[str, Any]) -> List[str]:
        """Get required packages."""
        requirements = [
            'torch>=2.0.0',
            'torchvision>=0.15.0',
            'numpy>=1.21.0'
        ]
        
        architecture = context.get('architecture', {})
        layers = architecture.get('layers', [])
        
        # Add additional requirements based on architecture
        if any(layer.get('type') == 'attention' for layer in layers):
            requirements.append('math')
        
        return requirements 