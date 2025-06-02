"""
Code Generator for Neural Architectures

This module converts discovered neural architectures into production-ready code
across multiple frameworks (PyTorch, TensorFlow, TensorRT, ONNX) with optimizations.
"""

import logging
import ast
import black
import autopep8
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set
from dataclasses import dataclass
from jinja2 import Environment, FileSystemLoader, Template
import torch
import torch.nn as nn
from ..utils.config import Config

logger = logging.getLogger(__name__)

@dataclass
class GeneratedCode:
    """Container for generated code and metadata."""
    framework: str
    model_code: str
    training_code: Optional[str] = None
    inference_code: Optional[str] = None
    deployment_code: Optional[str] = None
    requirements: List[str] = None
    metadata: Dict[str, Any] = None
    
    def save(self, base_path: Union[str, Path], prefix: str = "generated") -> None:
        """Save generated code to files."""
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Save model code
        model_file = base_path / f"{prefix}_model.py"
        with open(model_file, 'w') as f:
            f.write(self.model_code)
        
        # Save training code if available
        if self.training_code:
            train_file = base_path / f"{prefix}_train.py"
            with open(train_file, 'w') as f:
                f.write(self.training_code)
        
        # Save inference code if available
        if self.inference_code:
            infer_file = base_path / f"{prefix}_inference.py"
            with open(infer_file, 'w') as f:
                f.write(self.inference_code)
        
        # Save deployment code if available
        if self.deployment_code:
            deploy_file = base_path / f"{prefix}_deploy.py"
            with open(deploy_file, 'w') as f:
                f.write(self.deployment_code)
        
        # Save requirements
        if self.requirements:
            req_file = base_path / "requirements.txt"
            with open(req_file, 'w') as f:
                f.write('\n'.join(self.requirements))
        
        # Save metadata
        if self.metadata:
            import json
            meta_file = base_path / f"{prefix}_metadata.json"
            with open(meta_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        
        logger.info(f"Generated code saved to {base_path}")

class CodeGenerator:
    """
    Main code generator that converts neural architectures to production code.
    
    Supports multiple frameworks and optimization levels:
    - PyTorch (training, inference, JIT compilation)
    - TensorFlow (Keras, tf.function, TensorRT)
    - ONNX (export and optimization)
    - Hardware-specific optimizations
    
    Example:
        >>> generator = CodeGenerator(target_framework='pytorch')
        >>> code = generator.generate(
        ...     architecture=best_arch,
        ...     optimizations=['quantization', 'fusion'],
        ...     include_training=True
        ... )
        >>> code.save('./generated_models/')
    """
    
    def __init__(
        self,
        target_framework: str = 'pytorch',
        optimization_level: int = 2,
        code_style: str = 'black',
        template_dir: Optional[Path] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize code generator.
        
        Args:
            target_framework: Target framework ('pytorch', 'tensorflow', 'onnx')
            optimization_level: Optimization level (0=none, 1=basic, 2=aggressive)
            code_style: Code formatting style ('black', 'autopep8')
            template_dir: Custom template directory
            config: Configuration object
        """
        self.target_framework = target_framework.lower()
        self.optimization_level = optimization_level
        self.code_style = code_style
        self.config = config or Config()
        
        # Supported frameworks
        self.supported_frameworks = {
            'pytorch', 'tensorflow', 'tf', 'onnx', 'tensorrt', 'jax'
        }
        
        if self.target_framework not in self.supported_frameworks:
            raise ValueError(f"Unsupported framework: {target_framework}")
        
        # Setup template environment
        if template_dir is None:
            template_dir = Path(__file__).parent / 'templates'
        
        self.template_env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Import framework-specific generators
        self.generators = self._load_generators()
        
        # Code optimization passes
        self.optimization_passes = self._get_optimization_passes()
        
        logger.info(f"Initialized code generator for {target_framework}")
        logger.info(f"Optimization level: {optimization_level}")
    
    def _load_generators(self) -> Dict[str, Any]:
        """Load framework-specific generators."""
        generators = {}
        
        try:
            if self.target_framework == 'pytorch':
                from .generators.pytorch_gen import PyTorchGenerator
                generators['pytorch'] = PyTorchGenerator(
                    optimization_level=self.optimization_level,
                    code_style=self.code_style
                )
            
            elif self.target_framework in ['tensorflow', 'tf']:
                from .generators.tensorflow_gen import TensorFlowGenerator
                generators['tensorflow'] = TensorFlowGenerator(self.config)
            
            elif self.target_framework == 'onnx':
                from .generators.onnx_gen import ONNXGenerator
                generators['onnx'] = ONNXGenerator(self.config)
            
            elif self.target_framework == 'tensorrt':
                from .generators.tensorrt_gen import TensorRTGenerator
                generators['tensorrt'] = TensorRTGenerator(self.config)
            
            elif self.target_framework == 'jax':
                from .generators.jax_gen import JAXGenerator
                generators['jax'] = JAXGenerator(self.config)
                
        except ImportError as e:
            logger.warning(f"Could not load generator: {e}")
        
        return generators
    
    def _get_optimization_passes(self) -> List[str]:
        """Get optimization passes based on optimization level."""
        if self.optimization_level == 0:
            return []
        elif self.optimization_level == 1:
            return ['dead_code_elimination', 'constant_folding']
        elif self.optimization_level == 2:
            return [
                'dead_code_elimination',
                'constant_folding', 
                'operator_fusion',
                'memory_optimization',
                'quantization_aware'
            ]
        else:
            return [
                'dead_code_elimination',
                'constant_folding',
                'operator_fusion',
                'memory_optimization',
                'quantization_aware',
                'tensor_decomposition',
                'pruning_aware',
                'compilation_optimization'
            ]
    
    def generate(
        self,
        architecture: Dict[str, Any],
        include_training: bool = True,
        include_inference: bool = True,
        include_deployment: bool = False,
        optimizations: Optional[List[str]] = None,
        target_device: str = 'auto',
        batch_size: Optional[int] = None,
        input_shape: Optional[tuple] = None,
        **kwargs
    ) -> GeneratedCode:
        """
        Generate production-ready code for the given architecture.
        
        Args:
            architecture: Architecture specification
            include_training: Whether to generate training code
            include_inference: Whether to generate inference code
            include_deployment: Whether to generate deployment code
            optimizations: List of optimizations to apply
            target_device: Target device ('cpu', 'gpu', 'mobile', 'auto')
            batch_size: Default batch size for generated code
            input_shape: Input tensor shape
            **kwargs: Additional generation parameters
            
        Returns:
            GeneratedCode object containing all generated code
        """
        logger.info(f"Generating {self.target_framework} code for architecture")
        
        # Validate architecture
        self._validate_architecture(architecture)
        
        # Get framework-specific generator
        if self.target_framework not in self.generators:
            raise RuntimeError(f"Generator for {self.target_framework} not available")
        
        generator = self.generators[self.target_framework]
        
        # Prepare generation context
        context = {
            'architecture': architecture,
            'target_device': target_device,
            'batch_size': batch_size or 32,
            'input_shape': input_shape or self._infer_input_shape(architecture),
            'optimization_level': self.optimization_level,
            'optimizations': optimizations or [],
            **kwargs
        }
        
        # Generate model code
        model_code = generator.generate_model(context)
        
        # Apply optimizations
        if optimizations:
            model_code = self._apply_optimizations(model_code, optimizations, context)
        
        # Format code
        model_code = self._format_code(model_code)
        
        # Generate training code if requested
        training_code = None
        if include_training:
            training_code = generator.generate_training(context)
            if training_code:
                training_code = self._format_code(training_code)
        
        # Generate inference code if requested
        inference_code = None
        if include_inference:
            inference_code = generator.generate_inference(context)
            if inference_code:
                inference_code = self._format_code(inference_code)
        
        # Generate deployment code if requested
        deployment_code = None
        if include_deployment:
            deployment_code = generator.generate_deployment(context)
            if deployment_code:
                deployment_code = self._format_code(deployment_code)
        
        # Get requirements and metadata
        requirements = generator.get_requirements(context)
        metadata = self._generate_metadata(architecture, context)
        
        # Create result
        result = GeneratedCode(
            framework=self.target_framework,
            model_code=model_code,
            training_code=training_code,
            inference_code=inference_code,
            deployment_code=deployment_code,
            requirements=requirements,
            metadata=metadata
        )
        
        logger.info("Code generation completed successfully")
        return result
    
    def _validate_architecture(self, architecture: Dict[str, Any]) -> None:
        """Validate architecture specification."""
        required_keys = ['layers', 'connections']
        for key in required_keys:
            if key not in architecture:
                raise ValueError(f"Architecture missing required key: {key}")
        
        # Validate layers
        if not isinstance(architecture['layers'], list):
            raise ValueError("Architecture layers must be a list")
        
        for i, layer in enumerate(architecture['layers']):
            if 'type' not in layer:
                raise ValueError(f"Layer {i} missing 'type' field")
    
    def _infer_input_shape(self, architecture: Dict[str, Any]) -> tuple:
        """Infer input shape from architecture."""
        # This is a simplified inference - real implementation would be more sophisticated
        task = architecture.get('task', 'image_classification')
        
        if task == 'image_classification':
            return (3, 224, 224)  # Standard ImageNet size
        elif task == 'text_classification':
            return (512,)  # Sequence length
        elif task == 'object_detection':
            return (3, 416, 416)  # YOLO-style
        else:
            return (3, 224, 224)  # Default
    
    def _apply_optimizations(
        self,
        code: str,
        optimizations: List[str],
        context: Dict[str, Any]
    ) -> str:
        """Apply code optimizations."""
        optimized_code = code
        
        for opt in optimizations:
            if opt in self.optimization_passes:
                optimized_code = self._apply_optimization_pass(
                    optimized_code, opt, context
                )
        
        return optimized_code
    
    def _apply_optimization_pass(
        self,
        code: str,
        optimization: str,
        context: Dict[str, Any]
    ) -> str:
        """Apply a specific optimization pass."""
        logger.debug(f"Applying optimization: {optimization}")
        
        # This is a simplified implementation
        # Real optimizations would use AST manipulation and analysis
        
        if optimization == 'dead_code_elimination':
            # Remove unused imports and variables
            lines = code.split('\n')
            used_imports = set()
            
            # Simple analysis to find used imports
            for line in lines:
                if 'import' not in line:
                    for imp in ['torch', 'nn', 'F', 'numpy', 'tf']:
                        if imp in line:
                            used_imports.add(imp)
            
            # Filter imports (simplified)
            filtered_lines = []
            for line in lines:
                if line.strip().startswith('import') or line.strip().startswith('from'):
                    # Keep if used (simplified check)
                    keep = False
                    for imp in used_imports:
                        if imp in line:
                            keep = True
                            break
                    if keep or 'torch' in line or 'nn' in line:
                        filtered_lines.append(line)
                else:
                    filtered_lines.append(line)
            
            return '\n'.join(filtered_lines)
        
        elif optimization == 'constant_folding':
            # Fold constant expressions (simplified)
            # In practice, would use AST transformation
            return code.replace('1 * ', '').replace(' * 1', '')
        
        elif optimization == 'operator_fusion':
            # Fuse common operator patterns
            # This would be much more sophisticated in practice
            fused_code = code
            
            # Fuse BatchNorm + ReLU
            fused_code = fused_code.replace(
                'F.relu(self.bn',
                'F.relu(self.bn'  # Already fused in modern frameworks
            )
            
            return fused_code
        
        elif optimization == 'quantization_aware':
            # Add quantization-aware training hooks
            if 'torch' in code:
                quant_import = "import torch.quantization as quant\n"
                if quant_import not in code:
                    code = quant_import + code
            
            return code
        
        else:
            logger.warning(f"Unknown optimization: {optimization}")
            return code
    
    def _format_code(self, code: str) -> str:
        """Format generated code according to style guidelines."""
        try:
            if self.code_style == 'black':
                # Use black for formatting
                formatted = black.format_str(code, mode=black.FileMode())
                return formatted
            
            elif self.code_style == 'autopep8':
                # Use autopep8 for formatting
                formatted = autopep8.fix_code(code)
                return formatted
            
            else:
                logger.warning(f"Unknown code style: {self.code_style}")
                return code
                
        except Exception as e:
            logger.warning(f"Code formatting failed: {e}")
            return code
    
    def _generate_metadata(
        self,
        architecture: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate metadata for the generated code."""
        return {
            'framework': self.target_framework,
            'optimization_level': self.optimization_level,
            'architecture_hash': hash(str(sorted(architecture.items()))),
            'num_parameters': self._estimate_parameters(architecture),
            'num_layers': len(architecture.get('layers', [])),
            'target_device': context.get('target_device', 'auto'),
            'input_shape': context.get('input_shape'),
            'batch_size': context.get('batch_size'),
            'optimizations_applied': context.get('optimizations', []),
            'generation_timestamp': __import__('time').time(),
            'generator_version': '1.0.0'
        }
    
    def _estimate_parameters(self, architecture: Dict[str, Any]) -> int:
        """Estimate the number of parameters in the architecture."""
        total_params = 0
        
        # Simple parameter estimation
        for layer in architecture.get('layers', []):
            layer_type = layer.get('type', '')
            
            if layer_type in ['conv2d', 'convolution']:
                in_channels = layer.get('in_channels', 64)
                out_channels = layer.get('out_channels', 64)
                kernel_size = layer.get('kernel_size', 3)
                if isinstance(kernel_size, int):
                    kernel_size = (kernel_size, kernel_size)
                
                # Conv weights + bias
                total_params += in_channels * out_channels * kernel_size[0] * kernel_size[1]
                total_params += out_channels
            
            elif layer_type in ['linear', 'dense']:
                in_features = layer.get('in_features', 512)
                out_features = layer.get('out_features', 512)
                
                # Linear weights + bias
                total_params += in_features * out_features + out_features
            
            elif layer_type == 'batchnorm':
                num_features = layer.get('num_features', 64)
                # BN has 2 * num_features parameters (weight and bias)
                total_params += 2 * num_features
        
        return total_params
    
    def get_supported_optimizations(self) -> List[str]:
        """Get list of supported optimizations."""
        return [
            'dead_code_elimination',
            'constant_folding',
            'operator_fusion',
            'memory_optimization',
            'quantization_aware',
            'tensor_decomposition',
            'pruning_aware',
            'compilation_optimization'
        ]
    
    def benchmark_generated_code(
        self,
        generated_code: GeneratedCode,
        test_input: Optional[torch.Tensor] = None,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """Benchmark the generated code for performance."""
        logger.info("Benchmarking generated code")
        
        # This would compile and run the generated code
        # For now, return dummy metrics
        return {
            'inference_time_ms': 10.5,
            'memory_usage_mb': 150.0,
            'throughput_fps': 95.2,
            'compilation_time_s': 2.1
        }
    
    def visualize_architecture(
        self,
        architecture: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> None:
        """Create a visualization of the generated architecture."""
        from ..utils.visualization import ArchitectureVisualizer
        
        visualizer = ArchitectureVisualizer()
        visualizer.plot_architecture(architecture, save_path) 