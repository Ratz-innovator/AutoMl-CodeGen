#!/usr/bin/env python3
"""
AutoML-CodeGen Code Generation Script

This script generates production-ready code from discovered neural architectures.
It supports multiple frameworks, optimization levels, and deployment targets.

Usage Examples:
    # Generate PyTorch code from architecture
    python generate.py --architecture results/best_arch.json --framework pytorch

    # Generate optimized TensorFlow code for mobile
    python generate.py --architecture results/best_arch.json \
                       --framework tensorflow \
                       --device mobile \
                       --optimizations quantization pruning

    # Generate deployment package
    python generate.py --architecture results/best_arch.json \
                       --framework pytorch \
                       --include-deployment \
                       --output generated_models/
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from automl_codegen import (
    CodeGenerator,
    Config,
    get_version
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_architecture(architecture_path: str) -> Dict[str, Any]:
    """Load architecture from file."""
    arch_path = Path(architecture_path)
    
    if not arch_path.exists():
        raise FileNotFoundError(f"Architecture file not found: {architecture_path}")
    
    try:
        if arch_path.suffix == '.json':
            with open(arch_path, 'r') as f:
                architecture = json.load(f)
        elif arch_path.suffix == '.pkl':
            import pickle
            with open(arch_path, 'rb') as f:
                data = pickle.load(f)
                # Extract architecture from search results if needed
                if hasattr(data, 'best_architecture'):
                    architecture = data.best_architecture
                elif isinstance(data, dict) and 'best_architecture' in data:
                    architecture = data['best_architecture']
                else:
                    architecture = data
        else:
            raise ValueError(f"Unsupported file format: {arch_path.suffix}")
        
        logger.info(f"Loaded architecture from {architecture_path}")
        return architecture
        
    except Exception as e:
        raise RuntimeError(f"Failed to load architecture: {e}")

def create_sample_architecture() -> Dict[str, Any]:
    """Create a sample architecture for demonstration."""
    return {
        'task': 'image_classification',
        'input_shape': [3, 224, 224],
        'num_classes': 10,
        'layers': [
            {
                'type': 'conv2d',
                'out_channels': 32,
                'kernel_size': 3,
                'stride': 1,
                'padding': 1
            },
            {
                'type': 'batchnorm',
                'num_features': 32
            },
            {
                'type': 'relu'
            },
            {
                'type': 'conv2d',
                'out_channels': 64,
                'kernel_size': 3,
                'stride': 2,
                'padding': 1
            },
            {
                'type': 'batchnorm',
                'num_features': 64
            },
            {
                'type': 'relu'
            },
            {
                'type': 'adaptive_avg_pool2d',
                'output_size': [1, 1]
            },
            {
                'type': 'flatten'
            },
            {
                'type': 'linear',
                'out_features': 128
            },
            {
                'type': 'relu'
            },
            {
                'type': 'dropout',
                'p': 0.5
            },
            {
                'type': 'linear',
                'out_features': 10
            }
        ],
        'connections': 'sequential',
        'metadata': {
            'num_parameters': 123456,
            'flops': 987654,
            'accuracy': 0.95,
            'latency': 15.2,
            'memory': 45.6
        }
    }

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AutoML-CodeGen: Generate production code from neural architectures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output
    parser.add_argument(
        '--architecture', '-a',
        type=str,
        help='Path to architecture file (JSON or pickle)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./generated_code',
        help='Output directory for generated code'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='generated',
        help='Prefix for generated files'
    )
    
    # Framework options
    parser.add_argument(
        '--framework', '-f',
        type=str,
        default='pytorch',
        choices=['pytorch', 'tensorflow', 'onnx', 'tensorrt', 'jax'],
        help='Target framework'
    )
    
    parser.add_argument(
        '--optimization-level',
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help='Code optimization level'
    )
    
    parser.add_argument(
        '--optimizations',
        nargs='+',
        choices=[
            'quantization', 'pruning', 'fusion', 'compilation',
            'memory_optimization', 'dead_code_elimination'
        ],
        help='Specific optimizations to apply'
    )
    
    # Code generation options
    parser.add_argument(
        '--include-training',
        action='store_true',
        default=True,
        help='Include training code'
    )
    
    parser.add_argument(
        '--include-inference',
        action='store_true',
        default=True,
        help='Include inference code'
    )
    
    parser.add_argument(
        '--include-deployment',
        action='store_true',
        help='Include deployment code'
    )
    
    parser.add_argument(
        '--code-style',
        type=str,
        default='black',
        choices=['black', 'autopep8'],
        help='Code formatting style'
    )
    
    # Hardware targeting
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['cpu', 'gpu', 'mobile', 'edge', 'auto'],
        help='Target device'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Default batch size for generated code'
    )
    
    parser.add_argument(
        '--input-shape',
        nargs='+',
        type=int,
        help='Input tensor shape (e.g., 3 224 224)'
    )
    
    # Additional options
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Benchmark generated code'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create architecture visualization'
    )
    
    parser.add_argument(
        '--sample-architecture',
        action='store_true',
        help='Generate code for a sample architecture (for demo)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Configuration file for code generation'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version=f'AutoML-CodeGen {get_version()}'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load or create architecture
        if args.sample_architecture:
            logger.info("Using sample architecture for demonstration")
            architecture = create_sample_architecture()
        elif args.architecture:
            architecture = load_architecture(args.architecture)
        else:
            parser.error("Either --architecture or --sample-architecture must be specified")
        
        # Load configuration
        if args.config:
            config = Config(args.config)
        else:
            config = Config()
        
        # Override config with command line arguments
        config.codegen.target_framework = args.framework
        config.codegen.optimization_level = args.optimization_level
        config.codegen.include_training = args.include_training
        config.codegen.include_inference = args.include_inference
        config.codegen.include_deployment = args.include_deployment
        config.codegen.code_style = args.code_style
        config.hardware.target_device = args.device
        
        if args.optimizations:
            config.codegen.optimizations = args.optimizations
        
        # Initialize code generator
        logger.info(f"Initializing code generator for {args.framework}")
        codegen = CodeGenerator(
            target_framework=args.framework,
            optimization_level=args.optimization_level,
            code_style=args.code_style,
            config=config
        )
        
        # Prepare generation parameters
        gen_params = {
            'architecture': architecture,
            'include_training': args.include_training,
            'include_inference': args.include_inference,
            'include_deployment': args.include_deployment,
            'target_device': args.device,
            'batch_size': args.batch_size
        }
        
        if args.optimizations:
            gen_params['optimizations'] = args.optimizations
        
        if args.input_shape:
            gen_params['input_shape'] = tuple(args.input_shape)
        
        # Generate code
        logger.info("Generating production code...")
        generated_code = codegen.generate(**gen_params)
        
        # Save generated code
        output_dir = Path(args.output)
        generated_code.save(output_dir, prefix=args.prefix)
        
        # Print generation summary
        print("\n" + "=" * 60)
        print("CODE GENERATION SUMMARY")
        print("=" * 60)
        print(f"Framework: {args.framework}")
        print(f"Optimization Level: {args.optimization_level}")
        print(f"Target Device: {args.device}")
        print(f"Output Directory: {output_dir.absolute()}")
        
        if generated_code.metadata:
            metadata = generated_code.metadata
            print(f"Estimated Parameters: {metadata.get('num_parameters', 'N/A'):,}")
            print(f"Architecture Layers: {metadata.get('num_layers', 'N/A')}")
            if 'optimizations_applied' in metadata:
                print(f"Optimizations: {', '.join(metadata['optimizations_applied'])}")
        
        files_generated = []
        if generated_code.model_code:
            files_generated.append(f"{args.prefix}_model.py")
        if generated_code.training_code:
            files_generated.append(f"{args.prefix}_train.py")
        if generated_code.inference_code:
            files_generated.append(f"{args.prefix}_inference.py")
        if generated_code.deployment_code:
            files_generated.append(f"{args.prefix}_deploy.py")
        if generated_code.requirements:
            files_generated.append("requirements.txt")
        
        print(f"Files Generated: {', '.join(files_generated)}")
        print("=" * 60)
        
        # Benchmark if requested
        if args.benchmark:
            logger.info("Benchmarking generated code...")
            try:
                benchmark_results = codegen.benchmark_generated_code(generated_code)
                
                print("\nBENCHMARK RESULTS")
                print("-" * 30)
                for metric, value in benchmark_results.items():
                    if isinstance(value, float):
                        print(f"{metric}: {value:.2f}")
                    else:
                        print(f"{metric}: {value}")
                
                # Save benchmark results
                import json
                benchmark_file = output_dir / f"{args.prefix}_benchmark.json"
                with open(benchmark_file, 'w') as f:
                    json.dump(benchmark_results, f, indent=2)
                
                print(f"Benchmark results saved to: {benchmark_file}")
                
            except Exception as e:
                logger.warning(f"Benchmarking failed: {e}")
        
        # Create visualization if requested
        if args.visualize:
            try:
                logger.info("Creating architecture visualization...")
                viz_path = output_dir / f"{args.prefix}_architecture.png"
                codegen.visualize_architecture(architecture, str(viz_path))
                print(f"Architecture visualization saved to: {viz_path}")
                
            except Exception as e:
                logger.warning(f"Visualization failed: {e}")
        
        # Show code preview
        if args.verbose and generated_code.model_code:
            print("\nMODEL CODE PREVIEW")
            print("-" * 30)
            lines = generated_code.model_code.split('\n')
            preview_lines = min(20, len(lines))
            for i, line in enumerate(lines[:preview_lines]):
                print(f"{i+1:3d}: {line}")
            if len(lines) > preview_lines:
                print(f"... ({len(lines) - preview_lines} more lines)")
        
        logger.info("Code generation completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Code generation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main()) 