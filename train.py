#!/usr/bin/env python3
"""
AutoML-CodeGen Training Script

This script provides a command-line interface for running neural architecture search
and generating production-ready code. It supports various configuration options,
distributed training, and comprehensive logging.

Usage Examples:
    # Basic training
    python train.py --dataset cifar10 --algorithm evolutionary

    # Advanced configuration
    python train.py --config configs/large_experiment.yaml \
                    --population-size 100 \
                    --generations 50 \
                    --distributed \
                    --gpu 4

    # Resume from checkpoint
    python train.py --resume checkpoints/experiment_gen_15.pkl
"""

import argparse
import sys
import os
import logging
import signal
import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import numpy as np

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from automl_codegen import (
    NeuralArchitectureSearch,
    CodeGenerator,
    Config,
    Logger,
    get_version,
    get_build_info
)
from automl_codegen.utils.visualization import SearchVisualizer
from automl_codegen.utils.profiler import PerformanceProfiler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('automl_training.log')
    ]
)
logger = logging.getLogger(__name__)

class TrainingManager:
    """Manages the complete training process including search and code generation."""
    
    def __init__(self, config: Config, args: argparse.Namespace):
        """
        Initialize training manager.
        
        Args:
            config: Configuration object
            args: Command line arguments
        """
        self.config = config
        self.args = args
        self.start_time = time.time()
        
        # Setup experiment directory
        self.experiment_dir = Path(args.output_dir) / f"experiment_{int(time.time())}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.nas = None
        self.codegen = None
        self.profiler = PerformanceProfiler()
        self.visualizer = None
        
        # Training state
        self.best_architecture = None
        self.search_results = None
        self.interrupted = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"Training manager initialized for experiment: {self.experiment_dir}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.interrupted = True
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete training process.
        
        Returns:
            Dictionary containing training results and metrics
        """
        try:
            logger.info("=" * 60)
            logger.info("Starting AutoML-CodeGen Training")
            logger.info("=" * 60)
            
            # Print system information
            build_info = get_build_info()
            logger.info(f"AutoML-CodeGen version: {get_version()}")
            logger.info(f"Python version: {build_info['python']}")
            logger.info(f"PyTorch version: {build_info['pytorch']}")
            logger.info(f"CUDA available: {build_info['cuda_available']}")
            if build_info['cuda_available']:
                logger.info(f"CUDA devices: {build_info['device_count']}")
            
            # Validate configuration
            if not self.config.validate():
                raise ValueError("Configuration validation failed")
            
            logger.info("Configuration Summary:")
            logger.info(self.config.get_summary())
            
            # Initialize wandb if enabled
            if self.config.logging.use_wandb and HAS_WANDB:
                self._setup_wandb()
            
            # Step 1: Initialize NAS
            self._initialize_nas()
            
            # Step 2: Run architecture search
            self._run_search()
            
            # Step 3: Generate code for best architecture
            if not self.interrupted and self.best_architecture:
                self._generate_code()
            
            # Step 4: Create visualizations and reports
            self._create_reports()
            
            # Calculate final metrics
            total_time = time.time() - self.start_time
            results = self._compile_results(total_time)
            
            logger.info("=" * 60)
            logger.info("Training completed successfully!")
            logger.info(f"Total time: {total_time:.2f} seconds")
            logger.info(f"Best architecture accuracy: {results.get('best_accuracy', 'N/A')}")
            logger.info(f"Results saved to: {self.experiment_dir}")
            logger.info("=" * 60)
            
            return results
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            return self._compile_results(time.time() - self.start_time, interrupted=True)
            
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        
        finally:
            # Cleanup
            if self.config.logging.use_wandb:
                wandb.finish()
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        wandb_config = {
            'algorithm': self.config.search.algorithm,
            'population_size': self.config.search.population_size,
            'num_generations': self.config.search.num_generations,
            'mutation_rate': self.config.search.mutation_rate,
            'crossover_rate': self.config.search.crossover_rate,
            'max_epochs': self.config.evaluation.max_epochs,
            'batch_size': self.config.evaluation.batch_size,
            'learning_rate': self.config.evaluation.learning_rate,
            'target_framework': self.config.codegen.target_framework,
            'target_device': self.config.hardware.target_device,
            'dataset': self.config.data.dataset_name
        }
        
        wandb.init(
            project=self.config.logging.wandb_project or "automl-codegen",
            entity=self.config.logging.wandb_entity,
            config=wandb_config,
            name=f"nas_{self.config.search.algorithm}_{int(time.time())}",
            dir=str(self.experiment_dir)
        )
        
        logger.info("Weights & Biases logging initialized")
    
    def _initialize_nas(self):
        """Initialize the Neural Architecture Search system."""
        logger.info("Initializing Neural Architecture Search...")
        
        with self.profiler.profile("nas_initialization"):
            self.nas = NeuralArchitectureSearch(
                task=getattr(self.args, 'task', 'image_classification'),
                dataset=self.config.data.dataset_name,
                objectives=getattr(self.args, 'objectives', ['accuracy', 'latency', 'memory']),
                hardware_target=self.config.hardware.target_device,
                algorithm=self.config.search.algorithm,
                config=self.config,
                seed=getattr(self.args, 'seed', None),
                save_dir=self.experiment_dir
            )
        
        # Add custom callbacks if specified
        if hasattr(self.args, 'callbacks') and self.args.callbacks:
            for callback_name in self.args.callbacks:
                callback = self._create_callback(callback_name)
                if callback:
                    self.nas.add_callback(callback)
        
        logger.info("NAS system initialized successfully")
    
    def _run_search(self):
        """Run the neural architecture search."""
        logger.info("Starting neural architecture search...")
        
        search_params = {
            'max_epochs': self.config.evaluation.max_epochs,
            'population_size': self.config.search.population_size,
            'num_generations': self.config.search.num_generations,
            'early_stopping': self.config.search.early_stopping,
            'patience': self.config.search.patience,
            'save_frequency': self.config.logging.checkpoint_frequency,
            'distributed': getattr(self.args, 'distributed', False),
            'num_workers': getattr(self.args, 'num_workers', None)
        }
        
        # Resume from checkpoint if specified
        if self.args.resume:
            search_params['resume_from'] = self.args.resume
            logger.info(f"Resuming from checkpoint: {self.args.resume}")
        
        with self.profiler.profile("architecture_search"):
            self.search_results = self.nas.search(**search_params)
        
        self.best_architecture = self.search_results.best_architecture
        
        if self.best_architecture:
            logger.info("Architecture search completed successfully")
            logger.info(f"Best architecture found with score: {self.best_architecture.get('aggregate_score', 'N/A')}")
        else:
            logger.warning("No best architecture found")
    
    def _generate_code(self):
        """Generate production-ready code for the best architecture."""
        logger.info("Generating production code...")
        
        # Initialize code generator
        self.codegen = CodeGenerator(
            target_framework=self.config.codegen.target_framework,
            optimization_level=self.config.codegen.optimization_level,
            code_style=self.config.codegen.code_style,
            config=self.config
        )
        
        # Generate code
        with self.profiler.profile("code_generation"):
            generated_code = self.codegen.generate(
                architecture=self.best_architecture,
                include_training=self.config.codegen.include_training,
                include_inference=self.config.codegen.include_inference,
                include_deployment=self.config.codegen.include_deployment,
                optimizations=self.config.codegen.optimizations,
                target_device=self.config.hardware.target_device
            )
        
        # Save generated code
        code_dir = self.experiment_dir / 'generated_code'
        generated_code.save(code_dir, prefix='best_model')
        
        logger.info(f"Code generated and saved to: {code_dir}")
        
        # Benchmark generated code if requested
        if getattr(self.args, 'benchmark', False):
            logger.info("Benchmarking generated code...")
            benchmark_results = self.codegen.benchmark_generated_code(generated_code)
            
            # Save benchmark results
            import json
            with open(self.experiment_dir / 'benchmark_results.json', 'w') as f:
                json.dump(benchmark_results, f, indent=2)
            
            logger.info(f"Benchmark results: {benchmark_results}")
    
    def _create_reports(self):
        """Create visualizations and reports."""
        logger.info("Creating visualizations and reports...")
        
        # Create search visualizations
        if self.search_results:
            self.visualizer = SearchVisualizer(
                self.search_results.search_history,
                self.search_results.pareto_front
            )
            
            viz_dir = self.experiment_dir / 'visualizations'
            viz_dir.mkdir(exist_ok=True)
            
            self.visualizer.plot_convergence(viz_dir / 'convergence.png')
            self.visualizer.plot_pareto_front(viz_dir / 'pareto_front.png')
            self.visualizer.plot_search_progress(viz_dir / 'search_progress.png')
            
            if self.best_architecture:
                self.visualizer.plot_architecture(
                    self.best_architecture,
                    viz_dir / 'best_architecture.png'
                )
        
        # Create performance report
        perf_report = self.profiler.get_report()
        with open(self.experiment_dir / 'performance_report.txt', 'w') as f:
            f.write(perf_report)
        
        # Save configuration used
        self.config.save_to_file(self.experiment_dir / 'config.yaml')
        
        logger.info("Reports created successfully")
    
    def _compile_results(self, total_time: float, interrupted: bool = False) -> Dict[str, Any]:
        """Compile final results."""
        results = {
            'experiment_dir': str(self.experiment_dir),
            'total_time': total_time,
            'interrupted': interrupted,
            'config': self.config.to_dict()
        }
        
        if self.search_results:
            results.update({
                'search_time': self.search_results.search_time,
                'total_evaluations': self.search_results.total_evaluations,
                'convergence_generation': self.search_results.convergence_generation,
                'pareto_front_size': len(self.search_results.pareto_front),
                'final_metrics': self.search_results.final_metrics
            })
            
            if self.best_architecture:
                results.update({
                    'best_accuracy': self.best_architecture.get('accuracy', None),
                    'best_latency': self.best_architecture.get('latency', None),
                    'best_memory': self.best_architecture.get('memory', None),
                    'best_score': self.best_architecture.get('aggregate_score', None)
                })
        
        # Save results
        import json
        with open(self.experiment_dir / 'final_results.json', 'w') as f:
            # Convert numpy types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            def deep_convert(obj):
                if isinstance(obj, dict):
                    return {k: deep_convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [deep_convert(v) for v in obj]
                else:
                    return convert_numpy(obj)
            
            json.dump(deep_convert(results), f, indent=2)
        
        return results
    
    def _create_callback(self, callback_name: str):
        """Create a callback function by name."""
        if callback_name == 'wandb_logger':
            def wandb_callback(nas, stats):
                if self.config.logging.use_wandb:
                    wandb.log(stats)
            return wandb_callback
        
        elif callback_name == 'progress_saver':
            def progress_callback(nas, stats):
                if stats['generation'] % 5 == 0:
                    nas._save_checkpoint(stats['generation'])
            return progress_callback
        
        return None

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AutoML-CodeGen: Neural Architecture Search with Code Generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic arguments
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar10',
        choices=['cifar10', 'cifar100', 'imagenet', 'custom'],
        help='Dataset to use for training'
    )
    
    parser.add_argument(
        '--task',
        type=str,
        default='image_classification',
        choices=['image_classification', 'object_detection', 'semantic_segmentation'],
        help='Task type'
    )
    
    parser.add_argument(
        '--algorithm',
        type=str,
        default='evolutionary',
        choices=['evolutionary', 'darts', 'reinforcement', 'hybrid'],
        help='Search algorithm to use'
    )
    
    # Search parameters
    parser.add_argument(
        '--population-size',
        type=int,
        default=50,
        help='Population size for evolutionary algorithm'
    )
    
    parser.add_argument(
        '--generations',
        type=int,
        default=20,
        help='Number of generations to run'
    )
    
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=50,
        help='Maximum epochs to train each architecture'
    )
    
    parser.add_argument(
        '--objectives',
        nargs='+',
        default=['accuracy', 'latency', 'memory'],
        choices=['accuracy', 'latency', 'memory', 'energy', 'flops'],
        help='Optimization objectives'
    )
    
    # Code generation
    parser.add_argument(
        '--framework',
        type=str,
        default='pytorch',
        choices=['pytorch', 'tensorflow', 'onnx', 'tensorrt'],
        help='Target framework for code generation'
    )
    
    parser.add_argument(
        '--optimization-level',
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help='Code optimization level'
    )
    
    # Hardware settings
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['cpu', 'gpu', 'mobile', 'edge', 'auto'],
        help='Target device'
    )
    
    parser.add_argument(
        '--distributed',
        action='store_true',
        help='Use distributed training'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        help='Number of parallel workers'
    )
    
    # Experiment management
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./experiments',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint to resume from'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )
    
    # Additional options
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Benchmark generated code'
    )
    
    parser.add_argument(
        '--callbacks',
        nargs='+',
        choices=['wandb_logger', 'progress_saver'],
        help='Additional callbacks to use'
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
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load configuration
        if args.config:
            config = Config(args.config)
        else:
            config = Config()
        
        # Override config with command line arguments
        if args.dataset:
            config.data.dataset_name = args.dataset
        if args.algorithm:
            config.search.algorithm = args.algorithm
        if args.population_size:
            config.search.population_size = args.population_size
        if args.generations:
            config.search.num_generations = args.generations
        if args.max_epochs:
            config.evaluation.max_epochs = args.max_epochs
        if args.framework:
            config.codegen.target_framework = args.framework
        if args.optimization_level is not None:
            config.codegen.optimization_level = args.optimization_level
        if args.device:
            config.hardware.target_device = args.device
        
        # Create and run training manager
        trainer = TrainingManager(config, args)
        results = trainer.run()
        
        # Print summary
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Experiment directory: {results['experiment_dir']}")
        print(f"Total time: {results['total_time']:.2f} seconds")
        
        if 'best_accuracy' in results and results['best_accuracy']:
            print(f"Best accuracy: {results['best_accuracy']:.4f}")
        if 'best_latency' in results and results['best_latency']:
            print(f"Best latency: {results['best_latency']:.2f} ms")
        if 'total_evaluations' in results:
            print(f"Total evaluations: {results['total_evaluations']}")
        
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 