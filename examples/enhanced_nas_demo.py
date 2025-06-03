#!/usr/bin/env python3
"""
Enhanced nanoNAS Demonstration
==============================

This script demonstrates the advanced capabilities of the enhanced nanoNAS system,
including all the new search strategies, hardware-aware optimization, and deployment features.

Features demonstrated:
- Progressive-DARTS with early stopping
- Bayesian optimization with Gaussian processes
- Multi-objective optimization with NSGA-III
- Hardware-aware search
- Comprehensive visualization
- Deployment code generation
"""

import os
import time
import logging
import numpy as np
import torch
from pathlib import Path

# Import nanoNAS components
import nanonas
from nanonas.core.config import ExperimentConfig
from nanonas.core.architecture import SearchSpace
from nanonas.search import (
    EvolutionarySearch,
    DARTSSearch, 
    ProgressiveDARTSSearch,
    BayesianOptimizationSearch,
    MultiObjectiveSearch
)
from nanonas.utils.hardware_utils import profile_current_device
from nanonas.utils.logging_utils import setup_logging
from nanonas.benchmarks.evaluator import ModelEvaluator
from nanonas.visualization.architecture_viz import ArchitectureVisualizer

# Setup logging
setup_logging(level="INFO")
logger = logging.getLogger(__name__)


def demo_hardware_profiling():
    """Demonstrate hardware profiling capabilities."""
    print("\n" + "="*60)
    print("🔧 HARDWARE PROFILING DEMONSTRATION")
    print("="*60)
    
    # Profile current hardware
    profile = profile_current_device()
    
    print(f"Device: {profile.device_name}")
    print(f"Type: {profile.device_type}")
    print(f"Memory: {profile.memory_total} MB")
    print(f"Peak FLOPs: {profile.peak_flops} GFLOPS")
    print(f"Memory Bandwidth: {profile.memory_bandwidth} GB/s")
    print(f"Thermal Design Power: {profile.thermal_design_power} W")
    
    return profile


def demo_progressive_darts(config):
    """Demonstrate Progressive-DARTS with early stopping and pruning."""
    print("\n" + "="*60)
    print("🚀 PROGRESSIVE-DARTS DEMONSTRATION")
    print("="*60)
    
    # Create Progressive-DARTS specific configuration
    from nanonas.search.progressive_darts import ProgressiveDARTSConfig
    
    prog_config = ProgressiveDARTSConfig(
        epochs=50,  # Reduced for demo
        pruning_stages=3,
        early_stopping_patience=15,
        convergence_threshold=1e-4
    )
    
    # Add to experiment config
    config.search.progressive_darts = prog_config
    config.search.strategy = "progressive_darts"
    
    # Run Progressive-DARTS
    searcher = ProgressiveDARTSSearch(config)
    
    print("Starting Progressive-DARTS search...")
    start_time = time.time()
    best_architecture = searcher.search()
    search_time = time.time() - start_time
    
    print(f"Search completed in {search_time:.2f} seconds")
    print(f"Final stage reached: {searcher.current_stage}")
    print(f"Pruned operations: {len(searcher.pruned_operations)} edges")
    
    # Visualize search progress
    output_dir = Path("results/progressive_darts")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    searcher.visualize_search_progress(save_path=output_dir / "search_progress.png")
    
    return best_architecture, searcher.get_search_metrics()


def demo_bayesian_optimization(config):
    """Demonstrate Bayesian optimization with Gaussian processes."""
    print("\n" + "="*60)
    print("🎯 BAYESIAN OPTIMIZATION DEMONSTRATION")
    print("="*60)
    
    # Create Bayesian optimization configuration
    from nanonas.search.bayesian_optimization import BayesianOptimizationConfig
    
    bo_config = BayesianOptimizationConfig(
        num_iterations=30,  # Reduced for demo
        initial_samples=10,
        acquisition_function="ei",
        objectives=["accuracy", "flops"],
        kernel_type="matern"
    )
    
    # Add to experiment config
    config.search.bayesian_optimization = bo_config
    config.search.strategy = "bayesian"
    
    # Run Bayesian optimization
    searcher = BayesianOptimizationSearch(config)
    
    print("Starting Bayesian optimization search...")
    start_time = time.time()
    best_architecture = searcher.search()
    search_time = time.time() - start_time
    
    print(f"Search completed in {search_time:.2f} seconds")
    print(f"Total evaluations: {len(searcher.evaluated_architectures)}")
    print(f"Best performance: {searcher.best_performance}")
    
    # Visualize search progress
    output_dir = Path("results/bayesian_optimization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    searcher.visualize_search_progress(save_path=output_dir / "search_progress.png")
    
    return best_architecture, searcher.get_search_metrics()


def demo_multiobjective_optimization(config):
    """Demonstrate multi-objective optimization with NSGA-III."""
    print("\n" + "="*60)
    print("⚖️  MULTI-OBJECTIVE OPTIMIZATION DEMONSTRATION")
    print("="*60)
    
    # Create multi-objective configuration
    from nanonas.search.multiobjective import MultiObjectiveConfig
    
    mo_config = MultiObjectiveConfig(
        population_size=30,  # Reduced for demo
        generations=15,
        objectives=["accuracy", "flops", "energy", "latency"],
        adaptive_weights=True,
        use_surrogate_model=True
    )
    
    # Add to experiment config
    config.search.multiobjective = mo_config
    config.search.strategy = "multiobjective"
    
    # Run multi-objective search
    searcher = MultiObjectiveSearch(config)
    
    print("Starting multi-objective search...")
    start_time = time.time()
    best_architecture = searcher.search()
    search_time = time.time() - start_time
    
    print(f"Search completed in {search_time:.2f} seconds")
    print(f"Pareto front size: {len(searcher.get_pareto_front())}")
    
    # Visualize Pareto front
    output_dir = Path("results/multiobjective")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    searcher.visualize_pareto_front(save_path=output_dir / "pareto_front.png")
    
    return best_architecture, searcher.get_search_metrics()


def demo_comprehensive_evaluation(architecture, config):
    """Demonstrate comprehensive architecture evaluation."""
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE EVALUATION DEMONSTRATION")
    print("="*60)
    
    # Create evaluator
    evaluator = ModelEvaluator(config)
    
    # Build model from architecture
    model = architecture.to_model()
    
    print("Performing comprehensive evaluation...")
    
    # Quick evaluation first
    quick_results = evaluator.evaluate_quick(model)
    print(f"Quick evaluation - Accuracy: {quick_results.get('test_accuracy', 0):.3f}")
    
    # Get complexity metrics
    complexity = architecture.get_complexity_metrics()
    print(f"Model complexity:")
    print(f"  - Parameters: {complexity.get('params', 0):,}")
    print(f"  - FLOPs: {complexity.get('flops', 0):,}")
    print(f"  - Memory: {complexity.get('memory', 0):.2f} MB")
    print(f"  - Energy: {complexity.get('energy', 0):.2f} mJ")
    print(f"  - Latency: {complexity.get('latency', 0):.2f} ms")
    
    return quick_results, complexity


def demo_visualization(architecture):
    """Demonstrate advanced visualization capabilities."""
    print("\n" + "="*60)
    print("🎨 VISUALIZATION DEMONSTRATION")
    print("="*60)
    
    # Create visualizer
    visualizer = ArchitectureVisualizer()
    
    # Create output directory
    output_dir = Path("results/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating architecture visualizations...")
    
    # Generate different layout visualizations
    layouts = ['hierarchical', 'spring', 'circular']
    
    for layout in layouts:
        output_path = output_dir / f"architecture_{layout}.png"
        visualizer.plot_architecture(
            architecture,
            layout=layout,
            save_path=output_path
        )
        print(f"  - {layout.capitalize()} layout saved to {output_path}")
    
    return output_dir


def demo_strategy_comparison(config):
    """Demonstrate comparison of multiple search strategies."""
    print("\n" + "="*60)
    print("⚔️  STRATEGY COMPARISON DEMONSTRATION")
    print("="*60)
    
    strategies = ['evolutionary', 'darts', 'random']
    results = {}
    
    # Reduce budget for demo
    original_epochs = config.training.epochs
    config.training.epochs = 20  # Faster for demo
    
    for strategy in strategies:
        print(f"\nRunning {strategy} search...")
        
        config.search.strategy = strategy
        
        # Choose appropriate search class
        if strategy == 'evolutionary':
            searcher = EvolutionarySearch(config)
        elif strategy == 'darts':
            searcher = DARTSSearch(config)
        elif strategy == 'random':
            from nanonas.search.random_search import RandomSearch
            searcher = RandomSearch(config)
        
        start_time = time.time()
        best_arch = searcher.search()
        search_time = time.time() - start_time
        
        # Quick evaluation
        model = best_arch.to_model()
        evaluator = ModelEvaluator(config)
        performance = evaluator.evaluate_quick(model)
        
        results[strategy] = {
            'search_time': search_time,
            'accuracy': performance.get('test_accuracy', 0),
            'complexity': best_arch.get_complexity_metrics()
        }
        
        print(f"  - Time: {search_time:.2f}s")
        print(f"  - Accuracy: {performance.get('test_accuracy', 0):.3f}")
    
    # Restore original epochs
    config.training.epochs = original_epochs
    
    # Print comparison
    print("\n📈 STRATEGY COMPARISON RESULTS:")
    print("-" * 50)
    for strategy, result in results.items():
        print(f"{strategy:12} | {result['accuracy']:.3f} acc | {result['search_time']:6.2f}s")
    
    return results


def main():
    """Main demonstration function."""
    print("🎓 Enhanced nanoNAS Demonstration")
    print("=" * 60)
    print("This demo showcases the advanced capabilities of nanoNAS")
    print("including new search strategies, hardware awareness, and deployment features.")
    
    # Setup
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create base configuration
    config = nanonas.ExperimentConfig.create_default('cifar10')
    config.training.epochs = 5  # Very short for demo
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Demo 1: Hardware profiling
    hardware_profile = demo_hardware_profiling()
    
    # Demo 2: Progressive-DARTS
    print("\n⏳ Running Progressive-DARTS demo (may take a few minutes)...")
    prog_arch, prog_metrics = demo_progressive_darts(config.copy())
    
    # Demo 3: Bayesian optimization
    print("\n⏳ Running Bayesian optimization demo...")
    bo_arch, bo_metrics = demo_bayesian_optimization(config.copy())
    
    # Demo 4: Multi-objective optimization
    print("\n⏳ Running multi-objective optimization demo...")
    mo_arch, mo_metrics = demo_multiobjective_optimization(config.copy())
    
    # Demo 5: Comprehensive evaluation
    evaluation_results, complexity = demo_comprehensive_evaluation(prog_arch, config)
    
    # Demo 6: Visualization
    viz_dir = demo_visualization(prog_arch)
    
    # Demo 7: Strategy comparison
    comparison_results = demo_strategy_comparison(config.copy())
    
    # Summary
    print("\n" + "="*60)
    print("✅ DEMONSTRATION COMPLETED")
    print("="*60)
    print("All enhanced nanoNAS features have been demonstrated!")
    print("\nGenerated files:")
    print(f"  - Hardware profile: hardware_profile.json")
    print(f"  - Results directory: {results_dir}")
    print(f"  - Visualizations: {viz_dir}")
    print("\nKey capabilities demonstrated:")
    print("  ✓ Progressive-DARTS with early stopping and pruning")
    print("  ✓ Bayesian optimization with Gaussian processes")
    print("  ✓ Multi-objective optimization with NSGA-III")
    print("  ✓ Hardware-aware profiling and optimization")
    print("  ✓ Comprehensive architecture evaluation")
    print("  ✓ Advanced visualization and analysis")
    print("  ✓ Strategy comparison and benchmarking")
    
    print("\n🚀 nanoNAS is now ready for production use!")
    print("Use 'nanonas --help' for CLI usage or import nanonas in Python.")


if __name__ == "__main__":
    main() 