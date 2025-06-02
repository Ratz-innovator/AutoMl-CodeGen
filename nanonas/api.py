"""
Main API Interface for nanoNAS
=============================

This module provides the main user-facing API for neural architecture search.
It offers both simple one-line interfaces and advanced configuration options.

Key Features:
- One-line architecture search: model = search()
- Flexible configuration system
- Automatic benchmarking and comparison
- Visualization integration
- Results logging and experiment tracking

Example Usage:
    >>> import nanonas
    >>> 
    >>> # Simple usage
    >>> model = nanonas.search(strategy='evolutionary', dataset='cifar10')
    >>> 
    >>> # Advanced usage with config
    >>> config = nanonas.SearchConfig(strategy='darts', epochs=100)
    >>> results = nanonas.search(config, return_results=True)
    >>> 
    >>> # Benchmarking
    >>> comparison = nanonas.benchmark(['evolutionary', 'darts'], dataset='cifar10')
"""

import time
import logging
import warnings
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import torch
import torch.nn as nn

from .core.config import ExperimentConfig, SearchConfig, get_default_config, load_config
from .core.architecture import Architecture, SearchSpace


def search(strategy: Union[str, SearchConfig, ExperimentConfig] = "evolutionary",
          dataset: str = "cifar10",
          search_space: str = "nano",
          epochs: Optional[int] = None,
          population_size: Optional[int] = None,
          device: str = "auto",
          output_dir: str = "./results",
          return_results: bool = False,
          verbose: bool = True,
          **kwargs) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """
    Perform neural architecture search with a simple, one-line interface.
    
    Args:
        strategy: Search strategy ('evolutionary', 'darts', 'reinforcement', 'multiobjective') 
                 or SearchConfig/ExperimentConfig object
        dataset: Dataset name ('cifar10', 'cifar100', 'mnist', 'fashion_mnist')
        search_space: Search space ('nano', 'mobile', 'resnet_like')
        epochs: Number of training epochs (overrides config defaults)
        population_size: Population size for evolutionary search
        device: Computing device ('auto', 'cpu', 'cuda', 'mps')
        output_dir: Directory for saving results
        return_results: Whether to return detailed results
        verbose: Whether to print progress information
        **kwargs: Additional arguments passed to search configuration
    
    Returns:
        PyTorch model (if return_results=False) or tuple of (model, results_dict)
    
    Examples:
        >>> # Basic usage
        >>> model = search('evolutionary', 'cifar10')
        >>> 
        >>> # Advanced usage
        >>> model, results = search(
        ...     strategy='darts',
        ...     dataset='cifar10', 
        ...     epochs=100,
        ...     return_results=True
        ... )
        >>> print(f"Best accuracy: {results['best_accuracy']:.3f}")
    """
    
    # Setup logging
    if verbose:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info("🚀 Starting nanoNAS architecture search...")
    
    # Create configuration
    if isinstance(strategy, str):
        # Create config from string strategy
        config = _create_config_from_params(
            strategy, dataset, search_space, epochs, population_size,
            device, output_dir, **kwargs
        )
    elif isinstance(strategy, SearchConfig):
        # Create full experiment config from search config
        config = ExperimentConfig(
            search=strategy,
            device=device,
            output_dir=output_dir
        )
        # Update dataset and other params
        config.dataset.name = dataset
        if epochs is not None:
            config.training.epochs = epochs
    elif isinstance(strategy, ExperimentConfig):
        # Use provided experiment config
        config = strategy
    else:
        raise ValueError(f"Unsupported strategy type: {type(strategy)}")
    
    # Validate configuration
    _validate_search_inputs(config)
    
    if verbose:
        logger.info(f"📊 Configuration: {config.search.strategy} on {config.dataset.name}")
        logger.info(f"🔍 Search space: {config.search.search_space}")
        logger.info(f"💾 Output directory: {config.output_dir}")
    
    # Perform architecture search
    start_time = time.time()
    
    try:
        # Get search strategy
        search_strategy = _get_search_strategy(config)
        
        # Run search
        if verbose:
            logger.info("🔍 Starting architecture search...")
        
        best_architecture = search_strategy.search()
        
        # Convert to model
        model = best_architecture.to_model(
            input_channels=config.model.input_channels,
            num_classes=config.model.num_classes,
            base_channels=config.model.base_channels
        )
        
        search_time = time.time() - start_time
        
        if verbose:
            logger.info(f"✅ Search completed in {search_time:.2f} seconds")
            logger.info(f"🎯 Found architecture: {best_architecture}")
        
        # Prepare results
        results = {
            'best_architecture': best_architecture,
            'search_time': search_time,
            'config': config,
            'model_params': sum(p.numel() for p in model.parameters()),
            'search_strategy': config.search.strategy,
        }
        
        # Add strategy-specific metrics
        if hasattr(search_strategy, 'get_search_metrics'):
            results.update(search_strategy.get_search_metrics())
        
        # Save results if requested
        if config.output_dir:
            _save_search_results(results, config.output_dir)
        
        if return_results:
            return model, results
        else:
            return model
            
    except Exception as e:
        if verbose:
            logger.error(f"❌ Search failed: {str(e)}")
        raise


def benchmark(strategies: List[str],
             dataset: str = "cifar10",
             search_space: str = "nano",
             num_runs: int = 3,
             epochs: int = 50,
             output_dir: str = "./benchmark_results",
             include_baselines: bool = True,
             verbose: bool = True) -> Dict[str, Any]:
    """
    Benchmark multiple search strategies and compare their performance.
    
    Args:
        strategies: List of search strategies to compare
        dataset: Dataset for evaluation
        search_space: Search space to use
        num_runs: Number of independent runs per strategy
        epochs: Training epochs for each run
        output_dir: Directory for saving benchmark results
        include_baselines: Whether to include baseline models (ResNet, MobileNet)
        verbose: Whether to print progress
    
    Returns:
        Dictionary containing benchmark results and statistics
    
    Examples:
        >>> results = benchmark(['evolutionary', 'darts', 'random'])
        >>> print(results['summary_table'])
    """
    
    if verbose:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info("🔬 Starting nanoNAS benchmark suite...")
        
    benchmark_results = {}
    
    # Run each strategy multiple times
    for strategy in strategies:
        if verbose:
            logger.info(f"📊 Benchmarking {strategy}...")
            
        strategy_results = []
        
        for run in range(num_runs):
            if verbose:
                logger.info(f"  Run {run + 1}/{num_runs}")
            
            try:
                # Create config for this run
                config = _create_config_from_params(
                    strategy, dataset, search_space, epochs,
                    output_dir=f"{output_dir}/{strategy}/run_{run}"
                )
                
                # Run search
                model, results = search(config, return_results=True, verbose=False)
                
                # Evaluate model
                evaluator = _get_model_evaluator(config)
                metrics = evaluator.evaluate(model)
                
                run_results = {
                    'run': run,
                    'architecture': results['best_architecture'],
                    'search_time': results['search_time'],
                    'model_params': results['model_params'],
                    **metrics
                }
                
                strategy_results.append(run_results)
                
            except Exception as e:
                if verbose:
                    logger.warning(f"  Run {run + 1} failed: {e}")
                continue
        
        benchmark_results[strategy] = strategy_results
    
    # Include baseline models if requested
    if include_baselines:
        benchmark_results.update(_evaluate_baselines(dataset, epochs, verbose))
    
    # Compute summary statistics
    summary = _compute_benchmark_summary(benchmark_results)
    
    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    _save_benchmark_results(benchmark_results, summary, output_dir)
    
    if verbose:
        logger.info("✅ Benchmark completed!")
        logger.info(f"📊 Results saved to {output_dir}")
        _print_benchmark_summary(summary)
    
    return {
        'results': benchmark_results,
        'summary': summary,
        'summary_table': _format_benchmark_table(summary)
    }


def visualize(architecture: Union[Architecture, nn.Module, str],
             output_path: Optional[str] = None,
             format: str = "png",
             show_operations: bool = True,
             show_connections: bool = True,
             interactive: bool = False) -> Optional[str]:
    """
    Visualize neural network architecture.
    
    Args:
        architecture: Architecture to visualize (Architecture, model, or path to saved arch)
        output_path: Path to save visualization
        format: Output format ('png', 'svg', 'pdf', 'html')
        show_operations: Whether to show operation details
        show_connections: Whether to show skip connections
        interactive: Whether to create interactive visualization
    
    Returns:
        Path to saved visualization file (if output_path provided)
    
    Examples:
        >>> model = search('evolutionary')
        >>> visualize(model, 'architecture.png')
        >>> 
        >>> # Interactive visualization
        >>> visualize(model, 'architecture.html', interactive=True)
    """
    
    from .visualization.architecture_viz import ArchitectureVisualizer
    
    # Convert inputs to Architecture object
    if isinstance(architecture, nn.Module):
        # Extract architecture from model (this would need model introspection)
        arch = _extract_architecture_from_model(architecture)
    elif isinstance(architecture, str):
        # Load architecture from file
        arch = _load_architecture_from_file(architecture)
    elif isinstance(architecture, Architecture):
        arch = architecture
    else:
        raise ValueError(f"Unsupported architecture type: {type(architecture)}")
    
    # Create visualizer
    visualizer = ArchitectureVisualizer()
    
    # Generate visualization
    if interactive:
        viz_path = visualizer.create_interactive_visualization(
            arch, output_path, show_operations, show_connections
        )
    else:
        viz_path = visualizer.create_static_visualization(
            arch, output_path, format, show_operations, show_connections
        )
    
    return viz_path


# Helper functions

def _create_config_from_params(strategy: str, dataset: str, search_space: str,
                              epochs: Optional[int] = None,
                              population_size: Optional[int] = None,
                              device: str = "auto",
                              output_dir: str = "./results",
                              **kwargs) -> ExperimentConfig:
    """Create experiment configuration from parameters."""
    
    # Start with default config based on search space
    if search_space == "nano":
        config = get_default_config("nano")
    elif search_space == "mobile":
        config = get_default_config("mobile")
    else:
        config = get_default_config("nano")
    
    # Update configuration
    config.search.strategy = strategy
    config.search.search_space = search_space
    config.dataset.name = dataset
    config.device = device
    config.output_dir = output_dir
    
    # Override specific parameters
    if epochs is not None:
        if strategy == "darts":
            config.search.darts_epochs = epochs
        else:
            config.search.generations = epochs // 5  # Rough conversion
    
    if population_size is not None:
        config.search.population_size = population_size
    
    # Update model config based on dataset
    if dataset == "cifar100":
        config.model.num_classes = 100
    elif dataset in ["mnist", "fashion_mnist"]:
        config.model.input_channels = 1
        config.model.num_classes = 10
    elif dataset == "imagenet":
        config.model.input_channels = 3
        config.model.num_classes = 1000
    
    # Apply additional kwargs
    for key, value in kwargs.items():
        if hasattr(config.search, key):
            setattr(config.search, key, value)
        elif hasattr(config.training, key):
            setattr(config.training, key, value)
        elif hasattr(config.model, key):
            setattr(config.model, key, value)
    
    return config


def _validate_search_inputs(config: ExperimentConfig):
    """Validate search configuration and inputs."""
    
    # Check if dataset is supported
    supported_datasets = ["cifar10", "cifar100", "mnist", "fashion_mnist"]
    if config.dataset.name not in supported_datasets:
        warnings.warn(f"Dataset {config.dataset.name} may not be fully supported")
    
    # Check if search strategy is valid
    if config.search.strategy not in ["evolutionary", "darts", "reinforcement", "multiobjective", "random"]:
        raise ValueError(f"Unknown search strategy: {config.search.strategy}")
    
    # Check device availability
    if config.device == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA requested but not available, falling back to CPU")
        config.device = "cpu"


def _get_search_strategy(config: ExperimentConfig):
    """Get the appropriate search strategy instance."""
    
    if config.search.strategy == "evolutionary":
        from .search.evolutionary import EvolutionarySearch
        return EvolutionarySearch(config)
    
    elif config.search.strategy == "darts":
        from .search.darts import DARTSSearch
        return DARTSSearch(config)
    
    elif config.search.strategy == "reinforcement":
        from .search.reinforcement import ReinforcementSearch
        return ReinforcementSearch(config)
    
    elif config.search.strategy == "multiobjective":
        from .search.multiobjective import MultiObjectiveSearch
        return MultiObjectiveSearch(config)
    
    elif config.search.strategy == "random":
        from .search.random_search import RandomSearch
        return RandomSearch(config)
    
    else:
        raise ValueError(f"Unknown search strategy: {config.search.strategy}")


def _get_model_evaluator(config: ExperimentConfig):
    """Get model evaluator for benchmarking."""
    from .benchmarks.evaluator import ModelEvaluator
    return ModelEvaluator(config)


def _save_search_results(results: Dict[str, Any], output_dir: str):
    """Save search results to disk."""
    import json
    import pickle
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save JSON summary
    json_results = {
        'search_time': results['search_time'],
        'model_params': results['model_params'],
        'search_strategy': results['search_strategy'],
        'architecture_encoding': results['best_architecture'].encoding if results['best_architecture'].encoding else None
    }
    
    with open(output_path / "results.json", "w") as f:
        json.dump(json_results, f, indent=2)
    
    # Save full results with pickle
    with open(output_path / "results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # Save configuration
    results['config'].save(output_path / "config.yaml")


def _evaluate_baselines(dataset: str, epochs: int, verbose: bool) -> Dict[str, List[Dict[str, Any]]]:
    """Evaluate baseline models for comparison."""
    
    baselines = {}
    
    # This would implement evaluation of standard architectures
    # For now, return placeholder results
    if verbose:
        logger = logging.getLogger(__name__)
        logger.info("📏 Evaluating baseline models...")
    
    # Placeholder baseline results
    baselines['resnet18'] = [{
        'run': 0,
        'architecture': 'ResNet-18',
        'search_time': 0.0,
        'model_params': 11173962,  # Actual ResNet-18 params
        'accuracy': 0.85,  # Placeholder
        'latency': 0.001,  # Placeholder
    }]
    
    baselines['mobilenet_v2'] = [{
        'run': 0,
        'architecture': 'MobileNet-V2',
        'search_time': 0.0,
        'model_params': 3504872,  # Actual MobileNet-V2 params
        'accuracy': 0.82,  # Placeholder
        'latency': 0.0005,  # Placeholder
    }]
    
    return baselines


def _compute_benchmark_summary(results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Compute summary statistics from benchmark results."""
    import numpy as np
    
    summary = {}
    
    for strategy, runs in results.items():
        if not runs:  # Skip empty results
            continue
            
        # Extract metrics
        accuracies = [run.get('accuracy', 0) for run in runs]
        search_times = [run.get('search_time', 0) for run in runs]
        params = [run.get('model_params', 0) for run in runs]
        
        summary[strategy] = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_search_time': np.mean(search_times),
            'std_search_time': np.std(search_times),
            'mean_params': np.mean(params),
            'std_params': np.std(params),
            'num_runs': len(runs)
        }
    
    return summary


def _save_benchmark_results(results: Dict, summary: Dict, output_dir: str):
    """Save benchmark results to disk."""
    import json
    import pickle
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save summary as JSON
    with open(output_path / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save full results as pickle
    with open(output_path / "benchmark_results.pkl", "wb") as f:
        pickle.dump(results, f)


def _print_benchmark_summary(summary: Dict[str, Any]):
    """Print benchmark summary to console."""
    logger = logging.getLogger(__name__)
    
    logger.info("\n📊 Benchmark Summary:")
    logger.info("-" * 80)
    logger.info(f"{'Strategy':<15} {'Accuracy':<12} {'Search Time':<12} {'Parameters':<12}")
    logger.info("-" * 80)
    
    for strategy, stats in summary.items():
        acc = f"{stats['mean_accuracy']:.3f}±{stats['std_accuracy']:.3f}"
        time = f"{stats['mean_search_time']:.1f}±{stats['std_search_time']:.1f}s"
        params = f"{stats['mean_params']:.0f}"
        logger.info(f"{strategy:<15} {acc:<12} {time:<12} {params:<12}")


def _format_benchmark_table(summary: Dict[str, Any]) -> str:
    """Format benchmark results as a table string."""
    
    table = "| Strategy | Accuracy | Search Time (s) | Parameters |\n"
    table += "|----------|----------|-----------------|------------|\n"
    
    for strategy, stats in summary.items():
        acc = f"{stats['mean_accuracy']:.3f}±{stats['std_accuracy']:.3f}"
        time = f"{stats['mean_search_time']:.1f}±{stats['std_search_time']:.1f}"
        params = f"{stats['mean_params']:.0f}"
        table += f"| {strategy} | {acc} | {time} | {params} |\n"
    
    return table


# Placeholder helper functions (would be implemented in respective modules)

def _extract_architecture_from_model(model: nn.Module) -> Architecture:
    """Extract architecture representation from PyTorch model."""
    # This would require model introspection
    # For now, return a placeholder
    return Architecture(encoding=[0, 1, 2, 3])


def _load_architecture_from_file(filepath: str) -> Architecture:
    """Load architecture from saved file."""
    # This would load from JSON/pickle file
    # For now, return a placeholder
    return Architecture(encoding=[0, 1, 2, 3]) 