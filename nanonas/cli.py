"""
Command Line Interface for nanoNAS
==================================

Comprehensive CLI for neural architecture search with support for:
- Multiple search strategies
- Configuration management
- Hardware-aware optimization
- Deployment generation
- Experiment tracking
"""

import click
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import time

from .core.config import ExperimentConfig
from .core.architecture import SearchSpace
from .search import (
    EvolutionarySearch, 
    DARTSSearch, 
    ProgressiveDARTSSearch,
    BayesianOptimizationSearch,
    MultiObjectiveSearch,
    RandomSearch
)
from .utils.logging_utils import setup_logging
from .utils.hardware_utils import profile_current_device
from .benchmarks.evaluator import ModelEvaluator
from .visualization.architecture_viz import ArchitectureVisualizer


logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--log-level', default='INFO', help='Logging level')
@click.option('--gpu', default=None, type=int, help='GPU device ID')
@click.pass_context
def cli(ctx, config, log_level, gpu):
    """nanoNAS: Neural Architecture Search Made Simple"""
    # Setup logging
    setup_logging(level=log_level)
    
    # Setup device
    if gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        device = torch.device(f'cuda:{gpu}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {device}")
    
    # Store context
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['device'] = device
    ctx.obj['log_level'] = log_level


@cli.command()
@click.option('--strategy', '-s', default='evolutionary', 
              type=click.Choice(['evolutionary', 'darts', 'progressive_darts', 'bayesian', 'multiobjective', 'random']),
              help='Search strategy to use')
@click.option('--dataset', '-d', default='cifar10',
              type=click.Choice(['cifar10', 'cifar100', 'mnist', 'fashion_mnist']),
              help='Dataset to search on')
@click.option('--search_space', default='nano',
              type=click.Choice(['nano', 'mobile', 'advanced']),
              help='Search space to use')
@click.option('--budget', '-b', default=100, type=int, help='Search budget (epochs/generations/iterations)')
@click.option('--output', '-o', default='results', help='Output directory')
@click.option('--seed', default=42, type=int, help='Random seed')
@click.option('--objectives', multiple=True, default=['accuracy', 'flops'],
              help='Objectives for multi-objective search')
@click.pass_context
def search(ctx, strategy, dataset, search_space, budget, output, seed, objectives):
    """Run neural architecture search"""
    logger.info(f"Starting architecture search with strategy: {strategy}")
    
    # Set random seed
    torch.manual_seed(seed)
    
    # Create output directory
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load or create configuration
    if ctx.obj['config_path']:
        with open(ctx.obj['config_path']) as f:
            config_dict = yaml.safe_load(f)
        config = ExperimentConfig.from_dict(config_dict)
    else:
        # Create default configuration
        config = _create_default_config(strategy, dataset, search_space, budget, objectives)
    
    # Override with CLI arguments
    config.dataset.name = dataset
    config.training.device = str(ctx.obj['device'])
    
    # Initialize search space
    if search_space == 'nano':
        space = SearchSpace.get_nano_search_space()
    elif search_space == 'mobile':
        space = SearchSpace.get_mobile_search_space()
    else:
        space = SearchSpace.get_advanced_search_space()
    
    # Initialize search strategy
    strategy_class = {
        'evolutionary': EvolutionarySearch,
        'darts': DARTSSearch,
        'progressive_darts': ProgressiveDARTSSearch,
        'bayesian': BayesianOptimizationSearch,
        'multiobjective': MultiObjectiveSearch,
        'random': RandomSearch
    }[strategy]
    
    searcher = strategy_class(config)
    
    # Run search
    start_time = time.time()
    best_architecture = searcher.search()
    search_time = time.time() - start_time
    
    # Evaluate best architecture
    model = best_architecture.to_model()
    evaluator = ModelEvaluator(config)
    results = evaluator.evaluate_full(model)
    
    # Save results
    results_dict = {
        'search_strategy': strategy,
        'search_space': search_space,
        'dataset': dataset,
        'search_time': search_time,
        'best_architecture': best_architecture.to_dict(),
        'performance': results,
        'search_metrics': searcher.get_search_metrics()
    }
    
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    
    # Generate visualizations
    visualizer = ArchitectureVisualizer()
    visualizer.plot_architecture(best_architecture, save_path=output_path / 'architecture.png')
    
    if hasattr(searcher, 'visualize_search_progress'):
        searcher.visualize_search_progress(save_path=output_path / 'search_progress.png')
    
    logger.info(f"Search completed in {search_time:.2f}s")
    logger.info(f"Best architecture accuracy: {results.get('test_accuracy', 0):.3f}")
    logger.info(f"Results saved to: {output_path}")


@cli.command()
@click.argument('results_path', type=click.Path(exists=True))
@click.option('--framework', default='pytorch',
              type=click.Choice(['pytorch', 'tensorflow', 'jax', 'onnx']),
              help='Target framework')
@click.option('--quantization', type=click.Choice(['int8', 'fp16']), help='Quantization level')
@click.option('--serving', is_flag=True, help='Generate serving code')
@click.option('--docker', is_flag=True, help='Generate Docker configuration')
@click.option('--kubernetes', is_flag=True, help='Generate Kubernetes manifests')
@click.option('--output', '-o', default='deployment', help='Output directory')
def deploy(results_path, framework, quantization, serving, docker, kubernetes, output):
    """Generate deployment code from search results"""
    logger.info(f"Generating deployment code for framework: {framework}")
    
    # Load results
    with open(results_path) as f:
        results = json.load(f)
    
    # Reconstruct architecture
    from .core.architecture import Architecture, SearchSpace
    
    search_space_name = results['search_space']
    if search_space_name == 'nano':
        search_space = SearchSpace.get_nano_search_space()
    elif search_space_name == 'mobile':
        search_space = SearchSpace.get_mobile_search_space()
    else:
        search_space = SearchSpace.get_advanced_search_space()
    
    architecture = Architecture.from_dict(results['best_architecture'], search_space)
    
    # Create output directory
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate code based on framework
    if framework == 'pytorch':
        from .codegen.pytorch_generator import PyTorchGenerator
        generator = PyTorchGenerator()
        code = generator.generate_model_code(architecture)
        
        with open(output_path / 'model.py', 'w') as f:
            f.write(code)
        
        if serving:
            serving_code = generator.generate_serving_code(architecture)
            with open(output_path / 'serve.py', 'w') as f:
                f.write(serving_code)
    
    elif framework == 'onnx':
        from .codegen.onnx_generator import ONNXGenerator
        generator = ONNXGenerator()
        onnx_model = generator.convert_to_onnx(architecture)
        generator.save_model(onnx_model, output_path / 'model.onnx')
    
    # Generate deployment configurations
    if docker:
        from .codegen.deployment_generator import DeploymentGenerator
        deployment_gen = DeploymentGenerator()
        dockerfile = deployment_gen.generate_dockerfile(framework, quantization)
        
        with open(output_path / 'Dockerfile', 'w') as f:
            f.write(dockerfile)
    
    if kubernetes:
        from .codegen.deployment_generator import DeploymentGenerator
        deployment_gen = DeploymentGenerator()
        k8s_manifests = deployment_gen.generate_k8s_manifests(framework)
        
        with open(output_path / 'deployment.yaml', 'w') as f:
            f.write(k8s_manifests)
    
    logger.info(f"Deployment code generated in: {output_path}")


@cli.command()
@click.option('--strategy', multiple=True, default=['evolutionary', 'darts', 'random'],
              help='Strategies to benchmark')
@click.option('--dataset', default='cifar10', help='Dataset for benchmarking')
@click.option('--runs', default=3, type=int, help='Number of runs per strategy')
@click.option('--budget', default=50, type=int, help='Budget per run')
@click.option('--output', '-o', default='benchmark_results', help='Output directory')
@click.pass_context
def benchmark(ctx, strategy, dataset, runs, budget, output):
    """Benchmark multiple search strategies"""
    logger.info(f"Benchmarking strategies: {list(strategy)}")
    
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for strat in strategy:
        logger.info(f"Benchmarking {strat}")
        strategy_results = []
        
        for run in range(runs):
            logger.info(f"Run {run + 1}/{runs}")
            
            # Create temporary config
            config = _create_default_config(strat, dataset, 'nano', budget, ['accuracy'])
            config.training.device = str(ctx.obj['device'])
            
            # Run search
            strategy_class = {
                'evolutionary': EvolutionarySearch,
                'darts': DARTSSearch,
                'progressive_darts': ProgressiveDARTSSearch,
                'bayesian': BayesianOptimizationSearch,
                'multiobjective': MultiObjectiveSearch,
                'random': RandomSearch
            }[strat]
            
            searcher = strategy_class(config)
            start_time = time.time()
            best_arch = searcher.search()
            search_time = time.time() - start_time
            
            # Quick evaluation
            model = best_arch.to_model()
            evaluator = ModelEvaluator(config)
            results_run = evaluator.evaluate_quick(model)
            
            strategy_results.append({
                'run': run,
                'search_time': search_time,
                'accuracy': results_run.get('test_accuracy', 0),
                'complexity': best_arch.get_complexity_metrics()
            })
        
        results[strat] = strategy_results
    
    # Save benchmark results
    with open(output_path / 'benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Generate summary
    summary = {}
    for strat, strat_results in results.items():
        accuracies = [r['accuracy'] for r in strat_results]
        times = [r['search_time'] for r in strat_results]
        
        summary[strat] = {
            'mean_accuracy': sum(accuracies) / len(accuracies),
            'std_accuracy': (sum((x - sum(accuracies)/len(accuracies))**2 for x in accuracies) / len(accuracies))**0.5,
            'mean_time': sum(times) / len(times),
            'std_time': (sum((x - sum(times)/len(times))**2 for x in times) / len(times))**0.5
        }
    
    with open(output_path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("Benchmarking completed")
    for strat, stats in summary.items():
        logger.info(f"{strat}: {stats['mean_accuracy']:.3f} ± {stats['std_accuracy']:.3f} accuracy")


@cli.command()
@click.pass_context
def profile_hardware(ctx):
    """Profile current hardware for hardware-aware optimization"""
    logger.info("Profiling hardware...")
    
    profile = profile_current_device()
    
    print("\n🔧 Hardware Profile")
    print("=" * 50)
    print(f"Device: {profile.device_name}")
    print(f"Type: {profile.device_type}")
    print(f"Memory: {profile.memory_total} MB")
    print(f"Compute Capability: {profile.compute_capability}")
    print(f"Core Count: {profile.core_count}")
    print(f"Peak FLOPs: {profile.peak_flops} GFLOPS")
    print(f"Memory Bandwidth: {profile.memory_bandwidth} GB/s")
    print(f"TDP: {profile.thermal_design_power} W")
    
    # Save profile
    profile_dict = profile.to_dict()
    with open('hardware_profile.json', 'w') as f:
        json.dump(profile_dict, f, indent=2)
    
    logger.info("Hardware profile saved to hardware_profile.json")


@cli.command()
@click.argument('architecture_path', type=click.Path(exists=True))
@click.option('--format', default='png', type=click.Choice(['png', 'pdf', 'svg']),
              help='Output format')
@click.option('--layout', default='hierarchical',
              type=click.Choice(['hierarchical', 'spring', 'circular']),
              help='Graph layout')
@click.option('--output', '-o', help='Output file path')
def visualize(architecture_path, format, layout, output):
    """Visualize architecture from results file"""
    logger.info("Generating architecture visualization...")
    
    # Load architecture
    with open(architecture_path) as f:
        results = json.load(f)
    
    from .core.architecture import Architecture, SearchSpace
    
    # Reconstruct search space and architecture
    search_space_name = results.get('search_space', 'nano')
    if search_space_name == 'nano':
        search_space = SearchSpace.get_nano_search_space()
    elif search_space_name == 'mobile':
        search_space = SearchSpace.get_mobile_search_space()
    else:
        search_space = SearchSpace.get_advanced_search_space()
    
    architecture = Architecture.from_dict(results['best_architecture'], search_space)
    
    # Generate visualization
    visualizer = ArchitectureVisualizer()
    
    if output is None:
        output = f"architecture_viz.{format}"
    
    visualizer.plot_architecture(
        architecture, 
        layout=layout,
        save_path=output,
        format=format
    )
    
    logger.info(f"Visualization saved to: {output}")


def _create_default_config(strategy: str, dataset: str, search_space: str, 
                          budget: int, objectives: list) -> ExperimentConfig:
    """Create default configuration for CLI"""
    from .core.config import (
        SearchConfig, DatasetConfig, TrainingConfig, 
        ModelConfig, EvaluationConfig
    )
    
    search_config = SearchConfig(strategy=strategy)
    
    # Set strategy-specific parameters
    if strategy == 'evolutionary':
        search_config.population_size = 50
        search_config.generations = budget
    elif strategy in ['darts', 'progressive_darts']:
        search_config.darts_epochs = budget
    elif strategy == 'bayesian':
        search_config.num_iterations = budget
    elif strategy == 'multiobjective':
        search_config.population_size = 50
        search_config.generations = budget
        search_config.objectives = list(objectives)
    elif strategy == 'random':
        search_config.num_samples = budget
    
    dataset_config = DatasetConfig(name=dataset)
    training_config = TrainingConfig(epochs=200)
    model_config = ModelConfig()
    evaluation_config = EvaluationConfig()
    
    return ExperimentConfig(
        search=search_config,
        dataset=dataset_config,
        training=training_config,
        model=model_config,
        evaluation=evaluation_config
    )


if __name__ == '__main__':
    cli() 