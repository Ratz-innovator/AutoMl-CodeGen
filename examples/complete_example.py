#!/usr/bin/env python3
"""
Complete nanoNAS Example - End-to-End Neural Architecture Search

This script demonstrates the complete nanoNAS workflow:
1. Setting up the environment and data
2. Running different search strategies
3. Training found architectures
4. Benchmarking and comparison
5. Visualization and analysis

Run with: python examples/complete_example.py
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from nanonas import NanoNAS
from nanonas.utils import (
    setup_logging, get_quick_loaders, Trainer, TrainingConfig,
    set_seed, get_device_info, ResultLogger
)
from nanonas.visualization import plot_architecture
from nanonas.benchmarks import ModelProfiler


def main():
    """Main function demonstrating complete nanoNAS workflow."""
    
    # =========================================================================
    # Setup and Configuration
    # =========================================================================
    
    print("🚀 nanoNAS Complete Example")
    print("=" * 50)
    
    # Setup logging
    logger = setup_logging(log_level='INFO', log_file='logs/complete_example.log')
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device information
    device_info = get_device_info()
    logger.info(f"Running on: {device_info['platform']}")
    logger.info(f"CUDA available: {device_info['cuda_available']}")
    
    if device_info['cuda_available']:
        logger.info(f"GPU: {device_info['gpus'][0]['name']}")
    
    # =========================================================================
    # Data Preparation
    # =========================================================================
    
    print("\\n📊 Setting up data...")
    
    # Get data loaders
    train_loader, val_loader = get_quick_loaders(
        dataset_name='cifar10',
        batch_size=128,
        data_dir='./data'
    )
    
    logger.info(f"Dataset loaded - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
    
    # =========================================================================
    # Architecture Search Comparison
    # =========================================================================
    
    print("\\n🔍 Running architecture search comparison...")
    
    # Define search strategies to compare
    search_configs = {
        'evolutionary': {
            'search_strategy': 'evolutionary',
            'population_size': 8,
            'generations': 3,
            'dataset': 'cifar10'
        },
        'random': {
            'search_strategy': 'random', 
            'num_architectures': 20,
            'dataset': 'cifar10'
        },
        'darts': {
            'search_strategy': 'darts',
            'epochs': 5,
            'dataset': 'cifar10'
        }
    }
    
    search_results = {}
    
    for strategy_name, config in search_configs.items():
        print(f"\\n  🔬 Running {strategy_name} search...")
        start_time = time.time()
        
        try:
            # Create NAS instance
            nas = NanoNAS(**config)
            
            # Run search
            best_architecture = nas.search()
            
            # Get architecture details
            arch_details = nas.get_architecture_details(best_architecture)
            
            search_time = time.time() - start_time
            
            search_results[strategy_name] = {
                'architecture': best_architecture,
                'fitness': arch_details['fitness'],
                'params': arch_details['total_params'],
                'flops': arch_details['flops'], 
                'search_time': search_time
            }
            
            logger.info(f"{strategy_name} search completed in {search_time:.2f}s")
            logger.info(f"  Fitness: {arch_details['fitness']:.4f}")
            logger.info(f"  Parameters: {arch_details['total_params']:,}")
            
        except Exception as e:
            logger.error(f"Error in {strategy_name} search: {e}")
            search_results[strategy_name] = None
    
    # =========================================================================
    # Model Training and Evaluation
    # =========================================================================
    
    print("\\n🏋️ Training best architectures...")
    
    training_results = {}
    
    for strategy_name, result in search_results.items():
        if result is None:
            continue
            
        print(f"\\n  📈 Training {strategy_name} architecture...")
        
        try:
            # Create model
            nas = NanoNAS(**search_configs[strategy_name])
            model = nas.create_model(result['architecture'])
            
            # Training configuration
            config = TrainingConfig(
                epochs=10,  # Quick training for demo
                learning_rate=0.001,
                batch_size=128,
                device='auto',
                early_stopping_patience=5,
                log_every_n_steps=50
            )
            
            # Create trainer
            trainer = Trainer(config)
            
            # Log model info
            trainer.result_logger.log_model_info(model)
            trainer.result_logger.log_config(config)
            
            # Train model
            start_time = time.time()
            history = trainer.train(model, train_loader, val_loader)
            training_time = time.time() - start_time
            
            # Store results
            training_results[strategy_name] = {
                'history': history,
                'final_train_acc': history['train_acc'][-1],
                'final_val_acc': history['val_acc'][-1],
                'best_val_acc': max(history['val_acc']),
                'training_time': training_time,
                'model': model
            }
            
            logger.info(f"{strategy_name} training completed in {training_time:.2f}s")
            logger.info(f"  Best validation accuracy: {max(history['val_acc']):.2f}%")
            
        except Exception as e:
            logger.error(f"Error training {strategy_name} model: {e}")
            training_results[strategy_name] = None
    
    # =========================================================================
    # Benchmarking and Profiling
    # =========================================================================
    
    print("\\n⚡ Benchmarking models...")
    
    profiler = ModelProfiler()
    benchmark_results = {}
    
    for strategy_name, result in training_results.items():
        if result is None or result['model'] is None:
            continue
            
        print(f"  🔍 Profiling {strategy_name} model...")
        
        try:
            model = result['model']
            
            # Profile model
            profile_results = profiler.profile_model(
                model, 
                input_shape=(3, 32, 32),
                batch_size=128
            )
            
            # Benchmark inference
            inference_results = profiler.benchmark_inference(
                model,
                input_shape=(3, 32, 32),
                batch_size=128,
                num_runs=50
            )
            
            benchmark_results[strategy_name] = {
                'profile': profile_results,
                'inference': inference_results
            }
            
            logger.info(f"{strategy_name} profiling completed")
            logger.info(f"  Inference time: {inference_results['avg_time_ms']:.2f}ms")
            logger.info(f"  Throughput: {inference_results['throughput_samples_per_sec']:.0f} samples/s")
            
        except Exception as e:
            logger.error(f"Error profiling {strategy_name} model: {e}")
            benchmark_results[strategy_name] = None
    
    # =========================================================================
    # Results Analysis and Visualization
    # =========================================================================
    
    print("\\n📊 Analyzing results...")
    
    # Create comprehensive comparison
    create_results_visualization(search_results, training_results, benchmark_results)
    
    # Print summary
    print_summary(search_results, training_results, benchmark_results)
    
    # =========================================================================
    # Architecture Visualization
    # =========================================================================
    
    print("\\n🎨 Creating architecture visualizations...")
    
    try:
        # Visualize best architectures
        fig, axes = plt.subplots(1, len(search_results), figsize=(6*len(search_results), 6))
        if len(search_results) == 1:
            axes = [axes]
        
        for i, (strategy_name, result) in enumerate(search_results.items()):
            if result is not None:
                plot_architecture(
                    result['architecture'], 
                    ax=axes[i],
                    title=f"{strategy_name.title()} Architecture\\n"
                          f"Fitness: {result['fitness']:.3f}"
                )
        
        plt.tight_layout()
        plt.savefig('results/architecture_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
    
    # =========================================================================
    # Save Final Results
    # =========================================================================
    
    print("\\n💾 Saving results...")
    
    # Save comprehensive results
    final_results = {
        'search_results': search_results,
        'training_results': {k: {
            'final_train_acc': v['final_train_acc'],
            'final_val_acc': v['final_val_acc'], 
            'best_val_acc': v['best_val_acc'],
            'training_time': v['training_time']
        } for k, v in training_results.items() if v is not None},
        'benchmark_results': benchmark_results,
        'system_info': device_info
    }
    
    # Save to JSON
    import json
    with open('results/complete_example_results.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    logger.info("Results saved to results/complete_example_results.json")
    
    print("\\n🎉 Complete example finished successfully!")
    print("Check the results/ directory for detailed outputs.")


def create_results_visualization(search_results, training_results, benchmark_results):
    """Create comprehensive results visualization."""
    
    # Prepare data for plotting
    strategies = []
    search_times = []
    training_times = []
    best_accuracies = []
    model_params = []
    inference_times = []
    
    for strategy in search_results.keys():
        if (search_results[strategy] is not None and 
            training_results.get(strategy) is not None):
            
            strategies.append(strategy.title())
            search_times.append(search_results[strategy]['search_time'])
            training_times.append(training_results[strategy]['training_time'])
            best_accuracies.append(training_results[strategy]['best_val_acc'])
            model_params.append(search_results[strategy]['params'])
            
            if benchmark_results.get(strategy) is not None:
                inference_times.append(benchmark_results[strategy]['inference']['avg_time_ms'])
            else:
                inference_times.append(0)
    
    if not strategies:
        print("No data to visualize")
        return
    
    # Create multi-panel visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Search time comparison
    axes[0, 0].bar(strategies, search_times, color='skyblue')
    axes[0, 0].set_title('Search Time Comparison')
    axes[0, 0].set_ylabel('Time (seconds)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Training time comparison  
    axes[0, 1].bar(strategies, training_times, color='lightcoral')
    axes[0, 1].set_title('Training Time Comparison')
    axes[0, 1].set_ylabel('Time (seconds)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Best accuracy comparison
    axes[0, 2].bar(strategies, best_accuracies, color='lightgreen')
    axes[0, 2].set_title('Best Validation Accuracy')
    axes[0, 2].set_ylabel('Accuracy (%)')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Model parameters comparison
    axes[1, 0].bar(strategies, model_params, color='wheat')
    axes[1, 0].set_title('Model Parameters')
    axes[1, 0].set_ylabel('Number of Parameters')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Inference time comparison
    if any(t > 0 for t in inference_times):
        axes[1, 1].bar(strategies, inference_times, color='plum')
        axes[1, 1].set_title('Inference Time')
        axes[1, 1].set_ylabel('Time (ms)')
        axes[1, 1].tick_params(axis='x', rotation=45)
    else:
        axes[1, 1].text(0.5, 0.5, 'No inference\\nbenchmark data', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Inference Time (No Data)')
    
    # Efficiency plot (Accuracy vs Parameters)
    axes[1, 2].scatter(model_params, best_accuracies, s=100, alpha=0.7)
    for i, strategy in enumerate(strategies):
        axes[1, 2].annotate(strategy, (model_params[i], best_accuracies[i]),
                           xytext=(5, 5), textcoords='offset points')
    axes[1, 2].set_title('Efficiency: Accuracy vs Parameters')
    axes[1, 2].set_xlabel('Number of Parameters')
    axes[1, 2].set_ylabel('Best Accuracy (%)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/complete_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_summary(search_results, training_results, benchmark_results):
    """Print comprehensive summary of results."""
    
    print("\\n" + "="*60)
    print("📋 EXPERIMENT SUMMARY")
    print("="*60)
    
    # Search results summary
    print("\\n🔍 SEARCH RESULTS:")
    print("-" * 40)
    for strategy, result in search_results.items():
        if result is not None:
            print(f"{strategy.title():>12}: Fitness={result['fitness']:.4f}, "
                  f"Params={result['params']:,}, Time={result['search_time']:.1f}s")
        else:
            print(f"{strategy.title():>12}: FAILED")
    
    # Training results summary
    print("\\n🏋️ TRAINING RESULTS:")
    print("-" * 40)
    for strategy, result in training_results.items():
        if result is not None:
            print(f"{strategy.title():>12}: Best={result['best_val_acc']:.2f}%, "
                  f"Final={result['final_val_acc']:.2f}%, Time={result['training_time']:.1f}s")
        else:
            print(f"{strategy.title():>12}: FAILED")
    
    # Benchmark results summary
    print("\\n⚡ BENCHMARK RESULTS:")
    print("-" * 40)
    for strategy, result in benchmark_results.items():
        if result is not None:
            inference = result['inference']
            print(f"{strategy.title():>12}: Inference={inference['avg_time_ms']:.2f}ms, "
                  f"Throughput={inference['throughput_samples_per_sec']:.0f} samples/s")
        else:
            print(f"{strategy.title():>12}: No benchmark data")
    
    # Best overall performance
    if training_results:
        best_strategy = max(
            (k for k, v in training_results.items() if v is not None),
            key=lambda k: training_results[k]['best_val_acc']
        )
        best_acc = training_results[best_strategy]['best_val_acc']
        
        print(f"\\n🏆 BEST PERFORMER: {best_strategy.title()} ({best_acc:.2f}% accuracy)")
    
    print("\\n" + "="*60)


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\\n❌ Interrupted by user")
    except Exception as e:
        print(f"\\n💥 Error: {e}")
        import traceback
        traceback.print_exc() 