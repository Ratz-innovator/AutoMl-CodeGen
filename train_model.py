#!/usr/bin/env python3
"""
Simple nanoNAS Training Script

This script provides a simple interface for training models found through
neural architecture search.

Usage:
    python train_model.py --strategy evolutionary --dataset cifar10 --epochs 50
    python train_model.py --strategy darts --dataset mnist --epochs 30
    python train_model.py --help
"""

import argparse
import logging
import time
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from nanonas import NanoNAS
from nanonas.utils import (
    setup_logging, get_quick_loaders, Trainer, TrainingConfig,
    set_seed, get_device_info
)
from nanonas.visualization import plot_architecture


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train neural architectures found by nanoNAS',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Search strategy
    parser.add_argument(
        '--strategy', 
        type=str, 
        default='evolutionary',
        choices=['evolutionary', 'random', 'darts'],
        help='Architecture search strategy'
    )
    
    # Dataset
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar10',
        choices=['cifar10', 'cifar100', 'mnist', 'fashion_mnist'],
        help='Dataset to use'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    
    # Search parameters
    parser.add_argument(
        '--population-size',
        type=int,
        default=20,
        help='Population size for evolutionary search'
    )
    
    parser.add_argument(
        '--generations',
        type=int,
        default=10,
        help='Number of generations for evolutionary search'
    )
    
    parser.add_argument(
        '--search-epochs',
        type=int,
        default=10,
        help='Number of epochs for DARTS search'
    )
    
    # Other options
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for training'
    )
    
    parser.add_argument(
        '--save-model',
        action='store_true',
        help='Save the trained model'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create visualizations'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / f"training_{args.strategy}_{args.dataset}.log"
    logger = setup_logging(
        log_level=args.log_level,
        log_file=str(log_file)
    )
    
    # Set random seed
    set_seed(args.seed)
    
    # Print configuration
    print("🚀 nanoNAS Training Script")
    print("=" * 50)
    print(f"Strategy: {args.strategy}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    
    # Get device info
    device_info = get_device_info()
    logger.info(f"Running on: {device_info['platform']}")
    logger.info(f"CUDA available: {device_info['cuda_available']}")
    
    # =========================================================================
    # Data Loading
    # =========================================================================
    
    print("\\n📊 Loading dataset...")
    
    train_loader, val_loader = get_quick_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        data_dir='./data'
    )
    
    logger.info(f"Dataset loaded - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
    
    # =========================================================================
    # Architecture Search
    # =========================================================================
    
    print("\\n🔍 Starting architecture search...")
    search_start_time = time.time()
    
    # Configure search based on strategy
    if args.strategy == 'evolutionary':
        nas_config = {
            'search_strategy': 'evolutionary',
            'population_size': args.population_size,
            'generations': args.generations,
            'dataset': args.dataset
        }
    elif args.strategy == 'random':
        nas_config = {
            'search_strategy': 'random',
            'num_architectures': args.population_size,
            'dataset': args.dataset
        }
    elif args.strategy == 'darts':
        nas_config = {
            'search_strategy': 'darts',
            'epochs': args.search_epochs,
            'dataset': args.dataset
        }
    
    # Create NAS instance and run search
    nas = NanoNAS(**nas_config)
    best_architecture = nas.search()
    
    search_time = time.time() - search_start_time
    
    # Get architecture details
    arch_details = nas.get_architecture_details(best_architecture)
    
    print(f"\\n✅ Search completed in {search_time:.2f} seconds")
    print(f"📊 Architecture details:")
    print(f"  - Fitness: {arch_details['fitness']:.4f}")
    print(f"  - Parameters: {arch_details['total_params']:,}")
    print(f"  - FLOPs: {arch_details['flops']:,}")
    
    logger.info(f"Best architecture: {best_architecture}")
    logger.info(f"Architecture fitness: {arch_details['fitness']:.4f}")
    
    # =========================================================================
    # Model Training
    # =========================================================================
    
    print("\\n🏋️ Training the best architecture...")
    
    # Create model
    model = nas.create_model(best_architecture)
    
    # Training configuration
    training_config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        device=args.device,
        early_stopping_patience=10,
        save_checkpoints=True,
        checkpoint_dir=str(output_dir / 'checkpoints'),
        log_dir=str(output_dir / 'logs')
    )
    
    # Create trainer
    trainer = Trainer(training_config)
    
    # Log model and config
    trainer.result_logger.log_model_info(model)
    trainer.result_logger.log_config(training_config)
    trainer.result_logger.log_system_info()
    
    # Train model
    training_start_time = time.time()
    history = trainer.train(model, train_loader, val_loader)
    training_time = time.time() - training_start_time
    
    print(f"\\n✅ Training completed in {training_time:.2f} seconds")
    print(f"📈 Final results:")
    print(f"  - Final training accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"  - Final validation accuracy: {history['val_acc'][-1]:.2f}%")
    print(f"  - Best validation accuracy: {max(history['val_acc']):.2f}%")
    
    # =========================================================================
    # Save Results
    # =========================================================================
    
    print("\\n💾 Saving results...")
    
    # Save training history
    import json
    results = {
        'search_config': nas_config,
        'training_config': training_config.__dict__,
        'architecture': best_architecture,
        'architecture_details': arch_details,
        'training_history': history,
        'search_time': search_time,
        'training_time': training_time,
        'final_results': {
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1],
            'best_val_acc': max(history['val_acc'])
        }
    }
    
    results_file = output_dir / f"results_{args.strategy}_{args.dataset}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {results_file}")
    
    # Save model if requested
    if args.save_model:
        model_file = output_dir / f"model_{args.strategy}_{args.dataset}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'architecture': best_architecture,
            'config': training_config.__dict__,
            'results': results['final_results']
        }, model_file)
        
        logger.info(f"Model saved to {model_file}")
    
    # =========================================================================
    # Visualization
    # =========================================================================
    
    if args.visualize:
        print("\\n🎨 Creating visualizations...")
        
        try:
            # Training history plot
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Loss plot
            axes[0].plot(history['train_loss'], label='Training Loss', color='blue')
            axes[0].plot(history['val_loss'], label='Validation Loss', color='red')
            axes[0].set_title('Training and Validation Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True)
            
            # Accuracy plot
            axes[1].plot(history['train_acc'], label='Training Accuracy', color='blue')
            axes[1].plot(history['val_acc'], label='Validation Accuracy', color='red')
            axes[1].set_title('Training and Validation Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy (%)')
            axes[1].legend()
            axes[1].grid(True)
            
            plt.tight_layout()
            history_plot = output_dir / f"training_history_{args.strategy}_{args.dataset}.png"
            plt.savefig(history_plot, dpi=300, bbox_inches='tight')
            plt.show()
            
            # Architecture visualization
            plt.figure(figsize=(10, 8))
            plot_architecture(
                best_architecture,
                title=f"{args.strategy.title()} Architecture\\n"
                      f"Dataset: {args.dataset.upper()} | "
                      f"Accuracy: {max(history['val_acc']):.2f}%"
            )
            
            arch_plot = output_dir / f"architecture_{args.strategy}_{args.dataset}.png"
            plt.savefig(arch_plot, dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info(f"Visualizations saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    print("\\n🎉 Training completed successfully!")
    print(f"📁 Results saved in: {output_dir}")
    
    return results


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\\n❌ Interrupted by user")
    except Exception as e:
        print(f"\\n💥 Error: {e}")
        import traceback
        traceback.print_exc() 