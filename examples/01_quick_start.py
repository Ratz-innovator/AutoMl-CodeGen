#!/usr/bin/env python3
"""
nanoNAS Quick Start Example
==========================

This example demonstrates the basic usage of nanoNAS for neural architecture search.
Run this script to see nanoNAS in action with minimal setup.

Requirements:
    pip install torch torchvision numpy matplotlib

Usage:
    python examples/01_quick_start.py
"""

import sys
import os
import time

# Add parent directory to path to import nanonas
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Please install: pip install torch torchvision numpy matplotlib")
    sys.exit(1)

# Import nanoNAS
try:
    from nanonas import nano_nas  # Educational implementation
    import nanonas  # Professional package
except ImportError as e:
    print(f"❌ Failed to import nanoNAS: {e}")
    print("Make sure you're running from the nanoNAS root directory")
    sys.exit(1)


def demo_educational_implementation():
    """Demonstrate the educational 300-line implementation."""
    print("🎓 Educational Implementation Demo")
    print("=" * 50)
    
    print("1. Quick Evolution Search (5 generations)...")
    start_time = time.time()
    
    # Run evolution with small parameters for demo
    model = nano_nas(
        strategy='evolution',
        population_size=8,
        generations=5,
        verbose=True
    )
    
    evolution_time = time.time() - start_time
    
    # Test the model
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        y = model(x)
    
    print(f"✅ Evolution completed in {evolution_time:.1f}s")
    print(f"   Model output shape: {y.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n2. Quick DARTS Search (3 epochs)...")
    start_time = time.time()
    
    # Run DARTS with minimal epochs for demo
    model = nano_nas(
        strategy='darts',
        epochs=3,
        verbose=True
    )
    
    darts_time = time.time() - start_time
    
    print(f"✅ DARTS completed in {darts_time:.1f}s")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")


def demo_professional_package():
    """Demonstrate the professional nanoNAS package."""
    print("\n🏢 Professional Package Demo")
    print("=" * 50)
    
    # Test search space creation
    print("1. Creating search spaces...")
    nano_space = nanonas.SearchSpace.get_nano_search_space()
    mobile_space = nanonas.SearchSpace.get_mobile_search_space()
    
    print(f"   Nano space: {len(nano_space.operations)} operations")
    print(f"   Mobile space: {len(mobile_space.operations)} operations")
    
    # Test architecture creation
    print("\n2. Creating and analyzing architectures...")
    arch = nanonas.Architecture([0, 1, 2, 3, 4], nano_space)
    complexity = arch.get_complexity_metrics()
    
    print(f"   Architecture: {arch}")
    print(f"   Depth: {complexity['depth']}")
    print(f"   Skip ratio: {complexity['skip_ratio']:.2f}")
    
    # Test model conversion
    print("\n3. Converting to PyTorch model...")
    model = arch.to_model()
    
    # Test inference
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        y = model(x)
    
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test mutation
    print("\n4. Testing architecture mutation...")
    mutated_arch = arch.mutate(mutation_rate=0.5)
    print(f"   Original:  {arch.encoding}")
    print(f"   Mutated:   {mutated_arch.encoding}")


def demo_visualization():
    """Demonstrate architecture visualization capabilities."""
    print("\n📊 Visualization Demo")
    print("=" * 50)
    
    # Create sample architectures for comparison
    search_space = nanonas.SearchSpace.get_nano_search_space()
    
    architectures = []
    metrics = []
    
    print("1. Generating sample architectures...")
    for i in range(5):
        arch = search_space.sample_random_architecture(num_blocks=6)
        complexity = arch.get_complexity_metrics()
        
        architectures.append(arch)
        metrics.append(complexity)
        
        print(f"   Arch {i+1}: depth={complexity['depth']}, "
              f"cost={complexity.get('total_op_cost', 0):.1f}")
    
    # Create comparison plot
    print("\n2. Creating architecture comparison plot...")
    try:
        depths = [m['depth'] for m in metrics]
        costs = [m.get('total_op_cost', 0) for m in metrics]
        
        plt.figure(figsize=(8, 6))
        plt.scatter(depths, costs, s=100, alpha=0.7)
        plt.xlabel('Architecture Depth')
        plt.ylabel('Computational Cost')
        plt.title('nanoNAS Architecture Comparison')
        plt.grid(True, alpha=0.3)
        
        for i, (d, c) in enumerate(zip(depths, costs)):
            plt.annotate(f'Arch {i+1}', (d, c), 
                        xytext=(5, 5), textcoords='offset points')
        
        # Save plot
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/architecture_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("   ✅ Plot saved to: results/architecture_comparison.png")
    except Exception as e:
        print(f"   ⚠️  Plotting failed: {e}")


def demo_hardware_profiling():
    """Demonstrate hardware profiling capabilities."""
    print("\n🖥️  Hardware Profiling Demo")
    print("=" * 50)
    
    try:
        # Test hardware profiling
        profile = nanonas.profile_current_device()
        
        print(f"1. Hardware Detection:")
        print(f"   Device: {profile.device_name}")
        print(f"   Memory: {profile.memory_total} MB")
        print(f"   Compute capability: {getattr(profile, 'compute_capability', 'N/A')}")
        
        # Test architecture efficiency analysis
        search_space = nanonas.SearchSpace.get_mobile_search_space()
        arch = search_space.sample_random_architecture()
        complexity = arch.get_complexity_metrics()
        
        print(f"\n2. Architecture Efficiency:")
        print(f"   Estimated FLOPs: {complexity.get('total_flops', 0):,.0f}")
        print(f"   Estimated Memory: {complexity.get('total_memory', 0):.1f} units")
        print(f"   Efficiency Score: {complexity.get('efficiency_score', 0):.2f}")
        
    except Exception as e:
        print(f"   ⚠️  Hardware profiling failed: {e}")


def demo_configuration():
    """Demonstrate configuration system."""
    print("\n⚙️  Configuration Demo")
    print("=" * 50)
    
    try:
        # Create sample configuration
        config = nanonas.ExperimentConfig(
            name="demo_experiment",
            search=nanonas.SearchConfig(
                strategy='evolutionary',
                population_size=10,
                generations=5
            ),
            training=nanonas.TrainingConfig(
                epochs=10,
                learning_rate=0.01
            )
        )
        
        print("1. Configuration created:")
        print(f"   Experiment: {config.name}")
        print(f"   Strategy: {config.search.strategy}")
        print(f"   Population: {config.search.population_size}")
        print(f"   Training epochs: {config.training.epochs}")
        
        # Save configuration
        config_path = 'results/demo_config.yaml'
        os.makedirs('results', exist_ok=True)
        config.save(config_path)
        
        print(f"\n2. Configuration saved to: {config_path}")
        
        # Load configuration back
        loaded_config = nanonas.ExperimentConfig.load(config_path)
        print(f"   Loaded successfully: {loaded_config.name}")
        
    except Exception as e:
        print(f"   ⚠️  Configuration demo failed: {e}")


def main():
    """Run all demonstrations."""
    print("🚀 nanoNAS Quick Start Demo")
    print("="*50)
    print("This demo showcases nanoNAS capabilities in ~30 seconds")
    print()
    
    start_time = time.time()
    
    try:
        # Run all demos
        demo_educational_implementation()
        demo_professional_package()
        demo_visualization()
        demo_hardware_profiling()
        demo_configuration()
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 Demo Complete!")
        print(f"Total time: {total_time:.1f}s")
        print(f"\nNext steps:")
        print(f"  • Check results/ directory for outputs")
        print(f"  • Run python examples/02_detailed_search.py for advanced usage")
        print(f"  • See examples/03_benchmarking.py for performance comparisons")
        
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Please check your installation and try again")


if __name__ == "__main__":
    main() 