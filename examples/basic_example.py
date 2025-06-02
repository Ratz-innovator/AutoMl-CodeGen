#!/usr/bin/env python3
"""
Basic AutoML-CodeGen Example

This example demonstrates how to use AutoML-CodeGen to:
1. Define a search space
2. Run neural architecture search
3. Generate code for the best architecture

Note: This uses stub implementations for demonstration purposes.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from automl_codegen import (
    NeuralArchitectureSearch,
    CodeGenerator,
    Config,
    SearchSpace,
    MultiObjectiveOptimizer
)

def main():
    """Run basic AutoML-CodeGen example."""
    print("=" * 60)
    print("AutoML-CodeGen Basic Example")
    print("=" * 60)
    
    # 1. Create configuration
    print("\n1. Setting up configuration...")
    config = Config()
    config.search.algorithm = 'evolutionary'
    config.search.population_size = 10  # Small for demo
    config.search.num_generations = 3   # Small for demo
    config.evaluation.max_epochs = 2    # Small for demo
    
    print(f"   - Algorithm: {config.search.algorithm}")
    print(f"   - Population: {config.search.population_size}")
    print(f"   - Generations: {config.search.num_generations}")
    
    # 2. Create search space
    print("\n2. Creating search space...")
    search_space = SearchSpace(
        task='image_classification',
        hardware_target='gpu',
        min_layers=3,
        max_layers=8
    )
    print(f"   - Task: {search_space.task}")
    print(f"   - Hardware target: {search_space.hardware_target}")
    print(f"   - Layer range: {search_space.min_layers}-{search_space.max_layers}")
    
    # 3. Sample a few architectures to show the search space
    print("\n3. Sampling example architectures...")
    for i in range(3):
        arch = search_space.sample_architecture()
        complexity = search_space.estimate_complexity(arch)
        print(f"   Architecture {i+1}:")
        print(f"     - Layers: {len(arch['layers'])}")
        print(f"     - Est. Parameters: {complexity['parameters']:,}")
        print(f"     - Est. FLOPs: {complexity['flops']:,}")
    
    # 4. Initialize NAS
    print("\n4. Initializing Neural Architecture Search...")
    nas = NeuralArchitectureSearch(
        search_space=search_space,
        objectives=['accuracy', 'latency', 'memory'],
        config=config
    )
    print("   - NAS engine initialized")
    print(f"   - Objectives: {nas.objectives}")
    
    # 5. Run search (demo version)
    print("\n5. Running architecture search...")
    print("   Note: Using stub implementations for demonstration")
    
    try:
        results = nas.search(
            dataset='cifar10',
            max_time_hours=0.1  # Very short for demo
        )
        
        best_arch = results['best_architecture']
        best_metrics = results['best_metrics']
        
        print("   Search completed!")
        print(f"   - Best accuracy: {best_metrics.get('accuracy', 'N/A'):.3f}")
        print(f"   - Best latency: {best_metrics.get('latency', 'N/A'):.1f}ms")
        print(f"   - Architecture layers: {len(best_arch['layers'])}")
        
    except Exception as e:
        print(f"   Search failed (expected with stubs): {e}")
        # Use a sample architecture for the rest of the demo
        best_arch = search_space.sample_architecture()
        print("   Using sample architecture for code generation demo...")
    
    # 6. Generate code
    print("\n6. Generating production code...")
    codegen = CodeGenerator(
        target_framework='pytorch',
        optimization_level=2,
        config=config
    )
    
    try:
        # This will use the stub implementation
        generated_code = codegen.generate(
            architecture=best_arch,
            include_training=True,
            include_inference=True
        )
        
        print("   Code generation completed!")
        print(f"   - Framework: pytorch")
        print(f"   - Optimization level: 2")
        print(f"   - Generated files: {len(generated_code.files) if hasattr(generated_code, 'files') else 'Multiple'}")
        
    except Exception as e:
        print(f"   Code generation demonstration: {e}")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("\nNext steps:")
    print("1. Implement the missing modules (see src/automl_codegen/)")
    print("2. Add real datasets and training logic")
    print("3. Implement framework-specific code generators")
    print("4. Add hardware profiling and benchmarking")
    print("=" * 60)

if __name__ == '__main__':
    main() 