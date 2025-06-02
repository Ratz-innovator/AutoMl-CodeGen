#!/usr/bin/env python3
"""
🚀 nanoNAS: Interactive Demo
============================

Experience Neural Architecture Search in action!
Watch algorithms find optimal architectures automatically.
"""

import time
import torch
from nanonas import nano_nas, Architecture, Operation
from nas_insights import ArchitectureDNA, SearchLandscape, ParetoAnalyzer

def print_header():
    """Print beautiful header."""
    print("\n" + "="*60)
    print("🧠  nanoNAS: Neural Architecture Search Made Simple")
    print("="*60)
    print("🎯 Automatic neural network design in <200 lines")
    print("🔬 Educational • Minimal • Powerful")
    print("="*60 + "\n")

def demo_architecture_basics():
    """Demonstrate basic architecture concepts."""
    print("📚 1. Understanding Neural Architectures")
    print("-" * 40)
    
    # Show architecture encoding
    arch = Architecture([0, 1, 2, 3])
    print(f"🧬 Architecture DNA: {arch.encoding}")
    print("   0=conv3x3, 1=conv5x5, 2=maxpool, 3=skip")
    
    # Show model creation
    model = arch.to_model()
    params = sum(p.numel() for p in model.parameters())
    print(f"🏗️  Generated model: {params:,} parameters")
    
    # Test forward pass
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        output = model(x)
    print(f"✅ Forward pass: {x.shape} → {output.shape}")
    print()

def demo_dna_concept():
    """Demonstrate the DNA metaphor."""
    print("🧬 2. Architecture DNA Evolution")
    print("-" * 40)
    
    dna_analyzer = ArchitectureDNA()
    
    # Show DNA encoding
    arch1 = Architecture([0, 1, 2, 3])
    arch2 = Architecture([3, 2, 1, 0])
    
    dna1 = dna_analyzer.encode_architecture(arch1)
    dna2 = dna_analyzer.encode_architecture(arch2)
    
    print(f"🧬 Architecture 1 DNA: {dna1}")
    print(f"🧬 Architecture 2 DNA: {dna2}")
    
    # Show mutation
    mutated_dna = dna_analyzer.mutate_dna(dna1, mutation_rate=0.5)
    print(f"🔄 After mutation:     {mutated_dna}")
    print("   → Just like biological evolution!")
    print()

def demo_evolution_search():
    """Demonstrate evolutionary search."""
    print("🧬 3. Evolutionary Search in Action")
    print("-" * 40)
    print("🔍 Searching for optimal architecture...")
    
    start_time = time.time()
    model = nano_nas('evolution', population_size=10, generations=4)
    search_time = time.time() - start_time
    
    params = sum(p.numel() for p in model.parameters())
    print(f"⏱️  Search completed in {search_time:.2f} seconds")
    print(f"🎯 Found model with {params:,} parameters")
    print("✨ Ready to train with standard PyTorch!")
    print()

def demo_darts_search():
    """Demonstrate DARTS search."""
    print("📈 4. DARTS: Gradient-Based Search")
    print("-" * 40)
    print("🔍 Using gradients to optimize architecture...")
    
    start_time = time.time()
    model = nano_nas('darts', epochs=6)
    search_time = time.time() - start_time
    
    params = sum(p.numel() for p in model.parameters())
    print(f"⏱️  Search completed in {search_time:.2f} seconds")
    print(f"🎯 Found model with {params:,} parameters")
    print("🚀 DARTS converges faster than evolution!")
    print()

def demo_insights():
    """Demonstrate educational insights."""
    print("🎓 5. Educational Insights Available")
    print("-" * 40)
    print("📊 Run these commands to explore:")
    print()
    print("   🧬 Architecture DNA Evolution:")
    print("   >>> from nas_insights import ArchitectureDNA")
    print("   >>> dna = ArchitectureDNA()")
    print("   >>> dna.visualize_dna_evolution()")
    print()
    print("   🏔️  Search Landscape Topology:")
    print("   >>> from nas_insights import SearchLandscape")
    print("   >>> landscape = SearchLandscape()")
    print("   >>> landscape.visualize_search_paths()")
    print()
    print("   🎯 Multi-Objective Pareto Analysis:")
    print("   >>> from nas_insights import ParetoAnalyzer")
    print("   >>> pareto = ParetoAnalyzer()")
    print("   >>> pareto.simulate_pareto_evolution()")
    print()

def demo_comparison():
    """Compare different approaches."""
    print("⚖️  6. Algorithm Comparison")
    print("-" * 40)
    
    print("🧬 Evolutionary Search:")
    print("   ✅ Robust exploration")
    print("   ✅ Handles noisy evaluation")
    print("   ⏱️  Slower convergence")
    print()
    
    print("📈 DARTS (Gradient-based):")
    print("   ✅ Fast convergence")
    print("   ✅ Memory efficient")
    print("   ⚠️  May get trapped in local optima")
    print()

def demo_usage_examples():
    """Show practical usage examples."""
    print("💻 7. Practical Usage Examples")
    print("-" * 40)
    
    print("🔥 One-line architecture search:")
    print("   model = nano_nas('evolution')")
    print()
    
    print("⚙️  Custom search parameters:")
    print("   model = nano_nas('evolution',")
    print("                   population_size=20,")
    print("                   generations=15)")
    print()
    
    print("🎯 Quick DARTS search:")
    print("   model = nano_nas('darts', epochs=20)")
    print()
    
    print("🏋️  Train your discovered model:")
    print("   optimizer = torch.optim.Adam(model.parameters())")
    print("   # Standard PyTorch training loop")
    print()

def main():
    """Run the complete demo."""
    print_header()
    
    try:
        demo_architecture_basics()
        demo_dna_concept()
        demo_evolution_search()
        demo_darts_search()
        demo_comparison()
        demo_usage_examples()
        demo_insights()
        
        print("🎉 Demo Complete!")
        print("=" * 60)
        print("🚀 You've seen Neural Architecture Search in action!")
        print("📚 Check README_KARPATHY.md for deep dive")
        print("🧪 Run test_nano_nas.py to verify everything works")
        print("🎯 Try examples/quickstart.py for hands-on experience")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("💡 Make sure you have: pip install torch numpy matplotlib seaborn")

if __name__ == "__main__":
    main() 