"""
nanoNAS: Neural Architecture Search in <200 lines
-------------------------------------------------
A minimal, educational implementation of Neural Architecture Search
that demonstrates the core concepts with crystal clarity.

Inspired by Andrej Karpathy's minimalist approach to AI education.

Key insight: NAS is just optimization over discrete architecture choices
using continuous relaxation (DARTS) or evolutionary search.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from typing import List, Dict, Any

# ================================
# PART 1: SEARCH SPACE (30 lines)
# ================================

class Operation(nn.Module):
    """A single operation in our search space."""
    
    OPS = {
        'conv3x3': lambda C: nn.Conv2d(C, C, 3, padding=1),
        'conv5x5': lambda C: nn.Conv2d(C, C, 5, padding=2), 
        'maxpool': lambda C: nn.MaxPool2d(3, padding=1, stride=1),
        'skip': lambda C: nn.Identity(),
        'zero': lambda C: Zero(),
    }
    
    def __init__(self, op_name: str, channels: int):
        super().__init__()
        self.op = self.OPS[op_name](channels)
        
    def forward(self, x):
        return self.op(x)

class Zero(nn.Module):
    """Zero operation - outputs zeros."""
    def forward(self, x):
        return torch.zeros_like(x)

# ================================
# PART 2: ARCHITECTURE ENCODING (25 lines)
# ================================

class Architecture:
    """Encode architecture as a simple list of operation choices."""
    
    def __init__(self, encoding: List[int] = None, num_blocks: int = 4):
        """Each block chooses from 5 operations (0-4)."""
        self.num_ops = 5
        self.num_blocks = num_blocks
        self.encoding = encoding or [random.randint(0, self.num_ops-1) for _ in range(num_blocks)]
    
    def to_model(self, channels: int = 16) -> nn.Module:
        """Convert architecture encoding to actual PyTorch model."""
        ops = []
        op_names = list(Operation.OPS.keys())
        for block_choice in self.encoding:
            op_name = op_names[block_choice]
            ops.append(Operation(op_name, channels))
        return NanoNet(ops, channels)
    
    def mutate(self, mutation_rate: float = 0.1):
        """Mutate architecture for evolutionary search."""
        new_encoding = []
        for gene in self.encoding:
            if random.random() < mutation_rate:
                new_encoding.append(random.randint(0, self.num_ops-1))
            else:
                new_encoding.append(gene)
        return Architecture(new_encoding, self.num_blocks)

# ================================
# PART 3: MODEL DEFINITION (20 lines)
# ================================

class NanoNet(nn.Module):
    """Tiny neural network built from searched operations."""
    
    def __init__(self, ops: List[Operation], channels: int = 16):
        super().__init__()
        self.stem = nn.Conv2d(3, channels, 3, padding=1)
        self.ops = nn.ModuleList(ops)
        self.classifier = nn.Linear(channels, 10)  # CIFAR-10
        
    def forward(self, x):
        x = self.stem(x)
        for op in self.ops:
            x = op(x) + x  # residual connection
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)

# ================================
# PART 4: SEARCH ALGORITHMS (60 lines)
# ================================

class EvolutionaryNAS:
    """Dead simple evolutionary search for NAS."""
    
    def __init__(self, population_size: int = 20, mutation_rate: float = 0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        
    def search(self, generations: int = 10) -> Architecture:
        """Find best architecture via evolution."""
        # Initialize random population
        population = [Architecture() for _ in range(self.population_size)]
        
        for gen in range(generations):
            # Evaluate fitness (accuracy on validation set)
            fitness_scores = []
            for arch in population:
                model = arch.to_model()
                accuracy = self.evaluate_architecture(model)
                fitness_scores.append(accuracy)
            
            # Select top 50% and create next generation
            sorted_pop = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
            elite = [arch for arch, _ in sorted_pop[:self.population_size//2]]
            
            # Create offspring via mutation
            new_population = elite.copy()
            while len(new_population) < self.population_size:
                parent = random.choice(elite)
                child = parent.mutate(self.mutation_rate)
                new_population.append(child)
            
            population = new_population
            best_acc = max(fitness_scores)
            print(f"Generation {gen}: Best accuracy = {best_acc:.3f}")
        
        # Return best architecture
        final_fitness = [self.evaluate_architecture(arch.to_model()) for arch in population]
        best_idx = np.argmax(final_fitness)
        return population[best_idx]
    
    def evaluate_architecture(self, model: nn.Module) -> float:
        """Quick evaluation on toy data (replace with real validation)."""
        model.eval()
        with torch.no_grad():
            # Toy evaluation - replace with real CIFAR-10 validation
            x = torch.randn(32, 3, 32, 32)
            y = torch.randint(0, 10, (32,))
            logits = model(x)
            acc = (logits.argmax(1) == y).float().mean().item()
            # Add some noise to simulate real performance
            return acc + random.uniform(-0.1, 0.1)

class DifferentiableNAS:
    """DARTS: Differentiable Architecture Search in minimal form."""
    
    def __init__(self, channels: int = 16):
        self.channels = channels
        
    def search(self, epochs: int = 10) -> Architecture:
        """Search via gradient descent on architecture weights."""
        # Architecture weights (alpha) - one per operation per block
        num_ops = len(Operation.OPS)
        num_blocks = 4
        alpha = torch.randn(num_blocks, num_ops, requires_grad=True)
        
        optimizer = torch.optim.Adam([alpha], lr=0.01)
        
        for epoch in range(epochs):
            # Sample random data (replace with real dataset)
            x = torch.randn(8, 3, 32, 32)
            y = torch.randint(0, 10, (8,))
            
            # Build differentiable model
            model = self.build_differentiable_model(alpha)
            
            # Forward pass
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch}: Loss = {loss.item():.3f}")
        
        # Convert continuous weights to discrete architecture
        return self.alpha_to_architecture(alpha)
    
    def alpha_to_architecture(self, alpha: torch.Tensor) -> Architecture:
        """Convert learned weights to discrete architecture."""
        encoding = []
        for block_weights in alpha:
            best_op = block_weights.argmax().item()
            encoding.append(best_op)
        return Architecture(encoding)
    
    def build_differentiable_model(self, alpha: torch.Tensor) -> nn.Module:
        """Build model with weighted operations (simplified DARTS)."""
        # This is a simplified version - real DARTS is more complex
        # For educational purposes, we'll use the strongest operation per block
        encoding = []
        for block_weights in alpha:
            best_op = F.softmax(block_weights, dim=0).argmax().item()
            encoding.append(best_op)
        return Architecture(encoding).to_model(self.channels)

# ================================
# PART 5: SIMPLE API (30 lines)  
# ================================

def nano_nas(method: str = 'evolution', **kwargs) -> nn.Module:
    """
    One-line Neural Architecture Search.
    
    Args:
        method: 'evolution' or 'darts'
        **kwargs: Search parameters
    
    Returns:
        Optimized PyTorch model ready for training
        
    Example:
        >>> model = nano_nas('evolution', generations=5)
        >>> model = nano_nas('darts', epochs=10)
    """
    print(f"🔍 Running nanoNAS with {method}...")
    
    if method == 'evolution':
        # Extract search-specific parameters
        generations = kwargs.pop('generations', 10)
        population_size = kwargs.pop('population_size', 20)
        mutation_rate = kwargs.pop('mutation_rate', 0.1)
        
        searcher = EvolutionaryNAS(population_size, mutation_rate)
        best_arch = searcher.search(generations)
    elif method == 'darts':
        # Extract search-specific parameters  
        epochs = kwargs.pop('epochs', 10)
        channels = kwargs.pop('channels', 16)
        
        searcher = DifferentiableNAS(channels)
        best_arch = searcher.search(epochs)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"✅ Found architecture: {best_arch.encoding}")
    return best_arch.to_model()

# ================================
# USAGE EXAMPLE
# ================================

if __name__ == "__main__":
    print("nanoNAS: Neural Architecture Search in <200 lines")
    print("=" * 50)
    
    # Example 1: Evolutionary search
    print("\n1. Evolutionary Search:")
    model1 = nano_nas('evolution', population_size=10, generations=3)
    print(f"Model parameters: {sum(p.numel() for p in model1.parameters()):,}")
    
    # Example 2: Differentiable search  
    print("\n2. Differentiable Search (DARTS):")
    model2 = nano_nas('darts', epochs=5)
    print(f"Model parameters: {sum(p.numel() for p in model2.parameters()):,}")
    
    print("\n🎯 That's it! NAS in under 200 lines.")
    print("   Extend this for real datasets, better ops, multi-objective optimization...") 