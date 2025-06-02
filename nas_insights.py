"""
NAS Insights: Novel Perspectives on Neural Architecture Search
==============================================================

This module provides unique insights and visualizations that make
Neural Architecture Search concepts immediately graspable.

Novel contributions:
1. Architecture DNA visualization
2. Search landscape topology
3. Convergence dynamics analysis
4. Multi-objective Pareto surfaces
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
from nanonas import Architecture, nano_nas

# ===========================
# NOVEL INSIGHT 1: Architecture DNA
# ===========================

class ArchitectureDNA:
    """
    Represent architectures as DNA sequences for intuitive understanding.
    
    Key insight: NAS is like biological evolution - architectures have
    'genetic codes' that determine their performance characteristics.
    """
    
    def __init__(self):
        self.bases = ['A', 'T', 'G', 'C', 'N']  # Architecture, Tensor, Gate, Connect, Null
        self.op_mapping = {
            0: 'A',  # conv3x3 - Activation pathway
            1: 'T',  # conv5x5 - Tensor pathway  
            2: 'G',  # maxpool - Gate pathway
            3: 'C',  # skip - Connection pathway
            4: 'N'   # zero - Null pathway
        }
    
    def encode_architecture(self, arch: Architecture) -> str:
        """Convert architecture to DNA string."""
        return ''.join([self.op_mapping[op] for op in arch.encoding])
    
    def decode_dna(self, dna: str) -> Architecture:
        """Convert DNA string back to architecture."""
        reverse_map = {v: k for k, v in self.op_mapping.items()}
        encoding = [reverse_map[base] for base in dna]
        return Architecture(encoding)
    
    def mutate_dna(self, dna: str, mutation_rate: float = 0.1) -> str:
        """Mutate DNA string (like biological mutation)."""
        mutated = []
        for base in dna:
            if np.random.random() < mutation_rate:
                mutated.append(np.random.choice(self.bases))
            else:
                mutated.append(base)
        return ''.join(mutated)
    
    def visualize_dna_evolution(self, generations: int = 10, population_size: int = 20):
        """Visualize how architecture DNA evolves over generations."""
        
        # Initialize population
        population = [Architecture() for _ in range(population_size)]
        dna_history = []
        fitness_history = []
        
        for gen in range(generations):
            # Convert to DNA
            dna_population = [self.encode_architecture(arch) for arch in population]
            dna_history.append(dna_population)
            
            # Evaluate fitness (simplified)
            fitness = [np.random.random() + 0.1 * gen for _ in population]  # Simulated improvement
            fitness_history.append(fitness)
            
            # Select and evolve
            sorted_pop = sorted(zip(population, fitness), key=lambda x: x[1], reverse=True)
            elite = [arch for arch, _ in sorted_pop[:population_size//2]]
            
            # Create next generation
            new_population = elite.copy()
            while len(new_population) < population_size:
                parent = np.random.choice(elite)
                child_dna = self.mutate_dna(self.encode_architecture(parent))
                new_population.append(self.decode_dna(child_dna))
            
            population = new_population
        
        # Visualize evolution
        self._plot_dna_evolution(dna_history, fitness_history)
    
    def _plot_dna_evolution(self, dna_history: List[List[str]], fitness_history: List[List[float]]):
        """Plot DNA evolution visualization."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: DNA diversity over time
        diversity_scores = []
        for gen_dna in dna_history:
            unique_sequences = len(set(gen_dna))
            diversity_scores.append(unique_sequences / len(gen_dna))
        
        generations = range(len(diversity_scores))
        ax1.plot(generations, diversity_scores, 'o-', linewidth=3, markersize=8, color='#FF6B6B')
        ax1.fill_between(generations, diversity_scores, alpha=0.3, color='#FF6B6B')
        ax1.set_xlabel('Generation', fontsize=12)
        ax1.set_ylabel('DNA Diversity', fontsize=12)
        ax1.set_title('🧬 Architecture DNA Diversity Evolution', fontsize=14, weight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Fitness progression
        mean_fitness = [np.mean(gen_fitness) for gen_fitness in fitness_history]
        max_fitness = [np.max(gen_fitness) for gen_fitness in fitness_history]
        
        ax2.plot(generations, mean_fitness, 'o-', label='Mean Fitness', color='#4ECDC4', linewidth=2)
        ax2.plot(generations, max_fitness, 's-', label='Best Fitness', color='#45B7D1', linewidth=2)
        ax2.fill_between(generations, mean_fitness, alpha=0.3, color='#4ECDC4')
        ax2.set_xlabel('Generation', fontsize=12)
        ax2.set_ylabel('Fitness Score', fontsize=12)
        ax2.set_title('📈 Architecture Fitness Evolution', fontsize=14, weight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("🧬 DNA Evolution Insights:")
        print(f"   • Initial diversity: {diversity_scores[0]:.2f}")
        print(f"   • Final diversity: {diversity_scores[-1]:.2f}")
        print(f"   • Fitness improvement: {max_fitness[-1] - max_fitness[0]:.2f}")


# ===========================
# NOVEL INSIGHT 2: Search Landscape Topology
# ===========================

class SearchLandscape:
    """
    Visualize the NAS search space as a fitness landscape.
    
    Key insight: Understanding the 'geography' of the search space
    helps explain why different algorithms work better in different regions.
    """
    
    def __init__(self, arch_space_size: int = 100):
        self.arch_space_size = arch_space_size
        
    def generate_landscape(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a 2D projection of the architecture search landscape."""
        
        # Create a grid of points in 2D space (reduced from high-dim architecture space)
        x = np.linspace(-5, 5, 50)
        y = np.linspace(-5, 5, 50)
        X, Y = np.meshgrid(x, y)
        
        # Generate fitness landscape with multiple peaks (multi-modal optimization)
        Z = (
            # Main peak (global optimum)
            2.0 * np.exp(-((X-1)**2 + (Y-1)**2)) +
            # Secondary peak (local optimum)
            1.5 * np.exp(-((X+2)**2 + (Y+2)**2)/2) +
            # Ridge (represents skip connections)
            0.8 * np.exp(-(X**2 + (Y-2)**2)/3) +
            # Valley (represents poor architectures)
            -0.5 * np.exp(-((X+1)**2 + (Y-1)**2)/4) +
            # Noise (represents stochastic evaluation)
            0.1 * np.random.random(X.shape)
        )
        
        return X, Y, Z
    
    def visualize_search_paths(self):
        """Visualize how different search algorithms navigate the landscape."""
        
        X, Y, Z = self.generate_landscape()
        
        # Simulate different search algorithms
        evo_path = self._simulate_evolutionary_path()
        gradient_path = self._simulate_gradient_path()
        
        # Create 3D visualization
        fig = plt.figure(figsize=(16, 6))
        
        # 3D landscape
        ax1 = fig.add_subplot(121, projection='3d')
        surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
        
        # Plot search paths
        ax1.plot(evo_path[:, 0], evo_path[:, 1], evo_path[:, 2], 
                'ro-', linewidth=3, markersize=6, label='Evolutionary')
        ax1.plot(gradient_path[:, 0], gradient_path[:, 1], gradient_path[:, 2], 
                'bs-', linewidth=3, markersize=6, label='Gradient-based')
        
        ax1.set_xlabel('Architecture Dimension 1')
        ax1.set_ylabel('Architecture Dimension 2') 
        ax1.set_zlabel('Fitness')
        ax1.set_title('🏔️ NAS Search Landscape')
        ax1.legend()
        
        # 2D contour plot
        ax2 = fig.add_subplot(122)
        contour = ax2.contour(X, Y, Z, levels=15, colors='gray', alpha=0.6)
        ax2.contourf(X, Y, Z, levels=15, cmap='viridis', alpha=0.7)
        
        # Plot search paths in 2D
        ax2.plot(evo_path[:, 0], evo_path[:, 1], 'ro-', linewidth=3, 
                markersize=8, label='Evolutionary', alpha=0.8)
        ax2.plot(gradient_path[:, 0], gradient_path[:, 1], 'bs-', linewidth=3, 
                markersize=8, label='Gradient-based', alpha=0.8)
        
        ax2.set_xlabel('Architecture Dimension 1')
        ax2.set_ylabel('Architecture Dimension 2')
        ax2.set_title('🗺️ Search Strategy Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("🏔️ Search Landscape Insights:")
        print("   • Multiple peaks = multiple good architectures")
        print("   • Evolutionary search explores broadly")
        print("   • Gradient search converges quickly but may get trapped")
        print("   • The landscape shape determines which algorithm works best")
    
    def _simulate_evolutionary_path(self) -> np.ndarray:
        """Simulate evolutionary search path."""
        path = []
        # Start at random location
        x, y = np.random.uniform(-3, 3, 2)
        
        for step in range(15):
            # Random exploration with bias toward better regions
            dx, dy = np.random.normal(0, 0.5, 2)
            if step > 5:  # Start converging after exploration
                dx *= 0.7
                dy *= 0.7
            
            x += dx
            y += dy
            
            # Evaluate fitness at this point
            fitness = (2.0 * np.exp(-((x-1)**2 + (y-1)**2)) + 
                      1.5 * np.exp(-((x+2)**2 + (y+2)**2)/2) +
                      0.1 * np.random.random())
            
            path.append([x, y, fitness])
        
        return np.array(path)
    
    def _simulate_gradient_path(self) -> np.ndarray:
        """Simulate gradient-based search path."""
        path = []
        # Start at random location
        x, y = np.random.uniform(-3, 3, 2)
        
        for step in range(15):
            # Gradient ascent toward nearest peak
            grad_x = 4.0 * (1-x) * np.exp(-((x-1)**2 + (y-1)**2))
            grad_y = 4.0 * (1-y) * np.exp(-((x-1)**2 + (y-1)**2))
            
            # Add some noise
            grad_x += np.random.normal(0, 0.1)
            grad_y += np.random.normal(0, 0.1)
            
            # Take gradient step
            x += 0.3 * grad_x
            y += 0.3 * grad_y
            
            # Evaluate fitness
            fitness = (2.0 * np.exp(-((x-1)**2 + (y-1)**2)) + 
                      1.5 * np.exp(-((x+2)**2 + (y+2)**2)/2) +
                      0.1 * np.random.random())
            
            path.append([x, y, fitness])
        
        return np.array(path)


# ===========================
# NOVEL INSIGHT 3: Multi-Objective Pareto Dynamics  
# ===========================

class ParetoAnalyzer:
    """
    Analyze multi-objective optimization dynamics in NAS.
    
    Key insight: Real-world NAS must balance multiple objectives
    (accuracy, speed, memory). The Pareto frontier reveals trade-offs.
    """
    
    def simulate_pareto_evolution(self, generations: int = 20):
        """Simulate how Pareto frontier evolves during multi-objective NAS."""
        
        pareto_history = []
        
        for gen in range(generations):
            # Generate random architectures with 3 objectives
            n_archs = 50
            accuracy = np.random.beta(2, 5, n_archs) * 100  # Accuracy (0-100%)
            latency = np.random.exponential(2, n_archs)      # Latency (ms)
            memory = np.random.gamma(2, 2, n_archs)          # Memory (MB)
            
            # Apply evolutionary pressure (improvement over time)
            if gen > 0:
                accuracy += gen * 0.5  # Accuracy improves
                latency *= (1 - gen * 0.02)  # Latency decreases
                memory *= (1 - gen * 0.01)   # Memory usage decreases
            
            # Compute Pareto frontier
            pareto_points = self._compute_pareto_frontier(accuracy, -latency, -memory)
            pareto_history.append(pareto_points)
        
        self._visualize_pareto_evolution(pareto_history)
    
    def _compute_pareto_frontier(self, obj1: np.ndarray, obj2: np.ndarray, obj3: np.ndarray) -> np.ndarray:
        """Compute Pareto frontier for 3 objectives (all maximization)."""
        objectives = np.column_stack([obj1, obj2, obj3])
        pareto_mask = np.ones(len(objectives), dtype=bool)
        
        for i, point in enumerate(objectives):
            if pareto_mask[i]:
                # Check if this point is dominated by any other
                dominated = np.all(objectives >= point, axis=1) & np.any(objectives > point, axis=1)
                pareto_mask[dominated] = False
        
        return objectives[pareto_mask]
    
    def _visualize_pareto_evolution(self, pareto_history: List[np.ndarray]):
        """Visualize how Pareto frontier evolves."""
        
        fig = plt.figure(figsize=(18, 6))
        
        # Plot 1: Accuracy vs Latency trade-off
        ax1 = fig.add_subplot(131)
        colors = plt.cm.viridis(np.linspace(0, 1, len(pareto_history)))
        
        for gen, (pareto_points, color) in enumerate(zip(pareto_history[::4], colors[::4])):
            ax1.scatter(pareto_points[:, 1], pareto_points[:, 0], 
                       c=[color], s=60, alpha=0.7, label=f'Gen {gen*4}')
        
        ax1.set_xlabel('Speed (negative latency)', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title('🎯 Accuracy vs Speed Trade-off', fontsize=14, weight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy vs Memory trade-off
        ax2 = fig.add_subplot(132)
        for gen, (pareto_points, color) in enumerate(zip(pareto_history[::4], colors[::4])):
            ax2.scatter(pareto_points[:, 2], pareto_points[:, 0], 
                       c=[color], s=60, alpha=0.7, label=f'Gen {gen*4}')
        
        ax2.set_xlabel('Memory Efficiency (negative usage)', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('💾 Accuracy vs Memory Trade-off', fontsize=14, weight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: 3D Pareto frontier evolution
        ax3 = fig.add_subplot(133, projection='3d')
        
        # Show initial and final frontiers
        initial_frontier = pareto_history[0]
        final_frontier = pareto_history[-1]
        
        ax3.scatter(initial_frontier[:, 0], initial_frontier[:, 1], initial_frontier[:, 2],
                   c='red', s=80, alpha=0.6, label='Initial Frontier')
        ax3.scatter(final_frontier[:, 0], final_frontier[:, 1], final_frontier[:, 2],
                   c='blue', s=80, alpha=0.8, label='Final Frontier')
        
        ax3.set_xlabel('Accuracy (%)')
        ax3.set_ylabel('Speed')
        ax3.set_zlabel('Memory Efficiency')
        ax3.set_title('🚀 3D Pareto Evolution')
        ax3.legend()
        
        plt.tight_layout()
        plt.show()
        
        print("🎯 Multi-Objective Insights:")
        print(f"   • Initial Pareto points: {len(pareto_history[0])}")
        print(f"   • Final Pareto points: {len(pareto_history[-1])}")
        print("   • Trade-offs become clearer as search progresses")
        print("   • Multiple optimal solutions exist for different use cases")


# ===========================
# EDUCATIONAL API - One-line insights
# ===========================

def explore_nas_concepts():
    """One function to explore all NAS insights interactively."""
    print("🧠 Exploring Neural Architecture Search Concepts")
    print("=" * 50)
    
    # Insight 1: Architecture DNA
    print("\n🧬 1. Architecture DNA Evolution")
    dna_analyzer = ArchitectureDNA()
    dna_analyzer.visualize_dna_evolution(generations=8, population_size=15)
    
    # Insight 2: Search Landscape
    print("\n🏔️ 2. Search Space Landscape")
    landscape = SearchLandscape()
    landscape.visualize_search_paths()
    
    # Insight 3: Pareto Dynamics
    print("\n🎯 3. Multi-Objective Trade-offs")
    pareto_analyzer = ParetoAnalyzer()
    pareto_analyzer.simulate_pareto_evolution(generations=15)
    
    print("\n✨ Key Insights Summary:")
    print("   • NAS is like biological evolution with architecture DNA")
    print("   • Search algorithms navigate complex fitness landscapes")
    print("   • Multiple objectives create rich trade-off spaces")
    print("   • Understanding these dynamics helps choose the right algorithm")


if __name__ == "__main__":
    # Run the complete insights exploration
    explore_nas_concepts() 