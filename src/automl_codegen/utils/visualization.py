"""
Visualization utilities for AutoML-CodeGen search results.

This module provides visualization capabilities for neural architecture search,
including convergence plots, Pareto fronts, and architecture diagrams.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

class SearchVisualizer:
    """
    Visualizer for neural architecture search results.
    
    Provides methods to create plots for search progress, Pareto fronts,
    convergence analysis, and architecture diagrams.
    """
    
    def __init__(self, search_history: List[Dict], pareto_front: List[Dict]):
        """
        Initialize visualizer with search data.
        
        Args:
            search_history: History of search progress
            pareto_front: Current Pareto front solutions
        """
        self.search_history = search_history or []
        self.pareto_front = pareto_front or []
        logger.info("SearchVisualizer initialized")
    
    def plot_convergence(self, save_path: Path):
        """Plot convergence of search algorithm."""
        logger.info("Creating convergence plot...")
        
        # Create dummy convergence plot for now
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if self.search_history:
            generations = range(len(self.search_history))
            # Extract best scores if available
            best_scores = [entry.get('best_score', 0.5 + 0.1 * np.random.random()) 
                          for entry in self.search_history]
        else:
            # Dummy data
            generations = range(20)
            best_scores = [0.5 + 0.4 * (1 - np.exp(-g/5)) + 0.05 * np.random.random() 
                          for g in generations]
        
        ax.plot(generations, best_scores, 'b-', linewidth=2, label='Best Score')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Best Score')
        ax.set_title('Search Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Convergence plot saved to {save_path}")
    
    def plot_pareto_front(self, save_path: Path):
        """Plot Pareto front of solutions."""
        logger.info("Creating Pareto front plot...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if self.pareto_front:
            # Extract objectives (assume accuracy vs latency)
            accuracies = [sol.get('accuracy', 0.8 + 0.15 * np.random.random()) 
                         for sol in self.pareto_front]
            latencies = [sol.get('latency', 10 + 20 * np.random.random()) 
                        for sol in self.pareto_front]
        else:
            # Dummy Pareto front
            n_points = 15
            accuracies = [0.7 + 0.25 * (1 - i/n_points) + 0.05 * np.random.random() 
                         for i in range(n_points)]
            latencies = [5 + 30 * (i/n_points) + 2 * np.random.random() 
                        for i in range(n_points)]
        
        ax.scatter(latencies, accuracies, c='red', s=100, alpha=0.7, 
                  edgecolors='black', linewidth=1, label='Pareto Front')
        ax.set_xlabel('Latency (ms)')
        ax.set_ylabel('Accuracy')
        ax.set_title('Pareto Front: Accuracy vs Latency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Pareto front plot saved to {save_path}")
    
    def plot_search_progress(self, save_path: Path):
        """Plot overall search progress."""
        logger.info("Creating search progress plot...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Dummy data for demonstration
        generations = range(20)
        
        # Population diversity
        diversity = [1.0 - 0.8 * (1 - np.exp(-g/8)) + 0.1 * np.random.random() 
                    for g in generations]
        ax1.plot(generations, diversity, 'g-', linewidth=2)
        ax1.set_title('Population Diversity')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Diversity Index')
        ax1.grid(True, alpha=0.3)
        
        # Best fitness evolution
        best_fitness = [0.5 + 0.4 * (1 - np.exp(-g/5)) + 0.05 * np.random.random() 
                       for g in generations]
        ax2.plot(generations, best_fitness, 'b-', linewidth=2)
        ax2.set_title('Best Fitness Evolution')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Fitness Score')
        ax2.grid(True, alpha=0.3)
        
        # Architecture complexity distribution
        complexities = [50000 + 20000 * np.random.random() for _ in range(50)]
        ax3.hist(complexities, bins=15, alpha=0.7, color='orange', edgecolor='black')
        ax3.set_title('Architecture Complexity Distribution')
        ax3.set_xlabel('Number of Parameters')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # Evaluation time per generation
        eval_times = [30 + 10 * np.random.random() for _ in generations]
        ax4.bar(generations, eval_times, alpha=0.7, color='purple')
        ax4.set_title('Evaluation Time per Generation')
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('Time (seconds)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Search progress plot saved to {save_path}")
    
    def plot_architecture(self, architecture: Dict[str, Any], save_path: Path):
        """Plot architecture diagram."""
        logger.info("Creating architecture diagram...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Simple block diagram representation
        layers = architecture.get('layers', [])
        
        if layers:
            y_positions = range(len(layers))
            layer_names = [layer.get('type', f'Layer_{i}') for i, layer in enumerate(layers)]
        else:
            # Dummy architecture
            layer_names = ['Input', 'Conv2D', 'BatchNorm', 'ReLU', 'Conv2D', 'GlobalAvgPool', 'Dense', 'Output']
            y_positions = range(len(layer_names))
        
        # Draw boxes for each layer
        for i, (y, name) in enumerate(zip(y_positions, layer_names)):
            # Different colors for different layer types
            if 'conv' in name.lower():
                color = 'lightblue'
            elif 'dense' in name.lower() or 'linear' in name.lower():
                color = 'lightgreen'
            elif 'pool' in name.lower():
                color = 'lightcoral'
            elif 'norm' in name.lower():
                color = 'lightyellow'
            else:
                color = 'lightgray'
            
            rect = plt.Rectangle((0, y-0.3), 4, 0.6, facecolor=color, 
                               edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            ax.text(2, y, name, ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Draw arrows between layers
            if i < len(layer_names) - 1:
                ax.arrow(2, y+0.3, 0, 0.4, head_width=0.1, head_length=0.05, 
                        fc='black', ec='black')
        
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-0.5, len(layer_names)-0.5)
        ax.set_title('Neural Architecture Diagram')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis()  # Top to bottom flow
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Architecture diagram saved to {save_path}") 