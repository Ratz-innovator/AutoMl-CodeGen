"""
Architecture Visualization for Neural Architecture Search
========================================================

This module provides advanced visualization capabilities for neural architectures
including DAG visualizations, operation flow diagrams, and interactive plots.

Key Features:
- NetworkX-based architecture graphs
- Matplotlib and Plotly visualizations
- Publication-ready diagrams
- Interactive exploration tools
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import seaborn as sns

from ..core.base import BaseVisualizer
from ..core.architecture import Architecture, SearchSpace
from ..core.config import ExperimentConfig

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class ArchitectureVisualizer(BaseVisualizer):
    """
    Advanced architecture visualization with multiple rendering backends.
    
    Supports creating publication-quality diagrams of neural architectures
    with various layout algorithms and styling options.
    """
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        """Initialize architecture visualizer."""
        super().__init__(config)
        
        # Visualization settings
        self.figsize = (12, 8)
        self.dpi = 300
        self.node_colors = {
            'conv': '#FF6B6B',      # Red for convolutions
            'pool': '#4ECDC4',      # Teal for pooling
            'skip': '#45B7D1',      # Blue for skip connections
            'zero': '#96CEB4',      # Green for zero operations
            'input': '#FECA57',     # Yellow for input
            'output': '#FF9FF3',    # Pink for output
        }
        
    def visualize(self, 
                 architecture: Architecture,
                 output_path: Optional[str] = None,
                 title: str = "Neural Architecture",
                 layout: str = "hierarchical",
                 show_operations: bool = True,
                 show_connections: bool = True) -> str:
        """
        Create architecture visualization.
        
        Args:
            architecture: Architecture to visualize
            output_path: Path to save visualization
            title: Plot title
            layout: Layout algorithm ('hierarchical', 'spring', 'circular')
            show_operations: Whether to show operation details
            show_connections: Whether to show skip connections
            
        Returns:
            Path to saved visualization
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        if architecture.graph is not None:
            # Graph-based architecture
            self._plot_graph_architecture(ax, architecture, layout, show_operations, show_connections)
        elif architecture.encoding is not None:
            # List-based architecture
            self._plot_list_architecture(ax, architecture, show_operations)
        else:
            raise ValueError("Architecture must have either graph or encoding representation")
        
        # Styling
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add metadata
        self._add_metadata_text(ax, architecture)
        
        plt.tight_layout()
        
        # Save if path provided
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            self.logger.info(f"Architecture visualization saved to {output_path}")
            plt.close()
            return output_path
        else:
            plt.show()
            return ""
    
    def _plot_graph_architecture(self, 
                                ax,
                                architecture: Architecture,
                                layout: str,
                                show_operations: bool,
                                show_connections: bool):
        """Plot graph-based architecture."""
        G = architecture.graph.copy()
        
        # Compute layout
        if layout == "hierarchical":
            pos = self._hierarchical_layout(G)
        elif layout == "spring":
            pos = nx.spring_layout(G, k=2, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Draw nodes
        for node in G.nodes():
            x, y = pos[node]
            
            # Get operation info
            if 'operation' in G.nodes[node]:
                op_idx = G.nodes[node]['operation']
                if op_idx < len(architecture.search_space.operations):
                    op_spec = architecture.search_space.operations[op_idx]
                    op_type = op_spec.type
                    op_name = op_spec.name if show_operations else op_type
                else:
                    op_type = 'unknown'
                    op_name = 'unknown'
            else:
                op_type = 'node'
                op_name = f'Node {node}'
            
            # Get color
            color = self.node_colors.get(op_type, '#D3D3D3')
            
            # Draw node
            circle = patches.Circle((x, y), 0.1, color=color, alpha=0.8, zorder=2)
            ax.add_patch(circle)
            
            # Add label
            if show_operations:
                ax.text(x, y-0.15, op_name, ha='center', va='top', 
                       fontsize=8, fontweight='bold')
        
        # Draw edges
        for edge in G.edges():
            x1, y1 = pos[edge[0]]
            x2, y2 = pos[edge[1]]
            
            if show_connections:
                ax.arrow(x1, y1, x2-x1, y2-y1, head_width=0.03, head_length=0.03,
                        fc='gray', ec='gray', alpha=0.6, zorder=1)
        
        # Set axis limits
        x_coords = [pos[node][0] for node in G.nodes()]
        y_coords = [pos[node][1] for node in G.nodes()]
        margin = 0.2
        ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
        ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
    
    def _plot_list_architecture(self, 
                               ax,
                               architecture: Architecture,
                               show_operations: bool):
        """Plot list-based architecture as a flowchart."""
        encoding = architecture.encoding
        search_space = architecture.search_space
        
        # Layout parameters
        spacing_x = 1.5
        spacing_y = 0.5
        start_x = 0
        start_y = 0
        
        # Draw operations
        for i, op_idx in enumerate(encoding):
            x = start_x + i * spacing_x
            y = start_y
            
            # Get operation info
            if op_idx < len(search_space.operations):
                op_spec = search_space.operations[op_idx]
                op_type = op_spec.type
                op_name = op_spec.name if show_operations else op_type
            else:
                op_type = 'unknown'
                op_name = 'unknown'
            
            # Get color
            color = self.node_colors.get(op_type, '#D3D3D3')
            
            # Draw operation box
            rect = patches.Rectangle((x-0.3, y-0.2), 0.6, 0.4, 
                                   facecolor=color, alpha=0.8, zorder=2)
            ax.add_patch(rect)
            
            # Add label
            if show_operations:
                ax.text(x, y, op_name, ha='center', va='center', 
                       fontsize=9, fontweight='bold')
            
            # Draw arrow to next operation
            if i < len(encoding) - 1:
                ax.arrow(x + 0.3, y, spacing_x - 0.6, 0, 
                        head_width=0.1, head_length=0.1,
                        fc='gray', ec='gray', alpha=0.6, zorder=1)
        
        # Set axis limits
        ax.set_xlim(-0.5, len(encoding) * spacing_x)
        ax.set_ylim(-0.5, 0.5)
    
    def _hierarchical_layout(self, G: nx.DiGraph) -> Dict[int, Tuple[float, float]]:
        """Create hierarchical layout for DAG."""
        try:
            # Use topological sort to assign levels
            levels = {}
            for node in nx.topological_sort(G):
                if G.in_degree(node) == 0:
                    levels[node] = 0
                else:
                    levels[node] = max(levels[pred] for pred in G.predecessors(node)) + 1
            
            # Group nodes by level
            level_groups = {}
            for node, level in levels.items():
                if level not in level_groups:
                    level_groups[level] = []
                level_groups[level].append(node)
            
            # Assign positions
            pos = {}
            for level, nodes in level_groups.items():
                for i, node in enumerate(nodes):
                    x = level * 2
                    y = (i - len(nodes)/2) * 1.5
                    pos[node] = (x, y)
            
            return pos
            
        except:
            # Fallback to spring layout
            return nx.spring_layout(G)
    
    def _add_metadata_text(self, ax, architecture: Architecture):
        """Add metadata text to the plot."""
        complexity = architecture.get_complexity_metrics()
        
        metadata_text = []
        if 'depth' in complexity:
            metadata_text.append(f"Depth: {complexity['depth']}")
        if 'total_op_cost' in complexity:
            metadata_text.append(f"Op Cost: {complexity['total_op_cost']:.2f}")
        if 'skip_ratio' in complexity:
            metadata_text.append(f"Skip Ratio: {complexity['skip_ratio']:.2f}")
        
        if metadata_text:
            ax.text(0.02, 0.98, "\n".join(metadata_text), 
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def create_comparison_plot(self, 
                              architectures: List[Architecture],
                              names: List[str],
                              output_path: Optional[str] = None) -> str:
        """
        Create comparison plot of multiple architectures.
        
        Args:
            architectures: List of architectures to compare
            names: Names for each architecture
            output_path: Path to save plot
            
        Returns:
            Path to saved plot
        """
        num_archs = len(architectures)
        cols = min(3, num_archs)
        rows = (num_archs + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), dpi=self.dpi)
        if num_archs == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (arch, name) in enumerate(zip(architectures, names)):
            if i < len(axes):
                ax = axes[i]
                
                # Plot architecture
                if arch.graph is not None:
                    self._plot_graph_architecture(ax, arch, "hierarchical", True, True)
                elif arch.encoding is not None:
                    self._plot_list_architecture(ax, arch, True)
                
                ax.set_title(name, fontsize=12, fontweight='bold')
                ax.axis('off')
        
        # Hide unused subplots
        for i in range(num_archs, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Architecture comparison saved to {output_path}")
            plt.close()
            return output_path
        else:
            plt.show()
            return ""
    
    def create_evolution_animation(self,
                                  architecture_history: List[Dict[str, Any]],
                                  output_path: str):
        """
        Create animation showing architecture evolution over time.
        
        Args:
            architecture_history: List of architecture snapshots
            output_path: Path to save animation
        """
        # This would create an animated GIF or video
        # For now, we'll create a static plot showing evolution
        
        num_snapshots = len(architecture_history)
        cols = min(4, num_snapshots)
        rows = (num_snapshots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), dpi=self.dpi)
        if num_snapshots == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, arch_info in enumerate(architecture_history):
            if i < len(axes):
                ax = axes[i]
                arch = arch_info['architecture']
                epoch = arch_info.get('epoch', i)
                
                # Plot architecture
                if arch.graph is not None:
                    self._plot_graph_architecture(ax, arch, "spring", False, True)
                elif arch.encoding is not None:
                    self._plot_list_architecture(ax, arch, False)
                
                ax.set_title(f"Epoch {epoch}", fontsize=10)
                ax.axis('off')
        
        # Hide unused subplots
        for i in range(num_snapshots, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle("Architecture Evolution During Search", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        self.logger.info(f"Evolution animation saved to {output_path}")
        plt.close()


def plot_architecture(architecture: Architecture,
                     output_path: Optional[str] = None,
                     title: str = "Neural Architecture",
                     **kwargs) -> str:
    """
    Convenience function to plot an architecture.
    
    Args:
        architecture: Architecture to plot
        output_path: Optional output path
        title: Plot title
        **kwargs: Additional arguments for visualization
        
    Returns:
        Path to saved plot
    """
    visualizer = ArchitectureVisualizer()
    return visualizer.visualize(architecture, output_path, title, **kwargs)


def create_operation_legend(search_space: SearchSpace, 
                           output_path: Optional[str] = None) -> str:
    """
    Create a legend showing all operations in the search space.
    
    Args:
        search_space: Search space to create legend for
        output_path: Optional output path
        
    Returns:
        Path to saved legend
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    visualizer = ArchitectureVisualizer()
    
    # Group operations by type
    op_groups = {}
    for op_spec in search_space.operations:
        op_type = op_spec.type
        if op_type not in op_groups:
            op_groups[op_type] = []
        op_groups[op_type].append(op_spec.name)
    
    # Create legend
    y_pos = 0.9
    for op_type, op_names in op_groups.items():
        color = visualizer.node_colors.get(op_type, '#D3D3D3')
        
        # Draw color patch
        rect = patches.Rectangle((0.1, y_pos-0.02), 0.05, 0.04, 
                               facecolor=color, alpha=0.8)
        ax.add_patch(rect)
        
        # Add text
        ax.text(0.2, y_pos, f"{op_type.upper()}: {', '.join(op_names)}", 
               fontsize=12, va='center')
        
        y_pos -= 0.1
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Operation Legend", fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.show()
        return "" 