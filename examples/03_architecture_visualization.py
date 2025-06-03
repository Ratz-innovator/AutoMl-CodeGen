#!/usr/bin/env python3
"""
nanoNAS Architecture Visualization Example
==========================================

This example demonstrates various ways to visualize neural architectures
found by nanoNAS, including network graphs, operation flows, and comparisons.

Requirements:
    pip install torch numpy matplotlib networkx plotly

Usage:
    python examples/03_architecture_visualization.py
"""

import sys
import os
import time
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyBboxPatch
    try:
        import networkx as nx
        HAS_NETWORKX = True
    except ImportError:
        HAS_NETWORKX = False
        print("⚠️  NetworkX not available - graph visualizations will be limited")
    
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        HAS_PLOTLY = True
    except ImportError:
        HAS_PLOTLY = False
        print("⚠️  Plotly not available - interactive plots will be skipped")
        
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Please install: pip install torch numpy matplotlib networkx plotly")
    sys.exit(1)

# Import nanoNAS
try:
    import nanonas
    from nanonas import nano_nas
except ImportError as e:
    print(f"❌ Failed to import nanoNAS: {e}")
    sys.exit(1)


class ArchitectureVisualizer:
    """Comprehensive architecture visualization toolkit."""
    
    def __init__(self, output_dir: str = "results/visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Define colors for different operation types
        self.operation_colors = {
            'conv': '#ff6b6b',      # Red for convolutions
            'pool': '#4ecdc4',      # Teal for pooling
            'skip': '#45b7d1',      # Blue for skip connections
            'zero': '#95a5a6',      # Gray for zero operations
            'attention': '#f39c12',  # Orange for attention
            'norm': '#e74c3c',      # Red for normalization
            'activation': '#9b59b6'  # Purple for activations
        }
    
    def visualize_architecture_sequence(self, architecture: nanonas.Architecture, 
                                      title: str = "Architecture Sequence") -> str:
        """Visualize architecture as a sequence of operations."""
        print(f"📊 Creating sequence visualization: {title}")
        
        if architecture.encoding is None:
            print("   ⚠️  Architecture has no sequence encoding")
            return ""
        
        operations = [architecture.search_space.operations[i] for i in architecture.encoding]
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Draw operation boxes
        box_width = 0.8
        box_height = 0.6
        
        for i, op in enumerate(operations):
            x = i
            y = 0
            
            # Get color based on operation type
            color = self.operation_colors.get(op.type, '#95a5a6')
            
            # Create fancy box
            box = FancyBboxPatch(
                (x - box_width/2, y - box_height/2),
                box_width, box_height,
                boxstyle="round,pad=0.1",
                facecolor=color,
                edgecolor='black',
                linewidth=1.5,
                alpha=0.8
            )
            ax.add_patch(box)
            
            # Add operation name
            ax.text(x, y, op.name, ha='center', va='center', 
                   fontsize=10, fontweight='bold', color='white')
            
            # Add operation index
            ax.text(x, y - 0.4, f"Op {i}", ha='center', va='center', 
                   fontsize=8, color='black')
            
            # Draw arrows between operations
            if i < len(operations) - 1:
                ax.arrow(x + box_width/2, y, 1 - box_width, 0,
                        head_width=0.1, head_length=0.1, 
                        fc='black', ec='black', alpha=0.6)
        
        # Add input and output
        # Input
        input_box = FancyBboxPatch(
            (-1 - box_width/2, -box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.1",
            facecolor='#2ecc71',
            edgecolor='black',
            linewidth=1.5,
            alpha=0.8
        )
        ax.add_patch(input_box)
        ax.text(-1, 0, 'Input\n3×32×32', ha='center', va='center', 
               fontsize=10, fontweight='bold', color='white')
        
        # Arrow from input
        ax.arrow(-1 + box_width/2, 0, 1 - box_width, 0,
                head_width=0.1, head_length=0.1, 
                fc='black', ec='black', alpha=0.6)
        
        # Output
        output_x = len(operations)
        output_box = FancyBboxPatch(
            (output_x - box_width/2, -box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.1",
            facecolor='#e74c3c',
            edgecolor='black',
            linewidth=1.5,
            alpha=0.8
        )
        ax.add_patch(output_box)
        ax.text(output_x, 0, 'Output\n10 classes', ha='center', va='center', 
               fontsize=10, fontweight='bold', color='white')
        
        # Arrow to output
        ax.arrow(len(operations) - 1 + box_width/2, 0, 1 - box_width, 0,
                head_width=0.1, head_length=0.1, 
                fc='black', ec='black', alpha=0.6)
        
        # Set limits and labels
        ax.set_xlim(-2, len(operations) + 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Add legend
        legend_elements = []
        for op_type, color in self.operation_colors.items():
            legend_elements.append(patches.Patch(color=color, label=op_type.title()))
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Save plot
        plot_path = os.path.join(self.output_dir, f'{title.lower().replace(" ", "_")}.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved to: {plot_path}")
        return plot_path
    
    def visualize_architecture_graph(self, architecture: nanonas.Architecture,
                                   title: str = "Architecture Graph") -> str:
        """Visualize architecture as a directed graph (if applicable)."""
        print(f"🕸️  Creating graph visualization: {title}")
        
        if not HAS_NETWORKX:
            print("   ⚠️  NetworkX not available")
            return ""
        
        # Create graph representation
        if architecture.graph is not None:
            G = architecture.graph.copy()
        elif architecture.encoding is not None:
            # Create linear graph from sequence
            G = nx.DiGraph()
            for i in range(len(architecture.encoding)):
                G.add_node(i, operation=architecture.encoding[i])
                if i > 0:
                    G.add_edge(i-1, i)
        else:
            print("   ⚠️  Cannot create graph visualization")
            return ""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create layout
        if len(G.nodes()) <= 10:
            pos = nx.spring_layout(G, k=2, iterations=50)
        else:
            pos = nx.shell_layout(G)
        
        # Draw nodes with colors based on operations
        for node in G.nodes():
            op_idx = G.nodes[node].get('operation', 0)
            if op_idx < len(architecture.search_space.operations):
                op = architecture.search_space.operations[op_idx]
                color = self.operation_colors.get(op.type, '#95a5a6')
                
                nx.draw_networkx_nodes(G, pos, nodelist=[node], 
                                     node_color=[color], node_size=1000,
                                     alpha=0.8, ax=ax)
                
                # Add operation labels
                x, y = pos[node]
                ax.text(x, y, f"{node}\n{op.name}", ha='center', va='center',
                       fontsize=8, fontweight='bold', color='white')
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='black', 
                             arrows=True, arrowsize=20, 
                             alpha=0.6, ax=ax)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Save plot
        plot_path = os.path.join(self.output_dir, f'{title.lower().replace(" ", "_")}.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved to: {plot_path}")
        return plot_path
    
    def visualize_architecture_comparison(self, architectures: List[nanonas.Architecture],
                                        labels: List[str] = None) -> str:
        """Compare multiple architectures side by side."""
        print("📊 Creating architecture comparison")
        
        if labels is None:
            labels = [f"Architecture {i+1}" for i in range(len(architectures))]
        
        n_archs = len(architectures)
        fig, axes = plt.subplots(n_archs, 1, figsize=(14, 3*n_archs))
        
        if n_archs == 1:
            axes = [axes]
        
        for i, (arch, label) in enumerate(zip(architectures, labels)):
            ax = axes[i]
            
            if arch.encoding is None:
                ax.text(0.5, 0.5, "No sequence encoding", ha='center', va='center')
                ax.set_title(label)
                continue
            
            operations = [arch.search_space.operations[j] for j in arch.encoding]
            
            # Draw operation sequence
            for j, op in enumerate(operations):
                color = self.operation_colors.get(op.type, '#95a5a6')
                
                rect = patches.Rectangle((j, 0), 0.8, 0.8, 
                                       facecolor=color, alpha=0.8,
                                       edgecolor='black')
                ax.add_patch(rect)
                
                ax.text(j + 0.4, 0.4, op.name[:8], ha='center', va='center',
                       fontsize=8, fontweight='bold', color='white')
            
            ax.set_xlim(-0.2, len(operations))
            ax.set_ylim(-0.2, 1)
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.set_aspect('equal')
            ax.axis('off')
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'architecture_comparison.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved to: {plot_path}")
        return plot_path
    
    def visualize_complexity_analysis(self, architectures: List[nanonas.Architecture],
                                    labels: List[str] = None) -> str:
        """Visualize complexity metrics of architectures."""
        print("📈 Creating complexity analysis")
        
        if labels is None:
            labels = [f"Arch {i+1}" for i in range(len(architectures))]
        
        # Calculate metrics
        metrics_data = []
        for arch in architectures:
            complexity = arch.get_complexity_metrics()
            metrics_data.append(complexity)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Depth comparison
        depths = [m['depth'] for m in metrics_data]
        axes[0, 0].bar(labels, depths, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Architecture Depth')
        axes[0, 0].set_ylabel('Number of Operations')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Operation cost
        costs = [m.get('total_op_cost', 0) for m in metrics_data]
        axes[0, 1].bar(labels, costs, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Computational Cost')
        axes[0, 1].set_ylabel('Relative Cost')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. FLOPs
        flops = [m.get('total_flops', 0) for m in metrics_data]
        axes[1, 0].bar(labels, flops, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('Estimated FLOPs')
        axes[1, 0].set_ylabel('FLOPs (millions)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Skip connection ratio
        skip_ratios = [m.get('skip_ratio', 0) for m in metrics_data]
        axes[1, 1].bar(labels, skip_ratios, color='orange', alpha=0.7)
        axes[1, 1].set_title('Skip Connection Ratio')
        axes[1, 1].set_ylabel('Ratio')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'complexity_analysis.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved to: {plot_path}")
        return plot_path
    
    def create_interactive_visualization(self, architectures: List[nanonas.Architecture],
                                       labels: List[str] = None) -> str:
        """Create interactive visualization using Plotly."""
        if not HAS_PLOTLY:
            print("   ⚠️  Plotly not available for interactive visualization")
            return ""
        
        print("🎯 Creating interactive visualization")
        
        if labels is None:
            labels = [f"Architecture {i+1}" for i in range(len(architectures))]
        
        # Prepare data
        metrics_data = []
        for i, arch in enumerate(architectures):
            complexity = arch.get_complexity_metrics()
            complexity['name'] = labels[i]
            complexity['index'] = i
            metrics_data.append(complexity)
        
        # Create interactive plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy vs Complexity', 'FLOPs vs Parameters', 
                          'Architecture Depth', 'Operation Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "pie"}]]
        )
        
        # Plot 1: Accuracy vs Complexity (simulated)
        complexities = [m.get('total_op_cost', 0) for m in metrics_data]
        accuracies = [np.random.normal(92, 2) for _ in range(len(architectures))]  # Simulated
        
        fig.add_trace(
            go.Scatter(x=complexities, y=accuracies, mode='markers+text',
                      text=labels, textposition="top center",
                      marker=dict(size=10, color='blue', opacity=0.7),
                      name='Architectures'),
            row=1, col=1
        )
        
        # Plot 2: FLOPs vs Parameters
        flops = [m.get('total_flops', 0) for m in metrics_data]
        params = [m['depth'] * 1000 for m in metrics_data]  # Simplified
        
        fig.add_trace(
            go.Scatter(x=params, y=flops, mode='markers+text',
                      text=labels, textposition="top center",
                      marker=dict(size=10, color='red', opacity=0.7),
                      name='FLOPs vs Params'),
            row=1, col=2
        )
        
        # Plot 3: Architecture depths
        depths = [m['depth'] for m in metrics_data]
        fig.add_trace(
            go.Bar(x=labels, y=depths, name='Depth',
                  marker_color='green', opacity=0.7),
            row=2, col=1
        )
        
        # Plot 4: Operation type distribution (for first architecture)
        if architectures and architectures[0].encoding:
            arch = architectures[0]
            op_types = [arch.search_space.operations[i].type for i in arch.encoding]
            op_counts = {}
            for op_type in op_types:
                op_counts[op_type] = op_counts.get(op_type, 0) + 1
            
            fig.add_trace(
                go.Pie(labels=list(op_counts.keys()), 
                      values=list(op_counts.values()),
                      name="Operations"),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="nanoNAS Architecture Analysis Dashboard",
            showlegend=False,
            height=800
        )
        
        # Save interactive plot
        plot_path = os.path.join(self.output_dir, 'interactive_analysis.html')
        fig.write_html(plot_path)
        
        print(f"   ✅ Saved to: {plot_path}")
        return plot_path
    
    def demonstrate_visualizations(self):
        """Run demonstration of all visualization capabilities."""
        print("🎨 nanoNAS Architecture Visualization Demo")
        print("=" * 60)
        
        # Create sample architectures
        search_space = nanonas.SearchSpace.get_nano_search_space()
        
        # Different architecture types
        architectures = [
            nanonas.Architecture([0, 1, 2, 3, 4], search_space),  # Sequential
            nanonas.Architecture([1, 0, 4, 2, 1], search_space),  # Different pattern
            nanonas.Architecture([2, 2, 3, 0, 4], search_space),  # Pool-heavy
        ]
        
        labels = ['Sequential Pattern', 'Mixed Operations', 'Pool-Heavy']
        
        # Generate all visualizations
        plot_paths = []
        
        # Individual sequence plots
        for i, (arch, label) in enumerate(zip(architectures, labels)):
            path = self.visualize_architecture_sequence(arch, f"{label} Sequence")
            if path:
                plot_paths.append(path)
        
        # Comparison plot
        path = self.visualize_architecture_comparison(architectures, labels)
        if path:
            plot_paths.append(path)
        
        # Complexity analysis
        path = self.visualize_complexity_analysis(architectures, labels)
        if path:
            plot_paths.append(path)
        
        # Interactive visualization
        path = self.create_interactive_visualization(architectures, labels)
        if path:
            plot_paths.append(path)
        
        print(f"\n🎉 Generated {len(plot_paths)} visualizations!")
        print(f"Check {self.output_dir}/ for all plots")
        
        return plot_paths


def main():
    """Run the architecture visualization example."""
    print("🚀 Starting nanoNAS Architecture Visualization Demo")
    
    # Create visualizer
    visualizer = ArchitectureVisualizer()
    
    # Run demonstration
    plot_paths = visualizer.demonstrate_visualizations()
    
    print(f"\nVisualization Complete!")
    print(f"Generated files:")
    for path in plot_paths:
        print(f"  • {path}")
    
    print(f"\nNext steps:")
    print(f"  • Open the generated images to see architecture visualizations")
    print(f"  • Try the interactive HTML file in your browser")
    print(f"  • Modify architectures and re-run to see different patterns")


if __name__ == "__main__":
    main() 