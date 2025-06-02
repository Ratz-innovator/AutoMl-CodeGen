"""
Neural Network Builders for Architecture Search
==============================================

This module provides network builders that can construct PyTorch models
from architecture representations, supporting both sequential and graph-based
architectures.

Key Features:
- Sequential network builder for list-encoded architectures
- Graph network builder for DAG-encoded architectures
- Automatic model construction from Architecture objects
- Flexible input/output handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple

from ..core.base import BaseModel
from ..core.architecture import Architecture, SearchSpace
from .operations import create_operation, get_operation_by_name


class SequentialNet(BaseModel):
    """
    Sequential network builder for list-encoded architectures.
    
    Builds a feedforward network from a sequence of operations.
    """
    
    def __init__(self,
                 operations: List[Any],
                 input_channels: int = 3,
                 num_classes: int = 10,
                 base_channels: int = 16):
        """
        Initialize sequential network.
        
        Args:
            operations: List of operation indices or operation objects
            input_channels: Number of input channels
            num_classes: Number of output classes
            base_channels: Base number of channels for operations
        """
        super().__init__(input_channels, num_classes)
        
        self.base_channels = base_channels
        
        # Stem: initial convolution to get to base channels
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Main operations
        self.operations = nn.ModuleList()
        current_channels = base_channels
        
        for i, op in enumerate(operations):
            if isinstance(op, int):
                # Operation index - need to convert to actual operation
                # This is a placeholder - in practice, you'd use the search space
                op_layer = self._create_operation_from_index(op, current_channels)
            else:
                # Already an operation object
                op_layer = op
            
            self.operations.append(op_layer)
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(current_channels, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _create_operation_from_index(self, op_idx: int, channels: int) -> nn.Module:
        """Create operation from index (placeholder implementation)."""
        # This would use the actual search space operations
        operations = ["conv3x3", "conv5x5", "maxpool3x3", "skip", "zero"]
        op_name = operations[op_idx % len(operations)]
        
        try:
            return get_operation_by_name(op_name, channels)
        except ValueError:
            # Fallback to conv3x3
            return get_operation_by_name("conv3x3", channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through sequential network."""
        x = self.stem(x)
        
        for operation in self.operations:
            x = operation(x)
        
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class GraphNet(BaseModel):
    """
    Graph network builder for DAG-encoded architectures.
    
    Builds a network from a directed acyclic graph (DAG) representation.
    """
    
    def __init__(self,
                 graph: nx.DiGraph,
                 search_space: SearchSpace,
                 input_channels: int = 3,
                 num_classes: int = 10,
                 base_channels: int = 16):
        """
        Initialize graph network.
        
        Args:
            graph: NetworkX DAG representing the architecture
            search_space: Search space for operation lookup
            input_channels: Number of input channels
            num_classes: Number of output classes
            base_channels: Base number of channels
        """
        super().__init__(input_channels, num_classes)
        
        self.graph = graph
        self.search_space = search_space
        self.base_channels = base_channels
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Create operations for each node
        self.node_operations = nn.ModuleDict()
        for node in graph.nodes():
            if 'operation' in graph.nodes[node]:
                op_idx = graph.nodes[node]['operation']
                if op_idx < len(search_space.operations):
                    op_spec = search_space.operations[op_idx]
                    operation = create_operation(op_spec, base_channels)
                    self.node_operations[str(node)] = operation
        
        # Output processing
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(base_channels, num_classes)
        
        # Compute execution order
        self.execution_order = list(nx.topological_sort(graph))
        
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through graph network."""
        x = self.stem(x)
        
        # Execute nodes in topological order
        node_outputs = {}
        
        # Input nodes
        input_nodes = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
        for node in input_nodes:
            node_outputs[node] = x
        
        # Execute intermediate nodes
        for node in self.execution_order:
            if node in node_outputs:
                continue  # Already computed (input node)
            
            # Collect inputs from predecessor nodes
            predecessors = list(self.graph.predecessors(node))
            if len(predecessors) == 1:
                node_input = node_outputs[predecessors[0]]
            else:
                # Combine multiple inputs (simple addition)
                node_input = sum(node_outputs[pred] for pred in predecessors)
            
            # Apply operation
            if str(node) in self.node_operations:
                node_output = self.node_operations[str(node)](node_input)
            else:
                node_output = node_input  # Identity if no operation
            
            node_outputs[node] = node_output
        
        # Use output from the last node
        final_node = self.execution_order[-1]
        output = node_outputs[final_node]
        
        # Classification head
        output = self.global_pool(output)
        output = torch.flatten(output, 1)
        output = self.classifier(output)
        
        return output
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class CellBasedNet(BaseModel):
    """
    Cell-based network for DARTS-style architectures.
    
    Stacks multiple cells to create the final network.
    """
    
    def __init__(self,
                 cell_type: nn.Module,
                 num_cells: int = 8,
                 input_channels: int = 3,
                 num_classes: int = 10,
                 base_channels: int = 16,
                 reduction_layers: List[int] = None):
        """
        Initialize cell-based network.
        
        Args:
            cell_type: Type of cell to stack
            num_cells: Number of cells to stack
            input_channels: Number of input channels
            num_classes: Number of output classes
            base_channels: Base number of channels
            reduction_layers: Layers where spatial resolution is reduced
        """
        super().__init__(input_channels, num_classes)
        
        self.num_cells = num_cells
        self.base_channels = base_channels
        self.reduction_layers = reduction_layers or [num_cells // 3, 2 * num_cells // 3]
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base_channels),
        )
        
        # Cells
        self.cells = nn.ModuleList()
        current_channels = base_channels
        
        for i in range(num_cells):
            if i in self.reduction_layers:
                # Reduction cell (doubles channels, halves spatial resolution)
                cell = cell_type(current_channels, reduction=True)
                current_channels *= 2
            else:
                # Normal cell
                cell = cell_type(current_channels, reduction=False)
            
            self.cells.append(cell)
        
        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(current_channels, num_classes)
        
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through cell-based network."""
        x = self.stem(x)
        
        s0 = s1 = x
        
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
        
        x = self.global_pool(s1)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def build_model_from_architecture(architecture: Architecture,
                                 input_channels: int = 3,
                                 num_classes: int = 10,
                                 base_channels: int = 16) -> BaseModel:
    """
    Build a PyTorch model from an Architecture object.
    
    Args:
        architecture: Architecture representation
        input_channels: Number of input channels
        num_classes: Number of output classes
        base_channels: Base number of channels
        
    Returns:
        PyTorch model
    """
    if architecture.encoding is not None:
        # List-based architecture
        return SequentialNet(
            operations=architecture.encoding,
            input_channels=input_channels,
            num_classes=num_classes,
            base_channels=base_channels
        )
    elif architecture.graph is not None:
        # Graph-based architecture
        return GraphNet(
            graph=architecture.graph,
            search_space=architecture.search_space,
            input_channels=input_channels,
            num_classes=num_classes,
            base_channels=base_channels
        )
    else:
        raise ValueError("Architecture must have either encoding or graph representation") 