"""
Architecture Representation and Search Space Definition
======================================================

This module provides sophisticated architecture encoding and search space
management for neural architecture search.

Key Features:
- Multiple encoding schemes (list-based, graph-based, hierarchical)
- Flexible search space definitions with constraints
- Architecture mutation and crossover operations
- Performance caching and architecture hashing
- Visualization-ready architecture representations
"""

import torch
import torch.nn as nn
import numpy as np
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import networkx as nx


@dataclass
class OperationSpec:
    """Specification for a single operation in the search space."""
    name: str
    type: str  # 'conv', 'pool', 'norm', 'activation', 'skip', 'zero'
    params: Dict[str, Any] = field(default_factory=dict)
    computational_cost: float = 1.0  # Relative computational cost
    memory_cost: float = 1.0  # Relative memory cost
    
    def __hash__(self):
        return hash((self.name, self.type, json.dumps(self.params, sort_keys=True)))


class SearchSpace:
    """
    Defines the search space for neural architectures.
    
    Supports multiple search space paradigms:
    - Cell-based: Repeating cells with internal connectivity
    - Layer-based: Sequential layer choices
    - Graph-based: Arbitrary directed acyclic graphs
    """
    
    def __init__(self, 
                 name: str,
                 operations: List[OperationSpec],
                 constraints: Optional[Dict[str, Any]] = None,
                 encoding_type: str = "list"):
        """
        Initialize search space.
        
        Args:
            name: Name of the search space
            operations: List of available operations
            constraints: Optional constraints (max_depth, max_params, etc.)
            encoding_type: Type of encoding ('list', 'graph', 'hierarchical')
        """
        self.name = name
        self.operations = operations
        self.constraints = constraints or {}
        self.encoding_type = encoding_type
        
        # Create operation lookup
        self.op_dict = {op.name: op for op in operations}
        self.op_names = [op.name for op in operations]
        
        # Set default constraints
        self._set_default_constraints()
    
    def _set_default_constraints(self):
        """Set default constraints if not provided."""
        defaults = {
            'max_depth': 8,
            'max_params': 10e6,  # 10M parameters
            'max_flops': 600e6,  # 600M FLOPs
            'min_accuracy': 0.0,
            'skip_connection_prob': 0.3,
        }
        for key, value in defaults.items():
            if key not in self.constraints:
                self.constraints[key] = value
    
    def sample_random_architecture(self, 
                                 num_blocks: Optional[int] = None) -> 'Architecture':
        """Sample a random architecture from this search space."""
        if num_blocks is None:
            num_blocks = np.random.randint(3, self.constraints['max_depth'])
        
        if self.encoding_type == "list":
            encoding = [np.random.randint(0, len(self.operations)) 
                       for _ in range(num_blocks)]
            return Architecture(encoding=encoding, search_space=self)
        elif self.encoding_type == "graph":
            return self._sample_graph_architecture(num_blocks)
        else:
            raise NotImplementedError(f"Encoding type {self.encoding_type} not implemented")
    
    def _sample_graph_architecture(self, num_nodes: int) -> 'Architecture':
        """Sample a graph-based architecture."""
        # Create random DAG
        graph = nx.DiGraph()
        
        # Add nodes with random operations
        for i in range(num_nodes):
            op_idx = np.random.randint(0, len(self.operations))
            graph.add_node(i, operation=op_idx)
        
        # Add random edges (ensuring DAG property)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if np.random.random() < 0.3:  # 30% connection probability
                    graph.add_edge(i, j)
        
        return Architecture(graph=graph, search_space=self)
    
    @classmethod
    def get_nano_search_space(cls) -> 'SearchSpace':
        """Get the nano search space (minimal but effective)."""
        operations = [
            OperationSpec("conv3x3", "conv", {"kernel_size": 3, "padding": 1}, 1.0, 1.0),
            OperationSpec("conv5x5", "conv", {"kernel_size": 5, "padding": 2}, 2.0, 1.5),
            OperationSpec("maxpool3x3", "pool", {"kernel_size": 3, "padding": 1}, 0.1, 0.1),
            OperationSpec("skip", "skip", {}, 0.0, 0.0),
            OperationSpec("zero", "zero", {}, 0.0, 0.0),
        ]
        
        constraints = {
            'max_depth': 6,
            'max_params': 1e6,
            'max_flops': 100e6,
        }
        
        return cls("nano", operations, constraints, "list")
    
    @classmethod
    def get_mobile_search_space(cls) -> 'SearchSpace':
        """Get MobileNet-inspired search space."""
        operations = [
            OperationSpec("conv1x1", "conv", {"kernel_size": 1}, 0.5, 0.5),
            OperationSpec("conv3x3", "conv", {"kernel_size": 3, "padding": 1}, 1.0, 1.0),
            OperationSpec("conv5x5", "conv", {"kernel_size": 5, "padding": 2}, 2.0, 1.5),
            OperationSpec("dw_conv3x3", "conv", {"kernel_size": 3, "padding": 1, "groups": "input_channels"}, 0.3, 0.8),
            OperationSpec("dw_conv5x5", "conv", {"kernel_size": 5, "padding": 2, "groups": "input_channels"}, 0.6, 1.2),
            OperationSpec("maxpool3x3", "pool", {"kernel_size": 3, "padding": 1}, 0.1, 0.1),
            OperationSpec("avgpool3x3", "pool", {"kernel_size": 3, "padding": 1, "type": "avg"}, 0.1, 0.1),
            OperationSpec("skip", "skip", {}, 0.0, 0.0),
            OperationSpec("zero", "zero", {}, 0.0, 0.0),
        ]
        
        constraints = {
            'max_depth': 12,
            'max_params': 5e6,
            'max_flops': 300e6,
        }
        
        return cls("mobile", operations, constraints, "list")


class Architecture:
    """
    Flexible architecture representation supporting multiple encodings.
    
    This class can represent architectures as:
    - List encoding: [op1, op2, op3, ...] (traditional)
    - Graph encoding: NetworkX DAG with operation labels
    - Hierarchical encoding: Nested structure for complex architectures
    """
    
    def __init__(self, 
                 encoding: Optional[List[int]] = None,
                 graph: Optional[nx.DiGraph] = None,
                 search_space: Optional[SearchSpace] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize architecture.
        
        Args:
            encoding: List-based encoding (for list-type search spaces)
            graph: Graph-based encoding (for graph-type search spaces)
            search_space: The search space this architecture belongs to
            metadata: Additional metadata (performance metrics, etc.)
        """
        self.encoding = encoding
        self.graph = graph
        self.search_space = search_space or SearchSpace.get_nano_search_space()
        self.metadata = metadata or {}
        
        # Performance cache
        self._performance_cache = {}
        self._model_cache = None
        self._hash_cache = None
        
        # Validate architecture
        self._validate()
    
    def _validate(self):
        """Validate the architecture representation."""
        if self.encoding is None and self.graph is None:
            raise ValueError("Architecture must have either encoding or graph")
        
        if self.encoding is not None:
            # Validate list encoding
            if not all(0 <= op < len(self.search_space.operations) for op in self.encoding):
                raise ValueError("Invalid operation indices in encoding")
        
        if self.graph is not None:
            # Validate graph encoding
            if not nx.is_directed_acyclic_graph(self.graph):
                raise ValueError("Architecture graph must be a DAG")
    
    def to_model(self, 
                 input_channels: int = 3,
                 num_classes: int = 10,
                 base_channels: int = 16) -> nn.Module:
        """Convert architecture to PyTorch model."""
        if self._model_cache is not None:
            return self._model_cache
        
        if self.encoding is not None:
            model = self._build_list_model(input_channels, num_classes, base_channels)
        elif self.graph is not None:
            model = self._build_graph_model(input_channels, num_classes, base_channels)
        else:
            raise ValueError("No valid encoding found")
        
        self._model_cache = model
        return model
    
    def _build_list_model(self, input_channels: int, num_classes: int, base_channels: int) -> nn.Module:
        """Build model from list encoding."""
        from ..models.networks import SequentialNet
        
        operations = []
        current_channels = base_channels
        
        for op_idx in self.encoding:
            op_spec = self.search_space.operations[op_idx]
            operation = self._create_operation(op_spec, current_channels)
            operations.append(operation)
        
        return SequentialNet(
            operations=operations,
            input_channels=input_channels,
            num_classes=num_classes,
            base_channels=base_channels
        )
    
    def _build_graph_model(self, input_channels: int, num_classes: int, base_channels: int) -> nn.Module:
        """Build model from graph encoding."""
        from ..models.networks import GraphNet
        
        # Convert graph to model
        return GraphNet(
            graph=self.graph,
            search_space=self.search_space,
            input_channels=input_channels,
            num_classes=num_classes,
            base_channels=base_channels
        )
    
    def _create_operation(self, op_spec: OperationSpec, channels: int):
        """Create a PyTorch operation from specification."""
        from ..models.operations import create_operation
        return create_operation(op_spec, channels)
    
    def mutate(self, mutation_rate: float = 0.1, mutation_std: float = 1.0) -> 'Architecture':
        """
        Create a mutated version of this architecture.
        
        Args:
            mutation_rate: Probability of mutating each component
            mutation_std: Standard deviation for mutation strength
        
        Returns:
            New mutated architecture
        """
        if self.encoding is not None:
            return self._mutate_list_encoding(mutation_rate, mutation_std)
        elif self.graph is not None:
            return self._mutate_graph_encoding(mutation_rate, mutation_std)
        else:
            raise ValueError("No valid encoding to mutate")
    
    def _mutate_list_encoding(self, mutation_rate: float, mutation_std: float) -> 'Architecture':
        """Mutate list-based encoding."""
        new_encoding = []
        num_ops = len(self.search_space.operations)
        
        for op_idx in self.encoding:
            if np.random.random() < mutation_rate:
                # Gaussian mutation with clipping
                delta = int(np.random.normal(0, mutation_std))
                new_op = np.clip(op_idx + delta, 0, num_ops - 1)
                new_encoding.append(new_op)
            else:
                new_encoding.append(op_idx)
        
        return Architecture(
            encoding=new_encoding,
            search_space=self.search_space,
            metadata=self.metadata.copy()
        )
    
    def _mutate_graph_encoding(self, mutation_rate: float, mutation_std: float) -> 'Architecture':
        """Mutate graph-based encoding."""
        new_graph = self.graph.copy()
        num_ops = len(self.search_space.operations)
        
        # Mutate node operations
        for node in new_graph.nodes():
            if np.random.random() < mutation_rate:
                current_op = new_graph.nodes[node]['operation']
                delta = int(np.random.normal(0, mutation_std))
                new_op = np.clip(current_op + delta, 0, num_ops - 1)
                new_graph.nodes[node]['operation'] = new_op
        
        # Mutate edges (add/remove)
        nodes = list(new_graph.nodes())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if np.random.random() < mutation_rate * 0.1:  # Lower rate for edges
                    if new_graph.has_edge(nodes[i], nodes[j]):
                        new_graph.remove_edge(nodes[i], nodes[j])
                    else:
                        new_graph.add_edge(nodes[i], nodes[j])
        
        return Architecture(
            graph=new_graph,
            search_space=self.search_space,
            metadata=self.metadata.copy()
        )
    
    def crossover(self, other: 'Architecture') -> Tuple['Architecture', 'Architecture']:
        """
        Perform crossover with another architecture.
        
        Args:
            other: Another architecture for crossover
            
        Returns:
            Tuple of two offspring architectures
        """
        if self.encoding is not None and other.encoding is not None:
            return self._crossover_list_encoding(other)
        elif self.graph is not None and other.graph is not None:
            return self._crossover_graph_encoding(other)
        else:
            raise ValueError("Architectures must have compatible encodings for crossover")
    
    def _crossover_list_encoding(self, other: 'Architecture') -> Tuple['Architecture', 'Architecture']:
        """Perform crossover on list encodings."""
        min_len = min(len(self.encoding), len(other.encoding))
        
        if min_len < 2:
            return self, other
        
        # Single-point crossover
        crossover_point = np.random.randint(1, min_len)
        
        child1_encoding = (self.encoding[:crossover_point] + 
                          other.encoding[crossover_point:len(other.encoding)])
        child2_encoding = (other.encoding[:crossover_point] + 
                          self.encoding[crossover_point:len(self.encoding)])
        
        child1 = Architecture(
            encoding=child1_encoding,
            search_space=self.search_space,
            metadata={}
        )
        
        child2 = Architecture(
            encoding=child2_encoding,
            search_space=self.search_space,
            metadata={}
        )
        
        return child1, child2
    
    def _crossover_graph_encoding(self, other: 'Architecture') -> Tuple['Architecture', 'Architecture']:
        """Perform crossover on graph encodings."""
        # Simple node-based crossover
        nodes1 = set(self.graph.nodes())
        nodes2 = set(other.graph.nodes())
        all_nodes = nodes1.union(nodes2)
        
        # Randomly assign nodes to children
        child1_nodes = set(np.random.choice(list(all_nodes), 
                                          size=len(all_nodes)//2, 
                                          replace=False))
        child2_nodes = all_nodes - child1_nodes
        
        child1_graph = self.graph.subgraph(child1_nodes).copy()
        child2_graph = other.graph.subgraph(child2_nodes).copy()
        
        return (Architecture(graph=child1_graph, search_space=self.search_space),
                Architecture(graph=child2_graph, search_space=self.search_space))
    
    def get_complexity_metrics(self) -> Dict[str, float]:
        """Calculate various complexity metrics for the architecture."""
        if 'complexity' in self._performance_cache:
            return self._performance_cache['complexity']
        
        metrics = {}
        
        if self.encoding is not None:
            # List-based metrics
            metrics['depth'] = len(self.encoding)
            metrics['avg_op_cost'] = np.mean([
                self.search_space.operations[op_idx].computational_cost
                for op_idx in self.encoding
            ])
            metrics['total_op_cost'] = sum([
                self.search_space.operations[op_idx].computational_cost
                for op_idx in self.encoding
            ])
            metrics['skip_ratio'] = sum([
                1 for op_idx in self.encoding
                if self.search_space.operations[op_idx].type == 'skip'
            ]) / len(self.encoding)
            
        elif self.graph is not None:
            # Graph-based metrics
            metrics['num_nodes'] = self.graph.number_of_nodes()
            metrics['num_edges'] = self.graph.number_of_edges()
            metrics['avg_degree'] = (2 * self.graph.number_of_edges() / 
                                   max(1, self.graph.number_of_nodes()))
            metrics['max_path_length'] = max([
                len(path) for path in nx.all_simple_paths(
                    self.graph, 
                    source=min(self.graph.nodes()),
                    target=max(self.graph.nodes())
                )
            ]) if self.graph.number_of_nodes() > 1 else 0
        
        self._performance_cache['complexity'] = metrics
        return metrics
    
    def __hash__(self) -> int:
        """Generate hash for architecture (useful for caching and deduplication)."""
        if self._hash_cache is not None:
            return self._hash_cache
        
        if self.encoding is not None:
            hash_input = str(self.encoding)
        elif self.graph is not None:
            # Create canonical string representation of graph
            nodes_str = str(sorted([(n, self.graph.nodes[n].get('operation', 0)) 
                                  for n in self.graph.nodes()]))
            edges_str = str(sorted(self.graph.edges()))
            hash_input = nodes_str + edges_str
        else:
            hash_input = "empty"
        
        hash_input += str(self.search_space.name)
        self._hash_cache = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        return self._hash_cache
    
    def __eq__(self, other) -> bool:
        """Check equality with another architecture."""
        if not isinstance(other, Architecture):
            return False
        return hash(self) == hash(other)
    
    def __str__(self) -> str:
        """String representation of architecture."""
        if self.encoding is not None:
            op_names = [self.search_space.operations[idx].name for idx in self.encoding]
            return f"Architecture(ops={op_names})"
        elif self.graph is not None:
            return f"Architecture(graph={self.graph.number_of_nodes()}nodes_{self.graph.number_of_edges()}edges)"
        else:
            return "Architecture(empty)"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert architecture to dictionary for serialization."""
        data = {
            'search_space': self.search_space.name,
            'metadata': self.metadata,
        }
        
        if self.encoding is not None:
            data['encoding'] = self.encoding
        
        if self.graph is not None:
            data['graph'] = {
                'nodes': [(n, self.graph.nodes[n]) for n in self.graph.nodes()],
                'edges': list(self.graph.edges()),
            }
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], search_space: SearchSpace) -> 'Architecture':
        """Create architecture from dictionary."""
        encoding = data.get('encoding')
        graph_data = data.get('graph')
        
        graph = None
        if graph_data:
            graph = nx.DiGraph()
            for node, attrs in graph_data['nodes']:
                graph.add_node(node, **attrs)
            graph.add_edges_from(graph_data['edges'])
        
        return cls(
            encoding=encoding,
            graph=graph,
            search_space=search_space,
            metadata=data.get('metadata', {})
        ) 