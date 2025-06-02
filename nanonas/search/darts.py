"""
DARTS: Differentiable Architecture Search
========================================

Implementation of Differentiable Architecture Search (DARTS) by Liu et al.
This module provides a gradient-based approach to neural architecture search
using continuous relaxation of the search space.

Key Features:
- Mixed operations with softmax-based weighting
- Bilevel optimization (architecture and weight parameters)
- Continuous to discrete architecture conversion
- Memory-efficient implementation with gradient approximation

Reference:
Liu, H., Simonyan, K., & Yang, Y. (2018). DARTS: Differentiable architecture search.
arXiv preprint arXiv:1806.09055.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

from ..core.base import BaseSearchStrategy
from ..core.architecture import Architecture, SearchSpace
from ..core.config import ExperimentConfig
from ..models.operations import MixedOperation, create_operation
from ..models.supernet import DARTSSupernet
from ..benchmarks.evaluator import ModelEvaluator


class DARTSCell(nn.Module):
    """
    DARTS cell with mixed operations and learnable architecture parameters.
    
    Each cell contains a set of nodes connected by mixed operations,
    where the operation weights are learned through gradient descent.
    """
    
    def __init__(self, 
                 search_space: SearchSpace,
                 channels: int,
                 num_nodes: int = 4,
                 concat_nodes: List[int] = None):
        """
        Initialize DARTS cell.
        
        Args:
            search_space: Search space containing operations
            channels: Number of input/output channels
            num_nodes: Number of intermediate nodes in the cell
            concat_nodes: Nodes to concatenate for output (default: last 2)
        """
        super().__init__()
        
        self.search_space = search_space
        self.channels = channels
        self.num_nodes = num_nodes
        self.concat_nodes = concat_nodes or list(range(num_nodes - 2, num_nodes))
        
        # Create mixed operations for each edge
        self.mixed_ops = nn.ModuleDict()
        self.edge_indices = []
        
        # Each node receives inputs from all previous nodes
        for i in range(2, num_nodes + 2):  # 0,1 are input nodes
            for j in range(i):
                edge_key = f"{j}_{i}"
                self.edge_indices.append((j, i))
                
                # Create mixed operation for this edge
                self.mixed_ops[edge_key] = MixedOperation(
                    operations=search_space.operations,
                    channels=channels
                )
        
        # Architecture parameters (alpha) - learnable weights for operations
        self.num_ops = len(search_space.operations)
        self.alphas = nn.Parameter(
            torch.randn(len(self.edge_indices), self.num_ops) * 1e-3
        )
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the DARTS cell.
        
        Args:
            x1: First input tensor
            x2: Second input tensor
            
        Returns:
            Output tensor from concatenated nodes
        """
        # Initialize node states
        states = [x1, x2]
        
        # Process each intermediate node
        for i in range(2, self.num_nodes + 2):
            node_sum = 0
            
            # Aggregate inputs from all previous nodes
            for j in range(i):
                edge_key = f"{j}_{i}"
                edge_idx = self.edge_indices.index((j, i))
                
                # Get operation weights for this edge
                edge_weights = F.softmax(self.alphas[edge_idx], dim=0)
                
                # Apply mixed operation
                mixed_output = self.mixed_ops[edge_key](states[j], edge_weights)
                node_sum = node_sum + mixed_output
            
            states.append(node_sum)
        
        # Concatenate specified nodes for output
        outputs = [states[i] for i in self.concat_nodes]
        return torch.cat(outputs, dim=1)
    
    def get_architecture_params(self) -> torch.Tensor:
        """Get architecture parameters (alphas)."""
        return self.alphas
    
    def get_discrete_architecture(self) -> List[Tuple[int, int, int]]:
        """
        Extract discrete architecture from learned alphas.
        
        Returns:
            List of (from_node, to_node, operation_idx) tuples
        """
        discrete_arch = []
        
        for edge_idx, (from_node, to_node) in enumerate(self.edge_indices):
            # Get the operation with highest weight
            op_weights = F.softmax(self.alphas[edge_idx], dim=0)
            best_op_idx = op_weights.argmax().item()
            
            discrete_arch.append((from_node, to_node, best_op_idx))
        
        return discrete_arch


class DARTSSearch(BaseSearchStrategy):
    """
    DARTS (Differentiable Architecture Search) implementation.
    
    This class implements the bilevel optimization approach from the DARTS paper,
    alternating between optimizing network weights and architecture parameters.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize DARTS search strategy.
        
        Args:
            config: Experiment configuration
        """
        super().__init__(config)
        
        # DARTS-specific configuration
        self.darts_config = self._create_darts_config()
        
        # Get search space
        if self.config.search.search_space == "nano":
            self.search_space = SearchSpace.get_nano_search_space()
        elif self.config.search.search_space == "mobile":
            self.search_space = SearchSpace.get_mobile_search_space()
        else:
            self.search_space = SearchSpace.get_nano_search_space()
        
        # Initialize supernet
        self.supernet = None
        self.device = torch.device(config.device)
        
        # Optimizers
        self.weight_optimizer = None
        self.arch_optimizer = None
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        
        # Search state
        self.current_epoch = 0
        self.architecture_history = []
        self.loss_history = {'train': [], 'val': []}
        
        # Setup evaluator
        self.evaluator = ModelEvaluator(config)
        
    def _create_darts_config(self) -> Dict[str, Any]:
        """Create DARTS-specific configuration."""
        return {
            'epochs': self.config.search.darts_epochs,
            'weight_lr': self.config.search.darts_lr,
            'weight_decay': self.config.search.darts_weight_decay,
            'arch_lr': 3e-4,  # Architecture learning rate
            'arch_weight_decay': 1e-3,
            'num_cells': 8,
            'num_nodes_per_cell': 4,
            'channels': 16,
            'layers': 8,
            'grad_clip': 5.0,
            'train_portion': 0.5,  # Portion of train data for weight training
        }
    
    def search(self) -> Architecture:
        """
        Run DARTS search to find the best architecture.
        
        Returns:
            Best architecture found during search
        """
        self.logger.info("🔍 Starting DARTS search...")
        self.logger.info(f"📊 Search space: {self.search_space.name}")
        self.logger.info(f"⏰ DARTS epochs: {self.darts_config['epochs']}")
        
        start_time = time.time()
        
        try:
            # Setup search components
            self._setup_search()
            
            # Main DARTS loop
            for epoch in range(self.darts_config['epochs']):
                self.current_epoch = epoch
                
                # Training phase
                train_loss, train_acc = self._train_epoch()
                
                # Validation phase
                val_loss, val_acc = self._validate_epoch()
                
                # Record history
                self.loss_history['train'].append(train_loss)
                self.loss_history['val'].append(val_loss)
                
                # Save architecture snapshots
                if epoch % 10 == 0:
                    current_arch = self._extract_current_architecture()
                    self.architecture_history.append({
                        'epoch': epoch,
                        'architecture': current_arch,
                        'val_accuracy': val_acc
                    })
                
                # Logging
                if epoch % self.config.log_frequency == 0:
                    self.logger.info(
                        f"Epoch {epoch + 1}/{self.darts_config['epochs']}: "
                        f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                        f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
                    )
                
                # Early stopping check
                if self._should_early_stop():
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            # Extract final architecture
            best_architecture = self._extract_final_architecture()
            
            search_time = time.time() - start_time
            self.search_metrics.update({
                'total_search_time': search_time,
                'epochs_completed': self.current_epoch + 1,
                'final_train_loss': self.loss_history['train'][-1],
                'final_val_loss': self.loss_history['val'][-1],
                'architecture_history': self.architecture_history,
                'loss_history': self.loss_history,
            })
            
            self.logger.info(f"✅ DARTS search completed in {search_time:.2f}s")
            self.logger.info(f"🎯 Final architecture: {best_architecture}")
            
            return best_architecture
            
        except Exception as e:
            self.logger.error(f"❌ DARTS search failed: {e}")
            raise
    
    def _setup_search(self):
        """Setup all components needed for DARTS search."""
        # Create supernet
        self.supernet = DARTSSupernet(
            search_space=self.search_space,
            num_cells=self.darts_config['num_cells'],
            num_nodes_per_cell=self.darts_config['num_nodes_per_cell'],
            channels=self.darts_config['channels'],
            num_classes=self.config.model.num_classes,
            layers=self.darts_config['layers']
        ).to(self.device)
        
        # Setup data loaders
        self._setup_data_loaders()
        
        # Setup optimizers
        self._setup_optimizers()
        
        self.logger.info(f"✅ DARTS setup completed")
        self.logger.info(f"   Supernet parameters: {sum(p.numel() for p in self.supernet.parameters()):,}")
        self.logger.info(f"   Architecture parameters: {sum(p.numel() for p in self.supernet.get_architecture_params()):,}")
    
    def _setup_data_loaders(self):
        """Setup train and validation data loaders for bilevel optimization."""
        from ..benchmarks.datasets import get_dataset
        
        # Get full data loaders
        full_train_loader, val_loader, _ = get_dataset(
            self.config.dataset.name,
            batch_size=self.config.dataset.batch_size,
            num_workers=self.config.dataset.num_workers,
            use_augmentation=self.config.dataset.use_augmentation
        )
        
        # Split training data for bilevel optimization
        full_dataset = full_train_loader.dataset
        train_size = int(self.darts_config['train_portion'] * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, arch_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.dataset.batch_size,
            shuffle=True,
            num_workers=self.config.dataset.num_workers,
            pin_memory=True
        )
        
        self.arch_loader = torch.utils.data.DataLoader(
            arch_dataset,
            batch_size=self.config.dataset.batch_size,
            shuffle=True,
            num_workers=self.config.dataset.num_workers,
            pin_memory=True
        )
        
        self.val_loader = val_loader
    
    def _setup_optimizers(self):
        """Setup optimizers for weights and architecture parameters."""
        # Weight optimizer
        self.weight_optimizer = optim.SGD(
            self.supernet.get_weight_params(),
            lr=self.darts_config['weight_lr'],
            momentum=0.9,
            weight_decay=self.darts_config['weight_decay']
        )
        
        # Architecture optimizer
        self.arch_optimizer = optim.Adam(
            self.supernet.get_architecture_params(),
            lr=self.darts_config['arch_lr'],
            weight_decay=self.darts_config['arch_weight_decay']
        )
    
    def _train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch with bilevel optimization.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.supernet.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Create iterators
        train_iter = iter(self.train_loader)
        arch_iter = iter(self.arch_loader)
        
        for step in range(len(self.train_loader)):
            # Get training and architecture data
            try:
                train_data, train_target = next(train_iter)
                arch_data, arch_target = next(arch_iter)
            except StopIteration:
                # Reset iterators if exhausted
                arch_iter = iter(self.arch_loader)
                arch_data, arch_target = next(arch_iter)
            
            train_data = train_data.to(self.device)
            train_target = train_target.to(self.device)
            arch_data = arch_data.to(self.device)
            arch_target = arch_target.to(self.device)
            
            # Phase 1: Update architecture parameters
            self.arch_optimizer.zero_grad()
            arch_output = self.supernet(arch_data)
            arch_loss = F.cross_entropy(arch_output, arch_target)
            arch_loss.backward()
            nn.utils.clip_grad_norm_(
                self.supernet.get_architecture_params(), 
                self.darts_config['grad_clip']
            )
            self.arch_optimizer.step()
            
            # Phase 2: Update weight parameters
            self.weight_optimizer.zero_grad()
            train_output = self.supernet(train_data)
            train_loss = F.cross_entropy(train_output, train_target)
            train_loss.backward()
            nn.utils.clip_grad_norm_(
                self.supernet.get_weight_params(), 
                self.darts_config['grad_clip']
            )
            self.weight_optimizer.step()
            
            # Statistics
            total_loss += train_loss.item()
            pred = train_output.argmax(dim=1)
            correct += pred.eq(train_target).sum().item()
            total += train_target.size(0)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate_epoch(self) -> Tuple[float, float]:
        """
        Validate the current supernet.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.supernet.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.supernet(data)
                loss = F.cross_entropy(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _should_early_stop(self) -> bool:
        """Check if search should stop early."""
        if len(self.loss_history['val']) < 10:
            return False
        
        # Check for validation loss stagnation
        recent_losses = self.loss_history['val'][-10:]
        improvement = max(recent_losses) - min(recent_losses)
        
        return improvement < 1e-4
    
    def _extract_current_architecture(self) -> Architecture:
        """Extract current architecture from supernet."""
        # Get discrete architecture from each cell
        discrete_cells = []
        for cell in self.supernet.cells:
            if isinstance(cell, DARTSCell):
                discrete_cells.append(cell.get_discrete_architecture())
        
        # Convert to Architecture object
        # For simplicity, we'll use the first cell's architecture
        if discrete_cells:
            # Extract operation sequence from first cell
            first_cell = discrete_cells[0]
            operation_sequence = [op_idx for _, _, op_idx in first_cell]
            
            return Architecture(
                encoding=operation_sequence,
                search_space=self.search_space,
                metadata={
                    'source': 'darts',
                    'epoch': self.current_epoch,
                    'discrete_cells': discrete_cells
                }
            )
        else:
            # Fallback to random architecture
            return self.search_space.sample_random_architecture()
    
    def _extract_final_architecture(self) -> Architecture:
        """Extract the final best architecture."""
        # Use the architecture from the best validation epoch
        if self.architecture_history:
            best_arch_info = max(
                self.architecture_history, 
                key=lambda x: x['val_accuracy']
            )
            return best_arch_info['architecture']
        else:
            return self._extract_current_architecture()
    
    def get_search_metrics(self) -> Dict[str, Any]:
        """Get comprehensive search metrics."""
        base_metrics = super().get_search_metrics()
        
        darts_metrics = {
            'search_strategy': 'darts',
            'supernet_params': sum(p.numel() for p in self.supernet.parameters()) if self.supernet else 0,
            'architecture_params': sum(p.numel() for p in self.supernet.get_architecture_params()) if self.supernet else 0,
            'bilevel_optimization': True,
            'grad_based_search': True,
        }
        
        return {**base_metrics, **darts_metrics} 