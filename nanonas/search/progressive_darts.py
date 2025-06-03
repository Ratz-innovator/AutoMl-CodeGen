"""
Progressive-DARTS: DARTS with Early Stopping and Progressive Pruning
================================================================

This module implements Progressive-DARTS, an enhanced version of DARTS that incorporates:
- Early stopping based on architecture parameter convergence
- Progressive pruning of weak operations during search
- Dynamic learning rate scheduling for architecture parameters
- Improved stability through regularization techniques

Key Features:
- Adaptive architecture parameter learning rates
- Operation strength-based pruning
- Multi-stage progressive search
- Convergence monitoring and early stopping
- Memory-efficient implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
from copy import deepcopy

from ..core.base import BaseSearchStrategy
from ..core.architecture import Architecture, SearchSpace
from ..core.config import ExperimentConfig
from ..models.supernet import DARTSSupernet
from ..benchmarks.evaluator import ModelEvaluator
from .darts import DARTSSearch

logger = logging.getLogger(__name__)


@dataclass
class ProgressiveDARTSConfig:
    """Configuration for Progressive-DARTS."""
    
    # Basic DARTS parameters
    epochs: int = 100
    learning_rate: float = 0.025
    learning_rate_min: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 3e-4
    
    # Architecture parameter optimization
    arch_learning_rate: float = 3e-4
    arch_weight_decay: float = 1e-3
    
    # Progressive pruning parameters
    pruning_stages: int = 4
    pruning_start_epoch: int = 20
    pruning_threshold: float = 0.01  # Minimum operation strength to keep
    operations_to_keep: int = 2  # Minimum operations per edge
    
    # Early stopping parameters
    early_stopping_patience: int = 20
    convergence_threshold: float = 1e-4
    min_epochs: int = 50
    
    # Regularization
    auxiliary_weight: float = 0.4
    drop_path_prob: float = 0.2
    grad_clip: float = 5.0
    
    # Progressive search stages
    warmup_epochs: int = 15
    progressive_stages: List[int] = None
    
    def __post_init__(self):
        """Initialize progressive stages if not provided."""
        if self.progressive_stages is None:
            # Default progressive stages
            self.progressive_stages = [
                self.warmup_epochs,
                self.epochs // 4,
                self.epochs // 2,
                3 * self.epochs // 4,
                self.epochs
            ]


class ProgressiveDARTSSearch(DARTSSearch):
    """
    Progressive-DARTS with early stopping and progressive pruning.
    
    This enhanced version of DARTS includes:
    - Progressive operation pruning based on architecture parameters
    - Early stopping when architecture parameters converge
    - Dynamic learning rate scheduling
    - Improved stability through regularization
    """
    
    def __init__(self, config: ExperimentConfig):
        """Initialize Progressive-DARTS search."""
        super().__init__(config)
        
        # Progressive-DARTS specific configuration
        if hasattr(config.search, 'progressive_darts'):
            self.prog_config = config.search.progressive_darts
        else:
            self.prog_config = ProgressiveDARTSConfig()
        
        # Progressive search state
        self.current_stage = 0
        self.pruned_operations: Dict[int, List[int]] = {}
        self.architecture_history: List[torch.Tensor] = []
        self.convergence_history: List[float] = []
        self.operation_strengths: List[Dict[str, float]] = []
        
        # Early stopping state
        self.best_arch_params = None
        self.best_valid_acc = 0.0
        self.patience_counter = 0
        self.converged = False
        
        logger.info(f"Progressive-DARTS initialized with {self.prog_config.pruning_stages} pruning stages")
    
    def search(self) -> Architecture:
        """
        Run Progressive-DARTS search.
        
        Returns:
            Best architecture found during search
        """
        logger.info("Starting Progressive-DARTS search")
        
        # Build supernet if not already built
        if self.supernet is None:
            self._build_supernet()
        
        # Initialize optimizers
        self._setup_optimizers()
        
        # Progressive search loop
        for epoch in range(self.prog_config.epochs):
            self.current_epoch = epoch
            
            # Check if we need to advance to next stage
            self._check_stage_progression(epoch)
            
            # Perform pruning if needed
            if self._should_prune(epoch):
                self._progressive_pruning()
            
            # Train for one epoch
            train_acc, train_loss, arch_loss = self._train_epoch()
            valid_acc, valid_loss = self._validate_epoch()
            
            # Update learning rates
            self._update_learning_rates(epoch)
            
            # Monitor convergence
            convergence_metric = self._check_convergence()
            self.convergence_history.append(convergence_metric)
            
            # Log progress
            if epoch % 10 == 0:
                self._log_progress(epoch, train_acc, valid_acc, train_loss, valid_loss, arch_loss)
            
            # Check for early stopping
            if self._check_early_stopping(valid_acc, convergence_metric):
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Generate final architecture
        final_architecture = self._generate_final_architecture()
        
        logger.info("Progressive-DARTS search completed")
        return final_architecture
    
    def _check_stage_progression(self, epoch: int) -> None:
        """Check if we should advance to the next progressive stage."""
        if self.current_stage < len(self.prog_config.progressive_stages) - 1:
            next_stage_epoch = self.prog_config.progressive_stages[self.current_stage + 1]
            if epoch >= next_stage_epoch:
                self.current_stage += 1
                logger.info(f"Advanced to progressive stage {self.current_stage}")
                
                # Adjust learning rates for new stage
                self._adjust_stage_learning_rates()
    
    def _should_prune(self, epoch: int) -> bool:
        """Check if pruning should be performed at this epoch."""
        if epoch < self.prog_config.pruning_start_epoch:
            return False
        
        # Prune at specific intervals
        pruning_interval = (self.prog_config.epochs - self.prog_config.pruning_start_epoch) // self.prog_config.pruning_stages
        return (epoch - self.prog_config.pruning_start_epoch) % pruning_interval == 0
    
    def _progressive_pruning(self) -> None:
        """Perform progressive pruning of weak operations."""
        logger.info("Performing progressive pruning")
        
        # Get current architecture parameters
        arch_params = self._get_architecture_parameters()
        
        # Calculate operation strengths
        operation_strengths = {}
        pruning_decisions = {}
        
        for edge_idx, params in enumerate(arch_params):
            # Convert to probabilities
            probs = F.softmax(params, dim=-1)
            
            # Find operations to prune (below threshold)
            strong_ops = []
            for op_idx, prob in enumerate(probs):
                if prob.item() > self.prog_config.pruning_threshold:
                    strong_ops.append(op_idx)
                
                operation_strengths[f"edge_{edge_idx}_op_{op_idx}"] = prob.item()
            
            # Ensure minimum number of operations
            if len(strong_ops) < self.prog_config.operations_to_keep:
                # Keep top-k operations
                _, top_indices = torch.topk(probs, self.prog_config.operations_to_keep)
                strong_ops = top_indices.tolist()
            
            pruning_decisions[edge_idx] = strong_ops
        
        # Apply pruning to supernet
        self._apply_pruning(pruning_decisions)
        
        # Store operation strengths for analysis
        self.operation_strengths.append(operation_strengths)
        
        logger.info(f"Pruning completed. Remaining operations: {pruning_decisions}")
    
    def _apply_pruning(self, pruning_decisions: Dict[int, List[int]]) -> None:
        """Apply pruning decisions to the supernet."""
        # Store pruning decisions
        self.pruned_operations.update(pruning_decisions)
        
        # Modify supernet to mask pruned operations
        for edge_idx, kept_ops in pruning_decisions.items():
            if hasattr(self.supernet, 'cells'):
                for cell in self.supernet.cells:
                    if hasattr(cell, 'edges') and edge_idx < len(cell.edges):
                        edge = cell.edges[edge_idx]
                        if hasattr(edge, 'operations'):
                            # Mask unused operations
                            for op_idx, op in enumerate(edge.operations):
                                if op_idx not in kept_ops:
                                    # Set operation to zero
                                    if hasattr(op, 'weight'):
                                        op.weight.data.zero_()
                                    if hasattr(op, 'bias') and op.bias is not None:
                                        op.bias.data.zero_()
    
    def _check_convergence(self) -> float:
        """Check architecture parameter convergence."""
        current_arch_params = self._get_architecture_parameters()
        
        # Store current parameters
        flattened_params = torch.cat([params.flatten() for params in current_arch_params])
        self.architecture_history.append(flattened_params.detach().clone())
        
        if len(self.architecture_history) < 2:
            return float('inf')
        
        # Calculate parameter change
        prev_params = self.architecture_history[-2]
        curr_params = self.architecture_history[-1]
        
        param_change = torch.norm(curr_params - prev_params).item()
        return param_change
    
    def _check_early_stopping(self, valid_acc: float, convergence_metric: float) -> bool:
        """Check if early stopping criteria are met."""
        if self.current_epoch < self.prog_config.min_epochs:
            return False
        
        # Check validation accuracy improvement
        if valid_acc > self.best_valid_acc:
            self.best_valid_acc = valid_acc
            self.best_arch_params = self._get_architecture_parameters()
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # Check convergence
        if convergence_metric < self.prog_config.convergence_threshold:
            logger.info(f"Architecture parameters converged (change: {convergence_metric:.6f})")
            return True
        
        # Check patience
        if self.patience_counter >= self.prog_config.early_stopping_patience:
            logger.info(f"Early stopping due to lack of improvement (patience: {self.patience_counter})")
            return True
        
        return False
    
    def _adjust_stage_learning_rates(self) -> None:
        """Adjust learning rates for new progressive stage."""
        # Reduce architecture learning rate as we progress
        stage_factor = 0.8 ** self.current_stage
        new_arch_lr = self.prog_config.arch_learning_rate * stage_factor
        
        for param_group in self.arch_optimizer.param_groups:
            param_group['lr'] = new_arch_lr
        
        logger.info(f"Adjusted architecture learning rate to {new_arch_lr:.6f}")
    
    def _update_learning_rates(self, epoch: int) -> None:
        """Update learning rates using cosine annealing."""
        # Weight optimizer learning rate
        lr = self.prog_config.learning_rate_min + \
             (self.prog_config.learning_rate - self.prog_config.learning_rate_min) * \
             (1 + np.cos(np.pi * epoch / self.prog_config.epochs)) / 2
        
        for param_group in self.weight_optimizer.param_groups:
            param_group['lr'] = lr
        
        # Architecture optimizer learning rate (slower decay)
        arch_lr = self.prog_config.arch_learning_rate * \
                  (1 + np.cos(np.pi * epoch / (2 * self.prog_config.epochs))) / 2
        
        for param_group in self.arch_optimizer.param_groups:
            param_group['lr'] = arch_lr
    
    def _generate_final_architecture(self) -> Architecture:
        """Generate final architecture from trained supernet."""
        if self.best_arch_params is not None:
            arch_params = self.best_arch_params
        else:
            arch_params = self._get_architecture_parameters()
        
        # Convert architecture parameters to discrete architecture
        architecture_encoding = []
        
        for edge_params in arch_params:
            probs = F.softmax(edge_params, dim=-1)
            best_op = torch.argmax(probs).item()
            
            # Consider pruning decisions
            edge_idx = len(architecture_encoding)
            if edge_idx in self.pruned_operations:
                kept_ops = self.pruned_operations[edge_idx]
                if best_op not in kept_ops:
                    # Choose best from remaining operations
                    masked_probs = probs.clone()
                    for op_idx in range(len(probs)):
                        if op_idx not in kept_ops:
                            masked_probs[op_idx] = -float('inf')
                    best_op = torch.argmax(masked_probs).item()
            
            architecture_encoding.append(best_op)
        
        # Create architecture object
        architecture = Architecture(
            encoding=architecture_encoding,
            search_space=self.search_space,
            metadata={
                'search_method': 'progressive_darts',
                'pruned_operations': self.pruned_operations,
                'final_stage': self.current_stage,
                'convergence_history': self.convergence_history,
                'operation_strengths': self.operation_strengths[-1] if self.operation_strengths else {}
            }
        )
        
        return architecture
    
    def _log_progress(self, epoch: int, train_acc: float, valid_acc: float, 
                     train_loss: float, valid_loss: float, arch_loss: float) -> None:
        """Log training progress."""
        convergence = self.convergence_history[-1] if self.convergence_history else 0.0
        
        logger.info(
            f"Epoch {epoch:3d} | Stage {self.current_stage} | "
            f"Train Acc: {train_acc:.3f} | Valid Acc: {valid_acc:.3f} | "
            f"Train Loss: {train_loss:.3f} | Valid Loss: {valid_loss:.3f} | "
            f"Arch Loss: {arch_loss:.3f} | Convergence: {convergence:.6f}"
        )
    
    def get_search_metrics(self) -> Dict[str, Any]:
        """Get comprehensive search metrics."""
        base_metrics = super().get_search_metrics()
        
        progressive_metrics = {
            'progressive_stages': self.prog_config.progressive_stages,
            'final_stage': self.current_stage,
            'pruned_operations': self.pruned_operations,
            'convergence_history': self.convergence_history,
            'operation_strengths_history': self.operation_strengths,
            'early_stopped': self.converged,
            'best_valid_accuracy': self.best_valid_acc
        }
        
        base_metrics.update(progressive_metrics)
        return base_metrics
    
    def visualize_search_progress(self, save_path: Optional[str] = None) -> None:
        """Visualize Progressive-DARTS search progress."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Convergence history
        if self.convergence_history:
            axes[0, 0].plot(self.convergence_history)
            axes[0, 0].set_title('Architecture Parameter Convergence')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Parameter Change')
            axes[0, 0].set_yscale('log')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Operation strengths over time
        if self.operation_strengths:
            for i, strengths in enumerate(self.operation_strengths):
                epoch = self.prog_config.pruning_start_epoch + i * \
                       ((self.prog_config.epochs - self.prog_config.pruning_start_epoch) // self.prog_config.pruning_stages)
                
                op_names = list(strengths.keys())[:10]  # Show first 10 operations
                op_values = [strengths[name] for name in op_names]
                
                axes[0, 1].bar([f"{epoch}_{name.split('_')[-1]}" for name in op_names], 
                              op_values, alpha=0.7, label=f'Epoch {epoch}')
            
            axes[0, 1].set_title('Operation Strengths Over Time')
            axes[0, 1].set_ylabel('Probability')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Progressive stages timeline
        if hasattr(self, 'train_acc_history') and self.train_acc_history:
            epochs = range(len(self.train_acc_history))
            axes[1, 0].plot(epochs, self.train_acc_history, label='Train Accuracy')
            if hasattr(self, 'valid_acc_history') and self.valid_acc_history:
                axes[1, 0].plot(epochs, self.valid_acc_history, label='Valid Accuracy')
            
            # Mark progressive stages
            for i, stage_epoch in enumerate(self.prog_config.progressive_stages):
                if stage_epoch <= len(self.train_acc_history):
                    axes[1, 0].axvline(x=stage_epoch, color='red', linestyle='--', alpha=0.7,
                                      label=f'Stage {i}' if i == 0 else "")
            
            axes[1, 0].set_title('Training Progress with Progressive Stages')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Pruning statistics
        if self.pruned_operations:
            edge_ids = list(self.pruned_operations.keys())
            remaining_ops = [len(ops) for ops in self.pruned_operations.values()]
            
            axes[1, 1].bar(edge_ids, remaining_ops)
            axes[1, 1].set_title('Remaining Operations After Pruning')
            axes[1, 1].set_xlabel('Edge Index')
            axes[1, 1].set_ylabel('Number of Operations')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Search progress visualization saved to {save_path}")
        
        plt.show() 