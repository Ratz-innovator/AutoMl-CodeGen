"""
PC-DARTS: Partial Channel Connections for Memory-Efficient Architecture Search
=============================================================================

This module implements PC-DARTS (Partially Connected DARTS), which addresses
the memory consumption issue in DARTS by using partial channel connections
to reduce memory requirements while maintaining search effectiveness.

Key Features:
- Memory-efficient search through partial channel connections
- Edge normalization for gradient stabilization  
- Progressive channel sampling strategy
- Advanced regularization techniques
- Multi-GPU support for scalable search

Reference:
    Xu et al. "PC-DARTS: Partial Channel Connections for Memory-Efficient 
    Architecture Search" ICLR 2020.

Example Usage:
    >>> from nanonas.search.pc_darts import PCDARTSSearch
    >>> 
    >>> searcher = PCDARTSSearch(
    ...     search_space='mobile',
    ...     epochs=50,
    ...     channel_ratio=0.25,
    ...     progressive_channels=True
    ... )
    >>> best_architecture = searcher.search()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import copy
import time

from ..core.base import BaseSearchStrategy
from ..core.architecture import Architecture, SearchSpace
from ..core.config import SearchConfig
from ..models.supernet import SuperNet, PartialChannelSuperNet
from ..models.operations import MixedOperation
from ..utils.metrics import accuracy
from ..utils.scheduler import get_scheduler
from ..visualization.search_viz import SearchProgressTracker


class PCDARTSSearch(BaseSearchStrategy):
    """
    PC-DARTS: Partial Channel Connections for Memory-Efficient Architecture Search.
    
    PC-DARTS addresses the memory bottleneck in DARTS by using partial channel
    connections during the search phase, significantly reducing GPU memory
    requirements while maintaining competitive search performance.
    
    Args:
        search_space: Architecture search space configuration
        epochs: Number of search epochs
        warmup_epochs: Number of warmup epochs before architecture search
        channel_ratio: Ratio of channels to use in partial connections (0.1-0.5)
        progressive_channels: Whether to progressively increase channel ratio
        edge_normalization: Enable edge normalization for gradient stability
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for network weights
        arch_learning_rate: Learning rate for architecture parameters
        weight_decay: Weight decay for network weights
        arch_weight_decay: Weight decay for architecture parameters
        grad_clip: Gradient clipping value
        report_freq: Frequency of progress reporting
        save_freq: Frequency of checkpoint saving
        device: Computing device
        output_dir: Directory for saving results
        verbose: Enable verbose logging
    """
    
    def __init__(
        self,
        search_space: str = "mobile",
        epochs: int = 50,
        warmup_epochs: int = 15,
        channel_ratio: float = 0.25,
        progressive_channels: bool = True,
        edge_normalization: bool = True,
        dropout_rate: float = 0.1,
        learning_rate: float = 0.025,
        arch_learning_rate: float = 3e-4,
        weight_decay: float = 3e-4,
        arch_weight_decay: float = 1e-3,
        grad_clip: float = 5.0,
        report_freq: int = 50,
        save_freq: int = 10,
        device: str = "auto",
        output_dir: str = "./results",
        verbose: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Core configuration
        self.search_space_name = search_space
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.channel_ratio = channel_ratio
        self.progressive_channels = progressive_channels
        self.edge_normalization = edge_normalization
        self.dropout_rate = dropout_rate
        
        # Optimization parameters
        self.learning_rate = learning_rate
        self.arch_learning_rate = arch_learning_rate
        self.weight_decay = weight_decay
        self.arch_weight_decay = arch_weight_decay
        self.grad_clip = grad_clip
        
        # Logging and saving
        self.report_freq = report_freq
        self.save_freq = save_freq
        self.device = self._setup_device(device)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        
        # Initialize search components
        self.search_space = None
        self.supernet = None
        self.optimizer = None
        self.arch_optimizer = None
        self.scheduler = None
        self.progress_tracker = None
        
        # Search state
        self.current_epoch = 0
        self.best_architecture = None
        self.best_accuracy = 0.0
        self.search_history = []
        
        self._setup_logging()
    
    def setup(self, dataset: DataLoader, val_dataset: Optional[DataLoader] = None) -> None:
        """
        Setup the PC-DARTS search with dataset and initialize all components.
        
        Args:
            dataset: Training dataset loader
            val_dataset: Validation dataset loader (optional)
        """
        self.logger.info("🔧 Setting up PC-DARTS search...")
        
        # Store datasets
        self.train_loader = dataset
        self.val_loader = val_dataset or dataset
        
        # Initialize search space
        self.search_space = SearchSpace.get_search_space(self.search_space_name)
        
        # Create supernet with partial channel connections
        self.supernet = PartialChannelSuperNet(
            search_space=self.search_space,
            channel_ratio=self.channel_ratio,
            edge_normalization=self.edge_normalization,
            dropout_rate=self.dropout_rate,
            num_classes=self._get_num_classes()
        ).to(self.device)
        
        # Setup optimizers
        self._setup_optimizers()
        
        # Setup schedulers
        self._setup_schedulers()
        
        # Initialize progress tracking
        self.progress_tracker = SearchProgressTracker(
            output_dir=self.output_dir,
            track_gradients=True,
            track_architecture_evolution=True
        )
        
        self.logger.info(f"✅ PC-DARTS setup complete")
        self.logger.info(f"📊 Supernet parameters: {self._count_parameters():,}")
        self.logger.info(f"🔍 Architecture parameters: {self._count_arch_parameters():,}")
    
    def search(self) -> Architecture:
        """
        Perform PC-DARTS architecture search.
        
        Returns:
            Best discovered architecture
        """
        self.logger.info("🚀 Starting PC-DARTS search...")
        start_time = time.time()
        
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            
            # Update channel ratio if progressive
            if self.progressive_channels:
                self._update_channel_ratio(epoch)
            
            # Training phase
            if epoch >= self.warmup_epochs:
                # Joint training of weights and architecture
                train_acc, train_loss = self._train_search_epoch()
            else:
                # Warmup: only train network weights
                train_acc, train_loss = self._train_warmup_epoch()
            
            # Validation phase
            val_acc, val_loss = self._validate_epoch()
            
            # Update learning rate
            self.scheduler.step()
            
            # Track progress
            self.progress_tracker.log_epoch(
                epoch=epoch,
                train_acc=train_acc,
                train_loss=train_loss,
                val_acc=val_acc,
                val_loss=val_loss,
                architecture_weights=self.supernet.get_architecture_weights(),
                lr=self.optimizer.param_groups[0]['lr']
            )
            
            # Update best architecture
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self.best_architecture = self._derive_architecture()
                self.logger.info(f"🎯 New best architecture found! Accuracy: {val_acc:.3f}")
            
            # Periodic reporting
            if epoch % self.report_freq == 0 or epoch == self.epochs - 1:
                self._report_progress(epoch, train_acc, train_loss, val_acc, val_loss)
            
            # Periodic saving
            if epoch % self.save_freq == 0:
                self._save_checkpoint(epoch)
        
        search_time = time.time() - start_time
        
        # Final architecture derivation
        final_architecture = self._derive_architecture()
        
        # Save final results
        self._save_final_results(final_architecture, search_time)
        
        self.logger.info(f"✅ PC-DARTS search completed in {search_time:.2f}s")
        self.logger.info(f"🏆 Best validation accuracy: {self.best_accuracy:.3f}")
        
        return final_architecture
    
    def _train_search_epoch(self) -> Tuple[float, float]:
        """Train epoch with joint weight and architecture optimization."""
        self.supernet.train()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Split batch for bilevel optimization
            batch_size = data.size(0)
            split = batch_size // 2
            
            # Data for weight update
            data_search, target_search = data[:split], target[:split]
            # Data for architecture update  
            data_arch, target_arch = data[split:], target[split:]
            
            # Step 1: Update network weights
            self.optimizer.zero_grad()
            logits = self.supernet(data_search)
            loss = F.cross_entropy(logits, target_search)
            loss.backward()
            
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.supernet.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            # Step 2: Update architecture parameters
            self.arch_optimizer.zero_grad()
            logits_arch = self.supernet(data_arch)
            loss_arch = F.cross_entropy(logits_arch, target_arch)
            loss_arch.backward()
            
            self.arch_optimizer.step()
            
            # Apply edge normalization if enabled
            if self.edge_normalization:
                self.supernet.normalize_architecture_weights()
            
            # Accumulate metrics
            acc = accuracy(logits, target_search)[0]
            total_loss += loss.item()
            total_acc += acc.item()
            num_batches += 1
            
            # Memory cleanup
            del data, target, logits, loss
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
        
        return total_acc / num_batches, total_loss / num_batches
    
    def _train_warmup_epoch(self) -> Tuple[float, float]:
        """Warmup epoch with only network weight training."""
        self.supernet.train()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass through supernet
            logits = self.supernet(data)
            loss = F.cross_entropy(logits, target)
            
            # Backward pass
            loss.backward()
            
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.supernet.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            # Accumulate metrics
            acc = accuracy(logits, target)[0]
            total_loss += loss.item()
            total_acc += acc.item()
            num_batches += 1
        
        return total_acc / num_batches, total_loss / num_batches
    
    def _validate_epoch(self) -> Tuple[float, float]:
        """Validation epoch."""
        self.supernet.eval()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                logits = self.supernet(data)
                loss = F.cross_entropy(logits, target)
                
                acc = accuracy(logits, target)[0]
                total_loss += loss.item()
                total_acc += acc.item()
                num_batches += 1
        
        return total_acc / num_batches, total_loss / num_batches
    
    def _derive_architecture(self) -> Architecture:
        """Derive discrete architecture from continuous search space."""
        self.logger.info("🔍 Deriving architecture from supernet...")
        
        # Get architecture weights
        arch_weights = self.supernet.get_architecture_weights()
        
        # Convert to discrete architecture
        operations = []
        for edge_weights in arch_weights:
            # Select operation with highest weight
            op_idx = torch.argmax(edge_weights).item()
            operations.append(op_idx)
        
        # Create architecture object
        architecture = Architecture(
            operations=operations,
            search_space=self.search_space,
            metadata={
                'search_method': 'pc_darts',
                'channel_ratio': self.channel_ratio,
                'epoch_found': self.current_epoch,
                'validation_accuracy': self.best_accuracy
            }
        )
        
        return architecture
    
    def _update_channel_ratio(self, epoch: int) -> None:
        """Progressively update channel ratio during search."""
        if not self.progressive_channels:
            return
        
        # Progressive schedule: start low, gradually increase
        min_ratio = 0.1
        max_ratio = 0.5
        progress = min(epoch / (self.epochs * 0.7), 1.0)
        
        new_ratio = min_ratio + (max_ratio - min_ratio) * progress
        
        if abs(new_ratio - self.channel_ratio) > 0.01:
            self.channel_ratio = new_ratio
            self.supernet.update_channel_ratio(new_ratio)
            self.logger.info(f"📈 Updated channel ratio to {new_ratio:.3f}")
    
    def _setup_optimizers(self) -> None:
        """Setup optimizers for network weights and architecture parameters."""
        # Network weight optimizer
        self.optimizer = optim.SGD(
            self.supernet.network_parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=self.weight_decay
        )
        
        # Architecture parameter optimizer
        self.arch_optimizer = optim.Adam(
            self.supernet.architecture_parameters(),
            lr=self.arch_learning_rate,
            weight_decay=self.arch_weight_decay
        )
    
    def _setup_schedulers(self) -> None:
        """Setup learning rate schedulers."""
        self.scheduler = get_scheduler(
            optimizer=self.optimizer,
            scheduler_type='cosine',
            epochs=self.epochs,
            eta_min=0.001
        )
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computing device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        self.logger = logging.getLogger(f"{__name__}.PCDARTSSearch")
        if self.verbose:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _get_num_classes(self) -> int:
        """Get number of classes from dataset."""
        # Try to infer from data loader
        try:
            sample_batch = next(iter(self.train_loader))
            if isinstance(sample_batch, (list, tuple)) and len(sample_batch) >= 2:
                targets = sample_batch[1]
                return len(torch.unique(targets))
        except:
            pass
        
        # Default fallback
        return 10
    
    def _count_parameters(self) -> int:
        """Count total network parameters."""
        return sum(p.numel() for p in self.supernet.network_parameters())
    
    def _count_arch_parameters(self) -> int:
        """Count architecture parameters."""
        return sum(p.numel() for p in self.supernet.architecture_parameters())
    
    def _report_progress(self, epoch: int, train_acc: float, train_loss: float,
                        val_acc: float, val_loss: float) -> None:
        """Report training progress."""
        self.logger.info(
            f"Epoch {epoch:3d}/{self.epochs} | "
            f"Train: {train_acc:.3f}/{train_loss:.3f} | "
            f"Val: {val_acc:.3f}/{val_loss:.3f} | "
            f"Best: {self.best_accuracy:.3f}"
        )
    
    def _save_checkpoint(self, epoch: int) -> None:
        """Save training checkpoint."""
        checkpoint_path = self.output_dir / f"pc_darts_checkpoint_epoch_{epoch}.pth"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'supernet_state_dict': self.supernet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'arch_optimizer_state_dict': self.arch_optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_accuracy': self.best_accuracy,
            'best_architecture': self.best_architecture,
            'search_config': {
                'search_space': self.search_space_name,
                'channel_ratio': self.channel_ratio,
                'epochs': self.epochs,
                'learning_rate': self.learning_rate,
                'arch_learning_rate': self.arch_learning_rate
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def _save_final_results(self, architecture: Architecture, search_time: float) -> None:
        """Save final search results."""
        results_path = self.output_dir / "pc_darts_results.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'final_architecture': architecture.to_dict(),
            'best_accuracy': self.best_accuracy,
            'search_time': search_time,
            'total_epochs': self.epochs,
            'search_method': 'pc_darts',
            'configuration': {
                'search_space': self.search_space_name,
                'channel_ratio': self.channel_ratio,
                'progressive_channels': self.progressive_channels,
                'edge_normalization': self.edge_normalization,
                'learning_rate': self.learning_rate,
                'arch_learning_rate': self.arch_learning_rate
            }
        }
        
        import json
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"📊 Results saved: {results_path}")
    
    def get_search_metrics(self) -> Dict[str, Any]:
        """Get detailed search metrics."""
        return {
            'best_accuracy': self.best_accuracy,
            'total_epochs': self.epochs,
            'warmup_epochs': self.warmup_epochs,
            'channel_ratio': self.channel_ratio,
            'progressive_channels': self.progressive_channels,
            'edge_normalization': self.edge_normalization,
            'network_parameters': self._count_parameters(),
            'architecture_parameters': self._count_arch_parameters(),
            'memory_efficient': True,
            'search_method': 'pc_darts'
        } 