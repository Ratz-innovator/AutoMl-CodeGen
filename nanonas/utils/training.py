"""
Training utilities for nanoNAS framework.

This module provides comprehensive training infrastructure including
trainers, training configurations, and evaluation utilities.
"""

import time
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np

from .logging_utils import ResultLogger
from .reproducibility import set_seed


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Basic training parameters
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    momentum: float = 0.9
    
    # Optimizer settings
    optimizer: str = 'adam'  # 'adam', 'sgd', 'adamw'
    scheduler: str = 'cosine'  # 'cosine', 'step', 'exponential', 'plateau'
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    
    # Training features
    use_amp: bool = True  # Automatic Mixed Precision
    gradient_clip: Optional[float] = 1.0
    early_stopping_patience: int = 10
    save_checkpoints: bool = True
    
    # Validation and monitoring
    val_every_n_epochs: int = 1
    log_every_n_steps: int = 100
    
    # Device and reproducibility
    device: str = 'auto'  # 'auto', 'cpu', 'cuda'
    seed: Optional[int] = 42
    
    # Paths
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create directories
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Set seed if specified
        if self.seed is not None:
            set_seed(self.seed)


class Trainer:
    """
    Comprehensive trainer for neural architecture search models.
    
    Supports training individual architectures found by NAS methods
    with full logging, checkpointing, and evaluation capabilities.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        self.logger = logging.getLogger(__name__)
        self.result_logger = ResultLogger(config.log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        
        # Mixed precision training
        self.scaler = GradScaler() if config.use_amp else None
        
        self.logger.info(f"Trainer initialized with device: {self.device}")
    
    def _create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        params = model.parameters()
        
        if self.config.optimizer.lower() == 'adam':
            return optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'adamw':
            return optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'sgd':
            return optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self, optimizer: optim.Optimizer) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on configuration."""
        if self.config.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.epochs,
                **self.config.scheduler_params
            )
        elif self.config.scheduler == 'step':
            step_size = self.config.scheduler_params.get('step_size', 30)
            gamma = self.config.scheduler_params.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif self.config.scheduler == 'exponential':
            gamma = self.config.scheduler_params.get('gamma', 0.95)
            return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        elif self.config.scheduler == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                **self.config.scheduler_params
            )
        elif self.config.scheduler == 'none':
            return None
        else:
            raise ValueError(f"Unsupported scheduler: {self.config.scheduler}")
    
    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        batch_times = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            start_time = time.time()
            
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            
            if self.config.use_amp and self.scaler:
                with autocast():
                    output = model(data)
                    loss = criterion(output, target)
                
                self.scaler.scale(loss).backward()
                
                if self.config.gradient_clip:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                if self.config.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                
                optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            batch_time = time.time() - start_time
            batch_times.append(batch_time)
            
            # Logging
            if batch_idx % self.config.log_every_n_steps == 0:
                accuracy = 100. * correct / total
                avg_batch_time = np.mean(batch_times[-100:])  # Last 100 batches
                
                self.logger.info(
                    f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                    f'Loss: {loss.item():.6f} '
                    f'Acc: {accuracy:.2f}% '
                    f'Time: {avg_batch_time:.3f}s/batch'
                )
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        avg_batch_time = np.mean(batch_times)
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'avg_batch_time': avg_batch_time
        }
    
    def _validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """Validate the model."""
        model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                if self.config.use_amp:
                    with autocast():
                        output = model(data)
                        loss = criterion(output, target)
                else:
                    output = model(data)
                    loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        val_loss = total_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        return {
            'loss': val_loss,
            'accuracy': val_acc
        }
    
    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        val_acc: float,
        is_best: bool = False
    ):
        """Save model checkpoint."""
        if not self.config.save_checkpoints:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'config': self.config
        }
        
        checkpoint_path = Path(self.config.checkpoint_dir) / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with accuracy: {val_acc:.2f}%")
    
    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model with comprehensive logging and evaluation.
        
        Args:
            model: Neural network model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            criterion: Loss function (defaults to CrossEntropyLoss)
        
        Returns:
            Dictionary containing training history
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        model = model.to(self.device)
        criterion = criterion.to(self.device)
        
        optimizer = self._create_optimizer(model)
        scheduler = self._create_scheduler(optimizer)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        self.logger.info(f"Starting training for {self.config.epochs} epochs")
        start_time = time.time()
        
        for epoch in range(1, self.config.epochs + 1):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self._train_epoch(model, train_loader, optimizer, criterion, epoch)
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            
            # Validation phase
            if val_loader is not None and epoch % self.config.val_every_n_epochs == 0:
                val_metrics = self._validate(model, val_loader, criterion)
                history['val_loss'].append(val_metrics['loss'])
                history['val_acc'].append(val_metrics['accuracy'])
                
                # Check for improvement
                is_best = val_metrics['accuracy'] > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_metrics['accuracy']
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
                
                # Save checkpoint
                self._save_checkpoint(model, optimizer, epoch, val_metrics['accuracy'], is_best)
                
                # Early stopping
                if self.epochs_without_improvement >= self.config.early_stopping_patience:
                    self.logger.info(f"Early stopping after {epoch} epochs")
                    break
                
                self.logger.info(
                    f'Epoch {epoch}: Train Loss: {train_metrics["loss"]:.4f}, '
                    f'Train Acc: {train_metrics["accuracy"]:.2f}%, '
                    f'Val Loss: {val_metrics["loss"]:.4f}, '
                    f'Val Acc: {val_metrics["accuracy"]:.2f}%'
                )
            else:
                self.logger.info(
                    f'Epoch {epoch}: Train Loss: {train_metrics["loss"]:.4f}, '
                    f'Train Acc: {train_metrics["accuracy"]:.2f}%'
                )
            
            # Learning rate scheduling
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if val_loader is not None:
                        scheduler.step(history['val_acc'][-1])
                else:
                    scheduler.step()
            
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Log to result logger
            epoch_results = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            
            if val_loader is not None and epoch % self.config.val_every_n_epochs == 0:
                epoch_results.update({
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy']
                })
            
            self.result_logger.log_epoch(epoch_results)
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Save final results
        final_results = {
            'total_epochs': self.current_epoch,
            'best_val_accuracy': self.best_val_acc,
            'total_training_time': total_time,
            'final_train_accuracy': history['train_acc'][-1] if history['train_acc'] else 0.0
        }
        
        self.result_logger.save_results(final_results)
        
        return history


def quick_train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = 'auto'
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Quick training function for simple model training.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to use for training
    
    Returns:
        Tuple of (trained_model, training_history)
    """
    config = TrainingConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        log_every_n_steps=50
    )
    
    trainer = Trainer(config)
    history = trainer.train(model, train_loader, val_loader)
    
    return model, history 