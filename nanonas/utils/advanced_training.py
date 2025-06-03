"""
Advanced training strategies for Neural Architecture Search.

This module provides sophisticated training techniques including:
- Progressive training with curriculum learning
- Knowledge distillation from teacher networks
- Self-supervised pre-training
- Adaptive optimization and learning rate scheduling
- Multi-task learning capabilities
- Regularization techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
from collections import defaultdict
import copy
import random
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for advanced training strategies."""
    
    # Basic training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    # Progressive training
    enable_progressive_training: bool = False
    progressive_stages: List[int] = field(default_factory=lambda: [32, 64, 128, 224])
    progressive_epochs_per_stage: int = 20
    
    # Knowledge distillation
    enable_knowledge_distillation: bool = False
    teacher_model_path: Optional[str] = None
    distillation_temperature: float = 4.0
    distillation_alpha: float = 0.7
    
    # Self-supervised learning
    enable_self_supervised: bool = False
    ssl_method: str = "simclr"  # simclr, byol, swav
    ssl_epochs: int = 50
    ssl_temperature: float = 0.1
    
    # Adaptive optimization
    optimizer_type: str = "adamw"  # adam, adamw, sgd, rmsprop
    scheduler_type: str = "cosine"  # cosine, step, exponential, plateau
    warmup_epochs: int = 5
    
    # Regularization
    dropout_rate: float = 0.1
    droppath_rate: float = 0.1
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    label_smoothing: float = 0.1
    
    # Multi-task learning
    enable_multitask: bool = False
    auxiliary_tasks: List[str] = field(default_factory=list)
    task_weights: Dict[str, float] = field(default_factory=dict)
    
    # Advanced techniques
    enable_gradient_clipping: bool = True
    gradient_clip_value: float = 1.0
    enable_ema: bool = False
    ema_decay: float = 0.999
    enable_swa: bool = False
    swa_start_epoch: int = 80


class ProgressiveTrainer:
    """Progressive training with curriculum learning."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.current_stage = 0
        self.current_resolution = config.progressive_stages[0] if config.progressive_stages else 224
    
    def should_advance_stage(self, epoch: int) -> bool:
        """Check if should advance to next progressive stage."""
        if not self.config.enable_progressive_training:
            return False
        
        stage_epoch = epoch % self.config.progressive_epochs_per_stage
        return (stage_epoch == 0 and 
                self.current_stage < len(self.config.progressive_stages) - 1)
    
    def advance_stage(self):
        """Advance to next progressive training stage."""
        if self.current_stage < len(self.config.progressive_stages) - 1:
            self.current_stage += 1
            self.current_resolution = self.config.progressive_stages[self.current_stage]
            logger.info(f"Advanced to progressive stage {self.current_stage}, "
                       f"resolution: {self.current_resolution}")
    
    def get_current_transform(self):
        """Get data transform for current progressive stage."""
        from torchvision import transforms
        
        return transforms.Compose([
            transforms.Resize((self.current_resolution, self.current_resolution)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


class KnowledgeDistillationLoss(nn.Module):
    """Knowledge distillation loss function."""
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """Compute knowledge distillation loss."""
        # Soft targets from teacher
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # Distillation loss
        distillation_loss = self.kl_div(student_log_probs, teacher_probs) * (self.temperature ** 2)
        
        # Standard cross-entropy loss
        ce_loss = self.ce_loss(student_logits, targets)
        
        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * ce_loss
        
        return total_loss


class SelfSupervisedLearning:
    """Self-supervised learning strategies."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.method = config.ssl_method
    
    def create_ssl_model(self, backbone: nn.Module, feature_dim: int = 128) -> nn.Module:
        """Create self-supervised learning model."""
        if self.method == "simclr":
            return SimCLRModel(backbone, feature_dim)
        elif self.method == "byol":
            return BYOLModel(backbone, feature_dim)
        elif self.method == "swav":
            return SwAVModel(backbone, feature_dim)
        else:
            raise ValueError(f"Unknown SSL method: {self.method}")
    
    def create_ssl_loss(self) -> nn.Module:
        """Create self-supervised learning loss."""
        if self.method == "simclr":
            return SimCLRLoss(temperature=self.config.ssl_temperature)
        elif self.method == "byol":
            return BYOLLoss()
        elif self.method == "swav":
            return SwAVLoss()
        else:
            raise ValueError(f"Unknown SSL method: {self.method}")


class SimCLRModel(nn.Module):
    """SimCLR model for contrastive learning."""
    
    def __init__(self, backbone: nn.Module, feature_dim: int = 128):
        super().__init__()
        self.backbone = backbone
        
        # Get backbone output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            backbone_output = backbone(dummy_input)
            backbone_dim = backbone_output.shape[-1]
        
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(),
            nn.Linear(backbone_dim, feature_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        projections = self.projection_head(features)
        return F.normalize(projections, dim=1)


class SimCLRLoss(nn.Module):
    """SimCLR contrastive loss."""
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0] // 2
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create labels for positive pairs
        labels = torch.cat([torch.arange(batch_size) + batch_size, 
                           torch.arange(batch_size)]).to(features.device)
        
        # Mask out self-similarity
        mask = torch.eye(features.shape[0], dtype=torch.bool).to(features.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


class BYOLModel(nn.Module):
    """BYOL model for self-supervised learning."""
    
    def __init__(self, backbone: nn.Module, feature_dim: int = 128):
        super().__init__()
        
        # Online network
        self.online_backbone = backbone
        
        # Get backbone output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            backbone_output = backbone(dummy_input)
            backbone_dim = backbone_output.shape[-1]
        
        self.online_projector = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.BatchNorm1d(backbone_dim),
            nn.ReLU(),
            nn.Linear(backbone_dim, feature_dim)
        )
        
        self.online_predictor = nn.Sequential(
            nn.Linear(feature_dim, backbone_dim // 2),
            nn.BatchNorm1d(backbone_dim // 2),
            nn.ReLU(),
            nn.Linear(backbone_dim // 2, feature_dim)
        )
        
        # Target network (EMA of online network)
        self.target_backbone = copy.deepcopy(backbone)
        self.target_projector = copy.deepcopy(self.online_projector)
        
        # Disable gradients for target network
        for param in self.target_backbone.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Online network
        online_features_1 = self.online_backbone(x1)
        online_proj_1 = self.online_projector(online_features_1)
        online_pred_1 = self.online_predictor(online_proj_1)
        
        online_features_2 = self.online_backbone(x2)
        online_proj_2 = self.online_projector(online_features_2)
        online_pred_2 = self.online_predictor(online_proj_2)
        
        # Target network
        with torch.no_grad():
            target_features_1 = self.target_backbone(x1)
            target_proj_1 = self.target_projector(target_features_1)
            
            target_features_2 = self.target_backbone(x2)
            target_proj_2 = self.target_projector(target_features_2)
        
        return (online_pred_1, online_pred_2), (target_proj_1, target_proj_2)
    
    def update_target_network(self, tau: float = 0.99):
        """Update target network with EMA."""
        for online_param, target_param in zip(self.online_backbone.parameters(), 
                                            self.target_backbone.parameters()):
            target_param.data = tau * target_param.data + (1 - tau) * online_param.data
        
        for online_param, target_param in zip(self.online_projector.parameters(), 
                                            self.target_projector.parameters()):
            target_param.data = tau * target_param.data + (1 - tau) * online_param.data


class BYOLLoss(nn.Module):
    """BYOL loss function."""
    
    def forward(self, online_preds: Tuple[torch.Tensor, torch.Tensor], 
                target_projs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        online_pred_1, online_pred_2 = online_preds
        target_proj_1, target_proj_2 = target_projs
        
        # Normalize predictions and projections
        online_pred_1 = F.normalize(online_pred_1, dim=1)
        online_pred_2 = F.normalize(online_pred_2, dim=1)
        target_proj_1 = F.normalize(target_proj_1, dim=1)
        target_proj_2 = F.normalize(target_proj_2, dim=1)
        
        # Compute loss
        loss_1 = 2 - 2 * (online_pred_1 * target_proj_2).sum(dim=1).mean()
        loss_2 = 2 - 2 * (online_pred_2 * target_proj_1).sum(dim=1).mean()
        
        return (loss_1 + loss_2) / 2


class SwAVModel(nn.Module):
    """SwAV model for self-supervised learning."""
    
    def __init__(self, backbone: nn.Module, feature_dim: int = 128, num_prototypes: int = 3000):
        super().__init__()
        self.backbone = backbone
        self.num_prototypes = num_prototypes
        
        # Get backbone output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            backbone_output = backbone(dummy_input)
            backbone_dim = backbone_output.shape[-1]
        
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.BatchNorm1d(backbone_dim),
            nn.ReLU(),
            nn.Linear(backbone_dim, feature_dim)
        )
        
        # Prototypes
        self.prototypes = nn.Linear(feature_dim, num_prototypes, bias=False)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        projections = self.projection_head(features)
        projections = F.normalize(projections, dim=1)
        
        # Compute prototype assignments
        prototype_scores = self.prototypes(projections)
        
        return projections, prototype_scores


class SwAVLoss(nn.Module):
    """SwAV loss function."""
    
    def __init__(self, temperature: float = 0.1, epsilon: float = 0.05):
        super().__init__()
        self.temperature = temperature
        self.epsilon = epsilon
    
    def sinkhorn(self, scores: torch.Tensor, num_iters: int = 3) -> torch.Tensor:
        """Sinkhorn-Knopp algorithm for optimal transport."""
        Q = torch.exp(scores / self.epsilon).T
        Q /= torch.sum(Q)
        
        K, B = Q.shape
        
        for _ in range(num_iters):
            # Normalize rows
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= K
            
            # Normalize columns
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B
        
        return Q.T
    
    def forward(self, prototype_scores_1: torch.Tensor, 
                prototype_scores_2: torch.Tensor) -> torch.Tensor:
        # Compute soft assignments
        with torch.no_grad():
            q1 = self.sinkhorn(prototype_scores_1)
            q2 = self.sinkhorn(prototype_scores_2)
        
        # Compute cross-entropy loss
        p1 = F.softmax(prototype_scores_1 / self.temperature, dim=1)
        p2 = F.softmax(prototype_scores_2 / self.temperature, dim=1)
        
        loss = -0.5 * (torch.sum(q1 * torch.log(p2), dim=1).mean() + 
                       torch.sum(q2 * torch.log(p1), dim=1).mean())
        
        return loss


class AdaptiveOptimizer:
    """Adaptive optimizer with advanced scheduling."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        if self.config.optimizer_type == "adam":
            return optim.Adam(model.parameters(), 
                            lr=self.config.learning_rate,
                            weight_decay=self.config.weight_decay)
        elif self.config.optimizer_type == "adamw":
            return optim.AdamW(model.parameters(),
                             lr=self.config.learning_rate,
                             weight_decay=self.config.weight_decay)
        elif self.config.optimizer_type == "sgd":
            return optim.SGD(model.parameters(),
                           lr=self.config.learning_rate,
                           momentum=0.9,
                           weight_decay=self.config.weight_decay)
        elif self.config.optimizer_type == "rmsprop":
            return optim.RMSprop(model.parameters(),
                               lr=self.config.learning_rate,
                               weight_decay=self.config.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")
    
    def create_scheduler(self, optimizer: optim.Optimizer) -> _LRScheduler:
        """Create learning rate scheduler."""
        if self.config.scheduler_type == "cosine":
            return CosineAnnealingWarmupScheduler(
                optimizer,
                warmup_epochs=self.config.warmup_epochs,
                max_epochs=self.config.epochs
            )
        elif self.config.scheduler_type == "step":
            return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif self.config.scheduler_type == "exponential":
            return optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        elif self.config.scheduler_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.scheduler_type}")


class CosineAnnealingWarmupScheduler(_LRScheduler):
    """Cosine annealing scheduler with warmup."""
    
    def __init__(self, optimizer: optim.Optimizer, warmup_epochs: int, max_epochs: int):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        super().__init__(optimizer)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [base_lr * self.last_epoch / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [base_lr * 0.5 * (1 + math.cos(math.pi * progress)) for base_lr in self.base_lrs]


class RegularizationManager:
    """Manager for various regularization techniques."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def apply_mixup(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply MixUp augmentation."""
        if self.config.mixup_alpha <= 0:
            return x, y, y, 1.0
        
        lam = np.random.beta(self.config.mixup_alpha, self.config.mixup_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def apply_cutmix(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply CutMix augmentation."""
        if self.config.cutmix_alpha <= 0:
            return x, y, y, 1.0
        
        lam = np.random.beta(self.config.cutmix_alpha, self.config.cutmix_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        # Generate random bounding box
        W, H = x.size(2), x.size(3)
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply cutmix
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        y_a, y_b = y, y[index]
        
        return x, y_a, y_b, lam
    
    def mixup_criterion(self, criterion: nn.Module, pred: torch.Tensor, 
                       y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
        """Compute loss for mixup/cutmix."""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class ExponentialMovingAverage:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class StochasticWeightAveraging:
    """Stochastic Weight Averaging implementation."""
    
    def __init__(self, model: nn.Module, start_epoch: int = 80):
        self.model = model
        self.start_epoch = start_epoch
        self.swa_model = copy.deepcopy(model)
        self.swa_n = 0
    
    def update(self, epoch: int):
        """Update SWA model."""
        if epoch >= self.start_epoch:
            self.swa_n += 1
            
            # Update SWA parameters
            for swa_param, param in zip(self.swa_model.parameters(), self.model.parameters()):
                swa_param.data = (swa_param.data * (self.swa_n - 1) + param.data) / self.swa_n
    
    def apply_swa(self):
        """Apply SWA parameters to model."""
        for param, swa_param in zip(self.model.parameters(), self.swa_model.parameters()):
            param.data = swa_param.data


class MultiTaskLearning:
    """Multi-task learning manager."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.task_weights = config.task_weights
        self.adaptive_weights = True
        self.task_losses = defaultdict(list)
    
    def compute_multitask_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute weighted multi-task loss."""
        if not self.config.enable_multitask:
            return losses.get('main', torch.tensor(0.0))
        
        total_loss = torch.tensor(0.0, device=next(iter(losses.values())).device)
        
        for task, loss in losses.items():
            weight = self.task_weights.get(task, 1.0)
            if self.adaptive_weights:
                weight = self._compute_adaptive_weight(task, loss)
            
            total_loss += weight * loss
            self.task_losses[task].append(loss.item())
        
        return total_loss
    
    def _compute_adaptive_weight(self, task: str, current_loss: torch.Tensor) -> float:
        """Compute adaptive task weight based on loss history."""
        if len(self.task_losses[task]) < 2:
            return self.task_weights.get(task, 1.0)
        
        # Use gradient-based adaptive weighting
        recent_losses = self.task_losses[task][-10:]  # Last 10 losses
        if len(recent_losses) >= 2:
            loss_trend = recent_losses[-1] - recent_losses[0]
            # Increase weight for tasks with increasing loss
            adaptive_factor = 1.0 + 0.1 * loss_trend
            base_weight = self.task_weights.get(task, 1.0)
            return max(0.1, min(2.0, base_weight * adaptive_factor))
        
        return self.task_weights.get(task, 1.0)


class AdvancedTrainer:
    """Advanced trainer with all sophisticated techniques."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.progressive_trainer = ProgressiveTrainer(config)
        self.ssl_learner = SelfSupervisedLearning(config)
        self.adaptive_optimizer = AdaptiveOptimizer(config)
        self.regularization_manager = RegularizationManager(config)
        self.multitask_manager = MultiTaskLearning(config)
        
        # Advanced components
        self.ema = None
        self.swa = None
        self.kd_loss = None
        
        # Training state
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.training_history = defaultdict(list)
    
    def setup_training(self, model: nn.Module, teacher_model: Optional[nn.Module] = None):
        """Setup training components."""
        # Setup optimizer and scheduler
        self.optimizer = self.adaptive_optimizer.create_optimizer(model)
        self.scheduler = self.adaptive_optimizer.create_scheduler(self.optimizer)
        
        # Setup EMA
        if self.config.enable_ema:
            self.ema = ExponentialMovingAverage(model, self.config.ema_decay)
        
        # Setup SWA
        if self.config.enable_swa:
            self.swa = StochasticWeightAveraging(model, self.config.swa_start_epoch)
        
        # Setup knowledge distillation
        if self.config.enable_knowledge_distillation and teacher_model is not None:
            self.kd_loss = KnowledgeDistillationLoss(
                self.config.distillation_temperature,
                self.config.distillation_alpha
            )
        
        logger.info("Advanced training setup completed")
    
    def train_epoch(self, model: nn.Module, dataloader, criterion: nn.Module,
                   teacher_model: Optional[nn.Module] = None) -> Dict[str, float]:
        """Train one epoch with advanced techniques."""
        model.train()
        if teacher_model is not None:
            teacher_model.eval()
        
        epoch_losses = defaultdict(float)
        epoch_metrics = defaultdict(float)
        num_batches = len(dataloader)
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            # Progressive training resolution adjustment
            if self.progressive_trainer.should_advance_stage(self.current_epoch):
                self.progressive_trainer.advance_stage()
            
            # Apply data augmentation
            if random.random() < 0.5:
                data, targets_a, targets_b, lam = self.regularization_manager.apply_mixup(data, targets)
                use_mixup = True
            else:
                data, targets_a, targets_b, lam = self.regularization_manager.apply_cutmix(data, targets)
                use_mixup = True
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = model(data)
            
            # Compute losses
            losses = {}
            
            if use_mixup:
                main_loss = self.regularization_manager.mixup_criterion(
                    criterion, outputs, targets_a, targets_b, lam
                )
            else:
                main_loss = criterion(outputs, targets)
            
            losses['main'] = main_loss
            
            # Knowledge distillation loss
            if self.kd_loss is not None and teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs = teacher_model(data)
                kd_loss = self.kd_loss(outputs, teacher_outputs, targets)
                losses['distillation'] = kd_loss
            
            # Multi-task loss
            total_loss = self.multitask_manager.compute_multitask_loss(losses)
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            if self.config.enable_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    self.config.gradient_clip_value
                )
            
            self.optimizer.step()
            
            # Update EMA
            if self.ema is not None:
                self.ema.update()
            
            # Update SWA
            if self.swa is not None:
                self.swa.update(self.current_epoch)
            
            # Record losses
            for loss_name, loss_value in losses.items():
                epoch_losses[loss_name] += loss_value.item()
            
            # Compute accuracy
            if not use_mixup:
                _, predicted = outputs.max(1)
                correct = predicted.eq(targets).sum().item()
                accuracy = correct / targets.size(0)
                epoch_metrics['accuracy'] += accuracy
        
        # Average losses and metrics
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        # Update scheduler
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(epoch_losses['main'])
        else:
            self.scheduler.step()
        
        # Record training history
        self.training_history['epoch'].append(self.current_epoch)
        for key, value in epoch_losses.items():
            self.training_history[f'train_{key}_loss'].append(value)
        for key, value in epoch_metrics.items():
            self.training_history[f'train_{key}'].append(value)
        
        self.current_epoch += 1
        
        return {**epoch_losses, **epoch_metrics}
    
    def validate(self, model: nn.Module, dataloader, criterion: nn.Module) -> Dict[str, float]:
        """Validate model."""
        model.eval()
        
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in dataloader:
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss /= len(dataloader)
        accuracy = correct / total
        
        # Record validation history
        self.training_history['val_loss'].append(val_loss)
        self.training_history['val_accuracy'].append(accuracy)
        
        # Update best accuracy
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
        
        return {'val_loss': val_loss, 'val_accuracy': accuracy}
    
    def save_checkpoint(self, model: nn.Module, filepath: str):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_accuracy': self.best_accuracy,
            'training_history': dict(self.training_history),
            'config': self.config
        }
        
        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.shadow
        
        if self.swa is not None:
            checkpoint['swa_state_dict'] = self.swa.swa_model.state_dict()
        
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, model: nn.Module, filepath: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_accuracy = checkpoint['best_accuracy']
        self.training_history = defaultdict(list, checkpoint['training_history'])
        
        if 'ema_state_dict' in checkpoint and self.ema is not None:
            self.ema.shadow = checkpoint['ema_state_dict']
        
        if 'swa_state_dict' in checkpoint and self.swa is not None:
            self.swa.swa_model.load_state_dict(checkpoint['swa_state_dict'])
        
        logger.info(f"Checkpoint loaded from {filepath}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        return {
            'total_epochs': self.current_epoch,
            'best_accuracy': self.best_accuracy,
            'final_train_loss': self.training_history['train_main_loss'][-1] if self.training_history['train_main_loss'] else 0,
            'final_val_loss': self.training_history['val_loss'][-1] if self.training_history['val_loss'] else 0,
            'final_val_accuracy': self.training_history['val_accuracy'][-1] if self.training_history['val_accuracy'] else 0,
            'training_history': dict(self.training_history),
            'config': self.config
        } 