"""
Data utilities for nanoNAS framework.

This module provides comprehensive data loading and preprocessing
utilities for common datasets used in neural architecture search.
"""

import os
import logging
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and preprocessing."""
    
    # Dataset parameters
    name: str = 'cifar10'  # 'cifar10', 'cifar100', 'mnist', 'fashion_mnist'
    data_dir: str = './data'
    download: bool = True
    
    # Data splits
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    use_test_as_val: bool = False  # Use test set as validation
    
    # Data loading
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_train: bool = True
    
    # Data augmentation
    use_augmentation: bool = True
    normalize: bool = True
    resize_size: Optional[int] = None
    crop_size: Optional[int] = None
    
    # Custom augmentation parameters
    cutout: bool = False
    cutout_length: int = 16
    auto_augment: bool = False
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.train_ratio + self.val_ratio > 1.0:
            raise ValueError("train_ratio + val_ratio cannot exceed 1.0")
        
        # Create data directory
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)


class Cutout:
    """Cutout augmentation."""
    
    def __init__(self, length: int):
        """
        Initialize Cutout augmentation.
        
        Args:
            length: Length of the cutout square
        """
        self.length = length
    
    def __call__(self, img):
        """Apply cutout to image."""
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        
        y = np.random.randint(h)
        x = np.random.randint(w)
        
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)
        
        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        
        return img


def get_dataset_stats(dataset_name: str) -> Dict[str, Any]:
    """
    Get dataset statistics (mean, std, number of classes, input shape).
    
    Args:
        dataset_name: Name of the dataset
    
    Returns:
        Dictionary containing dataset statistics
    """
    stats = {
        'cifar10': {
            'mean': (0.4914, 0.4822, 0.4465),
            'std': (0.2023, 0.1994, 0.2010),
            'num_classes': 10,
            'input_shape': (3, 32, 32),
            'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
        },
        'cifar100': {
            'mean': (0.5071, 0.4867, 0.4408),
            'std': (0.2675, 0.2565, 0.2761),
            'num_classes': 100,
            'input_shape': (3, 32, 32),
            'classes': None  # Too many to list
        },
        'mnist': {
            'mean': (0.1307,),
            'std': (0.3081,),
            'num_classes': 10,
            'input_shape': (1, 28, 28),
            'classes': [str(i) for i in range(10)]
        },
        'fashion_mnist': {
            'mean': (0.2860,),
            'std': (0.3530,),
            'num_classes': 10,
            'input_shape': (1, 28, 28),
            'classes': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        }
    }
    
    if dataset_name not in stats:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return stats[dataset_name]


def get_transforms(config: DatasetConfig, train: bool = True) -> transforms.Compose:
    """
    Get data transforms for training or validation.
    
    Args:
        config: Dataset configuration
        train: Whether to apply training transforms (augmentation)
    
    Returns:
        Composed transforms
    """
    dataset_stats = get_dataset_stats(config.name)
    
    transform_list = []
    
    # Resize if specified
    if config.resize_size is not None:
        transform_list.append(transforms.Resize(config.resize_size))
    
    # Crop if specified
    if config.crop_size is not None:
        if train and config.use_augmentation:
            transform_list.append(transforms.RandomCrop(config.crop_size, padding=4))
        else:
            transform_list.append(transforms.CenterCrop(config.crop_size))
    
    # Training augmentations
    if train and config.use_augmentation:
        if config.name in ['cifar10', 'cifar100']:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
            
            if config.auto_augment:
                transform_list.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
    
    # Convert to tensor
    transform_list.append(transforms.ToTensor())
    
    # Normalization
    if config.normalize:
        transform_list.append(
            transforms.Normalize(dataset_stats['mean'], dataset_stats['std'])
        )
    
    # Post-tensor augmentations
    if train and config.use_augmentation:
        if config.cutout:
            transform_list.append(Cutout(config.cutout_length))
    
    return transforms.Compose(transform_list)


def get_dataset_loaders(config: DatasetConfig) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Get data loaders for training, validation, and optionally test sets.
    
    Args:
        config: Dataset configuration
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger = logging.getLogger(__name__)
    
    # Get transforms
    train_transform = get_transforms(config, train=True)
    val_transform = get_transforms(config, train=False)
    
    # Load datasets
    if config.name == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root=config.data_dir,
            train=True,
            download=config.download,
            transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=config.data_dir,
            train=False,
            download=config.download,
            transform=val_transform
        )
    elif config.name == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(
            root=config.data_dir,
            train=True,
            download=config.download,
            transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=config.data_dir,
            train=False,
            download=config.download,
            transform=val_transform
        )
    elif config.name == 'mnist':
        train_dataset = torchvision.datasets.MNIST(
            root=config.data_dir,
            train=True,
            download=config.download,
            transform=train_transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root=config.data_dir,
            train=False,
            download=config.download,
            transform=val_transform
        )
    elif config.name == 'fashion_mnist':
        train_dataset = torchvision.datasets.FashionMNIST(
            root=config.data_dir,
            train=True,
            download=config.download,
            transform=train_transform
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root=config.data_dir,
            train=False,
            download=config.download,
            transform=val_transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {config.name}")
    
    # Handle validation split
    if config.use_test_as_val:
        # Use test set as validation
        val_dataset = test_dataset
        test_loader = None
        logger.info(f"Using test set as validation ({len(val_dataset)} samples)")
    else:
        # Split training set into train and validation
        total_train = len(train_dataset)
        train_size = int(config.train_ratio * total_train)
        val_size = total_train - train_size
        
        train_dataset, val_dataset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Keep test set separate
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
        
        logger.info(f"Dataset split - Train: {train_size}, Val: {val_size}, Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle_train,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True  # For consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    logger.info(f"Created data loaders for {config.name}")
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    return train_loader, val_loader, test_loader


def get_quick_loaders(
    dataset_name: str = 'cifar10',
    batch_size: int = 128,
    data_dir: str = './data'
) -> Tuple[DataLoader, DataLoader]:
    """
    Quick function to get basic train and validation loaders.
    
    Args:
        dataset_name: Name of the dataset
        batch_size: Batch size for data loaders
        data_dir: Directory to store data
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    config = DatasetConfig(
        name=dataset_name,
        batch_size=batch_size,
        data_dir=data_dir,
        use_test_as_val=True  # Simple setup
    )
    
    train_loader, val_loader, _ = get_dataset_loaders(config)
    return train_loader, val_loader


class MixUp:
    """
    MixUp augmentation for training.
    
    Reference: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2017)
    """
    
    def __init__(self, alpha: float = 0.2):
        """
        Initialize MixUp.
        
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply MixUp to a batch.
        
        Args:
            x: Input batch
            y: Target batch
        
        Returns:
            Tuple of (mixed_x, y_a, y_b, lambda)
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam


class CutMix:
    """
    CutMix augmentation for training.
    
    Reference: "CutMix: Regularization Strategy to Train Strong Classifiers" (Yun et al., 2019)
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize CutMix.
        
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix to a batch.
        
        Args:
            x: Input batch
            y: Target batch
        
        Returns:
            Tuple of (mixed_x, y_a, y_b, lambda)
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        y_a, y_b = y, y[index]
        
        # Get bounding box
        W, H = x.size(2), x.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        
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
        
        return x, y_a, y_b, lam


def calculate_dataset_stats(dataset, num_samples: int = 1000) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """
    Calculate mean and standard deviation for a dataset.
    
    Args:
        dataset: PyTorch dataset
        num_samples: Number of samples to use for calculation
    
    Returns:
        Tuple of (mean, std) for each channel
    """
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_samples += batch_samples
        
        if total_samples >= num_samples:
            break
    
    mean /= total_samples
    std /= total_samples
    
    return tuple(mean.tolist()), tuple(std.tolist()) 