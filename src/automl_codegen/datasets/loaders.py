"""
Dataset Loaders

This module provides data loading utilities for various datasets used in neural architecture search.
"""

import logging
from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

def get_cifar10_loaders(
    batch_size: int = 128,
    data_dir: str = './data',
    num_workers: int = 4,
    validation_split: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get CIFAR-10 data loaders for training, validation, and testing.
    
    Args:
        batch_size: Batch size for data loaders
        data_dir: Directory to store/load dataset
        num_workers: Number of worker processes for data loading
        validation_split: Fraction of training data to use for validation
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger.info(f"Loading CIFAR-10 dataset with batch_size={batch_size}")
    
    # Define transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=transform_test
    )
    
    # Split training set into train and validation
    train_size = int((1 - validation_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"CIFAR-10 loaded: {len(train_subset)} train, {len(val_subset)} val, {len(test_dataset)} test")
    
    return train_loader, val_loader, test_loader

def get_cifar100_loaders(
    batch_size: int = 128,
    data_dir: str = './data',
    num_workers: int = 4,
    validation_split: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get CIFAR-100 data loaders.
    
    Args:
        batch_size: Batch size for data loaders
        data_dir: Directory to store/load dataset
        num_workers: Number of worker processes for data loading
        validation_split: Fraction of training data to use for validation
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger.info(f"Loading CIFAR-100 dataset with batch_size={batch_size}")
    
    # Define transforms (same as CIFAR-10)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=False, transform=transform_test
    )
    
    # Split training set into train and validation
    train_size = int((1 - validation_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"CIFAR-100 loaded: {len(train_subset)} train, {len(val_subset)} val, {len(test_dataset)} test")
    
    return train_loader, val_loader, test_loader

def get_dataset_info(dataset_name: str) -> dict:
    """Get information about a dataset."""
    dataset_info = {
        'cifar10': {
            'num_classes': 10,
            'input_shape': (3, 32, 32),
            'mean': (0.4914, 0.4822, 0.4465),
            'std': (0.2023, 0.1994, 0.2010),
            'class_names': ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
        },
        'cifar100': {
            'num_classes': 100,
            'input_shape': (3, 32, 32),
            'mean': (0.5071, 0.4867, 0.4408),
            'std': (0.2675, 0.2565, 0.2761),
            'class_names': None  # Too many to list
        },
        'imagenet': {
            'num_classes': 1000,
            'input_shape': (3, 224, 224),
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
            'class_names': None  # Too many to list
        }
    }
    
    return dataset_info.get(dataset_name.lower(), {})

def get_dataloader(dataset_name: str, batch_size: int = 128, **kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get data loaders for any supported dataset.
    
    Args:
        dataset_name: Name of the dataset ('cifar10', 'cifar100', etc.)
        batch_size: Batch size for data loaders
        **kwargs: Additional arguments passed to dataset-specific loader
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'cifar10':
        return get_cifar10_loaders(batch_size=batch_size, **kwargs)
    elif dataset_name == 'cifar100':
        return get_cifar100_loaders(batch_size=batch_size, **kwargs)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

# For quick testing
if __name__ == "__main__":
    # Test CIFAR-10 loading
    train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=32)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test a batch
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"Batch {batch_idx}: data shape {data.shape}, target shape {target.shape}")
        break 