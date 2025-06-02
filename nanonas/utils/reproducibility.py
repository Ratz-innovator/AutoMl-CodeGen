"""
Reproducibility utilities for nanoNAS framework.

This module provides utilities for ensuring reproducible results
across different runs and environments.
"""

import os
import random
import logging
from typing import Optional, Dict, Any

import torch
import numpy as np


def set_seed(seed: int, deterministic: bool = True):
    """
    Set random seeds for reproducible results.
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms (slower but more reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For newer PyTorch versions
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    logging.info(f"Random seed set to {seed} (deterministic={deterministic})")


def get_reproducible_config(seed: int = 42) -> Dict[str, Any]:
    """
    Get configuration for reproducible experiments.
    
    Args:
        seed: Random seed
    
    Returns:
        Dictionary with reproducibility settings
    """
    return {
        'seed': seed,
        'deterministic': True,
        'benchmark': False,
        'num_workers': 0,  # Deterministic data loading
        'drop_last': True,  # Consistent batch sizes
    }


def save_environment_info(filepath: str):
    """
    Save environment information for reproducibility.
    
    Args:
        filepath: Path to save environment info
    """
    import platform
    import subprocess
    
    env_info = {
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'pytorch_version': torch.__version__,
        'numpy_version': np.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        env_info.update({
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_count': torch.cuda.device_count(),
        })
    
    # Try to get git commit if available
    try:
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], 
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
        env_info['git_commit'] = git_commit
    except:
        env_info['git_commit'] = 'unknown'
    
    # Save pip list
    try:
        pip_list = subprocess.check_output(
            ['pip', 'list'], 
            stderr=subprocess.DEVNULL
        ).decode('ascii')
        env_info['pip_packages'] = pip_list
    except:
        env_info['pip_packages'] = 'unknown'
    
    import json
    with open(filepath, 'w') as f:
        json.dump(env_info, f, indent=2)
    
    logging.info(f"Environment info saved to {filepath}")


class ReproducibleContext:
    """
    Context manager for reproducible experiments.
    """
    
    def __init__(self, seed: int = 42, deterministic: bool = True):
        """
        Initialize reproducible context.
        
        Args:
            seed: Random seed
            deterministic: Whether to use deterministic algorithms
        """
        self.seed = seed
        self.deterministic = deterministic
        self.original_state = {}
    
    def __enter__(self):
        """Enter context and set up reproducibility."""
        # Save original state
        self.original_state = {
            'random_state': random.getstate(),
            'numpy_state': np.random.get_state(),
            'torch_state': torch.get_rng_state(),
            'torch_cuda_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
        }
        
        # Set reproducible state
        set_seed(self.seed, self.deterministic)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore original state."""
        # Restore original state
        random.setstate(self.original_state['random_state'])
        np.random.set_state(self.original_state['numpy_state'])
        torch.set_rng_state(self.original_state['torch_state'])
        
        if torch.cuda.is_available() and self.original_state['torch_cuda_state'] is not None:
            torch.cuda.set_rng_state_all(self.original_state['torch_cuda_state'])
        
        torch.backends.cudnn.deterministic = self.original_state['cudnn_deterministic']
        torch.backends.cudnn.benchmark = self.original_state['cudnn_benchmark'] 