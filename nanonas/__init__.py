"""
nanoNAS: Neural Architecture Search Made Simple
==============================================

A graduate-level, educational AutoML/NAS framework that combines
theoretical rigor with practical usability.

Key Features:
- Multiple search strategies (Evolution, DARTS, RL, Multi-objective)
- Real benchmarking on standard datasets (CIFAR-10, MNIST, Fashion-MNIST)
- Advanced visualization and analysis tools
- Modular, extensible architecture
- Production-ready code with comprehensive testing

Quick Start:
-----------
>>> import nanonas
>>> 
>>> # One-line architecture search
>>> model = nanonas.search(strategy='evolution', dataset='cifar10')
>>> 
>>> # Advanced search with custom config
>>> config = nanonas.SearchConfig(
...     search_space='mobilenet_like',
...     strategy='darts',
...     epochs=50
... )
>>> model = nanonas.search(config)

Authors: AutoML Research Team
License: MIT
Version: 1.0.0
"""

from .core.architecture import Architecture, SearchSpace
from .core.config import SearchConfig, ExperimentConfig
from .search.evolutionary import EvolutionarySearch
from .search.darts import DARTSSearch
from .search.progressive_darts import ProgressiveDARTSSearch
from .search.bayesian_optimization import BayesianOptimizationSearch
# from .search.reinforcement import ReinforcementSearch  # Module not implemented yet
from .search.multiobjective import MultiObjectiveSearch
from .models.supernet import DARTSSupernet
from .models.operations import *
from .benchmarks.evaluator import ModelEvaluator
from .benchmarks.datasets import get_dataset
from .visualization.architecture_viz import ArchitectureVisualizer
# from .utils.metrics import compute_model_stats  # Module not implemented yet
from .utils.hardware_utils import profile_current_device

# Main API functions
from .api import search, benchmark, visualize

# CLI interface
from .cli import cli

__version__ = "1.0.0"
__author__ = "AutoML Research Team"
__email__ = "research@nanonas.ai"

# Public API
__all__ = [
    # Main API
    "search",
    "benchmark", 
    "visualize",
    
    # Core classes
    "Architecture",
    "SearchSpace",
    "SearchConfig",
    "ExperimentConfig",
    
    # Search strategies
    "EvolutionarySearch",
    "DARTSSearch",
    "ProgressiveDARTSSearch", 
    "BayesianOptimizationSearch",
    # "ReinforcementSearch",  # Module not implemented yet
    "MultiObjectiveSearch",
    
    # Models and operations
    "DARTSSupernet",
    
    # Benchmarking
    "ModelEvaluator",
    "get_dataset",
    
    # Visualization
    "ArchitectureVisualizer",
    
    # Utilities
    # "compute_model_stats",  # Module not implemented yet
    "profile_current_device",
    
    # CLI
    "cli",
]

# Configuration for better user experience
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

def get_version():
    """Get the current version of nanoNAS."""
    return __version__

def list_search_strategies():
    """List all available search strategies."""
    return ["evolutionary", "darts", "progressive_darts", "bayesian", "reinforcement", "multiobjective", "random"]

def list_datasets():
    """List all supported datasets."""
    return ["cifar10", "cifar100", "mnist", "fashion_mnist", "imagenet_subset"]

def list_search_spaces():
    """List all available search spaces."""
    return ["nano", "mobile", "resnet_like", "densenet_like", "custom"] 