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
from .search.reinforcement import ReinforcementSearch
from .search.multiobjective import MultiObjectiveSearch
from .models.supernet import SuperNet
from .models.operations import *
from .benchmarks.evaluator import ModelEvaluator
from .benchmarks.datasets import get_dataset
from .visualization.architecture_viz import ArchitectureVisualizer
from .utils.metrics import compute_model_stats

# Main API functions
from .api import search, benchmark, visualize

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
    "ReinforcementSearch",
    "MultiObjectiveSearch",
    
    # Models and operations
    "SuperNet",
    
    # Benchmarking
    "ModelEvaluator",
    "get_dataset",
    
    # Visualization
    "ArchitectureVisualizer",
    
    # Utilities
    "compute_model_stats",
]

# Configuration for better user experience
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

def get_version():
    """Get the current version of nanoNAS."""
    return __version__

def list_search_strategies():
    """List all available search strategies."""
    return ["evolutionary", "darts", "reinforcement", "multiobjective", "random"]

def list_datasets():
    """List all supported datasets."""
    return ["cifar10", "cifar100", "mnist", "fashion_mnist", "imagenet_subset"]

def list_search_spaces():
    """List all available search spaces."""
    return ["nano", "mobile", "resnet_like", "densenet_like", "custom"] 