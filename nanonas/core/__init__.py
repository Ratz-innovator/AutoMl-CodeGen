"""
Core components for nanoNAS
===========================

This module contains the fundamental building blocks for neural architecture search:
- Architecture representation and encoding
- Search space definitions
- Configuration management
- Base classes for search strategies
"""

from .architecture import Architecture, SearchSpace
from .config import SearchConfig, ExperimentConfig, load_config
from .base import BaseSearchStrategy, BaseOperation

__all__ = [
    "Architecture",
    "SearchSpace", 
    "SearchConfig",
    "ExperimentConfig",
    "load_config",
    "BaseSearchStrategy",
    "BaseOperation",
] 