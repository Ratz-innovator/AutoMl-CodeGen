"""
Neural Architecture Search Strategies
====================================

This module contains various search strategies for neural architecture search:
- Evolutionary algorithms
- DARTS (Differentiable Architecture Search)
- Progressive-DARTS with early stopping and pruning
- Bayesian optimization with Gaussian processes
- Reinforcement learning
- Multi-objective optimization
- Random search
"""

from .evolutionary import EvolutionarySearch
from .darts import DARTSSearch
from .progressive_darts import ProgressiveDARTSSearch
from .bayesian_optimization import BayesianOptimizationSearch
# from .reinforcement import ReinforcementSearch  # Module not implemented yet
from .multiobjective import MultiObjectiveSearch
from .random_search import RandomSearch

__all__ = [
    "EvolutionarySearch",
    "DARTSSearch",
    "ProgressiveDARTSSearch",
    "BayesianOptimizationSearch",
    # "ReinforcementSearch",  # Module not implemented yet
    "MultiObjectiveSearch",
    "RandomSearch",
] 