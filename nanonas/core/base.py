"""
Base Classes and Interfaces for nanoNAS
=======================================

This module provides abstract base classes and common interfaces
for search strategies, operations, and other core components.

Key Features:
- Abstract base classes for search strategies
- Common interfaces for operations and models
- Shared utility functions and decorators
- Type hints and documentation standards
"""

import abc
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn

from .architecture import Architecture
from .config import ExperimentConfig


class BaseSearchStrategy(abc.ABC):
    """
    Abstract base class for neural architecture search strategies.
    
    All search strategy implementations should inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize search strategy.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.search_metrics = {}
        
    @abc.abstractmethod
    def search(self) -> Architecture:
        """
        Run the search algorithm to find the best architecture.
        
        Returns:
            Best architecture found
        """
        pass
    
    def get_search_metrics(self) -> Dict[str, Any]:
        """
        Get metrics and statistics from the search process.
        
        Returns:
            Dictionary containing search metrics
        """
        return self.search_metrics.copy()
    
    def save_checkpoint(self, filepath: str):
        """
        Save search state for resuming later.
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'config': self.config,
            'search_metrics': self.search_metrics,
            'strategy_state': self._get_strategy_state()
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        Load search state from checkpoint.
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath)
        self.config = checkpoint['config']
        self.search_metrics = checkpoint['search_metrics']
        self._set_strategy_state(checkpoint['strategy_state'])
        self.logger.info(f"Checkpoint loaded from {filepath}")
    
    def _get_strategy_state(self) -> Dict[str, Any]:
        """
        Get strategy-specific state for checkpointing.
        
        Returns:
            Strategy state dictionary
        """
        return {}
    
    def _set_strategy_state(self, state: Dict[str, Any]):
        """
        Set strategy-specific state from checkpoint.
        
        Args:
            state: Strategy state dictionary
        """
        pass


class BaseOperation(nn.Module, abc.ABC):
    """
    Abstract base class for neural network operations.
    
    All operation implementations should inherit from this class.
    """
    
    def __init__(self, channels: int, **kwargs):
        """
        Initialize operation.
        
        Args:
            channels: Number of input/output channels
            **kwargs: Additional operation-specific parameters
        """
        super().__init__()
        self.channels = channels
        self.operation_type = self.__class__.__name__
        
    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the operation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass
    
    def get_operation_info(self) -> Dict[str, Any]:
        """
        Get information about this operation.
        
        Returns:
            Operation information dictionary
        """
        return {
            'type': self.operation_type,
            'channels': self.channels,
            'parameters': sum(p.numel() for p in self.parameters()),
        }
    
    def compute_flops(self, input_shape: Tuple[int, ...]) -> int:
        """
        Compute FLOPs for this operation given input shape.
        
        Args:
            input_shape: Input tensor shape (without batch dimension)
            
        Returns:
            Number of FLOPs
        """
        # Default implementation - should be overridden by specific operations
        return 0


class BaseModel(nn.Module, abc.ABC):
    """
    Abstract base class for neural network models built from architectures.
    """
    
    def __init__(self, 
                 input_channels: int = 3,
                 num_classes: int = 10,
                 **kwargs):
        """
        Initialize model.
        
        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes
            **kwargs: Additional model-specific parameters
        """
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        
    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Model information dictionary
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'input_channels': self.input_channels,
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        }


class BaseEvaluator(abc.ABC):
    """
    Abstract base class for model evaluators.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize evaluator.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abc.abstractmethod
    def evaluate(self, model: nn.Module) -> Dict[str, float]:
        """
        Evaluate a model and return metrics.
        
        Args:
            model: Model to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    @abc.abstractmethod
    def quick_evaluate(self, model: nn.Module) -> Dict[str, float]:
        """
        Quick evaluation for search (may use proxies or fewer epochs).
        
        Args:
            model: Model to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass


class BaseVisualizer(abc.ABC):
    """
    Abstract base class for visualization components.
    """
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        """
        Initialize visualizer.
        
        Args:
            config: Optional experiment configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abc.abstractmethod
    def visualize(self, *args, **kwargs) -> str:
        """
        Create visualization and return path to saved file.
        
        Returns:
            Path to visualization file
        """
        pass


# Utility decorators and functions

def timer(func):
    """
    Decorator to time function execution.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Log timing information
        logger = logging.getLogger(func.__module__)
        logger.debug(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        
        return result
    
    return wrapper


def validate_architecture(architecture: Architecture) -> bool:
    """
    Validate that an architecture is well-formed.
    
    Args:
        architecture: Architecture to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check if architecture has valid encoding or graph
        if architecture.encoding is None and architecture.graph is None:
            return False
        
        # Check if search space is valid
        if architecture.search_space is None:
            return False
        
        # Try to build model (basic validation)
        model = architecture.to_model()
        if model is None:
            return False
        
        return True
        
    except Exception:
        return False


def safe_model_forward(model: nn.Module, 
                      input_tensor: torch.Tensor,
                      max_retries: int = 3) -> Optional[torch.Tensor]:
    """
    Safely perform forward pass with error handling.
    
    Args:
        model: Model to run
        input_tensor: Input tensor
        max_retries: Maximum number of retries
        
    Returns:
        Output tensor or None if failed
    """
    for attempt in range(max_retries):
        try:
            with torch.no_grad():
                output = model(input_tensor)
                return output
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Forward pass attempt {attempt + 1} failed: {e}")
            
            if attempt == max_retries - 1:
                logger.error(f"All {max_retries} forward pass attempts failed")
                return None
    
    return None


@dataclass
class SearchResult:
    """
    Container for search results and metadata.
    """
    best_architecture: Architecture
    search_time: float
    total_evaluations: int
    convergence_history: List[float]
    search_strategy: str
    final_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'best_architecture': self.best_architecture.to_dict(),
            'search_time': self.search_time,
            'total_evaluations': self.total_evaluations,
            'convergence_history': self.convergence_history,
            'search_strategy': self.search_strategy,
            'final_metrics': self.final_metrics,
        }


class SearchProgressCallback:
    """
    Callback interface for monitoring search progress.
    """
    
    def on_search_start(self, strategy: BaseSearchStrategy):
        """Called when search starts."""
        pass
    
    def on_generation_start(self, generation: int, strategy: BaseSearchStrategy):
        """Called at the start of each generation/iteration."""
        pass
    
    def on_architecture_evaluated(self, 
                                 architecture: Architecture, 
                                 metrics: Dict[str, float],
                                 strategy: BaseSearchStrategy):
        """Called after each architecture evaluation."""
        pass
    
    def on_generation_end(self, generation: int, strategy: BaseSearchStrategy):
        """Called at the end of each generation/iteration."""
        pass
    
    def on_search_end(self, result: SearchResult, strategy: BaseSearchStrategy):
        """Called when search completes."""
        pass


class LoggingCallback(SearchProgressCallback):
    """
    Simple logging callback for search progress.
    """
    
    def __init__(self, log_frequency: int = 1):
        """
        Initialize logging callback.
        
        Args:
            log_frequency: How often to log (every N generations)
        """
        self.log_frequency = log_frequency
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def on_search_start(self, strategy: BaseSearchStrategy):
        """Log search start."""
        self.logger.info(f"🚀 Starting {strategy.__class__.__name__} search")
    
    def on_generation_end(self, generation: int, strategy: BaseSearchStrategy):
        """Log generation progress."""
        if generation % self.log_frequency == 0:
            metrics = strategy.get_search_metrics()
            if 'best_fitness_per_generation' in metrics:
                best_fitness = metrics['best_fitness_per_generation'][-1]
                self.logger.info(f"Generation {generation}: Best fitness = {best_fitness:.4f}")
    
    def on_search_end(self, result: SearchResult, strategy: BaseSearchStrategy):
        """Log search completion."""
        self.logger.info(f"✅ Search completed in {result.search_time:.2f}s")
        self.logger.info(f"🏆 Best architecture: {result.best_architecture}")


# Error classes

class NanoNASError(Exception):
    """Base exception for nanoNAS errors."""
    pass


class SearchError(NanoNASError):
    """Exception raised during architecture search."""
    pass


class EvaluationError(NanoNASError):
    """Exception raised during model evaluation."""
    pass


class ConfigurationError(NanoNASError):
    """Exception raised for configuration issues."""
    pass


class ArchitectureError(NanoNASError):
    """Exception raised for architecture-related issues."""
    pass 