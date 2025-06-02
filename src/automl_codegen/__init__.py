"""
AutoML-CodeGen: Neural Architecture Search with Automatic Code Generation

A revolutionary system that automatically discovers optimal neural network architectures
and generates production-ready deployment code.

Key Features:
- Multi-objective neural architecture search
- Automatic code generation for multiple frameworks
- Hardware-aware optimization
- Distributed training and evaluation
- Real-time monitoring and visualization

Example Usage:
    >>> from automl_codegen import NeuralArchitectureSearch, CodeGenerator
    >>> 
    >>> # Create NAS instance
    >>> nas = NeuralArchitectureSearch(
    ...     task='image_classification',
    ...     dataset='cifar10',
    ...     objectives=['accuracy', 'latency']
    ... )
    >>> 
    >>> # Search for optimal architecture
    >>> best_arch = nas.search(max_epochs=50, population_size=100)
    >>> 
    >>> # Generate production code
    >>> codegen = CodeGenerator(target_framework='pytorch')
    >>> code = codegen.generate(best_arch, optimizations=['quantization'])
    >>> code.save('generated_model.py')
"""

__version__ = "1.0.0"
__author__ = "AutoML-CodeGen Team"
__email__ = "automl-codegen@research.ai"
__license__ = "MIT"

# Core API imports - only import what's actually working
try:
    from .search.nas import NeuralArchitectureSearch
except ImportError:
    NeuralArchitectureSearch = None

try:
    from .codegen.generator import CodeGenerator
except ImportError:
    CodeGenerator = None

try:
    from .evaluation.trainer import ArchitectureTrainer
except ImportError:
    ArchitectureTrainer = None

try:
    from .utils.config import Config
    from .utils.logger import Logger
    from .utils.build_info import get_build_info
except ImportError:
    Config = Logger = get_build_info = None

# Search algorithms - load only if available
try:
    from .search.algorithms.evolutionary import EvolutionarySearch
    from .search.algorithms.darts import DARTSSearch
    from .search.algorithms.reinforcement import ReinforcementSearch
except ImportError:
    EvolutionarySearch = DARTSSearch = ReinforcementSearch = None

# Search space components - load only if available
try:
    from .search.space.search_space import SearchSpace, LayerSpec
except ImportError:
    SearchSpace = LayerSpec = None

# Objectives and metrics - load only if available  
try:
    from .search.objectives.multi_objective import MultiObjectiveOptimizer, ParetoOptimizer, Objective
except ImportError:
    MultiObjectiveOptimizer = ParetoOptimizer = Objective = None

# Utilities - load only if available
try:
    from .utils.profiler import PerformanceProfiler
    from .utils.monitoring import TrainingMonitor
except ImportError:
    PerformanceProfiler = TrainingMonitor = None

# Exception classes
class AutoMLCodeGenError(Exception):
    """Base exception for AutoML-CodeGen errors."""
    pass

class SearchError(AutoMLCodeGenError):
    """Raised when architecture search fails."""
    pass

class CodeGenerationError(AutoMLCodeGenError):
    """Raised when code generation fails."""
    pass

class EvaluationError(AutoMLCodeGenError):
    """Raised when architecture evaluation fails."""
    pass

class ConfigurationError(AutoMLCodeGenError):
    """Raised when configuration is invalid."""
    pass

# Public API - only include what's actually loaded
__all__ = [
    # Core classes (if available)
    "NeuralArchitectureSearch",
    "CodeGenerator", 
    "ArchitectureTrainer",
    "Config",
    "Logger",
    "get_build_info",
    
    # Search algorithms (if available)
    "EvolutionarySearch",
    
    # Search space (if available)
    "SearchSpace",
    "LayerSpec",
    
    # Objectives (if available)
    "MultiObjectiveOptimizer",
    "ParetoOptimizer",
    "Objective",
    
    # Utilities (if available)
    "PerformanceProfiler",
    "TrainingMonitor",
    
    # Exceptions (always available)
    "AutoMLCodeGenError",
    "SearchError",
    "CodeGenerationError",
    "EvaluationError", 
    "ConfigurationError",
]

# Remove None values from __all__
__all__ = [name for name in __all__ if globals().get(name) is not None]

# Version info
def get_version():
    """Get the current version of AutoML-CodeGen."""
    return __version__

# Configuration defaults
DEFAULT_CONFIG = {
    "search": {
        "algorithm": "evolutionary",
        "population_size": 50,
        "num_generations": 20,
        "mutation_rate": 0.1,
        "crossover_rate": 0.8,
        "early_stopping": True,
        "patience": 5,
    },
    "evaluation": {
        "max_epochs": 50,
        "batch_size": 128,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "scheduler": "cosine",
        "validation_split": 0.2,
    },
    "codegen": {
        "target_framework": "pytorch",
        "optimization_level": 2,
        "include_training": True,
        "include_inference": True,
        "code_style": "black",
    },
    "hardware": {
        "target_device": "gpu",
        "memory_limit": "8GB",
        "latency_target": 100,  # ms
        "energy_budget": 1.0,   # Watts
    },
    "logging": {
        "level": "INFO",
        "save_checkpoints": True,
        "log_frequency": 10,
        "use_wandb": False,
        "use_tensorboard": True,
    }
}

# Fast build info without heavy imports
def get_build_info_fast():
    """Get build information quickly without importing heavy libraries."""
    import sys
    
    # Only import torch if really needed
    torch_version = "not available"
    cuda_available = False
    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
    except ImportError:
        pass
    
    info = {
        "version": __version__,
        "python": sys.version.split()[0],
        "pytorch": torch_version,
        "cuda_available": cuda_available,
    }
    return info

# Initialize logging (lightweight)
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler()) 