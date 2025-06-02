"""
Configuration Management System

This module provides a flexible configuration system for AutoML-CodeGen
that supports YAML files, environment variables, and programmatic configuration.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class SearchConfig:
    """Configuration for neural architecture search."""
    algorithm: str = 'evolutionary'
    population_size: int = 50
    num_generations: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elitism_ratio: float = 0.1
    tournament_size: int = 3
    early_stopping: bool = True
    patience: int = 5
    max_age: int = 10
    adaptive_parameters: bool = True

@dataclass
class EvaluationConfig:
    """Configuration for architecture evaluation."""
    max_epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 0.001
    optimizer: str = 'adam'
    scheduler: str = 'cosine'
    validation_split: float = 0.2
    early_stopping: bool = True
    patience: int = 5
    min_epochs: int = 10
    use_mixed_precision: bool = True

@dataclass
class CodeGenConfig:
    """Configuration for code generation."""
    target_framework: str = 'pytorch'
    optimization_level: int = 2
    include_training: bool = True
    include_inference: bool = True
    include_deployment: bool = False
    code_style: str = 'black'
    optimizations: List[str] = field(default_factory=lambda: ['quantization', 'fusion'])
    target_device: str = 'auto'

@dataclass
class HardwareConfig:
    """Configuration for hardware-specific settings."""
    target_device: str = 'gpu'
    memory_limit: str = '8GB'
    latency_target: float = 100.0  # ms
    energy_budget: float = 1.0     # Watts
    batch_size_limit: int = 256
    use_tensorrt: bool = False
    use_quantization: bool = True

@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""
    level: str = 'INFO'
    save_checkpoints: bool = True
    checkpoint_frequency: int = 5
    log_frequency: int = 10
    use_wandb: bool = False
    use_tensorboard: bool = True
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None

@dataclass
class DataConfig:
    """Configuration for dataset handling."""
    dataset_name: str = 'cifar10'
    data_path: Optional[str] = None
    download: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    augmentation: bool = True
    normalization: bool = True
    cache_dataset: bool = False

class Config:
    """
    Main configuration manager for AutoML-CodeGen.
    
    Supports hierarchical configuration loading from:
    - Default values
    - YAML configuration files
    - Environment variables
    - Programmatic updates
    
    Example:
        >>> config = Config()
        >>> config.load_from_file('config.yaml')
        >>> config.search.population_size = 100
        >>> config.save_to_file('updated_config.yaml')
    """
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Optional path to configuration file to load
        """
        # Initialize with default configurations
        self.search = SearchConfig()
        self.evaluation = EvaluationConfig()
        self.codegen = CodeGenConfig()
        self.hardware = HardwareConfig()
        self.logging = LoggingConfig()
        self.data = DataConfig()
        
        # Load from file if provided
        if config_file:
            self.load_from_file(config_file)
        
        # Override with environment variables
        self._load_from_environment()
        
        logger.debug("Configuration initialized")
    
    def load_from_file(self, config_file: Union[str, Path]) -> None:
        """
        Load configuration from YAML file.
        
        Args:
            config_file: Path to configuration file
        """
        config_file = Path(config_file)
        
        if not config_file.exists():
            logger.warning(f"Configuration file not found: {config_file}")
            return
        
        try:
            with open(config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            self._update_from_dict(config_dict)
            logger.info(f"Loaded configuration from {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_file}: {e}")
            raise
    
    def save_to_file(self, config_file: Union[str, Path]) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            config_file: Path to save configuration file
        """
        config_file = Path(config_file)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            config_dict = self.to_dict()
            
            with open(config_file, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Saved configuration to {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_file}: {e}")
            raise
    
    def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        env_mappings = {
            'AUTOML_SEARCH_ALGORITHM': ('search', 'algorithm'),
            'AUTOML_POPULATION_SIZE': ('search', 'population_size'),
            'AUTOML_MUTATION_RATE': ('search', 'mutation_rate'),
            'AUTOML_CROSSOVER_RATE': ('search', 'crossover_rate'),
            'AUTOML_MAX_EPOCHS': ('evaluation', 'max_epochs'),
            'AUTOML_BATCH_SIZE': ('evaluation', 'batch_size'),
            'AUTOML_LEARNING_RATE': ('evaluation', 'learning_rate'),
            'AUTOML_TARGET_FRAMEWORK': ('codegen', 'target_framework'),
            'AUTOML_OPTIMIZATION_LEVEL': ('codegen', 'optimization_level'),
            'AUTOML_TARGET_DEVICE': ('hardware', 'target_device'),
            'AUTOML_MEMORY_LIMIT': ('hardware', 'memory_limit'),
            'AUTOML_LOG_LEVEL': ('logging', 'level'),
            'AUTOML_USE_WANDB': ('logging', 'use_wandb'),
            'AUTOML_WANDB_PROJECT': ('logging', 'wandb_project'),
            'AUTOML_DATASET': ('data', 'dataset_name'),
            'AUTOML_DATA_PATH': ('data', 'data_path'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert value to appropriate type
                converted_value = self._convert_env_value(value)
                setattr(getattr(self, section), key, converted_value)
                logger.debug(f"Set {section}.{key} = {converted_value} from {env_var}")
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Try boolean
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for section_name, section_config in config_dict.items():
            if hasattr(self, section_name) and isinstance(section_config, dict):
                section = getattr(self, section_name)
                for key, value in section_config.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'search': asdict(self.search),
            'evaluation': asdict(self.evaluation),
            'codegen': asdict(self.codegen),
            'hardware': asdict(self.hardware),
            'logging': asdict(self.logging),
            'data': asdict(self.data)
        }
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates in section.key format or nested dict
        """
        if isinstance(updates, dict):
            # Handle both flat and nested dictionaries
            for key, value in updates.items():
                if '.' in key:
                    # Handle dot notation (e.g., 'search.population_size')
                    section_name, attr_name = key.split('.', 1)
                    if hasattr(self, section_name):
                        section = getattr(self, section_name)
                        if hasattr(section, attr_name):
                            setattr(section, attr_name, value)
                elif hasattr(self, key) and isinstance(value, dict):
                    # Handle nested dictionary
                    section = getattr(self, key)
                    for attr_name, attr_value in value.items():
                        if hasattr(section, attr_name):
                            setattr(section, attr_name, attr_value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key in section.attribute format
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        if '.' in key:
            section_name, attr_name = key.split('.', 1)
            if hasattr(self, section_name):
                section = getattr(self, section_name)
                return getattr(section, attr_name, default)
        
        return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key in section.attribute format
            value: Value to set
        """
        if '.' in key:
            section_name, attr_name = key.split('.', 1)
            if hasattr(self, section_name):
                section = getattr(self, section_name)
                if hasattr(section, attr_name):
                    setattr(section, attr_name, value)
                else:
                    logger.warning(f"Unknown attribute: {attr_name} in section {section_name}")
            else:
                logger.warning(f"Unknown section: {section_name}")
        else:
            logger.warning(f"Invalid key format: {key}. Use section.attribute format.")
    
    def validate(self) -> bool:
        """
        Validate configuration values.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        errors = []
        
        # Validate search configuration
        if self.search.population_size <= 0:
            errors.append("search.population_size must be positive")
        
        if not 0 <= self.search.mutation_rate <= 1:
            errors.append("search.mutation_rate must be between 0 and 1")
        
        if not 0 <= self.search.crossover_rate <= 1:
            errors.append("search.crossover_rate must be between 0 and 1")
        
        if not 0 <= self.search.elitism_ratio <= 1:
            errors.append("search.elitism_ratio must be between 0 and 1")
        
        # Validate evaluation configuration
        if self.evaluation.max_epochs <= 0:
            errors.append("evaluation.max_epochs must be positive")
        
        if self.evaluation.batch_size <= 0:
            errors.append("evaluation.batch_size must be positive")
        
        if self.evaluation.learning_rate <= 0:
            errors.append("evaluation.learning_rate must be positive")
        
        # Validate code generation configuration
        valid_frameworks = ['pytorch', 'tensorflow', 'tf', 'onnx', 'tensorrt', 'jax']
        if self.codegen.target_framework not in valid_frameworks:
            errors.append(f"codegen.target_framework must be one of {valid_frameworks}")
        
        if not 0 <= self.codegen.optimization_level <= 3:
            errors.append("codegen.optimization_level must be between 0 and 3")
        
        # Validate hardware configuration
        valid_devices = ['gpu', 'cpu', 'mobile', 'edge', 'auto']
        if self.hardware.target_device not in valid_devices:
            errors.append(f"hardware.target_device must be one of {valid_devices}")
        
        # Log errors
        if errors:
            for error in errors:
                logger.error(f"Configuration validation error: {error}")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def get_summary(self) -> str:
        """Get a summary of the current configuration."""
        summary = []
        summary.append("AutoML-CodeGen Configuration Summary")
        summary.append("=" * 40)
        
        summary.append(f"Search Algorithm: {self.search.algorithm}")
        summary.append(f"Population Size: {self.search.population_size}")
        summary.append(f"Generations: {self.search.num_generations}")
        summary.append(f"Mutation Rate: {self.search.mutation_rate}")
        summary.append(f"Crossover Rate: {self.search.crossover_rate}")
        summary.append("")
        
        summary.append(f"Max Epochs: {self.evaluation.max_epochs}")
        summary.append(f"Batch Size: {self.evaluation.batch_size}")
        summary.append(f"Learning Rate: {self.evaluation.learning_rate}")
        summary.append(f"Optimizer: {self.evaluation.optimizer}")
        summary.append("")
        
        summary.append(f"Target Framework: {self.codegen.target_framework}")
        summary.append(f"Optimization Level: {self.codegen.optimization_level}")
        summary.append(f"Include Training: {self.codegen.include_training}")
        summary.append("")
        
        summary.append(f"Target Device: {self.hardware.target_device}")
        summary.append(f"Memory Limit: {self.hardware.memory_limit}")
        summary.append(f"Latency Target: {self.hardware.latency_target}ms")
        summary.append("")
        
        summary.append(f"Dataset: {self.data.dataset_name}")
        summary.append(f"Logging Level: {self.logging.level}")
        summary.append(f"Use W&B: {self.logging.use_wandb}")
        
        return '\n'.join(summary)
    
    def create_experiment_config(
        self,
        experiment_name: str,
        overrides: Optional[Dict[str, Any]] = None
    ) -> 'Config':
        """
        Create a configuration for a specific experiment.
        
        Args:
            experiment_name: Name of the experiment
            overrides: Configuration overrides for this experiment
            
        Returns:
            New configuration instance for the experiment
        """
        # Create a copy of current config
        experiment_config = Config()
        experiment_config._update_from_dict(self.to_dict())
        
        # Apply experiment-specific overrides
        if overrides:
            experiment_config.update(overrides)
        
        # Add experiment name to logging config
        if experiment_config.logging.wandb_project:
            experiment_config.logging.wandb_project = f"{experiment_config.logging.wandb_project}_{experiment_name}"
        
        return experiment_config
    
    def compare_with(self, other: 'Config') -> Dict[str, Any]:
        """
        Compare this configuration with another.
        
        Args:
            other: Another configuration instance
            
        Returns:
            Dictionary of differences
        """
        current_dict = self.to_dict()
        other_dict = other.to_dict()
        
        differences = {}
        
        for section in current_dict:
            if section in other_dict:
                section_diffs = {}
                for key in current_dict[section]:
                    if key in other_dict[section]:
                        current_val = current_dict[section][key]
                        other_val = other_dict[section][key]
                        if current_val != other_val:
                            section_diffs[key] = {
                                'current': current_val,
                                'other': other_val
                            }
                
                if section_diffs:
                    differences[section] = section_diffs
        
        return differences 