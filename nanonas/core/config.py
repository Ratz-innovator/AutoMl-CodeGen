"""
Configuration Management for nanoNAS
====================================

This module provides a comprehensive configuration system for neural architecture search
experiments, supporting YAML-based configs, hyperparameter validation, and reproducibility.

Key Features:
- Dataclass-based configuration with type checking
- YAML serialization/deserialization
- Hyperparameter validation and constraints
- Experiment reproducibility (seeds, logging)
- Modular configs for different components
"""

import os
import yaml
import json
import random
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from omegaconf import OmegaConf, DictConfig


@dataclass
class SearchConfig:
    """Configuration for neural architecture search."""
    
    # Search strategy
    strategy: str = "evolutionary"  # evolutionary, darts, reinforcement, multiobjective
    search_space: str = "nano"  # nano, mobile, resnet_like, custom
    
    # Search parameters
    population_size: int = 20
    generations: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elitism_ratio: float = 0.2
    
    # DARTS specific
    darts_epochs: int = 50
    darts_lr: float = 0.025
    darts_weight_decay: float = 3e-4
    
    # RL specific
    rl_episodes: int = 1000
    rl_lr: float = 1e-3
    rl_discount_factor: float = 0.99
    
    # Multi-objective specific
    objectives: List[str] = field(default_factory=lambda: ["accuracy", "latency"])
    pareto_front_size: int = 50
    
    # General search constraints
    max_search_time: float = 3600.0  # 1 hour
    early_stopping_patience: int = 10
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate configuration parameters."""
        if self.strategy not in ["evolutionary", "darts", "reinforcement", "multiobjective", "random"]:
            raise ValueError(f"Unknown search strategy: {self.strategy}")
        
        if self.population_size < 2:
            raise ValueError("Population size must be at least 2")
        
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("Mutation rate must be between 0 and 1")
        
        if not 0 <= self.crossover_rate <= 1:
            raise ValueError("Crossover rate must be between 0 and 1")
        
        if not 0 <= self.elitism_ratio <= 1:
            raise ValueError("Elitism ratio must be between 0 and 1")


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and preprocessing."""
    
    name: str = "cifar10"  # cifar10, cifar100, mnist, fashion_mnist, imagenet
    batch_size: int = 128
    num_workers: int = 4
    
    # Data augmentation
    use_augmentation: bool = True
    cutout: bool = True
    cutout_length: int = 16
    auto_augment: bool = False
    
    # Dataset splitting
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Data loading
    pin_memory: bool = True
    persistent_workers: bool = True
    
    def __post_init__(self):
        """Validate dataset configuration."""
        if abs(self.train_split + self.val_split + self.test_split - 1.0) > 1e-6:
            raise ValueError("Dataset splits must sum to 1.0")
        
        if self.batch_size < 1:
            raise ValueError("Batch size must be positive")


@dataclass
class TrainingConfig:
    """Configuration for model training and evaluation."""
    
    # Training parameters
    epochs: int = 200
    learning_rate: float = 0.025
    momentum: float = 0.9
    weight_decay: float = 3e-4
    
    # Learning rate scheduling
    lr_scheduler: str = "cosine"  # cosine, step, exponential, none
    lr_decay_epochs: List[int] = field(default_factory=lambda: [100, 150])
    lr_decay_rate: float = 0.1
    warmup_epochs: int = 5
    
    # Optimization
    optimizer: str = "sgd"  # sgd, adam, adamw, rmsprop
    gradient_clip_norm: float = 5.0
    label_smoothing: float = 0.1
    
    # Regularization
    dropout_rate: float = 0.2
    droppath_rate: float = 0.3
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 20
    min_delta: float = 1e-4
    
    # Checkpointing
    save_best: bool = True
    save_every_n_epochs: int = 50
    
    def __post_init__(self):
        """Validate training configuration."""
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        if self.epochs < 1:
            raise ValueError("Number of epochs must be positive")


@dataclass
class ModelConfig:
    """Configuration for model architecture and initialization."""
    
    # Model structure
    input_channels: int = 3
    num_classes: int = 10
    base_channels: int = 16
    
    # Architecture constraints
    max_depth: int = 8
    max_parameters: float = 10e6  # 10M parameters
    max_flops: float = 600e6  # 600M FLOPs
    
    # Model initialization
    init_scheme: str = "kaiming_normal"  # kaiming_normal, xavier_normal, normal
    init_gain: float = 1.0
    
    # Batch normalization
    use_bn: bool = True
    bn_momentum: float = 0.1
    bn_eps: float = 1e-5
    
    # Activation functions
    activation: str = "relu"  # relu, swish, gelu, leaky_relu
    
    def __post_init__(self):
        """Validate model configuration."""
        if self.input_channels < 1:
            raise ValueError("Input channels must be positive")
        
        if self.num_classes < 1:
            raise ValueError("Number of classes must be positive")


@dataclass
class ExperimentConfig:
    """Main experiment configuration combining all components."""
    
    # Experiment metadata
    name: str = "nanonas_experiment"
    description: str = "Neural Architecture Search Experiment"
    tags: List[str] = field(default_factory=list)
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Component configurations
    search: SearchConfig = field(default_factory=SearchConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # Logging and output
    output_dir: str = "./results"
    log_level: str = "INFO"
    log_frequency: int = 10
    
    # Hardware settings
    device: str = "auto"  # auto, cpu, cuda, mps
    num_gpus: int = 1
    mixed_precision: bool = True
    
    # Experiment tracking
    use_wandb: bool = False
    wandb_project: str = "nanonas"
    wandb_entity: Optional[str] = None
    
    def __post_init__(self):
        """Setup and validate experiment configuration."""
        # Set up reproducibility
        if self.deterministic:
            self.set_seed(self.seed)
        
        # Auto-detect device
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Make CuDNN deterministic
        if self.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def save(self, path: Union[str, Path]):
        """Save configuration to YAML file."""
        path = Path(path)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create configuration from dictionary."""
        # Create nested configs
        search_config = SearchConfig(**config_dict.get('search', {}))
        dataset_config = DatasetConfig(**config_dict.get('dataset', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        
        # Remove nested configs from main dict
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ['search', 'dataset', 'training', 'model']}
        
        return cls(
            search=search_config,
            dataset=dataset_config,
            training=training_config,
            model=model_config,
            **main_config
        )
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        path = Path(path)
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)


def load_config(config_path: Union[str, Path, Dict[str, Any]]) -> ExperimentConfig:
    """
    Load configuration from various sources.
    
    Args:
        config_path: Path to YAML file, dictionary, or OmegaConf object
        
    Returns:
        ExperimentConfig instance
    """
    if isinstance(config_path, (str, Path)):
        return ExperimentConfig.load(config_path)
    elif isinstance(config_path, dict):
        return ExperimentConfig.from_dict(config_path)
    elif isinstance(config_path, DictConfig):
        return ExperimentConfig.from_dict(OmegaConf.to_container(config_path))
    else:
        raise ValueError(f"Unsupported config type: {type(config_path)}")


def get_default_config(preset: str = "nano") -> ExperimentConfig:
    """
    Get default configuration for common experimental setups.
    
    Args:
        preset: Configuration preset ('nano', 'mobile', 'research', 'benchmark')
        
    Returns:
        ExperimentConfig with preset values
    """
    if preset == "nano":
        return ExperimentConfig(
            name=f"nano_nas_{preset}",
            search=SearchConfig(
                strategy="evolutionary",
                search_space="nano",
                population_size=20,
                generations=10
            ),
            dataset=DatasetConfig(name="cifar10", batch_size=128),
            training=TrainingConfig(epochs=50, learning_rate=0.025),
            model=ModelConfig(base_channels=16, max_depth=6)
        )
    
    elif preset == "mobile":
        return ExperimentConfig(
            name=f"nano_nas_{preset}",
            search=SearchConfig(
                strategy="darts",
                search_space="mobile",
                darts_epochs=50
            ),
            dataset=DatasetConfig(name="cifar10", batch_size=96),
            training=TrainingConfig(epochs=100, learning_rate=0.025),
            model=ModelConfig(base_channels=32, max_depth=12)
        )
    
    elif preset == "research":
        return ExperimentConfig(
            name=f"nano_nas_{preset}",
            search=SearchConfig(
                strategy="multiobjective",
                search_space="mobile",
                population_size=50,
                generations=25
            ),
            dataset=DatasetConfig(name="cifar10", batch_size=128),
            training=TrainingConfig(epochs=200, learning_rate=0.025),
            model=ModelConfig(base_channels=48, max_depth=15),
            use_wandb=True
        )
    
    elif preset == "benchmark":
        return ExperimentConfig(
            name=f"nano_nas_{preset}",
            search=SearchConfig(
                strategy="evolutionary",
                search_space="nano",
                population_size=100,
                generations=50
            ),
            dataset=DatasetConfig(name="cifar10", batch_size=256),
            training=TrainingConfig(epochs=300, learning_rate=0.1),
            model=ModelConfig(base_channels=64, max_depth=20)
        )
    
    else:
        raise ValueError(f"Unknown preset: {preset}")


def create_experiment_configs() -> Dict[str, ExperimentConfig]:
    """Create a suite of experiment configurations for comprehensive evaluation."""
    configs = {}
    
    # Ablation study configs
    strategies = ["evolutionary", "darts", "random"]
    for strategy in strategies:
        config = get_default_config("nano")
        config.name = f"ablation_{strategy}"
        config.search.strategy = strategy
        configs[f"ablation_{strategy}"] = config
    
    # Search space comparison
    search_spaces = ["nano", "mobile"]
    for space in search_spaces:
        config = get_default_config("nano")
        config.name = f"searchspace_{space}"
        config.search.search_space = space
        configs[f"searchspace_{space}"] = config
    
    # Dataset generalization
    datasets = ["cifar10", "cifar100", "mnist", "fashion_mnist"]
    for dataset in datasets:
        config = get_default_config("nano")
        config.name = f"dataset_{dataset}"
        config.dataset.name = dataset
        if dataset in ["cifar100"]:
            config.model.num_classes = 100
        elif dataset in ["mnist", "fashion_mnist"]:
            config.model.input_channels = 1
            config.model.num_classes = 10
        configs[f"dataset_{dataset}"] = config
    
    return configs


# Pre-defined configuration templates
CONFIG_TEMPLATES = {
    "nano": get_default_config("nano"),
    "mobile": get_default_config("mobile"), 
    "research": get_default_config("research"),
    "benchmark": get_default_config("benchmark"),
} 