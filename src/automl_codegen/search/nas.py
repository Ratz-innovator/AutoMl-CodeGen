"""
Neural Architecture Search (NAS) Engine

This module implements the main NAS interface that coordinates architecture search,
evaluation, and optimization across multiple objectives and hardware constraints.
"""

import logging
import time
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable, Any
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

from ..utils.config import Config
from ..utils.logger import Logger
from ..utils.profiler import PerformanceProfiler
from ..utils.monitoring import TrainingMonitor
from .algorithms.evolutionary import EvolutionarySearch
from .algorithms.darts import DARTSSearch
from .algorithms.reinforcement import ReinforcementSearch
# from .algorithms.hybrid import HybridSearch  # TODO: Implement hybrid algorithm
from .space.search_space import SearchSpace
from .objectives.multi_objective import MultiObjectiveOptimizer
from ..evaluation.trainer import ArchitectureTrainer
from ..evaluation.hardware import HardwareProfiler

logger = logging.getLogger(__name__)

@dataclass
class SearchResults:
    """Container for architecture search results."""
    best_architecture: Dict[str, Any]
    pareto_front: List[Dict[str, Any]]
    search_history: List[Dict[str, Any]]
    final_metrics: Dict[str, float]
    search_time: float
    total_evaluations: int
    convergence_generation: Optional[int] = None
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save search results to file."""
        filepath = Path(filepath)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'SearchResults':
        """Load search results from file."""
        filepath = Path(filepath)
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

class NeuralArchitectureSearch:
    """
    Main Neural Architecture Search engine.
    
    This class orchestrates the entire NAS process including:
    - Search space definition and constraints
    - Multi-objective optimization
    - Architecture evaluation and training
    - Hardware-aware optimization
    - Result analysis and visualization
    
    Example:
        >>> nas = NeuralArchitectureSearch(
        ...     task='image_classification',
        ...     dataset='cifar10',
        ...     objectives=['accuracy', 'latency', 'memory']
        ... )
        >>> results = nas.search(max_epochs=100, population_size=50)
        >>> print(f"Best accuracy: {results.best_architecture['accuracy']:.3f}")
    """
    
    def __init__(
        self,
        task: str,
        dataset: str,
        objectives: List[str],
        search_space: Optional[Dict[str, Any]] = None,
        hardware_target: str = 'gpu',
        algorithm: str = 'evolutionary',
        config: Optional[Config] = None,
        seed: Optional[int] = None,
        save_dir: Union[str, Path] = './nas_results',
        **kwargs
    ):
        """
        Initialize Neural Architecture Search.
        
        Args:
            task: Task type ('image_classification', 'object_detection', etc.)
            dataset: Dataset name or path
            objectives: List of optimization objectives
            search_space: Custom search space definition
            hardware_target: Target hardware ('gpu', 'cpu', 'mobile', 'edge')
            algorithm: Search algorithm ('evolutionary', 'darts', 'rl', 'hybrid')
            config: Configuration object
            seed: Random seed for reproducibility
            save_dir: Directory to save results and checkpoints
            **kwargs: Additional configuration parameters
        """
        self.task = task
        self.dataset = dataset
        self.objectives = objectives
        self.hardware_target = hardware_target
        self.algorithm = algorithm
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed for reproducibility
        if seed is not None:
            self._set_seed(seed)
            
        # Initialize configuration
        self.config = config or Config()
        self.config.update(kwargs)
        
        # Initialize components
        self.logger = Logger(save_dir=self.save_dir / 'logs')
        self.profiler = PerformanceProfiler()
        self.monitor = TrainingMonitor(log_dir=self.save_dir / 'monitoring')
        
        # Initialize search space
        self.search_space = SearchSpace(
            task=task,
            custom_space=search_space,
            hardware_target=hardware_target
        )
        
        # Initialize multi-objective optimizer
        self.multi_objective_optimizer = MultiObjectiveOptimizer(
            objectives=objectives,
            hardware_target=hardware_target
        )
        
        # Initialize search algorithm
        self.search_algorithm = self._create_search_algorithm()
        
        # Initialize architecture trainer
        self.trainer = ArchitectureTrainer(
            task=task,
            dataset=dataset,
            config=self.config,
            logger=self.logger
        )
        
        # Initialize hardware profiler
        device_map = {'gpu': 'cuda', 'cpu': 'cpu'}
        device = device_map.get(hardware_target, 'cuda')
        self.hardware_profiler = HardwareProfiler(device=device)
        
        # Search state
        self.population = []
        self.generation = 0
        self.total_evaluations = 0
        self.search_history = []
        self.pareto_front = []
        self.best_architecture = None
        self.start_time = None
        
        # Callbacks and hooks
        self.callbacks = []
        self.evaluation_hooks = []
        
        logger.info(f"Initialized NAS for {task} on {dataset}")
        logger.info(f"Objectives: {objectives}")
        logger.info(f"Algorithm: {algorithm}")
        logger.info(f"Hardware target: {hardware_target}")
    
    def _set_seed(self, seed: int) -> None:
        """Set random seed for all libraries."""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def _create_search_algorithm(self):
        """Create the specified search algorithm."""
        algorithm_map = {
            'evolutionary': EvolutionarySearch,
            'darts': DARTSSearch,
            'reinforcement': ReinforcementSearch,
            'rl': ReinforcementSearch,  # Alias for reinforcement
            # 'hybrid': HybridSearch  # TODO: Implement hybrid algorithm
        }
        
        if self.algorithm not in algorithm_map:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        algorithm_class = algorithm_map[self.algorithm]
        return algorithm_class(
            search_space=self.search_space,
            objectives=self.multi_objective_optimizer,
            config=self.config,
            logger=self.logger
        )
    
    def add_callback(self, callback: Callable) -> None:
        """Add a callback function called after each generation."""
        self.callbacks.append(callback)
    
    def add_evaluation_hook(self, hook: Callable) -> None:
        """Add a hook function called during architecture evaluation."""
        self.evaluation_hooks.append(hook)
    
    def search(
        self,
        max_epochs: int = 100,
        population_size: int = 50,
        num_generations: int = 20,
        early_stopping: bool = True,
        patience: int = 5,
        save_frequency: int = 5,
        resume_from: Optional[str] = None,
        distributed: bool = False,
        num_workers: Optional[int] = None,
        **kwargs
    ) -> SearchResults:
        """
        Execute neural architecture search.
        
        Args:
            max_epochs: Maximum training epochs per architecture
            population_size: Number of architectures in population
            num_generations: Number of evolution generations
            early_stopping: Whether to use early stopping
            patience: Patience for early stopping
            save_frequency: Frequency to save checkpoints
            resume_from: Path to checkpoint to resume from
            distributed: Whether to use distributed evaluation
            num_workers: Number of parallel workers
            **kwargs: Additional search parameters
            
        Returns:
            SearchResults containing the best architectures and metrics
        """
        self.start_time = time.time()
        
        # Resume from checkpoint if specified
        if resume_from:
            self._load_checkpoint(resume_from)
        
        # Initialize population if starting fresh
        if not self.population:
            self.population = self.search_algorithm.initialize_population(
                population_size
            )
        
        # Setup distributed evaluation if requested
        if distributed:
            self._setup_distributed_evaluation(num_workers)
        
        logger.info(f"Starting architecture search with {len(self.population)} individuals")
        logger.info(f"Target generations: {num_generations}")
        
        # Main search loop
        best_score = -np.inf
        patience_counter = 0
        
        try:
            for generation in range(self.generation, num_generations):
                self.generation = generation
                generation_start = time.time()
                
                logger.info(f"Generation {generation + 1}/{num_generations}")
                
                # Evaluate population
                with self.profiler.profile("population_evaluation"):
                    population_metrics = self._evaluate_population(
                        max_epochs=max_epochs,
                        distributed=distributed,
                        **kwargs
                    )
                
                # Update multi-objective optimization
                self.multi_objective_optimizer.update(
                    population=self.population,
                    metrics=population_metrics
                )
                
                # Get Pareto front
                self.pareto_front = self.multi_objective_optimizer.get_pareto_front()
                
                # Update best architecture
                current_best = self._select_best_architecture(self.pareto_front)
                if current_best:
                    current_score = self._calculate_aggregate_score(current_best)
                    if current_score > best_score:
                        best_score = current_score
                        self.best_architecture = current_best
                        patience_counter = 0
                    else:
                        patience_counter += 1
                
                # Record generation results
                generation_time = time.time() - generation_start
                generation_stats = {
                    'generation': generation,
                    'population_size': len(self.population),
                    'pareto_front_size': len(self.pareto_front),
                    'best_score': best_score,
                    'generation_time': generation_time,
                    'total_evaluations': self.total_evaluations
                }
                self.search_history.append(generation_stats)
                
                # Log generation results
                self._log_generation_results(generation_stats)
                
                # Execute callbacks
                for callback in self.callbacks:
                    callback(self, generation_stats)
                
                # Check early stopping
                if early_stopping and patience_counter >= patience:
                    logger.info(f"Early stopping at generation {generation}")
                    break
                
                # Save checkpoint
                if (generation + 1) % save_frequency == 0:
                    self._save_checkpoint(generation)
                
                # Evolve population for next generation
                if generation < num_generations - 1:
                    with self.profiler.profile("population_evolution"):
                        self.population = self.search_algorithm.evolve_population(
                            population=self.population,
                            metrics=population_metrics,
                            pareto_front=self.pareto_front
                        )
                
        except KeyboardInterrupt:
            logger.info("Search interrupted by user")
        except Exception as e:
            logger.error(f"Search failed with error: {e}")
            raise
        
        # Final evaluation and results
        search_time = time.time() - self.start_time
        
        # Ensure we have a best architecture
        if self.best_architecture is None and self.pareto_front:
            self.best_architecture = self._select_best_architecture(self.pareto_front)
        
        # Create search results
        results = SearchResults(
            best_architecture=self.best_architecture,
            pareto_front=self.pareto_front,
            search_history=self.search_history,
            final_metrics=self._calculate_final_metrics(),
            search_time=search_time,
            total_evaluations=self.total_evaluations,
            convergence_generation=generation if early_stopping else None
        )
        
        # Save final results
        results.save(self.save_dir / 'final_results.pkl')
        self._save_results_json(results)
        
        logger.info(f"Search completed in {search_time:.2f}s")
        logger.info(f"Total evaluations: {self.total_evaluations}")
        logger.info(f"Best architecture score: {best_score:.4f}")
        
        return results
    
    def _evaluate_population(
        self,
        max_epochs: int,
        distributed: bool = False,
        **kwargs
    ) -> List[Dict[str, float]]:
        """Evaluate all architectures in the current population."""
        if distributed:
            return self._evaluate_population_distributed(max_epochs, **kwargs)
        else:
            return self._evaluate_population_sequential(max_epochs, **kwargs)
    
    def _evaluate_population_sequential(
        self,
        max_epochs: int,
        **kwargs
    ) -> List[Dict[str, float]]:
        """Evaluate population sequentially."""
        metrics_list = []
        
        for i, architecture in enumerate(self.population):
            logger.info(f"Evaluating architecture {i+1}/{len(self.population)}")
            
            metrics = self._evaluate_architecture(
                architecture,
                max_epochs=max_epochs,
                **kwargs
            )
            metrics_list.append(metrics)
            self.total_evaluations += 1
            
            # Execute evaluation hooks
            for hook in self.evaluation_hooks:
                hook(architecture, metrics)
        
        return metrics_list
    
    def _evaluate_population_distributed(
        self,
        max_epochs: int,
        **kwargs
    ) -> List[Dict[str, float]]:
        """Evaluate population using distributed processing."""
        # This would implement distributed evaluation
        # For now, fall back to sequential
        logger.warning("Distributed evaluation not yet implemented, using sequential")
        return self._evaluate_population_sequential(max_epochs, **kwargs)
    
    def _evaluate_architecture(
        self,
        architecture: Dict[str, Any],
        max_epochs: int,
        **kwargs
    ) -> Dict[str, float]:
        """Evaluate a single architecture."""
        try:
            # Train the architecture
            training_metrics = self.trainer.train_architecture(
                architecture,
                dataset=self.dataset,
                max_epochs=max_epochs,
                **kwargs
            )
            
            # Profile hardware performance
            hardware_metrics = self.hardware_profiler.profile_architecture(
                architecture
            )
            
            # Combine all metrics
            metrics = {**training_metrics, **hardware_metrics}
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to evaluate architecture: {e}")
            # Return worst possible metrics
            return {obj: 0.0 if 'accuracy' in obj else float('inf') 
                   for obj in self.objectives}
    
    def _select_best_architecture(
        self,
        pareto_front: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Select the best architecture from Pareto front based on preferences."""
        if not pareto_front:
            return None
        
        # Simple strategy: select based on weighted sum
        # This could be made more sophisticated with user preferences
        weights = {
            'accuracy': 0.5,
            'latency': 0.3,
            'memory': 0.1,
            'energy': 0.1
        }
        
        best_arch = None
        best_score = -np.inf
        
        for arch in pareto_front:
            score = self._calculate_weighted_score(arch, weights)
            if score > best_score:
                best_score = score
                best_arch = arch
        
        return best_arch
    
    def _calculate_aggregate_score(self, architecture: Dict[str, Any]) -> float:
        """Calculate an aggregate score for an architecture."""
        weights = {'accuracy': 0.5, 'latency': -0.3, 'memory': -0.1, 'energy': -0.1}
        return self._calculate_weighted_score(architecture, weights)
    
    def _calculate_weighted_score(
        self,
        architecture: Dict[str, Any],
        weights: Dict[str, float]
    ) -> float:
        """Calculate weighted score for architecture."""
        score = 0.0
        for metric, weight in weights.items():
            if metric in architecture:
                value = architecture[metric]
                # Normalize metrics (this is simplified)
                if 'accuracy' in metric:
                    normalized = value  # Already 0-1
                else:
                    normalized = 1.0 / (1.0 + value)  # Invert for latency/memory/energy
                score += weight * normalized
        return score
    
    def _calculate_final_metrics(self) -> Dict[str, float]:
        """Calculate final summary metrics."""
        if not self.search_history:
            return {}
        
        final_gen = self.search_history[-1]
        return {
            'final_generation': final_gen['generation'],
            'total_time': sum(h['generation_time'] for h in self.search_history),
            'avg_generation_time': np.mean([h['generation_time'] for h in self.search_history]),
            'convergence_rate': final_gen['best_score'] / max(1, final_gen['generation']),
            'pareto_front_diversity': len(self.pareto_front),
            'search_efficiency': final_gen['best_score'] / self.total_evaluations
        }
    
    def _log_generation_results(self, stats: Dict[str, Any]) -> None:
        """Log results for a generation."""
        logger.info(f"Generation {stats['generation']} completed:")
        logger.info(f"  Population size: {stats['population_size']}")
        logger.info(f"  Pareto front size: {stats['pareto_front_size']}")
        logger.info(f"  Best score: {stats['best_score']:.4f}")
        logger.info(f"  Generation time: {stats['generation_time']:.2f}s")
        logger.info(f"  Total evaluations: {stats['total_evaluations']}")
    
    def _save_checkpoint(self, generation: int) -> None:
        """Save search checkpoint."""
        checkpoint = {
            'generation': generation,
            'population': self.population,
            'pareto_front': self.pareto_front,
            'best_architecture': self.best_architecture,
            'search_history': self.search_history,
            'total_evaluations': self.total_evaluations,
            'config': self.config.to_dict(),
            'algorithm_state': self.search_algorithm.get_state()
        }
        
        checkpoint_path = self.save_dir / f'checkpoint_gen_{generation}.pkl'
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        logger.info(f"Saved checkpoint at generation {generation}")
    
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load search checkpoint."""
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.generation = checkpoint['generation']
        self.population = checkpoint['population']
        self.pareto_front = checkpoint['pareto_front']
        self.best_architecture = checkpoint['best_architecture']
        self.search_history = checkpoint['search_history']
        self.total_evaluations = checkpoint['total_evaluations']
        
        if 'algorithm_state' in checkpoint:
            self.search_algorithm.load_state(checkpoint['algorithm_state'])
        
        logger.info(f"Loaded checkpoint from generation {self.generation}")
    
    def _save_results_json(self, results: SearchResults) -> None:
        """Save results in JSON format for easy analysis."""
        results_dict = results.to_dict()
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Recursively convert numpy types
        def deep_convert(obj):
            if isinstance(obj, dict):
                return {k: deep_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deep_convert(v) for v in obj]
            else:
                return convert_numpy(obj)
        
        results_dict = deep_convert(results_dict)
        
        with open(self.save_dir / 'results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)
    
    def _setup_distributed_evaluation(self, num_workers: Optional[int]) -> None:
        """Setup distributed evaluation infrastructure."""
        if num_workers is None:
            num_workers = min(mp.cpu_count(), len(self.population))
        
        logger.info(f"Setting up distributed evaluation with {num_workers} workers")
        # Implementation would go here
    
    def get_search_summary(self) -> Dict[str, Any]:
        """Get a summary of the search process."""
        return {
            'task': self.task,
            'dataset': self.dataset,
            'objectives': self.objectives,
            'algorithm': self.algorithm,
            'hardware_target': self.hardware_target,
            'generations_completed': self.generation,
            'total_evaluations': self.total_evaluations,
            'pareto_front_size': len(self.pareto_front),
            'best_architecture': self.best_architecture,
            'search_time': time.time() - self.start_time if self.start_time else 0
        }
    
    def visualize_search(self, save_path: Optional[str] = None) -> None:
        """Create visualizations of the search process."""
        from ..utils.visualization import SearchVisualizer
        
        visualizer = SearchVisualizer(self.search_history, self.pareto_front)
        visualizer.plot_convergence(save_path)
        visualizer.plot_pareto_front(save_path)
        visualizer.plot_search_progress(save_path) 