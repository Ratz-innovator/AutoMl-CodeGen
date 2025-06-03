"""
Multi-Objective Neural Architecture Search
==========================================

This module implements advanced multi-objective optimization algorithms for neural
architecture search, including NSGA-III for handling 4+ objectives.

Key Features:
- NSGA-III algorithm for many-objective optimization (4+ objectives)
- Dynamic objective weighting based on user constraints
- Uncertainty quantification using Gaussian processes
- Adaptive constraint handling for hardware limitations
- Pareto frontier analysis and visualization
- Reference point adaptation for better convergence

Objectives supported:
- Accuracy (maximize)
- FLOPs (minimize)
- Energy consumption (minimize)
- Memory bandwidth (minimize)
- Inference latency (minimize)
- Model size (minimize)
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

from ..core.base import BaseSearchStrategy
from ..core.architecture import Architecture, SearchSpace
from ..core.config import ExperimentConfig
from ..benchmarks.evaluator import ModelEvaluator


@dataclass
class MultiObjectiveConfig:
    """Configuration for multi-objective optimization."""
    population_size: int = 100
    generations: int = 50
    objectives: List[str] = field(default_factory=lambda: ["accuracy", "flops", "energy", "latency"])
    reference_points: Optional[np.ndarray] = None
    constraint_weights: Dict[str, float] = field(default_factory=dict)
    uncertainty_threshold: float = 0.1
    adaptive_weights: bool = True
    use_surrogate_model: bool = True
    pareto_archive_size: int = 200
    
    def __post_init__(self):
        """Validate configuration."""
        if len(self.objectives) < 2:
            raise ValueError("At least 2 objectives required for multi-objective optimization")
        
        # Set default constraint weights
        default_weights = {
            "accuracy": 1.0,
            "flops": 0.3,
            "energy": 0.2,
            "latency": 0.25,
            "memory": 0.15,
            "params": 0.1
        }
        for obj in self.objectives:
            if obj not in self.constraint_weights:
                self.constraint_weights[obj] = default_weights.get(obj, 1.0)


class NSGAIIIOptimizer:
    """
    NSGA-III optimizer for many-objective optimization.
    
    Implements the NSGA-III algorithm which is specifically designed
    for handling 4 or more objectives effectively.
    """
    
    def __init__(self, config: MultiObjectiveConfig):
        """Initialize NSGA-III optimizer."""
        self.config = config
        self.num_objectives = len(config.objectives)
        
        # Generate reference points
        if config.reference_points is None:
            self.reference_points = self._generate_reference_points()
        else:
            self.reference_points = config.reference_points
        
        # Initialize tracking variables
        self.generation = 0
        self.pareto_archive = []
        self.hypervolume_history = []
        self.convergence_metrics = []
        
    def _generate_reference_points(self) -> np.ndarray:
        """Generate uniformly distributed reference points."""
        # Use Das and Dennis method for uniform reference point generation
        num_points = max(50, self.config.population_size // 2)
        
        if self.num_objectives <= 3:
            # For 2-3 objectives, use simple uniform distribution
            points = []
            for i in range(num_points):
                point = np.random.dirichlet(np.ones(self.num_objectives))
                points.append(point)
            return np.array(points)
        else:
            # For 4+ objectives, use structured approach
            return self._das_dennis_reference_points(num_points)
    
    def _das_dennis_reference_points(self, num_points: int) -> np.ndarray:
        """Generate reference points using Das and Dennis method."""
        # Simplified implementation for demonstration
        # In practice, you'd use a more sophisticated implementation
        points = []
        
        # Generate points on unit simplex
        for _ in range(num_points):
            # Generate random point and normalize
            point = np.random.exponential(1, self.num_objectives)
            point = point / np.sum(point)
            points.append(point)
        
        return np.array(points)
    
    def optimize(self, population: List[Dict[str, Any]], 
                 objective_values: np.ndarray) -> List[int]:
        """
        Perform NSGA-III selection.
        
        Args:
            population: List of individuals with their data
            objective_values: Array of shape (pop_size, num_objectives)
            
        Returns:
            Indices of selected individuals
        """
        # Step 1: Non-dominated sorting
        fronts = self._non_dominated_sort(objective_values)
        
        # Step 2: Select individuals for next generation
        selected_indices = []
        front_idx = 0
        
        # Fill population with complete fronts
        while (len(selected_indices) + len(fronts[front_idx]) <= 
               self.config.population_size and front_idx < len(fronts)):
            selected_indices.extend(fronts[front_idx])
            front_idx += 1
        
        # If we need to select from the last front
        if len(selected_indices) < self.config.population_size and front_idx < len(fronts):
            last_front = fronts[front_idx]
            remaining_slots = self.config.population_size - len(selected_indices)
            
            # Use reference point based selection for the last front
            last_front_objectives = objective_values[last_front]
            selected_from_last = self._reference_point_selection(
                last_front_objectives, remaining_slots
            )
            
            selected_indices.extend([last_front[i] for i in selected_from_last])
        
        return selected_indices[:self.config.population_size]
    
    def _non_dominated_sort(self, objectives: np.ndarray) -> List[List[int]]:
        """Perform non-dominated sorting."""
        pop_size = len(objectives)
        domination_count = np.zeros(pop_size, dtype=int)
        dominated_solutions = [[] for _ in range(pop_size)]
        fronts = [[]]
        
        # Calculate domination relationships
        for i in range(pop_size):
            for j in range(pop_size):
                if i != j:
                    if self._dominates(objectives[i], objectives[j]):
                        dominated_solutions[i].append(j)
                    elif self._dominates(objectives[j], objectives[i]):
                        domination_count[i] += 1
            
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        # Build subsequent fronts
        front_idx = 0
        while len(fronts[front_idx]) > 0:
            next_front = []
            for i in fronts[front_idx]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            
            if len(next_front) > 0:
                fronts.append(next_front)
            front_idx += 1
        
        return fronts[:-1]  # Remove empty last front
    
    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """Check if obj1 dominates obj2 (assuming minimization for all objectives except accuracy)."""
        # Accuracy should be maximized, others minimized
        obj1_mod = obj1.copy()
        obj2_mod = obj2.copy()
        
        # Flip accuracy objective (assuming it's the first one)
        if "accuracy" in self.config.objectives:
            acc_idx = self.config.objectives.index("accuracy")
            obj1_mod[acc_idx] = -obj1_mod[acc_idx]
            obj2_mod[acc_idx] = -obj2_mod[acc_idx]
        
        return np.all(obj1_mod <= obj2_mod) and np.any(obj1_mod < obj2_mod)
    
    def _reference_point_selection(self, objectives: np.ndarray, 
                                 num_select: int) -> List[int]:
        """Select individuals based on reference points."""
        # Normalize objectives
        normalized_obj = self._normalize_objectives(objectives)
        
        # Calculate distances to reference points
        distances = cdist(normalized_obj, self.reference_points)
        
        # Associate each individual with closest reference point
        closest_ref_points = np.argmin(distances, axis=1)
        
        # Count associations for each reference point
        ref_point_counts = np.bincount(closest_ref_points, 
                                     minlength=len(self.reference_points))
        
        # Select individuals to maintain diversity
        selected = []
        remaining_individuals = list(range(len(objectives)))
        
        for _ in range(num_select):
            if not remaining_individuals:
                break
            
            # Find reference point with minimum associations
            min_count = np.min(ref_point_counts)
            candidate_ref_points = np.where(ref_point_counts == min_count)[0]
            
            # Randomly select one reference point
            selected_ref_point = np.random.choice(candidate_ref_points)
            
            # Find individuals associated with this reference point
            candidates = [i for i in remaining_individuals 
                         if closest_ref_points[i] == selected_ref_point]
            
            if candidates:
                # Select individual with minimum distance to reference point
                candidate_distances = [distances[i, selected_ref_point] for i in candidates]
                best_candidate = candidates[np.argmin(candidate_distances)]
                
                selected.append(best_candidate)
                remaining_individuals.remove(best_candidate)
                ref_point_counts[selected_ref_point] += 1
            else:
                # If no candidates, select randomly
                if remaining_individuals:
                    selected.append(remaining_individuals.pop(0))
        
        return selected
    
    def _normalize_objectives(self, objectives: np.ndarray) -> np.ndarray:
        """Normalize objectives to [0, 1] range."""
        min_vals = np.min(objectives, axis=0)
        max_vals = np.max(objectives, axis=0)
        
        # Avoid division by zero
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1.0
        
        return (objectives - min_vals) / ranges
    
    def update_reference_points(self, population_objectives: np.ndarray):
        """Adaptively update reference points based on population distribution."""
        if self.config.adaptive_weights and self.generation % 10 == 0:
            # Analyze current population distribution
            normalized_obj = self._normalize_objectives(population_objectives)
            
            # Update reference points to better cover sparse regions
            self._adapt_reference_points(normalized_obj)
    
    def _adapt_reference_points(self, normalized_objectives: np.ndarray):
        """Adapt reference points to improve coverage."""
        # Simple adaptation: add points in sparse regions
        distances = cdist(normalized_objectives, self.reference_points)
        min_distances = np.min(distances, axis=0)
        
        # Find reference points with large minimum distances (sparse regions)
        sparse_threshold = np.percentile(min_distances, 75)
        sparse_regions = np.where(min_distances > sparse_threshold)[0]
        
        # Add new reference points in sparse regions
        if len(sparse_regions) > 0:
            new_points = []
            for region_idx in sparse_regions[:5]:  # Limit to 5 new points
                # Generate point near sparse reference point with some noise
                base_point = self.reference_points[region_idx]
                noise = np.random.normal(0, 0.1, self.num_objectives)
                new_point = np.clip(base_point + noise, 0, 1)
                new_point = new_point / np.sum(new_point)  # Normalize to simplex
                new_points.append(new_point)
            
            if new_points:
                self.reference_points = np.vstack([self.reference_points, new_points])


class GaussianProcessSurrogate:
    """Gaussian Process surrogate model for uncertainty quantification."""
    
    def __init__(self, objectives: List[str]):
        """Initialize GP surrogate models for each objective."""
        self.objectives = objectives
        self.models = {}
        self.training_data = {'X': [], 'y': {obj: [] for obj in objectives}}
        
        # Initialize GP models
        for obj in objectives:
            kernel = ConstantKernel(1.0) * RBF(1.0)
            self.models[obj] = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5
            )
    
    def update(self, architecture_features: np.ndarray, objective_values: Dict[str, float]):
        """Update surrogate models with new data."""
        self.training_data['X'].append(architecture_features)
        for obj in self.objectives:
            self.training_data['y'][obj].append(objective_values.get(obj, 0.0))
        
        # Retrain models if we have enough data
        if len(self.training_data['X']) >= 10:
            self._retrain_models()
    
    def _retrain_models(self):
        """Retrain all GP models."""
        X = np.array(self.training_data['X'])
        
        for obj in self.objectives:
            y = np.array(self.training_data['y'][obj])
            if len(np.unique(y)) > 1:  # Only train if we have variance
                self.models[obj].fit(X, y)
    
    def predict(self, architecture_features: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Predict objective values and uncertainties.
        
        Returns:
            Tuple of (predictions, uncertainties)
        """
        predictions = {}
        uncertainties = {}
        
        for obj in self.objectives:
            if hasattr(self.models[obj], 'X_train_'):  # Model is trained
                mean, std = self.models[obj].predict(
                    architecture_features.reshape(1, -1), 
                    return_std=True
                )
                predictions[obj] = mean[0]
                uncertainties[obj] = std[0]
            else:
                # Default values if model not trained
                predictions[obj] = 0.5
                uncertainties[obj] = 1.0
        
        return predictions, uncertainties
    
    def acquisition_function(self, architecture_features: np.ndarray, 
                           current_best: Dict[str, float]) -> float:
        """Expected Improvement acquisition function."""
        predictions, uncertainties = self.predict(architecture_features)
        
        # Multi-objective Expected Improvement
        ei_values = []
        for obj in self.objectives:
            if obj == "accuracy":
                # For accuracy, we want to maximize
                improvement = predictions[obj] - current_best.get(obj, 0.0)
            else:
                # For other objectives, we want to minimize
                improvement = current_best.get(obj, float('inf')) - predictions[obj]
            
            if uncertainties[obj] > 0:
                z = improvement / uncertainties[obj]
                ei = improvement * self._normal_cdf(z) + uncertainties[obj] * self._normal_pdf(z)
            else:
                ei = 0.0
            
            ei_values.append(max(0.0, ei))
        
        # Combine EI values (could use different strategies)
        return np.mean(ei_values)
    
    def _normal_cdf(self, x: float) -> float:
        """Standard normal CDF approximation."""
        return 0.5 * (1 + np.tanh(x * np.sqrt(2 / np.pi)))
    
    def _normal_pdf(self, x: float) -> float:
        """Standard normal PDF."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


class MultiObjectiveSearch(BaseSearchStrategy):
    """
    Multi-objective neural architecture search using NSGA-III.
    
    This implementation supports 4+ objectives and includes:
    - NSGA-III algorithm for many-objective optimization
    - Gaussian Process surrogate models
    - Adaptive constraint handling
    - Uncertainty quantification
    - Dynamic objective weighting
    """
    
    def __init__(self, config: ExperimentConfig):
        """Initialize multi-objective search."""
        super().__init__(config)
        
        # Create multi-objective specific config
        self.mo_config = MultiObjectiveConfig(
            population_size=config.search.population_size,
            generations=config.search.generations,
            objectives=config.search.objectives
        )
        
        # Get search space
        if config.search.search_space == "nano":
            self.search_space = SearchSpace.get_nano_search_space()
        elif config.search.search_space == "mobile":
            self.search_space = SearchSpace.get_mobile_search_space()
        elif config.search.search_space == "advanced":
            self.search_space = SearchSpace.get_advanced_search_space()
        else:
            self.search_space = SearchSpace.get_nano_search_space()
        
        # Initialize components
        self.evaluator = ModelEvaluator(config)
        self.optimizer = NSGAIIIOptimizer(self.mo_config)
        self.surrogate = GaussianProcessSurrogate(self.mo_config.objectives)
        self.logger = logging.getLogger(__name__)
        
        # Search state
        self.population = []
        self.objective_history = []
        self.pareto_front = []
        self.generation = 0
        
        # Performance tracking
        self.evaluation_cache = {}
        self.search_metrics = {
            'generations_run': 0,
            'total_evaluations': 0,
            'pareto_front_size': [],
            'hypervolume': [],
            'convergence_metric': [],
            'uncertainty_reduction': [],
        }
    
    def search(self) -> Architecture:
        """
        Run multi-objective search to find Pareto-optimal architectures.
        
        Returns:
            Best architecture according to weighted objectives
        """
        self.logger.info(f"🎯 Starting multi-objective search with {len(self.mo_config.objectives)} objectives")
        self.logger.info(f"📊 Objectives: {', '.join(self.mo_config.objectives)}")
        self.logger.info(f"🧬 Population size: {self.mo_config.population_size}")
        
        start_time = time.time()
        
        try:
            # Initialize population
            self._initialize_population()
            
            # Evolution loop
            for generation in range(self.mo_config.generations):
                self.generation = generation
                self.logger.info(f"🧬 Generation {generation + 1}/{self.mo_config.generations}")
                
                # Evaluate population
                self._evaluate_population()
                
                # Update surrogate models
                if self.mo_config.use_surrogate_model:
                    self._update_surrogate_models()
                
                # Update Pareto front
                self._update_pareto_front()
                
                # Update statistics
                self._update_statistics()
                
                # Check for convergence
                if self._check_convergence():
                    self.logger.info(f"🎯 Converged at generation {generation + 1}")
                    break
                
                # Create next generation
                if generation < self.mo_config.generations - 1:
                    self._create_next_generation()
                
                # Adaptive reference point update
                self._update_reference_points()
                
                # Log progress
                self._log_generation_progress()
            
            search_time = time.time() - start_time
            self.search_metrics['total_search_time'] = search_time
            
            # Select best architecture from Pareto front
            best_architecture = self._select_best_architecture()
            
            self.logger.info(f"✅ Multi-objective search completed in {search_time:.2f}s")
            self.logger.info(f"🎯 Pareto front size: {len(self.pareto_front)}")
            self.logger.info(f"📊 Total evaluations: {self.search_metrics['total_evaluations']}")
            
            return best_architecture
            
        except Exception as e:
            self.logger.error(f"❌ Multi-objective search failed: {e}")
            raise
    
    def _initialize_population(self):
        """Initialize population with diverse architectures."""
        self.logger.info("🌱 Initializing population...")
        
        self.population = []
        for i in range(self.mo_config.population_size):
            # Create random architecture
            arch = self.search_space.sample_random_architecture()
            
            individual = {
                'architecture': arch,
                'objectives': None,
                'rank': None,
                'crowding_distance': 0.0,
                'uncertainty': {},
                'age': 0,
            }
            self.population.append(individual)
        
        self.logger.info(f"✅ Initialized {len(self.population)} individuals")
    
    def _evaluate_population(self):
        """Evaluate all individuals in the population."""
        self.logger.info("🔬 Evaluating population...")
        
        for individual in self.population:
            if individual['objectives'] is None:
                objectives = self._evaluate_architecture(individual['architecture'])
                individual['objectives'] = objectives
                individual['age'] += 1
    
    def _evaluate_architecture(self, architecture: Architecture) -> Dict[str, float]:
        """Evaluate a single architecture for all objectives."""
        # Check cache first
        arch_hash = hash(architecture)
        if arch_hash in self.evaluation_cache:
            return self.evaluation_cache[arch_hash]
        
        self.search_metrics['total_evaluations'] += 1
        
        try:
            # Convert to model
            model = architecture.to_model(
                input_channels=self.config.model.input_channels,
                num_classes=self.config.model.num_classes,
                base_channels=self.config.model.base_channels
            )
            
            # Evaluate model performance
            metrics = self.evaluator.quick_evaluate(model)
            
            # Get complexity metrics
            complexity = architecture.get_complexity_metrics()
            
            # Compute all objectives
            objectives = {}
            
            for obj_name in self.mo_config.objectives:
                if obj_name == "accuracy":
                    objectives[obj_name] = metrics.get('accuracy', 0.0)
                elif obj_name == "flops":
                    objectives[obj_name] = complexity.get('total_flops', 1e6)
                elif obj_name == "energy":
                    objectives[obj_name] = complexity.get('total_energy', 1.0)
                elif obj_name == "latency":
                    objectives[obj_name] = complexity.get('total_latency', 1.0)
                elif obj_name == "memory":
                    objectives[obj_name] = complexity.get('total_memory', 1.0)
                elif obj_name == "params":
                    objectives[obj_name] = complexity.get('total_op_cost', 1.0)
                else:
                    objectives[obj_name] = 0.0
            
            # Cache result
            self.evaluation_cache[arch_hash] = objectives
            
            return objectives
            
        except Exception as e:
            self.logger.warning(f"⚠️ Architecture evaluation failed: {e}")
            # Return poor objectives for invalid architectures
            return {obj: 0.0 if obj == "accuracy" else 1000.0 
                   for obj in self.mo_config.objectives}
    
    def _update_surrogate_models(self):
        """Update Gaussian Process surrogate models."""
        for individual in self.population:
            if individual['objectives'] is not None:
                # Extract architecture features
                features = self._extract_architecture_features(individual['architecture'])
                
                # Update surrogate model
                self.surrogate.update(features, individual['objectives'])
    
    def _extract_architecture_features(self, architecture: Architecture) -> np.ndarray:
        """Extract numerical features from architecture for surrogate model."""
        complexity = architecture.get_complexity_metrics()
        
        # Create feature vector
        features = []
        
        # Basic complexity features
        features.extend([
            complexity.get('depth', 0),
            complexity.get('total_op_cost', 0),
            complexity.get('skip_ratio', 0),
            complexity.get('attention_ratio', 0),
            complexity.get('conv_ratio', 0),
            complexity.get('norm_ratio', 0),
        ])
        
        # Architecture-specific features
        if architecture.encoding is not None:
            # For list encoding, use operation histogram
            op_hist = np.bincount(architecture.encoding, 
                                minlength=len(self.search_space.operations))
            features.extend(op_hist.tolist())
        elif architecture.graph is not None:
            # For graph encoding, use graph properties
            features.extend([
                complexity.get('num_nodes', 0),
                complexity.get('num_edges', 0),
                complexity.get('avg_degree', 0),
                complexity.get('max_path_length', 0),
            ])
        elif architecture.hierarchical_encoding is not None:
            # For hierarchical encoding, use cell statistics
            features.extend([
                complexity.get('num_cells', 0),
                complexity.get('avg_ops_per_cell', 0),
                complexity.get('micro_ratio', 0),
                complexity.get('macro_ratio', 0),
            ])
        
        return np.array(features)
    
    def _update_pareto_front(self):
        """Update the Pareto front with current population."""
        # Extract objective values
        objective_matrix = np.array([
            [ind['objectives'][obj] for obj in self.mo_config.objectives]
            for ind in self.population if ind['objectives'] is not None
        ])
        
        if len(objective_matrix) == 0:
            return
        
        # Find non-dominated solutions
        fronts = self.optimizer._non_dominated_sort(objective_matrix)
        
        # Update Pareto front
        if len(fronts) > 0:
            pareto_indices = fronts[0]
            self.pareto_front = [self.population[i] for i in pareto_indices]
    
    def _create_next_generation(self):
        """Create next generation using NSGA-III selection."""
        # Extract objective values
        objective_matrix = np.array([
            [ind['objectives'][obj] for obj in self.mo_config.objectives]
            for ind in self.population
        ])
        
        # Perform NSGA-III selection
        selected_indices = self.optimizer.optimize(self.population, objective_matrix)
        
        # Create offspring through crossover and mutation
        new_population = []
        
        # Keep selected individuals
        for idx in selected_indices:
            new_population.append(self.population[idx].copy())
        
        # Generate offspring to fill population
        while len(new_population) < self.mo_config.population_size:
            # Select parents
            parent1_idx = np.random.choice(selected_indices)
            parent2_idx = np.random.choice(selected_indices)
            
            parent1 = self.population[parent1_idx]['architecture']
            parent2 = self.population[parent2_idx]['architecture']
            
            # Crossover
            try:
                child1_arch, child2_arch = parent1.crossover(parent2)
                
                # Mutation
                if np.random.random() < 0.1:  # 10% mutation rate
                    child1_arch = child1_arch.mutate(0.1, 1.0)
                if np.random.random() < 0.1:
                    child2_arch = child2_arch.mutate(0.1, 1.0)
                
                # Add children to population
                for child_arch in [child1_arch, child2_arch]:
                    if len(new_population) < self.mo_config.population_size:
                        child = {
                            'architecture': child_arch,
                            'objectives': None,
                            'rank': None,
                            'crowding_distance': 0.0,
                            'uncertainty': {},
                            'age': 0,
                        }
                        new_population.append(child)
            except Exception as e:
                # If crossover fails, create random individual
                if len(new_population) < self.mo_config.population_size:
                    arch = self.search_space.sample_random_architecture()
                    individual = {
                        'architecture': arch,
                        'objectives': None,
                        'rank': None,
                        'crowding_distance': 0.0,
                        'uncertainty': {},
                        'age': 0,
                    }
                    new_population.append(individual)
        
        self.population = new_population[:self.mo_config.population_size]
    
    def _update_reference_points(self):
        """Update reference points adaptively."""
        if len(self.population) > 0:
            objective_matrix = np.array([
                [ind['objectives'][obj] for obj in self.mo_config.objectives]
                for ind in self.population if ind['objectives'] is not None
            ])
            
            if len(objective_matrix) > 0:
                self.optimizer.update_reference_points(objective_matrix)
    
    def _check_convergence(self) -> bool:
        """Check if the search has converged."""
        if len(self.search_metrics['hypervolume']) < 10:
            return False
        
        # Check hypervolume improvement
        recent_hv = self.search_metrics['hypervolume'][-10:]
        hv_improvement = max(recent_hv) - min(recent_hv)
        
        return hv_improvement < 0.001  # Convergence threshold
    
    def _update_statistics(self):
        """Update search statistics."""
        # Pareto front size
        self.search_metrics['pareto_front_size'].append(len(self.pareto_front))
        
        # Hypervolume calculation (simplified)
        if len(self.pareto_front) > 0:
            hv = self._calculate_hypervolume()
            self.search_metrics['hypervolume'].append(hv)
        
        # Convergence metric
        convergence = self._calculate_convergence_metric()
        self.search_metrics['convergence_metric'].append(convergence)
        
        # Update generation count
        self.search_metrics['generations_run'] = self.generation + 1
    
    def _calculate_hypervolume(self) -> float:
        """Calculate hypervolume indicator (simplified implementation)."""
        if len(self.pareto_front) == 0:
            return 0.0
        
        # Extract objective values from Pareto front
        objectives = np.array([
            [ind['objectives'][obj] for obj in self.mo_config.objectives]
            for ind in self.pareto_front
        ])
        
        # Normalize objectives
        normalized = self.optimizer._normalize_objectives(objectives)
        
        # Simple hypervolume approximation (for demonstration)
        # In practice, use a proper hypervolume calculation algorithm
        reference_point = np.ones(self.mo_config.num_objectives)
        
        # Calculate dominated volume (simplified)
        volume = 0.0
        for point in normalized:
            # Calculate volume of dominated region
            dominated_volume = np.prod(np.maximum(0, reference_point - point))
            volume += dominated_volume
        
        return volume
    
    def _calculate_convergence_metric(self) -> float:
        """Calculate convergence metric based on population diversity."""
        if len(self.population) < 2:
            return 1.0
        
        # Calculate objective diversity
        objectives = np.array([
            [ind['objectives'][obj] for obj in self.mo_config.objectives]
            for ind in self.population if ind['objectives'] is not None
        ])
        
        if len(objectives) < 2:
            return 1.0
        
        # Calculate standard deviation across objectives
        std_devs = np.std(objectives, axis=0)
        return np.mean(std_devs)
    
    def _select_best_architecture(self) -> Architecture:
        """Select the best architecture from Pareto front based on preferences."""
        if len(self.pareto_front) == 0:
            # Fallback to best individual in population
            if len(self.population) > 0:
                return self.population[0]['architecture']
            else:
                return self.search_space.sample_random_architecture()
        
        # Use weighted sum to select from Pareto front
        best_score = -float('inf')
        best_arch = self.pareto_front[0]['architecture']
        
        for individual in self.pareto_front:
            score = 0.0
            for obj_name in self.mo_config.objectives:
                weight = self.mo_config.constraint_weights.get(obj_name, 1.0)
                value = individual['objectives'][obj_name]
                
                if obj_name == "accuracy":
                    score += weight * value  # Maximize accuracy
                else:
                    score -= weight * value  # Minimize other objectives
            
            if score > best_score:
                best_score = score
                best_arch = individual['architecture']
        
        return best_arch
    
    def _log_generation_progress(self):
        """Log progress for current generation."""
        if len(self.pareto_front) > 0:
            # Calculate average objectives in Pareto front
            avg_objectives = {}
            for obj_name in self.mo_config.objectives:
                values = [ind['objectives'][obj_name] for ind in self.pareto_front]
                avg_objectives[obj_name] = np.mean(values)
            
            obj_str = ", ".join([f"{obj}: {val:.3f}" for obj, val in avg_objectives.items()])
            
            self.logger.info(f"  🎯 Pareto front size: {len(self.pareto_front)}")
            self.logger.info(f"  📊 Average objectives: {obj_str}")
            
            if len(self.search_metrics['hypervolume']) > 0:
                hv = self.search_metrics['hypervolume'][-1]
                self.logger.info(f"  📈 Hypervolume: {hv:.4f}")
    
    def get_search_metrics(self) -> Dict[str, Any]:
        """Get comprehensive search metrics."""
        return {
            **self.search_metrics,
            'pareto_front_architectures': len(self.pareto_front),
            'evaluation_cache_size': len(self.evaluation_cache),
            'final_hypervolume': self.search_metrics['hypervolume'][-1] if self.search_metrics['hypervolume'] else 0.0,
        }
    
    def get_pareto_front(self) -> List[Dict[str, Any]]:
        """Get the current Pareto front."""
        return [
            {
                'architecture': ind['architecture'],
                'objectives': ind['objectives'],
                'architecture_str': str(ind['architecture'])
            }
            for ind in self.pareto_front
        ]
    
    def visualize_pareto_front(self, save_path: Optional[str] = None):
        """Visualize the Pareto front."""
        if len(self.pareto_front) == 0:
            self.logger.warning("No Pareto front to visualize")
            return
        
        # Extract objectives
        objectives = np.array([
            [ind['objectives'][obj] for obj in self.mo_config.objectives]
            for ind in self.pareto_front
        ])
        
        num_objectives = len(self.mo_config.objectives)
        
        if num_objectives == 2:
            # 2D plot
            plt.figure(figsize=(10, 8))
            plt.scatter(objectives[:, 0], objectives[:, 1], 
                       c='blue', s=100, alpha=0.7, edgecolors='black')
            plt.xlabel(self.mo_config.objectives[0])
            plt.ylabel(self.mo_config.objectives[1])
            plt.title('Pareto Front')
            plt.grid(True, alpha=0.3)
            
        elif num_objectives == 3:
            # 3D plot
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2],
                      c='blue', s=100, alpha=0.7, edgecolors='black')
            ax.set_xlabel(self.mo_config.objectives[0])
            ax.set_ylabel(self.mo_config.objectives[1])
            ax.set_zlabel(self.mo_config.objectives[2])
            ax.set_title('3D Pareto Front')
            
        else:
            # Parallel coordinates plot for 4+ objectives
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Normalize objectives for better visualization
            normalized_obj = self.optimizer._normalize_objectives(objectives)
            
            for i, point in enumerate(normalized_obj):
                ax.plot(range(num_objectives), point, 'b-', alpha=0.6)
            
            ax.set_xticks(range(num_objectives))
            ax.set_xticklabels(self.mo_config.objectives, rotation=45)
            ax.set_ylabel('Normalized Objective Value')
            ax.set_title('Pareto Front - Parallel Coordinates')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Pareto front visualization saved to {save_path}")
        
        plt.show() 