"""
Bayesian Optimization for Neural Architecture Search
==================================================

This module implements Bayesian optimization with Gaussian process surrogate models
and Expected Improvement acquisition for efficient neural architecture search.

Key Features:
- Gaussian process surrogate models for performance prediction
- Expected Improvement (EI) and Upper Confidence Bound (UCB) acquisition functions
- Multi-objective acquisition functions
- Architecture encoding for GP input
- Intelligent initialization strategies
- Convergence monitoring and adaptive sampling
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

from ..core.base import BaseSearchStrategy
from ..core.architecture import Architecture, SearchSpace
from ..core.config import ExperimentConfig
from ..benchmarks.evaluator import ModelEvaluator

logger = logging.getLogger(__name__)


@dataclass
class BayesianOptimizationConfig:
    """Configuration for Bayesian optimization."""
    
    # Search parameters
    num_iterations: int = 100
    initial_samples: int = 20
    acquisition_function: str = "ei"  # "ei", "ucb", "pi", "multi_ei"
    
    # Gaussian process parameters
    kernel_type: str = "matern"  # "rbf", "matern"
    kernel_length_scale: float = 1.0
    kernel_nu: float = 2.5  # For Matern kernel
    noise_level: float = 1e-5
    
    # Acquisition function parameters
    xi: float = 0.01  # Exploration-exploitation trade-off
    kappa: float = 2.576  # UCB parameter (99% confidence)
    
    # Multi-objective parameters
    objectives: List[str] = field(default_factory=lambda: ["accuracy", "flops"])
    objective_weights: Dict[str, float] = field(default_factory=dict)
    
    # Optimization parameters
    acquisition_optimizer: str = "lbfgs"  # "lbfgs", "random", "genetic"
    n_restarts: int = 10
    max_iter: int = 1000
    
    # Architecture encoding parameters
    encoding_dimension: int = 50
    use_kernel_pca: bool = True
    pca_components: int = 20
    
    # Convergence parameters
    convergence_patience: int = 20
    min_improvement: float = 1e-4
    
    def __post_init__(self):
        """Set default objective weights."""
        if not self.objective_weights:
            self.objective_weights = {obj: 1.0 for obj in self.objectives}


class ArchitectureEncoder:
    """Encoder for converting architectures to fixed-dimension vectors."""
    
    def __init__(self, search_space: SearchSpace, dimension: int = 50):
        """Initialize architecture encoder."""
        self.search_space = search_space
        self.dimension = dimension
        self.scaler = StandardScaler()
        self.fitted = False
        
    def encode_architecture(self, architecture: Architecture) -> np.ndarray:
        """Encode architecture to fixed-dimension vector."""
        # Basic encoding features
        features = []
        
        # Architecture encoding (padded or truncated to fixed size)
        if architecture.encoding:
            encoding = architecture.encoding[:self.dimension//2]
            encoding.extend([0] * (self.dimension//2 - len(encoding)))
            features.extend(encoding)
        else:
            features.extend([0] * (self.dimension//2))
        
        # Complexity metrics
        complexity = architecture.get_complexity_metrics()
        complexity_features = [
            complexity.get('flops', 0),
            complexity.get('params', 0),
            complexity.get('memory', 0),
            complexity.get('latency', 0),
            complexity.get('energy', 0)
        ]
        features.extend(complexity_features)
        
        # Architecture statistics
        if architecture.encoding:
            encoding_array = np.array(architecture.encoding)
            stats = [
                np.mean(encoding_array),
                np.std(encoding_array),
                np.max(encoding_array),
                np.min(encoding_array),
                len(np.unique(encoding_array))
            ]
        else:
            stats = [0.0] * 5
        features.extend(stats)
        
        # Pad or truncate to target dimension
        if len(features) < self.dimension:
            features.extend([0.0] * (self.dimension - len(features)))
        else:
            features = features[:self.dimension]
        
        return np.array(features, dtype=np.float32)
    
    def encode_batch(self, architectures: List[Architecture]) -> np.ndarray:
        """Encode batch of architectures."""
        encoded = np.array([self.encode_architecture(arch) for arch in architectures])
        
        # Fit scaler on first batch
        if not self.fitted:
            self.scaler.fit(encoded)
            self.fitted = True
        
        return self.scaler.transform(encoded)


class GaussianProcessSurrogate:
    """Gaussian process surrogate model for architecture performance."""
    
    def __init__(self, config: BayesianOptimizationConfig):
        """Initialize GP surrogate."""
        self.config = config
        self.models: Dict[str, GaussianProcessRegressor] = {}
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Dict[str, np.ndarray] = {}
        
        # Initialize GP models for each objective
        for objective in config.objectives:
            self.models[objective] = self._create_gp_model()
    
    def _create_gp_model(self) -> GaussianProcessRegressor:
        """Create Gaussian process model."""
        # Define kernel
        if self.config.kernel_type == "rbf":
            kernel = ConstantKernel(1.0) * RBF(
                length_scale=self.config.kernel_length_scale,
                length_scale_bounds=(1e-3, 1e3)
            )
        elif self.config.kernel_type == "matern":
            kernel = ConstantKernel(1.0) * Matern(
                length_scale=self.config.kernel_length_scale,
                length_scale_bounds=(1e-3, 1e3),
                nu=self.config.kernel_nu
            )
        else:
            raise ValueError(f"Unknown kernel type: {self.config.kernel_type}")
        
        # Add noise kernel
        kernel += WhiteKernel(noise_level=self.config.noise_level)
        
        # Create GP model
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=self.config.n_restarts,
            random_state=42
        )
        
        return gp
    
    def update(self, X: np.ndarray, y: Dict[str, np.ndarray]) -> None:
        """Update GP models with new data."""
        if self.X_train is None:
            self.X_train = X.copy()
            self.y_train = {obj: y[obj].copy() for obj in y.keys()}
        else:
            self.X_train = np.vstack([self.X_train, X])
            for obj in y.keys():
                if obj in self.y_train:
                    self.y_train[obj] = np.concatenate([self.y_train[obj], y[obj]])
                else:
                    self.y_train[obj] = y[obj].copy()
        
        # Retrain models
        for objective in self.config.objectives:
            if objective in self.y_train:
                self.models[objective].fit(self.X_train, self.y_train[objective])
    
    def predict(self, X: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Predict mean and std for all objectives."""
        means = {}
        stds = {}
        
        for objective in self.config.objectives:
            if objective in self.models and objective in self.y_train:
                mean, std = self.models[objective].predict(X, return_std=True)
                means[objective] = mean
                stds[objective] = std
            else:
                # Default predictions for untrained models
                means[objective] = np.zeros(len(X))
                stds[objective] = np.ones(len(X))
        
        return means, stds


class AcquisitionFunction:
    """Acquisition functions for Bayesian optimization."""
    
    def __init__(self, config: BayesianOptimizationConfig, gp_surrogate: GaussianProcessSurrogate):
        """Initialize acquisition function."""
        self.config = config
        self.gp_surrogate = gp_surrogate
        
    def expected_improvement(self, X: np.ndarray, best_y: Dict[str, float]) -> np.ndarray:
        """Expected Improvement acquisition function."""
        means, stds = self.gp_surrogate.predict(X)
        
        # Compute weighted EI across objectives
        ei_values = np.zeros(len(X))
        
        for objective in self.config.objectives:
            if objective in means and objective in best_y:
                mean = means[objective]
                std = stds[objective]
                best = best_y[objective]
                weight = self.config.objective_weights.get(objective, 1.0)
                
                # For accuracy (maximize), use standard EI
                # For other metrics (minimize), flip the sign
                if objective == "accuracy":
                    improvement = mean - best - self.config.xi
                else:
                    improvement = best - mean - self.config.xi
                
                z = improvement / (std + 1e-9)
                ei = improvement * norm.cdf(z) + std * norm.pdf(z)
                ei_values += weight * ei
        
        return ei_values
    
    def upper_confidence_bound(self, X: np.ndarray) -> np.ndarray:
        """Upper Confidence Bound acquisition function."""
        means, stds = self.gp_surrogate.predict(X)
        
        # Compute weighted UCB across objectives
        ucb_values = np.zeros(len(X))
        
        for objective in self.config.objectives:
            if objective in means:
                mean = means[objective]
                std = stds[objective]
                weight = self.config.objective_weights.get(objective, 1.0)
                
                # For accuracy (maximize), use standard UCB
                # For other metrics (minimize), flip the sign
                if objective == "accuracy":
                    ucb = mean + self.config.kappa * std
                else:
                    ucb = -mean + self.config.kappa * std
                
                ucb_values += weight * ucb
        
        return ucb_values
    
    def probability_of_improvement(self, X: np.ndarray, best_y: Dict[str, float]) -> np.ndarray:
        """Probability of Improvement acquisition function."""
        means, stds = self.gp_surrogate.predict(X)
        
        # Compute weighted PI across objectives
        pi_values = np.ones(len(X))
        
        for objective in self.config.objectives:
            if objective in means and objective in best_y:
                mean = means[objective]
                std = stds[objective]
                best = best_y[objective]
                weight = self.config.objective_weights.get(objective, 1.0)
                
                # For accuracy (maximize), use standard PI
                # For other metrics (minimize), flip the sign
                if objective == "accuracy":
                    improvement = mean - best - self.config.xi
                else:
                    improvement = best - mean - self.config.xi
                
                z = improvement / (std + 1e-9)
                pi = norm.cdf(z)
                pi_values *= pi ** weight
        
        return pi_values
    
    def __call__(self, X: np.ndarray, best_y: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Compute acquisition function values."""
        if self.config.acquisition_function == "ei":
            return self.expected_improvement(X, best_y or {})
        elif self.config.acquisition_function == "ucb":
            return self.upper_confidence_bound(X)
        elif self.config.acquisition_function == "pi":
            return self.probability_of_improvement(X, best_y or {})
        else:
            raise ValueError(f"Unknown acquisition function: {self.config.acquisition_function}")


class BayesianOptimizationSearch(BaseSearchStrategy):
    """
    Bayesian optimization search strategy for neural architecture search.
    
    Uses Gaussian process surrogate models to efficiently explore the
    architecture space and find high-performing architectures.
    """
    
    def __init__(self, config: ExperimentConfig):
        """Initialize Bayesian optimization search."""
        super().__init__(config)
        
        # Bayesian optimization configuration
        if hasattr(config.search, 'bayesian_optimization'):
            self.bo_config = config.search.bayesian_optimization
        else:
            self.bo_config = BayesianOptimizationConfig()
        
        # Initialize components
        self.encoder = ArchitectureEncoder(self.search_space, self.bo_config.encoding_dimension)
        self.gp_surrogate = GaussianProcessSurrogate(self.bo_config)
        self.acquisition_function = AcquisitionFunction(self.bo_config, self.gp_surrogate)
        
        # Search state
        self.evaluated_architectures: List[Architecture] = []
        self.performance_history: List[Dict[str, float]] = []
        self.best_architecture: Optional[Architecture] = None
        self.best_performance: Dict[str, float] = {}
        self.convergence_history: List[float] = []
        
        logger.info(f"Bayesian optimization initialized with {self.bo_config.num_iterations} iterations")
    
    def search(self) -> Architecture:
        """
        Run Bayesian optimization search.
        
        Returns:
            Best architecture found during search
        """
        logger.info("Starting Bayesian optimization search")
        
        # Phase 1: Initial sampling
        self._initial_sampling()
        
        # Phase 2: Bayesian optimization loop
        for iteration in range(self.bo_config.num_iterations - self.bo_config.initial_samples):
            # Find next architecture to evaluate
            next_architecture = self._find_next_architecture()
            
            # Evaluate architecture
            performance = self._evaluate_architecture(next_architecture)
            
            # Update data
            self._update_data(next_architecture, performance)
            
            # Update best architecture
            self._update_best_architecture(next_architecture, performance)
            
            # Check convergence
            if self._check_convergence():
                logger.info(f"Converged at iteration {iteration + self.bo_config.initial_samples}")
                break
            
            # Log progress
            if (iteration + 1) % 10 == 0:
                self._log_progress(iteration + self.bo_config.initial_samples)
        
        logger.info("Bayesian optimization search completed")
        return self.best_architecture
    
    def _initial_sampling(self) -> None:
        """Perform initial random sampling to bootstrap GP."""
        logger.info(f"Initial sampling: {self.bo_config.initial_samples} architectures")
        
        for i in range(self.bo_config.initial_samples):
            # Sample random architecture
            architecture = self.search_space.sample_random_architecture()
            
            # Evaluate architecture
            performance = self._evaluate_architecture(architecture)
            
            # Update data
            self._update_data(architecture, performance)
            
            # Update best architecture
            self._update_best_architecture(architecture, performance)
            
            if (i + 1) % 5 == 0:
                logger.info(f"Initial sampling progress: {i + 1}/{self.bo_config.initial_samples}")
    
    def _find_next_architecture(self) -> Architecture:
        """Find next architecture to evaluate using acquisition function."""
        # Generate candidate architectures
        candidates = self._generate_candidates()
        
        # Encode candidates
        X_candidates = self.encoder.encode_batch(candidates)
        
        # Compute acquisition function values
        acquisition_values = self.acquisition_function(X_candidates, self.best_performance)
        
        # Select best candidate
        best_idx = np.argmax(acquisition_values)
        return candidates[best_idx]
    
    def _generate_candidates(self, num_candidates: int = 1000) -> List[Architecture]:
        """Generate candidate architectures for evaluation."""
        candidates = []
        
        # Random candidates
        for _ in range(num_candidates // 2):
            candidates.append(self.search_space.sample_random_architecture())
        
        # Mutation-based candidates (if we have existing architectures)
        if self.evaluated_architectures:
            for _ in range(num_candidates // 2):
                base_arch = np.random.choice(self.evaluated_architectures)
                mutated_arch = base_arch.mutate(mutation_rate=0.2)
                candidates.append(mutated_arch)
        else:
            # More random candidates if no existing architectures
            for _ in range(num_candidates // 2):
                candidates.append(self.search_space.sample_random_architecture())
        
        return candidates
    
    def _evaluate_architecture(self, architecture: Architecture) -> Dict[str, float]:
        """Evaluate architecture on all objectives."""
        performance = {}
        
        # Use evaluator to get performance metrics
        model = architecture.to_model()
        if self.evaluator is None:
            self.evaluator = ModelEvaluator(self.config)
        
        results = self.evaluator.evaluate_quick(model)
        
        # Map results to objectives
        performance['accuracy'] = results.get('test_accuracy', 0.0)
        
        # Get complexity metrics
        complexity = architecture.get_complexity_metrics()
        performance['flops'] = complexity.get('flops', 0.0)
        performance['params'] = complexity.get('params', 0.0)
        performance['memory'] = complexity.get('memory', 0.0)
        performance['latency'] = complexity.get('latency', 0.0)
        performance['energy'] = complexity.get('energy', 0.0)
        
        return performance
    
    def _update_data(self, architecture: Architecture, performance: Dict[str, float]) -> None:
        """Update training data for GP."""
        self.evaluated_architectures.append(architecture)
        self.performance_history.append(performance)
        
        # Encode architecture
        X_new = self.encoder.encode_batch([architecture])
        
        # Prepare performance data
        y_new = {}
        for objective in self.bo_config.objectives:
            if objective in performance:
                y_new[objective] = np.array([performance[objective]])
        
        # Update GP surrogate
        self.gp_surrogate.update(X_new, y_new)
    
    def _update_best_architecture(self, architecture: Architecture, performance: Dict[str, float]) -> None:
        """Update best architecture based on multi-objective performance."""
        if not self.best_performance:
            # First architecture
            self.best_architecture = architecture
            self.best_performance = performance.copy()
        else:
            # Check if this architecture is better
            is_better = False
            
            # Simple weighted sum approach for multi-objective
            current_score = 0.0
            best_score = 0.0
            
            for objective in self.bo_config.objectives:
                if objective in performance and objective in self.best_performance:
                    weight = self.bo_config.objective_weights.get(objective, 1.0)
                    
                    if objective == "accuracy":
                        # Higher is better
                        current_score += weight * performance[objective]
                        best_score += weight * self.best_performance[objective]
                    else:
                        # Lower is better (normalize and invert)
                        current_val = 1.0 / (1.0 + performance[objective])
                        best_val = 1.0 / (1.0 + self.best_performance[objective])
                        current_score += weight * current_val
                        best_score += weight * best_val
            
            if current_score > best_score:
                self.best_architecture = architecture
                self.best_performance = performance.copy()
                is_better = True
            
            # Track convergence
            improvement = abs(current_score - best_score) / max(abs(best_score), 1e-9)
            self.convergence_history.append(improvement)
    
    def _check_convergence(self) -> bool:
        """Check if search has converged."""
        if len(self.convergence_history) < self.bo_config.convergence_patience:
            return False
        
        # Check if improvements are consistently small
        recent_improvements = self.convergence_history[-self.bo_config.convergence_patience:]
        avg_improvement = np.mean(recent_improvements)
        
        return avg_improvement < self.bo_config.min_improvement
    
    def _log_progress(self, iteration: int) -> None:
        """Log search progress."""
        best_acc = self.best_performance.get('accuracy', 0.0)
        best_flops = self.best_performance.get('flops', 0.0)
        
        logger.info(
            f"Iteration {iteration:3d} | "
            f"Best Accuracy: {best_acc:.3f} | "
            f"Best FLOPs: {best_flops:.0f} | "
            f"Evaluated: {len(self.evaluated_architectures)}"
        )
    
    def get_search_metrics(self) -> Dict[str, Any]:
        """Get comprehensive search metrics."""
        return {
            'num_evaluations': len(self.evaluated_architectures),
            'best_performance': self.best_performance,
            'convergence_history': self.convergence_history,
            'performance_history': self.performance_history,
            'acquisition_function': self.bo_config.acquisition_function,
            'kernel_type': self.bo_config.kernel_type
        }
    
    def visualize_search_progress(self, save_path: Optional[str] = None) -> None:
        """Visualize Bayesian optimization search progress."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Performance over iterations
        if self.performance_history:
            iterations = range(len(self.performance_history))
            for objective in self.bo_config.objectives:
                if objective in self.performance_history[0]:
                    values = [perf[objective] for perf in self.performance_history]
                    axes[0, 0].plot(iterations, values, label=objective, marker='o', markersize=3)
            
            axes[0, 0].set_title('Performance Over Iterations')
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Performance')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Convergence history
        if self.convergence_history:
            axes[0, 1].plot(self.convergence_history)
            axes[0, 1].set_title('Convergence History')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Improvement')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Pareto front (if multi-objective)
        if len(self.bo_config.objectives) >= 2 and self.performance_history:
            obj1, obj2 = self.bo_config.objectives[:2]
            x_vals = [perf.get(obj1, 0) for perf in self.performance_history]
            y_vals = [perf.get(obj2, 0) for perf in self.performance_history]
            
            scatter = axes[1, 0].scatter(x_vals, y_vals, c=range(len(x_vals)), 
                                       cmap='viridis', alpha=0.7)
            axes[1, 0].set_title(f'Search Trajectory: {obj1} vs {obj2}')
            axes[1, 0].set_xlabel(obj1)
            axes[1, 0].set_ylabel(obj2)
            plt.colorbar(scatter, ax=axes[1, 0], label='Iteration')
        
        # Best performance evolution
        if self.performance_history:
            best_evolution = {}
            for objective in self.bo_config.objectives:
                best_evolution[objective] = []
                best_val = None
                
                for perf in self.performance_history:
                    if objective in perf:
                        current_val = perf[objective]
                        if best_val is None:
                            best_val = current_val
                        elif objective == "accuracy":
                            best_val = max(best_val, current_val)
                        else:
                            best_val = min(best_val, current_val)
                        best_evolution[objective].append(best_val)
            
            iterations = range(len(self.performance_history))
            for objective in self.bo_config.objectives:
                if objective in best_evolution:
                    axes[1, 1].plot(iterations, best_evolution[objective], 
                                   label=f'Best {objective}', marker='o', markersize=3)
            
            axes[1, 1].set_title('Best Performance Evolution')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Best Performance')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Search progress visualization saved to {save_path}")
        
        plt.show() 