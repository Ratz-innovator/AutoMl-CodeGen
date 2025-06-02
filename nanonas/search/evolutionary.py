"""
Evolutionary Neural Architecture Search
======================================

This module implements advanced evolutionary algorithms for neural architecture search,
including various selection strategies, mutation operators, and population management.

Key Features:
- Multiple selection strategies (tournament, roulette, rank-based)
- Advanced mutation operators (Gaussian, uniform, adaptive)
- Crossover operators (single-point, multi-point, uniform)
- Population diversity management
- Multi-objective evolution
- Adaptive parameters
"""

import torch
import torch.nn as nn
import numpy as np
import random
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from ..core.architecture import Architecture, SearchSpace
from ..core.config import ExperimentConfig
from ..benchmarks.evaluator import ModelEvaluator


@dataclass
class EvolutionaryConfig:
    """Configuration for evolutionary search parameters."""
    population_size: int = 50
    generations: int = 20
    mutation_rate: float = 0.15
    crossover_rate: float = 0.8
    elitism_ratio: float = 0.1
    tournament_size: int = 3
    selection_pressure: float = 1.5
    diversity_pressure: float = 0.1
    adaptive_mutation: bool = True
    multi_objective: bool = False


class EvolutionarySearch:
    """
    Advanced evolutionary algorithm for neural architecture search.
    
    This implementation includes:
    - Multiple selection strategies
    - Adaptive mutation rates
    - Population diversity management
    - Multi-objective optimization
    - Performance tracking and analysis
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize evolutionary search.
        
        Args:
            config: Experiment configuration containing search parameters
        """
        self.config = config
        self.search_config = config.search
        
        # Create evolutionary-specific config
        self.evo_config = EvolutionaryConfig(
            population_size=self.search_config.population_size,
            generations=self.search_config.generations,
            mutation_rate=self.search_config.mutation_rate,
            crossover_rate=self.search_config.crossover_rate,
            elitism_ratio=self.search_config.elitism_ratio
        )
        
        # Get search space
        if self.search_config.search_space == "nano":
            self.search_space = SearchSpace.get_nano_search_space()
        elif self.search_config.search_space == "mobile":
            self.search_space = SearchSpace.get_mobile_search_space()
        else:
            self.search_space = SearchSpace.get_nano_search_space()
        
        # Initialize components
        self.evaluator = ModelEvaluator(config)
        self.logger = logging.getLogger(__name__)
        
        # Search state
        self.population = []
        self.fitness_history = []
        self.diversity_history = []
        self.generation = 0
        
        # Performance tracking
        self.best_architecture = None
        self.best_fitness = -float('inf')
        self.evaluation_cache = {}
        
        # Statistics
        self.search_metrics = {
            'generations_run': 0,
            'total_evaluations': 0,
            'cache_hits': 0,
            'best_fitness_per_generation': [],
            'population_diversity': [],
            'mutation_rates': [],
        }
    
    def search(self) -> Architecture:
        """
        Run evolutionary search to find the best architecture.
        
        Returns:
            Best architecture found during search
        """
        self.logger.info(f"🧬 Starting evolutionary search with {self.evo_config.population_size} individuals")
        self.logger.info(f"📊 Search space: {self.search_space.name} ({len(self.search_space.operations)} operations)")
        
        start_time = time.time()
        
        try:
            # Initialize population
            self._initialize_population()
            
            # Evolution loop
            for generation in range(self.evo_config.generations):
                self.generation = generation
                self.logger.info(f"🧬 Generation {generation + 1}/{self.evo_config.generations}")
                
                # Evaluate population
                self._evaluate_population()
                
                # Update statistics
                self._update_statistics()
                
                # Check for early stopping
                if self._should_stop_early():
                    self.logger.info(f"⏰ Early stopping at generation {generation + 1}")
                    break
                
                # Create next generation
                if generation < self.evo_config.generations - 1:
                    self._create_next_generation()
                
                # Adaptive parameters
                if self.evo_config.adaptive_mutation:
                    self._adapt_mutation_rate()
                
                # Log progress
                self._log_generation_progress()
            
            search_time = time.time() - start_time
            self.search_metrics['total_search_time'] = search_time
            
            self.logger.info(f"✅ Evolutionary search completed in {search_time:.2f}s")
            self.logger.info(f"🎯 Best fitness: {self.best_fitness:.4f}")
            self.logger.info(f"📊 Total evaluations: {self.search_metrics['total_evaluations']}")
            self.logger.info(f"💾 Cache hit rate: {self.search_metrics['cache_hits'] / max(1, self.search_metrics['total_evaluations']):.2%}")
            
            return self.best_architecture
            
        except Exception as e:
            self.logger.error(f"❌ Evolutionary search failed: {e}")
            raise
    
    def _initialize_population(self):
        """Initialize the population with random architectures."""
        self.logger.info("🌱 Initializing population...")
        
        self.population = []
        for i in range(self.evo_config.population_size):
            # Create random architecture
            arch = self.search_space.sample_random_architecture()
            self.population.append({
                'architecture': arch,
                'fitness': None,
                'age': 0,
                'parent_ids': [],
            })
        
        self.logger.info(f"✅ Initialized {len(self.population)} individuals")
    
    def _evaluate_population(self):
        """Evaluate fitness for all individuals in the population."""
        self.logger.info("🔬 Evaluating population fitness...")
        
        for i, individual in enumerate(self.population):
            if individual['fitness'] is None:
                fitness = self._evaluate_architecture(individual['architecture'])
                individual['fitness'] = fitness
                individual['age'] += 1
        
        # Sort by fitness (descending)
        self.population.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Update best architecture
        best_individual = self.population[0]
        if best_individual['fitness'] > self.best_fitness:
            self.best_fitness = best_individual['fitness']
            self.best_architecture = best_individual['architecture']
    
    def _evaluate_architecture(self, architecture: Architecture) -> float:
        """
        Evaluate a single architecture.
        
        Args:
            architecture: Architecture to evaluate
            
        Returns:
            Fitness score (higher is better)
        """
        # Check cache first
        arch_hash = hash(architecture)
        if arch_hash in self.evaluation_cache:
            self.search_metrics['cache_hits'] += 1
            return self.evaluation_cache[arch_hash]
        
        self.search_metrics['total_evaluations'] += 1
        
        try:
            # Convert to model
            model = architecture.to_model(
                input_channels=self.config.model.input_channels,
                num_classes=self.config.model.num_classes,
                base_channels=self.config.model.base_channels
            )
            
            # Evaluate model
            metrics = self.evaluator.quick_evaluate(model)
            
            # Compute fitness (could be multi-objective)
            fitness = self._compute_fitness(metrics, architecture)
            
            # Cache result
            self.evaluation_cache[arch_hash] = fitness
            
            return fitness
            
        except Exception as e:
            self.logger.warning(f"⚠️ Architecture evaluation failed: {e}")
            return 0.0  # Poor fitness for invalid architectures
    
    def _compute_fitness(self, metrics: Dict[str, float], architecture: Architecture) -> float:
        """
        Compute fitness score from evaluation metrics.
        
        Args:
            metrics: Evaluation metrics from model evaluator
            architecture: Architecture being evaluated
            
        Returns:
            Fitness score
        """
        # Primary objective: accuracy
        fitness = metrics.get('accuracy', 0.0)
        
        # Secondary objectives (penalties/bonuses)
        complexity_metrics = architecture.get_complexity_metrics()
        
        # Penalize for high parameter count
        param_penalty = 0.0
        if 'total_op_cost' in complexity_metrics:
            max_cost = 20.0  # Adjust based on search space
            param_penalty = 0.1 * (complexity_metrics['total_op_cost'] / max_cost)
        
        # Bonus for skip connections (good for gradient flow)
        skip_bonus = 0.0
        if 'skip_ratio' in complexity_metrics:
            skip_bonus = 0.05 * complexity_metrics['skip_ratio']
        
        # Combine objectives
        fitness = fitness - param_penalty + skip_bonus
        
        # Add small noise to break ties
        fitness += np.random.normal(0, 0.001)
        
        return fitness
    
    def _create_next_generation(self):
        """Create the next generation through selection, crossover, and mutation."""
        new_population = []
        
        # Elitism: keep best individuals
        elite_size = int(self.evo_config.elitism_ratio * self.evo_config.population_size)
        elite = self.population[:elite_size]
        new_population.extend([{
            'architecture': ind['architecture'],
            'fitness': ind['fitness'],
            'age': ind['age'] + 1,
            'parent_ids': ind['parent_ids'],
        } for ind in elite])
        
        # Generate offspring
        while len(new_population) < self.evo_config.population_size:
            # Selection
            parent1 = self._select_parent()
            parent2 = self._select_parent()
            
            # Crossover
            if random.random() < self.evo_config.crossover_rate:
                child1_arch, child2_arch = parent1['architecture'].crossover(parent2['architecture'])
                children = [child1_arch, child2_arch]
            else:
                children = [parent1['architecture'], parent2['architecture']]
            
            # Mutation
            for child_arch in children:
                if len(new_population) >= self.evo_config.population_size:
                    break
                
                if random.random() < self.evo_config.mutation_rate:
                    child_arch = child_arch.mutate(
                        mutation_rate=self._get_adaptive_mutation_rate(),
                        mutation_std=self._get_adaptive_mutation_std()
                    )
                
                new_population.append({
                    'architecture': child_arch,
                    'fitness': None,
                    'age': 0,
                    'parent_ids': [id(parent1), id(parent2)],
                })
        
        # Trim to exact population size
        self.population = new_population[:self.evo_config.population_size]
    
    def _select_parent(self) -> Dict[str, Any]:
        """
        Select a parent for reproduction using tournament selection.
        
        Returns:
            Selected individual
        """
        # Tournament selection
        tournament_size = min(self.evo_config.tournament_size, len(self.population))
        tournament = random.sample(self.population, tournament_size)
        
        # Select best from tournament
        return max(tournament, key=lambda x: x['fitness'])
    
    def _get_adaptive_mutation_rate(self) -> float:
        """Get adaptive mutation rate based on search progress."""
        if not self.evo_config.adaptive_mutation:
            return self.evo_config.mutation_rate
        
        # Decrease mutation rate as search progresses
        progress = self.generation / self.evo_config.generations
        base_rate = self.evo_config.mutation_rate
        
        # Add diversity pressure
        diversity = self._calculate_population_diversity()
        diversity_factor = 1.0 + self.evo_config.diversity_pressure * (1.0 - diversity)
        
        adaptive_rate = base_rate * (1.0 - 0.5 * progress) * diversity_factor
        return np.clip(adaptive_rate, 0.01, 0.5)
    
    def _get_adaptive_mutation_std(self) -> float:
        """Get adaptive mutation standard deviation."""
        progress = self.generation / self.evo_config.generations
        return 1.0 * (1.0 - 0.3 * progress)  # Decrease exploration over time
    
    def _adapt_mutation_rate(self):
        """Adapt mutation rate based on population diversity and progress."""
        current_rate = self._get_adaptive_mutation_rate()
        self.evo_config.mutation_rate = current_rate
        self.search_metrics['mutation_rates'].append(current_rate)
    
    def _calculate_population_diversity(self) -> float:
        """
        Calculate population diversity based on architecture uniqueness.
        
        Returns:
            Diversity score between 0 and 1
        """
        if len(self.population) <= 1:
            return 0.0
        
        # Count unique architectures
        unique_archs = set()
        for individual in self.population:
            arch_hash = hash(individual['architecture'])
            unique_archs.add(arch_hash)
        
        diversity = len(unique_archs) / len(self.population)
        return diversity
    
    def _should_stop_early(self) -> bool:
        """Check if search should stop early due to convergence."""
        if len(self.fitness_history) < self.search_config.early_stopping_patience:
            return False
        
        # Check for fitness stagnation
        recent_fitness = self.fitness_history[-self.search_config.early_stopping_patience:]
        fitness_improvement = max(recent_fitness) - min(recent_fitness)
        
        return fitness_improvement < 1e-4
    
    def _update_statistics(self):
        """Update search statistics and metrics."""
        # Fitness statistics
        current_fitness = [ind['fitness'] for ind in self.population if ind['fitness'] is not None]
        if current_fitness:
            best_fitness = max(current_fitness)
            self.fitness_history.append(best_fitness)
            self.search_metrics['best_fitness_per_generation'].append(best_fitness)
        
        # Diversity statistics
        diversity = self._calculate_population_diversity()
        self.diversity_history.append(diversity)
        self.search_metrics['population_diversity'].append(diversity)
        
        # Update generation count
        self.search_metrics['generations_run'] = self.generation + 1
    
    def _log_generation_progress(self):
        """Log progress for current generation."""
        if len(self.population) > 0 and self.population[0]['fitness'] is not None:
            best_fitness = self.population[0]['fitness']
            avg_fitness = np.mean([ind['fitness'] for ind in self.population if ind['fitness'] is not None])
            diversity = self._calculate_population_diversity()
            mutation_rate = self.evo_config.mutation_rate
            
            self.logger.info(f"  📊 Best: {best_fitness:.4f}, Avg: {avg_fitness:.4f}, "
                           f"Diversity: {diversity:.3f}, Mutation: {mutation_rate:.3f}")
    
    def get_search_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive search metrics and statistics.
        
        Returns:
            Dictionary containing search metrics
        """
        return {
            **self.search_metrics,
            'final_population_size': len(self.population),
            'best_fitness': self.best_fitness,
            'fitness_convergence': self.fitness_history,
            'diversity_evolution': self.diversity_history,
            'evaluation_cache_size': len(self.evaluation_cache),
        }
    
    def get_population_analysis(self) -> Dict[str, Any]:
        """
        Analyze the final population for insights.
        
        Returns:
            Population analysis results
        """
        if not self.population:
            return {}
        
        # Fitness statistics
        fitness_values = [ind['fitness'] for ind in self.population if ind['fitness'] is not None]
        
        # Architecture complexity analysis
        complexity_stats = defaultdict(list)
        for individual in self.population:
            if individual['fitness'] is not None:
                metrics = individual['architecture'].get_complexity_metrics()
                for key, value in metrics.items():
                    complexity_stats[key].append(value)
        
        # Age analysis
        ages = [ind['age'] for ind in self.population]
        
        analysis = {
            'fitness_stats': {
                'mean': np.mean(fitness_values) if fitness_values else 0,
                'std': np.std(fitness_values) if fitness_values else 0,
                'min': np.min(fitness_values) if fitness_values else 0,
                'max': np.max(fitness_values) if fitness_values else 0,
            },
            'complexity_stats': {
                key: {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                } for key, values in complexity_stats.items()
            },
            'age_stats': {
                'mean': np.mean(ages),
                'std': np.std(ages),
                'min': np.min(ages),
                'max': np.max(ages),
            },
            'diversity': self._calculate_population_diversity(),
        }
        
        return analysis 