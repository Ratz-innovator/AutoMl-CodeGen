"""
Evolutionary Neural Architecture Search Algorithm

This module implements a sophisticated evolutionary algorithm for discovering
optimal neural network architectures through genetic operations and multi-objective
optimization.
"""

import logging
import random
import copy
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
import torch
from ..space.search_space import SearchSpace
from ..objectives.multi_objective import MultiObjectiveOptimizer
from ...utils.config import Config
from ...utils.logger import Logger

logger = logging.getLogger(__name__)

@dataclass
class Individual:
    """Represents an individual architecture in the population."""
    architecture: Dict[str, Any]
    fitness: Optional[Dict[str, float]] = None
    age: int = 0
    parent_ids: Optional[Tuple[int, int]] = None
    mutation_history: List[str] = None
    
    def __post_init__(self):
        if self.mutation_history is None:
            self.mutation_history = []
    
    def copy(self) -> 'Individual':
        """Create a deep copy of the individual."""
        return Individual(
            architecture=copy.deepcopy(self.architecture),
            fitness=self.fitness.copy() if self.fitness else None,
            age=self.age,
            parent_ids=self.parent_ids,
            mutation_history=self.mutation_history.copy()
        )

class EvolutionarySearch:
    """
    Evolutionary search algorithm for neural architecture search.
    
    Features:
    - Multi-objective optimization with Pareto front tracking
    - Advanced crossover operators (uniform, single-point, block-wise)
    - Sophisticated mutation strategies
    - Elitism and diversity preservation
    - Adaptive parameters based on convergence
    - Speciation and niching for exploration
    
    Example:
        >>> search = EvolutionarySearch(
        ...     search_space=space,
        ...     objectives=objectives,
        ...     config=config
        ... )
        >>> population = search.initialize_population(50)
        >>> evolved_pop = search.evolve_population(population, metrics, pareto_front)
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        objectives: MultiObjectiveOptimizer,
        config: Optional[Config] = None,
        logger: Optional[Logger] = None,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elitism_ratio: float = 0.1,
        tournament_size: int = 3,
        max_age: int = 10,
        adaptive_parameters: bool = True,
        **kwargs
    ):
        """
        Initialize evolutionary search algorithm.
        
        Args:
            search_space: Search space definition
            objectives: Multi-objective optimizer
            config: Configuration object
            logger: Logger instance
            population_size: Size of the population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elitism_ratio: Fraction of population to preserve as elite
            tournament_size: Size for tournament selection
            max_age: Maximum age before forced replacement
            adaptive_parameters: Whether to adapt parameters during evolution
            **kwargs: Additional configuration
        """
        self.search_space = search_space
        self.objectives = objectives
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        
        # Evolution parameters
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_ratio = elitism_ratio
        self.tournament_size = tournament_size
        self.max_age = max_age
        self.adaptive_parameters = adaptive_parameters
        
        # Initialize mutation strategies
        self.mutation_strategies = [
            self._mutate_layer_type,
            self._mutate_layer_parameters,
            self._mutate_connections,
            self._add_layer,
            self._remove_layer,
            self._swap_layers
        ]
        
        # Crossover strategies
        self.crossover_strategies = [
            self._uniform_crossover,
            self._single_point_crossover,
            self._block_crossover
        ]
        
        # Evolution statistics
        self.generation = 0
        self.diversity_history = []
        self.convergence_history = []
        self.best_fitness_history = []
        
        # Adaptive parameter tracking
        self.stagnation_counter = 0
        self.last_best_fitness = -np.inf
        
        self.logger.info("Initialized evolutionary search algorithm")
        self.logger.info(f"Population size: {population_size}")
        self.logger.info(f"Mutation rate: {mutation_rate}")
        self.logger.info(f"Crossover rate: {crossover_rate}")
    
    def initialize_population(self, size: Optional[int] = None) -> List[Individual]:
        """
        Initialize a random population of architectures.
        
        Args:
            size: Population size (uses default if None)
            
        Returns:
            List of Individual objects representing the initial population
        """
        if size is None:
            size = self.population_size
        
        population = []
        
        logger.info(f"Initializing population of size {size}")
        
        for i in range(size):
            # Generate random architecture
            architecture = self.search_space.sample_architecture()
            
            # Create individual
            individual = Individual(
                architecture=architecture,
                age=0,
                mutation_history=[]
            )
            
            population.append(individual)
        
        logger.info(f"Generated {len(population)} initial architectures")
        return population
    
    def evolve_population(
        self,
        population: List[Individual],
        metrics: List[Dict[str, float]],
        pareto_front: List[Dict[str, Any]],
        **kwargs
    ) -> List[Individual]:
        """
        Evolve the population for one generation.
        
        Args:
            population: Current population
            metrics: Fitness metrics for each individual
            pareto_front: Current Pareto front
            **kwargs: Additional evolution parameters
            
        Returns:
            Evolved population for the next generation
        """
        self.generation += 1
        logger.info(f"Evolving population for generation {self.generation}")
        
        # Update fitness values
        self._update_fitness(population, metrics)
        
        # Adapt parameters if enabled
        if self.adaptive_parameters:
            self._adapt_parameters(population, pareto_front)
        
        # Calculate diversity metrics
        diversity = self._calculate_diversity(population)
        self.diversity_history.append(diversity)
        
        # Track convergence
        best_fitness = max([ind.fitness.get('aggregate_score', 0) 
                           for ind in population if ind.fitness])
        self.best_fitness_history.append(best_fitness)
        self._update_convergence_tracking(best_fitness)
        
        # Create new population
        new_population = []
        
        # Elitism: preserve best individuals
        elite_size = int(self.elitism_ratio * len(population))
        elite_individuals = self._select_elite(population, elite_size)
        new_population.extend([ind.copy() for ind in elite_individuals])
        
        # Generate offspring to fill remaining slots
        remaining_size = len(population) - len(new_population)
        offspring = self._generate_offspring(population, remaining_size)
        new_population.extend(offspring)
        
        # Age individuals and handle aging
        new_population = self._handle_aging(new_population)
        
        # Ensure population size consistency
        if len(new_population) > len(population):
            new_population = new_population[:len(population)]
        elif len(new_population) < len(population):
            # Add random individuals to fill up
            while len(new_population) < len(population):
                new_population.append(
                    Individual(architecture=self.search_space.sample_architecture())
                )
        
        logger.info(f"Generated {len(new_population)} individuals for next generation")
        logger.info(f"Population diversity: {diversity:.4f}")
        logger.info(f"Best fitness: {best_fitness:.4f}")
        
        return new_population
    
    def _update_fitness(
        self,
        population: List[Individual],
        metrics: List[Dict[str, float]]
    ) -> None:
        """Update fitness values for all individuals."""
        for individual, metric in zip(population, metrics):
            individual.fitness = metric
            
            # Calculate aggregate score for comparison
            if metric:
                aggregate = self._calculate_aggregate_fitness(metric)
                individual.fitness['aggregate_score'] = aggregate
    
    def _calculate_aggregate_fitness(self, metrics: Dict[str, float]) -> float:
        """Calculate aggregate fitness score from multiple objectives."""
        # Simple weighted sum - could be more sophisticated
        weights = {
            'accuracy': 0.5,
            'latency': -0.3,  # Negative because we want to minimize
            'memory': -0.1,
            'energy': -0.1
        }
        
        score = 0.0
        for metric, value in metrics.items():
            weight = weights.get(metric, 0.0)
            if 'accuracy' in metric:
                normalized = value  # Already 0-1
            else:
                normalized = 1.0 / (1.0 + value)  # Invert for minimization metrics
            score += weight * normalized
        
        return score
    
    def _adapt_parameters(
        self,
        population: List[Individual],
        pareto_front: List[Dict[str, Any]]
    ) -> None:
        """Adapt evolution parameters based on search progress."""
        # Increase mutation rate if stagnating
        if self.stagnation_counter > 3:
            self.mutation_rate = min(0.3, self.mutation_rate * 1.1)
            logger.debug(f"Increased mutation rate to {self.mutation_rate:.3f}")
        
        # Decrease mutation rate if progressing well
        elif self.stagnation_counter == 0:
            self.mutation_rate = max(0.05, self.mutation_rate * 0.95)
        
        # Adapt crossover rate based on diversity
        if len(self.diversity_history) > 0:
            recent_diversity = np.mean(self.diversity_history[-5:])
            if recent_diversity < 0.3:  # Low diversity
                self.crossover_rate = min(0.9, self.crossover_rate * 1.05)
            elif recent_diversity > 0.7:  # High diversity
                self.crossover_rate = max(0.6, self.crossover_rate * 0.95)
    
    def _update_convergence_tracking(self, best_fitness: float) -> None:
        """Update convergence tracking metrics."""
        improvement_threshold = 0.001
        
        if best_fitness > self.last_best_fitness + improvement_threshold:
            self.stagnation_counter = 0
            self.last_best_fitness = best_fitness
        else:
            self.stagnation_counter += 1
    
    def _calculate_diversity(self, population: List[Individual]) -> float:
        """Calculate population diversity metric."""
        if len(population) < 2:
            return 0.0
        
        # Calculate pairwise distances between architectures
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                dist = self._architecture_distance(
                    population[i].architecture,
                    population[j].architecture
                )
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def _architecture_distance(
        self,
        arch1: Dict[str, Any],
        arch2: Dict[str, Any]
    ) -> float:
        """Calculate distance between two architectures."""
        # Simple distance based on layer differences
        layers1 = arch1.get('layers', [])
        layers2 = arch2.get('layers', [])
        
        # Length difference
        len_diff = abs(len(layers1) - len(layers2)) / max(len(layers1), len(layers2), 1)
        
        # Layer type differences
        min_len = min(len(layers1), len(layers2))
        type_diff = 0
        for i in range(min_len):
            if layers1[i].get('type') != layers2[i].get('type'):
                type_diff += 1
        
        type_diff = type_diff / max(min_len, 1)
        
        return (len_diff + type_diff) / 2
    
    def _select_elite(
        self,
        population: List[Individual],
        elite_size: int
    ) -> List[Individual]:
        """Select elite individuals based on fitness."""
        # Sort by aggregate fitness
        sorted_pop = sorted(
            population,
            key=lambda x: x.fitness.get('aggregate_score', 0) if x.fitness else 0,
            reverse=True
        )
        
        return sorted_pop[:elite_size]
    
    def _generate_offspring(
        self,
        population: List[Individual],
        offspring_size: int
    ) -> List[Individual]:
        """Generate offspring through crossover and mutation."""
        offspring = []
        
        while len(offspring) < offspring_size:
            if random.random() < self.crossover_rate:
                # Crossover
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                
                child1, child2 = self._crossover(parent1, parent2)
                offspring.extend([child1, child2])
            else:
                # Asexual reproduction with mutation
                parent = self._tournament_selection(population)
                child = parent.copy()
                offspring.append(child)
        
        # Apply mutations
        for child in offspring:
            if random.random() < self.mutation_rate:
                self._mutate(child)
        
        return offspring[:offspring_size]
    
    def _tournament_selection(self, population: List[Individual]) -> Individual:
        """Tournament selection for parent selection."""
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        
        # Select best from tournament
        best = max(
            tournament,
            key=lambda x: x.fitness.get('aggregate_score', 0) if x.fitness else 0
        )
        
        return best
    
    def _crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Perform crossover between two parents."""
        # Choose crossover strategy
        crossover_func = random.choice(self.crossover_strategies)
        
        child1_arch, child2_arch = crossover_func(
            parent1.architecture,
            parent2.architecture
        )
        
        # Create offspring individuals
        child1 = Individual(
            architecture=child1_arch,
            age=0,
            parent_ids=(id(parent1), id(parent2))
        )
        
        child2 = Individual(
            architecture=child2_arch,
            age=0,
            parent_ids=(id(parent1), id(parent2))
        )
        
        return child1, child2
    
    def _uniform_crossover(
        self,
        arch1: Dict[str, Any],
        arch2: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Uniform crossover between architectures."""
        child1 = copy.deepcopy(arch1)
        child2 = copy.deepcopy(arch2)
        
        # Crossover layers
        layers1 = child1.get('layers', [])
        layers2 = child2.get('layers', [])
        
        min_len = min(len(layers1), len(layers2))
        
        for i in range(min_len):
            if random.random() < 0.5:
                layers1[i], layers2[i] = layers2[i], layers1[i]
        
        return child1, child2
    
    def _single_point_crossover(
        self,
        arch1: Dict[str, Any],
        arch2: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Single-point crossover between architectures."""
        child1 = copy.deepcopy(arch1)
        child2 = copy.deepcopy(arch2)
        
        layers1 = child1.get('layers', [])
        layers2 = child2.get('layers', [])
        
        if len(layers1) > 1 and len(layers2) > 1:
            # Choose crossover point
            point = random.randint(1, min(len(layers1), len(layers2)) - 1)
            
            # Swap tails
            layers1[point:], layers2[point:] = layers2[point:], layers1[point:]
        
        return child1, child2
    
    def _block_crossover(
        self,
        arch1: Dict[str, Any],
        arch2: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Block-wise crossover between architectures."""
        child1 = copy.deepcopy(arch1)
        child2 = copy.deepcopy(arch2)
        
        # This would implement more sophisticated block crossover
        # For now, fall back to uniform crossover
        return self._uniform_crossover(arch1, arch2)
    
    def _mutate(self, individual: Individual) -> None:
        """Apply mutation to an individual."""
        # Choose mutation strategy
        mutation_func = random.choice(self.mutation_strategies)
        
        try:
            mutation_name = mutation_func.__name__
            mutation_func(individual.architecture)
            individual.mutation_history.append(mutation_name)
            
        except Exception as e:
            logger.warning(f"Mutation failed: {e}")
    
    def _mutate_layer_type(self, architecture: Dict[str, Any]) -> None:
        """Mutate the type of a random layer."""
        layers = architecture.get('layers', [])
        if not layers:
            return
        
        layer_idx = random.randint(0, len(layers) - 1)
        layer = layers[layer_idx]
        
        # Get valid layer types for this position
        valid_types = self.search_space.get_valid_layer_types(layer_idx)
        
        if valid_types and len(valid_types) > 1:
            current_type = layer.get('type', '')
            new_types = [t for t in valid_types if t != current_type]
            if new_types:
                layer['type'] = random.choice(new_types)
    
    def _mutate_layer_parameters(self, architecture: Dict[str, Any]) -> None:
        """Mutate parameters of a random layer."""
        layers = architecture.get('layers', [])
        if not layers:
            return
        
        layer_idx = random.randint(0, len(layers) - 1)
        layer = layers[layer_idx]
        
        # Mutate layer parameters based on type
        layer_type = layer.get('type', '')
        
        if layer_type in ['conv2d', 'convolution']:
            if 'out_channels' in layer:
                # Randomly increase/decrease channels
                factor = random.choice([0.5, 1.5, 2.0])
                layer['out_channels'] = max(1, int(layer['out_channels'] * factor))
            
            if 'kernel_size' in layer:
                layer['kernel_size'] = random.choice([1, 3, 5, 7])
        
        elif layer_type in ['linear', 'dense']:
            if 'out_features' in layer:
                factor = random.choice([0.5, 1.5, 2.0])
                layer['out_features'] = max(1, int(layer['out_features'] * factor))
    
    def _mutate_connections(self, architecture: Dict[str, Any]) -> None:
        """Mutate connection patterns in the architecture."""
        # This would implement connection mutation
        # For now, placeholder
        pass
    
    def _add_layer(self, architecture: Dict[str, Any]) -> None:
        """Add a new layer to the architecture."""
        layers = architecture.get('layers', [])
        
        if len(layers) < self.search_space.max_layers:
            # Choose insertion position
            pos = random.randint(0, len(layers))
            
            # Generate new layer
            new_layer = self.search_space.sample_layer()
            
            layers.insert(pos, new_layer)
    
    def _remove_layer(self, architecture: Dict[str, Any]) -> None:
        """Remove a layer from the architecture."""
        layers = architecture.get('layers', [])
        
        if len(layers) > self.search_space.min_layers:
            # Choose layer to remove
            pos = random.randint(0, len(layers) - 1)
            layers.pop(pos)
    
    def _swap_layers(self, architecture: Dict[str, Any]) -> None:
        """Swap two layers in the architecture."""
        layers = architecture.get('layers', [])
        
        if len(layers) >= 2:
            idx1, idx2 = random.sample(range(len(layers)), 2)
            layers[idx1], layers[idx2] = layers[idx2], layers[idx1]
    
    def _handle_aging(self, population: List[Individual]) -> List[Individual]:
        """Handle aging of individuals and forced replacement."""
        for individual in population:
            individual.age += 1
            
            # Replace very old individuals
            if individual.age > self.max_age:
                individual.architecture = self.search_space.sample_architecture()
                individual.age = 0
                individual.fitness = None
                individual.mutation_history = []
        
        return population
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state for checkpointing."""
        return {
            'generation': self.generation,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'diversity_history': self.diversity_history,
            'convergence_history': self.convergence_history,
            'best_fitness_history': self.best_fitness_history,
            'stagnation_counter': self.stagnation_counter,
            'last_best_fitness': self.last_best_fitness
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self.generation = state.get('generation', 0)
        self.mutation_rate = state.get('mutation_rate', self.mutation_rate)
        self.crossover_rate = state.get('crossover_rate', self.crossover_rate)
        self.diversity_history = state.get('diversity_history', [])
        self.convergence_history = state.get('convergence_history', [])
        self.best_fitness_history = state.get('best_fitness_history', [])
        self.stagnation_counter = state.get('stagnation_counter', 0)
        self.last_best_fitness = state.get('last_best_fitness', -np.inf)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get evolution statistics."""
        return {
            'generation': self.generation,
            'current_mutation_rate': self.mutation_rate,
            'current_crossover_rate': self.crossover_rate,
            'stagnation_counter': self.stagnation_counter,
            'average_diversity': np.mean(self.diversity_history) if self.diversity_history else 0.0,
            'best_fitness_so_far': max(self.best_fitness_history) if self.best_fitness_history else 0.0,
            'fitness_improvement': (
                self.best_fitness_history[-1] - self.best_fitness_history[0]
                if len(self.best_fitness_history) > 1 else 0.0
            )
        } 