"""
Random Search for Neural Architecture Search
==========================================

Implementation of random search as a baseline for neural architecture search.
While simple, random search often provides surprisingly competitive results
and serves as an important baseline for evaluating more sophisticated methods.

Key Features:
- Pure random sampling from search space
- Configurable evaluation budget
- Simple but effective baseline
- Statistical analysis of random architectures
"""

import torch
import numpy as np
import time
import logging
import random
from typing import Dict, Any, List, Optional

from ..core.base import BaseSearchStrategy
from ..core.architecture import Architecture, SearchSpace
from ..core.config import ExperimentConfig
from ..benchmarks.evaluator import ModelEvaluator


class RandomSearch(BaseSearchStrategy):
    """
    Random search strategy for neural architecture search.
    
    This strategy serves as a simple but important baseline by randomly
    sampling architectures from the search space and evaluating them.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize random search strategy.
        
        Args:
            config: Experiment configuration
        """
        super().__init__(config)
        
        # Get search space
        if self.config.search.search_space == "nano":
            self.search_space = SearchSpace.get_nano_search_space()
        elif self.config.search.search_space == "mobile":
            self.search_space = SearchSpace.get_mobile_search_space()
        else:
            self.search_space = SearchSpace.get_nano_search_space()
        
        # Search configuration
        self.num_samples = config.search.population_size * config.search.generations
        self.max_time = config.search.max_search_time
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator(config)
        
        # Search state
        self.evaluated_architectures = []
        self.best_architecture = None
        self.best_fitness = -float('inf')
        
        # Statistics
        self.search_metrics = {
            'total_evaluations': 0,
            'architectures_evaluated': [],
            'fitness_history': [],
            'complexity_stats': [],
            'unique_architectures': set(),
        }
    
    def search(self) -> Architecture:
        """
        Run random search to find the best architecture.
        
        Returns:
            Best architecture found during search
        """
        self.logger.info("🎲 Starting random search...")
        self.logger.info(f"📊 Search space: {self.search_space.name}")
        self.logger.info(f"🔢 Number of samples: {self.num_samples}")
        self.logger.info(f"⏰ Max time: {self.max_time}s")
        
        start_time = time.time()
        
        try:
            for i in range(self.num_samples):
                # Check time limit
                if time.time() - start_time > self.max_time:
                    self.logger.info(f"⏰ Time limit reached after {i} evaluations")
                    break
                
                # Sample random architecture
                architecture = self._sample_random_architecture()
                
                # Evaluate architecture
                fitness = self._evaluate_architecture(architecture)
                
                # Update best architecture
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_architecture = architecture
                    self.logger.info(f"🆕 New best architecture at sample {i + 1}: fitness={fitness:.4f}")
                
                # Record statistics
                self._record_statistics(architecture, fitness)
                
                # Progress logging
                if (i + 1) % 10 == 0:
                    self.logger.info(f"🔍 Evaluated {i + 1}/{self.num_samples} architectures, "
                                   f"best fitness: {self.best_fitness:.4f}")
            
            search_time = time.time() - start_time
            
            # Final statistics
            self._finalize_statistics(search_time)
            
            self.logger.info(f"✅ Random search completed in {search_time:.2f}s")
            self.logger.info(f"🎯 Best fitness: {self.best_fitness:.4f}")
            self.logger.info(f"📊 Total evaluations: {self.search_metrics['total_evaluations']}")
            self.logger.info(f"🔀 Unique architectures: {len(self.search_metrics['unique_architectures'])}")
            
            return self.best_architecture
            
        except Exception as e:
            self.logger.error(f"❌ Random search failed: {e}")
            raise
    
    def _sample_random_architecture(self) -> Architecture:
        """Sample a random architecture from the search space."""
        # For list-based search spaces
        if self.search_space.encoding_type == "list":
            # Random depth between 3 and max_depth
            depth = np.random.randint(3, self.search_space.constraints.get('max_depth', 8))
            
            # Random operation indices
            encoding = [np.random.randint(0, len(self.search_space.operations)) 
                       for _ in range(depth)]
            
            return Architecture(
                encoding=encoding,
                search_space=self.search_space,
                metadata={'source': 'random_search'}
            )
        
        elif self.search_space.encoding_type == "graph":
            # For graph-based search spaces
            return self.search_space.sample_random_architecture()
        
        else:
            # Fallback to search space's random sampling
            return self.search_space.sample_random_architecture()
    
    def _evaluate_architecture(self, architecture: Architecture) -> float:
        """
        Evaluate a single architecture.
        
        Args:
            architecture: Architecture to evaluate
            
        Returns:
            Fitness score
        """
        try:
            # Convert to model
            model = architecture.to_model(
                input_channels=self.config.model.input_channels,
                num_classes=self.config.model.num_classes,
                base_channels=self.config.model.base_channels
            )
            
            # Quick evaluation for speed
            metrics = self.evaluator.quick_evaluate(model)
            
            # Use accuracy as primary fitness
            fitness = metrics.get('accuracy', 0.0)
            
            self.search_metrics['total_evaluations'] += 1
            
            return fitness
            
        except Exception as e:
            self.logger.warning(f"⚠️ Architecture evaluation failed: {e}")
            return 0.0
    
    def _record_statistics(self, architecture: Architecture, fitness: float):
        """Record statistics for analysis."""
        # Record architecture and fitness
        self.evaluated_architectures.append({
            'architecture': architecture,
            'fitness': fitness,
            'evaluation_order': len(self.evaluated_architectures)
        })
        
        self.search_metrics['fitness_history'].append(fitness)
        
        # Track unique architectures
        arch_hash = hash(architecture)
        self.search_metrics['unique_architectures'].add(arch_hash)
        
        # Record complexity metrics
        complexity = architecture.get_complexity_metrics()
        self.search_metrics['complexity_stats'].append(complexity)
        
        # Store architecture info for analysis
        self.search_metrics['architectures_evaluated'].append({
            'encoding': architecture.encoding,
            'fitness': fitness,
            'complexity': complexity
        })
    
    def _finalize_statistics(self, search_time: float):
        """Finalize search statistics."""
        self.search_metrics.update({
            'total_search_time': search_time,
            'best_fitness': self.best_fitness,
            'avg_fitness': np.mean(self.search_metrics['fitness_history']),
            'std_fitness': np.std(self.search_metrics['fitness_history']),
            'unique_architecture_ratio': len(self.search_metrics['unique_architectures']) / max(1, self.search_metrics['total_evaluations']),
        })
        
        # Complexity statistics
        if self.search_metrics['complexity_stats']:
            complexity_keys = self.search_metrics['complexity_stats'][0].keys()
            for key in complexity_keys:
                values = [stats[key] for stats in self.search_metrics['complexity_stats']]
                self.search_metrics[f'avg_{key}'] = np.mean(values)
                self.search_metrics[f'std_{key}'] = np.std(values)
    
    def get_search_metrics(self) -> Dict[str, Any]:
        """Get comprehensive search metrics."""
        base_metrics = super().get_search_metrics()
        
        random_metrics = {
            'search_strategy': 'random',
            'sampling_method': 'uniform_random',
            'exploration_coverage': len(self.search_metrics['unique_architectures']) / max(1, self.search_metrics['total_evaluations']),
            'convergence_rate': self._compute_convergence_rate(),
        }
        
        return {**base_metrics, **random_metrics}
    
    def _compute_convergence_rate(self) -> float:
        """Compute how quickly the search finds good architectures."""
        if len(self.search_metrics['fitness_history']) < 2:
            return 0.0
        
        # Find when we reached 90% of final best fitness
        target_fitness = 0.9 * self.best_fitness
        fitness_history = self.search_metrics['fitness_history']
        
        for i, fitness in enumerate(fitness_history):
            if fitness >= target_fitness:
                return i / len(fitness_history)
        
        return 1.0  # Didn't reach 90% until the end
    
    def analyze_search_distribution(self) -> Dict[str, Any]:
        """
        Analyze the distribution of sampled architectures.
        
        Returns:
            Analysis of search distribution and coverage
        """
        if not self.evaluated_architectures:
            return {}
        
        # Analyze operation frequencies
        operation_counts = {}
        depth_counts = {}
        
        for arch_info in self.evaluated_architectures:
            architecture = arch_info['architecture']
            
            if architecture.encoding:
                # Count operation frequencies
                for op_idx in architecture.encoding:
                    op_name = self.search_space.operations[op_idx].name
                    operation_counts[op_name] = operation_counts.get(op_name, 0) + 1
                
                # Count depth distribution
                depth = len(architecture.encoding)
                depth_counts[depth] = depth_counts.get(depth, 0) + 1
        
        # Fitness distribution analysis
        fitnesses = [arch['fitness'] for arch in self.evaluated_architectures]
        
        analysis = {
            'operation_frequency': operation_counts,
            'depth_distribution': depth_counts,
            'fitness_distribution': {
                'mean': np.mean(fitnesses),
                'std': np.std(fitnesses),
                'min': np.min(fitnesses),
                'max': np.max(fitnesses),
                'percentiles': {
                    '25th': np.percentile(fitnesses, 25),
                    '50th': np.percentile(fitnesses, 50),
                    '75th': np.percentile(fitnesses, 75),
                    '90th': np.percentile(fitnesses, 90),
                }
            },
            'search_efficiency': {
                'unique_ratio': len(self.search_metrics['unique_architectures']) / max(1, self.search_metrics['total_evaluations']),
                'best_found_at': next((i for i, arch in enumerate(self.evaluated_architectures) if arch['fitness'] == self.best_fitness), -1),
                'improvement_rate': self._compute_improvement_rate(),
            }
        }
        
        return analysis
    
    def _compute_improvement_rate(self) -> float:
        """Compute rate of fitness improvement over time."""
        if len(self.search_metrics['fitness_history']) < 2:
            return 0.0
        
        improvements = 0
        current_best = -float('inf')
        
        for fitness in self.search_metrics['fitness_history']:
            if fitness > current_best:
                improvements += 1
                current_best = fitness
        
        return improvements / len(self.search_metrics['fitness_history']) 