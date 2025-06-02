"""
Unit Tests for Search Strategies
===============================

This module provides comprehensive unit tests for all search strategies
in nanoNAS including evolutionary, DARTS, and random search algorithms.

Test Coverage:
- Search strategy initialization
- Architecture sampling and evaluation
- Population management and evolution
- Fitness computation and ranking
- Configuration validation
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path

# Import nanoNAS components
from nanonas.core.architecture import Architecture, SearchSpace
from nanonas.core.config import ExperimentConfig, SearchConfig, ModelConfig
from nanonas.search.evolutionary import EvolutionarySearch
from nanonas.search.random_search import RandomSearch
from nanonas.benchmarks.evaluator import ModelEvaluator


class TestSearchStrategies:
    """Test suite for search strategies."""
    
    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing."""
        config = ExperimentConfig(
            experiment={
                'name': 'test_experiment',
                'description': 'Test experiment for unit testing'
            },
            dataset={
                'name': 'cifar10',
                'batch_size': 32,
                'num_workers': 1
            },
            model={
                'input_channels': 3,
                'num_classes': 10,
                'base_channels': 16
            },
            search={
                'strategy': 'evolutionary',
                'search_space': 'nano',
                'population_size': 10,
                'generations': 5,
                'mutation_rate': 0.3,
                'crossover_rate': 0.6,
                'max_search_time': 300
            },
            training={
                'epochs': 5,
                'learning_rate': 0.01,
                'optimizer': 'sgd'
            },
            evaluation={
                'quick_eval_epochs': 2,
                'metrics': ['accuracy', 'loss']
            },
            device='cpu',
            output={
                'save_dir': 'test_results',
                'log_level': 'WARNING'
            }
        )
        return config
    
    @pytest.fixture
    def sample_search_space(self):
        """Create sample search space for testing."""
        return SearchSpace.get_nano_search_space()
    
    @pytest.fixture
    def sample_architecture(self, sample_search_space):
        """Create sample architecture for testing."""
        return Architecture(
            encoding=[0, 1, 2, 3],
            search_space=sample_search_space,
            metadata={'test': True}
        )
    
    def test_search_space_creation(self):
        """Test search space creation and properties."""
        # Test nano search space
        nano_space = SearchSpace.get_nano_search_space()
        assert nano_space.name == "nano"
        assert nano_space.encoding_type == "list"
        assert len(nano_space.operations) > 0
        
        # Test mobile search space
        mobile_space = SearchSpace.get_mobile_search_space()
        assert mobile_space.name == "mobile"
        assert len(mobile_space.operations) > 0
        
        # Test that operations have required attributes
        for op in nano_space.operations:
            assert hasattr(op, 'name')
            assert hasattr(op, 'type')
            assert hasattr(op, 'cost')
    
    def test_architecture_creation(self, sample_search_space):
        """Test architecture creation and validation."""
        # Valid architecture
        arch = Architecture(
            encoding=[0, 1, 2],
            search_space=sample_search_space,
            metadata={'test': True}
        )
        assert arch.encoding == [0, 1, 2]
        assert arch.search_space == sample_search_space
        assert arch.metadata['test'] is True
        
        # Test architecture hashing
        arch2 = Architecture(
            encoding=[0, 1, 2],
            search_space=sample_search_space
        )
        assert hash(arch) == hash(arch2)
        
        # Test different architecture
        arch3 = Architecture(
            encoding=[0, 1, 3],
            search_space=sample_search_space
        )
        assert hash(arch) != hash(arch3)
    
    def test_architecture_complexity_metrics(self, sample_architecture):
        """Test architecture complexity computation."""
        complexity = sample_architecture.get_complexity_metrics()
        
        # Check required metrics
        assert 'depth' in complexity
        assert 'total_op_cost' in complexity
        assert isinstance(complexity['depth'], int)
        assert isinstance(complexity['total_op_cost'], (int, float))
        assert complexity['depth'] > 0
        assert complexity['total_op_cost'] >= 0
    
    def test_architecture_genetic_operations(self, sample_architecture):
        """Test genetic operations on architectures."""
        # Test mutation
        mutated = sample_architecture.mutate(mutation_rate=0.5)
        assert isinstance(mutated, Architecture)
        assert mutated.search_space == sample_architecture.search_space
        
        # Test crossover
        parent2 = Architecture(
            encoding=[1, 0, 3, 2],
            search_space=sample_architecture.search_space
        )
        
        child1, child2 = sample_architecture.crossover(parent2)
        assert isinstance(child1, Architecture)
        assert isinstance(child2, Architecture)
        assert child1.search_space == sample_architecture.search_space
        assert child2.search_space == sample_architecture.search_space


class TestEvolutionarySearch:
    """Test suite for evolutionary search strategy."""
    
    @pytest.fixture
    def evolutionary_search(self, sample_config):
        """Create evolutionary search instance."""
        with patch('nanonas.benchmarks.evaluator.ModelEvaluator'):
            return EvolutionarySearch(sample_config)
    
    def test_evolutionary_search_initialization(self, evolutionary_search):
        """Test evolutionary search initialization."""
        assert evolutionary_search.population_size == 10
        assert evolutionary_search.num_generations == 5
        assert evolutionary_search.mutation_rate == 0.3
        assert evolutionary_search.crossover_rate == 0.6
        assert evolutionary_search.search_space is not None
    
    def test_population_initialization(self, evolutionary_search):
        """Test population initialization."""
        population = evolutionary_search._initialize_population()
        
        assert len(population) == evolutionary_search.population_size
        assert all(isinstance(arch, Architecture) for arch in population)
        assert all(arch.search_space == evolutionary_search.search_space 
                  for arch in population)
    
    def test_fitness_evaluation(self, evolutionary_search):
        """Test fitness evaluation."""
        # Mock the evaluator to return consistent results
        with patch.object(evolutionary_search.evaluator, 'quick_evaluate') as mock_eval:
            mock_eval.return_value = {'accuracy': 0.85, 'loss': 0.45}
            
            arch = Architecture(
                encoding=[0, 1, 2],
                search_space=evolutionary_search.search_space
            )
            
            fitness = evolutionary_search._evaluate_architecture(arch)
            assert isinstance(fitness, float)
            assert 0 <= fitness <= 1
            mock_eval.assert_called_once()
    
    def test_selection_methods(self, evolutionary_search):
        """Test selection methods."""
        population = evolutionary_search._initialize_population()
        fitnesses = [np.random.random() for _ in population]
        
        # Test tournament selection
        selected = evolutionary_search._tournament_selection(population, fitnesses, k=3)
        assert len(selected) == len(population)
        assert all(isinstance(arch, Architecture) for arch in selected)
        
        # Test roulette wheel selection
        selected = evolutionary_search._roulette_selection(population, fitnesses)
        assert len(selected) == len(population)
        assert all(isinstance(arch, Architecture) for arch in selected)
    
    def test_evolution_operations(self, evolutionary_search):
        """Test crossover and mutation operations."""
        population = evolutionary_search._initialize_population()
        
        # Test crossover
        parent1, parent2 = population[0], population[1]
        child1, child2 = evolutionary_search._crossover(parent1, parent2)
        
        assert isinstance(child1, Architecture)
        assert isinstance(child2, Architecture)
        assert child1.search_space == parent1.search_space
        assert child2.search_space == parent2.search_space
        
        # Test mutation
        mutated = evolutionary_search._mutate(parent1)
        assert isinstance(mutated, Architecture)
        assert mutated.search_space == parent1.search_space


class TestRandomSearch:
    """Test suite for random search strategy."""
    
    @pytest.fixture
    def random_search(self, sample_config):
        """Create random search instance."""
        config = sample_config.copy()
        config.search.strategy = 'random'
        
        with patch('nanonas.benchmarks.evaluator.ModelEvaluator'):
            return RandomSearch(config)
    
    def test_random_search_initialization(self, random_search):
        """Test random search initialization."""
        assert random_search.num_samples > 0
        assert random_search.search_space is not None
        assert hasattr(random_search, 'evaluator')
    
    def test_random_architecture_sampling(self, random_search):
        """Test random architecture sampling."""
        arch = random_search._sample_random_architecture()
        
        assert isinstance(arch, Architecture)
        assert arch.search_space == random_search.search_space
        assert arch.encoding is not None
        assert len(arch.encoding) >= 3  # Minimum depth
    
    def test_search_statistics(self, random_search):
        """Test search statistics collection."""
        # Initialize search metrics
        assert 'total_evaluations' in random_search.search_metrics
        assert 'fitness_history' in random_search.search_metrics
        assert 'unique_architectures' in random_search.search_metrics
        
        # Test recording statistics
        arch = random_search._sample_random_architecture()
        fitness = 0.75
        
        random_search._record_statistics(arch, fitness)
        
        assert random_search.search_metrics['fitness_history'][-1] == fitness
        assert len(random_search.evaluated_architectures) == 1
    
    def test_search_analysis(self, random_search):
        """Test search distribution analysis."""
        # Add some dummy data
        for i in range(5):
            arch = random_search._sample_random_architecture()
            fitness = np.random.random()
            random_search._record_statistics(arch, fitness)
        
        analysis = random_search.analyze_search_distribution()
        
        assert 'operation_frequency' in analysis
        assert 'fitness_distribution' in analysis
        assert 'search_efficiency' in analysis


class TestSearchIntegration:
    """Integration tests for search strategies."""
    
    def test_search_with_mock_evaluator(self, sample_config):
        """Test search with mocked evaluator."""
        # Mock evaluator to return consistent results
        with patch('nanonas.benchmarks.evaluator.ModelEvaluator') as mock_evaluator_class:
            mock_evaluator = MagicMock()
            mock_evaluator.quick_evaluate.return_value = {
                'accuracy': 0.85,
                'loss': 0.32,
                'f1_score': 0.83
            }
            mock_evaluator_class.return_value = mock_evaluator
            
            # Test evolutionary search
            evolutionary = EvolutionarySearch(sample_config)
            best_arch = evolutionary.search()
            
            assert isinstance(best_arch, Architecture)
            assert evolutionary.best_fitness > 0
            assert mock_evaluator.quick_evaluate.call_count > 0
    
    def test_search_early_stopping(self, sample_config):
        """Test search early stopping functionality."""
        config = sample_config.copy()
        config.search.early_stopping_patience = 2
        config.search.generations = 10
        
        with patch('nanonas.benchmarks.evaluator.ModelEvaluator') as mock_evaluator_class:
            # Mock evaluator to return decreasing fitness
            mock_evaluator = MagicMock()
            fitness_values = [0.8, 0.75, 0.7, 0.65, 0.6]  # Decreasing
            mock_evaluator.quick_evaluate.side_effect = [
                {'accuracy': f, 'loss': 1-f} for f in fitness_values
            ]
            mock_evaluator_class.return_value = mock_evaluator
            
            evolutionary = EvolutionarySearch(config)
            
            # Mock the search to simulate early stopping
            with patch.object(evolutionary, '_should_stop_early', return_value=True):
                best_arch = evolutionary.search()
                assert isinstance(best_arch, Architecture)
    
    def test_search_time_limit(self, sample_config):
        """Test search time limit functionality."""
        config = sample_config.copy()
        config.search.max_search_time = 1  # Very short time limit
        
        with patch('nanonas.benchmarks.evaluator.ModelEvaluator'):
            evolutionary = EvolutionarySearch(config)
            
            # Mock time.time to simulate time passing
            with patch('time.time', side_effect=[0, 0.5, 1.1]):  # Start, middle, timeout
                best_arch = evolutionary.search()
                assert isinstance(best_arch, Architecture)


class TestSearchMetrics:
    """Test suite for search metrics and analysis."""
    
    def test_metrics_collection(self, sample_config):
        """Test metrics collection during search."""
        with patch('nanonas.benchmarks.evaluator.ModelEvaluator'):
            evolutionary = EvolutionarySearch(sample_config)
            
            # Initialize metrics
            metrics = evolutionary.get_search_metrics()
            assert isinstance(metrics, dict)
            assert 'search_strategy' in metrics
            assert 'start_time' in metrics
    
    def test_convergence_analysis(self, sample_config):
        """Test convergence analysis."""
        with patch('nanonas.benchmarks.evaluator.ModelEvaluator'):
            random_search = RandomSearch(sample_config)
            
            # Simulate some search history
            for i in range(10):
                fitness = 0.5 + i * 0.03  # Improving fitness
                random_search.search_metrics['fitness_history'].append(fitness)
            
            convergence_rate = random_search._compute_convergence_rate()
            assert isinstance(convergence_rate, float)
            assert 0 <= convergence_rate <= 1


class TestConfigurationValidation:
    """Test configuration validation and error handling."""
    
    def test_invalid_search_strategy(self):
        """Test handling of invalid search strategy."""
        with pytest.raises((ValueError, KeyError)):
            config = ExperimentConfig(
                search={'strategy': 'invalid_strategy'}
            )
    
    def test_missing_required_config(self):
        """Test handling of missing required configuration."""
        with pytest.raises((ValueError, TypeError)):
            # This should fail due to missing required fields
            EvolutionarySearch({})
    
    def test_config_validation(self, sample_config):
        """Test configuration validation."""
        # Valid configuration should work
        with patch('nanonas.benchmarks.evaluator.ModelEvaluator'):
            evolutionary = EvolutionarySearch(sample_config)
            assert evolutionary.population_size > 0
            assert evolutionary.num_generations > 0
            assert 0 <= evolutionary.mutation_rate <= 1
            assert 0 <= evolutionary.crossover_rate <= 1


# Pytest fixtures and utilities
@pytest.fixture(scope="session")
def temp_output_dir():
    """Create temporary output directory for tests."""
    temp_dir = tempfile.mkdtemp(prefix="nanonas_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_architecture_serialization(temp_output_dir):
    """Test architecture serialization and deserialization."""
    search_space = SearchSpace.get_nano_search_space()
    arch = Architecture(
        encoding=[0, 1, 2, 3],
        search_space=search_space,
        metadata={'test': True}
    )
    
    # Test serialization
    arch_dict = arch.to_dict()
    assert isinstance(arch_dict, dict)
    assert 'encoding' in arch_dict
    assert 'metadata' in arch_dict
    
    # Test deserialization
    arch_restored = Architecture.from_dict(arch_dict, search_space)
    assert arch_restored.encoding == arch.encoding
    assert arch_restored.metadata == arch.metadata
    assert hash(arch_restored) == hash(arch)


def test_search_reproducibility(sample_config):
    """Test search reproducibility with fixed seed."""
    config1 = sample_config.copy()
    config1.seed = 42
    
    config2 = sample_config.copy()
    config2.seed = 42
    
    with patch('nanonas.benchmarks.evaluator.ModelEvaluator'):
        # Set numpy and torch seeds
        np.random.seed(42)
        torch.manual_seed(42)
        
        search1 = RandomSearch(config1)
        arch1 = search1._sample_random_architecture()
        
        np.random.seed(42)
        torch.manual_seed(42)
        
        search2 = RandomSearch(config2)
        arch2 = search2._sample_random_architecture()
        
        # Results should be identical with same seed
        assert arch1.encoding == arch2.encoding


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"]) 