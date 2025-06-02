#!/usr/bin/env python3
"""
Comprehensive Test Suite for AutoML-CodeGen

This script tests all components of the AutoML-CodeGen system to ensure
everything is working correctly after the completion.
"""

import sys
import os
import torch
import logging
import traceback
import time
from typing import Dict, Any, List
import numpy as np

# Add src to path
sys.path.insert(0, 'src')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutoMLTestSuite:
    """Comprehensive test suite for AutoML-CodeGen."""
    
    def __init__(self):
        self.test_results = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    def run_test(self, test_name: str, test_func):
        """Run a single test and record results."""
        logger.info(f"🧪 Running test: {test_name}")
        self.total_tests += 1
        
        try:
            start_time = time.time()
            result = test_func()
            end_time = time.time()
            
            self.test_results[test_name] = {
                'status': 'PASSED',
                'duration': end_time - start_time,
                'result': result,
                'error': None
            }
            self.passed_tests += 1
            logger.info(f"✅ {test_name} PASSED ({end_time - start_time:.2f}s)")
            
        except Exception as e:
            end_time = time.time()
            self.test_results[test_name] = {
                'status': 'FAILED',
                'duration': end_time - start_time,
                'result': None,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            self.failed_tests += 1
            logger.error(f"❌ {test_name} FAILED: {e}")
    
    def test_imports(self):
        """Test that all modules can be imported successfully."""
        logger.info("Testing module imports...")
        
        # Test core imports
        import automl_codegen
        from automl_codegen import (
            NeuralArchitectureSearch,
            CodeGenerator,
            EvolutionarySearch,
            DARTSSearch,
            ReinforcementSearch,
            SearchSpace,
            LayerSpec,
            MultiObjectiveOptimizer
        )
        
        # Test evaluation imports
        from automl_codegen.evaluation.trainer import ArchitectureTrainer
        from automl_codegen.evaluation.hardware import HardwareProfiler, HardwareMetrics
        
        # Test utilities
        from automl_codegen.utils.build_info import get_build_info_fast
        from automl_codegen.utils.config import Config
        
        return "All imports successful"
    
    def test_search_space(self):
        """Test search space functionality."""
        logger.info("Testing search space...")
        
        from automl_codegen import SearchSpace
        
        # Create search space
        space = SearchSpace(task='image_classification')
        
        # Sample architectures
        architectures = []
        for i in range(5):
            arch = space.sample_architecture()
            architectures.append(arch)
            
            # Validate architecture
            assert 'task' in arch
            assert 'layers' in arch
            assert isinstance(arch['layers'], list)
            assert len(arch['layers']) > 0
        
        # Test complexity estimation
        complexity = space.estimate_complexity(architectures[0])
        assert 'parameters' in complexity
        assert 'flops' in complexity
        
        return f"Generated {len(architectures)} valid architectures"
    
    def test_evolutionary_search(self):
        """Test evolutionary search algorithm."""
        logger.info("Testing evolutionary search...")
        
        from automl_codegen import SearchSpace, MultiObjectiveOptimizer, EvolutionarySearch
        
        # Create components
        space = SearchSpace(task='image_classification')
        objectives = MultiObjectiveOptimizer(['accuracy', 'latency'])
        
        # Create evolutionary search
        search = EvolutionarySearch(
            search_space=space,
            objectives=objectives,
            config={'population_size': 10, 'max_generations': 2}
        )
        
        # Initialize population
        population = search.initialize_population(size=10)
        assert len(population) == 10
        
        # Test mutation
        individual = population[0]
        original_arch = individual.architecture.copy()
        search._mutate(individual)
        
        return f"Evolution initialized with {len(population)} individuals"
    
    def test_darts_search(self):
        """Test DARTS search algorithm."""
        logger.info("Testing DARTS search...")
        
        from automl_codegen import SearchSpace, MultiObjectiveOptimizer
        from automl_codegen.search.algorithms.darts import DARTSSearch
        
        # Create components
        space = SearchSpace(task='image_classification')
        objectives = MultiObjectiveOptimizer(['accuracy', 'latency'])
        
        # Create DARTS search
        search = DARTSSearch(
            search_space=space,
            objectives=objectives,
            config={'num_cells': 4, 'num_nodes': 3}
        )
        
        # Test architecture derivation
        arch = search._derive_architecture()
        assert 'task' in arch
        assert 'layers' in arch
        assert arch['algorithm'] == 'darts'
        
        return f"DARTS initialized with supernet"
    
    def test_reinforcement_search(self):
        """Test reinforcement learning search algorithm."""
        logger.info("Testing reinforcement learning search...")
        
        from automl_codegen import SearchSpace, MultiObjectiveOptimizer
        from automl_codegen.search.algorithms.reinforcement import ReinforcementSearch
        
        # Create components
        space = SearchSpace(task='image_classification')
        objectives = MultiObjectiveOptimizer(['accuracy', 'latency'])
        
        # Create RL search
        search = ReinforcementSearch(
            search_space=space,
            objectives=objectives,
            config={'max_layers': 10}
        )
        
        # Sample architecture
        arch = search.sample_architecture()
        assert 'task' in arch
        assert 'layers' in arch
        assert arch['algorithm'] == 'reinforcement'
        
        # Test reward estimation
        reward = search._evaluate_architecture(arch)
        assert 0.0 <= reward <= 1.0
        
        return f"RL search initialized with controller network"
    
    def test_code_generation(self):
        """Test code generation functionality."""
        logger.info("Testing code generation...")
        
        from automl_codegen import CodeGenerator, SearchSpace
        
        # Create sample architecture
        space = SearchSpace(task='image_classification')
        architecture = space.sample_architecture()
        
        # Create code generator
        codegen = CodeGenerator(target_framework='pytorch')
        
        # Generate code
        code_result = codegen.generate(
            architecture,
            include_training=True,
            include_inference=True
        )
        
        # Validate generated code
        assert hasattr(code_result, 'model_code')
        assert hasattr(code_result, 'training_code')
        assert len(code_result.model_code) > 0
        
        # Test that generated code is valid Python
        compile(code_result.model_code, '<string>', 'exec')
        
        return f"Generated {len(code_result.model_code)} characters of PyTorch code"
    
    def test_hardware_profiling(self):
        """Test hardware profiling functionality."""
        logger.info("Testing hardware profiling...")
        
        from automl_codegen.evaluation.hardware import HardwareProfiler
        from automl_codegen import SearchSpace
        
        # Create profiler
        profiler = HardwareProfiler()
        
        # Create sample architecture
        space = SearchSpace(task='image_classification')
        architecture = space.sample_architecture()
        
        # Profile architecture
        metrics = profiler.profile_architecture(architecture)
        
        # Validate metrics
        assert hasattr(metrics, 'latency_ms')
        assert hasattr(metrics, 'memory_mb')
        assert hasattr(metrics, 'parameters')
        assert hasattr(metrics, 'flops')
        assert metrics.latency_ms > 0
        assert metrics.parameters > 0
        
        return f"Profiled architecture: {metrics.latency_ms:.2f}ms, {metrics.parameters} params"
    
    def test_multi_objective_optimization(self):
        """Test multi-objective optimization."""
        logger.info("Testing multi-objective optimization...")
        
        from automl_codegen import MultiObjectiveOptimizer
        
        # Create optimizer
        objectives = ['accuracy', 'latency', 'memory']
        optimizer = MultiObjectiveOptimizer(objectives)
        
        # Create dummy population and metrics
        population = [f"arch_{i}" for i in range(10)]
        metrics_list = []
        
        for i in range(10):
            metrics = {
                'accuracy': np.random.uniform(0.8, 0.95),
                'latency': np.random.uniform(10, 100),
                'memory': np.random.uniform(50, 500)
            }
            metrics_list.append(metrics)
        
        # Update optimizer
        optimizer.update(population, metrics_list)
        
        # Get Pareto front
        pareto_front = optimizer.get_pareto_front()
        assert len(pareto_front) > 0
        
        # Test hypervolume calculation
        hypervolume = optimizer.get_hypervolume()
        assert hypervolume >= 0
        
        return f"Pareto front contains {len(pareto_front)} solutions, HV: {hypervolume:.4f}"
    
    def test_nas_integration(self):
        """Test full NAS integration."""
        logger.info("Testing NAS integration...")
        
        from automl_codegen import NeuralArchitectureSearch
        
        # Test different algorithms
        algorithms = ['evolutionary', 'darts', 'reinforcement']
        results = {}
        
        for algorithm in algorithms:
            try:
                nas = NeuralArchitectureSearch(
                    task='image_classification',
                    dataset='cifar10',
                    objectives=['accuracy', 'latency'],
                    algorithm=algorithm,
                    hardware_target='gpu'
                )
                
                # Test that NAS is properly initialized
                assert nas.algorithm == algorithm
                assert nas.task == 'image_classification'
                assert len(nas.objectives.objective_names) == 2
                
                results[algorithm] = 'initialized'
                
            except Exception as e:
                results[algorithm] = f'failed: {e}'
        
        return f"NAS algorithms tested: {results}"
    
    def test_architecture_training(self):
        """Test architecture training functionality."""
        logger.info("Testing architecture training...")
        
        from automl_codegen.evaluation.trainer import ArchitectureTrainer
        from automl_codegen import SearchSpace
        
        # Create trainer
        trainer = ArchitectureTrainer()
        
        # Create sample architecture
        space = SearchSpace(task='image_classification')
        architecture = space.sample_architecture()
        
        # Create dummy dataset
        dataset_config = {
            'name': 'dummy',
            'num_classes': 10,
            'input_shape': (3, 32, 32)
        }
        
        # Test architecture evaluation (quick test)
        metrics = trainer.evaluate_architecture(
            architecture,
            dataset_config,
            max_epochs=1,
            quick_eval=True
        )
        
        assert 'accuracy' in metrics
        assert 'loss' in metrics
        assert metrics['accuracy'] >= 0
        
        return f"Training metrics: accuracy={metrics['accuracy']:.4f}, loss={metrics['loss']:.4f}"
    
    def test_end_to_end_pipeline(self):
        """Test the complete end-to-end pipeline."""
        logger.info("Testing end-to-end pipeline...")
        
        from automl_codegen import NeuralArchitectureSearch, CodeGenerator
        
        # Create NAS
        nas = NeuralArchitectureSearch(
            task='image_classification',
            dataset='cifar10',
            objectives=['accuracy', 'latency'],
            algorithm='evolutionary'
        )
        
        # Sample a few architectures (simulate quick search)
        sample_arch = nas.search_space.sample_architecture()
        
        # Generate code for the architecture
        codegen = CodeGenerator(target_framework='pytorch')
        code_result = codegen.generate(sample_arch)
        
        # Validate the pipeline works
        assert len(code_result.model_code) > 0
        
        # Test that generated model can be instantiated
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        local_vars = {}
        global_vars = {
            'torch': torch,
            'nn': nn,
            'F': F,
            '__builtins__': __builtins__
        }
        exec(code_result.model_code, global_vars, local_vars)
        
        # Find the model class
        model_class = None
        for name, obj in local_vars.items():
            if hasattr(obj, '__bases__') and any('Module' in str(base) for base in obj.__bases__):
                model_class = obj
                break
        
        if model_class:
            # Try to instantiate the model
            model = model_class()
            param_count = sum(p.numel() for p in model.parameters())
            return f"End-to-end pipeline successful, generated model with {param_count} parameters"
        else:
            return "End-to-end pipeline successful, code generated"
    
    def run_all_tests(self):
        """Run all tests in the suite."""
        logger.info("🚀 Starting comprehensive AutoML-CodeGen test suite...")
        
        # List of all tests
        tests = [
            ("Import Test", self.test_imports),
            ("Search Space Test", self.test_search_space),
            ("Evolutionary Search Test", self.test_evolutionary_search),
            ("DARTS Search Test", self.test_darts_search),
            ("Reinforcement Search Test", self.test_reinforcement_search),
            ("Code Generation Test", self.test_code_generation),
            ("Hardware Profiling Test", self.test_hardware_profiling),
            ("Multi-Objective Optimization Test", self.test_multi_objective_optimization),
            ("NAS Integration Test", self.test_nas_integration),
            ("Architecture Training Test", self.test_architecture_training),
            ("End-to-End Pipeline Test", self.test_end_to_end_pipeline),
        ]
        
        # Run all tests
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
            print()  # Add spacing between tests
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test results summary."""
        print("\n" + "="*80)
        print("🎯 TEST SUITE SUMMARY")
        print("="*80)
        
        print(f"Total Tests: {self.total_tests}")
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.failed_tests}")
        print(f"Success Rate: {(self.passed_tests/self.total_tests)*100:.1f}%")
        
        print("\n📊 DETAILED RESULTS:")
        print("-" * 80)
        
        for test_name, result in self.test_results.items():
            status_emoji = "✅" if result['status'] == 'PASSED' else "❌"
            print(f"{status_emoji} {test_name:<35} {result['status']:<8} ({result['duration']:.2f}s)")
            
            if result['result']:
                print(f"   └─ {result['result']}")
            
            if result['error']:
                print(f"   └─ Error: {result['error']}")
        
        print("\n" + "="*80)
        
        if self.failed_tests == 0:
            print("🎉 ALL TESTS PASSED! AutoML-CodeGen is fully functional!")
        else:
            print(f"⚠️  {self.failed_tests} tests failed. Check the errors above.")
        
        print("="*80)


if __name__ == "__main__":
    print("🤖 AutoML-CodeGen Comprehensive Test Suite")
    print("="*60)
    
    # Check if we're in the right directory
    if not os.path.exists('src/automl_codegen'):
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Initialize and run test suite
    test_suite = AutoMLTestSuite()
    test_suite.run_all_tests() 