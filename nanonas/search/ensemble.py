"""
Ensemble Neural Architecture Search
===================================

This module implements an ensemble approach to Neural Architecture Search that
combines multiple search strategies to leverage their complementary strengths
and achieve more robust architecture discovery.

Key Features:
- Multi-strategy ensemble combining evolutionary, gradient-based, and RL methods
- Adaptive strategy weighting based on performance
- Cross-validation for robust architecture evaluation
- Consensus mechanisms for final architecture selection
- Parallel search execution for efficiency
- Advanced architecture ranking and selection

Algorithm Overview:
1. Initialize multiple search strategies in parallel
2. Execute search with different random seeds
3. Collect candidate architectures from all strategies  
4. Rank architectures using ensemble evaluation
5. Select final architecture using consensus mechanisms

Example Usage:
    >>> from nanonas.search.ensemble import EnsembleSearch
    >>> 
    >>> searcher = EnsembleSearch(
    ...     strategies=['evolutionary', 'darts', 'pc_darts'],
    ...     strategy_weights={'evolutionary': 0.4, 'darts': 0.3, 'pc_darts': 0.3},
    ...     consensus_method='weighted_voting'
    ... )
    >>> best_architecture = searcher.search()
"""

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import copy
import time
import concurrent.futures
from collections import defaultdict, Counter
import json

from ..core.base import BaseSearchStrategy
from ..core.architecture import Architecture, SearchSpace
from ..core.config import SearchConfig
from .evolutionary import EvolutionarySearch
from .darts import DARTSSearch
from .pc_darts import PCDARTSSearch
from .bayesian_optimization import BayesianOptimizationSearch
from ..utils.metrics import accuracy, compute_model_stats
from ..benchmarks.evaluator import ModelEvaluator
from ..visualization.search_viz import EnsembleTracker


class EnsembleSearch(BaseSearchStrategy):
    """
    Ensemble Neural Architecture Search combining multiple search strategies.
    
    This approach leverages the strengths of different search methods by running
    them in parallel and using consensus mechanisms to select the best architecture.
    The ensemble provides improved robustness and exploration compared to single
    strategy approaches.
    
    Args:
        strategies: List of strategy names to include in ensemble
        strategy_weights: Weights for each strategy in final consensus
        consensus_method: Method for combining strategy results
        num_seeds: Number of random seeds per strategy
        parallel_execution: Whether to run strategies in parallel
        cross_validation_folds: Number of CV folds for architecture evaluation
        architecture_pool_size: Maximum size of candidate architecture pool
        final_evaluation_epochs: Epochs for final architecture evaluation
        diversity_weight: Weight for architecture diversity in selection
        device: Computing device
        output_dir: Directory for saving results
        verbose: Enable verbose logging
    """
    
    def __init__(
        self,
        strategies: List[str] = ["evolutionary", "darts", "pc_darts"],
        strategy_weights: Optional[Dict[str, float]] = None,
        consensus_method: str = "weighted_voting",
        num_seeds: int = 3,
        parallel_execution: bool = True,
        cross_validation_folds: int = 3,
        architecture_pool_size: int = 50,
        final_evaluation_epochs: int = 100,
        diversity_weight: float = 0.2,
        device: str = "auto",
        output_dir: str = "./ensemble_results",
        verbose: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Core configuration
        self.strategies = strategies
        self.strategy_weights = strategy_weights or self._get_default_weights()
        self.consensus_method = consensus_method
        self.num_seeds = num_seeds
        self.parallel_execution = parallel_execution
        self.cross_validation_folds = cross_validation_folds
        self.architecture_pool_size = architecture_pool_size
        self.final_evaluation_epochs = final_evaluation_epochs
        self.diversity_weight = diversity_weight
        
        # Setup
        self.device = self._setup_device(device)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        
        # Search components
        self.search_instances = {}
        self.candidate_architectures = []
        self.strategy_results = defaultdict(list)
        self.final_rankings = []
        
        # Tracking and evaluation
        self.ensemble_tracker = None
        self.evaluator = None
        
        self._setup_logging()
        self._validate_configuration()
    
    def setup(self, dataset: DataLoader, val_dataset: Optional[DataLoader] = None) -> None:
        """
        Setup ensemble search with datasets and initialize all search strategies.
        
        Args:
            dataset: Training dataset loader
            val_dataset: Validation dataset loader (optional)
        """
        self.logger.info("🔧 Setting up Ensemble Search...")
        
        # Store datasets
        self.train_loader = dataset
        self.val_loader = val_dataset or dataset
        
        # Initialize tracking
        self.ensemble_tracker = EnsembleTracker(
            strategies=self.strategies,
            output_dir=self.output_dir
        )
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator(
            device=self.device,
            cross_validation_folds=self.cross_validation_folds
        )
        
        # Initialize search strategies
        self._initialize_search_strategies()
        
        self.logger.info(f"✅ Ensemble search setup complete")
        self.logger.info(f"🔍 Strategies: {', '.join(self.strategies)}")
        self.logger.info(f"⚖️ Weights: {self.strategy_weights}")
        self.logger.info(f"🎯 Consensus method: {self.consensus_method}")
    
    def search(self) -> Architecture:
        """
        Perform ensemble architecture search.
        
        Returns:
            Best architecture from ensemble consensus
        """
        self.logger.info("🚀 Starting Ensemble Search...")
        start_time = time.time()
        
        # Phase 1: Execute multiple search strategies
        self.logger.info("📊 Phase 1: Multi-strategy search execution")
        strategy_results = self._execute_parallel_search()
        
        # Phase 2: Collect and pool candidate architectures
        self.logger.info("🏊 Phase 2: Architecture candidate pooling")
        candidate_pool = self._collect_candidate_architectures(strategy_results)
        
        # Phase 3: Comprehensive evaluation of candidates
        self.logger.info("📈 Phase 3: Cross-validation evaluation")
        evaluation_results = self._evaluate_architecture_pool(candidate_pool)
        
        # Phase 4: Consensus-based final selection
        self.logger.info("🎯 Phase 4: Consensus-based selection")
        final_architecture = self._consensus_selection(evaluation_results)
        
        # Phase 5: Final validation and reporting
        self.logger.info("✅ Phase 5: Final validation")
        final_metrics = self._final_validation(final_architecture)
        
        search_time = time.time() - start_time
        
        # Save comprehensive results
        self._save_ensemble_results(
            final_architecture, 
            strategy_results, 
            evaluation_results,
            final_metrics,
            search_time
        )
        
        self.logger.info(f"🏆 Ensemble search completed in {search_time:.2f}s")
        self.logger.info(f"🎖️ Final architecture: {final_architecture}")
        
        return final_architecture
    
    def _execute_parallel_search(self) -> Dict[str, List[Dict[str, Any]]]:
        """Execute multiple search strategies in parallel."""
        strategy_results = defaultdict(list)
        
        if self.parallel_execution and len(self.strategies) > 1:
            # Parallel execution
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=min(len(self.strategies), mp.cpu_count())
            ) as executor:
                
                futures = {}
                for strategy_name in self.strategies:
                    for seed in range(self.num_seeds):
                        future = executor.submit(
                            self._run_single_strategy,
                            strategy_name,
                            seed
                        )
                        futures[future] = (strategy_name, seed)
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(futures):
                    strategy_name, seed = futures[future]
                    try:
                        result = future.result()
                        strategy_results[strategy_name].append(result)
                        self.logger.info(
                            f"✅ {strategy_name} (seed {seed}) completed: "
                            f"acc={result['accuracy']:.3f}"
                        )
                    except Exception as e:
                        self.logger.error(f"❌ {strategy_name} (seed {seed}) failed: {e}")
        else:
            # Sequential execution
            for strategy_name in self.strategies:
                for seed in range(self.num_seeds):
                    try:
                        result = self._run_single_strategy(strategy_name, seed)
                        strategy_results[strategy_name].append(result)
                        self.logger.info(
                            f"✅ {strategy_name} (seed {seed}) completed: "
                            f"acc={result['accuracy']:.3f}"
                        )
                    except Exception as e:
                        self.logger.error(f"❌ {strategy_name} (seed {seed}) failed: {e}")
        
        return strategy_results
    
    def _run_single_strategy(self, strategy_name: str, seed: int) -> Dict[str, Any]:
        """Run a single search strategy with specified seed."""
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Get strategy instance
        strategy = self._get_strategy_instance(strategy_name)
        
        # Setup strategy
        strategy.setup(self.train_loader, self.val_loader)
        
        # Execute search
        architecture = strategy.search()
        
        # Evaluate architecture
        model = architecture.to_model()
        accuracy_score = self._quick_evaluate(model)
        
        return {
            'strategy': strategy_name,
            'seed': seed,
            'architecture': architecture,
            'accuracy': accuracy_score,
            'search_metrics': strategy.get_search_metrics() if hasattr(strategy, 'get_search_metrics') else {}
        }
    
    def _collect_candidate_architectures(
        self, 
        strategy_results: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Collect and deduplicate candidate architectures from all strategies."""
        candidates = []
        architecture_hashes = set()
        
        for strategy_name, results in strategy_results.items():
            strategy_weight = self.strategy_weights.get(strategy_name, 1.0)
            
            for result in results:
                architecture = result['architecture']
                arch_hash = architecture.get_hash()
                
                # Avoid duplicates
                if arch_hash not in architecture_hashes:
                    architecture_hashes.add(arch_hash)
                    
                    candidates.append({
                        'architecture': architecture,
                        'strategy': strategy_name,
                        'seed': result['seed'],
                        'initial_accuracy': result['accuracy'],
                        'strategy_weight': strategy_weight,
                        'search_metrics': result['search_metrics']
                    })
        
        # Sort by initial accuracy and take top candidates
        candidates.sort(key=lambda x: x['initial_accuracy'], reverse=True)
        
        if len(candidates) > self.architecture_pool_size:
            candidates = candidates[:self.architecture_pool_size]
        
        self.logger.info(f"🏊 Collected {len(candidates)} unique candidate architectures")
        
        return candidates
    
    def _evaluate_architecture_pool(
        self, 
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Comprehensively evaluate all candidate architectures."""
        evaluation_results = []
        
        for i, candidate in enumerate(candidates):
            self.logger.info(f"📊 Evaluating candidate {i+1}/{len(candidates)}")
            
            architecture = candidate['architecture']
            model = architecture.to_model()
            
            # Cross-validation evaluation
            cv_results = self.evaluator.cross_validate(
                model, 
                self.train_loader,
                epochs=self.final_evaluation_epochs // 2
            )
            
            # Model statistics
            model_stats = compute_model_stats(model)
            
            # Diversity score
            diversity_score = self._compute_diversity_score(
                architecture, 
                [c['architecture'] for c in candidates]
            )
            
            evaluation_results.append({
                **candidate,
                'cv_accuracy_mean': cv_results['accuracy_mean'],
                'cv_accuracy_std': cv_results['accuracy_std'],
                'cv_scores': cv_results['fold_scores'],
                'model_params': model_stats['parameters'],
                'model_flops': model_stats['flops'],
                'model_memory': model_stats['memory'],
                'diversity_score': diversity_score,
                'evaluation_time': cv_results['evaluation_time']
            })
        
        return evaluation_results
    
    def _consensus_selection(
        self, 
        evaluation_results: List[Dict[str, Any]]
    ) -> Architecture:
        """Select final architecture using consensus method."""
        
        if self.consensus_method == "weighted_voting":
            return self._weighted_voting_consensus(evaluation_results)
        elif self.consensus_method == "pareto_optimal":
            return self._pareto_optimal_consensus(evaluation_results)
        elif self.consensus_method == "ensemble_ranking":
            return self._ensemble_ranking_consensus(evaluation_results)
        else:
            raise ValueError(f"Unknown consensus method: {self.consensus_method}")
    
    def _weighted_voting_consensus(
        self, 
        evaluation_results: List[Dict[str, Any]]
    ) -> Architecture:
        """Select architecture using weighted voting based on multiple criteria."""
        
        # Normalize metrics
        accuracies = [r['cv_accuracy_mean'] for r in evaluation_results]
        diversities = [r['diversity_score'] for r in evaluation_results]
        
        acc_min, acc_max = min(accuracies), max(accuracies)
        div_min, div_max = min(diversities), max(diversities)
        
        # Compute composite scores
        for result in evaluation_results:
            # Normalize accuracy (higher is better)
            norm_acc = (result['cv_accuracy_mean'] - acc_min) / (acc_max - acc_min + 1e-8)
            
            # Normalize diversity (higher is better)
            norm_div = (result['diversity_score'] - div_min) / (div_max - div_min + 1e-8)
            
            # Normalize efficiency (lower parameters is better)
            norm_eff = 1.0 / (1.0 + result['model_params'] / 1e6)
            
            # Strategy weight
            strategy_weight = result['strategy_weight']
            
            # Composite score
            result['consensus_score'] = (
                (1.0 - self.diversity_weight) * norm_acc +
                self.diversity_weight * norm_div +
                0.1 * norm_eff
            ) * strategy_weight
        
        # Select best architecture
        best_result = max(evaluation_results, key=lambda x: x['consensus_score'])
        
        self.logger.info(
            f"🎯 Consensus selection: {best_result['strategy']} "
            f"(score: {best_result['consensus_score']:.3f})"
        )
        
        return best_result['architecture']
    
    def _pareto_optimal_consensus(
        self, 
        evaluation_results: List[Dict[str, Any]]
    ) -> Architecture:
        """Select from Pareto-optimal architectures."""
        
        # Extract objectives (accuracy, efficiency)
        objectives = []
        for result in evaluation_results:
            accuracy = result['cv_accuracy_mean']
            efficiency = 1.0 / (result['model_params'] + 1e6)  # Inverse of parameters
            objectives.append((accuracy, efficiency))
        
        # Find Pareto front
        pareto_indices = self._find_pareto_front(objectives)
        pareto_results = [evaluation_results[i] for i in pareto_indices]
        
        # Select from Pareto front using strategy weights
        best_result = max(
            pareto_results, 
            key=lambda x: x['cv_accuracy_mean'] * x['strategy_weight']
        )
        
        self.logger.info(
            f"🎯 Pareto optimal selection: {len(pareto_results)} candidates, "
            f"selected {best_result['strategy']}"
        )
        
        return best_result['architecture']
    
    def _ensemble_ranking_consensus(
        self, 
        evaluation_results: List[Dict[str, Any]]
    ) -> Architecture:
        """Select using ensemble ranking across multiple criteria."""
        
        criteria = ['cv_accuracy_mean', 'diversity_score', 'model_params']
        rankings = {}
        
        # Rank by each criterion
        for criterion in criteria:
            if criterion == 'model_params':
                # Lower is better for parameters
                sorted_results = sorted(
                    evaluation_results, 
                    key=lambda x: x[criterion]
                )
            else:
                # Higher is better for accuracy and diversity
                sorted_results = sorted(
                    evaluation_results, 
                    key=lambda x: x[criterion], 
                    reverse=True
                )
            
            for rank, result in enumerate(sorted_results):
                arch_id = id(result['architecture'])
                if arch_id not in rankings:
                    rankings[arch_id] = {'result': result, 'total_rank': 0}
                rankings[arch_id]['total_rank'] += rank
        
        # Select architecture with best overall ranking
        best_arch_id = min(rankings.keys(), key=lambda x: rankings[x]['total_rank'])
        best_result = rankings[best_arch_id]['result']
        
        self.logger.info(
            f"🎯 Ensemble ranking selection: {best_result['strategy']} "
            f"(rank sum: {rankings[best_arch_id]['total_rank']})"
        )
        
        return best_result['architecture']
    
    def _compute_diversity_score(
        self, 
        target_arch: Architecture, 
        all_architectures: List[Architecture]
    ) -> float:
        """Compute diversity score for an architecture relative to others."""
        if len(all_architectures) <= 1:
            return 1.0
        
        # Compare operations
        target_ops = target_arch.operations
        diversity_scores = []
        
        for other_arch in all_architectures:
            if other_arch == target_arch:
                continue
            
            other_ops = other_arch.operations
            # Hamming distance normalized by length
            hamming_dist = sum(
                o1 != o2 for o1, o2 in zip(target_ops, other_ops)
            ) / len(target_ops)
            
            diversity_scores.append(hamming_dist)
        
        return np.mean(diversity_scores) if diversity_scores else 1.0
    
    def _find_pareto_front(self, objectives: List[Tuple[float, float]]) -> List[int]:
        """Find Pareto-optimal solutions."""
        pareto_indices = []
        
        for i, obj_i in enumerate(objectives):
            is_dominated = False
            
            for j, obj_j in enumerate(objectives):
                if i == j:
                    continue
                
                # Check if obj_i is dominated by obj_j
                if all(obj_j[k] >= obj_i[k] for k in range(len(obj_i))) and \
                   any(obj_j[k] > obj_i[k] for k in range(len(obj_i))):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_indices.append(i)
        
        return pareto_indices
    
    def _final_validation(self, architecture: Architecture) -> Dict[str, Any]:
        """Perform final comprehensive validation of selected architecture."""
        self.logger.info("🔬 Performing final validation...")
        
        model = architecture.to_model()
        
        # Extended training evaluation
        final_results = self.evaluator.evaluate(
            model,
            self.train_loader,
            self.val_loader,
            epochs=self.final_evaluation_epochs
        )
        
        # Model analysis
        model_stats = compute_model_stats(model)
        
        return {
            **final_results,
            **model_stats,
            'architecture_complexity': len(architecture.operations),
            'search_method': 'ensemble'
        }
    
    def _get_strategy_instance(self, strategy_name: str) -> BaseSearchStrategy:
        """Get instance of specified search strategy."""
        
        strategy_map = {
            'evolutionary': EvolutionarySearch,
            'darts': DARTSSearch,
            'pc_darts': PCDARTSSearch,
            'bayesian': BayesianOptimizationSearch,
        }
        
        if strategy_name not in strategy_map:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        # Create instance with appropriate configuration
        strategy_class = strategy_map[strategy_name]
        
        # Basic configuration that works for all strategies
        config = {
            'device': self.device,
            'output_dir': self.output_dir / strategy_name,
            'verbose': False  # Reduce noise during ensemble execution
        }
        
        return strategy_class(**config)
    
    def _quick_evaluate(self, model: nn.Module) -> float:
        """Quick evaluation for initial screening."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # Quick evaluation - don't need full dataset
                if total >= 1000:
                    break
        
        return correct / total if total > 0 else 0.0
    
    def _get_default_weights(self) -> Dict[str, float]:
        """Get default strategy weights."""
        num_strategies = len(self.strategies)
        return {strategy: 1.0 / num_strategies for strategy in self.strategies}
    
    def _initialize_search_strategies(self) -> None:
        """Initialize all search strategy instances."""
        for strategy_name in self.strategies:
            self.search_instances[strategy_name] = self._get_strategy_instance(strategy_name)
    
    def _validate_configuration(self) -> None:
        """Validate ensemble configuration."""
        valid_strategies = ['evolutionary', 'darts', 'pc_darts', 'bayesian']
        valid_consensus = ['weighted_voting', 'pareto_optimal', 'ensemble_ranking']
        
        # Check strategies
        for strategy in self.strategies:
            if strategy not in valid_strategies:
                raise ValueError(f"Invalid strategy: {strategy}")
        
        # Check consensus method
        if self.consensus_method not in valid_consensus:
            raise ValueError(f"Invalid consensus method: {self.consensus_method}")
        
        # Check weights
        if set(self.strategy_weights.keys()) != set(self.strategies):
            self.logger.warning("Strategy weights don't match strategies, using defaults")
            self.strategy_weights = self._get_default_weights()
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computing device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        self.logger = logging.getLogger(f"{__name__}.EnsembleSearch")
        if self.verbose:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _save_ensemble_results(
        self,
        final_architecture: Architecture,
        strategy_results: Dict[str, List[Dict[str, Any]]],
        evaluation_results: List[Dict[str, Any]],
        final_metrics: Dict[str, Any],
        search_time: float
    ) -> None:
        """Save comprehensive ensemble search results."""
        
        results_path = self.output_dir / "ensemble_results.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare serializable results
        serializable_results = {
            'final_architecture': final_architecture.to_dict(),
            'final_metrics': final_metrics,
            'search_time': search_time,
            'ensemble_configuration': {
                'strategies': self.strategies,
                'strategy_weights': self.strategy_weights,
                'consensus_method': self.consensus_method,
                'num_seeds': self.num_seeds,
                'parallel_execution': self.parallel_execution
            },
            'strategy_summary': {},
            'evaluation_summary': {
                'total_candidates': len(evaluation_results),
                'consensus_method': self.consensus_method,
                'final_accuracy': final_metrics.get('accuracy', 0.0)
            }
        }
        
        # Summarize strategy results
        for strategy, results in strategy_results.items():
            accuracies = [r['accuracy'] for r in results]
            serializable_results['strategy_summary'][strategy] = {
                'num_runs': len(results),
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'best_accuracy': max(accuracies),
                'weight': self.strategy_weights[strategy]
            }
        
        # Save to file
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"📊 Ensemble results saved: {results_path}")
    
    def get_search_metrics(self) -> Dict[str, Any]:
        """Get comprehensive ensemble search metrics."""
        return {
            'search_method': 'ensemble',
            'strategies': self.strategies,
            'strategy_weights': self.strategy_weights,
            'consensus_method': self.consensus_method,
            'num_seeds': self.num_seeds,
            'parallel_execution': self.parallel_execution,
            'architecture_pool_size': self.architecture_pool_size,
            'cross_validation_folds': self.cross_validation_folds,
            'diversity_weight': self.diversity_weight
        } 