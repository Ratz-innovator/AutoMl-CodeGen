"""
Multi-Objective Optimization for Neural Architecture Search

This module implements multi-objective optimization algorithms for balancing
accuracy, latency, memory usage, and energy consumption in neural architectures.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import copy

@dataclass
class Objective:
    """Definition of a single optimization objective."""
    name: str
    direction: str  # 'maximize' or 'minimize'
    weight: float = 1.0
    threshold: Optional[float] = None
    normalization: Optional[Tuple[float, float]] = None  # (min, max) for normalization

class MultiObjectiveOptimizer:
    """
    Multi-objective optimizer for neural architecture search.
    
    This class handles optimization across multiple objectives such as:
    - Accuracy (maximize)
    - Latency (minimize)
    - Memory usage (minimize)
    - Energy consumption (minimize)
    - FLOPs (minimize)
    
    Uses Pareto front analysis to find architectures that represent
    optimal trade-offs between competing objectives.
    
    Example:
        >>> objectives = ['accuracy', 'latency', 'memory']
        >>> optimizer = MultiObjectiveOptimizer(objectives)
        >>> optimizer.update(population, metrics)
        >>> pareto_front = optimizer.get_pareto_front()
    """
    
    def __init__(
        self,
        objectives: List[str],
        hardware_target: str = 'gpu',
        objective_weights: Optional[Dict[str, float]] = None,
        normalization_strategy: str = 'dynamic',
        **kwargs
    ):
        """
        Initialize multi-objective optimizer.
        
        Args:
            objectives: List of objective names to optimize
            hardware_target: Target hardware platform
            objective_weights: Custom weights for objectives
            normalization_strategy: How to normalize objectives ('dynamic', 'fixed')
        """
        self.objective_names = objectives
        self.hardware_target = hardware_target
        self.normalization_strategy = normalization_strategy
        
        # Create objective definitions
        self.objectives = self._create_objectives(objective_weights)
        
        # Pareto front tracking
        self.pareto_front: List[Dict[str, Any]] = []
        self.all_solutions: List[Dict[str, Any]] = []
        
        # Normalization bounds (updated dynamically)
        self.normalization_bounds: Dict[str, Tuple[float, float]] = {}
        
        # Statistics
        self.generation_count = 0
        self.dominated_count = 0
        
    def _create_objectives(self, custom_weights: Optional[Dict[str, float]] = None) -> List[Objective]:
        """Create objective definitions with default or custom weights."""
        default_configs = {
            'accuracy': {'direction': 'maximize', 'weight': 0.5},
            'latency': {'direction': 'minimize', 'weight': 0.3},
            'memory': {'direction': 'minimize', 'weight': 0.1},
            'energy': {'direction': 'minimize', 'weight': 0.1},
            'flops': {'direction': 'minimize', 'weight': 0.05},
            'parameters': {'direction': 'minimize', 'weight': 0.05}
        }
        
        # Adjust weights based on hardware target
        if self.hardware_target == 'mobile':
            default_configs['latency']['weight'] = 0.4
            default_configs['memory']['weight'] = 0.2
            default_configs['energy']['weight'] = 0.2
        elif self.hardware_target == 'edge':
            default_configs['latency']['weight'] = 0.5
            default_configs['memory']['weight'] = 0.25
            default_configs['energy']['weight'] = 0.15
        
        objectives = []
        for obj_name in self.objective_names:
            if obj_name in default_configs:
                config = default_configs[obj_name].copy()
                
                # Override with custom weights if provided
                if custom_weights and obj_name in custom_weights:
                    config['weight'] = custom_weights[obj_name]
                
                objectives.append(Objective(
                    name=obj_name,
                    direction=config['direction'],
                    weight=config['weight']
                ))
            else:
                # Unknown objective - assume minimize with equal weight
                objectives.append(Objective(
                    name=obj_name,
                    direction='minimize',
                    weight=1.0 / len(self.objective_names)
                ))
        
        return objectives
    
    def update(
        self,
        population: List[Any],
        metrics: List[Dict[str, float]],
        **kwargs
    ) -> None:
        """
        Update optimizer with new population and metrics.
        
        Args:
            population: List of architectures/individuals
            metrics: List of metric dictionaries for each individual
        """
        self.generation_count += 1
        
        # Create solutions from population and metrics
        solutions = []
        for individual, metric_dict in zip(population, metrics):
            if metric_dict:  # Skip failed evaluations
                solution = {
                    'individual': individual,
                    'metrics': metric_dict,
                    'objectives': self._extract_objectives(metric_dict),
                    'generation': self.generation_count
                }
                solutions.append(solution)
        
        # Add to all solutions
        self.all_solutions.extend(solutions)
        
        # Update normalization bounds
        self._update_normalization_bounds(solutions)
        
        # Normalize objectives
        for solution in solutions:
            solution['normalized_objectives'] = self._normalize_objectives(solution['objectives'])
        
        # Update Pareto front
        self._update_pareto_front(solutions)
    
    def _extract_objectives(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Extract objective values from metrics."""
        objectives = {}
        for obj in self.objectives:
            if obj.name in metrics:
                objectives[obj.name] = metrics[obj.name]
            else:
                # Use default value if metric not available
                if obj.direction == 'maximize':
                    objectives[obj.name] = 0.0  # Worst case for maximization
                else:
                    objectives[obj.name] = float('inf')  # Worst case for minimization
        
        return objectives
    
    def _update_normalization_bounds(self, solutions: List[Dict[str, Any]]) -> None:
        """Update normalization bounds based on observed solutions."""
        if self.normalization_strategy != 'dynamic':
            return
        
        for obj in self.objectives:
            obj_name = obj.name
            values = []
            
            # Collect values from current solutions
            for solution in solutions:
                if obj_name in solution['objectives']:
                    values.append(solution['objectives'][obj_name])
            
            # Collect values from all historical solutions
            for solution in self.all_solutions:
                if obj_name in solution['objectives']:
                    values.append(solution['objectives'][obj_name])
            
            if values:
                current_min = min(values)
                current_max = max(values)
                
                if obj_name in self.normalization_bounds:
                    # Update bounds
                    old_min, old_max = self.normalization_bounds[obj_name]
                    new_min = min(old_min, current_min)
                    new_max = max(old_max, current_max)
                else:
                    # Initialize bounds
                    new_min, new_max = current_min, current_max
                
                # Ensure bounds are not equal
                if new_min == new_max:
                    if obj.direction == 'maximize':
                        new_min = new_max * 0.9 if new_max > 0 else new_max - 0.1
                    else:
                        new_max = new_min * 1.1 if new_min > 0 else new_min + 0.1
                
                self.normalization_bounds[obj_name] = (new_min, new_max)
    
    def _normalize_objectives(self, objectives: Dict[str, float]) -> Dict[str, float]:
        """Normalize objective values to [0, 1] range."""
        normalized = {}
        
        for obj in self.objectives:
            obj_name = obj.name
            if obj_name not in objectives:
                continue
            
            value = objectives[obj_name]
            
            if obj_name in self.normalization_bounds:
                min_val, max_val = self.normalization_bounds[obj_name]
                
                if max_val > min_val:
                    # Normalize to [0, 1]
                    normalized_value = (value - min_val) / (max_val - min_val)
                    
                    # For maximization objectives, flip the normalized value
                    if obj.direction == 'maximize':
                        normalized_value = 1.0 - normalized_value
                    
                    normalized[obj_name] = max(0.0, min(1.0, normalized_value))
                else:
                    normalized[obj_name] = 0.0
            else:
                # No normalization bounds available
                if obj.direction == 'maximize':
                    normalized[obj_name] = 1.0 - value if value <= 1.0 else 0.0
                else:
                    normalized[obj_name] = value
        
        return normalized
    
    def _update_pareto_front(self, new_solutions: List[Dict[str, Any]]) -> None:
        """Update Pareto front with new solutions."""
        # Combine current Pareto front with new solutions
        all_candidates = self.pareto_front + new_solutions
        
        # Find new Pareto front
        pareto_front = []
        
        for i, solution_i in enumerate(all_candidates):
            is_pareto_optimal = True
            
            for j, solution_j in enumerate(all_candidates):
                if i == j:
                    continue
                
                if self._dominates(solution_j, solution_i):
                    is_pareto_optimal = False
                    break
            
            if is_pareto_optimal:
                pareto_front.append(solution_i)
        
        # Remove duplicates (same individual)
        unique_pareto = []
        seen_individuals = set()
        
        for solution in pareto_front:
            individual_id = id(solution['individual'])
            if individual_id not in seen_individuals:
                unique_pareto.append(solution)
                seen_individuals.add(individual_id)
        
        self.pareto_front = unique_pareto
    
    def _dominates(self, solution_a: Dict[str, Any], solution_b: Dict[str, Any]) -> bool:
        """Check if solution A dominates solution B."""
        objectives_a = solution_a['normalized_objectives']
        objectives_b = solution_b['normalized_objectives']
        
        # A dominates B if A is at least as good in all objectives
        # and strictly better in at least one objective
        at_least_as_good = True
        strictly_better = False
        
        for obj in self.objectives:
            obj_name = obj.name
            
            if obj_name not in objectives_a or obj_name not in objectives_b:
                continue
            
            value_a = objectives_a[obj_name]
            value_b = objectives_b[obj_name]
            
            # For normalized objectives, lower is always better
            if value_a > value_b:
                at_least_as_good = False
                break
            elif value_a < value_b:
                strictly_better = True
        
        return at_least_as_good and strictly_better
    
    def get_pareto_front(self) -> List[Dict[str, Any]]:
        """Get current Pareto front."""
        return self.pareto_front.copy()
    
    def get_pareto_front_individuals(self) -> List[Any]:
        """Get individuals from the Pareto front."""
        return [solution['individual'] for solution in self.pareto_front]
    
    def select_solution_by_preference(
        self,
        preference_weights: Optional[Dict[str, float]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Select a single solution from Pareto front based on preferences.
        
        Args:
            preference_weights: Custom weights for selection
            
        Returns:
            Selected solution or None if Pareto front is empty
        """
        if not self.pareto_front:
            return None
        
        if len(self.pareto_front) == 1:
            return self.pareto_front[0]
        
        # Use provided weights or default objective weights
        weights = preference_weights or {obj.name: obj.weight for obj in self.objectives}
        
        best_solution = None
        best_score = float('inf')
        
        for solution in self.pareto_front:
            # Calculate weighted sum of normalized objectives
            score = 0.0
            total_weight = 0.0
            
            for obj_name, weight in weights.items():
                if obj_name in solution['normalized_objectives']:
                    score += weight * solution['normalized_objectives'][obj_name]
                    total_weight += weight
            
            if total_weight > 0:
                score /= total_weight
                
                if score < best_score:
                    best_score = score
                    best_solution = solution
        
        return best_solution
    
    def get_hypervolume(self, reference_point: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate hypervolume indicator for the Pareto front.
        
        Args:
            reference_point: Reference point for hypervolume calculation
            
        Returns:
            Hypervolume value
        """
        if not self.pareto_front:
            return 0.0
        
        # Use worst values as reference point if not provided
        if reference_point is None:
            reference_point = {}
            for obj in self.objectives:
                obj_name = obj.name
                if obj_name in self.normalization_bounds:
                    reference_point[obj_name] = 1.0  # Worst normalized value
                else:
                    reference_point[obj_name] = 1.0
        
        # Simplified 2D hypervolume calculation for common case
        if len(self.objectives) == 2:
            return self._calculate_2d_hypervolume(reference_point)
        else:
            # For higher dimensions, use approximate calculation
            return self._calculate_approximate_hypervolume(reference_point)
    
    def _calculate_2d_hypervolume(self, reference_point: Dict[str, float]) -> float:
        """Calculate exact 2D hypervolume."""
        if len(self.objectives) != 2:
            return 0.0
        
        obj1_name = self.objectives[0].name
        obj2_name = self.objectives[1].name
        
        points = []
        for solution in self.pareto_front:
            obj_vals = solution['normalized_objectives']
            if obj1_name in obj_vals and obj2_name in obj_vals:
                points.append((obj_vals[obj1_name], obj_vals[obj2_name]))
        
        if not points:
            return 0.0
        
        # Sort points by first objective
        points.sort()
        
        # Calculate area
        area = 0.0
        ref_x = reference_point.get(obj1_name, 1.0)
        ref_y = reference_point.get(obj2_name, 1.0)
        
        for i, (x, y) in enumerate(points):
            if i == 0:
                width = ref_x - x
                height = ref_y - y
            else:
                prev_x, prev_y = points[i-1]
                width = ref_x - x
                height = prev_y - y
            
            area += max(0, width) * max(0, height)
            ref_y = min(ref_y, y)
        
        return area
    
    def _calculate_approximate_hypervolume(self, reference_point: Dict[str, float]) -> float:
        """Calculate approximate hypervolume for higher dimensions."""
        if not self.pareto_front:
            return 0.0
        
        # Use sum of individual dominated volumes as approximation
        total_volume = 0.0
        
        for solution in self.pareto_front:
            obj_vals = solution['normalized_objectives']
            volume = 1.0
            
            for obj in self.objectives:
                obj_name = obj.name
                if obj_name in obj_vals and obj_name in reference_point:
                    dim_contribution = reference_point[obj_name] - obj_vals[obj_name]
                    volume *= max(0, dim_contribution)
            
            total_volume += volume
        
        return total_volume
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            'pareto_front_size': len(self.pareto_front),
            'total_solutions_evaluated': len(self.all_solutions),
            'generations': self.generation_count,
            'objectives': [obj.name for obj in self.objectives],
            'normalization_bounds': self.normalization_bounds.copy(),
            'hypervolume': self.get_hypervolume()
        }
    
    def reset(self) -> None:
        """Reset optimizer state."""
        self.pareto_front.clear()
        self.all_solutions.clear()
        self.normalization_bounds.clear()
        self.generation_count = 0
        self.dominated_count = 0

# Alias for backwards compatibility
ParetoOptimizer = MultiObjectiveOptimizer 