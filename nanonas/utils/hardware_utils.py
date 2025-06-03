"""
Hardware-aware optimization utilities for Neural Architecture Search.

This module provides comprehensive hardware profiling, energy estimation,
latency prediction, and hardware-specific optimization capabilities.
"""

import time
import psutil
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from collections import defaultdict
import threading
import subprocess
import platform
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class HardwareProfile:
    """Hardware profile containing device specifications and capabilities."""
    
    device_name: str
    device_type: str  # 'cpu', 'gpu', 'mobile', 'edge'
    compute_capability: Optional[str] = None
    memory_total: int = 0  # MB
    memory_bandwidth: float = 0.0  # GB/s
    peak_flops: float = 0.0  # GFLOPS
    power_budget: float = 0.0  # Watts
    thermal_design_power: float = 0.0  # Watts
    cache_sizes: Dict[str, int] = field(default_factory=dict)  # L1, L2, L3 in KB
    core_count: int = 0
    base_frequency: float = 0.0  # GHz
    boost_frequency: float = 0.0  # GHz
    architecture: str = ""
    vendor: str = ""
    driver_version: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'device_name': self.device_name,
            'device_type': self.device_type,
            'compute_capability': self.compute_capability,
            'memory_total': self.memory_total,
            'memory_bandwidth': self.memory_bandwidth,
            'peak_flops': self.peak_flops,
            'power_budget': self.power_budget,
            'thermal_design_power': self.thermal_design_power,
            'cache_sizes': self.cache_sizes,
            'core_count': self.core_count,
            'base_frequency': self.base_frequency,
            'boost_frequency': self.boost_frequency,
            'architecture': self.architecture,
            'vendor': self.vendor,
            'driver_version': self.driver_version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HardwareProfile':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PerformanceMetrics:
    """Performance metrics for architecture evaluation."""
    
    latency_ms: float = 0.0
    throughput_fps: float = 0.0
    energy_consumption_mj: float = 0.0  # millijoules
    memory_usage_mb: float = 0.0
    flops: int = 0
    parameters: int = 0
    memory_bandwidth_utilization: float = 0.0  # percentage
    compute_utilization: float = 0.0  # percentage
    cache_hit_rate: float = 0.0  # percentage
    thermal_throttling: bool = False
    power_efficiency: float = 0.0  # GFLOPS/Watt
    
    def efficiency_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted efficiency score."""
        if weights is None:
            weights = {
                'latency': 0.3,
                'energy': 0.3,
                'throughput': 0.2,
                'memory': 0.1,
                'power_efficiency': 0.1
            }
        
        # Normalize metrics (higher is better for efficiency)
        normalized_latency = 1.0 / (1.0 + self.latency_ms / 100.0)
        normalized_energy = 1.0 / (1.0 + self.energy_consumption_mj / 1000.0)
        normalized_throughput = min(self.throughput_fps / 100.0, 1.0)
        normalized_memory = 1.0 / (1.0 + self.memory_usage_mb / 1000.0)
        normalized_power_eff = min(self.power_efficiency / 10.0, 1.0)
        
        score = (
            weights['latency'] * normalized_latency +
            weights['energy'] * normalized_energy +
            weights['throughput'] * normalized_throughput +
            weights['memory'] * normalized_memory +
            weights['power_efficiency'] * normalized_power_eff
        )
        
        return score


class HardwareProfiler:
    """Hardware profiler for device characterization and monitoring."""
    
    def __init__(self):
        self.profiles: Dict[str, HardwareProfile] = {}
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_data: Dict[str, List[float]] = defaultdict(list)
        
    def profile_device(self, device: Optional[torch.device] = None) -> HardwareProfile:
        """Profile hardware device and return comprehensive profile."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        profile = HardwareProfile(
            device_name=self._get_device_name(device),
            device_type=self._get_device_type(device)
        )
        
        if device.type == 'cuda':
            profile = self._profile_gpu(device, profile)
        else:
            profile = self._profile_cpu(profile)
        
        self.profiles[str(device)] = profile
        return profile
    
    def _get_device_name(self, device: torch.device) -> str:
        """Get device name."""
        if device.type == 'cuda':
            return torch.cuda.get_device_name(device.index or 0)
        else:
            return platform.processor() or "CPU"
    
    def _get_device_type(self, device: torch.device) -> str:
        """Determine device type."""
        if device.type == 'cuda':
            return 'gpu'
        else:
            return 'cpu'
    
    def _profile_gpu(self, device: torch.device, profile: HardwareProfile) -> HardwareProfile:
        """Profile GPU device."""
        if not torch.cuda.is_available():
            return profile
        
        device_idx = device.index or 0
        
        # Get GPU properties
        props = torch.cuda.get_device_properties(device_idx)
        profile.compute_capability = f"{props.major}.{props.minor}"
        profile.memory_total = props.total_memory // (1024 * 1024)  # MB
        profile.core_count = props.multi_processor_count
        
        # Estimate memory bandwidth (rough approximation)
        if "RTX" in profile.device_name or "GTX" in profile.device_name:
            # NVIDIA consumer GPUs
            if "4090" in profile.device_name:
                profile.memory_bandwidth = 1008.0  # GB/s
                profile.peak_flops = 83000.0  # GFLOPS (FP32)
                profile.thermal_design_power = 450.0
            elif "4080" in profile.device_name:
                profile.memory_bandwidth = 717.0
                profile.peak_flops = 48700.0
                profile.thermal_design_power = 320.0
            elif "3090" in profile.device_name:
                profile.memory_bandwidth = 936.0
                profile.peak_flops = 35600.0
                profile.thermal_design_power = 350.0
            elif "3080" in profile.device_name:
                profile.memory_bandwidth = 760.0
                profile.peak_flops = 29800.0
                profile.thermal_design_power = 320.0
            else:
                # Generic estimation
                profile.memory_bandwidth = 500.0
                profile.peak_flops = 20000.0
                profile.thermal_design_power = 250.0
        
        profile.vendor = "NVIDIA"
        profile.architecture = self._get_gpu_architecture(profile.device_name)
        
        return profile
    
    def _profile_cpu(self, profile: HardwareProfile) -> HardwareProfile:
        """Profile CPU device."""
        # Get CPU information
        profile.core_count = psutil.cpu_count(logical=False)
        profile.memory_total = psutil.virtual_memory().total // (1024 * 1024)  # MB
        
        # Get CPU frequency
        freq_info = psutil.cpu_freq()
        if freq_info:
            profile.base_frequency = freq_info.min / 1000.0  # GHz
            profile.boost_frequency = freq_info.max / 1000.0  # GHz
        
        # Estimate performance based on CPU model
        cpu_info = platform.processor().lower()
        if "intel" in cpu_info:
            profile.vendor = "Intel"
            if "i9" in cpu_info:
                profile.peak_flops = 1000.0  # GFLOPS
                profile.thermal_design_power = 125.0
            elif "i7" in cpu_info:
                profile.peak_flops = 800.0
                profile.thermal_design_power = 95.0
            elif "i5" in cpu_info:
                profile.peak_flops = 600.0
                profile.thermal_design_power = 65.0
            else:
                profile.peak_flops = 400.0
                profile.thermal_design_power = 65.0
        elif "amd" in cpu_info:
            profile.vendor = "AMD"
            profile.peak_flops = 800.0
            profile.thermal_design_power = 105.0
        
        # Estimate memory bandwidth (typical DDR4/DDR5)
        profile.memory_bandwidth = 50.0  # GB/s (conservative estimate)
        
        return profile
    
    def _get_gpu_architecture(self, device_name: str) -> str:
        """Get GPU architecture name."""
        device_name = device_name.lower()
        if "rtx 40" in device_name or "4090" in device_name or "4080" in device_name:
            return "Ada Lovelace"
        elif "rtx 30" in device_name or "3090" in device_name or "3080" in device_name:
            return "Ampere"
        elif "rtx 20" in device_name:
            return "Turing"
        elif "gtx 16" in device_name:
            return "Turing"
        elif "gtx 10" in device_name:
            return "Pascal"
        else:
            return "Unknown"
    
    @contextmanager
    def monitor_performance(self, interval: float = 0.1):
        """Context manager for performance monitoring."""
        self.start_monitoring(interval)
        try:
            yield self.monitoring_data
        finally:
            self.stop_monitoring()
    
    def start_monitoring(self, interval: float = 0.1):
        """Start performance monitoring in background thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_data.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
    
    def _monitoring_loop(self, interval: float):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent()
                memory_info = psutil.virtual_memory()
                
                self.monitoring_data['cpu_percent'].append(cpu_percent)
                self.monitoring_data['memory_percent'].append(memory_info.percent)
                self.monitoring_data['memory_used_mb'].append(
                    memory_info.used / (1024 * 1024)
                )
                
                # GPU metrics (if available)
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                    gpu_memory_cached = torch.cuda.memory_reserved() / (1024 * 1024)  # MB
                    
                    self.monitoring_data['gpu_memory_mb'].append(gpu_memory)
                    self.monitoring_data['gpu_memory_cached_mb'].append(gpu_memory_cached)
                
                # Timestamp
                self.monitoring_data['timestamp'].append(time.time())
                
                time.sleep(interval)
                
            except Exception as e:
                logger.warning(f"Monitoring error: {e}")
                time.sleep(interval)


class EnergyEstimator:
    """Energy consumption estimator for neural architectures."""
    
    def __init__(self, hardware_profile: HardwareProfile):
        self.hardware_profile = hardware_profile
        self.operation_energy_map = self._build_energy_map()
    
    def _build_energy_map(self) -> Dict[str, float]:
        """Build energy consumption map for different operations."""
        # Energy per operation in picojoules (pJ)
        # Based on research literature and hardware specifications
        
        base_energy = {
            'conv2d': 100.0,
            'linear': 50.0,
            'relu': 1.0,
            'gelu': 2.0,
            'swish': 3.0,
            'mish': 4.0,
            'prelu': 2.0,
            'batch_norm': 10.0,
            'layer_norm': 15.0,
            'group_norm': 20.0,
            'rms_norm': 12.0,
            'attention': 200.0,
            'channel_attention': 150.0,
            'spatial_attention': 180.0,
            'pooling': 5.0,
            'adaptive_pool': 8.0,
            'dropout': 1.0,
            'identity': 0.1,
        }
        
        # Scale based on hardware type
        if self.hardware_profile.device_type == 'gpu':
            # GPUs are more energy efficient for parallel operations
            scale_factor = 0.3
        elif self.hardware_profile.device_type == 'mobile':
            # Mobile devices have lower power but less efficiency
            scale_factor = 2.0
        elif self.hardware_profile.device_type == 'edge':
            # Edge devices optimized for efficiency
            scale_factor = 0.5
        else:
            # CPU baseline
            scale_factor = 1.0
        
        return {op: energy * scale_factor for op, energy in base_energy.items()}
    
    def estimate_operation_energy(self, operation_type: str, input_shape: Tuple[int, ...],
                                output_shape: Tuple[int, ...], **kwargs) -> float:
        """Estimate energy consumption for a single operation."""
        base_energy = self.operation_energy_map.get(operation_type, 50.0)
        
        # Calculate computational complexity
        input_size = np.prod(input_shape)
        output_size = np.prod(output_shape)
        
        if operation_type == 'conv2d':
            kernel_size = kwargs.get('kernel_size', 3)
            channels_in = input_shape[1] if len(input_shape) > 1 else 1
            channels_out = output_shape[1] if len(output_shape) > 1 else 1
            
            # Energy scales with MAC operations
            mac_ops = output_size * channels_in * (kernel_size ** 2)
            energy = base_energy * mac_ops / 1000.0  # Normalize
            
        elif operation_type == 'linear':
            input_features = input_shape[-1] if input_shape else 1
            output_features = output_shape[-1] if output_shape else 1
            batch_size = input_shape[0] if input_shape else 1
            
            mac_ops = batch_size * input_features * output_features
            energy = base_energy * mac_ops / 1000.0
            
        elif operation_type == 'attention':
            seq_len = input_shape[1] if len(input_shape) > 1 else 1
            embed_dim = input_shape[-1] if input_shape else 1
            
            # Attention has quadratic complexity in sequence length
            attention_ops = seq_len ** 2 * embed_dim
            energy = base_energy * attention_ops / 1000.0
            
        else:
            # For other operations, energy scales with output size
            energy = base_energy * output_size / 1000.0
        
        return energy  # Return in picojoules
    
    def estimate_architecture_energy(self, architecture, input_shape: Tuple[int, ...]) -> float:
        """Estimate total energy consumption for an architecture."""
        total_energy = 0.0
        current_shape = input_shape
        
        # This would need to be integrated with the actual architecture representation
        # For now, provide a placeholder implementation
        
        # Estimate based on architecture complexity metrics
        if hasattr(architecture, 'get_complexity_metrics'):
            metrics = architecture.get_complexity_metrics()
            flops = metrics.get('flops', 0)
            params = metrics.get('parameters', 0)
            
            # Rough estimation: energy scales with FLOPs and parameters
            flop_energy = flops * 0.1  # pJ per FLOP
            param_energy = params * 0.01  # pJ per parameter (memory access)
            
            total_energy = flop_energy + param_energy
        
        return total_energy


class LatencyPredictor:
    """Latency predictor for neural architectures."""
    
    def __init__(self, hardware_profile: HardwareProfile):
        self.hardware_profile = hardware_profile
        self.operation_latency_map = self._build_latency_map()
        self.calibration_data: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    
    def _build_latency_map(self) -> Dict[str, float]:
        """Build latency map for different operations."""
        # Base latency per operation in microseconds (μs)
        base_latency = {
            'conv2d': 100.0,
            'linear': 20.0,
            'relu': 1.0,
            'gelu': 3.0,
            'swish': 4.0,
            'mish': 5.0,
            'prelu': 2.0,
            'batch_norm': 5.0,
            'layer_norm': 8.0,
            'group_norm': 12.0,
            'rms_norm': 6.0,
            'attention': 150.0,
            'channel_attention': 80.0,
            'spatial_attention': 100.0,
            'pooling': 3.0,
            'adaptive_pool': 5.0,
            'dropout': 0.5,
            'identity': 0.1,
        }
        
        # Scale based on hardware capabilities
        if self.hardware_profile.device_type == 'gpu':
            # GPUs have high parallelism
            scale_factor = 0.1
        elif self.hardware_profile.device_type == 'mobile':
            # Mobile devices are slower
            scale_factor = 5.0
        elif self.hardware_profile.device_type == 'edge':
            # Edge devices vary widely
            scale_factor = 2.0
        else:
            # CPU baseline
            scale_factor = 1.0
        
        return {op: latency * scale_factor for op, latency in base_latency.items()}
    
    def predict_operation_latency(self, operation_type: str, input_shape: Tuple[int, ...],
                                output_shape: Tuple[int, ...], **kwargs) -> float:
        """Predict latency for a single operation."""
        base_latency = self.operation_latency_map.get(operation_type, 10.0)
        
        # Calculate computational complexity
        input_size = np.prod(input_shape)
        output_size = np.prod(output_shape)
        
        if operation_type == 'conv2d':
            kernel_size = kwargs.get('kernel_size', 3)
            channels_in = input_shape[1] if len(input_shape) > 1 else 1
            
            # Latency scales with MAC operations and memory access
            mac_ops = output_size * channels_in * (kernel_size ** 2)
            memory_access = input_size + output_size
            
            compute_latency = base_latency * mac_ops / 10000.0
            memory_latency = memory_access / self.hardware_profile.memory_bandwidth * 1000.0
            
            latency = compute_latency + memory_latency
            
        elif operation_type == 'linear':
            input_features = input_shape[-1] if input_shape else 1
            output_features = output_shape[-1] if output_shape else 1
            batch_size = input_shape[0] if input_shape else 1
            
            mac_ops = batch_size * input_features * output_features
            memory_access = input_size + output_size
            
            compute_latency = base_latency * mac_ops / 10000.0
            memory_latency = memory_access / self.hardware_profile.memory_bandwidth * 1000.0
            
            latency = compute_latency + memory_latency
            
        elif operation_type == 'attention':
            seq_len = input_shape[1] if len(input_shape) > 1 else 1
            embed_dim = input_shape[-1] if input_shape else 1
            
            # Attention has quadratic complexity
            attention_ops = seq_len ** 2 * embed_dim
            latency = base_latency * attention_ops / 10000.0
            
        else:
            # For other operations, latency scales with output size
            latency = base_latency * output_size / 10000.0
        
        return latency  # Return in microseconds
    
    def predict_architecture_latency(self, architecture, input_shape: Tuple[int, ...],
                                   batch_size: int = 1) -> float:
        """Predict total latency for an architecture."""
        total_latency = 0.0
        
        # This would need to be integrated with the actual architecture representation
        # For now, provide a placeholder implementation
        
        if hasattr(architecture, 'get_complexity_metrics'):
            metrics = architecture.get_complexity_metrics()
            flops = metrics.get('flops', 0)
            params = metrics.get('parameters', 0)
            
            # Rough estimation based on FLOPs and memory access
            compute_latency = flops / self.hardware_profile.peak_flops * 1000.0  # ms
            memory_latency = params * 4 / self.hardware_profile.memory_bandwidth  # ms (4 bytes per param)
            
            total_latency = compute_latency + memory_latency
        
        return total_latency  # Return in milliseconds
    
    def calibrate_with_measurement(self, operation_type: str, complexity: int, measured_latency: float):
        """Calibrate predictor with actual measurements."""
        self.calibration_data[operation_type].append((complexity, measured_latency))
        
        # Update latency map based on calibration data
        if len(self.calibration_data[operation_type]) >= 3:
            # Simple linear regression to update base latency
            complexities = [x[0] for x in self.calibration_data[operation_type]]
            latencies = [x[1] for x in self.calibration_data[operation_type]]
            
            if len(set(complexities)) > 1:  # Avoid division by zero
                slope = np.polyfit(complexities, latencies, 1)[0]
                self.operation_latency_map[operation_type] = slope * 1000.0  # Convert to base units


class HardwareConstraintManager:
    """Manager for hardware-specific constraints and optimization."""
    
    def __init__(self, hardware_profile: HardwareProfile):
        self.hardware_profile = hardware_profile
        self.constraints = self._build_constraints()
    
    def _build_constraints(self) -> Dict[str, Any]:
        """Build hardware-specific constraints."""
        constraints = {
            'max_memory_mb': self.hardware_profile.memory_total * 0.8,  # 80% of total memory
            'max_power_watts': self.hardware_profile.thermal_design_power * 0.9,  # 90% of TDP
            'max_latency_ms': 100.0,  # Default 100ms latency budget
            'min_throughput_fps': 10.0,  # Minimum 10 FPS
            'max_energy_mj': 1000.0,  # Maximum 1000 mJ per inference
        }
        
        # Adjust constraints based on device type
        if self.hardware_profile.device_type == 'mobile':
            constraints.update({
                'max_latency_ms': 50.0,  # Stricter latency for mobile
                'max_energy_mj': 100.0,  # Much stricter energy budget
                'max_power_watts': 5.0,  # Mobile power budget
            })
        elif self.hardware_profile.device_type == 'edge':
            constraints.update({
                'max_latency_ms': 20.0,  # Real-time requirements
                'max_energy_mj': 50.0,  # Very strict energy budget
                'max_power_watts': 10.0,  # Edge device power budget
            })
        elif self.hardware_profile.device_type == 'gpu':
            constraints.update({
                'max_latency_ms': 200.0,  # More relaxed for high-performance
                'min_throughput_fps': 100.0,  # Higher throughput expectations
            })
        
        return constraints
    
    def check_constraints(self, metrics: PerformanceMetrics) -> Dict[str, bool]:
        """Check if performance metrics satisfy hardware constraints."""
        results = {}
        
        results['memory_constraint'] = metrics.memory_usage_mb <= self.constraints['max_memory_mb']
        results['latency_constraint'] = metrics.latency_ms <= self.constraints['max_latency_ms']
        results['throughput_constraint'] = metrics.throughput_fps >= self.constraints['min_throughput_fps']
        results['energy_constraint'] = metrics.energy_consumption_mj <= self.constraints['max_energy_mj']
        
        # Power constraint (estimated from energy and latency)
        estimated_power = metrics.energy_consumption_mj / metrics.latency_ms if metrics.latency_ms > 0 else 0
        results['power_constraint'] = estimated_power <= self.constraints['max_power_watts']
        
        return results
    
    def get_constraint_violations(self, metrics: PerformanceMetrics) -> List[str]:
        """Get list of constraint violations."""
        violations = []
        constraint_results = self.check_constraints(metrics)
        
        for constraint, satisfied in constraint_results.items():
            if not satisfied:
                violations.append(constraint)
        
        return violations
    
    def suggest_optimizations(self, metrics: PerformanceMetrics) -> List[str]:
        """Suggest optimizations based on constraint violations."""
        suggestions = []
        violations = self.get_constraint_violations(metrics)
        
        if 'memory_constraint' in violations:
            suggestions.extend([
                "Reduce model width (number of channels)",
                "Use depth-wise separable convolutions",
                "Apply model pruning techniques",
                "Use gradient checkpointing during training"
            ])
        
        if 'latency_constraint' in violations:
            suggestions.extend([
                "Reduce model depth (number of layers)",
                "Use more efficient operations (e.g., MobileNet blocks)",
                "Apply quantization (INT8/FP16)",
                "Optimize memory access patterns"
            ])
        
        if 'energy_constraint' in violations:
            suggestions.extend([
                "Use energy-efficient activations (ReLU over GELU)",
                "Reduce attention mechanisms",
                "Apply aggressive pruning",
                "Use lower precision arithmetic"
            ])
        
        if 'throughput_constraint' in violations:
            suggestions.extend([
                "Increase batch size if memory allows",
                "Use tensor parallelism",
                "Optimize data loading pipeline",
                "Use mixed precision training"
            ])
        
        return suggestions


def benchmark_operation(operation_type: str, input_shape: Tuple[int, ...], 
                       device: torch.device, num_runs: int = 100) -> PerformanceMetrics:
    """Benchmark a specific operation on given hardware."""
    # This is a placeholder for actual operation benchmarking
    # In practice, this would create and run the actual operation
    
    metrics = PerformanceMetrics()
    
    # Simulate benchmarking
    import random
    metrics.latency_ms = random.uniform(1.0, 100.0)
    metrics.throughput_fps = 1000.0 / metrics.latency_ms
    metrics.memory_usage_mb = np.prod(input_shape) * 4 / (1024 * 1024)  # 4 bytes per float
    metrics.energy_consumption_mj = metrics.latency_ms * 10.0  # Rough estimate
    
    return metrics


def save_hardware_profile(profile: HardwareProfile, filepath: Union[str, Path]):
    """Save hardware profile to file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(profile.to_dict(), f, indent=2)


def load_hardware_profile(filepath: Union[str, Path]) -> HardwareProfile:
    """Load hardware profile from file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return HardwareProfile.from_dict(data)


# Global hardware profiler instance
_global_profiler: Optional[HardwareProfiler] = None


def get_hardware_profiler() -> HardwareProfiler:
    """Get global hardware profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = HardwareProfiler()
    return _global_profiler


def profile_current_device() -> HardwareProfile:
    """Profile the current PyTorch device."""
    profiler = get_hardware_profiler()
    return profiler.profile_device()


def estimate_architecture_performance(architecture, hardware_profile: HardwareProfile,
                                    input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> PerformanceMetrics:
    """Estimate performance metrics for an architecture on given hardware."""
    energy_estimator = EnergyEstimator(hardware_profile)
    latency_predictor = LatencyPredictor(hardware_profile)
    
    metrics = PerformanceMetrics()
    
    # Estimate energy and latency
    metrics.energy_consumption_mj = energy_estimator.estimate_architecture_energy(architecture, input_shape)
    metrics.latency_ms = latency_predictor.predict_architecture_latency(architecture, input_shape)
    
    # Calculate derived metrics
    if metrics.latency_ms > 0:
        metrics.throughput_fps = 1000.0 / metrics.latency_ms
    
    # Estimate memory usage (rough approximation)
    if hasattr(architecture, 'get_complexity_metrics'):
        complexity = architecture.get_complexity_metrics()
        metrics.parameters = complexity.get('parameters', 0)
        metrics.flops = complexity.get('flops', 0)
        metrics.memory_usage_mb = metrics.parameters * 4 / (1024 * 1024)  # 4 bytes per parameter
    
    # Calculate power efficiency
    if metrics.energy_consumption_mj > 0 and metrics.latency_ms > 0:
        power_watts = metrics.energy_consumption_mj / metrics.latency_ms
        if power_watts > 0:
            metrics.power_efficiency = metrics.flops / (power_watts * 1e9)  # GFLOPS/Watt
    
    return metrics 