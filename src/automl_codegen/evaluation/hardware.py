"""
Hardware Profiler for Neural Architecture Evaluation

This module provides hardware-aware profiling capabilities for neural architectures,
measuring actual performance metrics like latency, memory usage, and energy consumption.
"""

import logging
import time
import torch
import torch.nn as nn
import numpy as np
import psutil
import gc
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import platform
import threading

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

try:
    import gpustat
    GPUSTAT_AVAILABLE = True
except ImportError:
    GPUSTAT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class HardwareMetrics:
    """Container for hardware performance metrics."""
    latency_ms: float
    memory_mb: float
    gpu_memory_mb: float
    energy_consumption: float
    flops: int
    parameters: int
    throughput_fps: float
    cpu_utilization: float
    gpu_utilization: float


class MemoryMonitor:
    """Monitor memory usage during model execution."""
    
    def __init__(self):
        self.peak_memory = 0
        self.current_memory = 0
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start memory monitoring in background thread."""
        self.peak_memory = 0
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> float:
        """Stop monitoring and return peak memory usage."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        return self.peak_memory
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            if torch.cuda.is_available():
                current = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                self.peak_memory = max(self.peak_memory, current)
            time.sleep(0.01)  # Monitor every 10ms


class HardwareProfiler:
    """Hardware-aware profiler for neural architectures."""
    
    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        warmup_runs: int = 5,
        benchmark_runs: int = 20,
        energy_monitoring: bool = True
    ):
        """
        Initialize hardware profiler.
        
        Args:
            device: Target device for profiling
            warmup_runs: Number of warmup runs before benchmarking
            benchmark_runs: Number of runs for accurate timing
            energy_monitoring: Whether to monitor energy consumption
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
            
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.energy_monitoring = energy_monitoring
        
        # Initialize GPU monitoring if available
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available and PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.pynvml_ready = True
            except Exception as e:
                logger.warning(f"Failed to initialize NVIDIA-ML: {e}")
                self.pynvml_ready = False
        else:
            self.pynvml_ready = False
        
        # Memory monitor
        self.memory_monitor = MemoryMonitor()
        
        logger.info(f"Hardware profiler initialized for device: {self.device}")
    
    def profile_model(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        batch_size: int = 1,
        detailed: bool = True
    ) -> HardwareMetrics:
        """
        Profile a PyTorch model for hardware performance.
        
        Args:
            model: PyTorch model to profile
            input_shape: Input tensor shape (without batch dimension)
            batch_size: Batch size for profiling
            detailed: Whether to perform detailed profiling
            
        Returns:
            HardwareMetrics with performance measurements
        """
        logger.info(f"Profiling model with input shape: {input_shape}, batch size: {batch_size}")
        
        # Move model to device
        model = model.to(self.device)
        model.eval()
        
        # Create input tensor
        full_input_shape = (batch_size,) + input_shape
        dummy_input = torch.randn(full_input_shape, device=self.device)
        
        # Warm up the model
        self._warmup_model(model, dummy_input)
        
        # Profile latency
        latency_ms = self._profile_latency(model, dummy_input)
        
        # Profile memory usage
        memory_mb, gpu_memory_mb = self._profile_memory(model, dummy_input)
        
        # Calculate model statistics
        parameters = self._count_parameters(model)
        flops = self._estimate_flops(model, dummy_input)
        
        # Calculate throughput
        throughput_fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
        
        # Profile system utilization
        cpu_util, gpu_util = self._profile_utilization()
        
        # Energy profiling (simplified)
        energy_consumption = self._estimate_energy_consumption(latency_ms)
        
        metrics = HardwareMetrics(
            latency_ms=latency_ms,
            memory_mb=memory_mb,
            gpu_memory_mb=gpu_memory_mb,
            energy_consumption=energy_consumption,
            flops=flops,
            parameters=parameters,
            throughput_fps=throughput_fps,
            cpu_utilization=cpu_util,
            gpu_utilization=gpu_util
        )
        
        logger.info(f"Profiling complete - Latency: {latency_ms:.2f}ms, Memory: {memory_mb:.1f}MB")
        return metrics
    
    def _warmup_model(self, model: nn.Module, dummy_input: torch.Tensor):
        """Warm up the model with several forward passes."""
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = model(dummy_input)
        
        # Synchronize GPU if available
        if self.gpu_available:
            torch.cuda.synchronize()
    
    def _profile_latency(self, model: nn.Module, dummy_input: torch.Tensor) -> float:
        """Profile model inference latency."""
        latencies = []
        
        with torch.no_grad():
            for _ in range(self.benchmark_runs):
                # Start timing
                if self.gpu_available:
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
                
                # Forward pass
                _ = model(dummy_input)
                
                # End timing
                if self.gpu_available:
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                latency = (end_time - start_time) * 1000  # Convert to milliseconds
                latencies.append(latency)
        
        # Return median latency for robustness
        return float(np.median(latencies))
    
    def _profile_memory(self, model: nn.Module, dummy_input: torch.Tensor) -> Tuple[float, float]:
        """Profile memory usage during inference."""
        # Clear memory
        if self.gpu_available:
            torch.cuda.empty_cache()
        gc.collect()
        
        # Measure memory before
        memory_before = psutil.virtual_memory().used / (1024 ** 2)  # MB
        gpu_memory_before = 0.0
        if self.gpu_available:
            gpu_memory_before = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        
        # Start memory monitoring
        self.memory_monitor.start_monitoring()
        
        # Run inference
        with torch.no_grad():
            _ = model(dummy_input)
        
        # Stop memory monitoring
        peak_gpu_memory = self.memory_monitor.stop_monitoring()
        
        # Measure memory after
        memory_after = psutil.virtual_memory().used / (1024 ** 2)  # MB
        gpu_memory_after = 0.0
        if self.gpu_available:
            gpu_memory_after = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        
        # Calculate memory usage
        memory_usage = memory_after - memory_before
        gpu_memory_usage = max(peak_gpu_memory, gpu_memory_after - gpu_memory_before)
        
        return max(0.0, memory_usage), max(0.0, gpu_memory_usage)
    
    def _count_parameters(self, model: nn.Module) -> int:
        """Count total number of parameters in the model."""
        return sum(p.numel() for p in model.parameters())
    
    def _estimate_flops(self, model: nn.Module, dummy_input: torch.Tensor) -> int:
        """Estimate FLOPs for the model (simplified calculation)."""
        total_flops = 0
        
        def flop_count_hook(module, input, output):
            nonlocal total_flops
            
            if isinstance(module, nn.Conv2d):
                # FLOPs for convolution: N * C_out * H_out * W_out * (C_in * K_h * K_w)
                batch_size, in_channels, in_height, in_width = input[0].shape
                out_channels = module.out_channels
                kernel_h, kernel_w = module.kernel_size
                
                if len(output.shape) == 4:
                    out_height, out_width = output.shape[2], output.shape[3]
                    flops = batch_size * out_channels * out_height * out_width * (in_channels * kernel_h * kernel_w)
                    total_flops += flops
                    
            elif isinstance(module, nn.Linear):
                # FLOPs for linear layer: N * input_features * output_features
                batch_size = input[0].shape[0]
                in_features = module.in_features
                out_features = module.out_features
                flops = batch_size * in_features * out_features
                total_flops += flops
        
        # Register hooks
        handles = []
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                handle = module.register_forward_hook(flop_count_hook)
                handles.append(handle)
        
        # Run forward pass
        with torch.no_grad():
            _ = model(dummy_input)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        return total_flops
    
    def _profile_utilization(self) -> Tuple[float, float]:
        """Profile CPU and GPU utilization."""
        # CPU utilization
        cpu_util = psutil.cpu_percent(interval=0.1)
        
        # GPU utilization
        gpu_util = 0.0
        if self.pynvml_ready:
            try:
                gpu_info = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_util = gpu_info.gpu
            except Exception as e:
                logger.debug(f"Failed to get GPU utilization: {e}")
        
        return cpu_util, gpu_util
    
    def _estimate_energy_consumption(self, latency_ms: float) -> float:
        """Estimate energy consumption (simplified model)."""
        # This is a very simplified energy estimation
        # In practice, you'd use actual power measurement tools
        
        base_power_watts = 10.0  # Base CPU power
        if self.gpu_available:
            base_power_watts += 150.0  # Add GPU power estimate
        
        # Energy = Power * Time
        energy_joules = base_power_watts * (latency_ms / 1000.0)
        return energy_joules
    
    def profile_architecture(self, architecture: Dict[str, Any]) -> HardwareMetrics:
        """
        Profile an architecture specification.
        
        Args:
            architecture: Architecture dictionary
            
        Returns:
            HardwareMetrics for the architecture
        """
        # Create a PyTorch model from architecture
        model = self._create_model_from_architecture(architecture)
        
        # Get input shape
        input_shape = architecture.get('input_shape', (3, 224, 224))
        
        # Profile the model
        return self.profile_model(model, input_shape)
    
    def _create_model_from_architecture(self, architecture: Dict[str, Any]) -> nn.Module:
        """Create a PyTorch model from architecture specification."""
        layers = architecture.get('layers', [])
        
        if not layers:
            raise ValueError("Architecture must have layers")
        
        # Create a more robust model with proper dimension handling
        class DynamicModel(nn.Module):
            def __init__(self, layer_specs, input_shape):
                super().__init__()
                self.layers = nn.ModuleList()
                self.input_shape = input_shape
                
                current_channels = input_shape[0]
                spatial_size = input_shape[1] * input_shape[2]  # H * W
                is_flattened = False
                
                for i, layer_spec in enumerate(layer_specs):
                    layer_type = layer_spec.get('type')
                    
                    if layer_type == 'conv2d':
                        if is_flattened:
                            # Can't add conv after flatten
                            continue
                        out_channels = layer_spec.get('out_channels', 64)
                        kernel_size = layer_spec.get('kernel_size', 3)
                        stride = layer_spec.get('stride', 1)
                        padding = layer_spec.get('padding', 1)
                        
                        layer = nn.Conv2d(current_channels, out_channels, kernel_size, 
                                        stride=stride, padding=padding)
                        self.layers.append(layer)
                        current_channels = out_channels
                        
                        # Update spatial size
                        spatial_size = spatial_size // (stride * stride)
                        
                    elif layer_type == 'batchnorm':
                        if is_flattened:
                            layer = nn.BatchNorm1d(current_channels)
                        else:
                            layer = nn.BatchNorm2d(current_channels)
                        self.layers.append(layer)
                        
                    elif layer_type == 'relu':
                        layer = nn.ReLU(inplace=layer_spec.get('inplace', False))
                        self.layers.append(layer)
                        
                    elif layer_type == 'gelu':
                        layer = nn.GELU()
                        self.layers.append(layer)
                        
                    elif layer_type == 'adaptive_pool':
                        if not is_flattened:
                            output_size = layer_spec.get('output_size', (1, 1))
                            layer = nn.AdaptiveAvgPool2d(output_size)
                            self.layers.append(layer)
                            spatial_size = output_size[0] * output_size[1]
                        
                    elif layer_type == 'flatten':
                        if not is_flattened:
                            layer = nn.Flatten()
                            self.layers.append(layer)
                            current_channels = current_channels * spatial_size
                            is_flattened = True
                        
                    elif layer_type == 'linear':
                        if not is_flattened:
                            # Add flatten before linear if not already flattened
                            self.layers.append(nn.Flatten())
                            current_channels = current_channels * spatial_size
                            is_flattened = True
                        
                        out_features = layer_spec.get('out_features', 10)
                        in_features = layer_spec.get('in_features', current_channels)
                        layer = nn.Linear(in_features, out_features)
                        self.layers.append(layer)
                        current_channels = out_features
                        
                    elif layer_type == 'dropout':
                        p = layer_spec.get('p', 0.5)
                        layer = nn.Dropout(p)
                        self.layers.append(layer)
            
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
        
        input_shape = architecture.get('input_shape', (3, 32, 32))
        return DynamicModel(layers, input_shape)
    
    def compare_architectures(
        self,
        architectures: List[Dict[str, Any]],
        names: Optional[List[str]] = None
    ) -> Dict[str, HardwareMetrics]:
        """
        Compare multiple architectures.
        
        Args:
            architectures: List of architecture specifications
            names: Optional names for architectures
            
        Returns:
            Dictionary mapping architecture names to metrics
        """
        results = {}
        
        if names is None:
            names = [f"Architecture_{i}" for i in range(len(architectures))]
        
        for arch, name in zip(architectures, names):
            logger.info(f"Profiling {name}...")
            try:
                metrics = self.profile_architecture(arch)
                results[name] = metrics
            except Exception as e:
                logger.error(f"Failed to profile {name}: {e}")
                # Create dummy metrics for failed architectures
                results[name] = HardwareMetrics(
                    latency_ms=float('inf'),
                    memory_mb=float('inf'),
                    gpu_memory_mb=float('inf'),
                    energy_consumption=float('inf'),
                    flops=0,
                    parameters=0,
                    throughput_fps=0.0,
                    cpu_utilization=0.0,
                    gpu_utilization=0.0
                )
        
        return results
    
    def get_hardware_constraints(self) -> Dict[str, Any]:
        """Get current hardware constraints and capabilities."""
        constraints = {
            'device': str(self.device),
            'gpu_available': self.gpu_available,
            'cpu_count': psutil.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / (1024 ** 3)
        }
        
        if self.gpu_available:
            constraints['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            constraints['gpu_name'] = torch.cuda.get_device_name(0)
        
        return constraints 