"""
Performance profiling and monitoring utilities for Neural Architecture Search.

This module provides comprehensive profiling capabilities including:
- Real-time performance monitoring
- Memory usage tracking
- Training dynamics analysis
- Search process profiling
- Bottleneck identification
- Performance regression detection
"""

import time
import psutil
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from collections import defaultdict, deque
import threading
import functools
import traceback
from contextlib import contextmanager
import pickle
import warnings
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class ProfilerConfig:
    """Configuration for performance profiler."""
    
    enable_memory_profiling: bool = True
    enable_timing_profiling: bool = True
    enable_gpu_profiling: bool = True
    enable_io_profiling: bool = True
    sampling_interval: float = 0.1  # seconds
    max_samples: int = 10000
    profile_gradients: bool = False
    profile_activations: bool = False
    save_traces: bool = True
    trace_directory: str = "traces"
    auto_save_interval: int = 100  # samples


@dataclass
class PerformanceSnapshot:
    """Single performance measurement snapshot."""
    
    timestamp: float
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    gpu_memory_mb: float = 0.0
    gpu_utilization: float = 0.0
    gpu_temperature: float = 0.0
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_io_sent_mb: float = 0.0
    network_io_recv_mb: float = 0.0
    process_threads: int = 0
    open_files: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'cpu_percent': self.cpu_percent,
            'memory_mb': self.memory_mb,
            'memory_percent': self.memory_percent,
            'gpu_memory_mb': self.gpu_memory_mb,
            'gpu_utilization': self.gpu_utilization,
            'gpu_temperature': self.gpu_temperature,
            'disk_io_read_mb': self.disk_io_read_mb,
            'disk_io_write_mb': self.disk_io_write_mb,
            'network_io_sent_mb': self.network_io_sent_mb,
            'network_io_recv_mb': self.network_io_recv_mb,
            'process_threads': self.process_threads,
            'open_files': self.open_files
        }


@dataclass
class FunctionProfile:
    """Profile data for a specific function."""
    
    name: str
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    memory_usage: List[float] = field(default_factory=list)
    gpu_memory_usage: List[float] = field(default_factory=list)
    exceptions: List[str] = field(default_factory=list)
    
    def update(self, execution_time: float, memory_mb: float = 0.0, gpu_memory_mb: float = 0.0):
        """Update profile with new measurement."""
        self.call_count += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.avg_time = self.total_time / self.call_count
        
        if memory_mb > 0:
            self.memory_usage.append(memory_mb)
        if gpu_memory_mb > 0:
            self.gpu_memory_usage.append(gpu_memory_mb)
    
    def add_exception(self, exception: str):
        """Add exception to profile."""
        self.exceptions.append(exception)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            'name': self.name,
            'call_count': self.call_count,
            'total_time': self.total_time,
            'min_time': self.min_time if self.min_time != float('inf') else 0.0,
            'max_time': self.max_time,
            'avg_time': self.avg_time,
            'memory_stats': {
                'mean': np.mean(self.memory_usage) if self.memory_usage else 0.0,
                'std': np.std(self.memory_usage) if self.memory_usage else 0.0,
                'max': np.max(self.memory_usage) if self.memory_usage else 0.0,
            },
            'gpu_memory_stats': {
                'mean': np.mean(self.gpu_memory_usage) if self.gpu_memory_usage else 0.0,
                'std': np.std(self.gpu_memory_usage) if self.gpu_memory_usage else 0.0,
                'max': np.max(self.gpu_memory_usage) if self.gpu_memory_usage else 0.0,
            },
            'exception_count': len(self.exceptions),
            'exceptions': self.exceptions[-5:]  # Last 5 exceptions
        }


class PerformanceProfiler:
    """Comprehensive performance profiler for NAS systems."""
    
    def __init__(self, config: Optional[ProfilerConfig] = None):
        self.config = config or ProfilerConfig()
        self.is_profiling = False
        self.snapshots: deque = deque(maxlen=self.config.max_samples)
        self.function_profiles: Dict[str, FunctionProfile] = {}
        self.monitoring_thread: Optional[threading.Thread] = None
        self.start_time: Optional[float] = None
        self.process = psutil.Process()
        
        # Initialize baseline measurements
        self.baseline_io = self._get_io_stats()
        self.baseline_network = self._get_network_stats()
        
        # Create trace directory
        Path(self.config.trace_directory).mkdir(parents=True, exist_ok=True)
    
    def start_profiling(self):
        """Start performance profiling."""
        if self.is_profiling:
            logger.warning("Profiler is already running")
            return
        
        self.is_profiling = True
        self.start_time = time.time()
        self.snapshots.clear()
        
        if self.config.enable_memory_profiling or self.config.enable_gpu_profiling:
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self.monitoring_thread.start()
        
        logger.info("Performance profiling started")
    
    def stop_profiling(self) -> Dict[str, Any]:
        """Stop profiling and return results."""
        if not self.is_profiling:
            logger.warning("Profiler is not running")
            return {}
        
        self.is_profiling = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        
        results = self._generate_report()
        
        if self.config.save_traces:
            self._save_traces(results)
        
        logger.info("Performance profiling stopped")
        return results
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        sample_count = 0
        
        while self.is_profiling:
            try:
                snapshot = self._take_snapshot()
                self.snapshots.append(snapshot)
                
                sample_count += 1
                if (sample_count % self.config.auto_save_interval == 0 and 
                    self.config.save_traces):
                    self._auto_save_checkpoint()
                
                time.sleep(self.config.sampling_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(self.config.sampling_interval)
    
    def _take_snapshot(self) -> PerformanceSnapshot:
        """Take a performance snapshot."""
        snapshot = PerformanceSnapshot(timestamp=time.time())
        
        try:
            # CPU and memory
            snapshot.cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            snapshot.memory_mb = memory_info.rss / (1024 * 1024)
            snapshot.memory_percent = self.process.memory_percent()
            
            # Process info
            snapshot.process_threads = self.process.num_threads()
            snapshot.open_files = len(self.process.open_files())
            
            # I/O stats
            if self.config.enable_io_profiling:
                io_stats = self._get_io_stats()
                if io_stats and self.baseline_io:
                    snapshot.disk_io_read_mb = (io_stats.read_bytes - self.baseline_io.read_bytes) / (1024 * 1024)
                    snapshot.disk_io_write_mb = (io_stats.write_bytes - self.baseline_io.write_bytes) / (1024 * 1024)
                
                network_stats = self._get_network_stats()
                if network_stats and self.baseline_network:
                    snapshot.network_io_sent_mb = (network_stats.bytes_sent - self.baseline_network.bytes_sent) / (1024 * 1024)
                    snapshot.network_io_recv_mb = (network_stats.bytes_recv - self.baseline_network.bytes_recv) / (1024 * 1024)
            
            # GPU stats
            if self.config.enable_gpu_profiling and torch.cuda.is_available():
                snapshot.gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                
                try:
                    # Try to get GPU utilization (requires nvidia-ml-py)
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    snapshot.gpu_utilization = utilization.gpu
                    
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    snapshot.gpu_temperature = temperature
                except ImportError:
                    pass  # pynvml not available
                except Exception as e:
                    logger.debug(f"GPU monitoring error: {e}")
        
        except Exception as e:
            logger.error(f"Snapshot error: {e}")
        
        return snapshot
    
    def _get_io_stats(self):
        """Get I/O statistics."""
        try:
            return self.process.io_counters()
        except (AttributeError, OSError):
            return None
    
    def _get_network_stats(self):
        """Get network statistics."""
        try:
            return psutil.net_io_counters()
        except (AttributeError, OSError):
            return None
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile function execution."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self.is_profiling:
                return func(*args, **kwargs)
            
            func_name = f"{func.__module__}.{func.__name__}"
            
            # Get initial memory
            initial_memory = self.process.memory_info().rss / (1024 * 1024)
            initial_gpu_memory = 0.0
            if torch.cuda.is_available():
                initial_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Calculate metrics
                execution_time = time.time() - start_time
                final_memory = self.process.memory_info().rss / (1024 * 1024)
                memory_delta = final_memory - initial_memory
                
                gpu_memory_delta = 0.0
                if torch.cuda.is_available():
                    final_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                    gpu_memory_delta = final_gpu_memory - initial_gpu_memory
                
                # Update profile
                if func_name not in self.function_profiles:
                    self.function_profiles[func_name] = FunctionProfile(func_name)
                
                self.function_profiles[func_name].update(
                    execution_time, memory_delta, gpu_memory_delta
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                if func_name not in self.function_profiles:
                    self.function_profiles[func_name] = FunctionProfile(func_name)
                
                self.function_profiles[func_name].update(execution_time)
                self.function_profiles[func_name].add_exception(str(e))
                
                raise
        
        return wrapper
    
    @contextmanager
    def profile_block(self, block_name: str):
        """Context manager to profile a code block."""
        if not self.is_profiling:
            yield
            return
        
        initial_memory = self.process.memory_info().rss / (1024 * 1024)
        initial_gpu_memory = 0.0
        if torch.cuda.is_available():
            initial_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        
        start_time = time.time()
        
        try:
            yield
            
            execution_time = time.time() - start_time
            final_memory = self.process.memory_info().rss / (1024 * 1024)
            memory_delta = final_memory - initial_memory
            
            gpu_memory_delta = 0.0
            if torch.cuda.is_available():
                final_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                gpu_memory_delta = final_gpu_memory - initial_gpu_memory
            
            if block_name not in self.function_profiles:
                self.function_profiles[block_name] = FunctionProfile(block_name)
            
            self.function_profiles[block_name].update(
                execution_time, memory_delta, gpu_memory_delta
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            if block_name not in self.function_profiles:
                self.function_profiles[block_name] = FunctionProfile(block_name)
            
            self.function_profiles[block_name].update(execution_time)
            self.function_profiles[block_name].add_exception(str(e))
            
            raise
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.snapshots:
            return {}
        
        # Convert snapshots to arrays for analysis
        timestamps = [s.timestamp for s in self.snapshots]
        cpu_usage = [s.cpu_percent for s in self.snapshots]
        memory_usage = [s.memory_mb for s in self.snapshots]
        gpu_memory_usage = [s.gpu_memory_mb for s in self.snapshots]
        
        # Calculate statistics
        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        
        report = {
            'profiling_duration': duration,
            'total_samples': len(self.snapshots),
            'sampling_rate': len(self.snapshots) / duration if duration > 0 else 0,
            
            'cpu_stats': {
                'mean': np.mean(cpu_usage),
                'std': np.std(cpu_usage),
                'min': np.min(cpu_usage),
                'max': np.max(cpu_usage),
                'p95': np.percentile(cpu_usage, 95),
                'p99': np.percentile(cpu_usage, 99)
            },
            
            'memory_stats': {
                'mean_mb': np.mean(memory_usage),
                'std_mb': np.std(memory_usage),
                'min_mb': np.min(memory_usage),
                'max_mb': np.max(memory_usage),
                'peak_mb': np.max(memory_usage),
                'growth_mb': memory_usage[-1] - memory_usage[0] if len(memory_usage) > 1 else 0
            },
            
            'gpu_memory_stats': {
                'mean_mb': np.mean(gpu_memory_usage),
                'std_mb': np.std(gpu_memory_usage),
                'min_mb': np.min(gpu_memory_usage),
                'max_mb': np.max(gpu_memory_usage),
                'peak_mb': np.max(gpu_memory_usage),
                'growth_mb': gpu_memory_usage[-1] - gpu_memory_usage[0] if len(gpu_memory_usage) > 1 else 0
            },
            
            'function_profiles': {
                name: profile.get_statistics() 
                for name, profile in self.function_profiles.items()
            },
            
            'bottlenecks': self._identify_bottlenecks(),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        # Function-level bottlenecks
        sorted_functions = sorted(
            self.function_profiles.items(),
            key=lambda x: x[1].total_time,
            reverse=True
        )
        
        for name, profile in sorted_functions[:5]:  # Top 5 time consumers
            if profile.total_time > 1.0:  # More than 1 second total
                bottlenecks.append({
                    'type': 'function',
                    'name': name,
                    'total_time': profile.total_time,
                    'avg_time': profile.avg_time,
                    'call_count': profile.call_count,
                    'severity': 'high' if profile.avg_time > 0.1 else 'medium'
                })
        
        # Memory bottlenecks
        if self.snapshots:
            memory_usage = [s.memory_mb for s in self.snapshots]
            memory_growth = memory_usage[-1] - memory_usage[0] if len(memory_usage) > 1 else 0
            
            if memory_growth > 1000:  # More than 1GB growth
                bottlenecks.append({
                    'type': 'memory_leak',
                    'growth_mb': memory_growth,
                    'severity': 'high' if memory_growth > 5000 else 'medium'
                })
            
            peak_memory = max(memory_usage)
            if peak_memory > 8000:  # More than 8GB peak usage
                bottlenecks.append({
                    'type': 'high_memory_usage',
                    'peak_mb': peak_memory,
                    'severity': 'high' if peak_memory > 16000 else 'medium'
                })
        
        return bottlenecks
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        bottlenecks = self._identify_bottlenecks()
        
        for bottleneck in bottlenecks:
            if bottleneck['type'] == 'function':
                if bottleneck['avg_time'] > 0.1:
                    recommendations.append(
                        f"Optimize function '{bottleneck['name']}' - "
                        f"average execution time is {bottleneck['avg_time']:.3f}s"
                    )
            
            elif bottleneck['type'] == 'memory_leak':
                recommendations.append(
                    f"Investigate memory leak - growth of {bottleneck['growth_mb']:.1f}MB detected"
                )
            
            elif bottleneck['type'] == 'high_memory_usage':
                recommendations.append(
                    f"Consider memory optimization - peak usage is {bottleneck['peak_mb']:.1f}MB"
                )
        
        # General recommendations based on patterns
        if self.snapshots:
            cpu_usage = [s.cpu_percent for s in self.snapshots]
            avg_cpu = np.mean(cpu_usage)
            
            if avg_cpu < 20:
                recommendations.append("Low CPU utilization detected - consider increasing parallelism")
            elif avg_cpu > 90:
                recommendations.append("High CPU utilization detected - consider reducing computational load")
        
        return recommendations
    
    def _save_traces(self, report: Dict[str, Any]):
        """Save profiling traces to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save report
        report_path = Path(self.config.trace_directory) / f"profile_report_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save raw snapshots
        snapshots_path = Path(self.config.trace_directory) / f"snapshots_{timestamp}.pkl"
        with open(snapshots_path, 'wb') as f:
            pickle.dump(list(self.snapshots), f)
        
        logger.info(f"Profiling traces saved to {self.config.trace_directory}")
    
    def _auto_save_checkpoint(self):
        """Auto-save checkpoint during profiling."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = Path(self.config.trace_directory) / f"checkpoint_{timestamp}.pkl"
            
            checkpoint_data = {
                'snapshots': list(self.snapshots),
                'function_profiles': self.function_profiles,
                'start_time': self.start_time
            }
            
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
                
        except Exception as e:
            logger.error(f"Auto-save checkpoint failed: {e}")
    
    def visualize_performance(self, save_path: Optional[str] = None) -> None:
        """Create performance visualization plots."""
        if not self.snapshots:
            logger.warning("No profiling data available for visualization")
            return
        
        # Prepare data
        timestamps = [(s.timestamp - self.snapshots[0].timestamp) / 60 for s in self.snapshots]  # Minutes
        cpu_usage = [s.cpu_percent for s in self.snapshots]
        memory_usage = [s.memory_mb for s in self.snapshots]
        gpu_memory_usage = [s.gpu_memory_mb for s in self.snapshots]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Profiling Results', fontsize=16)
        
        # CPU Usage
        axes[0, 0].plot(timestamps, cpu_usage, 'b-', alpha=0.7)
        axes[0, 0].set_title('CPU Usage Over Time')
        axes[0, 0].set_xlabel('Time (minutes)')
        axes[0, 0].set_ylabel('CPU Usage (%)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Memory Usage
        axes[0, 1].plot(timestamps, memory_usage, 'r-', alpha=0.7)
        axes[0, 1].set_title('Memory Usage Over Time')
        axes[0, 1].set_xlabel('Time (minutes)')
        axes[0, 1].set_ylabel('Memory Usage (MB)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # GPU Memory Usage
        axes[1, 0].plot(timestamps, gpu_memory_usage, 'g-', alpha=0.7)
        axes[1, 0].set_title('GPU Memory Usage Over Time')
        axes[1, 0].set_xlabel('Time (minutes)')
        axes[1, 0].set_ylabel('GPU Memory Usage (MB)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Function Performance (Top 10)
        if self.function_profiles:
            sorted_functions = sorted(
                self.function_profiles.items(),
                key=lambda x: x[1].total_time,
                reverse=True
            )[:10]
            
            names = [name.split('.')[-1] for name, _ in sorted_functions]  # Short names
            times = [profile.total_time for _, profile in sorted_functions]
            
            axes[1, 1].barh(names, times)
            axes[1, 1].set_title('Top 10 Functions by Total Time')
            axes[1, 1].set_xlabel('Total Time (seconds)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance visualization saved to {save_path}")
        else:
            plt.show()
    
    def get_memory_timeline(self) -> Tuple[List[float], List[float]]:
        """Get memory usage timeline."""
        if not self.snapshots:
            return [], []
        
        timestamps = [(s.timestamp - self.snapshots[0].timestamp) for s in self.snapshots]
        memory_usage = [s.memory_mb for s in self.snapshots]
        
        return timestamps, memory_usage
    
    def detect_memory_leaks(self, threshold_mb: float = 100.0) -> List[Dict[str, Any]]:
        """Detect potential memory leaks."""
        if len(self.snapshots) < 10:
            return []
        
        memory_usage = [s.memory_mb for s in self.snapshots]
        timestamps = [s.timestamp for s in self.snapshots]
        
        # Calculate memory growth rate
        window_size = min(len(memory_usage) // 4, 100)  # 25% of data or 100 samples
        growth_rates = []
        
        for i in range(window_size, len(memory_usage)):
            start_idx = i - window_size
            start_memory = np.mean(memory_usage[start_idx:start_idx + 5])
            end_memory = np.mean(memory_usage[i-5:i])
            time_diff = timestamps[i] - timestamps[start_idx]
            
            if time_diff > 0:
                growth_rate = (end_memory - start_memory) / time_diff  # MB/second
                growth_rates.append(growth_rate)
        
        # Identify sustained growth periods
        leaks = []
        if growth_rates:
            avg_growth_rate = np.mean(growth_rates)
            if avg_growth_rate > threshold_mb / 3600:  # threshold per hour converted to per second
                leaks.append({
                    'type': 'sustained_growth',
                    'growth_rate_mb_per_hour': avg_growth_rate * 3600,
                    'total_growth_mb': memory_usage[-1] - memory_usage[0],
                    'severity': 'high' if avg_growth_rate * 3600 > threshold_mb else 'medium'
                })
        
        return leaks


class SearchProfiler:
    """Specialized profiler for NAS search processes."""
    
    def __init__(self, profiler: PerformanceProfiler):
        self.profiler = profiler
        self.search_metrics: Dict[str, List[float]] = defaultdict(list)
        self.generation_times: List[float] = []
        self.evaluation_times: List[float] = []
        self.architecture_complexities: List[Dict[str, float]] = []
    
    def log_generation(self, generation: int, best_fitness: float, avg_fitness: float,
                      diversity: float, generation_time: float):
        """Log search generation metrics."""
        self.search_metrics['generation'].append(generation)
        self.search_metrics['best_fitness'].append(best_fitness)
        self.search_metrics['avg_fitness'].append(avg_fitness)
        self.search_metrics['diversity'].append(diversity)
        self.generation_times.append(generation_time)
    
    def log_evaluation(self, architecture_id: str, fitness: float, evaluation_time: float,
                      complexity_metrics: Optional[Dict[str, float]] = None):
        """Log architecture evaluation metrics."""
        self.evaluation_times.append(evaluation_time)
        
        if complexity_metrics:
            complexity_metrics['architecture_id'] = architecture_id
            complexity_metrics['fitness'] = fitness
            complexity_metrics['evaluation_time'] = evaluation_time
            self.architecture_complexities.append(complexity_metrics)
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get comprehensive search statistics."""
        stats = {
            'total_generations': len(self.generation_times),
            'total_evaluations': len(self.evaluation_times),
            'total_search_time': sum(self.generation_times),
            'avg_generation_time': np.mean(self.generation_times) if self.generation_times else 0,
            'avg_evaluation_time': np.mean(self.evaluation_times) if self.evaluation_times else 0,
        }
        
        if self.search_metrics['best_fitness']:
            stats.update({
                'final_best_fitness': self.search_metrics['best_fitness'][-1],
                'fitness_improvement': (
                    self.search_metrics['best_fitness'][-1] - self.search_metrics['best_fitness'][0]
                    if len(self.search_metrics['best_fitness']) > 1 else 0
                ),
                'convergence_rate': self._calculate_convergence_rate()
            })
        
        if self.architecture_complexities:
            complexities = [arch['fitness'] for arch in self.architecture_complexities]
            stats['architecture_stats'] = {
                'best_architecture_fitness': max(complexities),
                'worst_architecture_fitness': min(complexities),
                'avg_architecture_fitness': np.mean(complexities),
                'fitness_std': np.std(complexities)
            }
        
        return stats
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate search convergence rate."""
        if len(self.search_metrics['best_fitness']) < 2:
            return 0.0
        
        fitness_values = self.search_metrics['best_fitness']
        improvements = []
        
        for i in range(1, len(fitness_values)):
            improvement = fitness_values[i] - fitness_values[i-1]
            improvements.append(improvement)
        
        # Calculate rate of improvement decay
        if len(improvements) > 5:
            recent_improvements = improvements[-5:]
            early_improvements = improvements[:5]
            
            recent_avg = np.mean(recent_improvements)
            early_avg = np.mean(early_improvements)
            
            if early_avg > 0:
                convergence_rate = recent_avg / early_avg
            else:
                convergence_rate = 0.0
        else:
            convergence_rate = 0.0
        
        return convergence_rate
    
    def visualize_search_progress(self, save_path: Optional[str] = None):
        """Visualize search progress."""
        if not self.search_metrics['generation']:
            logger.warning("No search data available for visualization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('NAS Search Progress', fontsize=16)
        
        generations = self.search_metrics['generation']
        
        # Fitness evolution
        axes[0, 0].plot(generations, self.search_metrics['best_fitness'], 'b-', label='Best Fitness')
        axes[0, 0].plot(generations, self.search_metrics['avg_fitness'], 'r--', label='Avg Fitness')
        axes[0, 0].set_title('Fitness Evolution')
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Fitness')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Diversity
        axes[0, 1].plot(generations, self.search_metrics['diversity'], 'g-')
        axes[0, 1].set_title('Population Diversity')
        axes[0, 1].set_xlabel('Generation')
        axes[0, 1].set_ylabel('Diversity')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Generation times
        axes[1, 0].plot(generations, self.generation_times, 'orange')
        axes[1, 0].set_title('Generation Times')
        axes[1, 0].set_xlabel('Generation')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Evaluation time distribution
        if self.evaluation_times:
            axes[1, 1].hist(self.evaluation_times, bins=30, alpha=0.7, color='purple')
            axes[1, 1].set_title('Evaluation Time Distribution')
            axes[1, 1].set_xlabel('Evaluation Time (seconds)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Search progress visualization saved to {save_path}")
        else:
            plt.show()


# Global profiler instance
_global_profiler: Optional[PerformanceProfiler] = None


def get_profiler(config: Optional[ProfilerConfig] = None) -> PerformanceProfiler:
    """Get global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler(config)
    return _global_profiler


def profile_function(func: Callable) -> Callable:
    """Decorator to profile function execution."""
    profiler = get_profiler()
    return profiler.profile_function(func)


@contextmanager
def profile_block(block_name: str):
    """Context manager to profile a code block."""
    profiler = get_profiler()
    with profiler.profile_block(block_name):
        yield


def start_profiling(config: Optional[ProfilerConfig] = None):
    """Start global profiling."""
    profiler = get_profiler(config)
    profiler.start_profiling()


def stop_profiling() -> Dict[str, Any]:
    """Stop global profiling and return results."""
    profiler = get_profiler()
    return profiler.stop_profiling()


def benchmark_function(func: Callable, *args, num_runs: int = 10, **kwargs) -> Dict[str, float]:
    """Benchmark a function with multiple runs."""
    times = []
    memory_usage = []
    
    process = psutil.Process()
    
    for _ in range(num_runs):
        # Measure memory before
        initial_memory = process.memory_info().rss / (1024 * 1024)
        
        # Time the function
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Measure memory after
        final_memory = process.memory_info().rss / (1024 * 1024)
        
        times.append(end_time - start_time)
        memory_usage.append(final_memory - initial_memory)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'mean_memory_delta': np.mean(memory_usage),
        'max_memory_delta': np.max(memory_usage)
    } 