"""
Performance Profiling System for AutoML-CodeGen

This module provides comprehensive performance profiling capabilities including
timing, memory usage tracking, and GPU utilization monitoring.
"""

import time
import psutil
import threading
from contextlib import contextmanager
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path
import json

try:
    import pynvml
    NVML_AVAILABLE = True
    pynvml.nvmlInit()
except ImportError:
    NVML_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

@dataclass
class ProfileEntry:
    """Single profiling entry."""
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_start: Optional[float] = None
    memory_end: Optional[float] = None
    memory_peak: Optional[float] = None
    gpu_memory_start: Optional[Dict[int, float]] = None
    gpu_memory_end: Optional[Dict[int, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def finish(self, end_time: float, memory_end: float, gpu_memory_end: Optional[Dict[int, float]] = None) -> None:
        """Mark the profile entry as finished."""
        self.end_time = end_time
        self.duration = end_time - self.start_time
        self.memory_end = memory_end
        if gpu_memory_end:
            self.gpu_memory_end = gpu_memory_end

class PerformanceProfiler:
    """
    Comprehensive performance profiler for AutoML-CodeGen.
    
    Features:
    - Execution time profiling
    - Memory usage tracking (CPU and GPU)
    - Nested profiling contexts
    - Detailed performance reports
    - Real-time monitoring
    
    Example:
        >>> profiler = PerformanceProfiler()
        >>> with profiler.profile("training"):
        ...     # Training code here
        ...     pass
        >>> report = profiler.get_report()
        >>> print(report)
    """
    
    def __init__(self, enable_gpu_monitoring: bool = True, sampling_interval: float = 0.1):
        """
        Initialize performance profiler.
        
        Args:
            enable_gpu_monitoring: Whether to monitor GPU memory
            sampling_interval: Interval for continuous monitoring
        """
        self.enable_gpu_monitoring = enable_gpu_monitoring and NVML_AVAILABLE
        self.sampling_interval = sampling_interval
        
        # Profiling data
        self.entries: List[ProfileEntry] = []
        self.active_profiles: Dict[str, ProfileEntry] = {}
        self.profile_stack: List[str] = []
        
        # Monitoring
        self.monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_data: List[Dict[str, Any]] = []
        
        # GPU info
        self.gpu_count = 0
        if self.enable_gpu_monitoring:
            try:
                self.gpu_count = pynvml.nvmlDeviceGetCount()
            except:
                self.enable_gpu_monitoring = False
                self.gpu_count = 0
    
    @contextmanager
    def profile(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for profiling a code block.
        
        Args:
            name: Name of the profiling section
            metadata: Additional metadata to store
        """
        self.start_profile(name, metadata)
        try:
            yield
        finally:
            self.end_profile(name)
    
    def start_profile(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Start profiling a section."""
        if name in self.active_profiles:
            raise ValueError(f"Profile '{name}' is already active")
        
        # Get current memory usage
        memory_start = self._get_memory_usage()
        gpu_memory_start = self._get_gpu_memory_usage() if self.enable_gpu_monitoring else None
        
        # Create profile entry
        entry = ProfileEntry(
            name=name,
            start_time=time.time(),
            memory_start=memory_start,
            gpu_memory_start=gpu_memory_start,
            metadata=metadata or {}
        )
        
        self.active_profiles[name] = entry
        self.profile_stack.append(name)
    
    def end_profile(self, name: str) -> None:
        """End profiling a section."""
        if name not in self.active_profiles:
            raise ValueError(f"Profile '{name}' is not active")
        
        # Get current memory usage
        memory_end = self._get_memory_usage()
        gpu_memory_end = self._get_gpu_memory_usage() if self.enable_gpu_monitoring else None
        
        # Finish the profile entry
        entry = self.active_profiles[name]
        entry.finish(time.time(), memory_end, gpu_memory_end)
        
        # Calculate memory peak if we have torch
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                entry.memory_peak = torch.cuda.max_memory_allocated() / 1024**3  # GB
            except:
                pass
        
        # Move to completed entries
        self.entries.append(entry)
        del self.active_profiles[name]
        
        # Remove from stack
        if name in self.profile_stack:
            self.profile_stack.remove(name)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024**3  # Convert to GB
        except:
            return 0.0
    
    def _get_gpu_memory_usage(self) -> Optional[Dict[int, float]]:
        """Get GPU memory usage for all devices."""
        if not self.enable_gpu_monitoring:
            return None
        
        try:
            gpu_memory = {}
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory[i] = mem_info.used / 1024**3  # Convert to GB
            return gpu_memory
        except:
            return None
    
    def start_monitoring(self) -> None:
        """Start continuous performance monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitoring_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop continuous performance monitoring."""
        self.monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
    
    def _monitor_loop(self) -> None:
        """Monitoring loop that runs in a separate thread."""
        while self.monitoring:
            timestamp = time.time()
            cpu_percent = psutil.cpu_percent()
            memory_usage = self._get_memory_usage()
            
            data_point = {
                'timestamp': timestamp,
                'cpu_percent': cpu_percent,
                'memory_gb': memory_usage,
                'active_profiles': list(self.active_profiles.keys())
            }
            
            # Add GPU info if available
            if self.enable_gpu_monitoring:
                gpu_memory = self._get_gpu_memory_usage()
                if gpu_memory:
                    data_point['gpu_memory_gb'] = gpu_memory
                
                # Get GPU utilization
                try:
                    gpu_util = {}
                    for i in range(self.gpu_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_util[i] = util.gpu
                    data_point['gpu_utilization'] = gpu_util
                except:
                    pass
            
            self.monitoring_data.append(data_point)
            time.sleep(self.sampling_interval)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.entries:
            return {}
        
        # Calculate overall statistics
        total_time = sum(entry.duration for entry in self.entries if entry.duration)
        avg_time = total_time / len(self.entries) if self.entries else 0
        
        # Group by profile name
        by_name = {}
        for entry in self.entries:
            if entry.name not in by_name:
                by_name[entry.name] = []
            by_name[entry.name].append(entry)
        
        # Calculate statistics per profile name
        name_stats = {}
        for name, entries in by_name.items():
            durations = [e.duration for e in entries if e.duration]
            memory_usage = [e.memory_end - e.memory_start for e in entries 
                          if e.memory_end and e.memory_start]
            
            name_stats[name] = {
                'count': len(entries),
                'total_time': sum(durations),
                'avg_time': sum(durations) / len(durations) if durations else 0,
                'min_time': min(durations) if durations else 0,
                'max_time': max(durations) if durations else 0,
                'avg_memory_delta': sum(memory_usage) / len(memory_usage) if memory_usage else 0
            }
        
        summary = {
            'total_profiles': len(self.entries),
            'total_time': total_time,
            'avg_time_per_profile': avg_time,
            'by_name': name_stats
        }
        
        # Add monitoring summary if available
        if self.monitoring_data:
            cpu_usage = [d['cpu_percent'] for d in self.monitoring_data]
            memory_usage = [d['memory_gb'] for d in self.monitoring_data]
            
            summary['monitoring'] = {
                'samples': len(self.monitoring_data),
                'avg_cpu_percent': sum(cpu_usage) / len(cpu_usage),
                'max_cpu_percent': max(cpu_usage),
                'avg_memory_gb': sum(memory_usage) / len(memory_usage),
                'max_memory_gb': max(memory_usage)
            }
            
            # GPU monitoring summary
            if self.enable_gpu_monitoring and self.monitoring_data:
                gpu_data = [d for d in self.monitoring_data if 'gpu_memory_gb' in d]
                if gpu_data:
                    gpu_memory_all = []
                    for d in gpu_data:
                        gpu_memory_all.extend(d['gpu_memory_gb'].values())
                    
                    summary['monitoring']['avg_gpu_memory_gb'] = sum(gpu_memory_all) / len(gpu_memory_all)
                    summary['monitoring']['max_gpu_memory_gb'] = max(gpu_memory_all)
        
        return summary
    
    def get_report(self) -> str:
        """Generate a detailed performance report."""
        summary = self.get_summary()
        
        if not summary:
            return "No profiling data available."
        
        lines = []
        lines.append("=" * 60)
        lines.append("PERFORMANCE PROFILING REPORT")
        lines.append("=" * 60)
        
        # Overall statistics
        lines.append(f"Total Profiles: {summary['total_profiles']}")
        lines.append(f"Total Time: {summary['total_time']:.3f}s")
        lines.append(f"Average Time per Profile: {summary['avg_time_per_profile']:.3f}s")
        lines.append("")
        
        # Per-name statistics
        if 'by_name' in summary:
            lines.append("Profile Breakdown:")
            lines.append("-" * 40)
            for name, stats in summary['by_name'].items():
                lines.append(f"{name}:")
                lines.append(f"  Count: {stats['count']}")
                lines.append(f"  Total Time: {stats['total_time']:.3f}s")
                lines.append(f"  Average Time: {stats['avg_time']:.3f}s")
                lines.append(f"  Min Time: {stats['min_time']:.3f}s")
                lines.append(f"  Max Time: {stats['max_time']:.3f}s")
                lines.append(f"  Avg Memory Delta: {stats['avg_memory_delta']:.3f}GB")
                lines.append("")
        
        # Monitoring statistics
        if 'monitoring' in summary:
            mon = summary['monitoring']
            lines.append("System Monitoring:")
            lines.append("-" * 40)
            lines.append(f"Samples Collected: {mon['samples']}")
            lines.append(f"Average CPU Usage: {mon['avg_cpu_percent']:.1f}%")
            lines.append(f"Peak CPU Usage: {mon['max_cpu_percent']:.1f}%")
            lines.append(f"Average Memory Usage: {mon['avg_memory_gb']:.2f}GB")
            lines.append(f"Peak Memory Usage: {mon['max_memory_gb']:.2f}GB")
            
            if 'avg_gpu_memory_gb' in mon:
                lines.append(f"Average GPU Memory: {mon['avg_gpu_memory_gb']:.2f}GB")
                lines.append(f"Peak GPU Memory: {mon['max_gpu_memory_gb']:.2f}GB")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def save_report(self, filepath: Union[str, Path]) -> None:
        """Save performance report to file."""
        filepath = Path(filepath)
        
        # Save text report
        with open(filepath.with_suffix('.txt'), 'w') as f:
            f.write(self.get_report())
        
        # Save JSON data
        summary = self.get_summary()
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(summary, f, indent=2)
    
    def reset(self) -> None:
        """Reset all profiling data."""
        self.entries.clear()
        self.active_profiles.clear()
        self.profile_stack.clear()
        self.monitoring_data.clear()
    
    def get_active_profiles(self) -> List[str]:
        """Get list of currently active profiles."""
        return list(self.active_profiles.keys())
    
    def get_profile_stack(self) -> List[str]:
        """Get the current profile stack (nested profiles)."""
        return self.profile_stack.copy() 