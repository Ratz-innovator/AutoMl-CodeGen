"""
Logging utilities for nanoNAS framework.

This module provides comprehensive logging infrastructure including
result tracking, experiment logging, and visualization utilities.
"""

import os
import json
import logging
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import csv

import torch
import numpy as np


class ResultLogger:
    """
    Comprehensive result logger for experiments.
    
    Logs training metrics, model performance, and experiment metadata
    with support for multiple output formats.
    """
    
    def __init__(self, log_dir: str, experiment_name: Optional[str] = None):
        """
        Initialize result logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment (auto-generated if None)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"
        
        self.experiment_name = experiment_name
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize log files
        self.metrics_file = self.experiment_dir / "metrics.json"
        self.csv_file = self.experiment_dir / "training_log.csv"
        self.config_file = self.experiment_dir / "config.json"
        
        # Storage for metrics
        self.epoch_metrics: List[Dict[str, Any]] = []
        self.best_metrics: Dict[str, Any] = {}
        self.experiment_metadata: Dict[str, Any] = {}
        
        # Initialize CSV file
        self._init_csv()
        
        logging.info(f"ResultLogger initialized - Experiment: {experiment_name}")
    
    def _init_csv(self):
        """Initialize CSV file with headers."""
        if not self.csv_file.exists():
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'epoch', 'train_loss', 'train_accuracy', 'val_loss', 
                    'val_accuracy', 'learning_rate', 'timestamp'
                ])
    
    def log_epoch(self, metrics: Dict[str, Any]):
        """
        Log metrics for an epoch.
        
        Args:
            metrics: Dictionary containing epoch metrics
        """
        # Add timestamp
        metrics['timestamp'] = datetime.now().isoformat()
        
        # Store in memory
        self.epoch_metrics.append(metrics.copy())
        
        # Update best metrics
        if 'val_accuracy' in metrics:
            if 'best_val_accuracy' not in self.best_metrics or \
               metrics['val_accuracy'] > self.best_metrics['best_val_accuracy']:
                self.best_metrics.update({
                    'best_val_accuracy': metrics['val_accuracy'],
                    'best_epoch': metrics.get('epoch', len(self.epoch_metrics)),
                    'best_train_accuracy': metrics.get('train_accuracy', 0.0),
                    'best_val_loss': metrics.get('val_loss', float('inf'))
                })
        
        # Write to CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics.get('epoch', ''),
                metrics.get('train_loss', ''),
                metrics.get('train_accuracy', ''),
                metrics.get('val_loss', ''),
                metrics.get('val_accuracy', ''),
                metrics.get('learning_rate', ''),
                metrics['timestamp']
            ])
        
        # Save JSON (overwrite each time for most recent state)
        self._save_json()
    
    def _save_json(self):
        """Save metrics to JSON file."""
        data = {
            'experiment_name': self.experiment_name,
            'epoch_metrics': self.epoch_metrics,
            'best_metrics': self.best_metrics,
            'metadata': self.experiment_metadata,
            'total_epochs': len(self.epoch_metrics)
        }
        
        with open(self.metrics_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def log_config(self, config: Any):
        """
        Log experiment configuration.
        
        Args:
            config: Configuration object (will be converted to dict)
        """
        if hasattr(config, '__dict__'):
            config_dict = config.__dict__
        elif hasattr(config, '_asdict'):  # namedtuple
            config_dict = config._asdict()
        elif isinstance(config, dict):
            config_dict = config
        else:
            config_dict = {'config': str(config)}
        
        # Make serializable
        serializable_config = self._make_serializable(config_dict)
        
        with open(self.config_file, 'w') as f:
            json.dump(serializable_config, f, indent=2)
        
        # Store in metadata
        self.experiment_metadata['config'] = serializable_config
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return str(obj)
    
    def log_model_info(self, model: torch.nn.Module):
        """
        Log model architecture information.
        
        Args:
            model: PyTorch model
        """
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'architecture': str(model)
        }
        
        self.experiment_metadata['model_info'] = model_info
        logging.info(f"Model info logged - Total params: {total_params:,}")
    
    def log_system_info(self):
        """Log system and environment information."""
        import platform
        import psutil
        
        system_info = {
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            system_info.update({
                'cuda_version': torch.version.cuda,
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3)
            })
        
        self.experiment_metadata['system_info'] = system_info
    
    def save_results(self, final_results: Dict[str, Any]):
        """
        Save final experiment results.
        
        Args:
            final_results: Dictionary containing final results
        """
        self.experiment_metadata['final_results'] = final_results
        self._save_json()
        
        # Create summary file
        summary_file = self.experiment_dir / "summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Experiment: {self.experiment_name}\\n")
            f.write(f"Completed: {datetime.now()}\\n\\n")
            
            f.write("=== Final Results ===\\n")
            for key, value in final_results.items():
                f.write(f"{key}: {value}\\n")
            
            f.write("\\n=== Best Metrics ===\\n")
            for key, value in self.best_metrics.items():
                f.write(f"{key}: {value}\\n")
        
        logging.info(f"Results saved to {self.experiment_dir}")
    
    def get_metrics_df(self):
        """
        Get metrics as pandas DataFrame.
        
        Returns:
            DataFrame with all epoch metrics
        """
        try:
            import pandas as pd
            return pd.read_csv(self.csv_file)
        except ImportError:
            logging.warning("pandas not available, returning raw metrics")
            return self.epoch_metrics


def setup_logging(
    log_level: str = 'INFO',
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional file to write logs to
        format_string: Custom format string
    
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=format_string,
        handlers=[]
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    
    # File handler (if specified)
    handlers = [console_handler]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    # Get logger
    logger = logging.getLogger('nanonas')
    logger.handlers = handlers
    logger.setLevel(getattr(logging, log_level.upper()))
    
    return logger


class MetricsTracker:
    """
    Simple metrics tracker for accumulating and averaging metrics.
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics: Dict[str, List[float]] = {}
    
    def update(self, **kwargs):
        """
        Update metrics with new values.
        
        Args:
            **kwargs: Metric name-value pairs
        """
        for name, value in kwargs.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(float(value))
    
    def avg(self, name: str) -> float:
        """
        Get average value for a metric.
        
        Args:
            name: Metric name
        
        Returns:
            Average value
        """
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0.0
        return sum(self.metrics[name]) / len(self.metrics[name])
    
    def last(self, name: str) -> float:
        """
        Get last value for a metric.
        
        Args:
            name: Metric name
        
        Returns:
            Last value
        """
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0.0
        return self.metrics[name][-1]
    
    def sum(self, name: str) -> float:
        """
        Get sum of values for a metric.
        
        Args:
            name: Metric name
        
        Returns:
            Sum of values
        """
        if name not in self.metrics:
            return 0.0
        return sum(self.metrics[name])
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all metrics.
        
        Returns:
            Dictionary with summary stats for each metric
        """
        summary = {}
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'last': values[-1],
                    'count': len(values)
                }
        return summary


class ProgressLogger:
    """
    Simple progress logger for long-running operations.
    """
    
    def __init__(self, total: int, name: str = "Progress"):
        """
        Initialize progress logger.
        
        Args:
            total: Total number of items to process
            name: Name of the operation
        """
        self.total = total
        self.name = name
        self.current = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
    
    def update(self, count: int = 1):
        """
        Update progress.
        
        Args:
            count: Number of items processed
        """
        self.current += count
        current_time = time.time()
        
        # Log every 10% or every 30 seconds
        progress_pct = (self.current / self.total) * 100
        time_since_last_log = current_time - self.last_log_time
        
        if progress_pct % 10 < (count / self.total) * 100 or time_since_last_log > 30:
            elapsed = current_time - self.start_time
            rate = self.current / elapsed if elapsed > 0 else 0
            eta = (self.total - self.current) / rate if rate > 0 else 0
            
            logging.info(
                f"{self.name}: {self.current}/{self.total} "
                f"({progress_pct:.1f}%) - "
                f"Rate: {rate:.2f}/s - "
                f"ETA: {eta:.1f}s"
            )
            
            self.last_log_time = current_time 