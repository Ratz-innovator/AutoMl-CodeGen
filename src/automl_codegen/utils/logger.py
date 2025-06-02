"""
Advanced Logging System for AutoML-CodeGen

This module provides a sophisticated logging system with structured logging,
file rotation, and integration with monitoring systems like Weights & Biases.
"""

import logging
import logging.handlers
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
import sys

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def format(self, record):
        """Format log record with structured data."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_entry['data'] = record.extra_data
        
        return json.dumps(log_entry)

class Logger:
    """
    Advanced logger for AutoML-CodeGen with structured logging and monitoring integration.
    
    Features:
    - Structured JSON logging
    - File rotation
    - Multiple log levels and handlers
    - Integration with monitoring systems
    - Performance tracking
    
    Example:
        >>> logger = Logger(save_dir='./logs', use_wandb=True)
        >>> logger.info("Starting experiment", extra_data={'experiment_id': 123})
        >>> logger.log_metrics({'accuracy': 0.95, 'loss': 0.1})
    """
    
    def __init__(
        self,
        name: str = 'automl_codegen',
        save_dir: Union[str, Path] = './logs',
        level: str = 'INFO',
        use_file_handler: bool = True,
        use_console_handler: bool = True,
        use_structured_format: bool = False,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        use_wandb: bool = False,
        **kwargs
    ):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            save_dir: Directory to save log files
            level: Logging level
            use_file_handler: Whether to log to files
            use_console_handler: Whether to log to console
            use_structured_format: Whether to use JSON structured format
            max_file_size: Maximum file size before rotation
            backup_count: Number of backup files to keep
            use_wandb: Whether to integrate with Weights & Biases
            **kwargs: Additional configuration
        """
        self.name = name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup handlers
        if use_console_handler:
            self._setup_console_handler(use_structured_format)
        
        if use_file_handler:
            self._setup_file_handler(use_structured_format, max_file_size, backup_count)
        
        # Initialize metrics tracking
        self.metrics_history = []
        self.start_time = time.time()
        
        self.info(f"Logger initialized: {name}")
    
    def _setup_console_handler(self, structured: bool = False) -> None:
        """Setup console logging handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        
        if structured:
            console_handler.setFormatter(StructuredFormatter())
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(
        self,
        structured: bool = False,
        max_size: int = 10 * 1024 * 1024,
        backup_count: int = 5
    ) -> None:
        """Setup file logging handler with rotation."""
        log_file = self.save_dir / f"{self.name}.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_size,
            backupCount=backup_count
        )
        
        if structured:
            file_handler.setFormatter(StructuredFormatter())
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        
        # Also create a separate JSON log file for structured data
        json_file = self.save_dir / f"{self.name}_structured.jsonl"
        json_handler = logging.FileHandler(json_file)
        json_handler.setFormatter(StructuredFormatter())
        json_handler.setLevel(logging.INFO)
        self.logger.addHandler(json_handler)
    
    def debug(self, message: str, extra_data: Optional[Dict[str, Any]] = None) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, extra_data)
    
    def info(self, message: str, extra_data: Optional[Dict[str, Any]] = None) -> None:
        """Log info message."""
        self._log(logging.INFO, message, extra_data)
    
    def warning(self, message: str, extra_data: Optional[Dict[str, Any]] = None) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, extra_data)
    
    def error(self, message: str, extra_data: Optional[Dict[str, Any]] = None) -> None:
        """Log error message."""
        self._log(logging.ERROR, message, extra_data)
    
    def critical(self, message: str, extra_data: Optional[Dict[str, Any]] = None) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, message, extra_data)
    
    def _log(self, level: int, message: str, extra_data: Optional[Dict[str, Any]] = None) -> None:
        """Internal logging method."""
        record = self.logger.makeRecord(
            self.logger.name, level, __file__, 0, message, (), None
        )
        
        if extra_data:
            record.extra_data = extra_data
        
        self.logger.handle(record)
        
        # Log to wandb if enabled
        if self.use_wandb and level >= logging.INFO:
            try:
                import wandb
                log_data = {'log_level': logging.getLevelName(level), 'message': message}
                if extra_data:
                    log_data.update(extra_data)
                wandb.log(log_data)
            except ImportError:
                pass  # wandb not available
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = ""
    ) -> None:
        """
        Log metrics for monitoring.
        
        Args:
            metrics: Dictionary of metric values
            step: Optional step/iteration number
            prefix: Optional prefix for metric names
        """
        timestamp = time.time()
        
        # Format metrics for logging
        metric_strs = []
        wandb_metrics = {}
        
        for key, value in metrics.items():
            metric_name = f"{prefix}{key}" if prefix else key
            metric_strs.append(f"{metric_name}={value:.4f}")
            wandb_metrics[metric_name] = value
        
        message = f"Metrics: {', '.join(metric_strs)}"
        if step is not None:
            message += f" (step={step})"
        
        self.info(message, extra_data={'metrics': metrics, 'step': step})
        
        # Store metrics history
        self.metrics_history.append({
            'timestamp': timestamp,
            'step': step,
            'metrics': metrics
        })
        
        # Log to wandb
        if self.use_wandb:
            try:
                import wandb
                if step is not None:
                    wandb.log(wandb_metrics, step=step)
                else:
                    wandb.log(wandb_metrics)
            except ImportError:
                pass
    
    def log_architecture(
        self,
        architecture: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
        generation: Optional[int] = None
    ) -> None:
        """Log discovered architecture with its performance."""
        log_data = {
            'architecture': architecture,
            'generation': generation
        }
        
        if metrics:
            log_data['metrics'] = metrics
        
        message = f"Architecture discovered"
        if generation is not None:
            message += f" (generation {generation})"
        if metrics:
            accuracy = metrics.get('accuracy', 'N/A')
            latency = metrics.get('latency', 'N/A')
            message += f" - accuracy: {accuracy}, latency: {latency}"
        
        self.info(message, extra_data=log_data)
    
    def log_experiment_start(
        self,
        config: Dict[str, Any],
        experiment_name: Optional[str] = None
    ) -> None:
        """Log experiment start with configuration."""
        message = "Experiment started"
        if experiment_name:
            message += f": {experiment_name}"
        
        self.info(message, extra_data={
            'event': 'experiment_start',
            'config': config,
            'experiment_name': experiment_name
        })
    
    def log_experiment_end(
        self,
        results: Dict[str, Any],
        duration: Optional[float] = None
    ) -> None:
        """Log experiment completion with results."""
        if duration is None:
            duration = time.time() - self.start_time
        
        message = f"Experiment completed in {duration:.2f}s"
        
        self.info(message, extra_data={
            'event': 'experiment_end',
            'results': results,
            'duration': duration
        })
    
    def create_child_logger(self, suffix: str) -> 'Logger':
        """Create a child logger with a name suffix."""
        child_name = f"{self.name}.{suffix}"
        return Logger(
            name=child_name,
            save_dir=self.save_dir,
            use_wandb=self.use_wandb
        )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of logged metrics."""
        if not self.metrics_history:
            return {}
        
        # Calculate summary statistics
        all_metrics = {}
        for entry in self.metrics_history:
            for key, value in entry['metrics'].items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)
        
        summary = {}
        for key, values in all_metrics.items():
            summary[key] = {
                'count': len(values),
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'latest': values[-1]
            }
        
        return summary
    
    def flush(self) -> None:
        """Flush all handlers."""
        for handler in self.logger.handlers:
            handler.flush()
    
    def close(self) -> None:
        """Close all handlers."""
        for handler in self.logger.handlers:
            handler.close()
        self.logger.handlers.clear() 