"""
Production deployment utilities for Neural Architecture Search.

This module provides comprehensive deployment capabilities including:
- Model serving and inference optimization
- Model quantization and pruning
- Cloud deployment automation
- Production monitoring and logging
- A/B testing framework
- Model versioning and rollback
"""

import torch
import torch.nn as nn
import torch.quantization as quant
import torch.jit as jit
import onnx
import onnxruntime as ort
import numpy as np
import json
import time
import logging
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import hashlib
import threading
from collections import defaultdict, deque
import psutil
import subprocess
from abc import ABC, abstractmethod
import yaml
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""
    
    # Model optimization
    enable_quantization: bool = True
    quantization_backend: str = "fbgemm"  # fbgemm, qnnpack
    enable_pruning: bool = False
    pruning_sparsity: float = 0.5
    enable_jit_compilation: bool = True
    enable_onnx_export: bool = True
    
    # Serving configuration
    serving_framework: str = "torchserve"  # torchserve, triton, custom
    batch_size: int = 1
    max_batch_delay_ms: int = 100
    max_concurrent_requests: int = 100
    timeout_seconds: int = 30
    
    # Performance optimization
    enable_tensorrt: bool = False
    enable_openvino: bool = False
    target_device: str = "cpu"  # cpu, cuda, tensorrt
    precision: str = "fp32"  # fp32, fp16, int8
    
    # Monitoring
    enable_monitoring: bool = True
    metrics_collection_interval: int = 60  # seconds
    log_level: str = "INFO"
    enable_health_checks: bool = True
    
    # A/B Testing
    enable_ab_testing: bool = False
    traffic_split: Dict[str, float] = field(default_factory=lambda: {"model_a": 0.5, "model_b": 0.5})
    
    # Cloud deployment
    cloud_provider: str = "aws"  # aws, gcp, azure
    container_registry: str = ""
    deployment_environment: str = "staging"  # staging, production
    auto_scaling: bool = True
    min_replicas: int = 1
    max_replicas: int = 10


class ModelOptimizer:
    """Model optimization for production deployment."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
    
    def optimize_model(self, model: nn.Module, sample_input: torch.Tensor) -> nn.Module:
        """Apply comprehensive model optimization."""
        optimized_model = model
        
        # Apply quantization
        if self.config.enable_quantization:
            optimized_model = self.quantize_model(optimized_model, sample_input)
        
        # Apply pruning
        if self.config.enable_pruning:
            optimized_model = self.prune_model(optimized_model)
        
        # Apply JIT compilation
        if self.config.enable_jit_compilation:
            optimized_model = self.jit_compile_model(optimized_model, sample_input)
        
        return optimized_model
    
    def quantize_model(self, model: nn.Module, sample_input: torch.Tensor) -> nn.Module:
        """Apply post-training quantization."""
        logger.info("Applying post-training quantization")
        
        # Prepare model for quantization
        model.eval()
        model.qconfig = quant.get_default_qconfig(self.config.quantization_backend)
        
        # Prepare the model
        prepared_model = quant.prepare(model, inplace=False)
        
        # Calibrate with sample data
        with torch.no_grad():
            prepared_model(sample_input)
        
        # Convert to quantized model
        quantized_model = quant.convert(prepared_model, inplace=False)
        
        logger.info("Model quantization completed")
        return quantized_model
    
    def prune_model(self, model: nn.Module) -> nn.Module:
        """Apply structured pruning to model."""
        logger.info(f"Applying pruning with sparsity {self.config.pruning_sparsity}")
        
        import torch.nn.utils.prune as prune
        
        # Apply global magnitude pruning
        parameters_to_prune = []
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
        
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.config.pruning_sparsity,
        )
        
        # Remove pruning reparameterization
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        logger.info("Model pruning completed")
        return model
    
    def jit_compile_model(self, model: nn.Module, sample_input: torch.Tensor) -> jit.ScriptModule:
        """Apply TorchScript JIT compilation."""
        logger.info("Applying JIT compilation")
        
        model.eval()
        with torch.no_grad():
            traced_model = jit.trace(model, sample_input)
        
        # Optimize the traced model
        optimized_model = jit.optimize_for_inference(traced_model)
        
        logger.info("JIT compilation completed")
        return optimized_model
    
    def export_onnx(self, model: nn.Module, sample_input: torch.Tensor, 
                   output_path: str) -> str:
        """Export model to ONNX format."""
        logger.info(f"Exporting model to ONNX: {output_path}")
        
        model.eval()
        torch.onnx.export(
            model,
            sample_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify the exported model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        logger.info("ONNX export completed")
        return output_path
    
    def optimize_onnx_model(self, onnx_path: str) -> str:
        """Optimize ONNX model for inference."""
        import onnxoptimizer
        
        logger.info("Optimizing ONNX model")
        
        # Load ONNX model
        model = onnx.load(onnx_path)
        
        # Apply optimizations
        optimized_model = onnxoptimizer.optimize(model)
        
        # Save optimized model
        optimized_path = onnx_path.replace('.onnx', '_optimized.onnx')
        onnx.save(optimized_model, optimized_path)
        
        logger.info(f"Optimized ONNX model saved to {optimized_path}")
        return optimized_path


class InferenceEngine:
    """High-performance inference engine."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.model = None
        self.session = None
        self.device = torch.device(config.target_device)
        
        # Performance tracking
        self.inference_times = deque(maxlen=1000)
        self.request_count = 0
        self.error_count = 0
        
    def load_model(self, model_path: str, model_type: str = "pytorch"):
        """Load model for inference."""
        if model_type == "pytorch":
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
        elif model_type == "onnx":
            providers = ['CPUExecutionProvider']
            if self.config.target_device == "cuda":
                providers.insert(0, 'CUDAExecutionProvider')
            
            self.session = ort.InferenceSession(model_path, providers=providers)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        logger.info(f"Model loaded from {model_path}")
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on input data."""
        start_time = time.time()
        
        try:
            if self.model is not None:
                # PyTorch inference
                with torch.no_grad():
                    input_tensor = torch.from_numpy(input_data).to(self.device)
                    output = self.model(input_tensor)
                    result = output.cpu().numpy()
            
            elif self.session is not None:
                # ONNX inference
                input_name = self.session.get_inputs()[0].name
                result = self.session.run(None, {input_name: input_data})[0]
            
            else:
                raise RuntimeError("No model loaded")
            
            # Track performance
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.request_count += 1
            
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Inference error: {e}")
            raise
    
    def batch_predict(self, input_batch: List[np.ndarray]) -> List[np.ndarray]:
        """Run batch inference."""
        if not input_batch:
            return []
        
        # Stack inputs into batch
        batch_input = np.stack(input_batch, axis=0)
        batch_output = self.predict(batch_input)
        
        # Split batch output
        return [batch_output[i] for i in range(len(input_batch))]
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get inference performance metrics."""
        if not self.inference_times:
            return {}
        
        times = list(self.inference_times)
        return {
            'avg_inference_time_ms': np.mean(times) * 1000,
            'p50_inference_time_ms': np.percentile(times, 50) * 1000,
            'p95_inference_time_ms': np.percentile(times, 95) * 1000,
            'p99_inference_time_ms': np.percentile(times, 99) * 1000,
            'total_requests': self.request_count,
            'error_rate': self.error_count / max(self.request_count, 1),
            'throughput_rps': len(times) / (times[-1] - times[0]) if len(times) > 1 else 0
        }


class ModelServer:
    """Production model serving server."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.engines: Dict[str, InferenceEngine] = {}
        self.request_queue = asyncio.Queue(maxsize=config.max_concurrent_requests)
        self.batch_processor = None
        self.monitoring = ProductionMonitoring(config)
        
    async def start_server(self, host: str = "0.0.0.0", port: int = 8080):
        """Start the model serving server."""
        from aiohttp import web
        
        app = web.Application()
        app.router.add_post('/predict', self.handle_predict)
        app.router.add_get('/health', self.handle_health)
        app.router.add_get('/metrics', self.handle_metrics)
        app.router.add_post('/models/{model_name}/load', self.handle_load_model)
        
        # Start batch processor
        if self.config.batch_size > 1:
            self.batch_processor = asyncio.create_task(self._batch_processing_loop())
        
        # Start monitoring
        if self.config.enable_monitoring:
            self.monitoring.start_monitoring()
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        
        logger.info(f"Model server started on {host}:{port}")
    
    async def handle_predict(self, request):
        """Handle prediction requests."""
        try:
            data = await request.json()
            model_name = data.get('model', 'default')
            input_data = np.array(data['input'])
            
            if model_name not in self.engines:
                return web.json_response(
                    {'error': f'Model {model_name} not found'}, 
                    status=404
                )
            
            # Add to queue for batch processing or process immediately
            if self.config.batch_size > 1:
                future = asyncio.Future()
                await self.request_queue.put((model_name, input_data, future))
                result = await future
            else:
                result = self.engines[model_name].predict(input_data)
            
            return web.json_response({
                'prediction': result.tolist(),
                'model': model_name,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return web.json_response(
                {'error': str(e)}, 
                status=500
            )
    
    async def handle_health(self, request):
        """Handle health check requests."""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'models': list(self.engines.keys()),
            'queue_size': self.request_queue.qsize() if self.request_queue else 0
        }
        
        # Add performance metrics
        for name, engine in self.engines.items():
            health_status[f'{name}_metrics'] = engine.get_performance_metrics()
        
        return web.json_response(health_status)
    
    async def handle_metrics(self, request):
        """Handle metrics requests."""
        metrics = {}
        
        # Collect metrics from all engines
        for name, engine in self.engines.items():
            metrics[name] = engine.get_performance_metrics()
        
        # Add system metrics
        if self.config.enable_monitoring:
            metrics['system'] = self.monitoring.get_system_metrics()
        
        return web.json_response(metrics)
    
    async def handle_load_model(self, request):
        """Handle model loading requests."""
        try:
            model_name = request.match_info['model_name']
            data = await request.json()
            model_path = data['model_path']
            model_type = data.get('model_type', 'pytorch')
            
            # Create new inference engine
            engine = InferenceEngine(self.config)
            engine.load_model(model_path, model_type)
            self.engines[model_name] = engine
            
            return web.json_response({
                'message': f'Model {model_name} loaded successfully',
                'model_path': model_path,
                'model_type': model_type
            })
            
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            return web.json_response(
                {'error': str(e)}, 
                status=500
            )
    
    async def _batch_processing_loop(self):
        """Process requests in batches."""
        while True:
            try:
                batch = []
                futures = []
                model_name = None
                
                # Collect batch
                start_time = time.time()
                while (len(batch) < self.config.batch_size and 
                       (time.time() - start_time) * 1000 < self.config.max_batch_delay_ms):
                    
                    try:
                        item = await asyncio.wait_for(
                            self.request_queue.get(), 
                            timeout=0.01
                        )
                        name, input_data, future = item
                        
                        if model_name is None:
                            model_name = name
                        elif model_name != name:
                            # Different model, put back and process current batch
                            await self.request_queue.put(item)
                            break
                        
                        batch.append(input_data)
                        futures.append(future)
                        
                    except asyncio.TimeoutError:
                        break
                
                # Process batch
                if batch and model_name in self.engines:
                    try:
                        results = self.engines[model_name].batch_predict(batch)
                        for future, result in zip(futures, results):
                            future.set_result(result)
                    except Exception as e:
                        for future in futures:
                            future.set_exception(e)
                
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                await asyncio.sleep(0.1)


class ProductionMonitoring:
    """Production monitoring and alerting."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.metrics_history = defaultdict(deque)
        self.alerts = []
        self.monitoring_active = False
        self.monitoring_thread = None
        
    def start_monitoring(self):
        """Start monitoring in background thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Production monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                metrics = self.get_system_metrics()
                timestamp = time.time()
                
                for metric_name, value in metrics.items():
                    self.metrics_history[metric_name].append((timestamp, value))
                    
                    # Keep only recent history
                    while (len(self.metrics_history[metric_name]) > 0 and 
                           timestamp - self.metrics_history[metric_name][0][0] > 3600):
                        self.metrics_history[metric_name].popleft()
                
                # Check for alerts
                self._check_alerts(metrics)
                
                time.sleep(self.config.metrics_collection_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(self.config.metrics_collection_interval)
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0,
            'open_files': len(psutil.Process().open_files()),
            'network_connections': len(psutil.net_connections()),
        }
    
    def _check_alerts(self, metrics: Dict[str, float]):
        """Check for alert conditions."""
        alerts = []
        
        # CPU usage alert
        if metrics['cpu_percent'] > 90:
            alerts.append({
                'type': 'high_cpu',
                'message': f"High CPU usage: {metrics['cpu_percent']:.1f}%",
                'severity': 'warning',
                'timestamp': datetime.now().isoformat()
            })
        
        # Memory usage alert
        if metrics['memory_percent'] > 90:
            alerts.append({
                'type': 'high_memory',
                'message': f"High memory usage: {metrics['memory_percent']:.1f}%",
                'severity': 'warning',
                'timestamp': datetime.now().isoformat()
            })
        
        # Disk usage alert
        if metrics['disk_percent'] > 90:
            alerts.append({
                'type': 'high_disk',
                'message': f"High disk usage: {metrics['disk_percent']:.1f}%",
                'severity': 'warning',
                'timestamp': datetime.now().isoformat()
            })
        
        # Add new alerts
        for alert in alerts:
            self.alerts.append(alert)
            logger.warning(f"Alert: {alert['message']}")
        
        # Keep only recent alerts
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.alerts = [
            alert for alert in self.alerts 
            if datetime.fromisoformat(alert['timestamp']) > cutoff_time
        ]


class ABTestingFramework:
    """A/B testing framework for model deployment."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.models: Dict[str, InferenceEngine] = {}
        self.traffic_split = config.traffic_split
        self.experiment_results = defaultdict(list)
        
    def add_model(self, model_name: str, engine: InferenceEngine):
        """Add model to A/B test."""
        self.models[model_name] = engine
        logger.info(f"Added model {model_name} to A/B test")
    
    def route_request(self, request_id: str) -> str:
        """Route request to appropriate model based on traffic split."""
        # Use hash of request ID for consistent routing
        hash_value = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        normalized_hash = (hash_value % 1000) / 1000.0
        
        cumulative_weight = 0.0
        for model_name, weight in self.traffic_split.items():
            cumulative_weight += weight
            if normalized_hash <= cumulative_weight:
                return model_name
        
        # Fallback to first model
        return list(self.models.keys())[0]
    
    def predict_with_routing(self, request_id: str, input_data: np.ndarray) -> Tuple[np.ndarray, str]:
        """Make prediction with A/B routing."""
        model_name = self.route_request(request_id)
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in A/B test")
        
        start_time = time.time()
        result = self.models[model_name].predict(input_data)
        inference_time = time.time() - start_time
        
        # Record experiment result
        self.experiment_results[model_name].append({
            'request_id': request_id,
            'inference_time': inference_time,
            'timestamp': datetime.now().isoformat()
        })
        
        return result, model_name
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get A/B test experiment summary."""
        summary = {}
        
        for model_name, results in self.experiment_results.items():
            if not results:
                continue
            
            inference_times = [r['inference_time'] for r in results]
            summary[model_name] = {
                'total_requests': len(results),
                'avg_inference_time': np.mean(inference_times),
                'p95_inference_time': np.percentile(inference_times, 95),
                'traffic_percentage': self.traffic_split.get(model_name, 0) * 100
            }
        
        return summary


class CloudDeployment:
    """Cloud deployment automation."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.provider = config.cloud_provider
        
    def create_deployment_manifest(self, model_path: str, image_name: str) -> Dict[str, Any]:
        """Create deployment manifest for cloud deployment."""
        if self.provider == "aws":
            return self._create_aws_manifest(model_path, image_name)
        elif self.provider == "gcp":
            return self._create_gcp_manifest(model_path, image_name)
        elif self.provider == "azure":
            return self._create_azure_manifest(model_path, image_name)
        else:
            raise ValueError(f"Unsupported cloud provider: {self.provider}")
    
    def _create_aws_manifest(self, model_path: str, image_name: str) -> Dict[str, Any]:
        """Create AWS EKS deployment manifest."""
        return {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': 'nanonas-model-server',
                'labels': {'app': 'nanonas-model-server'}
            },
            'spec': {
                'replicas': self.config.min_replicas,
                'selector': {'matchLabels': {'app': 'nanonas-model-server'}},
                'template': {
                    'metadata': {'labels': {'app': 'nanonas-model-server'}},
                    'spec': {
                        'containers': [{
                            'name': 'model-server',
                            'image': image_name,
                            'ports': [{'containerPort': 8080}],
                            'env': [
                                {'name': 'MODEL_PATH', 'value': model_path},
                                {'name': 'BATCH_SIZE', 'value': str(self.config.batch_size)},
                                {'name': 'TARGET_DEVICE', 'value': self.config.target_device}
                            ],
                            'resources': {
                                'requests': {'cpu': '500m', 'memory': '1Gi'},
                                'limits': {'cpu': '2', 'memory': '4Gi'}
                            },
                            'livenessProbe': {
                                'httpGet': {'path': '/health', 'port': 8080},
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            },
                            'readinessProbe': {
                                'httpGet': {'path': '/health', 'port': 8080},
                                'initialDelaySeconds': 5,
                                'periodSeconds': 5
                            }
                        }]
                    }
                }
            }
        }
    
    def _create_gcp_manifest(self, model_path: str, image_name: str) -> Dict[str, Any]:
        """Create GCP GKE deployment manifest."""
        # Similar to AWS but with GCP-specific configurations
        manifest = self._create_aws_manifest(model_path, image_name)
        
        # Add GCP-specific annotations
        manifest['metadata']['annotations'] = {
            'cloud.google.com/load-balancer-type': 'Internal'
        }
        
        return manifest
    
    def _create_azure_manifest(self, model_path: str, image_name: str) -> Dict[str, Any]:
        """Create Azure AKS deployment manifest."""
        # Similar to AWS but with Azure-specific configurations
        manifest = self._create_aws_manifest(model_path, image_name)
        
        # Add Azure-specific annotations
        manifest['metadata']['annotations'] = {
            'service.beta.kubernetes.io/azure-load-balancer-internal': 'true'
        }
        
        return manifest
    
    def create_dockerfile(self, model_path: str, requirements_path: str) -> str:
        """Create Dockerfile for model deployment."""
        dockerfile_content = f"""
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY {requirements_path} requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and application code
COPY {model_path} /app/model/
COPY nanonas/ /app/nanonas/
COPY deployment_server.py /app/

# Expose port
EXPOSE 8080

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/model/
ENV BATCH_SIZE={self.config.batch_size}
ENV TARGET_DEVICE={self.config.target_device}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["python", "deployment_server.py"]
"""
        return dockerfile_content
    
    def deploy_to_cloud(self, manifest: Dict[str, Any], namespace: str = "default"):
        """Deploy to cloud using kubectl."""
        import tempfile
        
        # Save manifest to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(manifest, f)
            manifest_path = f.name
        
        try:
            # Apply manifest using kubectl
            result = subprocess.run(
                ['kubectl', 'apply', '-f', manifest_path, '-n', namespace],
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Deployment successful: {result.stdout}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Deployment failed: {e.stderr}")
            return False
        
        finally:
            # Clean up temporary file
            Path(manifest_path).unlink()


class ModelVersioning:
    """Model versioning and rollback system."""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.versions_file = self.storage_path / "versions.json"
        self.versions = self._load_versions()
    
    def _load_versions(self) -> Dict[str, Any]:
        """Load version metadata."""
        if self.versions_file.exists():
            with open(self.versions_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_versions(self):
        """Save version metadata."""
        with open(self.versions_file, 'w') as f:
            json.dump(self.versions, f, indent=2)
    
    def register_version(self, model_name: str, model_path: str, 
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """Register a new model version."""
        version_id = f"v{int(time.time())}"
        
        # Copy model to versioned storage
        version_dir = self.storage_path / model_name / version_id
        version_dir.mkdir(parents=True, exist_ok=True)
        
        import shutil
        model_file = version_dir / "model.pt"
        shutil.copy2(model_path, model_file)
        
        # Store metadata
        version_info = {
            'version_id': version_id,
            'model_path': str(model_file),
            'created_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        if model_name not in self.versions:
            self.versions[model_name] = []
        
        self.versions[model_name].append(version_info)
        self._save_versions()
        
        logger.info(f"Registered model version {version_id} for {model_name}")
        return version_id
    
    def get_version(self, model_name: str, version_id: str) -> Optional[Dict[str, Any]]:
        """Get specific model version."""
        if model_name not in self.versions:
            return None
        
        for version in self.versions[model_name]:
            if version['version_id'] == version_id:
                return version
        
        return None
    
    def get_latest_version(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get latest model version."""
        if model_name not in self.versions or not self.versions[model_name]:
            return None
        
        return self.versions[model_name][-1]
    
    def rollback_to_version(self, model_name: str, version_id: str) -> bool:
        """Rollback to specific version."""
        version_info = self.get_version(model_name, version_id)
        if not version_info:
            logger.error(f"Version {version_id} not found for model {model_name}")
            return False
        
        # Mark as current version by moving to end of list
        self.versions[model_name] = [
            v for v in self.versions[model_name] 
            if v['version_id'] != version_id
        ]
        self.versions[model_name].append(version_info)
        self._save_versions()
        
        logger.info(f"Rolled back model {model_name} to version {version_id}")
        return True
    
    def list_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """List all versions for a model."""
        return self.versions.get(model_name, [])


def create_production_deployment(config: DeploymentConfig, model_path: str, 
                                model_name: str = "nanonas_model") -> Dict[str, str]:
    """Create complete production deployment."""
    logger.info("Creating production deployment")
    
    # Initialize components
    optimizer = ModelOptimizer(config)
    cloud_deployment = CloudDeployment(config)
    versioning = ModelVersioning("./model_versions")
    
    # Load and optimize model
    model = torch.jit.load(model_path)
    sample_input = torch.randn(1, 3, 224, 224)
    
    optimized_model = optimizer.optimize_model(model, sample_input)
    
    # Save optimized model
    optimized_path = f"./optimized_{model_name}.pt"
    torch.jit.save(optimized_model, optimized_path)
    
    # Export to ONNX if enabled
    onnx_path = None
    if config.enable_onnx_export:
        onnx_path = optimizer.export_onnx(optimized_model, sample_input, 
                                        f"./optimized_{model_name}.onnx")
    
    # Register version
    version_id = versioning.register_version(
        model_name, 
        optimized_path,
        {
            'quantized': config.enable_quantization,
            'pruned': config.enable_pruning,
            'jit_compiled': config.enable_jit_compilation,
            'onnx_exported': config.enable_onnx_export
        }
    )
    
    # Create deployment artifacts
    dockerfile = cloud_deployment.create_dockerfile(optimized_path, "requirements.txt")
    manifest = cloud_deployment.create_deployment_manifest(optimized_path, 
                                                          f"{model_name}:latest")
    
    # Save artifacts
    with open("Dockerfile", "w") as f:
        f.write(dockerfile)
    
    with open("deployment.yaml", "w") as f:
        yaml.dump(manifest, f)
    
    logger.info("Production deployment created successfully")
    
    return {
        'optimized_model_path': optimized_path,
        'onnx_model_path': onnx_path,
        'version_id': version_id,
        'dockerfile_path': "Dockerfile",
        'manifest_path': "deployment.yaml"
    } 