"""
Code Generation for Neural Architecture Search
=============================================

This module provides comprehensive code generation capabilities for 
deploying neural architectures found through NAS to various frameworks
and deployment targets.

Features:
- PyTorch, TensorFlow, JAX, and ONNX code generation
- Automatic quantization (INT8, FP16) integration
- Docker containers and Kubernetes manifests
- Model serving with FastAPI/TorchServe
- Optimization passes (operator fusion, memory layout)
"""

from .pytorch_generator import PyTorchGenerator
from .tensorflow_generator import TensorFlowGenerator
from .jax_generator import JAXGenerator
from .onnx_generator import ONNXGenerator
from .deployment_generator import DeploymentGenerator
from .optimization_passes import OptimizationPasses

__all__ = [
    "PyTorchGenerator",
    "TensorFlowGenerator", 
    "JAXGenerator",
    "ONNXGenerator",
    "DeploymentGenerator",
    "OptimizationPasses",
] 