#!/usr/bin/env python3
"""
AutoML-CodeGen: Neural Architecture Search with Automatic Code Generation
A revolutionary system for automated neural network design and deployment.
"""

from setuptools import setup, find_packages
import os
import sys

# Ensure Python 3.8+
if sys.version_info < (3, 8):
    raise RuntimeError("Python 3.8 or higher is required")

# Read long description from README
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

# Core dependencies
install_requires = [
    # Core ML/DL frameworks
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    # "tensorflow>=2.13.0",  # Not available for Python 3.13 yet
    "jax>=0.4.0",
    "jaxlib>=0.4.0",
    "flax>=0.7.0",
    
    # Optimization and search
    "optuna>=3.2.0",
    "ray[tune]>=2.5.0",
    "hyperopt>=0.2.7",
    "scikit-optimize>=0.9.0",
    "deap>=1.3.3",
    "pymoo>=0.6.0",
    
    # Data processing
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scipy>=1.10.0",
    "scikit-learn>=1.3.0",
    "datasets>=2.13.0",
    "transformers>=4.30.0",
    
    # Visualization and monitoring
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "plotly>=5.15.0",
    "wandb>=0.15.0",
    "tensorboard>=2.13.0",
    "visdom>=0.2.0",
    
    # Code generation and compilation
    "jinja2>=3.1.0",
    "black>=23.0.0",
    "autopep8>=2.0.0",
    "ast-tools>=0.1.0",
    "astunparse>=1.6.0",
    
    # Performance and profiling
    "memory-profiler>=0.60.0",
    "py-spy>=0.3.0",
    "pynvml>=11.5.0",
    "psutil>=5.9.0",
    "gpustat>=1.1.0",
    
    # Utilities
    "tqdm>=4.65.0",
    "rich>=13.0.0",
    "click>=8.1.0",
    "pyyaml>=6.0",
    "toml>=0.10.0",
    "jsonschema>=4.17.0",
    "requests>=2.31.0",
    
    # Testing and development
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-xdist>=3.3.0",
    "hypothesis>=6.75.0",
]

# Optional dependencies for specific features
extras_require = {
    "distributed": [
        "ray[default]>=2.5.0",
        "horovod>=0.28.0",
        "mpi4py>=3.1.0",
        "dask[complete]>=2023.6.0",
    ],
    "cloud": [
        "boto3>=1.26.0",
        "google-cloud-storage>=2.9.0",
        "azure-storage-blob>=12.16.0",
        "kubernetes>=26.1.0",
    ],
    "hardware": [
        "onnx>=1.14.0",
        "onnxruntime>=1.15.0",
        "tensorrt>=8.6.0",
        "openvino>=2023.0.0",
        "tflite-runtime>=2.13.0",
    ],
    "web": [
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "streamlit>=1.25.0",
        "gradio>=3.35.0",
        "dash>=2.11.0",
    ],
    "research": [
        "jupyter>=1.0.0",
        "jupyterlab>=4.0.0",
        "ipywidgets>=8.0.0",
        "papermill>=2.4.0",
        "nbconvert>=7.6.0",
    ],
    "dev": [
        "pre-commit>=3.3.0",
        "mypy>=1.4.0",
        "flake8>=6.0.0",
        "isort>=5.12.0",
        "bandit>=1.7.0",
        "safety>=2.3.0",
    ]
}

# All optional dependencies
extras_require["all"] = list(set(sum(extras_require.values(), [])))

setup(
    name="automl-codegen",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@university.edu",
    description="Neural Architecture Search with Automatic Code Generation",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/automl-codegen",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Software Development :: Compilers",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "automl-search=automl_codegen.cli.search:main",
            "automl-generate=automl_codegen.cli.generate:main",
            "automl-benchmark=automl_codegen.cli.benchmark:main",
            "automl-serve=automl_codegen.cli.serve:main",
        ],
    },
    include_package_data=True,
    package_data={
        "automl_codegen": [
            "templates/*.j2",
            "configs/*.yaml",
            "data/*.json",
            "assets/*",
        ]
    },
    zip_safe=False,
    keywords=[
        "neural architecture search",
        "automl",
        "code generation",
        "deep learning",
        "machine learning",
        "optimization",
        "artificial intelligence",
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-username/automl-codegen/issues",
        "Source": "https://github.com/your-username/automl-codegen",
        "Documentation": "https://automl-codegen.readthedocs.io/",
        "Research Paper": "https://arxiv.org/abs/2024.xxxxx",
    },
) 