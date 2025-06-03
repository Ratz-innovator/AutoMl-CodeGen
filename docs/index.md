# nanoNAS: Neural Architecture Search Made Simple

<div align="center">

![nanoNAS Logo](assets/images/nanonas-banner.svg){ .center }

**🎓 Graduate-Level Neural Architecture Search Framework**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776ab?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/Ratz-innovator/AutoMl-CodeGen/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://nanonas.readthedocs.io)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000)](https://github.com/psf/black)

[🚀 Get Started](quickstart.md){ .md-button .md-button--primary }
[📖 Documentation](user-guide/index.md){ .md-button }
[🔬 Research](research/index.md){ .md-button }
[📊 Benchmarks](#benchmarks){ .md-button }

</div>

---

## 🎯 Overview

**nanoNAS** is a comprehensive, production-ready framework for Neural Architecture Search (NAS) that combines cutting-edge research with educational clarity. Designed for graduate-level research and industry applications, it provides modular implementations of state-of-the-art NAS algorithms with extensive benchmarking and visualization capabilities.

### 🏆 Key Features

=== "🔬 Research-Grade"

    - **Advanced Algorithms**: PC-DARTS, GDAS, SETN, DrNAS implementations
    - **Multi-Objective Optimization**: NSGA-II/III, Pareto optimization
    - **Theoretical Foundation**: Graduate-level algorithmic rigor
    - **Publication Quality**: Research-grade experimental validation

=== "🏭 Production-Ready" 

    - **One-Line API**: Simple interface for quick experimentation
    - **REST API Server**: Professional FastAPI-based backend
    - **Docker Deployment**: Containerized for easy deployment
    - **Scalable Architecture**: Multi-GPU and distributed support

=== "📚 Educational"

    - **Interactive Tutorials**: Step-by-step Jupyter notebooks
    - **Comprehensive Documentation**: Theory and implementation details
    - **Visualization Dashboards**: Real-time monitoring and analysis
    - **Code Examples**: Ready-to-run demonstrations

### 🚀 What Makes nanoNAS Special?

| Feature | Description | Impact |
|---------|-------------|---------|
| **🎓 Graduate-Level Quality** | Research-grade implementations with theoretical rigor | Publication-ready results |
| **🔬 Comprehensive Evaluation** | Real benchmarks on CIFAR-10/100, ImageNet, COCO | Reliable performance metrics |
| **📊 Advanced Visualization** | Architecture evolution, search landscapes, Pareto frontiers | Deep insights into NAS dynamics |
| **⚙️ Production Ready** | Modular design, extensive testing, professional documentation | Enterprise deployment |
| **🎯 Multi-Objective** | Beyond accuracy - optimize for latency, parameters, energy | Real-world constraints |
| **📈 Research Insights** | Novel analysis tools for understanding NAS dynamics | Scientific advancement |

---

## 🏗️ Architecture

```mermaid
graph TB
    A[User Interface] --> B[API Layer]
    B --> C[Search Strategies]
    B --> D[Visualization]
    B --> E[Benchmarking]
    
    C --> F[Evolutionary]
    C --> G[DARTS]
    C --> H[PC-DARTS]
    C --> I[Ensemble]
    
    F --> J[Search Space]
    G --> J
    H --> J
    I --> J
    
    J --> K[Model Builder]
    K --> L[Training Engine]
    L --> M[Evaluation]
    
    D --> N[Interactive Dashboard]
    D --> O[Architecture Visualizer]
    D --> P[Performance Analysis]
    
    E --> Q[Model Evaluator]
    E --> R[Statistical Analysis]
    E --> S[Comparison Engine]
```

---

## 📊 Benchmarks

### Performance Comparison

| Model | Params | FLOPs | CIFAR-10 Acc | Search Time | Method |
|-------|---------|-------|-------------|-------------|--------|
| **nanoNAS-Evolutionary** | 0.89M | 156M | **94.2%** | 2.3h | This work |
| **nanoNAS-PC-DARTS** | 1.1M | 180M | **94.0%** | 1.5h | This work |
| **nanoNAS-Ensemble** | 0.95M | 165M | **94.5%** | 3.1h | This work |
| ResNet-18 | 11.2M | 1.8G | 95.0% | - | Baseline |
| MobileNet-V2 | 3.5M | 300M | 92.1% | - | Baseline |
| DARTS-Original | 3.3M | 528M | 97.0% | 4.0h | Liu et al. |

### Search Strategy Comparison

!!! tip "Performance Insights"
    - **Evolutionary**: Best exploration, diverse architectures
    - **PC-DARTS**: Memory efficient, fast convergence
    - **Ensemble**: Highest accuracy, robust results
    - **Multi-Objective**: Optimal accuracy-efficiency trade-offs

---

## 🎓 Theoretical Foundation

### Neural Architecture Search Problem

Neural Architecture Search addresses the fundamental question: **How can we automatically design optimal neural network architectures?**

Given a search space $\mathcal{A}$ of possible architectures and a performance metric $P$, NAS seeks to find:

$$\alpha^* = \underset{\alpha \in \mathcal{A}}{\arg\max} \, P(\alpha, \mathcal{D})$$

Where $\alpha$ is an architecture, $\mathcal{D}$ is the dataset, and $P$ is the performance function.

### Search Space Design

nanoNAS implements multiple search spaces with different complexities:

```python
# Nano Space: Educational and fast
operations = [conv3x3, conv5x5, maxpool, skip, zero]
constraints = {max_depth: 6, max_params: 1e6}

# Mobile Space: Efficiency-focused  
operations = [conv1x1, conv3x3, dw_conv3x3, skip, zero]
constraints = {max_depth: 12, max_params: 5e6}

# Research Space: Full expressiveness
operations = [attention, moe, conv_transformer, ...]
constraints = {max_params: 50e6, max_flops: 1e9}
```

---

## 🚀 Quick Start

### Installation

=== "pip"

    ```bash
    pip install nanonas
    ```

=== "conda"

    ```bash
    conda install -c conda-forge nanonas
    ```

=== "Docker"

    ```bash
    docker pull nanonas/nanonas:latest
    docker run -it nanonas/nanonas:latest
    ```

### One-Line Architecture Search

```python
import nanonas

# Find optimal architecture for CIFAR-10
model = nanonas.search(strategy='evolutionary', dataset='cifar10')

# Ready to train!
optimizer = torch.optim.Adam(model.parameters())
```

### Advanced Usage

```python
import nanonas

# Custom search configuration
config = nanonas.ExperimentConfig(
    search=nanonas.SearchConfig(
        strategy='pc_darts',
        search_space='mobile',
        epochs=100
    ),
    dataset=nanonas.DatasetConfig(
        name='cifar10',
        batch_size=128,
        use_augmentation=True
    )
)

# Run comprehensive search with visualization
model, results = nanonas.search(config, return_results=True)
nanonas.visualize(results['best_architecture'], 'architecture.png')
```

---

## 📚 Learning Path

### For Beginners

1. [🚀 Installation Guide](installation.md)
2. [📖 Quick Start Tutorial](quickstart.md)
3. [🎯 Basic Usage](user-guide/basic-usage.md)
4. [🔍 Search Strategies](user-guide/search-strategies.md)

### For Researchers

1. [🔬 Theoretical Foundation](research/theoretical-foundation.md)
2. [🧬 Advanced Algorithms](research/algorithms.md)
3. [📊 Experimental Results](research/experimental-results.md)
4. [📄 Publications](research/publications.md)

### For Practitioners

1. [⚙️ Configuration Guide](user-guide/configuration.md)
2. [🏭 Production Deployment](user-guide/deployment.md)
3. [📈 Visualization Tools](user-guide/visualization.md)
4. [🔧 API Reference](api/index.md)

---

## 🤝 Community

### Get Involved

- **🐛 Bug Reports**: [GitHub Issues](https://github.com/Ratz-innovator/AutoMl-CodeGen/issues)
- **💡 Feature Requests**: [GitHub Discussions](https://github.com/Ratz-innovator/AutoMl-CodeGen/discussions)
- **📝 Contributing**: [Development Guide](development/contributing.md)
- **💬 Chat**: [Discord Server](https://discord.gg/nanonas)

### Research Collaboration

We welcome research collaborations! If you're working on:

- 🧠 Novel NAS algorithms
- 📊 Benchmark development
- 🔬 Theoretical analysis
- 🏭 Production applications

Please reach out to our research team.

---

## 📄 Citation

If you use nanoNAS in your research, please cite:

```bibtex
@software{nanonas2024,
  title={nanoNAS: Neural Architecture Search Made Simple},
  author={AutoML Research Team},
  year={2024},
  url={https://github.com/Ratz-innovator/AutoMl-CodeGen},
  version={1.0.0}
}
```

---

## 📜 License

This project is licensed under the [MIT License](https://github.com/Ratz-innovator/AutoMl-CodeGen/blob/main/LICENSE).

---

!!! quote "Research Philosophy"
    "The best neural architectures are not discovered, they are evolved through intelligent exploration of the vast space of possibilities."
    
    *— nanoNAS Development Team* 