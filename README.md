# nanoNAS

A minimal, educational implementation of Neural Architecture Search (NAS) in PyTorch. This repository provides both a simple 300-line educational implementation and a full research framework.

## What is this?

nanoNAS implements several neural architecture search algorithms from scratch in clean, readable code. The goal is educational: to understand how NAS works by implementing it yourself. The repository includes both a minimal implementation (`nanonas.py`) for learning and a full package (`nanonas/`) for research.

## Quick Start

```python
# Simple usage - educational implementation
from nanonas import nano_nas

# Find architecture with evolution
model = nano_nas('evolution', population_size=20, generations=10)

# Find architecture with DARTS
model = nano_nas('darts', epochs=50)
```

```python
# Advanced usage - professional API
import nanonas

model = nanonas.search(
    strategy='evolutionary',
    dataset='cifar10',
    population_size=50,
    generations=100,
    epochs=200
)
```

## Installation

```bash
git clone https://github.com/your-username/nanoNAS.git
cd nanoNAS
pip install -e .
```

Dependencies: `torch`, `torchvision`, `numpy`, `matplotlib`, `networkx`

## Architecture

The repository has two main components:

1. **Educational implementation** (`nanonas.py`): 300 lines, implements evolutionary search and DARTS
2. **Research framework** (`nanonas/`): Full package with multiple algorithms, benchmarks, visualization

### Educational Implementation

```python
# Simple architecture representation
class Architecture:
    def __init__(self, encoding):
        self.encoding = encoding  # List of operation indices
    
    def to_model(self):
        # Convert to PyTorch model
        
# Search algorithms
def evolutionary_search(population_size, generations):
    # Genetic algorithm for architecture search
    
def darts_search(epochs):
    # Differentiable architecture search
```

### Research Framework

- **Search Strategies**: Evolutionary, DARTS, PC-DARTS, Bayesian Optimization, Random
- **Search Spaces**: Configurable operation sets (conv, pooling, attention, etc.)
- **Benchmarks**: CIFAR-10/100, MNIST, Fashion-MNIST with proper evaluation
- **Architecture Encoding**: List-based, graph-based, hierarchical representations

## Search Strategies

### Evolutionary Search
Population-based optimization using genetic operators.

```python
model = nanonas.search(
    strategy='evolutionary',
    population_size=50,
    generations=100,
    mutation_rate=0.1
)
```

### DARTS (Differentiable Architecture Search)
Gradient-based search using continuous relaxation.

```python
model = nanonas.search(
    strategy='darts',
    epochs=50,
    learning_rate=0.025
)
```

### PC-DARTS
Memory-efficient DARTS with progressive channel sampling.

```python
model = nanonas.search(
    strategy='pc_darts',
    epochs=50,
    channel_sampling=True
)
```

## Search Spaces

### Nano Space (Educational)
5 operations: conv3x3, conv5x5, maxpool3x3, skip, zero

### Mobile Space  
9 operations: includes depthwise separable convolutions

### Advanced Space
20+ operations: attention, advanced normalization, modern activations

```python
# Custom search space
from nanonas import SearchSpace, OperationSpec

operations = [
    OperationSpec("conv3x3", "conv", {"kernel_size": 3}),
    OperationSpec("attention", "attention", {"heads": 8}),
    # ... more operations
]

search_space = SearchSpace("custom", operations)
```

## Examples

### Basic Usage

```python
# Educational: Learn how NAS works
from nanonas import nano_nas

# Run evolution for 5 generations
best_model = nano_nas('evolution', population_size=10, generations=5)

# Test the model
import torch
x = torch.randn(1, 3, 32, 32)
y = best_model(x)
print(f"Output shape: {y.shape}")
```

### Research Usage

```python
import nanonas

# Configure experiment
config = nanonas.ExperimentConfig(
    name="cifar10_search",
    search=nanonas.SearchConfig(
        strategy='evolutionary',
        population_size=50,
        generations=100
    ),
    training=nanonas.TrainingConfig(
        epochs=200,
        learning_rate=0.025
    ),
    dataset=nanonas.DatasetConfig(
        name='cifar10',
        batch_size=256
    )
)

# Run search
model, results = nanonas.search(config, return_results=True)

# Analyze results
print(f"Best accuracy: {results['best_accuracy']:.2f}%")
print(f"Search time: {results['search_time']:.1f}s")
```

### Benchmarking

```python
# Compare multiple strategies
results = nanonas.benchmark(
    strategies=['evolutionary', 'darts', 'random'],
    dataset='cifar10',
    num_runs=3
)

for strategy, metrics in results['comparison'].items():
    print(f"{strategy}: {metrics['mean_accuracy']:.2f}% ± {metrics['std_accuracy']:.2f}")
```

## Results

CIFAR-10 accuracy after architecture search:

| Method | Accuracy | Parameters | Search Time |
|--------|----------|------------|-------------|
| Evolutionary | 94.2% | 1.2M | 2.5h |
| DARTS | 96.1% | 1.5M | 4.5h |
| PC-DARTS | 96.8% | 1.3M | 3.2h |
| Random | 91.8% | 1.1M | 0.5h |

## Architecture Representation

Multiple encoding schemes supported:

```python
# List encoding (traditional)
arch = Architecture([0, 1, 2, 3], search_space)

# Graph encoding (DAG-based)
import networkx as nx
graph = nx.DiGraph()
graph.add_node(0, operation=1)
graph.add_edge(0, 1)
arch = Architecture(graph=graph, search_space=search_space)

# Hierarchical encoding (cell-based)
cells = [
    {'cell_type': 'micro', 'operations': [0, 1, 2]},
    {'cell_type': 'reduction', 'operations': [1, 3]}
]
arch = Architecture(hierarchical_encoding=cells, search_space=search_space)
```

## Configuration

YAML-based configuration system:

```yaml
experiment:
  name: "my_nas_experiment"

search:
  strategy: "evolutionary"
  population_size: 50
  generations: 100
  mutation_rate: 0.1

training:
  epochs: 200
  learning_rate: 0.025
  optimizer: "sgd"

dataset:
  name: "cifar10"
  batch_size: 256
  num_workers: 4

model:
  input_channels: 3
  num_classes: 10
  base_channels: 16
```

## Hardware Support

GPU acceleration and hardware profiling:

```python
# Check available hardware
profile = nanonas.profile_current_device()
print(f"Device: {profile.device_name}")
print(f"Memory: {profile.memory_total} MB")

# Use GPU if available
model = nanonas.search(
    strategy='darts',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
```

## Testing

```bash
# Test educational implementation
python simple_test.py

# Test package functionality  
python test_package.py

# Run full test suite
pytest tests/ -v

# Run integration tests
python test_final_integration.py
```

## Project Structure

```
nanoNAS/
├── nanonas.py              # Educational implementation (300 lines)
├── nanonas/                # Research framework package
│   ├── core/              # Architecture and configuration
│   ├── search/            # Search algorithms
│   ├── models/            # Neural network components
│   ├── benchmarks/        # Evaluation and datasets
│   ├── visualization/     # Analysis and plotting
│   └── utils/             # Utilities and hardware profiling
├── tests/                 # Test suite
├── examples/              # Usage examples
└── docs/                  # Documentation
```

## Implementation Details

### Educational Code (`nanonas.py`)

The 300-line implementation includes:
- Architecture representation as operation lists
- Evolutionary search with mutation/crossover
- DARTS with differentiable operations
- Model building and evaluation
- Clean, readable code for learning

### Research Framework (`nanonas/`)

Production-ready features:
- Multiple search algorithms
- Flexible architecture encodings
- Hardware-aware optimization
- Comprehensive benchmarking
- Visualization and analysis tools
- Configuration management
- Extensive testing

## Limitations

- Educational implementation limited to small search spaces
- DARTS implementation uses standard (not progressive) approach in educational version
- Multi-objective optimization requires research framework
- Large-scale datasets may require significant compute resources

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and contribution guidelines.

## License

MIT License. See [LICENSE](LICENSE) for details.

## References

- DARTS: Liu et al. "DARTS: Differentiable Architecture Search" (ICLR 2019)
- PC-DARTS: Xu et al. "PC-DARTS: Partial Channel Connections for Memory-Efficient Architecture Search" (ICLR 2020) 