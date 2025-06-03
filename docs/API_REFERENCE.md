# nanoNAS API Reference

Complete API documentation for the nanoNAS Neural Architecture Search framework.

## Core Classes

### Architecture

Main class for representing neural architectures.

```python
class Architecture:
    def __init__(self, encoding=None, graph=None, hierarchical_encoding=None, 
                 search_space=None, metadata=None)
```

**Parameters:**
- `encoding` (List[int], optional): List-based architecture encoding
- `graph` (nx.DiGraph, optional): Graph-based architecture encoding  
- `hierarchical_encoding` (List[Dict], optional): Hierarchical cell-based encoding
- `search_space` (SearchSpace, optional): Search space definition
- `metadata` (Dict, optional): Additional metadata

**Key Methods:**

#### `to_model(input_channels=3, num_classes=10, base_channels=16) -> nn.Module`
Convert architecture to PyTorch model.

```python
arch = Architecture([0, 1, 2, 3], search_space)
model = arch.to_model()
```

#### `mutate(mutation_rate=0.1, mutation_std=1.0) -> Architecture`
Create mutated version of architecture.

```python
mutated = arch.mutate(mutation_rate=0.2)
```

#### `crossover(other) -> Tuple[Architecture, Architecture]`
Perform crossover with another architecture.

```python
child1, child2 = arch1.crossover(arch2)
```

#### `get_complexity_metrics() -> Dict[str, float]`
Calculate architecture complexity metrics.

```python
metrics = arch.get_complexity_metrics()
print(f"Depth: {metrics['depth']}")
print(f"FLOPs: {metrics['total_flops']}")
```

#### `save(path) / load(path)`
Save/load architecture to/from file.

```python
arch.save("architecture.json")
loaded = Architecture.load("architecture.json")
```

### SearchSpace

Defines the operation search space for architectures.

```python
class SearchSpace:
    def __init__(self, name, operations, constraints=None, encoding_type="list", 
                 hierarchical_cells=None)
```

**Class Methods:**

#### `get_nano_search_space() -> SearchSpace`
Get minimal 5-operation search space.

```python
space = SearchSpace.get_nano_search_space()
# Operations: conv3x3, conv5x5, maxpool3x3, skip, zero
```

#### `get_mobile_search_space() -> SearchSpace`
Get MobileNet-inspired search space.

```python
space = SearchSpace.get_mobile_search_space()
# 9 operations including depthwise separable convolutions
```

#### `get_advanced_search_space() -> SearchSpace`
Get advanced search space with modern operations.

```python
space = SearchSpace.get_advanced_search_space()
# 20+ operations: attention, advanced normalization, etc.
```

**Instance Methods:**

#### `sample_random_architecture(num_blocks=None) -> Architecture`
Sample random architecture from search space.

```python
arch = search_space.sample_random_architecture(num_blocks=6)
```

### OperationSpec

Specification for neural operations.

```python
@dataclass
class OperationSpec:
    name: str
    type: str  # 'conv', 'pool', 'skip', 'attention', etc.
    params: Dict[str, Any]
    computational_cost: float = 1.0
    memory_cost: float = 1.0
    energy_cost: float = 1.0
    latency_cost: float = 1.0
```

**Example:**
```python
op = OperationSpec(
    name="conv3x3",
    type="conv",
    params={"kernel_size": 3, "padding": 1},
    computational_cost=1.0
)
```

## Configuration System

### ExperimentConfig

Main configuration class for experiments.

```python
class ExperimentConfig:
    def __init__(self, name="nanonas_experiment", search=None, training=None, 
                 dataset=None, model=None, device="auto")
```

**Example:**
```python
config = ExperimentConfig(
    name="my_experiment",
    search=SearchConfig(strategy='evolutionary', population_size=50),
    training=TrainingConfig(epochs=100, learning_rate=0.025),
    dataset=DatasetConfig(name='cifar10', batch_size=128)
)
```

#### `save(path) / load(path)`
Save/load configuration to/from YAML file.

```python
config.save("experiment.yaml")
loaded = ExperimentConfig.load("experiment.yaml")
```

### SearchConfig

Configuration for search algorithms.

```python
@dataclass
class SearchConfig:
    strategy: str = 'evolutionary'
    search_space: str = 'nano'
    population_size: int = 20
    generations: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    epochs: int = 50  # For DARTS
    num_samples: int = 100  # For random search
```

### TrainingConfig

Configuration for model training.

```python
@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 0.025
    optimizer: str = 'sgd'
    weight_decay: float = 3e-4
    momentum: float = 0.9
```

### DatasetConfig

Configuration for datasets.

```python
@dataclass
class DatasetConfig:
    name: str = 'cifar10'
    data_path: str = './data'
    batch_size: int = 128
    num_workers: int = 4
    augmentation: bool = True
```

## High-Level API

### search()

Main function for neural architecture search.

```python
def search(strategy=None, dataset='cifar10', search_space='nano', 
           config=None, return_results=False, **kwargs) -> Union[nn.Module, Tuple]
```

**Parameters:**
- `strategy` (str): Search strategy ('evolutionary', 'darts', 'random', etc.)
- `dataset` (str): Dataset name ('cifar10', 'cifar100', 'mnist')
- `search_space` (str): Search space name ('nano', 'mobile', 'advanced')
- `config` (ExperimentConfig, optional): Full configuration object
- `return_results` (bool): Whether to return detailed results
- `**kwargs`: Additional parameters for search strategy

**Examples:**

#### Basic Search
```python
import nanonas

# Simple evolutionary search
model = nanonas.search(
    strategy='evolutionary',
    dataset='cifar10',
    population_size=20,
    generations=50
)
```

#### Advanced Search with Configuration
```python
config = nanonas.ExperimentConfig(
    name="advanced_search",
    search=nanonas.SearchConfig(
        strategy='darts',
        search_space='mobile',
        epochs=100
    ),
    training=nanonas.TrainingConfig(
        epochs=200,
        learning_rate=0.025
    )
)

model, results = nanonas.search(config=config, return_results=True)
```

#### DARTS Search
```python
model = nanonas.search(
    strategy='darts',
    dataset='cifar10',
    search_space='mobile',
    epochs=50,
    learning_rate=0.025
)
```

### benchmark()

Compare multiple search strategies.

```python
def benchmark(strategies, dataset='cifar10', search_space='nano', 
              num_runs=3, **kwargs) -> Dict[str, Any]
```

**Example:**
```python
results = nanonas.benchmark(
    strategies=['evolutionary', 'darts', 'random'],
    dataset='cifar10',
    num_runs=5
)

for strategy, metrics in results['comparison'].items():
    print(f"{strategy}: {metrics['mean_accuracy']:.2f}% ± {metrics['std_accuracy']:.2f}")
```

### nano_nas()

Educational implementation function.

```python
def nano_nas(strategy='evolution', **kwargs) -> nn.Module
```

**Examples:**
```python
# Evolutionary search
model = nano_nas('evolution', population_size=20, generations=10)

# DARTS search  
model = nano_nas('darts', epochs=25)
```

## Search Algorithms

### EvolutionarySearch

Population-based genetic algorithm.

```python
class EvolutionarySearch(BaseSearchStrategy):
    def __init__(self, config, population_size=20, generations=50, 
                 mutation_rate=0.1, crossover_rate=0.7)
```

**Key Methods:**
- `search() -> Architecture`: Run evolutionary search
- `mutate(architecture) -> Architecture`: Mutate architecture
- `crossover(arch1, arch2) -> Tuple[Architecture, Architecture]`: Crossover operation

### DARTSSearch

Differentiable Architecture Search.

```python
class DARTSSearch(BaseSearchStrategy):
    def __init__(self, config, epochs=50, learning_rate=0.025)
```

**Key Methods:**
- `search() -> Architecture`: Run DARTS search
- `train_supernet() -> None`: Train supernet with mixed operations

### RandomSearch

Baseline random sampling.

```python
class RandomSearch(BaseSearchStrategy):
    def __init__(self, config, num_samples=100)
```

## Hardware Profiling

### profile_current_device()

Get current device information.

```python
def profile_current_device() -> DeviceProfile

profile = nanonas.profile_current_device()
print(f"Device: {profile.device_name}")
print(f"Memory: {profile.memory_total} MB")
```

### DeviceProfile

Hardware profile information.

```python
@dataclass
class DeviceProfile:
    device_name: str
    device_type: str  # 'cuda', 'cpu', 'mps'
    memory_total: int  # MB
    memory_available: int  # MB
    compute_capability: Optional[str] = None
    num_cores: Optional[int] = None
```

## Utilities

### ModelEvaluator

Evaluate model performance.

```python
class ModelEvaluator:
    def __init__(self, dataset='cifar10', device='auto')
    
    def evaluate(self, model) -> Dict[str, float]:
        """Evaluate model accuracy and efficiency metrics."""
        
    def quick_evaluate(self, model) -> float:
        """Quick accuracy estimation."""
```

**Example:**
```python
evaluator = nanonas.ModelEvaluator('cifar10')
metrics = evaluator.evaluate(model)
print(f"Accuracy: {metrics['accuracy']:.2f}%")
```

## Visualization

### ArchitectureVisualizer

Comprehensive architecture visualization.

```python
class ArchitectureVisualizer:
    def __init__(self, output_dir="results/visualizations")
    
    def visualize_architecture_sequence(self, arch, title) -> str:
        """Create sequence diagram of architecture."""
        
    def visualize_complexity_analysis(self, archs, labels) -> str:
        """Create complexity comparison plots."""
        
    def create_interactive_visualization(self, archs, labels) -> str:
        """Create interactive Plotly dashboard."""
```

**Example:**
```python
viz = nanonas.ArchitectureVisualizer()
viz.visualize_architecture_sequence(architecture, "My Architecture")
viz.create_interactive_visualization([arch1, arch2], ["Arch A", "Arch B"])
```

## Error Handling

### Common Exceptions

```python
class NanoNASError(Exception):
    """Base exception for nanoNAS."""

class SearchSpaceError(NanoNASError):
    """Invalid search space configuration."""

class ArchitectureError(NanoNASError):
    """Invalid architecture definition."""

class ConfigurationError(NanoNASError):
    """Invalid configuration."""
```

## Examples

### Complete Workflow Example

```python
import nanonas

# 1. Create custom search space
operations = [
    nanonas.OperationSpec("conv3x3", "conv", {"kernel_size": 3}),
    nanonas.OperationSpec("conv5x5", "conv", {"kernel_size": 5}),
    nanonas.OperationSpec("maxpool", "pool", {"kernel_size": 3}),
    nanonas.OperationSpec("skip", "skip", {}),
]

search_space = nanonas.SearchSpace("custom", operations)

# 2. Configure experiment
config = nanonas.ExperimentConfig(
    name="custom_experiment",
    search=nanonas.SearchConfig(
        strategy='evolutionary',
        population_size=30,
        generations=25
    ),
    training=nanonas.TrainingConfig(
        epochs=100,
        learning_rate=0.01
    )
)

# 3. Run search
model, results = nanonas.search(config=config, return_results=True)

# 4. Analyze results
best_arch = results['best_architecture']
print(f"Best architecture: {best_arch}")
print(f"Search time: {results['search_time']:.1f}s")

# 5. Visualize
viz = nanonas.ArchitectureVisualizer()
viz.visualize_architecture_sequence(best_arch, "Best Found Architecture")

# 6. Save everything
config.save("experiment_config.yaml")
best_arch.save("best_architecture.json")
```

### Batch Architecture Analysis

```python
# Create multiple architectures
search_space = nanonas.SearchSpace.get_mobile_search_space()
architectures = []

for i in range(10):
    arch = search_space.sample_random_architecture()
    architectures.append(arch)

# Analyze all architectures
for i, arch in enumerate(architectures):
    metrics = arch.get_complexity_metrics()
    print(f"Arch {i}: depth={metrics['depth']}, "
          f"flops={metrics['total_flops']:,.0f}")

# Compare architectures
viz = nanonas.ArchitectureVisualizer()
viz.visualize_architecture_comparison(
    architectures[:5], 
    [f"Architecture {i+1}" for i in range(5)]
)
```

## Type Hints

All public APIs include complete type hints:

```python
from typing import List, Dict, Any, Optional, Tuple, Union
import torch.nn as nn

def search(
    strategy: Optional[str] = None,
    dataset: str = 'cifar10',
    search_space: str = 'nano',
    config: Optional[ExperimentConfig] = None,
    return_results: bool = False,
    **kwargs: Any
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    ...
```

## Performance Notes

- **List encoding**: Fastest for sequential architectures
- **Graph encoding**: More memory intensive but flexible
- **Hierarchical encoding**: Best for complex cell-based architectures
- **Hardware profiling**: Automatic GPU detection and optimization
- **Caching**: Architecture metrics cached for performance

## Version Compatibility

- **Python**: 3.8+
- **PyTorch**: 1.12+
- **NumPy**: 1.20+
- **Optional**: NetworkX, Plotly, pandas for advanced features 