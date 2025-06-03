# Enhanced nanoNAS Usage Guide

## 🚀 Quick Start with New Features

### Command Line Interface

The enhanced nanoNAS now includes a comprehensive CLI:

```bash
# Basic search with Progressive-DARTS
nanonas search --strategy progressive_darts --dataset cifar10 --budget 100

# Bayesian optimization with multiple objectives
nanonas search --strategy bayesian --dataset cifar10 --objectives accuracy flops energy

# Multi-objective search with NSGA-III
nanonas search --strategy multiobjective --dataset cifar10 --objectives accuracy flops latency memory

# Hardware profiling
nanonas profile-hardware

# Benchmark multiple strategies
nanonas benchmark --strategy evolutionary darts progressive_darts bayesian --runs 3

# Visualize architecture from results
nanonas visualize results/results.json --format png --layout hierarchical

# Generate deployment code
nanonas deploy results/results.json --framework pytorch --serving --docker --kubernetes
```

### Python API with Enhanced Features

```python
import nanonas

# 1. Progressive-DARTS with early stopping and pruning
from nanonas.search.progressive_darts import ProgressiveDARTSConfig, ProgressiveDARTSSearch

config = nanonas.ExperimentConfig.create_default('cifar10')
prog_config = ProgressiveDARTSConfig(
    epochs=100,
    pruning_stages=4,
    early_stopping_patience=20,
    convergence_threshold=1e-4
)
config.search.progressive_darts = prog_config

searcher = ProgressiveDARTSSearch(config)
best_architecture = searcher.search()

# Visualize search progress
searcher.visualize_search_progress('progressive_darts_progress.png')

# 2. Bayesian Optimization with Gaussian Processes
from nanonas.search.bayesian_optimization import BayesianOptimizationConfig, BayesianOptimizationSearch

bo_config = BayesianOptimizationConfig(
    num_iterations=100,
    initial_samples=20,
    acquisition_function="ei",
    objectives=["accuracy", "flops", "energy"],
    kernel_type="matern"
)
config.search.bayesian_optimization = bo_config

bo_searcher = BayesianOptimizationSearch(config)
best_architecture = bo_searcher.search()

# 3. Multi-Objective Optimization with NSGA-III
from nanonas.search.multiobjective import MultiObjectiveConfig, MultiObjectiveSearch

mo_config = MultiObjectiveConfig(
    population_size=100,
    generations=50,
    objectives=["accuracy", "flops", "energy", "latency", "memory"],
    adaptive_weights=True,
    use_surrogate_model=True
)
config.search.multiobjective = mo_config

mo_searcher = MultiObjectiveSearch(config)
best_architecture = mo_searcher.search()

# Visualize Pareto front
mo_searcher.visualize_pareto_front('pareto_front.png')
```

## 🔧 Hardware-Aware Optimization

### Hardware Profiling

```python
from nanonas.utils.hardware_utils import profile_current_device, HardwareProfiler

# Profile current device
profile = profile_current_device()
print(f"Device: {profile.device_name}")
print(f"Peak FLOPs: {profile.peak_flops} GFLOPS")
print(f"Memory Bandwidth: {profile.memory_bandwidth} GB/s")
print(f"TDP: {profile.thermal_design_power} W")

# Use hardware profiler for detailed monitoring
profiler = HardwareProfiler()
with profiler.monitor_performance():
    # Run architecture search
    best_arch = nanonas.search(strategy='evolutionary', dataset='cifar10')
```

### Energy and Latency Estimation

```python
from nanonas.utils.hardware_utils import EnergyEstimator, LatencyPredictor

# Create estimators
energy_estimator = EnergyEstimator(hardware_profile)
latency_predictor = LatencyPredictor(hardware_profile)

# Estimate for architecture
energy = energy_estimator.estimate_architecture_energy(architecture, input_shape=(1, 3, 224, 224))
latency = latency_predictor.predict_architecture_latency(architecture, input_shape=(1, 3, 224, 224))

print(f"Estimated energy: {energy:.2f} mJ")
print(f"Estimated latency: {latency:.2f} ms")
```

## 📊 Advanced Search Spaces

### Hierarchical Search Spaces

```python
from nanonas.core.architecture import SearchSpace, OperationSpec, HierarchicalCell

# Define modern operations
operations = [
    OperationSpec("conv3x3", "conv", {"kernel_size": 3}),
    OperationSpec("self_attention", "attention", {"num_heads": 8}),
    OperationSpec("layer_norm", "advanced_norm", {"eps": 1e-6}),
    OperationSpec("swish", "activation", {"type": "swish"}),
    OperationSpec("gelu", "activation", {"type": "gelu"}),
]

# Define hierarchical cells
micro_cell = HierarchicalCell(
    name="micro_cell",
    cell_type="micro",
    operations=operations,
    num_nodes=4,
    skip_connections=True
)

# Create hierarchical search space
hierarchical_space = SearchSpace(
    name="advanced_hierarchical",
    operations=operations,
    encoding_type="hierarchical",
    hierarchical_cells=[micro_cell]
)

# Use in search
architecture = hierarchical_space.sample_random_architecture()
```

### Graph Neural Architecture Search

```python
# Create graph-based search space
graph_space = SearchSpace.get_graph_neural_architecture_space()

# Sample graph architecture
graph_architecture = graph_space.sample_random_architecture()

# Visualize graph structure
visualizer = nanonas.ArchitectureVisualizer()
visualizer.plot_architecture(graph_architecture, layout='spring')
```

## 🎯 Multi-Objective Optimization

### Advanced NSGA-III Configuration

```python
from nanonas.search.multiobjective import NSGAIIIOptimizer

# Configure NSGA-III for many objectives (4+)
nsga3_config = MultiObjectiveConfig(
    population_size=100,
    objectives=["accuracy", "flops", "energy", "latency", "memory", "params"],
    reference_points=None,  # Auto-generated
    constraint_weights={
        "accuracy": 1.0,
        "flops": 0.3,
        "energy": 0.25,
        "latency": 0.25,
        "memory": 0.15,
        "params": 0.1
    },
    adaptive_weights=True
)

# Run search
searcher = MultiObjectiveSearch(config)
best_architecture = searcher.search()

# Analyze Pareto front
pareto_front = searcher.get_pareto_front()
print(f"Pareto front contains {len(pareto_front)} solutions")
```

### Dynamic Objective Weighting

```python
# Configure dynamic objective weighting based on hardware constraints
from nanonas.utils.hardware_utils import HardwareConstraintManager

constraint_manager = HardwareConstraintManager(hardware_profile)

# Adapt weights based on constraints
if constraint_manager.check_constraints(performance_metrics)['energy']:
    mo_config.constraint_weights['energy'] = 0.5  # Increase energy importance
if constraint_manager.check_constraints(performance_metrics)['latency']:
    mo_config.constraint_weights['latency'] = 0.4  # Increase latency importance
```

## 🚀 Deployment and Code Generation

### Framework-Specific Code Generation

```python
from nanonas.codegen import PyTorchGenerator, TensorFlowGenerator, ONNXGenerator

# PyTorch code generation
pytorch_gen = PyTorchGenerator()
pytorch_code = pytorch_gen.generate_model_code(architecture)

# TensorFlow code generation
tf_gen = TensorFlowGenerator()
tf_code = tf_gen.generate_model_code(architecture)

# ONNX conversion
onnx_gen = ONNXGenerator()
onnx_model = onnx_gen.convert_to_onnx(architecture)
onnx_gen.save_model(onnx_model, 'model.onnx')
```

### Production Deployment

```python
from nanonas.codegen import DeploymentGenerator

deploy_gen = DeploymentGenerator()

# Generate Docker configuration
dockerfile = deploy_gen.generate_dockerfile('pytorch', quantization='fp16')

# Generate Kubernetes manifests
k8s_manifests = deploy_gen.generate_k8s_manifests('pytorch')

# Generate serving code with FastAPI
serving_code = deploy_gen.generate_fastapi_serving(architecture)
```

### Quantization and Optimization

```python
from nanonas.codegen import OptimizationPasses

optimizer = OptimizationPasses()

# Apply optimization passes
optimized_model = optimizer.apply_operator_fusion(model)
optimized_model = optimizer.apply_quantization(optimized_model, precision='int8')
optimized_model = optimizer.optimize_memory_layout(optimized_model)
```

## 📈 Advanced Visualization and Analysis

### Search Dynamics Visualization

```python
from nanonas.visualization import SearchAnalyzer

analyzer = SearchAnalyzer()

# Visualize search evolution
evolution_plot = analyzer.plot_evolution_dynamics(search_results)
diversity_plot = analyzer.plot_population_diversity(search_results)
landscape_viz = analyzer.visualize_search_landscape(search_results)

# Architecture DNA analysis
dna_viz = analyzer.visualize_architecture_dna(architectures)
```

### Interactive Dashboards

```python
from nanonas.visualization import InteractiveDashboard

dashboard = InteractiveDashboard()

# Create real-time search monitoring
dashboard.create_search_monitor(searcher)
dashboard.start_server(port=8080)  # Access at http://localhost:8080
```

## 🧪 Advanced Training Strategies

### Curriculum Learning and Progressive Training

```python
from nanonas.utils.advanced_training import CurriculumTrainer, ProgressiveTrainer

# Curriculum learning for architecture complexity
curriculum_trainer = CurriculumTrainer(
    complexity_schedule='linear',
    start_complexity=0.1,
    end_complexity=1.0,
    curriculum_epochs=50
)

# Progressive training with increasing resolution
progressive_trainer = ProgressiveTrainer(
    resolution_schedule=[(32, 10), (64, 20), (128, 30)],
    transition_epochs=5
)

# Use in search
config.training.curriculum_trainer = curriculum_trainer
config.training.progressive_trainer = progressive_trainer
```

### Weight Sharing and SuperNet Training

```python
from nanonas.search.oneshot_nas import OneShotNAS, SuperNetTrainer

# Configure one-shot NAS with SuperNet
oneshot_config = OneShotNASConfig(
    supernet_epochs=100,
    architecture_epochs=50,
    path_sampling_strategy='uniform',
    weight_sharing=True
)

# Train SuperNet once, then sample architectures
oneshot_searcher = OneShotNAS(config)
supernet = oneshot_searcher.train_supernet()

# Sample multiple architectures from trained SuperNet
for _ in range(100):
    arch = oneshot_searcher.sample_architecture(supernet)
    performance = oneshot_searcher.evaluate_architecture(arch, supernet)
```

## 📊 Comprehensive Benchmarking

### Statistical Significance Testing

```python
from nanonas.benchmarks import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()

# Compare search strategies with statistical tests
comparison_results = analyzer.compare_strategies(
    strategies=['evolutionary', 'darts', 'progressive_darts', 'bayesian'],
    num_runs=10,
    significance_level=0.05
)

print(analyzer.generate_statistical_report(comparison_results))
```

### Cross-Domain Transfer Analysis

```python
from nanonas.benchmarks import TransferAnalyzer

transfer_analyzer = TransferAnalyzer()

# Analyze architecture transfer across domains
transfer_results = transfer_analyzer.evaluate_transfer(
    source_dataset='cifar10',
    target_datasets=['cifar100', 'fashion_mnist'],
    architectures=discovered_architectures
)

transfer_analyzer.visualize_transfer_matrix(transfer_results)
```

## 🔒 Reproducibility and Experiment Management

### Comprehensive Experiment Tracking

```python
from nanonas.utils.reproducibility import ExperimentTracker

# Initialize experiment tracking
tracker = ExperimentTracker(
    experiment_name="advanced_nas_comparison",
    tracking_backend="wandb",  # or "tensorboard", "mlflow"
    project_name="nanonas-research"
)

# Track search experiment
with tracker.track_experiment():
    tracker.log_config(config)
    tracker.log_hardware_profile(hardware_profile)
    
    best_arch = searcher.search()
    
    tracker.log_architecture(best_arch)
    tracker.log_metrics(search_results)
    tracker.log_artifacts(['model.pt', 'search_progress.png'])
```

### Seed Management and Deterministic Execution

```python
from nanonas.utils.reproducibility import ReproducibilityManager

# Ensure reproducible experiments
repro_manager = ReproducibilityManager(
    global_seed=42,
    deterministic_mode=True,
    benchmark_mode=False
)

with repro_manager.reproducible_context():
    # All operations will be deterministic
    results = nanonas.search(strategy='evolutionary', dataset='cifar10')
```

## 📚 Examples and Tutorials

### Research-Grade Experiment

```python
# Complete research-grade experiment setup
def run_research_experiment():
    # Hardware-aware configuration
    hardware_profile = nanonas.profile_current_device()
    
    # Multi-objective search with custom objectives
    objectives = ["accuracy", "flops", "energy", "latency"]
    
    config = nanonas.ExperimentConfig(
        search=nanonas.SearchConfig(
            strategy="multiobjective",
            objectives=objectives
        ),
        dataset=nanonas.DatasetConfig(name="cifar10"),
        training=nanonas.TrainingConfig(epochs=200),
        evaluation=nanonas.EvaluationConfig(
            metrics=["accuracy", "f1", "precision", "recall"]
        )
    )
    
    # Run search with tracking
    with ExperimentTracker("research_experiment"):
        searcher = nanonas.MultiObjectiveSearch(config)
        best_arch = searcher.search()
        
        # Comprehensive evaluation
        evaluator = nanonas.ModelEvaluator(config)
        results = evaluator.evaluate_full(best_arch.to_model())
        
        # Generate deployment code
        pytorch_gen = nanonas.PyTorchGenerator()
        deployment_code = pytorch_gen.generate_serving_code(best_arch)
        
        return best_arch, results, deployment_code

# Run the experiment
architecture, results, deployment = run_research_experiment()
```

This enhanced nanoNAS system is now a world-class Neural Architecture Search framework suitable for both research and production use. The comprehensive feature set includes state-of-the-art search algorithms, hardware-aware optimization, multi-objective capabilities, and production-ready deployment tools. 