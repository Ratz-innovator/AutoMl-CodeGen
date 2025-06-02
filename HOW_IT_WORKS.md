# How AutoML-CodeGen Works

**A Technical Deep Dive into Neural Architecture Search + Code Generation**

---

## 🏗️ System Architecture Overview

AutoML-CodeGen is built as a modular pipeline that transforms a problem specification into production-ready neural network code. Here's how each component works:

```
Input Problem → Search Space → NAS Algorithm → Best Architecture → Code Generator → Production Code
```

---

## 📋 Core Components

### 1. **Search Space Definition** (`src/automl_codegen/search/space/`)

The search space defines what architectures the system can explore:

```python
class SearchSpace:
    def __init__(self, task, min_layers, max_layers, hardware_target):
        # Define available operations
        self.operations = {
            'conv3x3': ConvOperation(kernel_size=3),
            'conv5x5': ConvOperation(kernel_size=5), 
            'depthwise_conv': DepthwiseConvOperation(),
            'max_pool': PoolingOperation(type='max'),
            'avg_pool': PoolingOperation(type='avg'),
            'skip_connect': SkipConnection(),
            'sep_conv3x3': SeparableConvOperation(kernel_size=3),
            'dil_conv3x3': DilatedConvOperation(kernel_size=3)
        }
        
        # Define architecture constraints
        self.layer_constraints = {
            'min_layers': min_layers,
            'max_layers': max_layers,
            'channel_options': [16, 32, 64, 128, 256, 512],
            'max_connections': 2  # Max skip connections per layer
        }
```

**How it works:**
- Defines a hierarchical graph representation of neural architectures
- Each architecture is encoded as a sequence of operations with connections
- Supports hardware-specific constraints (mobile, GPU, edge devices)
- Automatically estimates complexity (FLOPs, parameters, memory)

### 2. **Neural Architecture Search Algorithms** (`src/automl_codegen/search/algorithms/`)

#### A. **Evolutionary Search** (`evolutionary.py`)
```python
class EvolutionarySearch:
    def evolve_population(self, population):
        # 1. Evaluate fitness of each architecture
        fitness_scores = []
        for arch in population:
            performance = self.evaluate_architecture(arch)
            fitness = self.calculate_multi_objective_fitness(performance)
            fitness_scores.append(fitness)
        
        # 2. Select parents using tournament selection
        parents = self.tournament_selection(population, fitness_scores)
        
        # 3. Create offspring through crossover
        offspring = []
        for i in range(0, len(parents), 2):
            child1, child2 = self.crossover(parents[i], parents[i+1])
            offspring.extend([child1, child2])
        
        # 4. Apply mutations
        for child in offspring:
            if random.random() < self.mutation_rate:
                self.mutate_architecture(child)
        
        # 5. Select next generation (elitism + new offspring)
        next_population = self.environmental_selection(
            population + offspring, fitness_scores
        )
        
        return next_population
```

**Key Innovation:** Custom crossover operators that preserve architectural building blocks while exploring new combinations.

#### B. **DARTS (Differentiable Architecture Search)** (`darts.py`)
```python
class DARTSSearch:
    def __init__(self):
        # Architecture weights (learnable parameters)
        self.alphas = nn.Parameter(torch.randn(num_edges, num_operations))
        
    def forward(self, x):
        # Mixed operation: weighted sum of all possible operations
        for edge in self.edges:
            # Softmax to get operation probabilities
            weights = F.softmax(self.alphas[edge], dim=-1)
            
            # Compute weighted sum of operations
            edge_output = sum(
                weight * operation(x) 
                for weight, operation in zip(weights, self.operations)
            )
            
        return edge_output
    
    def derive_discrete_architecture(self):
        # Convert continuous weights to discrete architecture
        discrete_arch = []
        for edge_alphas in self.alphas:
            # Select operation with highest weight
            best_op = torch.argmax(edge_alphas)
            discrete_arch.append(self.operations[best_op])
        
        return discrete_arch
```

**Key Innovation:** Gradient-based optimization of architecture weights, then discretization to final architecture.

#### C. **Reinforcement Learning Search** (`reinforcement.py`)
```python
class RLController:
    def __init__(self):
        # LSTM controller that generates architectures
        self.controller = nn.LSTM(input_size=embedding_dim, 
                                 hidden_size=hidden_dim,
                                 num_layers=2)
        
    def sample_architecture(self):
        hidden = self.init_hidden()
        architecture = []
        
        for layer_idx in range(max_layers):
            # Sample operation type
            op_logits, hidden = self.controller(embedded_input, hidden)
            op_probs = F.softmax(op_logits, dim=-1)
            operation = torch.multinomial(op_probs, 1)
            
            # Sample hyperparameters for this operation
            params = self.sample_operation_params(operation, hidden)
            
            architecture.append({
                'operation': operation,
                'params': params
            })
            
        return architecture
    
    def update_controller(self, architectures, rewards):
        # REINFORCE algorithm
        loss = 0
        for arch, reward in zip(architectures, rewards):
            # Compute log probability of architecture
            log_prob = self.compute_log_prob(arch)
            
            # Policy gradient with baseline
            loss -= log_prob * (reward - self.baseline)
        
        # Update controller parameters
        loss.backward()
        self.optimizer.step()
```

**Key Innovation:** Neural controller learns to generate high-performing architectures through trial and reward.

### 3. **Multi-Objective Optimization** (`src/automl_codegen/search/objectives/`)

```python
class MultiObjectiveOptimizer:
    def __init__(self, objectives=['accuracy', 'latency', 'memory', 'energy']):
        self.objectives = objectives
        self.weights = self.initialize_weights()
    
    def compute_pareto_frontier(self, population, scores):
        """Find architectures that aren't dominated by others"""
        pareto_front = []
        
        for i, candidate in enumerate(population):
            dominated = False
            
            for j, other in enumerate(population):
                if i != j and self.dominates(scores[j], scores[i]):
                    dominated = True
                    break
            
            if not dominated:
                pareto_front.append(candidate)
        
        return pareto_front
    
    def dominates(self, scores_a, scores_b):
        """Check if solution A dominates solution B"""
        better_in_all = True
        better_in_at_least_one = False
        
        for obj in self.objectives:
            if obj in ['accuracy']:  # Higher is better
                if scores_a[obj] < scores_b[obj]:
                    better_in_all = False
                if scores_a[obj] > scores_b[obj]:
                    better_in_at_least_one = True
            else:  # Lower is better (latency, memory, energy)
                if scores_a[obj] > scores_b[obj]:
                    better_in_all = False
                if scores_a[obj] < scores_b[obj]:
                    better_in_at_least_one = True
        
        return better_in_all and better_in_at_least_one
```

### 4. **Architecture Evaluation** (`src/automl_codegen/evaluation/`)

#### A. **Training Pipeline** (`trainer.py`)
```python
class ArchitectureTrainer:
    def evaluate_architecture(self, architecture):
        # 1. Build PyTorch model from architecture
        model = self.build_model_from_architecture(architecture)
        
        # 2. Train for a few epochs (early estimation)
        train_acc, val_acc = self.quick_training(model, epochs=5)
        
        # 3. Estimate final performance using learning curves
        estimated_final_acc = self.extrapolate_performance(
            train_acc, val_acc
        )
        
        return {
            'accuracy': estimated_final_acc,
            'train_time': self.last_train_time,
            'convergence_speed': self.compute_convergence_rate()
        }
    
    def quick_training(self, model, epochs=5):
        """Fast training for architecture evaluation"""
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        
        for epoch in range(epochs):
            # Training loop with reduced dataset
            train_acc = self.train_epoch(model, optimizer, 
                                       subset_ratio=0.1)  # Use 10% of data
            val_acc = self.validate_epoch(model, subset_ratio=0.1)
            
        return train_acc, val_acc
```

#### B. **Hardware Profiling** (`hardware.py`)
```python
class HardwareProfiler:
    def profile_architecture(self, architecture, device='cuda'):
        """Get real hardware performance metrics"""
        model = self.build_model(architecture)
        model = model.to(device)
        
        # 1. Memory profiling
        torch.cuda.empty_cache()
        input_tensor = torch.randn(1, 3, 224, 224).to(device)
        
        # Measure memory before
        memory_before = torch.cuda.memory_allocated()
        
        # Forward pass
        with torch.no_grad():
            output = model(input_tensor)
        
        # Measure memory after
        memory_after = torch.cuda.memory_allocated()
        memory_usage = memory_after - memory_before
        
        # 2. Latency profiling
        latencies = []
        for _ in range(100):  # Multiple runs for stability
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = model(input_tensor)
            torch.cuda.synchronize()  # Wait for GPU
            end_time = time.perf_counter()
            latencies.append(end_time - start_time)
        
        avg_latency = np.mean(latencies[10:])  # Skip warmup runs
        
        # 3. Energy estimation (model-based)
        energy_per_inference = self.estimate_energy_consumption(
            model, device
        )
        
        return {
            'latency_ms': avg_latency * 1000,
            'memory_mb': memory_usage / 1024 / 1024,
            'energy_mj': energy_per_inference
        }
```

### 5. **Code Generation** (`src/automl_codegen/codegen/`)

```python
class PyTorchGenerator:
    def generate_model(self, architecture):
        """Generate PyTorch code from architecture specification"""
        
        # 1. Analyze architecture structure
        layers = self.parse_architecture_layers(architecture)
        connections = self.parse_skip_connections(architecture)
        
        # 2. Generate class definition
        class_code = f"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoGeneratedNet(nn.Module):
    def __init__(self, num_classes={architecture['num_classes']}):
        super(AutoGeneratedNet, self).__init__()
        
        # Generated layers
{self.generate_layer_definitions(layers)}
        
        # Skip connections
{self.generate_skip_connections(connections)}
        
    def forward(self, x):
{self.generate_forward_pass(layers, connections)}
        
        return x
"""
        
        # 3. Add optimizations
        optimized_code = self.apply_optimizations(class_code)
        
        # 4. Add training script
        training_code = self.generate_training_script(architecture)
        
        # 5. Add inference script
        inference_code = self.generate_inference_script(architecture)
        
        return {
            'model.py': optimized_code,
            'train.py': training_code,
            'inference.py': inference_code,
            'requirements.txt': self.generate_requirements()
        }
    
    def generate_layer_definitions(self, layers):
        """Convert layer specifications to PyTorch code"""
        layer_code = []
        
        for i, layer in enumerate(layers):
            if layer['type'] == 'conv':
                code = f"self.conv_{i} = nn.Conv2d({layer['in_channels']}, "
                code += f"{layer['out_channels']}, {layer['kernel_size']}, "
                code += f"padding={layer['padding']})"
                
            elif layer['type'] == 'depthwise_conv':
                code = f"self.depthwise_{i} = nn.Conv2d({layer['channels']}, "
                code += f"{layer['channels']}, {layer['kernel_size']}, "
                code += f"groups={layer['channels']}, padding={layer['padding']})"
                
            # Add more operation types...
            
            layer_code.append(f"        {code}")
        
        return '\n'.join(layer_code)
    
    def apply_optimizations(self, code):
        """Apply code optimizations"""
        # 1. Operator fusion
        code = self.fuse_conv_batchnorm(code)
        
        # 2. Memory optimizations
        code = self.add_gradient_checkpointing(code)
        
        # 3. JIT compilation hints
        code = self.add_jit_annotations(code)
        
        return code
```

---

## 🔄 Complete Workflow

### Phase 1: Problem Setup
```python
# User defines the task
nas = NeuralArchitectureSearch(
    task='image_classification',
    dataset='cifar10',
    objectives=['accuracy', 'latency', 'memory'],
    hardware_target='mobile'
)
```

### Phase 2: Search Space Creation
```python
# System creates constrained search space
search_space = SearchSpace(
    operations=['conv3x3', 'conv5x5', 'depthwise_conv', 'skip_connect'],
    hardware_constraints=mobile_constraints,
    complexity_limits={'max_params': 5e6, 'max_flops': 500e6}
)
```

### Phase 3: Architecture Search
```python
# Multi-algorithm search
best_architectures = []

# Run evolutionary search
evo_results = evolutionary_search.search(generations=20)
best_architectures.extend(evo_results)

# Run DARTS search  
darts_results = darts_search.search(epochs=50)
best_architectures.extend(darts_results)

# Run RL search
rl_results = rl_search.search(iterations=1000)
best_architectures.extend(rl_results)

# Find Pareto optimal solutions
pareto_front = multi_objective_optimizer.compute_pareto_frontier(
    best_architectures
)
```

### Phase 4: Architecture Selection
```python
# Select final architecture based on user preferences
final_arch = pareto_front.select_best(
    accuracy_weight=0.7,
    latency_weight=0.2, 
    memory_weight=0.1
)
```

### Phase 5: Code Generation
```python
# Generate production code
codegen = CodeGenerator(target_framework='pytorch')
generated_files = codegen.generate(final_arch)

# Files created:
# - model.py: Neural network implementation
# - train.py: Training script with data loaders
# - inference.py: Optimized inference pipeline
# - deploy.py: Deployment utilities
# - requirements.txt: Dependencies
```

---

## 🧠 Key Innovations

### 1. **Multi-Algorithm Ensemble**
- Combines strengths of evolutionary, gradient-based, and RL approaches
- Different algorithms explore different regions of search space
- Ensemble voting for robust architecture selection

### 2. **Real Hardware Profiling**
- Actual GPU memory measurements during search
- Real-world latency profiling on target hardware
- Energy consumption modeling for mobile deployment

### 3. **Progressive Search Refinement**
- Starts with coarse search across large space
- Progressively refines around promising regions
- Adaptive resource allocation based on search progress

### 4. **Production-Ready Code Generation**
- Not just architecture discovery, but complete deployment pipeline
- Automatic optimization passes (operator fusion, quantization)
- Multi-framework support with consistency guarantees

---

## 🔧 Technical Implementation Details

### Memory Efficiency
- Gradient checkpointing for reduced memory usage
- Mixed precision training support
- Efficient architecture encoding (graph compression)

### Distributed Computing
- Asynchronous architecture evaluation across multiple GPUs
- Population-based search with worker synchronization
- Fault tolerance and automatic recovery

### Performance Optimization
- JIT compilation of generated models
- Automatic operator fusion detection
- Hardware-specific optimization passes

---

## 📊 Validation & Testing

The system includes 11 comprehensive tests:

1. **Search Space Validation**: Ensures all generated architectures are valid
2. **Algorithm Correctness**: Verifies each NAS algorithm implementation
3. **Multi-Objective Optimization**: Tests Pareto frontier computation
4. **Hardware Profiling Accuracy**: Validates performance measurements
5. **Code Generation Correctness**: Ensures generated code compiles and runs
6. **End-to-End Pipeline**: Full system integration test
7. **Performance Regression**: Monitors search quality over time
8. **Memory Usage**: Prevents memory leaks during long searches
9. **Distributed Computing**: Tests multi-GPU coordination
10. **Error Handling**: Robust failure recovery
11. **Code Quality**: Syntax validation and optimization verification

---

This architecture enables AutoML-CodeGen to automatically discover high-performing neural networks and generate production-ready code, bridging the gap between research and deployment. 