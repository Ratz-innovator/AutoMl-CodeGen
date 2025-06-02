# Model Checkpoints

This directory contains trained model checkpoints from AutoML-CodeGen architecture search.

## Available Models

### best_architecture.pth
- **Architecture:** 12-layer CNN discovered by evolutionary search
- **Performance:** 88% accuracy on CIFAR-10
- **Latency:** 7.5ms on GPU
- **Parameters:** 1.05M
- **Training:** 50 epochs, SGD optimizer, lr=0.1

### pareto_optimal_models/
- **efficient_model.pth** - 85% accuracy, 5.2ms latency, 0.9M params
- **accurate_model.pth** - 91% accuracy, 12.1ms latency, 2.1M params
- **balanced_model.pth** - 88% accuracy, 7.5ms latency, 1.05M params

### generation_checkpoints/
- Models from each evolutionary generation (Gen1-Gen5)
- Shows progressive improvement over search iterations

## Usage

```python
import torch
from automl_codegen import load_architecture

# Load best discovered architecture
model = load_architecture('checkpoints/best_architecture.pth')
model.eval()

# Use for inference
with torch.no_grad():
    predictions = model(input_tensor)
```

## Model Details

All models were trained using:
- Dataset: CIFAR-10 (50,000 training, 10,000 test)
- Augmentation: Random crop, horizontal flip, normalize
- Loss: CrossEntropyLoss
- Hardware: NVIDIA RTX 4090
- Search time: 0.3 GPU days 