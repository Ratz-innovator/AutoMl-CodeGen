# nanoNAS: Neural Architecture Search Made Simple

**The educational way to learn and use Neural Architecture Search**

---

## ⚡ Quick Start

```bash
# Install dependencies
pip install torch numpy matplotlib seaborn

# Run architecture search
python nanonas.py
```

```python
# Use in your code
from nanonas import nano_nas

model = nano_nas('evolution', generations=10)
# That's it! Your model is ready to train
```

---

## 🎯 What is nanoNAS?

**Neural Architecture Search** automatically finds optimal neural network architectures. Instead of manually designing networks, let evolution or gradients do the work!

**nanoNAS** makes this **accessible to everyone** with:
- 🔬 **<200 lines of code** - Understand every detail
- 🎓 **Educational focus** - Learn concepts, not just use tools
- ⚡ **Immediate utility** - Working models in minutes
- 🧬 **Novel insights** - Unique visualizations and metaphors

---

## 📁 Project Structure

```
nanoNAS/
├── nanonas.py              # Core NAS in <200 lines
├── nas_insights.py         # Educational visualizations
├── test_nano_nas.py        # Simple tests
├── examples/quickstart.py  # Get started example
├── README_KARPATHY.md      # Deep educational guide
└── requirements.txt        # Minimal dependencies
```

---

## 🚀 Features

### 🔬 **Minimal Core Implementation**
- **Evolutionary Search** - Let architectures evolve like biology
- **DARTS** - Gradient-based differentiable search  
- **Clean API** - One line to get optimal models

### 🧠 **Educational Insights**
- **Architecture DNA** - See architectures as genetic sequences
- **Search Landscapes** - Visualize the optimization terrain
- **Pareto Frontiers** - Understand multi-objective trade-offs

### ⚡ **Immediate Utility**
- **Ready-to-train models** - Standard PyTorch outputs
- **Customizable search** - Tune for your needs
- **Real research methods** - Not toy implementations

---

## 🎓 Learn More

- **📖 [Educational Guide](README_KARPATHY.md)** - Deep dive into concepts
- **🚀 [Quickstart Example](examples/quickstart.py)** - Get started immediately
- **🧪 [Run Tests](test_nano_nas.py)** - Verify everything works

---

## 🧬 Core Concepts

### Architecture = DNA
```python
# Every architecture is just a sequence of choices
architecture = [0, 1, 2, 3]  # conv3x3, conv5x5, maxpool, skip

# Evolution finds better sequences  
better_arch = [1, 3, 3, 0]   # Through mutation and selection
```

### Search Strategies
- **🧬 Evolution** - Broad exploration, robust results
- **📈 DARTS** - Fast convergence, gradient-based

---

## 🎯 Why nanoNAS?

| Traditional NAS | nanoNAS |
|-----------------|---------|
| ❌ Complex frameworks | ✅ <200 lines of clear code |
| ❌ Hidden implementations | ✅ Transparent algorithms |
| ❌ Hard to modify | ✅ Easy to extend |
| ❌ Performance focus | ✅ Educational focus |

---

## 🤝 Contributing

**Found this helpful?**
- ⭐ **Star the repo** - Help others discover it
- 🐛 **Report issues** - Help us improve
- 💡 **Suggest features** - What would make it better?
- 📚 **Improve docs** - Make it clearer for others

---

## 📜 License

MIT License - Educational use encouraged!

---

*Making Neural Architecture Search as simple as training a model* ✨ 