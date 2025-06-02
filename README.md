# nanoNAS: Neural Architecture Search Made Simple

<div align="center">

**🧠 The educational way to learn and use Neural Architecture Search**

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Educational](https://img.shields.io/badge/Purpose-Educational-green)](README_KARPATHY.md)
[![Code Style](https://img.shields.io/badge/Code%20Style-Karpathy--Level-gold)](nanonas.py)

[🚀 Quick Start](#-quick-start) • 
[📖 Learn Concepts](#-core-concepts) • 
[🎓 Deep Dive](README_KARPATHY.md) • 
[🧪 Run Demo](#-demo)

</div>

---

## ⚡ Quick Start

```bash
# 1. Install dependencies (only 4 packages!)
pip install torch numpy matplotlib seaborn

# 2. Run architecture search
python nanonas.py

# 3. Try the interactive demo
python demo.py
```

**That's it!** Neural Architecture Search in 3 commands.

### Use in Your Code
```python
from nanonas import nano_nas

# One line to get an optimal model
model = nano_nas('evolution', generations=10)

# Ready to train with standard PyTorch!
optimizer = torch.optim.Adam(model.parameters())
```

---

## 🎯 What is nanoNAS?

**Neural Architecture Search** automatically finds optimal neural network architectures. Instead of manually designing networks, let evolution or gradients do the work!

**nanoNAS** makes this **accessible to everyone**:

| Traditional NAS | nanoNAS |
|-----------------|---------|
| ❌ 1000+ lines of complex code | ✅ <200 lines of clear code |
| ❌ Dozens of dependencies | ✅ 4 essential packages |
| ❌ Hidden implementations | ✅ Transparent algorithms |
| ❌ Performance focus | ✅ **Educational focus** |

---

## 📁 Crystal Clear Structure

```
nanoNAS/
├── 📖 README.md              # You are here
├── 📚 README_KARPATHY.md     # Deep educational guide
├── 🔬 nanonas.py             # Core NAS in <200 lines
├── 🧠 nas_insights.py        # Novel educational visualizations
├── 🧪 test_nano_nas.py       # Simple comprehensive tests
├── 🎬 demo.py                # Beautiful interactive demo
├── 📦 requirements.txt       # Minimal dependencies
└── 📁 examples/
    └── quickstart.py         # Get started in 3 lines
```

**Everything has a purpose. Nothing is cluttered.**

---

## 🚀 Core Features

### 🔬 **Minimal Core Implementation**
- **🧬 Evolutionary Search** - Let architectures evolve like biology
- **📈 DARTS** - Gradient-based differentiable search  
- **⚡ One-line API** - `model = nano_nas('evolution')`

### 🧠 **Educational Insights (Novel!)**
- **🧬 Architecture DNA** - See architectures as genetic sequences
- **🏔️ Search Landscapes** - Visualize the optimization terrain
- **🎯 Pareto Frontiers** - Understand multi-objective trade-offs

### ⚡ **Immediate Utility**
- **🏗️ Ready-to-train models** - Standard PyTorch outputs
- **⚙️ Customizable search** - Tune for your needs
- **🔬 Real research methods** - Not toy implementations

---

## 🧬 Core Concepts

### Architecture = DNA
```python
# Every architecture is just a sequence of choices
architecture = [0, 1, 2, 3]  # conv3x3, conv5x5, maxpool, skip

# Evolution finds better sequences  
better_arch = [1, 3, 3, 0]   # Through mutation and selection
```

### Two Search Strategies
- **🧬 Evolution** - Broad exploration, robust results
- **📈 DARTS** - Fast convergence, gradient-based optimization

---

## 🎬 Demo

Run the beautiful interactive demo:

```bash
python demo.py
```

You'll see:
- 🧬 Architecture DNA evolution in action
- ⚖️ Comparison of search algorithms  
- 🎯 Live architecture generation
- 📊 Educational insights and visualizations

---

## 🎓 Learn More

| Resource | Description |
|----------|-------------|
| 📖 **[Educational Guide](README_KARPATHY.md)** | Deep dive into NAS concepts with novel insights |
| 🚀 **[Quickstart](examples/quickstart.py)** | Working example in 3 lines |
| 🧪 **[Tests](test_nano_nas.py)** | Verify everything works perfectly |
| 🎬 **[Demo](demo.py)** | Interactive showcase of all features |

---

## ✅ Verified Working

Run the tests to verify everything works:
```bash
python test_nano_nas.py
# ✅ All operations working!
# ✅ Architecture DNA working!
# ✅ Evolutionary search working!
# ✅ DARTS search working!
# 🎉 All tests passed!
```

---

## 🌟 Why nanoNAS is Special

### 📚 **Educational Excellence**
- **Karpathy-level clarity** - Every line is understandable
- **Novel metaphors** - Architecture DNA, Search Landscapes
- **Learning > Performance** - Understanding over optimization

### 🔬 **Minimal Brilliance**
- **<200 lines** capture NAS essence
- **4 dependencies** instead of dozens
- **Zero bloat** - every component serves education

### 🚀 **Immediate Impact**
- **One-line usage** - instant neural architecture search
- **Real PyTorch models** - ready to train immediately
- **Production quality** - not just educational toys

---

## 🤝 Contributing

**Found this helpful?**
- ⭐ **Star the repo** - Help others discover it
- 🐛 **Report issues** - Help us improve
- 💡 **Suggest features** - What would make it better?
- 📚 **Improve docs** - Make it clearer for others

---

## 🎯 Perfect For

- **🎓 Students** learning neural architecture search
- **🔬 Researchers** needing clean reference implementations
- **👩‍💻 Practitioners** wanting automated architecture design
- **🏫 Educators** teaching AI concepts with clarity
- **🧠 Anyone curious** about how NAS actually works

---

## 📜 License

Open source educational project - Use freely for learning and research!

---

<div align="center">

**🧠 Making Neural Architecture Search as simple as training a model ✨**

*Built with ❤️ for the AI education community*

</div> 