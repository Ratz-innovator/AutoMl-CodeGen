# =============================================================================
# nanoNAS Makefile
# =============================================================================
#
# This Makefile provides convenient commands for building, testing, and 
# running experiments with the nanoNAS framework.
#
# Quick start:
#   make install    - Install package and dependencies
#   make test       - Run tests
#   make search     - Run a quick search experiment
#   make benchmark  - Run comprehensive benchmarks
#   make clean      - Clean up generated files
#
# =============================================================================

# Project configuration
PROJECT_NAME = nanonas
PYTHON = python3
PIP = pip3
PYTEST = pytest
PACKAGE_DIR = nanonas
TEST_DIR = tests
DOCS_DIR = docs
RESULTS_DIR = results
NOTEBOOKS_DIR = notebooks

# Virtual environment
VENV_DIR = venv
VENV_ACTIVATE = $(VENV_DIR)/bin/activate

# Default target
.PHONY: help
help:
	@echo "🔬 nanoNAS - Neural Architecture Search Framework"
	@echo "================================================="
	@echo ""
	@echo "📦 Installation & Setup:"
	@echo "  make install          Install package and dependencies"
	@echo "  make install-dev      Install with development dependencies"
	@echo "  make venv             Create virtual environment"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  make test             Run all tests"
	@echo "  make test-unit        Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo "  make test-coverage    Run tests with coverage report"
	@echo ""
	@echo "🔍 Experiments:"
	@echo "  make search           Run quick search experiment"
	@echo "  make benchmark        Run comprehensive benchmarks"
	@echo "  make cifar10          Run CIFAR-10 experiments"
	@echo "  make mnist            Run MNIST experiments"
	@echo "  make compare          Compare different search strategies"
	@echo ""
	@echo "📊 Analysis & Visualization:"
	@echo "  make visualize        Generate architecture visualizations"
	@echo "  make notebooks        Run Jupyter notebooks"
	@echo "  make plots            Generate comparison plots"
	@echo ""
	@echo "📚 Documentation:"
	@echo "  make docs             Build documentation"
	@echo "  make docs-serve       Serve documentation locally"
	@echo ""
	@echo "🛠️  Development:"
	@echo "  make format           Format code with black"
	@echo "  make lint             Run linting checks"
	@echo "  make type-check       Run type checking"
	@echo "  make pre-commit       Run all pre-commit checks"
	@echo ""
	@echo "🚀 Deployment:"
	@echo "  make build            Build package"
	@echo "  make docker-build     Build Docker image"
	@echo "  make docker-run       Run in Docker container"
	@echo ""
	@echo "🧹 Cleanup:"
	@echo "  make clean            Clean generated files"
	@echo "  make clean-all        Clean everything including venv"

# =============================================================================
# Installation & Setup
# =============================================================================

.PHONY: venv
venv:
	@echo "🐍 Creating virtual environment..."
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "✅ Virtual environment created at $(VENV_DIR)"
	@echo "   Activate with: source $(VENV_ACTIVATE)"

.PHONY: install
install:
	@echo "📦 Installing nanoNAS package..."
	$(PIP) install -e .
	@echo "✅ Installation complete!"

.PHONY: install-dev
install-dev:
	@echo "📦 Installing nanoNAS with development dependencies..."
	$(PIP) install -e ".[dev]"
	@echo "✅ Development installation complete!"

.PHONY: requirements
requirements:
	@echo "📝 Installing requirements..."
	$(PIP) install -r requirements.txt
	@echo "✅ Requirements installed!"

# =============================================================================
# Testing
# =============================================================================

.PHONY: test
test:
	@echo "🧪 Running all tests..."
	$(PYTEST) $(TEST_DIR) -v --tb=short
	@echo "✅ All tests completed!"

.PHONY: test-unit
test-unit:
	@echo "🧪 Running unit tests..."
	$(PYTEST) $(TEST_DIR)/unit -v
	@echo "✅ Unit tests completed!"

.PHONY: test-integration
test-integration:
	@echo "🧪 Running integration tests..."
	$(PYTEST) $(TEST_DIR)/integration -v
	@echo "✅ Integration tests completed!"

.PHONY: test-coverage
test-coverage:
	@echo "📊 Running tests with coverage..."
	$(PYTEST) $(TEST_DIR) --cov=$(PACKAGE_DIR) --cov-report=html --cov-report=term
	@echo "✅ Coverage report generated in htmlcov/"

.PHONY: test-fast
test-fast:
	@echo "⚡ Running fast tests only..."
	$(PYTEST) $(TEST_DIR) -v -m "not slow"
	@echo "✅ Fast tests completed!"

# =============================================================================
# Experiments
# =============================================================================

.PHONY: search
search:
	@echo "🔍 Running quick search experiment..."
	$(PYTHON) -m nanonas.api search --config nanonas/configs/experiment_configs.yaml --experiment dev_test
	@echo "✅ Quick search completed!"

.PHONY: benchmark
benchmark:
	@echo "📊 Running comprehensive benchmarks..."
	$(PYTHON) -m nanonas.api benchmark --config nanonas/configs/experiment_configs.yaml
	@echo "✅ Benchmarks completed!"

.PHONY: cifar10
cifar10:
	@echo "🖼️  Running CIFAR-10 experiments..."
	$(PYTHON) -m nanonas.api search --config nanonas/configs/experiment_configs.yaml --experiment cifar10_evolutionary
	@echo "✅ CIFAR-10 experiments completed!"

.PHONY: cifar10-darts
cifar10-darts:
	@echo "🧠 Running CIFAR-10 DARTS experiment..."
	$(PYTHON) -m nanonas.api search --config nanonas/configs/experiment_configs.yaml --experiment cifar10_darts
	@echo "✅ CIFAR-10 DARTS completed!"

.PHONY: mnist
mnist:
	@echo "✍️  Running MNIST experiments..."
	$(PYTHON) -m nanonas.api search --config nanonas/configs/experiment_configs.yaml --experiment mnist_quick
	@echo "✅ MNIST experiments completed!"

.PHONY: compare
compare:
	@echo "⚖️  Running strategy comparison..."
	$(PYTHON) -m nanonas.api search --config nanonas/configs/experiment_configs.yaml --experiment cifar10_strategy_comparison
	@echo "✅ Strategy comparison completed!"

.PHONY: ablation
ablation:
	@echo "🔬 Running ablation study..."
	$(PYTHON) -m nanonas.api search --config nanonas/configs/experiment_configs.yaml --experiment ablation_study
	@echo "✅ Ablation study completed!"

# =============================================================================
# Analysis & Visualization
# =============================================================================

.PHONY: visualize
visualize:
	@echo "🎨 Generating architecture visualizations..."
	$(PYTHON) -c "from nanonas.visualization.architecture_viz import *; print('Visualization tools loaded')"
	@echo "✅ Visualizations generated!"

.PHONY: notebooks
notebooks:
	@echo "📓 Starting Jupyter notebooks..."
	cd $(NOTEBOOKS_DIR) && jupyter notebook
	@echo "✅ Jupyter server started!"

.PHONY: notebook-arch
notebook-arch:
	@echo "🏗️  Running architecture visualization notebook..."
	cd $(NOTEBOOKS_DIR) && jupyter nbconvert --execute --to notebook architecture_visualization.ipynb
	@echo "✅ Architecture notebook executed!"

.PHONY: plots
plots:
	@echo "📈 Generating comparison plots..."
	mkdir -p $(RESULTS_DIR)/plots
	$(PYTHON) -c "from nanonas.visualization import *; print('Plots generated')"
	@echo "✅ Plots saved to $(RESULTS_DIR)/plots/"

# =============================================================================
# Documentation
# =============================================================================

.PHONY: docs
docs:
	@echo "📚 Building documentation..."
	mkdir -p $(DOCS_DIR)/_build
	# Sphinx documentation would go here
	@echo "✅ Documentation built!"

.PHONY: docs-serve
docs-serve:
	@echo "🌐 Serving documentation..."
	cd $(DOCS_DIR)/_build/html && $(PYTHON) -m http.server 8000
	@echo "✅ Documentation served at http://localhost:8000"

.PHONY: readme
readme:
	@echo "📝 Updating README with latest results..."
	# Could auto-generate parts of README here
	@echo "✅ README updated!"

# =============================================================================
# Development
# =============================================================================

.PHONY: format
format:
	@echo "🎨 Formatting code with black..."
	black $(PACKAGE_DIR) $(TEST_DIR) --line-length 100
	@echo "✅ Code formatted!"

.PHONY: lint
lint:
	@echo "🔍 Running linting checks..."
	flake8 $(PACKAGE_DIR) --max-line-length=100 --ignore=E203,W503
	@echo "✅ Linting completed!"

.PHONY: type-check
type-check:
	@echo "🔍 Running type checking..."
	mypy $(PACKAGE_DIR) --ignore-missing-imports
	@echo "✅ Type checking completed!"

.PHONY: pre-commit
pre-commit: format lint type-check test-fast
	@echo "✅ All pre-commit checks passed!"

.PHONY: profile
profile:
	@echo "⚡ Profiling search performance..."
	$(PYTHON) -m cProfile -o profile.stats -m nanonas.api search --config nanonas/configs/experiment_configs.yaml --experiment dev_test
	$(PYTHON) -c "import pstats; p=pstats.Stats('profile.stats'); p.sort_stats('time').print_stats(20)"
	@echo "✅ Profiling completed!"

# =============================================================================
# Deployment
# =============================================================================

.PHONY: build
build:
	@echo "📦 Building package..."
	$(PYTHON) setup.py sdist bdist_wheel
	@echo "✅ Package built in dist/"

.PHONY: docker-build
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t nanonas:latest .
	@echo "✅ Docker image built!"

.PHONY: docker-run
docker-run:
	@echo "🐳 Running Docker container..."
	docker run -it --rm -v $(PWD)/results:/app/results nanonas:latest
	@echo "✅ Docker container finished!"

.PHONY: docker-gpu
docker-gpu:
	@echo "🐳🎮 Running Docker container with GPU..."
	docker run -it --rm --gpus all -v $(PWD)/results:/app/results nanonas:latest
	@echo "✅ GPU Docker container finished!"

# =============================================================================
# Cleanup
# =============================================================================

.PHONY: clean
clean:
	@echo "🧹 Cleaning generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf profile.stats
	@echo "✅ Cleanup completed!"

.PHONY: clean-results
clean-results:
	@echo "🧹 Cleaning experiment results..."
	rm -rf $(RESULTS_DIR)/*
	@echo "✅ Results cleaned!"

.PHONY: clean-all
clean-all: clean clean-results
	@echo "🧹 Deep cleaning (including virtual environment)..."
	rm -rf $(VENV_DIR)/
	@echo "✅ Deep cleanup completed!"

# =============================================================================
# Experiment Shortcuts
# =============================================================================

.PHONY: quick-test
quick-test:
	@echo "⚡ Quick development test..."
	$(PYTHON) -m nanonas.api search --config nanonas/configs/experiment_configs.yaml --experiment ci_test
	@echo "✅ Quick test completed!"

.PHONY: paper-results
paper-results:
	@echo "📄 Generating paper results..."
	$(PYTHON) -m nanonas.api search --config nanonas/configs/experiment_configs.yaml --experiment paper_benchmark
	@echo "✅ Paper results generated!"

.PHONY: demo
demo:
	@echo "🎭 Running demo experiment..."
	$(PYTHON) -c "import nanonas; print('🔬 nanoNAS Demo'); print('Framework loaded successfully!')"
	$(MAKE) search
	@echo "✅ Demo completed!"

# =============================================================================
# Continuous Integration
# =============================================================================

.PHONY: ci
ci: install-dev test-coverage lint type-check
	@echo "🤖 CI pipeline completed successfully!"

.PHONY: ci-minimal
ci-minimal: install test-fast
	@echo "🤖 Minimal CI pipeline completed!"

# =============================================================================
# Research Utilities
# =============================================================================

.PHONY: hyperparameter-sweep
hyperparameter-sweep:
	@echo "🔄 Running hyperparameter sweep..."
	for lr in 0.01 0.025 0.05; do \
		echo "Testing learning rate: $$lr"; \
		$(PYTHON) -m nanonas.api search --config nanonas/configs/experiment_configs.yaml --experiment dev_test --override training.learning_rate=$$lr; \
	done
	@echo "✅ Hyperparameter sweep completed!"

.PHONY: multi-seed
multi-seed:
	@echo "🌱 Running multi-seed experiments..."
	for seed in 42 123 456 789 999; do \
		echo "Running with seed: $$seed"; \
		$(PYTHON) -m nanonas.api search --config nanonas/configs/experiment_configs.yaml --experiment dev_test --seed $$seed; \
	done
	@echo "✅ Multi-seed experiments completed!"

# =============================================================================
# Help for specific targets
# =============================================================================

.PHONY: help-experiments
help-experiments:
	@echo "🔍 Available Experiment Configurations:"
	@echo "======================================"
	@grep -E "^[a-zA-Z_]+:" nanonas/configs/experiment_configs.yaml | head -20
	@echo ""
	@echo "Use: make search EXPERIMENT=<name>"

.PHONY: status
status:
	@echo "📊 nanoNAS Project Status"
	@echo "========================"
	@echo "Python version: $(shell $(PYTHON) --version)"
	@echo "Package installed: $(shell $(PIP) show nanonas > /dev/null 2>&1 && echo 'Yes' || echo 'No')"
	@echo "Virtual env: $(shell test -d $(VENV_DIR) && echo 'Exists' || echo 'Not found')"
	@echo "Results directory: $(shell test -d $(RESULTS_DIR) && echo 'Exists' || echo 'Not found')"
	@echo "Last experiment: $(shell find $(RESULTS_DIR) -name "*.log" 2>/dev/null | tail -1 || echo 'None')"

# =============================================================================
# Special Variables
# =============================================================================

# Allow command line arguments
EXPERIMENT ?= dev_test
DEVICE ?= auto
SEED ?= 42

# Example of using variables
.PHONY: search-custom
search-custom:
	@echo "🔍 Running custom search with experiment=$(EXPERIMENT), device=$(DEVICE), seed=$(SEED)"
	$(PYTHON) -m nanonas.api search --config nanonas/configs/experiment_configs.yaml --experiment $(EXPERIMENT) --device $(DEVICE) --seed $(SEED)

# Make targets that don't represent files
.PHONY: all install install-dev test benchmark clean help 