# 🤝 Contributing to nanoNAS

We love your input! We want to make contributing to nanoNAS as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## 🎯 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Research Contributions](#research-contributions)
- [Community](#community)

## 📜 Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to ensure a welcoming environment for everyone.

## 🚀 Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

### Pull Requests

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## 🛠️ Development Setup

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Git

### Setup

```bash
# Clone your fork
git clone https://github.com/your-username/nanoNAS.git
cd nanoNAS

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Development Dependencies

```bash
pip install pytest pytest-cov black mypy flake8 pre-commit
```

## 🧪 Testing

### Running Tests

```bash
# Run basic functionality tests
python simple_test.py
python test_package.py

# Run full test suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=nanonas --cov-report=html
```

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names
- Follow the existing test structure
- Aim for >90% code coverage

Example test:
```python
def test_architecture_creation():
    """Test that architectures can be created properly."""
    search_space = SearchSpace.get_nano_search_space()
    arch = Architecture([0, 1, 2], search_space)
    assert arch.encoding == [0, 1, 2]
    assert arch.search_space == search_space
```

## 📝 Code Style

We use several tools to maintain code quality:

### Formatting

```bash
# Format code with black
black nanonas/ tests/

# Check formatting
black --check nanonas/ tests/
```

### Type Checking

```bash
# Run mypy
mypy nanonas/
```

### Linting

```bash
# Run flake8
flake8 nanonas/ tests/
```

### Pre-commit Hooks

We use pre-commit to run checks automatically:

```bash
# Install hooks
pre-commit install

# Run all hooks manually
pre-commit run --all-files
```

## 📚 Documentation

### Code Documentation

- Use docstrings for all public functions and classes
- Follow Google-style docstrings
- Include type hints for all function signatures

Example:
```python
def search(strategy: str, dataset: str, **kwargs) -> nn.Module:
    """
    Perform neural architecture search.
    
    Args:
        strategy: Search strategy to use ('evolutionary', 'darts', etc.)
        dataset: Dataset name ('cifar10', 'cifar100', etc.)
        **kwargs: Additional search parameters
        
    Returns:
        Best found model
        
    Raises:
        ValueError: If strategy or dataset is invalid
    """
```

### README Updates

If your changes affect the user interface:
- Update the README.md
- Add examples if relevant
- Update the quick start guide

## 🐛 Bug Reports

Use GitHub issues to report bugs. Include:

1. **Quick summary** of the bug
2. **Steps to reproduce**
3. **Expected behavior**
4. **Actual behavior**
5. **Environment details** (OS, Python version, PyTorch version)
6. **Code sample** if applicable

**Great Bug Reports** tend to have:

- Clear, specific summary
- Minimal reproducible example
- Stack trace if there's an exception
- Environment information

## 💡 Feature Requests

We love feature requests! Use GitHub issues with the "enhancement" label.

Please provide:
- **Use case**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives considered**: What other approaches did you consider?

## 🏗️ Architecture Guidelines

### Adding New Search Strategies

1. Create a new file in `nanonas/search/`
2. Inherit from `BaseSearchStrategy`
3. Implement required methods
4. Add tests
5. Update documentation

Example structure:
```python
class NewSearchStrategy(BaseSearchStrategy):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
    
    def search(self) -> Architecture:
        # Implementation here
        pass
```

### Adding New Operations

1. Add operation to `nanonas/models/operations.py`
2. Update search spaces if needed
3. Add tests
4. Document the operation

### Adding New Datasets

1. Add dataset loader to `nanonas/benchmarks/datasets.py`
2. Update configuration options
3. Add tests
4. Document usage

## 🎯 Contribution Ideas

### For Beginners

- Fix typos in documentation
- Add more test cases
- Improve error messages
- Add type hints to uncovered code

### For Intermediate

- Implement new search strategies
- Add visualization features
- Improve performance of existing algorithms
- Add new neural operations

### For Advanced

- Add support for new hardware (TPU, mobile)
- Implement advanced NAS techniques
- Add multi-objective optimization features
- Performance optimizations

## 📋 Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- MAJOR: incompatible API changes
- MINOR: backwards-compatible functionality
- PATCH: backwards-compatible bug fixes

### Release Checklist

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Run full test suite
4. Update documentation
5. Create release on GitHub
6. Deploy to PyPI (maintainers only)

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: research@nanonas.ai for sensitive issues

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be:
- Listed in the CONTRIBUTORS.md file
- Mentioned in release notes for significant contributions
- Invited to join the maintainer team for outstanding contributions

Thank you for contributing to nanoNAS! 🎉

## 🚀 Getting Started

### Prerequisites

- Python 3.8+ 
- PyTorch 1.12+
- Git
- Basic knowledge of neural architecture search concepts

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/nanonas.git
   cd nanonas
   ```

2. **Set up Development Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install in development mode
   make install-dev
   # or manually:
   pip install -e ".[dev]"
   ```

3. **Verify Installation**
   ```bash
   make test
   python -m nanonas.api search --experiment dev_test
   ```

### Repository Structure

```
nanonas/
├── nanonas/           # Main package
│   ├── core/         # Core framework components
│   ├── search/       # Search strategies
│   ├── models/       # Neural network components
│   ├── benchmarks/   # Evaluation and metrics
│   ├── visualization/ # Plotting and analysis
│   └── configs/      # Configuration files
├── tests/            # Test suite
├── notebooks/        # Jupyter notebooks
├── docs/            # Documentation
└── results/         # Experiment results
```

## 🔄 Development Workflow

### Branch Strategy

- `main`: Stable, production-ready code
- `develop`: Integration branch for new features
- `feature/xxx`: Feature development branches
- `bugfix/xxx`: Bug fix branches
- `hotfix/xxx`: Critical fixes for main

### Workflow Steps

1. **Create Feature Branch**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   make pre-commit  # Runs formatting, linting, type-checking, tests
   ```

4. **Commit and Push**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**
   - Use our PR template
   - Link to relevant issues
   - Provide detailed description

## 🎨 Code Style Guidelines

### Python Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: 100 characters (not 79)
- **Imports**: Use absolute imports, group them logically
- **Docstrings**: Use Google-style docstrings
- **Type hints**: Use type hints for all public functions

### Code Formatting

We use automated tools for consistent formatting:

```bash
# Format code
make format

# Check formatting
black --check nanonas tests

# Sort imports
isort nanonas tests

# Lint code
make lint
```

### Example Code Style

```python
"""
Module docstring describing the purpose.

This module implements advanced search strategies for neural architecture search.
"""

from typing import Dict, List, Optional, Tuple
import torch
import numpy as np

from ..core.base import BaseSearchStrategy
from ..core.architecture import Architecture


class EvolutionarySearch(BaseSearchStrategy):
    """
    Evolutionary search strategy for neural architecture search.
    
    This class implements a genetic algorithm approach to finding optimal
    neural architectures through evolution of a population.
    
    Args:
        config: Experiment configuration object
        population_size: Size of the population for evolution
        
    Attributes:
        population: Current population of architectures
        best_fitness: Best fitness score found
    """
    
    def __init__(self, config: ExperimentConfig, population_size: int = 50):
        super().__init__(config)
        self.population_size = population_size
        self.population: List[Architecture] = []
        self.best_fitness: float = -float('inf')
    
    def search(self) -> Architecture:
        """
        Run evolutionary search to find the best architecture.
        
        Returns:
            Best architecture found during search
            
        Raises:
            ValueError: If search configuration is invalid
        """
        # Implementation here
        pass
```

## 🧪 Testing

### Test Structure

- `tests/unit/`: Unit tests for individual components
- `tests/integration/`: Integration tests for workflows
- `tests/benchmarks/`: Performance and accuracy benchmarks

### Writing Tests

```python
import pytest
import torch
from unittest.mock import Mock, patch

from nanonas.search.evolutionary import EvolutionarySearch
from nanonas.core.architecture import Architecture


class TestEvolutionarySearch:
    """Test suite for evolutionary search strategy."""
    
    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing."""
        # Configuration setup
        pass
    
    def test_initialization(self, sample_config):
        """Test proper initialization of evolutionary search."""
        search = EvolutionarySearch(sample_config)
        assert search.population_size > 0
        assert isinstance(search.population, list)
    
    @patch('nanonas.benchmarks.evaluator.ModelEvaluator')
    def test_search_execution(self, mock_evaluator, sample_config):
        """Test search execution with mocked evaluator."""
        # Test implementation
        pass
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test categories
make test-unit
make test-integration

# Run with coverage
make test-coverage

# Run specific test file
pytest tests/unit/test_search.py -v
```

## 📚 Documentation

### Docstring Style

Use Google-style docstrings:

```python
def mutate(self, architecture: Architecture, mutation_rate: float = 0.1) -> Architecture:
    """
    Apply mutation to an architecture.
    
    This method randomly modifies operations in the architecture based on
    the specified mutation rate.
    
    Args:
        architecture: Architecture to mutate
        mutation_rate: Probability of mutating each operation
        
    Returns:
        New mutated architecture
        
    Raises:
        ValueError: If mutation_rate is not in [0, 1]
        
    Example:
        >>> mutated = search.mutate(original_arch, mutation_rate=0.2)
        >>> assert mutated != original_arch
    """
```

### Adding New Documentation

1. **API Documentation**: Add docstrings to all public methods
2. **Configuration**: Document new configuration options
3. **Examples**: Add usage examples in docstrings and notebooks
4. **README**: Update README for significant changes

## 🔀 Submitting Changes

### Pull Request Process

1. **Follow the PR Template**: Fill out all relevant sections
2. **Link Issues**: Reference related issues using "Fixes #123"
3. **Describe Changes**: Provide clear description of what changed
4. **Include Tests**: Add tests for new functionality
5. **Update Documentation**: Update docs for user-facing changes

### Review Process

1. **Automated Checks**: CI must pass (tests, linting, etc.)
2. **Code Review**: At least one maintainer review required
3. **Testing**: Reviewers may test changes locally
4. **Documentation**: Check that docs are updated appropriately

### Merge Criteria

- ✅ All CI checks pass
- ✅ Code review approved
- ✅ Documentation updated
- ✅ Tests included and passing
- ✅ No conflicts with target branch

## 🔬 Research Contributions

### Implementing New Search Strategies

1. **Inherit from BaseSearchStrategy**
   ```python
   from nanonas.core.base import BaseSearchStrategy
   
   class YourSearchStrategy(BaseSearchStrategy):
       def search(self) -> Architecture:
           # Your implementation
           pass
   ```

2. **Add Configuration Support**
   ```yaml
   search:
     strategy: "your_strategy"
     your_param: value
   ```

3. **Write Comprehensive Tests**
4. **Add Benchmarking**
5. **Document Algorithm and Parameters**

### Adding New Operations

1. **Inherit from BaseOperation**
   ```python
   from nanonas.core.base import BaseOperation
   
   class YourOperation(BaseOperation):
       def forward(self, x: torch.Tensor) -> torch.Tensor:
           # Your implementation
           pass
   ```

2. **Register in Operation Registry**
3. **Add to Search Space Definitions**
4. **Include in Tests and Benchmarks**

### Research Guidelines

- **Reproducibility**: Ensure experiments are reproducible
- **Baselines**: Compare against existing methods
- **Documentation**: Provide clear algorithmic description
- **References**: Cite relevant papers and inspiration
- **Statistical Significance**: Use proper statistical analysis

## 🏷️ Commit Message Guidelines

Use conventional commits format:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(search): add DARTS search strategy implementation

fix(visualization): resolve NetworkX compatibility issue

docs(readme): update installation instructions for Windows

test(benchmarks): add GPU memory usage tests
```

## 🌟 Recognition

Contributors will be recognized in:

- `CONTRIBUTORS.md` file
- Release notes for significant contributions
- Research paper acknowledgments (for research contributions)
- GitHub contributors page

## 🆘 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Discord/Slack**: [Link to community chat]
- **Email**: For private inquiries

## 📋 Checklist for New Contributors

- [ ] Read this contributing guide
- [ ] Set up development environment
- [ ] Run tests successfully
- [ ] Read code style guidelines
- [ ] Understand the project structure
- [ ] Join community channels
- [ ] Look for "good first issue" labels

## 🙏 Thank You

Thank you for contributing to nanoNAS! Your contributions help advance the field of neural architecture search and make the framework better for everyone.

---

*This document is living and will be updated as the project evolves. Please suggest improvements!* 