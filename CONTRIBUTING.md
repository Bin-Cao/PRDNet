# Contributing to PRDNet

Thank you for your interest in contributing to PRDNet! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- Git

### Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/your-username/PRDNet.git
   cd PRDNet
   ```

2. **Create a development environment:**
   ```bash
   conda create -n prdnet-dev python=3.9 -y
   conda activate prdnet-dev
   ```

3. **Install dependencies:**
   ```bash
   # Install PyTorch (adjust CUDA version as needed)
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   
   # Install the package in development mode
   pip install -e .
   
   # Install development dependencies
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

## Contributing Process

1. **Create an issue** describing the bug fix or feature you want to work on
2. **Fork the repository** and create a new branch from `main`
3. **Make your changes** following our coding standards
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Submit a pull request**

### Branch Naming Convention

- `feature/description` - for new features
- `bugfix/description` - for bug fixes
- `docs/description` - for documentation updates
- `refactor/description` - for code refactoring

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [flake8](https://flake8.pycqa.org/) for linting
- Maximum line length: 88 characters (Black default)

### Code Quality

- Write clear, self-documenting code
- Add type hints for function parameters and return values
- Include docstrings for all public functions and classes
- Follow Google-style docstring format

### Example Function Documentation

```python
def train_model(config: TrainingConfig, data_loader: DataLoader) -> Dict[str, float]:
    """Train a PRDNet model with the given configuration.
    
    Args:
        config: Training configuration object containing hyperparameters
        data_loader: PyTorch DataLoader with training data
        
    Returns:
        Dictionary containing training metrics (loss, MAE, etc.)
        
    Raises:
        ValueError: If configuration is invalid
        RuntimeError: If CUDA is required but not available
    """
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=prdnet

# Run specific test file
pytest tests/test_model.py
```

### Writing Tests

- Write unit tests for all new functions
- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies when appropriate

### Test Structure

```python
def test_model_forward_pass():
    """Test that model forward pass produces expected output shape."""
    config = PrdnetConfig(node_features=64, output_features=1)
    model = Prdnet(config)
    
    # Create dummy input
    batch_size = 4
    num_nodes = 10
    x = torch.randn(batch_size * num_nodes, config.node_features)
    
    # Test forward pass
    output = model(x)
    assert output.shape == (batch_size, config.output_features)
```

## Documentation

### Code Documentation

- Document all public APIs
- Include usage examples in docstrings
- Update README.md for significant changes

### Adding Examples

When adding new features, include:
- Usage examples in docstrings
- Example scripts in `examples/` directory
- Updates to relevant documentation

## Submitting Changes

### Pull Request Process

1. **Update your branch:**
   ```bash
   git checkout main
   git pull upstream main
   git checkout your-feature-branch
   git rebase main
   ```

2. **Run quality checks:**
   ```bash
   black prdnet/
   flake8 prdnet/
   pytest
   ```

3. **Create pull request:**
   - Use a clear, descriptive title
   - Reference related issues
   - Describe changes and motivation
   - Include test results

### Pull Request Template

```markdown
## Description
Brief description of changes

## Related Issues
Fixes #123

## Changes Made
- Added new feature X
- Fixed bug in Y
- Updated documentation for Z

## Testing
- [ ] All existing tests pass
- [ ] Added tests for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

## Performance Considerations

- Profile code for performance bottlenecks
- Use appropriate data types (float32 vs float64)
- Consider memory usage for large datasets
- Optimize GPU utilization when possible

## Questions?

If you have questions about contributing, please:
1. Check existing issues and discussions
2. Create a new issue with the "question" label
3. Contact the maintainers

Thank you for contributing to PRDNet!
