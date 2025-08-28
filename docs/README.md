# PRDNET Documentation

Welcome to the PRDNET documentation! This directory contains comprehensive guides and API documentation for the Pseudo-particle Ray Diffraction Network.

## 📚 Documentation Structure

### Core Documentation
- **[API.md](API.md)** - Complete API reference for all classes and functions
- **[../README.md](../README.md)** - Main project documentation with installation and usage
- **[../CONTRIBUTING.md](../CONTRIBUTING.md)** - Guidelines for contributing to the project
- **[../CHANGELOG.md](../CHANGELOG.md)** - Version history and release notes

## 🚀 Quick Navigation

### Getting Started
1. **Installation**: See [README.md](../README.md#installation) for setup instructions
2. **Quick Start**: Follow the [Quick Start guide](../README.md#quick-start) for basic usage
3. **API Reference**: Check [API.md](API.md) for detailed function documentation

### Key Components

#### Model Architecture
- **`Prdnet`**: Main neural network model
- **`PrdnetConfig`**: Model configuration and hyperparameters
- **`DiffractionIntegration`**: Physics-informed diffraction module

#### Training System
- **`PrdnetTrainer`**: High-level training interface
- **`TrainingConfig`**: Training configuration management
- **`train_dgl`**: Core training function

#### Data Processing
- **`PygStructureDataset`**: Crystal structure dataset
- **`get_train_val_loaders`**: Data loading utilities
- **`CachedPrdnetDataset`**: Cached dataset for performance

### Advanced Features

#### Physics Integration
PRDNET's key innovation is integrating X-ray diffraction physics:
- Structure factor calculations
- Miller index (HKL) selection
- Physics-graph feature fusion

#### Performance Optimization
- Distributed training across multiple GPUs
- Intelligent data caching system
- Memory-efficient graph processing
- Automatic mixed precision training

#### Production Features
- Comprehensive experiment tracking with WandB
- Model checkpointing and recovery
- Extensive configuration validation
- Professional logging and monitoring

## 🔧 Configuration Guide

### Model Configuration
```python
from prdnet.model import PrdnetConfig

config = PrdnetConfig(
    conv_layers=6,           # Graph convolution layers
    node_features=256,       # Node feature dimensions
    use_diffraction=True,    # Enable physics integration
    diffraction_max_hkl=5    # Maximum Miller indices
)
```

### Training Configuration
```python
from prdnet.config import TrainingConfig

config = TrainingConfig(
    dataset="dft_3d",
    target="formation_energy_peratom",
    epochs=100,
    batch_size=32,
    learning_rate=0.001
)
```

## 🎯 Use Cases

### Crystal Property Prediction
- Formation energy prediction
- Band gap estimation
- Mechanical properties (elastic moduli)
- Custom property prediction

### Research Applications
- High-throughput materials screening
- Materials discovery and design
- Structure-property relationship studies
- Physics-informed machine learning research

## 🛠️ Development

### Code Organization
```
prdnet/
├── model.py          # Core model implementation
├── config.py         # Configuration management
├── train.py          # Training utilities
├── data.py           # Data loading and processing
├── graphs.py         # Graph neural network components
├── diffraction.py    # Physics integration
├── transformer.py    # Attention mechanisms
└── utils.py          # Shared utilities
```

### Testing and Quality
- Comprehensive unit tests
- Code formatting with Black
- Linting with flake8
- Type checking with mypy
- Pre-commit hooks for quality assurance

## 📊 Performance

### Benchmarks
- **Formation Energy**: MAE < 0.05 eV/atom on Materials Project
- **Band Gap**: Competitive performance across crystal systems
- **Training Speed**: Optimized for distributed GPU training
- **Memory Efficiency**: Intelligent caching reduces memory usage

### Scalability
- Supports datasets with millions of crystal structures
- Distributed training across multiple nodes
- Efficient graph processing for large crystal systems
- Automatic batch size optimization

## 🤝 Community

### Getting Help
- 📖 Check this documentation first
- 🐛 Report bugs via [GitHub Issues](https://github.com/your-username/PRDNET/issues)
- 💬 Ask questions in [Discussions](https://github.com/your-username/PRDNET/discussions)
- 📧 Contact maintainers for collaboration

### Contributing
We welcome contributions! See [CONTRIBUTING.md](../CONTRIBUTING.md) for:
- Development setup instructions
- Coding standards and guidelines
- Pull request process
- Testing requirements

## 📄 License

PRDNET is released under the MIT License. See [LICENSE](../LICENSE) for details.

## 🙏 Acknowledgments

This project builds upon excellent open-source tools:
- PyTorch and PyTorch Geometric for deep learning
- ASE for atomic structure manipulation
- JARVIS for materials science datasets
- Materials Project for crystal databases

---

**Need more help?** Check the [main README](../README.md) or [API documentation](API.md).
