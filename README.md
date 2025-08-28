# PRDNET: Pseudo-particle Ray Diffraction Network

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-coming_soon-b31b1b.svg)](https://arxiv.org/)

**PRDNET** is a state-of-the-art physics-informed neural network that design for crystal property prediction by combining:
- **Graph Neural Networks (GNNs)** for crystal structure representation
- **Pseudo-particle Ray Diffraction** physics integration
- **Advanced Transformer Attention** mechanisms
  

## 🚀 Key Features

- **Physics-Informed Architecture**: Integrates Pseudo-particle Ray diffraction physics directly into the neural network
- **Advanced GNN Design**: Custom transformer-based graph convolutions with multi-head attention
- **High Performance**: Optimized for distributed training with automatic mixed precision
- **Flexible Data Handling**: Supports ASE databases, JARVIS datasets, and custom crystal structures
- **Production Ready**: Comprehensive caching, checkpointing, and monitoring with WandB integration

## 📊 Performance

PRDNET achieves state-of-the-art performance on crystal property prediction benchmarks:
- **Formation Energy**: MAE < 0.03 eV/atom on Materials Project dataset
- **Band Gap**: Accurate prediction across diverse crystal systems
- **Mechanical Properties**: Superior performance on kinds of moduli prediction

## 🔬 Scientific Background

PRDNET introduces a novel paradigm by incorporating diffraction physics—the fundamental interaction between Pseudo-particle-rays and crystal lattices—directly into the learning process. This physics-informed approach enables the model to understand the relationship between atomic arrangements and their measurable diffraction signatures, leading to more accurate and interpretable predictions.


## 🛠️ Installation

### Quick Install

```bash
# Clone the repository
git clone https://github.com/your-username/PRDNET.git
cd PRDNET

# Install with pip
pip install -e .
```

### Development Install

```bash
# Clone and install in development mode
git clone https://github.com/your-username/PRDNET.git
cd PRDNET

# Create conda environment
conda create -n prdnet python=3.9 -y
conda activate prdnet

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install PRDNET with development dependencies
pip install -e ".[dev]"
```

### Manual Installation

If you prefer to install dependencies manually:

#### 1. Environment Setup
```bash
conda create -n prdnet python=3.9 -y
conda activate prdnet
```

#### 2. Core Dependencies
```bash
# PyTorch and PyTorch Geometric
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torch-geometric pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv

# Scientific computing
pip install numpy scipy pandas scikit-learn matplotlib
pip install ase jarvis-tools h5py tqdm

# Machine learning
pip install wandb ignite pydantic pydantic-settings
```

#### 3. Optional Dependencies
```bash
# Visualization
pip install seaborn plotly

# Distributed training
pip install accelerate

# Development tools
pip install pytest black flake8 mypy pre-commit
```

### 4. Verify Installation

```bash
# Quick verification
python -c "
import torch
import prdnet
from prdnet.model import Prdnet, PrdnetConfig
print('✅ PRDNET installed successfully!')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
"
```

## 🚀 Quick Start

### Basic Usage

```python
import torch
from prdnet.model import Prdnet, PrdnetConfig
from prdnet.data import get_train_val_loaders

# Create model configuration
config = PrdnetConfig(
    name="prdnet",
    node_features=256,
    conv_layers=6,
    use_diffraction=True,
    diffraction_max_hkl=5
)

# Initialize model
model = Prdnet(config)

# Load your crystal structure data
train_loader, val_loader, test_loader, prepare_batch, mean, std = get_train_val_loaders(
    dataset="dft_3d",  # or your custom dataset
    target="formation_energy_peratom",
    batch_size=32,
    atom_features="cgcnn"
)

# Train the model
from prdnet.train import train_dgl
from prdnet.config import TrainingConfig

training_config = TrainingConfig(
    epochs=100,
    learning_rate=0.001,
    batch_size=32
)

results = train_dgl(
    config=training_config,
    model=model,
    train_val_test_loaders=[train_loader, val_loader, test_loader, prepare_batch]
)
```

### Using the Trainer Class

```python
from trainer import PrdnetTrainer, create_trainer_config

# Create configuration
config = create_trainer_config(
    train_db_path="path/to/train.db",
    val_db_path="path/to/val.db",
    target_property="formation_energy",
    epochs=200,
    batch_size=64,
    use_cache=True
)

# Initialize and run trainer
trainer = PrdnetTrainer(config)
results = trainer.train(
    train_db_path=config.train_db_path,
    val_db_path=config.val_db_path
)
```

## 🎯 Training

### Command Line Training

```bash
# Basic training
python trainer.py

# Custom parameters
python trainer.py \
    --epochs 200 \
    --batch_size 64 \
    --learning_rate 1e-4

# Using console script (after installation)
prdnet-train --epochs 100 --batch_size 32
```

### Distributed Training

```bash
# Multi-GPU training with torchrun
torchrun --nproc_per_node=4 trainer.py \
    --epochs 500 \
    --batch_size 96 \
    --learning_rate 1e-4

# With custom data paths
torchrun --nproc_per_node=4 trainer.py \
    --train_db_path data/train.db \
    --val_db_path data/val.db \
    --test_db_path data/test.db
```

### Configuration Options

```bash
# Disable WandB logging
python trainer.py --no_wandb

# Custom cache directory
python trainer.py --cache_dir ./custom_cache

# Force cache rebuild
python trainer.py --force_cache_rebuild

# Custom model configuration
python trainer.py \
    --conv_layers 8 \
    --node_features 512 \
    --use_diffraction True \
    --diffraction_max_hkl 6
```

## Monitoring Training

### WandB Integration

1. **Setup WandB account** (optional but recommended):
```bash
# Install and login to WandB
pip install wandb
wandb login
```

2. **Training with WandB logging**:
```bash
python prd_trainer.py --epochs 100
# Training metrics will be automatically logged to WandB
```

### Local Monitoring

Training logs and checkpoints are saved to:
```
./prdnet_training_output/
├── training.log          # Detailed training logs
├── checkpoints/          # Model checkpoints
├── best_model.pth       # Best model weights
└── training_curves.png  # Training visualization
```

## 📚 API Documentation

### Core Classes

#### `Prdnet` - Main Model Class

```python
from prdnet.model import Prdnet, PrdnetConfig

config = PrdnetConfig(
    name="prdnet",
    conv_layers=6,              # Number of graph convolution layers
    node_features=256,          # Node feature dimensions
    edge_features=256,          # Edge feature dimensions
    fc_layers=2,                # Number of fully connected layers
    fc_features=512,            # FC layer dimensions
    output_features=1,          # Output dimensions
    use_diffraction=True,       # Enable diffraction integration
    diffraction_max_hkl=5,      # Maximum HKL index for diffraction
    dropout=0.1,                # Dropout rate
    use_residual_connections=True,  # Enable residual connections
    activation="silu"           # Activation function
)

model = Prdnet(config)
```

#### `PrdnetTrainer` - Training Interface

```python
from trainer import PrdnetTrainer, create_trainer_config

# Create training configuration
config = create_trainer_config(
    train_db_path="train.db",
    target_property="formation_energy",
    epochs=200,
    batch_size=64,
    learning_rate=0.001,
    use_cache=True,
    cache_dir="./cache"
)

# Initialize trainer
trainer = PrdnetTrainer(config)

# Start training
results = trainer.train(
    train_db_path=config.train_db_path,
    val_db_path=config.val_db_path
)
```

#### `TrainingConfig` - Configuration Management

```python
from prdnet.config import TrainingConfig

config = TrainingConfig(
    dataset="dft_3d",
    target="formation_energy_peratom",
    atom_features="cgcnn",
    neighbor_strategy="k-nearest",
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    model=PrdnetConfig(name="prdnet")
)
```

### Data Loading

#### Working with ASE Databases

```python
from ase.db import connect
from trainer import PrdnetTrainer

# Your ASE database should contain crystal structures with target properties
db = connect("my_crystals.db")

# Example: Add a crystal structure to database
from ase import Atoms
atoms = Atoms(...)  # Your crystal structure
db.write(atoms, formation_energy=-2.5)  # Add with target property

# Train with ASE database
config = create_trainer_config(
    train_db_path="my_crystals.db",
    target_property="formation_energy"
)
trainer = PrdnetTrainer(config)
results = trainer.train()
```

#### Custom Datasets

```python
from prdnet.data import get_train_val_loaders

# Load data with custom parameters
train_loader, val_loader, test_loader, prepare_batch, mean, std = get_train_val_loaders(
    dataset="custom",           # Use custom dataset
    dataset_array=my_data,      # Your data array
    target="my_property",       # Target property name
    atom_features="cgcnn",      # Feature type
    neighbor_strategy="k-nearest",
    batch_size=32,
    cutoff=8.0,                 # Cutoff radius for neighbors
    max_neighbors=12,           # Maximum number of neighbors
    use_lattice=True,           # Include lattice information
    line_graph=True             # Use line graph representation
)
```




## 🔧 Advanced Features

### Physics-Informed Diffraction Integration

PRDNET's key innovation is the integration of X-ray diffraction physics:

```python
from prdnet.diffraction import DiffractionIntegration

# Enable diffraction integration in model
config = PrdnetConfig(
    use_diffraction=True,
    diffraction_max_hkl=5,      # Maximum Miller indices
    diffraction_num_hkl=300     # Number of HKL reflections
)

# The model automatically computes structure factors
# and integrates them with graph features
model = Prdnet(config)
```

### Distributed Training with Caching

```python
# Automatic data caching for faster training
config = create_trainer_config(
    use_cache=True,
    cache_dir="./prdnet_cache",
    cache_format="pt",          # or "hdf5"
    force_cache_rebuild=False
)

# Distributed training automatically handles caching
# Only rank 0 generates cache, others wait and load
```

### Custom Property Prediction

```python
# Extend for new properties
config = TrainingConfig(
    target="my_custom_property",
    classification=False,        # Set True for classification tasks
    output_features=1           # Adjust for multi-output
)
```

## 📊 Datasets

PRDNET supports various crystal structure datasets:

- **Materials Project**: Formation energies, band gaps, elastic properties
- **JARVIS-DFT**: Comprehensive DFT calculations
- **Custom ASE Databases**: Your own crystal structures
- **OQMD**: Open Quantum Materials Database

### Data Format

Your data should be in ASE database format with crystal structures and target properties:

```python
from ase.db import connect
from ase import Atoms

db = connect("my_data.db")
atoms = Atoms(...)  # Your crystal structure
db.write(atoms, formation_energy=-2.5, band_gap=1.2)
```

## 🚀 Performance Tips

1. **Use GPU acceleration**: Ensure CUDA is available for training
2. **Enable caching**: Use `use_cache=True` for repeated training
3. **Distributed training**: Use multiple GPUs with `torchrun`
4. **Batch size optimization**: Adjust based on GPU memory
5. **Mixed precision**: Automatically enabled for better performance

## 🐛 Troubleshooting

### Common Issues

**CUDA out of memory**:
```bash
# Reduce batch size
python trainer.py --batch_size 16

# Or reduce model size
python trainer.py --node_features 128 --conv_layers 4
```

**Import errors**:
```bash
# Reinstall with dependencies
pip install -e ".[dev]"
```

**Slow training**:
```bash
# Enable caching
python trainer.py --use_cache True

# Use distributed training
torchrun --nproc_per_node=4 trainer.py
```

## 📖 Citation

If you use PRDNET in your research, please cite:

```bibtex
@article{cao2024prdnet,
  title={PRDNET: Pseudo-particle Ray Diffraction Network for Crystal Property Prediction},
  author={Cao, Bin and others},
  journal={arXiv preprint},
  year={2024},
  note={Paper in preparation}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Contribution Steps

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run tests: `pytest`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact & Support

- **Author**: Bin Cao
- **Email**: bcao686@connect.hkust-gz.edu.cn
- **Affiliations**:
  - Hong Kong University of Science and Technology (Guangzhou)
  - City University of Hong Kong

### Getting Help

- 📖 Check the [documentation](README.md)
- 🐛 Report bugs via [GitHub Issues](https://github.com/your-username/PRDNET/issues)
- 💬 Ask questions in [Discussions](https://github.com/your-username/PRDNET/discussions)
- 📧 Email for collaboration inquiries

## 🙏 Acknowledgments

- PyTorch Geometric team for excellent graph neural network tools
- JARVIS team for materials science datasets and tools
- Materials Project for crystal structure databases
- ASE developers for atomic simulation environment

---

**⭐ If you find PRDNET useful, please star the repository!**
