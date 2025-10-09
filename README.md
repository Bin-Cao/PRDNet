**⭐ If you find PRDNet useful, please star the repository!**

# PRDNet: Pseudo-particle Ray Diffraction Network

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-PDF-b31b1b.svg)](https://arxiv.org/pdf/2509.21778)


**PRDNET** is a SOTA invariant neural network that design for crystal property prediction by combining:
- **Graph Neural Networks (GNNs)** for crystal structure representation
- **Pseudo-particle Ray Diffraction** physics integration
- **Advanced Transformer Attention** mechanisms
  

## Key Features

- **Physics-Informed Architecture**: Integrates Pseudo-particle Ray diffraction physics directly into the neural network
- **Advanced GNN Design**: Custom transformer-based graph convolutions with multi-head attention
- **High Performance**: Optimized for distributed training with automatic mixed precision
- **Production Ready**: Comprehensive caching, checkpointing, and monitoring with WandB integration

## Performance

PRDNet achieves state-of-the-art performance on crystal property prediction benchmarks:
- **Formation Energy**: MAE < 0.03 eV/atom on Materials Project dataset
- **Band Gap**: MAE < 0.16 eV on Materials Project dataset 
- **Mechanical Properties**: Superior performance on kinds of moduli prediction

## Background

PRDNet, a novel architecture that integrates graph embeddings with a learned pseudo-particle diffraction module. It generates synthetic diffraction patterns that are invariant to crystallographic symmetries.


## Installation

### Quick Install

```bash
# Clone the repository
git clone https://github.com/your-username/PRDNET.git
cd PRDNet

# Install with pip
pip install -e .
```

### Development Install

```bash
# Clone and install in development mode
git clone https://github.com/your-username/PRDNET.git
cd PRDNet

# Create conda environment
conda create -n prdnet python=3.9 -y
conda activate prdnet

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install PRDNET with development dependencies
pip install -e ".[dev]"
```

### Manual Installation

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
print('✅ PRDNet installed successfully!')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
"
```

## Quick Start

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

## Training

### Command Line Training

```bash
# Basic training
python trainer.py

# Custom parameters
python trainer.py \
    --epochs 200 \
    --batch_size 64 \
    --learning_rate 1e-4

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
python trainer.py --epochs 300
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

### Data Loading

#### Working with ASE Databases

```python
from ase.db import connect
from trainer import PrdnetTrainer

# Your ASE database should contain crystal structures with target properties

# Example: Add a crystal structure to database
"""
from ase import Atoms
atoms = Atoms(...)  # Your crystal structure
db.write(atoms, formation_energy=-2.5)  # Add with target property
"""

# Train with ASE database
python trainer.py 
        --epochs 300 \
        --train_db_path MP_100_bgfe_train.db \ 
        --val_db_path MP_100_bgfe_val.db \
        --test_db_path MP_100_bgfe_test.db \
        --target_property formation_energy \
        --cache_dir prdnet_cache
```


## Datasets

PRDNET supports various crystal structure datasets:

- **[Materials Project](https://huggingface.co/datasets/caobin/CPPbenchmark)**
- **JARVIS-DFT**
- **MatBench**


### Data Format

Your data should be in ASE database format with crystal structures and target properties:

```python
from ase.db import connect
from ase import Atoms

db = connect("my_data.db")
atoms = Atoms(...)  # Your crystal structure
db.write(atoms, formation_energy=-2.5, band_gap=1.2)
```

## Troubleshooting

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

## Citation

If you use PRDNet in your research, please cite:

```bibtex
@misc{cao2025structureinvariantcrystalproperty,
      title={Beyond Structure: Invariant Crystal Property Prediction with Pseudo-Particle Ray Diffraction}, 
      author={Bin Cao and Yang Liu and Longhan Zhang and Yifan Wu and Zhixun Li and Yuyu Luo and Hong Cheng and Yang Ren and Tong-Yi Zhang},
      year={2025},
      eprint={2509.21778},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2509.21778}, 
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact & Support

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

## Acknowledgments

- PyTorch Geometric team for excellent graph neural network tools
- JARVIS team for materials science datasets and tools
- Materials Project for crystal structure databases
- ASE developers for atomic simulation environment




