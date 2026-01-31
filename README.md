# PRDNet: Pseudo-particle Ray Diffraction Network


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ICLR 2026](https://img.shields.io/badge/ICLR-OpenReview-4b44ce.svg)](https://openreview.net/forum?id=OfmurJrzlT)

<img width="638"  alt="Screenshot 2026-01-31 at 12 01 13" src="https://github.com/user-attachments/assets/47396371-c413-418d-a1db-e90d6ed11c61" />

**PRDNet** is a physics-informed graph neural network for crystal property prediction that combines:
- **Graph Neural Networks** for crystal structure representation
- **Pseudo-particle Ray Diffraction** physics integration
- **Multi-head Attention** mechanisms

## Quick Start



### 1. Installation

```bash
# Clone repository
git clone https://github.com/Bin-Cao/PRDNet.git
cd PRDNet

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import prdnet; print('PRDNet installed successfully!')"
```

### 2. Prepare Data

Your data should be in ASE database format. Here's how to create one:

```python
from ase.db import connect
from ase import Atoms
from ase.build import bulk

# Create database
db = connect("my_data.db")

# Example: Add some crystal structures
# Silicon crystal
si = bulk('Si', 'diamond', a=5.43)
db.write(si, formation_energy=-5.42, band_gap=1.12)

# NaCl crystal
nacl = bulk('NaCl', 'rocksalt', a=5.64)
db.write(nacl, formation_energy=-8.23, band_gap=8.5)

print(f"Database created with {len(db)} structures")
```

**Using existing datasets:**
- Download from [Materials Project](https://huggingface.co/datasets/caobin/CPPbenchmark)
- Or use your own ASE-compatible database files

### 3. Train Model

**Option A: Edit trainer.py (Recommended)**

1. Open `trainer.py` and modify the database paths in the `main()` function:
```python
config = create_trainer_config(
    train_db_path="your_train.db",      # ← Change this
    val_db_path="your_val.db",          # ← Change this
    test_db_path="your_test.db",        # ← Change this
    target_property="formation_energy", # ← Change if needed
    # ... other settings
)
```

2. Run training:
```bash
python trainer.py
```

**Option B: Python API**

```python
from trainer import PrdnetTrainer, create_trainer_config

config = create_trainer_config(
    train_db_path="my_data.db",
    target_property="formation_energy",
    epochs=100,
    batch_size=32
)

trainer = PrdnetTrainer(config)
results = trainer.train()
```

## Configuration

### Key Parameters

Edit these parameters in `trainer.py`:

```python
config = create_trainer_config(
    # Data paths
    train_db_path="path/to/train.db",
    val_db_path="path/to/val.db",      # Optional
    target_property="formation_energy", # Property to predict

    # Training settings
    epochs=500,
    batch_size=32,
    learning_rate=0.0005,

    # Model architecture
    model_config={
        "conv_layers": 6,           # Graph convolution layers
        "node_features": 256,       # Node embedding dimension
        "use_diffraction": True,    # Enable physics integration
        "diffraction_max_hkl": 5,   # Miller index range
        "node_layer_head": 8,       # Attention heads
    }
)
```

## Advanced Usage

### Distributed Training

```bash
# Multi-GPU training
torchrun --nproc_per_node=4 trainer.py

# Custom parameters
torchrun --nproc_per_node=4 trainer.py \
    --epochs 500 \
    --batch_size 96
```

### Data Caching

PRDNet automatically caches preprocessed data for faster training:

```python
config = create_trainer_config(
    cache_dir="./prdnet_cache",  # Cache directory
    use_cache=True,              # Enable caching
    force_cache_rebuild=False    # Rebuild cache if needed
)
```

### Monitoring with WandB

```bash
# Install WandB (optional)
pip install wandb
wandb login

# Training with logging
python trainer.py  # Metrics automatically logged
```

## Supported Properties

PRDNET can predict various materials properties:

- Formation energy (`formation_energy`)
- Band gap (`band_gap`)
- Bulk modulus (`bulk_modulus`)
- Shear modulus (`shear_modulus`)
- Custom properties (any numeric property in your database)

## Datasets

- **[Materials Project](https://huggingface.co/datasets/caobin/CPPbenchmark)** - Formation energies, band gaps, elastic properties
- **JARVIS-DFT** - Comprehensive DFT calculations
- **Custom databases** - ASE database format

## Troubleshooting

### Common Issues

**CUDA out of memory**:
```python
# Reduce batch size and model size
batch_size=16
model_config={"node_features": 128, "conv_layers": 4}
```

**Database format errors**:
```bash
# Check your database
python -c "from ase.db import connect; print(len(connect('your_data.db')))"
```

**Missing dependencies**:
```bash
# Install PyTorch first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install other dependencies
pip install -r requirements.txt
```


## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@article{cao2025beyond,
  title={Beyond Structure: Invariant Crystal Property Prediction with Pseudo-Particle Ray Diffraction},
  author={Cao, Bin and Liu, Yang and Zhang, Longhan and Wu, Yifan and Li, Zhixun and Luo, Yuyu and Cheng, Hong and Ren, Yang and Zhang, Tong-Yi},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2026}
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Contact

- **Author**: Bin Cao
- **Email**: bcao686@connect.hkust-gz.edu.cn
- **Affiliations**: HKUST(GZ), CityU HK




