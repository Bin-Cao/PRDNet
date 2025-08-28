# PRDNET API Documentation

This document provides detailed API documentation for PRDNET components.

## Core Classes

### `prdnet.model.Prdnet`

Main PRDNET model class implementing the physics-informed neural network.

```python
class Prdnet(nn.Module):
    def __init__(self, config: PrdnetConfig = PrdnetConfig(name="prdnet"))
```

**Parameters:**
- `config` (PrdnetConfig): Model configuration object

**Key Methods:**
- `forward(data)`: Forward pass through the network
- `get_attention_weights()`: Extract attention weights for visualization

### `prdnet.model.PrdnetConfig`

Configuration class for PRDNET model hyperparameters.

**Key Parameters:**
- `conv_layers` (int): Number of graph convolution layers (default: 6)
- `node_features` (int): Node feature dimensions (default: 256)
- `edge_features` (int): Edge feature dimensions (default: 256)
- `use_diffraction` (bool): Enable diffraction integration (default: True)
- `diffraction_max_hkl` (int): Maximum HKL index (default: 5)
- `dropout` (float): Dropout rate (default: 0.1)

### `trainer.PrdnetTrainer`

High-level training interface with caching and distributed training support.

```python
class PrdnetTrainer:
    def __init__(self, config: Union[Dict, PrdnetTrainingConfig, TrainingConfig])
    def train(self, train_db_path: str, val_db_path: str = None, test_db_path: str = None)
```

**Key Features:**
- Automatic data caching for improved performance
- Distributed training support
- ASE database integration
- Comprehensive logging and monitoring

### `prdnet.config.TrainingConfig`

Training configuration with validation and defaults.

**Key Parameters:**
- `dataset` (str): Dataset name or "custom"
- `target` (str): Target property name
- `epochs` (int): Number of training epochs
- `batch_size` (int): Training batch size
- `learning_rate` (float): Learning rate
- `model` (PrdnetConfig): Model configuration

## Data Loading

### `prdnet.data.get_train_val_loaders`

Create PyTorch data loaders for training.

```python
def get_train_val_loaders(
    dataset: str,
    target: str,
    batch_size: int = 32,
    atom_features: str = "cgcnn",
    neighbor_strategy: str = "k-nearest",
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Callable, float, float]
```

**Returns:**
- Training DataLoader
- Validation DataLoader  
- Test DataLoader
- Batch preparation function
- Training mean (for normalization)
- Training std (for normalization)

### `prdnet.graphs.PygStructureDataset`

PyTorch Geometric dataset for crystal structures.

```python
class PygStructureDataset(torch.utils.data.Dataset):
    def __init__(self, df, graphs, target, **kwargs)
```

## Physics Integration

### `prdnet.diffraction.DiffractionIntegration`

Diffraction physics integration module.

```python
class DiffractionIntegration(nn.Module):
    def __init__(self, node_features: int, graph_features: int, 
                 output_features: int, max_hkl: int = 5, num_hkl: int = 300)
```

**Key Components:**
- `StructureFactorCalculator`: Computes X-ray structure factors
- `HKLSelector`: Selects relevant Miller indices
- `DiffractionFusion`: Fuses diffraction with graph features

## Utilities

### `trainer.create_trainer_config`

Convenience function to create training configurations.

```python
def create_trainer_config(
    train_db_path: str,
    target_property: str = "formation_energy",
    **kwargs
) -> PrdnetTrainingConfig
```

### `prdnet.utils.plot_learning_curve`

Plot training curves from saved history files.

```python
def plot_learning_curve(results_dir: Union[str, Path], key: str = "mae")
```

## Configuration Examples

### Basic Model Configuration

```python
from prdnet.model import PrdnetConfig

config = PrdnetConfig(
    name="prdnet",
    conv_layers=6,
    node_features=256,
    use_diffraction=True
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
    model=PrdnetConfig(name="prdnet")
)
```

### Trainer Configuration

```python
from trainer import create_trainer_config

config = create_trainer_config(
    train_db_path="train.db",
    target_property="formation_energy",
    epochs=200,
    use_cache=True
)
```

## Error Handling

Common exceptions and their meanings:

- `ValueError`: Invalid configuration parameters
- `RuntimeError`: CUDA/training runtime errors
- `FileNotFoundError`: Missing data files
- `ImportError`: Missing dependencies

## Performance Tips

1. **Enable caching**: Use `use_cache=True` for repeated training
2. **GPU optimization**: Ensure CUDA is available and properly configured
3. **Batch size**: Adjust based on GPU memory (start with 32)
4. **Distributed training**: Use `torchrun` for multi-GPU setups
5. **Data workers**: Set `num_workers=4` for faster data loading

## Version Compatibility

- Python: 3.8+
- PyTorch: 2.0+
- PyTorch Geometric: 2.3+
- CUDA: 11.8+ (optional but recommended)

For detailed examples and tutorials, see the main README.md file.
