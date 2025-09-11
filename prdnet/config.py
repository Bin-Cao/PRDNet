"""
Description: Configuration models and validation for Prdnet.
"""

import subprocess
from typing import Optional, Union
import os
from pydantic import model_validator
from typing import Literal
from prdnet.utils import BaseSettings
from prdnet.model import PrdnetConfig

try:
    VERSION = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    )
except Exception as exp:
    VERSION = "0.01"


FEATURESET_SIZE = {"basic": 11, "atomic_number": 1, "cfid": 438, "cgcnn": 92}


TARGET_ENUM = Literal[
    "formation_energy_peratom",
    "formation_energy",
    "band_gap",
    "bulk_modulus",
    "shear_modulus",
    "youngs_modulus",
    "poissons_ratio",
    "jdft2d",
    "mp_e_form",
    "log_gvrh",
    "dft_3d_formation_energy",
    "dft_3d_band_gap",
    "dft_3d_bulk_modulus",
    "dft_3d_shear_modulus"

]


class TrainingConfig(BaseSettings):
    """Training config defaults and validation."""

    version: str = VERSION

    # dataset configuration
    dataset: Literal[
        "dft_3d",
        "megnet",
    ] = "dft_3d"
    target: TARGET_ENUM = "formation_energy_peratom"
    atom_features: Literal["basic", "atomic_number", "cfid", "cgcnn"] = "cgcnn"
    neighbor_strategy: Literal["k-nearest", "voronoi", "pairwise-k-nearest"] = "k-nearest"
    id_tag: Literal["jid", "id", "_oqmd_entry_id"] = "jid"

    # logging configuration

    # training configuration
    random_seed: Optional[int] = 123
    classification_threshold: Optional[float] = None
    n_val: Optional[int] = None
    n_test: Optional[int] = None
    n_train: Optional[int] = None
    train_ratio: Optional[float] = 0.8
    val_ratio: Optional[float] = 0.1
    test_ratio: Optional[float] = 0.1
    target_multiplication_factor: Optional[float] = None
    epochs: int = 300
    batch_size: int = 64
    weight_decay: float = 0
    learning_rate: float = 1e-2
    filename: str = "sample"
    warmup_steps: int = 2000
    criterion: Literal["mse", "l1", "huber", "smooth_l1", "log_cosh", "poisson", "zig"] = "mse"
    optimizer: Literal["adamw", "adam", "sgd", "rmsprop"] = "adamw"
    scheduler: Literal["onecycle", "none", "step", "cosine", "cosine_warm", "exponential", "plateau"] = "onecycle"

    # Advanced optimization parameters
    grad_clip_norm: float = 1.0  # Gradient clipping norm
    use_ema: bool = False  # Exponential moving average
    ema_decay: float = 0.999  # EMA decay rate
    pin_memory: bool = False
    save_dataloader: bool = False
    write_checkpoint: bool = True
    write_predictions: bool = True
    store_outputs: bool = True
    progress: bool = True
    log_tensorboard: bool = False
    standard_scalar_and_pca: bool = False
    use_canonize: bool = True
    num_workers: int = 2
    cutoff: float = 8.0
    max_neighbors: int = 12
    keep_data_order: bool = False
    distributed: bool = False
    n_early_stopping: Optional[int] = None  # typically 50
    output_dir: str = os.path.abspath(".")  # typically 50
    matrix_input: bool = False
    pyg_input: bool = False
    use_lattice: bool = False
    use_angle: bool = False

    # model configuration
    model: PrdnetConfig = PrdnetConfig(name="prdnet")
    @model_validator(mode='before')
    @classmethod
    def set_input_size(cls, values):
        """Automatically configure node feature dimensionality."""
        if isinstance(values, dict):
            if "model" in values and "atom_features" in values:
                # Handle case where model is a dict
                if isinstance(values["model"], dict):
                    values["model"]["atom_input_features"] = FEATURESET_SIZE[
                        values["atom_features"]
                    ]
                else:
                    # Handle case where model is already a MatformerConfig object
                    values["model"].atom_input_features = FEATURESET_SIZE[
                        values["atom_features"]
                    ]
        return values
