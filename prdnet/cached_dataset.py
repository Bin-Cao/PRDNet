"""
Description: Cached dataset implementation for Prdnet training.
This module provides efficient data loading through caching mechanisms.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import torch
import pickle as pk

from prdnet.graphs import PygStructureDataset

logger = logging.getLogger(__name__)


class CachedPrdnetDataset(PygStructureDataset):
    """Dataset for loading preprocessed cached data for Prdnet training."""

    def __init__(self, db_path: str, config):
        """Initialize cached dataset.

        Args:
            db_path: Path to original database file
            config: Training configuration with caching parameters
        """
        self.db_path = db_path
        self.config = config
        self.cache_dir = Path(config.cache_dir)

        graphs, targets, df = self._load_cached_data()

        super().__init__(
            df=df,
            graphs=graphs,
            target=config.target,
            atom_features=getattr(config, 'atom_features', 'atomic_number'),
            line_graph=True,  # Model requires line graph
            classification=False,
            id_tag='jid',
            neighbor_strategy=config.neighbor_strategy,
        )

    def _load_cached_data(self) -> Tuple[List, List, pd.DataFrame]:
        """Load data from cache files."""
        if self.config.cache_format == "hdf5":
            return self._load_hdf5_data()
        else:
            return self._load_pytorch_data()

    def _load_pytorch_data(self):
        """Load data from PyTorch cache files."""
        db_name = Path(self.db_path).stem

        # Load graphs, targets, and dataframe
        graphs_path = self.cache_dir / "graphs" / f"{db_name}_graphs.pt"
        targets_path = self.cache_dir / "targets" / f"{db_name}_targets.pt"
        df_path = self.cache_dir / "dataframe" / f"{db_name}_dataframe.pt"

        if not all(p.exists() for p in [graphs_path, targets_path, df_path]):
            raise FileNotFoundError(f"Cache files not found for {db_name}")

        graphs = torch.load(graphs_path)['data']
        targets = torch.load(targets_path)['data']
        df = torch.load(df_path)['data']

        return graphs, targets, df

    def _load_hdf5_data(self):
        """Load data from HDF5 cache files."""
        import h5py
        import io

        db_name = Path(self.db_path).stem

        # Load graphs
        graphs_path = self.cache_dir / "graphs" / f"{db_name}_graphs.h5"
        with h5py.File(graphs_path, 'r') as f:
            serialized_graphs = f['data'][:]
            graphs = [torch.load(io.BytesIO(data)) for data in serialized_graphs]

        # Load targets
        targets_path = self.cache_dir / "targets" / f"{db_name}_targets.h5"
        with h5py.File(targets_path, 'r') as f:
            targets = f['data'][:]

        # Load dataframe
        df_path = self.cache_dir / "dataframe" / f"{db_name}_dataframe.h5"
        with h5py.File(df_path, 'r') as f:
            df_data = f['data'][:]
            df = pd.read_json(df_data[0])  # Assuming single serialized dataframe

        return graphs, targets, df
