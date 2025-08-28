#!/usr/bin/env python3
"""
Author: Bin Cao (bcao686@connect.hkust-gz.edu.cn)
Affiliations:
- Hong Kong University of Science and Technology (Guangzhou)
- City University of Hong Kong

Description: Prdnet Trainer with Data Caching - Provides training functionality
for Prdnet models with data caching support. Supports ASE database files and
configurable caching strategies.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import io

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

# ASE for database handling
from ase.db import connect
from ase import Atoms

# Prdnet imports
from prdnet.train import train_dgl
from prdnet.config import TrainingConfig
from prdnet.data import get_train_val_loaders
from prdnet.model import PrdnetConfig
from prdnet.graphs import PygStructureDataset
from prdnet.cached_dataset import CachedPrdnetDataset

# Set up logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Additional imports for graph processing
from jarvis.core.atoms import Atoms as JarvisAtoms
from torch_geometric.data import Batch



class PrdnetTrainer:
    """Main trainer class for Prdnet training with data caching support."""

    def __init__(self, config: Union[Dict[str, Any], "PrdnetTrainingConfig", TrainingConfig]):
        """Initialize the PrdnetTrainer.

        Args:
            config: Training configuration object or dictionary
        """
        # Setup distributed training first
        self._setup_distributed()

        # Convert to PrdnetTrainingConfig if needed
        if isinstance(config, dict):
            # Add caching-specific parameters to config if not present
            config = self._add_caching_config(config)
            # Import here to avoid circular import
            from trainer import PrdnetTrainingConfig
            self.config = PrdnetTrainingConfig(**config)
        elif isinstance(config, TrainingConfig):
            # Convert TrainingConfig to PrdnetTrainingConfig
            config_dict = config.dict() if hasattr(config, 'dict') else config.model_dump()
            config_dict = self._add_caching_config(config_dict)
            # Import here to avoid circular import
            from trainer import PrdnetTrainingConfig
            self.config = PrdnetTrainingConfig(**config_dict)
        else:
            self.config = config

        # Setup logging
        self._setup_logging()

        # Initialize caching system
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_format = self.config.cache_format
        self.use_cache = self.config.use_cache
        self.force_cache_rebuild = self.config.force_cache_rebuild

        if self.is_main_process:
            logger.info(f"PrdnetTrainer initialized with caching: {self.use_cache}")
            logger.info(f"Cache directory: {self.cache_dir}")
            logger.info(f"Cache format: {self.cache_format}")
            if self.is_distributed:
                logger.info(f"Distributed training: rank {self.rank}/{self.world_size}")
    
    def _setup_distributed(self):
        """Setup distributed training parameters."""
        self.is_distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ

        if self.is_distributed:
            import torch.distributed as dist
            if dist.is_initialized():
                self.rank = dist.get_rank()
                self.world_size = dist.get_world_size()
                self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
                self.is_main_process = self.rank == 0
            else:
                # Fallback if not initialized yet
                self.rank = int(os.environ.get('RANK', 0))
                self.world_size = int(os.environ.get('WORLD_SIZE', 1))
                self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
                self.is_main_process = self.rank == 0
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            self.is_main_process = True

    def _add_caching_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Add caching-specific and distributed training configuration parameters."""
        caching_defaults = {
            'cache_dir': './prdnet_cache',
            'cache_format': 'pt',  # 'pt' or 'hdf5'
            'use_cache': True,
            'force_cache_rebuild': False,
            'cache_validation': True,
        }

        # Add distributed training parameters
        distributed_defaults = {
            'distributed': self.is_distributed,
            'rank': self.rank,
            'world_size': self.world_size,
            'local_rank': self.local_rank,
        }

        # Merge all defaults
        all_defaults = {**caching_defaults, **distributed_defaults}

        for key, value in all_defaults.items():
            if key not in config:
                config[key] = value

        return config
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(self.config, 'log_level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def _distributed_barrier(self, timeout_minutes=120):
        """
        Synchronize all distributed processes with robust error handling.

        Args:
            timeout_minutes: Maximum time to wait for barrier in minutes (default: 2 hours)
        """
        if self.is_distributed:
            try:
                import torch.distributed as dist

                if dist.is_initialized():
                    logger.info(f"Rank {self.rank}: Waiting at distributed barrier (timeout: {timeout_minutes} minutes)")

                    # Retry mechanism for barrier operations
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            # For PyTorch 2.2.2, barrier() doesn't support opts parameter
                            # Use the default barrier with the global timeout set during init_process_group
                            dist.barrier()
                            logger.info(f"Rank {self.rank}: Successfully passed distributed barrier")
                            return

                        except Exception as e:
                            if attempt < max_retries - 1:
                                logger.warning(f"Rank {self.rank}: Barrier attempt {attempt + 1} failed: {e}. Retrying...")
                                import time
                                time.sleep(10)  # Wait 10 seconds before retry
                            else:
                                logger.error(f"Rank {self.rank}: All barrier attempts failed: {e}")
                                raise
                else:
                    logger.warning("Distributed training requested but torch.distributed not initialized")
            except ImportError:
                logger.warning("torch.distributed not available, skipping barrier")
        else:
            # No-op for single process training
            pass
    
    def train(self, train_db_path: str, val_db_path: str = None, test_db_path: str = None):
        """
        Main training function that integrates caching with prdnet training.

        Args:
            train_db_path: Path to training database
            val_db_path: Path to validation database (optional)
            test_db_path: Path to test database (optional)
        """
        logger.info("Starting Prdnet training with data caching")

        # Setup data paths
        self.train_db_path = train_db_path
        self.val_db_path = val_db_path or train_db_path
        self.test_db_path = test_db_path or val_db_path or train_db_path

        # Check if we should use cached data
        if self.use_cache:
            # Distributed-safe data generation: only rank 0 generates data
            if self.is_distributed:
                if self.rank == 0:
                    logger.info("Rank 0: Generating/validating cached data for all processes")
                    train_loader, val_loader, test_loader = self._create_cached_data_loaders()
                    logger.info("Rank 0: Data generation complete, signaling other processes")
                else:
                    logger.info(f"Rank {self.rank}: Waiting for rank 0 to complete data generation")

                # Synchronize all processes - wait for rank 0 to finish data generation
                if self.rank != 0:
                    logger.info(f"Rank {self.rank}: Waiting for rank 0 to complete all data generation")
                    self._distributed_barrier(timeout_minutes=180)  # 3 hour timeout for large datasets

                    # Now non-rank-0 processes can safely load the cached data
                    logger.info(f"Rank {self.rank}: Loading cached data generated by rank 0")
                    train_loader, val_loader, test_loader = self._create_cached_data_loaders()
                else:
                    # Rank 0 signals completion to other processes
                    logger.info("Rank 0: Signaling completion to other processes")
                    self._distributed_barrier(timeout_minutes=180)
            else:
                # Single process training
                train_loader, val_loader, test_loader = self._create_cached_data_loaders()

            # Create prepare_batch function for cached data loaders
            def prepare_batch(batch, device=None, non_blocking=False):
                """Prepare batch for training with device handling."""
                # Check if using line graph
                if train_loader.dataset.line_graph:
                    # For line graph: batch = (batched_graph, batched_line_graph, lattice, targets)
                    batched_graph, batched_line_graph, lattice, targets = batch

                    # Move to device if specified
                    if device is not None:
                        batched_graph = batched_graph.to(device, non_blocking=non_blocking)
                        batched_line_graph = batched_line_graph.to(device, non_blocking=non_blocking)
                        lattice = lattice.to(device, non_blocking=non_blocking)
                        targets = targets.to(device, non_blocking=non_blocking)

                    return (batched_graph, batched_line_graph, lattice), targets
                else:
                    # For regular graph: batch = (batched_data, targets)
                    batched_data, targets = batch

                    # Move to device if specified
                    if device is not None:
                        if hasattr(batched_data, 'to'):
                            batched_data = batched_data.to(device, non_blocking=non_blocking)
                        if hasattr(targets, 'to'):
                            targets = targets.to(device, non_blocking=non_blocking)

                    return batched_data, targets
        else:
            train_loader, val_loader, test_loader, prepare_batch, mean_train, std_train = self._create_standard_data_loaders()

        # Start prdnet training with prepared data loaders
        result = train_dgl(
            config=self.config,
            train_val_test_loaders=[train_loader, val_loader, test_loader, prepare_batch]
        )

        logger.info(f"Training completed with result: {result}")
        return result

    def _create_cached_data_loaders(self):
        """
        Create data loaders using cached preprocessed data for improved performance.

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if self.is_main_process:
            logger.info("Creating cached data loaders for improved performance")

        # Check and create cache for each dataset
        datasets = {
            'train': self.train_db_path,
            'val': self.val_db_path,
            'test': self.test_db_path
        }

        data_loaders = {}

        # Track cache statistics
        cache_hits = 0
        cache_misses = 0

        for split, db_path in datasets.items():
            if self.is_main_process:
                logger.info(f"Processing {split} dataset: {Path(db_path).name}")

            # Check if cache exists and is valid
            cache_valid = self._validate_cache(db_path) and not self.force_cache_rebuild

            if not cache_valid:
                # Only rank 0 creates cache in distributed training
                if self.is_distributed:
                    if self.rank == 0:
                        logger.info(f"Rank 0: Creating cache for {split} dataset: {db_path}")
                        try:
                            self._preprocess_and_cache_data(db_path)
                            logger.info(f"Rank 0: Cache creation complete for {split}")
                        except Exception as e:
                            logger.error(f"Rank 0: Failed to create cache for {split}: {e}")
                            raise

                    # All processes wait for rank 0 to finish caching this dataset
                    # Use extended timeout for data preprocessing operations
                    logger.info(f"Rank {self.rank}: Waiting for cache creation to complete for {split}")
                    self._distributed_barrier(timeout_minutes=180)  # 3 hour timeout for large datasets
                else:
                    # Single process training
                    if self.is_main_process:
                        logger.info(f"Creating cache for {split} dataset: {db_path}")
                        self._preprocess_and_cache_data(db_path)

                cache_misses += 1
            else:
                if self.is_main_process:
                    logger.info(f"Loading cached data for {split} dataset")
                cache_hits += 1

            # Create cached dataset
            cached_dataset = CachedPrdnetDataset(db_path, self.config)

            # Use prdnet's collate function
            collate_fn = cached_dataset.collate
            if cached_dataset.line_graph:  # 检查数据集的 line_graph 属性
                collate_fn = cached_dataset.collate_line_graph

            # Create data loader
            data_loaders[split] = DataLoader(
                cached_dataset,
                batch_size=self.config.batch_size,
                shuffle=(split == 'train'),
                num_workers=getattr(self.config, 'num_workers', 0),
                pin_memory=True,
                collate_fn=collate_fn,
                drop_last=True
            )

        # Display cache statistics summary
        total_datasets = len(datasets)
        logger.info(f"Cache Statistics Summary:")
        logger.info(f"  Cache Hits: {cache_hits}/{total_datasets} ({cache_hits/total_datasets*100:.1f}%)")
        logger.info(f"  Cache Misses: {cache_misses}/{total_datasets} ({cache_misses/total_datasets*100:.1f}%)")

        if cache_hits == total_datasets:
            logger.info(f"  All data loaded from cache.")
        elif cache_hits > 0:
            logger.info(f"  Some data loaded from cache.")
        else:
            logger.info(f"  All data generated fresh.")

        return data_loaders['train'], data_loaders['val'], data_loaders['test']

    def _create_standard_data_loaders(self):
        """
        Create standard data loaders using prdnet's built-in functionality.

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        logger.info("Creating standard data loaders (no caching)")

        # Convert ASE databases to prdnet format
        dataset_array = self._convert_ase_to_prdnet_format()

        # Split data into train/val/test
        total_size = len(dataset_array)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)

        train_data = dataset_array[:train_size]
        val_data = dataset_array[train_size:train_size + val_size]
        test_data = dataset_array[train_size + val_size:]

        # Create data loaders using prdnet's function
        train_loader, val_loader, test_loader, prepare_batch_fn, mean_train, std_train = get_train_val_loaders(
            dataset="custom",
            dataset_array=train_data,
            target=self.config.target,
            atom_features=self.config.atom_features,
            neighbor_strategy=self.config.neighbor_strategy,
            batch_size=self.config.batch_size,
            workers=getattr(self.config, 'num_workers', 0),
            cutoff=getattr(self.config, 'cutoff', 8.0),
            max_neighbors=getattr(self.config, 'max_neighbors', 12),
            use_lattice=getattr(self.config, 'use_lattice', True),
            use_angle=getattr(self.config, 'use_angle', False),
            line_graph=True,  # 启用 line graph，与模型要求一致
            output_dir=self.config.output_dir,  # 提供输出目录
        )

        return train_loader, val_loader, test_loader, prepare_batch_fn, mean_train, std_train

    def _validate_cache(self, db_path: str) -> bool:
        """
        Validate if cache exists and is up-to-date with detailed logging.

        Args:
            db_path: Path to the database file

        Returns:
            True if cache is valid, False otherwise
        """
        db_name = Path(db_path).stem

        # Check if cache files exist
        graphs_path = self._get_cache_path(db_path, "graphs")
        targets_path = self._get_cache_path(db_path, "targets")
        df_path = self._get_cache_path(db_path, "dataframe")

        cache_files = [graphs_path, targets_path, df_path]
        cache_file_names = ["graphs", "targets", "dataframe"]

        # Log cache file status (only from main process)
        if self.is_main_process:
            logger.info(f"Cache validation for {db_name}:")

        cache_exists = True
        for file_path, file_type in zip(cache_files, cache_file_names):
            if file_path.exists():
                if self.is_main_process:
                    file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                    logger.info(f"  {file_type}: {file_path.name} ({file_size:.2f} MB)")
            else:
                if self.is_main_process:
                    logger.info(f"  {file_type}: {file_path.name} (missing)")
                cache_exists = False

        if not cache_exists:
            if self.is_main_process:
                logger.info(f"Cache status: MISS - Some cache files are missing")
            return False

        # Check if cache is newer than database
        try:
            db_mtime = Path(db_path).stat().st_mtime

            if self.cache_format == "hdf5":
                import h5py
                with h5py.File(graphs_path, 'r') as f:
                    cache_mtime = f.attrs.get('db_mtime', 0)
            else:
                cache_data = torch.load(graphs_path)
                cache_mtime = cache_data.get('db_mtime', 0)

            # Convert timestamps to readable format
            import datetime
            db_time_str = datetime.datetime.fromtimestamp(db_mtime).strftime('%Y-%m-%d %H:%M:%S')
            cache_time_str = datetime.datetime.fromtimestamp(cache_mtime).strftime('%Y-%m-%d %H:%M:%S')

            logger.info(f"  Database modified: {db_time_str}")
            logger.info(f"  Cache modified: {cache_time_str}")

            if cache_mtime >= db_mtime:
                logger.info(f"Cache status: HIT - Cache is up-to-date and will be used")
                return True
            else:
                logger.info(f"Cache status: MISS - Cache is outdated, will regenerate")
                return False

        except Exception as e:
            logger.warning(f"Cache validation failed: {e}")
            logger.info(f"Cache status: MISS - Validation error")
            return False

    def _get_cache_path(self, db_path: str, data_type: str) -> Path:
        """
        Get the cache file path for a specific data type.

        Args:
            db_path: Original database path
            data_type: Type of data ('graphs', 'targets', etc.)

        Returns:
            Path to cache file
        """
        db_name = Path(db_path).stem
        cache_subdir = self.cache_dir / data_type
        cache_subdir.mkdir(parents=True, exist_ok=True)

        if self.cache_format == "hdf5":
            return cache_subdir / f"{db_name}_{data_type}.h5"
        else:
            return cache_subdir / f"{db_name}_{data_type}.pt"

    def _preprocess_and_cache_data(self, db_path: str):
        """
        Preprocess database data and cache for fast loading with progress reporting.

        Args:
            db_path: Path to the database file to preprocess
        """
        import time
        start_time = time.time()
        logger.info(f"Preprocessing and caching data from {db_path}")

        try:
            # Create cache directory
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Connect to ASE database
            db = connect(db_path)

            # Convert ASE database to prdnet format
            structures_data = []
            targets = []

            total_structures = len(db)
            logger.info(f"Processing {total_structures} structures from database")

            # Memory management counters
            processed_count = 0
            batch_size = 200  # Clean cache every 200 structures
            save_interval = 1000  # Save to disk every 1000 structures
            progress_interval = 100  # Report progress every 100 structures

            # Send periodic heartbeat to prevent timeout
            last_heartbeat = time.time()
            heartbeat_interval = 60  # Send heartbeat every 60 seconds

            for row in tqdm(db.select(), desc="Converting structures"):
                atoms = row.toatoms()

                # Convert ASE Atoms to JARVIS-compatible dict format
                atoms_dict = {
                    'lattice_mat': atoms.cell.array.tolist(),
                    'coords': atoms.get_scaled_positions().tolist(),
                    'elements': atoms.get_chemical_symbols(),
                    'cartesian': False,  # coords are fractional
                    'props': {}  # Required by JARVIS Atoms.from_dict
                }

                # Create structure dictionary for prdnet
                structure_dict = {
                    'atoms': atoms_dict,
                    'jid': str(row.id),  # Use database ID as identifier
                }

                # Extract target property
                target_value = row.get(self.config.target, None)
                if target_value is not None:
                    structures_data.append(structure_dict)
                    targets.append(target_value)
                    processed_count += 1

                    # Progress reporting
                    if processed_count % progress_interval == 0:
                        elapsed = time.time() - start_time
                        rate = processed_count / elapsed
                        eta = (total_structures - processed_count) / rate if rate > 0 else 0
                        logger.info(f"Progress: {processed_count}/{total_structures} ({processed_count/total_structures*100:.1f}%) "
                                  f"Rate: {rate:.1f} structures/sec, ETA: {eta/60:.1f} min")

                    # Memory management: cleanup every batch_size structures
                    if processed_count % batch_size == 0:
                        import gc
                        gc.collect()
                        logger.debug(f"Memory cleanup after {processed_count} structures")

                    # Periodic saving: save intermediate results every save_interval
                    if processed_count % save_interval == 0:
                        logger.info(f"Intermediate save at {processed_count} structures")
                        self._save_intermediate_data(db_path, structures_data, targets, processed_count)

                # Send heartbeat to prevent distributed timeout
                current_time = time.time()
                if current_time - last_heartbeat > heartbeat_interval:
                    logger.debug(f"Heartbeat: Processing structure {processed_count}/{total_structures}")
                    last_heartbeat = current_time

            logger.info(f"Converted {len(structures_data)} valid structures")

            # Create pandas DataFrame for prdnet compatibility
            df = pd.DataFrame(structures_data)

            # Add target column to dataframe
            df[self.config.target] = targets

            # Generate graphs using prdnet's graph generation
            from prdnet.graphs import PygGraph

            graphs = []
            graph_count = 0
            graph_start_time = time.time()

            logger.info(f"Starting graph generation for {len(df)} structures")

            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating graphs"):
                try:
                    # Convert atoms dict back to JARVIS Atoms object for graph generation
                    atoms_obj = JarvisAtoms.from_dict(row['atoms'])

                    # Generate graph using prdnet's method
                    result = PygGraph.atom_dgl_multigraph(
                        atoms=atoms_obj,
                        neighbor_strategy=self.config.neighbor_strategy,
                        cutoff=getattr(self.config, 'cutoff', 8.0),
                        max_neighbors=getattr(self.config, 'max_neighbors', 12),
                        atom_features="atomic_number",
                        compute_line_graph=True,  # 生成 line graph
                        use_canonize=False,
                        use_lattice=getattr(self.config, 'use_lattice', True),
                        use_angle=getattr(self.config, 'use_angle', False),
                    )

                    # When compute_line_graph=True, result is (graph, line_graph)
                    if isinstance(result, tuple):
                        graph, _ = result  # Use _ for unused line_graph variable
                        graphs.append(graph)  # 只保存主图，line graph 会在数据集中重新生成
                    else:
                        graphs.append(result)

                    graph_count += 1

                    # Progress reporting for graph generation
                    if graph_count % progress_interval == 0:
                        elapsed = time.time() - graph_start_time
                        rate = graph_count / elapsed if elapsed > 0 else 0
                        eta = (len(df) - graph_count) / rate if rate > 0 else 0
                        logger.info(f"Graph generation progress: {graph_count}/{len(df)} ({graph_count/len(df)*100:.1f}%) "
                                  f"Rate: {rate:.1f} graphs/sec, ETA: {eta/60:.1f} min")

                    # Memory management for graph generation
                    if graph_count % batch_size == 0:
                        import gc
                        gc.collect()
                        logger.debug(f"Memory cleanup after {graph_count} graphs")

                    # Send heartbeat during graph generation
                    current_time = time.time()
                    if current_time - last_heartbeat > heartbeat_interval:
                        logger.debug(f"Graph generation heartbeat: {graph_count}/{len(df)} graphs processed")
                        last_heartbeat = current_time

                except Exception as e:
                    logger.error(f"Failed to generate graph for structure {idx}: {e}")
                    logger.error(f"Structure data: {row['atoms']}")
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    continue

            logger.info(f"Generated {len(graphs)} graphs")

            # Save cached data
            self._save_cached_data(db_path, {
                'graphs': graphs,
                'targets': targets[:len(graphs)],  # Match length with successful graphs
                'dataframe': df.iloc[:len(graphs)]  # Store dataframe for dataset creation
            })

            logger.info(f"Cached {len(graphs)} structures from {db_path}")

        except Exception as e:
            logger.error(f"Failed to preprocess data: {e}")
            raise

    def _save_intermediate_data(self, db_path: str, structures_data: list, targets: list, count: int):
        """
        Save intermediate data to prevent memory overflow and enable recovery.

        Args:
            db_path: Original database path
            structures_data: List of structure dictionaries
            targets: List of target values
            count: Current count of processed structures
        """
        try:
            db_name = Path(db_path).stem
            intermediate_dir = self.cache_dir / "intermediate"
            intermediate_dir.mkdir(parents=True, exist_ok=True)

            # Save intermediate structures with timestamp for recovery
            import time
            timestamp = int(time.time())
            intermediate_file = intermediate_dir / f"{db_name}_intermediate_{count}_{timestamp}.pt"
            torch.save({
                'structures': structures_data,
                'targets': targets,
                'count': count,
                'timestamp': time.time()
            }, intermediate_file)

            logger.debug(f"Saved intermediate data to {intermediate_file}")

        except Exception as e:
            logger.warning(f"Failed to save intermediate data: {e}")

    def _save_cached_data(self, db_path: str, data: dict):
        """
        Save preprocessed data to cache files.

        Args:
            db_path: Original database path
            data: Dictionary containing preprocessed data
        """
        db_mtime = Path(db_path).stat().st_mtime

        if self.cache_format == "hdf5":
            import h5py

            # Save each data type to separate HDF5 files
            for data_type, data_content in data.items():
                cache_path = self._get_cache_path(db_path, data_type)

                with h5py.File(cache_path, 'w') as f:
                    f.attrs['db_mtime'] = db_mtime

                    if data_type == 'targets':
                        f.attrs['num_samples'] = len(data_content)
                        f.create_dataset('data', data=np.array(data_content))
                    elif data_type == 'dataframe':
                        # Serialize dataframe to JSON
                        df_json = data_content.to_json()
                        f.create_dataset('data', data=[df_json])
                    else:
                        # Store as serialized tensors for complex data (graphs)
                        f.attrs['num_samples'] = len(data_content)
                        serialized_data = []
                        for item in data_content:
                            buffer = io.BytesIO()
                            torch.save(item, buffer)
                            serialized_data.append(buffer.getvalue())
                        f.create_dataset('data', data=serialized_data)
        else:
            # Save as PyTorch files
            for data_type, data_content in data.items():
                cache_path = self._get_cache_path(db_path, data_type)

                save_data = {
                    'data': data_content,
                    'db_mtime': db_mtime,
                }

                if data_type != 'dataframe':
                    save_data['num_samples'] = len(data_content)

                torch.save(save_data, cache_path)

        logger.info(f"Saved cached data to {self.cache_dir}")

    def _convert_ase_to_prdnet_format(self) -> List[Dict]:
        """
        Convert ASE database files to prdnet-compatible format.

        Returns:
            List of structure dictionaries compatible with prdnet
        """
        all_structures = []

        for db_path in [self.train_db_path, self.val_db_path, self.test_db_path]:
            if db_path and Path(db_path).exists():
                logger.info(f"Converting database: {db_path}")
                db = connect(db_path)

                for row in tqdm(db.select(), desc=f"Converting {Path(db_path).stem}"):
                    atoms = row.toatoms()
                    target_value = row.get(self.config.target, None)

                    if target_value is not None:
                        # Convert ASE Atoms to dict format for prdnet
                        atoms_dict = {
                            'numbers': atoms.numbers.tolist(),
                            'positions': atoms.positions.tolist(),
                            'cell': atoms.cell.array.tolist(),
                            'pbc': atoms.pbc.tolist(),
                        }

                        structure_dict = {
                            'atoms': atoms_dict,
                            'jid': str(row.id),
                            self.config.target: target_value
                        }
                        all_structures.append(structure_dict)

        logger.info(f"Converted {len(all_structures)} structures total")
        return all_structures

    def convert_single_database(self, db_path: str) -> List[Dict]:
        """
        Convert a single ASE database to prdnet format.

        Args:
            db_path: Path to ASE database file

        Returns:
            List of structure dictionaries
        """
        structures = []

        if not Path(db_path).exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

        logger.info(f"Converting database: {db_path}")
        db = connect(db_path)

        for row in tqdm(db.select(), desc=f"Converting {Path(db_path).stem}"):
            atoms = row.toatoms()
            target_value = row.get(self.config.target, None)

            if target_value is not None:
                # Convert ASE Atoms to JARVIS-compatible dict format
                atoms_dict = {
                    'lattice_mat': atoms.cell.array.tolist(),
                    'coords': atoms.get_scaled_positions().tolist(),
                    'elements': atoms.get_chemical_symbols(),
                    'cartesian': False,  # coords are fractional
                    'props': {}  # Required by JARVIS Atoms.from_dict
                }

                structure_dict = {
                    'atoms': atoms_dict,
                    'jid': str(row.id),
                    self.config.target: target_value
                }
                structures.append(structure_dict)

        logger.info(f"Converted {len(structures)} structures from {db_path}")
        return structures




class PrdnetTrainingConfig(TrainingConfig):
    """
    Extended TrainingConfig that includes caching parameters for PrdnetTrainer.

    This class extends prdnet's TrainingConfig to add data caching functionality
    while maintaining compatibility with the original training pipeline.
    """

    # Caching configuration
    cache_dir: str = "./prdnet_cache"
    cache_format: str = "pt"  # 'pt' or 'hdf5'
    use_cache: bool = True
    force_cache_rebuild: bool = False
    cache_validation: bool = True

    # Database paths for ASE databases
    train_db_path: Optional[str] = None
    val_db_path: Optional[str] = None
    test_db_path: Optional[str] = None

    # Additional system parameters
    log_level: str = "INFO"

    # Allow extra fields for flexibility
    class Config:
        extra = "allow"


def create_trainer_config(
    train_db_path: str,
    val_db_path: str = None,
    test_db_path: str = None,
    target_property: str = "formation_energy",
    cache_dir: str = "./prdnet_cache",
    use_cache: bool = True,
    **kwargs
) -> "PrdnetTrainingConfig":
    """
    Create a configuration object for PrdnetTrainer.

    Args:
        train_db_path: Path to training database
        val_db_path: Path to validation database
        test_db_path: Path to test database
        target_property: Target property to predict
        cache_dir: Directory for caching preprocessed data
        use_cache: Whether to use data caching
        **kwargs: Additional configuration parameters

    Returns:
        PrdnetTrainingConfig object
    """
    config_dict = {
        # Data paths
        'train_db_path': train_db_path,
        'val_db_path': val_db_path or train_db_path,
        'test_db_path': test_db_path or val_db_path or train_db_path,

        # Target and features
        'target': target_property,
        'atom_features': 'cgcnn',
        'neighbor_strategy': 'k-nearest',

        # Enhanced model configuration with diffraction integration
        'model': {
            'name': 'prdnet',
            'conv_layers': 6,
            'node_features': 256,
            'edge_features': 256,
            'fc_layers': 2,
            'fc_features': 512,
            'output_features': 1,
            'node_layer_head': 8,
            'dropout': 0.1,
            'use_residual_connections': True,
            'use_layer_norm': True,
            'activation': 'silu',
            'use_diffraction': True,
            'diffraction_max_hkl': 5,
            'diffraction_num_hkl': 300,
        },

        # Training parameters
        'epochs': 500,
        'batch_size': 64,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'criterion': 'mse',

        # Data processing
        'cutoff': 8.0,
        'max_neighbors': 12,
        'use_lattice': True,
        'use_angle': False,

        # Caching configuration
        'cache_dir': cache_dir,
        'cache_format': 'pt',  # 'pt' or 'hdf5'
        'use_cache': use_cache,
        'force_cache_rebuild': False,
        'cache_validation': True,

        # System parameters
        'num_workers': 0,
        'output_dir': './prdnet_output',
        'log_level': 'INFO',
    }

    # Handle model configuration override before updating kwargs
    if 'model_config' in kwargs:
        model_config = kwargs.pop('model_config')
        config_dict['model'].update(model_config)

    # Update with any additional parameters
    config_dict.update(kwargs)

    # Create and return PrdnetTrainingConfig object
    return PrdnetTrainingConfig(**config_dict)


def main():
    """
    Example usage of PrdnetTrainer with data caching.
    """
    # Optimized configuration for MP_100_bgfe dataset with enhanced hyperparameters and diffraction integration
    config = create_trainer_config(
        train_db_path="MP_100_bgfe_train.db",
        val_db_path="MP_100_bgfe_val.db",
        test_db_path="MP_100_bgfe_test.db",
        target_property="formation_energy",
        cache_dir="./prdnet_cache",
        use_cache=True,
        epochs=500,
        batch_size=32,  
        learning_rate=0.0005,  # Reduced for stability with enhanced model
        optimizer="adamw",
        scheduler="cosine_warm",  # Better convergence
        criterion="huber",  # More robust to outliers
        grad_clip_norm=1.0,  # Gradient clipping for stability
        weight_decay=0.01,  # Regularization
        model_config={
            "use_diffraction": True,
            "diffraction_max_hkl": 4,  # Further reduced for memory efficiency
            "diffraction_num_hkl": 150,  # Further reduced for memory efficiency
            "conv_layers": 5,  # Reduced from 6 for memory efficiency
            "node_features": 192,  # Reduced from 256 for memory efficiency
            "edge_features": 192,  # Reduced from 256 for memory efficiency
            "fc_layers": 2,
            "fc_features": 384,  # Reduced from 512 for memory efficiency
            "node_layer_head": 6,  # Reduced from 8 for memory efficiency
            "dropout": 0.15,  # Slightly increased for regularization
            "use_residual_connections": True,
            "use_layer_norm": True,
            "activation": "silu"
        }
    )

    # Create and run trainer
    trainer = PrdnetTrainer(config)

    # Start training
    result = trainer.train(
        train_db_path=config.train_db_path,
        val_db_path=config.val_db_path,
        test_db_path=config.test_db_path
    )

    print(f"Training completed with MAE: {result}")


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Prdnet Training with Distributed Support')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    args = parser.parse_args()

    # Set environment variables for memory optimization and deterministic behavior
    import os
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Enable memory optimization
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Enable memory efficient attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except:
            pass

    # Handle distributed training properly
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        import torch.distributed as dist
        from datetime import timedelta

        # Set environment variables for better distributed training stability
        os.environ.setdefault('NCCL_TIMEOUT', '3600')  # 1 hour timeout for NCCL operations
        os.environ.setdefault('NCCL_BLOCKING_WAIT', '1')  # Enable blocking wait for better error reporting
        os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')  # Enable async error handling
        os.environ.setdefault('NCCL_DEBUG', 'INFO')  # Enable debug info for troubleshooting

        # Set longer timeout for distributed operations (especially for data preprocessing)
        timeout = timedelta(hours=2)  # 2 hour timeout for long data preprocessing operations

        # Initialize distributed training with extended timeout
        dist.init_process_group(
            backend='nccl',
            timeout=timeout
        )

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        # Set device for this process
        torch.cuda.set_device(local_rank)

        if rank == 0:
            print(f"Initialized distributed training with {world_size} processes")
            print(f"Distributed timeout set to: {timeout}")

        try:
            # All processes run main, but only rank 0 creates cache
            main()
        finally:
            # Cleanup
            if dist.is_initialized():
                dist.destroy_process_group()
    else:
        # Single process training
        main()
