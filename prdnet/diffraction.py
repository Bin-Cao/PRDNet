"""
Description: Diffraction Information Integration Module for Prdnet.
This module implements crystal diffraction information extraction and integration
It calculates structure factors from crystal graphs
and fuses this information with existing graph representations.

Key Features:
- Extract atomic diffraction factors from aggregated node attributes
- Calculate structure factors for selected HKL indices
- Generate structure factor maps
- Fuse diffraction information with graph data
- Residual connections for training stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
from torch_geometric.data import Data
from torch_geometric.utils import scatter


class HKLSelector:
    """Select diverse, representative HKL indices for structure factor calculation."""
    
    def __init__(self, max_hkl: int = 3, num_indices: int = 100):
        """
        Initialize HKL selector with simplified parameters for memory efficiency.

        Args:
            max_hkl: Maximum value for h, k, l indices (reduced for memory efficiency)
            num_indices: Target number of HKL indices to select (reduced for memory efficiency)
        """
        self.max_hkl = max_hkl
        self.num_indices = num_indices
        self.hkl_indices = self._generate_hkl_indices()
    
    def _generate_hkl_indices(self) -> torch.Tensor:
        """Generate symmetry-closed HKL indices ensuring closure under symmetry operations."""
        import itertools

        # Step 1: Generate initial set H₀ with basic constraints
        initial_hkl_list = []

        # Generate all possible HKL combinations for initial set
        for h in range(-self.max_hkl, self.max_hkl + 1):
            for k in range(-self.max_hkl, self.max_hkl + 1):
                for l in range(-self.max_hkl, self.max_hkl + 1):
                    # Skip (0,0,0)
                    if h == 0 and k == 0 and l == 0:
                        continue

                    # Apply basic Friedel's law to reduce initial set size
                    # Keep only one of each Friedel pair for initial generation
                    if h > 0 or (h == 0 and k > 0) or (h == 0 and k == 0 and l > 0):
                        initial_hkl_list.append([h, k, l])

        # Step 2: Apply symmetry closure to ensure H is closed under symmetry operations
        # H = {±perm(h, k, l) : (h, k, l) ∈ H₀}
        symmetry_closed_hkl = set()

        for hkl in initial_hkl_list:
            h, k, l = hkl

            # Generate all permutations of (h, k, l)
            for perm in itertools.permutations([h, k, l]):
                # Apply both positive and negative (inversion symmetry)
                symmetry_closed_hkl.add(tuple(perm))
                symmetry_closed_hkl.add(tuple([-x for x in perm]))

        # Convert back to list and remove (0,0,0) if it was added
        symmetry_closed_hkl.discard((0, 0, 0))
        all_hkl_list = list(symmetry_closed_hkl)

        # Convert to tensor
        all_hkl = torch.tensor(all_hkl_list, dtype=torch.float32)

        # Step 3: Select diverse subset based on d-spacing distribution
        if len(all_hkl) > self.num_indices:
            # Calculate d-spacing for each HKL (assuming cubic lattice for selection)
            d_spacings = 1.0 / torch.sqrt(torch.sum(all_hkl**2, dim=1) + 1e-8)

            # Sort by d-spacing and select evenly distributed indices
            sorted_indices = torch.argsort(d_spacings, descending=True)
            step = len(sorted_indices) // self.num_indices
            selected_indices = sorted_indices[::step][:self.num_indices]
            selected_hkl = all_hkl[selected_indices]
        else:
            selected_hkl = all_hkl

        return selected_hkl


class StructureFactorCalculator(nn.Module):
    """Calculate structure factors from crystal graph data with per-HKL form factors."""

    def __init__(self, hkl_indices: torch.Tensor, node_features: int = 256):
        """
        Initialize structure factor calculator with per-HKL form factor learning.

        Args:
            hkl_indices: HKL indices for structure factor calculation
            node_features: Number of node features in the graph
        """
        super().__init__()
        self.register_buffer('hkl_indices', hkl_indices)
        self.node_features = node_features
        self.num_hkl = hkl_indices.shape[0]

        # Per-HKL form factor network: f*ᵢ(H) = MLPform(h⁽ᴸ⁾ᵢ) ∈ ℝ^(Nhkl)
        # This maps node embeddings to scattering strengths for each HKL index
        self.form_factor_net = nn.Sequential(
            nn.Linear(node_features, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, self.num_hkl)  # Output dimension is Nhkl
        )
    
    def forward(self, node_features: torch.Tensor, pos: torch.Tensor,
                batch: torch.Tensor, lattice: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate structure factors with per-HKL form factors.

        Args:
            node_features: Node features from graph [num_nodes, node_features]
            pos: Atomic positions [num_nodes, 3]
            batch: Batch indices [num_nodes]
            lattice: Lattice parameters [batch_size, 6] (optional)

        Returns:
            Structure factor map [batch_size, num_hkl, 2] (real and imaginary parts)
        """
        batch_size = batch.max().item() + 1
        num_hkl = self.hkl_indices.shape[0]
        device = node_features.device

        # Calculate per-HKL form factors from node features
        # f*i(H) = MLPform(h(L)1) ∈ R^(Nhkl)
        form_factors_per_hkl = self.form_factor_net(node_features)  # [num_nodes, num_hkl]

        # Initialize structure factor tensor
        structure_factors = torch.zeros(batch_size, num_hkl, 2, device=device)

        # Calculate structure factors for each crystal in the batch
        for batch_idx in range(batch_size):
            # Get atoms for this crystal
            mask = batch == batch_idx
            if not mask.any():
                continue

            crystal_pos = pos[mask]  # [num_atoms_in_crystal, 3]
            crystal_form_factors = form_factors_per_hkl[mask]  # [num_atoms_in_crystal, num_hkl]

            # Calculate structure factors for all HKL indices
            # F(hkl) = sum_j f_j(hkl) * exp(2πi * (h*x_j + k*y_j + l*z_j))
            hkl_dot_pos = torch.matmul(self.hkl_indices, crystal_pos.T)  # [num_hkl, num_atoms]
            phase_angles = 2 * np.pi * hkl_dot_pos  # [num_hkl, num_atoms]

            # Calculate real and imaginary parts
            cos_phases = torch.cos(phase_angles)  # [num_hkl, num_atoms]
            sin_phases = torch.sin(phase_angles)  # [num_hkl, num_atoms]

            # Weight by per-HKL form factors and sum over atoms
            # crystal_form_factors is [num_atoms, num_hkl], we need [num_hkl, num_atoms]
            crystal_form_factors_T = crystal_form_factors.T  # [num_hkl, num_atoms]

            real_parts = torch.sum(crystal_form_factors_T * cos_phases, dim=1)  # [num_hkl]
            imag_parts = torch.sum(crystal_form_factors_T * sin_phases, dim=1)  # [num_hkl]

            structure_factors[batch_idx, :, 0] = real_parts
            structure_factors[batch_idx, :, 1] = imag_parts

        return structure_factors


class DiffractionFusionLayer(nn.Module):
    """Fuse diffraction information with graph features."""
    
    def __init__(self, graph_features: int, diffraction_features: int, 
                 output_features: int, use_residual: bool = True):
        """
        Initialize diffraction fusion layer.
        
        Args:
            graph_features: Number of graph features
            diffraction_features: Number of diffraction features (2 * num_hkl)
            output_features: Number of output features
            use_residual: Whether to use residual connections
        """
        super().__init__()
        self.use_residual = use_residual
        
        # Diffraction processing network
        self.diffraction_net = nn.Sequential(
            nn.Linear(diffraction_features, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, graph_features)
        )
        
        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(graph_features * 2, output_features),
            nn.LayerNorm(output_features),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(output_features, output_features)
        )
        
        # Residual projection if needed
        if self.use_residual and graph_features != output_features:
            self.residual_proj = nn.Linear(graph_features, output_features)
        else:
            self.residual_proj = None
    
    def forward(self, graph_features: torch.Tensor, 
                structure_factors: torch.Tensor) -> torch.Tensor:
        """
        Fuse graph features with diffraction information.
        
        Args:
            graph_features: Graph features [batch_size, graph_features]
            structure_factors: Structure factors [batch_size, num_hkl, 2]
        
        Returns:
            Fused features [batch_size, output_features]
        """
        # Flatten structure factors
        batch_size = structure_factors.shape[0]
        diffraction_flat = structure_factors.view(batch_size, -1)
        
        # Process diffraction information
        diffraction_processed = self.diffraction_net(diffraction_flat)
        
        # Concatenate and fuse
        combined = torch.cat([graph_features, diffraction_processed], dim=-1)
        fused = self.fusion_net(combined)
        
        # Apply residual connection
        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(graph_features)
            else:
                residual = graph_features
            fused = fused + residual
        
        return fused


class DiffractionIntegration(nn.Module):
    """Complete diffraction information integration module."""
    
    def __init__(self, node_features: int = 256, graph_features: int = 512, 
                 output_features: int = 512, max_hkl: int = 5, num_hkl: int = 300):
        """
        Initialize complete diffraction integration.
        
        Args:
            node_features: Number of node features
            graph_features: Number of graph features before fusion
            output_features: Number of output features after fusion
            max_hkl: Maximum HKL index value
            num_hkl: Number of HKL indices to use
        """
        super().__init__()
        
        # Initialize HKL selector and get indices
        hkl_selector = HKLSelector(max_hkl=max_hkl, num_indices=num_hkl)
        self.hkl_indices = hkl_selector.hkl_indices

        # Structure factor calculator
        self.structure_factor_calc = StructureFactorCalculator(
            self.hkl_indices, node_features
        )

        # Diffraction fusion layer
        # Use actual number of HKL indices (may be different due to symmetry closure)
        actual_num_hkl = self.hkl_indices.shape[0]
        diffraction_features = actual_num_hkl * 2  # Real and imaginary parts
        self.fusion_layer = DiffractionFusionLayer(
            graph_features, diffraction_features, output_features, use_residual=True
        )
    
    def forward(self, graph_features: torch.Tensor, node_features: torch.Tensor,
                pos: torch.Tensor, batch: torch.Tensor, 
                lattice: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Integrate diffraction information with graph features.
        
        Args:
            graph_features: Aggregated graph features [batch_size, graph_features]
            node_features: Node features [num_nodes, node_features]
            pos: Atomic positions [num_nodes, 3]
            batch: Batch indices [num_nodes]
            lattice: Lattice parameters [batch_size, 6] (optional)
        
        Returns:
            Enhanced features with diffraction information [batch_size, output_features]
        """
        # Calculate structure factors
        structure_factors = self.structure_factor_calc(
            node_features, pos, batch, lattice
        )
        
        # Fuse with graph features
        enhanced_features = self.fusion_layer(graph_features, structure_factors)
        
        return enhanced_features
