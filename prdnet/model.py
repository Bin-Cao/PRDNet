"""
Description: Prdnet model implementation with attention mechanisms.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from typing import Literal
from torch import nn
from prdnet.model_utils import RBFExpansion
from prdnet.utils import BaseSettings
from prdnet.features import angle_emb_mp
from prdnet.diffraction import DiffractionIntegration
from torch_scatter import scatter
from prdnet.transformer import PrdnetConv


class PrdnetConfig(BaseSettings):
    """Configuration schema for Prdnet model hyperparameters."""

    name: Literal["prdnet"]
    conv_layers: int = 6
    edge_layers: int = 0
    atom_input_features: int = 92
    edge_features: int = 256
    triplet_input_features: int = 40
    node_features: int = 256
    fc_layers: int = 2
    fc_features: int = 512
    output_features: int = 1
    node_layer_head: int = 8
    edge_layer_head: int = 4
    nn_based: bool = False

    # New optimization parameters
    dropout: float = 0.1  # Add dropout for regularization
    use_residual_connections: bool = True  # Enhanced residual connections
    use_layer_norm: bool = True  # Layer normalization
    activation: Literal["silu", "gelu", "swish"] = "silu"  # Better activation functions

    # Diffraction integration parameters
    use_diffraction: bool = True  # Enable diffraction information integration
    diffraction_max_hkl: int = 5  # Maximum HKL index value
    diffraction_num_hkl: int = 300  # Number of HKL indices to use

    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False
    use_angle: bool = False
    angle_lattice: bool = False
    classification: bool = False

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


class Prdnet(nn.Module):
    """Prdnet transformer implementation."""

    def __init__(self, config: PrdnetConfig = PrdnetConfig(name="prdnet")):
        """Set up att modules."""
        super().__init__()
        self.classification = config.classification
        self.use_angle = config.use_angle
        self.dropout = config.dropout
        self.use_residual_connections = config.use_residual_connections
        self.use_layer_norm = config.use_layer_norm
        self.use_diffraction = config.use_diffraction

        # Enhanced atom embedding with residual connection
        self.atom_embedding = nn.Sequential(
            nn.Linear(config.atom_input_features, config.node_features),
            nn.LayerNorm(config.node_features) if config.use_layer_norm else nn.Identity(),
            self._get_activation(config.activation),
            nn.Dropout(config.dropout),
            nn.Linear(config.node_features, config.node_features),
        )

        # Enhanced RBF with deeper network
        self.rbf = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=config.edge_features,
            ),
            nn.Linear(config.edge_features, config.node_features),
            nn.LayerNorm(config.node_features) if config.use_layer_norm else nn.Identity(),
            self._get_activation(config.activation),
            nn.Dropout(config.dropout),
            nn.Linear(config.node_features, config.node_features),
            nn.LayerNorm(config.node_features) if config.use_layer_norm else nn.Identity(),
            self._get_activation(config.activation),
            nn.Linear(config.node_features, config.node_features),
        )
        self.angle_lattice = config.angle_lattice
        if self.angle_lattice: ## module not used
            print('use angle lattice')
            self.lattice_rbf = nn.Sequential(
                RBFExpansion(
                    vmin=0,
                    vmax=8.0,
                    bins=config.edge_features,
                ),
                nn.Linear(config.edge_features, config.node_features),
                nn.Softplus(),
                nn.Linear(config.node_features, config.node_features)
            )

            self.lattice_angle = nn.Sequential(
                RBFExpansion(
                    vmin=-1,
                    vmax=1.0,
                    bins=config.triplet_input_features,
                ),
                nn.Linear(config.triplet_input_features, config.node_features),
                nn.Softplus(),
                nn.Linear(config.node_features, config.node_features)
            )

            self.lattice_emb = nn.Sequential(
                nn.Linear(config.node_features * 6, config.node_features),
                nn.Softplus(),
                nn.Linear(config.node_features, config.node_features)
            )

            self.lattice_atom_emb = nn.Sequential(
                nn.Linear(config.node_features * 2, config.node_features),
                nn.Softplus(),
                nn.Linear(config.node_features, config.node_features)
            )


        self.edge_init = nn.Sequential( ## module not used
            nn.Linear(3 * config.node_features, config.node_features),
            nn.Softplus(),
            nn.Linear(config.node_features, config.node_features)
        )

        self.sbf = angle_emb_mp(num_spherical=3, num_radial=40, cutoff=8.0) ## module not used

        self.angle_init_layers = nn.Sequential( ## module not used
            nn.Linear(120, config.node_features),
            nn.Softplus(),
            nn.Linear(config.node_features, config.node_features)
        )

        # Enhanced attention layers with residual connections and normalization
        self.att_layers = nn.ModuleList(
            [
                PrdnetConv(in_channels=config.node_features, out_channels=config.node_features, heads=config.node_layer_head, edge_dim=config.node_features)
                for _ in range(config.conv_layers)
            ]
        )

        # Layer normalization for each attention layer
        if config.use_layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(config.node_features) for _ in range(config.conv_layers)
            ])

        # Dropout layers
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(config.dropout) for _ in range(config.conv_layers)
        ])
        
        self.edge_update_layers = nn.ModuleList( ## module not used
            [
                PrdnetConv(in_channels=config.node_features, out_channels=config.node_features, heads=config.edge_layer_head, edge_dim=config.node_features)
                for _ in range(config.edge_layers)
            ]
        )

        # Enhanced final layers with multiple FC layers, normalization and dropout
        self.fc_layers = nn.ModuleList()
        self.fc_norms = nn.ModuleList()
        self.fc_dropouts = nn.ModuleList()

        # Build multiple FC layers based on config
        if config.fc_layers > 1:
            # First layer
            self.fc_layers.append(nn.Linear(config.node_features, config.fc_features))
            if config.use_layer_norm:
                self.fc_norms.append(nn.LayerNorm(config.fc_features))
            self.fc_dropouts.append(nn.Dropout(config.dropout))

            # Hidden layers
            for _ in range(config.fc_layers - 2):
                self.fc_layers.append(nn.Linear(config.fc_features, config.fc_features))
                if config.use_layer_norm:
                    self.fc_norms.append(nn.LayerNorm(config.fc_features))
                self.fc_dropouts.append(nn.Dropout(config.dropout))

            # Output layer
            if self.classification:
                self.fc_out = nn.Linear(config.fc_features, 2)
                self.softmax = nn.LogSoftmax(dim=1)
            else:
                self.fc_out = nn.Linear(config.fc_features, config.output_features)
        else:
            # Single layer case
            if self.classification:
                self.fc_layers.append(nn.Linear(config.node_features, config.fc_features))
                self.fc_out = nn.Linear(config.fc_features, 2)
                self.softmax = nn.LogSoftmax(dim=1)
            else:
                self.fc_out = nn.Linear(config.node_features, config.output_features)

        self.sigmoid = nn.Sigmoid()

        # Diffraction integration module
        if self.use_diffraction:
            self.diffraction_integration = DiffractionIntegration(
                node_features=config.node_features,
                graph_features=config.fc_features,
                output_features=config.fc_features,
                max_hkl=config.diffraction_max_hkl,
                num_hkl=config.diffraction_num_hkl
            )

        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x
        elif config.link == "log":
            self.link = torch.exp
            avg_gap = 0.7  # magic number -- average bandgap in dft_3d
            if not self.zero_inflated:
                self.fc_out.bias.data = torch.tensor(
                    np.log(avg_gap), dtype=torch.float
                )
        elif config.link == "logit":
            self.link = torch.sigmoid

    def _get_activation(self, activation_name):
        """Get activation function by name."""
        activations = {
            "silu": nn.SiLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),  # SiLU is the same as Swish
            "relu": nn.ReLU(),
            "softplus": nn.Softplus()
        }
        return activations.get(activation_name, nn.SiLU())

    def forward(self, data) -> torch.Tensor:
        data, ldata, lattice = data
        # initial node features: atom feature network...
            
        node_features = self.atom_embedding(data.x)
        edge_feat = torch.norm(data.edge_attr, dim=1)
        
        edge_features = self.rbf(edge_feat)
        if self.angle_lattice: ## module not used
            lattice_len = torch.norm(lattice, dim=-1) # batch * 3 * 1
            lattice_edge = self.lattice_rbf(lattice_len.view(-1)).view(-1, 3 * 128) # batch * 3 * 128
            cos1 = self.lattice_angle(torch.clamp(torch.sum(lattice[:,0,:] * lattice[:,1,:], dim=-1) / (torch.norm(lattice[:,0,:], dim=-1) * torch.norm(lattice[:,1,:], dim=-1)), -1, 1).unsqueeze(-1)).view(-1, 128)
            cos2 = self.lattice_angle(torch.clamp(torch.sum(lattice[:,0,:] * lattice[:,2,:], dim=-1) / (torch.norm(lattice[:,0,:], dim=-1) * torch.norm(lattice[:,2,:], dim=-1)), -1, 1).unsqueeze(-1)).view(-1, 128)
            cos3 = self.lattice_angle(torch.clamp(torch.sum(lattice[:,1,:] * lattice[:,2,:], dim=-1) / (torch.norm(lattice[:,1,:], dim=-1) * torch.norm(lattice[:,2,:], dim=-1)), -1, 1).unsqueeze(-1)).view(-1, 128)
            lattice_emb = self.lattice_emb(torch.cat((lattice_edge, cos1, cos2, cos3), dim=-1))
            node_features = self.lattice_atom_emb(torch.cat((node_features, lattice_emb[data.batch]), dim=-1))
        
        # Enhanced gated graph convolution layers with residual connections and normalization
        for i, att_layer in enumerate(self.att_layers):
            # Store input for residual connection
            residual = node_features

            # Apply attention layer
            node_features = att_layer(node_features, data.edge_index, edge_features)

            # Apply layer normalization if enabled
            if self.use_layer_norm and hasattr(self, 'layer_norms') and i < len(self.layer_norms):
                node_features = self.layer_norms[i](node_features)

            # Apply dropout
            if i < len(self.dropout_layers):
                node_features = self.dropout_layers[i](node_features)

            # Add residual connection if enabled
            if self.use_residual_connections and residual.shape == node_features.shape:
                node_features = node_features + residual


        # crystal-level readout
        features = scatter(node_features, data.batch, dim=0, reduce="mean")

        if self.angle_lattice:
            # features *= F.sigmoid(lattice_emb)
            features += lattice_emb

        # Integrate diffraction information if enabled
        if self.use_diffraction:
            # Get atomic positions (assuming they are available in data.pos)
            if hasattr(data, 'pos') and data.pos is not None:
                features = self.diffraction_integration(
                    graph_features=features,
                    node_features=node_features,
                    pos=data.pos,
                    batch=data.batch,
                    lattice=lattice if hasattr(self, 'lattice') else None
                )

        # Enhanced final layers
        if len(self.fc_layers) > 0:
            for i, fc_layer in enumerate(self.fc_layers):
                if i == 0:
                    features = fc_layer(features)
                else:
                    # Store input for potential residual connection
                    residual = features
                    features = fc_layer(features)

                    # Add residual connection for same-size layers
                    if self.use_residual_connections and residual.shape == features.shape:
                        features = features + residual

                # Apply normalization and activation (except for last layer)
                if i < len(self.fc_layers) - 1:
                    if self.use_layer_norm and i < len(self.fc_norms):
                        features = self.fc_norms[i](features)
                    features = self._get_activation("silu")(features)
                    if i < len(self.fc_dropouts):
                        features = self.fc_dropouts[i](features)

        out = self.fc_out(features)
        if self.link:
            out = self.link(out)
        if self.classification:
            out = self.softmax(out)

        return torch.squeeze(out)


