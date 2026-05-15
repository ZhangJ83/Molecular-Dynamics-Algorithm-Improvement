import torch
import torch.nn as nn

class EquivariantMessagePassing(nn.Module):
    """E(n)-Equivariant Message Passing Layer"""
    def __init__(self, hidden_nf, edge_attr_dim=0):
        super(EquivariantMessagePassing, self).__init__()
        # Message MLP: processes node features and distances
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_nf * 2 + 1 + edge_attr_dim, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf)
        )
        
        # Radial basis for distance-dependent functions
        self.radial_mlp = nn.Sequential(
            nn.Linear(1, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, 1)
        )
        
        # Node update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf * 2, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf)
        )

    def forward(self, h, x, edge_index, edge_attr=None):
        """
        h: node features [N, hidden_nf]
        x: coordinates [N, 3]
        edge_index: [2, num_edges]
        """
        row, col = edge_index
        coord_diff = x[row] - x[col]  # [num_edges, 3]
        radial_sq = torch.sum(coord_diff ** 2, dim=1, keepdim=True)  # [num_edges, 1]
        
        # Edge features
        edge_input = [h[row], h[col], radial_sq]
        if edge_attr is not None:
            edge_input.append(edge_attr)
        edge_feat = torch.cat(edge_input, dim=1)
        
        # Message
        m_ij = self.edge_mlp(edge_feat)  # [num_edges, hidden_nf]
        
        # Radial basis weight (for equivariance)
        radial_weight = self.radial_mlp(radial_sq)  # [num_edges, 1]
        
        # Aggregate messages
        m_i = torch.zeros(h.size(0), m_ij.size(1), device=h.device, dtype=h.dtype)
        m_i.index_add_(0, row, m_ij)
        
        # Node update
        h_out = self.node_mlp(torch.cat([h, m_i], dim=1))
        
        return h_out, radial_weight, coord_diff

class EGNNEnergyModel(nn.Module):
    """
    E(n)-Equivariant Graph Neural Network that predicts ENERGY (not forces).
    Forces are computed via automatic differentiation: F = -∇_x E
    
    This ensures strict energy conservation in MD simulations.
    """
    def __init__(self, in_node_nf=1, hidden_nf=64, out_nf=1, n_layers=4, edge_attr_dim=0):
        super(EGNNEnergyModel, self).__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        
        # Input embedding
        self.embedding_in = nn.Linear(in_node_nf, hidden_nf)
        
        # Equivariant message passing layers
        self.layers = nn.ModuleList([
            EquivariantMessagePassing(hidden_nf, edge_attr_dim)
            for _ in range(n_layers)
        ])
        
        # Energy output head (scalar)
        self.energy_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, out_nf)
        )

    def forward(self, h, x, edge_index, edge_attr=None):
        """
        Args:
            h: [N, in_node_nf] - node features (e.g., atomic numbers one-hot)
            x: [N, 3] - atomic coordinates (MUST require_grad=True for force computation)
            edge_index: [2, num_edges] - graph connectivity
            edge_attr: [num_edges, edge_attr_dim] - edge features (optional)
        
        Returns:
            energy: [] - total system energy (scalar)
        """
        h = self.embedding_in(h)
        
        for layer in self.layers:
            h, radial_weight, coord_diff = layer(h, x, edge_index, edge_attr)
        
        # Readout: sum node representations to get system property
        # (invariant to permutation)
        energy = self.energy_mlp(h).sum()
        
        return energy
    
    def compute_forces(self, h, x, edge_index, edge_attr=None):
        """
        Compute forces via automatic differentiation.
        F_i = -∇_{x_i} E
        
        Args:
            h: node features
            x: coordinates with requires_grad=True
            edge_index: graph connectivity
            edge_attr: optional edge features
        
        Returns:
            forces: [N, 3] - forces on each atom
        """
        x.requires_grad_(True)
        
        if x.grad is not None:
            x.grad.zero_()
        
        energy = self.forward(h, x, edge_index, edge_attr)
        energy.backward()
        
        forces = -x.grad.clone()  # F = -∇E
        x.requires_grad_(False)
        
        return forces
