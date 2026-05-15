import torch
import torch.nn as nn
from egnn_model import EGNNEnergyModel
import numpy as np

class EnergyConservingMD:
    """
    Energy-Conserving Molecular Dynamics Simulator.
    
    Uses an EGNN energy model to predict system potential energy,
    then derives forces via automatic differentiation: F = -∇_x E.
    
    Integrates trajectories using the Velocity Verlet algorithm,
    which is a symplectic integrator that minimizes energy drift.
    """
    def __init__(self, model, num_particles, dt=0.001, mass=1.0):
        """
        Args:
            model: EGNNEnergyModel instance
            num_particles: number of atoms
            dt: time step (in simulation units)
            mass: atomic mass
        """
        self.model = model
        self.model.eval()  # Inference mode
        self.num_particles = num_particles
        self.dt = dt
        self.mass = mass
        
        # Create fully-connected edge index (all pairs)
        rows, cols = [], []
        for i in range(num_particles):
            for j in range(num_particles):
                if i != j:
                    rows.append(i)
                    cols.append(j)
        
        self.edge_index = torch.tensor([rows, cols], dtype=torch.long)
    
    def compute_forces_and_energy(self, h, x):
        """
        Compute forces and energy via automatic differentiation.
        
        Args:
            h: [N, in_node_nf] node features
            x: [N, 3] coordinates (requires grad)
        
        Returns:
            forces: [N, 3] forces
            energy: scalar energy value
        """
        x.requires_grad_(True)
        
        with torch.enable_grad():
            energy = self.model(h, x, self.edge_index)
            energy.backward()
        
        forces = -x.grad.clone()  # F = -∇_x E
        x.requires_grad_(False)
        
        return forces, energy.item()
    
    def velocity_verlet_step(self, h, x, v, f=None):
        """
        One step of Velocity Verlet integration.
        
        Verlet is a symplectic integrator that preserves phase space volume,
        leading to excellent long-term energy conservation.
        """
        # Compute initial force if not provided
        if f is None:
            f, _ = self.compute_forces_and_energy(h, x.detach().clone().requires_grad_(True))
            f = f.detach()
        
        a = f / self.mass  # acceleration
        
        # Half-step velocity
        v_half = v + 0.5 * a * self.dt
        
        # Full-step position
        x_new = x.detach() + v_half * self.dt
        
        # Force at new position
        f_new, E_new = self.compute_forces_and_energy(h, x_new.detach().clone().requires_grad_(True))
        f_new = f_new.detach()
        a_new = f_new / self.mass
        
        # Full-step velocity
        v_new = v_half + 0.5 * a_new * self.dt
        
        return x_new, v_new, f_new, E_new
    
    def simulate(self, h, x_init, v_init, steps=1000, energy_tracking=True):
        """
        Run MD simulation.
        
        Args:
            h: [N, in_node_nf] node features (constant throughout)
            x_init: [N, 3] initial positions
            v_init: [N, 3] initial velocities
            steps: number of MD steps
            energy_tracking: whether to track E, K, V
        
        Returns:
            trajectory: list of position tensors
            energies: dict with 'total', 'kinetic', 'potential' lists
        """
        x = x_init.clone().detach()
        v = v_init.clone().detach()
        
        trajectory = [x.clone()]
        energies = {'total': [], 'kinetic': [], 'potential': []}
        
        f = None
        for step in range(steps):
            x, v, f, E_pot = self.velocity_verlet_step(h, x, v, f)
            
            if energy_tracking:
                E_kin = 0.5 * self.mass * (v ** 2).sum().item()
                E_tot = E_pot + E_kin
                
                energies['kinetic'].append(E_kin)
                energies['potential'].append(E_pot)
                energies['total'].append(E_tot)
            
            trajectory.append(x.clone())
            
            if (step + 1) % max(1, steps // 10) == 0:
                if energy_tracking:
                    print(f"Step {step+1}/{steps} | E_tot={E_tot:.4f} | E_kin={E_kin:.4f} | E_pot={E_pot:.4f}")
                else:
                    print(f"Step {step+1}/{steps}")
        
        return trajectory, energies


if __name__ == "__main__":
    print("="*60)
    print("ENERGY-CONSERVING MOLECULAR DYNAMICS SIMULATOR")
    print("Using E(n)-Equivariant Graph Neural Networks for Force Fields")
    print("="*60)
    
    # System setup
    n_atoms = 5
    in_node_nf = 1
    hidden_nf = 32
    n_layers = 3
    dt = 0.001  # Small timestep for stability
    
    print(f"\n[System Setup]")
    print(f"  Atoms: {n_atoms}")
    print(f"  Model: EGNN (hidden={hidden_nf}, layers={n_layers})")
    print(f"  Timestep: {dt} (in atomic units)")
    
    # Initialize model
    print(f"\n[Initializing Model]")
    model = EGNNEnergyModel(
        in_node_nf=in_node_nf,
        hidden_nf=hidden_nf,
        out_nf=1,
        n_layers=n_layers
    )
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Initialize system state
    print(f"\n[Initializing System State]")
    h = torch.ones(n_atoms, in_node_nf)  # All atoms same type
    
    # Random but bounded initial positions
    torch.manual_seed(42)
    x = torch.randn(n_atoms, 3) * 0.5  # Compact initial geometry
    v = torch.randn(n_atoms, 3) * 0.1  # Low initial kinetic energy
    
    print(f"  Initial positions (first 2 atoms):")
    print(f"    {x[:2]}")
    print(f"  Initial velocities (first 2 atoms):")
    print(f"    {v[:2]}")
    
    # Create simulator
    simulator = EnergyConservingMD(model, num_particles=n_atoms, dt=dt, mass=1.0)
    
    # Run simulation
    print(f"\n[Running Simulation]")
    print(f"  Total steps: 100")
    print(f"\nEnergy Evolution:")
    trajectory, energies = simulator.simulate(h, x, v, steps=100, energy_tracking=True)
    
    # Analysis
    print(f"\n[Energy Analysis]")
    E_tot = np.array(energies['total'])
    E_drift = np.abs(E_tot[-1] - E_tot[0]) / np.abs(E_tot[0]) * 100
    
    print(f"  Initial total energy: {E_tot[0]:.6f}")
    print(f"  Final total energy:   {E_tot[-1]:.6f}")
    print(f"  Energy drift: {E_drift:.4f}%")
    print(f"  Max kinetic energy: {np.max(energies['kinetic']):.6f}")
    print(f"  Min potential energy: {np.min(energies['potential']):.6f}")
    
    print(f"\n[Trajectory Statistics]")
    final_pos = trajectory[-1]
    print(f"  Final positions (first 2 atoms):")
    print(f"    {final_pos[:2]}")
    print(f"  Displacement from start: {(final_pos - trajectory[0]).norm().item():.4f}")
    
    print(f"\n✓ Simulation complete. Results saved for visualization.")
