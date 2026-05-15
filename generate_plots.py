import torch
import torch.nn as nn
from egnn_model import EGNNEnergyModel
from md_simulation import EnergyConservingMD
import matplotlib.pyplot as plt
import numpy as np

def run_simulation_and_plot():
    """
    Run actual MD simulation and generate publication-quality plots
    showing REAL energy conservation from the EGNN force field.
    """
    
    print("="*70)
    print("REAL MD SIMULATION & VISUALIZATION")
    print("="*70)
    
    # ============ SYSTEM INITIALIZATION ============
    n_atoms = 6
    in_node_nf = 1
    hidden_nf = 32
    n_layers = 3
    dt = 0.0005  # Smaller timestep for better energy conservation
    
    print(f"\n[System Configuration]")
    print(f"  Number of atoms: {n_atoms}")
    print(f"  EGNN hidden dim: {hidden_nf}")
    print(f"  EGNN layers: {n_layers}")
    print(f"  Timestep: {dt}")
    
    # Initialize model
    model = EGNNEnergyModel(
        in_node_nf=in_node_nf,
        hidden_nf=hidden_nf,
        out_nf=1,
        n_layers=n_layers
    )
    
    # Initialize atoms with torch seed
    torch.manual_seed(42)
    h = torch.ones(n_atoms, in_node_nf)
    x = torch.randn(n_atoms, 3) * 0.3  # Compact initial configuration
    v = torch.randn(n_atoms, 3) * 0.2  # Initial thermal motion
    
    # Create simulator
    simulator = EnergyConservingMD(model, num_particles=n_atoms, dt=dt, mass=1.0)
    
    # ============ RUN SIMULATION ============
    print(f"\n[Running Simulation: 500 MD steps]")
    trajectory, energies = simulator.simulate(h, x, v, steps=500, energy_tracking=True)
    
    print(f"\n[Simulation Results]")
    E_tot = np.array(energies['total'])
    E_kin = np.array(energies['kinetic'])
    E_pot = np.array(energies['potential'])
    
    E_drift_pct = np.abs(E_tot[-1] - E_tot[0]) / (np.abs(E_tot[0]) + 1e-10) * 100
    
    print(f"  Initial total energy: {E_tot[0]:.6f}")
    print(f"  Final total energy:   {E_tot[-1]:.6f}")
    print(f"  Energy drift: {E_drift_pct:.4f}%")
    print(f"  Avg kinetic energy: {np.mean(E_kin):.6f}")
    print(f"  Avg potential energy: {np.mean(E_pot):.6f}")
    
    # ============ PLOT 1: ENERGY CONSERVATION ============
    print(f"\n[Generating Plot 1: Energy Conservation]")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=150)
    
    steps_array = np.arange(len(E_tot))
    
    # Top subplot: All three energies
    ax1.plot(steps_array, E_pot, label='Potential Energy', color='#1f77b4', linewidth=2, alpha=0.8)
    ax1.plot(steps_array, E_kin, label='Kinetic Energy', color='#ff7f0e', linewidth=2, alpha=0.8)
    ax1.plot(steps_array, E_tot, label='Total Energy (Hamiltonian)', color='#2ca02c', linewidth=2.5, linestyle='-', alpha=0.9)
    
    ax1.set_xlabel('Simulation Steps', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Energy (atomic units)', fontsize=12, fontweight='bold')
    ax1.set_title('Energy Conservation via Automatic Differentiation\n$F = -\\nabla_x E_{\\theta}(x)$ using Velocity Verlet', 
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_facecolor('#f8f9fa')
    
    # Bottom subplot: Energy drift (absolute deviation from initial)
    E_deviation = E_tot - E_tot[0]
    ax2.fill_between(steps_array, -np.abs(E_deviation), np.abs(E_deviation), alpha=0.3, color='#d62728', label='±Energy Deviation')
    ax2.plot(steps_array, E_deviation, color='#d62728', linewidth=2, label='Cumulative Drift')
    
    ax2.set_xlabel('Simulation Steps', fontsize=12, fontweight='bold')
    ax2.set_ylabel('ΔE (atomic units)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Energy Stability Analysis (Total Drift: {E_drift_pct:.3f}%)', 
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=1)
    ax2.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig('energy_conservation.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: energy_conservation.png")
    plt.close()
    
    # ============ PLOT 2: 3D ATOMIC TRAJECTORIES ============
    print(f"\n[Generating Plot 2: Atomic Trajectories]")
    
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert trajectory to numpy for plotting
    traj_np = [t.detach().cpu().numpy() for t in trajectory]
    
    # Color map for atoms
    colors = plt.cm.Set3(np.linspace(0, 1, n_atoms))
    
    for atom_idx in range(min(n_atoms, 6)):  # Plot max 6 atoms for clarity
        positions = np.array([t[atom_idx] for t in traj_np])
        
        # Plot trajectory
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                color=colors[atom_idx], linewidth=1.5, alpha=0.7, label=f'Atom {atom_idx+1}')
        
        # Plot starting position
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                  color=colors[atom_idx], s=100, marker='o', edgecolors='black', linewidth=1.5, zorder=5)
        
        # Plot ending position
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                  color=colors[atom_idx], s=150, marker='*', edgecolors='red', linewidth=2, zorder=5)
    
    ax.set_xlabel('X (Å)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y (Å)', fontsize=11, fontweight='bold')
    ax.set_zlabel('Z (Å)', fontsize=11, fontweight='bold')
    ax.set_title(f'EGNN-Driven N-Body Trajectories\n{n_atoms} Atoms, {len(trajectory)} MD Steps, dt={dt}', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('trajectory.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: trajectory.png")
    plt.close()
    
    # ============ PLOT 3: PHASE SPACE (E_kin vs E_pot) ============
    print(f"\n[Generating Plot 3: Phase Space Portrait]")
    
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    
    # Create scatter plot colored by step
    scatter = ax.scatter(E_pot, E_kin, c=steps_array, cmap='viridis', s=30, alpha=0.6, edgecolors='none')
    
    # Add arrows to show time direction
    for i in range(0, len(E_pot)-1, 20):
        ax.annotate('', xy=(E_pot[i+1], E_kin[i+1]), xytext=(E_pot[i], E_kin[i]),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='black', alpha=0.3))
    
    ax.set_xlabel('Potential Energy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Kinetic Energy', fontsize=12, fontweight='bold')
    ax.set_title('Phase Space Portrait: Energy Coupling\n(Closed trajectory indicates energy conservation)', 
                fontsize=13, fontweight='bold')
    
    cbar = plt.colorbar(scatter, ax=ax, label='Simulation Step')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig('phase_space.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: phase_space.png")
    plt.close()
    
    print(f"\n{'='*70}")
    print(f"✓ All visualizations complete!")
    print(f"{'='*70}\n")
    
    return energies, trajectory

if __name__ == "__main__":
    energies, trajectory = run_simulation_and_plot()
