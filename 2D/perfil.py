import numpy as np
import matplotlib.pyplot as plt
from becFort import Grid, TrapPotential, Simulation


def test_comparacion_tf():
    dt = 0.001
    N = (256, 256)
    L = (40.0, 40.0)
    grid = Grid(N, L)

    sim = Simulation(grid          = grid, 
                     gamma         = (1.0, 1.0), 
                     beta          = 1000.0, 
                     Omega         = 0.0, 
                     n_vortex      = 0, 
                     vortex_charge = None, 
                     positions     = None
                     )
    
    print("Calculando estado fundamental...")
    sim.cooling(dt, max_iter=80000)
    
    density_sim = sim.wf.density()

    potential = sim.potential

    V = potential(grid.X, grid.Y)
    density_tf = (sim.mutf - V) / sim.beta
    density_tf[density_tf < 0] = 0
    
    mid_idx = N[1] // 2
    x_axis = grid.x
    profile_sim = density_sim[:, mid_idx]
    profile_tf = density_tf[:, mid_idx]

    diff = np.abs(profile_tf - profile_sim)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    zoom = [0, (sim.rtf + 1.5)]

    ax1.plot(x_axis, diff, color='red', alpha=0.8)
    ax1.set_title(r"Errore erlatiboa")
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_xlim([0, sim.rtf])

    ax2.plot(x_axis, profile_sim, label="Simulazioa", color='blue', alpha=0.8)
    ax2.plot(x_axis, profile_tf, '--', label="Thomas-Fermi (teorikoa)", color='black', linewidth=2)
    
    ax2.set_title(fr"$Y=0$ ebaketa ($\beta={sim.beta}$)")
    ax2.set_xlabel(r"$x$")
    ax2.set_ylabel(r"$|\Psi|^2$")
    ax2.set_xlim(zoom)
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_comparacion_tf()