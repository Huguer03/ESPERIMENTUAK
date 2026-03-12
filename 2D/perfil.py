import numpy as np
import matplotlib.pyplot as plt
from becFort import Grid, TrapPotential, Simulation


def test_comparacion_tf():
    dt = 0.001
    N = (256, 256)
    L = (40.0, 40.0)
    grid = Grid(N, L)
    
    g_param = 500.0
    omega_trap = (1.0, 1.0) 
    potential = TrapPotential(omega=omega_trap)

    sim = Simulation(
        grid=grid, 
        potential=potential, 
        g=g_param, 
        Omega=0.0,
        n_vortex=0,
        seed=42
    )
    
    print("Calculando estado fundamental...")
    sim.cooling(dt, max_iter=80000)
    
    density_sim = sim.wf.density()
    
    mu = np.max(density_sim) * g_param
    
    V = potential(grid.X, grid.Y)
    density_tf = (mu - V) / g_param
    density_tf[density_tf < 0] = 0
    
    mid_idx = N[1] // 2
    x_axis = grid.x
    profile_sim = density_sim[:, mid_idx]
    profile_tf = density_tf[:, mid_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    zoom = [-10, 10]

    im = ax1.imshow(density_sim.T, extent=[-L[0]/2, L[0]/2, -L[1]/2, L[1]/2], 
                    origin='lower', cmap='viridis')
    ax1.set_title("Densidad GPE (Simulación)")
    ax1.set_xlim(zoom); ax1.set_ylim(zoom)
    fig.colorbar(im, ax=ax1)

    ax2.plot(x_axis, profile_sim, label="GPE (Simulación)", color='blue', alpha=0.8)
    ax2.plot(x_axis, profile_tf, '--', label="Thomas-Fermi (Teórico)", color='black', linewidth=2)
    
    ax2.set_title(f"Sección en Y=0 (g={g_param})")
    ax2.set_xlabel("x")
    ax2.set_ylabel(r"$|\Psi|^2$")
    ax2.set_xlim(zoom)
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_comparacion_tf()