import numpy as np
import matplotlib.pyplot as plt
from becFort import Grid, TrapPotential, Simulation

def test_vortex_central():
    dt = 0.001
    N = (256, 256)
    L = (20.0, 20.0)
    grid = Grid(N, L)

    potential = TrapPotential(omega=(1.0, 1.0))

    sim = Simulation(
        grid          = grid, 
        potential     = potential, 
        g             = 500.0, 
        Omega         = 0.0, 
        n_vortex      = 1, 
        vortex_charge = [1], 
        positions     = [(0.0, 0.0)] 
    )
    
    print("Relajando el condensado con un vórtice central s=1...")
    sim.cooling(dt, max_iter=50000) 

    density_sim1 = sim.wf.density() 

    mid_idx = N[1] // 2
    x_axis = grid.x
    profile_s1 = density_sim1[:, mid_idx]

    sim = Simulation(
        grid          = grid, 
        potential     = potential, 
        g             = 500.0, 
        Omega         = 0.0, 
        n_vortex      = 0, 
        vortex_charge = None, 
        positions     = None 
    )

    sim.cooling(dt, max_iter=50000) 

    density_sim = sim.wf.density()

    print("calculando TF")
    mu = np.max(density_sim) * 500.0

    V = potential(grid.X, grid.Y)
    density_tf = (mu - V) / 500.0
    density_tf[density_tf < 0] = 0 
    
    mid_idx = N[1] // 2
    x_axis = grid.x
    profile_sim = density_sim[:, mid_idx]
    profile_tf = density_tf[:, mid_idx]

    plt.figure(figsize=(10, 6))
    zoom = [-6, 6]

    plt.plot(grid.x, profile_s1, label="Carga topológica s=1", color='blue', linewidth=2)
    plt.plot(grid.x, profile_tf, linestyle='--', label="Aproximación Thomas-Fermi", color='black', linewidth=1)

    plt.axvline(0, color='red', linestyle=':', alpha=0.5)
    
    plt.xlabel("x")
    plt.ylabel("Densidad")
    plt.xlim(zoom) 
    
    plt.title("Comparación de perfiles de vórtice según su carga")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_vortex_central()