import numpy as np
import matplotlib.pyplot as plt
from becFort import Grid, TrapPotential, Simulation

def test():
    # 1. Configuración de la malla (Grid)
    # N: número de puntos, L: tamaño de la caja
    dt = 0.001
    N = (256, 256)
    L = (40.0, 40.0)
    grid = Grid(N, L)
    vortex_charges = [1, 1, 1, 1]
    positions = [
        (2.0, 0.0),   
        (-2.0, 0.0),
        (0.0, 2.0),
        (0.0, -2.0)
    ]

    # 2. Definir el potencial (Trampa armónica)
    # omega_x = 1.0, omega_y = 1.0 (trampa simétrica)
    potential = TrapPotential(omega=(1.0, 1.5))

    # 3. Crear la simulación
    sim = Simulation(grid          = grid, 
                     potential     = potential, 
                     g             = 500.0, 
                     Omega         = 0.9, 
                     n_vortex      = 20, 
                     vortex_charge = None, 
                     positions     = None
                     )
    
    print("Iniciando proceso de cooling (Gradient descent)...")
    
    # 4. Ejecutar el cooling
    sim.cooling(dt, max_iter=100000)

    print("Cooling finalizado.")
    density0 = sim.wf.density()

    # 5. Vamos a simular la hidrodinamica
    sim.hydrodynamics(5.0,dt=dt)
    density5 = sim.wf.density()
    print(5)

    sim.hydrodynamics(5.0,dt=dt)
    density10 = sim.wf.density()
    print(10)

    sim.hydrodynamics(5.0,dt=dt)
    density15 = sim.wf.density()
    print(15)

    sim.hydrodynamics(5.0,dt=dt)
    density20 = sim.wf.density()
    print(20)

    sim.hydrodynamics(5.0,dt=dt)
    density25 = sim.wf.density()
    print(25)

    # 6. Visualización de resultados
    fig, ax = plt.subplots(2, 3, figsize=(12, 5))
    zoom_region = [-8, 8, -8, 8]

    im0 = ax[0,0].imshow(density0, extent=[-L[0]/2, L[0]/2, -L[1]/2, L[1]/2])
    ax[0,0].set_title(r"$|\Psi|^2 (t=0s)$")
    ax[0,0].axis(zoom_region)
    fig.colorbar(im0, ax=ax[0,0])

    im1 = ax[0,1].imshow(density5, extent=[-L[0]/2, L[0]/2, -L[1]/2, L[1]/2])
    ax[0,1].set_title(r"$|\Psi|^2 (t=5s)$")
    ax[0,1].axis(zoom_region)
    fig.colorbar(im1, ax=ax[0,1])

    im2 = ax[0,2].imshow(density10, extent=[-L[0]/2, L[0]/2, -L[1]/2, L[1]/2])
    ax[0,2].set_title(r"$|\Psi|^2 (t=10s)$")
    ax[0,2].axis(zoom_region)
    fig.colorbar(im2, ax=ax[0,2])

    im3 = ax[1,0].imshow(density15, extent=[-L[0]/2, L[0]/2, -L[1]/2, L[1]/2])
    ax[1,0].set_title(r"$|\Psi|^2 (t=15s)$")
    ax[1,0].axis(zoom_region)
    fig.colorbar(im3, ax=ax[1,0])

    im4 = ax[1,1].imshow(density20, extent=[-L[0]/2, L[0]/2, -L[1]/2, L[1]/2])
    ax[1,1].set_title(r"$|\Psi|^2 (t=20s)$")
    ax[1,1].axis(zoom_region)
    fig.colorbar(im4, ax=ax[1,1])

    im5 = ax[1,2].imshow(density25, extent=[-L[0]/2, L[0]/2, -L[1]/2, L[1]/2])
    ax[1,2].set_title(r"$|\Psi|^2 (t=25s)$")
    ax[1,2].axis(zoom_region)
    fig.colorbar(im5, ax=ax[1,2])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test()