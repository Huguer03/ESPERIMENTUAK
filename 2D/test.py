import numpy as np
import matplotlib.pyplot as plt
from becFort import Grid, TrapPotential, Simulation, ThomasFermi
import scienceplots
plt.style.use(['science'])

def test():
    beta = 3000.0
    gamma = (500.0, 500.0)
    Omega = 0.6
    tf = ThomasFermi(gamma, Omega, beta)
    N = (2**7, 2**7)
    L = (5*tf.rtf, 5*tf.rtf)
    grid = Grid(N, L)
    print(tf.rtf, L)
    vortex_charges = [1, 1, 1]
    positions = [
        (tf.rtf/5.0, 0.0),
        (0.0, -tf.rtf/5.0),
        (-tf.rtf/5.0, tf.rtf/5.0)
    ]

    sim = Simulation(grid          = grid, 
                     gamma         = gamma, 
                     beta          = beta, 
                     Omega         = Omega, 
                     n_vortex      = 3, 
                     vortex_charge = vortex_charges, 
                     positions     = positions
                     )
    
    print("Iniciando proceso de cooling (Gradient descent)...")
    
    # 4. Ejecutar el cooling
    sim.cooling(1e-6, max_iter=100000)

    print("Cooling finalizado.")
    density0 = sim.wf.density()
    print(sim.wf.norma())

    # 5. Vamos a simular la hidrodinamica
    sim.hydrodynamics(t_max=1.0,dt=1e-5)
    density5 = sim.wf.density()
    print(5)

    # 6. Visualización de resultados
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    zoom_region = [-1.5*tf.rtf, 1.5*tf.rtf, -1.5*tf.rtf, 1.5*tf.rtf]

    im0 = ax[0].imshow(density0, extent=[-L[0]/2, L[0]/2, -L[1]/2, L[1]/2], cmap='inferno')
    ax[0].set_title(r"$|\Psi|^2 (t=0s)$")
    ax[0].axis(zoom_region)
    fig.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(density5, extent=[-L[0]/2, L[0]/2, -L[1]/2, L[1]/2], cmap='inferno')
    ax[1].set_title(r"$|\Psi|^2 (t=5s)$")
    ax[1].axis(zoom_region)
    fig.colorbar(im1, ax=ax[1])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test()