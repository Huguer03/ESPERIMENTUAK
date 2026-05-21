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
    t = [0,5,10,15,20,25]
    norma = []

    print("Cooling finalizado.")
    density0 = sim.wf.density()
    norma.append(1-sim.wf.norma())

    # 5. Vamos a simular la hidrodinamica
    sim.hydrodynamics(1.0,dt=1e-5)
    density_sim = sim.wf.density()
    norma.append(1-sim.wf.norma())
    print(5)

    sim.hydrodynamics(1.0,dt=1e-5)
    density_sim = sim.wf.density()
    norma.append(1-sim.wf.norma())
    print(10)

    sim.hydrodynamics(1.0,dt=1e-5)
    density_sim = sim.wf.density()
    norma.append(1-sim.wf.norma())
    print(15)

    sim.hydrodynamics(1.0,dt=1e-5)
    density_sim = sim.wf.density()
    norma.append(1-sim.wf.norma())
    print(20)

    sim.hydrodynamics(1.0,dt=1e-5)
    density_sim = sim.wf.density()
    norma.append(1-sim.wf.norma())
    print(25)

    # 6. Visualización de resultados
    plt.figure(figsize=(8, 6))
    plt.plot(t, norma, label='norma') 
    plt.xlabel(r'$t(s)$')
    plt.ylabel(r'$Norma$')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    test()