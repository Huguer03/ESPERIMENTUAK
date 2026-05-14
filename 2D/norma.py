import numpy as np
import matplotlib.pyplot as plt
from becFort import Grid, TrapPotential, Simulation
import scienceplots
plt.style.use(['science'])

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
    potential = TrapPotential(omega=(1.0, 1.0))

    # 3. Crear la simulación
    sim = Simulation(grid          = grid, 
                     potential     = potential, 
                     g             = 3000.0, 
                     Omega         = 0.999, 
                     n_vortex      = 0, 
                     vortex_charge = None, 
                     positions     = None
                     )
    
    print("Iniciando proceso de cooling (Gradient descent)...")
    
    # 4. Ejecutar el cooling
    sim.cooling(dt, max_iter=100000)
    t = [0,5,10,15,20,25]
    norma = []

    print("Cooling finalizado.")
    density0 = sim.wf.density()
    norma.append(sim.wf.norma())

    # 5. Vamos a simular la hidrodinamica
    sim.hydrodynamics(1.0,dt=dt)
    density_sim = sim.wf.density()
    norma.append(sim.wf.norma())
    print(5)

    sim.hydrodynamics(1.0,dt=dt)
    density_sim = sim.wf.density()
    norma.append(sim.wf.norma())
    print(10)

    sim.hydrodynamics(1.0,dt=dt)
    density_sim = sim.wf.density()
    norma.append(sim.wf.norma())
    print(15)

    sim.hydrodynamics(1.0,dt=dt)
    density_sim = sim.wf.density()
    norma.append(sim.wf.norma())
    print(20)

    sim.hydrodynamics(1.0,dt=dt)
    density_sim = sim.wf.density()
    norma.append(sim.wf.norma())
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