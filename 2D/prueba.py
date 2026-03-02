import numpy as np
import matplotlib.pyplot as plt
from bec2d import Grid, TrapPotential, Simulation

def test():
    # 1. Configuración de la malla (Grid)
    # N: número de puntos, L: tamaño de la caja
    N = (256, 256)
    L = (40.0, 40.0)
    grid = Grid(N, L)
    vortex_charges = [1, -1]
    positions = [
        (2.0, 0.0),   # Vórtice +1
        (-2.0, 0.0),  # Vórtice +1
    ]

    # 2. Definir el potencial (Trampa armónica)
    # omega_x = 1.0, omega_y = 1.0 (trampa simétrica)
    potential = TrapPotential(omega=(0.8, 0.85))

    # 3. Crear la simulación
    sim = Simulation(grid=grid, 
                     potential=potential, 
                     g=200.0, 
                     Omega=0.7, 
                     n_vortex=2, 
                     vortex_charge=vortex_charges, 
                     positions=positions
                     )
    
    print("Iniciando proceso de cooling (Tiempo Imaginario)...")
    
    # 4. Ejecutar el cooling
    # tau_max: tiempo total de evolución imaginaria
    # dt: paso de tiempo (debe ser pequeño para estabilidad)
    sim.cooling(dt=0.01)

    initial_density = sim.wf.density().copy()

    print("Cooling finalizado.")

    # 5. Vamos a simular la hidrodinamica
    sim.hydrodynamics(25.0,dt=0.01)

    # 6. Visualización de resultados
    final_density = sim.wf.density()

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Gráfico Estado Inicial
    im0 = ax[0].imshow(initial_density, extent=[-4, 4, -4, 4])
    ax[0].set_title("Densidad Inicial (Gaussiana)")
    fig.colorbar(im0, ax=ax[0])

    # Gráfico Estado Final (tras cooling)
    im1 = ax[1].imshow(final_density, extent=[-4, 4, -4, 4])
    ax[1].set_title("Estado Rotado")
    fig.colorbar(im1, ax=ax[1])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test()