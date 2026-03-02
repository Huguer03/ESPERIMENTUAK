import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from bec2d import Grid, TrapPotential, Simulation

def test():
    # Configuración de la malla
    N = (128, 128)
    L = (25.0, 25.0)
    grid = Grid(N, L)
    
    # Vórtices
    vortex_charges = [1, 1]
    positions = [(2.0, 0.0), (-2.0, 0.0)]

    # Potencial
    potential = TrapPotential(omega=(0.8, 0.85))

    # Simulación
    sim = Simulation(
        grid=grid, 
        potential=potential, 
        g=200.0, 
        Omega=0.4, 
        n_vortex=2, 
        vortex_charge=vortex_charges, 
        positions=positions
    )
    
    # Cooling
    sim.cooling(dt=0.005, converge=1e-8)

    print("Enfriamiento terminado")
    
    # Listas para guardar evolución
    times = []
    densities = []
    
    def callback(t, sim):
        times.append(t)
        densities.append(sim.wf.density().copy())
    
    # Hidrodinámica
    sim.hydrodynamics(5.0, dt=0.001, callback=callback)

    print("Terminada la simulacion")
    
    # ANIMACIÓN
    fig, ax = plt.subplots(figsize=(6, 6))
    
    im = ax.imshow(densities[0], extent=[-8, 8, -8, 8], 
                   cmap='viridis', vmin=0, vmax=np.max(densities))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_title('Evolución del Condensado')
    plt.colorbar(im, ax=ax, label='Densidad')
    
    def animate(i):
        im.set_array(densities[i])
        ax.set_title(f't = {times[i]:.2f}')
        return [im]
    
    ani = FuncAnimation(fig, animate, frames=len(times), interval=50, blit=True)
    plt.tight_layout()
    # guardar animación
    #ani.save('condensado.gif', writer='pillow', fps=20, dpi=150)
    plt.show()

if __name__ == "__main__":
    test()