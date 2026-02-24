import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from bec2d import Grid, TrapPotential, Simulation

def run_simulation(Omega, N=128, L=20, omega_trap=(0.1, 0.1), g=1000, cooling_time=50, t_max=100, dt=0.01):
    """
    Ejecuta una simulación completa para un valor dado de Omega
    """
    # Crear grid
    grid = Grid((N, N), (L, L))
    
    # Crear potencial
    potential = TrapPotential(omega_trap)
    
    # Crear simulación
    sim = Simulation(grid, potential, g=g, Omega=Omega)
    
    print(f"\n--- Simulación con Omega = {Omega} ---")
    
    # Cooling (obtener estado fundamental)
    print("  Cooling en progreso...")
    sim.cooling(cooling_time, dt)
    print("  Cooling completado")
    
    # Evolución en tiempo real (para desarrollar vórtices si Omega > 0)
    print("  Evolución en tiempo real...")
    sim.hydrodynamics(t_max, dt)
    print("  Evolución completada")
    
    return sim

def plot_results(sim_0, sim_07, sim_10):
    """
    Crea una figura con tres paneles mostrando las densidades
    """
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)
    
    # Panel 1: Omega = 0
    ax1 = fig.add_subplot(gs[0, 0])
    density1 = sim_0.wf.density()
    im1 = ax1.imshow(density1.T, origin='lower', extent=[-sim_0.grid.Lx/2, sim_0.grid.Lx/2, -sim_0.grid.Ly/2, sim_0.grid.Ly/2],cmap='viridis')
    ax1.set_title(f'$\Omega = 0$')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(im1, ax=ax1, label='$|\psi|^2$')
    
    # Panel 2: Omega = 0.7
    ax2 = fig.add_subplot(gs[0, 1])
    density2 = sim_07.wf.density()
    im2 = ax2.imshow(density2.T, origin='lower', extent=[-sim_07.grid.Lx/2, sim_07.grid.Lx/2, -sim_07.grid.Ly/2, sim_07.grid.Ly/2],cmap='viridis')
    ax2.set_title(f'$\Omega = 0.7$')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(im2, ax=ax2, label='$|\psi|^2$')
    
    # Panel 3: Omega = 1.0
    ax3 = fig.add_subplot(gs[0, 2])
    density3 = sim_10.wf.density()
    im3 = ax3.imshow(density3.T, origin='lower', extent=[-sim_10.grid.Lx/2, sim_10.grid.Lx/2, -sim_10.grid.Ly/2, sim_10.grid.Ly/2],cmap='viridis')
    ax3.set_title(f'$\Omega = 1.0$')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    plt.colorbar(im3, ax=ax3, label='$|\psi|^2$')
    
    plt.suptitle('Evolución de vórtices en condensado de Bose-Einstein en rotación', fontsize=14)
    plt.tight_layout()
    plt.savefig('vortex_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Función principal que ejecuta las tres simulaciones
    """
    # Parámetros ajustados para observar vórtices
    params = {
        'N': 256,                    # Resolución más alta para mejor visualización
        'L': 30,                      # Tamaño del dominio
        'omega_trap': (0.05, 0.05),    # Frecuencia de la trampa (más suave)
        'g': 2000,                     # No linealidad fuerte
        'cooling_time': 100,            # Tiempo de enfriamiento
        't_max': 200,                   # Tiempo de evolución para vórtices
        'dt': 0.005                     # Paso de tiempo pequeño para estabilidad
    }
    
    print("Iniciando simulaciones...")
    print("Este proceso puede tomar varios minutos...")
    
    # Ejecutar las tres simulaciones
    sim_0 = run_simulation(Omega=0.0, **params)
    sim_07 = run_simulation(Omega=0.7, **params)
    sim_10 = run_simulation(Omega=1.0, **params)
    
    # Mostrar resultados
    plot_results(sim_0, sim_07, sim_10)
    
    print("\n¡Simulaciones completadas!")
    print("Las imágenes han sido guardadas como 'vortex_comparison.png'")

if __name__ == "__main__":
    main()
