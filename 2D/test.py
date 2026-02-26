import numpy as np
import matplotlib.pyplot as plt
from bec2d import Grid, TrapPotential, Simulation

def test_vortex_final():
    # Configuración ÓPTIMA para vórtices
    N = (128, 128)
    L = (30.0, 30.0)  # Caja suficientemente grande
    grid = Grid(N, L)
    
    # Potencial con asimetría (10% es suficiente)
    potential = TrapPotential(omega=(1.0, 1.1))
    
    # Parámetros críticos
    g = 800.0      # Interacción fuerte (necesaria para vórtices)
    Omega = 0.7     # Velocidad de rotación
    n_vortex = 1    # Número de vórtices a sembrar
    sigma = (4.0, 4.0)  # Gaussiana ancha
    
    print("="*60)
    print("CREANDO SIMULACIÓN CON VÓRTICE")
    print("="*60)
    print(f"Grid: {N[0]}x{N[1]}, L={L}")
    print(f"Potencial: ω=({potential.omega_x:.2f}, {potential.omega_y:.2f})")
    print(f"g={g}, Ω={Omega}, n_vortex={n_vortex}")
    
    # Crear simulación
    sim = Simulation(grid, potential, g=g, Omega=Omega, 
                     n_vortex=n_vortex, sigma=sigma)
    
    # =========================================================
    # VERIFICACIÓN DEL ESTADO INICIAL
    # =========================================================
    print("\n" + "="*60)
    print("ESTADO INICIAL")
    print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Densidad inicial
    im1 = axes[0,0].imshow(sim.wf.density().T, origin='lower', 
                           extent=[-L[0]/2, L[0]/2, -L[1]/2, L[1]/2])
    axes[0,0].set_title('Densidad Inicial')
    plt.colorbar(im1, ax=axes[0,0])
    
    # Fase inicial
    phase0 = np.angle(sim.wf.psi)
    im2 = axes[0,1].imshow(phase0.T, origin='lower', cmap='twilight',
                           extent=[-L[0]/2, L[0]/2, -L[1]/2, L[1]/2])
    axes[0,1].set_title('Fase Inicial')
    plt.colorbar(im2, ax=axes[0,1])
    
    # Verificar winding number
    theta = np.arctan2(grid.Y, grid.X)
    winding = np.abs(np.sum(np.exp(1j * (phase0 - theta))) / grid.N)
    axes[0,2].text(0.1, 0.5, f'Winding = {winding:.3f}\n(Debería ser 1.0)', 
                  fontsize=14, transform=axes[0,2].transAxes)
    axes[0,2].axis('off')
    
    # Perfil radial inicial
    r = np.sqrt(grid.X**2 + grid.Y**2)
    density0 = sim.wf.density()
    r_bins = np.linspace(0, 15, 50)
    radial0 = [np.mean(density0[(r >= r_bins[i]) & (r < r_bins[i+1])]) 
               for i in range(len(r_bins)-1)]
    
    axes[1,0].plot(r_bins[:-1], radial0, 'b-', linewidth=2)
    axes[1,0].set_xlabel('r')
    axes[1,0].set_ylabel('Densidad promedio')
    axes[1,0].set_title('Perfil Radial Inicial')
    axes[1,0].grid(True)
    
    # Zoom al centro (fase)
    zoom = 20
    center = N[0]//2
    im3 = axes[1,1].imshow(phase0[center-zoom:center+zoom, center-zoom:center+zoom].T,
                           origin='lower', cmap='twilight')
    axes[1,1].set_title('Zoom centro (fase inicial)')
    plt.colorbar(im3, ax=axes[1,1])
    
    axes[1,2].axis('off')
    
    plt.suptitle('ESTADO INICIAL - Verificar que la fase tiene espiral')
    plt.tight_layout()
    plt.show()
    
    # =========================================================
    # COOLING (TIEMPO IMAGINARIO)
    # =========================================================
    print("\n" + "="*60)
    print("EJECUTANDO COOLING (TIEMPO IMAGINARIO)")
    print("="*60)
    
    # Para guardar evolución
    densities = []
    phases = []
    times = []
    windings = []
    
    def cooling_callback(t, sim_state):
        densities.append(sim_state.wf.density().copy())
        phases.append(np.angle(sim_state.wf.psi).copy())
        times.append(t)
        
        # Calcular winding durante evolución
        phase_t = np.angle(sim_state.wf.psi)
        winding_t = np.abs(np.sum(np.exp(1j * (phase_t - theta))) / grid.N)
        windings.append(winding_t)
        
        if len(times) % 20 == 0:
            max_dens = np.max(sim_state.wf.density())
            print(f"  t={t:.3f}, max|ψ|²={max_dens:.3f}, winding={winding_t:.3f}")
    
    # Cooling con paso pequeño
    sim.cooling(dt=0.002, converge=1e-5, callback=cooling_callback)
    
    print("\n" + "="*60)
    print("RESULTADO FINAL DEL COOLING")
    print("="*60)
    
    # =========================================================
    # VISUALIZACIÓN DE LA EVOLUCIÓN
    # =========================================================
    n_frames = min(16, len(densities))
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    
    for i in range(n_frames):
        idx = i * len(densities) // n_frames
        ax = axes[i//4, i%4]
        
        # Mostrar fase con contorno de densidad
        phase_frame = phases[idx]
        density_frame = densities[idx]
        
        ax.imshow(phase_frame.T, origin='lower', cmap='twilight')
        ax.contour(density_frame.T, levels=[0.5*np.max(density_frame)], 
                  colors='white', linewidths=1)
        ax.set_title(f't={times[idx]:.2f}\nw={windings[idx]:.2f}')
        ax.axis('off')
    
    plt.suptitle('Evolución de la fase durante cooling (blanco = 50% densidad máxima)')
    plt.tight_layout()
    plt.show()
    
    # =========================================================
    # RESULTADO FINAL DETALLADO
    # =========================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Densidad final
    density_final = sim.wf.density()
    im1 = axes[0,0].imshow(density_final.T, origin='lower', cmap='viridis')
    axes[0,0].set_title('Densidad Final')
    plt.colorbar(im1, ax=axes[0,0])
    
    # Fase final
    phase_final = np.angle(sim.wf.psi)
    im2 = axes[0,1].imshow(phase_final.T, origin='lower', cmap='twilight')
    axes[0,1].set_title('Fase Final')
    plt.colorbar(im2, ax=axes[0,1])
    
    # Winding final
    winding_final = np.abs(np.sum(np.exp(1j * (phase_final - theta))) / grid.N)
    axes[0,2].text(0.1, 0.5, f'Winding final = {winding_final:.3f}', 
                  fontsize=14, transform=axes[0,2].transAxes)
    axes[0,2].axis('off')
    
    # Perfil radial final
    radial_final = [np.mean(density_final[(r >= r_bins[i]) & (r < r_bins[i+1])]) 
                    for i in range(len(r_bins)-1)]
    
    axes[1,0].plot(r_bins[:-1], radial_final, 'b-', linewidth=2, label='Final')
    axes[1,0].plot(r_bins[:-1], radial0, 'r--', linewidth=2, label='Inicial')
    axes[1,0].set_xlabel('r')
    axes[1,0].set_ylabel('Densidad promedio')
    axes[1,0].set_title('Perfil Radial')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Zoom al centro (densidad)
    zoom = 20
    im3 = axes[1,1].imshow(density_final[center-zoom:center+zoom, center-zoom:center+zoom].T,
                           origin='lower', cmap='viridis')
    axes[1,1].set_title('Zoom centro (densidad)')
    plt.colorbar(im3, ax=axes[1,1])
    
    # Zoom al centro (fase)
    im4 = axes[1,2].imshow(phase_final[center-zoom:center+zoom, center-zoom:center+zoom].T,
                           origin='lower', cmap='twilight')
    axes[1,2].set_title('Zoom centro (fase)')
    plt.colorbar(im4, ax=axes[1,2])
    
    plt.suptitle('RESULTADO FINAL - ¿Hay vórtice?')
    plt.tight_layout()
    plt.show()
    
    # =========================================================
    # DIAGNÓSTICO FINAL
    # =========================================================
    center_dens = density_final[center, center]
    max_dens = np.max(density_final)
    ratio = center_dens / max_dens
    
    print("\n" + "="*60)
    print("DIAGNÓSTICO FINAL")
    print("="*60)
    print(f"Densidad en centro / máximo = {ratio:.4f}")
    print(f"Winding number final = {winding_final:.4f}")
    
    if ratio < 0.5 and winding_final > 0.9:
        print("\n✅ ¡VÓRTICE DETECTADO!")
        print("   La densidad tiene un hueco en el centro")
        print("   La fase tiene la topología correcta")
    elif ratio < 0.5:
        print("\n⚠️  Hay hueco en densidad pero la fase no es correcta")
    elif winding_final > 0.9:
        print("\n⚠️  La fase es correcta pero no hay hueco en densidad")
    else:
        print("\n❌ NO HAY VÓRTICE")
        print("   Posibles problemas:")
        print("   - Aumentar g (interacción)")
        print("   - Ajustar Omega (rotación)")
        print("   - Aumentar asimetría del potencial")
        print("   - Reducir dt en cooling")
    
    return sim

# EJECUTAR
if __name__ == "__main__":
    sim = test_vortex_final()
    
    # Opcional: evolución en tiempo real para ver dinámica
    print("\n" + "="*60)
    print("¿Ejecutar hidrodinámica? (10 segundos de simulación)")
    print("="*60)
    respuesta = input("¿Continuar? (s/n): ")
    
    if respuesta.lower() == 's':
        print("Ejecutando hidrodinámica...")
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        
        def hydro_callback(t, sim_state):
            if int(t*10) % 10 == 0:  # Cada 0.1 unidades de tiempo
                idx = int(t * 2)  # Para guardar 20 frames
                if idx < 10:
                    ax1 = axes[0, idx]
                    ax2 = axes[1, idx]
                    
                    ax1.imshow(sim_state.wf.density().T, origin='lower', cmap='viridis')
                    ax1.set_title(f't={t:.2f}')
                    ax1.axis('off')
                    
                    ax2.imshow(np.angle(sim_state.wf.psi).T, origin='lower', cmap='twilight')
                    ax2.axis('off')
        
        sim.hydrodynamics(t_max=5.0, dt=0.01, callback=hydro_callback)
        plt.suptitle('Evolución en tiempo real')
        plt.tight_layout()
        plt.show()