import numpy as np 
from scipy.fft import fft2, ifft2, fftfreq 
from scipy.ndimage import map_coordinates

class Grid:
    """
    Definituko dugu espazioaren diskretizazioa
    """
    def __init__(self, N, L):
        """
        N:   Zenbat puntutan banatuko dugu, tupla (Nx, Ny)
        L:   Kaxaren luzera, tupla (Lx, Ly) eta [-Li/2,Li/2]
        """
        self.Nx = N[0]
        self.Ny = N[1]
        self.N  = N[0]*N[1]
        self.Lx = L[0]
        self.Ly = L[1]
        self.dx = self.Lx/self.Nx
        self.dy = self.Ly/self.Ny

        # Espazio erreala
        self.x = np.linspace(-self.Lx/2, self.Lx/2, self.Nx, endpoint=False)
        self.y = np.linspace(-self.Ly/2, self.Ly/2, self.Ny, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        # k espazioa lortu
        self.kx = 2.0 * np.pi * fftfreq(self.Nx, self.dx)
        self.ky = 2.0 * np.pi * fftfreq(self.Ny, self.dy)
        self.Kx, self.Ky = np.meshgrid(self.kx, self.ky, indexing='ij')
        self.K2 = self.Kx**2 + self.Ky**2

        self.mesh_shape = (self.Nx, self.Ny)

    def fft(self, psi):
        return fft2(psi)

    def ifft(self, psi):
        return ifft2(psi)

    def laplacian(self, psi_k):
        """
        Laplazearra k espazioan: -|K|^2 * psi_k
        """
        return - self.K2 * psi_k

class WaveFunction:
    """
    Uhin funtzioaren logika 2D-tan.
    """
    def __init__(self, grid, sigma=(1.0,1.0), psi=None):
        self.grid         = grid
        self.sigma_x      = sigma[0]
        self.sigma_y      = sigma[1]
        if psi == None:
            # Perfil Gausstarra emango diogu
            psi = np.exp(-0.5 * ((self.grid.X / self.sigma_x)**2 + (self.grid.Y / self.sigma_y)**2), dtype=complex)

        self.psi = psi
        self.normalize()

    def normalize(self, A=1.0):
        """
        Normalizatuko dugu uhin funtzioa. A normalizazio helburua da.
        """
        norma = np.sum(np.abs(self.psi)**2) * self.grid.dx * self.grid.dy 
        self.psi *= np.sqrt(A / norma)

    def density(self):
        """
        Dentsitatea bueltatzen du |psi|^2
        """
        return np.abs(self.psi)**2 

class TrapPotential:
    """
    Erabiliko dugu potentzial harmonikoa
    """
    def __init__(self, omega):
        """
        omega: tupla bat da non (omega_x, omega_y)
        """
        self.omega_x, self.omega_y = omega

    def __call__(self, X, Y):
        return 0.5 * (self.omega_x * X**2 + self.omega_y * Y**2)

class SSFM:
    """
    Bigarren  ordeneko   splip-step  Fourier   metodoa 
    erabiliko dugu GPE ekuazioa biraketarekin ebazteko.
    """
    def __init__(self, grid, potential, g=0, Omega=0):
        self.grid      = grid
        self.potential = potential
        self.g         = g
        self.Omega     = Omega
    
    def rot(self, psi, angle):
        X, Y = self.grid.X, self.grid.Y
    
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # Rotación de coordenadas ANTIHORARIA (corregida)
        X_rot =  X * cos_a - Y * sin_a  # Cambio aquí: -Y * sin_a
        Y_rot =  X * sin_a + Y * cos_a  # Cambio aquí: +X * sin_a
        
        # Convertir a índices para interpolación
        I = (X_rot + self.grid.Lx/2) / self.grid.dx
        J = (Y_rot + self.grid.Ly/2) / self.grid.dy
        
        # Asegurar que los índices están dentro de los límites
        I = np.clip(I, 0, self.grid.Nx - 1)
        J = np.clip(J, 0, self.grid.Ny - 1)
        
        coords = np.array([I.ravel(), J.ravel()])
        
        # Interpolación
        psi_real = map_coordinates(psi.real, coords, order=3, mode='wrap').reshape(psi.shape)
        psi_imag = map_coordinates(psi.imag, coords, order=3, mode='wrap').reshape(psi.shape)
        psi_rotated = psi_real + 1j * psi_imag
        return psi_rotated

    def step(self, psi, dt, imag=False):
        """
        dt pausu bat egiten du, denbora irudikarian (oinarrizko egoera bilatzeko)
        ez du biraketarik  egiten,  baina  denbora  errealean  bai  egiten duela.
        """
        V = self.potential(self.grid.X, self.grid.Y)

        if imag:
            # Calcular el potencial químico aproximado
            # Energía cinética en espacio k
            psi_k = self.grid.fft(psi)
            kin_energy = 0.5 * self.grid.ifft(self.grid.K2 * psi_k)
            
            # Energía potencial + interacción
            pot_energy = (V + 0.5 * self.g * np.abs(psi)**2) * psi
            
            # Potencial químico (promedio)
            mu = np.real(np.sum(np.conj(psi) * (kin_energy + pot_energy)) * self.grid.dx * self.grid.dy)
            
            # Propagación en tiempo imaginario CON resta de μ
            # Medio paso en espacio real
            psi1 = np.exp(-(V + self.g * np.abs(psi)**2 - mu) * dt/2) * psi
            
            # Paso en espacio k
            psi_k = self.grid.fft(psi1)
            psi_k *= np.exp(-0.5 * self.grid.K2 * dt)
            psi2 = self.grid.ifft(psi_k)
            
            # Segundo medio paso
            psi_new = np.exp(-(V + self.g * np.abs(psi2)**2 - mu) * dt/2) * psi2
            
            # Normalizar (importante en tiempo imaginario)
            norm = np.sum(np.abs(psi_new)**2) * self.grid.dx * self.grid.dy
            psi_new /= np.sqrt(norm)
            
            return psi_new

        else:
            # Potentzialean erdipausua
            psi1 = np.exp(-1j * (V + self.g * np.abs(psi)**2) * dt/2) * psi

            # Biraketa egotekotan, erdi pausua sartu
            if self.Omega != 0:
                psi2 = self.rot(psi1, self.Omega * dt/2)
            else:
                psi2 = psi1

            # Pausua K espazioan
            psi_k  = self.grid.fft(psi2)
            psi_k *= np.exp(-0.5j * self.grid.K2 * dt)
            psi3   = self.grid.ifft(psi_k)

            # Biraketa egotekotan, erdi pausua sartu
            if self.Omega != 0:
                psi4 = self.rot(psi3, self.Omega * dt/2)
            else:
                psi4 = psi3

            # Pausua berriz Potentzialean
            psi_new = np.exp(-1j * (V + self.g * np.abs(psi4)**2) * dt/2) * psi4
            return psi_new

    def evol(self, psi, final_time, dt, imag=False, callback=None):
        steps       = int(final_time / dt)
        t           = 0.0
        psi_current = psi.copy()

        for step in range(steps):
            psi_current = self.step(psi_current, dt, imag=imag)
            t += dt

            if callback is not None:
                    callback(t, psi_current)

        return psi_current
    
    def evolcool(self, psi, dt, n_vortex, vortex_charges=None, positions=None, converge=1E-10, imag=True, callback=None, random_seed=None):
        t           = 0.0
        psi_current = psi.copy()
        psi_previus = np.zeros(self.grid.mesh_shape, dtype=complex)
            
        while not self.check_convergence(psi_previus, psi_current, converge):
            psi_previus = psi_current.copy()
            psi_current = self.step(psi_current, dt, imag=imag)
            t += dt

            if callback is not None:
                    callback(t, psi_current)

        psi_current  = self.vortex_phase_mask(psi_current, n_vortex, vortex_charges, positions, random_seed) 
        psi_current /= np.sqrt(np.sum(np.abs(psi_current)**2) * self.grid.dx * self.grid.dy)

        return psi_current
    
    def vortex_phase_mask(self, psi, n_vortex, vortex_charges, positions, random_seed):
        if n_vortex == 0:
            return psi
        
        # Establecer semilla aleatoria si se proporciona
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Definir cargas de los vórtices
        if vortex_charges is None:
            # Por defecto: todos carga +1
            vortex_charges = [1] * n_vortex
        else:
            # Asegurar que tenemos suficientes cargas
            if len(vortex_charges) < n_vortex:
                raise TypeError("LESS vortex charges than vortex")
            elif len(vortex_charges) > n_vortex:
                raise TypeError("MORE vortex charges than vortex")
        
        # Definir posiciones de los vórtices
        if positions is None:
            # Generar posiciones aleatorias dentro de un radio
            max_radius = min(3.0, self.grid.Lx/4)  # Radio máximo para evitar bordes
            
            positions = []
            for i in range(n_vortex):
                # Distribución uniforme en círculo
                r     = np.random.uniform(0.5, max_radius)  # Evitar el centro exacto
                theta = np.random.uniform(0, 2*np.pi)
                x     = r * np.cos(theta)
                y     = r * np.sin(theta)
                positions.append((x, y))
        else:
            # Usar posiciones proporcionadas
            if len(positions) < n_vortex:
                raise TypeError("ERROR: LESS positions than vortex")
            elif len(positions) > n_vortex:
                raise TypeError("MORE positions than vortex")
        
        # Aplicar todos los vórtices
        psi_with_vortices = psi.copy()

        for i, (charge, (x, y)) in enumerate(zip(vortex_charges, positions)):         
            # Calcular la fase del vórtice en cada punto de la malla
            phase = np.atan2(self.grid.Y - y, self.grid.X - x)
            
            # Crear la máscara de fase con la carga correcta
            vortex_phase = np.exp(1j * charge * phase)
            
            # Multiplicar la función de onda por la fase de este vórtice
            psi_with_vortices *= vortex_phase
        return psi_with_vortices

    def energy(self, psi):
        """
        Energia kalkulatu
        """
        psi_k  = fft2(psi)
        dx_psi = ifft2(1j * self.grid.Kx * psi_k)
        dy_psi = ifft2(1j * self.grid.Ky * psi_k)

        kin   = 0.5 * (np.abs(dx_psi)**2 + np.abs(dy_psi)**2)
        E_kin = np.sum(kin) * self.grid.dx * self.grid.dy
        V     = self.potential(self.grid.X, self.grid.Y)
        E_pot = np.sum(V * np.abs(psi)**2) * self.grid.dx * self.grid.dy
        E_int = (self.g / 2.0) * np.sum(np.abs(psi)**4) * self.grid.dx * self.grid.dy

        Lz_psi      = -1j * (self.grid.X * dy_psi - self.grid.Y * dx_psi)
        expectation = np.sum(np.conj(psi) * Lz_psi) * self.grid.dx * self.grid.dy
        E_rot       = -self.Omega * expectation.real

        return E_kin + E_pot + E_int + E_rot

    def check_convergence(self, psi_previus, psi_current, converge):
        E1 = self.energy(psi_current)
        E0 = self.energy(psi_previus)
        if E0 == 0: return False
        rel_diff = np.abs((E1 - E0) / E0)
        return rel_diff < converge

class Simulation:
    """
    Simulazioa manipulatzeko erabiliko duguna
    """
    def __init__(self, grid, potential, g=0, Omega=0, n_vortex=0, vortex_charge=None, positions=None, sigma=(1.0,1.0), psi=None, seed=None):
        self.grid      = grid
        self.potential = potential
        self.vortex    = n_vortex
        self.v_charge  = vortex_charge
        self.positions = positions
        self.g         = g
        self.Omega     = Omega
        self.seed      = seed
        self.wf        = WaveFunction(grid, sigma, psi)
        self.ssfm      = SSFM(grid, potential, g, Omega)

    def cooling(self, dt, converge=1E-10, callback=None):
        def wrapped_callback(t, psi):
            self.wf.psi = psi
            if callback:
                callback(t, self)

        self.wf.psi = self.ssfm.evolcool(self.wf.psi, dt, n_vortex=self.vortex, vortex_charges=self.v_charge, positions=self.positions, converge=converge, imag=True,callback=wrapped_callback, random_seed=self.seed)
        self.wf.normalize()
    
    def hydrodynamics(self, t_max, dt, callback=None):
        def wrapped_callback(t, psi):
            self.wf.psi = psi
            if callback:
                callback(t, self)

        self.wf.psi = self.ssfm.evol(self.wf.psi, t_max, dt, imag=False, callback=wrapped_callback)

