import numpy as np 
from scipy.fft import fft2, ifft2, fftfreq 
from scipy.ndimage import map_coordinates
import gpemod

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
    
    def phase(self):
        return np.angle(self.psi)

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
        return 0.5 * (self.omega_x**2 * X**2 + self.omega_y**2 * Y**2)

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
    
    def rot(self, psi, angle, n=2):
        """
        Angelu txikietarako biraketa
        """
        if angle < 0.01:
            psi_k  = self.grid.fft(psi)
            dx_psi = self.grid.ifft(1j * self.grid.Kx * psi_k)
            dy_psi = self.grid.ifft(1j * self.grid.Ky * psi_k)
            Lz_psi = -1j * (self.grid.X * dy_psi - self.grid.Y * dx_psi)

            num = psi - 0.5j * angle * Lz_psi
            
            psi_rot = num.copy()
            for i in range(n):
                psi_rot_k = self.grid.fft(psi_rot)
                dx_rot    = self.grid.ifft(1j * self.grid.Kx * psi_rot_k)
                dy_rot    = self.grid.ifft(1j * self.grid.Ky * psi_rot_k)
                Lz_rot    = -1j * (self.grid.X * dy_rot - self.grid.Y * dx_rot)
                psi_rot   = num - 0.5j * angle * Lz_rot

            return psi_rot
        else:
            raise ValueError(f"Angle to big {angle}, please reduce the angular speed or the time step")

    def step(self, psi, dt):
        """
        dt pausu bat egiten du, denbora irudikarian (oinarrizko egoera bilatzeko)
        ez du biraketarik  egiten,  baina  denbora  errealean  bai  egiten duela.
        """
        V = self.potential(self.grid.X, self.grid.Y)

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
    
    def gradient_descent_step(self, psi, dt):
        """
        Egingo duguna izango psi_new = psi - factor * grad(E) <-- hau deribatu funtzionala izaten
        """
        
        V = self.potential(self.grid.X, self.grid.Y)

        psi_out = np.asfortranarray(psi.copy()).astype(np.complex128)
        v = np.asfortranarray(V).astype(np.float64)
        kx = np.asfortranarray(self.grid.Kx).astype(np.float64)
        ky = np.asfortranarray(self.grid.Ky).astype(np.float64)
        k2 = np.asfortranarray(self.grid.K2).astype(np.float64)
        x = np.asfortranarray(self.grid.X).astype(np.float64)
        y = np.asfortranarray(self.grid.Y).astype(np.float64)

        gpemod.gpe_solver.gradient_descent_step(
                psi_out, v, kx, ky, k2, x, y,
                self.grid.Nx, self.grid.Ny,
                self.grid.dx, self.grid.dy,
                self.g, self.Omega, dt)

        return psi_out

    def evol(self, psi, final_time, dt, callback=None):
        steps       = int(final_time / dt)
        t           = 0.0
        psi_current = psi.copy()

        for step in range(steps):
            psi_current = self.step(psi_current, dt)
            t += dt

            if callback is not None:
                    callback(t, psi_current)

        return psi_current
    
    def gradient_descent_evol(self, psi, dt, tol=1E-6, max_iter=1000000, callback=None):
        t           = 0.0
        psi_current = psi.copy()
        energy_old  = self.energy(psi_current)

        for i in range(max_iter):
            t          += dt
            psi_current = self.gradient_descent_step(psi_current, dt)

            if callback:
                callback(t, psi_current)
            
            if i % 10 == 0:
                energy_new = self.energy(psi_current)
                rel_diff = np.abs((energy_new - energy_old) / (energy_old + 1e-10))
                
                if rel_diff < tol:
                    print(i)
                    return psi_current
                    
                energy_old = energy_new
        
        raise ValueError(f"Maximun iterations reached, the whave function does not converge. Final energy relative diference: {energy_new}")

    
    def evolcool(self, psi, dt, n_vortex=0, vortex_charges=None, positions=None, tol=1E-6, callback=None, random_seed=None, max_iter=1000000):
        psi_current = psi.copy()

        if n_vortex > 0: 
            psi_current = self.vortex_phase_mask(psi            = psi_current, 
                                                 n_vortex       = n_vortex, 
                                                 vortex_charges = vortex_charges, 
                                                 positions      = positions, 
                                                 random_seed    = random_seed
                                                 )
            
        psi = self.gradient_descent_evol(psi      = psi_current, 
                                         dt       = dt,
                                         tol      = tol,
                                         max_iter = max_iter, 
                                         callback = callback
                                         )

        return psi
    
    def vortex_phase_mask(self, psi, n_vortex, vortex_charges, positions, random_seed):
        if n_vortex == 0:
            return psi
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        if vortex_charges is None:
            vortex_charges = [1] * n_vortex
        else:
            if len(vortex_charges) < n_vortex:
                raise TypeError("LESS vortex charges than vortex")
            elif len(vortex_charges) > n_vortex:
                raise TypeError("MORE vortex charges than vortex")
        
        if positions is None:
            max_radius = min(3.0, self.grid.Lx/4)
            
            positions = []
            for i in range(n_vortex):
                r     = np.random.uniform(0.5, max_radius)
                theta = np.random.uniform(0, 2*np.pi)
                x     = r * np.cos(theta)
                y     = r * np.sin(theta)
                positions.append((x, y))
        else:
            if len(positions) < n_vortex:
                raise TypeError("ERROR: LESS positions than vortex")
            elif len(positions) > n_vortex:
                raise TypeError("MORE positions than vortex")
        
        # Aplicar todos los vórtices
        psi_with_vortices = psi.copy()

        for i, (charge, (x, y)) in enumerate(zip(vortex_charges, positions)):         
            phase = np.atan2(self.grid.Y - y, self.grid.X - x)
            
            vortex_phase = np.exp(1j * charge * phase)
            
            psi_with_vortices *= vortex_phase
        return psi_with_vortices

    def energy(self, psi):
        """
        Energia kalkulatu
        """
        psi_k  = fft2(psi)
        dx_psi = ifft2(1j * self.grid.Kx * psi_k)
        dy_psi = ifft2(1j * self.grid.Ky * psi_k)

        kin    = 0.5 * (np.abs(dx_psi)**2 + np.abs(dy_psi)**2)
        E_kin  = np.sum(kin) * self.grid.dx * self.grid.dy

        V      = self.potential(self.grid.X, self.grid.Y)
        E_pot  = np.sum(V * np.abs(psi)**2) * self.grid.dx * self.grid.dy

        E_int  = self.g * 0.5 * np.sum(np.abs(psi)**4) * self.grid.dx * self.grid.dy

        if self.Omega != 0:
            Lz_psi      = -1j * (self.grid.X * dy_psi - self.grid.Y * dx_psi)
            expectation = np.sum(np.conj(psi) * Lz_psi) * self.grid.dx * self.grid.dy
            E_rot       = -self.Omega * expectation.real
        else: 
            E_rot = 0

        return E_kin + E_pot + E_int + E_rot
    
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

    def cooling(self, dt, tol=1E-6, callback=None, max_iter=10000):
        print(self.Omega * dt/2)
        def wrapped_callback(t, psi):
            self.wf.psi = psi
            if callback:
                callback(t, self)

        self.wf.psi = self.ssfm.evolcool(psi           = self.wf.psi,
                                        dt             = dt, 
                                        n_vortex       = self.vortex, 
                                        vortex_charges = self.v_charge, 
                                        positions      = self.positions, 
                                        tol            = tol,
                                        callback       = wrapped_callback, 
                                        random_seed    = self.seed,
                                        max_iter       = max_iter
                                        )
        self.wf.normalize()
    
    def hydrodynamics(self, t_max, dt, callback=None):
        def wrapped_callback(t, psi):
            self.wf.psi = psi
            if callback:
                callback(t, self)

        self.wf.psi = self.ssfm.evol(psi        = self.wf.psi, 
                                     final_time = t_max, 
                                     dt         = dt, 
                                     callback   = wrapped_callback
                                     )