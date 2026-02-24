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
        dim: Dimentsioa (W.I.P)
        """
        self.Nx = N[0]
        self.Ny = N[1]
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

    def fft(self, psi):
        return fft2(psi)

    def ifft(self, psi):
        return ifft2(psi)

    def laplacian(self, psi_k):
        """
        Laplazearra k espazioan: -|K|^2 * psi_k
        """
        return self.K2**2 * psi_k

class WaveFunction:
    """
    Uhin funtzioaren logika 2D-tan.
    """
    def __init__(self, grid, psi=None):
        self.grid = grid
        if psi == None:
            # Perfil Gausstarra emango diogu
            sigma_x = 1.0
            sigma_y = 1.0

            psi = np.exp(-0.5 * ((self.grid.X / sigma_x)**2 + (self.grid.Y / sigma_y)**2), dtype=complex)

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
        """
        Uhin-funtzioaren biraketa egiten du. psi'(r) = psi(R^-1 r)
        """
        X, Y = self.grid.X, self.grid.Y

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        X_rot =  X * cos_a + Y * sin_a
        Y_rot = -X * sin_a + Y * cos_a

        coords   = np.array([X_rot.ravel(), Y_rot.ravel()])
        psi_flat = map_coordinates(psi.real, coords, order=1, mode='wrap') \
                 + 1j * map_coordinates(psi.imag, coords, order=1, mode='wrap')

        return psi_flat.reshape(psi.shape)

    def step(self, psi, dt, imag=False):
        """
        dt pausu bat egiten du, denbora irudikarian (oinarrizko egoera bilatzeko)
        ez du biraketarik  egiten,  baina  denbora  errealean  bai  egiten duela.
        """
        V = self.potential(self.grid.X, self.grid.Y)

        if imag:
            # Potentzialean erdipausua
            psi1 = np.exp(-(V * self.g * np.abs(psi)**2) * dt/2) * psi

            # Pausua K espazioan
            psi_k  = self.grid.fft(psi1)
            psi_k *= np.exp(-0.5 * self.grid.K2 * dt)
            psi2   = self.grid.ifft(psi_k)

            # Pausua berriz Potentzialean
            psi_new = np.exp(-(V * self.g * np.abs(psi2)**2) * dt/2) * psi2

            norma    = np.sum(np.abs(psi_new)**2) * self.grid.dx * self.grid.dy 
            psi_new /= np.sqrt(norma)
            return psi_new

        else:
            # Potentzialean erdipausua
            psi1 = np.exp(-1j * (V * self.g * np.abs(psi)**2) * dt/2) * psi

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
            psi_new = np.exp(-1j * (V * self.g * np.abs(psi4)**2) * dt/2) * psi2
            return psi_new

    def evol(self, psi, final_time, dt, imag=False, callback=None):
        steps       = int(final_time / dt)
        t           = 0.0
        psi_current = psi.copy()

        for step in range(steps):
            psi_current = self.step(psi, dt, imag=imag)
            t += dt

            if callback is not None:
                    callback(t, psi_current)

        return psi_current

class Simulation:
    """
    Simulazioa manipulatzeko erabiliko duguna
    """
    def __init__(self, grid, potential, g=0, Omega=0):
        self.grid      = grid
        self.potential = potential
        self.g         = g
        self.Omega     = Omega
        self.wf        = WaveFunction(grid)
        self.ssfm      = SSFM(grid, potential, g, Omega)

    def cooling(self, tau_max, dt, callback=None):
        def wrapped_callback(t, psi):
            self.wf.psi = psi
            if callback:
                callback(t, self)

        self.wf.psi = self.ssfm.evol(self.wf.psi, tau_max, dt, imag=True, callback=wrapped_callback)
        self.wf.normalize()
    
    def hydrodynamics(self, t_max, dt, callback=None):
        def wrapped_callback(t, psi):
            self.wf.psi = psi
            if callback:
                callback(t, self)

        self.wf.psi = self.ssfm.evol(self.wf.psi, t_max, dt, imag=False, callback=wrapped_callback)

