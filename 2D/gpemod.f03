! para compilar: python -m numpy.f2py -c gpemod.f03 -m gpe_solver -L$CONDA_PREFIX/lib -lfftw3 --f90flags="-I$CONDA_PREFIX/include"

module gpe_solver
    use iso_c_binding
    implicit none

    include 'fftw3.f03'

    complex(8), parameter :: zi = (0.0d0, 1.0d0)
    type(C_PTR), private  :: plan_forward, plan_backward
    logical, private      :: plans_created = .false.

contains

    subroutine create_fft_plans(nx, ny)
        integer, intent(in)     :: nx, ny
        complex(8), allocatable :: temp(:,:)
        
        allocate(temp(nx, ny))
        plan_forward = fftw_plan_dft_2d(nx, ny, temp, temp, FFTW_FORWARD, FFTW_MEASURE)
        plan_backward = fftw_plan_dft_2d(nx, ny, temp, temp, FFTW_BACKWARD, FFTW_MEASURE)
        deallocate(temp)
        plans_created = .true.
    end subroutine create_fft_plans

    subroutine destroy_fft_plans()
        if (plans_created) then
            call fftw_destroy_plan(plan_forward)
            call fftw_destroy_plan(plan_backward)
            plans_created = .false.
        end if
    end subroutine destroy_fft_plans

    subroutine fft2(psi, nx, ny, direction)
        complex(8), intent(inout) :: psi(nx, ny)
        integer, intent(in)       :: nx, ny, direction
        
        if (.not. plans_created) call create_fft_plans(nx, ny)
        
        if (direction == 1) then
            call fftw_execute_dft(plan_forward, psi, psi)
        else
            call fftw_execute_dft(plan_backward, psi, psi)
            psi = psi / real(nx * ny, 8)
        end if
    end subroutine fft2

    subroutine gradient_descent_step(psi, v, kx, ky, k2, x, y, nx, ny, dx, dy, &
                                    g, omega, dt)
        complex(8), intent(inout) :: psi(nx, ny)
        real(8), intent(in) :: v(nx, ny), kx(nx, ny), ky(nx, ny), k2(nx, ny)
        real(8), intent(in) :: x(nx, ny), y(nx, ny)
        integer, intent(in) :: nx, ny
        real(8), intent(in) :: dx, dy, g, omega, dt
        
        complex(8) :: psi_k(nx, ny), grad(nx, ny)
        complex(8) :: dx_psi(nx, ny), dy_psi(nx, ny)
        complex(8) :: temp(nx, ny)
        real(8) :: norm

        psi_k = psi
        call fft2(psi_k, nx, ny, 1)
        dx_psi = zi * kx * psi_k
        dy_psi = zi * ky * psi_k

        temp = -k2 * psi_k
        call fft2(temp, nx, ny, -1)

        grad = -0.5d0 * temp + v * psi + g * abs(psi)**2 * psi

        if (omega /= 0.0d0) then
            call fft2(dx_psi, nx, ny, -1)
            call fft2(dy_psi, nx, ny, -1)

            grad = grad - zi * omega * (x * dy_psi - y * dx_psi)
        end if

        psi = psi - dt * grad 

        norm = sum(abs(psi)**2) * dx * dy
        psi = psi / sqrt(norm)
    end subroutine gradient_descent_step

    function energy(psi, v, kx, ky, x, y, nx, ny, dx, dy,&
                                     g, omega) result(E)
        complex(8), intent(in) :: psi(nx, ny)
        real(8), intent(in) :: v(nx, ny), kx(nx, ny), ky(nx, ny)
        real(8), intent(in) :: x(nx, ny), y(nx, ny)
        integer, intent(in) :: nx, ny
        real(8), intent(in) :: dx, dy, g, omega
        
        complex(8) :: psi_k(nx, ny)
        complex(8) :: dx_psi(nx, ny), dy_psi(nx, ny)
        complex(8) :: Lz_psi(nx,ny)
        complex(8) :: expextation
        real(8) :: laplacian(nx,ny)
        real(8) :: E_kin, E_pot, E_g, E_rot, E

        psi_k = psi
        call fft2(psi_k, nx, ny, 1)

        dx_psi = zi * kx * psi_k
        dy_psi = zi * ky * psi_k
        call fft2(dx_psi, nx, ny, -1)
        call fft2(dy_psi, nx, ny, -1)

        laplacian = (abs(dx_psi)**2 + abs(dy_psi)**2)
        E_kin     = 0.5d0 * sum(laplacian) * dx * dy

        E_pot = sum(abs(psi)**2 * v) * dx * dy

        E_g = 0.5d0 * g * sum(abs(psi)**4) * dx * dy

        E = E_kin + E_pot + E_g

        if (omega /= 0.0d0) then
            Lz_psi = -zi * (x * dy_psi - y * dx_psi)
            expextation = sum(conjg(psi) * Lz_psi) * dx * dy
            E_rot       = -omega * real(expextation)
        else 
            E_rot = 0.0d0
        end if

        E = E + E_rot
    end function energy

    subroutine gradient_descent_evol(psi, v, kx, ky, k2, x, y, nx, ny, dx, dy, &
                                    g, omega, dt, max_iter, tol, converge)
        complex(8), intent(inout) :: psi(nx, ny)
        real(8), intent(in)  :: v(nx, ny), kx(nx, ny), ky(nx, ny), k2(nx, ny)
        real(8), intent(in)  :: x(nx, ny), y(nx, ny)
        integer, intent(in)  :: nx, ny
        real(8), intent(in)  :: dx, dy, g, omega, dt
        logical, intent(out) :: converge
        integer, optional    :: max_iter
        real(8), optional    :: tol

        real(8)    :: E_rel, E_new, E_old
        integer    :: i

        if (.not. present(max_iter)) max_iter = 100000
        if (.not. present(tol)) tol = 1e-6

        E_old = energy(psi, v, kx, ky, x, y, nx, ny, dx, dy, g, omega)

        do i = 1, max_iter
            call gradient_descent_step(psi, v, kx, ky, k2, x, y, nx, ny,&
                                         dx, dy, g, omega, dt)
            if ( modulo(i,10) == 0 ) then
                E_new = energy(psi, v, kx, ky, x, y, nx, ny, dx, dy, g, omega)
                E_rel = abs(E_new - E_old) / E_old
                E_old = E_new
                if ( E_rel < tol ) then
                    print*, "Iterations needed for convergence: ", i
                    converge = .true.
                    return
                end if
            end if
        end do

        if ( i == max_iter ) then
            converge = .false.
        end if

        call destroy_fft_plans()
    end subroutine gradient_descent_evol 
    
end module gpe_solver