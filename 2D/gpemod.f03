! para compilar: python -m numpy.f2py -c gpemod.f03 -m gpe_solver -L$CONDA_PREFIX/lib -lfftw3 --f90flags="-I$CONDA_PREFIX/include"

module gpe_solver
    use iso_c_binding
    implicit none

    include 'fftw3.f03'

    complex(8), parameter :: zi = (0.0d0, 1.0d0)
    type(C_PTR), private  :: plan_foreward_1d_x, plan_backward_1d_x
    type(C_PTR), private  :: plan_foreward_1d_y, plan_backward_1d_y
    type(C_PTR), private  :: plan_foreward_2d, plan_backward_2d

contains

    subroutine fftw_create_plans_1d(nx, ny)
        integer, intent(in)       :: nx,ny
        complex(8), allocatable   :: psi(:,:)
        integer :: n(1)

        allocate(psi(nx,ny))
        n(1) = nx
        plan_foreward_1d_x = fftw_plan_many_dft(1, n, ny, &
            psi, n, 1, nx, psi, n, 1, nx, FFTW_FORWARD, FFTW_MEASURE)
        plan_backward_1d_x = fftw_plan_many_dft(1, n, ny, &
            psi, n, 1, nx, psi, n, 1, nx, FFTW_BACKWARD, FFTW_MEASURE)

        n(1) = ny
        plan_foreward_1d_y = fftw_plan_many_dft(1, n, nx, &
            psi, n, nx, 1, psi, n, nx, 1, FFTW_FORWARD, FFTW_MEASURE)
        plan_backward_1d_y = fftw_plan_many_dft(1, n, nx, &
            psi, n, nx, 1, psi, n, nx, 1, FFTW_BACKWARD, FFTW_MEASURE)
        deallocate(psi)
    end subroutine fftw_create_plans_1d

    subroutine fftw_create_plans_2d(nx, ny)
        integer, intent(in)     :: nx,ny
        complex(8), allocatable :: psi(:,:)

        allocate(psi(nx,ny))
        plan_foreward_2d  = fftw_plan_dft_2d(nx, ny, psi, psi, FFTW_FORWARD, FFTW_MEASURE)
        plan_backward_2d = fftw_plan_dft_2d(nx, ny, psi, psi, FFTW_BACKWARD, FFTW_MEASURE)
        deallocate(psi)
    end subroutine fftw_create_plans_2d

    subroutine fftw_create_plans(nx, ny, dim)
        integer, intent(in) :: nx,ny,dim

        if (dim == 1) then
            call fftw_create_plans_1d(nx, ny)
        elseif (dim == 2) then
            call fftw_create_plans_2d(nx, ny)
        else 
            call fftw_create_plans_1d(nx, ny)
            call fftw_create_plans_2d(nx, ny)
        end if
    end subroutine fftw_create_plans

    subroutine fftw_destroy_plans(dim)
        integer, intent(in) :: dim

        if ( dim == 1 ) then
            call fftw_destroy_plan(plan_foreward_1d_x)
            call fftw_destroy_plan(plan_backward_1d_x)
            call fftw_destroy_plan(plan_foreward_1d_y)
            call fftw_destroy_plan(plan_backward_1d_y)
        elseif ( dim == 2) then
            call fftw_destroy_plan(plan_foreward_2d)
            call fftw_destroy_plan(plan_backward_2d)
        else 
            call fftw_destroy_plan(plan_foreward_1d_x)
            call fftw_destroy_plan(plan_backward_1d_x)
            call fftw_destroy_plan(plan_foreward_1d_y)
            call fftw_destroy_plan(plan_backward_1d_y)
            call fftw_destroy_plan(plan_foreward_2d)
            call fftw_destroy_plan(plan_backward_2d)
        end if
    end subroutine fftw_destroy_plans

    subroutine fft(psi, nx, ny, axis, direction)
        complex(8), intent(inout) :: psi(nx, ny)
        integer, intent(in)       :: nx, ny, direction, axis

        if (axis == 1) then 
            if (direction == 1) then
                call fftw_execute_dft(plan_foreward_1d_x, psi, psi)
            else
                call fftw_execute_dft(plan_backward_1d_x, psi, psi)
                psi = psi / real(nx, 8)
            end if
        else if (axis == 2) then
            if (direction == 1) then
                call fftw_execute_dft(plan_foreward_1d_y, psi, psi)
            else
                call fftw_execute_dft(plan_backward_1d_y, psi, psi)
                psi = psi / real(ny, 8)
            end if
        end if
    end subroutine fft

    subroutine fft2(psi, nx, ny, direction)
        complex(8), intent(inout) :: psi(nx, ny)
        integer, intent(in)       :: nx, ny, direction

        if (direction == 1) then
            call fftw_execute_dft(plan_foreward_2d, psi, psi)
        else
            call fftw_execute_dft(plan_backward_2d, psi, psi)
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

        call fftw_create_plans(nx, ny, 2)

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

        call fftw_destroy_plans(2)
    end subroutine gradient_descent_evol 

    subroutine rot(psi, angle, kx, ky, x, y, nx, ny)
        complex(8), intent(inout) :: psi(nx,ny)
        real(8), intent(in)       :: kx(nx,ny), ky(nx,ny)
        real(8), intent(in)       :: x(nx,ny), y(nx,ny)
        real(8), intent(in)       :: angle
        integer, intent(in)       :: nx, ny

        real(8) :: alpha, beta
        integer :: j, i
        real(8) :: kx_vec(nx), ky_vec(ny)
        real(8) :: x_vec(nx), y_vec(ny)

        kx_vec = kx(:, 1)
        ky_vec = ky(1, :)
        x_vec  = x(:, 1)
        y_vec  = y(1, :)

        alpha = -tan(angle / 2.0d0)
        beta  = sin(angle)

        call fft(psi, nx, ny, 1, 1)
        do j = 1, ny
            psi(:, j) = psi(:, j) * exp(-zi * kx_vec * (alpha * y_vec(j)))
        end do
        call fft(psi, nx, ny, 1, -1)

        call fft(psi, nx, ny, 2, 1)
        do i = 1, nx
            psi(i, :) = psi(i, :) * exp(-zi * ky_vec * (beta * x_vec(i)))
        end do
        call fft(psi, nx, ny, 2, -1)

        call fft(psi, nx, ny, 1, 1)
        do j = 1, ny
            psi(:, j) = psi(:, j) * exp(-zi * kx_vec * (alpha * y_vec(j)))
        end do
        call fft(psi, nx, ny, 1, -1)
    end subroutine rot

    subroutine ssfm_step(psi, v, kx, ky, k2, x, y, nx, ny, &
                                    g, omega, dt)
        complex(8), intent(inout) :: psi(nx,ny)
        real(8), intent(in)       :: v(nx,ny)
        real(8), intent(in)       :: x(nx,ny), y(nx,ny)
        real(8), intent(in)       :: kx(nx,ny), ky(nx,ny)
        real(8), intent(in)       :: k2(nx,ny)
        real(8), intent(in)       :: dt, g, omega
        integer, intent(in)       :: nx, ny

        real(8)    :: angle

        angle = omega * dt

        call fft2(psi, nx, ny, 1)
        psi = exp(-0.5d0 * zi * k2 * dt/2.0d0) * psi
        call fft2(psi, nx, ny, -1)

        psi   = exp(-zi * (v + g * abs(psi)**2) * dt/2.0d0) * psi

        if (omega /= 0.0d0) then
            call rot(psi, angle, kx, ky, x, y, nx, ny)
        end if

        psi = exp(-zi * (v + g * abs(psi)**2) * dt/2.0d0) * psi

        call fft2(psi, nx, ny, 1)
        psi = exp(-0.5d0 * zi * k2 * dt/2.0d0) * psi
        call fft2(psi, nx, ny, -1)
    end subroutine ssfm_step

    subroutine ssfm_evol(psi, v, kx, ky, k2, x, y, nx, ny,&
                                    g, omega, final_time, dt)
        complex(8), intent(inout) :: psi(nx,ny)
        real(8), intent(in)       :: v(nx,ny)
        real(8), intent(in)       :: x(nx,ny), y(nx,ny)
        real(8), intent(in)       :: kx(nx,ny), ky(nx,ny)
        real(8), intent(in)       :: k2(nx,ny)
        real(8), intent(in)       :: dt, g, omega, final_time
        integer, intent(in)       :: nx, ny

        integer :: i, steps

        call fftw_create_plans(nx, ny, 0)
        
        steps = int(final_time / dt)

        do i = 1,steps
            call ssfm_step(psi, v, kx, ky, k2, x, y, nx, ny, &
                                    g, omega, dt)
        enddo
        
        call fftw_destroy_plans(0)
    end subroutine ssfm_evol

end module gpe_solver