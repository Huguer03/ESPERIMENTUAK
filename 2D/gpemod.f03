module gpe_solver
    
    implicit none

    complex(8), parameter :: zi = (0.0d0, 1.0d0)

contains

    subroutine fft2(psi, nx, ny, direction)
        complex(8), intent(inout) :: psi(nx, ny)
        integer, intent(in) :: nx, ny
        integer, intent(in) :: direction
        
        real(8) :: work(2*nx*ny + 2*nx + 2*ny + 30)
        real(8) :: rwork(2*nx*ny)
        integer :: i, j, k
        
        do j = 1, ny
            do i = 1, nx
                k = 2*((j-1)*nx + i-1) + 1
                rwork(k)   = real(psi(i, j))
                rwork(k+1) = aimag(psi(i, j))
            end do
        end do
        
        if (direction == 1) then
            call zfft2f(0, nx, ny, rwork, nx, work, 2*nx*ny, work(2*nx*ny+1), 2*(nx+ny))
        else
            call zfft2b(0, nx, ny, rwork, nx, work, 2*nx*ny, work(2*nx*ny+1), 2*(nx+ny))
            rwork = rwork / (nx * ny)
        end if
        
        do j = 1, ny
            do i = 1, nx
                k = 2*((j-1)*nx + i-1) + 1
                psi(i, j) = cmplx(rwork(k), rwork(k+1), kind=8)
            end do
        end do
        
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

        !FFT
        psi_k = psi
        call fft2(psi_k, nx, ny, 1)

        temp = -k2 * psi_k
        call fft2(temp, nx, ny, -1)

        if (omega /= 0.0d0) then
            dx_psi = zi * kx * psi_k
            dy_psi = zi * ky * psi_k

            call fft2(dx_psi, nx, ny, -1)
            call fft2(dy_psi, nx, ny, -1)
        end if

        grad = -0.5d0 * temp + v * psi + g * abs(psi)**2 * psi

        if (omega /= 0.0d0) then
            grad = grad - zi * (x * dy_psi - y * dx_psi)
        end if

        psi = psi - dt * grad

        norm = sum(abs(psi)**2) * dx * dy
        psi = psi / sqrt(norm)
    end subroutine gradient_descent_step

    function energy(psi, v, kx, ky, k2, x, y, nx, ny, dx, dy,&
                                     g, omega) result(E)
        complex(8), intent(in) :: psi(nx, ny)
        real(8), intent(in) :: v(nx, ny), kx(nx, ny), ky(nx, ny), k2(nx, ny)
        real(8), intent(in) :: x(nx, ny), y(nx, ny)
        integer, intent(in) :: nx, ny
        real(8), intent(in) :: dx, dy, g, omega
        
        complex(8) :: psi_k(nx, ny)
        complex(8) :: dx_psi(nx, ny), dy_psi(nx, ny)
        complex(8) :: temp_c(nx, ny)
        real(8) :: temp_r(nx,ny)
        real(8) :: E_kin, E_pot, E_g, E_rot, E

        psi_k = psi
        call fft2(psi_k, nx, ny, 1)

        dx_psi = zi * kx * psi_k
        dy_psi = zi * ky * psi_k
        call fft2(dx_psi, nx, ny, -1)
        call fft2(dy_psi, nx, ny, -1)

        temp_r = (abs(dx_psi)**2 + abs(dy_psi)**2)
        E_kin  = 0.5d0 * sum(temp_r) * dx * dy

        E_pot = sum(psi2 * v) * dx * dy




    end function energy
    
end module gpe_solver