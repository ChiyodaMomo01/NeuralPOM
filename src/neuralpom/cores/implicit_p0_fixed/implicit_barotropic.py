import math
import numpy as np
try:
    import netCDF4 as nc
except ModuleNotFoundError:
    nc = None
import torch
import torch.nn.functional as F


torch.set_default_dtype(torch.float32)


class ImplicitBarotropicMixin:
    def apply_elliptic_operator(self, eta, d_2d):
        """
        Apply L(eta) = eta - 4*dti^2*g*alpha*div(H*grad(eta)) on interior T-points (Arakawa C).
        Fluxes multiplied by dum/dvm for zero-flux at land boundaries.
        """
        s_i = slice(1, self.imm1)
        s_j = slice(1, self.jmm1)
        s_i_p1 = slice(2, self.imm1 + 1)
        s_i_m1 = slice(0, self.imm1 - 1)
        s_j_p1 = slice(2, self.jmm1 + 1)
        s_j_m1 = slice(0, self.jmm1 - 1)
        # H at U/V faces (Arakawa C)
        H_u_east = 0.5 * (d_2d[s_i_p1, s_j] + d_2d[s_i, s_j])
        H_u_west = 0.5 * (d_2d[s_i, s_j] + d_2d[s_i_m1, s_j])
        H_v_north = 0.5 * (d_2d[s_i, s_j_p1] + d_2d[s_i, s_j])
        H_v_south = 0.5 * (d_2d[s_i, s_j] + d_2d[s_i, s_j_m1])
        dx_east = (self.dx[s_i_p1, s_j] + self.dx[s_i, s_j]) * 0.5
        dx_west = (self.dx[s_i, s_j] + self.dx[s_i_m1, s_j]) * 0.5
        dy_east = (self.dy[s_i_p1, s_j] + self.dy[s_i, s_j]) * 0.5
        dy_west = (self.dy[s_i, s_j] + self.dy[s_i_m1, s_j]) * 0.5
        dy_north = (self.dy[s_i, s_j_p1] + self.dy[s_i, s_j]) * 0.5
        dy_south = (self.dy[s_i, s_j] + self.dy[s_i, s_j_m1]) * 0.5
        dx_north = (self.dx[s_i, s_j_p1] + self.dx[s_i, s_j]) * 0.5
        dx_south = (self.dx[s_i, s_j] + self.dx[s_i, s_j_m1]) * 0.5
        # Fluxes with land mask (dum @ U-faces, dvm @ V-faces)
        flux_x_east = H_u_east * (eta[s_i_p1, s_j] - eta[s_i, s_j]) / (dx_east + 1e-20) * dy_east * self.dum[s_i_p1, s_j]
        flux_x_west = H_u_west * (eta[s_i, s_j] - eta[s_i_m1, s_j]) / (dx_west + 1e-20) * dy_west * self.dum[s_i, s_j]
        flux_y_north = H_v_north * (eta[s_i, s_j_p1] - eta[s_i, s_j]) / (dy_north + 1e-20) * dx_north * self.dvm[s_i, s_j_p1]
        flux_y_south = H_v_south * (eta[s_i, s_j] - eta[s_i, s_j_m1]) / (dy_south + 1e-20) * dx_south * self.dvm[s_i, s_j]
        div_flux = (flux_x_east - flux_x_west + flux_y_north - flux_y_south) / (self.art[s_i, s_j] + 1e-20)
        # Correct coefficient: 4*dti^2*g*alpha (semi-implicit u = u* - g*dt*grad(eta) => dt^2*g)
        coeff = 4.0 * (self.dti ** 2) * self.grav * self.alpha
        out = eta.clone()
        out[s_i, s_j] = eta[s_i, s_j] - coeff * div_flux
        return out

    def _enforce_eta_neumann_bc(self, eta):
        """Enforce the case boundary policy for eta in the CG solver."""
        self.enforce_implicit_eta_boundary(eta)

    def solve_implicit_eta_cg(self, rhs, d_2d, max_iter=200, tol=1e-7, print_cg_residual=True):
        """
        Solve L(eta) = rhs by Conjugate Gradient. L(eta)=eta - 4*dti^2*g*alpha*div(H*grad(eta)).
        rhs, d_2d: (im, jm). Returns eta solution (im, jm).
        Boundary conditions are provided by the active implicit boundary policy.
        """
        s_i = slice(1, self.imm1)
        s_j = slice(1, self.jmm1)

        # [OPTIMIZATION] Warm Start: use linear extrapolation in time as initial guess
        # to significantly reduce CG iterations (eta^{n+1} ~ 2*eta^n - eta^{n-1}).
        x = (2.0 * self.el - self.elb).clone()
        # Apply BC to initial guess (Dirichlet/Neumann as appropriate)
        self._enforce_eta_neumann_bc(x)
        self.prepare_implicit_eta_rhs(rhs)

        r = rhs - self.apply_elliptic_operator(x, d_2d)
        p = r.clone()
        rs_old = (r[s_i, s_j] * r[s_i, s_j]).sum().item()
        if print_cg_residual:
            print(f"  [CG] iter 0  loss(residual^2) = {rs_old:.6e}")
        for it in range(1, max_iter + 1):
            self._enforce_eta_neumann_bc(p)
            Ap = self.apply_elliptic_operator(p, d_2d)
            alpha = rs_old / ((p[s_i, s_j] * Ap[s_i, s_j]).sum().item() + 1e-30)
            x = x + alpha * p
            self._enforce_eta_neumann_bc(x)
            r = r - alpha * Ap
            rs_new = (r[s_i, s_j] * r[s_i, s_j]).sum().item()
            if print_cg_residual:
                print(f"  [CG] iter {it:3d}  loss(residual^2) = {rs_new:.6e}")
            if rs_new <= tol * tol or rs_new <= 1e-30:
                break
            beta = rs_new / (rs_old + 1e-30)
            p = r + beta * p
            self._enforce_eta_neumann_bc(p)
            rs_old = rs_new
        self._enforce_eta_neumann_bc(x)
        return x
