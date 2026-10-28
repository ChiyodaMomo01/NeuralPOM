import os

import torch
from torch._higher_order_ops import while_loop as torch_while_loop

from .reference_core import POM2K_Core as _ReferencePOM2KCore

try:
    from .triton_kernels import (
        apply_elliptic_operator_triton,
        cg_update_p_dirichlet_triton,
        cg_update_x_r_dirichlet_triton,
        triton_available,
    )
except Exception:  # pragma: no cover - Triton is optional.
    apply_elliptic_operator_triton = None
    cg_update_p_dirichlet_triton = None
    cg_update_x_r_dirichlet_triton = None

    def triton_available():
        return False


class _ImplicitEtaSolveFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, core, rhs, d_2d, max_iter, tol, print_cg_residual, initial_guess):
        x = core._solve_implicit_eta_cg_fast(
            rhs,
            d_2d,
            max_iter=max_iter,
            tol=tol,
            print_cg_residual=print_cg_residual,
            initial_guess=initial_guess,
        )
        # Retaining the live core creates core -> state.grad_fn -> ctx -> core
        # cycles and retains entire checkpointed trajectories across batches.
        # The adjoint only needs fixed geometry, solver options and scratch.
        ctx.core = core._make_eta_adjoint_solver()
        ctx.max_iter = max_iter
        ctx.tol = tol
        ctx.save_for_backward(x, d_2d)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        core = ctx.core
        x, d_2d = ctx.saved_tensors

        with torch.no_grad():
            lambda_eta = core._solve_implicit_eta_adjoint(
                grad_output.detach(), d_2d, max_iter=ctx.max_iter, tol=ctx.tol
            )
            grad_d = core._apply_elliptic_operator_depth_adjoint(
                x.detach(),
                lambda_eta,
                d_2d,
            )

        return None, lambda_eta, -grad_d, None, None, None, None


class POM2K_Core(_ReferencePOM2KCore):
    """
    P0-corrected implicit core used for all NeuralPOM 1.0 production paths.

    The underlying elliptic problem is unchanged. This class upgrades the
    default implicit core with cached geometry terms, configurable PCG
    preconditioning, and reusable scratch buffers for simulation efficiency.
    """

    NUMERICS_VERSION = "implicit-dg-v2-aligned-1e5-p0-barotropic-coupling-fixed"
    BAROTROPIC_ENDPOINT_TRANSPORT_COEFFICIENT = 0.25
    AUTOGRAD_VERSION = "area-transpose-scaled-adjoint-zero-strain-v2"
    DEFAULT_ETA_SOLVER_REL_TOL = 1e-5
    DEFAULT_ETA_SOLVER_PRECONDITIONER = "spectral"
    DEFAULT_SPATIAL_SHAPIRO_GAMMA = 0.05

    def __init__(self, im, jm, *args, **kwargs):
        super().__init__(im, jm, *args, **kwargs)
        spatial_shapiro = os.environ.get(
            "NEURALPOM_IMPLICIT_SPATIAL_SHAPIRO", "0"
        ).strip().lower()
        if spatial_shapiro not in {"0", "1", "false", "true", "off", "on", "no", "yes"}:
            raise ValueError(
                "NEURALPOM_IMPLICIT_SPATIAL_SHAPIRO must be one of "
                "0/1, false/true, off/on, or no/yes"
            )
        self.spatial_shapiro_enabled = spatial_shapiro in {"1", "true", "on", "yes"}
        spatial_shapiro_gamma = (
            self.DEFAULT_SPATIAL_SHAPIRO_GAMMA if self.spatial_shapiro_enabled else 0.0
        )
        self.barotropic_shapiro = spatial_shapiro_gamma
        self.tracer_shapiro = spatial_shapiro_gamma
        self.velocity_shapiro = spatial_shapiro_gamma
        self._accel_cache_ready = False
        self._accel_cache = None
        self._cg_work = None
        self.eta_solver_rel_tol = float(
            os.environ.get(
                "NEURALPOM_ETA_SOLVER_REL_TOL",
                str(self.DEFAULT_ETA_SOLVER_REL_TOL),
            )
        )
        self.eta_solver_preconditioner = os.environ.get(
            "NEURALPOM_ETA_SOLVER_PRECONDITIONER",
            self.DEFAULT_ETA_SOLVER_PRECONDITIONER,
        ).strip().lower()
        self.compile_eta_solver = os.environ.get("NEURALPOM_COMPILE_ETA_SOLVER", "0") == "1"
        self.use_while_loop_eta_solver = os.environ.get("NEURALPOM_USE_WHILE_LOOP_ETA", "0") == "1"
        self.use_triton_elliptic_operator = (
            os.environ.get("NEURALPOM_TRITON_ELLIPTIC", "0") == "1" and triton_available()
        )
        self.use_triton_cg_updates = (
            os.environ.get("NEURALPOM_TRITON_CG_UPDATES", "0") == "1" and triton_available()
        )
        self.graph_eta_max_iter = int(os.environ.get("NEURALPOM_GRAPH_ETA_MAX_ITER", "0") or 0)
        self._eta_compile_failed = False
        self._eta_while_loop_failed = False
        self._eta_solver_iteration_history: list[int] = []

    def _build_acceleration_cache(self):
        s_i = slice(1, self.imm1)
        s_j = slice(1, self.jmm1)
        s_i_p1 = slice(2, self.imm1 + 1)
        s_i_m1 = slice(0, self.imm1 - 1)
        s_j_p1 = slice(2, self.jmm1 + 1)
        s_j_m1 = slice(0, self.jmm1 - 1)

        dx_east = 0.5 * (self.dx[s_i_p1, s_j] + self.dx[s_i, s_j])
        dx_west = 0.5 * (self.dx[s_i, s_j] + self.dx[s_i_m1, s_j])
        dy_east = 0.5 * (self.dy[s_i_p1, s_j] + self.dy[s_i, s_j])
        dy_west = 0.5 * (self.dy[s_i, s_j] + self.dy[s_i_m1, s_j])
        dy_north = 0.5 * (self.dy[s_i, s_j_p1] + self.dy[s_i, s_j])
        dy_south = 0.5 * (self.dy[s_i, s_j] + self.dy[s_i, s_j_m1])
        dx_north = 0.5 * (self.dx[s_i, s_j_p1] + self.dx[s_i, s_j])
        dx_south = 0.5 * (self.dx[s_i, s_j] + self.dx[s_i, s_j_m1])

        s_i_full = slice(1, self.im)
        s_j_full = slice(1, self.jm)
        s_i_full_m1 = slice(0, self.im - 1)
        s_j_full_m1 = slice(0, self.jm - 1)
        s_imm1 = slice(1, self.imm1)
        s_jmm1 = slice(1, self.jmm1)

        dx_sum_4 = (
            self.dx[s_i_full, s_j_full]
            + self.dx[s_i_full_m1, s_j_full]
            + self.dx[s_i_full, s_j_full_m1]
            + self.dx[s_i_full_m1, s_j_full_m1]
        )
        dy_sum_4 = (
            self.dy[s_i_full, s_j_full]
            + self.dy[s_i_full_m1, s_j_full]
            + self.dy[s_i_full, s_j_full_m1]
            + self.dy[s_i_full_m1, s_j_full_m1]
        )

        self._accel_cache = {
            "s_i": s_i,
            "s_j": s_j,
            "s_i_p1": s_i_p1,
            "s_i_m1": s_i_m1,
            "s_j_p1": s_j_p1,
            "s_j_m1": s_j_m1,
            "s_i_full": s_i_full,
            "s_j_full": s_j_full,
            "s_i_full_m1": s_i_full_m1,
            "s_j_full_m1": s_j_full_m1,
            "s_imm1": s_imm1,
            "s_jmm1": s_jmm1,
            "dy_over_dx_east": (dy_east / (dx_east + 1e-20)).contiguous(),
            "dy_over_dx_west": (dy_west / (dx_west + 1e-20)).contiguous(),
            "dx_over_dy_north": (dx_north / (dy_north + 1e-20)).contiguous(),
            "dx_over_dy_south": (dx_south / (dy_south + 1e-20)).contiguous(),
            "mask_u_east": self.dum[s_i_p1, s_j].contiguous(),
            "mask_u_west": self.dum[s_i, s_j].contiguous(),
            "mask_v_north": self.dvm[s_i, s_j_p1].contiguous(),
            "mask_v_south": self.dvm[s_i, s_j].contiguous(),
            "inv_art": (1.0 / (self.art[s_i, s_j] + 1e-20)).contiguous(),
            "implicit_coeff": 4.0 * (self.dti ** 2) * self.grav * self.alpha,
            "dx_sum_4": dx_sum_4,
            "dy_sum_4": dy_sum_4,
            "dx_sum_4_quarter": 0.25 * dx_sum_4,
            "dy_sum_4_quarter": 0.25 * dy_sum_4,
        }
        self._cg_work = {
            "ap": torch.zeros((self.im, self.jm), device=self.device, dtype=self.el.dtype),
            "rhs": torch.zeros((self.im, self.jm), device=self.device, dtype=self.el.dtype),
            "r": torch.zeros((self.im, self.jm), device=self.device, dtype=self.el.dtype),
            "p": torch.zeros((self.im, self.jm), device=self.device, dtype=self.el.dtype),
            "z": torch.zeros((self.im, self.jm), device=self.device, dtype=self.el.dtype),
            "diag": torch.ones((self.im, self.jm), device=self.device, dtype=self.el.dtype),
            "curv3d": torch.zeros((self.im, self.jm, self.kb), device=self.device, dtype=self.u.dtype),
            "xflux3d": torch.zeros((self.im, self.jm, self.kb), device=self.device, dtype=self.u.dtype),
            "yflux3d": torch.zeros((self.im, self.jm, self.kb), device=self.device, dtype=self.u.dtype),
            "advx3d": torch.zeros((self.im, self.jm, self.kb), device=self.device, dtype=self.u.dtype),
            "advy3d": torch.zeros((self.im, self.jm, self.kb), device=self.device, dtype=self.u.dtype),
        }
        self._accel_cache_ready = True

    def _ensure_acceleration_cache(self):
        if not self._accel_cache_ready:
            self._build_acceleration_cache()

    def _make_eta_adjoint_solver(self):
        """A solver context with no references to evolving 3-D autograd state."""
        solver = object.__new__(type(self))
        for name, value in vars(self).items():
            if isinstance(value, (str, int, float, bool, type(None), torch.device)):
                setattr(solver, name, value)
        for name in ('h', 'dx', 'dy', 'art', 'dum', 'dvm', 'fsm', 'aru', 'arv', 'el', 'elb'):
            setattr(solver, name, getattr(self, name).detach())
        solver._implicit_boundary_policy = self.implicit_boundary_policy()
        solver._accel_cache = {
            name: value.detach() if torch.is_tensor(value) else value
            for name, value in self._accel_cache.items()
        }
        solver._accel_cache_ready = True
        spectral = getattr(self, '_spectral_eta_preconditioner_cache', None)
        solver._spectral_eta_preconditioner_cache = (
            {name: value.detach() if torch.is_tensor(value) else value
             for name, value in spectral.items()} if spectral is not None else None
        )
        solver._cg_work = {name: torch.empty_like(self.el)
                           for name in ('rhs', 'ap', 'r', 'p', 'z', 'diag')}
        solver._eta_solver_iteration_history = []
        solver.compile_eta_solver = False
        solver.runtime_cuda_graph_fast_path = False
        return solver

    def _solve_implicit_eta_adjoint(self, grad_output, d_2d, *, max_iter, tol):
        """Solve the Euclidean transpose, retaining the primal PCG tolerance.

        The finite-volume operator is A = I - M^-1 L, with diagonal cell-area
        matrix M and symmetric face-flux matrix L. Thus A^-T = M A^-1 M^-1.
        Relative area weights avoid changing the RHS units by an area of 1e9.
        Homogeneous RHS scaling keeps CG dot products finite for large BPTT
        gradients; the final adjoint is restored to its original magnitude.
        """
        area = torch.ones_like(grad_output)
        area[1:-1, 1:-1] = self.art[1:-1, 1:-1] / self.art[1, 1]
        rhs = grad_output / area
        scale = rhs.abs().amax().clamp_min(1.0)
        normalized_rhs = rhs / scale
        # The relative convergence target is invariant to this scaling. Scale
        # the absolute tolerance too, so the stopping condition is unchanged.
        scaled_tol = float(tol) / float(scale)
        solution = self._solve_implicit_eta_cg_fast(
            normalized_rhs, d_2d, max_iter=max_iter, tol=scaled_tol,
            print_cg_residual=False, initial_guess=torch.zeros_like(grad_output),
        )
        return solution * area * scale

    def run_initialization(self):
        result = super().run_initialization()
        self._build_acceleration_cache()
        if self._use_spectral_eta_preconditioner():
            self._get_spectral_eta_preconditioner_cache()
        return result

    def load_checkpoint(self, filename, *, exact_restart=False):
        result = super().load_checkpoint(filename, exact_restart=exact_restart)
        self._build_acceleration_cache()
        if self._use_spectral_eta_preconditioner():
            self._get_spectral_eta_preconditioner_cache()
        return result

    def _solve_implicit_eta_cg_fast(
        self,
        rhs,
        d_2d,
        max_iter=200,
        tol=1e-7,
        print_cg_residual=True,
        initial_guess=None,
    ):
        if getattr(self, "runtime_cuda_graph_fast_path", False) and not torch.is_grad_enabled() and not print_cg_residual:
            return self._solve_implicit_eta_cg_eager(
                rhs,
                d_2d,
                max_iter=max_iter,
                tol=tol,
                print_cg_residual=print_cg_residual,
                initial_guess=initial_guess,
            )
        if self._can_use_compiled_eta_solver(max_iter, tol, print_cg_residual):
            try:
                return self._get_compiled_eta_solver()(rhs, d_2d, initial_guess)
            except Exception as exc:
                if not self._eta_compile_failed:
                    print(f"[NeuralPOM] Falling back to eager eta solver after torch.compile failure: {exc}")
                self._eta_compile_failed = True
                self.compile_eta_solver = False
        return self._solve_implicit_eta_cg_eager(
            rhs,
            d_2d,
            max_iter=max_iter,
            tol=tol,
            print_cg_residual=print_cg_residual,
            initial_guess=initial_guess,
        )

    def _can_use_compiled_eta_solver(self, max_iter, tol, print_cg_residual):
        return (
            self.compile_eta_solver
            and not self._eta_compile_failed
            and hasattr(torch, "compile")
            and int(max_iter) == 200
            and abs(float(tol) - 1e-12) <= 1e-30
            and not print_cg_residual
        )

    def _get_compiled_eta_solver(self):
        compiled = getattr(self, "_compiled_eta_solver", None)
        if compiled is None:
            try:
                import torch._dynamo

                torch._dynamo.config.capture_scalar_outputs = True
            except Exception:
                pass

            def _compiled_entry(rhs, d_2d, initial_guess):
                return self._solve_implicit_eta_cg_eager(
                    rhs,
                    d_2d,
                    max_iter=200,
                    tol=1e-12,
                    print_cg_residual=False,
                    initial_guess=initial_guess,
                )

            compiled = torch.compile(_compiled_entry, mode="reduce-overhead")
            self._compiled_eta_solver = compiled
        return compiled

    def _get_compiled_eta_solver_while(self):
        compiled = getattr(self, "_compiled_eta_solver_while", None)
        if compiled is None:
            try:
                import torch._dynamo

                torch._dynamo.config.capture_scalar_outputs = True
            except Exception:
                pass

            def _compiled_entry(rhs, d_2d, initial_guess):
                return self._solve_implicit_eta_cg_while(
                    rhs,
                    d_2d,
                    max_iter=200,
                    tol=1e-12,
                    print_cg_residual=False,
                    initial_guess=initial_guess,
                )

            compiled = torch.compile(_compiled_entry, mode="reduce-overhead", fullgraph=False, dynamic=False)
            self._compiled_eta_solver_while = compiled
        return compiled

    def _solve_implicit_eta_cg_while(
        self,
        rhs,
        d_2d,
        max_iter=200,
        tol=1e-7,
        print_cg_residual=True,
        initial_guess=None,
    ):
        self._ensure_acceleration_cache()
        cache = self._accel_cache
        s_i = cache["s_i"]
        s_j = cache["s_j"]

        if initial_guess is None:
            x = (2.0 * self.el - self.elb).clone()
        else:
            x = initial_guess.detach().clone()
        self._enforce_eta_neumann_bc(x)

        rhs_work = rhs.detach().clone()
        rhs_work[0, :] = 0.0
        rhs_work[self.im - 1, :] = 0.0
        rhs_work[:, 0] = 0.0
        rhs_work[:, self.jm - 1] = 0.0

        ap = x.clone()
        self._apply_elliptic_operator_into(x, d_2d, ap)
        r = rhs_work - ap
        if self._use_spectral_eta_preconditioner():
            z = torch.zeros_like(r)
            self._apply_spectral_preconditioner_into(r, z)
        else:
            diag = torch.zeros_like(d_2d)
            self._build_jacobi_preconditioner_into(d_2d, diag)
            z = torch.zeros_like(r)
            self._apply_jacobi_preconditioner_into(r, diag, z)
        p = z.clone()
        rho_old = (r[s_i, s_j] * z[s_i, s_j]).sum()
        rs_old = (r[s_i, s_j] * r[s_i, s_j]).sum()
        capture_safe = getattr(self, "runtime_cuda_graph_fast_path", False)
        if capture_safe:
            initial_residual = rs_old.detach()
            tol_sq = tol * tol
            eps = 1e-30
            rel_tol_sq = (self.eta_solver_rel_tol ** 2) * initial_residual
            self._last_eta_solver_iterations = max_iter
            self._last_eta_solver_residual = None
        else:
            tol_sq = rhs.new_tensor(tol * tol)
            eps = rhs.new_tensor(1e-30)
            initial_residual = rs_old.item()
            rel_tol_sq = rhs.new_tensor((self.eta_solver_rel_tol ** 2) * max(initial_residual, 1e-30))
            self._last_eta_solver_iterations = 0
            self._last_eta_solver_residual = initial_residual
        max_iter_t = max_iter
        it0 = (rs_old.detach() * 0).to(dtype=torch.int64)

        def cond_fn(it, x, r, p, rho_old, rs_old):
            return (it < max_iter_t) & (rs_old > tol_sq) & (rs_old > rel_tol_sq) & (rs_old > eps)

        def body_fn(it, x, r, p, rho_old, rs_old):
            p_bc = p.clone()
            self._enforce_eta_neumann_bc(p_bc)
            ap = p_bc.clone()
            self._apply_elliptic_operator_into(p_bc, d_2d, ap)
            denom = (p_bc[s_i, s_j] * ap[s_i, s_j]).sum()
            alpha = rho_old / (denom + eps)
            x = x + alpha * p_bc
            x = x.clone()
            self._enforce_eta_neumann_bc(x)
            r = r - alpha * ap
            rs_new = (r[s_i, s_j] * r[s_i, s_j]).sum()
            if self._use_spectral_eta_preconditioner():
                z = torch.zeros_like(r)
                self._apply_spectral_preconditioner_into(r, z)
            else:
                diag = torch.zeros_like(d_2d)
                self._build_jacobi_preconditioner_into(d_2d, diag)
                z = torch.zeros_like(r)
                self._apply_jacobi_preconditioner_into(r, diag, z)
            rho_new = (r[s_i, s_j] * z[s_i, s_j]).sum()
            beta = rho_new / (rho_old + eps)
            p = z + beta * p_bc
            p = p.clone()
            self._enforce_eta_neumann_bc(p)
            return (it + 1, x, r, p, rho_new, rs_new)

        it, x, r, p, rho_old, rs_old = torch_while_loop(cond_fn, body_fn, (it0, x, r, p, rho_old, rs_old))
        if capture_safe:
            self._last_eta_solver_iterations = max_iter
            self._last_eta_solver_residual = None
        else:
            self._last_eta_solver_iterations = int(it.item())
            self._last_eta_solver_residual = float(rs_old.item())
        self._eta_solver_iteration_history.append(int(self._last_eta_solver_iterations))
        self._enforce_eta_neumann_bc(x)
        return x

    def _solve_implicit_eta_cg_eager(
        self,
        rhs,
        d_2d,
        max_iter=200,
        tol=1e-7,
        print_cg_residual=True,
        initial_guess=None,
    ):
        self._ensure_acceleration_cache()
        cache = self._accel_cache
        s_i = cache["s_i"]
        s_j = cache["s_j"]

        if initial_guess is None:
            x = (2.0 * self.el - self.elb).clone()
        else:
            x = initial_guess.detach().clone()
        self._enforce_eta_neumann_bc(x)

        rhs_work = self._cg_work["rhs"]
        rhs_work.copy_(rhs)
        rhs_work[0, :] = 0.0
        rhs_work[self.im - 1, :] = 0.0
        rhs_work[:, 0] = 0.0
        rhs_work[:, self.jm - 1] = 0.0

        ap = self._cg_work["ap"]
        self._apply_elliptic_operator_into(x, d_2d, ap)
        r = self._cg_work["r"]
        p = self._cg_work["p"]
        z = self._cg_work["z"]
        diag = self._cg_work["diag"]
        r.copy_(rhs_work)
        r.sub_(ap)
        if self._use_spectral_eta_preconditioner():
            self._apply_spectral_preconditioner_into(r, z)
        else:
            self._build_jacobi_preconditioner_into(d_2d, diag)
            self._apply_jacobi_preconditioner_into(r, diag, z)
        p.copy_(z)
        rho_old = (r[s_i, s_j] * z[s_i, s_j]).sum()
        rs_old = (r[s_i, s_j] * r[s_i, s_j]).sum()

        capture_safe = getattr(self, "runtime_cuda_graph_fast_path", False)
        if capture_safe:
            initial_residual = rs_old.detach()
            effective_max_iter = max_iter
            if self.graph_eta_max_iter > 0:
                effective_max_iter = min(max_iter, int(self.graph_eta_max_iter))
            self._last_eta_solver_iterations = effective_max_iter
            self._last_eta_solver_residual = None

            tol_sq = tol * tol
            rel_tol_sq = (self.eta_solver_rel_tol ** 2) * torch.clamp_min(initial_residual, 1e-30)
            eps = 1e-30
            active = initial_residual > -1.0
            for _ in range(effective_max_iter):
                self._enforce_eta_neumann_bc(p)
                self._apply_elliptic_operator_into(p, d_2d, ap)
                denom = (p[s_i, s_j] * ap[s_i, s_j]).sum()
                alpha_raw = rho_old / (denom + eps)
                alpha = alpha_raw * active.to(dtype=rhs.dtype)
                if self.use_triton_cg_updates:
                    if not cg_update_x_r_dirichlet_triton(self, x, r, p, ap, alpha):
                        x.add_(alpha * p)
                        self._enforce_eta_neumann_bc(x)
                        r.sub_(alpha * ap)
                else:
                    x.add_(alpha * p)
                    self._enforce_eta_neumann_bc(x)
                    r.sub_(alpha * ap)
                rs_new = (r[s_i, s_j] * r[s_i, s_j]).sum()
                if self._use_spectral_eta_preconditioner():
                    self._apply_spectral_preconditioner_into(r, z)
                else:
                    self._apply_jacobi_preconditioner_into(r, diag, z)
                rho_new = (r[s_i, s_j] * z[s_i, s_j]).sum()
                converged = (rs_new <= tol_sq) | (rs_new <= rel_tol_sq) | (rs_new <= eps)
                continue_mask = active & (~converged)
                beta = rho_new / (rho_old + eps)
                if self.use_triton_cg_updates:
                    continue_mask_tensor = continue_mask.to(dtype=torch.bool)
                    if not cg_update_p_dirichlet_triton(self, p, z, p, beta, continue_mask_tensor):
                        p_candidate = z + beta * p
                        p.copy_(torch.where(continue_mask, p_candidate, p))
                        self._enforce_eta_neumann_bc(p)
                else:
                    p_candidate = z + beta * p
                    p.copy_(torch.where(continue_mask, p_candidate, p))
                    self._enforce_eta_neumann_bc(p)
                rho_old = torch.where(continue_mask, rho_new, rho_old)
                active = continue_mask
            self._enforce_eta_neumann_bc(x)
            self._eta_solver_iteration_history.append(int(self._last_eta_solver_iterations))
            return x

        initial_residual = rs_old.item()
        self._last_eta_solver_iterations = 0
        self._last_eta_solver_residual = initial_residual

        tol_sq = tol * tol
        rel_tol_sq = (self.eta_solver_rel_tol ** 2) * max(initial_residual, 1e-30)
        for it in range(1, max_iter + 1):
            self._enforce_eta_neumann_bc(p)
            self._apply_elliptic_operator_into(p, d_2d, ap)
            denom = (p[s_i, s_j] * ap[s_i, s_j]).sum()
            alpha = rho_old / (denom + 1e-30)
            x.add_(alpha * p)
            self._enforce_eta_neumann_bc(x)
            r.sub_(alpha * ap)
            rs_new = (r[s_i, s_j] * r[s_i, s_j]).sum()
            rs_new_value = rs_new.item()
            self._last_eta_solver_iterations = it
            self._last_eta_solver_residual = rs_new_value
            if rs_new_value <= tol_sq or rs_new_value <= rel_tol_sq or rs_new_value <= 1e-30:
                break
            if self._use_spectral_eta_preconditioner():
                self._apply_spectral_preconditioner_into(r, z)
            else:
                self._apply_jacobi_preconditioner_into(r, diag, z)
            rho_new = (r[s_i, s_j] * z[s_i, s_j]).sum()
            beta = rho_new / (rho_old + 1e-30)
            p.mul_(beta)
            p.add_(z)
            self._enforce_eta_neumann_bc(p)
            rho_old = rho_new
        self._eta_solver_iteration_history.append(int(self._last_eta_solver_iterations))
        self._enforce_eta_neumann_bc(x)
        return x

    def _apply_elliptic_operator_into(self, eta, d_2d, out):
        self._ensure_acceleration_cache()
        cache = self._accel_cache

        if self.use_triton_elliptic_operator and eta.is_cuda and d_2d.is_cuda and out.is_cuda:
            if apply_elliptic_operator_triton(self, eta, d_2d, out):
                return out

        s_i = cache["s_i"]
        s_j = cache["s_j"]
        s_i_p1 = cache["s_i_p1"]
        s_i_m1 = cache["s_i_m1"]
        s_j_p1 = cache["s_j_p1"]
        s_j_m1 = cache["s_j_m1"]

        h_u_east = 0.5 * (d_2d[s_i_p1, s_j] + d_2d[s_i, s_j])
        h_u_west = 0.5 * (d_2d[s_i, s_j] + d_2d[s_i_m1, s_j])
        h_v_north = 0.5 * (d_2d[s_i, s_j_p1] + d_2d[s_i, s_j])
        h_v_south = 0.5 * (d_2d[s_i, s_j] + d_2d[s_i, s_j_m1])

        flux_x_east = (
            h_u_east
            * (eta[s_i_p1, s_j] - eta[s_i, s_j])
            * cache["dy_over_dx_east"]
            * cache["mask_u_east"]
        )
        flux_x_west = (
            h_u_west
            * (eta[s_i, s_j] - eta[s_i_m1, s_j])
            * cache["dy_over_dx_west"]
            * cache["mask_u_west"]
        )
        flux_y_north = (
            h_v_north
            * (eta[s_i, s_j_p1] - eta[s_i, s_j])
            * cache["dx_over_dy_north"]
            * cache["mask_v_north"]
        )
        flux_y_south = (
            h_v_south
            * (eta[s_i, s_j] - eta[s_i, s_j_m1])
            * cache["dx_over_dy_south"]
            * cache["mask_v_south"]
        )

        div_flux = (flux_x_east - flux_x_west + flux_y_north - flux_y_south) * cache["inv_art"]
        out.copy_(eta)
        out[s_i, s_j] = eta[s_i, s_j] - cache["implicit_coeff"] * div_flux
        return out

    def apply_elliptic_operator(self, eta, d_2d):
        out = eta.clone()
        return self._apply_elliptic_operator_into(eta, d_2d, out)

    def _apply_elliptic_operator_depth_adjoint(self, eta, lambda_eta, d_2d):
        self._ensure_acceleration_cache()
        cache = self._accel_cache

        s_i = cache["s_i"]
        s_j = cache["s_j"]
        s_i_p1 = cache["s_i_p1"]
        s_i_m1 = cache["s_i_m1"]
        s_j_p1 = cache["s_j_p1"]
        s_j_m1 = cache["s_j_m1"]

        coeff = cache["implicit_coeff"]
        inv_art = cache["inv_art"]

        grad_d = torch.zeros_like(d_2d)
        lam = lambda_eta[s_i, s_j] * coeff * inv_art

        east = 0.5 * lam * (eta[s_i_p1, s_j] - eta[s_i, s_j]) * cache["dy_over_dx_east"] * cache["mask_u_east"]
        west = 0.5 * lam * (eta[s_i, s_j] - eta[s_i_m1, s_j]) * cache["dy_over_dx_west"] * cache["mask_u_west"]
        north = 0.5 * lam * (eta[s_i, s_j_p1] - eta[s_i, s_j]) * cache["dx_over_dy_north"] * cache["mask_v_north"]
        south = 0.5 * lam * (eta[s_i, s_j] - eta[s_i, s_j_m1]) * cache["dx_over_dy_south"] * cache["mask_v_south"]

        grad_d[s_i_p1, s_j] += -east
        grad_d[s_i, s_j] += -east
        grad_d[s_i, s_j] += west
        grad_d[s_i_m1, s_j] += west
        grad_d[s_i, s_j_p1] += -north
        grad_d[s_i, s_j] += -north
        grad_d[s_i, s_j] += south
        grad_d[s_i, s_j_m1] += south
        return grad_d

    def _build_jacobi_preconditioner_into(self, d_2d, diag_out):
        self._ensure_acceleration_cache()
        cache = self._accel_cache

        s_i = cache["s_i"]
        s_j = cache["s_j"]
        s_i_p1 = cache["s_i_p1"]
        s_i_m1 = cache["s_i_m1"]
        s_j_p1 = cache["s_j_p1"]
        s_j_m1 = cache["s_j_m1"]

        h_u_east = 0.5 * (d_2d[s_i_p1, s_j] + d_2d[s_i, s_j])
        h_u_west = 0.5 * (d_2d[s_i, s_j] + d_2d[s_i_m1, s_j])
        h_v_north = 0.5 * (d_2d[s_i, s_j_p1] + d_2d[s_i, s_j])
        h_v_south = 0.5 * (d_2d[s_i, s_j] + d_2d[s_i, s_j_m1])

        diag_out.fill_(1.0)
        diag_out[s_i, s_j] = 1.0 + cache["implicit_coeff"] * cache["inv_art"] * (
            h_u_east * cache["dy_over_dx_east"] * cache["mask_u_east"]
            + h_u_west * cache["dy_over_dx_west"] * cache["mask_u_west"]
            + h_v_north * cache["dx_over_dy_north"] * cache["mask_v_north"]
            + h_v_south * cache["dx_over_dy_south"] * cache["mask_v_south"]
        )
        return diag_out

    def _apply_jacobi_preconditioner_into(self, residual, diag, out):
        cache = self._accel_cache
        s_i = cache["s_i"]
        s_j = cache["s_j"]
        out.copy_(residual)
        out[s_i, s_j] = residual[s_i, s_j] / (diag[s_i, s_j] + 1e-30)
        return out

    def _use_spectral_eta_preconditioner(self):
        return self.eta_solver_preconditioner == "spectral"

    @staticmethod
    def _dst1(x, dim):
        n = x.shape[dim]
        shape = list(x.shape)
        shape[dim] = 2 * (n + 1)
        y = x.new_zeros(shape)

        sl = [slice(None)] * x.ndim
        sl[dim] = slice(1, n + 1)
        y[tuple(sl)] = x

        sl[dim] = slice(n + 2, 2 * (n + 1))
        y[tuple(sl)] = -torch.flip(x, dims=[dim])

        transformed = torch.fft.fft(y, dim=dim)
        return -transformed.imag.narrow(dim, 1, n)

    def _get_spectral_eta_preconditioner_cache(self):
        spectral_cache = getattr(self, "_spectral_eta_preconditioner_cache", None)
        if spectral_cache is not None:
            return spectral_cache

        self._ensure_acceleration_cache()
        cache = self._accel_cache
        s_i = cache["s_i"]
        s_j = cache["s_j"]
        ni = self.imm1 - 1
        nj = self.jmm1 - 1
        dtype = self.el.dtype
        device = self.device

        kx = torch.arange(1, ni + 1, device=device, dtype=dtype).view(-1, 1)
        ky = torch.arange(1, nj + 1, device=device, dtype=dtype).view(1, -1)
        dx0 = self.dx[s_i, s_j].median()
        dy0 = self.dy[s_i, s_j].median()
        h0 = self.h[s_i, s_j].median()
        lam_x = 4.0 * torch.sin(torch.pi * kx / (2.0 * (ni + 1))).square() / (dx0 * dx0 + 1e-30)
        lam_y = 4.0 * torch.sin(torch.pi * ky / (2.0 * (nj + 1))).square() / (dy0 * dy0 + 1e-30)
        inv_eigs = 1.0 / (1.0 + cache["implicit_coeff"] * h0 * (lam_x + lam_y))

        spectral_cache = {
            "s_i": s_i,
            "s_j": s_j,
            "inv_eigs": inv_eigs,
            "inverse_scale": 4.0 * (ni + 1) * (nj + 1),
        }
        self._spectral_eta_preconditioner_cache = spectral_cache
        return spectral_cache

    def _apply_spectral_preconditioner_into(self, residual, out):
        cache = getattr(self, "_spectral_eta_preconditioner_cache", None)
        if cache is None:
            cache = self._get_spectral_eta_preconditioner_cache()
        s_i = cache["s_i"]
        s_j = cache["s_j"]
        coeff = self._dst1(self._dst1(residual[s_i, s_j], 0), 1)
        coeff = coeff * cache["inv_eigs"].clone()
        inner = self._dst1(self._dst1(coeff, 0), 1) / cache["inverse_scale"]
        out.zero_()
        out[s_i, s_j] = inner
        return out

    def solve_implicit_eta_cg(self, rhs, d_2d, max_iter=200, tol=1e-7, print_cg_residual=True, initial_guess=None):
        if torch.is_grad_enabled() and (rhs.requires_grad or d_2d.requires_grad):
            return _ImplicitEtaSolveFunction.apply(
                self,
                rhs,
                d_2d,
                max_iter,
                tol,
                print_cg_residual,
                initial_guess,
            )
        return self._solve_implicit_eta_cg_fast(
            rhs,
            d_2d,
            max_iter=max_iter,
            tol=tol,
            print_cg_residual=print_cg_residual,
            initial_guess=initial_guess,
        )

    def _compute_barotropic_predictor(self):
        s_i = slice(1, self.imm1)
        s_j = slice(1, self.jmm1)
        s_i_m1 = slice(0, self.imm1 - 1)
        s_j_m1 = slice(0, self.jmm1 - 1)
        s_i_p1 = slice(2, self.imm1 + 1)
        s_j_p1 = slice(2, self.jmm1 + 1)

        cor_term = 0.25 * (
            self.cor[s_i, s_j] * self.d[s_i, s_j] * (self.va[s_i, s_j_p1] + self.va[s_i, s_j]) +
            self.cor[s_i_m1, s_j] * self.d[s_i_m1, s_j] * (self.va[s_i_m1, s_j_p1] + self.va[s_i_m1, s_j])
        )
        cor_term_v = 0.25 * (
            self.cor[s_i, s_j] * self.d[s_i, s_j] * (self.ua[s_i_p1, s_j] + self.ua[s_i, s_j]) +
            self.cor[s_i, s_j_m1] * self.d[s_i, s_j_m1] * (self.ua[s_i_p1, s_j_m1] + self.ua[s_i, s_j_m1])
        )

        grad_term_u_expl = 0.25 * self.grav * (self.dy[s_i, s_j] + self.dy[s_i_m1, s_j]) * \
                      (self.d[s_i, s_j] + self.d[s_i_m1, s_j]) * (
                          (1.0 - 2.0 * self.alpha) * (self.el[s_i, s_j] - self.el[s_i_m1, s_j]) +
                          self.alpha * (self.elb[s_i, s_j] - self.elb[s_i_m1, s_j]) +
                          self.e_atmos[s_i, s_j] - self.e_atmos[s_i_m1, s_j]
                      )
        grad_term_v_expl = 0.25 * self.grav * (self.dx[s_i, s_j] + self.dx[s_i, s_j_m1]) * \
                      (self.d[s_i, s_j] + self.d[s_i, s_j_m1]) * (
                          (1.0 - 2.0 * self.alpha) * (self.el[s_i, s_j] - self.el[s_i, s_j_m1]) +
                          self.alpha * (self.elb[s_i, s_j] - self.elb[s_i, s_j_m1]) +
                          self.e_atmos[s_i, s_j] - self.e_atmos[s_i, s_j_m1]
                      )

        uaf_prov_expl = self.adx2d[s_i, s_j] + self.advua[s_i, s_j] - \
                        self.aru[s_i, s_j] * cor_term + grad_term_u_expl + \
                        self.drx2d[s_i, s_j] - self.aru[s_i, s_j] * (-self.wusurf[s_i, s_j] + self.wubot[s_i, s_j])
        vaf_prov_expl = self.ady2d[s_i, s_j] + self.advva[s_i, s_j] + \
                        self.arv[s_i, s_j] * cor_term_v + grad_term_v_expl + \
                        self.dry2d[s_i, s_j] - self.arv[s_i, s_j] * (-self.wvsurf[s_i, s_j] + self.wvbot[s_i, s_j])

        u_star = self.ua.clone()
        v_star = self.va.clone()

        denom_u_star = (self.h[s_i, s_j] + self.el[s_i, s_j] + self.h[s_i_m1, s_j] + self.el[s_i_m1, s_j]) * self.aru[s_i, s_j]
        numer_u_star = (self.h[s_i, s_j] + self.elb[s_i, s_j] + self.h[s_i_m1, s_j] + self.elb[s_i_m1, s_j]) * self.aru[s_i, s_j] * self.uab[s_i, s_j] - \
                       4.0 * self.dti * uaf_prov_expl

        denom_v_star = (self.h[s_i, s_j] + self.el[s_i, s_j] + self.h[s_i, s_j_m1] + self.el[s_i, s_j_m1]) * self.arv[s_i, s_j]
        numer_v_star = (self.h[s_i, s_j] + self.elb[s_i, s_j] + self.h[s_i, s_j_m1] + self.elb[s_i, s_j_m1]) * self.arv[s_i, s_j] * self.vab[s_i, s_j] - \
                       4.0 * self.dti * vaf_prov_expl

        u_star[s_i, s_j] = (numer_u_star / (denom_u_star + 1e-20)) * self.dum[s_i, s_j]
        v_star[s_i, s_j] = (numer_v_star / (denom_v_star + 1e-20)) * self.dvm[s_i, s_j]
        return u_star, v_star, denom_u_star, denom_v_star

    def _use_ip4_triton_predictor(self) -> bool:
        return False

    def _use_ip4_triton_corrector(self) -> bool:
        return False

    def _barotropic_predictor_with_bc(self):
        u_star, v_star, denom_u_star, denom_v_star = self._compute_barotropic_predictor()
        _, u_star, v_star, _, _, _ = self.bcond(2, self.el, u_star, v_star, self.uf, self.vf, self.w)
        return u_star, v_star, denom_u_star, denom_v_star

    def _compute_barotropic_transport_and_rhs(self, u_star, v_star):
        s_i = slice(1, self.imm1)
        s_j = slice(1, self.jmm1)
        s_i_full = slice(1, self.im)
        s_j_full = slice(1, self.jm)
        s_i_full_m1 = slice(0, self.imm1)
        s_j_full_m1 = slice(0, self.jmm1)

        d_avg_u = 0.25 * (self.d[s_i_full, s_j_full] + self.d[s_i_full_m1, s_j_full])
        dy_avg_u = (self.dy[s_i_full, s_j_full] + self.dy[s_i_full_m1, s_j_full])
        fluxua_inner = d_avg_u * dy_avg_u * u_star[s_i_full, s_j_full]

        d_avg_v = 0.25 * (self.d[s_i_full, s_j_full] + self.d[s_i_full, s_j_full_m1])
        dx_avg_v = (self.dx[s_i_full, s_j_full] + self.dx[s_i_full, s_j_full_m1])
        fluxva_inner = d_avg_v * dx_avg_v * v_star[s_i_full, s_j_full]

        flux_div = (
            fluxua_inner[1:, :-1] - fluxua_inner[:-1, :-1]
            + fluxva_inner[:-1, 1:] - fluxva_inner[:-1, :-1]
        )

        rhs_eta = self.elb.clone()
        rhs_eta[s_i, s_j] = self.elb[s_i, s_j] - 2.0 * self.dti * (flux_div / (self.art[s_i, s_j] + 1e-20) + self.vfluxf[s_i, s_j])
        return fluxua_inner, fluxva_inner, rhs_eta

    def _barotropic_transport_rhs_full(self, u_star, v_star):
        fluxua_part, fluxva_part, rhs_eta = self._compute_barotropic_transport_and_rhs(u_star, v_star)
        if fluxua_part.shape == self.fluxua.shape and fluxva_part.shape == self.fluxva.shape:
            return fluxua_part, fluxva_part, rhs_eta
        fluxua = torch.zeros_like(self.fluxua)
        fluxva = torch.zeros_like(self.fluxva)
        fluxua[1:self.im, 1:self.jm] = fluxua_part
        fluxva[1:self.im, 1:self.jm] = fluxva_part
        return fluxua, fluxva, rhs_eta

    def _apply_barotropic_shapiro(self, elf):
        s_i = slice(1, self.imm1)
        s_j = slice(1, self.jmm1)
        s_i_p1 = slice(2, self.imm1 + 1)
        s_i_m1 = slice(0, self.imm1 - 1)
        s_j_p1 = slice(2, self.jmm1 + 1)
        s_j_m1 = slice(0, self.jmm1 - 1)
        gamma_shapiro = float(
            getattr(self, "barotropic_shapiro", self.DEFAULT_SPATIAL_SHAPIRO_GAMMA)
        )
        elf_smooth = elf.clone()
        elf_smooth[s_i, s_j] = elf[s_i, s_j] + (gamma_shapiro / 4.0) * (
            elf[s_i_p1, s_j] + elf[s_i_m1, s_j] +
            elf[s_i, s_j_p1] + elf[s_i, s_j_m1] -
            4.0 * elf[s_i, s_j]
        )
        out = elf.clone()
        out[s_i, s_j] = elf_smooth[s_i, s_j] * self.fsm[s_i, s_j]
        return out

    def _surface_bc_after_solver(self, elf):
        elf = self._apply_barotropic_shapiro(elf)
        elf, _, _, _, _, _ = self.bcond(1, elf, self.uaf, self.vaf, self.uf, self.vf, self.w)
        return elf

    def _compute_barotropic_corrector(self, u_star, v_star, denom_u_star, denom_v_star):
        s_i = slice(1, self.imm1)
        s_j = slice(1, self.jmm1)
        s_i_m1 = slice(0, self.imm1 - 1)
        s_j_m1 = slice(0, self.jmm1 - 1)

        grad_term_u_impl = 0.25 * self.grav * (self.dy[s_i, s_j] + self.dy[s_i_m1, s_j]) * \
                      (self.d[s_i, s_j] + self.d[s_i_m1, s_j]) * self.alpha * (self.elf[s_i, s_j] - self.elf[s_i_m1, s_j])
        grad_term_v_impl = 0.25 * self.grav * (self.dx[s_i, s_j] + self.dx[s_i, s_j_m1]) * \
                      (self.d[s_i, s_j] + self.d[s_i, s_j_m1]) * self.alpha * (self.elf[s_i, s_j] - self.elf[s_i, s_j_m1])

        denom_u_final = (self.h[s_i, s_j] + self.elf[s_i, s_j] + self.h[s_i_m1, s_j] + self.elf[s_i_m1, s_j]) * self.aru[s_i, s_j]
        denom_v_final = (self.h[s_i, s_j] + self.elf[s_i, s_j] + self.h[s_i, s_j_m1] + self.elf[s_i, s_j_m1]) * self.arv[s_i, s_j]

        uaf_inner = (u_star[s_i, s_j] * denom_u_star - 4.0 * self.dti * grad_term_u_impl) / (denom_u_final + 1e-20) * self.dum[s_i, s_j]
        vaf_inner = (v_star[s_i, s_j] * denom_v_star - 4.0 * self.dti * grad_term_v_impl) / (denom_v_final + 1e-20) * self.dvm[s_i, s_j]
        return uaf_inner, vaf_inner

    def _barotropic_corrector_with_bc(self, u_star, v_star, denom_u_star, denom_v_star):
        uaf_inner, vaf_inner = self._compute_barotropic_corrector(u_star, v_star, denom_u_star, denom_v_star)
        uaf = self.uaf.clone()
        vaf = self.vaf.clone()
        uaf[1:self.imm1, 1:self.jmm1] = uaf_inner
        vaf[1:self.imm1, 1:self.jmm1] = vaf_inner
        _, uaf, vaf, _, _, _ = self.bcond(2, self.elf, uaf, vaf, self.uf, self.vf, self.w)
        return uaf, vaf

    def _capture_generation_reference_snapshot(self):
        return {
            "time": float(self.time),
            "u": self.u.detach().clone(),
            "ub": self.ub.detach().clone(),
            "v": self.v.detach().clone(),
            "vb": self.vb.detach().clone(),
            "t": self.t.detach().clone(),
            "s": self.s.detach().clone(),
            "el": self.el.detach().clone(),
            "ua": self.ua.detach().clone(),
            "va": self.va.detach().clone(),
            "uab": self.uab.detach().clone(),
            "vab": self.vab.detach().clone(),
        }

    def run_generation_day(self, steps_per_day: int, capture_interval_steps: int = 8, capture_count: int = 12):
        self.generation_fast_path = True
        self._generation_capture_interval_steps = int(capture_interval_steps)
        self._generation_capture_limit_steps = int(capture_interval_steps) * int(capture_count)
        self._generation_reference_records = []
        self.time0 = self.time
        self.iint = 0
        self.iend = int(steps_per_day)
        try:
            self.run_time_loop()
            return list(self._generation_reference_records)
        finally:
            self.generation_fast_path = False
            self._generation_capture_interval_steps = 0
            self._generation_capture_limit_steps = 0
            self._generation_reference_records = []


ImplicitP0FixedPOM2KCore = POM2K_Core
