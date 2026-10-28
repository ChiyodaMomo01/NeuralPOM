"""P0-corrected implicit time loop with the legacy implementation preserved separately."""

import math
import numpy as np
try:
    import netCDF4 as nc
except ModuleNotFoundError:
    nc = None
import torch
import torch.nn.functional as F

from .autograd import strain_magnitude


torch.set_default_dtype(torch.float32)


# With raw adjacent-depth sums, face averaging and endpoint time averaging
# each contribute 1/2, hence 1/4 overall.
BAROTROPIC_ENDPOINT_TRANSPORT_COEFFICIENT = 0.25


def barotropic_endpoint_transport(
    start_velocity,
    start_adjacent_depth_sum,
    end_velocity,
    end_adjacent_depth_sum,
):
    """Return the face-centred, endpoint-time-mean volume transport."""

    return BAROTROPIC_ENDPOINT_TRANSPORT_COEFFICIENT * (
        start_velocity * start_adjacent_depth_sum
        + end_velocity * end_adjacent_depth_sum
    )


class TimeLoopMixin:
    def run_time_loop(self):
        # Create slice objects for frequent ranges to match MATLAB indices
        s_i = slice(1, self.imm1) # 1 to im-2
        s_j = slice(1, self.jmm1) # 1 to jm-2

        s_i_m1 = slice(0, self.imm1 - 1)
        s_i_p1 = slice(2, self.imm1 + 1)
        s_j_m1 = slice(0, self.jmm1 - 1)
        s_j_p1 = slice(2, self.jmm1 + 1)

        generation_fast_path = (
            getattr(self, "generation_fast_path", False)
            or getattr(self, "runtime_no_grad_fast_path", False)
        )
        vamax_safe_fast_path = generation_fast_path and getattr(self, "runtime_vamax_safe_fast_path", False)

        def refresh_ip6_scalar2d(field):
            return self.apply_implicit_boundary_scalar2d(field)

        def refresh_ip6_velocity2d(u_field, v_field):
            return self.apply_implicit_boundary_velocity2d(u_field, v_field)

        def refresh_ip6_scalar3d(field):
            return self.apply_implicit_boundary_scalar3d(field)

        def refresh_ip6_velocity3d(u_field, v_field):
            return self.apply_implicit_boundary_velocity3d(u_field, v_field)

        if not generation_fast_path:
            # =========================================================
            # CRITICAL FIX FOR AUTOGRAD: Clone Leaf Variables
            # =========================================================
            # Convert leaf tensors from the initial condition into intermediate tensors so later in-place updates are valid.
            self.u = self.u.clone()
            self.v = self.v.clone()
            self.w = self.w.clone()
            self.t = self.t.clone()
            self.s = self.s.clone()
            self.rho = self.rho.clone()

            self.ua = self.ua.clone()
            self.va = self.va.clone()
            self.el = self.el.clone()
            self.et = self.et.clone()
            self.elb = self.elb.clone()
            self.etb = self.etb.clone()

            self.q2 = self.q2.clone()
            self.q2l = self.q2l.clone()
            self.km = self.km.clone()
            self.kh = self.kh.clone()
            self.kq = self.kq.clone()

            self.uab = self.uab.clone()
            self.vab = self.vab.clone()

            # Surface forcing variables
            self.wusurf = self.wusurf.clone()
            self.wvsurf = self.wvsurf.clone()
            self.wtsurf = self.wtsurf.clone()
            self.wssurf = self.wssurf.clone()
            self.swrad = self.swrad.clone()
            self.vfluxf = self.vfluxf.clone()

        # ---------------------------------------------------------
        # Begin internal (3-D) mode
        # ---------------------------------------------------------
        for iint in range(1, self.iend + 1):
            self.iint = iint
            self.time = self.dti * iint * 1.0 / 86400.0 + self.time0

            # Ramp function
            if self.lramp: # Python bool check
                if generation_fast_path and getattr(self, "runtime_ramp_saturated_fast_path", False):
                    self.ramp = 1.0
                else:
                    self.ramp = self.time / self.period
                    if self.ramp > 1.0:
                        self.ramp = 1.0
            else:
                self.ramp = 1.0

            # -----------------------------------------------------
            # Set time dependent, surface and lateral boundary conditions
            # -----------------------------------------------------

            self.apply_implicit_case_surface_forcing(generation_fast_path=generation_fast_path)

            # -----------------------------------------------------
            # Lateral Viscosity (Smagorinsky) & Internal Mode Prep
            # -----------------------------------------------------
            if self.mode != 2:
                # Call ADVCT
                xflux_new, yflux_new, curv_new, advx_new, advy_new = self.advct(
                    self.a, self.c, self.ee, self.advx, self.advy,
                    self.dt, self.u, self.v, self.w, self.ub, self.vb,
                    self.dti2, self.dx, self.dy, self.dz,
                    self.h, self.dum, self.dvm, self.art, self.aru, self.arv,
                    self.cbc, self.uab, self.vab,
                    self.im, self.jm, self.kb, self.imm1, self.jmm1, self.kbm1,
                    self.drx2d, self.dry2d, self.cor, self.curv, self.aam
                    )

                # Update class attributes with new tensors
                self.advx = advx_new
                self.advy = advy_new
                self.curv = curv_new
                # self.xflux/yflux usually are scratchpads, updating them is optional but safe
                # self.a = xflux_new
                # self.c = yflux_new

                # Call BAROPG
                self.rho, self.drhox, self.drhoy = self.baropg()

                # Calculate Smagorinsky lateral viscosity
                dx_sl = self.dx[s_i, s_j].unsqueeze(2)
                dy_sl = self.dy[s_i, s_j].unsqueeze(2)

                u_curr = self.u[s_i, s_j, :self.kbm1]
                u_ip1  = self.u[s_i_p1, s_j, :self.kbm1]
                v_curr = self.v[s_i, s_j, :self.kbm1]
                v_jp1  = self.v[s_i, s_j_p1, :self.kbm1]

                term1 = ((u_ip1 - u_curr) / dx_sl) ** 2
                term2 = ((v_jp1 - v_curr) / dy_sl) ** 2

                u_term = 0.25 * (
                    self.u[s_i, s_j_p1, :self.kbm1] + self.u[s_i_p1, s_j_p1, :self.kbm1] -
                    self.u[s_i, s_j_m1, :self.kbm1] - self.u[s_i_p1, s_j_m1, :self.kbm1]
                ) / dy_sl

                v_term = 0.25 * (
                    self.v[s_i_p1, s_j, :self.kbm1] + self.v[s_i_p1, s_j_p1, :self.kbm1] -
                    self.v[s_i_m1, s_j, :self.kbm1] - self.v[s_i_m1, s_j_p1, :self.kbm1]
                ) / dx_sl

                term3 = 0.5 * (u_term + v_term) ** 2

                # =========================================================
                # Combine Smagorinsky viscosity with background viscosity (MITgcm style).
                # =========================================================

                # 1. Compute the pure Smagorinsky contribution.
                # Keep the active implicit operator on the pre-V3 forward
                # semantics.  A zero strain invariant must produce exactly
                # zero Smagorinsky viscosity; safe_sqrt would introduce a
                # positive floor and therefore change the resolved physics.
                smag_visc = self.horcon * dx_sl * dy_sl * strain_magnitude(term1 + term2 + term3)
                aam_min = float(getattr(self, "aam_min", 0.0) or 0.0)
                if aam_min > 0.0:
                    smag_visc = torch.clamp(smag_visc, min=aam_min)

                if generation_fast_path:
                    self.aam[s_i, s_j, :self.kbm1].copy_(smag_visc)
                else:
                    aam_new = self.aam.clone()

                    # 2. Apply the sliced updates on a copy.
                    aam_new[s_i, s_j, :self.kbm1] = smag_visc

                    # 3. Rebind self.aam to the updated copy.
                    self.aam = aam_new

                self.aam = refresh_ip6_scalar3d(self.aam)

                # =========================================================

                # =========================================================

                # Vertical averages of 3-D fields
                dz_w = self.dz[:self.kbm1].view(1, 1, -1)

                adx2d = torch.sum(self.advx[:, :, :self.kbm1] * dz_w, dim=2)
                ady2d = torch.sum(self.advy[:, :, :self.kbm1] * dz_w, dim=2)
                drx2d = torch.sum(self.drhox[:, :, :self.kbm1] * dz_w, dim=2)
                dry2d = torch.sum(self.drhoy[:, :, :self.kbm1] * dz_w, dim=2)
                aam2d = torch.sum(self.aam[:, :, :self.kbm1] * dz_w, dim=2)
                if generation_fast_path:
                    self.adx2d.copy_(adx2d)
                    self.ady2d.copy_(ady2d)
                    self.drx2d.copy_(drx2d)
                    self.dry2d.copy_(dry2d)
                    self.aam2d.copy_(aam2d)
                else:
                    self.adx2d = adx2d
                    self.ady2d = ady2d
                    self.drx2d = drx2d
                    self.dry2d = dry2d
                    self.aam2d = aam2d

                # Call ADVAVE
                # Passing self.psi as a temporary buffer for curv2d
                self.psi, self.advua, self.advva, self.fluxua, self.fluxva, self.wubot, self.wvbot, self.tps = \
                    self.advave(self.psi, self.advua, self.advva, self.fluxua, self.fluxva,
                                self.wubot, self.wvbot, self.tps,
                                self.mode, self.aam2d,
                                self.uab, self.vab, self.dx, self.dy, self.ua, self.va,
                                self.cbc, self.aru, self.arv, self.d)

                self.adx2d -= self.advua
                self.ady2d -= self.advva

            # -----------------------------------------------------
            # Prepare for External Mode
            # -----------------------------------------------------
            egf = self.el * self.ispi
            if generation_fast_path:
                self.egf.copy_(egf)
            else:
                self.egf = egf
            # Barotropic depth based on previous free surface (used for utf/vtf trapezoidal averaging)
            d_baro = self.h + self.elb

            d_term_u = self.d[1:self.im, :] + self.d[0:self.imm1, :]
            self.utf[1:self.im, :] = self.ua[1:self.im, :] * d_term_u * self.isp2i

            d_term_v = self.d[:, 1:self.jm] + self.d[:, 0:self.jmm1]
            self.vtf[:, 1:self.jm] = self.va[:, 1:self.jm] * d_term_v * self.isp2i
            self.utf, self.vtf = refresh_ip6_velocity2d(self.utf, self.vtf)

            # -----------------------------------------------------
            # External (2-D) Mode: STRICT SEMI-IMPLICIT (Fractional Step Method)
            # -----------------------------------------------------
            s_i = slice(1, self.imm1)
            s_j = slice(1, self.jmm1)
            s_i_m1 = slice(0, self.imm1 - 1)
            s_i_p1 = slice(2, self.imm1 + 1)
            s_j_m1 = slice(0, self.jmm1 - 1)
            s_j_p1 = slice(2, self.jmm1 + 1)

            # === STEP 2: Calculate Provisional Velocities (U*, V*) ===
            u_star, v_star, denom_u_star, denom_v_star = self._barotropic_predictor_with_bc()

            # === STEP 3: Form the RHS of the Elliptic Equation ===
            self.fluxua, self.fluxva, rhs_eta = self._barotropic_transport_rhs_full(u_star, v_star)

            # === STEP 4: Solve Implicit Surface (elf) ===
            prev_elf = getattr(self, "_eta_prev_solution", None)
            if prev_elf is not None:
                initial_guess = 2.0 * self.elf - prev_elf
            else:
                initial_guess = getattr(self, "elf", None)
            eta_solver_options = self.implicit_numerics_policy().eta_solver_options(self)
            eta_preconditioner = eta_solver_options.get("preconditioner")
            if eta_preconditioner is not None:
                self.eta_solver_preconditioner = str(eta_preconditioner)
            self.elf = self.solve_implicit_eta_cg(
                rhs_eta,
                self.d,
                max_iter=int(eta_solver_options.get("max_iter", 200)),
                tol=float(eta_solver_options.get("tol", 1.0e-12)),
                print_cg_residual=bool(eta_solver_options.get("print_cg_residual", False)),
                initial_guess=initial_guess,
            )

            self.elf = self._surface_bc_after_solver(self.elf)
            if generation_fast_path:
                prev_elf = getattr(self, "_eta_prev_solution", None)
                if prev_elf is None or prev_elf.shape != self.elf.shape or prev_elf.dtype != self.elf.dtype:
                    self._eta_prev_solution = self.elf.detach().clone()
                else:
                    prev_elf.copy_(self.elf.detach())

            # Update surface geometry
            d_new = self.h + self.elf
            if generation_fast_path:
                self.d.copy_(d_new)
            else:
                self.d = d_new

            # === STEP 5: Final Momentum Correction ===
            uaf, vaf = self._barotropic_corrector_with_bc(u_star, v_star, denom_u_star, denom_v_star)
            if generation_fast_path:
                self.uaf.copy_(uaf)
                self.vaf.copy_(vaf)
            else:
                self.uaf = uaf
                self.vaf = vaf

            # Time filter (single step: etf = elf * fsm)
            self.etf[:] = self.elf[:] * self.fsm[:]

            vamax = torch.max(torch.abs(self.vaf))

            if vamax_safe_fast_path or vamax <= self.vmaxl:
                # Robert-Asselin Time Filter
                if generation_fast_path:
                    self.ua.add_(0.5 * self.smoth * (self.uab - 2.0 * self.ua + self.uaf))
                    self.va.add_(0.5 * self.smoth * (self.vab - 2.0 * self.va + self.vaf))
                    self.el.add_(0.5 * self.smoth * (self.elb - 2.0 * self.el + self.elf))
                    self.elb.copy_(self.el)
                    self.el.copy_(self.elf)
                    d_new = self.h + self.el
                    if generation_fast_path:
                        self.d.copy_(d_new)
                    else:
                        self.d = d_new
                    self.uab.copy_(self.ua)
                    self.ua.copy_(self.uaf)
                    self.vab.copy_(self.va)
                    self.va.copy_(self.vaf)
                else:
                    self.ua = self.ua + 0.5 * self.smoth * (self.uab - 2.0 * self.ua + self.uaf)
                    self.va = self.va + 0.5 * self.smoth * (self.vab - 2.0 * self.va + self.vaf)
                    self.el = self.el + 0.5 * self.smoth * (self.elb - 2.0 * self.el + self.elf)
                    self.elb = self.el.clone()
                    self.el = self.elf.clone()
                    d_new = self.h + self.el
                    if generation_fast_path:
                        self.d.copy_(d_new)
                    else:
                        self.d = d_new
                    self.uab = self.ua.clone()
                    self.ua = self.uaf.clone()
                    self.vab = self.va.clone()
                    self.va = self.vaf.clone()
                self.elb = refresh_ip6_scalar2d(self.elb)
                self.el = refresh_ip6_scalar2d(self.el)
                self.d = refresh_ip6_scalar2d(self.d)
                self.uab, self.vab = refresh_ip6_velocity2d(self.uab, self.vab)
                self.ua, self.va = refresh_ip6_velocity2d(self.ua, self.va)
                # Single implicit step: egf, utf, vtf for baroclinic correction
                # Explicit accumulates N face transports with isp2i=1/(2N),
                # yielding the interval mean of velocity times face depth.
                # P0 fix: d_sum is the sum of the two adjacent T-point depths.
                # Multiplying by 1/2 forms the face depth and multiplying by
                # another 1/2 forms the start/end trapezoidal time mean, so
                # the coefficient on the endpoint sum is exactly 1/4.
                egf = (self.elb + self.el) * 0.5
                if generation_fast_path:
                    self.egf.copy_(egf)
                else:
                    self.egf = egf
                d_sum_u_new = self.d[1:self.im, :] + self.d[0:self.imm1, :]
                d_sum_u_baro = d_baro[1:self.im, :] + d_baro[0:self.imm1, :]
                self.utf[1:self.im, :] = barotropic_endpoint_transport(
                    self.uab[1:self.im, :],
                    d_sum_u_baro,
                    self.ua[1:self.im, :],
                    d_sum_u_new,
                )
                d_sum_v_new = self.d[:, 1:self.jm] + self.d[:, 0:self.jmm1]
                d_sum_v_baro = d_baro[:, 1:self.jm] + d_baro[:, 0:self.jmm1]
                self.vtf[:, 1:self.jm] = barotropic_endpoint_transport(
                    self.vab[:, 1:self.jm],
                    d_sum_v_baro,
                    self.va[:, 1:self.jm],
                    d_sum_v_new,
                )
                self.utf, self.vtf = refresh_ip6_velocity2d(self.utf, self.vtf)

            # -----------------------------------------------------
            # End of external mode (implicit)
            # -----------------------------------------------------

            if vamax_safe_fast_path or vamax <= self.vmaxl:
                if (iint != 1 or self.time0 != 0.0) and self.mode != 2:

                    dz_w = self.dz[:self.kbm1].view(1, 1, -1)

                    # --------------------- Fix U ---------------------
                    tps_u = torch.sum(self.u[:, :, :self.kbm1] * dz_w, dim=2)

                    dt_sum_u = self.dt[1:self.im, :] + self.dt[0:self.imm1, :]
                    correction_u = (self.utb[1:self.im, :] + self.utf[1:self.im, :]) / (dt_sum_u + 1e-20)

                    u_slice = self.u[1:self.im, :, :self.kbm1]
                    tps_slice = tps_u[1:self.im, :].unsqueeze(2)
                    corr_slice = correction_u.unsqueeze(2)

                    # 🔥 FIX: Clone u to avoid in-place modification of the graph leaf/node
                    if generation_fast_path:
                        self.u[1:self.im, :, :self.kbm1].copy_((u_slice - tps_slice) + corr_slice)
                    else:
                        u_new = self.u.clone()
                        u_new[1:self.im, :, :self.kbm1] = (u_slice - tps_slice) + corr_slice
                        self.u = u_new  # Update reference to the new tensor

                    # --------------------- Fix V ---------------------
                    tps_v = torch.sum(self.v[:, :, :self.kbm1] * dz_w, dim=2)
                    dt_sum_v = self.dt[:, 1:self.jm] + self.dt[:, 0:self.jmm1]
                    correction_v = (self.vtb[:, 1:self.jm] + self.vtf[:, 1:self.jm]) / (dt_sum_v + 1e-20)

                    v_slice = self.v[:, 1:self.jm, :self.kbm1]
                    tps_v_slice = tps_v[:, 1:self.jm].unsqueeze(2)
                    corr_v_slice = correction_v.unsqueeze(2)

                    # 🔥 FIX: Clone v to avoid in-place modification
                    if generation_fast_path:
                        self.v[:, 1:self.jm, :self.kbm1].copy_((v_slice - tps_v_slice) + corr_v_slice)
                    else:
                        v_new = self.v.clone()
                        v_new[:, 1:self.jm, :self.kbm1] = (v_slice - tps_v_slice) + corr_v_slice
                        self.v = v_new  # Update reference to the new tensor

                    self.u, self.v = refresh_ip6_velocity3d(self.u, self.v)

                    # Vertical Velocity
                    self.a, self.c, self.w = self.vertvl(
                        self.a, self.c, self.w, self.dx, self.dy, self.dz, self.dt,
                        self.u, self.v, self.vfluxb, self.vfluxf, self.etf, self.etb,
                        self.dti2, self.im, self.jm, self.imm1, self.jmm1, self.kbm1
                    )

                    # BC 5
                    self.elf, self.uaf, self.vaf, self.uf, self.vf, self.w = self.bcond(
                        5, self.elf, self.uaf, self.vaf, self.uf, self.vf, self.w
                    )

                    self.vf.zero_()
                    self.uf.zero_()

                    # Turbulence
                    self.q2b, self.q2, self.uf, self.a, self.c = self.advq(
                        self.q2b, self.q2, self.uf, self.a, self.c,
                        self.dt, self.dx, self.dy, self.dz, self.u, self.v, self.w,
                        self.aam, self.h, self.dum, self.dvm, self.art, self.etb, self.etf,
                        self.im, self.jm, self.imm1, self.jmm1, self.kbm1, self.dti2
                    )

                    self.q2lb, self.q2l, self.vf, self.a, self.c = self.advq(
                        self.q2lb, self.q2l, self.vf, self.a, self.c,
                        self.dt, self.dx, self.dy, self.dz, self.u, self.v, self.w,
                        self.aam, self.h, self.dum, self.dvm, self.art, self.etb, self.etf,
                        self.im, self.jm, self.imm1, self.jmm1, self.kbm1, self.dti2
                    )

                    self.a, self.c, self.tps, self.dtef, self.ee, self.gg, self.l, \
                    self.kq, self.km, self.kh, self.uf, self.vf, self.q2b, self.q2lb, self.a, self.c = \
                        self.profq(
                            self.a, self.c, self.tps, self.dtef, self.ee, self.gg, self.l,
                            self.kq, self.km, self.kh, self.uf, self.vf, self.q2, self.q2b, self.q2lb, self.a, self.c,
                            self.h, self.etf, self.dti2, self.umol, self.dzz, self.grav, self.rho, self.kappa,
                            self.u, self.v, self.dt, self.small, self.fsm, self.im, self.jm, self.kb,
                            self.imm1, self.jmm1, self.kbm1, self.tbias, self.sbias, self.dz,
                            self.wusurf, self.wubot, self.wvsurf, self.wvbot, self.t, self.s, self.rhoref, self.zz, self.z
                        )

                    # BC 6
                    self.elf, self.uaf, self.vaf, self.uf, self.vf, self.w = self.bcond(
                        6, self.elf, self.uaf, self.vaf, self.uf, self.vf, self.w
                    )

                    # Filter Turbulence
                    if generation_fast_path:
                        self.q2.add_(0.5 * self.smoth * (self.uf + self.q2b - 2.0 * self.q2))
                        self.q2l.add_(0.5 * self.smoth * (self.vf + self.q2lb - 2.0 * self.q2l))
                        self.q2b.copy_(self.q2)
                        self.q2.copy_(self.uf)
                        self.q2lb.copy_(self.q2l)
                        self.q2l.copy_(self.vf)
                    else:
                        self.q2 = self.q2 + 0.5 * self.smoth * (self.uf + self.q2b - 2.0 * self.q2)
                        self.q2l = self.q2l + 0.5 * self.smoth * (self.vf + self.q2lb - 2.0 * self.q2l)

                        self.q2b = self.q2.clone()
                        self.q2 = self.uf.clone()
                        self.q2lb = self.q2l.clone()
                        self.q2l = self.vf.clone()

                    for turb_field in (self.q2b, self.q2, self.q2lb, self.q2l):
                        refresh_ip6_scalar3d(turb_field)

                    vertical_mixing_max = getattr(self, "vertical_mixing_max", None)
                    if vertical_mixing_max is not None:
                        vmax_mix = float(vertical_mixing_max)
                        if generation_fast_path:
                            self.kh.clamp_(min=0.0, max=vmax_mix)
                            self.km.clamp_(min=0.0, max=vmax_mix)
                            self.kq.clamp_(min=0.0, max=vmax_mix)
                        else:
                            self.kh = torch.clamp(self.kh, min=0.0, max=vmax_mix)
                            self.km = torch.clamp(self.km, min=0.0, max=vmax_mix)
                            self.kq = torch.clamp(self.kq, min=0.0, max=vmax_mix)

                    for mix_field in (self.kh, self.km, self.kq):
                        refresh_ip6_scalar3d(mix_field)

                    # T/S
                    if self.mode != 4:
                        self.implicit_numerics_policy().advect_tracers(self)

                        self.uf, self.wtsurf, self.tsurf, self.nbct, self.tps, self.a, self.c, self.ee, self.gg = \
                            self.proft(self.uf, self.wtsurf, self.tsurf, self.nbct, self.tps, self.a, self.c, self.ee, self.gg,
                                    self.h, self.etf, self.dti2, self.dz, self.dzz, self.swrad, self.ntp,
                                    self.im, self.jm, self.kb, self.kbm1, self.kbm2, self.kh, self.umol)

                        self.vf, self.wssurf, self.ssurf, self.nbcs, self.tps, self.a, self.c, self.ee, self.gg = \
                            self.proft(self.vf, self.wssurf, self.ssurf, self.nbcs, self.tps, self.a, self.c, self.ee, self.gg,
                                    self.h, self.etf, self.dti2, self.dz, self.dzz, self.swrad, self.ntp,
                                    self.im, self.jm, self.kb, self.kbm1, self.kbm2, self.kh, self.umol)

                        self.elf, self.uaf, self.vaf, self.uf, self.vf, self.w = self.bcond(
                            4, self.elf, self.uaf, self.vaf, self.uf, self.vf, self.w
                        )

                        self.t = self.t + 0.5 * self.smoth * (self.uf + self.tb - 2.0 * self.t)
                        self.s = self.s + 0.5 * self.smoth * (self.vf + self.sb - 2.0 * self.s)

                        self.uf, self.vf = self.implicit_numerics_policy().filter_tracers(
                            self, self.uf, self.vf, generation_fast_path=generation_fast_path
                        )

                        if generation_fast_path:
                            self.tb.copy_(self.t)
                            self.t.copy_(self.uf)
                            self.sb.copy_(self.s)
                            self.s.copy_(self.vf)
                        else:
                            self.tb = self.t.clone()
                            self.t = self.uf.clone()
                            self.sb = self.s.clone()
                            self.s = self.vf.clone()

                        self.implicit_numerics_policy().refresh_tracer_ghosts(
                            self, generation_fast_path=generation_fast_path
                        )

                        self.s, self.t, self.rho = self.dens()

                    # Internal Velocity
                    self.uf = self.advu(self.advx, self.aru, self.dz, self.cor, self.dt, self.e_atmos, self.dy,
                                        self.drhox, self.h, self.dti2, self.ub, self.uf, self.u, self.v, self.w,
                                        self.im, self.jm, self.kb, self.imm1, self.jmm1, self.kbm1, self.grav,
                                        self.egf, self.egb, self.etf, self.etb)

                    self.vf = self.advv(self.advy, self.arv, self.dz, self.cor, self.dt, self.e_atmos, self.dx,
                                        self.drhoy, self.h, self.dti2, self.vb, self.vf, self.u, self.v, self.w,
                                        self.im, self.jm, self.kb, self.imm1, self.jmm1, self.kbm1, self.grav,
                                        self.egf, self.egb, self.etf, self.etb)

                    self.a, self.c, self.ee, self.gg, self.tps, self.uf, self.wubot = \
                        self.profu(self.a, self.c, self.ee, self.gg, self.tps, self.uf, self.wubot,
                                   self.etf, self.h, self.km, self.dti2, self.umol, self.dz, self.dzz,
                                   self.wusurf, self.cbc, self.dum, self.im, self.jm, self.kb,
                                   self.imm1, self.jmm1, self.kbm1, self.kbm2, self.ub, self.vb)

                    self.a, self.c, self.ee, self.gg, self.tps, self.vf, self.wvbot = \
                        self.profv(self.a, self.c, self.ee, self.gg, self.tps, self.vf, self.wvbot,
                                   self.dvm, self.dz, self.dzz, self.im, self.jm, self.kb,
                                   self.imm1, self.jmm1, self.kbm1, self.kbm2, self.km, self.cbc,
                                   self.ub, self.vb, self.umol, self.wvsurf, self.h, self.etf, self.dti2)

                    self.elf, self.uaf, self.vaf, self.uf, self.vf, self.w = self.bcond(
                        3, self.elf, self.uaf, self.vaf, self.uf, self.vf, self.w
                    )

                    # Filter U/V
                    dz_w = self.dz[:self.kbm1].view(1, 1, -1)

                    # --- U Filter ---
                    term_u = self.uf[:,:,:self.kbm1] + self.ub[:,:,:self.kbm1] - 2.0 * self.u[:,:,:self.kbm1]

                    # Store tps in a local variable to avoid mutating self.tps.
                    tps_u = torch.sum(term_u * dz_w, dim=2)
                    tps_unsqueezed = tps_u.unsqueeze(2)

                    if generation_fast_path:
                        u_filtered = self.u
                        u_filtered[:,:,:self.kbm1].copy_(
                            self.u[:,:,:self.kbm1] + 0.5 * self.smoth * (term_u - tps_unsqueezed)
                        )
                    else:
                        # Update a fresh copy instead of modifying self.u in place.
                        u_filtered = self.u.clone()
                        u_filtered[:,:,:self.kbm1] = self.u[:,:,:self.kbm1] + 0.5 * self.smoth * (term_u - tps_unsqueezed)

                    # --- V Filter ---
                    term_v = self.vf[:,:,:self.kbm1] + self.vb[:,:,:self.kbm1] - 2.0 * self.v[:,:,:self.kbm1]

                    tps_v = torch.sum(term_v * dz_w, dim=2)
                    tps_unsqueezed_v = tps_v.unsqueeze(2)

                    if generation_fast_path:
                        v_filtered = self.v
                        v_filtered[:,:,:self.kbm1].copy_(
                            self.v[:,:,:self.kbm1] + 0.5 * self.smoth * (term_v - tps_unsqueezed_v)
                        )
                    else:
                        # Create new copy for V to avoid in-place modification
                        v_filtered = self.v.clone()
                        v_filtered[:,:,:self.kbm1] = self.v[:,:,:self.kbm1] + 0.5 * self.smoth * (term_v - tps_unsqueezed_v)

                    u_filtered, v_filtered = self.implicit_numerics_policy().filter_internal_velocity(
                        self, u_filtered, v_filtered, generation_fast_path=generation_fast_path
                    )

                    # --- Shift Variables ---
                    # The current state (filtered) becomes the old state (backward) for the next step.
                    if generation_fast_path:
                        self.ub.copy_(u_filtered)
                        self.u.copy_(self.uf)
                        self.vb.copy_(v_filtered)
                        self.v.copy_(self.vf)
                    else:
                        self.ub = u_filtered
                        self.u = self.uf.clone()

                        self.vb = v_filtered
                        self.v = self.vf.clone()
                    self.ub, self.vb = refresh_ip6_velocity3d(self.ub, self.vb)
                    self.u, self.v = refresh_ip6_velocity3d(self.u, self.v)

            # Shifts
            if generation_fast_path:
                self.egb.copy_(self.egf)
                self.etb.copy_(self.et)
                self.et.copy_(self.etf)
            else:
                self.egb = self.egf.clone()
                self.etb = self.et.clone()
                self.et = self.etf.clone()
            dt_new = self.h + self.et
            if generation_fast_path:
                self.dt.copy_(dt_new)
            else:
                self.dt = dt_new
            if generation_fast_path:
                self.utb.copy_(self.utf)
                self.vtb.copy_(self.vtf)
                self.vfluxb.copy_(self.vfluxf)
            else:
                self.utb = self.utf.clone()
                self.vtb = self.vtf.clone()
                self.vfluxb = self.vfluxf.clone()

            self.egb = refresh_ip6_scalar2d(self.egb)
            self.etb = refresh_ip6_scalar2d(self.etb)
            self.et = refresh_ip6_scalar2d(self.et)
            self.dt = refresh_ip6_scalar2d(self.dt)
            self.utb, self.vtb = refresh_ip6_velocity2d(self.utb, self.vtb)

            if generation_fast_path:
                capture_interval = getattr(self, "_generation_capture_interval_steps", 0)
                capture_limit = getattr(self, "_generation_capture_limit_steps", 0)
                if capture_interval > 0 and self.iint <= capture_limit and self.iint % capture_interval == 0:
                    self._generation_reference_records.append(self._capture_generation_reference_snapshot())

            # Print
            if self.iint >= self.iswtch:
                self.iprint = int(self.prtd2 * 24.0 * 3600.0 / self.dti + 0.5)

            if (getattr(self, "runtime_diagnostics", True) and self.iint % self.iprint == 0) or ((not vamax_safe_fast_path) and vamax >= self.vmaxl):
                self.printall(self.im, self.jm, self.imm1, self.jmm1, self.iskp, self.jskp,
                              self.uab, self.vab, self.elb, self.d, self.dx, self.dy,
                              self.time, self.u, self.v, self.w, self.t, self.s, self.rho,
                              self.aam, self.km, self.kb, self.mode, self.dt, self.zz, self.z)
                if (not vamax_safe_fast_path) and vamax > self.vmaxl:
                    raise FloatingPointError(
                        f"Barotropic velocity {float(vamax)} exceeds limit {self.vmaxl} at time {self.time}"
                    )
