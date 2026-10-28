from __future__ import annotations

import torch


class DoubleGyreNumerics:
    name = "double_gyre"

    def initialize(self, core) -> None:
        return None

    def apply_surface_flux_defaults(self, core, s_i: slice, s_j: slice, *, generation_fast_path: bool = False) -> None:
        if not hasattr(core, "_inner_mask") or (core._inner_mask.shape != core.e_atmos.shape):
            inner_mask = torch.zeros_like(core.e_atmos, dtype=torch.bool)
            inner_mask[s_i, s_j] = True
            core._inner_mask = inner_mask

        m = core._inner_mask
        if generation_fast_path:
            core.e_atmos.masked_fill_(m, 0.0)
            core.vfluxf.masked_fill_(m, 0.0)
            core.swrad.masked_fill_(m, 0.0)
        else:
            core.e_atmos = core.e_atmos.masked_fill(m, 0.0)
            core.vfluxf = core.vfluxf.masked_fill(m, 0.0)
            core.swrad = core.swrad.masked_fill(m, 0.0)

    def apply_pre_step_filters(self, core) -> None:
        return None

    def apply_post_tracer_filters(self, core) -> None:
        return None

    def refresh_tracer_ghosts(self, core, *, generation_fast_path: bool = False) -> None:
        core.apply_implicit_tracer_boundary()
        if core.implicit_boundary_refreshes_tracer_backups():
            if generation_fast_path:
                core.tb.copy_(core.t)
                core.sb.copy_(core.s)
            else:
                core.tb = core.t.clone()
                core.sb = core.s.clone()

    def advect_tracers(self, core) -> None:
        if core.nadv == 1:
            core.tb, core.t, core.tclim, core.uf, core.a, core.c, core.zflux = core.advt1(
                core.tb, core.t, core.tclim, core.uf, core.a, core.c, core.zflux,
                core.dt, core.u, core.v, core.aam, core.tprni, core.dum, core.dvm, core.w,
                core.art, core.etb, core.etf, core.dti2, core.dx, core.dy, core.dz, core.h,
                core.im, core.jm, core.kb, core.imm1, core.jmm1, core.kbm1
            )
            core.sb, core.s, core.sclim, core.vf, core.a, core.c, core.zflux = core.advt1(
                core.sb, core.s, core.sclim, core.vf, core.a, core.c, core.zflux,
                core.dt, core.u, core.v, core.aam, core.tprni, core.dum, core.dvm, core.w,
                core.art, core.etb, core.etf, core.dti2, core.dx, core.dy, core.dz, core.h,
                core.im, core.jm, core.kb, core.imm1, core.jmm1, core.kbm1
            )
            return

        if core.nadv == 2:
            core.tb, core.t, core.tclim, core.uf, core.a, core.c, core.nitera, core.sw, core.zflux = \
                core.advt2(core.tb, core.t, core.tclim, core.uf, core.a, core.c, core.nitera, core.sw, core.zflux,
                           core.im, core.jm, core.kb, core.imm1, core.jmm1, core.kbm1, core.dti2,
                           core.etb, core.etf, core.w, core.art, core.fsm, core.dt, core.aam, core.tprni,
                           core.h, core.dum, core.dvm, core.dx, core.dy, core.u, core.v, core.aru, core.arv,
                           core.dz, core.dzz)

            core.sb, core.s, core.sclim, core.vf, core.a, core.c, core.nitera, core.sw, core.zflux = \
                core.advt2(core.sb, core.s, core.sclim, core.vf, core.a, core.c, core.nitera, core.sw, core.zflux,
                           core.im, core.jm, core.kb, core.imm1, core.jmm1, core.kbm1, core.dti2,
                           core.etb, core.etf, core.w, core.art, core.fsm, core.dt, core.aam, core.tprni,
                           core.h, core.dum, core.dvm, core.dx, core.dy, core.u, core.v, core.aru, core.arv,
                           core.dz, core.dzz)
            return

        raise ValueError(f"Unsupported implicit tracer advection scheme nadv={core.nadv}")

    def filter_tracers(self, core, t_field, s_field, *, generation_fast_path: bool = False):
        gamma_ts = float(getattr(core, "tracer_shapiro", 0.05))
        if generation_fast_path:
            t_shapiro = t_field
            s_shapiro = s_field
        else:
            t_shapiro = t_field.clone()
            s_shapiro = s_field.clone()

        s_i_in = slice(1, core.im - 1)
        s_j_in = slice(1, core.jm - 1)
        s_k = slice(0, core.kbm1)

        if generation_fast_path:
            if gamma_ts > 0.0:
                t_shapiro[s_i_in, s_j_in, s_k].copy_(
                    t_field[s_i_in, s_j_in, s_k] + (gamma_ts / 4.0) * (
                        t_field[2:core.im, s_j_in, s_k]
                        + t_field[0:core.im - 2, s_j_in, s_k]
                        + t_field[s_i_in, 2:core.jm, s_k]
                        + t_field[s_i_in, 0:core.jm - 2, s_k]
                        - 4.0 * t_field[s_i_in, s_j_in, s_k]
                    )
                )
                s_shapiro[s_i_in, s_j_in, s_k].copy_(
                    s_field[s_i_in, s_j_in, s_k] + (gamma_ts / 4.0) * (
                        s_field[2:core.im, s_j_in, s_k]
                        + s_field[0:core.im - 2, s_j_in, s_k]
                        + s_field[s_i_in, 2:core.jm, s_k]
                        + s_field[s_i_in, 0:core.jm - 2, s_k]
                        - 4.0 * s_field[s_i_in, s_j_in, s_k]
                    )
                )
            t_field.mul_(core.fsm.unsqueeze(2))
            s_field.mul_(core.fsm.unsqueeze(2))
            return t_field, s_field

        if gamma_ts > 0.0:
            t_shapiro[s_i_in, s_j_in, s_k] = t_field[s_i_in, s_j_in, s_k] + (gamma_ts / 4.0) * (
                t_field[2:core.im, s_j_in, s_k]
                + t_field[0:core.im - 2, s_j_in, s_k]
                + t_field[s_i_in, 2:core.jm, s_k]
                + t_field[s_i_in, 0:core.jm - 2, s_k]
                - 4.0 * t_field[s_i_in, s_j_in, s_k]
            )
            s_shapiro[s_i_in, s_j_in, s_k] = s_field[s_i_in, s_j_in, s_k] + (gamma_ts / 4.0) * (
                s_field[2:core.im, s_j_in, s_k]
                + s_field[0:core.im - 2, s_j_in, s_k]
                + s_field[s_i_in, 2:core.jm, s_k]
                + s_field[s_i_in, 0:core.jm - 2, s_k]
                - 4.0 * s_field[s_i_in, s_j_in, s_k]
            )
        return t_shapiro * core.fsm.unsqueeze(2), s_shapiro * core.fsm.unsqueeze(2)

    def filter_internal_velocity(self, core, u_field, v_field, *, generation_fast_path: bool = False):
        gamma_3d = float(getattr(core, "velocity_shapiro", 0.05))
        if generation_fast_path:
            u_shapiro = u_field
            v_shapiro = v_field
        else:
            u_shapiro = u_field.clone()
            v_shapiro = v_field.clone()

        s_i_in = slice(1, core.im - 1)
        s_j_in = slice(1, core.jm - 1)
        s_k = slice(0, core.kbm1)

        if generation_fast_path:
            if gamma_3d > 0.0:
                u_shapiro[s_i_in, s_j_in, s_k].copy_(
                    u_field[s_i_in, s_j_in, s_k] + (gamma_3d / 4.0) * (
                        u_field[2:core.im, s_j_in, s_k]
                        + u_field[0:core.im - 2, s_j_in, s_k]
                        + u_field[s_i_in, 2:core.jm, s_k]
                        + u_field[s_i_in, 0:core.jm - 2, s_k]
                        - 4.0 * u_field[s_i_in, s_j_in, s_k]
                    )
                )
                v_shapiro[s_i_in, s_j_in, s_k].copy_(
                    v_field[s_i_in, s_j_in, s_k] + (gamma_3d / 4.0) * (
                        v_field[2:core.im, s_j_in, s_k]
                        + v_field[0:core.im - 2, s_j_in, s_k]
                        + v_field[s_i_in, 2:core.jm, s_k]
                        + v_field[s_i_in, 0:core.jm - 2, s_k]
                        - 4.0 * v_field[s_i_in, s_j_in, s_k]
                    )
                )
            u_field.mul_(core.dum.unsqueeze(2))
            v_field.mul_(core.dvm.unsqueeze(2))
            return u_field, v_field

        if gamma_3d > 0.0:
            u_shapiro[s_i_in, s_j_in, s_k] = u_field[s_i_in, s_j_in, s_k] + (gamma_3d / 4.0) * (
                u_field[2:core.im, s_j_in, s_k]
                + u_field[0:core.im - 2, s_j_in, s_k]
                + u_field[s_i_in, 2:core.jm, s_k]
                + u_field[s_i_in, 0:core.jm - 2, s_k]
                - 4.0 * u_field[s_i_in, s_j_in, s_k]
            )
            v_shapiro[s_i_in, s_j_in, s_k] = v_field[s_i_in, s_j_in, s_k] + (gamma_3d / 4.0) * (
                v_field[2:core.im, s_j_in, s_k]
                + v_field[0:core.im - 2, s_j_in, s_k]
                + v_field[s_i_in, 2:core.jm, s_k]
                + v_field[s_i_in, 0:core.jm - 2, s_k]
                - 4.0 * v_field[s_i_in, s_j_in, s_k]
            )
        return u_shapiro * core.dum.unsqueeze(2), v_shapiro * core.dvm.unsqueeze(2)

    def eta_solver_options(self, core) -> dict[str, object]:
        return {
            "max_iter": getattr(core, "eta_solver_max_iter", 200),
            "tol": getattr(core, "eta_solver_tol", 1.0e-12),
            "preconditioner": getattr(core, "eta_solver_preconditioner", "jacobi"),
        }
