from __future__ import annotations

import torch


def _assign_tensor(core, name: str, value, *, generation_fast_path: bool = False) -> None:
    current = getattr(core, name)
    if generation_fast_path and torch.is_tensor(current) and current.shape == value.shape and current.dtype == value.dtype:
        current.copy_(value)
    else:
        setattr(core, name, value)


class SurfaceHeatSaltFluxForcing:
    name = "surface_heat_salt_flux"

    def initialize(self, core) -> None:
        return None

    def update_before_step(self, core, time_days: float, *, generation_fast_path: bool = False) -> None:
        s_i = slice(1, core.imm1)
        s_j = slice(1, core.jmm1)
        self._clear_inner_surface_fluxes(core, s_i, s_j, generation_fast_path=generation_fast_path)
        self._apply_vertical_surface_fluxes(core, s_i, s_j, generation_fast_path=generation_fast_path)

    def _clear_inner_surface_fluxes(self, core, s_i: slice, s_j: slice, *, generation_fast_path: bool = False) -> None:
        if not hasattr(core, "_inner_mask") or (core._inner_mask.shape != core.e_atmos.shape):
            inner_mask = torch.zeros_like(core.e_atmos, dtype=torch.bool)
            inner_mask[s_i, s_j] = True
            core._inner_mask = inner_mask

        mask = core._inner_mask
        if generation_fast_path:
            core.e_atmos.masked_fill_(mask, 0.0)
            core.vfluxf.masked_fill_(mask, 0.0)
            core.swrad.masked_fill_(mask, 0.0)
        else:
            core.e_atmos = core.e_atmos.masked_fill(mask, 0.0)
            core.vfluxf = core.vfluxf.masked_fill(mask, 0.0)
            core.swrad = core.swrad.masked_fill(mask, 0.0)

    def _apply_vertical_surface_fluxes(self, core, s_i: slice, s_j: slice, *, generation_fast_path: bool = False) -> None:
        if generation_fast_path:
            core.w[s_i, s_j, 0].copy_(core.vfluxf[s_i, s_j])
        else:
            w_new = core.w.clone()
            w_new[s_i, s_j, 0] = core.vfluxf[s_i, s_j]
            core.w = w_new

        t_surf = core.t[s_i, s_j, 0]
        tatm = t_surf + core.tbias
        delta_wts = core.vfluxf[s_i, s_j] * (tatm - t_surf - core.tbias)
        if generation_fast_path:
            core.wtsurf[s_i, s_j].add_(delta_wts)
        else:
            wtsurf_new = core.wtsurf.clone()
            wtsurf_new[s_i, s_j] = wtsurf_new[s_i, s_j] + delta_wts
            core.wtsurf = wtsurf_new

        s_surf = core.s[s_i, s_j, 0]
        wssurf_inner = core.vfluxf[s_i, s_j] * (0.0 - s_surf - core.sbias)
        if generation_fast_path:
            core.wssurf[s_i, s_j].copy_(wssurf_inner)
        else:
            wssurf_new = core.wssurf.clone()
            wssurf_new[s_i, s_j] = wssurf_inner
            core.wssurf = wssurf_new


class DoubleGyreWindForcing(SurfaceHeatSaltFluxForcing):
    name = "double_gyre_wind"

    def update_before_step(self, core, time_days: float, *, generation_fast_path: bool = False) -> None:
        tau0 = float(getattr(core, "double_gyre_wind_stress_pa", 0.1))
        kinematic_factor = tau0 / core.rhoref
        arg = 2.0 * core.pi * (core.north_e - 30.0) / 20.0
        wind_stress_x = -kinematic_factor * (1.0 - torch.cos(arg)) * core.dum
        _assign_tensor(core, "wusurf", wind_stress_x, generation_fast_path=generation_fast_path)
        if generation_fast_path:
            core.wvsurf.zero_()
        else:
            core.wvsurf = torch.zeros_like(core.wvsurf)
        core.double_gyre_wind_stress_pa = tau0
        super().update_before_step(core, time_days, generation_fast_path=generation_fast_path)
