"""Independent POM08 Jiaozhou Bay (pom_jzb_nowad) open-boundary adapter; not installed in production.
Explicit elevation (459 M2 tidal nodes + outer ghost copy), 2-D Sommerfeld radiation, and
3-D internal velocity radiation translated directly from archived pom_jzb_nowad/pom08.f bcond(1..6).
Implicit extends its elliptic solve with a fixed-node tidal lift and area-weighted CG.
"""
import types
import numpy as np
import torch

try:
    from .kernels.coastal_explicit_baroclinic import BaroclinicMixin
except ImportError:
    from neuralpom.cores.explicit.baroclinic import BaroclinicMixin
from neuralpom.cores.implicit_p0_fixed.boundary.closed_basin import ClosedBasinBoundary


def tide(c, seconds):
    return c._tide_amp * torch.cos(
        (28.98410424 * 3.1416 / 180.0) * (seconds / 3600.0)
        - c._tide_phase * 3.1416 / 180.0
    )


def _apply_outer_eta_copy(c, elf):
    # pom_jzb_nowad/pom08.f lines 3308-3323: West/East first, then South/North, then * fsm
    elf[0, :] = elf[1, :]
    elf[-1, :] = elf[-2, :]
    elf[:, 0] = elf[:, 1]
    elf[:, -1] = elf[:, -2]
    return elf * c.fsm


def bcond(c, idx, elf, uaf, vaf, uf, vf, w):
    if idx == 1:
        if c._validation_family == "explicit":
            c._tide_substep += 1
            sec = c.time * 86400.0 - c.dti + c.dte * (c._tide_substep - 1)
        else:
            sec = c.time * 86400.0 - c.dte
        target = tide(c, sec) * c.fsm[c._tide_i, c._tide_j]
        if c._validation_family == "explicit" and hasattr(c, "_coastal_tide_overwrite_sum_m3"):
            delta = (
                (target - elf[c._tide_i, c._tide_j])
                * c.art[c._tide_i, c._tide_j]
                * c.fsm[c._tide_i, c._tide_j]
            ).sum()
            c._coastal_tide_overwrite_last_m3.copy_(delta)
            c._coastal_tide_overwrite_sum_m3.add_(delta)
        elf[c._tide_i, c._tide_j] = target
        elf = _apply_outer_eta_copy(c, elf)
        return elf, uaf, vaf, uf, vf, w

    if idx == 2:
        j = slice(1, -1)
        i = slice(1, -1)
        ramp = float(c.ramp)
        # East (im)
        mask_e = c.d[-1, j] > 0.0
        gae_e = c.dte * torch.sqrt(torch.clamp(c.grav * c.d[-1, j], min=0.0)) / c.dx[-1, j]
        uaf[-1, j] = torch.where(
            mask_e, (gae_e * c.ua[-2, j] + (1.0 - gae_e) * c.ua[-1, j]) * ramp, uaf[-1, j]
        )
        vaf[-1, j] = torch.where(mask_e, torch.zeros_like(vaf[-1, j]), vaf[-1, j])
        # West (2)
        mask_w = c.d[1, j] > 0.0
        gae_w = c.dte * torch.sqrt(torch.clamp(c.grav * c.d[1, j], min=0.0)) / c.dx[1, j]
        uaf[1, j] = torch.where(
            mask_w, ramp * (gae_w * c.ua[2, j] + (1.0 - gae_w) * c.ua[1, j]), uaf[1, j]
        )
        vaf[0, j] = torch.where(mask_w, torch.zeros_like(vaf[0, j]), vaf[0, j])
        # North (jm)
        mask_n = c.d[i, -1] > 0.0
        gae_n = c.dte * torch.sqrt(torch.clamp(c.grav * c.d[i, -1], min=0.0)) / c.dy[i, -1]
        vaf[i, -1] = torch.where(
            mask_n, (gae_n * c.va[i, -2] + (1.0 - gae_n) * c.va[i, -1]) * ramp, vaf[i, -1]
        )
        uaf[i, -1] = torch.where(mask_n, torch.zeros_like(uaf[i, -1]), uaf[i, -1])
        # South (2)
        mask_s = c.d[i, 1] > 0.0
        gae_s = c.dte * torch.sqrt(torch.clamp(c.grav * c.d[i, 1], min=0.0)) / c.dy[i, 1]
        vaf[i, 1] = torch.where(
            mask_s, (gae_s * c.va[i, 2] + (1.0 - gae_s) * c.va[i, 1]) * ramp, vaf[i, 1]
        )
        uaf[i, 0] = torch.where(mask_s, torch.zeros_like(uaf[i, 0]), uaf[i, 0])
        return elf, uaf * c.dum, vaf * c.dvm, uf, vf, w

    if idx == 3:
        k = slice(0, c.kbm1)
        j = slice(1, c.jm - 1)
        jm1 = slice(0, c.jm - 2)
        jp1 = slice(2, c.jm)
        i = slice(1, c.im - 1)
        im1 = slice(0, c.im - 2)
        ip1 = slice(2, c.im)
        ramp = float(c.ramp)

        # East (im)
        mask_e = (c.d[-1, j] > 0.0).unsqueeze(1)
        ga_e = torch.sqrt(torch.clamp(c.d[-1, j] / c.hmax, min=0.0)).unsqueeze(1)
        u_in_e = 0.25 * c.u[-2, jm1, k] + 0.50 * c.u[-2, j, k] + 0.25 * c.u[-2, jp1, k]
        u_out_e = 0.25 * c.u[-1, jm1, k] + 0.50 * c.u[-1, j, k] + 0.25 * c.u[-1, jp1, k]
        uf[-1, j, k] = torch.where(
            mask_e, (ga_e * u_in_e + (1.0 - ga_e) * u_out_e) * ramp, uf[-1, j, k]
        )
        vf[-1, j, k] = torch.where(mask_e, torch.zeros_like(vf[-1, j, k]), vf[-1, j, k])

        # West (2) - pom08.f line 3408 uses d(1,j) i.e. d[0,j]
        mask_w = (c.d[0, j] > 0.0).unsqueeze(1)
        ga_w = torch.sqrt(torch.clamp(c.d[0, j] / c.hmax, min=0.0)).unsqueeze(1)
        u_in_w = 0.25 * c.u[2, jm1, k] + 0.50 * c.u[2, j, k] + 0.25 * c.u[2, jp1, k]
        u_out_w = 0.25 * c.u[1, jm1, k] + 0.50 * c.u[1, j, k] + 0.25 * c.u[1, jp1, k]
        uf_w = (ga_w * u_in_w + (1.0 - ga_w) * u_out_w) * ramp
        uf[1, j, k] = torch.where(mask_w, uf_w, uf[1, j, k])
        uf[0, j, k] = torch.where(mask_w, uf[1, j, k], uf[0, j, k])
        vf[0, j, k] = torch.where(mask_w, torch.zeros_like(vf[0, j, k]), vf[0, j, k])

        # North (jm) - pom08.f line 3425 does not multiply by ramp
        mask_n = (c.d[i, -1] > 0.0).unsqueeze(1)
        ga_n = torch.sqrt(torch.clamp(c.d[i, -1] / c.hmax, min=0.0)).unsqueeze(1)
        v_in_n = 0.25 * c.v[im1, -2, k] + 0.50 * c.v[i, -2, k] + 0.25 * c.v[ip1, -2, k]
        v_out_n = 0.25 * c.v[im1, -1, k] + 0.50 * c.v[i, -1, k] + 0.25 * c.v[ip1, -1, k]
        vf[i, -1, k] = torch.where(
            mask_n, ga_n * v_in_n + (1.0 - ga_n) * v_out_n, vf[i, -1, k]
        )
        uf[i, -1, k] = torch.where(mask_n, torch.zeros_like(uf[i, -1, k]), uf[i, -1, k])

        # South (2) - pom08.f line 3437 uses d(i,2) i.e. d[i,1] and multiplies by ramp
        mask_s = (c.d[i, 1] > 0.0).unsqueeze(1)
        ga_s = torch.sqrt(torch.clamp(c.d[i, 1] / c.hmax, min=0.0)).unsqueeze(1)
        v_in_s = 0.25 * c.v[im1, 2, k] + 0.50 * c.v[i, 2, k] + 0.25 * c.v[ip1, 2, k]
        v_out_s = 0.25 * c.v[im1, 1, k] + 0.50 * c.v[i, 1, k] + 0.25 * c.v[ip1, 1, k]
        vf_s = (ga_s * v_in_s + (1.0 - ga_s) * v_out_s) * ramp
        vf[i, 1, k] = torch.where(mask_s, vf_s, vf[i, 1, k])
        vf[i, 0, k] = torch.where(mask_s, vf[i, 1, k], vf[i, 0, k])
        uf[i, 0, k] = torch.where(mask_s, torch.zeros_like(uf[i, 0, k]), uf[i, 0, k])

        uf[:, :, k] = uf[:, :, k] * c.dum.unsqueeze(2)
        vf[:, :, k] = vf[:, :, k] * c.dvm.unsqueeze(2)
        return elf, uaf, vaf, uf, vf, w

    return BaroclinicMixin.bcond(c, idx, elf, uaf, vaf, uf, vf, w)


class Boundary(ClosedBasinBoundary):
    name = "independent_original_pom08_jzb_tide_adapter"

    def apply_bcond(self, c, *args):
        return bcond(c, *args)

    def enforce_eta(self, c, x):
        x[c._fixed] = 0.0
        x[c._tide_i, c._tide_j] = tide(c, c.time * 86400.0 - c.dte) * c.fsm[c._tide_i, c._tide_j]
        _apply_outer_eta_copy(c, x)

    def prepare_eta_rhs(self, c, rhs):
        return None


def solve(c, rhs, d_2d, max_iter=200, tol=1e-12, print_cg_residual=False, initial_guess=None):
    # Boundary nodes are not unknowns. Subtract the tidal lift before solving
    # the homogeneous problem; weight by cell area for the flux-form operator.
    lift = torch.zeros_like(rhs)
    if hasattr(c, "_tide_i"):
        lift[c._tide_i, c._tide_j] = (
            tide(c, c.time * 86400.0 - c.dte) * c.fsm[c._tide_i, c._tide_j]
        )
        lift = _apply_outer_eta_copy(c, lift)
    free = ~c._fixed
    weight = c.art * free

    def op(v):
        v = v.masked_fill(c._fixed, 0.0)
        return c.apply_elliptic_operator(v, d_2d).masked_fill(c._fixed, 0.0)

    b = (rhs - c.apply_elliptic_operator(lift, d_2d)).masked_fill(c._fixed, 0.0)
    x = torch.zeros_like(rhs)
    r = b.clone()
    p = r.clone()
    rr = (weight * r * r).sum()
    rr0 = rr
    it = 0
    for it in range(min(max_iter, 500)):
        if float(rr) <= max(1e-30, float(rr0) * 1e-24):
            break
        Ap = op(p)
        den = (weight * p * Ap).sum()
        a = rr / (den + 1e-40)
        x = x + a * p
        r = r - a * Ap
        new = (weight * r * r).sum()
        p = r + (new / (rr + 1e-40)) * p
        rr = new
    c._validation_eta_iterations = it + 1
    c._validation_eta_relative_residual = float(torch.sqrt(rr / (rr0 + 1e-40)))
    return _apply_outer_eta_copy(c, x + lift)


def install(c, family, path):
    data = np.loadtxt(path, max_rows=459)
    c._validation_family = family
    c._tide_substep = 0
    c._tide_i = torch.as_tensor(data[:, 0] - 1, dtype=torch.long, device=c.device)
    c._tide_j = torch.as_tensor(data[:, 1] - 1, dtype=torch.long, device=c.device)
    c._tide_amp = torch.as_tensor(data[:, 2], dtype=c.el.dtype, device=c.device)
    c._tide_phase = torch.as_tensor(data[:, 3], dtype=c.el.dtype, device=c.device)
    c.rfe = c.rfw = c.rfn = c.rfs = 1.0
    c._fixed = torch.ones_like(c.fsm, dtype=torch.bool)
    c._fixed[1:-1, 1:-1] = False
    c._fixed |= c.fsm == 0
    c._fixed[c._tide_i, c._tide_j] = True
    if family == "explicit":
        c.bcond = types.MethodType(bcond, c)
    else:
        c._implicit_boundary_policy = Boundary()
        c.solve_implicit_eta_cg = types.MethodType(solve, c)
