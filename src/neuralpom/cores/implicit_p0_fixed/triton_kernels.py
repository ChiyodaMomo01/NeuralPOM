from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised on systems without Triton.
    triton = None
    tl = None


def triton_available() -> bool:
    return triton is not None and tl is not None


if triton_available():

    @triton.jit
    def _elliptic_operator_kernel(
        eta,
        d_2d,
        out,
        dy_over_dx_east,
        dy_over_dx_west,
        dx_over_dy_north,
        dx_over_dy_south,
        mask_u_east,
        mask_u_west,
        mask_v_north,
        mask_v_south,
        inv_art,
        coeff: tl.constexpr,
        im: tl.constexpr,
        jm: tl.constexpr,
        inner_i: tl.constexpr,
        inner_j: tl.constexpr,
        n_total: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        valid = offsets < n_total
        i = offsets // jm
        j = offsets - i * jm
        interior = valid & (i > 0) & (i < im - 1) & (j > 0) & (j < jm - 1)

        eta_c = tl.load(eta + offsets, mask=valid, other=0.0)
        d_c = tl.load(d_2d + offsets, mask=interior, other=0.0)
        east = offsets + jm
        west = offsets - jm
        north = offsets + 1
        south = offsets - 1

        d_e = tl.load(d_2d + east, mask=interior, other=0.0)
        d_w = tl.load(d_2d + west, mask=interior, other=0.0)
        d_n = tl.load(d_2d + north, mask=interior, other=0.0)
        d_s = tl.load(d_2d + south, mask=interior, other=0.0)

        eta_e = tl.load(eta + east, mask=interior, other=0.0)
        eta_w = tl.load(eta + west, mask=interior, other=0.0)
        eta_n = tl.load(eta + north, mask=interior, other=0.0)
        eta_s = tl.load(eta + south, mask=interior, other=0.0)

        cache_idx = (i - 1) * inner_j + (j - 1)
        h_u_east = 0.5 * (d_e + d_c)
        h_u_west = 0.5 * (d_c + d_w)
        h_v_north = 0.5 * (d_n + d_c)
        h_v_south = 0.5 * (d_c + d_s)

        flux_x_east = (
            h_u_east
            * (eta_e - eta_c)
            * tl.load(dy_over_dx_east + cache_idx, mask=interior, other=0.0)
            * tl.load(mask_u_east + cache_idx, mask=interior, other=0.0)
        )
        flux_x_west = (
            h_u_west
            * (eta_c - eta_w)
            * tl.load(dy_over_dx_west + cache_idx, mask=interior, other=0.0)
            * tl.load(mask_u_west + cache_idx, mask=interior, other=0.0)
        )
        flux_y_north = (
            h_v_north
            * (eta_n - eta_c)
            * tl.load(dx_over_dy_north + cache_idx, mask=interior, other=0.0)
            * tl.load(mask_v_north + cache_idx, mask=interior, other=0.0)
        )
        flux_y_south = (
            h_v_south
            * (eta_c - eta_s)
            * tl.load(dx_over_dy_south + cache_idx, mask=interior, other=0.0)
            * tl.load(mask_v_south + cache_idx, mask=interior, other=0.0)
        )
        div_flux = (flux_x_east - flux_x_west + flux_y_north - flux_y_south) * tl.load(
            inv_art + cache_idx,
            mask=interior,
            other=0.0,
        )
        result = tl.where(interior, eta_c - coeff * div_flux, eta_c)
        tl.store(out + offsets, result, mask=valid)


    @triton.jit
    def _elliptic_operator_batch_kernel(
        eta,
        d_2d,
        out,
        dy_over_dx_east,
        dy_over_dx_west,
        dx_over_dy_north,
        dx_over_dy_south,
        mask_u_east,
        mask_u_west,
        mask_v_north,
        mask_v_south,
        inv_art,
        coeff: tl.constexpr,
        im: tl.constexpr,
        jm: tl.constexpr,
        inner_j: tl.constexpr,
        plane_size: tl.constexpr,
        n_total: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        valid = offsets < n_total
        b = offsets // plane_size
        rem = offsets - b * plane_size
        i = rem // jm
        j = rem - i * jm
        plane_base = b * plane_size
        interior = valid & (i > 0) & (i < im - 1) & (j > 0) & (j < jm - 1)

        eta_c = tl.load(eta + offsets, mask=valid, other=0.0)
        d_c = tl.load(d_2d + offsets, mask=interior, other=0.0)
        east = plane_base + (i + 1) * jm + j
        west = plane_base + (i - 1) * jm + j
        north = plane_base + i * jm + (j + 1)
        south = plane_base + i * jm + (j - 1)

        d_e = tl.load(d_2d + east, mask=interior, other=0.0)
        d_w = tl.load(d_2d + west, mask=interior, other=0.0)
        d_n = tl.load(d_2d + north, mask=interior, other=0.0)
        d_s = tl.load(d_2d + south, mask=interior, other=0.0)

        eta_e = tl.load(eta + east, mask=interior, other=0.0)
        eta_w = tl.load(eta + west, mask=interior, other=0.0)
        eta_n = tl.load(eta + north, mask=interior, other=0.0)
        eta_s = tl.load(eta + south, mask=interior, other=0.0)

        cache_idx = (i - 1) * inner_j + (j - 1)
        h_u_east = 0.5 * (d_e + d_c)
        h_u_west = 0.5 * (d_c + d_w)
        h_v_north = 0.5 * (d_n + d_c)
        h_v_south = 0.5 * (d_c + d_s)

        flux_x_east = (
            h_u_east
            * (eta_e - eta_c)
            * tl.load(dy_over_dx_east + cache_idx, mask=interior, other=0.0)
            * tl.load(mask_u_east + cache_idx, mask=interior, other=0.0)
        )
        flux_x_west = (
            h_u_west
            * (eta_c - eta_w)
            * tl.load(dy_over_dx_west + cache_idx, mask=interior, other=0.0)
            * tl.load(mask_u_west + cache_idx, mask=interior, other=0.0)
        )
        flux_y_north = (
            h_v_north
            * (eta_n - eta_c)
            * tl.load(dx_over_dy_north + cache_idx, mask=interior, other=0.0)
            * tl.load(mask_v_north + cache_idx, mask=interior, other=0.0)
        )
        flux_y_south = (
            h_v_south
            * (eta_c - eta_s)
            * tl.load(dx_over_dy_south + cache_idx, mask=interior, other=0.0)
            * tl.load(mask_v_south + cache_idx, mask=interior, other=0.0)
        )
        div_flux = (flux_x_east - flux_x_west + flux_y_north - flux_y_south) * tl.load(
            inv_art + cache_idx,
            mask=interior,
            other=0.0,
        )
        result = tl.where(interior, eta_c - coeff * div_flux, eta_c)
        tl.store(out + offsets, result, mask=valid)


    @triton.jit
    def _cg_update_x_r_dirichlet_kernel(
        x,
        r,
        p,
        ap,
        alpha_ptr,
        im: tl.constexpr,
        jm: tl.constexpr,
        n_total: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        valid = offsets < n_total
        i = offsets // jm
        j = offsets - i * jm
        boundary = valid & ((i == 0) | (i == im - 1) | (j == 0) | (j == jm - 1))
        alpha = tl.load(alpha_ptr)
        x_old = tl.load(x + offsets, mask=valid, other=0.0)
        r_old = tl.load(r + offsets, mask=valid, other=0.0)
        p_val = tl.load(p + offsets, mask=valid, other=0.0)
        ap_val = tl.load(ap + offsets, mask=valid, other=0.0)
        x_new = tl.where(boundary, 0.0, x_old + alpha * p_val)
        r_new = r_old - alpha * ap_val
        tl.store(x + offsets, x_new, mask=valid)
        tl.store(r + offsets, r_new, mask=valid)


    @triton.jit
    def _cg_update_p_dirichlet_kernel(
        p,
        z,
        p_old,
        beta_ptr,
        continue_ptr,
        im: tl.constexpr,
        jm: tl.constexpr,
        n_total: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        valid = offsets < n_total
        i = offsets // jm
        j = offsets - i * jm
        boundary = valid & ((i == 0) | (i == im - 1) | (j == 0) | (j == jm - 1))
        beta = tl.load(beta_ptr)
        do_continue = tl.load(continue_ptr)
        z_val = tl.load(z + offsets, mask=valid, other=0.0)
        p_old_val = tl.load(p_old + offsets, mask=valid, other=0.0)
        candidate = z_val + beta * p_old_val
        p_new = tl.where(do_continue, candidate, p_old_val)
        p_new = tl.where(boundary, 0.0, p_new)
        tl.store(p + offsets, p_new, mask=valid)


def apply_elliptic_operator_triton(core, eta: torch.Tensor, d_2d: torch.Tensor, out: torch.Tensor) -> bool:
    if not triton_available() or not eta.is_cuda or not d_2d.is_cuda or not out.is_cuda:
        return False
    cache = core._accel_cache
    required = (
        "dy_over_dx_east",
        "dy_over_dx_west",
        "dx_over_dy_north",
        "dx_over_dy_south",
        "mask_u_east",
        "mask_u_west",
        "mask_v_north",
        "mask_v_south",
        "inv_art",
    )
    if any(not torch.is_tensor(cache.get(name)) or not cache[name].is_cuda for name in required):
        return False

    im = int(core.im)
    jm = int(core.jm)
    n_total = im * jm
    block_size = 256
    grid = (triton.cdiv(n_total, block_size),)
    _elliptic_operator_kernel[grid](
        eta,
        d_2d,
        out,
        cache["dy_over_dx_east"],
        cache["dy_over_dx_west"],
        cache["dx_over_dy_north"],
        cache["dx_over_dy_south"],
        cache["mask_u_east"],
        cache["mask_u_west"],
        cache["mask_v_north"],
        cache["mask_v_south"],
        cache["inv_art"],
        float(cache["implicit_coeff"]),
        im,
        jm,
        im - 2,
        jm - 2,
        n_total,
        block_size,
        # Match the eager stencil rounding; FMA changes CG trajectories.
        enable_fp_fusion=False,
    )
    return True


def apply_elliptic_operator_batch_triton(core, eta: torch.Tensor, d_2d: torch.Tensor, out: torch.Tensor) -> bool:
    if not triton_available() or eta.dim() != 3 or d_2d.dim() != 3 or out.dim() != 3:
        return False
    if not eta.is_cuda or not d_2d.is_cuda or not out.is_cuda:
        return False
    if eta.shape != d_2d.shape or eta.shape != out.shape:
        return False
    batch, im, jm = eta.shape
    if int(im) != int(core.im) or int(jm) != int(core.jm):
        return False
    cache = core._accel_cache
    required = (
        "dy_over_dx_east",
        "dy_over_dx_west",
        "dx_over_dy_north",
        "dx_over_dy_south",
        "mask_u_east",
        "mask_u_west",
        "mask_v_north",
        "mask_v_south",
        "inv_art",
    )
    if any(not torch.is_tensor(cache.get(name)) or not cache[name].is_cuda for name in required):
        return False

    eta_c = eta.contiguous()
    d_c = d_2d.contiguous()
    plane_size = int(im) * int(jm)
    n_total = int(batch) * plane_size
    block_size = 256
    grid = (triton.cdiv(n_total, block_size),)
    _elliptic_operator_batch_kernel[grid](
        eta_c,
        d_c,
        out,
        cache["dy_over_dx_east"],
        cache["dy_over_dx_west"],
        cache["dx_over_dy_north"],
        cache["dx_over_dy_south"],
        cache["mask_u_east"],
        cache["mask_u_west"],
        cache["mask_v_north"],
        cache["mask_v_south"],
        cache["inv_art"],
        float(cache["implicit_coeff"]),
        int(im),
        int(jm),
        int(jm) - 2,
        plane_size,
        n_total,
        block_size,
        # Match the eager stencil rounding; FMA changes CG trajectories.
        enable_fp_fusion=False,
    )
    return True


def cg_update_x_r_dirichlet_triton(core, x: torch.Tensor, r: torch.Tensor, p: torch.Tensor, ap: torch.Tensor, alpha: torch.Tensor) -> bool:
    if not triton_available() or not x.is_cuda or not r.is_cuda or not p.is_cuda or not ap.is_cuda:
        return False
    if not torch.is_tensor(alpha) or not alpha.is_cuda:
        return False
    im = int(core.im)
    jm = int(core.jm)
    n_total = im * jm
    block_size = 256
    grid = (triton.cdiv(n_total, block_size),)
    _cg_update_x_r_dirichlet_kernel[grid](x, r, p, ap, alpha, im, jm, n_total, block_size)
    return True


def cg_update_p_dirichlet_triton(
    core,
    p: torch.Tensor,
    z: torch.Tensor,
    p_old: torch.Tensor,
    beta: torch.Tensor,
    continue_mask: torch.Tensor,
) -> bool:
    if not triton_available() or not p.is_cuda or not z.is_cuda or not p_old.is_cuda:
        return False
    if not torch.is_tensor(beta) or not beta.is_cuda or not torch.is_tensor(continue_mask) or not continue_mask.is_cuda:
        return False
    im = int(core.im)
    jm = int(core.jm)
    n_total = im * jm
    block_size = 256
    grid = (triton.cdiv(n_total, block_size),)
    _cg_update_p_dirichlet_kernel[grid](p, z, p_old, beta, continue_mask, im, jm, n_total, block_size)
    return True
