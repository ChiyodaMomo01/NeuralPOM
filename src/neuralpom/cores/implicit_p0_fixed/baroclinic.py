import math
import numpy as np
try:
    import netCDF4 as nc
except ModuleNotFoundError:
    nc = None
import torch
import torch.nn.functional as F


torch.set_default_dtype(torch.float32)


class BaroclinicMixin:
    def advct(self, a, c, ee, advx, advy, dt, u, v, w, ub, vb, 
              dti2, dx, dy, dz, h, dum, dvm, art, aru, arv, 
              cbc, uab, vab, im, jm, kb, imm1, jmm1, kbm1, 
              drx2d, dry2d, cor, curv, aam):
        """
        Calculates the horizontal portions of momentum advection.
        [FIXED for Autograd]: 
        1. Localized output tensors (xflux, yflux, curv, advx, advy) to avoid in-place modification of class attributes.
        2. Ensures 'curv' and fluxes are fresh tensors for each step.
        """
        
        # Helper for 2D -> 3D broadcasting
        def _b(tensor_2d):
            return tensor_2d.unsqueeze(2)

        # =========================================================
        # 🔥 FIX: Localize Variables (Create Fresh Tensors)
        # =========================================================
        # Instead of using passed-in self.advx/self.curv which carry history/versions,
        # we create fresh tensors for the current time step's calculation.
        
        # Initialize Output/Scratchpad Tensors locally
        curv = torch.zeros_like(u)
        xflux = torch.zeros_like(u)
        yflux = torch.zeros_like(u) # Re-used for both U and V fluxes, reset in between
        advx = torch.zeros_like(u)
        advy = torch.zeros_like(v)

        # Slice limits for K loop (0 to kbm1-1)
        k_sl = slice(0, kbm1)

        # =================================================================
        # 1. Calculate Curvature (curv)
        # =================================================================
        
        s_i = slice(1, imm1)      # i
        s_j = slice(1, jmm1)      # j
        
        s_ip1 = slice(2, imm1+1)  # i+1
        s_im1 = slice(0, imm1-1)  # i-1
        s_jp1 = slice(2, jmm1+1)  # j+1
        s_jm1 = slice(0, jmm1-1)  # j-1

        # Term 1: (v(i,j+1)+v(i,j)) * (dy(i+1,j)-dy(i-1,j))
        term1 = (v[s_i, s_jp1, k_sl] + v[s_i, s_j, k_sl]) * \
                _b(dy[s_ip1, s_j] - dy[s_im1, s_j])
        
        # Term 2: (u(i+1,j)+u(i,j)) * (dx(i,j+1)-dx(i,j-1))
        term2 = (u[s_ip1, s_j, k_sl] + u[s_i, s_j, k_sl]) * \
                _b(dx[s_i, s_jp1] - dx[s_i, s_jm1])
        
        # Denom: dx(i,j)*dy(i,j)
        denom = _b(dx[s_i, s_j] * dy[s_i, s_j])

        # Apply to curv
        curv[s_i, s_j, k_sl] = 0.25 * (term1 - term2) / (denom + 1e-20)

        # =================================================================
        # 2. Calculate X-component of velocity advection (ADVX)
        # =================================================================
        
        # --- 2.1 Calculate horizontal advective fluxes (X-Flux) ---
        s_i_xflux = slice(1, imm1)
        s_ip1_xflux = slice(2, imm1 + 1)
        s_im1_xflux = slice(0, imm1 - 1)
        
        term_u1 = ( _b(dt[s_ip1_xflux, :]) + _b(dt[s_i_xflux, :]) ) * u[s_ip1_xflux, :, k_sl]
        term_u2 = ( _b(dt[s_i_xflux, :]) + _b(dt[s_im1_xflux, :]) ) * u[s_i_xflux, :, k_sl]
        term_u_sum = u[s_ip1_xflux, :, k_sl] + u[s_i_xflux, :, k_sl]
        
        xflux[s_i_xflux, :, k_sl] = 0.125 * (term_u1 + term_u2) * term_u_sum

        # --- 2.2 Calculate horizontal advective fluxes (Y-Flux for U equation) ---
        s_j_yflux = slice(1, jm)
        s_jm1_yflux = slice(0, jm - 1)
        s_i_yflux = slice(1, im)
        s_im1_yflux = slice(0, im - 1) 

        dt_ij = _b(dt[s_i_yflux, s_j_yflux])
        dt_ijm1 = _b(dt[s_i_yflux, s_jm1_yflux])
        dt_im1j = _b(dt[s_im1_yflux, s_j_yflux])
        dt_im1jm1 = _b(dt[s_im1_yflux, s_jm1_yflux])
        
        v_ij = v[s_i_yflux, s_j_yflux, k_sl]
        v_im1j = v[s_im1_yflux, s_j_yflux, k_sl]
        
        term_v1 = (dt_ij + dt_ijm1) * v_ij
        term_v2 = (dt_im1j + dt_im1jm1) * v_im1j
        
        u_sum = u[s_i_yflux, s_j_yflux, k_sl] + u[s_i_yflux, s_jm1_yflux, k_sl]
        
        yflux[s_i_yflux, s_j_yflux, k_sl] = 0.125 * (term_v1 + term_v2) * u_sum

        # --- 2.3 Add horizontal diffusive fluxes (X-Momentum) ---
        s_i = slice(1, imm1)
        s_j = slice(1, jm)
        s_ip1 = slice(2, imm1+1)
        # s_im1 = slice(0, imm1-1) # Already defined
        s_jm1 = slice(0, jm-1)
        
        # Update xflux (Diffusive part)
        dx_val = _b(dx[s_i, s_j])
        grad_u = (ub[s_ip1, s_j, k_sl] - ub[s_i, s_j, k_sl]) / (dx_val + 1e-20)
        visc_term_x = _b(dt[s_i, s_j]) * aam[s_i, s_j, k_sl] * 2.0 * grad_u
        
        xflux[s_i, s_j, k_sl] = xflux[s_i, s_j, k_sl] - visc_term_x

        # Calculate dtaam (4-point average)
        dt_avg = 0.25 * (dt[s_i, s_j] + dt[s_im1, s_j] + dt[s_i, s_jm1] + dt[s_im1, s_jm1])
        aam_sum = aam[s_i, s_j, k_sl] + aam[s_im1, s_j, k_sl] + \
                  aam[s_i, s_jm1, k_sl] + aam[s_im1, s_jm1, k_sl]
        dtaam = _b(dt_avg) * aam_sum
        
        # Update yflux (Diffusive part)
        denom_dy = _b(dy[s_i, s_j] + dy[s_im1, s_j] + dy[s_i, s_jm1] + dy[s_im1, s_jm1])
        denom_dx = _b(dx[s_i, s_j] + dx[s_im1, s_j] + dx[s_i, s_jm1] + dx[s_im1, s_jm1])
        
        term_du_dy = (ub[s_i, s_j, k_sl] - ub[s_i, s_jm1, k_sl]) / (denom_dy + 1e-20)
        term_dv_dx = (vb[s_i, s_j, k_sl] - vb[s_im1, s_j, k_sl]) / (denom_dx + 1e-20)
        
        visc_term_y = dtaam * (term_du_dy + term_dv_dx)
        
        yflux[s_i, s_j, k_sl] = yflux[s_i, s_j, k_sl] - visc_term_y

        # Scale fluxes by geometry
        xflux[s_i, s_j, k_sl] *= _b(dy[s_i, s_j])
        dx_sum = dx[s_i, s_j] + dx[s_im1, s_j] + dx[s_i, s_jm1] + dx[s_im1, s_jm1]
        yflux[s_i, s_j, k_sl] *= _b(0.25 * dx_sum)

        xflux, yflux = self.apply_implicit_flux3d_boundary(xflux, yflux, k_sl)

        # --- 2.4 Calculate ADVX (Flux Divergence) ---
        s_i = slice(1, imm1)
        s_im1 = slice(0, imm1-1)
        s_j = slice(1, jmm1)
        s_jp1 = slice(2, jmm1+1)
        
        advx[s_i, s_j, k_sl] = xflux[s_i, s_j, k_sl] - xflux[s_im1, s_j, k_sl] + \
                               yflux[s_i, s_jp1, k_sl] - yflux[s_i, s_j, k_sl]

        # --- 2.5 Curvature Correction for ADVX ---
        s_i_c = slice(2, imm1)
        s_im1_c = slice(1, imm1-1)
        # Note: s_j is same (1:jmm1)
        
        term_curv1 = curv[s_i_c, s_j, k_sl] * _b(dt[s_i_c, s_j]) * \
                     (v[s_i_c, s_jp1, k_sl] + v[s_i_c, s_j, k_sl])
                     
        term_curv2 = curv[s_im1_c, s_j, k_sl] * _b(dt[s_im1_c, s_j]) * \
                     (v[s_im1_c, s_jp1, k_sl] + v[s_im1_c, s_j, k_sl])
        
        advx[s_i_c, s_j, k_sl] -= aru[s_i_c, s_j].unsqueeze(2) * 0.25 * (term_curv1 + term_curv2)

        # =================================================================
        # 3. Calculate Y-component of velocity advection (ADVY)
        # =================================================================
        
        # Reset fluxes for Y calculation
        # 🔥 FIX: Re-initialize local tensors instead of zero_() to be safe with autograd versions
        xflux = torch.zeros_like(u)
        yflux = torch.zeros_like(u)

        # --- 3.1 Horizontal advective fluxes (X-Flux for V equation) ---
        s_i = slice(1, im)
        s_im1 = slice(0, im-1)
        s_j = slice(1, jm)
        s_jm1 = slice(0, jm-1)
        
        dt_ij = _b(dt[s_i, s_j])
        dt_im1j = _b(dt[s_im1, s_j])
        dt_ijm1 = _b(dt[s_i, s_jm1])
        dt_im1jm1 = _b(dt[s_im1, s_jm1])
        
        u_ij = u[s_i, s_j, k_sl]
        u_ijm1 = u[s_i, s_jm1, k_sl]
        
        term_u1 = (dt_ij + dt_im1j) * u_ij
        term_u2 = (dt_ijm1 + dt_im1jm1) * u_ijm1
        v_sum = v[s_i, s_j, k_sl] + v[s_im1, s_j, k_sl]
        
        xflux[s_i, s_j, k_sl] = 0.125 * (term_u1 + term_u2) * v_sum

        # --- 3.2 Horizontal advective fluxes (Y-Flux) ---
        s_j = slice(1, jmm1)
        s_jp1 = slice(2, jmm1+1)
        s_jm1 = slice(0, jmm1-1)
        
        term_v1 = ( _b(dt[:, s_jp1]) + _b(dt[:, s_j]) ) * v[:, s_jp1, k_sl]
        term_v2 = ( _b(dt[:, s_j]) + _b(dt[:, s_jm1]) ) * v[:, s_j, k_sl]
        v_sum = v[:, s_jp1, k_sl] + v[:, s_j, k_sl]
        
        yflux[:, s_j, k_sl] = 0.125 * (term_v1 + term_v2) * v_sum

        # --- 3.3 Add horizontal diffusive fluxes (Y-Momentum) ---
        s_j = slice(1, jmm1)
        s_i = slice(1, im)
        s_im1 = slice(0, im-1)
        s_jm1 = slice(0, jmm1-1)
        s_jp1 = slice(2, jmm1+1)

        dt_avg = 0.25 * (dt[s_i, s_j] + dt[s_im1, s_j] + dt[s_i, s_jm1] + dt[s_im1, s_jm1])
        aam_sum = aam[s_i, s_j, k_sl] + aam[s_im1, s_j, k_sl] + \
                  aam[s_i, s_jm1, k_sl] + aam[s_im1, s_jm1, k_sl]
        dtaam = _b(dt_avg) * aam_sum

        denom_dy = _b(dy[s_i, s_j] + dy[s_im1, s_j] + dy[s_i, s_jm1] + dy[s_im1, s_jm1])
        denom_dx = _b(dx[s_i, s_j] + dx[s_im1, s_j] + dx[s_i, s_jm1] + dx[s_im1, s_jm1])
        
        term_du_dy = (ub[s_i, s_j, k_sl] - ub[s_i, s_jm1, k_sl]) / (denom_dy + 1e-20)
        term_dv_dx = (vb[s_i, s_j, k_sl] - vb[s_im1, s_j, k_sl]) / (denom_dx + 1e-20)
        
        xflux[s_i, s_j, k_sl] = xflux[s_i, s_j, k_sl] - dtaam * (term_du_dy + term_dv_dx)

        grad_v = (vb[s_i, s_jp1, k_sl] - vb[s_i, s_j, k_sl]) / (_b(dy[s_i, s_j]) + 1e-20)
        visc_term_y = _b(dt[s_i, s_j]) * aam[s_i, s_j, k_sl] * 2.0 * grad_v
        
        yflux[s_i, s_j, k_sl] = yflux[s_i, s_j, k_sl] - visc_term_y

        # Scale fluxes
        dy_sum = dy[s_i, s_j] + dy[s_im1, s_j] + dy[s_i, s_jm1] + dy[s_im1, s_jm1]
        xflux[s_i, s_j, k_sl] *= _b(0.25 * dy_sum)
        yflux[s_i, s_j, k_sl] *= _b(dx[s_i, s_j])

        xflux, yflux = self.apply_implicit_flux3d_boundary(xflux, yflux, k_sl)

        # --- 3.4 Calculate ADVY (Flux Divergence) ---
        s_i = slice(1, imm1)
        s_ip1 = slice(2, imm1+1)
        s_j = slice(1, jmm1)
        s_jm1 = slice(0, jmm1-1)
        
        advy[s_i, s_j, k_sl] = xflux[s_ip1, s_j, k_sl] - xflux[s_i, s_j, k_sl] + \
                               yflux[s_i, s_j, k_sl] - yflux[s_i, s_jm1, k_sl]

        # --- 3.5 Curvature Correction for ADVY ---
        s_j_c = slice(2, jmm1)
        s_jm1_c = slice(1, jmm1-1)
        s_i = slice(1, imm1)
        s_ip1 = slice(2, imm1+1)
        
        term_curv1 = curv[s_i, s_j_c, k_sl] * _b(dt[s_i, s_j_c]) * \
                     (u[s_ip1, s_j_c, k_sl] + u[s_i, s_j_c, k_sl])
                     
        term_curv2 = curv[s_i, s_jm1_c, k_sl] * _b(dt[s_i, s_jm1_c]) * \
                     (u[s_ip1, s_jm1_c, k_sl] + u[s_i, s_jm1_c, k_sl])
        
        advy[s_i, s_j_c, k_sl] += arv[s_i, s_j_c].unsqueeze(2) * 0.25 * (term_curv1 + term_curv2)

        # Return the LOCAL tensors. 
        # The caller (run_time_loop) will update self.advx, self.advy etc with these values.
        return xflux, yflux, curv, advx, advy

    def advave(self, curv2d, advua, advva, fluxua, fluxva, wubot, wvbot, tps, 
               mode, aam2d, 
               uab, vab, dx, dy, ua, va, cbc, aru, arv, d):
        """
        Calculates horizontal advection and diffusion (2D External Mode).
        """
        
        # Reset output arrays (gradients safe)
        advua = torch.zeros_like(advua)
        # Note: advva is reset later in the V-section, as per MATLAB code
        
        # Slices for common ranges
        # MATLAB: 2:jm   -> Python: 1:jm
        # MATLAB: 2:imm1 -> Python: 1:imm1
        # MATLAB: 2:im   -> Python: 1:im
        
        s_i = slice(1, self.im)
        s_j = slice(1, self.jm)
        
        s_imm1 = slice(1, self.imm1) # 1 to im-2
        s_jmm1 = slice(1, self.jmm1) # 1 to jm-2
        
        # Stencil slices
        s_ip1 = slice(2, self.im + 1)
        s_im1 = slice(0, self.im - 1)
        s_jp1 = slice(2, self.jm + 1)
        s_jm1 = slice(0, self.jm - 1)

        # =============================================================
        # 1. U-Advection and Diffusion
        # =============================================================
        
        # --- 1.1 Flux UA (Advective) ---
        # Loop: j=2:jm, i=2:imm1
        # Python i -> 1:imm1
        s_i_u = s_imm1
        s_ip1_u = slice(2, self.imm1 + 1)
        s_im1_u = slice(0, self.imm1 - 1)
        
        # Term 1: (d(i+1)+d(i)) * ua(i+1)
        t1 = (d[s_ip1_u, s_j] + d[s_i_u, s_j]) * ua[s_ip1_u, s_j]
        # Term 2: (d(i)+d(i-1)) * ua(i)
        t2 = (d[s_i_u, s_j] + d[s_im1_u, s_j]) * ua[s_i_u, s_j]
        
        fluxua[s_i_u, s_j] = 0.125 * (t1 + t2) * (ua[s_ip1_u, s_j] + ua[s_i_u, s_j])

        # --- 1.2 Flux VA (Advective) ---
        # Loop: j=2:jm, i=2:im
        # Python i -> 1:im
        s_i_v = s_i
        s_im1_v = s_im1
        # j-1 needed
        s_jm1_v = s_jm1
        
        # Term 1: (d(i,j)+d(i,j-1)) * va(i,j)
        t1 = (d[s_i_v, s_j] + d[s_i_v, s_jm1_v]) * va[s_i_v, s_j]
        # Term 2: (d(i-1,j)+d(i-1,j-1)) * va(i-1,j)
        t2 = (d[s_im1_v, s_j] + d[s_im1_v, s_jm1_v]) * va[s_im1_v, s_j]
        
        fluxva[s_i_v, s_j] = 0.125 * (t1 + t2) * (ua[s_i_v, s_j] + ua[s_i_v, s_jm1_v])

        # --- 1.3 Add Viscous Fluxes to FluxUA ---
        # Loop: j=2:jm, i=2:imm1
        # fluxua = fluxua - d * 2 * aam * (uab(i+1)-uab(i)) / dx
        diff_u = (uab[s_ip1_u, s_j] - uab[s_i_u, s_j]) / dx[s_i_u, s_j]
        visc = d[s_i_u, s_j] * 2.0 * aam2d[s_i_u, s_j] * diff_u
        fluxua[s_i_u, s_j] -= visc

        # --- 1.4 Calculate TPS and Update Fluxes ---
        # Loop: j=2:jm, i=2:im
        # Note: This is a complex 4-point stencil
        
        # Sum of Depths (d)
        d_sum = d[s_i, s_j] + d[s_im1, s_j] + d[s_i, s_jm1] + d[s_im1, s_jm1]
        
        # Sum of Viscosity (aam2d)
        aam_sum = aam2d[s_i, s_j] + aam2d[s_i, s_jm1] + \
                  aam2d[s_im1, s_j] + aam2d[s_im1, s_jm1]
        
        # Sum of DY and DX
        dy_sum = dy[s_i, s_j] + dy[s_im1, s_j] + dy[s_i, s_jm1] + dy[s_im1, s_jm1]
        dx_sum = dx[s_i, s_j] + dx[s_im1, s_j] + dx[s_i, s_jm1] + dx[s_im1, s_jm1]
        
        # Strain rates
        du_dy = (uab[s_i, s_j] - uab[s_i, s_jm1]) / dy_sum
        dv_dx = (vab[s_i, s_j] - vab[s_im1, s_j]) / dx_sum
        
        tps[s_i, s_j] = 0.25 * d_sum * aam_sum * (du_dy + dv_dx)
        
        # Update fluxua (Scaling by dy)
        fluxua[s_i, s_j] *= dy[s_i, s_j]
        
        # Update fluxva (Subtract TPS and Scale by avg DX)
        fluxva[s_i, s_j] = (fluxva[s_i, s_j] - tps[s_i, s_j]) * 0.25 * dx_sum

        fluxua, fluxva = self.apply_implicit_flux2d_boundary(fluxua, fluxva)

        # --- 1.5 Calculate ADVUA (Divergence) ---
        # Loop: j=2:jmm1, i=2:imm1
        s_i_a = s_imm1
        s_im1_a = slice(0, self.imm1 - 1)
        s_j_a = s_jmm1
        s_jp1_a = slice(2, self.jmm1 + 1)
        
        advua[s_i_a, s_j_a] = fluxua[s_i_a, s_j_a] - fluxua[s_im1_a, s_j_a] + \
                              fluxva[s_i_a, s_jp1_a] - fluxva[s_i_a, s_j_a]

        # =============================================================
        # 2. V-Advection and Diffusion
        # =============================================================
        
        advva = torch.zeros_like(advva)
        
        # Note: We reuse fluxua and fluxva variables, effectively resetting them 
        # by overwriting them in the loops below.
        
        # --- 2.1 Flux UA (Advective - REUSED for V-eqn) ---
        # Loop: j=2:jm, i=2:im
        # Python i -> 1:im
        s_i_u = s_i
        s_im1_u = s_im1
        s_jm1_u = s_jm1
        
        # .125 * ((d(i,j)+d(i-1,j))*ua(i,j) + (d(i,j-1)+d(i-1,j-1))*ua(i,j-1)) * (va(i-1,j)+va(i,j))
        
        t1 = (d[s_i_u, s_j] + d[s_im1_u, s_j]) * ua[s_i_u, s_j]
        t2 = (d[s_i_u, s_jm1_u] + d[s_im1_u, s_jm1_u]) * ua[s_i_u, s_jm1_u]
        
        fluxua[s_i_u, s_j] = 0.125 * (t1 + t2) * (va[s_im1_u, s_j] + va[s_i_u, s_j])
        
        # --- 2.2 Flux VA (Advective) ---
        # Loop: j=2:jmm1, i=2:im
        # Python j -> 1:jmm1
        s_j_v = s_jmm1
        s_jp1_v = slice(2, self.jmm1 + 1)
        s_jm1_v = slice(0, self.jmm1 - 1)
        s_i_v = s_i
        
        # .125 * ((d(i,j+1)+d(i,j))*va(i,j+1) + (d(i,j)+d(i,j-1))*va(i,j)) * (va(i,j+1)+va(i,j))
        
        t1 = (d[s_i_v, s_jp1_v] + d[s_i_v, s_j_v]) * va[s_i_v, s_jp1_v]
        t2 = (d[s_i_v, s_j_v] + d[s_i_v, s_jm1_v]) * va[s_i_v, s_j_v]
        
        fluxva[s_i_v, s_j_v] = 0.125 * (t1 + t2) * (va[s_i_v, s_jp1_v] + va[s_i_v, s_j_v])
        
        # --- 2.3 Add Viscous Fluxes to FluxVA ---
        # Loop: j=2:jmm1, i=2:im
        # fluxva = fluxva - d * 2 * aam * (vab(j+1)-vab(j)) / dy
        diff_v = (vab[s_i_v, s_jp1_v] - vab[s_i_v, s_j_v]) / dy[s_i_v, s_j_v]
        visc = d[s_i_v, s_j_v] * 2.0 * aam2d[s_i_v, s_j_v] * diff_v
        fluxva[s_i_v, s_j_v] -= visc
        
        # --- 2.4 Update Fluxes (Metric Scaling & TPS Re-use) ---
        # Loop: j=2:jm, i=2:im
        # Note: TPS is reused from Section 1!
        
        # fluxva scaled by dx
        fluxva[s_i, s_j] *= dx[s_i, s_j]
        
        # fluxua subtracts TPS then scales by avg DY
        dy_sum = dy[s_i, s_j] + dy[s_im1, s_j] + dy[s_i, s_jm1] + dy[s_im1, s_jm1]
        fluxua[s_i, s_j] = (fluxua[s_i, s_j] - tps[s_i, s_j]) * 0.25 * dy_sum
        
        fluxua, fluxva = self.apply_implicit_flux2d_boundary(fluxua, fluxva)

        # --- 2.5 Calculate ADVVA (Divergence) ---
        # Loop: j=2:jmm1, i=2:imm1
        s_i_a = s_imm1
        s_ip1_a = slice(2, self.imm1 + 1)
        s_j_a = s_jmm1
        s_jm1_a = slice(0, self.jmm1 - 1)
        
        # advva = fluxua(i+1) - fluxua(i) + fluxva(j) - fluxva(j-1)
        advva[s_i_a, s_j_a] = fluxua[s_ip1_a, s_j_a] - fluxua[s_i_a, s_j_a] + \
                              fluxva[s_i_a, s_j_a] - fluxva[s_i_a, s_jm1_a]

        # =============================================================
        # 3. Mode 2 Specifics (Bottom Stress & Curvature)
        # =============================================================
        if mode == 2:
            # Slices: j=2:jmm1, i=2:imm1
            s_i = s_imm1
            s_im1 = slice(0, self.imm1 - 1)
            s_ip1 = slice(2, self.imm1 + 1)
            
            s_j = s_jmm1
            s_jm1 = slice(0, self.jmm1 - 1)
            s_jp1 = slice(2, self.jmm1 + 1)
            
            # --- 3.1 wubot calculation ---
            # V averaged at U point
            v_avg_sq = (0.25 * (vab[s_i, s_j] + vab[s_i, s_jp1] + 
                                vab[s_im1, s_j] + vab[s_im1, s_jp1])) ** 2
            
            wubot[s_i, s_j] = -0.50 * (cbc[s_i, s_j] + cbc[s_im1, s_j]) * \
                              torch.sqrt(uab[s_i, s_j]**2 + v_avg_sq) * uab[s_i, s_j]
            
            # --- 3.2 wvbot calculation ---
            # U averaged at V point
            u_avg_sq = (0.25 * (uab[s_i, s_j] + uab[s_ip1, s_j] + 
                                uab[s_i, s_jm1] + uab[s_ip1, s_jm1])) ** 2
            
            wvbot[s_i, s_j] = -0.50 * (cbc[s_i, s_j] + cbc[s_i, s_jm1]) * \
                              torch.sqrt(vab[s_i, s_j]**2 + u_avg_sq) * vab[s_i, s_j]
            
            # --- 3.3 Curvature (curv2d) ---
            # Same formula as in advct but 2D
            t1 = (va[s_i, s_jp1] + va[s_i, s_j]) * (dy[s_ip1, s_j] - dy[s_im1, s_j])
            t2 = (ua[s_ip1, s_j] + ua[s_i, s_j]) * (dx[s_i, s_jp1] - dx[s_i, s_jm1])
            
            curv2d[s_i, s_j] = 0.25 * (t1 - t2) / (dx[s_i, s_j] * dy[s_i, s_j])
            
            # --- 3.4 Curvature Corrections ---
            
            # advua correction (i=3:imm1 -> s_i start+1)
            # Python i: 2:imm1
            s_i_c = slice(2, self.imm1)
            s_im1_c = slice(1, self.imm1 - 1) # relative to original indices
            # j is 2:jmm1 (same)
            
            term1 = curv2d[s_i_c, s_j] * d[s_i_c, s_j] * (va[s_i_c, s_jp1] + va[s_i_c, s_j])
            term2 = curv2d[s_im1_c, s_j] * d[s_im1_c, s_j] * (va[s_im1_c, s_jp1] + va[s_im1_c, s_j])
            
            advua[s_i_c, s_j] -= aru[s_i_c, s_j] * 0.25 * (term1 + term2)
            
            # advva correction (j=3:jmm1 -> s_j start+1)
            # Python j: 2:jmm1
            s_j_c = slice(2, self.jmm1)
            s_jm1_c = slice(1, self.jmm1 - 1)
            # i is 2:imm1 (same)
            
            term1 = curv2d[s_i, s_j_c] * d[s_i, s_j_c] * (ua[s_ip1, s_j_c] + ua[s_i, s_j_c])
            term2 = curv2d[s_i, s_jm1_c] * d[s_i, s_jm1_c] * (ua[s_ip1, s_jm1_c] + ua[s_i, s_jm1_c])
            
            advva[s_i, s_j_c] += arv[s_i, s_j_c] * 0.25 * (term1 + term2)

        return curv2d, advua, advva, fluxua, fluxva, wubot, wvbot, tps

    def bcond(self, idx, elf, uaf, vaf, uf, vf, w):
        """Apply the active implicit boundary policy.

        idx: 1=Elevation, 2=2D Velocity, 3=3D Velocity, 4=T/S, 5=W, 6=Turbulence.
        """
        return self.apply_implicit_bcond(idx, elf, uaf, vaf, uf, vf, w)

    def vertvl(self, xflux, yflux, w, dx, dy, dz, dt, u, v, vfluxb, vfluxf, 
               etf, etb, dti2, im, jm, imm1, jmm1, kbm1):
        """
        Calculates vertical velocity (w) by integrating the continuity equation.
        """
        
        # Helper for 2D -> 3D broadcasting
        def _b(tensor_2d):
            return tensor_2d.unsqueeze(2)

        # Slice for K loop (0 to kbm1-1) representing layers
        k_sl = slice(0, kbm1)

        # =================================================================
        # 1. Calculate Horizontal Fluxes (XFlux)
        # =================================================================
        # Loop: j=2:jm, i=2:im
        # Python: j -> 1:jm, i -> 1:im
        
        s_i = slice(1, im)
        s_im1 = slice(0, im - 1)
        s_j = slice(1, jm)
        
        # Formula: .25 * (dy(i,j)+dy(i-1,j)) * (dt(i,j)+dt(i-1,j)) * u(i,j,k)
        
        term_dy = _b(dy[s_i, s_j] + dy[s_im1, s_j])
        term_dt = _b(dt[s_i, s_j] + dt[s_im1, s_j])
        
        xflux[s_i, s_j, k_sl] = 0.25 * term_dy * term_dt * u[s_i, s_j, k_sl]

        # =================================================================
        # 2. Calculate Horizontal Fluxes (YFlux)
        # =================================================================
        # Loop: j=2:jm, i=2:im
        # Python: j -> 1:jm, i -> 1:im
        
        s_jm1 = slice(0, jm - 1)
        
        # Formula: .25 * (dx(i,j)+dx(i,j-1)) * (dt(i,j)+dt(i,j-1)) * v(i,j,k)
        
        term_dx = _b(dx[s_i, s_j] + dx[s_i, s_jm1])
        term_dt = _b(dt[s_i, s_j] + dt[s_i, s_jm1])
        
        yflux[s_i, s_j, k_sl] = 0.25 * term_dx * term_dt * v[s_i, s_j, k_sl]

        # =================================================================
        # 3. Surface Boundary Condition
        # =================================================================
        # Loop: j=2:jmm1, i=2:imm1
        # Python: j -> 1:jmm1, i -> 1:imm1
        
        s_i_w = slice(1, imm1)
        s_j_w = slice(1, jmm1)
        
        # w(i,j,1) = 0.5 * (vfluxb + vfluxf) -> Python index 0
        w[s_i_w, s_j_w, 0] = 0.5 * (vfluxb[s_i_w, s_j_w] + vfluxf[s_i_w, s_j_w])

        # =================================================================
        # 4. Integrate Vertical Velocity (The Continuity Equation)
        # =================================================================
        # We compute the increment for all K layers at once and then use cumsum.
        
        # Slices for gradients
        # xflux(i+1) - xflux(i)
        s_ip1_w = slice(2, imm1 + 1)
        # yflux(j+1) - yflux(j)
        s_jp1_w = slice(2, jmm1 + 1)
        
        # Divergence Term: (dFluxX + dFluxY) / (dx*dy)
        flux_div = (xflux[s_ip1_w, s_j_w, k_sl] - xflux[s_i_w, s_j_w, k_sl] +
                    yflux[s_i_w, s_jp1_w, k_sl] - yflux[s_i_w, s_j_w, k_sl])
        
        area = _b(dx[s_i_w, s_j_w] * dy[s_i_w, s_j_w])
        term_div = flux_div / area
        
        # Eta Term: (etf - etb) / dti2
        term_eta = _b((etf[s_i_w, s_j_w] - etb[s_i_w, s_j_w]) / dti2)
        
        # Total Increment per layer (before multiplying by dz)
        increment_base = term_div + term_eta
        
        # Multiply by dz(k)
        # dz is 1D (kb). Slice it to kbm1. Reshape to (1, 1, kbm1) for broadcast.
        dz_k = dz[k_sl].view(1, 1, -1)
        
        w_increments = increment_base * dz_k
        
        # Cumulative Sum
        # w(k+1) = w(k) + increment
        # This is equivalent to cumsum starting from w_surface
        w_integral = torch.cumsum(w_increments, dim=2)
        
        # Assign to w
        # w indices: 1 to kbm1 (Python) corresponds to MATLAB k+1 (where k=1..kbm1)
        # w[:,:,0] is already set. We set w[:,:,1:kb-1]
        # Note: kbm1 in Python slice is exclusive, so we need kbm1+1 to include index kbm1?
        # MATLAB loop: k=1 -> w(2) = w(1) + inc(1)
        # ...
        # MATLAB loop: k=kbm1 -> w(kbm1+1) = w(kbm1) + inc(kbm1)
        # So we update indices 1 to kbm1 (inclusive count: kbm1).
        
        # w_integral has shape (..., kbm1).
        # w_surface has shape (..., 1) after unsqueeze.
        
        w_surface = w[s_i_w, s_j_w, 0].unsqueeze(2)
        
        # Assign result to w layers 1 to kbm1 (indices 1 to kbm1)
        # Python slice [1 : kbm1 + 1] has length kbm1.
        w[s_i_w, s_j_w, 1 : kbm1 + 1] = w_surface + w_integral

        return xflux, yflux, w
