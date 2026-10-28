import math
import numpy as np
import netCDF4 as nc
import torch
import torch.nn.functional as F

from neuralpom.utils import safe_clamp_min, safe_sqrt


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
        """
        Applies the explicit Double-gyre lateral boundary conditions.
        idx: 1=Elevation, 2=2D Velocity, 3=3D Velocity, 4=T/S, 5=W, 6=Turbulence
        """
        
        # Helper indices
        im_end = self.im - 1  # Last index
        im_m1  = self.im - 2  # Second to last
        jm_end = self.jm - 1  # Last index
        jm_m1  = self.jm - 2  # Second to last

        # IDX = 1: External (2-D) Elevation
        # =================================================================
        if idx == 1:
            # East: elf(im, :) = elf(imm1, :)
            elf[im_end, :] = elf[im_m1, :]
            # West: elf(1, :) = elf(2, :) -> Python 0 = 1
            elf[0, :] = elf[1, :]
            
            # North: elf(:, jm) = elf(:, jmm1)
            elf[:, jm_end] = elf[:, jm_m1]
            # South: elf(:, 1) = elf(:, 2) -> Python 0 = 1
            elf[:, 0] = elf[:, 1]
            
            # Masking
            elf = elf * self.fsm
            
            return elf, uaf, vaf, uf, vf, w

        # =================================================================
        # IDX = 2: External (2-D) Velocity
        # =================================================================
        elif idx == 2:
            # Range: j=2:jmm1 -> Python 1:jm-1
            s_j = slice(1, self.jm - 1)
            
            # --- East Boundary (uaf) ---
            # uaf(im,j) = uabe(j) + rfe * sqrt(g/h(imm1,j)) * (el(imm1,j) - ele(j))
            term_e = self.rfe * safe_sqrt(self.grav / safe_clamp_min(self.h[im_m1, s_j], 1e-8)) * \
                     (self.el[im_m1, s_j] - self.ele[s_j])
            uaf[im_end, s_j] = (self.uabe[s_j] + term_e) * self.ramp
            vaf[im_end, s_j] = 0.0

            # --- West Boundary (uaf) ---
            # uaf(2,j) (index 1) calculated first
            term_w = self.rfw * safe_sqrt(self.grav / safe_clamp_min(self.h[1, s_j], 1e-8)) * \
                     (self.el[1, s_j] - self.elw[s_j])
            uaf[1, s_j] = (self.uabw[s_j] - term_w) * self.ramp
            # uaf(1,j) = uaf(2,j)
            uaf[0, s_j] = uaf[1, s_j]
            vaf[0, s_j] = 0.0

            # Range: i=2:imm1 -> Python 1:im-1
            s_i = slice(1, self.im - 1)

            # --- North Boundary (vaf) ---
            term_n = self.rfn * safe_sqrt(self.grav / safe_clamp_min(self.h[s_i, jm_m1], 1e-8)) * \
                     (self.el[s_i, jm_m1] - self.eln[s_i])
            vaf[s_i, jm_end] = (self.vabn[s_i] + term_n) * self.ramp
            uaf[s_i, jm_end] = 0.0

            # --- South Boundary (vaf) ---
            # vaf(i,2) (index 1)
            term_s = self.rfs * safe_sqrt(self.grav / safe_clamp_min(self.h[s_i, 1], 1e-8)) * \
                     (self.el[s_i, 1] - self.els[s_i])
            vaf[s_i, 1] = (self.vabs[s_i] - term_s) * self.ramp
            # vaf(i,1) = vaf(i,2)
            vaf[s_i, 0] = vaf[s_i, 1]
            uaf[s_i, 0] = 0.0

            # Masking
            uaf = uaf * self.dum
            vaf = vaf * self.dvm
            
            return elf, uaf, vaf, uf, vf, w

        # =================================================================
        # IDX = 3: Internal (3-D) Velocity
        # =================================================================
        elif idx == 3:
            # Slices for smoothing (tangential direction)
            # J range: 2:jmm1 -> 1:jm-1
            s_j = slice(1, self.jm - 1)
            s_jm1 = slice(0, self.jm - 2) # j-1
            s_jp1 = slice(2, self.jm)     # j+1
            
            # Note: 3D variables (im, jm, kb)
            
            # --- East Boundary (uf) ---
            ga = safe_sqrt(self.h[im_end, s_j] / self.hmax).unsqueeze(1) # (j_dim, 1) for broadcasting over k
            
            # Smooth neighbor (imm1)
            u_inner = 0.25 * self.u[im_m1, s_jm1, :self.kbm1] + \
                      0.50 * self.u[im_m1, s_j,   :self.kbm1] + \
                      0.25 * self.u[im_m1, s_jp1, :self.kbm1]
            
            # Smooth boundary (im)
            u_outer = 0.25 * self.u[im_end, s_jm1, :self.kbm1] + \
                      0.50 * self.u[im_end, s_j,   :self.kbm1] + \
                      0.25 * self.u[im_end, s_jp1, :self.kbm1]
            
            uf[im_end, s_j, :self.kbm1] = ga * u_inner + (1.0 - ga) * u_outer
            vf[im_end, s_j, :self.kbm1] = 0.0

            # --- West Boundary (uf) ---
            ga = safe_sqrt(self.h[0, s_j] / self.hmax).unsqueeze(1)
            
            # Inner is index 2 (python), Outer is index 1 (python)
            # Note: MATLAB code uses u(3,...) for inner smoothing of west boundary? 
            # Code: u(3,j-1)... Let's check logic. West BC at i=2 (python 1). Inner point is i=3 (python 2).
            # Yes.
            
            u_inner = 0.25 * self.u[2, s_jm1, :self.kbm1] + \
                      0.50 * self.u[2, s_j,   :self.kbm1] + \
                      0.25 * self.u[2, s_jp1, :self.kbm1]
                      
            u_outer = 0.25 * self.u[1, s_jm1, :self.kbm1] + \
                      0.50 * self.u[1, s_j,   :self.kbm1] + \
                      0.25 * self.u[1, s_jp1, :self.kbm1]
            
            uf[1, s_j, :self.kbm1] = ga * u_inner + (1.0 - ga) * u_outer
            uf[0, s_j, :self.kbm1] = uf[1, s_j, :self.kbm1]
            vf[0, s_j, :self.kbm1] = 0.0

            # I range: 2:imm1 -> 1:im-1
            s_i = slice(1, self.im - 1)
            s_im1_i = slice(0, self.im - 2)
            s_ip1_i = slice(2, self.im)

            # --- North Boundary (vf) ---
            ga = safe_sqrt(self.h[s_i, jm_end] / self.hmax).unsqueeze(1)
            
            v_inner = 0.25 * self.v[s_im1_i, jm_m1, :self.kbm1] + \
                      0.50 * self.v[s_i,     jm_m1, :self.kbm1] + \
                      0.25 * self.v[s_ip1_i, jm_m1, :self.kbm1]
            
            v_outer = 0.25 * self.v[s_im1_i, jm_end, :self.kbm1] + \
                      0.50 * self.v[s_i,     jm_end, :self.kbm1] + \
                      0.25 * self.v[s_ip1_i, jm_end, :self.kbm1]
                      
            vf[s_i, jm_end, :self.kbm1] = ga * v_inner + (1.0 - ga) * v_outer
            uf[s_i, jm_end, :self.kbm1] = 0.0

            # --- South Boundary (vf) ---
            ga = safe_sqrt(self.h[s_i, 0] / self.hmax).unsqueeze(1)
            
            # Inner index 2 (python), Outer index 1 (python)
            v_inner = 0.25 * self.v[s_im1_i, 2, :self.kbm1] + \
                      0.50 * self.v[s_i,     2, :self.kbm1] + \
                      0.25 * self.v[s_ip1_i, 2, :self.kbm1]
            
            v_outer = 0.25 * self.v[s_im1_i, 1, :self.kbm1] + \
                      0.50 * self.v[s_i,     1, :self.kbm1] + \
                      0.25 * self.v[s_ip1_i, 1, :self.kbm1]

            vf[s_i, 1, :self.kbm1] = ga * v_inner + (1.0 - ga) * v_outer
            vf[s_i, 0, :self.kbm1] = vf[s_i, 1, :self.kbm1]
            uf[s_i, 0, :self.kbm1] = 0.0

            # Masking
            # Expand masks to 3D: (im, jm, 1) * (im, jm, kb)
            uf = uf * self.dum.unsqueeze(2)
            vf = vf * self.dvm.unsqueeze(2)
            
            return elf, uaf, vaf, uf, vf, w

        # =================================================================
        # IDX = 4: Temperature and Salinity (using uf/vf as placeholders)
        # =================================================================
        elif idx == 4:
            # Logic: Upstream Advection scheme
            
            # --- East Boundary ---
            # u1 = 2 * u(im,j,k) * dti / (dx(im,j) + dx(imm1,j))
            dx_sum = self.dx[im_end, :] + self.dx[im_m1, :]
            u1 = 2.0 * self.u[im_end, :, :self.kbm1] * self.dti / dx_sum.unsqueeze(1)
            
            # Mask for Inflow (u1 <= 0) / Outflow
            is_inflow = u1 <= 0.0
            
            # Inflow: t = t - u1*(tbe - t)
            # Outflow: t = t - u1*(t - t_inner)
            t_curr = self.t[im_end, :, :self.kbm1]
            s_curr = self.s[im_end, :, :self.kbm1]
            t_inner = self.t[im_m1, :, :self.kbm1]
            s_inner = self.s[im_m1, :, :self.kbm1]
            t_ext = self.tbe[:, :self.kbm1]
            s_ext = self.sbe[:, :self.kbm1]
            
            uf_in = t_curr - u1 * (t_ext - t_curr)
            vf_in = s_curr - u1 * (s_ext - s_curr)
            
            uf_out = t_curr - u1 * (t_curr - t_inner)
            vf_out = s_curr - u1 * (s_curr - s_inner)
            
            uf[im_end, :, :self.kbm1] = torch.where(is_inflow, uf_in, uf_out)
            vf[im_end, :, :self.kbm1] = torch.where(is_inflow, vf_in, vf_out)
            
            # Vertical Correction for Outflow (k != 1 and k != kbm1)
            # Python k indices: 1 to kbm1-2
            if self.kbm1 > 2:
                # Vertical velocity interpolation
                # wm = 0.5 * (w(imm1, j, k) + w(imm1, j, k+1)) * dti / ((zz(k-1)-zz(k+1))*dt)
                # This logic is complex to vectorise fully across K due to k dependence.
                # However, we can slice k=1:kbm1-1 (Python) corresponding to MATLAB k=2:kbm1-1
                
                k_mid = slice(1, self.kbm1 - 1)
                k_minus = slice(0, self.kbm1 - 2)
                k_plus = slice(2, self.kbm1)
                
                w_term = 0.5 * (self.w[im_m1, :, k_mid] + self.w[im_m1, :, k_plus]) * self.dti
                
                # zz terms: zz[k-1] - zz[k+1] -> Python: zz[k_minus] - zz[k_plus]
                dz_term = (self.zz[k_minus] - self.zz[k_plus]).view(1, -1) * self.dt[im_m1, :].unsqueeze(1)
                
                wm = w_term / dz_term
                
                # Correction
                t_grad_z = self.t[im_m1, :, k_minus] - self.t[im_m1, :, k_plus]
                s_grad_z = self.s[im_m1, :, k_minus] - self.s[im_m1, :, k_plus]
                
                correction_t = wm * t_grad_z
                correction_s = wm * s_grad_z
                
                # Apply only to Outflow (u1 > 0)
                # Need u1 slice for k_mid
                u1_mid = u1[:, k_mid]
                is_outflow_mid = u1_mid > 0.0
                
                uf[im_end, :, k_mid] = torch.where(is_outflow_mid, uf[im_end, :, k_mid] - correction_t, uf[im_end, :, k_mid])
                vf[im_end, :, k_mid] = torch.where(is_outflow_mid, vf[im_end, :, k_mid] - correction_s, vf[im_end, :, k_mid])

            # --- West Boundary ---
            dx_sum = self.dx[0, :] + self.dx[1, :]
            u1 = 2.0 * self.u[1, :, :self.kbm1] * self.dti / dx_sum.unsqueeze(1)
            
            is_inflow = u1 >= 0.0 # Note: code says >= 0 for west inflow
            
            t_curr = self.t[0, :, :self.kbm1]
            s_curr = self.s[0, :, :self.kbm1]
            t_inner = self.t[1, :, :self.kbm1]
            s_inner = self.s[1, :, :self.kbm1]
            t_ext = self.tbw[:, :self.kbm1]
            s_ext = self.sbw[:, :self.kbm1]
            
            uf_in = t_curr - u1 * (t_curr - t_ext)
            vf_in = s_curr - u1 * (s_curr - s_ext)
            
            uf_out = t_curr - u1 * (t_inner - t_curr)
            vf_out = s_curr - u1 * (s_inner - s_curr)
            
            uf[0, :, :self.kbm1] = torch.where(is_inflow, uf_in, uf_out)
            vf[0, :, :self.kbm1] = torch.where(is_inflow, vf_in, vf_out)
            
            # Vertical Correction West
            if self.kbm1 > 2:
                k_mid = slice(1, self.kbm1 - 1)
                k_minus = slice(0, self.kbm1 - 2)
                k_plus = slice(2, self.kbm1)
                
                w_term = 0.5 * (self.w[1, :, k_mid] + self.w[1, :, k_plus]) * self.dti
                dz_term = (self.zz[k_minus] - self.zz[k_plus]).view(1, -1) * self.dt[1, :].unsqueeze(1)
                wm = w_term / dz_term
                
                t_grad_z = self.t[1, :, k_minus] - self.t[1, :, k_plus]
                s_grad_z = self.s[1, :, k_minus] - self.s[1, :, k_plus]
                
                u1_mid = u1[:, k_mid]
                is_outflow_mid = u1_mid < 0.0 # Outflow at West is u < 0
                
                uf[0, :, k_mid] = torch.where(is_outflow_mid, uf[0, :, k_mid] - wm*t_grad_z, uf[0, :, k_mid])
                vf[0, :, k_mid] = torch.where(is_outflow_mid, vf[0, :, k_mid] - wm*s_grad_z, vf[0, :, k_mid])

            # --- North Boundary ---
            dy_sum = self.dy[:, jm_end] + self.dy[:, jm_m1]
            u1 = 2.0 * self.v[:, jm_end, :self.kbm1] * self.dti / dy_sum.unsqueeze(1)
            
            is_inflow = u1 <= 0.0
            
            t_curr = self.t[:, jm_end, :self.kbm1]
            s_curr = self.s[:, jm_end, :self.kbm1]
            t_inner = self.t[:, jm_m1, :self.kbm1]
            s_inner = self.s[:, jm_m1, :self.kbm1]
            t_ext = self.tbn[:, :self.kbm1]
            s_ext = self.sbn[:, :self.kbm1]
            
            uf_in = t_curr - u1 * (t_ext - t_curr)
            vf_in = s_curr - u1 * (s_ext - s_curr)
            uf_out = t_curr - u1 * (t_curr - t_inner)
            vf_out = s_curr - u1 * (s_curr - s_inner)
            
            uf[:, jm_end, :self.kbm1] = torch.where(is_inflow, uf_in, uf_out)
            vf[:, jm_end, :self.kbm1] = torch.where(is_inflow, vf_in, vf_out)
            
            if self.kbm1 > 2:
                k_mid = slice(1, self.kbm1 - 1)
                w_term = 0.5 * (self.w[:, jm_m1, k_mid] + self.w[:, jm_m1, k_plus]) * self.dti
                dz_term = (self.zz[k_minus] - self.zz[k_plus]).view(1, -1) * self.dt[:, jm_m1].unsqueeze(1)
                wm = w_term / dz_term
                
                t_grad_z = self.t[:, jm_m1, k_minus] - self.t[:, jm_m1, k_plus]
                s_grad_z = self.s[:, jm_m1, k_minus] - self.s[:, jm_m1, k_plus]
                
                u1_mid = u1[:, k_mid]
                is_outflow = u1_mid > 0.0
                uf[:, jm_end, k_mid] = torch.where(is_outflow, uf[:, jm_end, k_mid] - wm*t_grad_z, uf[:, jm_end, k_mid])
                vf[:, jm_end, k_mid] = torch.where(is_outflow, vf[:, jm_end, k_mid] - wm*s_grad_z, vf[:, jm_end, k_mid])

            # --- South Boundary ---
            dy_sum = self.dy[:, 0] + self.dy[:, 1]
            u1 = 2.0 * self.v[:, 1, :self.kbm1] * self.dti / dy_sum.unsqueeze(1)
            
            is_inflow = u1 >= 0.0
            
            t_curr = self.t[:, 0, :self.kbm1]
            s_curr = self.s[:, 0, :self.kbm1]
            t_inner = self.t[:, 1, :self.kbm1]
            s_inner = self.s[:, 1, :self.kbm1]
            t_ext = self.tbs[:, :self.kbm1]
            s_ext = self.sbs[:, :self.kbm1]
            
            uf_in = t_curr - u1 * (t_curr - t_ext)
            vf_in = s_curr - u1 * (s_curr - s_ext)
            uf_out = t_curr - u1 * (t_inner - t_curr)
            vf_out = s_curr - u1 * (s_inner - s_curr)
            
            uf[:, 0, :self.kbm1] = torch.where(is_inflow, uf_in, uf_out)
            vf[:, 0, :self.kbm1] = torch.where(is_inflow, vf_in, vf_out)
            
            if self.kbm1 > 2:
                k_mid = slice(1, self.kbm1 - 1)
                w_term = 0.5 * (self.w[:, 1, k_mid] + self.w[:, 1, k_plus]) * self.dti
                dz_term = (self.zz[k_minus] - self.zz[k_plus]).view(1, -1) * self.dt[:, 1].unsqueeze(1)
                wm = w_term / dz_term
                
                t_grad_z = self.t[:, 1, k_minus] - self.t[:, 1, k_plus]
                s_grad_z = self.s[:, 1, k_minus] - self.s[:, 1, k_plus]
                
                u1_mid = u1[:, k_mid]
                is_outflow = u1_mid < 0.0
                uf[:, 0, k_mid] = torch.where(is_outflow, uf[:, 0, k_mid] - wm*t_grad_z, uf[:, 0, k_mid])
                vf[:, 0, k_mid] = torch.where(is_outflow, vf[:, 0, k_mid] - wm*s_grad_z, vf[:, 0, k_mid])
                
            # Masking
            uf = uf * self.fsm.unsqueeze(2)
            vf = vf * self.fsm.unsqueeze(2)
            
            return elf, uaf, vaf, uf, vf, w

        # =================================================================
        # IDX = 5: Vertical Velocity
        # =================================================================
        elif idx == 5:
            w[:, :, :self.kbm1] = w[:, :, :self.kbm1] * self.fsm.unsqueeze(2)
            return elf, uaf, vaf, uf, vf, w

        # =================================================================
        # IDX = 6: Turbulence (q2, q2l)
        # =================================================================
        elif idx == 6:
            # Similar to T/S, but simpler (no vertical correction, inflow is small)
            
            # --- East ---
            dx_sum = self.dx[im_end, :] + self.dx[im_m1, :]
            # FIX: dx_sum is (jm,). self.u is (jm, kb). 
            # Need dx_sum -> (jm, 1) to broadcast against (jm, kb)
            u1 = 2.0 * self.u[im_end, :] * self.dti / dx_sum.unsqueeze(1)
            
            is_inflow = u1 <= 0.0
            
            uf_in = self.q2[im_end, :] - u1 * (self.small - self.q2[im_end, :])
            vf_in = self.q2l[im_end, :] - u1 * (self.small - self.q2l[im_end, :])
            uf_out = self.q2[im_end, :] - u1 * (self.q2[im_end, :] - self.q2[im_m1, :])
            vf_out = self.q2l[im_end, :] - u1 * (self.q2l[im_end, :] - self.q2l[im_m1, :])
            
            uf[im_end, :] = torch.where(is_inflow, uf_in, uf_out)
            vf[im_end, :] = torch.where(is_inflow, vf_in, vf_out)

            # --- West ---
            dx_sum = self.dx[0, :] + self.dx[1, :]
            # FIX: Add unsqueeze(1)
            u1 = 2.0 * self.u[1, :] * self.dti / dx_sum.unsqueeze(1)
            
            is_inflow = u1 >= 0.0
            
            uf_in = self.q2[0, :] - u1 * (self.q2[0, :] - self.small)
            vf_in = self.q2l[0, :] - u1 * (self.q2l[0, :] - self.small)
            uf_out = self.q2[0, :] - u1 * (self.q2[1, :] - self.q2[0, :])
            vf_out = self.q2l[0, :] - u1 * (self.q2l[1, :] - self.q2l[0, :])
            
            uf[0, :] = torch.where(is_inflow, uf_in, uf_out)
            vf[0, :] = torch.where(is_inflow, vf_in, vf_out)
            
            # --- North ---
            dy_sum = self.dy[:, jm_end] + self.dy[:, jm_m1]
            # FIX: dy_sum is (im,). self.v is (im, kb).
            # Need dy_sum -> (im, 1)
            u1 = 2.0 * self.v[:, jm_end] * self.dti / dy_sum.unsqueeze(1)
            
            is_inflow = u1 <= 0.0
            
            uf_in = self.q2[:, jm_end] - u1 * (self.small - self.q2[:, jm_end])
            vf_in = self.q2l[:, jm_end] - u1 * (self.small - self.q2l[:, jm_end])
            uf_out = self.q2[:, jm_end] - u1 * (self.q2[:, jm_end] - self.q2[:, jm_m1])
            vf_out = self.q2l[:, jm_end] - u1 * (self.q2l[:, jm_end] - self.q2l[:, jm_m1])
            
            uf[:, jm_end] = torch.where(is_inflow, uf_in, uf_out)
            vf[:, jm_end] = torch.where(is_inflow, vf_in, vf_out)
            
            # --- South ---
            dy_sum = self.dy[:, 0] + self.dy[:, 1]
            # FIX: Add unsqueeze(1)
            u1 = 2.0 * self.v[:, 1] * self.dti / dy_sum.unsqueeze(1)
            
            is_inflow = u1 >= 0.0
            
            uf_in = self.q2[:, 0] - u1 * (self.q2[:, 0] - self.small)
            vf_in = self.q2l[:, 0] - u1 * (self.q2l[:, 0] - self.small)
            uf_out = self.q2[:, 0] - u1 * (self.q2[:, 1] - self.q2[:, 0])
            vf_out = self.q2l[:, 0] - u1 * (self.q2l[:, 1] - self.q2l[:, 0])
            
            uf[:, 0] = torch.where(is_inflow, uf_in, uf_out)
            vf[:, 0] = torch.where(is_inflow, vf_in, vf_out)

            # Masking and stabilization (add epsilon)
            uf = uf * self.fsm.unsqueeze(2) + 1.e-10
            vf = vf * self.fsm.unsqueeze(2) + 1.e-10
            
            return elf, uaf, vaf, uf, vf, w

        else:
            raise ValueError(f"Invalid bcond index: {idx}")

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
