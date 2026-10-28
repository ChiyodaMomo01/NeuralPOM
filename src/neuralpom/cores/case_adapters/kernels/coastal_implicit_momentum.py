import math
import numpy as np
import netCDF4 as nc
import torch
import torch.nn.functional as F


torch.set_default_dtype(torch.float32)


class MomentumMixin:
    def advu(self, advx, aru, dz, cor, dt, e_atmos, dy, drhox, h, dti2, 
             ub, uf, u, v, w, im, jm, kb, imm1, jmm1, kbm1, grav, 
             egf, egb, etf, etb):
        """
        Calculates horizontal and vertical advection of U-momentum, 
        including Coriolis, surface slope, and baroclinic terms.
        """

        # Helper for 2D -> 3D broadcasting
        def _b(tensor_2d):
            return tensor_2d.unsqueeze(2)

        # Initialize Output
        uf = torch.zeros((im, jm, kb), device=self.device, dtype=self.t.dtype)
        
        # Temporary tensor for vertical flux to avoid overwriting issues
        vert_flux = torch.zeros((im, jm, kb), device=self.device, dtype=self.t.dtype)

        # =================================================================
        # 1. Vertical Advection Flux
        # =================================================================
        # Loop: k=2:kbm1, j=1:jm, i=2:im
        # Python: k -> 1:kbm1 (indices 1 to kbm1-1)
        
        k_sl = slice(1, kbm1)     # Current k
        k_m1 = slice(0, kbm1 - 1) # k-1
        
        s_i = slice(1, im)
        s_im1 = slice(0, im - 1)
        s_j = slice(0, jm) # Full J range
        
        # Formula: .25 * (w(i)+w(i-1)) * (u(k)+u(k-1))
        # Note: w is on interfaces, u is at layer centers
        
        w_avg = 0.25 * (w[s_i, s_j, k_sl] + w[s_im1, s_j, k_sl])
        u_avg = u[s_i, s_j, k_sl] + u[s_i, s_j, k_m1]
        
        vert_flux[s_i, s_j, k_sl] = w_avg * u_avg

        # =================================================================
        # 2. Combine Terms (RHS of Momentum Equation)
        # =================================================================
        # Loop: k=1:kbm1, j=2:jmm1, i=2:imm1
        # Python: k -> 0:kbm1
        
        k_range = slice(0, kbm1)
        k_plus1 = slice(1, kbm1 + 1)
        
        s_i = slice(1, imm1)      # i
        s_im1 = slice(0, imm1-1)  # i-1
        s_j = slice(1, jmm1)      # j
        s_jp1 = slice(2, jmm1+1)  # j+1
        
        # --- Term A: Vertical Flux Divergence ---
        # (flux(k) - flux(k+1)) * aru / dz(k)
        # Note: vert_flux size is kb. indices k and k+1 work.
        
        dz_k = dz[k_range].view(1, 1, -1)
        
        vert_div = (vert_flux[s_i, s_j, k_range] - vert_flux[s_i, s_j, k_plus1]) * \
                   _b(aru[s_i, s_j]) / dz_k
                   
        # --- Term B: Coriolis ---
        # -aru * .25 * (cor(i,j)*dt(i,j)*(v(i,j+1)+v(i,j)) + cor(i-1,j)*dt(i-1,j)*(v(i-1,j+1)+v(i-1,j)))
        
        # V part 1 (at i, j)
        v_term1 = v[s_i, s_jp1, k_range] + v[s_i, s_j, k_range]
        cor_term1 = _b(cor[s_i, s_j] * dt[s_i, s_j]) * v_term1
        
        # V part 2 (at i-1, j)
        v_term2 = v[s_im1, s_jp1, k_range] + v[s_im1, s_j, k_range]
        cor_term2 = _b(cor[s_im1, s_j] * dt[s_im1, s_j]) * v_term2
        
        coriolis = -_b(aru[s_i, s_j]) * 0.25 * (cor_term1 + cor_term2)
        
        # --- Term C: Surface Slope (Pressure Gradient) ---
        # grav * .125 * (dt(i)+dt(i-1)) * (grad_eg + 2*grad_atmos) * (dy(i)+dy(i-1))
        
        dt_sum = dt[s_i, s_j] + dt[s_im1, s_j]
        dy_sum = dy[s_i, s_j] + dy[s_im1, s_j]
        
        # Gradient of Elevation (Future + Back)
        grad_eg = (egf[s_i, s_j] - egf[s_im1, s_j]) + (egb[s_i, s_j] - egb[s_im1, s_j])
        
        # Gradient of Atmos Pressure
        grad_pa = 2.0 * (e_atmos[s_i, s_j] - e_atmos[s_im1, s_j])
        
        press_grad = grav * 0.125 * _b(dt_sum * (grad_eg + grad_pa) * dy_sum)
        
        # --- Combine All ---
        # uf = advx + vert_div + coriolis + press_grad + drhox
        
        uf[s_i, s_j, k_range] = advx[s_i, s_j, k_range] + vert_div + \
                                coriolis + press_grad + drhox[s_i, s_j, k_range]

        # =================================================================
        # 3. Step Forward in Time
        # =================================================================
        # uf = (Old_Flux - 2*dti2*RHS) / New_Depth
        
        # Depths
        h_sum = h[s_i, s_j] + h[s_im1, s_j]
        
        # Denominator: (H + Elf + H(i-1) + Elf(i-1)) * aru
        depth_new = (h_sum + etf[s_i, s_j] + etf[s_im1, s_j]) * aru[s_i, s_j]
        denom = _b(depth_new)
        
        # Numerator Term 1: (H + Elb + ...) * aru * ub
        depth_old = (h_sum + etb[s_i, s_j] + etb[s_im1, s_j]) * aru[s_i, s_j]
        term_old = _b(depth_old) * ub[s_i, s_j, k_range]
        
        # Numerator Term 2: 2 * dti2 * uf (the RHS computed above)
        term_rhs = 2.0 * dti2 * uf[s_i, s_j, k_range]
        
        # Final Update
        uf[s_i, s_j, k_range] = (term_old - term_rhs) / (denom + 1e-20)

        return uf

    def advv(self, advy, arv, dz, cor, dt, e_atmos, dx, drhoy, h, dti2, 
             vb, vf, u, v, w, im, jm, kb, imm1, jmm1, kbm1, grav, 
             egf, egb, etf, etb):
        """
        Calculates horizontal and vertical advection of V-momentum, 
        including Coriolis, surface slope, and baroclinic terms.
        """

        # Helper for 2D -> 3D broadcasting
        def _b(tensor_2d):
            return tensor_2d.unsqueeze(2)

        # Initialize Output
        vf = torch.zeros((im, jm, kb), device=self.device, dtype=self.t.dtype)
        
        # Temporary tensor for vertical flux
        vert_flux = torch.zeros((im, jm, kb), device=self.device, dtype=self.t.dtype)

        # =================================================================
        # 1. Vertical Advection Flux
        # =================================================================
        # Loop: k=2:kbm1, j=2:jm, i=1:im
        # Python: k -> 1:kbm1
        
        k_sl = slice(1, kbm1)     # Current k
        k_m1 = slice(0, kbm1 - 1) # k-1
        
        s_i = slice(0, im) # Full I range
        s_j = slice(1, jm)
        s_jm1 = slice(0, jm - 1)
        
        # Formula: .25 * (w(i,j)+w(i,j-1)) * (v(k)+v(k-1))
        
        w_avg = 0.25 * (w[s_i, s_j, k_sl] + w[s_i, s_jm1, k_sl])
        v_avg = v[s_i, s_j, k_sl] + v[s_i, s_j, k_m1]
        
        vert_flux[s_i, s_j, k_sl] = w_avg * v_avg

        # =================================================================
        # 2. Combine Terms (RHS of Momentum Equation)
        # =================================================================
        # Loop: k=1:kbm1, j=2:jmm1, i=2:imm1
        # Python: k -> 0:kbm1
        
        k_range = slice(0, kbm1)
        k_plus1 = slice(1, kbm1 + 1)
        
        s_i = slice(1, imm1)      # i
        s_ip1 = slice(2, imm1+1)  # i+1
        s_j = slice(1, jmm1)      # j
        s_jm1 = slice(0, jmm1-1)  # j-1
        
        # --- Term A: Vertical Flux Divergence ---
        # (flux(k) - flux(k+1)) * arv / dz(k)
        
        dz_k = dz[k_range].view(1, 1, -1)
        
        vert_div = (vert_flux[s_i, s_j, k_range] - vert_flux[s_i, s_j, k_plus1]) * \
                   _b(arv[s_i, s_j]) / dz_k

        # --- Term B: Coriolis ---
        # arv * .25 * (cor(i,j)*dt(i,j)*(u(i+1,j)+u(i,j)) + cor(i,j-1)*dt(i,j-1)*(u(i+1,j-1)+u(i,j-1)))
        
        # U part 1 (at i, j)
        u_term1 = u[s_ip1, s_j, k_range] + u[s_i, s_j, k_range]
        cor_term1 = _b(cor[s_i, s_j] * dt[s_i, s_j]) * u_term1
        
        # U part 2 (at i, j-1)
        u_term2 = u[s_ip1, s_jm1, k_range] + u[s_i, s_jm1, k_range]
        cor_term2 = _b(cor[s_i, s_jm1] * dt[s_i, s_jm1]) * u_term2
        
        coriolis = _b(arv[s_i, s_j]) * 0.25 * (cor_term1 + cor_term2)

        # --- Term C: Surface Slope (Pressure Gradient) ---
        # grav * .125 * (dt(j)+dt(j-1)) * (grad_eg + 2*grad_atmos) * (dx(j)+dx(j-1))
        
        dt_sum = dt[s_i, s_j] + dt[s_i, s_jm1]
        dx_sum = dx[s_i, s_j] + dx[s_i, s_jm1]
        
        # Gradient of Elevation (Future + Back)
        grad_eg = (egf[s_i, s_j] - egf[s_i, s_jm1]) + (egb[s_i, s_j] - egb[s_i, s_jm1])
        
        # Gradient of Atmos Pressure
        grad_pa = 2.0 * (e_atmos[s_i, s_j] - e_atmos[s_i, s_jm1])
        
        press_grad = grav * 0.125 * _b(dt_sum * (grad_eg + grad_pa) * dx_sum)

        # --- Combine All ---
        # vf = advy + vert_div + coriolis + press_grad + drhoy
        
        vf[s_i, s_j, k_range] = advy[s_i, s_j, k_range] + vert_div + \
                                coriolis + press_grad + drhoy[s_i, s_j, k_range]

        # =================================================================
        # 3. Step Forward in Time
        # =================================================================
        # vf = (Old_Flux - 2*dti2*RHS) / New_Depth
        
        # Depths
        h_sum = h[s_i, s_j] + h[s_i, s_jm1]
        
        # Denominator: (H + Elf(j) + H(j-1) + Elf(j-1)) * arv
        depth_new = (h_sum + etf[s_i, s_j] + etf[s_i, s_jm1]) * arv[s_i, s_j]
        denom = _b(depth_new)
        
        # Numerator Term 1: (H + Elb + ...) * arv * vb
        depth_old = (h_sum + etb[s_i, s_j] + etb[s_i, s_jm1]) * arv[s_i, s_j]
        term_old = _b(depth_old) * vb[s_i, s_j, k_range]
        
        # Numerator Term 2: 2 * dti2 * vf (the RHS computed above)
        term_rhs = 2.0 * dti2 * vf[s_i, s_j, k_range]
        
        # Final Update
        vf[s_i, s_j, k_range] = (term_old - term_rhs) / (denom + 1e-20)

        return vf

    def profu(self, a, c, ee, gg, tps, uf, wubot, 
              etf, h, km, dti2, umol, dz, dzz, wusurf, cbc, dum, 
              im, jm, kb, imm1, jmm1, kbm1, kbm2, ub, vb):
        """
        [FIXED FINAL - DEEP FREEZE]: 
        1. Snapshot ALL Inputs.
        2. Freeze 'a' and 'c' immediately after construction.
        3. Freeze 'ee' and 'gg' after forward sweep.
        """
        def _b(tensor_2d): return tensor_2d.unsqueeze(2)

        # 1. Snapshot Inputs (To isolate from external mods)
        uf_in = uf.clone(); uf_out = uf.clone()
        wubot = wubot.clone()
        etf_in = etf.clone(); km_in = km.clone()
        wusurf_in = wusurf.clone(); cbc_in = cbc.clone()
        ub_in = ub.clone(); vb_in = vb.clone(); h_in = h.clone()

        # 2. Local Scratchpads
        a = torch.zeros_like(uf_out); c = torch.zeros_like(uf_out)
        ee = torch.zeros_like(uf_out); gg_store = torch.zeros_like(uf_out) 
        tps_local = torch.zeros_like(uf_out[:, :, 0]) 

        s_i = slice(1, im); s_im1 = slice(0, im - 1); s_j = slice(1, jm)
        
        dh_u = 0.5 * (h_in[s_i, s_j] + etf_in[s_i, s_j] + h_in[s_im1, s_j] + etf_in[s_im1, s_j])
        dh_3d = _b(dh_u)
        
        km_u = torch.zeros_like(km_in)
        km_u[s_i, s_j, :] = 0.5 * (km_in[s_i, s_j, :] + km_in[s_im1, s_j, :])

        dh2 = dh_3d[:, :, 0] ** 2
        for k in range(1, kbm1):
            num = -dti2 * (km_u[s_i, s_j, k] + umol)
            den_a = dz[k-1] * dzz[k-1] * dh2
            a[s_i, s_j, k-1] = num / den_a
            den_c = dz[k] * dzz[k-1] * dh2
            c[s_i, s_j, k] = num / den_c

        # 🔥 DEEP FREEZE 1: Freeze 'a' and 'c'
        # Detach the accumulated history and use read-only copies afterward.
        a_frozen = a.clone()
        c_frozen = c.clone()

        # --- Forward Sweep (Use Frozen a/c) ---
        denom = a_frozen[s_i, s_j, 0] - 1.0
        ee[s_i, s_j, 0] = a_frozen[s_i, s_j, 0] / (denom + 1e-20)
        
        term_surf = -dti2 * wusurf_in[s_i, s_j] / (-dz[0] * dh_u)
        gg_prev = (term_surf - uf_in[s_i, s_j, 0]) / (denom + 1e-20)
        gg_store[s_i, s_j, 0] = gg_prev

        for k in range(1, kbm2):
            # Use a_frozen, c_frozen
            denom = a_frozen[s_i, s_j, k] + c_frozen[s_i, s_j, k] * (1.0 - ee[s_i, s_j, k-1]) - 1.0
            inv_denom = 1.0 / (denom + 1e-20)
            ee[s_i, s_j, k] = a_frozen[s_i, s_j, k] * inv_denom
            
            gg_curr = (c_frozen[s_i, s_j, k] * gg_prev - uf_in[s_i, s_j, k]) * inv_denom
            gg_store[s_i, s_j, k] = gg_curr
            gg_prev = gg_curr

        # 🔥 DEEP FREEZE 2: Freeze 'ee' and 'gg'
        ee_frozen = ee.clone()
        gg_frozen = gg_store.clone()

        # --- Bottom BC (Use Frozen everything) ---
        s_i_in = slice(1, imm1); s_im1_in = slice(0, imm1 - 1); s_j_in = slice(1, jmm1)
        
        cbc_avg = 0.5 * (cbc_in[s_i_in, s_j_in] + cbc_in[s_im1_in, s_j_in])
        k_bot = kbm1 - 1
        
        u_sq = ub_in[s_i_in, s_j_in, k_bot] ** 2
        s_jp1_in = slice(2, jmm1 + 1)
        v_sum = vb_in[s_i_in, s_j_in, k_bot] + vb_in[s_i_in, s_jp1_in, k_bot] + \
                vb_in[s_im1_in, s_j_in, k_bot] + vb_in[s_im1_in, s_jp1_in, k_bot]
        v_avg_sq = (0.25 * v_sum) ** 2
        mag_vel = torch.sqrt(u_sq + v_avg_sq)
        
        tps_local[s_i_in, s_j_in] = cbc_avg * mag_vel
        
        k_prev = k_bot - 1
        term_friction = tps_local[s_i_in, s_j_in] * dti2 / (-dz[k_bot] * dh_u[0:imm1-1, 0:jmm1-1])
        
        # Use a_frozen, c_frozen, ee_frozen
        denom_bot = term_friction - 1.0 - (ee_frozen[s_i_in, s_j_in, k_prev] - 1.0) * c_frozen[s_i_in, s_j_in, k_bot]
        
        # Use c_frozen, gg_frozen
        numer_bot = c_frozen[s_i_in, s_j_in, k_bot] * gg_frozen[s_i_in, s_j_in, k_prev] - uf_in[s_i_in, s_j_in, k_bot]
        
        val_bot = numer_bot / (denom_bot + 1e-20)
        val_bot = val_bot * dum[s_i_in, s_j_in]
        
        uf_out[s_i_in, s_j_in, k_bot] = val_bot
        uf_next = val_bot

        # --- Backward Substitution ---
        for k in range(k_bot - 1, -1, -1):
            val = ee_frozen[s_i_in, s_j_in, k] * uf_next + gg_frozen[s_i_in, s_j_in, k]
            val = val * dum[s_i_in, s_j_in]
            
            uf_out[s_i_in, s_j_in, k] = val
            uf_next = val

        uf_bot_slice = uf_out[s_i_in, s_j_in, k_bot].clone()
        wubot[s_i_in, s_j_in] = -tps_local[s_i_in, s_j_in] * uf_bot_slice

        return a, c, ee, gg_store, tps_local, uf_out, wubot

    def profv(self, a, c, ee, gg, tps, vf, wvbot, 
              dvm, dz, dzz, im, jm, kb, imm1, jmm1, kbm1, kbm2, 
              km, cbc, ub, vb, umol, wvsurf, h, etf, dti2):
        """
        [FIXED FINAL - DEEP FREEZE]: 
        1. Snapshot ALL Inputs.
        2. Freeze 'a' and 'c' immediately after construction.
        3. Freeze 'ee' and 'gg' after forward sweep.
        """
        def _b(tensor_2d): return tensor_2d.unsqueeze(2)

        # 1. Snapshot Inputs
        vf_in = vf.clone(); vf_out = vf.clone()
        wvbot = wvbot.clone()
        etf_in = etf.clone(); km_in = km.clone()
        wvsurf_in = wvsurf.clone(); cbc_in = cbc.clone()
        ub_in = ub.clone(); vb_in = vb.clone(); h_in = h.clone()

        # 2. Local Scratchpads
        a = torch.zeros_like(vf_out); c = torch.zeros_like(vf_out)
        ee = torch.zeros_like(vf_out); gg_store = torch.zeros_like(vf_out) 
        tps_local = torch.zeros_like(vf_out[:, :, 0]) 

        s_i = slice(1, im); s_j = slice(1, jm); s_jm1 = slice(0, jm - 1)
        
        dh_v = 0.5 * (h_in[s_i, s_j] + etf_in[s_i, s_j] + h_in[s_i, s_jm1] + etf_in[s_i, s_jm1])
        dh_3d = _b(dh_v)
        
        km_v = torch.zeros_like(km_in)
        km_v[s_i, s_j, :] = 0.5 * (km_in[s_i, s_j, :] + km_in[s_i, s_jm1, :])

        dh2 = dh_3d[:, :, 0] ** 2
        for k in range(1, kbm1):
            num = -dti2 * (km_v[s_i, s_j, k] + umol)
            den_a = dz[k-1] * dzz[k-1] * dh2
            a[s_i, s_j, k-1] = num / den_a
            den_c = dz[k] * dzz[k-1] * dh2
            c[s_i, s_j, k] = num / den_c

        # 🔥 DEEP FREEZE 1: Freeze 'a' and 'c'
        a_frozen = a.clone()
        c_frozen = c.clone()

        # --- Forward Sweep (Use Frozen a/c) ---
        denom = a_frozen[s_i, s_j, 0] - 1.0
        ee[s_i, s_j, 0] = a_frozen[s_i, s_j, 0] / (denom + 1e-20)
        
        term_surf = -dti2 * wvsurf_in[s_i, s_j] / (-dz[0] * dh_v)
        gg_prev = (term_surf - vf_in[s_i, s_j, 0]) / (denom + 1e-20)
        gg_store[s_i, s_j, 0] = gg_prev

        for k in range(1, kbm2):
            # Use a_frozen, c_frozen
            denom = a_frozen[s_i, s_j, k] + c_frozen[s_i, s_j, k] * (1.0 - ee[s_i, s_j, k-1]) - 1.0
            inv_denom = 1.0 / (denom + 1e-20)
            ee[s_i, s_j, k] = a_frozen[s_i, s_j, k] * inv_denom
            
            gg_curr = (c_frozen[s_i, s_j, k] * gg_prev - vf_in[s_i, s_j, k]) * inv_denom
            gg_store[s_i, s_j, k] = gg_curr
            gg_prev = gg_curr

        # 🔥 DEEP FREEZE 2: Freeze 'ee' and 'gg'
        ee_frozen = ee.clone()
        gg_frozen = gg_store.clone()

        # --- Bottom BC (Use Frozen everything) ---
        s_i_in = slice(1, imm1); s_j_in = slice(1, jmm1); s_jm1_in = slice(0, jmm1 - 1)
        
        cbc_avg = 0.5 * (cbc_in[s_i_in, s_j_in] + cbc_in[s_i_in, s_jm1_in])
        k_bot = kbm1 - 1
        
        v_sq = vb_in[s_i_in, s_j_in, k_bot] ** 2
        s_ip1_in = slice(2, imm1 + 1)
        u_sum = ub_in[s_i_in, s_j_in, k_bot] + ub_in[s_ip1_in, s_j_in, k_bot] + \
                ub_in[s_i_in, s_jm1_in, k_bot] + ub_in[s_ip1_in, s_jm1_in, k_bot]
        u_avg_sq = (0.25 * u_sum) ** 2
        mag_vel = torch.sqrt(u_avg_sq + v_sq)
        
        tps_local[s_i_in, s_j_in] = cbc_avg * mag_vel
        
        k_prev = k_bot - 1
        term_friction = tps_local[s_i_in, s_j_in] * dti2 / (-dz[k_bot] * dh_v[0:imm1-1, 0:jmm1-1])
        
        # Use a_frozen, c_frozen, ee_frozen
        denom_bot = term_friction - 1.0 - (ee_frozen[s_i_in, s_j_in, k_prev] - 1.0) * c_frozen[s_i_in, s_j_in, k_bot]
        
        # Use c_frozen, gg_frozen
        numer_bot = c_frozen[s_i_in, s_j_in, k_bot] * gg_frozen[s_i_in, s_j_in, k_prev] - vf_in[s_i_in, s_j_in, k_bot]
        
        val_bot = numer_bot / (denom_bot + 1e-20)
        val_bot = val_bot * dvm[s_i_in, s_j_in]
        
        vf_out[s_i_in, s_j_in, k_bot] = val_bot
        vf_next = val_bot

        # --- Backward Substitution ---
        for k in range(k_bot - 1, -1, -1):
            val = ee_frozen[s_i_in, s_j_in, k] * vf_next + gg_frozen[s_i_in, s_j_in, k]
            val = val * dvm[s_i_in, s_j_in]
            
            vf_out[s_i_in, s_j_in, k] = val
            vf_next = val

        vf_bot_slice = vf_out[s_i_in, s_j_in, k_bot].clone()
        wvbot[s_i_in, s_j_in] = -tps_local[s_i_in, s_j_in] * vf_bot_slice

        return a, c, ee, gg_store, tps_local, vf_out, wvbot

    def dens(self):
        """
        Calculates (density-1000.)/rhoref using Mellor (1991) Equation of State.
        Inputs: self.t, self.s (Potential Temp, Salinity)
        Outputs: self.rho (In-place update)
        """

        # Helper for broadcasting
        def _b(tensor_2d):
            return tensor_2d.unsqueeze(2)

        # Slice for K loop (k=1:kbm1 in MATLAB -> 0:kbm1 in Python)
        k_sl = slice(0, self.kbm1)

        # 1. Apply Bias
        # tr = ti + tbias
        tr = self.t[:, :, k_sl] + self.tbias
        # sr = si + sbias
        sr = self.s[:, :, k_sl] + self.sbias

        # 2. Pre-calculate Powers of T
        tr2 = tr * tr
        tr3 = tr2 * tr
        tr4 = tr3 * tr

        # 3. Calculate Pressure (in Bars)
        # p = grav * rhoref * (-zz(k) * h(i,j)) * 1.e-5
        # h is (im, jm) -> (im, jm, 1)
        # zz is (kb) -> slice to kbm1 -> (1, 1, kbm1)
        
        h_3d = _b(self.h)
        zz_3d = self.zz[k_sl].view(1, 1, -1)
        
        # Note: zz is negative, so -zz is positive depth factor
        p = self.grav * self.rhoref * (-zz_3d * h_3d) * 1.0e-5

        # 4. Calculate Density (rhor) - Seawater EOS at Atmospheric Pressure
        # Coefficients from Mellor (1991) / POM code
        
        # Pure water density terms
        rhor = -0.157406 + 6.793952e-2 * tr - 9.095290e-3 * tr2 + \
               1.001685e-4 * tr3 - 1.120083e-6 * tr4 + 6.536332e-9 * tr4 * tr

        # Salinity terms
        rhor += (0.824493 - 4.0899e-3 * tr + 7.6438e-5 * tr2 - 8.2467e-7 * tr3 + \
                 5.3875e-9 * tr4) * sr
                 
        # Salinity^1.5 terms
        rhor += (-5.72466e-3 + 1.0227e-4 * tr - 1.6546e-6 * tr2) * torch.pow(torch.clamp(torch.abs(sr), min=1e-8), 1.5)
        
        # Salinity^2 terms
        rhor += 4.8314e-4 * sr * sr

        # 5. Calculate Sound Speed Squared (cr^2) and Pressure Correction
        # Note: Original code calculates cr, then uses cr*cr. 
        # We calculate cr, then adjust rhor.
        
        # cr = 1449.1 + ...
        cr = 1449.1 + 0.0821 * p + 4.55 * tr - 0.045 * tr2 + 1.34 * (sr - 35.0)
        
        # Denominator cr^2
        cr2 = cr * cr
        
        # rhor = rhor + 1e5 * p / cr^2 * (1 - 2*p/cr^2)
        term_p = 1.0e5 * p / cr2
        rhor += term_p * (1.0 - 2.0 * p / cr2)

        # 6. Finalize Output (rhoo)
        # rhoo(i,j,k) = rhor / rhoref * fsm(i,j)
        
        self.rho[:, :, k_sl] = (rhor / self.rhoref) * _b(self.fsm)

        # Return for compatibility with calling signature if needed, 
        # though we updated self.rho in-place.
        return self.s, self.t, self.rho

    def baropg(self):
        """
        Calculates baroclinic pressure gradient (drhox, drhoy).
        Includes sigma coordinate correction terms.
        """
        
        # Helper for 2D -> 3D broadcasting
        def _b(tensor_2d):
            return tensor_2d.unsqueeze(2)

        # 1. Subtract Mean Density (Anomaly)
        self.rho = self.rho - self.rmean
        
        # Initialize outputs
        drhox = torch.zeros((self.im, self.jm, self.kb), device=self.device, dtype=self.t.dtype)
        drhoy = torch.zeros((self.im, self.jm, self.kb), device=self.device, dtype=self.t.dtype)
        
        # Slices for inner domain (j=2:jmm1, i=2:imm1) -> Python (1:jmm1, 1:imm1)
        s_i = slice(1, self.imm1)
        s_im1 = slice(0, self.imm1 - 1)
        s_j = slice(1, self.jmm1)
        s_jm1 = slice(0, self.jmm1 - 1)
        
        # =================================================================
        # X-Component (drhox)
        # =================================================================
        
        # --- Surface Layer (k=1 in MATLAB -> k=0 in Python) ---
        # drhox(1) = 0.5 * g * (-zz(1)) * (dt(i)+dt(i-1)) * (rho(i)-rho(i-1))
        
        term_dt = self.dt[s_i, s_j] + self.dt[s_im1, s_j]
        drho = self.rho[s_i, s_j, 0] - self.rho[s_im1, s_j, 0]
        
        # zz is 1D. zz[0] is scalar for this layer
        drhox[s_i, s_j, 0] = 0.5 * self.grav * (-self.zz[0]) * term_dt * drho
        
        # --- Internal Layers (Integration) ---
        # We need to compute increments for k=2:kbm1 (Python 1:kbm1)
        # But formula relies on recursive sum: val(k) = val(k-1) + increment(k)
        # We can calculate all increments and then cumsum.
        
        # Slices for K
        # We need k (current) and k-1 (previous)
        # Python k ranges from 1 to kbm1-1.
        # To vectorize, we compute terms for k=1..kbm1-1 based on k-1 and k.
        
        k_sl = slice(1, self.kbm1)     # k (1 to kbm1-1)
        k_m1 = slice(0, self.kbm1 - 1) # k-1 (0 to kbm1-2)
        
        # zz terms: Need shape (1, 1, k_len)
        zz_k = self.zz[k_sl].view(1, 1, -1)
        zz_km1 = self.zz[k_m1].view(1, 1, -1)
        
        # Term 1: grav * 0.25 * (zz(k-1)-zz(k)) * (dt(i)+dt(i-1)) * (rho_diff_sum)
        # rho_diff_sum = (rho(i,k)-rho(i-1,k)) + (rho(i,k-1)-rho(i-1,k-1))
        
        rho_i_k   = self.rho[s_i, s_j, k_sl]
        rho_im1_k = self.rho[s_im1, s_j, k_sl]
        rho_i_km1 = self.rho[s_i, s_j, k_m1]
        rho_im1_km1 = self.rho[s_im1, s_j, k_m1]
        
        diff_k = rho_i_k - rho_im1_k
        diff_km1 = rho_i_km1 - rho_im1_km1
        
        term1 = self.grav * 0.25 * (zz_km1 - zz_k) * _b(term_dt) * (diff_k + diff_km1)
        
        # Term 2: grav * 0.25 * (zz(k-1)+zz(k)) * (dt(i)-dt(i-1)) * (rho_sum_diff)
        # rho_sum_diff = (rho(i,k)+rho(i-1,k)) - (rho(i,k-1)+rho(i-1,k-1))
        
        dt_diff = self.dt[s_i, s_j] - self.dt[s_im1, s_j]
        
        sum_k = rho_i_k + rho_im1_k
        sum_km1 = rho_i_km1 + rho_im1_km1
        
        term2 = self.grav * 0.25 * (zz_km1 + zz_k) * _b(dt_diff) * (sum_k - sum_km1)
        
        # Total Increment
        increment = term1 + term2
        
        # Cumulative Sum (Integration)
        # Note: drhox[k] = drhox[k-1] + increment
        # We have initial value at k=0.
        # result[0] = init + inc[0] (for k=1)
        # result[1] = init + inc[0] + inc[1] (for k=2)
        
        integral = torch.cumsum(increment, dim=2)
        
        # Add the surface value (k=0) to all subsequent layers
        drhox[s_i, s_j, k_sl] = _b(drhox[s_i, s_j, 0]) + integral
        
        # --- Metric Scaling ---
        # drhox = .25 * (dt(i)+dt(i-1)) * drhox * dum * (dy(i)+dy(i-1))
        
        # Need full range k=0:kbm1
        k_full = slice(0, self.kbm1)
        
        dy_sum = self.dy[s_i, s_j] + self.dy[s_im1, s_j]
        
        scale_factor = 0.25 * term_dt * self.dum[s_i, s_j] * dy_sum
        drhox[s_i, s_j, k_full] *= _b(scale_factor)

        # =================================================================
        # Y-Component (drhoy)
        # =================================================================
        
        # --- Surface Layer (k=0) ---
        # drhoy(1) = 0.5 * g * (-zz(1)) * (dt(j)+dt(j-1)) * (rho(j)-rho(j-1))
        
        term_dt = self.dt[s_i, s_j] + self.dt[s_i, s_jm1]
        drho = self.rho[s_i, s_j, 0] - self.rho[s_i, s_jm1, 0]
        
        drhoy[s_i, s_j, 0] = 0.5 * self.grav * (-self.zz[0]) * term_dt * drho
        
        # --- Internal Layers (Integration) ---
        # Same logic as X-component but with J-indices
        
        rho_j_k   = self.rho[s_i, s_j, k_sl]
        rho_jm1_k = self.rho[s_i, s_jm1, k_sl]
        rho_j_km1 = self.rho[s_i, s_j, k_m1]
        rho_jm1_km1 = self.rho[s_i, s_jm1, k_m1]
        
        diff_k = rho_j_k - rho_jm1_k
        diff_km1 = rho_j_km1 - rho_jm1_km1
        
        term1 = self.grav * 0.25 * (zz_km1 - zz_k) * _b(term_dt) * (diff_k + diff_km1)
        
        dt_diff = self.dt[s_i, s_j] - self.dt[s_i, s_jm1]
        
        sum_k = rho_j_k + rho_jm1_k
        sum_km1 = rho_j_km1 + rho_jm1_km1
        
        term2 = self.grav * 0.25 * (zz_km1 + zz_k) * _b(dt_diff) * (sum_k - sum_km1)
        
        increment = term1 + term2
        integral = torch.cumsum(increment, dim=2)
        
        drhoy[s_i, s_j, k_sl] = _b(drhoy[s_i, s_j, 0]) + integral
        
        # --- Metric Scaling ---
        # drhoy = .25 * (dt(j)+dt(j-1)) * drhoy * dvm * (dx(j)+dx(j-1))
        
        dx_sum = self.dx[s_i, s_j] + self.dx[s_i, s_jm1]
        
        scale_factor = 0.25 * term_dt * self.dvm[s_i, s_j] * dx_sum
        drhoy[s_i, s_j, k_full] *= _b(scale_factor)

        # =================================================================
        # Finalize
        # =================================================================
        
        # Apply Ramp
        drhox *= self.ramp
        drhoy *= self.ramp
        
        # Restore Mean Density
        self.rho = self.rho + self.rmean
        
        return self.rho, drhox, drhoy
