import math
import numpy as np
import netCDF4 as nc
import torch
import torch.nn.functional as F


torch.set_default_dtype(torch.float32)


class TracersMixin:
    def advq(self, qb, q, qf, xflux, yflux, 
             dt, dx, dy, dz, u, v, w, aam, h, dum, dvm, art, etb, etf, 
             im, jm, imm1, jmm1, kbm1, dti2):
        """
        Calculates horizontal advection and diffusion, and vertical advection 
        for turbulent quantities (q2, q2l).
        """

        # Helper for 2D -> 3D broadcasting
        def _b(tensor_2d):
            return tensor_2d.unsqueeze(2)

        # Slice Definitions for K Loop (k=2:kbm1 in MATLAB -> 1:kbm1 in Python)
        k_sl = slice(1, kbm1)       # Current k (index 1 to kbm1-1)
        k_m1 = slice(0, kbm1 - 1)   # Previous k (index 0 to kbm1-2)
        
        # Ranges for Horizontal Loops (j=2:jm, i=2:im) -> Python 1:jm, 1:im
        s_i = slice(1, im)
        s_im1 = slice(0, im - 1)
        s_j = slice(1, jm)
        s_jm1 = slice(0, jm - 1)

        # =================================================================
        # 1. Horizontal Advection
        # =================================================================
        
        # --- XFlux ---
        # 0.125 * (q(i)+q(i-1)) * (dt(i)+dt(i-1)) * (u(k)+u(k-1))
        
        term_q = q[s_i, s_j, k_sl] + q[s_im1, s_j, k_sl]
        term_dt = _b(dt[s_i, s_j] + dt[s_im1, s_j])
        term_u = u[s_i, s_j, k_sl] + u[s_i, s_j, k_m1]
        
        xflux[s_i, s_j, k_sl] = 0.125 * term_q * term_dt * term_u
        
        # --- YFlux ---
        # 0.125 * (q(j)+q(j-1)) * (dt(j)+dt(j-1)) * (v(k)+v(k-1))
        
        term_q = q[s_i, s_j, k_sl] + q[s_i, s_jm1, k_sl]
        term_dt = _b(dt[s_i, s_j] + dt[s_i, s_jm1])
        term_v = v[s_i, s_j, k_sl] + v[s_i, s_j, k_m1]
        
        yflux[s_i, s_j, k_sl] = 0.125 * term_q * term_dt * term_v

        # =================================================================
        # 2. Horizontal Diffusion
        # =================================================================
        
        # --- XFlux Update ---
        # Formula: -0.25 * (aam_sum) * (h_sum) * (qb_diff) * dum / dx_sum
        
        aam_sum = aam[s_i, s_j, k_sl] + aam[s_im1, s_j, k_sl] + \
                  aam[s_i, s_j, k_m1] + aam[s_im1, s_j, k_m1]
                  
        h_sum = _b(h[s_i, s_j] + h[s_im1, s_j])
        qb_diff = qb[s_i, s_j, k_sl] - qb[s_im1, s_j, k_sl]
        dx_sum = _b(dx[s_i, s_j] + dx[s_im1, s_j])
        dum_b = _b(dum[s_i, s_j])
        
        diff_term_x = 0.25 * aam_sum * h_sum * qb_diff * dum_b / dx_sum
        
        xflux[s_i, s_j, k_sl] -= diff_term_x
        
        # --- YFlux Update ---
        # Formula: -0.25 * (aam_sum) * (h_sum) * (qb_diff) * dvm / dy_sum
        
        aam_sum = aam[s_i, s_j, k_sl] + aam[s_i, s_jm1, k_sl] + \
                  aam[s_i, s_j, k_m1] + aam[s_i, s_jm1, k_m1]
                  
        h_sum = _b(h[s_i, s_j] + h[s_i, s_jm1])
        qb_diff = qb[s_i, s_j, k_sl] - qb[s_i, s_jm1, k_sl]
        dy_sum = _b(dy[s_i, s_j] + dy[s_i, s_jm1])
        dvm_b = _b(dvm[s_i, s_j])
        
        diff_term_y = 0.25 * aam_sum * h_sum * qb_diff * dvm_b / dy_sum
        
        yflux[s_i, s_j, k_sl] -= diff_term_y
        
        # --- Final Scaling ---
        # xflux = 0.5 * (dy + dy(i-1)) * xflux
        dy_sum = _b(dy[s_i, s_j] + dy[s_im1, s_j])
        xflux[s_i, s_j, k_sl] *= 0.5 * dy_sum
        
        # yflux = 0.5 * (dx + dx(j-1)) * yflux
        dx_sum = _b(dx[s_i, s_j] + dx[s_i, s_jm1])
        yflux[s_i, s_j, k_sl] *= 0.5 * dx_sum

        # =================================================================
        # 3. Vertical Advection and Time Stepping
        # =================================================================
        # Ranges: j=2:jmm1, i=2:imm1 -> Python 1:jmm1, 1:imm1
        s_i = slice(1, imm1)
        s_ip1 = slice(2, imm1 + 1)
        s_j = slice(1, jmm1)
        s_jp1 = slice(2, jmm1 + 1)
        
        # Vertical Indices (k-1, k, k+1)
        # Note: In loop k=2:kbm1.
        # k-1 corresponds to Python index k-1 (0 to kbm1-2)
        # k+1 corresponds to Python index k+1 (2 to kbm1) [Requires q has size kbm1+1 or kb]
        # Our tensors are (im, jm, kb). kbm1 = kb-1.
        # So range k=1 to kbm1-1 (exclusive of kbm1).
        # k-1: 0 to kbm1-2.
        # k+1: 2 to kbm1.
        # This fits within [0, kb-1].
        
        k_sl = slice(1, kbm1)
        k_m1 = slice(0, kbm1 - 1)
        k_p1 = slice(2, kbm1 + 1)
        
        # --- Vertical Flux Term ---
        # (w(k-1)*q(k-1) - w(k+1)*q(k+1)) * art / (dz(k) + dz(k-1))
        
        # Note: dz is 1D (kb). 
        # dz(k) in MATLAB is Python dz[k] (index 1..). 
        # dz(k-1) in MATLAB is Python dz[k-1] (index 0..).
        dz_sum = (dz[k_sl] + dz[k_m1]).view(1, 1, -1) # Broadcast to 3D
        
        vert_term = (w[s_i, s_j, k_m1] * q[s_i, s_j, k_m1] - 
                     w[s_i, s_j, k_p1] * q[s_i, s_j, k_p1]) * \
                    _b(art[s_i, s_j]) / dz_sum
        
        # --- Horizontal Flux Divergence ---
        # xflux(i+1) - xflux(i) + yflux(j+1) - yflux(j)
        horiz_div = xflux[s_ip1, s_j, k_sl] - xflux[s_i, s_j, k_sl] + \
                    yflux[s_i, s_jp1, k_sl] - yflux[s_i, s_j, k_sl]
        
        # Combine into qf provisional
        qf[s_i, s_j, k_sl] = vert_term + horiz_div
        
        # --- Time Stepping (Leapfrog/Forward) ---
        # qf = ((h + etb)*art*qb - dti2*qf) / ((h + etf)*art)
        
        term_b = _b((h[s_i, s_j] + etb[s_i, s_j]) * art[s_i, s_j])
        term_f = _b((h[s_i, s_j] + etf[s_i, s_j]) * art[s_i, s_j])
        
        numer = term_b * qb[s_i, s_j, k_sl] - dti2 * qf[s_i, s_j, k_sl]
        denom = term_f
        
        # Avoid division by zero
        qf[s_i, s_j, k_sl] = numer / (denom + 1e-20)

        return qb, q, qf, xflux, yflux

    def profq(self, sm, sh, dh, cc, 
              ee, gg, l, kq, km, kh, 
              uf, vf, q2, q2b, q2lb, a, c, 
              h, etf, dti2, umol, dzz, grav, rho, kappa, u, v, dt, small, fsm, 
              im, jm, kb, imm1, jmm1, kbm1, tbias, sbias, dz, 
              wusurf, wubot, wvsurf, wvbot, t, s, rhoref, zz, z):
        """
        [FIXED FINAL - FULL UNROLL]: 
        1. Snapshot Inputs.
        2. Use Python Lists for BOTH Forward (ee/gg) and Backward sweeps.
        3. No in-place tensor modifications at all in the Thomas Algorithm.
        """
        def _b(tensor_2d): return tensor_2d.unsqueeze(2)

        # =========================================================
        # 1. SNAPSHOT INPUTS (Read-Only)
        # =========================================================
        l_in = l.clone()
        km_in = km.clone(); kh_in = kh.clone(); kq_in = kq.clone()
        q2_in = q2.clone(); q2b_in = q2b.clone()
        q2lb_in = q2lb.clone()
        uf_in = uf.clone(); vf_in = vf.clone()
        
        h_in = h.clone(); etf_in = etf.clone()
        u_in = u.clone(); v_in = v.clone()
        t_in = t.clone(); s_in = s.clone()
        rho_in = rho.clone()
        wusurf_in = wusurf.clone(); wvsurf_in = wvsurf.clone()
        wubot_in = wubot.clone(); wvbot_in = wvbot.clone()

        # Scratchpads
        # Note: We do NOT pre-allocate a, c, ee, gg tensors anymore.
        # We compute them layer by layer.
        
        kn = torch.zeros((im, jm, kb), device=self.device)
        boygr = torch.zeros((im, jm, kb), device=self.device)
        gh = torch.zeros((im, jm, kb), device=self.device)
        stf = torch.zeros((im, jm, kb), device=self.device)
        dtef = torch.zeros((im, jm, kb), device=self.device)
        l0 = torch.zeros((im, jm), device=self.device)

        # Constants
        a1=0.92; b1=16.6; a2=0.74; b2=10.1; c1=0.08
        e1=1.8; e2=1.33; sef=1.0
        cbcnst=100.0; surfl=2.0e5; shiw=0.0
        
        dh_2d = h_in + etf_in
        dh_3d = _b(dh_2d)

        # --- 2. Calculate Diffusion Coefficients ---
        # a and c are still computed as tensors first since they are simple parallel ops
        # (Though technically a and c lists would be safer, they are not recurrent)
        # We will keep them as tensors but clone them immediately.
        
        # Pre-allocate a and c just for this step
        a_tensor = torch.zeros((im, jm, kb), device=self.device)
        c_tensor = torch.zeros((im, jm, kb), device=self.device)

        for k in range(1, kbm1):
            denom_a = dzz[k-1] * dz[k] * dh_3d[:,:,0] * dh_3d[:,:,0]
            denom_c = dzz[k-1] * dz[k-1] * dh_3d[:,:,0] * dh_3d[:,:,0]
            a_tensor[:, :, k] = -dti2 * (kq_in[:, :, k+1] + kq_in[:, :, k] + 2.0 * umol) * 0.5 / denom_a
            c_tensor[:, :, k] = -dti2 * (kq_in[:, :, k-1] + kq_in[:, :, k] + 2.0 * umol) * 0.5 / denom_c
        
        # Freeze a/c
        a_frozen = a_tensor.clone()
        c_frozen = c_tensor.clone()

        # --- 3. Surface & Bottom Prep ---
        const1 = (16.6**(2.0/3.0)) * sef
        
        s_i = slice(0, imm1); s_ip1 = slice(1, imm1 + 1)
        s_j = slice(0, jmm1); s_jp1 = slice(1, jmm1 + 1)
        
        u_stress = 0.5 * (wusurf_in[s_i, s_j] + wusurf_in[s_ip1, s_j])
        v_stress = 0.5 * (wvsurf_in[s_i, s_j] + wvsurf_in[s_i, s_jp1])
        utau2 = torch.sqrt(u_stress**2 + v_stress**2 + 1e-8)
        
        l0[s_i, s_j] = surfl * utau2 / grav
        
        u_bot_val = 0.5 * (wubot_in[s_i, s_j] + wubot_in[s_ip1, s_j])
        v_bot_val = 0.5 * (wvbot_in[s_i, s_j] + wvbot_in[s_i, s_jp1])
        
        uf_bottom_val = torch.zeros((im, jm), device=self.device)
        uf_bottom_val[s_i, s_j] = torch.sqrt(u_bot_val**2 + v_bot_val**2 + 1e-8) * const1

        # --- 4. Buoyancy ---
        tp = t_in[:, :, :kbm1] + tbias
        sp = s_in[:, :, :kbm1] + sbias
        zz_sl = zz[:kbm1].view(1, 1, -1)
        p = grav * rhoref * (-zz_sl * dh_3d[:,:,:kbm1]) * 1.e-4
        
        cc_slice = 1449.1 + 0.00821*p + 4.55*tp - 0.045*(tp**2) + 1.34*(sp - 35.0)
        cc_slice_safe = torch.where(torch.abs(cc_slice) < 1e-6, torch.full_like(cc_slice, 1e-6), cc_slice)
        radicand_cc = (1.0 - 0.01642 * p / cc_slice_safe) * (1.0 - 0.40 * p / (cc_slice_safe**2))
        denom_cc = torch.sqrt(torch.clamp(radicand_cc, min=1e-8))
        cc_val = cc_slice_safe / denom_cc
        
        for k in range(1, kbm1):
            drho = rho_in[:, :, k-1] - rho_in[:, :, k]
            term1 = grav * drho / (dzz[k-1] * dh_2d)
            term2 = (grav**2) * 2.0 / (cc_val[:, :, k-1]**2 + cc_val[:, :, k]**2)
            boygr[:, :, k] = term1 + term2

        # --- 5. Length Scale ---
        q2b_abs = torch.abs(q2b_in)
        q2lb_abs = torch.abs(q2lb_in)
        
        l_out = torch.zeros((im, jm, kb), device=self.device)
        l_out[:, :, 1:kbm1] = q2lb_abs[:, :, 1:kbm1] / (q2b_abs[:, :, 1:kbm1] + 1e-20)
        
        l0_3d = _b(l0)
        z_tensor = z[:kbm1].view(1, 1, -1)
        mask_surf = z_tensor > -0.5
        
        l_slice_safe = l_out[:,:,:kbm1].clone()
        l_update = torch.max(l_slice_safe, kappa * l0_3d)
        l_out[:,:,:kbm1] = torch.where(mask_surf, l_update, l_slice_safe)
        
        l_slice_for_gh = l_out[:, :, 1:kbm1].clone()
        gh[:, :, 1:kbm1] = (l_slice_for_gh**2) * boygr[:, :, 1:kbm1] / (q2b_abs[:, :, 1:kbm1] + 1e-20)
        gh = torch.clamp(gh, max=0.028)
        
        l_out[:, :, 0] = kappa * l0
        l_out[:, :, kb-1] = 0.0
        gh[:, :, 0] = 0.0
        gh[:, :, kb-1] = 0.0

        # --- 6. Shear Production ---
        s_i_sh = slice(1, imm1); s_ip1_sh = slice(2, imm1+1)
        s_j_sh = slice(1, jmm1); s_jp1_sh = slice(2, jmm1+1)
        
        for k in range(1, kbm1):
            du_sq = (u_in[s_i_sh, s_j_sh, k] - u_in[s_i_sh, s_j_sh, k-1] + 
                     u_in[s_ip1_sh, s_j_sh, k] - u_in[s_ip1_sh, s_j_sh, k-1])**2
            dv_sq = (v_in[s_i_sh, s_j_sh, k] - v_in[s_i_sh, s_j_sh, k-1] + 
                     v_in[s_i_sh, s_jp1_sh, k] - v_in[s_i_sh, s_jp1_sh, k-1])**2
            
            denom = (dzz[k-1] * dh_2d[s_i_sh, s_j_sh])**2
            kn[s_i_sh, s_j_sh, k] = km_in[s_i_sh, s_j_sh, k] * 0.25 * sef * (du_sq + dv_sq) / denom
            kn[s_i_sh, s_j_sh, k] = kn[s_i_sh, s_j_sh, k] - shiw * km_in[s_i_sh, s_j_sh, k] * boygr[s_i_sh, s_j_sh, k] + \
                                    kh_in[s_i_sh, s_j_sh, k] * boygr[s_i_sh, s_j_sh, k]

        # ==========================================
        # 7. Solve for q2 (TKE) - FULL LIST METHOD
        # ==========================================
        stf.fill_(1.0)
        # Guard against near-zero q2b values.
        dtef = torch.sqrt(q2b_abs + 1e-8) * stf / (b1 * l_out + small)
        
        # Lists for Forward Sweep
        ee_list = [torch.zeros((im, jm), device=self.device) for _ in range(kb)]
        gg_list = [torch.zeros((im, jm), device=self.device) for _ in range(kb)]
        
        # Surface BC
        # ee_list[0] is 0.0
        gg_list[0][s_i, s_j] = (15.8 * cbcnst)**(2.0/3.0) * utau2

        # Forward Sweep (q2)
        for k in range(1, kbm1):
            # Read from previous step in list
            ee_prev = ee_list[k-1]
            gg_prev = gg_list[k-1]
            
            denom = a_frozen[:,:,k] + c_frozen[:,:,k] * (1.0 - ee_prev) - (2.0 * dti2 * dtef[:,:,k] + 1.0)
            ee_val = a_frozen[:,:,k] / (denom + 1e-20)
            
            rhs = -2.0 * dti2 * kn[:,:,k] + c_frozen[:,:,k] * gg_prev - uf_in[:,:,k]
            gg_val = rhs / (denom + 1e-20)
            
            # Store new tensors in list
            ee_list[k] = ee_val
            gg_list[k] = gg_val

        # Convert to Tensor for Backward Sweep (Optional, but convenient)
        # Using stack breaks the history link to list elements if we access them by index?
        # Actually, let's keep using the list for backward sweep for consistency.
        
        # Backward Sweep (q2)
        uf_list = [None] * kb
        uf_list[kb-1] = uf_bottom_val

        for k in range(kbm1 - 1, -1, -1):
            val_next = uf_list[k+1]
            # val_curr = ee[k] * val_next + gg[k]
            val_curr = ee_list[k] * val_next + gg_list[k]
            uf_list[k] = val_curr
            
        uf_out = torch.stack(uf_list, dim=2)

        # ==========================================
        # 8. Solve for q2l (Length Scale) - FULL LIST METHOD
        # ==========================================
        # Fresh Lists
        ee_list_q2l = [torch.zeros((im, jm), device=self.device) for _ in range(kb)]
        gg_list_q2l = [torch.zeros((im, jm), device=self.device) for _ in range(kb)]
        
        z_surf = z[0]; z_bot = z[kb-1]
        
        # Forward Sweep (q2l)
        for k in range(1, kbm1):
            ee_prev = ee_list_q2l[k-1]
            gg_prev = gg_list_q2l[k-1]
            
            wall_term = (1.0 / (torch.abs(z[k] - z_surf) + 1e-8) + 1.0 / (torch.abs(z[k] - z_bot) + 1e-8)) * \
                        l_out[:,:,k] / (dh_2d * kappa)
            dtef_k = dtef[:,:,k] * (1.0 + e2 * (wall_term**2))
            
            denom = a_frozen[:,:,k] + c_frozen[:,:,k] * (1.0 - ee_prev) - (dti2 * dtef_k + 1.0)
            ee_val = a_frozen[:,:,k] / (denom + 1e-20)
            
            rhs = dti2 * (-kn[:,:,k] * l_out[:,:,k] * e1) + c_frozen[:,:,k] * gg_prev - vf_in[:,:,k]
            gg_val = rhs / (denom + 1e-20)
            
            ee_list_q2l[k] = ee_val
            gg_list_q2l[k] = gg_val

        # Backward Sweep (q2l)
        vf_list = [None] * kb
        vf_list[kb-1] = torch.zeros((im, jm), device=self.device) # Bottom BC
        
        for k in range(kbm1 - 1, -1, -1):
            val_next = vf_list[k+1]
            val_curr = ee_list_q2l[k] * val_next + gg_list_q2l[k]
            vf_list[k] = val_curr
            
        vf_out = torch.stack(vf_list, dim=2)

        # ==========================================
        # 9. Clip & Update
        # ==========================================
        s_k = slice(1, kbm1)
        mask_small = (uf_out[:, :, s_k] <= small) | (vf_out[:, :, s_k] <= small)
        
        uf_sl = uf_out[:, :, s_k]
        vf_sl = vf_out[:, :, s_k]
        dt_sl = _b(dt)
        
        uf_final = uf_out.clone()
        vf_final = vf_out.clone()
        
        uf_sl_new = torch.where(mask_small, uf_sl * 0.0 + small, uf_sl)
        vf_sl_new = torch.where(mask_small, 0.1 * dt_sl * small, vf_sl)
        
        uf_final[:, :, s_k] = uf_sl_new
        vf_final[:, :, s_k] = vf_sl_new

        # --- Update Coefs ---
        coef4 = 18.0 * a1 * a1 + 9.0 * a1 * a2
        coef5 = 9.0 * a1 * a2
        coef1 = a2 * (1.0 - 6.0 * a1 / b1)
        coef2 = 3.0 * a2 * b2 + 18.0 * a1 * a2
        coef3 = a1 * (1.0 - 3.0 * c1 - 6.0 * a1 / b1)
        
        denom_sh = 1.0 - coef2 * gh
        sh = coef1 / (denom_sh + 1e-20)
        
        num_sm = coef3 + sh * coef4 * gh
        denom_sm = 1.0 - coef5 * gh
        sm = num_sm / (denom_sm + 1e-20)
        
        # Guard the square root against zeros and very small values.
        kn_final = l_out * torch.sqrt(torch.abs(q2_in) + 1e-8)
        
        kq_out = (kn_final * 0.41 * sh + kq_in) * 0.5
        km_out = (kn_final * sm + km_in) * 0.5
        kh_out = (kn_final * sh + kh_in) * 0.5
        
        for arr in [km_out, kh_out]:
            arr[im-1, :, :] = arr[im-2, :, :] * fsm[im-1, :].unsqueeze(1)
            arr[0, :, :]    = arr[1, :, :]    * fsm[0, :].unsqueeze(1)
            arr[:, jm-1, :] = arr[:, jm-2, :] * fsm[:, jm-1].unsqueeze(1)
            arr[:, 0, :]    = arr[:, 1, :]    * fsm[:, 0].unsqueeze(1)

        return sm, sh, dh, cc_val, ee, gg, l_out, kq_out, km_out, kh_out, uf_final, vf_final, q2b_in, q2lb_in, a_tensor, c_tensor

    def advt1(self, fb, f, fclim, ff, xflux, yflux, zflux, 
              dt, u, v, aam, tprni, dum, dvm, w, art, etb, etf, 
              dti2, dx, dy, dz, h, im, jm, kb, imm1, jmm1, kbm1):
        """
        Integrates conservative scalar equations (T/S) using Centered Scheme.
        [FIXED for Autograd]: 
        1. Cloned inputs (f, fb) to allow boundary modification without error.
        2. Localized scratchpads (xflux, yflux, zflux, ff) to avoid overwriting class attributes.
        """

        # Helper for 2D -> 3D broadcasting
        def _b(tensor_2d):
            return tensor_2d.unsqueeze(2)

        # Clone inputs and local scratch tensors to avoid autograd issues from
        # later in-place updates.
        f = f.clone()
        fb = fb.clone()
        u = u.clone()
        v = v.clone()
        w = w.clone()
        dt = dt.clone()
        art = art.clone()

        # Use local scratch tensors instead of global cached buffers.
        xflux = torch.zeros_like(f)
        yflux = torch.zeros_like(f)
        zflux = torch.zeros_like(f)
        ff = torch.zeros_like(f)
        # =========================================================

        # 1. Fill Bottom Boundary (kb)
        # f(i,j,kb) = f(i,j,kbm1) -> Python index kb-1 = kbm1-1
        f[:, :, kb-1] = f[:, :, kbm1-1]
        fb[:, :, kb-1] = fb[:, :, kbm1-1]

        # Slice definitions
        k_sl = slice(0, kbm1)
        
        s_i = slice(1, im)
        s_im1 = slice(0, im - 1)
        s_j = slice(1, jm)
        s_jm1 = slice(0, jm - 1)

        # =================================================================
        # 2. Advective Fluxes (Horizontal)
        # =================================================================
        
        # --- XFlux ---
        term_dt = _b(dt[s_i, s_j] + dt[s_im1, s_j])
        term_f = f[s_i, s_j, k_sl] + f[s_im1, s_j, k_sl]
        xflux[s_i, s_j, k_sl] = 0.25 * term_dt * term_f * u[s_i, s_j, k_sl]
        
        # --- YFlux ---
        term_dt = _b(dt[s_i, s_j] + dt[s_i, s_jm1])
        term_f = f[s_i, s_j, k_sl] + f[s_i, s_jm1, k_sl]
        yflux[s_i, s_j, k_sl] = 0.25 * term_dt * term_f * v[s_i, s_j, k_sl]

        # =================================================================
        # 3. Diffusive Fluxes (Horizontal)
        # =================================================================
        
        # Calculate anomaly: fb = fb - fclim
        fb_anom = fb - fclim
        
        # --- XFlux Update (Diffusion) ---
        aam_avg = 0.5 * (aam[s_i, s_j, k_sl] + aam[s_im1, s_j, k_sl])
        h_sum = _b(h[s_i, s_j] + h[s_im1, s_j])
        fb_diff = fb_anom[s_i, s_j, k_sl] - fb_anom[s_im1, s_j, k_sl]
        dx_sum = _b(dx[s_i, s_j] + dx[s_im1, s_j])
        dum_b = _b(dum[s_i, s_j])
        
        diff_x = aam_avg * h_sum * tprni * fb_diff * dum_b / dx_sum
        xflux[s_i, s_j, k_sl] -= diff_x
        
        # Scale XFlux by average DY
        dy_avg = _b(0.5 * (dy[s_i, s_j] + dy[s_im1, s_j]))
        xflux[s_i, s_j, k_sl] *= dy_avg
        
        # --- YFlux Update (Diffusion) ---
        aam_avg = 0.5 * (aam[s_i, s_j, k_sl] + aam[s_i, s_jm1, k_sl])
        h_sum = _b(h[s_i, s_j] + h[s_i, s_jm1])
        fb_diff = fb_anom[s_i, s_j, k_sl] - fb_anom[s_i, s_jm1, k_sl]
        dy_sum = _b(dy[s_i, s_j] + dy[s_i, s_jm1])
        dvm_b = _b(dvm[s_i, s_j])
        
        diff_y = aam_avg * h_sum * tprni * fb_diff * dvm_b / dy_sum
        yflux[s_i, s_j, k_sl] -= diff_y
        
        # Scale YFlux by average DX
        dx_avg = _b(0.5 * (dx[s_i, s_j] + dx[s_i, s_jm1]))
        yflux[s_i, s_j, k_sl] *= dx_avg

        # =================================================================
        # 4. Vertical Advection (ZFlux)
        # =================================================================
        s_i_in = slice(1, imm1)
        s_j_in = slice(1, jmm1)
        
        # Surface Flux
        zflux[s_i_in, s_j_in, 0] = f[s_i_in, s_j_in, 0] * w[s_i_in, s_j_in, 0] * art[s_i_in, s_j_in]
        
        # Bottom Flux
        zflux[s_i_in, s_j_in, kb-1] = 0.0
        
        # Internal Fluxes
        k_sl_z = slice(1, kbm1)     # k
        k_m1_z = slice(0, kbm1 - 1) # k-1
        
        term_f = 0.5 * (f[s_i_in, s_j_in, k_m1_z] + f[s_i_in, s_j_in, k_sl_z])
        zflux[s_i_in, s_j_in, k_sl_z] = term_f * w[s_i_in, s_j_in, k_sl_z] * _b(art[s_i_in, s_j_in])

        # =================================================================
        # 5. Time Stepping
        # =================================================================
        s_ip1 = slice(2, imm1 + 1)
        s_jp1 = slice(2, jmm1 + 1)
        
        dz_k = dz[0:kbm1].view(1, 1, -1)
        
        k_sl = slice(0, kbm1)
        k_p1 = slice(1, kbm1 + 1)
        
        horiz_div = xflux[s_ip1, s_j_in, k_sl] - xflux[s_i_in, s_j_in, k_sl] + \
                    yflux[s_i_in, s_jp1, k_sl] - yflux[s_i_in, s_j_in, k_sl]
                    
        vert_div = (zflux[s_i_in, s_j_in, k_sl] - zflux[s_i_in, s_j_in, k_p1]) / dz_k
        
        ff[s_i_in, s_j_in, k_sl] = horiz_div + vert_div
        
        # Step Forward
        term_b = _b( (h[s_i_in, s_j_in] + etb[s_i_in, s_j_in]) * art[s_i_in, s_j_in] )
        term_f = _b( (h[s_i_in, s_j_in] + etf[s_i_in, s_j_in]) * art[s_i_in, s_j_in] )
        
        numer = term_b * fb[s_i_in, s_j_in, k_sl] - dti2 * ff[s_i_in, s_j_in, k_sl]
        
        ff[s_i_in, s_j_in, k_sl] = numer / (term_f + 1e-20)

        return fb, f, fclim, ff, xflux, yflux, zflux

    def smol_adif(self, xmassflux, ymassflux, zwflux, ff, sw, 
                  im, jm, kb, imm1, jmm1, kbm1, dti2, fsm, aru, arv, dzz, dt):
        """
        Calculates the antidiffusive velocity to reduce numerical diffusion 
        associated with the upstream differencing scheme (MPDATA).
        """
        
        value_min = 1.0e-9
        epsilon = 1.0e-14
        
        # Helper for 2D -> 3D broadcasting
        def _b(tensor_2d):
            return tensor_2d.unsqueeze(2)

        # =================================================================
        # 1. Apply Mask to Tracer Field (ff)
        # =================================================================
        # ff = ff * fsm
        ff = ff * _b(fsm)

        # =================================================================
        # 2. Recalculate X Mass Fluxes
        # =================================================================
        # Loop: k=1:kbm1, j=2:jmm1, i=2:im
        # Python: k -> 0:kbm1, j -> 1:jmm1, i -> 1:im
        
        s_i = slice(1, im)
        s_im1 = slice(0, im - 1)
        s_j = slice(1, jmm1)
        k_sl = slice(0, kbm1)
        
        # Condition 1: Check for small values
        mask_small = (ff[s_i, s_j, k_sl] < value_min) | (ff[s_im1, s_j, k_sl] < value_min)
        
        # Calculate terms
        # udx = abs(xmassflux)
        udx = torch.abs(xmassflux[s_i, s_j, k_sl])
        
        # u2dt = dti2 * flux^2 * 2.0 / (aru * (dt(i-1)+dt(i)))
        dt_sum = _b(dt[s_im1, s_j] + dt[s_i, s_j])
        aru_sl = _b(aru[s_i, s_j])
        
        u2dt = dti2 * (xmassflux[s_i, s_j, k_sl]**2) * 2.0 / (aru_sl * dt_sum + epsilon)
        
        # mol = (ff(i) - ff(i-1)) / (ff(i-1) + ff(i) + eps)
        ff_curr = ff[s_i, s_j, k_sl]
        ff_prev = ff[s_im1, s_j, k_sl]
        mol = (ff_curr - ff_prev) / (ff_prev + ff_curr + epsilon)
        
        # Calculate provisional flux
        # flux = (udx - u2dt) * mol * sw
        flux_new = (udx - u2dt) * mol * sw
        
        # Condition 2: Stability Check (abs(udx) < abs(u2dt))
        mask_unstable = torch.abs(udx) < torch.abs(u2dt)
        
        # Apply Masks (Set to 0.0 if small or unstable)
        # Note: logic in MATLAB is "if A ... else ... calc ... if B set 0".
        # This means if A is true, 0. If A is false, calc, then if B is true, 0.
        # Equivalent to: flux = 0 where (A or B).
        
        final_mask = mask_small | mask_unstable
        xmassflux[s_i, s_j, k_sl] = torch.where(final_mask, flux_new * 0.0, flux_new)

        # =================================================================
        # 3. Recalculate Y Mass Fluxes
        # =================================================================
        # Loop: k=1:kbm1, j=2:jm, i=2:imm1
        # Python: k -> 0:kbm1, j -> 1:jm, i -> 1:imm1
        
        s_i = slice(1, imm1)
        s_j = slice(1, jm)
        s_jm1 = slice(0, jm - 1)
        
        # Condition 1
        mask_small = (ff[s_i, s_j, k_sl] < value_min) | (ff[s_i, s_jm1, k_sl] < value_min)
        
        vdy = torch.abs(ymassflux[s_i, s_j, k_sl])
        
        # v2dt calculation
        dt_sum = _b(dt[s_i, s_jm1] + dt[s_i, s_j])
        arv_sl = _b(arv[s_i, s_j])
        
        v2dt = dti2 * (ymassflux[s_i, s_j, k_sl]**2) * 2.0 / (arv_sl * dt_sum + epsilon)
        
        # mol calculation
        ff_curr = ff[s_i, s_j, k_sl]
        ff_prev = ff[s_i, s_jm1, k_sl]
        mol = (ff_curr - ff_prev) / (ff_prev + ff_curr + epsilon)
        
        flux_new = (vdy - v2dt) * mol * sw
        
        # Condition 2
        mask_unstable = torch.abs(vdy) < torch.abs(v2dt)
        
        final_mask = mask_small | mask_unstable
        ymassflux[s_i, s_j, k_sl] = torch.where(final_mask, flux_new * 0.0, flux_new)

        # =================================================================
        # 4. Recalculate Z Fluxes (Vertical)
        # =================================================================
        # Loop: k=2:kbm1, j=2:jmm1, i=2:imm1
        # Python: k -> 1:kbm1 (indices 1 to kbm1-1)
        # Slices need to handle k (current) and k-1 (previous)
        
        k_sl = slice(1, kbm1)     # k
        k_m1 = slice(0, kbm1 - 1) # k-1
        
        s_i = slice(1, imm1)
        s_j = slice(1, jmm1)
        
        # Condition 1: ff(k) < min or ff(k-1) < min
        mask_small = (ff[s_i, s_j, k_sl] < value_min) | (ff[s_i, s_j, k_m1] < value_min)
        
        wdz = torch.abs(zwflux[s_i, s_j, k_sl])
        
        # w2dt = dti2 * flux^2 / (dzz(k-1) * dt(i,j))
        # dzz is 1D (kb). slice k-1 matches k_m1 indices.
        dzz_sl = dzz[k_m1].view(1, 1, -1)
        dt_sl = _b(dt[s_i, s_j])
        
        w2dt = dti2 * (zwflux[s_i, s_j, k_sl]**2) / (dzz_sl * dt_sl + epsilon)
        
        # mol = (ff(k-1) - ff(k)) / (ff(k) + ff(k-1))
        # Note order: k-1 minus k (Top minus Bottom, assuming k increases downwards)
        ff_curr = ff[s_i, s_j, k_sl]
        ff_prev = ff[s_i, s_j, k_m1]
        mol = (ff_prev - ff_curr) / (ff_curr + ff_prev + epsilon)
        
        flux_new = (wdz - w2dt) * mol * sw
        
        # Condition 2
        mask_unstable = torch.abs(wdz) < torch.abs(w2dt)
        
        final_mask = mask_small | mask_unstable
        zwflux[s_i, s_j, k_sl] = torch.where(final_mask, flux_new * 0.0, flux_new)

        return xmassflux, ymassflux, zwflux, ff, sw

    def advt2(self, fb, f, fclim, ff, xflux, yflux, nitera, sw, zflux,
              im, jm, kb, imm1, jmm1, kbm1, dti2,
              etb, etf, w, art, fsm, dt, aam, tprni, h, dum, dvm, dx, dy, u, v, aru, arv, dz, dzz):
        """
        Integrates conservative scalar equations using Smolarkiewicz iterative upstream scheme.
        """

        # Helper for 2D -> 3D broadcasting
        def _b(tensor_2d):
            return tensor_2d.unsqueeze(2)

        # 1. Initialize Mass Fluxes
        xmassflux = torch.zeros((im, jm, kb), device=self.device, dtype=self.t.dtype)
        ymassflux = torch.zeros((im, jm, kb), device=self.device, dtype=self.t.dtype)

        # Slices
        s_i = slice(1, im)
        s_im1 = slice(0, im - 1)
        s_j = slice(1, jm)
        s_jm1 = slice(0, jm - 1)
        k_sl = slice(0, kbm1)

        # 2. Calculate Initial Horizontal Mass Fluxes
        # X Mass Flux
        term_dy = _b(dy[s_im1, s_j] + dy[s_i, s_j])
        term_dt = _b(dt[s_im1, s_j] + dt[s_i, s_j])
        xmassflux[s_i, s_j, k_sl] = 0.25 * term_dy * term_dt * u[s_i, s_j, k_sl]

        # Y Mass Flux
        term_dx = _b(dx[s_i, s_jm1] + dx[s_i, s_j])
        term_dt = _b(dt[s_i, s_jm1] + dt[s_i, s_j])
        ymassflux[s_i, s_j, k_sl] = 0.25 * term_dx * term_dt * v[s_i, s_j, k_sl]

        # 3. Fill Bottom Boundary
        fb[:, :, kb-1] = fb[:, :, kbm1-1]

        # 4. Iteration Setup
        eta = etb.clone()
        zwflux = w.clone()
        fbmem = fb.clone()

        # =============================================================
        # Start Smolarkiewicz Scheme Loop
        # =============================================================
        for itera in range(1, nitera + 1):
            
            # --- Upwind Advection Scheme (Horizontal) ---
            # XFlux
            # .5 * ((Flux + |Flux|)*In + (Flux - |Flux|)*Out)
            flux = xmassflux[s_i, s_j, k_sl]
            abs_flux = torch.abs(flux)
            val_in = fbmem[s_im1, s_j, k_sl]
            val_out = fbmem[s_i, s_j, k_sl]
            
            xflux[s_i, s_j, k_sl] = 0.5 * ((flux + abs_flux) * val_in + 
                                           (flux - abs_flux) * val_out)
            
            # YFlux
            flux = ymassflux[s_i, s_j, k_sl]
            abs_flux = torch.abs(flux)
            val_in = fbmem[s_i, s_jm1, k_sl]
            val_out = fbmem[s_i, s_j, k_sl]
            
            yflux[s_i, s_j, k_sl] = 0.5 * ((flux + abs_flux) * val_in + 
                                           (flux - abs_flux) * val_out)

            # --- Upwind Advection Scheme (Vertical) ---
            # Surface Flux (k=0)
            zflux[:, :, 0] = 0.0
            if itera == 1:
                # MATLAB: zflux(i,j,1)=w(i,j,1)*f(i,j,1)*art(i,j)
                # Range j=2:jmm1, i=2:imm1 -> Python 1:jmm1, 1:imm1
                s_i_in = slice(1, imm1)
                s_j_in = slice(1, jmm1)
                zflux[s_i_in, s_j_in, 0] = w[s_i_in, s_j_in, 0] * f[s_i_in, s_j_in, 0] * _b(art[s_i_in, s_j_in]).squeeze(2)
            
            zflux[:, :, kb-1] = 0.0

            # Internal Fluxes (k=2:kbm1 -> Python 1:kbm1)
            # zflux(k) uses zwflux(k) and fbmem(k), fbmem(k-1)
            k_z = slice(1, kbm1)
            k_zm1 = slice(0, kbm1 - 1)
            
            flux = zwflux[s_i_in, s_j_in, k_z]
            abs_flux = torch.abs(flux)
            val_in = fbmem[s_i_in, s_j_in, k_z]   # k
            val_out = fbmem[s_i_in, s_j_in, k_zm1] # k-1
            
            z_val = 0.5 * ((flux + abs_flux) * val_in + (flux - abs_flux) * val_out)
            zflux[s_i_in, s_j_in, k_z] = z_val * _b(art[s_i_in, s_j_in])

            # --- Add Net Advective Fluxes & Step Forward ---
            # Range: j=2:jmm1, i=2:imm1 -> Python 1:jmm1, 1:imm1
            # ff = xflux(i+1) - xflux(i) + yflux(j+1) - yflux(j) + (zflux(k) - zflux(k+1))/dz
            
            s_ip1 = slice(2, imm1 + 1)
            s_jp1 = slice(2, jmm1 + 1)
            
            k_sl_full = slice(0, kbm1)
            k_p1_full = slice(1, kbm1 + 1)
            
            # Note: dz is 1D.
            dz_k = dz[k_sl_full].view(1, 1, -1)
            
            horiz = xflux[s_ip1, s_j_in, k_sl_full] - xflux[s_i_in, s_j_in, k_sl_full] + \
                    yflux[s_i_in, s_jp1, k_sl_full] - yflux[s_i_in, s_j_in, k_sl_full]
            
            vert = (zflux[s_i_in, s_j_in, k_sl_full] - zflux[s_i_in, s_j_in, k_p1_full]) / dz_k
            
            ff[s_i_in, s_j_in, k_sl_full] = horiz + vert
            
            # Time Step Update
            # ff = (fbmem*(h+eta)*art - dti2*ff) / ((h+etf)*art)
            term_old = _b( (h[s_i_in, s_j_in] + eta[s_i_in, s_j_in]) * art[s_i_in, s_j_in] )
            term_new = _b( (h[s_i_in, s_j_in] + etf[s_i_in, s_j_in]) * art[s_i_in, s_j_in] )
            
            numer = term_old * fbmem[s_i_in, s_j_in, k_sl_full] - dti2 * ff[s_i_in, s_j_in, k_sl_full]
            ff[s_i_in, s_j_in, k_sl_full] = numer / (term_new + 1e-20)

            # --- Calculate Antidiffusion Velocity (Smol_adif) ---
            # Passing necessary variables to the stub/function
            xmassflux, ymassflux, zwflux, ff, sw = \
                self.smol_adif(xmassflux, ymassflux, zwflux, ff, sw,
                               im, jm, kb, imm1, jmm1, kbm1, dti2, fsm, aru, arv, dzz, dt)

            # Update loop variables
            eta = etf.clone()
            fbmem = ff.clone()
        
        # =============================================================
        # End Iteration Loop
        # =============================================================

        # 5. Add Horizontal Diffusive Fluxes
        
        # Calculate Anomaly
        fb_anom = fb - fclim
        
        # Calculate Diffusive Mass Fluxes (borrowing xmassflux variable to save memory, per POM)
        # xmassflux = 0.5 * (aam(i) + aam(i-1))
        xmassflux[s_i, s_j, k_sl] = 0.5 * (aam[s_i, s_j, k_sl] + aam[s_im1, s_j, k_sl])
        ymassflux[s_i, s_j, k_sl] = 0.5 * (aam[s_i, s_j, k_sl] + aam[s_i, s_jm1, k_sl])
        
        # Calculate Diffusive Fluxes
        # XFlux
        h_sum = _b(h[s_i, s_j] + h[s_im1, s_j])
        fb_diff = fb_anom[s_i, s_j, k_sl] - fb_anom[s_im1, s_j, k_sl]
        dy_sum = _b(dy[s_i, s_j] + dy[s_im1, s_j])
        dx_sum = _b(dx[s_i, s_j] + dx[s_im1, s_j])
        dum_b = _b(dum[s_i, s_j])
        
        # -xmassflux * h_sum * tprni * fb_diff * dum * dy_sum * 0.5 / dx_sum
        xflux[s_i, s_j, k_sl] = -xmassflux[s_i, s_j, k_sl] * h_sum * tprni * \
                                fb_diff * dum_b * dy_sum * 0.5 / dx_sum
        
        # YFlux
        h_sum = _b(h[s_i, s_j] + h[s_i, s_jm1])
        fb_diff = fb_anom[s_i, s_j, k_sl] - fb_anom[s_i, s_jm1, k_sl]
        dx_sum = _b(dx[s_i, s_j] + dx[s_i, s_jm1])
        dy_sum = _b(dy[s_i, s_j] + dy[s_i, s_jm1])
        dvm_b = _b(dvm[s_i, s_j])
        
        yflux[s_i, s_j, k_sl] = -ymassflux[s_i, s_j, k_sl] * h_sum * tprni * \
                                fb_diff * dvm_b * dx_sum * 0.5 / dy_sum

        # Restore FB (Just implies we are done with anomaly)
        # fb = fb + fclim (Not strictly needed as we used fb_anom temp variable)
        
        # 6. Step Forward with Diffusion
        # ff = ff - dti2 * (divergence) / vol
        
        term_new = _b((h[s_i_in, s_j_in] + etf[s_i_in, s_j_in]) * art[s_i_in, s_j_in])
        
        horiz_div = xflux[s_ip1, s_j_in, k_sl_full] - xflux[s_i_in, s_j_in, k_sl_full] + \
                    yflux[s_i_in, s_jp1, k_sl_full] - yflux[s_i_in, s_j_in, k_sl_full]
                    
        ff[s_i_in, s_j_in, k_sl_full] -= dti2 * horiz_div / (term_new + 1e-20)

        return fb, f, fclim, ff, xflux, yflux, nitera, sw, zflux

    def proft(self, f, wfsurf, fsurf, nbc, dh, 
              a, c, ee, gg, 
              h, etf, dti2, dz, dzz, swrad, ntp, im, jm, kb, kbm1, kbm2, kh, umol):
        """
        Solves for vertical diffusion of temperature and salinity (Implicit Scheme).
        [FIXED FINAL]: Breaks autograd dependency chain using temporary variables.
        """

        # Helper for 2D -> 3D broadcasting
        def _b(tensor_2d): return tensor_2d.unsqueeze(2)

        # =========================================================
        # Separate inputs/outputs and keep scratch variables local.
        # =========================================================
        f_in = f             # Input (Read-only)
        f = f.clone()        # Output (Write)
        
        # Local scratchpads
        a = torch.zeros_like(f)
        c = torch.zeros_like(f)
        ee = torch.zeros_like(f)
        gg_store = torch.zeros_like(f) # Replaces 'gg' for storage
        # =========================================================

        # 1. Irradiance Parameters
        R   = [0.58, 0.62, 0.67, 0.77, 0.78]
        AD1 = [0.35, 0.60, 1.00, 1.50, 1.40]
        AD2 = [0.23, 20.0, 17.0, 14.0, 7.90]
        
        ntp_idx = int(ntp) - 1
        r_val   = R[ntp_idx]; ad1_val = AD1[ntp_idx]; ad2_val = AD2[ntp_idx]

        # 2. Coefficients
        dh = h + etf; dh_3d = _b(dh)
        rad = torch.zeros((im, jm, kb), device=self.device, dtype=self.t.dtype)

        for k in range(1, kbm1):
            dh2 = dh_3d[:, :, 0] ** 2
            num = -dti2 * (kh[:, :, k] + umol)
            den_a = dz[k-1] * dzz[k-1] * dh2
            a[:, :, k-1] = num / den_a
            den_c = dz[k] * dzz[k-1] * dh2
            c[:, :, k] = num / den_c

        # 3. Radiation
        if nbc == 2 or nbc == 4:
            for k in range(kbm1):
                z_term = self.z[k] * dh
                term1 = r_val * torch.exp(z_term / ad1_val)
                term2 = (1.0 - r_val) * torch.exp(z_term / ad2_val)
                rad[:, :, k] = swrad * (term1 + term2)

        # 4. Surface BC (Forward Sweep Start)
        gg_prev = None # Temp variable

        if nbc == 1:
            denom = a[:, :, 0] - 1.0
            ee[:, :, 0] = a[:, :, 0] / denom
            term_flux = -dti2 * wfsurf / (-dz[0] * dh)
            # 🔥 FIX: Calc to temp var
            gg_prev = (term_flux - f_in[:, :, 0]) / denom
            
        elif nbc == 2:
            denom = a[:, :, 0] - 1.0
            ee[:, :, 0] = a[:, :, 0] / denom
            rad_diff = rad[:, :, 0] - rad[:, :, 1]
            term_flux = dti2 * (wfsurf + rad_diff) / (dz[0] * dh)
            # 🔥 FIX: Calc to temp var
            gg_prev = (term_flux - f_in[:, :, 0]) / denom
            
        elif nbc == 3 or nbc == 4:
            ee[:, :, 0] = 0.0
            # 🔥 FIX: Calc to temp var
            gg_prev = fsurf

        # Store first value
        gg_store[:, :, 0] = gg_prev

        # 5. Forward Sweep (Middle Layers)
        for k in range(1, kbm2):
            denom = a[:, :, k] + c[:, :, k] * (1.0 - ee[:, :, k-1]) - 1.0
            inv_denom = 1.0 / (denom + 1e-20)
            ee[:, :, k] = a[:, :, k] * inv_denom
            
            rad_term = dti2 * (rad[:, :, k] - rad[:, :, k+1]) / (dh * dz[k])
            
            # 🔥 FIX: Use gg_prev (break dependency on tensor slice)
            source = c[:, :, k] * gg_prev - f_in[:, :, k] + rad_term
            gg_curr = source * inv_denom
            
            # Update store & prev
            gg_store[:, :, k] = gg_curr
            gg_prev = gg_curr

        # 6. Bottom BC
        k_bot = kbm1 - 1
        k_prev = k_bot - 1
        
        rad_term = dti2 * (rad[:, :, k_bot] - rad[:, :, k_bot+1]) / (dh * dz[k_bot])
        
        # Read from gg_store (Safe now)
        numer = c[:, :, k_bot] * gg_store[:, :, k_prev] - f_in[:, :, k_bot] + rad_term
        denom = c[:, :, k_bot] * (1.0 - ee[:, :, k_prev]) - 1.0
        
        val_bot = numer / (denom + 1e-20)
        
        # Write to output
        f[:, :, k_bot] = val_bot
        
        # 🔥 FIX: Carrier for backward loop
        f_next = val_bot

        # 7. Backward Substitution
        for k in range(kbm1 - 2, -1, -1):
            # 🔥 FIX: Use f_next
            val = ee[:, :, k] * f_next + gg_store[:, :, k]
            
            f[:, :, k] = val
            f_next = val

        return f, wfsurf, fsurf, nbc, dh, a, c, ee, gg_store
