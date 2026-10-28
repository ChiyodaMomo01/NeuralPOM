import math
import numpy as np
import netCDF4 as nc
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float32)


def smagorinsky_strain_magnitude(du_dx, dv_dy, shear_sum):
    """Return the V2-Align Smagorinsky strain norm with a finite zero gradient."""
    components = torch.stack(
        (du_dx, dv_dy, shear_sum * math.sqrt(0.5)),
        dim=0,
    )
    return torch.linalg.vector_norm(components, dim=0)


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
            
            # Wind Stress & Surface Fluxes
            if self.iproblem == 4:
                # === Double Gyre Wind Stress (Paper Eq. 5) ===
                # tau_x = tau_0 * [1 - cos(2*pi * (lat - 30) / 20)]
                # tau_y = 0
                
                tau0 = 0.1 # N/m^2
                # Kinematic stress = tau / rho0
                kinematic_factor = tau0 / self.rhoref 
                
                # Latitude grid (using north_e as approximation for u-points center)
                lat_grid = self.north_e
                
                # Calculate the cosine argument: 2*pi * (lat - 30) / 20
                # Lat range is 30 to 50, so (lat-30)/20 goes from 0 to 1
                arg = 2.0 * self.pi * (lat_grid - 30.0) / 20.0
                
                # Calculate wind stress pattern (Time invariant part)
                wind_stress = kinematic_factor * (1.0 - torch.cos(arg))
                
                # # Apply Ramp and Mask
                # self.wusurf[s_i, s_j] = wind_stress[s_i, s_j] * self.ramp * self.dum[s_i, s_j]
                # self.wvsurf[s_i, s_j] = 0.0
            
            if not hasattr(self, "_inner_mask") or (self._inner_mask.shape != self.e_atmos.shape):
                inner_mask = torch.zeros_like(self.e_atmos, dtype=torch.bool)
                inner_mask[s_i, s_j] = True
                self._inner_mask = inner_mask

            # Common surface settings
            m = self._inner_mask
            if generation_fast_path:
                self.e_atmos.masked_fill_(m, 0.0)
                self.vfluxf.masked_fill_(m, 0.0)
                self.swrad.masked_fill_(m, 0.0)
            else:
                self.e_atmos = self.e_atmos.masked_fill(m, 0.0)
                self.vfluxf = self.vfluxf.masked_fill(m, 0.0)
                self.swrad = self.swrad.masked_fill(m, 0.0)
            
            # w(i,j,1) = vfluxf(i,j) -> w[:,:,0] in Python
            if generation_fast_path:
                self.w[s_i, s_j, 0].copy_(self.vfluxf[s_i, s_j])
            else:
                w_new = self.w.clone()
                w_new[s_i, s_j, 0] = self.vfluxf[s_i, s_j]
                self.w = w_new
            
            # self.wtsurf[s_i, s_j] = 0.0
            
            # Heat flux approximation
            t_surf = self.t[s_i, s_j, 0]
            tatm = t_surf + self.tbias
            delta_wts = self.vfluxf[s_i, s_j] * (tatm - t_surf - self.tbias)
            if generation_fast_path:
                self.wtsurf[s_i, s_j].add_(delta_wts)
            else:
                wtsurf_new = self.wtsurf.clone()
                wtsurf_new[s_i, s_j] = wtsurf_new[s_i, s_j] + delta_wts
                self.wtsurf = wtsurf_new
            
            # Salinity flux
            satm = 0.0
            s_surf = self.s[s_i, s_j, 0]
            wssurf_inner = self.vfluxf[s_i, s_j] * (satm - s_surf - self.sbias)
            if generation_fast_path:
                self.wssurf[s_i, s_j].copy_(wssurf_inner)
            else:
                wssurf_new = self.wssurf.clone()
                wssurf_new[s_i, s_j] = wssurf_inner
                self.wssurf = wssurf_new

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
                
                du_dx = (u_ip1 - u_curr) / dx_sl
                dv_dy = (v_jp1 - v_curr) / dy_sl
                
                u_term = 0.25 * (
                    self.u[s_i, s_j_p1, :self.kbm1] + self.u[s_i_p1, s_j_p1, :self.kbm1] -
                    self.u[s_i, s_j_m1, :self.kbm1] - self.u[s_i_p1, s_j_m1, :self.kbm1]
                ) / dy_sl
                
                v_term = 0.25 * (
                    self.v[s_i_p1, s_j, :self.kbm1] + self.v[s_i_p1, s_j_p1, :self.kbm1] -
                    self.v[s_i_m1, s_j, :self.kbm1] - self.v[s_i_m1, s_j_p1, :self.kbm1]
                ) / dx_sl
                
                shear_sum = u_term + v_term
                
                # =========================================================
                # Combine Smagorinsky viscosity with background viscosity (MITgcm style).
                # =========================================================
                
                # 1. Compute the pure Smagorinsky contribution.
                strain_magnitude = smagorinsky_strain_magnitude(du_dx, dv_dy, shear_sum)
                smag_visc = self.horcon * dx_sl * dy_sl * strain_magnitude

                if generation_fast_path:
                    self.aam[s_i, s_j, :self.kbm1].copy_(smag_visc)
                else:
                    aam_new = self.aam.clone()

                    # 2. Apply the sliced updates on a copy.
                    aam_new[s_i, s_j, :self.kbm1] = smag_visc

                    # 3. Rebind self.aam to the updated copy.
                    self.aam = aam_new
                
                # =========================================================

                # =========================================================

                # Vertical averages of 3-D fields
                dz_w = self.dz[:self.kbm1].view(1, 1, -1)
                
                adx2d = torch.sum(self.advx[:,:,:self.kbm1] * dz_w, dim=2)
                ady2d = torch.sum(self.advy[:,:,:self.kbm1] * dz_w, dim=2)
                drx2d = torch.sum(self.drhox[:,:,:self.kbm1] * dz_w, dim=2)
                dry2d = torch.sum(self.drhoy[:,:,:self.kbm1] * dz_w, dim=2)
                aam2d = torch.sum(self.aam[:,:,:self.kbm1] * dz_w, dim=2)
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
            
            d_term_u = self.d[1:self.im, :] + self.d[0:self.imm1, :]
            self.utf[1:self.im, :] = self.ua[1:self.im, :] * d_term_u * self.isp2i
            
            d_term_v = self.d[:, 1:self.jm] + self.d[:, 0:self.jmm1]
            self.vtf[:, 1:self.jm] = self.va[:, 1:self.jm] * d_term_v * self.isp2i

            # -----------------------------------------------------
            # External (2-D) Mode Loop
            # -----------------------------------------------------
            vamax = 0.0
            
            for iext in range(1, self.isplit + 1):
                # Calculate Fluxes
                s_i_full = slice(1, self.im)
                s_j_full = slice(1, self.jm)
                s_i_full_m1 = slice(0, self.imm1)
                s_j_full_m1 = slice(0, self.jmm1)
                
                d_avg_u = 0.25 * (self.d[s_i_full, s_j_full] + self.d[s_i_full_m1, s_j_full])
                dy_avg_u = (self.dy[s_i_full, s_j_full] + self.dy[s_i_full_m1, s_j_full])
                self.fluxua[s_i_full, s_j_full] = d_avg_u * dy_avg_u * self.ua[s_i_full, s_j_full]
                
                d_avg_v = 0.25 * (self.d[s_i_full, s_j_full] + self.d[s_i_full, s_j_full_m1])
                dx_avg_v = (self.dx[s_i_full, s_j_full] + self.dx[s_i_full, s_j_full_m1])
                self.fluxva[s_i_full, s_j_full] = d_avg_v * dx_avg_v * self.va[s_i_full, s_j_full]

                # Update Elevation (elf)
                flux_div = (self.fluxua[s_i_p1, s_j] - self.fluxua[s_i, s_j] +
                            self.fluxva[s_i, s_j_p1] - self.fluxva[s_i, s_j])
                
                term = -(flux_div / self.art[s_i, s_j]) - self.vfluxf[s_i, s_j]
                self.elf[s_i, s_j] = self.elb[s_i, s_j] + self.dte2 * term

                # Boundary Conds (Step 1)
                self.elf, self.uaf, self.vaf, self.uf, self.vf, self.w = \
                    self.bcond(1, self.elf, self.uaf, self.vaf, self.uf, self.vf, self.w)

                if (iext % self.ispadv) == 0:
                    self.psi, self.advua, self.advva, self.fluxua, self.fluxva, self.wubot, self.wvbot, self.tps = \
                        self.advave(self.psi, self.advua, self.advva, self.fluxua, self.fluxva, 
                                    self.wubot, self.wvbot, self.tps, 
                                    self.mode, self.aam2d, 
                                    self.uab, self.vab, self.dx, self.dy, self.ua, self.va, 
                                    self.cbc, self.aru, self.arv, self.d)

                # Update uaf
                cor_term = 0.25 * (
                    self.cor[s_i, s_j] * self.d[s_i, s_j] * (self.va[s_i, s_j_p1] + self.va[s_i, s_j]) +
                    self.cor[s_i_m1, s_j] * self.d[s_i_m1, s_j] * (self.va[s_i_m1, s_j_p1] + self.va[s_i_m1, s_j])
                )
                
                grad_term = 0.25 * self.grav * (self.dy[s_i, s_j] + self.dy[s_i_m1, s_j]) * \
                            (self.d[s_i, s_j] + self.d[s_i_m1, s_j]) * (
                                (1.0 - 2.0 * self.alpha) * (self.el[s_i, s_j] - self.el[s_i_m1, s_j]) +
                                self.alpha * (self.elb[s_i, s_j] - self.elb[s_i_m1, s_j] + 
                                              self.elf[s_i, s_j] - self.elf[s_i_m1, s_j]) +
                                self.e_atmos[s_i, s_j] - self.e_atmos[s_i_m1, s_j]
                            )
                
                ######################################## CRTICAL FIX ########################################
                # -self.aru * (self.wusurf + self.wubot) ---> -self.aru * (-self.wusurf + self.wubot)
                #############################################################################################
                uaf_prov = self.adx2d[s_i, s_j] + self.advua[s_i, s_j] - \
                           self.aru[s_i, s_j] * cor_term + grad_term + \
                           self.drx2d[s_i, s_j] - self.aru[s_i, s_j] * (- self.wusurf[s_i, s_j] + self.wubot[s_i, s_j])
                
                self.uaf[s_i, s_j] = uaf_prov

                denom = (self.h[s_i, s_j] + self.elf[s_i, s_j] + self.h[s_i_m1, s_j] + self.elf[s_i_m1, s_j]) * self.aru[s_i, s_j]
                numer = (self.h[s_i, s_j] + self.elb[s_i, s_j] + self.h[s_i_m1, s_j] + self.elb[s_i_m1, s_j]) * self.aru[s_i, s_j] * self.uab[s_i, s_j] - \
                        4.0 * self.dte * self.uaf[s_i, s_j]
                
                self.uaf[s_i, s_j] = numer / (denom + 1e-20)

                # Update vaf
                cor_term_v = 0.25 * (
                    self.cor[s_i, s_j] * self.d[s_i, s_j] * (self.ua[s_i_p1, s_j] + self.ua[s_i, s_j]) +
                    self.cor[s_i, s_j_m1] * self.d[s_i, s_j_m1] * (self.ua[s_i_p1, s_j_m1] + self.ua[s_i, s_j_m1])
                )
                
                grad_term_v = 0.25 * self.grav * (self.dx[s_i, s_j] + self.dx[s_i, s_j_m1]) * \
                              (self.d[s_i, s_j] + self.d[s_i, s_j_m1]) * (
                                  (1.0 - 2.0 * self.alpha) * (self.el[s_i, s_j] - self.el[s_i, s_j_m1]) +
                                  self.alpha * (self.elb[s_i, s_j] - self.elb[s_i, s_j_m1] + 
                                                self.elf[s_i, s_j] - self.elf[s_i, s_j_m1]) +
                                  self.e_atmos[s_i, s_j] - self.e_atmos[s_i, s_j_m1]
                              )

                vaf_prov = self.ady2d[s_i, s_j] + self.advva[s_i, s_j] + \
                           self.arv[s_i, s_j] * cor_term_v + grad_term_v + \
                           self.dry2d[s_i, s_j] - self.arv[s_i, s_j] * (- self.wvsurf[s_i, s_j] + self.wvbot[s_i, s_j])
                
                self.vaf[s_i, s_j] = vaf_prov

                denom_v = (self.h[s_i, s_j] + self.elf[s_i, s_j] + self.h[s_i, s_j_m1] + self.elf[s_i, s_j_m1]) * self.arv[s_i, s_j]
                numer_v = (self.h[s_i, s_j] + self.elb[s_i, s_j] + self.h[s_i, s_j_m1] + self.elb[s_i, s_j_m1]) * self.arv[s_i, s_j] * self.vab[s_i, s_j] - \
                          4.0 * self.dte * self.vaf[s_i, s_j]
                
                self.vaf[s_i, s_j] = numer_v / (denom_v + 1e-20)

                # Boundary Conds (Step 2)
                self.elf, self.uaf, self.vaf, self.uf, self.vf, self.w = \
                    self.bcond(2, self.elf, self.uaf, self.vaf, self.uf, self.vf, self.w)
                
                # Time filtering
                if iext == (self.isplit - 2):
                    self.etf[:] = 0.25 * self.smoth * self.elf[:]
                elif iext == (self.isplit - 1):
                    self.etf[:] += 0.5 * (1.0 - 0.5 * self.smoth) * self.elf[:]
                elif iext == self.isplit:
                    self.etf[:] = (self.etf[:] + 0.5 * self.elf[:]) * self.fsm[:]

                vamax = torch.max(torch.abs(self.vaf))

                if vamax_safe_fast_path or vamax <= self.vmaxl:
                    # Robert-Asselin Time Filter
                    if generation_fast_path:
                        self.ua.add_(0.5 * self.smoth * (self.uab - 2.0 * self.ua + self.uaf))
                        self.va.add_(0.5 * self.smoth * (self.vab - 2.0 * self.va + self.vaf))
                        self.el.add_(0.5 * self.smoth * (self.elb - 2.0 * self.el + self.elf))
                        self.elb.copy_(self.el)
                        self.el.copy_(self.elf)
                        self.d.copy_(self.h + self.el)
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
                        self.d = self.h + self.el

                        self.uab = self.ua.clone()
                        self.ua = self.uaf.clone()

                        self.vab = self.va.clone()
                        self.va = self.vaf.clone()
                    
                    if iext != self.isplit:
                        if generation_fast_path:
                            self.egf.add_(self.el * self.ispi)
                        else:
                            self.egf = self.egf + self.el * self.ispi
                        d_sum_u = self.d[1:self.im, :] + self.d[0:self.imm1, :]
                        self.utf[1:self.im, :] = self.utf[1:self.im, :] + self.ua[1:self.im, :] * d_sum_u * self.isp2i
                        d_sum_v = self.d[:, 1:self.jm] + self.d[:, 0:self.jmm1]
                        self.vtf[:, 1:self.jm] = self.vtf[:, 1:self.jm] + self.va[:, 1:self.jm] * d_sum_v * self.isp2i

            # -----------------------------------------------------
            # End of external mode
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
                    
                    u_inner = (u_slice - tps_slice) + corr_slice
                    if generation_fast_path:
                        self.u[1:self.im, :, :self.kbm1].copy_(u_inner)
                    else:
                        # Clone u to avoid in-place modification of the graph leaf/node.
                        u_new = self.u.clone()
                        u_new[1:self.im, :, :self.kbm1] = u_inner
                        self.u = u_new  # Update reference to the new tensor

                    # --------------------- Fix V ---------------------
                    tps_v = torch.sum(self.v[:, :, :self.kbm1] * dz_w, dim=2)
                    dt_sum_v = self.dt[:, 1:self.jm] + self.dt[:, 0:self.jmm1]
                    correction_v = (self.vtb[:, 1:self.jm] + self.vtf[:, 1:self.jm]) / (dt_sum_v + 1e-20)
                    
                    v_slice = self.v[:, 1:self.jm, :self.kbm1]
                    tps_v_slice = tps_v[:, 1:self.jm].unsqueeze(2)
                    corr_v_slice = correction_v.unsqueeze(2)

                    v_inner = (v_slice - tps_v_slice) + corr_v_slice
                    if generation_fast_path:
                        self.v[:, 1:self.jm, :self.kbm1].copy_(v_inner)
                    else:
                        # Clone v to avoid in-place modification.
                        v_new = self.v.clone()
                        v_new[:, 1:self.jm, :self.kbm1] = v_inner
                        self.v = v_new  # Update reference to the new tensor

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

                    # T/S
                    if self.mode != 4:
                        if self.nadv == 1:
                            self.tb, self.t, self.tclim, self.uf, self.a, self.c, self.zflux = self.advt1(
                                self.tb, self.t, self.tclim, self.uf, self.a, self.c, self.zflux,
                                self.dt, self.u, self.v, self.aam, self.tprni, self.dum, self.dvm, self.w, 
                                self.art, self.etb, self.etf, self.dti2, self.dx, self.dy, self.dz, self.h, 
                                self.im, self.jm, self.kb, self.imm1, self.jmm1, self.kbm1
                            )
                            self.sb, self.s, self.sclim, self.vf, self.a, self.c, self.zflux = self.advt1(
                                self.sb, self.s, self.sclim, self.vf, self.a, self.c, self.zflux,
                                self.dt, self.u, self.v, self.aam, self.tprni, self.dum, self.dvm, self.w, 
                                self.art, self.etb, self.etf, self.dti2, self.dx, self.dy, self.dz, self.h, 
                                self.im, self.jm, self.kb, self.imm1, self.jmm1, self.kbm1
                            )
                        
                        elif self.nadv == 2:
                            self.tb, self.t, self.tclim, self.uf, self.a, self.c, self.nitera, self.sw, self.zflux = \
                                self.advt2(self.tb, self.t, self.tclim, self.uf, self.a, self.c, self.nitera, self.sw, self.zflux,
                                           self.im, self.jm, self.kb, self.imm1, self.jmm1, self.kbm1, self.dti2,
                                           self.etb, self.etf, self.w, self.art, self.fsm, self.dt, self.aam, self.tprni, 
                                           self.h, self.dum, self.dvm, self.dx, self.dy, self.u, self.v, self.aru, self.arv, 
                                           self.dz, self.dzz)
                            
                            self.sb, self.s, self.sclim, self.vf, self.a, self.c, self.nitera, self.sw, self.zflux = \
                                self.advt2(self.sb, self.s, self.sclim, self.vf, self.a, self.c, self.nitera, self.sw, self.zflux,
                                           self.im, self.jm, self.kb, self.imm1, self.jmm1, self.kbm1, self.dti2,
                                           self.etb, self.etf, self.w, self.art, self.fsm, self.dt, self.aam, self.tprni, 
                                           self.h, self.dum, self.dvm, self.dx, self.dy, self.u, self.v, self.aru, self.arv, 
                                           self.dz, self.dzz)

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

                        if generation_fast_path:
                            self.t.add_(0.5 * self.smoth * (self.uf + self.tb - 2.0 * self.t))
                            self.s.add_(0.5 * self.smoth * (self.vf + self.sb - 2.0 * self.s))
                            self.tb.copy_(self.t)
                            self.t.copy_(self.uf)
                            self.sb.copy_(self.s)
                            self.s.copy_(self.vf)
                        else:
                            self.t = self.t + 0.5 * self.smoth * (self.uf + self.tb - 2.0 * self.t)
                            self.s = self.s + 0.5 * self.smoth * (self.vf + self.sb - 2.0 * self.s)

                            self.tb = self.t.clone()
                            self.t = self.uf.clone()
                            self.sb = self.s.clone()
                            self.s = self.vf.clone()

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
                    
                    u_filtered_inner = self.u[:,:,:self.kbm1] + 0.5 * self.smoth * (term_u - tps_unsqueezed)
                    if generation_fast_path:
                        u_filtered = self.ub
                        u_filtered[:,:,:self.kbm1].copy_(u_filtered_inner)
                        u_filtered[:,:,self.kbm1:].copy_(self.u[:,:,self.kbm1:])
                    else:
                        # Update a fresh copy instead of modifying self.u in place.
                        u_filtered = self.u.clone()
                        u_filtered[:,:,:self.kbm1] = u_filtered_inner

                    # --- V Filter ---
                    term_v = self.vf[:,:,:self.kbm1] + self.vb[:,:,:self.kbm1] - 2.0 * self.v[:,:,:self.kbm1]
                    
                    tps_v = torch.sum(term_v * dz_w, dim=2)
                    tps_unsqueezed_v = tps_v.unsqueeze(2)
                    
                    v_filtered_inner = self.v[:,:,:self.kbm1] + 0.5 * self.smoth * (term_v - tps_unsqueezed_v)
                    if generation_fast_path:
                        v_filtered = self.vb
                        v_filtered[:,:,:self.kbm1].copy_(v_filtered_inner)
                        v_filtered[:,:,self.kbm1:].copy_(self.v[:,:,self.kbm1:])
                    else:
                        # Update a fresh copy instead of modifying self.v in place.
                        v_filtered = self.v.clone()
                        v_filtered[:,:,:self.kbm1] = v_filtered_inner

                    # --- Shift Variables ---
                    # After Robert-Asselin filtering, the filtered current value becomes the next-step old value (ub).
                    if generation_fast_path:
                        self.u.copy_(self.uf)
                        self.v.copy_(self.vf)
                    else:
                        self.ub = u_filtered
                        self.u = self.uf.clone()

                        self.vb = v_filtered
                        self.v = self.vf.clone()

            # Shifts
            if generation_fast_path:
                self.egb.copy_(self.egf)
                self.etb.copy_(self.et)
                self.et.copy_(self.etf)
                self.dt.copy_(self.h + self.et)
                self.utb.copy_(self.utf)
                self.vtb.copy_(self.vtf)
                self.vfluxb.copy_(self.vfluxf)
            else:
                self.egb = self.egf.clone()
                self.etb = self.et.clone()
                self.et = self.etf.clone()
                self.dt = self.h + self.et
                self.utb = self.utf.clone()
                self.vtb = self.vtf.clone()
                self.vfluxb = self.vfluxf.clone()

            # Print
            if self.iint >= self.iswtch:
                self.iprint = int(self.prtd2 * 24.0 * 3600.0 / self.dti + 0.5)
            
            if (not generation_fast_path) and ((self.iint % self.iprint == 0) or (vamax >= self.vmaxl)):
                self.printall(self.im, self.jm, self.imm1, self.jmm1, self.iskp, self.jskp, 
                              self.uab, self.vab, self.elb, self.d, self.dx, self.dy, 
                              self.time, self.u, self.v, self.w, self.t, self.s, self.rho, 
                              self.aam, self.km, self.kb, self.mode, self.dt, self.zz, self.z)
                if vamax > self.vmaxl:
                    print("Abnormal end")
                    return
