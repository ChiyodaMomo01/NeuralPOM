import math
import os
import torch


torch.set_default_dtype(torch.float32)


class CasesMixin:
    def _vertical_interp(self, z_src, val_src, z_dst):
        """Interpolate a descending 1D depth profile onto a 3D depth grid."""
        z_src_flip = torch.flip(z_src, dims=[0])
        val_src_flip = torch.flip(val_src, dims=[0])
        idxs = torch.bucketize(z_dst, z_src_flip)
        idxs = torch.clamp(idxs, min=1, max=z_src.shape[0] - 1)

        z0 = z_src_flip[idxs - 1]
        z1 = z_src_flip[idxs]
        v0 = val_src_flip[idxs - 1]
        v1 = val_src_flip[idxs]
        denom = z1 - z0
        denom = torch.where(
            torch.abs(denom) < 1e-6,
            torch.tensor(1.0, device=self.device, dtype=torch.float32),
            denom,
        )
        weight = (z_dst - z0) / denom
        return v0 * (1.0 - weight) + v1 * weight

    def double_gyre(self):
        """
        Sets up the Wind-Driven Double Gyre configuration with 20 Vertical Layers.
        Includes explicit interpolation from Z-levels to Sigma-levels.
        """
        # print(f"Initializing Double Gyre (High-Res 1/16 deg) with {self.kb} layers...")
        # print("Applying Z-to-Sigma interpolation for stratification...")
        
        # 1. Constants & Grid Setup
        deg2rad = self.pi / 180.0
        re = 6371.0e3
        omega = 7.2921e-5
        
        lat_min, lat_max = 30.0, 50.0
        lon_min, lon_max = 0.0, 22.0
        
        # Grid Spacing
        dlat = (lat_max - lat_min) / float(self.jmm1 - 1)
        dlon = (lon_max - lon_min) / float(self.imm1 - 1)
        
        # Create Grid Indices
        j_idx = torch.arange(self.jm, device=self.device, dtype=torch.float32).view(1, -1).repeat(self.im, 1)
        i_idx = torch.arange(self.im, device=self.device, dtype=torch.float32).view(-1, 1).repeat(1, self.jm)
        
        self.north_e = lat_min + (j_idx - 1.0) * dlat
        self.east_e  = lon_min + (i_idx - 1.0) * dlon
        
        # Metric terms
        self.dy = torch.full_like(self.dy, re * dlat * deg2rad)
        self.dx = re * torch.cos(self.north_e * deg2rad) * dlon * deg2rad
        self.cor = 2.0 * omega * torch.sin(self.north_e * deg2rad)
        
        # 2. Flat-bottom Double-gyre topography
        self.h.fill_(2500.0)

        self.h = torch.clamp(self.h, min=50.0)
        self.h[0, :] = 1.0; self.h[self.im-1, :] = 1.0
        self.h[:, 0] = 1.0; self.h[:, self.jm-1] = 1.0
        
        # Get Max Depth for Reference Profile definition
        h_max_real = torch.max(self.h).item()

        # ==========================================================
        # 3. Vertical Coordinate (Auto-generate for N Layers)
        # ==========================================================
        # Uniform Sigma Distribution (0 to -1)
        for k in range(self.kb):
            self.z[k] = -float(k) / float(self.kb - 1)

        # Calculate Mid-points (zz) and Thickness (dz, dzz)
        for k in range(self.kb - 1):
            self.dz[k] = self.z[k] - self.z[k+1]
            self.zz[k] = 0.5 * (self.z[k] + self.z[k+1])
            self.dzz[k] = self.dz[k] # Approx

        self.dz[self.kb-1] = 0.0
        self.dzz[self.kb-1] = 0.0
        self.zz[self.kb-1] = 2.0 * self.zz[self.kb-2] - self.zz[self.kb-3]

        # 4. Masks & Velocity Initialization
        self.areas_masks()
        self.ub.zero_()
        self.vb.zero_()
        self.uab.zero_()
        self.vab.zero_()
        self.elb.zero_()

        # ==========================================================
        # 5. Stratification: Interpolation Method
        # ==========================================================
        self.sb.fill_(35.0)
        
        # A. Define the 1D Reference Vertical Profile (Horizontally Uniform)
        # ------------------------------------------------------------------
        t_surf = 30.0 # Surface Temperature
        t_bot  = 2.0  # Bottom Temperature
        h_scale = 800.0 # e-folding scale
        
        # Create a high-res Z-level grid for the reference profile
        # From 0 down to slightly deeper than the deepest hole to ensure coverage
        z_ref_levels = torch.linspace(0, -(h_max_real + 100.0), steps=1000, device=self.device, dtype=torch.float32)
        
        # Define T on these Z-levels (Stable Stratification)
        t_ref_profile = t_bot + (t_surf - t_bot) * torch.exp(z_ref_levels / h_scale)
        
        # B. Calculate Physical Depths of Sigma Layers in the Model
        # ------------------------------------------------------------------
        # Model depths: D(i,j,k) = H(i,j) * Sigma(k)
        # self.z is (kb), self.h is (im, jm)
        sigma_3d = self.z.view(1, 1, -1)     # Shape (1, 1, kb)
        h_3d     = self.h.unsqueeze(2)       # Shape (im, jm, 1)
        
        # Broadcast to (im, jm, kb)
        model_z_depths = sigma_3d * h_3d 
        
        # C. Interpolate Reference Profile onto Model Depths
        # ------------------------------------------------------------------
        self.tb = self._vertical_interp(z_ref_levels, t_ref_profile, model_z_depths)
        
        # Finalize Thermodynamics
        self.tclim[:] = self.tb[:]
        self.sclim[:] = self.sb[:]
        self.s, self.t, self.rho = self.dens()
        self.rmean[:] = self.rho[:]

        # 6. Wind Stress
        tau0 = 0.1
        kinematic_factor = tau0 / self.rhoref 
        lat_grid = self.north_e
        arg = 2.0 * self.pi * (lat_grid - 30.0) / 20.0
        wind_stress_x = - kinematic_factor * (1.0 - torch.cos(arg))
        self.wusurf = wind_stress_x * self.dum
        self.wvsurf.zero_()

        # 7. Time Step Calculation
        g_grav = 9.81
        c_wave = math.sqrt(g_grav * torch.max(self.h).item())
        min_d = min(torch.min(self.dx).item(), torch.min(self.dy).item())
        cfl_limit = min_d / c_wave
        safety_factor = 0.4 
        self.dt = float(int(cfl_limit * safety_factor))
        
        self.horcon = float(os.environ.get("NEURALPOM_HORCON", "0.05"))
        
        # print(f"Set dt: {self.dt} s (CFL limit: {cfl_limit:.2f} s)")

        # 9. BC Flags
        self.rfe = 0.0; self.rfw = 0.0; self.rfn = 0.0; self.rfs = 0.0

        return (self.dx, self.dy, self.cor, self.east_c, self.north_c, self.east_e, self.north_e, 
                self.east_u, self.north_u, self.east_v, self.north_v, self.h, self.art, self.aru, self.arv, 
                self.fsm, self.dum, self.dvm, self.tb, self.sb, self.tclim, self.sclim, self.ub, self.uab, 
                self.elb, self.etb, self.dt, self.aam2d, self.rho, self.rmean, self.rfe, self.rfw, self.rfn, self.rfs,
                self.uabw, self.uabe, self.ele, self.elw, self.tbe, self.tbw, self.sbe, self.sbw, 
                self.tbn, self.tbs, self.sbn, self.sbs)
