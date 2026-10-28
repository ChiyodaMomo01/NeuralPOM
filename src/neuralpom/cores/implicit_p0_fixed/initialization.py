import torch


torch.set_default_dtype(torch.float32)


class InitializationMixin:
    def run_initialization(self):
        if not self.implicit_case_defines_vertical_grid():
            self.dz, self.z, self.zz, self.dz, self.dzz = self.depth()

        self.run_implicit_case_setup()

        # Inertial period calculation
        mid_i = (self.im) // 2
        mid_j = (self.jm) // 2
        
        cor_val = torch.abs(self.cor[mid_i, mid_j]) + 1e-20
        self.period = (2.0 * self.pi) / cor_val / 86400.0

        # Initialize time
        self.time0 = 0.0
        self.time = 0.0

        # Initial conditions assignments
        self.ua = self.uab.clone()
        self.va = self.vab.clone()
        self.el = self.elb.clone()
        self.et = self.etb.clone()
        self.etf = self.et.clone()
        
        self.d = self.h + self.el
        self.dt = self.h + self.et
        
        self.w[:, :, 0] = self.vfluxf

        # Loop Initialization
        self.l = 0.1 * self.dt.unsqueeze(2).expand(-1, -1, self.kb)
        self.q2b = torch.full_like(self.q2b, self.small)
        self.q2lb = self.l * self.q2b
        # Guard the square root against tiny initial q2 values.
        self.kh = self.l * torch.sqrt(self.q2b + 1e-8)
        self.km = self.kh.clone()
        self.kq = self.kh.clone()
        # Support both tensor-valued and scalar-valued viscosity initialization.
        if isinstance(self.aam_init, torch.Tensor):
            if self.aam_init.shape == self.aam.shape:
                self.aam = self.aam_init.clone()
            else:
                self.aam = torch.full_like(self.aam, 500.0) 
        else:
            self.aam = torch.full_like(self.aam, self.aam_init)

        # Slicing setup for 3D
        self.q2[:, :, :self.kbm1]  = self.q2b[:, :, :self.kbm1]
        self.q2l[:, :, :self.kbm1] = self.q2lb[:, :, :self.kbm1]
        self.t[:, :, :self.kbm1]   = self.tb[:, :, :self.kbm1]
        self.s[:, :, :self.kbm1]   = self.sb[:, :, :self.kbm1]
        self.u[:, :, :self.kbm1]   = self.ub[:, :, :self.kbm1]
        self.v[:, :, :self.kbm1]   = self.vb[:, :, :self.kbm1]

        # Call DENS
        self.s, self.t, self.rho = self.dens()

        # Call BAROPG
        self.rho, self.drhox, self.drhoy = self.baropg()

        # Integral of density gradients
        drhox_slice = self.drhox[:, :, :self.kbm1]
        drhoy_slice = self.drhoy[:, :, :self.kbm1]
        dz_slice    = self.dz[:self.kbm1]
        dz_reshaped = dz_slice.view(1, 1, -1)
        
        self.drx2d = torch.sum(drhox_slice * dz_reshaped, dim=2)
        self.dry2d = torch.sum(drhoy_slice * dz_reshaped, dim=2)

        # Bottom Friction Coefficient (CBC)
        zz_val = self.zz[self.kbm1 - 1]
        term = (1.0 + zz_val) * self.h / self.z0b
        term = torch.clamp(term, min=1e-12) 
        numer = self.kappa
        denom = torch.log(term)
        raw_cbc = (numer / denom) ** 2
        self.cbc = torch.clamp(raw_cbc, min=self.cbcmin, max=self.cbcmax)

        # External (2-D) CFL time step calculation
        dx_safe = torch.clamp(self.dx, min=1e-6)
        dy_safe = torch.clamp(self.dy, min=1e-6)
        h_safe  = self.h + self.small
        geom_term = torch.sqrt(1.0 / (dx_safe**2) + 1.0 / (dy_safe**2))
        grav_term = torch.sqrt(self.grav * h_safe)
        self.tps = (0.5 / geom_term) / grav_term * self.fsm

        # Final adjustments
        self.d = self.h + self.el
        self.dt = self.h + self.et
        self.time = self.time0
