import math
import numpy as np
import netCDF4 as nc
import torch
import torch.nn.functional as F


torch.set_default_dtype(torch.float32)


class DiagnosticsMixin:
    def printall(self, im, jm, imm1, jmm1, iskp, jskp, uab, vab, elb, d, dx, dy, 
                 time, u, v, w, t, s, rho, aam, km, kb, mode, dt, zz, z):
        """
        Prints a set of outputs (Diagnostics).
        Adapted to print statistics for large grids or matrices for small grids.
        """
        
        print(f"\n--- Diagnostics at Time = {time:.4f} ---")

        # --- 2-D horizontal fields ---
        self.prxy('Depth-averaged u, uab', time, uab, im, iskp, jm, jskp, 0.0)
        self.prxy('Depth-averaged v, vab', time, vab, im, iskp, jm, jskp, 0.0)
        self.prxy('Surface elevation, elb', time, elb, im, iskp, jm, jskp, 0.0)

        # --- Calculate and print streamfunction ---
        self.findpsi(im, jm, imm1, jmm1, time, iskp, jskp, uab, vab, d, dx, dy)

        if mode != 2:
            # --- 3-D fields ---
            
            # Set levels for output (Python indices)
            # MATLAB: 1, floor(kb/2), kb-1
            # Python: 0, kb//2, kb-2
            ko = [0, kb // 2, kb - 2] 
            
            self.prxyz('x-velocity, u', time, u, im, iskp, jm, jskp, kb, ko, 3, 0.0)
            self.prxyz('y-velocity, v', time, v, im, iskp, jm, jskp, kb, ko, 3, 0.0)
            
            # W usually wants slightly different levels in POM, keeping standard here
            self.prxyz('z-velocity, w', time, w, im, iskp, jm, jskp, kb, ko, 3, 0.0)
            
            self.prxyz('Potential temperature, t', time, t, im, iskp, jm, jskp, kb, ko, 3, 1.e-2)
            self.prxyz('Salinity, s', time, s, im, iskp, jm, jskp, kb, ko, 3, 1.e-2)
            self.prxyz('(density-1000)/rhoref, rho', time, rho, im, iskp, jm, jskp, kb, ko, 3, 1.e-5)
            
            self.prxyz('Horizontal kinematic viscosity, aam', time, aam, im, iskp, jm, jskp, kb, ko, 3, 0.0)
            self.prxyz('Vertical kinematic viscosity, km', time, km, im, iskp, jm, jskp, kb, ko, 3, 0.0)

            # --- Sections (Cross-sections) ---
            # Y-Sections (Fixed J)
            # MATLAB: 1, floor(jm/2), jm-1
            # Python: 0, jm//2, jm-2
            jo = [0, jm // 2, jm - 2]
            
            self.prxz('x-velocity, u', time, u, im, iskp, jm, kb, jo, 3, 0.0, dt, zz)
            self.prxz('z-velocity, w', time, w, im, iskp, jm, kb, jo, 3, 0.0, dt, z)
            self.prxz('Potential temperature, t', time, t, im, iskp, jm, kb, jo, 3, 1.e-2, dt, zz)
            self.prxz('Salinity, s', time, s, im, iskp, jm, kb, jo, 3, 1.e-2, dt, zz)

            # X-Sections (Fixed I)
            # Python: 0, im//2, im-2
            io = [0, im // 2, im - 2]
            
            self.pryz('x-velocity, u', time, u, im, jm, jskp, kb, io, 3, 0.0, dt, zz)
            self.pryz('z-velocity, w', time, w, im, jm, jskp, kb, io, 3, 0.0, dt, zz)
            self.pryz('Potential temperature, t', time, t, im, jm, jskp, kb, io, 3, 1.e-2, dt, zz)

        print("---------------------------------------")

    def prxy(self, label, time, f, im, iskp, jm, jskp, scale):
        """Prints statistics or values for a 2D field."""
        f_np = f.detach().cpu().numpy()
        
        if im < 15 and jm < 15:
            print(f"\n{label} (Time: {time:.2f})")
            # Simple matrix print
            print(f_np)
        else:
            # Stats print
            vmin = f_np.min()
            vmax = f_np.max()
            vmean = f_np.mean()
            print(f"{label:35s}: Min={vmin:+.4e}  Max={vmax:+.4e}  Mean={vmean:+.4e}")

    def prxyz(self, label, time, f, im, iskp, jm, jskp, kb, ko, num_levels, scale):
        """Prints statistics for specific horizontal slices of a 3D field."""
        f_np = f.detach().cpu().numpy()
        
        for k_idx in ko:
            # Safety check for index
            if k_idx < 0 or k_idx >= kb: continue
            
            layer_data = f_np[:, :, k_idx]
            layer_label = f"{label} [K={k_idx}]"
            
            if im < 15 and jm < 15:
                print(f"\n{layer_label}")
                print(layer_data)
            else:
                vmin = layer_data.min()
                vmax = layer_data.max()
                vmean = layer_data.mean()
                print(f"{layer_label:35s}: Min={vmin:+.4e}  Max={vmax:+.4e}  Mean={vmean:+.4e}")

    def prxz(self, label, time, f, im, iskp, jm, kb, jo, num_secs, scale, dt, zz):
        """Prints statistics for X-Z cross sections (Fixed J)."""
        f_np = f.detach().cpu().numpy()
        
        for j_idx in jo:
            if j_idx < 0 or j_idx >= jm: continue
            
            # Section: All I, Fixed J, All K
            sec_data = f_np[:, j_idx, :]
            sec_label = f"{label} [J={j_idx}]"
            
            vmin = sec_data.min()
            vmax = sec_data.max()
            print(f"{sec_label:35s}: Min={vmin:+.4e}  Max={vmax:+.4e}")

    def pryz(self, label, time, f, im, jm, jskp, kb, io, num_secs, scale, dt, zz):
        """Prints statistics for Y-Z cross sections (Fixed I)."""
        f_np = f.detach().cpu().numpy()
        
        for i_idx in io:
            if i_idx < 0 or i_idx >= im: continue
            
            # Section: Fixed I, All J, All K
            sec_data = f_np[i_idx, :, :]
            sec_label = f"{label} [I={i_idx}]"
            
            vmin = sec_data.min()
            vmax = sec_data.max()
            print(f"{sec_label:35s}: Min={vmin:+.4e}  Max={vmax:+.4e}")

    def findpsi(self, im, jm, imm1, jmm1, time, iskp, jskp, uab, vab, d, dx, dy):
        """
        Calculates transport streamfunction.
        Simplified implementation: Just calculates total transport range.
        """
        # Transport U = uab * d * dy
        # Transport V = vab * d * dx
        # In a real implementation, we would integrate these.
        
        trans_u = uab * d * dy
        trans_v = vab * d * dx
        
        t_u_val = trans_u.detach().cpu().numpy()
        t_v_val = trans_v.detach().cpu().numpy()
        
        # Just print max transport for now
        print(f"{'Streamfunction (Transport Proxy)':35s}: MaxU={np.max(np.abs(t_u_val)):.4e}  MaxV={np.max(np.abs(t_v_val)):.4e}")
