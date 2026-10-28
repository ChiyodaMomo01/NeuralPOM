import math
import numpy as np
import netCDF4 as nc
import torch
import torch.nn.functional as F


torch.set_default_dtype(torch.float32)


class GridMixin:
    def depth(self):
        """
        Generates vertical sigma-coordinate grid (z, zz, dz, dzz).
        Distributes layers logarithmically near surface/bottom and linearly in between.
        """
        
        # KDZ sequence: Powers of 2 (1, 1, 2, 4, ...)
        # Used to define layer thickness ratios
        kdz_list = [1.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0]
        kdz = torch.tensor(kdz_list, device=self.device, dtype=torch.float32)

        # Reset arrays to zero
        self.z.zero_()
        self.zz.zero_()
        self.dz.zero_()
        self.dzz.zero_()

        # ==========================================
        # 1. Surface Logarithmic Layers
        # ==========================================
        # MATLAB: for k=2:kl1
        # Python: k from 1 to kl1-1
        for k in range(1, self.kl1):
            # z(k) = z(k-1) + kdz(k-1)
            # Note: MATLAB k-1 accesses previous Z. 
            # kdz index: MATLAB k-1 (if k=2, idx 1). Python k-1 (if k=1, idx 0).
            self.z[k] = self.z[k-1] + kdz[k-1]

        # Calculate delta for linear portion
        # delz = z(kl1) - z(kl1-1)
        # Python indices: kl1-1 and kl1-2
        delz = self.z[self.kl1 - 1] - self.z[self.kl1 - 2]

        # ==========================================
        # 2. Middle Linear Layers
        # ==========================================
        # MATLAB: for k=kl1+1:kl2
        # Python: k from kl1 to kl2-1
        for k in range(self.kl1, self.kl2):
            self.z[k] = self.z[k-1] + delz

        # ==========================================
        # 3. Bottom Logarithmic Layers
        # ==========================================
        # MATLAB: for k=kl2+1:kb
        # Python: k from kl2 to kb-1
        
        # Constant denominator: kdz(kb-kl2)
        # Python index for kdz: (kb-kl2) - 1
        denom = kdz[self.kb - self.kl2 - 1]
        
        for k in range(self.kl2, self.kb):
            # dz(k) = kdz(kb-k+1) * delz / denom
            # kdz index logic:
            # If k = kb (MATLAB), index is 1.
            # If k = kb-1 (Python), we want index 0.
            # Formula: self.kb - 1 - k
            kdz_val = kdz[self.kb - 1 - k]
            
            dz_val = kdz_val * delz / denom
            self.dz[k] = dz_val # Temporary storage in dz, will be overwritten later
            self.z[k] = self.z[k-1] + dz_val

        # ==========================================
        # 4. Normalization (0 to -1)
        # ==========================================
        # z(k) = -z(k) / z(kb)
        total_depth = self.z[self.kb - 1].clone() # Use clone to avoid in-place issues
        self.z = -self.z / total_depth

        # ==========================================
        # 5. Calculate Mid-points (zz)
        # ==========================================
        # MATLAB: for k=1:kb-1 -> zz(k) = 0.5*(z(k)+z(k+1))
        # Vectorized
        s_k = slice(0, self.kb - 1)
        s_kp1 = slice(1, self.kb)
        
        self.zz[s_k] = 0.5 * (self.z[s_k] + self.z[s_kp1])

        # Extrapolate bottom zz
        # MATLAB: zz(kb) = 2.0*zz(kb-1) - zz(kb-2)
        # Python indices: kb-1, kb-2, kb-3
        self.zz[self.kb - 1] = 2.0 * self.zz[self.kb - 2] - self.zz[self.kb - 3]

        # ==========================================
        # 6. Calculate Layer Thicknesses (dz, dzz)
        # ==========================================
        # MATLAB: for k=1:kb-1
        # dz(k) = z(k) - z(k+1)
        # dzz(k) = zz(k) - zz(k+1)
        
        self.dz[s_k] = self.z[s_k] - self.z[s_kp1]
        self.dzz[s_k] = self.zz[s_k] - self.zz[s_kp1]

        # Bottom boundary (kb)
        self.dz[self.kb - 1] = 0.0
        self.dzz[self.kb - 1] = 0.0

        return kdz, self.z, self.zz, self.dz, self.dzz

    def areas_masks(self):
        """
        Calculates areas (art, aru, arv) and masks (fsm, dum, dvm).
        Exact translation of the provided MATLAB code.
        """
        
        # 1. Calculate areas of "t" and "s" cells (art)
        self.art = self.dx * self.dy
        
        # 2. Calculate areas of "u" and "v" cells (aru, arv)
        s_i = slice(1, self.im)
        s_im1 = slice(0, self.im - 1)
        s_j = slice(1, self.jm)
        s_jm1 = slice(0, self.jm - 1)
        
        dx_sum_u = self.dx[s_i, s_j] + self.dx[s_im1, s_j]
        dy_sum_u = self.dy[s_i, s_j] + self.dy[s_im1, s_j]
        self.aru[s_i, s_j] = 0.25 * dx_sum_u * dy_sum_u
        
        dx_sum_v = self.dx[s_i, s_j] + self.dx[s_i, s_jm1]
        dy_sum_v = self.dy[s_i, s_j] + self.dy[s_i, s_jm1]
        self.arv[s_i, s_j] = 0.25 * dx_sum_v * dy_sum_v
        
        # Boundary Extrapolation for Areas
        self.aru[0, :] = self.aru[1, :]
        self.arv[0, :] = self.arv[1, :]
        self.aru[:, 0] = self.aru[:, 1]
        self.arv[:, 0] = self.arv[:, 1]
        
        # 3. Free surface mask (fsm)
        self.fsm.zero_()
        self.dum.zero_()
        self.dvm.zero_()
        
        self.fsm = torch.where(self.h > 1.0, 
                               torch.tensor(1.0, device=self.device, dtype=torch.float32), 
                               torch.tensor(0.0, device=self.device, dtype=torch.float32))
        
        # 4. Set up velocity masks (dum, dvm)
        self.dum[s_i, s_j] = self.fsm[s_i, s_j] * self.fsm[s_im1, s_j]
        self.dvm[s_i, s_j] = self.fsm[s_i, s_j] * self.fsm[s_i, s_jm1]
        
        # ==========================================================
        # Boundary extrapolation for masks.
        # ==========================================================
        # Keep ghost-cell masks consistent with the boundary so bcond does not zero them out.
        self.dum[0, :] = self.dum[1, :]
        self.dvm[0, :] = self.dvm[1, :]
        
        self.dum[:, 0] = self.dum[:, 1]
        self.dvm[:, 0] = self.dvm[:, 1]
        # ==========================================================
        
        return self.art, self.aru, self.arv, self.fsm, self.dum, self.dvm

    def slpmax(self, slmax):
        """
        Adjusts bottom topography (h) so that cell-to-cell variations 
        do not exceed parameter slmax.
        Limit: |h1 - h2| / (h1 + h2) <= slmax
        """
        
        # Number of smoothing iterations
        num_loops = 10
        
        # Masks for inner domain
        # Range: j=2:jm-1, i=2:im-1 -> Python 1:jm-1, 1:im-1
        # Note: Python slices exclude end index
        s_i = slice(1, self.im - 1)
        s_ip1 = slice(2, self.im)
        s_j = slice(1, self.jm - 1)
        s_jp1 = slice(2, self.jm)
        
        # Loop
        for _ in range(num_loops):
            
            # =============================================================
            # 1. Sweep Right (Check i vs i+1)
            # =============================================================
            # Condition: fsm(i,j) != 0 AND fsm(i+1,j) != 0
            mask_valid = (self.fsm[s_i, s_j] != 0) & (self.fsm[s_ip1, s_j] != 0)
            
            # Calculate Slope Ratio
            # r = |h(i+1) - h(i)| / (h(i) + h(i+1))
            h_i = self.h[s_i, s_j]
            h_ip1 = self.h[s_ip1, s_j]
            
            diff = h_ip1 - h_i
            summ = h_i + h_ip1
            ratio = torch.abs(diff) / (summ + 1e-20)
            
            # Mask for exceeding limit
            mask_exceed = mask_valid & (ratio >= slmax)
            
            if torch.any(mask_exceed):
                # Calculate adjustments
                # mean = (h(i+1) + h(i)) / 2
                mean = summ * 0.5
                
                # del = sign(diff) * slmax
                del_val = torch.sign(diff) * slmax
                
                # Update values (Vectorized)
                # h(i+1) = mean * (1 + del)
                # h(i)   = mean * (1 - del)
                
                new_h_ip1 = mean * (1.0 + del_val)
                new_h_i   = mean * (1.0 - del_val)
                
                # Apply only where mask_exceed is True
                self.h[s_ip1, s_j] = torch.where(mask_exceed, new_h_ip1, self.h[s_ip1, s_j])
                self.h[s_i, s_j]   = torch.where(mask_exceed, new_h_i,   self.h[s_i, s_j])

            # =============================================================
            # 2. Sweep Left (Check i+1 vs i, reverse order logic)
            # =============================================================
            # In a vectorized approach, Right and Left sweeps cover the same pairs.
            # However, to mimic the iterative effect of sweeping back, we repeat the check.
            # Since we updated self.h in place above, re-running the same logic 
            # effectively acts as a second pass. The "direction" matters less in parallel.
            
            # Let's repeat the exact same logic block to simulate the "Left Sweep"
            # checking the exact same neighbors (i, i+1) but with updated h.
            
            h_i = self.h[s_i, s_j]
            h_ip1 = self.h[s_ip1, s_j]
            diff = h_ip1 - h_i
            summ = h_i + h_ip1
            ratio = torch.abs(diff) / (summ + 1e-20)
            mask_exceed = mask_valid & (ratio >= slmax)
            
            if torch.any(mask_exceed):
                mean = summ * 0.5
                del_val = torch.sign(diff) * slmax
                new_h_ip1 = mean * (1.0 + del_val)
                new_h_i   = mean * (1.0 - del_val)
                self.h[s_ip1, s_j] = torch.where(mask_exceed, new_h_ip1, self.h[s_ip1, s_j])
                self.h[s_i, s_j]   = torch.where(mask_exceed, new_h_i,   self.h[s_i, s_j])

            # =============================================================
            # 3. Sweep Up (Check j vs j+1)
            # =============================================================
            # Condition: fsm(i,j) != 0 AND fsm(i,j+1) != 0
            mask_valid_y = (self.fsm[s_i, s_j] != 0) & (self.fsm[s_i, s_jp1] != 0)
            
            h_j = self.h[s_i, s_j]
            h_jp1 = self.h[s_i, s_jp1]
            
            diff = h_jp1 - h_j
            summ = h_j + h_jp1
            ratio = torch.abs(diff) / (summ + 1e-20)
            
            mask_exceed_y = mask_valid_y & (ratio >= slmax)
            
            if torch.any(mask_exceed_y):
                mean = summ * 0.5
                del_val = torch.sign(diff) * slmax
                
                new_h_jp1 = mean * (1.0 + del_val)
                new_h_j   = mean * (1.0 - del_val)
                
                self.h[s_i, s_jp1] = torch.where(mask_exceed_y, new_h_jp1, self.h[s_i, s_jp1])
                self.h[s_i, s_j]   = torch.where(mask_exceed_y, new_h_j,   self.h[s_i, s_j])

            # =============================================================
            # 4. Sweep Down (Check j vs j+1, second pass)
            # =============================================================
            h_j = self.h[s_i, s_j]
            h_jp1 = self.h[s_i, s_jp1]
            diff = h_jp1 - h_j
            summ = h_j + h_jp1
            ratio = torch.abs(diff) / (summ + 1e-20)
            mask_exceed_y = mask_valid_y & (ratio >= slmax)
            
            if torch.any(mask_exceed_y):
                mean = summ * 0.5
                del_val = torch.sign(diff) * slmax
                new_h_jp1 = mean * (1.0 + del_val)
                new_h_j   = mean * (1.0 - del_val)
                self.h[s_i, s_jp1] = torch.where(mask_exceed_y, new_h_jp1, self.h[s_i, s_jp1])
                self.h[s_i, s_j]   = torch.where(mask_exceed_y, new_h_j,   self.h[s_i, s_j])

        return self.h
