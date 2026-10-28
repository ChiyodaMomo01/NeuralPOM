import math
import numpy as np
import netCDF4 as nc
import torch
import torch.nn.functional as F


torch.set_default_dtype(torch.float32)


class CheckpointingMixin:
    def serialize_checkpoint_data(self):
        def to_np(tensor):
            return tensor.detach().cpu().numpy()

        data_dict = {
            'time': self.time, 'time0': self.time0, 'iint': self.iint,
            'u': to_np(self.u),   'ub': to_np(self.ub),
            'v': to_np(self.v),   'vb': to_np(self.vb),
            't': to_np(self.t),   'tb': to_np(self.tb),
            's': to_np(self.s),   'sb': to_np(self.sb),
            'q2': to_np(self.q2),   'q2b': to_np(self.q2b),
            'q2l': to_np(self.q2l), 'q2lb': to_np(self.q2lb),
            'km': to_np(self.km),   'kh': to_np(self.kh),   'kq': to_np(self.kq),
            'el': to_np(self.el),   'elb': to_np(self.elb),
            'et': to_np(self.et),   'etb': to_np(self.etb),  'etf': to_np(self.etf),
            'ua': to_np(self.ua),   'uab': to_np(self.uab),
            'va': to_np(self.va),   'vab': to_np(self.vab),
            'utf': to_np(self.utf), 'utb': to_np(self.utb),
            'vtf': to_np(self.vtf), 'vtb': to_np(self.vtb),
            'adx2d': to_np(self.adx2d), 'ady2d': to_np(self.ady2d),
            'drx2d': to_np(self.drx2d), 'dry2d': to_np(self.dry2d),
            'aam': to_np(self.aam),
            'vfluxf': to_np(self.vfluxf), 'vfluxb': to_np(self.vfluxb),
            'wubot': to_np(self.wubot),   'wvbot': to_np(self.wvbot),
            'egb': to_np(self.egb),
            'rho': to_np(self.rho),
            'w': to_np(self.w),
            'd': to_np(self.d), 'dt': to_np(self.dt)
        }
        if hasattr(self, 'dhb') and torch.is_tensor(self.dhb):
            data_dict['dhb'] = to_np(self.dhb)
        if hasattr(self, 'uaf'):
            data_dict['uaf'] = to_np(self.uaf)
        if hasattr(self, 'vaf'):
            data_dict['vaf'] = to_np(self.vaf)
        if hasattr(self, 'elf'):
            data_dict['elf'] = to_np(self.elf)
        eta_prev_solution = getattr(self, '_eta_prev_solution', None)
        if torch.is_tensor(eta_prev_solution):
            data_dict['_eta_prev_solution'] = to_np(eta_prev_solution)
        return data_dict

    def _apply_bc_after_load(self):
        """
        Re-apply boundary conditions after checkpoint restore by calling bcond()
        with each field, ensuring identical logic to the time loop.
        """
        # Elevation: use bcond(1) for el, elb, elf, et, etb, etf
        self.el, _, _, _, _, _ = self.bcond(1, self.el, self.uaf, self.vaf, self.uf, self.vf, self.w)
        self.elb, _, _, _, _, _ = self.bcond(1, self.elb, self.uaf, self.vaf, self.uf, self.vf, self.w)
        self.elf, _, _, _, _, _ = self.bcond(1, self.elf, self.uaf, self.vaf, self.uf, self.vf, self.w)
        self.et, _, _, _, _, _ = self.bcond(1, self.et, self.uaf, self.vaf, self.uf, self.vf, self.w)
        self.etb, _, _, _, _, _ = self.bcond(1, self.etb, self.uaf, self.vaf, self.uf, self.vf, self.w)
        self.etf, _, _, _, _, _ = self.bcond(1, self.etf, self.uaf, self.vaf, self.uf, self.vf, self.w)

        # 2D velocity: use bcond(2)
        _, self.ua, self.va, _, _, _ = self.bcond(2, self.elf, self.ua, self.va, self.uf, self.vf, self.w)
        _, self.uab, self.vab, _, _, _ = self.bcond(2, self.elf, self.uab, self.vab, self.uf, self.vf, self.w)
        _, self.uaf, self.vaf, _, _, _ = self.bcond(2, self.elf, self.uaf, self.vaf, self.uf, self.vf, self.w)

        # 3D velocity: use bcond(3)
        _, _, _, self.u, self.v, _ = self.bcond(3, self.elf, self.uaf, self.vaf, self.u, self.v, self.w)
        _, _, _, self.ub, self.vb, _ = self.bcond(3, self.elf, self.uaf, self.vaf, self.ub, self.vb, self.w)

        # W: use bcond(5)
        _, _, _, _, _, self.w = self.bcond(5, self.elf, self.uaf, self.vaf, self.uf, self.vf, self.w)

        self.d[:] = self.h[:] + self.el[:]
        self.dt[:] = self.h[:] + self.et[:]

    def load_checkpoint(self, filename="checkpoint.npz", *, exact_restart=False):
        import numpy as np
        import os
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Checkpoint file {filename} not found.")

        print(f"Loading checkpoint from {filename}...")
        data = np.load(filename)

        def from_np(np_arr):
            return torch.from_numpy(np_arr).to(self.device, dtype=torch.float32)

        # 1. Restore Meta
        self.time = float(data['time'])
        self.time0 = float(data['time0']) if 'time0' in data else self.time
        self.iint = int(data['iint'])

        # 2. Restore 3D Prognostic
        self.u[:] = from_np(data['u']);   self.ub[:] = from_np(data['ub'])
        self.v[:] = from_np(data['v']);   self.vb[:] = from_np(data['vb'])
        self.t[:] = from_np(data['t']);   self.tb[:] = from_np(data['tb'])
        self.s[:] = from_np(data['s']);   self.sb[:] = from_np(data['sb'])
        
        # 3. Restore Turbulence (Critical)
        self.q2[:] = from_np(data['q2']);   self.q2b[:] = from_np(data['q2b'])
        self.q2l[:] = from_np(data['q2l']); self.q2lb[:] = from_np(data['q2lb'])
        # Load the mixing coefficients directly and do not recompute them.
        if 'km' in data:
            self.km[:] = from_np(data['km'])
            self.kh[:] = from_np(data['kh'])
            self.kq[:] = from_np(data['kq'])
        else:
            print("⚠️ Warning: KM/KH missing. Bitwise restart impossible.")

        # 4. Restore 2D External Mode
        self.el[:] = from_np(data['el']); self.elb[:] = from_np(data['elb'])
        self.ua[:] = from_np(data['ua']); self.uab[:] = from_np(data['uab'])
        self.va[:] = from_np(data['va']); self.vab[:] = from_np(data['vab'])
        
        # Check whether et exists because older checkpoints may not contain it.
        if 'et' in data:
            self.et[:] = from_np(data['et'])
            self.etb[:] = from_np(data['etb']); self.etf[:] = from_np(data['etf'])
            self.utf[:] = from_np(data['utf']); self.utb[:] = from_np(data['utb'])
            self.vtf[:] = from_np(data['vtf']); self.vtb[:] = from_np(data['vtb'])
            
            self.adx2d[:] = from_np(data['adx2d']); self.ady2d[:] = from_np(data['ady2d'])
            self.drx2d[:] = from_np(data['drx2d']); self.dry2d[:] = from_np(data['dry2d'])
            
            self.aam[:] = from_np(data['aam'])
            self.vfluxf[:] = from_np(data['vfluxf']); self.vfluxb[:] = from_np(data['vfluxb'])
            
            # Core prognostic variables required for restart.
            if 'wubot' in data:
                self.wubot[:] = from_np(data['wubot'])
                self.wvbot[:] = from_np(data['wvbot'])
                self.egb[:] = from_np(data['egb'])
            
            # Filtered Ghost Cells
            if hasattr(self, 'uaf') and 'uaf' in data: self.uaf[:] = from_np(data['uaf'])
            if hasattr(self, 'vaf') and 'vaf' in data: self.vaf[:] = from_np(data['vaf'])
            if hasattr(self, 'elf') and 'elf' in data: self.elf[:] = from_np(data['elf'])
            
        else:
            print("❌ Critical: New checkpoint format required for restart.")
            # Fallback logic removed for brevity, assume strictly new checkpoint
            
        # 5. Restore Diagnostics (Directly from file)
        # Load rho and w directly to avoid tiny floating-point drift or state resets from dens() and vertvl().
        if 'rho' in data: self.rho[:] = from_np(data['rho'])
        if 'w' in data:   self.w[:]   = from_np(data['w'])
        if '_eta_prev_solution' in data:
            self._eta_prev_solution = from_np(data['_eta_prev_solution']).clone()
        else:
            self._eta_prev_solution = None
            
        # 6. Consistency Updates (Algebraic only)
        # Same-grid exact replay must preserve the saved auxiliary depth fields;
        # recomputing them changes the trajectory when an injected checkpoint
        # intentionally carries its historical values.
        if exact_restart and 'dt' in data and 'd' in data:
            self.dt[:] = from_np(data['dt'])
            self.d[:] = from_np(data['d'])
            if 'dhb' in data:
                self.dhb = from_np(data['dhb'])
            elif hasattr(self, 'dhb'):
                delattr(self, 'dhb')
        else:
            self.dt[:] = self.h[:] + self.et[:]
            self.d[:]  = self.h[:] + self.el[:]
            self.dhb   = self.h[:] + self.elb[:]  # Needed for GOMO-like logic if used

        # -------------------------------------------------------------
        # 7. Re-apply Boundary Conditions (Critical for Restart)
        # Checkpoint may have been saved with ghost cells from a different
        # grid/mask, or BCs must be enforced before first time step.
        # -------------------------------------------------------------
        if exact_restart:
            # Same-grid closed-loop continuation must preserve the saved halo
            # and history fields exactly.  Reapplying bcond here changes the
            # state that the uninterrupted trajectory would consume.
            pass
        else:
            self._apply_bc_after_load()

        print(f"Checkpoint loaded successfully at step {self.iint}. Ready to resume.")

    def save_checkpoint(self, filename="checkpoint.npz"):
        import numpy as np
        print(f"Saving checkpoint to {filename} at time={self.time:.2f}...")
        data_dict = self.serialize_checkpoint_data()
        np.savez_compressed(filename, **data_dict)
        print("Checkpoint saved successfully.")
