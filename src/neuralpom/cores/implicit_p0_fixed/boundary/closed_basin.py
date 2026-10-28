from __future__ import annotations

class ClosedBasinBoundary:
    name = "closed_double_gyre"

    def apply_scalar2d(self, core, field):
        return field

    def apply_velocity2d(self, core, u_field, v_field):
        return u_field, v_field

    def apply_scalar3d(self, core, field):
        return field

    def apply_velocity3d(self, core, u_field, v_field):
        return u_field, v_field

    def apply_tracers(self, core):
        return core.elf, core.uaf, core.vaf, core.t, core.s, core.w

    def apply_flux2d(self, core, xflux, yflux):
        return xflux, yflux

    def apply_flux3d(self, core, xflux, yflux, k_slice):
        return xflux, yflux

    def apply_bcond(self, core, idx: int, elf, uaf, vaf, uf, vf, w):
        im_end = core.im - 1
        jm_end = core.jm - 1

        if idx == 1:
            elf = elf * core.fsm
            elf[0, :] = 0.0
            elf[im_end, :] = 0.0
            elf[:, 0] = 0.0
            elf[:, jm_end] = 0.0
            return elf, uaf, vaf, uf, vf, w

        if idx == 2:
            uaf = uaf * core.dum
            vaf = vaf * core.dvm
            uaf[0, :] = 0.0
            uaf[im_end, :] = 0.0
            uaf[:, 0] = 0.0
            uaf[:, jm_end] = 0.0
            vaf[0, :] = 0.0
            vaf[im_end, :] = 0.0
            vaf[:, 0] = 0.0
            vaf[:, jm_end] = 0.0
            return elf, uaf, vaf, uf, vf, w

        if idx == 3:
            uf = uf * core.dum.unsqueeze(2)
            vf = vf * core.dvm.unsqueeze(2)
            uf[0, :, :] = 0.0
            uf[im_end, :, :] = 0.0
            uf[:, 0, :] = 0.0
            uf[:, jm_end, :] = 0.0
            vf[0, :, :] = 0.0
            vf[im_end, :, :] = 0.0
            vf[:, 0, :] = 0.0
            vf[:, jm_end, :] = 0.0
            return elf, uaf, vaf, uf, vf, w

        if idx == 4:
            im_m1 = core.im - 2
            jm_m1 = core.jm - 2
            uf[0, :, :] = uf[1, :, :]
            uf[im_end, :, :] = uf[im_m1, :, :]
            uf[:, 0, :] = uf[:, 1, :]
            uf[:, jm_end, :] = uf[:, jm_m1, :]
            vf[0, :, :] = vf[1, :, :]
            vf[im_end, :, :] = vf[im_m1, :, :]
            vf[:, 0, :] = vf[:, 1, :]
            vf[:, jm_end, :] = vf[:, jm_m1, :]
            return elf, uaf, vaf, uf, vf, w

        if idx == 5:
            w = w * core.fsm.unsqueeze(2)
            return elf, uaf, vaf, uf, vf, w

        if idx == 6:
            uf = uf * core.fsm.unsqueeze(2)
            vf = vf * core.fsm.unsqueeze(2)
            return elf, uaf, vaf, uf, vf, w

        return elf, uaf, vaf, uf, vf, w

    def refresh_tracer_backups_after_apply(self) -> bool:
        return False

    def enforce_eta(self, core, eta) -> None:
        im_end = core.im - 1
        jm_end = core.jm - 1
        eta[0, :] = 0.0
        eta[im_end, :] = 0.0
        eta[:, 0] = 0.0
        eta[:, jm_end] = 0.0

    def prepare_eta_rhs(self, core, rhs) -> None:
        self.enforce_eta(core, rhs)
