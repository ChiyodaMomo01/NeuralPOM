"""Private POM08-WAD nwad=0 Jiaozhou Bay (pom_jzb_nowad) compatibility; production untouched."""
import types
import torch

ADAPTER_VERSION = "jzb-pom08-cbc-frozen-gradient-mass-v1"


def update_jzb_cbc(self):
    # Matches pom_jzb_nowad/pom08.f lines 1577-1584:
    #   cbc(i,j) = cbcmin
    #   if (d(i,j) >= hc .and. wetmask(i,j) > 0) then
    #       cbc(i,j) = (kappa / log((zsh + (1.e0 + zz(kbm1)) * d(i,j)) / z0b)) ** 2
    #   endif
    #   cbc(i,j) = min(cbcmax, max(cbcmin, cbc(i,j)))
    zz_b = self.zz[self.kbm1 - 1]
    zsh = float(getattr(self, "zsh", 0.01))
    z0b = float(getattr(self, "z0b", 0.01))
    cbcmin = float(getattr(self, "cbcmin", 0.0025))
    cbcmax = float(getattr(self, "cbcmax", 1.0))
    hc = float(getattr(self, "hc", 0.05))
    valid = (self.fsm > 0) & (self.d >= hc)
    safe_d = torch.where(valid, self.d, torch.ones_like(self.d))
    raw = (float(self.kappa) / torch.log((zsh + (1.0 + zz_b) * safe_d) / z0b)) ** 2
    cbc = torch.where(valid, raw, torch.full_like(raw, cbcmin))
    self.cbc = torch.clamp(cbc, min=cbcmin, max=cbcmax)
    self._validation_cbc_updates = getattr(self, "_validation_cbc_updates", 0) + 1
    return self.cbc


def install(c, closure=True):
    c.mode = 4
    c.lramp = True
    c.period = 3.0
    c.horcon = 0.1
    c.aam_init = 50.0
    c.zsh = 0.01
    c.z0b = 0.01
    c.hc = 0.05
    c.cbcmin = 0.0025
    c.cbcmax = 1.0
    c.hmax = 60.0
    c.tprni = 0.2
    c.umol = 2.0e-5
    c._validation_cbc_updates = 0
    c._coastal_tide_overwrite_last_m3 = torch.zeros((), device=c.device, dtype=c.el.dtype)
    c._coastal_tide_overwrite_sum_m3 = torch.zeros_like(c._coastal_tide_overwrite_last_m3)
    c._validation_update_cbc = types.MethodType(update_jzb_cbc, c)
    # Imported initial cbc is retained before the first dynamics call.
    c.barotropic_shapiro = c.tracer_shapiro = c.velocity_shapiro = 0.0
    c.generation_fast_path = True
    c.runtime_no_grad_fast_path = True
    c.runtime_vamax_safe_fast_path = True
    c.runtime_diagnostics = False
    if hasattr(c, "_accel_cache_ready"):
        c._accel_cache_ready = False
    if closure:
        try:
            from .fortran_compat import install as install_closure
        except ImportError:
            from fortran_compat import install as install_closure
        install_closure(c, zsh=0.01)
    c._coastal_adapter_version = ADAPTER_VERSION
    c._jzb_adapter_version = ADAPTER_VERSION
