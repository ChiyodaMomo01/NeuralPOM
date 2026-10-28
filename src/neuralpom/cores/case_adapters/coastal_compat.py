"""Private POM08-WAD nwad=0 coastal compatibility; production untouched."""
import types,torch
ADAPTER_VERSION='coastal-pom08-cbc-frozen-gradient-mass-v2'
def update_cbc(c):
    # pom08.f after EACH external history/depth update (lines 1560–1570).
    # With nwad=0 wetmask=fsm, zsh=0.01 m, z0b=0.005 m.
    ratio=(c.zsh+(1+c.zz[c.kbm1-1])*c.d)/c.z0b
    raw=(c.kappa/torch.log(ratio))**2
    c.cbc.copy_(torch.where(c.fsm>0,raw,torch.full_like(raw,c.cbcmin)).clamp(c.cbcmin,c.cbcmax))
    c._validation_cbc_updates+=1

def install(c,closure=True):
    c.zsh=.01;c.z0b=.005;c.cbcmin=.0025;c.cbcmax=1.
    c._validation_cbc_updates=0
    c._coastal_tide_overwrite_last_m3=torch.zeros((),device=c.device,dtype=c.el.dtype)
    c._coastal_tide_overwrite_sum_m3=torch.zeros_like(c._coastal_tide_overwrite_last_m3)
    c._validation_update_cbc=types.MethodType(update_cbc,c)
    # Imported initial cbc is retained before the first dynamics call.
    c.barotropic_shapiro=c.tracer_shapiro=c.velocity_shapiro=0.
    if closure:
        from .fortran_compat import install as install_closure
        install_closure(c)
    c._coastal_adapter_version=ADAPTER_VERSION
