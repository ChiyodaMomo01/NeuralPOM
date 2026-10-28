"""Explicit registry for private, reproducible Double Gyre validation cases."""
from dataclasses import dataclass, asdict
from .fortran_compat import install, VERSION

@dataclass(frozen=True)
class DoubleGyrePolicy:
    reference_case: str
    dte: int
    dti: int
    duration_seconds: int = 86400
    zsh: float = .01
    dtype: str = 'float64'
    def apply(self, core, family, filter_mode):
        install(core,zsh=self.zsh)
        core.smoth = .1 if family=='explicit' or filter_mode=='matched' else .15
        # Alpha is the external-mode algorithm parameter, not an Asselin filter.
        core.alpha = .225 if family=='explicit' else .333
        if family=='implicit':
            from .dg_boundary import install as install_dg_boundary
            install_dg_boundary(core)
            core.eta_solver_rel_tol=1e-11
            core.eta_solver_tol=1e-12
            core.eta_solver_max_iter=200
        core._registered_dg_policy = self
        return core
    def receipt(self,family,filter_mode):
        r=asdict(self)
        r.update(case_id='double_gyre_closed_wind_pdens',family=family,filter=filter_mode,
                 steps=self.duration_seconds//self.dti,closure_policy=VERSION,
                 alpha=.225 if family=='explicit' else .333,
                 smoth=.1 if family=='explicit' or filter_mode=='matched' else .15,
                 reference_alpha=.225,reference_smoth=.1,
                 filter_matching_scope='smoth only; implicit alpha retained .333; numerical algorithms remain different',
                 scratch_dtype='float64 state/profq; legacy explicit float32 momentum/radiative scratch retained',nwad=0,
                 dg_boundary_policy='POM08 idx6 + conservative old-face-depth corrector' if family=='implicit' else 'archived explicit boundary algorithm',
                 policy_generation='v3' if family=='implicit' else 'v1',
                 eta_solver_rel_tol=1e-11 if family=='implicit' else None,
                 reference_physics='archived POM08-WAD; active T/S restored; DG pdens initialized; no physics-kernel edits')
        return r

CASES={}
for label,dte,dti in [('baseline',30,900),('dt900',10,900),('dt300',10,300),('dt100',10,100)]:
    CASES[label]=DoubleGyrePolicy(f'double_gyre_active_ts_compat_dt{dti}_dte{dte}',dte,dti)
