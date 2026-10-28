"""Numerical identities and production settings for interchangeable V1 cores."""
from dataclasses import dataclass

from neuralpom.core_factory import resolve_core_class


@dataclass(frozen=True)
class CoreSpec:
    name: str
    numerics_version: str
    physical_step_seconds: int = 600
    external_step_seconds: int = 20
    split_count: int = 30
    alpha: float = 0.
    smoth: float = 0.
    horcon: float = .05

    @property
    def core_class(self):
        return resolve_core_class(self.name)

    def configure(self, core):
        if self.name == 'implicit_p0_fixed':
            core.spatial_shapiro_enabled = False
            core.barotropic_shapiro = 0.
            core.tracer_shapiro = 0.
            core.velocity_shapiro = 0.
            core.eta_solver_rel_tol = 1e-5
            core.eta_solver_preconditioner = 'spectral'
        core.horcon = self.horcon
        core.alpha = self.alpha
        core.smoth = self.smoth
        core.prtd1 = core.prtd2 = 1e9
        core.iprint = 10**9
        if (core.NUMERICS_VERSION != self.numerics_version or
                core.dti != self.physical_step_seconds or
                core.dte != self.external_step_seconds or
                core.isplit != self.split_count or core.kbm1 != 15):
            raise ValueError(f'{self.name} core does not match the registered numerical settings')


CORE_SPECS = {
    'implicit_p0_fixed': CoreSpec('implicit_p0_fixed',
        'implicit-dg-v2-aligned-1e5-p0-barotropic-coupling-fixed', alpha=.333, smoth=.15),
    'explicit': CoreSpec('explicit', 'explicit-dg-v2-split-smag-v2align-closure',
                         alpha=.225, smoth=.10),
}


def core_spec(name):
    try:
        return CORE_SPECS[name]
    except KeyError as exc:
        raise ValueError(f'Unsupported V1 core {name!r}; choose from {sorted(CORE_SPECS)}') from exc
