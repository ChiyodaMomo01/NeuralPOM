"""B1: separate momentum/tracer coefficients without changing Q2/Q2L diffusion.

The stored ``aam`` remains the production reference (horcon=0.05). Scaling
at the consuming operator preserves injected initial aam and the native
old/new coefficient timing, including advct before the Smagorinsky update.
No production core, dataset identity, or frozen evaluation is modified.
"""
from functools import lru_cache
import inspect
import math

from neuralpom.core_factory import CORE_REGISTRY
from neuralpom.hybrid import build_initialized_core
from neuralpom.v1.core_spec import core_spec

REFERENCE_CM = 0.05
REFERENCE_CC = 0.01
CLOSURE_VERSION = 'b1-separated-smag-v1'


@lru_cache(None)
def _signature(function):
    return inspect.signature(function)


def _rescaled_call(method, args, kwargs, name, ratio):
    # Preserve the exact production arithmetic for B0, not multiply by 1.
    if ratio == 1.0:
        return method(*args, **kwargs)
    bound = _signature(method.__func__).bind(method.__self__, *args, **kwargs)
    bound.arguments[name] = bound.arguments[name] * ratio
    return method(*bound.args[1:], **bound.kwargs)


class SeparatedSmagorinsky:
    """Mixin state is scalar-only so the existing full-state restart captures it."""
    def advct(self, *args, **kwargs):
        return _rescaled_call(super().advct, args, kwargs, 'aam',
                              self.baseline_cm / REFERENCE_CM)

    def advave(self, *args, **kwargs):
        return _rescaled_call(super().advave, args, kwargs, 'aam2d',
                              self.baseline_cm / REFERENCE_CM)

    def advt1(self, *args, **kwargs):
        return _rescaled_call(super().advt1, args, kwargs, 'tprni',
                              self.baseline_cc / REFERENCE_CC)

    # advq intentionally inherited unchanged: turbulence diffusion uses aam.


class ExplicitSmagorinsky(SeparatedSmagorinsky, CORE_REGISTRY['explicit']):
    pass


class ImplicitSmagorinsky(SeparatedSmagorinsky, CORE_REGISTRY['implicit_p0_fixed']):
    pass


BASELINE_CLASSES = {'explicit': ExplicitSmagorinsky,
                    'implicit_p0_fixed': ImplicitSmagorinsky}


def build_smagorinsky_core(core_name, device, im=90, jm=82, *, cm=.05, cc=.01):
    for name, value in [('cm', cm), ('cc', cc)]:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    spec = core_spec(core_name)
    core = build_initialized_core(BASELINE_CLASSES[core_name], device, im, jm)
    spec.configure(core)
    if not math.isclose(float(core.tprni), .2, rel_tol=0, abs_tol=1e-12):
        raise ValueError('B1 requires production tprni=0.2')
    core.baseline_cm = float(cm)
    core.baseline_cc = float(cc)
    core.baseline_cq = REFERENCE_CM
    core.baseline_closure_version = CLOSURE_VERSION
    core.baseline_reference_core = core_name
    return core


def closure_contract(core):
    """Must be included alongside the reference data/core contract in manifests."""
    return dict(family='B1', closure_version=core.baseline_closure_version,
                reference_core=core.baseline_reference_core,
                reference_numerics_version=core.NUMERICS_VERSION,
                cm=core.baseline_cm, cc=core.baseline_cc, cq=core.baseline_cq,
                reference_aam_coefficient=REFERENCE_CM,
                coefficient_timing='production-including-injected-initial-aam')
