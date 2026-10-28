"""Install frozen forward policies on one existing, initialized core instance.

These policies are validated for physical forward comparisons. They do not
register a training adjoint. Use ``torch.no_grad()`` for model advancement.
"""
from importlib import import_module
from pathlib import Path
from threading import RLock
import hashlib
import math
import types

import torch

_DTYPE_LOCK = RLock()
_GEOMETRY = {
    'double_gyre': (90, 82, 16),
    'coastal_tide': (161, 101, 5),
    'seamount': (65, 49, 16),
    'jiaozhou_bay': (312, 213, 17),
}
_CLOCKS = {
    'double_gyre': {(30., 900.), (10., 900.), (10., 300.), (10., 100.)},
    'coastal_tide': {(1., 10.)},
    'seamount': {(5., 150.)},
    'jiaozhou_bay': {(1., 10.)},
}
_COASTAL_TIDE_SHA256 = '8e33add7787e334a4ca3a5b306182a4f90d9349abd106badfd139796a4785bac'
_JIAOZHOU_BAY_SHA256 = '4341da5432389865380ba43fcd5bff5b952ac10e5269d1d7873b3dc0af6db6b2'


def _bind_mixin(core, mixin):
    for name, method in vars(mixin).items():
        if callable(method) and not name.startswith('__'):
            setattr(core, name, types.MethodType(method, core))


def _scoped_forward(core, *args, **kwargs):
    if torch.is_grad_enabled():
        raise RuntimeError('Registered case policies require advancement inside torch.no_grad(); their training adjoints are unvalidated.')
    # Some archived kernels allocate default-dtype scratch. Scope that process
    # setting to this advancement and restore it even if a model step fails.
    with _DTYPE_LOCK:
        previous = torch.get_default_dtype()
        requested_steps = int(core.iend)
        if requested_steps < 1 or requested_steps != core.iend:
            raise ValueError('iend must be a positive integer number of internal steps')
        start_time = float(core.time)
        try:
            torch.set_default_dtype(core.t.dtype)
            result = None
            # The registered runners advance one internal step per call. Keep
            # that clock/history protocol for callers requesting several steps.
            for _ in range(requested_steps):
                core.time0 = float(core.time)
                core.iend = 1
                if core._case_adapter_registration['case'] in ('coastal_tide', 'jiaozhou_bay'):
                    core._tide_substep = 0
                result = core._case_adapter_forward(*args, **kwargs)
            return result
        finally:
            core.iend = requested_steps
            core.time0 = start_time
            torch.set_default_dtype(previous)


def apply_case_adaptation(core, case, family, *, tide_path=None,
                          filter_mode='matched', strict=True):
    """Apply the registered case policy after importing its common initial state.

    ``case`` is double_gyre, coastal_tide, seamount or jiaozhou_bay (alias jzb_tide).
    ``family`` is explicit or implicit. The registered geometry includes the dummy bottom layer.
    ``strict=False`` permits an unvalidated geometry/clock and marks the
    returned registration accordingly. No original class is modified.
    """
    if case == 'jzb_tide':
        case = 'jiaozhou_bay'
    if case not in _GEOMETRY or family not in ('explicit', 'implicit'):
        raise ValueError(f'Unsupported case/family: {case}/{family}')
    if hasattr(core, '_case_adapter_registration'):
        raise RuntimeError('A case policy is already installed on this instance')
    is_implicit = any('neuralpom.cores.implicit_p0_fixed' in cls.__module__ for cls in type(core).__mro__)
    is_explicit = any('neuralpom.cores.explicit' in cls.__module__ for cls in type(core).__mro__)
    if not (is_implicit if family == 'implicit' else is_explicit):
        raise ValueError('family does not match the supplied physical core class')
    if core.t.dtype != torch.float64:
        raise ValueError('Registered comparisons require float64 state imported from the common reference')
    if filter_mode not in ('native', 'matched'):
        raise ValueError('filter_mode must be native or matched')
    geometry = (int(core.im), int(core.jm), int(core.kb))
    clock = (float(core.dte), float(core.dti))
    im, jm, kb = geometry
    for name in ('h', 'el', 'd', 'dt', 'dx', 'dy', 'fsm', 'u', 'v', 't', 's', 'q2', 'q2l', 'dz'):
        value = getattr(core, name)
        shape = (kb,) if name == 'dz' else ((im, jm, kb) if name in ('u', 'v', 't', 's', 'q2', 'q2l') else (im, jm))
        if tuple(value.shape) != shape or value.dtype != torch.float64 or value.device != core.t.device:
            raise ValueError(f'Common-reference state shape/dtype/device mismatch: {name}')
    expected_split = clock[1] / clock[0]
    if expected_split < 1 or not math.isclose(expected_split, round(expected_split), abs_tol=1e-12):
        raise ValueError('The internal timestep must be an integer multiple of the external timestep')
    derived = {'isplit': expected_split, 'dti2': 2 * clock[1], 'dte2': 2 * clock[0],
               'ispi': 1 / expected_split, 'isp2i': 1 / (2 * expected_split)}
    for name, expected in derived.items():
        if not math.isclose(float(getattr(core, name)), expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f'Inconsistent derived clock setting: {name}')
    registered = geometry == _GEOMETRY[case] and clock in _CLOCKS[case]
    filters_disabled = family != 'implicit' or (
        not bool(getattr(core, 'spatial_shapiro_enabled', False)) and
        all(float(getattr(core, name, 0.)) == 0.
            for name in ('barotropic_shapiro', 'tracer_shapiro', 'velocity_shapiro')))
    if strict and not registered:
        raise ValueError(f'Unvalidated {case} geometry/clock: {geometry}/{clock}')
    if strict and not filters_disabled:
        raise ValueError('Registered comparisons require the optional Implicit spatial Shapiro filters to be disabled')
    tide_sha = None
    expected_tide_sha = None
    if case in ('coastal_tide', 'jiaozhou_bay'):
        default_tide_file = 'data/amph_pom.dat' if case == 'coastal_tide' else 'data/jzb_amph_pom.dat'
        expected_tide_sha = _COASTAL_TIDE_SHA256 if case == 'coastal_tide' else _JIAOZHOU_BAY_SHA256
        tide_path = Path(tide_path) if tide_path is not None else Path(__file__).parent / default_tide_file
        if not tide_path.is_file():
            raise ValueError(f'{case} requires the registered amph_pom.dat tide_path')
        tide_sha = hashlib.sha256(tide_path.read_bytes()).hexdigest()
        if strict and tide_sha != expected_tide_sha:
            raise ValueError('Tide forcing differs from the registered reference; use strict=False only for a separately evaluated protocol')
    with _DTYPE_LOCK:
        previous = torch.get_default_dtype()
        try:
            if case == 'double_gyre':
                policy_module = import_module('.dg_policy', __package__)
                label = next((name for name, policy in policy_module.CASES.items()
                              if (float(policy.dte), float(policy.dti)) == clock), None)
                policy = policy_module.CASES[label] if label else policy_module.DoubleGyrePolicy('unvalidated_clock', int(core.dte), int(core.dti))
                policy.apply(core, family, filter_mode)
                version = 'dg-pom08-policy-v3' if family == 'implicit' else 'dg-pom08-policy-v1'
            elif case == 'coastal_tide':
                core.alpha = .225 if family == 'explicit' else .333
                core.smoth = .1 if family == 'explicit' else .15
                for module, class_name in (('momentum', 'MomentumMixin'), ('tracers', 'TracersMixin'), ('time_loop', 'TimeLoopMixin')):
                    implementation = import_module(f'.kernels.coastal_{family}_{module}', __package__)
                    _bind_mixin(core, getattr(implementation, class_name))
                boundary = import_module('.coastal_boundary', __package__)
                compatibility = import_module('.coastal_compat', __package__)
                boundary.install(core, family, Path(tide_path))
                compatibility.install(core)
                version = compatibility.ADAPTER_VERSION
            elif case == 'jiaozhou_bay':
                core.alpha = .225 if family == 'explicit' else .333
                core.smoth = .1 if family == 'explicit' else .15
                for module, class_name in (('momentum', 'MomentumMixin'), ('tracers', 'TracersMixin'), ('time_loop', 'TimeLoopMixin')):
                    implementation = import_module(f'.kernels.coastal_{family}_{module}', __package__)
                    _bind_mixin(core, getattr(implementation, class_name))
                boundary = import_module('.jzb_boundary', __package__)
                compatibility = import_module('.jzb_compat', __package__)
                boundary.install(core, family, Path(tide_path))
                compatibility.install(core)
                version = compatibility.ADAPTER_VERSION
            else:
                implementation = import_module('.seamount_runtime', __package__)
                version = implementation.install(core, family)
            core._case_adapter_forward = core.run_time_loop
            core.run_time_loop = types.MethodType(_scoped_forward, core)
            core._case_adapter_registration = {'case': case, 'family': family,
                'version': version, 'geometry': geometry, 'clock_seconds': clock,
                'registered_geometry_and_clock': registered, 'forward_validation_only': True,
                'registered_optional_filters': filters_disabled,
                'filter_mode': filter_mode if case in ('double_gyre', 'jiaozhou_bay') else 'native',
                'tide_forcing_sha256': tide_sha,
                'registered_tide_forcing': (tide_sha == expected_tide_sha) if expected_tide_sha is not None else None,
                'source_manifest': str(Path(__file__).with_name('source_manifest.json'))}
        finally:
            torch.set_default_dtype(previous)
    return core
