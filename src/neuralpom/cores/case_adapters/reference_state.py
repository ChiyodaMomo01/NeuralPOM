"""Initialise the common-state contract used by the registered comparisons."""
from pathlib import Path
from collections.abc import Mapping
import numpy as np
import torch

from .registry import _DTYPE_LOCK, apply_case_adaptation

_SC = 'time dte dti dti2 grav rhoref tbias sbias hmax kappa tprni umol alpha small slmax'.split()
_F1 = 'z zz dz dzz'.split()
_F2 = 'h dx dy cor art aru arv fsm dum dvm east_e north_e ua uab va vab el elb et etb etf egb utb vtb d dt wusurf wvsurf wtsurf wssurf swrad vfluxf cbc aam2d advua advva adx2d ady2d drx2d dry2d wubot wvbot'.split()
_F3 = 'u ub v vb w t tb s sb rho rmean tclim sclim km kh kq l q2 q2b q2l q2lb aam drhox drhoy'.split()


def core_from_reference(initial_state, case, family, *, device=None,
                        tide_path=None, filter_mode='matched', strict=True):
    """Build a case core from an exported full Fortran initial-state NPZ/mapping.

    This is the registered physical-case initialisation, including history
    layers and the dummy bottom layer. It is not a production restart loader.
    """
    if case == 'jzb_tide':
        case = 'jiaozhou_bay'
    if family not in ('explicit', 'implicit'):
        raise ValueError(f'Unsupported family: {family}')
    if isinstance(initial_state, (str, Path)):
        with np.load(initial_state) as stream:
            state = {key: stream[key].copy() for key in stream.files}
    elif isinstance(initial_state, Mapping):
        state = {key: np.asarray(value) for key, value in initial_state.items()}
    else:
        raise TypeError('initial_state must be an NPZ path or a full-state mapping')
    required = _SC + _F1 + _F2 + _F3
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f'Missing reference state/history fields: {missing}')
    if float(state.get('time', 0.)) != 0.:
        raise ValueError('The registered initial-state contract starts at time zero')
    im, jm = state['h'].shape
    kb = len(state['z'])
    for fields, shape in ((_SC, ()), (_F1, (kb,)), (_F2, (im, jm)), (_F3, (im, jm, kb))):
        for key in fields:
            if state[key].shape != shape or not np.isfinite(state[key]).all():
                raise ValueError(f'Invalid common-reference field shape/finite values: {key}')
    with _DTYPE_LOCK:
        previous = torch.get_default_dtype()
        try:
            from neuralpom.cores.explicit import ExplicitPOM2KCore
            from neuralpom.cores.implicit_p0_fixed import POM2K_Core
            from neuralpom.cores.implicit_p0_fixed.forcing.surface import SurfaceHeatSaltFluxForcing
            core = (ExplicitPOM2KCore if family == 'explicit' else POM2K_Core)(im, jm)
            # The registered comparisons disable optional spatial filters.
            # A constructor's ambient environment must not silently alter this
            # explicitly selected common-reference initialisation protocol.
            if family == 'implicit':
                core.spatial_shapiro_enabled = False
                core.barotropic_shapiro = core.tracer_shapiro = core.velocity_shapiro = 0.
            core.device = torch.device(device) if device is not None else core.device
            for key, value in vars(core).copy().items():
                if not torch.is_tensor(value):
                    continue
                if value.is_floating_point():
                    shape = list(value.shape)
                    if len(shape) == 3 and shape[:2] == [im, jm]:
                        shape[-1] = kb
                    elif len(shape) == 2 and shape[-1] == 16 and shape[0] in (im, jm):
                        shape[-1] = kb
                    elif len(shape) == 1 and shape[0] == 16:
                        shape[0] = kb
                    value = value.to(core.device, torch.float64) if shape == list(value.shape) else torch.zeros(shape, device=core.device, dtype=torch.float64)
                setattr(core, key, value.to(core.device))
            core.kb, core.kbm1, core.kbm2 = kb, kb - 1, kb - 2
            for key, value in state.items():
                setattr(core, key, torch.as_tensor(value, device=core.device, dtype=torch.float64).clone() if value.shape else float(value))
            core.isplit = round(core.dti / core.dte)
            core.dte2 = 2 * core.dte
            core.ispi, core.isp2i = 1 / core.isplit, 1 / (2 * core.isplit)
            core.mode = 4 if case in ('coastal_tide', 'jiaozhou_bay') else 3
            core.lramp, core.ramp = (True, 0.) if case == 'jiaozhou_bay' else (False, 1.)
            core.horcon, core.aam_init = (.1, 50.) if case == 'jiaozhou_bay' else (.05, 500.)
            core.z0b, core.cbcmin, core.cbcmax = (.01 if case == 'jiaozhou_bay' else .005), .0025, 1.
            core.smoth, core.period, core.nadv = .1, 3., 1
            core.time0, core.time = 0., 0.
            core.prtd1 = core.prtd2 = core.swtch = 1000.
            core.iprint = core.iswtch = 999999
            core.runtime_no_grad_fast_path = True
            core.runtime_vamax_safe_fast_path = True
            core.tsurf, core.ssurf = core.t[:, :, 0].clone(), core.s[:, :, 0].clone()
            core._implicit_forcing_provider = SurfaceHeatSaltFluxForcing()
            core.iend = 1
            return apply_case_adaptation(core, case, family, tide_path=tide_path,
                                         filter_mode=filter_mode, strict=strict)
        finally:
            torch.set_default_dtype(previous)
