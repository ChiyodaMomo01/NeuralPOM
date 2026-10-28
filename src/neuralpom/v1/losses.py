"""Composable endpoint objectives shared by both frequencies and all stages."""
from __future__ import annotations
import math
import torch
from neuralpom.training.daily_objective import daily_objective
from neuralpom.hybrid.sparse_coupling import LEVELS, volume_weights


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def spectral_loss(core, reference, floor=1e-12):
    """Windowed native-grid velocity energy per level, plus depth mean.

    Land contributes zero. No interpolation or phase alignment is introduced.
    The positive energy floor makes derivatives finite even at zero velocity.
    """
    floor = _positive(floor, 'spectral floor')
    dz = core.dz[:LEVELS]
    dz = dz / dz.sum()
    nx, ny = core.u.shape[:2]
    window = torch.hann_window(nx, periodic=False, device=core.u.device, dtype=core.u.dtype)[:, None] * torch.hann_window(ny, periodic=False, device=core.u.device, dtype=core.u.dtype)[None, :]
    energies = []
    for fields in ((core.u, core.v), (reference['u'], reference['v'])):
        components = []
        for field, wet in zip(fields, (core.dum, core.dvm)):
            values = field[..., :LEVELS].permute(2, 0, 1)
            values = torch.cat((values, (values * dz[:, None, None]).sum(0, keepdim=True)))
            weight = window * wet
            mean = (values * weight).sum((-2, -1), keepdim=True) / weight.sum().clamp_min(1e-20)
            fft = torch.fft.fft2((values - mean) * weight / weight.square().mean().sqrt().clamp_min(1e-20), norm='forward')
            components.append(fft.abs().square())
        energies.append(.5 * (components[0] + components[1]))
    predicted, target = energies
    return (torch.sqrt((predicted + floor) / (target.detach() + floor)) - 1).square().mean()


def physical_loss(core, reference, scales, include_eta=True):
    """Squared tracer-inventory and free-surface mean errors on common geometry."""
    terms = []
    for field in (('t', 's', 'el') if include_eta else ('t', 's')):
        weights = core.art * core.fsm if field == 'el' else volume_weights(core, field)
        predicted = core.el if field == 'el' else getattr(core, field)[..., :LEVELS]
        target = reference[field] if field == 'el' else reference[field][..., :LEVELS]
        mean_error = ((predicted - target) * weights).sum() / weights.sum().clamp_min(1e-20)
        terms.append((mean_error / _positive(scales[field], f'physical scale {field}')).square())
    return torch.stack(terms).mean()


class CompositeLoss:
    def __init__(self, config, scales, front_rms=None):
        self.scales = scales
        self.front_rms = front_rms
        if 'loss' in config:
            raise ValueError('V1 uses losses; the legacy loss key is not accepted')
        terms = config.get('losses', [{'name': 'weighted_mse', 'weight': 1.0}])
        if isinstance(terms, dict):
            terms = [{'name': name, **(value if isinstance(value, dict) else {'weight': value})} for name, value in terms.items()]
        if not terms:
            raise ValueError('At least one loss is required')
        self.terms = []
        seen = set()
        for term in terms:
            name = term['name']
            if name not in ('weighted_mse', 'spectral', 'physical'):
                raise ValueError(f'Unknown loss: {name}')
            if name in seen:
                raise ValueError(f'Duplicate loss component: {name}')
            seen.add(name)
            weight = float(term.get('weight', 1.0))
            if not math.isfinite(weight) or weight < 0:
                raise ValueError('Loss weights must be finite and nonnegative')
            self.terms.append((name, weight, _positive(term.get('scale', 1.), 'loss scale'), term))
        if not any(term[1] > 0 for term in self.terms):
            raise ValueError('At least one loss weight must be positive')

    def __call__(self, core, reference, include_eta=True):
        total = core.t.sum() * 0
        details = {}
        for name, weight, scale, term in self.terms:
            if weight == 0:
                continue
            if name == 'weighted_mse':
                if self.front_rms is None:
                    raise ValueError('Weighted MSE requires training-only front scales')
                front = torch.as_tensor(self.front_rms, device=core.t.device, dtype=core.t.dtype)
                if front.shape != (4,) or not bool(torch.isfinite(front).all()) or not bool((front > 0).all()):
                    raise ValueError('Invalid adjacent-difference normalization')
                value, components = daily_objective(core, reference, self.scales, front, include_eta)
                details.update(components)
            elif name == 'spectral':
                value = spectral_loss(core, reference, term.get('floor', 1e-12))
            else:
                value = physical_loss(core, reference, self.scales, include_eta=include_eta)
            contribution = weight * value / scale**2
            total = total + contribution
            details['loss/' + name] = value
            details['weighted_loss/' + name] = contribution
        details['loss/total'] = total
        return total, details


def build_loss(config, scales, front_rms=None):
    return CompositeLoss(config, scales, front_rms)
