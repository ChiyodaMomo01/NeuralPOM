"""Daily objectives; physical correction constraints are shared with sparse_coupling."""
import torch

from neuralpom.hybrid.sparse_coupling import (
    LEVELS, pressure_velocity_error, volume_weights, weighted_mse,
)


def adjacent_error_moments(core, reference):
    """Four (squared error sum, weight sum) pairs, in T-x/T-y/S-x/S-y order.

    This is an index-space adjacent difference, not a physical gradient.
    Both cells must be wet and strictly inside the outer domain boundary.
    Area and sigma thickness weights do not depend on predicted surface height.
    """
    area = core.art[1:-1, 1:-1]
    wet = core.fsm[1:-1, 1:-1] > .5
    dz = core.dz[:LEVELS]
    pairs = []
    for field in ('t', 's'):
        error = (getattr(core, field) - reference[field])[1:-1, 1:-1, :LEVELS]
        for axis in (0, 1):
            left = [slice(None), slice(None)]
            right = left.copy()
            left[axis] = slice(None, -1)
            right[axis] = slice(1, None)
            left, right = tuple(left), tuple(right)
            weights = (.5 * (area[left] + area[right]) * (wet[left] & wet[right]))[..., None] * dz
            difference = error[right] - error[left]
            pairs.append(torch.stack(((difference.square() * weights).sum(), weights.sum())))
    return torch.stack(pairs)


def state_metrics(core, reference):
    out = {f: weighted_mse(getattr(core, f)[..., :LEVELS], reference[f][..., :LEVELS],
                           volume_weights(core, f)) for f in ('u', 'ub', 'v', 'vb', 't', 's')}
    out['el'] = weighted_mse(core.el, reference['el'], core.art * core.fsm)
    ku, kv = pressure_velocity_error(core, reference['t'], reference['s'])
    out['pressure'] = .5 * (weighted_mse(ku, torch.zeros_like(ku), volume_weights(core, 'u')) +
                             weighted_mse(kv, torch.zeros_like(kv), volume_weights(core, 'v')))
    return out


def daily_objective(core, reference, scales, front_rms, include_eta=True):
    metrics = state_metrics(core, reference)
    n = {f: value / max(scales[f] ** 2, 1e-16) for f, value in metrics.items()}
    momentum = torch.stack([n[f] for f in ('u', 'ub', 'v', 'vb')]).mean()
    moments = adjacent_error_moments(core, reference)
    front_components = moments[:, 0] / moments[:, 1].clamp_min(1e-20) / front_rms.square()
    front = front_components.mean()
    loss = .25 * momentum + .25 * n['t'] + .25 * n['s'] + .15 * n['pressure'] + .10 * front
    if include_eta:
        loss = loss + .10 * n['el']
    details = {**{'mse/' + k: v for k, v in metrics.items()},
               **{'normalized/' + k: v for k, v in n.items()},
               'normalized/momentum': momentum, 'normalized/front_ts': front,
               **{'front/' + k: v for k, v in zip(('t_x', 't_y', 's_x', 's_y'), front_components)},
               'state_loss': loss}
    return loss, details
