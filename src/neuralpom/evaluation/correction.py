"""Ablate neural constraints while preserving endpoint/history/EOS semantics."""
import torch

from neuralpom.hybrid.state import pack_pom_batch
from neuralpom.hybrid.sparse_coupling import LEVELS, OUTPUT_FIELDS, eos, apply_joint_correction


def apply_raw(core, raw, scales):
    """No projection, mask, clipping, budget correction, or spectral filtering.

    Training's channel scales decode units. Paired history writes and EOS
    recomputation are the common state interface, not stabilization operations.
    Native physical boundary conditions run normally on the next physical step.
    """
    if raw.shape != (4 * LEVELS, core.im, core.jm):
        raise ValueError('Unexpected corrector output shape')
    for index, (field, history) in enumerate(zip(OUTPUT_FIELDS, ('ub', 'vb', 'tb', 'sb'))):
        delta = raw[index * LEVELS:(index + 1) * LEVELS].permute(1, 2, 0) * scales[field]
        for name in (field, history):
            previous = getattr(core, name)
            setattr(core, name, torch.cat((previous[..., :LEVELS] + delta, previous[..., LEVELS:]), -1))
    core.rho = eos(core, core.t, core.s)
    return {'ts_limit_factor': torch.ones((), device=raw.device)}


@torch.no_grad()
def correct(runtime, model, policy):
    raw = model(pack_pom_batch([runtime.core], runtime.normalizer).to(runtime.device))[0]
    if not bool(torch.isfinite(raw).all()):
        raise FloatingPointError('Nonfinite neural output')
    if policy == 'raw':
        result = apply_raw(runtime.core, raw, runtime.correction_scales)
    elif policy == 'constrained':
        result = apply_joint_correction(runtime.core, raw, runtime.correction_scales, cap=5.)
    elif policy == 'density75':
        from .density import apply_density75
        result = apply_density75(runtime.core, raw, runtime.correction_scales)
    else:
        raise ValueError(f'Unknown neural correction policy {policy}')
    return {key: float(value) for key, value in result.items()}
