"""Compile both autograd directions without touching a live rollout graph."""
import torch


def prewarm_forward_backward(run, values):
    # Tracing a non-leaf input can reach upstream saved-tensor hooks. Use fresh
    # leaves with identical metadata, so no model/earlier-checkpoint graph is
    # reachable. run() itself still clones every physical state tensor.
    inputs = tuple(value.detach().clone().requires_grad_(value.requires_grad)
                   for value in values)
    outputs = run(*inputs)
    differentiable_inputs = tuple(value for value in inputs if value.requires_grad)
    differentiable_outputs = tuple(value for value in outputs if value.requires_grad)
    if differentiable_inputs and differentiable_outputs:
        # AOT/Inductor may defer backward compilation until its first call. Do
        # that here, before the real physics checkpoint installs its hooks.
        # These gradients are discarded; model gradients and optimizer state
        # are never involved in this initialization.
        torch.autograd.grad(differentiable_outputs, differentiable_inputs,
                            grad_outputs=tuple(torch.ones_like(value)
                                               for value in differentiable_outputs),
                            allow_unused=True)


def prewarm_strict_vertical(core):
    """Prepare tracer VJPs from core metadata before installing checkpoint hooks."""
    from neuralpom.cores.implicit_p0_fixed.strict_vertical import (
        is_strict_proft_backward_warm, warmup_strict_proft_backward,
    )
    if any(value.requires_grad for value in (core.z, core.dz, core.dzz)):
        raise ValueError("Compiled tracer adjoints are validated for fixed sigma geometry")
    for field, flux, surface, boundary in ((core.t, core.wtsurf, core.tsurf, core.nbct),
                                            (core.s, core.wssurf, core.ssurf, core.nbcs)):
        args = (core.z, field, flux, surface, boundary, None, None, None, None, None,
                core.h, core.etf, core.dti2, core.dz, core.dzz, core.swrad, core.ntp,
                core.im, core.jm, core.kb, core.kb-1, core.kb-2, core.kh, core.umol)
        if not is_strict_proft_backward_warm(*args):
            warmup_strict_proft_backward(*args)
