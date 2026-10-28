"""Strict-order vertical tracer solve; optional Triton forward, eager adjoint.

The public signature is ``strict_proft(z, *proft_args)``. CUDA float32 inputs
use one column-wise Thomas kernel with rounded division and FMA disabled.
CPU inputs exercise the same custom-autograd interface with the original
Torch forward. Neither path mutates an input. This module does not select a
backend on a live core; callers must explicitly integrate and validate it.
"""
import torch
from torch.autograd.function import once_differentiable

from .tracers import TracersMixin

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU installations do not need Triton.
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _column_solve(F, WF, FS, H, ETF, DZ, DZZ, KH, RAD,
                      OUT, DH, A, C, EE, GG,
                      N: tl.constexpr, KB: tl.constexpr, NBC: tl.constexpr,
                      DTI2: tl.constexpr, UMOL: tl.constexpr,
                      BLOCK: tl.constexpr):
        col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = col < N
        base = col * KB
        dh = tl.load(H + col, valid, 0.) + tl.load(ETF + col, valid, 0.)
        dh2 = dh * dh
        tl.store(DH + col, dh, valid)
        for k in tl.static_range(KB):
            tl.store(OUT + base + k, tl.load(F + base + k, valid, 0.), valid)
            tl.store(A + base + k, 0., valid)
            tl.store(C + base + k, 0., valid)
            tl.store(EE + base + k, 0., valid)
            tl.store(GG + base + k, 0., valid)
        for k in tl.static_range(1, KB - 1):
            num = (-DTI2) * (tl.load(KH + base + k, valid, 0.) + UMOL)
            dzz = tl.load(DZZ + k - 1)
            den_a = (tl.load(DZ + k - 1) * dzz) * dh2
            den_c = (tl.load(DZ + k) * dzz) * dh2
            tl.store(A + base + k - 1, tl.div_rn(num, den_a), valid)
            tl.store(C + base + k, tl.div_rn(num, den_c), valid)
        a0 = tl.load(A + base, valid, 0.)
        dz0 = tl.load(DZ)
        if NBC == 1:
            denom = a0 - 1.
            e_prev = tl.div_rn(a0, denom)
            flux = tl.div_rn((-DTI2) * tl.load(WF + col, valid, 0.), (-dz0) * dh)
            g_prev = tl.div_rn(flux - tl.load(F + base, valid, 0.), denom)
        elif NBC == 2:
            denom = a0 - 1.
            e_prev = tl.div_rn(a0, denom)
            rad_diff = tl.load(RAD + base, valid, 0.) - tl.load(RAD + base + 1, valid, 0.)
            flux = tl.div_rn(DTI2 * (tl.load(WF + col, valid, 0.) + rad_diff), dz0 * dh)
            g_prev = tl.div_rn(flux - tl.load(F + base, valid, 0.), denom)
        else:
            e_prev = tl.full((BLOCK,), 0., tl.float32)
            g_prev = tl.load(FS + col, valid, 0.)
        tl.store(EE + base, e_prev, valid)
        tl.store(GG + base, g_prev, valid)
        for k in tl.static_range(1, KB - 2):
            ak = tl.load(A + base + k, valid, 0.)
            ck = tl.load(C + base + k, valid, 0.)
            denom = (ak + ck * (1. - e_prev)) - 1.
            inverse = tl.div_rn(1., denom + 1.e-20)
            e_prev = ak * inverse
            if NBC == 2 or NBC == 4:
                rad_diff = tl.load(RAD + base + k, valid, 0.) - tl.load(RAD + base + k + 1, valid, 0.)
            else:
                rad_diff = tl.full((BLOCK,), 0., tl.float32)
            radiation = tl.div_rn(DTI2 * rad_diff, dh * tl.load(DZ + k))
            source = (ck * g_prev - tl.load(F + base + k, valid, 0.)) + radiation
            g_prev = source * inverse
            tl.store(EE + base + k, e_prev, valid)
            tl.store(GG + base + k, g_prev, valid)
        k_bot: tl.constexpr = KB - 2
        ck = tl.load(C + base + k_bot, valid, 0.)
        if NBC == 2 or NBC == 4:
            rad_diff = tl.load(RAD + base + k_bot, valid, 0.) - tl.load(RAD + base + k_bot + 1, valid, 0.)
        else:
            rad_diff = tl.full((BLOCK,), 0., tl.float32)
        radiation = tl.div_rn(DTI2 * rad_diff, dh * tl.load(DZ + k_bot))
        numerator = (ck * g_prev - tl.load(F + base + k_bot, valid, 0.)) + radiation
        denom = ck * (1. - e_prev) - 1.
        f_next = tl.div_rn(numerator, denom + 1.e-20)
        tl.store(OUT + base + k_bot, f_next, valid)
        for index in tl.static_range(KB - 2):
            k = KB - 3 - index
            f_next = tl.load(EE + base + k, valid, 0.) * f_next + tl.load(GG + base + k, valid, 0.)
            tl.store(OUT + base + k, f_next, valid)


class _TracerContext:
    def __init__(self, z, device):
        self.z, self.device = z, device


def _reference(inputs, settings):
    z, f, wfsurf, fsurf, h, etf, dz, dzz, swrad, kh = inputs
    nbc, dti2, ntp, umol = settings
    im, jm, kb = f.shape
    # proft replaces these scratch arguments before reading any of them.
    result = TracersMixin.proft(_TracerContext(z, f.device),
        f, wfsurf, fsurf, nbc, None, None, None, None, None,
        h, etf, dti2, dz, dzz, swrad, ntp, im, jm, kb, kb-1, kb-2, kh, umol)
    return tuple(result[index] for index in (0, 4, 5, 6, 7, 8))


def _forward(inputs, settings):
    z, f, wfsurf, fsurf, h, etf, dz, dzz, swrad, kh = inputs
    nbc, dti2, ntp, umol = settings
    if f.device.type == 'cpu':
        return _reference(inputs, settings)
    if f.device.type != 'cuda' or triton is None:
        raise RuntimeError('Strict proft CUDA forward requires Triton')
    # Keep Torch's exp and scalar-division rounding for radiative BCs. The
    # production nbct=nbcs=1 path needs no radiation field or extra launch.
    if nbc in (2, 4):
        depth = h + etf
        r = (0.58, 0.62, 0.67, 0.77, 0.78)[ntp-1]
        ad1 = (0.35, 0.60, 1.00, 1.50, 1.40)[ntp-1]
        ad2 = (0.23, 20., 17., 14., 7.90)[ntp-1]
        rad = torch.zeros_like(f)
        for k in range(f.shape[2]-1):
            z_term = z[k] * depth
            rad[..., k] = swrad * (r * torch.exp(z_term / ad1) + (1.-r) * torch.exp(z_term / ad2))
    else:
        rad = f  # constexpr branch never dereferences RAD for flux-only BCs.
    output = (torch.empty_like(f), torch.empty_like(h), *(torch.empty_like(f) for _ in range(4)))
    _column_solve[(triton.cdiv(h.numel(), 128),)](
        f, wfsurf, fsurf, h, etf, dz, dzz, kh, rad, *output,
        h.numel(), f.shape[2], nbc, dti2, umol, 128,
        enable_fp_fusion=False, num_warps=4)
    return output


class _StrictProft(torch.autograd.Function):
    @staticmethod
    def forward(ctx, settings, *inputs):
        ctx.settings = settings
        ctx.save_for_backward(*inputs)
        ctx.set_materialize_grads(False)
        return _forward(inputs, settings)

    @staticmethod
    @once_differentiable
    def backward(ctx, *output_gradients):
        # The eager reference preserves every input path, including depth,
        # diffusivity, surface forcing and sigma geometry. No core is captured.
        with torch.enable_grad():
            compiled_key = getattr(ctx, 'compiled_adjoint_key', None)
            inputs = [value.detach().requires_grad_(needed or compiled_key is not None)
                      for value, needed in zip(ctx.saved_tensors, ctx.needs_input_grad[1:])]
            outputs = (_COMPILED_ADJOINT[compiled_key](*inputs) if compiled_key is not None
                       else _reference(inputs, ctx.settings))
            if compiled_key is not None:
                # One warmed cotangent layout, including unused scratch outputs.
                output_gradients = tuple(torch.zeros_like(value) if gradient is None
                                         else gradient.contiguous()
                                         for value, gradient in zip(outputs, output_gradients))
            active_outputs = [(value, gradient) for value, gradient in zip(outputs, output_gradients)
                              if value.requires_grad and gradient is not None]
            positions = [i for i, value in enumerate(inputs) if value.requires_grad]
            result = [None] * len(inputs)
            if active_outputs and positions:
                gradients = torch.autograd.grad(
                    [value for value, _ in active_outputs], [inputs[i] for i in positions],
                    [gradient for _, gradient in active_outputs], allow_unused=True)
                for index, gradient in zip(positions, gradients):
                    result[index] = gradient if ctx.needs_input_grad[index + 1] else None
        return (None, *result)


_COMPILED_ADJOINT = {}


def _adjoint_key(inputs, settings):
    return (settings, tuple((tuple(value.shape), value.dtype, value.device) for value in inputs))


class _StrictCompiledAdjoint(_StrictProft):
    @staticmethod
    def forward(ctx, settings, *inputs):
        key = _adjoint_key(inputs, settings)
        if key not in _COMPILED_ADJOINT:
            raise RuntimeError('Call warmup_strict_proft_backward outside gradient checkpoint first')
        ctx.compiled_adjoint_key = key
        return _StrictProft.forward(ctx, settings, *inputs)


def _invoke_proft(autograd_function, z, f, wfsurf, fsurf, nbc, dh, a, c, ee, gg,
                 h, etf, dti2, dz, dzz, swrad, ntp, im, jm, kb, kbm1, kbm2, kh, umol):
    if nbc not in (1, 2, 3, 4) or ntp not in range(1, 6):
        raise ValueError('Invalid tracer surface boundary or irradiance type')
    if (im, jm, kb) != tuple(f.shape) or kb < 3 or (kbm1, kbm2) != (kb-1, kb-2):
        raise ValueError('Strict proft grid dimensions disagree')
    if not isinstance(dti2, (float, int)) or not isinstance(umol, (float, int)):
        raise TypeError('Strict proft numerical settings must be Python scalars')
    inputs = (z, f, wfsurf, fsurf, h, etf, dz, dzz, swrad, kh)
    shapes = ((kb,), (im,jm,kb), *((im,jm),)*4, (kb,), (kb,), (im,jm), (im,jm,kb))
    for value, shape in zip(inputs, shapes):
        if tuple(value.shape) != shape or value.device != f.device or value.dtype != f.dtype:
            raise ValueError('Strict proft tensor shape, dtype or device disagrees')
    if f.dtype not in (torch.float32, torch.float64) or (f.is_cuda and f.dtype != torch.float32):
        raise TypeError('Strict proft CUDA path requires float32')
    result = autograd_function.apply((nbc, dti2, int(ntp), umol), *(value.contiguous() for value in inputs))
    return result[0], wfsurf, fsurf, nbc, *result[1:]


def strict_proft(z, *args, **kwargs):
    """Default strict forward and eager recomputation, with full first derivatives.

    Accepts the original proft arguments after explicit sigma coordinates z.
    Numerical settings remain Python scalars. CUDA requires float32; CPU also
    supports float64 finite differences. Higher derivatives are unsupported.
    """
    return _invoke_proft(_StrictProft, z, *args, **kwargs)


def strict_proft_compiled_backward(z, *args, **kwargs):
    """Experimental strict forward plus externally warmed compiled recomputation.

    Only explicit strict_vertical_compiled mode selects this operator; the
    default strict_proft retains its eager Torch backward.
    Every distinct grid/device/dtype/boundary/settings signature must first be
    passed to warmup_strict_proft_backward outside any gradient checkpoint.
    """
    return _invoke_proft(_StrictCompiledAdjoint, z, *args, **kwargs)


def is_strict_proft_backward_warm(z, f, wfsurf, fsurf, nbc, dh, a, c, ee, gg,
                                  h, etf, dti2, dz, dzz, swrad, ntp,
                                  im, jm, kb, kbm1, kbm2, kh, umol):
    """Read only metadata; no tensor copies, compilation, or GPU work."""
    inputs = (z, f, wfsurf, fsurf, h, etf, dz, dzz, swrad, kh)
    return _adjoint_key(inputs, (nbc, dti2, int(ntp), umol)) in _COMPILED_ADJOINT


def warmup_strict_proft_backward(z, *args, backend='inductor', **kwargs):
    """Compile the complete reference forward/adjoint before checkpoint execution.

    Warm all actual input derivatives and all six output cotangents together.
    Runtime input requires_grad subsets do not create new AOT signatures. No
    physical state is mutated and no input .grad is populated. aot_eager is
    useful for CPU interface tests; performance validation uses inductor.
    """
    class Prepare:
        @staticmethod
        def apply(settings, *inputs):
            key = _adjoint_key(inputs, settings)
            if key in _COMPILED_ADJOINT:
                raise ValueError('Strict proft adjoint signature is already prepared')
            def reference(*values):
                return _reference(values, settings)
            options = {'triton.cudagraphs': False} if backend == 'inductor' else None
            compiled = torch.compile(reference, backend=backend, fullgraph=True,
                                     dynamic=False, options=options)
            with torch.enable_grad():
                leaves = [value.detach().clone().requires_grad_(True) for value in inputs]
                outputs = compiled(*leaves)
                active = [value for value in outputs if value.requires_grad]
                torch.autograd.grad(active, leaves, [torch.ones_like(value) for value in active],
                                    allow_unused=True)
            _COMPILED_ADJOINT[key] = compiled
            return tuple(value.detach() for value in outputs)
    _invoke_proft(Prepare, z, *args, **kwargs)
