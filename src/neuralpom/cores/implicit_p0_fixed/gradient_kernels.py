"""Optional fusion of existing vertical mixing equations on the autograd path.

No live core is captured by a compiler closure. This matters because checkpoint
chunks replace their tensor state, and eta adjoints outlive individual cores.
The original methods remain the single source of the discretization.
"""
from functools import lru_cache

import torch

from .core import POM2K_Core
from .tracers import TracersMixin


COMPILED_KERNEL_NAMES = ("profq", "proft")


class _TracerContext:
    def __init__(self, device, z=None):
        self.device = device
        self.z = z


def _profq(*args):
    return TracersMixin.profq(_TracerContext(args[0].device), *args)


def _proft(z, *args):
    return TracersMixin.proft(_TracerContext(args[0].device, z), *args)


@lru_cache(maxsize=2)
def _compiled(name):
    # Fail loudly if a complete kernel cannot be compiled. Never silently use
    # an eager fallback and label its timing as compiled. CUDA graphs stay off:
    # many saved activations coexist until the full-day backward completes.
    return torch.compile(
        {"profq": _profq, "proft": _proft}[name],
        backend="inductor", fullgraph=True, dynamic=False,
        options={"triton.cudagraphs": False},
    )


class CompiledVerticalCore(POM2K_Core):
    """Fuse vertical mixing in inference and full-BPTT physical advancement."""

    def profq(self, *args):
        if "profq" in self._gradient_kernel_names:
            return _compiled("profq")(*args)
        return super().profq(*args)

    def proft(self, *args):
        if "proft" in self._gradient_kernel_names:
            return _compiled("proft")(self.z, *args)
        return super().proft(*args)

    # Advection fusion alone reproduced the full-day tracer mismatch and had
    # only a 1.03x speedup. Keep advq/advt1 on their original path in every variant.


class StrictVerticalCore(POM2K_Core):
    """Strict-rounding fused tracer forward with the original Torch adjoint."""

    def proft(self, *args):
        from .strict_vertical import strict_proft
        return strict_proft(self.z, *args)


class StrictCompiledVerticalCore(POM2K_Core):
    """Strict tracer forward plus externally warmed Inductor first derivatives."""

    def proft(self, *args):
        from .strict_vertical import strict_proft, strict_proft_compiled_backward
        operator = strict_proft_compiled_backward if torch.is_grad_enabled() else strict_proft
        return operator(self.z, *args)
