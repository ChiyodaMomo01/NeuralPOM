"""Fixed-core advancement with optional fusion and full-state checkpointing."""
import copy

import torch
from torch.utils.checkpoint import checkpoint

from neuralpom.cores.implicit_p0_fixed import POM2K_Core
from neuralpom.cores.explicit import POM2K_Core as ExplicitCore
from neuralpom.cores.implicit_p0_fixed.gradient_kernels import (
    COMPILED_KERNEL_NAMES,
    CompiledVerticalCore,
    StrictVerticalCore,
    StrictCompiledVerticalCore,
)
from neuralpom.hybrid.compile_prewarm import prewarm_forward_backward, prewarm_strict_vertical


class SparsePhysics:
    def __init__(self, device, rank=0, cuda_graph=False, chunk_steps=6,
                 acceleration="eager", checkpoint_physics=True, cg_backend="torch",
                 compiled_kernels=("proft",), core_name="implicit_p0_fixed"):
        if core_name not in ("implicit_p0_fixed", "explicit"):
            raise ValueError(f"Unsupported physics core: {core_name}")
        if core_name == "explicit" and (acceleration != "eager" or cg_backend != "torch" or
                                        cuda_graph or compiled_kernels):
            raise ValueError("Explicit physics requires eager mode without implicit solver or kernels")
        if acceleration not in {"eager", "compiled_vertical", "strict_vertical", "strict_vertical_compiled"}:
            raise ValueError(f"Unknown physics acceleration: {acceleration}")
        if acceleration == "strict_vertical_compiled" and torch.device(device).type != "cuda":
            raise ValueError("Compiled strict tracer adjoints require CUDA")
        compiled_kernels = tuple(compiled_kernels)
        if acceleration in {"strict_vertical", "strict_vertical_compiled"} and compiled_kernels != ("proft",):
            raise ValueError("Strict vertical acceleration supports only proft")
        if core_name == "implicit_p0_fixed" and ((not compiled_kernels and acceleration != 'eager') or not set(compiled_kernels) <= set(COMPILED_KERNEL_NAMES)):
            raise ValueError("Compiled kernels must be a nonempty subset of profq/proft")
        if cg_backend not in {"torch", "triton"}:
            raise ValueError(f"Unknown CG operator backend: {cg_backend}")
        if cg_backend == "triton":
            from neuralpom.cores.implicit_p0_fixed.triton_kernels import triton_available
            if torch.device(device).type != "cuda" or not triton_available():
                raise ValueError("Triton CG requires CUDA and Triton")
        if cuda_graph:
            raise ValueError("Use compiled_vertical; the legacy inference CUDA graph is archived")
        if chunk_steps < 1:
            raise ValueError("chunk_steps must be positive")
        self.device = torch.device(device)
        self.global_rank = rank
        self.core_name = core_name
        self.core_cls = ExplicitCore if core_name == "explicit" else POM2K_Core
        self.chunk_steps = int(chunk_steps)
        self.acceleration = acceleration
        self.cg_backend = cg_backend
        self.compiled_kernels = compiled_kernels
        self.checkpoint_physics = checkpoint_physics
        self._compiled_chunk_signatures = set()

    @staticmethod
    def release_completed_rollout(core):
        """Detach after backward, at a TBPTT or completed-batch boundary."""
        for name, value in list(vars(core).items()):
            if torch.is_tensor(value) and (value.requires_grad or value.grad_fn is not None):
                setattr(core, name, value.detach())

    def _configure(self, core):
        if self.core_name == "explicit":
            if not isinstance(core, ExplicitCore):
                raise TypeError("Explicit physics requires the Explicit V2 core")
            core.runtime_no_grad_fast_path = False
            core.generation_fast_path = False
            core.runtime_cuda_graph_fast_path = False
            core.runtime_vamax_safe_fast_path = False
            core.runtime_ramp_saturated_fast_path = False
            return
        if not isinstance(core, POM2K_Core):
            raise TypeError("SparsePhysics requires implicit_p0_fixed")
        core.__class__ = {"eager": POM2K_Core, "compiled_vertical": CompiledVerticalCore,
                          "strict_vertical": StrictVerticalCore,
                          "strict_vertical_compiled": StrictCompiledVerticalCore}[self.acceleration]
        core._gradient_kernel_names = self.compiled_kernels
        core.use_triton_elliptic_operator = self.cg_backend == "triton"
        core.use_triton_cg_updates = False
        core.compile_eta_solver = False
        core.generation_fast_path = False
        core.runtime_diagnostics = False
        # Keep the same converged CG algorithm on both eager/compiled paths.
        core.runtime_no_grad_fast_path = False
        core.runtime_cuda_graph_fast_path = False
        core.runtime_vamax_safe_fast_path = False
        core.runtime_ramp_saturated_fast_path = False

    def advance(self, core, steps=6, differentiable=False):
        steps = int(steps)
        if steps < 1:
            raise ValueError("steps must be positive")
        self._configure(core)
        if differentiable and self.acceleration == "strict_vertical_compiled":
            prewarm_strict_vertical(core)
        if not differentiable:
            core.time0 = float(core.time)
            core.iint = 0
            core.iend = steps
            with torch.no_grad():
                core.run_time_loop()
            return
        for start in range(0, steps, self.chunk_steps):
            self._checkpoint_chunk(core, min(self.chunk_steps, steps - start))

    def _checkpoint_chunk(self, core, steps):
        names = tuple(sorted(k for k, v in vars(core).items() if torch.is_tensor(v)))
        values = tuple(getattr(core, k) for k in names)
        template = copy.copy(core)
        start = float(core.time)

        def run(*inputs):
            local = copy.copy(template)
            for name, value in zip(names, inputs):
                setattr(local, name, value.clone())
            local._cg_work = {}
            local._accel_cache = {}
            local._accel_cache_ready = False
            local._eta_solver_iteration_history = []
            local.time0 = start
            local.time = start
            local.iint = 0
            local.iend = steps
            local.run_time_loop()
            if local.iint != steps:
                raise RuntimeError("Physical core returned before the checkpoint chunk completed")
            return tuple(getattr(local, name) for name in names)

        if self.acceleration == "compiled_vertical" and self.checkpoint_physics:
            signature = (steps, self.compiled_kernels, tuple((name, tuple(v.shape), tuple(v.stride()),
                                      v.dtype, v.device, v.requires_grad)
                                     for name, v in zip(names, values)))
            if signature not in self._compiled_chunk_signatures:
                # AOT backward must compile before checkpoint saved-tensor hooks.
                prewarm_forward_backward(run, values)
                self._compiled_chunk_signatures.add(signature)
        if self.checkpoint_physics:
            outputs = checkpoint(run, *values, use_reentrant=False, preserve_rng_state=False)
        else:
            outputs = run(*values)
        for name, value in zip(names, outputs):
            setattr(core, name, value)
        core.time0 = start
        core.time = start + steps * core.dti / 86400.0
        core.iint = steps
        core.iend = steps
        core.ramp = min(core.time / float(core.period), 1.0) if core.lramp else 1.0
