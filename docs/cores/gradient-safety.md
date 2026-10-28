# Differentiation and restart

## Physical gradients

Midtrain differentiates through the physical intervals connecting corrected endpoints. The semi-implicit core supplies backward operators for its free-surface and vertical solves. Physical checkpointing recomputes integration chunks during backward while retaining the complete state required by each chunk.

The implementations are in `cores/implicit_p0_fixed/autograd.py`, `strict_vertical.py`, `gradient_kernels.py` and `hybrid/sparse_physics.py`. The Explicit core uses its PyTorch integration operators.

## Regression coverage

`tests/test_fixed_core_runtime.py` compares checkpointed and direct gradients, checks the sea-surface adjoint and verifies complete-state restart. `tests/test_strict_vertical.py` checks vertical operator gradients and finite differences. Training restart tests additionally preserve model, optimizer, sampler, numerical state and random-number state.

Run the suite with `python -m pytest` after the [development installation](../guide/installation.md).
