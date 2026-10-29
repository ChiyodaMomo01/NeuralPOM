# The semi-implicit free-surface solve

**The semi-implicit core couples surface elevation and depth-averaged transport through an elliptic equation.** Its forward solve uses preconditioned conjugate gradients (PCG), with a discrete sine transform preconditioner in the published configuration. Its backward pass differentiates the solved system implicitly. ([Manuscript, Section 2.1](../paper.md))

## From a surface-pressure response to a linear system

Let $\eta$ be the unknown surface elevation, $D$ the depth entering the operator, $\nabla_h$ the horizontal gradient, and $\kappa$ the positive coefficient determined by the time discretization. With $b$ collecting known forcing, history, and boundary contributions, the operator has the form

$$
A(D)\eta=b,\qquad
A(D)\eta=\eta-\kappa\,\nabla_h\cdot(D\nabla_h\eta).
$$

The C-grid implementation evaluates face fluxes and their divergence with the grid metrics and masks. This connects adjacent surface cells without explicitly assembling a dense matrix. Exact coefficients and boundary handling are defined in [`implicit_barotropic.py`](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/src/neuralpom/cores/implicit_p0_fixed/implicit_barotropic.py).

## Forward solve and preconditioning

PCG repeatedly applies the discrete operator and a preconditioner to reduce the residual. The preconditioner accelerates convergence; the physical operator still defines the solved equation. The portable semi-implicit configurations select the registered Triton backend and compiled vertical operators. The published coarse physical step is **600 s**. ([Core configuration](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/src/neuralpom/v1/core_spec.py), [portable examples](../guide/configuration.md))

## Differentiate the solution

Let $\mathcal L$ be a scalar loss, $\lambda$ an adjoint vector, and $A^\top$ the transpose of the discrete linear operator under the chosen algebraic representation. Given the loss derivative with respect to the solved elevation, the adjoint system is

$$
A^\top\lambda=\frac{\partial\mathcal L}{\partial\eta}.
$$

For variations $\mathrm db$ and $\mathrm dA$, the corresponding loss variation is

$$
\mathrm d\mathcal L=\lambda^\top(\mathrm db-\mathrm dA\,\eta).
$$

This relation propagates sensitivity through the solved equation. The production backward operator includes its geometry and boundary conventions; it is implemented in [`autograd.py`](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/src/neuralpom/cores/implicit_p0_fixed/autograd.py). The sea-surface adjoint checks are in [`tests/test_fixed_core_runtime.py`](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/tests/test_fixed_core_runtime.py).

## Why Midtrain needs this derivative

An earlier correction changes the state entering the next physical interval. The free-surface solve is one part of that physical response. Midtrain uses its derivative to connect the later loss back to the earlier network parameters. See [Staged training](../hybrid/training.md) for the complete gradient path.
