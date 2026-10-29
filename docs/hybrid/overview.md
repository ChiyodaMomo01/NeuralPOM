# Learning inside an ocean trajectory

**The physical model remains responsible for advancing the ocean.** NeuralPOM learns endpoint increments in velocity, temperature, and salinity and applies them through constraints that preserve the model's transport and tracer budgets. The next physical interval starts from this corrected numerical state. ([Manuscript, Section 2.2](../paper.md))

## One forecast cycle

Let $n$ index correction endpoints, $x_n^+$ be the complete state after correction, and $\Delta$ the correction interval. Let $\Phi_\Delta$ advance the physical model, $\mathcal N$ normalize its network inputs, $f_\theta$ be the network with parameters $\theta$, and $\mathcal C$ decode and apply a constrained correction. A superscript minus denotes the endpoint before correction:

$$
x_{n+1}^-=\Phi_\Delta(x_n^+),\qquad
x_{n+1}^+=\mathcal C\!\left(x_{n+1}^-,f_\theta(\mathcal N(x_{n+1}^-))\right).
$$

The published configurations use $\Delta=1$ hour or $1$ day. At a 600 s internal step, those intervals contain 6 or 144 physical steps. The [Explicit and semi-implicit cores](../cores/overview.md) share this coupling interface.

## What the correction preserves

For active vertical layer $k$, let $\Delta\sigma_k$ be its fractional thickness. For horizontal cell $(i,j)$, let $w_{ijk}$ be the wet cell volume used by the correction. The constrained increments $\delta u$, $\delta v$, $\delta T$, and $\delta S$ satisfy

$$
\sum_k\Delta\sigma_k\,\delta u_{ijk}=0,
\qquad \sum_k\Delta\sigma_k\,\delta v_{ijk}=0,
$$

$$
\sum_{ij}w_{ijk}\,\delta T_{ijk}=0,
\qquad \sum_{ij}w_{ijk}\,\delta S_{ijk}=0.
$$

The velocity constraints let the network change vertical shear while keeping the depth-integrated transport under the physical external-mode equations. The tracer constraints redistribute heat and salt horizontally within a layer without adding a net layer source.

A common limiting factor for each variable pair caps standardized increments while preserving these zero-mean constraints. The same increment is added to the current and backward leapfrog fields. Density is then refreshed from the nonlinear equation of state. Sea-surface elevation, turbulence variables, and other physical fields evolve during the next physical integration. ([Correction implementation](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/src/neuralpom/hybrid/sparse_coupling.py))

## From architecture to deployment

| Component | Scientific role | Read next |
|---|---|---|
| Residual U-Net | Predict endpoint increments from standardized physical fields | [CorrectorNet](./corrector-net.md) |
| Staged optimization | Learn endpoint accuracy and the consequences of corrections | [Training](./training.md) |
| Density-normal damping | Attenuate the constrained density-modifying tracer projection at inference | [DND](./density-normal-damping.md) |
| Frozen forecast | Advance corrected states with fixed network weights | [Reproduction](../reproduction.md) |

<figure><img src="/figures/method_flow.png" alt="NeuralPOM method diagram from the manuscript, showing physical integration, constrained correction, and staged learning" loading="lazy"><figcaption>Method diagram reproduced from the repository manuscript. The detailed numerical protocol is given in its Section 2 and Appendix B.</figcaption></figure>
