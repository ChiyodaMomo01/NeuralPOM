# Fast surface waves and internal motion

**Mode splitting separates the depth-averaged flow from its vertical structure.** External gravity waves propagate rapidly, while the stratified interior evolves on longer time scales. NeuralPOM provides Explicit and semi-implicit treatments of the external mode within the same hydrostatic framework. ([Manuscript, Section 2.1](../paper.md))

## Depth average and vertical shear

Let $H$ be resting depth, $\eta$ surface elevation, $D=H+\eta$ total depth, and $\boldsymbol u_h(z)$ horizontal velocity. Define its depth average $\overline{\boldsymbol u}$ and departure $\boldsymbol u'$ by

$$
\overline{\boldsymbol u}=\frac1D\int_{-H}^{\eta}\boldsymbol u_h\,\mathrm dz,
\qquad \boldsymbol u'=\boldsymbol u_h-\overline{\boldsymbol u}.
$$

The external continuity equation, with zero net surface volume flux, is

$$
\partial_t\eta+\nabla_h\cdot(D\overline{\boldsymbol u})=0,
$$

where $\nabla_h\cdot$ is horizontal divergence. The internal mode represents the vertical shear, tracer evolution, and associated baroclinic pressure gradients. The modes exchange transport and forcing through the numerical coupling.

## Two treatments of the external mode

| Core | Fast-mode treatment | Published coarse time stepping |
|---|---|---|
| Explicit | Free-surface and depth-averaged updates subcycle within the internal step | 20 s external; 600 s internal |
| Semi-implicit | A coupled free-surface elliptic solve handles the fast pressure response | 600 s internal step |

The values describe the published coarse configurations. See [Time stepping](../numerics/time-stepping.md) and the [elliptic solver](../numerics/elliptic-solver.md) for their numerical roles. ([Manuscript, Section 2.1](../paper.md), [`v1/core_spec.py`](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/src/neuralpom/v1/core_spec.py))

## Why the correction preserves column transport

A learned velocity increment with zero thickness-weighted column mean changes shear without directly changing the depth average. This keeps the neural update compatible with the transport already governed by the external continuity equation. Temperature and salinity increments still affect subsequent circulation through density and baroclinic pressure gradients.

The [constrained correction](../hybrid/overview.md#what-the-correction-preserves) therefore connects a numerical requirement—consistent external and internal modes—to the scientific design of the hybrid model.
