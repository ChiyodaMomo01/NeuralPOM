# Time steps, corrections, and history

**Three different clocks appear in NeuralPOM:** the external-mode substep, the internal physical step, and the neural correction interval. Keeping them separate makes training horizons and restart states easier to interpret. ([Manuscript, Sections 2.1–2.3](../paper.md))

| Clock | Published setting | Role |
|---|---|---|
| Explicit external step | 20 s | Resolve the fast surface mode through subcycling |
| Internal physical step | 600 s | Advance the three-dimensional physical state |
| Neural correction interval | 1 hour or 1 day | Apply the constrained network increment |

The Explicit core takes 30 external steps per internal step. Both correction schedules use the same 600 s coarse internal step; hourly and daily coupling correspond to 6 and 144 internal steps. The semi-implicit core uses a [free-surface solve](./elliptic-solver.md) for the fast-mode pressure response.

## Why the past state is part of the present model

Let $\phi$ be a prognostic field, $n$ the physical time index, $\Delta t$ the step, and $F$ its tendency. The elementary leapfrog update is

$$
\phi^{n+1}=\phi^{n-1}+2\Delta t\,F(\phi^n).
$$

This illustrates why a current field alone is insufficient to restart a multilevel solver. POM also applies time filtering and mode coupling. Fields with a `b` suffix store backward levels, and fields with an `f` suffix commonly store forward values during a step. See the core's [`time_loop.py`](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/src/neuralpom/cores/explicit/time_loop.py).

## Apply a correction consistently

For learned increment $\delta\phi$, the correction updates both current and backward fields:

$$
\phi^n\leftarrow\phi^n+\delta\phi,\qquad
\phi^{n-1}\leftarrow\phi^{n-1}+\delta\phi.
$$

The leapfrog difference $\phi^n-\phi^{n-1}$ is preserved. The corrected temperature and salinity also require a fresh density evaluation before the next physical advance. These operations are part of [`apply_joint_correction`](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/src/neuralpom/hybrid/sparse_coupling.py).

## Restore a complete trajectory

Reference checkpoints and runtime restarts include histories, turbulence variables, transports, and auxiliary fields alongside visible prognostic fields. Their stored model clock identifies the physical time. Training restart additionally restores optimizer and sampling state. Continue to [Differentiation and restart](../cores/gradient-safety.md) and [Reference format](../data/format.md).
