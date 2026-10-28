# Dynamical cores

NeuralPOM provides **Explicit** and **semi-implicit** hydrostatic POM implementations through a common initialization, state and training interface.

| Core | Registry name | Implementation |
|---|---|---|
| Explicit | `explicit` | `src/neuralpom/cores/explicit/` |
| Semi-implicit | `implicit_p0_fixed` | `src/neuralpom/cores/implicit_p0_fixed/` |

Both use the [Arakawa C grid](../numerics/arakawa-c-grid.md), terrain-following vertical coordinates, tracer transport and vertical turbulence closure. The Explicit implementation advances the external mode through subcycling. The semi-implicit implementation solves an elliptic free-surface problem. See [Explicit](./explicit.md), [Semi-implicit](./implicit.md) and [Governing equations](../physics/equations.md).

The matching reference ensemble, core configuration and calibration assets define each training run. The [portable templates](../guide/configuration.md) retain the configured backend for each core.
