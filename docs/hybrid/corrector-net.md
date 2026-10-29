# CorrectorNet

## Input and output

**CorrectorNet** is the residual U-Net in `src/neuralpom/models/corrector_net.py`. The hybrid input includes current and historical velocities, temperature and salinity, sea-surface elevation and depth-averaged transports. Six three-dimensional fields over **16 storage levels** and five two-dimensional fields give **101 input channels**.

The output contains increments for zonal velocity, meridional velocity, temperature and salinity over **15 active layers**, giving **60 output channels**. The portable templates use a network width of **256**.

## The channel map

| Input group | Fields | Channels |
|---|---|---|
| Three-dimensional fields | `u`, `ub`, `v`, `vb`, `t`, `s` | 6 × 16 = 96 |
| Two-dimensional fields | `el`, `ua`, `uab`, `va`, `vab` | 5 |
| Total input | Packed in the order above | **101** |
| Output increments | `u`, `v`, `t`, `s` on active layers | 4 × 15 = **60** |

The `b` suffix marks backward velocity or transport levels. Temperature and salinity inputs use their current fields. The input contract is defined in [`hybrid/constants.py`](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/src/neuralpom/hybrid/constants.py), and packing is implemented in [`hybrid/state.py`](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/src/neuralpom/hybrid/state.py).

## Why predict increments

An increment is applied to an already advanced physical endpoint. The residual U-Net can represent corrections at several horizontal scales while the physical solver continues to carry the prognostic state. The output passes through the physical constraints before it becomes a state change. Consequently, an output tensor alone is not a complete corrected restart.

## Physical application

The network output is decoded using the matching training calibration. `hybrid/sparse_coupling.py` applies velocity column constraints, tracer layer budgets and joint limiting, then updates the physical state and density. This corrected endpoint is carried into the following physical interval. The correction map is shared by the [training](./training.md) and frozen forecast interfaces.
