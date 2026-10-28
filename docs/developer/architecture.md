# Architecture

## Package structure

| Package | Responsibility |
|---|---|
| `cores/explicit` | Explicit free-surface POM |
| `cores/implicit_p0_fixed` | Semi-implicit free-surface POM |
| `cores/case_adapters` | Case boundaries, compatibility operators and tidal input data |
| `cases` | Shared Double-gyre settings |
| `models` | Residual U-Net and Triton architecture |
| `hybrid` | State packing, constrained correction and physical execution |
| `v1` | Data loading, caches, staged training, restart and tracking |
| `evaluation` | Frozen inference, deployment policies and diagnostics |
| `physics_baselines` | Physical closure comparisons |
| `ai_baselines` | Direct full-state neural forecasts |
| `training` | Shared daily objective |

The production core registry in `src/neuralpom/core_factory.py` exposes `explicit` and `implicit_p0_fixed`.

## State and correction

The physical runtime restores the complete numerical state and advances it to a correction endpoint. The network input contains **101 channels** from six three-dimensional fields over 16 storage levels and five two-dimensional fields. The network predicts velocity, temperature and salinity increments on **15 active layers**, giving **60 output channels**.

`hybrid/state.py` packs and restores the numerical state. `hybrid/sparse_coupling.py` applies column and layer constraints, updates the required history fields, and refreshes density before physical integration continues. `v1/runtime.py` connects this operation to data, calibration and the physical core.

## Entry points

The `neuralpom` console command dispatches to `v1/cli.py`; `neuralpom-ai` dispatches to `ai_baselines/cli.py`. The short scripts under `scripts/v1/` expose plan, preparation, training, evaluation and reporting from a checkout. [Reproduction](../reproduction.md) describes the complete workflow.
