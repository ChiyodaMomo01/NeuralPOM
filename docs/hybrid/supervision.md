# Supervision

## Corrected endpoints

The staged trainer scores **corrected fields** against matching reference endpoints. The default objective calls `src/neuralpom/training/daily_objective.py`; `v1/losses.py` composes configured losses and records their contributions.

Let $K$ be the set of configured loss components. For component $k\in K$, let $w_k$ be its weight, $s_k$ its normalization scale, and $\mathcal L_k$ its raw loss. The combined objective $\mathcal L$ is

$$
\mathcal L=\sum_{k\in K}\frac{w_k}{s_k^2}\mathcal L_k.
$$

The portable templates select `weighted_mse`. The composition interface also includes velocity spectral-energy and physical-mean losses. The configuration stores each component's name, weight and scale.

## Spatial weighting and calibration

The objective uses wet-cell geometry and the physical quantities defined by the daily objective. Calibration uses training inputs. Pretrain excludes the sea-surface contribution according to the stage settings; the physical runtime and later stages retain the complete state.

The data and gradient checks are in `tests/test_v1_data_losses.py`. See [Configuration](../guide/configuration.md) for the template fields and [Training](./training.md) for stage horizons.
