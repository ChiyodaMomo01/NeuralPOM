# Data and calibration pipeline

## Reference ensembles

Each production core uses its own reference ensemble. `FixedCoreDataset` reads `manifest.json` and per-member `provenance.json` to identify the generating core, numerical settings and consistent downsampling. [Reference data format](./format.md) describes the directory and time-index conventions.

The full ensembles and model weights are distributed separately from this checkout. The [reproduction guide](../reproduction.md) lists the assets required for training and frozen inference.

## Preparation

```bash
torchrun --standalone --nproc_per_node=8 \
  scripts/v1/prepare.py --config configs/local/explicit_1day.json
```

Preparation builds endpoint caches and calibration through `src/neuralpom/v1/cache.py`. Input statistics use the configured training inputs and wet-grid geometry. Assets and cache records identify the core, grid, interval, sampling and numerical settings used to produce them.

## Training and evaluation

The trainer reads the prepared assets and matching references. Frozen evaluation reads the corresponding stage weights and calibration, advances the predicted state, and scores against reference fields. References enter the diagnostics after prediction. [Training](../hybrid/training.md) and [Reproduction](../reproduction.md) give the commands.
