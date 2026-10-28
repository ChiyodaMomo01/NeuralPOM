# Quick start

## Install and select a configuration

Follow [Installation](./installation.md), then copy a portable template from the repository root:

```bash
mkdir -p configs/local
cp configs/examples/explicit_1day.json configs/local/explicit_1day.json
```

Set `data_root` to the matching reference ensemble and `run_root` to the output directory. Relative paths resolve from the JSON file's directory. The four training templates cover Explicit and semi-implicit cores at hourly and daily correction intervals; see [Configuration](./configuration.md).

## Inspect the training plan

```bash
python scripts/v1/plan.py --config configs/local/explicit_1day.json
```

This command reports sample counts and update budgets without loading reference arrays or starting training.

## Prepare and train

After supplying the reference ensemble, use the configured DDP world size:

```bash
CONFIG=configs/local/explicit_1day.json

torchrun --standalone --nproc_per_node=8 \
  scripts/v1/prepare.py --config "$CONFIG"

set -e
for stage in pretrain midtrain4 posttrain; do
  torchrun --standalone --nproc_per_node=8 \
    scripts/v1/train.py --config "$CONFIG" --stage "$stage"
done
```

Preparation builds physical endpoint caches and training-only calibration. The stages train a new network, propagate gradients across four corrected endpoints, and then update along ten-day model-generated trajectories. Stage weights are written under `run_root/frozen/`. Repeating the matching training command resumes a compatible incomplete stage. See [Training](../hybrid/training.md).

## Evaluate frozen forecasts

Copy `configs/examples/evaluation_100d.json` to `configs/local/evaluation_100d.json` and set each selected group's `config` path to its matching training configuration. Campaign group paths resolve from the repository root. Run a single configured forecast variant:

```bash
python -m neuralpom.evaluation.campaign worker \
  --protocol configs/local/evaluation_100d.json \
  --variant explicit_1day_posttrain_density75 --device cuda:0
```

The same-core physical baseline is `explicit_1day_physics_physics`. The `density75` policy selects the paper's frozen tracer correction. See [Reproduction](../reproduction.md) for assets, forecast outputs and scoring.
