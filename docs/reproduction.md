# Reproduction

## Paper and source data

The repository includes the manuscript, supplement and publication source tables under `paper/`:

- [Manuscript](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/paper/NeuralPOM.pdf).
- [Supplement](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/paper/Supplement.pdf).
- [Source data](https://github.com/ChiyodaMomo01/NeuralPOM/tree/main/paper/data).

The source tables contain the saved numerical results used by the paper. The full reference ensembles, trained weights and matching calibration/cache assets are supplied separately from the code checkout.

## Required assets

| Workflow | Assets |
|---|---|
| Read or replot source tables | CSV files in `paper/data/` |
| Train a hybrid model | Matching reference ensemble and a portable training configuration |
| Resume training | Run assets, complete restart checkpoint and matching configuration |
| Evaluate frozen weights | Matching weights, reference ensemble, calibration and configuration |

A reference root contains `manifest.json` and eight member directories with provenance, daily initial checkpoints and intraday reference fields. The [data format](./data/format.md) defines the layout. Calibration and prepared caches live under the training run root; frozen stage weights live under `frozen/`.

## Train a new model

Follow [Installation](./guide/installation.md), copy the corresponding template from `configs/examples/` to `configs/local/`, and set the data/output paths. The [quick start](./guide/quickstart.md) provides the plan, prepare and staged training commands.

The training stages are **Pretrain**, **Midtrain** and **Posttrain**. Training uses canonical constrained correction. Each core and interval has its own matching assets. The templates record the optimizer, sampling and numerical settings used by the training interface.

## Frozen inference and scoring

Copy `configs/examples/evaluation_100d.json` to `configs/local/` and point its groups at the matching local training configurations. The protocol specifies **16 origins** and a **100-day** horizon. Paths in the paired protocol resolve from the repository root.

```bash
python -m neuralpom.evaluation.campaign worker \
  --protocol configs/local/evaluation_100d.json \
  --variant explicit_1day_posttrain_density75 --device cuda:0
```

The variant selects a group, frozen stage and correction policy. `density75` applies the paper's tracer deployment policy; `constrained` applies canonical constrained correction; `raw` selects the unconstrained comparison. The physical control for this group is `explicit_1day_physics_physics`. Repeat the worker command for the desired variants and use `python -m neuralpom.evaluation.campaign --help` for campaign reporting.

Forecasts save daily fields, geometry, metrics, complete-state restarts and trajectory outcomes. Scoring uses the matching reference fields and wet-grid geometry. Model weights remain fixed during forecasts. `scripts/v1/evaluate.py` supplies the training-oriented evaluator with canonical correction.
