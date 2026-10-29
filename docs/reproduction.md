# Reproduce the experiment

Choose one physical core and correction interval, then keep its references, training configuration, calibration, and frozen weights together. This page connects the manuscript assets to the supported command-line workflow.

## Paper and source data

The repository includes the manuscript, supplement and publication source tables under `paper/`:

- [Manuscript](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/paper/NeuralPOM.pdf).
- [Supplement](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/paper/Supplement.pdf).
- [Source data](https://github.com/ChiyodaMomo01/NeuralPOM/tree/main/paper/data).

The source tables contain the saved numerical results used by the paper. The full reference ensembles, trained weights and matching calibration/cache assets are supplied separately from the code checkout.

## Find the evidence

| Question | Source table or directory | Interpretation |
|---|---|---|
| How does aggregate forecast error change? | `paper/data/main_metrics.csv`, `baseline_comparisons.csv` | Use the stated core, field, policy, and evaluation window |
| How does error evolve with lead time? | `paper/data/daily_curves.csv`, `daily_mse.csv` | Compare paired origins at a common lead time |
| What changes across training stages? | `paper/data/training_stage_comparisons.csv` | Compare stage weights within a matched setup |
| What does the tracer policy change? | `paper/data/policy_comparisons/`, `tracer_decomposition.csv` | Separate projection diagnostics from forecast errors |
| Which spatial scales change? | `paper/data/spectral_curves.csv`, `spectral_metrics.csv` | Use the paper's wavenumber and normalization definitions |
| How does skill vary with depth? | `paper/data/layer_profiles.csv` | Retain the layer geometry and variable identity |

All paths are relative to the repository root. The [paper guide](./paper.md) explains the scientific comparisons and displays the corresponding manuscript figures.

## Required assets

| Workflow | Assets |
|---|---|
| Read or replot source tables | CSV files in `paper/data/` |
| Train a hybrid model | Matching reference ensemble and a portable training configuration |
| Resume training | Run assets, complete restart checkpoint and matching configuration |
| Evaluate frozen weights | Matching weights, reference ensemble, calibration and configuration |

A reference root contains `manifest.json` and eight member directories with provenance, daily initial checkpoints and intraday reference fields. The [data format](./data/format.md) defines the layout. Calibration and prepared caches live under the training run root; frozen stage weights live under `frozen/`.

Reference data are hosted on [ModelScope](https://www.modelscope.cn/datasets/chiyodamomo01/NeuralPOM). Follow the dataset's current instructions for asset availability and access, and preserve the extracted member-directory structure.

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


## Inspect the result

Check the variant's trajectory outcomes alongside the saved metrics: the completed forecast window and matched reference support determine which errors can be compared. Use the campaign's reporting commands to inspect saved daily diagnostics, and use the paper's source tables when reproducing its published numerical summaries.

For a restart, keep the complete-state checkpoint with its configuration and run assets. A plot or a network weight file does not contain the physical histories needed to continue an ocean trajectory. See [Reference format](./data/format.md) and [Differentiation and restart](./cores/gradient-safety.md).
