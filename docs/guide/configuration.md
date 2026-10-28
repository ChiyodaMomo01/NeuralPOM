# Configuration

## Portable templates

The JSON templates in `configs/examples/` contain the scientific training settings and relative paths.

| Core | Correction interval | Template |
|---|---|---|
| Explicit | 1 hour | `explicit_1h.json` |
| Explicit | 1 day | `explicit_1day.json` |
| Semi-implicit | 1 hour | `implicit_1h.json` |
| Semi-implicit | 1 day | `implicit_1day.json` |

`evaluation_100d.json` describes paired frozen forecasts. `rl_posttrain.json` demonstrates the optional score-function trainer described in [Posttrain](../v1/posttrain.md).

## Training fields

| Field | Meaning |
|---|---|
| `core` | `explicit` or `implicit_p0_fixed` |
| `frequency` | `1h` or `1day` correction interval |
| `data_root` | Matching reference ensemble |
| `run_root` | Caches, calibration, weights and run records |
| `world_size` | Number of distributed training ranks |
| `grid` | Coarse horizontal dimensions |
| `model.hidden_dim` | CorrectorNet width |
| `split` | Members, model-day bounds and sample strides |
| `stages` | Stage budgets, rollout horizons and checkpoint intervals |
| `losses` | Loss names, weights and normalization scales |
| `acceleration` | Physical execution and solver backends |
| `wandb` | Project, entity and `online` or `offline` mode |

Training paths resolve relative to the configuration file. Defaults and their merge rules are implemented in `src/neuralpom/v1/config.py`. The plan command prints resolved budgets:

```bash
python scripts/v1/plan.py --config configs/local/explicit_1day.json
```

The portable examples select **offline W&B**. Set `wandb.entity` to your own account or team and `wandb.project` to your project when enabling online tracking.

## Evaluation protocol

The paired evaluator uses a separate protocol with `groups`, `cases`, `horizons_days`, `correction_policies` and `output_root`. Each group names a core, correction interval and training configuration. Each case names a reference member and initial model day. The worker's `--variant` selects one group, stage and correction policy; the [quick start](./quickstart.md) shows a complete command.

The training evaluator in `scripts/v1/evaluate.py` uses canonical constrained correction. The campaign evaluator in `neuralpom.evaluation.campaign` additionally exposes `raw`, `constrained` and `density75` policies.
