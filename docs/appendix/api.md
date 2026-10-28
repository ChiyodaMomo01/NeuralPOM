# API and entry points

## Command interfaces

| Interface | Purpose |
|---|---|
| `neuralpom --config CONFIG prepare` | Build caches and calibration |
| `neuralpom --config CONFIG train --stage STAGE` | Train or resume a hybrid stage |
| `neuralpom-ai` | Direct neural forecast preparation, training and evaluation |
| `python scripts/v1/plan.py --config CONFIG` | Inspect hybrid sampling and budgets |
| `python scripts/v1/evaluate.py --help` | Training-oriented frozen evaluation |
| `python scripts/v1/report.py --help` | Training run report |
| `python -m neuralpom.evaluation.campaign --help` | Paired frozen forecast campaigns |

The hybrid training stages are `pretrain`, `midtrain4` and `posttrain`.

## Core and state modules

`neuralpom.core_factory.resolve_core_class(name)` resolves `explicit` or `implicit_p0_fixed`. `neuralpom.hybrid.build_initialized_core` constructs an initialized core. The `v1` package supplies configuration, data, cache and checkpoint interfaces; the `evaluation` package supplies deployment policies and forecast diagnostics. See [Architecture](../developer/architecture.md).
