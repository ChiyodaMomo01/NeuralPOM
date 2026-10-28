# Staged training

## Stage definitions

| Stage | Network input | Parameter-gradient path | Parent |
|---|---|---|---|
| **Pretrain** | Cached physical endpoints | One constrained correction | New network |
| **Midtrain** | Four corrected endpoints | Full backpropagation through the connecting physical intervals | Selected Pretrain weights |
| **Posttrain** | Ten-day model-generated trajectories | One daily update; state detached between days | Final Midtrain weights |

Midtrain uses four-hour trajectories for hourly correction and four-day trajectories for daily correction. Hourly Posttrain makes 24 corrections per day and differentiates only the final correction. Daily Posttrain makes one correction per day. After an update, the generated corrected state continues into the next day.

## Execution and restart

The [quick start](../guide/quickstart.md) shows the prepare and train commands. Preparation fits calibration using training inputs and builds physical endpoint caches. W&B tracking is enabled in offline mode by the portable examples.

Restart checkpoints preserve model and optimizer state, sampling cursor, each rank's random-number state and the complete physical state. Repeating a compatible incomplete stage resumes its committed checkpoint. Frozen stage weights are written under `run_root/frozen/`.

The training implementation is `src/neuralpom/v1/engine.py`; Posttrain implementations are described in [Posttrain interfaces](../v1/posttrain.md).
