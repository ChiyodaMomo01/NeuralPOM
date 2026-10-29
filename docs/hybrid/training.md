# From endpoints to trajectories

**Staged learning changes both the states seen by the corrector and the gradient path used to update it.** Pretrain establishes a local correction, Midtrain learns through the physical response to earlier corrections, and Posttrain continues learning on model-generated trajectories. ([Manuscript, Section 2.3 and Appendix B.5](../paper.md))

## Three stages, three learning problems

| Stage | States presented to the network | Parameter-gradient path | Initialization |
|---|---|---|---|
| **Pretrain** | Cached physical endpoints | A single constrained correction | New network |
| **Midtrain** | Four connected corrected endpoints | Through corrections and the three connecting physical intervals | Selected Pretrain weights |
| **Posttrain** | Ten-day model-generated trajectories | One update per day through the final correction of that day | Final Midtrain weights |

### Pretrain: learn the endpoint correction

Preparation advances reference initial states with the selected coarse physical model and caches their endpoints. The network learns to correct those endpoints against the matching reference targets. Since the physical endpoint is cached, the optimizer differentiates through the correction and objective without replaying the preceding physical interval.

### Midtrain: learn the physical consequences

The corrected endpoint becomes the next interval's initial condition. A later loss therefore depends on earlier neural corrections through the physical evolution. Four corrected endpoints give a four-hour sequence for hourly coupling or a four-day sequence for daily coupling, with three physical intervals connecting the endpoints inside the gradient path.

Let $\mathcal L$ be an endpoint loss, $x$ the physical state, $\theta$ the network parameters, and $\Phi_\Delta$ the physical advance over correction interval $\Delta$. The chain rule includes factors of $\partial\Phi_\Delta/\partial x$: they tell the optimizer how a change in one corrected state affects a later state. This is why the solver's [backward operators](../cores/gradient-safety.md) are part of the learning method.

### Posttrain: adapt along generated trajectories

A trajectory spans ten days. With hourly coupling, the network makes 24 corrections each day, but only the final correction contributes a parameter-gradient path for that day's update. With daily coupling, there is one correction and one update per day. The corrected state continues into the next day after detachment, so the next training input reflects the accumulated forecast trajectory.

The ten-day rollout horizon and the parameter-gradient horizon describe different aspects of this stage. The standard trainer is implemented in [`v1/engine.py`](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/src/neuralpom/v1/engine.py); [Posttrain interfaces](../v1/posttrain.md) documents the separate optional interfaces.

## Execute the sequence

After [installation](../guide/installation.md), configuration, and reference preparation, run from the repository root:

```bash
CONFIG=configs/local/explicit_1day.json
set -e
for stage in pretrain midtrain4 posttrain; do
  torchrun --standalone --nproc_per_node=8 \
    scripts/v1/train.py --config "$CONFIG" --stage "$stage"
done
```

The command uses the eight-rank topology in the portable templates. `midtrain4` is the command-line name for the four-endpoint Midtrain stage. The [plan command](../guide/quickstart.md#inspect-the-training-plan) reports the resolved sampling and update budgets before training.

## Restart and freeze

A training restart restores the network, optimizer, sampler cursor, per-rank random-number states, and complete physical state. Repeat the matching command to resume a compatible incomplete stage. Stage weights intended for evaluation are saved under `run_root/frozen/`.

During a [frozen forecast](../reproduction.md#frozen-inference-and-scoring), network parameters stay fixed. DND is selected by the evaluation policy after training; the staged training workflow uses the canonical constrained correction. ([Runtime](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/src/neuralpom/v1/runtime.py), [evaluation correction](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/src/neuralpom/evaluation/correction.py))
