# Posttrain interfaces

## Truncated training

The default **truncated Posttrain** method advances ten-day model-generated trajectories with one parameter update per day. Hourly correction differentiates only the final daily correction; daily correction differentiates its one correction. The corrected state continues after each update and is detached between days.

The implementation is `src/neuralpom/v1/posttrain/truncated.py`. The ordinary staged workflow is shown in [Quick start](../guide/quickstart.md).

## Optional score-function training

The optional **RL Posttrain** method uses a fixed-variance Gaussian policy and grouped leave-one-out returns. The network stays fixed while collecting a complete trajectory. Detached observations and raw actions are replayed through the network for score-gradient updates. The policy, rollout and trainer modules are under `src/neuralpom/v1/posttrain/`.

`configs/examples/rl_posttrain.json` supplies a portable configuration. Select a matching frozen Midtrain parent and use the ordinary training interface:

```bash
neuralpom --config configs/local/rl_posttrain.json train \
  --stage posttrain --parent runs/matching_parent/frozen/midtrain4.pt
```

Run the command through torchrun with the configuration's world size for distributed training. The example's `method` selects the trainer; existing truncated configurations retain their original method.
