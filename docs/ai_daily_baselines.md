# Daily full-state AI baselines

## Forecast task

The direct neural models predict a complete daily state without POM integration during prediction. `StateCodec` encodes **43 spatial fields** as **313 channels** on the **90 × 82 × 16** storage grid. Frozen training-input means and standard deviations normalize each channel. Static geometry remains external.

The registry includes **Triton**, **ResUNet**, **SimVP**, **ConvLSTM** and **PastNet**. `src/neuralpom/ai_baselines/upstream.py` records the external source identities; `scripts/fetch_ai_upstream.py` fetches the pinned architecture files required by the corresponding adapters.

## Training

| Stage | Trajectory | Update |
|---|---|---|
| **Pretrain** | One next-day forecast | One standardized next-state loss |
| **Posttrain** | Ten-day model-generated forecast | One daily update with detached carry |

The sample lists use `neuralpom.v1.sampling.stage_samples`. Pretrain selects validation-best weights. Posttrain carries its actual generated state forward after each parameter update. The full-state objective uses standardized channel errors and wet-grid weights.

Use `neuralpom-ai --help` to inspect preparation, planning, training and frozen evaluation commands. The matching core configuration identifies the reference ensemble and geometry. See [Configuration](./guide/configuration.md) and [Reproduction](./reproduction.md) for assets and outputs.

## Diagnostics

Frozen AI forecasts use the common state diagnostics for physical-unit field errors, density, tracer decomposition, profiles and velocity spectra. The evaluation runner records reference pairing, weight identities and completed origins with the forecast outputs. The implementation is in `src/neuralpom/ai_baselines/`.
