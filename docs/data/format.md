# Reference data format

## Ensemble layout

`FixedCoreDataset`, implemented in `src/neuralpom/v1/data.py`, reads the matching core's manifest, member provenance and NPZ arrays:

```text
REFERENCE_ROOT/
  manifest.json
  member_0_gpu0/
    provenance.json
    checkpoints_lowres_consistent/
      checkpoint_Day0999.npz
    reference_data_lowres_consistent/
      ref_Day1000_seg02.npz
  ...
  member_7_gpu7/
```

Each member follows the same directory layout. Model day **d** starts from `checkpoint_Day{d-1:04d}.npz`; the array's stored clock provides its model time. The coarse reference grid has **90 × 82 × 16** stored points/levels.

## Initial state and labels

The initial state includes current and historical velocity, tracers, turbulence variables, density, transport and free-surface fields. The reader defines these explicitly as `INITIAL_3D_FIELDS` and `INITIAL_2D_FIELDS`. Training labels use the fields in `LABEL_FIELDS`.

Intraday reference files exist every **20 minutes in the first four hours**. The hourly labels use segments **02, 05, 08 and 11**. A one-day label uses the following daily checkpoint.

## Restart state

A reference initial state starts a new physical trajectory. A training or inference restart additionally restores the low-resolution model's complete numerical histories, solver state, clock and integration counters. The serialization interface is `src/neuralpom/v1/checkpoint.py`.
