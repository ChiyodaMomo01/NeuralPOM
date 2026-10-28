# Wind-driven Double gyre

## Physical configuration

The primary case is a wind-driven basin with rotation, a free surface, stratification and vertical mixing. Both production cores obtain their case settings from `src/neuralpom/cases/`. The [governing equations](../physics/equations.md) and [turbulence closure](../physics/turbulence.md) describe the physical model.

## Hybrid workflow

The coarse training grid has **90 × 82 horizontal points**, **16 storage levels** and **15 active layers**. Hourly and daily models apply constrained velocity, temperature and salinity corrections after the corresponding physical interval. Each core uses a matching reference ensemble.

Use the four templates listed in [Configuration](../guide/configuration.md). Preparation, staged training and frozen inference are documented in [Quick start](../guide/quickstart.md).
