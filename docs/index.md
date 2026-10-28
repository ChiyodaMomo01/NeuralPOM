---
layout: home
hero:
  name: NeuralPOM
  text: A Hybrid Ocean Forecast Model
  tagline: Differentiable POM dynamics and staged learning
  actions:
    - theme: brand
      text: Get started
      link: /guide/quickstart
    - theme: alt
      text: Reproduction
      link: /reproduction
  image:
    src: /logo.png
    alt: NeuralPOM
features:
  - title: Differentiable dynamics
    details: Explicit and semi-implicit POM implementations share state and training interfaces.
    link: /cores/overview
  - title: Endpoint corrections
    details: A residual U-Net corrects velocity, temperature and salinity inside the physical trajectory.
    link: /hybrid/overview
  - title: Staged learning
    details: Pretrain, Midtrain and Posttrain connect physical endpoints to model-generated trajectories.
    link: /hybrid/training
---

## Model and workflow

**NeuralPOM** combines differentiable ocean dynamics with learned endpoint corrections. The package includes the physical cores, constrained correction, staged training, restart and frozen forecast diagnostics. The manuscript, supplement and numerical source tables are described in [Reproduction](./reproduction.md).

Start with [Installation](./guide/installation.md), [Quick start](./guide/quickstart.md) and [Configuration](./guide/configuration.md). The scientific reference covers [governing equations](./physics/equations.md), [numerics](./numerics/arakawa-c-grid.md) and [hybrid training](./hybrid/training.md).
