---
title: Start here
description: A reading path from the NeuralPOM scientific idea to a reproducible forecast.
---
# A field guide to NeuralPOM

**NeuralPOM** couples a differentiable ocean model with a learned correction. The physical model advances the ocean; a residual U-Net updates its endpoint; the corrected state then enters the next physical interval. The central question is how to make a useful local correction remain useful as the ocean evolves. ([Manuscript, Sections 2–5](../paper.md))

## Choose your starting point

| Your goal | Start with | Then continue to |
|---|---|---|
| Understand the scientific idea | [Hybrid model](../hybrid/overview.md) | [Staged learning](../hybrid/training.md) → [Density-normal damping](../hybrid/density-normal-damping.md) |
| Run the published workflow | [Installation](./installation.md) | [Quick start](./quickstart.md) → [Reproduction](../reproduction.md) |
| Study the physical model | [Governing equations](../physics/equations.md) | [Mode splitting](../physics/barotropic-baroclinic.md) → [Two dynamical cores](../cores/overview.md) |
| Work on the implementation | [Architecture](../developer/architecture.md) | [Differentiation and restart](../cores/gradient-safety.md) → [API](../appendix/api.md) |

## Four ideas to keep in view

**The numerical state includes its history.** A leapfrog solver needs more than the current velocity, temperature, and salinity fields. A correction updates the appropriate current and backward fields together, and density is recomputed before the physical model continues. See [Time stepping](../numerics/time-stepping.md).

**A correction interval is not a physical time step.** The coarse model advances at 600 s. Corrections arrive every hour or every day. The Explicit core also resolves the fast external mode with shorter substeps. See [Configuration](./configuration.md).

**Training changes the trajectory presented to the network.** Pretrain learns from cached physical endpoints. Midtrain learns through connected physical intervals. Posttrain adapts on states generated over ten-day trajectories. See [Staged training](../hybrid/training.md).

**Tracer corrections change density as well as tracer error.** Temperature and salinity enter the equation of state, which controls buoyancy and baroclinic pressure gradients. Density-normal damping acts on a constrained density-modifying projection of the tracer increment. See [the interactive explanation](../hybrid/density-normal-damping.md).

These definitions follow the [manuscript](../paper.md), [`hybrid/sparse_coupling.py`](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/src/neuralpom/hybrid/sparse_coupling.py), and [`v1/engine.py`](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/src/neuralpom/v1/engine.py).

## A useful first session

Install the package and run the plan command in [Quick start](./quickstart.md). This checks the configuration and exposes the sampling and update budgets before you prepare data. Then read the [reference format](../data/format.md), match one core and correction interval to its dataset, and follow the preparation and training sequence.

To inspect results immediately, open the [paper reading guide](../paper.md). The manuscript figures and numerical source tables are included in the code repository.
