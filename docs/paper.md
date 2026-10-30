---
title: Read the paper
description: The NeuralPOM manuscript, its scientific argument, figures, source data, and implementation map.
---
# New wine in an old bottle

**NeuralPOM: A Hybrid Ocean Forecast Model with Differentiable Dynamics and Staged Learning**

Ruiqi Shu · Guangliang Liu · Ruijian Gou · Tianrun Gao · Haomin Wen · Hao Li · Xiaomeng Huang

[Read the manuscript ↗](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/paper/NeuralPOM.pdf) · [Supplement ↗](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/paper/Supplement.pdf) · [Numerical source tables ↗](https://github.com/ChiyodaMomo01/NeuralPOM/tree/main/paper/data)

## The scientific question

A coarse ocean model misses parts of the resolved evolution of a finer reference model. A learned endpoint correction can reduce local error, but its effect is then carried through pressure gradients, advection, and mixing. The paper examines how differentiable dynamics, staged learning, and constrained corrections shape that subsequent evolution. (Manuscript, Sections 1–2.)

## The reference ocean

The primary experiment is a wind-driven double gyre. Each physical core has its own eight-member fine-grid reference ensemble, restricted consistently to the coarse forecast grid. The fine horizontal grid is **354 × 322**, and the coarse grid is **90 × 82**, with **15 active vertical layers**. (Manuscript, Section 3.1.)

<figure><img src="/figures/developed_surface_fields.png" alt="Developed double-gyre reference fields: normalized vorticity, sea-surface elevation, temperature, zonal and meridional velocity, and salinity" loading="lazy"><figcaption>Developed reference surface fields, reproduced from the manuscript. The full panels retain their axes and color scales.</figcaption></figure>

## Three parts of the method

| Part | Question it answers | Tutorial |
|---|---|---|
| Differentiable physics | How does an earlier correction change a later state? | [Physical framework](./physics/equations.md), [adjoint solve](./numerics/elliptic-solver.md) |
| Staged learning | Which states and gradient paths train the network? | [Pretrain → Midtrain → Posttrain](./hybrid/training.md) |
| Constrained correction and DND | Which budgets and density responses should the update preserve or attenuate? | [Correction](./hybrid/overview.md), [DND explorer](./hybrid/density-normal-damping.md) |

## Density response and forecast error

For hourly coupling, the paper reports DND reductions relative to the same-weight Undamped update of roughly **5%–9% in zonal velocity and SSH error**, **12%–13% in temperature**, **38%–43% in salinity**, and **9%–10% in density** across the two physical cores. These are the paper's reported error comparisons; the projection's **43.75% squared-norm reduction** follows separately from its retention factor. (Manuscript, Section 5.2.)

<figure><img src="/figures/retention_theory.png" alt="Manuscript diagnostics comparing density-normal damping and uniform damping, including projection scaling and retained components" loading="lazy"><figcaption>Projection and density-response diagnostics from the manuscript. Read Section 5 and Appendix C for the definitions of each plotted quantity.</figcaption></figure>

<figure><img src="/figures/forecast_tracer_curves.png" alt="Manuscript lead-time curves for tracer and density forecast errors across correction policies" loading="lazy"><figcaption>Forecast tracer diagnostics from the manuscript. Numerical results are available in paper/data/ and paper/data/policy_comparisons/.</figcaption></figure>

## Read a forecast comparison correctly

The paired protocol uses **16 forecast origins** and **100-day trajectories**, initialized from reference member 7 at model days $8000+32r$, where $r=0,\ldots,15$. Model weights remain fixed during evaluation. The paper compares physical closures, learning stages, correction policies, and direct neural baselines. Their settings and comparison windows are documented in Section 3 and Appendix D.

For a specific comparison, match the physical core, correction interval, initial state, variable, spatial weighting, and lead-time window. The [reproduction guide](./reproduction.md) maps the saved source tables and evaluation commands to these choices.

## Trace the paper to the repository

| Evidence or method | Repository location |
|---|---|
| Model architecture | [`models/corrector_net.py`](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/src/neuralpom/models/corrector_net.py) |
| State constraints | [`hybrid/sparse_coupling.py`](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/src/neuralpom/hybrid/sparse_coupling.py) |
| Training stages | [`v1/engine.py`](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/src/neuralpom/v1/engine.py) |
| Density-normal policy | [`evaluation/density.py`](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/src/neuralpom/evaluation/density.py) |
| Forecast protocol | [`configs/examples/evaluation_100d.json`](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/configs/examples/evaluation_100d.json) |
| Forecast and stage metrics | [`paper/data/`](https://github.com/ChiyodaMomo01/NeuralPOM/tree/main/paper/data) |

## Authors and availability

**Ruiqi Shu, Guangliang Liu, and Ruijian Gou contributed equally**, in this author order.

The [repository README](https://github.com/ChiyodaMomo01/NeuralPOM) and manuscript contain the full author affiliations. Corresponding authors are **Hao Li** ([lihao_lh@fudan.edu.cn](mailto:lihao_lh@fudan.edu.cn)) and **Xiaomeng Huang** ([hxm@tsinghua.edu.cn](mailto:hxm@tsinghua.edu.cn)).

Code is available on [GitHub](https://github.com/ChiyodaMomo01/NeuralPOM) under Apache-2.0. Data are hosted in the [NeuralPOM ModelScope repository](https://www.modelscope.cn/datasets/chiyodamomo01/NeuralPOM). See its dataset documentation for available assets and access.
