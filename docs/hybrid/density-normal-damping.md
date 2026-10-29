---
title: Density-normal damping
description: How NeuralPOM attenuates the density-modifying tracer projection while preserving layer budgets.
---
# Density-normal damping

**Temperature and salinity corrections also change the forces that drive the ocean.** Through the equation of state, a tracer increment can alter density, stratification, and baroclinic pressure gradients. Density-normal damping (DND) targets the density-modifying part of a constrained tracer increment. ([Manuscript, Sections 2.2 and 5.1; Appendix C](../paper.md))

## Separate the correction by its density response

Work within one active vertical layer. Let $i$ index wet horizontal cells and let $s_T,s_S$ be the positive correction scales for temperature and salinity. For constrained tracer increments $\delta T_i,\delta S_i$, define the standardized vector $z_i$ and the scaled equation-of-state gradient $g_i$:

$$
z_i=\begin{pmatrix}\delta T_i/s_T\\\delta S_i/s_S\end{pmatrix},\qquad
g_i=\begin{pmatrix}s_T\,\partial_T\rho_i\\s_S\,\partial_S\rho_i\end{pmatrix}.
$$

Here $\rho_i$ is density evaluated at the pre-correction state. Its first-order response is $g_i^\top z_i$. Let $n_i=g_i/\lVert g_i\rVert_2$ be the local unit density normal where the gradient is nonzero, and let $w_i$ be normalized wet-volume weights.

The projection must also respect the two layer budgets. Define the density-normal subspace $\mathcal S$ as vectors $p$ with a scalar amplitude $\beta_i$ at each cell:

$$
\mathcal S=\left\{p:p_i=\beta_i n_i,\quad\sum_i w_i\beta_i n_i=0\right\}.
$$

Let $E$ be the orthogonal projection onto $\mathcal S$ under the volume-weighted inner product, and $I$ the identity. The decomposition $z=Ez+(I-E)z$ separates the constrained density-normal projection from its orthogonal remainder. Both components retain the zero layer means.

## Retain three quarters of the projection

For retention factor $c\in[0,1]$, the damped increment is

$$
z_c=(I-E)z+cEz.
$$

The paper uses **$c=3/4$**. The density-normal projection keeps **75% of its amplitude** and **56.25% of its squared norm**, before the final paired limiter. Its squared-norm reduction is $1-c^2=43.75\%$. The orthogonal component is retained in full. These percentages describe the projection operation; forecast error is measured separately.

<DensityExplorer />

## How the code applies it

[`evaluation/density.py`](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/src/neuralpom/evaluation/density.py) first applies the canonical constrained correction, evaluates the EOS gradient at the original tracer state, and obtains a layer-budget-preserving density-normal amplitude. It changes that amplitude by the factor $c-1$, applies the paired tracer limiter, writes consistent current and backward fields, and recomputes density with the full EOS.

The implementation solves a two-constraint problem per layer using a small Gram matrix and its pseudoinverse. The constraints couple the local density-normal directions across the layer. A pointwise multiplication of all temperature and salinity increments by 0.75 would implement a different operation.

## Select it in a frozen forecast

```bash
python -m neuralpom.evaluation.campaign worker \
  --protocol configs/local/evaluation_100d.json \
  --variant explicit_1day_posttrain_density75 --device cuda:0
```

The protocol names the DND policy `density75`. The `constrained` policy is the Undamped constrained update, and `raw` selects the unconstrained comparison. See [Reproduction](../reproduction.md) to match the configuration, weights, calibration, and reference data.

## Read the evidence

The manuscript compares DND against Undamped correction and norm-matched uniform damping using the same network weights. This separates the effect of direction from the effect of overall increment size. Its projection diagnostics and forecast comparisons are reproduced in the [paper guide](../paper.md#density-response-and-forecast-error), with numerical values in [`paper/data/policy_comparisons/`](https://github.com/ChiyodaMomo01/NeuralPOM/tree/main/paper/data/policy_comparisons).
