# Hybrid physical-neural model

Let $n$ index correction endpoints, and let $x_n^+$ be the complete numerical state after correction. Let $\Delta$ be the correction interval, $\Phi_\Delta$ the physical advance over that interval, and $\mathcal N$ the input normalization. Let $f_\theta$ be the network with parameters $\theta$, and $\mathcal C$ the operation that decodes and constrains its output, updates numerical histories and refreshes density. One interval is

$$
x_{n+1}^-=\Phi_\Delta(x_n^+),\qquad
x_{n+1}^+=\mathcal C\!\left(x_{n+1}^-,f_\theta(\mathcal N(x_{n+1}^-))\right).
$$

The correction interval is **one hour or one day**. The residual U-Net reads **101 channels** and predicts **60 increment channels**. Velocity corrections satisfy column constraints; temperature and salinity corrections satisfy layer budgets. The corrected physical state becomes the initial state for the next interval.

The implementation is in `hybrid/sparse_coupling.py` and `v1/runtime.py`. See [CorrectorNet](./corrector-net.md), [Training](./training.md) and [Supervision](./supervision.md).
