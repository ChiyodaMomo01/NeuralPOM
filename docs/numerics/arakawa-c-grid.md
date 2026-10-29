# A staggered ocean grid

**The Arakawa C grid places scalar fields at cell centers and horizontal velocities on their corresponding cell faces.** This gives a direct representation of the fluxes entering a cell's volume and tracer balances. NeuralPOM uses this arrangement in both physical cores. ([Manuscript, Section 2.1](../paper.md), [grid implementation](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/src/neuralpom/cores/explicit/grid.py))

## Field locations

| Location | Physical quantities | Code fields or geometry |
|---|---|---|
| Cell center | Temperature, salinity, density, surface elevation | `t`, `s`, `rho`, `el`; area `art`, mask `fsm` |
| Face normal to the first horizontal direction | Zonal velocity and its depth average | `u`, `ua`; area `aru`, mask `dum` |
| Face normal to the second horizontal direction | Meridional velocity and its depth average | `v`, `va`; area `arv`, mask `dvm` |
| Vertical interfaces | Transformed vertical transport | `w`; sigma interfaces `z` |
| Vertical layer centers | Tracers and horizontal velocity | sigma midpoints `zz` |

The first two array axes are the horizontal indices `i,j`; the third is vertical index `k`. Boundary rows and columns carry the case's boundary treatment. Center and face masks have different locations, so a scalar wet mask cannot replace every velocity mask.

## Sigma levels and physical volumes

Let $\sigma_k$ and $\sigma_{k+1}$ be successive vertical interfaces. The fractional layer thickness is $\Delta\sigma_k=\sigma_k-\sigma_{k+1}$. Let $A_{ij}$ be horizontal cell area, $D_{ij}$ total depth, and $m_{ij}$ the wet mask. The geometric volume of a cell is

$$
V_{ijk}=A_{ij}D_{ij}\Delta\sigma_km_{ij}.
$$

The model stores interfaces in `z`, midpoints in `zz`, interface differences in `dz`, and midpoint differences in `dzz`. The correction uses the volume weights defined by its physical-state convention in [`volume_weights`](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/src/neuralpom/hybrid/sparse_coupling.py).

## Storage and active layers

The published setup has **16 vertical storage levels and 15 active layers**. CorrectorNet reads all 16 storage levels for its six three-dimensional inputs, but predicts increments for 15 active layers. The inactive bottom storage level is left unchanged. See [CorrectorNet](../hybrid/corrector-net.md) for the full channel map.

The reference simulation uses a **354 × 322** horizontal grid, and the forecast model a **90 × 82** grid. The archived references are consistently restricted to the forecast grid, so target arrays and coarse states share the model's geometry. ([Manuscript, Section 3.1 and Appendix B.1](../paper.md))

## Practical consequence

Use each field's own mask, area, and vertical weights when calculating errors or budgets. Preserve staggering during data conversion. The dataset reader and evaluation code already encode these conventions; [Data format](../data/format.md) and [Reproduction](../reproduction.md) describe the supported workflow.
