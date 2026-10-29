# The physical foundation

**NeuralPOM advances the three-dimensional hydrostatic primitive equations** with the Boussinesq approximation on an Arakawa C grid and a terrain-following vertical coordinate. The neural corrector acts at endpoints of this physical evolution. The formulation follows the Princeton Ocean Model; the implementation and numerical choices are described in the [manuscript, Section 2.1](../paper.md) and [Blumberg & Mellor (1987)](https://doi.org/10.1029/CO004p0001).

## Momentum and hydrostatic pressure

Let $x,y,z$ be eastward, northward, and upward coordinates, $t$ time, and $\boldsymbol u_h=(u,v)$ horizontal velocity. Let $w$ be vertical velocity, $f$ the Coriolis parameter, $\widehat{\boldsymbol k}$ the upward unit vector, $p$ pressure, $\rho$ density, $\rho_0$ constant reference density, and $g$ gravitational acceleration. With horizontal gradient $\nabla_h$, vertical viscosity $K_M$, and horizontal momentum mixing $\boldsymbol F_h$, the physical-coordinate equations are

$$
\left(\partial_t+\boldsymbol u_h\cdot\nabla_h+w\partial_z\right)\boldsymbol u_h
+f\widehat{\boldsymbol k}\times\boldsymbol u_h
=-\frac{1}{\rho_0}\nabla_h p
+\partial_z(K_M\partial_z\boldsymbol u_h)+\boldsymbol F_h,
$$

$$
\partial_zp=-\rho g,\qquad
\nabla_h\cdot\boldsymbol u_h+\partial_zw=0.
$$

Let $\eta$ denote free-surface elevation and $\rho'=\rho-\rho_0$ the density anomaly. With spatially uniform atmospheric pressure $p_{\rm atm}$, hydrostatic pressure is

$$
p(z)=p_{\rm atm}+\rho_0g(\eta-z)+g\int_z^{\eta}\rho'(z')\,\mathrm dz'.
$$

Its horizontal gradient contains the surface-slope contribution and a depth-dependent contribution from density. This connects tracer corrections to subsequent velocity evolution. The transformed pressure-gradient operators are implemented in [`momentum.py`](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/src/neuralpom/cores/explicit/momentum.py).

## Tracers and density

For tracer $C$, representing temperature $T$ or salinity $S$, let $K_H$ be vertical diffusivity and $F_C$ horizontal tracer mixing. The interior transport equation is

$$
\left(\partial_t+\boldsymbol u_h\cdot\nabla_h+w\partial_z\right)C
=\partial_z(K_H\partial_z C)+F_C.
$$

Surface and bottom fluxes enter through the selected case's boundary conditions. The model evaluates density from temperature, salinity, and pressure through the POM equation of state. The EOS reference is [Mellor (1991)](https://doi.org/10.1175/1520-0426(1991)008%3C0609:AEOSFN%3E2.0.CO;2); vertical mixing follows the separate [Mellor–Yamada closure](./turbulence.md).

## Follow the seabed with sigma coordinates

Let $H(x,y)>0$ be resting depth and $D=H+\eta$ the total water-column depth. Define

$$
\sigma=\frac{z-\eta}{H+\eta},\qquad -1\leq\sigma\leq0.
$$

The seabed is $\sigma=-1$ and the free surface is $\sigma=0$. The numerical operators account for this coordinate transformation in transport and pressure gradients. The vertical arrays `z` and `zz` store sigma interfaces and midpoints; `dz` and `dzz` store their separations. The published coarse configuration has 15 active layers within 16 vertical storage levels. ([Grid implementation](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/src/neuralpom/cores/explicit/grid.py), [model contract](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/src/neuralpom/hybrid/constants.py))

## Connect equations to the numerical model

| Physical operation | Implementation family | Explanation |
|---|---|---|
| Momentum and density | `momentum.py` | Pressure gradients, vertical momentum mixing, EOS |
| Tracer transport | `tracers.py` | Advection and vertical diffusion |
| Horizontal momentum and turbulence | `baroclinic.py` | Internal-mode operators and mixing closure |
| Free-surface evolution | `time_loop.py`, `implicit_barotropic.py` | [Explicit or semi-implicit external mode](../cores/overview.md) |
| Geometry | `grid.py` | [Staggered grid and masks](../numerics/arakawa-c-grid.md) |

The two core directories retain their own numerical implementation while sharing the hybrid state and correction interfaces. Continue with [mode splitting](./barotropic-baroclinic.md) to see why the fast surface mode receives special treatment.
