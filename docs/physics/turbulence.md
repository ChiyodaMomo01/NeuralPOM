# Vertical turbulence closure

**Vertical mixing connects shear, stratification, momentum, and tracers.** NeuralPOM retains the POM Mellor–Yamada level-2.5 closure in the physical integration. The neural corrector updates resolved velocity and tracer fields; the closure responds as the model advances. ([Mellor & Yamada, 1982](https://doi.org/10.1029/RG020i004p00851), [manuscript, Section 2.1](../paper.md))

## Prognostic turbulence quantities

Let $q^2$ be twice the turbulent kinetic energy per unit mass and $l$ a turbulence length scale. The closure evolves $q^2$ and $q^2l$, balancing transport, shear production, buoyancy effects, and dissipation. These fields, their histories, and the resulting mixing coefficients form part of the complete restart state.

For dimensionless stability functions $S_M$ and $S_H$, the closure relates the velocity scale $q$ and length scale $l$ to vertical viscosity $K_M$ and diffusivity $K_H$:

$$
K_M=qlS_M,\qquad K_H=qlS_H.
$$

The implemented operators also include background mixing, time averaging, boundary conditions, and numerical limits. Their definitions are in [`baroclinic.py`](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/src/neuralpom/cores/explicit/baroclinic.py).

## Shear and stratification have different effects

Let $u,v$ be horizontal velocity components and $z$ the upward coordinate. Shear production $P_s$ is

$$
P_s=K_M\left[(\partial_z u)^2+(\partial_z v)^2\right].
$$

Let $g$ be gravitational acceleration, $\rho_0$ reference density, and $\rho$ the density profile. The Boussinesq buoyancy frequency satisfies $N^2=-(g/\rho_0)\partial_z\rho$. In the usual TKE sign convention, the downgradient buoyancy contribution is $-K_HN^2$: stable stratification consumes turbulent energy, while unstable stratification can produce it. The POM implementation evaluates its buoyancy term with its EOS and compressibility treatment. ([Mellor & Yamada, 1982](https://doi.org/10.1029/RG020i004p00851), [physical operators](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/src/neuralpom/cores/explicit/baroclinic.py))

## Why this matters to learning

A tracer correction changes density; a velocity correction changes shear. Both can change the vertical mixing produced in the next physical interval. During Midtrain, the relevant physical response lies inside the gradient path. The model consequently learns with the dynamics that act on its corrected fields.

The vertical diffusion and closure solves are part of the [differentiation checks](../cores/gradient-safety.md). To follow the tracer-to-density connection, continue to [density-normal damping](../hybrid/density-normal-damping.md).
