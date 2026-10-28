# Governing Equations

This page describes the fundamental equations solved by NeuralPOM, a neural differentiable implementation of the Princeton Ocean Model (POM). The model solves the three-dimensional primitive equations under the Boussinesq and hydrostatic approximations on a terrain-following sigma vertical coordinate system.

---

## Boussinesq Approximation

The Boussinesq approximation simplifies the Navier-Stokes equations by assuming that density variations are small relative to the background density $\rho_0$, and can therefore be neglected except in the buoyancy term of the vertical momentum equation. Formally:

$$
\rho = \rho_0 + \rho'(x, y, z, t), \qquad |\rho'| \ll \rho_0
$$

The reference density $\rho_0 = 1000$ kg/m$^3$ is used throughout. Density anomalies are handled by the equation of state, and the density field contributes only to the baroclinic pressure gradient and buoyancy production in the turbulence closure.

---

## Primitive Equations on a Rotating Sphere

### Horizontal Momentum Equations

The momentum balance on a rotating Earth (on an $f$-plane or $\beta$-plane) governs the evolution of the horizontal velocity components $(u, v)$:

$$
\frac{\partial u}{\partial t} + \vec{v} \cdot \nabla u - f v = -\frac{1}{\rho_0} \frac{\partial p}{\partial x} + \frac{\partial}{\partial z} \left( K_M \frac{\partial u}{\partial z} \right) + F_u
$$

$$
\frac{\partial v}{\partial t} + \vec{v} \cdot \nabla v + f u = -\frac{1}{\rho_0} \frac{\partial p}{\partial y} + \frac{\partial}{\partial z} \left( K_M \frac{\partial v}{\partial z} \right) + F_v
$$

where:

| Symbol | Description |
|--------|-------------|
| $u, v$ | Horizontal velocity components in the $x$ (eastward) and $y$ (northward) directions |
| $f$ | Coriolis parameter; $f = 2\Omega\sin\phi$ on a sphere, linear $f = f_0 + \beta y$ on a $\beta$-plane |
| $\rho_0$ | Reference density (1000 kg/m$^3$) |
| $p$ | Pressure |
| $K_M$ | Vertical eddy viscosity (from Mellor-Yamada 2.5 closure) |
| $F_u, F_v$ | Horizontal mixing terms (Smagorinsky or constant Laplacian diffusion) |
| $\vec{v} \cdot \nabla$ | Three-dimensional advection operator: $u \frac{\partial}{\partial x} + v \frac{\partial}{\partial y} + w \frac{\partial}{\partial z}$ |

In the code, these equations are implemented in `advu` and `advv` (`momentum.py`), where the terms are assembled as:

```python
uf = advx + vert_div + coriolis + press_grad + drhox
```

The time discretization uses a leapfrog scheme:

```python
uf[s_i, s_j, k_range] = (term_old - 2.0 * dti2 * uf[s_i, s_j, k_range]) / (denom + 1e-20)
```

where `term_old` represents $D^{n-1} u^{n-1}$, `dti2` is $2\Delta t$, and `denom` is $D^{n+1}$ (the total depth at the new time level).

### Hydrostatic Approximation

In ocean circulation, the vertical scale of motion ($H \sim 10^2$--$10^3$ m) is much smaller than the horizontal scale ($L \sim 10^4$--$10^6$ m). The hydrostatic approximation therefore replaces the vertical momentum equation with:

$$
\frac{\partial p}{\partial z} = -\rho g
$$

This yields the vertical distribution of pressure by integration from the surface:

$$
p(z) = p_{\text{atm}} + \rho_0 g \eta + g \int_z^{\eta} \rho \, dz'
$$

The pressure gradient is decomposed into barotropic (surface slope) and baroclinic (density) components:

$$
-\frac{1}{\rho_0} \nabla_h p = -g \nabla_h \eta - \frac{g}{\rho_0} \nabla_h \int_z^{\eta} \rho' \, dz'
$$

In the code, the baroclinic pressure gradient is computed in `baropg()` through vertical integration of density gradients with sigma-coordinate correction terms.

### Continuity Equation

The continuity equation under the Boussinesq approximation reduces to volume conservation:

$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z} = 0
$$

After sigma-coordinate transformation (see below), this becomes the equation for the free surface elevation $\eta$.

---

## Temperature and Salinity Conservation

The scalar transport equations for potential temperature $T$ and salinity $S$ are:

$$
\frac{\partial T}{\partial t} + \vec{v} \cdot \nabla T = \frac{\partial}{\partial z} \left( K_H \frac{\partial T}{\partial z} \right) + F_T
$$

$$
\frac{\partial S}{\partial t} + \vec{v} \cdot \nabla S = \frac{\partial}{\partial z} \left( K_H \frac{\partial S}{\partial z} \right) + F_S
$$

where:

| Symbol | Description |
|--------|-------------|
| $T$ | Potential temperature |
| $S$ | Salinity (psu) |
| $K_H$ | Vertical eddy diffusivity (from turbulence closure) |
| $F_T, F_S$ | Horizontal mixing terms |

These are solved using a conservative flux-form centered-difference scheme (`advt1`) or the Smolarkiewicz iterative upstream scheme (`advt2`) to suppress numerical diffusion. Vertical diffusion is solved implicitly via a Thomas algorithm in `proft()`.

---

## Equation of State

NeuralPOM implements the Mellor (1991) seawater equation of state. The density is computed as a function of potential temperature, salinity, and pressure:

$$
\rho = \rho(T, S, p)
$$

The computation in `dens()` follows the polynomial form:

```python
# Atmospheric pressure contribution
rhor = -0.157406 + 6.793952e-2 * tr - 9.095290e-3 * tr^2
     + 1.001685e-4 * tr^3 - 1.120083e-6 * tr^4 + 6.536332e-9 * tr^5

# Salinity contribution
rhor += (0.824493 - 4.0899e-3 * tr + 7.6438e-5 * tr^2
         - 8.2467e-7 * tr^3 + 5.3875e-9 * tr^4) * sr

# Salinity^1.5 terms
rhor += (-5.72466e-3 + 1.0227e-4 * tr - 1.6546e-6 * tr^2) * |sr|^{1.5}

# Salinity^2 terms
rhor += 4.8314e-4 * sr^2
```

The pressure correction uses the speed of sound $c_s$:

$$
\rho = \rho_0 + \frac{10^5 p}{c_s^2} \left(1 - \frac{2p}{c_s^2}\right)
$$

The final density anomaly is output as $(\rho - 1000) / \rho_{\text{ref}}$:

```python
self.rho[:, :, k_sl] = (rhor / self.rhoref) * _b(self.fsm)
```

---

## Sigma-Coordinate Transformation

The vertical coordinate is transformed from the physical vertical coordinate $z$ to the terrain-following sigma coordinate $\sigma$:

$$
\sigma = \frac{z - \eta}{H + \eta}, \qquad -1 \leq \sigma \leq 0
$$

where:

| Symbol | Description |
|--------|-------------|
| $z$ | Physical vertical coordinate (positive upward, $z=0$ at mean sea level) |
| $\eta(x, y, t)$ | Free surface elevation |
| $H(x, y)$ | Resting water depth |
| $\sigma$ | Sigma coordinate ($\sigma = 0$ at surface, $\sigma = -1$ at bottom) |

This transformation maps the variable ocean domain to a fixed computational domain, simplifying the implementation of surface and bottom boundary conditions. The sigma levels are defined through the `z` and `zz` arrays:

```python
# z(k): vertical position of sigma level k
# zz(k): vertical spacing between sigma levels (for finite differences)
```

The transformation introduces metric correction terms in the baroclinic pressure gradient computation to avoid the "sigma coordinate pressure gradient error" over steep topography.

---

## Vertical Boundary Conditions

### Surface Boundary Conditions

**Momentum (wind stress):**

$$
K_M \frac{\partial u}{\partial z}\bigg|_{z=\eta} = \frac{\tau_s^x}{\rho_0}, \qquad
K_M \frac{\partial v}{\partial z}\bigg|_{z=\eta} = \frac{\tau_s^y}{\rho_0}
$$

where $\tau_s^x, \tau_s^y$ are the surface wind stress components provided by `wusurf` and `wvsurf`.

**Tracers (surface heat and freshwater flux):**

$$
K_H \frac{\partial T}{\partial z}\bigg|_{z=\eta} = \frac{Q_{\text{net}}}{\rho_0 c_p}, \qquad
K_H \frac{\partial S}{\partial z}\bigg|_{z=\eta} = S_0 (E - P)
$$

**Turbulence (boundary conditions for $q^2$):**

Surface boundary condition for TKE is proportional to the friction velocity $u_*$:

$$
q^2|_{z=\eta} = (15.8 C_{\text{bc}})^{2/3} u_*^2
$$

### Bottom Boundary Conditions

**Momentum (bottom friction):**

$$
K_M \frac{\partial u}{\partial z}\bigg|_{z=-H} = C_d |\vec{v}_b| u_b, \qquad
K_M \frac{\partial v}{\partial z}\bigg|_{z=-H} = C_d |\vec{v}_b| v_b
$$

where $C_d$ is the bottom drag coefficient (`cbc`), and $|\vec{v}_b|$ is the bottom velocity magnitude. The implementation in `profu`/`profv` computes:

```python
mag_vel = safe_sqrt(u_sq + v_avg_sq)
tps_local[s_i_in, s_j_in] = cbc_avg * mag_vel
wubot[s_i_in, s_j_in] = -tps_local[s_i_in, s_j_in] * uf_bot_slice
```

**Turbulence:**

Bottom boundary condition for $q^2l$:

$$
q^2 l|_{z=-H} = 0
$$

---

## Numerical Framework

### Spatial Discretization

NeuralPOM uses the Arakawa C-grid on a curvilinear orthogonal horizontal grid:

- $u$ and $v$ velocity components are staggered with respect to the tracer/scalar grid
- `aru` and `arv` are area factors at $u$ and $v$ points
- `dt` (the metric factor) represents $\sqrt{g_{xx}}$ at tracer points
- `dx`, `dy` are grid spacing arrays

### Temporal Discretization

The model uses a leapfrog time integration scheme with an Asselin time filter to suppress the computational mode. The baroclinic and barotropic modes are split, with the barotropic time step typically much smaller (see [Barotropic-Baroclinic Mode Splitting](./barotropic-baroclinic)).

### Python Implementation

The governing equations are implemented as PyTorch tensor operations in the following modules:

| Module | File | Contents |
|--------|------|----------|
| `MomentumMixin` | `momentum.py` | `advu`, `advv`, `profu`, `profv`, `dens`, `baropg` |
| `BaroclinicMixin` | `baroclinic.py` | `advct` (horizontal momentum advection) |
| `TracersMixin` | `tracers.py` | `advt1`, `advt2`, `proft`, `smol_adif` |

All tensor operations are fully differentiable, enabling gradient-based optimization and neural network coupling.

Scientific references are listed in the [bibliography](../appendix/references.md); the production configuration is described in [Dynamical cores](../cores/overview.md).
