# Arakawa-C Grid

## Overview

The Arakawa-C grid is a staggered horizontal grid arrangement widely used in geophysical fluid dynamics models, including the Princeton Ocean Model (POM) family on which NeuralPOM is based. In the Arakawa-C staggering, scalar quantities (temperature, salinity, density, pressure, free surface elevation) are placed at cell centers, while the velocity components are positioned at cell faces normal to their respective directions. This staggering provides excellent conservation properties for energy, enstrophy, and mass, and naturally avoids the pressure-velocity decoupling (checkerboard) problem that plagues collocated (Arakawa-A) grids.

NeuralPOM implements the Arakawa-C grid in both its explicit and implicit cores through the `GridMixin` class (located at `src/neuralpom/cores/explicit/grid.py` and `src/neuralpom/cores/implicit_p0_fixed/grid.py`).

## Grid Indexing Convention

The grid uses three spatial indices:

| Index | Direction   | Range (1-based, MATLAB style) | Python (0-based) |
|-------|-------------|-------------------------------|-------------------|
| `i`   | East-West   | `1, 2, ..., im`              | `0, 1, ..., im-1` |
| `j`   | North-South | `1, 2, ..., jm`              | `0, 1, ..., jm-1` |
| `k`   | Vertical    | `1, 2, ..., kb`              | `0, 1, ..., kb-1` |

The code defines derived indices for boundary handling:

```python
self.imm1 = self.im - 1    # im minus 1
self.imm2 = self.im - 2    # im minus 2
self.jmm1 = self.jm - 1    # jm minus 1
self.jmm2 = self.jm - 2    # jm minus 2
self.kbm1 = self.kb - 1    # kb minus 1
self.kbm2 = self.kb - 2    # kb minus 2
```

Ghost cells exist along all four horizontal boundaries (rows `0` and `im-1`, columns `0` and `jm-1`) to apply boundary conditions. The interior domain for computations is `s_i = slice(1, self.imm1)` and `s_j = slice(1, self.jmm1)` -- i.e., rows `1` through `im-2` and columns `1` through `jm-2`.

## Variable Placement on the Staggered Grid

### Cell Center Quantities (T-points)

Scalar variables reside at the geometric center of each grid cell, indexed by `(i, j, k)`:

| Variable | Symbol | Description |
|----------|--------|-------------|
| Free surface elevation | `el`, `elf`, `elb` | $\eta$ (current, future, backward) |
| Total depth | `dt`, `d` | $D = h + \eta$ |
| Potential temperature | `t` | $T$ |
| Salinity | `s` | $S$ |
| Density anomaly | `rho` | $\rho$ |
| Turbulent kinetic energy | `q2` | $q^2$ |
| TKE $\times$ length scale | `q2l` | $q^2 l$ |
| Vertical eddy viscosity | `km` | $K_M$ |
| Vertical eddy diffusivity | `kh` | $K_H$ |
| Vertical eddy diffusivity (TKE) | `kq` | $K_q$ |
| Horizontal viscosity | `aam` | $A_M$ |
| Coriolis parameter | `cor` | $f$ |

### East-West Face Quantities (U-points)

The zonal (east-west) velocity components are placed at the centers of cell faces normal to the $x$-direction, indexed by `(i, j, k)` but offset by half a grid cell in the $i$ direction:

| Variable | Symbol | Description |
|----------|--------|-------------|
| 3D zonal velocity | `u`, `uf`, `ub` | $u$ (current, future, backward) |
| Depth-averaged zonal velocity | `ua`, `uaf`, `uab` | $\bar{u}$ |

These are computed at horizontal positions midway between T-points $(i,j)$ and $(i-1,j)$. The U-momentum equation is evaluated using area elements `aru` and masks `dum`.

### North-South Face Quantities (V-points)

The meridional (north-south) velocity components are placed at the centers of cell faces normal to the $y$-direction, indexed by `(i, j, k)` but offset by half a grid cell in the $j$ direction:

| Variable | Symbol | Description |
|----------|--------|-------------|
| 3D meridional velocity | `v`, `vf`, `vb` | $v$ (current, future, backward) |
| Depth-averaged meridional velocity | `va`, `vaf`, `vab` | $\bar{v}$ |

These are computed at horizontal positions midway between T-points $(i,j)$ and $(i,j-1)$. The V-momentum equation uses `arv` and `dvm`.

### Top-Bottom Face Quantities (W-points)

The vertical velocity $w$ is placed at the centers of cell faces normal to the $z$-direction (sigma-level interfaces):

| Variable | Symbol | Description |
|----------|--------|-------------|
| Vertical velocity | `w` | $w$ on sigma interfaces |

In the sigma-coordinate system, $w = 0$ at the surface ($k=0$) and $w = 0$ at the bottom ($k=kb-1$).

## Grid Metrics

### Horizontal Grid Spacing

The grid spacings `dx` and `dy` are 2D arrays `(im, jm)` representing the physical distance between adjacent grid points:

```python
self.dx  = torch.ones((im, jm), device=self.device)  # East-West spacing [m]
self.dy  = torch.ones((im, jm), device=self.device)  # North-South spacing [m]
```

For spherical coordinates, these are set based on latitude and longitude resolution. For the default Double Gyre configuration, the resolution is approximately $1/16^\circ$.

### Cell Areas

Three distinct area arrays are computed for the staggered grid:

**T-cell area** (cell centers):
$$ art(i,j) = dx(i,j) \cdot dy(i,j) $$

**U-cell area** (east-west faces):
$$ aru(i,j) = \frac{1}{4}(dx_i + dx_{i-1})(dy_i + dy_{i-1}) \quad \text{for } i=1,\dots,im-1 $$

**V-cell area** (north-south faces):
$$ arv(i,j) = \frac{1}{4}(dx_i + dx_i)(dy_j + dy_{j-1}) \quad \text{for } j=1,\dots,jm-1 $$

Boundary values are extrapolated from the interior:
```python
self.aru[0, :] = self.aru[1, :]
self.arv[0, :] = self.arv[1, :]
self.aru[:, 0] = self.aru[:, 1]
self.arv[:, 0] = self.arv[:, 1]
```

### Vertical Grid (Sigma Coordinates)

The vertical coordinate system uses terrain-following sigma levels, normalized so that $z = 0$ at the surface and $z = -1$ at the bottom. The grid is generated by the `depth()` method in `GridMixin`.

Layer interfaces `z` (at cell faces) and midpoints `zz` (at cell centers) are computed:

$$ zz(k) = \frac{1}{2}\left[z(k) + z(k+1)\right] \quad k = 0,\dots,kb-2 $$

The layer thicknesses are:
$$ dz(k) = z(k) - z(k+1) $$
$$ dzz(k) = zz(k) - zz(k+1) $$

The vertical distribution has three zones:
1. **Surface logarithmic layers** ($k = 0,\dots,kl1-1$): thicknesses follow a doubling sequence $(1, 1, 2, 4, 8, \dots)$.
2. **Middle linear layers** ($k = kl1,\dots,kl2-1$): constant thickness $\Delta z$.
3. **Bottom logarithmic layers** ($k = kl2,\dots,kb-1$): symmetrically decreasing thickness.

The total depth is then normalized to $[-1, 0]$:
$$ z(k) = -\frac{z(k)}{z(kb-1)} $$

## Coriolis Parameter

The Coriolis parameter $f$ is stored at T-points (cell centers) and is interpolated to U and V faces as needed during momentum calculations:

$$ f = 2\Omega \sin\phi $$

where $\Omega = 7.2921 \times 10^{-5}$ rad/s is the Earth's rotation rate and $\phi$ is the latitude.

In the momentum equations, the Coriolis terms use 4-point averaging to obtain values at the appropriate staggered locations. For example, the Coriolis term in the U-momentum equation averages $v$ to the U-point:

```python
cor_term = 0.25 * (
    cor(i,j) * dt(i,j) * (v(i,j+1) + v(i,j)) +
    cor(i-1,j) * dt(i-1,j) * (v(i-1,j+1) + v(i-1,j))
)
```

## Land/Sea Masking

Land-sea masking is handled through three mask arrays:

| Mask | Grid Position | Meaning |
|------|---------------|---------|
| `fsm` | T-points | Free surface mask: $1$ where $h > 1$ m (water), $0$ otherwise (land) |
| `dum` | U-points | Velocity mask for $u$: $dum(i,j) = fsm(i,j) \cdot fsm(i-1,j)$ |
| `dvm` | V-points | Velocity mask for $v$: $dvm(i,j) = fsm(i,j) \cdot fsm(i,j-1)$ |

These masks ensure zero velocity at land boundaries and prevent fluxes through land cells. A velocity point is active only if both adjacent T-points are water cells.

```python
self.fsm = torch.where(self.h > 1.0, 1.0, 0.0)
self.dum[1:im, 1:jm] = self.fsm[1:im, 1:jm] * self.fsm[0:imm1, 1:jm]
self.dvm[1:im, 1:jm] = self.fsm[1:im, 1:jm] * self.fsm[1:im, 0:jmm1]
```

## Bottom Topography Smoothing

The method `slpmax()` smooths the bottom topography $h(i,j)$ to limit the cell-to-cell slope ratio. The constraint enforced is:

$$ \frac{|h_1 - h_2|}{h_1 + h_2} \leq slmax $$

Four directional sweeps (right, left, up, down) are applied iteratively (10 passes). When the slope constraint is violated at a pair of adjacent cells, both depths are adjusted to satisfy the limit while preserving their mean:

$$ h_1 = \bar{h}(1 - slmax \cdot \mathrm{sgn}(h_2 - h_1)) $$
$$ h_2 = \bar{h}(1 + slmax \cdot \mathrm{sgn}(h_2 - h_1)) $$

where $\bar{h} = (h_1 + h_2)/2$.

## Grid Initialization in Python

The `GridMixin` class in both `src/neuralpom/cores/explicit/grid.py` and `src/neuralpom/cores/implicit_p0_fixed/grid.py` provides the following methods for grid initialization:

- `depth()`: Generates the vertical sigma-coordinate arrays `(z, zz, dz, dzz)`.
- `areas_masks()`: Computes cell areas `(art, aru, arv)` and land-sea masks `(fsm, dum, dvm)`.
- `slpmax(slmax)`: Smooths bottom topography to enforce the slope constraint.

Grid initialization follows this sequence within `run_initialization()`:
1. Set horizontal grid spacing `dx`, `dy` based on the case configuration.
2. Set the Coriolis parameter `cor` from latitude.
3. Read or set the bottom topography `h`.
4. Call `slpmax()` to smooth the bathymetry.
5. Call `depth()` to build the vertical grid.
6. Call `areas_masks()` to compute areas and masks.
7. Initialize `d = h + el` (total water depth) and `dt = h + et`.

## Advantages of Arakawa-C for Geophysical Fluid Dynamics

1. **Natural representation of pressure gradients**: Scalar quantities (pressure, density) are naturally centered, while the pressure gradient force is computed using differences of scalars at velocity points, avoiding the 2$\Delta x$ computational mode.

2. **Mass and tracer conservation**: The continuity equation is evaluated at T-points, and fluxes through cell faces are computed directly from the face-normal velocities $u$, $v$, $w$, ensuring exact conservation.

3. **Coriolis term accuracy**: The staggered arrangement allows the Coriolis terms to be computed from the appropriate average of the cross-face velocity at each momentum point.

4. **Geostrophic balance**: The Arakawa-C grid accurately represents geostrophic balance, which is fundamental to large-scale ocean circulation.

5. **Gravity wave dispersion**: The staggering correctly captures the dispersion relation for both external (barotropic) and internal (baroclinic) gravity waves.

6. **Computational efficiency**: With all quantities in their natural grid positions, minimal interpolation is required, reducing computational overhead and numerical diffusion.

The Arakawa-C grid has been the standard for ocean circulation models for decades and is used in POM, ROMS, MITgcm, NEMO, and many other community ocean models.

Scientific references are listed in the [bibliography](../appendix/references.md); the production configuration is described in [Dynamical cores](../cores/overview.md).
