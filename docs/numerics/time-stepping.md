# Time Stepping

## Overview

NeuralPOM employs a split-explicit time-stepping scheme that separates the fast barotropic (external) mode from the slower baroclinic (internal) 3D mode. This approach, inherited from the Princeton Ocean Model (POM), uses a leapfrog time discretization for the internal mode with an Asselin-Robert time filter to control the computational mode.

## Time Scales and Step Splitting

The model operates with two distinct time steps:

| Parameter | Symbol | Description | Registered coarse-grid value |
|-----------|--------|-------------|------------------------------|
| External time step | `dte` | $\Delta t_e$ | 20 s |
| Internal time step | `dti` | $\Delta t_i$ | 600 s |
| Splitting ratio | `isplit` | $N_{split} = \Delta t_i / \Delta t_e$ | 30 |

The splitting ratio is defined as:

$$ \Delta t_i = N_{split} \cdot \Delta t_e $$

where $N_{split}$ is stored as `self.isplit` in the code. Derived quantities include:

```python
self.dte2 = self.dte * 2.0      # 2 * external time step
self.dti2 = self.dti * 2.0      # 2 * internal time step
self.ispi = 1.0 / self.isplit   # inverse splitting ratio
self.isp2i = 1.0 / (2 * self.isplit)  # half-inverse splitting ratio
```

## Leapfrog Time Discretization

The basic time integration uses a three-time-level leapfrog scheme centered at time level $n$:

$$ \frac{\partial \phi}{\partial t} \approx \frac{\phi^{n+1} - \phi^{n-1}}{2\Delta t} $$

For a generic prognostic equation $\partial_t \phi = F(\phi)$, the leapfrog discretization is:

$$ \phi^{n+1} = \phi^{n-1} + 2\Delta t \cdot F(\phi^n) $$

The leapfrog scheme is formally second-order accurate in time and is computationally efficient since it is explicit and requires only one evaluation of the right-hand side per time step.

### Three Time-Level Storage

NeuralPOM maintains three time levels for each prognostic variable:

| Time Level | Variable Suffix | Meaning |
|------------|----------------|---------|
| $n-1$ (backward) | `b` (e.g., `ub`, `vb`, `tb`, `elb`) | Previous time step |
| $n$ (current) | none (e.g., `u`, `v`, `t`, `el`) | Current time step |
| $n+1$ (forward) | `f` (e.g., `uf`, `vf`, `tf`, `elf`) | Next time step |

### Computational Mode and the Asselin-Robert Filter

The leapfrog scheme has a well-known weakness: it supports a spurious computational mode that oscillates with period $2\Delta t$ and can grow over time. To suppress this mode, NeuralPOM applies the Asselin-Robert time filter after each time step:

$$ \phi^n \leftarrow \phi^n + \frac{\nu}{2}\left(\phi^{n+1} + \phi^{n-1} - 2\phi^n\right) $$

where $\nu$ is the filter coefficient stored as `self.smoth`. Typical values are:
- Explicit core: `smoth = 0.10`
- Implicit core: `smoth = 0.15`

The filter is applied in the code as:

```python
# Barotropic variables (depth-averaged)
self.ua = self.ua + 0.5 * self.smoth * (self.uab - 2.0 * self.ua + self.uaf)
self.va = self.va + 0.5 * self.smoth * (self.vab - 2.0 * self.va + self.vaf)
self.el = self.el + 0.5 * self.smoth * (self.elb - 2.0 * self.el + self.elf)

# Baroclinic variables (3D fields)
u_filtered = self.u + 0.5 * self.smoth * (self.uf + self.ub - 2.0 * self.u - tps_u)
v_filtered = self.v + 0.5 * self.smoth * (self.vf + self.vb - 2.0 * self.v - tps_v)
```

For 3D velocity fields, the depth-mean is subtracted before filtering and restored afterward, following standard POM practice, to ensure that the depth-averaged transport is conserved during the filter step.

After filtering, the three time levels are shifted:
$$ \phi^{n-1} \leftarrow \phi^n \text{ (filtered)}, \quad \phi^n \leftarrow \phi^{n+1} $$

## CFL Constraints

### External Mode CFL

The fast barotropic mode involves surface gravity waves with phase speed $c = \sqrt{gH}$. The CFL condition for the external time step is:

$$ \Delta t_e < \frac{\min(\Delta x, \Delta y)}{\sqrt{g H_{max}}} $$

For the Double Gyre configuration ($\Delta x \approx 7$ km, $H_{max} = 2000$ m):
$$ \sqrt{gH_{max}} \approx \sqrt{9.81 \times 2000} \approx 140 \text{ m/s}, \quad \Delta t_e < \frac{7000}{140} \approx 50 \text{ s} $$

The registered integration settings are defined in `src/neuralpom/v1/core_spec.py`. See [Dynamical cores](../cores/overview.md) for the production implementations.

### Internal Mode CFL

The 3D internal mode is constrained by the maximum advective velocity and internal wave speed:

$$ \Delta t_i < \frac{\min(\Delta x, \Delta y)}{|u_{max}| + c_{int}} $$

where $c_{int} \approx 2-3$ m/s is the first-mode baroclinic wave speed. For velocities up to 1 m/s:
$$ \Delta t_i < \frac{7000}{3} \approx 2333 \text{ s} $$

The registered coarse-grid internal time step is **600 s**, comprising **30 external steps of 20 s**. These values are defined in `src/neuralpom/v1/core_spec.py`.

## Barotropic-Baroclinic Time Step Splitting

### Explicit Core: Sub-cycled External Mode

In the explicit core (`src/neuralpom/cores/explicit/time_loop.py`), the external mode is sub-cycled `isplit` times for each internal time step. The structure is:

```
for iint in range(1, iend + 1):           # Internal mode loop
    # 1. Apply surface forcing, boundary conditions
    # 2. Compute baroclinic terms (advection, pressure gradient, viscosity)

    for iext in range(1, isplit + 1):      # External mode sub-cycle
        # 2a. Compute depth-averaged fluxes
        # 2b. Update free surface (elf) using external time step

        # Every ispadv-th step: compute depth-averaged advection

        # 2c. Update depth-averaged velocity (uaf, vaf)
        # 2d. Apply boundary conditions
        # 2e. Time filter at final sub-steps

    # 3. Accumulate egf, utf, vtf (trapezoidal averages over external steps)
    # 4. Correct 3D velocity (remove barotropic, add averaged barotropic)
    # 5. Compute vertical velocity
    # 6. Solve turbulence (q2, q2l, km, kh)
    # 7. Advect and diffuse T/S
    # 8. Update density
    # 9. Solve 3D momentum with vertical diffusion
    # 10. Apply Asselin filter to all fields
```

The free-surface elevation `elf` is updated at each external sub-step using the 2D continuity equation:

$$ \eta_f(i,j) = \eta_b(i,j) - \frac{2\Delta t_e}{art(i,j)}\left[ \Phi_u(i+1,j) - \Phi_u(i,j) + \Phi_v(i,j+1) - \Phi_v(i,j) \right] - 2\Delta t_e \cdot vfluxf(i,j) $$

where $\Phi_u$, $\Phi_v$ are the depth-integrated transports at U- and V-faces.

The accumulated external mode contributions `egf`, `utf`, `vtf` are computed as trapezoidal averages over all `isplit` sub-steps:

```python
if iext != self.isplit:
    self.egf = self.egf + self.el * self.ispi
    self.utf[1:im, :] = self.utf[1:im, :] + self.ua[1:im, :] * d_sum_u * self.isp2i
    self.vtf[:, 1:jm] = self.vtf[:, 1:jm] + self.va[:, 1:jm] * d_sum_v * self.isp2i
```

These averages are then used to correct the 3D velocity:

$$ u_{corrected}(i,j,k) = u(i,j,k) - \langle u \rangle + \frac{utb + utf}{D(i,j) + D(i-1,j)} $$

### Implicit Core: Single-Step Semi-Implicit External Mode

In the implicit core (`src/neuralpom/cores/implicit_p0_fixed/time_loop.py`), the external mode is treated in a single step using a fractional step (predictor-corrector) approach:

```
for iint in range(1, iend + 1):           # Internal mode loop
    # 1. Apply surface forcing, boundary conditions
    # 2. Compute baroclinic terms

    # 3. Predictor step: compute provisional velocities U*, V* (explicit momentum
    #    minus terms depending on unknown future free surface)

    # 4. Form RHS of elliptic equation for eta^{n+1}

    # 5. Solve elliptic equation: L(eta) = rhs using CG/PCG

    # 6. Corrector step: update uaf, vaf using solved eta^{n+1}

    # 7. Apply Asselin filter (single step)

    # 8. Compute trapezoidal averages (egf, utf, vtf)

    # 9-14. Same baroclinic sequence as explicit core
```

The predictor step computes provisional depth-averaged velocities $U^*$, $V^*$ that exclude the implicit pressure gradient correction:

$$ U^* = \frac{D_b \cdot U_b - 4\Delta t_i \cdot F_{expl}}{D^n + D^{n-1}} $$

where $F_{expl}$ includes all explicit forcing terms (advection, Coriolis, explicit part of pressure gradient, wind stress, bottom friction) but excludes the $\alpha$-weighted implicit gradient of $\eta^{n+1}$.

The free surface is then solved implicitly (see [Elliptic Solver](./elliptic-solver.md)), and the corrector applies the missing pressure gradient:

$$ U^{n+1} = \frac{D_b \cdot U^* - 4\Delta t_i \cdot g \cdot \alpha \cdot \nabla\eta^{n+1} \cdot D \cdot A}{D^{n+1}} $$

## The `iint` Counter and Step Management

The simulation is driven by the `iint` loop counter:

```python
for iint in range(1, self.iend + 1):
    self.iint = iint
    self.time = self.dti * iint * 1.0 / 86400.0 + self.time0
```

Key parameters:
- `iend`: Total number of internal time steps, computed as:

  $$ iend = \max\left(\left\lfloor\frac{\mathrm{days} \times 86400}{\Delta t_i} + 0.5\right\rfloor, 2\right) $$

- `time0`: Initial model time offset in days (for restart runs).
- `time`: Current model time in days since the simulation start.
- `iprint`: Print interval in internal steps.

## Ramp Function

A smooth spin-up is implemented using a linear ramp for the first `period` days:

```python
if self.lramp:
    self.ramp = self.time / self.period
    if self.ramp > 1.0:
        self.ramp = 1.0
```

The ramp is used to gradually introduce surface forcing, reducing inertial shock during initialization.

Scientific references are listed in the [bibliography](../appendix/references.md); the production configuration is described in [Dynamical cores](../cores/overview.md).
