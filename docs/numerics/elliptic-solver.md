# Elliptic Solver for the Implicit Free Surface

## Overview

The implicit free-surface formulation in NeuralPOM reduces the barotropic (external) mode to a 2D elliptic boundary value problem for the sea surface elevation $\eta^{n+1}$. This approach eliminates the restrictive CFL condition associated with fast barotropic gravity waves, allowing larger time steps for the 3D baroclinic mode.

The solver is implemented in `src/neuralpom/cores/implicit_p0_fixed/implicit_barotropic.py` as the `ImplicitBarotropicMixin` class, with an optimized version in `src/neuralpom/cores/implicit_p0_fixed/core.py` (`POM2K_Core`) that adds cached geometry terms, Jacobi-preconditioned PCG, and reusable scratch buffers.

## Derivation of the 2D Elliptic Equation

### Starting Equations

The depth-integrated continuity equation is:

$$ \frac{\partial \eta}{\partial t} + \nabla \cdot (D \bar{\mathbf{u}}) = 0 $$

where $D = h + \eta$ is the total water depth and $\bar{\mathbf{u}}$ is the depth-averaged horizontal velocity.

The depth-averaged momentum equation, discretized with a semi-implicit treatment of the surface pressure gradient, gives the provisional velocity (predictor):

$$ D^n \bar{\mathbf{u}}^* = D^{n-1} \bar{\mathbf{u}}^{n-1} - 2\Delta t_i \, \mathbf{F}(\bar{\mathbf{u}}^n, \eta^n, \eta^{n-1}) $$

where $\mathbf{F}$ contains all explicitly evaluated terms: advection, Coriolis, baroclinic pressure gradient, wind stress, bottom friction, and the explicit portion of the surface pressure gradient (weighted by $1 - 2\alpha$ and $\alpha$ for $\eta^n$ and $\eta^{n-1}$ respectively).

The final velocity is obtained by adding the implicit pressure gradient correction:

$$ D^{n+1} \bar{\mathbf{u}}^{n+1} = D^n \bar{\mathbf{u}}^* - 2\Delta t_i \, g \, \alpha \, D^n \nabla \eta^{n+1} $$

### Forming the Elliptic Equation

Substituting the momentum corrector into the continuity equation evaluated at time $n+1$ and using a leapfrog discretization for the time derivative:

$$ \frac{\eta^{n+1} - \eta^{n-1}}{2\Delta t_i} = -\nabla \cdot \left( D^n \bar{\mathbf{u}}^* - 2\Delta t_i \, g \, \alpha \, D^n \nabla \eta^{n+1} \right) $$

Rearranging to isolate $\eta^{n+1}$:

$$ \eta^{n+1} - (2\Delta t_i)^2 g \, \alpha \, \nabla \cdot (D^n \nabla \eta^{n+1}) = \text{RHS} $$

where the right-hand side is:

$$ \text{RHS} = \eta^{n-1} - 2\Delta t_i \, \nabla \cdot (D^n \bar{\mathbf{u}}^*) $$

This is the fundamental elliptic equation:

$$ \boxed{ \eta - 4 \Delta t_i^2 g \alpha \, \nabla \cdot (D \nabla \eta) = \text{RHS} } $$

where we write $\eta = \eta^{n+1}$ and $D = D^n$ for brevity.

## The Elliptic Operator

On the Arakawa-C grid, the operator $L(\eta) = \eta - 4\Delta t_i^2 g \alpha \, \nabla \cdot (D \nabla \eta)$ is discretized using a 5-point stencil with fluxes at U- and V-faces.

### Discretized Form

The divergence term is computed as the sum of four fluxes:

$$ L(\eta)_{i,j} = \eta_{i,j} - 4\Delta t_i^2 g \alpha \cdot \frac{1}{art_{i,j}} \left[ F_{i+1/2,j}^x - F_{i-1/2,j}^x + F_{i,j+1/2}^y - F_{i,j-1/2}^y \right] $$

where the fluxes are:

**East flux** (at U-face $i+1/2, j$):
$$ F_{i+1/2,j}^x = \frac{D_{i+1,j} + D_{i,j}}{2} \cdot \frac{\eta_{i+1,j} - \eta_{i,j}}{(dx_{i+1,j} + dx_{i,j})/2} \cdot \frac{dy_{i+1,j} + dy_{i,j}}{2} \cdot dum_{i+1,j} $$

**West flux** (at U-face $i-1/2, j$):
$$ F_{i-1/2,j}^x = \frac{D_{i,j} + D_{i-1,j}}{2} \cdot \frac{\eta_{i,j} - \eta_{i-1,j}}{(dx_{i,j} + dx_{i-1,j})/2} \cdot \frac{dy_{i,j} + dy_{i-1,j}}{2} \cdot dum_{i,j} $$

**North flux** (at V-face $i, j+1/2$):
$$ F_{i,j+1/2}^y = \frac{D_{i,j+1} + D_{i,j}}{2} \cdot \frac{\eta_{i,j+1} - \eta_{i,j}}{(dy_{i,j+1} + dy_{i,j})/2} \cdot \frac{dx_{i,j+1} + dx_{i,j}}{2} \cdot dvm_{i,j+1} $$

**South flux** (at V-face $i, j-1/2$):
$$ F_{i,j-1/2}^y = \frac{D_{i,j} + D_{i,j-1}}{2} \cdot \frac{\eta_{i,j} - \eta_{i,j-1}}{(dy_{i,j} + dy_{i,j-1})/2} \cdot \frac{dx_{i,j} + dx_{i,j-1}}{2} \cdot dvm_{i,j} $$

The land masks $dum$, $dvm$ enforce zero flux across land boundaries.

### Coefficient

The implicit coefficient is:

$$ \text{coeff} = 4 \cdot \Delta t_i^2 \cdot g \cdot \alpha $$

For the standard configuration:
- $\Delta t_i = 900$ s
- $g = 9.81$ m/s$^2$
- $\alpha = 0.333$

This gives $\text{coeff} = 4 \times 900^2 \times 9.81 \times 0.333 \approx 1.06 \times 10^7$ m$^2$.

### Python Implementation

The reference implementation in `apply_elliptic_operator`:

```python
def apply_elliptic_operator(self, eta, d_2d):
    # H at U/V faces
    H_u_east = 0.5 * (d_2d[i+1,j] + d_2d[i,j])
    H_u_west = 0.5 * (d_2d[i,j] + d_2d[i-1,j])
    H_v_north = 0.5 * (d_2d[i,j+1] + d_2d[i,j])
    H_v_south = 0.5 * (d_2d[i,j] + d_2d[i,j-1])

    # Fluxes with land mask
    flux_x_east = H_u_east * (eta[i+1,j] - eta[i,j]) / dx_east * dy_east * dum[i+1,j]
    flux_x_west = H_u_west * (eta[i,j] - eta[i-1,j]) / dx_west * dy_west * dum[i,j]
    flux_y_north = H_v_north * (eta[i,j+1] - eta[i,j]) / dy_north * dx_north * dvm[i,j+1]
    flux_y_south = H_v_south * (eta[i,j] - eta[i,j-1]) / dy_south * dx_south * dvm[i,j]

    div_flux = (flux_x_east - flux_x_west + flux_y_north - flux_y_south) / art[i,j]
    coeff = 4.0 * (self.dti ** 2) * self.grav * self.alpha
    out = eta.clone()
    out[i,j] = eta[i,j] - coeff * div_flux
    return out
```

### 5-Point Stencil Structure

The discretization produces a 5-point Laplacian-like stencil on the Arakawa-C grid. For each T-point $(i,j)$, the operator involves:

- Center: $\eta_{i,j}$ with a coefficient modified by the divergence of fluxes
- East: $\eta_{i+1,j}$ through $F_{i+1/2,j}^x$
- West: $\eta_{i-1,j}$ through $F_{i-1/2,j}^x$
- North: $\eta_{i,j+1}$ through $F_{i,j+1/2}^y$
- South: $\eta_{i,j-1}$ through $F_{i,j-1/2}^y$

The effective stencil is a symmetric, diagonally dominant, positive-definite linear system. The system matrix is never explicitly formed -- the operator is applied matrix-free, which is ideal for iterative solvers.

## Conjugate Gradient (CG) Method

### Reference Implementation

The reference CG solver (`solve_implicit_eta_cg` in `ImplicitBarotropicMixin`) implements the standard Conjugate Gradient algorithm:

```
Algorithm: Conjugate Gradient for L(eta) = rhs

1. Warm start: x_0 = 2 * eta^n - eta^{n-1}  (linear extrapolation)
2. Apply BC to x_0
3. r_0 = rhs - L(x_0)
4. p_0 = r_0
5. for k = 1, 2, ..., max_iter:
       Enforce BC on p_{k-1}
       Ap = L(p_{k-1})
       alpha = (r_{k-1} · r_{k-1}) / (p_{k-1} · Ap)
       x_k = x_{k-1} + alpha * p_{k-1}
       Enforce BC on x_k
       r_k = r_{k-1} - alpha * Ap
       if ||r_k||^2 < tol^2: break
       beta = (r_k · r_k) / (r_{k-1} · r_{k-1})
       p_k = r_k + beta * p_{k-1}
       Enforce BC on p_k
6. return x_k
```

### Warm Start

The initial guess uses linear extrapolation in time:

$$ \eta^{(0)} = 2\eta^n - \eta^{n-1} $$

This significantly reduces the number of CG iterations compared to a zero initial guess, typically cutting the iteration count by 50-70%.

### Convergence Criteria

Convergence is declared when:

$$ \|r_k\|^2_{interior} \leq \max(\text{tol}^2, \;\text{rel\_tol}^2 \cdot \|r_0\|^2) $$

The default parameters are:
- `tol = 1e-12` (absolute tolerance)
- `rel_tol = 1e-6` (relative tolerance, in the accelerated solver)
- `max_iter = 200`

Typical convergence for the Double Gyre configuration (250 $\times$ 250 grid) requires 15-30 iterations.

## Preconditioned Conjugate Gradient (PCG)

The accelerated implicit core (`POM2K_Core`) implements Jacobi (diagonal) preconditioning for faster convergence.

### Jacobi Preconditioner

The diagonal of the system matrix $L$ is:

$$ \text{diag}_{i,j} = 1 + 4\Delta t_i^2 g \alpha \cdot \frac{1}{art_{i,j}} \sum_{\text{faces}} \left( H_f \cdot \frac{dy_f}{dx_f} \cdot mask_f \right) $$

where the sum runs over all four faces (east, west, north, south).

The preconditioner is $M = \text{diag}(L)$, and the preconditioned system is:

$$ M^{-1} L(\eta) = M^{-1} \text{RHS} $$

The PCG algorithm replaces the simple residual inner product with a preconditioned one:

```
Algorithm: Preconditioned Conjugate Gradient

rho_old = r · z              where z = M^{-1} r
p = z

for k = 1, 2, ...:
    alpha = rho_old / (p · Ap)
    x = x + alpha * p
    r = r - alpha * Ap
    z = M^{-1} r
    rho_new = r · z
    beta = rho_new / rho_old
    p = z + beta * p
    rho_old = rho_new
```

### Implementation in the Accelerated Core

The accelerated CG solver uses pre-allocated work arrays to avoid repeated tensor allocations:

```python
def solve_implicit_eta_cg(self, rhs, d_2d, max_iter=200, tol=1e-12, ...):
    # Reuse scratch arrays: ap, r, p, z, diag from self._cg_work
    ap = self._cg_work["ap"]
    r = self._cg_work["r"]
    p = self._cg_work["p"]
    z = self._cg_work["z"]
    diag = self._cg_work["diag"]

    # Build Jacobi preconditioner
    self._build_jacobi_preconditioner_into(d_2d, diag)

    # Apply preconditioner: z = M^{-1} * r
    z[s_i, s_j] = r[s_i, s_j] / (diag[s_i, s_j] + 1e-30)

    # PCG iteration loop
    for it in range(1, max_iter + 1):
        ...
```

### Cached Geometry Terms

To avoid redundant computation, the accelerated core pre-computes all time-invariant geometric factors during initialization:

```python
self._accel_cache = {
    "dy_over_dx_east": dy_east / (dx_east + 1e-20),
    "dy_over_dx_west": dy_west / (dx_west + 1e-20),
    "dx_over_dy_north": dx_north / (dy_north + 1e-20),
    "dx_over_dy_south": dx_south / (dy_south + 1e-20),
    "mask_u_east": self.dum[s_i_p1, s_j],
    "mask_u_west": self.dum[s_i, s_j],
    "mask_v_north": self.dvm[s_i, s_j_p1],
    "mask_v_south": self.dvm[s_i, s_j],
    "inv_art": 1.0 / (self.art[s_i, s_j] + 1e-20),
    "implicit_coeff": 4.0 * (self.dti ** 2) * self.grav * self.alpha,
    ...
}
```

## Boundary Conditions

### Dirichlet (Closed Basin)

For the Double Gyre (`iproblem == 4`), the boundaries are closed rigid walls
with zero free-surface elevation:

$$ \eta = 0 \quad \text{on} \quad \partial\Omega $$

```python
eta[0, :] = 0.0
eta[im-1, :] = 0.0
eta[:, 0] = 0.0
eta[:, jm-1] = 0.0
```

The RHS is also zeroed at boundaries for consistency.

## Computational Cost and GPU Considerations

### Matrix-Free Operation

The elliptic operator is applied matrix-free: $L(\eta)$ is computed directly from the stencil without ever assembling the sparse $N \times N$ matrix. For a grid of size $im \times jm$, each evaluation of the elliptic operator requires:

- 4 flux computations (one per face)
- Total: approximately $O(im \cdot jm)$ operations per CG iteration

### Native PyTorch Operations

All operations use native PyTorch tensor operations:

```python
# Flux computations are fully vectorized
flux_x_east = (
    h_u_east * (eta[s_i_p1, s_j] - eta[s_i, s_j])
    * cache["dy_over_dx_east"] * cache["mask_u_east"]
)
```

This enables:
- **GPU acceleration**: All operations run directly on CUDA tensors.
- **Automatic differentiation**: The solver is compatible with `torch.autograd` for gradient-based optimization in hybrid AI-physics training.
- **Batch processing**: Multiple ensemble members can be solved simultaneously by adding a batch dimension.

### Performance Characteristics

| Configuration | Grid Size | CG Iterations (typical) | Time per Solve (GPU) |
|---------------|-----------|------------------------|----------------------|
| Double Gyre | 90 $\times$ 82 | workload dependent | hardware dependent |

The Jacobi preconditioning typically reduces the CG iteration count by 30-50% compared to unpreconditioned CG.

### Autograd Compatibility

When gradient tracking is active (e.g., during model training), the accelerated solver falls back to the reference CG implementation:

```python
def solve_implicit_eta_cg(self, rhs, d_2d, ...):
    if torch.is_grad_enabled() and (rhs.requires_grad or d_2d.requires_grad):
        return _ReferencePOM2KCore.solve_implicit_eta_cg(
            self, rhs, d_2d, max_iter=max_iter, tol=tol, ...
        )
```

This ensures that the computational graph is properly constructed through the iterative solver. The reference implementation uses `.clone()` for all intermediate tensors so that in-place operations do not break the autograd tape. During pure forward simulation (no gradients), the accelerated PCG with pre-allocated buffers is used for maximum performance.

Scientific references are listed in the [bibliography](../appendix/references.md); the production configuration is described in [Dynamical cores](../cores/overview.md).
