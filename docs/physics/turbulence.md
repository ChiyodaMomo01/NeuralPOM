# Mellor-Yamada Level-2.5 Turbulence Closure

NeuralPOM implements the Mellor-Yamada level-2.5 turbulence closure model (Mellor & Yamada, 1982) to parameterize vertical mixing. This second-moment closure scheme solves two prognostic equations for turbulence quantities, from which the vertical eddy viscosity $K_M$ and diffusivity $K_H$ are diagnostically determined.

---

## Overview of the Mellor-Yamada Hierarchy

The Mellor-Yamada turbulence closure hierarchy classifies models by the level of complexity in the Reynolds stress equations:

| Level | Description | Prognostic Variables |
|-------|-------------|---------------------|
| Level-2 | Full algebraic equilibrium; no transport of turbulence | None |
| Level-2.5 | Turbulence kinetic energy and length scale prognostic | $q^2$, $q^2 l$ |
| Level-3 | Adds temperature variance equation | $q^2$, $q^2 l$, $\overline{\theta^2}$ |
| Level-4 | Full Reynolds stress transport | All second moments |

Level-2.5 provides a good balance between physical fidelity (it captures the evolution of turbulence in non-equilibrium conditions) and computational cost (only two additional prognostic equations).

---

## Prognostic Equations

### Turbulent Kinetic Energy ($q^2/2$)

The quantity $q^2 \equiv \overline{u_i u_i}$ is twice the turbulent kinetic energy (TKE). Its prognostic equation is:

$$
\frac{\partial q^2}{\partial t} = \frac{\partial}{\partial z} \left( K_q \frac{\partial q^2}{\partial z} \right) + 2 \left( P_s + P_b - \varepsilon \right)
$$

### Turbulence Length Scale Product ($q^2 l$)

The master length scale $l$ is carried through the product $q^2 l$:

$$
\frac{\partial q^2 l}{\partial t} = \frac{\partial}{\partial z} \left( K_q \frac{\partial q^2 l}{\partial z} \right) + l E_1 \left( P_s + P_b - \frac{\tilde{W}}{E_1} \varepsilon \right)
$$

where:

| Symbol | Description |
|--------|-------------|
| $q^2$ | Twice the turbulent kinetic energy ($q^2 \equiv 2k$) |
| $l$ | Turbulence master length scale |
| $K_q$ | Vertical diffusivity for turbulence quantities ($K_q = q l S_q$) |
| $P_s$ | Shear production |
| $P_b$ | Buoyancy production/destruction |
| $\varepsilon$ | Dissipation rate ($\varepsilon = q^3 / (B_1 l)$) |
| $\tilde{W}$ | Wall proximity function |
| $E_1$ | Empirical constant |

---

## Production and Dissipation Terms

### Shear Production $P_s$

Shear production represents the conversion of mean kinetic energy to turbulence by the Reynolds stresses working against the mean shear:

$$
P_s = K_M \left[ \left( \frac{\partial u}{\partial z} \right)^2 + \left( \frac{\partial v}{\partial z} \right)^2 \right]
$$

In the code (`profq`), the shear production is computed at internal interfaces:

```python
for k in range(1, kbm1):
    du_sq = (u[s_i, s_j, k] - u[s_i, s_j, k-1] +
             u[s_ip1, s_j, k] - u[s_ip1, s_j, k-1])**2
    dv_sq = (v[s_i, s_j, k] - v[s_i, s_j, k-1] +
             v[s_i, s_jp1, k] - v[s_i, s_jp1, k-1])**2

    denom = (dzz[k-1] * dh[s_i, s_j])**2
    kn[s_i, s_j, k] = km[s_i, s_j, k] * 0.25 * sef * (du_sq + dv_sq) / denom
```

where `sef = 1.0` is a scaling factor and `dzz` is the vertical layer spacing.

### Buoyancy Production/Destruction $P_b$

Buoyancy effects can either produce turbulence (in unstable stratification) or destroy it (in stable stratification):

$$
P_b = K_H N^2
$$

where $N$ is the Brunt-Vaisala frequency computed from density gradients:

$$
N^2 = -\frac{g}{\rho_0} \frac{\partial \rho}{\partial z}
$$

The implementation includes a sound speed correction:

```python
for k in range(1, kbm1):
    drho = rho[:, :, k-1] - rho[:, :, k]
    term1 = grav * drho / (dzz[k-1] * dh)
    term2 = (grav**2) * 2.0 / (cc[:, :, k-1]**2 + cc[:, :, k]**2)
    boygr[:, :, k] = term1 + term2
```

### Dissipation $\varepsilon$

The dissipation rate follows Kolmogorov scaling:

$$
\varepsilon = \frac{q^3}{B_1 l}
$$

where $B_1 = 16.6$ is an empirical constant. The dissipation timescale is controlled by:

```python
dtef = torch.sqrt(torch.abs(q2b) + 1e-8) * stf / (b1 * l + small)
```

Enhancing dissipation near boundaries:

$$
\varepsilon_{\text{wall}} = \varepsilon \cdot \left( 1 + E_2 \tilde{W}^2 \right)
$$

---

## Wall Proximity Function $\tilde{W}$

The wall proximity function accounts for the suppression of the turbulence length scale near solid boundaries (both surface and bottom). It is defined as:

$$
\tilde{W} = \frac{l}{\kappa} \left( \frac{1}{|z - z_s|} + \frac{1}{|z - z_b|} \right) \cdot \frac{1}{D}
$$

where:

| Symbol | Description |
|--------|-------------|
| $\kappa = 0.41$ | von Karman constant |
| $z_s$ | Vertical position of the free surface |
| $z_b$ | Vertical position of the bottom |
| $D = H + \eta$ | Total water depth |
| $l$ | Turbulence length scale |

This forces $l \to 0$ as the boundaries are approached. In the code:

```python
wall_term = (1.0 / torch.abs(z[k] - z_surf) + 1.0 / torch.abs(z[k] - z_bot)) * l[:, :, k] / (dh * kappa)
```

---

## Stability Functions $S_M$ and $S_H$

The stability functions are algebraic expressions derived from the level-2.5 closure assumptions. They depend on the Richardson number through the quantity $G_H$:

$$
G_H = -\frac{l^2}{q^2} N^2
$$

The stability functions are:

$$
S_H = \frac{A_2 \left( 1 - \frac{6A_1}{B_1} \right)}{1 - (3A_2 B_2 + 18 A_1 A_2) G_H}
$$

$$
S_M = \frac{A_1 \left( 1 - 3C_1 - \frac{6A_1}{B_1} \right) + S_H \left( 18A_1^2 + 9A_1 A_2 \right) G_H}{1 - 9A_1 A_2 G_H}
$$

These are computed in the code using the pre-defined coef1--coef5:

```python
coef4 = 18.0 * a1 * a1 + 9.0 * a1 * a2
coef5 = 9.0 * a1 * a2
coef1 = a2 * (1.0 - 6.0 * a1 / b1)
coef2 = 3.0 * a2 * b2 + 18.0 * a1 * a2
coef3 = a1 * (1.0 - 3.0 * c1 - 6.0 * a1 / b1)

denom_sh = 1.0 - coef2 * gh
sh = coef1 / (denom_sh + 1e-20)

num_sm = coef3 + sh * coef4 * gh
denom_sm = 1.0 - coef5 * gh
sm = num_sm / (denom_sm + 1e-20)
```

The quantity `gh` is clamped to a maximum value of 0.028 to prevent numerical instability:

```python
gh = torch.clamp(gh, max=0.028)
```

---

## Vertical Eddy Viscosity and Diffusivity

The turbulence quantities determine the vertical mixing coefficients:

$$
K_M = q l S_M
$$

$$
K_H = q l S_H
$$

$$
K_q = 0.41 q l S_H = 0.2 q l
$$

(with $S_q / S_H = 0.41$, from the closure assumptions).

The final values are time-smoothed (averaged with previous values):

```python
kn_final = l_out * torch.sqrt(torch.abs(q2_in) + 1e-8)

kq_out = (kn_final * 0.41 * sh + kq_in) * 0.5
km_out = (kn_final * sm + km_in) * 0.5
kh_out = (kn_final * sh + kh_in) * 0.5
```

Boundary values of $K_M$ and $K_H$ are set by extrapolation from interior points, masked by the land-sea mask `fsm`.

---

## Boundary Conditions

### Surface Boundary Condition for $q^2$

At the free surface, the TKE is proportional to the friction velocity $u_*$ from wind stress:

$$
q^2|_{z=\eta} = (15.8 \, C_{\text{bc}})^{2/3} u_*^2
$$

where $C_{\text{bc}} = 100.0$ is a boundary constant, and $u_*$ is computed from the wind stress components:

```python
gg_list[0][s_i, s_j] = (15.8 * cbcnst)**(2.0/3.0) * utau2
```

The surface friction velocity is:

$$
u_*^2 = \sqrt{ \left( \frac{\tau_s^x}{\rho_0} \right)^2 + \left( \frac{\tau_s^y}{\rho_0} \right)^2 }
$$

### Surface Length Scale

The surface value of the length scale is constrained by a surface roughness length $L_0$:

$$
l_0 = \frac{\text{surfl} \cdot u_*^2}{g}
$$

where `surfl = 2.0e5` is an empirical constant. The length scale in the upper layer is limited:

```python
l0[s_i, s_j] = surfl * utau2 / grav
l_out[:, :, 0] = kappa * l0
```

### Bottom Boundary Condition

At the bottom, the turbulence length scale goes to zero:

$$
q^2 l |_{z=-H} = 0
$$

The bottom TKE is determined by bottom friction:

$$
q^2|_{z=-H} = B_1^{2/3} |\vec{\tau}_b / \rho_0|
$$

```python
uf_bottom_val[s_i, s_j] = torch.sqrt(u_bot_val**2 + v_bot_val**2 + 1e-8) * const1
```

where `const1 = (16.6**(2.0/3.0)) * sef`.

---

## Implementation: The `advq` and `profq` Subroutines

### `advq`: Horizontal Advection and Diffusion

The `advq` subroutine advects and diffuses $q^2$ and $q^2 l$ in the horizontal. Since these are positive-definite fields, the advection scheme uses the same centered-in-space formulation as momentum, with a blend of two time levels for stability.

Key steps in `advq`:

1. **Horizontal advection** at interior U/V points
2. **Horizontal diffusion** using the `aam` (ambient horizontal viscosity) field
3. **Vertical advection** term: $\frac{\partial (w q)}{\partial z}$
4. **Time stepping**: Leapfrog update with depth scaling

### `profq`: Vertical Diffusion and Production

The `profq` subroutine is the core of the Mellor-Yamada 2.5 implementation. It performs a vertical implicit solve for $q^2$ and $q^2 l$ using the Thomas algorithm (tridiagonal solver), with complete autograd safety via list-based forward/backward sweeps.

The sequence in `profq`:

```
1. Compute diffusion coefficients (a/c matrices) for Kq
2. Calculate buoyancy frequency (N^2) at interfaces
3. Determine length scale l = |q^2l| / |q^2|
4. Apply wall proximity correction to l
5. Calculate shear production (P_s)
6. Forward sweep for q^2 (Thomas algorithm)
7. Backward sweep for q^2
8. Forward sweep for q^2l (with wall-enhanced dissipation)
9. Backward sweep for q^2l
10. Clip negative values to small positive floor
11. Compute stability functions S_M, S_H from gh
12. Update K_M, K_H, K_q
```

---

## Empirical Constants

The standard Mellor-Yamada constants used in NeuralPOM are:

| Symbol | Value | Description |
|--------|-------|-------------|
| $A_1$ (`a1`) | 0.92 | Turbulence closure constant |
| $B_1$ (`b1`) | 16.6 | Turbulence closure constant |
| $A_2$ (`a2`) | 0.74 | Turbulence closure constant |
| $B_2$ (`b2`) | 10.1 | Turbulence closure constant |
| $C_1$ (`c1`) | 0.08 | Turbulence closure constant |
| $E_1$ (`e1`) | 1.8 | Wall proximity constant |
| $E_2$ (`e2`) | 1.33 | Wall proximity constant |
| $E_3$ | -- | (not directly used in level-2.5) |
| $\kappa$ (`kappa`) | 0.4 | von Karman constant |
| $C_{\text{bc}}$ (`cbcnst`) | 100.0 | Surface boundary constant |
| $\text{surfl}$ | $2 \times 10^5$ | Surface length scale constant |
| $\text{sef}$ | 1.0 | Shear energy factor |

These constants are defined at the beginning of `profq`:

```python
a1=0.92; b1=16.6; a2=0.74; b2=10.1; c1=0.08
e1=1.8; e2=1.33; sef=1.0
cbcnst=100.0; surfl=2.0e5; shiw=0.0
```

---

## The `G_H` Limiter

To prevent unrealistically large mixing in strongly stratified conditions, the stability-related quantity $G_H$ is clipped:

$$
G_H^{\text{limit}} = 0.028
$$

This is equivalent to a critical gradient Richardson number of approximately $Ri_g \approx 0.2$--$0.25$, beyond which turbulence is strongly suppressed. The limit prevents the denominator in the stability functions from approaching zero (which would cause $K_M, K_H \to \infty$ in the original algebraic formulation).

```python
gh[:, :, 1:kbm1] = (l_out[:, :, 1:kbm1]**2) * boygr[:, :, 1:kbm1] / (q2b_abs[:, :, 1:kbm1] + 1e-20)
gh = torch.clamp(gh, max=0.028)
```

---

## Background Vertical Mixing

A molecular background viscosity `umol` is added to the turbulent coefficients to prevent zero mixing and maintain numerical stability:

$$
K_M^{\text{total}} = K_M^{\text{turbulent}} + u_{\text{mol}}
$$

$$
K_H^{\text{total}} = K_H^{\text{turbulent}} + u_{\text{mol}}
$$

In the Thomas algorithm tridiagonal coefficients:

```python
num = -dti2 * (km_u[s_i, s_j, k] + umol)
```

A small floor value `small` is enforced on $q^2$ and $q^2 l$ to prevent negative values that would cause `sqrt` errors in the stability function computation:

```python
uf_sl_new = torch.where(mask_small, torch.tensor(small), uf_sl)
vf_sl_new = torch.where(mask_small, 0.1 * dt_sl * small, vf_sl)
```

---

## References

- Mellor, G. L., & Yamada, T. (1982). Development of a turbulence closure model for geophysical fluid problems. *Reviews of Geophysics*, 20(4), 851--875.
- Mellor, G. L. (2004). *Users guide for a three-dimensional, primitive equation, numerical ocean model*. Princeton University.
- Burchard, H., & Bolding, K. (2001). Comparative analysis of four second-moment turbulence closure models for the oceanic mixed layer. *Journal of Physical Oceanography*, 31(8), 1943--1968.

Scientific references are listed in the [bibliography](../appendix/references.md); the production configuration is described in [Dynamical cores](../cores/overview.md).
