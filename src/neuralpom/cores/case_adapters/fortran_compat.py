"""Private POM08-WAD profq compatibility policy; never edits production cores.

Forward equations follow archived pom08.f:4761-5131 (nwad=0, zsh=.01 m).
No epsilon is added to stress/TKE square roots. torch.linalg.vector_norm has
the exact zero forward and a defined zero subgradient at zero stress. sqrt(q)
retains its mathematical singular derivative at q=0: this evaluation policy
does not claim a validated training adjoint for zero TKE. Tested via no_grad.
All newly allocated arrays have the state's dtype. install also makes private
process default floating allocations use that dtype (other legacy kernels
otherwise create float32 scratch). It does not change the archived source.
"""
import types
import torch

VERSION = 'pom08-wad-profq-v1'


def _edge_cosmetics(x, fsm):
    # Fortran order: north/south, then east/west, including corner overwrite.
    y = torch.cat((x[:, 1:2] * fsm[:, :1, None], x[:, 1:-1],
                   x[:, -2:-1] * fsm[:, -1:, None]), dim=1)
    return torch.cat((y[1:2] * fsm[:1, :, None], y[1:-1],
                      y[-2:-1] * fsm[-1:, :, None]), dim=0)


def profq(self, sm, sh, dh, cc, ee, gg, l, kq, km, kh,
          uf, vf, q2, q2b, q2lb, a, c,
          h, etf, dti2, umol, dzz, grav, rho, kappa, u, v, dt, small, fsm,
          im, jm, kb, imm1, jmm1, kbm1, tbias, sbias, dz,
          wusurf, wubot, wvsurf, wvbot, t, s, rhoref, zz, z):
    zero = torch.zeros_like(h)
    dh = h + etf
    a_layers = [zero]
    c_layers = [zero]
    for k in range(1, kbm1):
        a_layers.append(-dti2 * (kq[:, :, k+1] + kq[:, :, k] + 2*umol)*.5
                        / (dzz[k-1]*dz[k]*dh*dh))
        c_layers.append(-dti2 * (kq[:, :, k-1] + kq[:, :, k] + 2*umol)*.5
                        / (dzz[k-1]*dz[k-1]*dh*dh))
    a_layers.append(zero); c_layers.append(zero)
    aa = torch.stack(a_layers, 2); ca = torch.stack(c_layers, 2)

    # Original stress boundary only covers i<im-1,j<jm-1.
    us = .5*(wusurf[:-1, :-1]+wusurf[1:, :-1])
    vs = .5*(wvsurf[:-1, :-1]+wvsurf[:-1, 1:])
    ub = .5*(wubot[:-1, :-1]+wubot[1:, :-1])
    vb = .5*(wvbot[:-1, :-1]+wvbot[:-1, 1:])
    utau2 = torch.linalg.vector_norm(torch.stack((us, vs)), dim=0)
    btau2 = torch.linalg.vector_norm(torch.stack((ub, vb)), dim=0)
    pad = lambda x: torch.nn.functional.pad(x, (0, 1, 0, 1))
    l0 = pad(2.e5*utau2/grav)
    surface_q = pad((15.8*100.)**(2./3.)*utau2)
    bottom = uf[:, :, -1].clone()
    bottom[:-1, :-1] = btau2*(16.6**(2./3.))

    # POM08-WAD sound speed uses fixed pdens, not h+etf.
    p = self._fortran_pdens[:, :, :kbm1]*10.
    tp = t[:, :, :kbm1]+tbias; sp = s[:, :, :kbm1]+sbias
    sound = 1449.1+.00821*p+4.55*tp-.045*tp**2+1.34*(sp-35.)
    sound = sound/torch.sqrt((1.-.01642*p/sound)*(1.-.40*p/sound**2))
    cc_out = torch.cat((sound, cc[:, :, -1:]), dim=2)
    qb = torch.cat((q2b[:, :, :1], torch.abs(q2b[:, :, 1:kbm1]),
                    q2b[:, :, -1:]), dim=2)
    qlb = torch.cat((q2lb[:, :, :1], torch.abs(q2lb[:, :, 1:kbm1]),
                     q2lb[:, :, -1:]), dim=2)
    buoy = [zero]
    lengths = [kappa*l0]
    stability = [zero]
    # et is the tracer/density grid before this step, unlike new etf.
    density_depth = h+self.et
    for k in range(1, kbm1):
        b = grav*(rho[:, :, k-1]-rho[:, :, k])/(dzz[k-1]*density_depth)
        b = b+grav**2*2./(sound[:, :, k-1]**2+sound[:, :, k]**2)
        ll = torch.abs(qlb[:, :, k]/qb[:, :, k])
        if z[k] > -.5:
            ll = torch.maximum(ll, kappa*l0)
        buoy.append(b); lengths.append(ll)
        stability.append(torch.clamp(ll**2*b/qb[:, :, k], max=.028))
    buoy.append(zero); lengths.append(zero); stability.append(zero)
    boygr = torch.stack(buoy, 2)
    l_out = torch.stack(lengths, 2)
    gh = torch.stack(stability, 2)

    prod = [zero]
    for k in range(1, kbm1):
        du = u[1:-1, 1:-1, k]-u[1:-1, 1:-1, k-1]+u[2:, 1:-1, k]-u[2:, 1:-1, k-1]
        dv = v[1:-1, 1:-1, k]-v[1:-1, 1:-1, k-1]+v[1:-1, 2:, k]-v[1:-1, 2:, k-1]
        pr = km[1:-1, 1:-1, k]*.25*(du**2+dv**2)/(dzz[k-1]*dh[1:-1, 1:-1])**2
        pr = pr+kh[1:-1, 1:-1, k]*boygr[1:-1, 1:-1, k]
        prod.append(torch.nn.functional.pad(pr, (1, 1, 1, 1)))
    prod.append(zero)
    production = torch.stack(prod, 2)
    diss = torch.sqrt(torch.abs(qb))/(16.6*l_out+small)

    e = [zero]; g = [surface_q]
    for k in range(1, kbm1):
        inv = 1./(aa[:, :, k]+ca[:, :, k]*(1.-e[k-1])-(2.*dti2*diss[:, :, k]+1.))
        e.append(aa[:, :, k]*inv)
        g.append((-2.*dti2*production[:, :, k]+ca[:, :, k]*g[k-1]-uf[:, :, k])*inv)
    e.append(zero); g.append(zero)
    usol = [None]*kb; usol[-1] = bottom
    for k in range(kbm1-1, -1, -1):
        usol[k] = e[k]*usol[k+1]+g[k]
    uf_out = torch.stack(usol, 2)

    # q2l has Dirichlet conditions at the SECOND and PENULTIMATE layers.
    # The legacy Torch implementation incorrectly solves both as interiors.
    e2l = [zero, zero]
    g2l = [zero, -kappa*z[1]*dh*q2[:, :, 1]]
    ds_layers = [diss[:, :, 0]]
    for k in range(1, kbm1):
        wall = (1./torch.abs(z[k]-z[0])+1./torch.abs(z[k]-z[-1]))*l_out[:, :, k]/(dh*kappa)
        ds_layers.append(diss[:, :, k]*(1.+1.33*wall**2))
    ds_layers.append(diss[:, :, -1])
    for k in range(2, kbm1):
        inv = 1./(aa[:, :, k]+ca[:, :, k]*(1.-e2l[k-1])-(dti2*ds_layers[k]+1.))
        e2l.append(aa[:, :, k]*inv)
        # Fortran sets vf(kb-1) boundary BEFORE forward sweep, then overwrites
        # it during back substitution; preserve precisely this operation.
        rhs_v = kappa*(1.+z[kbm1-1])*dh*q2[:, :, kbm1-1] if k==kbm1-1 else vf[:, :, k]
        g2l.append((dti2*(-production[:, :, k]*l_out[:, :, k]*1.8)
                    +ca[:, :, k]*g2l[k-1]-rhs_v)*inv)
    e2l.append(zero); g2l.append(zero)
    vsol = [None]*kb; vsol[0] = zero; vsol[-1] = zero
    for k in range(kbm1-1, 0, -1):
        vsol[k] = e2l[k]*vsol[k+1]+g2l[k]
    vf_out = torch.stack(vsol, 2)
    uf_out = torch.cat((uf_out[:, :, :1], torch.abs(uf_out[:, :, 1:kbm1]), uf_out[:, :, -1:]), 2)
    vf_out = torch.cat((vf_out[:, :, :1], torch.abs(vf_out[:, :, 1:kbm1]), vf_out[:, :, -1:]), 2)

    coef1 = .74*(1.-6.*.92/16.6)
    coef2 = 3.*.74*10.1+18.*.92*.74
    coef3 = .92*(1.-3.*.08-6.*.92/16.6)
    coef4 = 18.*.92*.92+9.*.92*.74
    coef5 = 9.*.92*.74
    sh_out = coef1/(1.-coef2*gh)
    sm_out = (coef3+sh_out*coef4*gh)/(1.-coef5*gh)
    kn = (kappa*self._fortran_zsh+l_out)*torch.sqrt(torch.abs(q2))
    kq_out = (kn*.20+kq)*.5
    km_out = _edge_cosmetics((kn*sm_out+km)*.5, fsm)
    kh_out = _edge_cosmetics((kn*sh_out+kh)*.5, fsm)
    return (sm_out, sh_out, dh, cc_out, torch.stack(e2l, 2),
            torch.stack(g2l, 2), l_out, kq_out, km_out, kh_out,
            uf_out, vf_out, qb, qlb, aa, ca)


def install(core, *, pdens=None, zsh=.01, set_scratch_default=True):
    """Bind private compatibility operator after importing the reference state.

    Override pdens for cases with nonzero MSL/vertical reference shift hhi.
    Default is archived nwad=0,hhi=0 formula. pdens stays fixed in time.
    """
    if pdens is None:
        pdens = core.grav*core.rhoref*(-core.zz[None, None, :]*torch.clamp(core.h[:, :, None], min=0.))*1.e-5
    core._fortran_pdens = torch.as_tensor(pdens, device=core.h.device, dtype=core.h.dtype).clone()
    core._fortran_zsh = float(zsh)
    core._fortran_compat_version = VERSION
    core.profq = types.MethodType(profq, core)
    if set_scratch_default:
        torch.set_default_dtype(core.h.dtype)
    return core
