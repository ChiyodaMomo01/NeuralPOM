"""Read-only local vertical-mode / Rossby diagnostics for the planned B3 mask.

Rigid-lid hydrostatic vertical modes; N2 uses the same compressibility
correction as POM profq. The positive N2 floor regularizes the eigenproblem
ONLY: its incidence must be reported, never silently interpreted as observed
stable stratification. No core state is altered.
"""
import torch


def first_mode_speed(n2, thickness, centre_distance, floor=1e-8):
    """Return first baroclinic c in m/s from -d/dz(1/N2 dphi/dz)=phi/c².

    Cell-centred velocity modes use zero normal derivative at top/bottom.
    thickness has n layers, centre_distance and n2 have n-1 interfaces.
    Leading dimensions broadcast; all computations should use float64.
    """
    if n2.shape[-1]!=thickness.shape[-1]-1:raise ValueError('Vertical size mismatch')
    if bool((thickness<=0).any() or (centre_distance<=0).any()):raise ValueError('Positive vertical geometry required')
    n=thickness.shape[-1]
    conductance=1/(n2.clamp_min(floor)*centre_distance)
    diagonal=torch.zeros_like(thickness)
    diagonal[...,:-1]+=conductance;diagonal[...,1:]+=conductance
    invmass=thickness.rsqrt()
    matrix=torch.diag_embed(diagonal*invmass.square())
    off=-conductance*invmass[...,:-1]*invmass[...,1:]
    matrix+=torch.diag_embed(off,offset=1)+torch.diag_embed(off,offset=-1)
    eigenvalues=torch.linalg.eigvalsh(matrix)
    if bool((eigenvalues[...,1]<=0).any()):raise ValueError('Nonpositive first baroclinic eigenvalue')
    return eigenvalues[...,1].rsqrt()


def stratification(core):
    """POM's N²=-boygr; rho is (density-1000)/rhoref, not kg/m³."""
    k=core.kbm1
    rho=core.rho[...,:k].double();depth=(core.h+core.et).double()
    temperature=core.t[...,:k].double()+core.tbias
    salinity=core.s[...,:k].double()+core.sbias
    zz=core.zz[:k].double()
    pressure=core.grav*core.rhoref*(-zz[None,None,:]*depth[...,None])*1e-4
    sound=1449.1+.00821*pressure+4.55*temperature-.045*temperature.square()+1.34*(salinity-35)
    sound=sound/torch.sqrt(((1-.01642*pressure/sound)*(1-.40*pressure/sound.square())).clamp_min(1e-8))
    distance=depth[...,None]*(zz[:-1]-zz[1:])
    n2=core.grav*(rho[...,1:]-rho[...,:-1])/distance-2*core.grav**2/(sound[...,:-1].square()+sound[...,1:].square())
    return n2,depth[...,None]*core.dz[:k].double(),distance


def rossby_at_centres(core):
    """Maximum |zeta/f| over layers and the four surrounding wet corners."""
    u=core.u[...,:core.kbm1].double();v=core.v[...,:core.kbm1].double()
    dx=core.dx.double();dy=core.dy.double();f=core.cor.double();wet=core.fsm.bool()
    corner_wet=wet[1:,1:]&wet[:-1,1:]&wet[1:,:-1]&wet[:-1,:-1]
    fc=.25*(f[1:,1:]+f[:-1,1:]+f[1:,:-1]+f[:-1,:-1])
    dxc=.5*(dx[1:,1:]+dx[:-1,1:]);dyc=.5*(dy[1:,1:]+dy[1:,:-1])
    zeta=(v[1:,1:]-v[:-1,1:])/dxc[...,None]-(u[1:,1:]-u[1:,:-1])/dyc[...,None]
    ro=(zeta.abs()/fc.abs().clamp_min(1e-12)[...,None]).amax(-1)
    ro=torch.where(corner_wet,ro,torch.zeros_like(ro))
    result=torch.zeros_like(f)
    for i,j in [(slice(1,None),slice(1,None)),(slice(None,-1),slice(1,None)),(slice(1,None),slice(None,-1)),(slice(None,-1),slice(None,-1))]:
        result[i,j]=torch.maximum(result[i,j],ro)
    return result


def eligibility_diagnostics(core,n2_floor=1e-8):
    n2,thickness,distance=stratification(core)
    wet=core.fsm.bool()
    c=first_mode_speed(n2[wet],thickness[wet],distance[wet],n2_floor)
    radius=torch.zeros_like(core.h,dtype=torch.float64)
    radius[wet]=c/core.cor.double()[wet].abs().clamp_min(1e-12)
    ratio=radius/(core.dx.double()*core.dy.double()).sqrt()
    ro=rossby_at_centres(core)
    s=(2*ratio-1).clamp(0,1);a=((.30-ro)/.15).clamp(0,1)
    mask=s.square()*(3-2*s)*a.square()*(3-2*a)*wet
    # Convective/near-neutral interfaces remain explicitly identified.
    return dict(n2=n2,radius=radius,ratio=ratio,rossby=ro,mask=mask,
                regularized_fraction=(n2<n2_floor).double().mean(-1),wet=wet)
