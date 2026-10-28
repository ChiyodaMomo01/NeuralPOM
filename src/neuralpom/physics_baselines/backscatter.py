"""Conservative, supply-limited negative-diffusion substep on velocity faces.

Positive nu is the magnitude of negative viscosity. Wet/wet edges only:
backscatter does not apply negative wall stress. Volume and edge distances
come from the native POM velocity control cells. No core state is mutated.
"""
import torch
from .energy import return_energy_pair
from .pom_stress import face_average
from .reservoir import allocate_faces,debit_faces


def _component(field,volume,energy,viscosity,dx,dy,seconds):
    q=field.clone();budget=energy.clone();paid=0.;limited=0;active=0
    # Symmetric ordering of the four disjoint edge sets. The flow of each
    # pair with frozen viscosity is integrated exactly before supply limiting.
    passes=((0,0,.5),(0,1,.5),(1,0,.5),(1,1,1.),(1,0,.5),(0,1,.5),(0,0,.5))
    for axis,parity,fraction in passes:
        left=[slice(None)]*3;right=left.copy()
        left[axis]=slice(parity,field.shape[axis]-1,2)
        right[axis]=slice(parity+1,field.shape[axis],2)
        left=tuple(left);right=tuple(right)
        vl=volume[left];vr=volume[right];wet=(vl>0)&(vr>0)
        spacing=dx if axis==0 else dy
        # Broadcast 2D geometry and viscosity over layers.
        d=spacing[...,None].expand_as(field);nu=viscosity[...,None].expand_as(field)
        distance=.5*(d[left]+d[right])
        conductance=.5*(vl+vr)/distance.square()
        magnitude=.5*(nu[left]+nu[right])
        inverse=1/vl[wet]+1/vr[wet]
        rate=conductance[wet]*magnitude[wet]*inverse
        transport=torch.expm1(seconds*fraction*rate)/inverse
        if not bool(torch.isfinite(transport).all()):raise FloatingPointError('Backscatter pair exponential overflow')
        ql,qr,el,er,audit=return_energy_pair(q[left][wet],q[right][wet],vl[wet],vr[wet],budget[left][wet],budget[right][wet],transport)
        # Disjoint pairs per pass: no simultaneous cell can overspend.
        view=q[left];view[wet]=ql;view=q[right];view[wet]=qr
        view=budget[left];view[wet]=el;view=budget[right];view[wet]=er
        paid+=float(audit['paid'].sum());limited+=int(audit['limited'].sum());active+=int(wet.sum())
    return q,budget,dict(paid=paid,limited_pairs=limited,active_pairs=active)


def backscatter_exchange(operator,u,v,energy,nu,seconds):
    c=operator.core
    u=u.double();v=v.double();energy=energy.double();nu=nu.double()
    mu=operator.mu.double();mv=operator.mv.double()
    allocated_u,allocated_v,_=allocate_faces(energy,mu,mv)
    dz=c.dz.double().clone();dz[c.kbm1:]=0
    if abs(float(dz.sum())-1)>1e-6:raise ValueError('Column layer fractions do not sum to one')
    dz=dz/dz.sum()
    eu=allocated_u[...,None]*dz;ev=allocated_v[...,None]*dz
    un,eu_new,au=_component(u,operator.wu.double(),eu,face_average(nu,0),face_average(c.dx.double(),0),face_average(c.dy.double(),0),seconds)
    vn,ev_new,av=_component(v,operator.wv.double(),ev,face_average(nu,1),face_average(c.dx.double(),1),face_average(c.dy.double(),1),seconds)
    used_u=(eu-eu_new).sum(-1);used_v=(ev-ev_new).sum(-1)
    remaining,debit=debit_faces(energy,mu,mv,used_u,used_v)
    gain=(operator.wu*.5*(un+u)*(un-u)).sum()+(operator.wv*.5*(vn+v)*(vn-v)).sum()
    paid=used_u.sum()+used_v.sum();residual=gain-paid
    scale=(operator.wu*u.square()).sum()+(operator.wv*v.square()).sum()+energy.sum()
    if abs(float(residual))>max(float(scale)*1e-11,1e-12):raise RuntimeError('Finite-step backscatter budget failed')
    return un,vn,remaining,dict(kinetic_gain=float(gain),reservoir_debit=float(paid),energy_residual=float(residual),
            limited_pairs=au['limited_pairs']+av['limited_pairs'],active_pairs=au['active_pairs']+av['active_pairs'],**debit)
