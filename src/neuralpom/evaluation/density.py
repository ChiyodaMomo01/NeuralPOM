"""Retained density75 policy, transferred without retuning to all checkpoints.

Arithmetic follows the preserved confirmation1/density_policy.py implementation:
canonical correction, normalized EOS direction, two layer budget constraints,
75% density component, paired cap, paired history write, exact EOS refresh.
"""
import copy

import torch

from neuralpom.hybrid.sparse_coupling import LEVELS, eos, volume_weights, paired_limit, apply_joint_correction


def eos_gradient(core, t, s):
    with torch.enable_grad():
        tt=t.detach().double().requires_grad_(True)
        ss=s.detach().double().requires_grad_(True)
        proxy=copy.copy(core)
        proxy.t=tt.clone();proxy.s=ss.clone();proxy.rho=core.rho.double().clone()
        density=proxy.dens()[2][..., :LEVELS].sum()
        derivatives=torch.autograd.grad(density,(tt,ss))
    return tuple(d[..., :LEVELS].detach() for d in derivatives)


def constrained_amplitude(amplitude, direction, weights):
    w=weights/weights.sum((0,1),keepdim=True).clamp_min(1e-30)
    gram=torch.einsum('ijk,ijka,ijkb->kab',w,direction,direction)
    rhs=torch.einsum('ijk,ijka,ijk->ka',w,direction,amplitude)
    eigenvalues,basis=torch.linalg.eigh(gram)
    cutoff=eigenvalues.amax(-1,keepdim=True)*1e-10
    inverse=torch.where(eigenvalues>cutoff,eigenvalues.clamp_min(1e-30).reciprocal(),0.)
    coefficients=torch.einsum('kai,ki,kbi,kb->ka',basis,inverse,basis,rhs)
    adjusted=amplitude-torch.einsum('ijka,ka->ijk',direction,coefficients)
    residual=torch.einsum('ijk,ijka,ijk->ka',w,direction,adjusted)
    return adjusted,residual


@torch.no_grad()
def apply_density75(core,raw,correction_scales):
    before={f:getattr(core,f).clone() for f in ('t','tb','s','sb')}
    diagnostics=apply_joint_correction(core,raw,correction_scales,cap=5.)
    scales=torch.tensor([correction_scales[f] for f in ('t','s')],device=raw.device,dtype=torch.float64)
    x=torch.stack([(getattr(core,f)-before[f])[..., :LEVELS].double() for f in ('t','s')],-1)/scales
    if not bool(x.any()):
        return diagnostics
    dt,ds=eos_gradient(core,before['t'],before['s'])
    g=torch.stack((dt,ds),-1)*scales
    direction=g/torch.linalg.vector_norm(g,dim=-1,keepdim=True).clamp_min(1e-30)
    weights=volume_weights(core).double()
    amplitude=(x*direction).sum(-1)
    adjusted,residual=constrained_amplitude(amplitude,direction,weights)
    change=(.75-1)*adjusted[...,None]*direction
    proposed=(x+change)*scales
    increments,factor=paired_limit([proposed[...,i] for i in range(2)],list(scales),5.)
    diagnostics['ts_limit_factor']*=factor
    diagnostics['density_component_limit_factor']=factor
    diagnostics['density_constraint_residual_max_abs']=residual.abs().max()
    for f,history,delta in zip(('t','s'),('tb','sb'),increments):
        delta=delta.to(before[f].dtype)
        for name in (f,history):
            value=before[name]
            setattr(core,name,torch.cat((value[..., :LEVELS]+delta,value[..., LEVELS:]),2))
    core.rho=eos(core,core.t,core.s)
    return diagnostics
