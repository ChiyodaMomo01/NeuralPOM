"""Conservative allocation of a 2-D column energy reservoir to C-grid faces.

Temporary allocation is only a spending limit. Unspent energy stays at its
original tracer cell, so allocation/reclamation cannot smooth the reservoir.
All energies are extensive per reference density, in m^5/s^2.
"""
import torch


def allocate_faces(energy,mask_u,mask_v):
    count=mask_u+mask_v
    count=count.clone();count[:-1]+=mask_u[1:];count[:,:-1]+=mask_v[:,1:]
    share=torch.where(count>0,energy/torch.where(count>0,count,1.),0.)
    eu=torch.zeros_like(energy);ev=torch.zeros_like(energy)
    eu[1:]=(share[1:]+share[:-1])*mask_u[1:]
    ev[:,1:]=(share[:,1:]+share[:,:-1])*mask_v[:,1:]
    return eu,ev,share


def debit_faces(energy,mask_u,mask_v,used_u,used_v):
    eu,ev,share=allocate_faces(energy,mask_u,mask_v)
    fractions=[]
    for used,available in ((used_u,eu),(used_v,ev)):
        tolerance=available.abs()*1e-12+1e-20
        if bool((used < -tolerance).any() or (used-available>tolerance).any()):
            raise ValueError('Face spending exceeded its allocated reservoir')
        frac=torch.where(available>0,used/torch.where(available>0,available,1.),0.)
        fractions.append(frac.clamp(0,1))
    fu,fv=fractions
    factor=mask_u*fu+mask_v*fv
    factor=factor.clone();factor[:-1]+=mask_u[1:]*fu[1:];factor[:,:-1]+=mask_v[:,1:]*fv[:,1:]
    remaining=energy-share*factor
    # Compare actual debit to spending before accepting any roundoff-level
    # negativity. An overdraw above tolerance was rejected before updating.
    residual=(energy-remaining).sum()-used_u.sum()-used_v.sum()
    if abs(float(residual))>max(float(energy.abs().sum())*1e-12,1e-20):raise RuntimeError('Reservoir debit ledger mismatch')
    return remaining,dict(debit_residual=float(residual))


def face_dissipation_to_columns(diss_u,diss_v):
    """Split each positive wet-face loss equally over its adjacent cells."""
    if diss_u.ndim==3:diss_u=diss_u.sum(-1);diss_v=diss_v.sum(-1)
    result=.5*(diss_u+diss_v)
    result=result.clone();result[:-1]+=.5*diss_u[1:];result[:,:-1]+=.5*diss_v[:,1:]
    return result


def diffuse_energy(energy,volume,conductance_x,conductance_y,seconds):
    """Positive explicit substeps for dR/dt=div(C grad(R/V)).

    C includes Ke (m²/s), and has units m³/s. No-flux dry/boundary edges
    are encoded by zero conductance. Subcycling is derived from the actual
    outgoing-rate sum; it is recorded rather than clipping negative energy.
    """
    import math
    if seconds<0:raise ValueError('Negative duration')
    outgoing=torch.zeros_like(volume)
    outgoing[:-1]+=conductance_x;outgoing[1:]+=conductance_x
    outgoing[:,:-1]+=conductance_y;outgoing[:,1:]+=conductance_y
    safe_volume=torch.where(volume>0,volume,1.)
    maximum=float((outgoing/safe_volume).max())
    substeps=max(1,math.ceil(seconds*maximum/.8));dt=seconds/substeps
    result=energy.clone()
    for _ in range(substeps):
        intensive=torch.where(volume>0,result/safe_volume,0.)
        fx=conductance_x*(intensive[1:]-intensive[:-1])
        fy=conductance_y*(intensive[:,1:]-intensive[:,:-1])
        change=torch.zeros_like(result)
        change[:-1]+=fx;change[1:]-=fx
        change[:,:-1]+=fy;change[:,1:]-=fy
        result=result+dt*change
        if bool((result < -energy.abs().max()*1e-12).any()):raise RuntimeError('Reservoir diffusion lost positivity')
    residual=float(result.sum()-energy.sum())
    if abs(residual)>max(float(energy.abs().sum())*1e-12,1e-20):raise RuntimeError('Reservoir diffusion lost conservation')
    return result,dict(substeps=substeps,conservation_residual=residual)
