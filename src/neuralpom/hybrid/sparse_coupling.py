"""Shared endpoint correction constraints for hourly and daily models."""
import copy
import torch

OUTPUT_FIELDS=('u','v','t','s')
LEVELS=15
OUTPUT_CHANNELS=4*LEVELS


def eos(core,t,s):
    """Functional evaluation of the actual core EOS, including its ghost rule."""
    proxy=copy.copy(core)
    proxy.t=t.clone();proxy.s=s.clone();proxy.rho=core.rho.clone()
    _,_,rho=proxy.dens()
    return rho


def volume_weights(core,field='t'):
    mask=core.dum if field in ('u','ub') else core.dvm if field in ('v','vb') else core.fsm
    return (core.art*mask*(core.h+core.et))[:,:,None]*core.dz[:LEVELS][None,None,:]


def weighted_mse(x,y,w):
    return ((x-y).square()*w).sum()/w.expand_as(x).sum().clamp_min(1e-20)


def budget_project(increment,weights):
    wet=weights>0
    # The neural increment may redistribute tracers horizontally within a
    # physical layer. The unchanged physical core owns the mean vertical flux.
    mean=(increment*weights).sum((0,1),keepdim=True)/weights.sum((0,1),keepdim=True).clamp_min(1e-20)
    return (increment-mean)*wet


def paired_limit(increments,scales,cap):
    # A SINGLE multiplier for T and S preserves their first-order compensation.
    peak=torch.stack([(x/s).abs().amax() for x,s in zip(increments,scales)]).amax()
    factor=(cap/peak.clamp_min(cap)).clamp(max=1.)
    return [factor*x for x in increments],factor


def apply_joint_correction(core,raw,scales,cap=5.):
    if core.kbm1!=LEVELS or raw.shape!=(OUTPUT_CHANNELS,core.im,core.jm):
        raise ValueError(f'Unexpected grid/output contract: {core.kbm1}, {raw.shape}')
    pieces={f:raw[i*LEVELS:(i+1)*LEVELS].permute(1,2,0)*scales[f]
            for i,f in enumerate(OUTPUT_FIELDS)}
    dz=core.dz[:LEVELS].view(1,1,-1)
    for f,mask in (('u',core.dum),('v',core.dvm)):
        pieces[f]=(pieces[f]-(pieces[f]*dz).sum(2,keepdim=True)/dz.sum())*mask[:,:,None]
    w=volume_weights(core)
    for f in ('t','s'):
        pieces[f]=budget_project(pieces[f],w)
    tracer,factor=paired_limit([pieces['t'],pieces['s']],[scales['t'],scales['s']],cap)
    pieces['t'],pieces['s']=tracer
    velocity,_=paired_limit([pieces['u'],pieces['v']],[scales['u'],scales['v']],cap)
    pieces['u'],pieces['v']=velocity
    for f,history in (('u','ub'),('v','vb'),('t','tb'),('s','sb')):
        for attr in (f,history):
            previous=getattr(core,attr)
            updated=torch.cat((previous[:,:,:LEVELS]+pieces[f],previous[:,:,LEVELS:]),dim=2)
            # Keep the core-produced ghost; its normal stencil fills happen in
            # advt1. A zero network must not alter legacy ghost/time-layer state.
            setattr(core,attr,updated)
    core.rho=eos(core,core.t,core.s)
    return {'ts_limit_factor':factor,
            't_budget':(w*pieces['t']).sum()/w.sum(),
            's_budget':(w*pieces['s']).sum()/w.sum()}


def pressure_velocity_error(core,reference_t,reference_s):
    """Exact baropg difference on common geometry, converted to advu/v velocity units."""
    proxy=copy.copy(core)
    # baropg is linear in rho at fixed geometry. A zero rmean computes its difference.
    proxy.rho=eos(core,core.t,core.s)-eos(core,reference_t,reference_s)
    proxy.rmean=torch.zeros_like(core.rmean)
    _,px,py=proxy.baropg()
    depth=core.h+core.etf
    du=(depth+torch.roll(depth,1,0))*core.aru
    dv=(depth+torch.roll(depth,1,1))*core.arv
    kick_u=-2.*core.dti2*px[:,:,:LEVELS]/du[:,:,None].clamp_min(1e-20)
    kick_v=-2.*core.dti2*py[:,:,:LEVELS]/dv[:,:,None].clamp_min(1e-20)
    return kick_u*core.dum[:,:,None],kick_v*core.dvm[:,:,None]
