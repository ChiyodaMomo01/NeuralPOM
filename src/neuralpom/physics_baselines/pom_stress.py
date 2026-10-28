"""Original POM C-grid vector stress and its exact volume-weighted adjoint.

Uses the existing tensor-stress stencil, with advection disabled. The input
coefficient is one m²/s: its numerical result is L u (units velocity/m²).
L need not be presumed self-adjoint on the nonuniform masked sigma grid.
An exact vector-Jacobian product constructs -L† B4 L with the supplied kinetic
energy volumes. This module is a verified operator, not a time integrator.
"""
import inspect
import torch
from neuralpom.v1.core_spec import core_spec


def face_average(value,axis):
    result=value.clone()
    if axis==0:result[1:]=.5*(value[1:]+value[:-1])
    elif axis==1:result[:,1:]=.5*(value[:,1:]+value[:,:-1])
    else:raise ValueError('Horizontal face axis must be 0 or 1')
    return result


class POMStress:
    def __init__(self,core,core_name,*,barotropic=False,depth=None):
        if core.mode!=3:raise ValueError('B2 integration is defined for three-dimensional mode 3')
        self.core=core;self.barotropic=barotropic
        self.depth=core.d if barotropic and depth is None else core.dt if depth is None else depth
        self.method=getattr(core_spec(core_name).core_class,'advave' if barotropic else 'advct')
        self.mu=core.dum.clone();self.mv=core.dvm.clone()
        for mask in (self.mu,self.mv):
            mask[0]=0;mask[-1]=0;mask[:,0]=0;mask[:,-1]=0
        hu=face_average(self.depth,0);hv=face_average(self.depth,1)
        self.area_depth_u=core.aru*hu;self.area_depth_v=core.arv*hv
        if barotropic:
            self.wu=self.area_depth_u*self.mu;self.wv=self.area_depth_v*self.mv
        else:
            dz=core.dz.clone();dz[core.kbm1:]=0
            self.wu=(self.area_depth_u*self.mu)[...,None]*dz
            self.wv=(self.area_depth_v*self.mv)[...,None]*dz

    def __call__(self,u,v):
        c=self.core
        scratch={'curv2d','advua','advva','fluxua','fluxva','tps','aam2d'} if self.barotropic else set()
        args={name:(torch.zeros_like(u) if name in scratch else getattr(c,name)) for name in inspect.signature(self.method).parameters if name!='self'}
        if self.barotropic:
            args.update(ua=torch.zeros_like(u),va=torch.zeros_like(v),uab=u*(self.wu>0),vab=v*(self.wv>0),
                        aam2d=torch.ones_like(u),d=self.depth)
        else:
            args.update(u=torch.zeros_like(u),v=torch.zeros_like(v),ub=u*(self.wu>0),vb=v*(self.wv>0),
                        aam=torch.ones_like(u),dt=self.depth)
        # advave reuses its scratch arrays; keep application side-effect free.
        if self.barotropic:
            for key in ('curv2d','advua','advva','fluxua','fluxva','wubot','wvbot','tps'):
                args[key]=args[key].clone()
        out=self.method(c,**args)
        ru,rv=(out[1],out[2]) if self.barotropic else (out[3],out[4])
        du=self.area_depth_u if self.barotropic else self.area_depth_u[...,None]
        dv=self.area_depth_v if self.barotropic else self.area_depth_v[...,None]
        return (-ru/du.clamp_min(1e-30)*(self.wu>0),-rv/dv.clamp_min(1e-30)*(self.wv>0))

    def dissipative_fourth(self,u,v,b4u,b4v):
        """Force and local positive dissipation density, frozen B4 and geometry.

        Derivatives are only with respect to independent velocity leaves;
        there is no neural model or optimizer and no gradient through B4.
        """
        with torch.enable_grad():
            left=u.detach().requires_grad_(True);right=v.detach().requires_grad_(True)
            lu,lv=self(left,right)
            density_u=b4u.detach()*lu.square();density_v=b4v.detach()*lv.square()
            energy=.5*((self.wu*density_u).sum()+(self.wv*density_v).sum())
            gu,gv=torch.autograd.grad(energy,(left,right))
        fu=-gu/torch.where(self.wu>0,self.wu,1.)
        fv=-gv/torch.where(self.wv>0,self.wv,1.)
        return fu.detach(),fv.detach(),density_u.detach(),density_v.detach()
