"""Experimental common B2/B3 closure splitting with separately auditable work.

Not yet a calibrated production baseline. Each internal step is followed by one closure step on both augmented
velocity layers. Both leapfrog velocity layers have their own reservoir;
reservoirs rotate with the prognostic/history layers, never collect energy
from the native core. Temporal convergence and transport consistency are
mandatory gates because this is a change of numerical integration.
"""
import torch
from neuralpom.v1.checkpoint import capture
from neuralpom.v1.core_spec import core_spec
from .smagorinsky import ExplicitSmagorinsky,ImplicitSmagorinsky,_rescaled_call
from .runtime import SmagorinskyRuntime
from .pom_stress import POMStress,face_average
from .midpoint import dissipative_midpoint
from .backscatter import backscatter_exchange
from .reservoir import face_dissipation_to_columns,diffuse_energy
from .eligibility import eligibility_diagnostics


class NoNativeMomentumDiffusion:
    def advct(self,*args,**kwargs):
        method=core_spec(self.baseline_reference_core).core_class.advct.__get__(self,type(self))
        return _rescaled_call(method,args,kwargs,'aam',0.)
    def advave(self,*args,**kwargs):
        method=core_spec(self.baseline_reference_core).core_class.advave.__get__(self,type(self))
        return _rescaled_call(method,args,kwargs,'aam2d',0.)


class ExplicitBudgetCore(NoNativeMomentumDiffusion,ExplicitSmagorinsky):pass
class ImplicitBudgetCore(NoNativeMomentumDiffusion,ImplicitSmagorinsky):pass


def configure_budget(core,*,gamma,beta,family):
    if family not in ('B2','B3') or not 0<=beta<=1 or not 0<gamma<=2:raise ValueError('Invalid budget closure parameters')
    if family=='B2' and beta!=0:raise ValueError('B2 cannot return energy')
    core.__class__={'explicit':ExplicitBudgetCore,'implicit_p0_fixed':ImplicitBudgetCore}[core.baseline_reference_core]
    core.baseline_family=family;core.baseline_gamma=float(gamma);core.baseline_beta=float(beta)
    core.baseline_cb=.4;core.baseline_ke=600.;core.baseline_grid_cap=.25
    core.baseline_closure_version='budget-poststep-midpoint-prototype-v2'
    core.closure_energy=torch.zeros_like(core.h,dtype=torch.float64)
    core.closure_energy_b=torch.zeros_like(core.h,dtype=torch.float64)
    core.closure_steps=0
    for layer in ('current','history'):
        for key in ('dissipation','reservoir_credit','backscatter','heat','roundoff','midpoint_residual','backscatter_residual','limited_pairs','active_pairs','cap_cells','eligible_cells'):
            setattr(core,'closure_'+layer+'_'+key,0.)
    return core


def _write_velocity_increment(core,du,dv,history):
    k=core.kbm1;dz=core.dz[:k]
    mean_u=(du[...,:k]*dz).sum(-1);mean_v=(dv[...,:k]*dz).sum(-1)
    if history:
        core.uab=core.uab+mean_u;core.vab=core.vab+mean_v
    else:
        core.ua=core.ua+mean_u;core.va=core.va+mean_v
        core.uaf=core.uaf+mean_u;core.vaf=core.vaf+mean_v
        # Preserve the existing relation between internal column mean and
        # averaged external transport; do not force initially distinct means
        # to be equal. Actual transport invariant is tested independently.
        hu=face_average(core.dt,0);hv=face_average(core.dt,1)
        core.utf=core.utf+hu*mean_u;core.utb=core.utb+hu*mean_u
        core.vtf=core.vtf+hv*mean_v;core.vtb=core.vtb+hv*mean_v


def closure_substep(core,seconds):
    if core.baseline_beta:
        eligibility=eligibility_diagnostics(core)
        mask=eligibility['mask']*(eligibility['regularized_fraction']==0)
    else:mask=torch.zeros_like(core.h,dtype=torch.float64)
    symbol=4*(core.dx.double().square().reciprocal()+core.dy.double().square().reciprocal())
    b4=core.baseline_gamma*(core.baseline_cm/.05)*core.aam.double()/symbol[...,None]
    b4u=face_average(b4,0);b4v=face_average(b4,1)
    # A vertically uniform negative viscosity must also respect the least
    # damped active layer at the grid scale. Its restrictiveness is reported.
    cap=core.baseline_grid_cap*b4[...,:core.kbm1].amin(-1)*symbol
    for history in (False,True):
        label='history' if history else 'current'
        uk,vk,rk,ek=('ub','vb','closure_energy_b','etb') if history else ('u','v','closure_energy','et')
        old_u=getattr(core,uk);old_v=getattr(core,vk);old_energy=getattr(core,rk)
        depth=core.h+getattr(core,ek)
        op=POMStress(core,core.baseline_reference_core,depth=depth)
        u,v,loss_u,loss_v,mid=dissipative_midpoint(op,old_u,old_v,b4u,b4v,seconds)
        loss=face_dissipation_to_columns(loss_u,loss_v)
        credit=core.baseline_beta*mask*loss
        reservoir=old_energy+credit
        back=dict(kinetic_gain=0.,energy_residual=0.,limited_pairs=0,active_pairs=0)
        cap_cells=eligible_cells=0
        if core.baseline_beta:
            volume=(core.art*depth*core.fsm).double()
            e=reservoir/torch.where(volume>0,volume,1.)
            nu=core.baseline_cb*(core.dx.double()*core.dy.double()).sqrt()*torch.sqrt(2*e.clamp_min(0))*mask
            cap_cells=int(((nu>cap)&(mask>0)).sum());eligible_cells=int((mask>0).sum())
            nu=torch.minimum(nu,cap)
            u,v,reservoir,back=backscatter_exchange(op,u,v,reservoir,nu,seconds)
            wet=core.fsm.bool();dx=core.dx.double();dy=core.dy.double()
            cx=core.baseline_ke*.5*(volume[1:]+volume[:-1])/(.5*(dx[1:]+dx[:-1])).square()*(wet[1:]&wet[:-1])
            cy=core.baseline_ke*.5*(volume[:,1:]+volume[:,:-1])/(.5*(dy[:,1:]+dy[:,:-1])).square()*(wet[:,1:]&wet[:,:-1])
            reservoir,_=diffuse_energy(reservoir,volume,cx,cy,seconds)
        new_u=u.to(old_u.dtype);new_v=v.to(old_v.dtype)
        rounding=.5*((op.wu*(new_u.double().square()-u.square())).sum()+(op.wv*(new_v.double().square()-v.square())).sum())
        setattr(core,uk,new_u);setattr(core,vk,new_v);setattr(core,rk,reservoir)
        _write_velocity_increment(core,new_u-old_u,new_v-old_v,history)
        values=dict(dissipation=mid['actual_loss'],reservoir_credit=float(credit.sum()),backscatter=back['kinetic_gain'],
                    heat=float(loss.sum()-credit.sum()),roundoff=float(rounding),midpoint_residual=mid['energy_residual'],
                    backscatter_residual=back['energy_residual'],limited_pairs=back['limited_pairs'],active_pairs=back['active_pairs'],
                    cap_cells=cap_cells,eligible_cells=eligible_cells)
        for key,value in values.items():
            attr='closure_'+label+'_'+key;setattr(core,attr,getattr(core,attr)+value)


def advance_budget(core,physics,steps):
    for _ in range(steps):
        current,history=core.closure_energy,core.closure_energy_b
        physics.advance(core,steps=1,differentiable=False)
        core.closure_energy=history;core.closure_energy_b=current
        closure_substep(core,core.dti)
        core.closure_steps+=1


class BudgetRuntime(SmagorinskyRuntime):
    def __init__(self,config,device,*,cm,cc,gamma,beta,family):
        super().__init__(config,device,cm=cm,cc=cc)
        configure_budget(self.core,gamma=gamma,beta=beta,family=family)
        self.baseline=capture(self.core,cpu=False)
    def advance(self,differentiable=False,seconds=None):
        if differentiable:raise ValueError('Physics budget calibration does not train a model')
        seconds=self.interval_seconds if seconds is None else seconds
        steps,remainder=divmod(seconds,self.core.dti)
        if remainder or steps<=0:raise ValueError('Positive whole physical steps required')
        start=float(self.core.time);advance_budget(self.core,self.physics,int(steps))
        self.assert_time(start+seconds/86400);self.physical_forward_steps+=int(steps)
    def contract(self):
        value=super().contract();value['closure'].update(family=self.core.baseline_family,gamma=self.core.baseline_gamma,beta=self.core.baseline_beta,
            cb=self.core.baseline_cb,ke=self.core.baseline_ke,grid_cap=self.core.baseline_grid_cap,
            integration='prototype v2: full closure step after native step; augmented history layer',
            energy_units='extensive per reference density m5/s2',status='not calibrated; convergence gates pending')
        return value
