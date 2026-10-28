"""Private closed-DG Fortran turbulence BC and conservative implicit transport."""
import types
import torch
from neuralpom.cores.implicit_p0_fixed.boundary.closed_basin import ClosedBasinBoundary
from neuralpom.cores.explicit.baroclinic import BaroclinicMixin

class FortranTurbulenceClosedBasin(ClosedBasinBoundary):
    name='registered_dg_closed_with_pom08_turbulence_boundary'
    def apply_bcond(self,c,idx,*args):
        if idx==6:return BaroclinicMixin.bcond(c,idx,*args)
        return super().apply_bcond(c,idx,*args)

def _predictor(c):
    c._dg_operator_depth=c.d.clone()
    c._dg_continuity_previous_eta=c.elb.clone()
    return c._dg_native_predictor()

def _rhs(c,*args):
    fu,fv,r=c._dg_native_rhs(*args)
    c._dg_rhs=r.clone()
    return fu,fv,r

def residual(c,u,v):
    new=c.h+c.elf
    fx=.25*(new[1:,1:]+new[:-1,1:])*(c.dy[1:,1:]+c.dy[:-1,1:])*u[1:,1:]
    fy=.25*(new[1:,1:]+new[1:,:-1])*(c.dx[1:,1:]+c.dx[1:,:-1])*v[1:,1:]
    div=fx[1:,:-1]-fx[:-1,:-1]+fy[:-1,1:]-fy[:-1,:-1]
    r=c.elf[1:-1,1:-1]-c._dg_continuity_previous_eta[1:-1,1:-1]+2*c.dti*(div/c.art[1:-1,1:-1]+c.vfluxf[1:-1,1:-1])
    wet=c.fsm[1:-1,1:-1]>0
    a=c.apply_elliptic_operator(c.elf,c._dg_operator_depth)-c._dg_rhs
    return {'actual_continuity_max_m':float(abs(r[wet]).max()),'actual_continuity_rms_m':float(torch.sqrt((r[wet]**2).mean())),
            'elliptic_residual_max_m':float(abs(a[1:-1,1:-1][wet]).max()),'elliptic_residual_rms_m':float(torch.sqrt((a[1:-1,1:-1][wet]**2).mean()))}

def _corrector(c,us,vs,denu,denv):
    old=c._dg_operator_depth;new=c.h+c.elf
    u=us.clone();v=vs.clone()
    # Transport equation uses the same old face H as the elliptic operator.
    ou=old[1:-1,1:-1]+old[:-2,1:-1];ov=old[1:-1,1:-1]+old[1:-1,:-2]
    nu=new[1:-1,1:-1]+new[:-2,1:-1];nv=new[1:-1,1:-1]+new[1:-1,:-2]
    grad_u=.25*c.grav*(c.dy[1:-1,1:-1]+c.dy[:-2,1:-1])*ou*c.alpha*(c.elf[1:-1,1:-1]-c.elf[:-2,1:-1])
    grad_v=.25*c.grav*(c.dx[1:-1,1:-1]+c.dx[1:-1,:-2])*ov*c.alpha*(c.elf[1:-1,1:-1]-c.elf[1:-1,:-2])
    u[1:-1,1:-1]=(us[1:-1,1:-1]*denu-4*c.dti*grad_u)/(nu*c.aru[1:-1,1:-1])*c.dum[1:-1,1:-1]
    v[1:-1,1:-1]=(vs[1:-1,1:-1]*denv-4*c.dti*grad_v)/(nv*c.arv[1:-1,1:-1])*c.dvm[1:-1,1:-1]
    _,u,v,_,_,_=c.bcond(2,c.elf,u,v,c.uf,c.vf,c.w)
    c._dg_continuity_diagnostic=residual(c,u,v)
    return u,v

def _legacy_corrector_probe(c,*args):
    u,v=c._dg_native_corrector(*args)
    c._dg_continuity_diagnostic=residual(c,u,v)
    return u,v

def install(c,conservative=True):
    c._implicit_boundary_policy=FortranTurbulenceClosedBasin()
    c._dg_native_predictor=c._barotropic_predictor_with_bc
    c._dg_native_rhs=c._barotropic_transport_rhs_full
    c._dg_native_corrector=c._barotropic_corrector_with_bc
    c._barotropic_predictor_with_bc=types.MethodType(_predictor,c)
    c._barotropic_transport_rhs_full=types.MethodType(_rhs,c)
    c._barotropic_corrector_with_bc=types.MethodType(_corrector if conservative else _legacy_corrector_probe,c)
    c._dg_geometry_policy='same_old_face_depth_operator_corrector; new_depth_velocity_conversion' if conservative else 'legacy_corrector_with_diagnostic'
    return c
