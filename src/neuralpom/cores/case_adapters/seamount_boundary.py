"""Registered native seamount bcondorl and conservative implicit Robin adapter.
Archived original kernels remain immutable. Both sides use zero incident flow,
fixed exterior eta=0, E/W characteristic radiation, and Orlanski 3-D boundaries.
"""
import torch,types
from neuralpom.cores.implicit_p0_fixed.boundary.closed_basin import ClosedBasinBoundary

VERSION='seamount-bcondorl-robin-v4-body-continuity'
def eta_ghost(c,x):
 x[0,:]=x[1,:];x[-1,:]=x[-2,:]
 return x*c.fsm

def bcond(c,idx,elf,uaf,vaf,uf,vf,w):
 j=slice(1,-1);n=c.kbm1
 if idx==1:return eta_ghost(c,elf),uaf,vaf,uf,vf,w
 if idx==2:
  uaf[1,j]=c.ramp*c.uabw[j]-torch.sqrt(c.grav/c.h[1,j])*(c.el[1,j]-c.elw[j]);uaf[0,j]=uaf[1,j];vaf[0,j]=0
  uaf[-1,j]=c.ramp*c.uabe[j]+torch.sqrt(c.grav/c.h[-2,j])*(c.el[-2,j]-c.ele[j]);vaf[-1,j]=0
  return elf,uaf*c.dum,vaf*c.dvm,uf,vf,w
 if idx==3:
  # Fortran bcondorl: exact zero-denominator guard, clipping and history.
  for edge,near,nexti in [(1,2,3),(-1,-2,-3)]:
   den=uf[near,j,:n]+c.ub[near,j,:n]-2*c.u[nexti,j,:n]
   den=torch.where(den==0,torch.full_like(den,.01),den)
   cl=((c.ub[near,j,:n]-uf[near,j,:n])/den).clamp(0,1)
   uf[edge,j,:n]=(c.ub[edge,j,:n]*(1-cl)+2*cl*c.u[near,j,:n])/(1+cl)
   vf[0 if edge==1 else -1,j,:n]=0
  uf[0,j,:n]=uf[1,j,:n]
  uf[:,:,:n]*=c.dum[:,:,None];vf[:,:,:n]*=c.dvm[:,:,None]
 elif idx==4:
  for edge,near,nexti,velocity,t_ext,s_ext,sign in [(0,1,2,c.ub[1,:,:n],c.tbw,c.sbw,1),(-1,-2,-3,c.ub[-1,:,:n],c.tbe,c.sbe,-1)]:
   for out,past,current,exterior in [(uf,c.tb,c.t,t_ext),(vf,c.sb,c.s,s_ext)]:
    den=out[near,:,:n]+past[near,:,:n]-2*current[nexti,:,:n]
    den=torch.where(den==0,torch.full_like(den,.01),den)
    cl=((past[near,:,:n]-out[near,:,:n])/den).clamp(0,1)
    value=(past[edge,:,:n]*(1-cl)+2*cl*current[near,:,:n])/(1+cl)
    out[edge,:,:n]=torch.where((cl==0)&(sign*velocity>=0),exterior[:,:n],value)
  uf[:,:,:n]*=c.fsm[:,:,None];vf[:,:,:n]*=c.fsm[:,:,None]
 elif idx==5:w[:,:,:n]*=c.fsm[:,:,None]
 elif idx==6:
  uf[[0,-1],:,:]=1e-10;vf[[0,-1],:,:]=1e-10
  uf=uf*c.fsm[:,:,None]+1e-10;vf=vf*c.fsm[:,:,None]+1e-10
 return elf,uaf,vaf,uf,vf,w

class Boundary(ClosedBasinBoundary):
 name=VERSION
 def apply_bcond(self,c,*args):return bcond(c,*args)
 def enforce_eta(self,c,x):x.copy_(eta_ghost(c,x))
 def prepare_eta_rhs(self,c,rhs):pass
 def apply_velocity2d(self,c,u,v):
  if u is c.utf and getattr(c,'_sea_coupling_ready',False):
   # Adjacent time intervals retain their own transport histories. RA-filtered
   # start velocity and its matching RA-filtered eta define the first endpoint.
   hb=c.h+c.elb;hn=c.h+c.el
   u[1:,:]=.25*(c.uab[1:,:]*(hb[1:,:]+hb[:-1,:])+c.ua[1:,:]*(hn[1:,:]+hn[:-1,:]))
   v[:,1:]=.25*(c.vab[:,1:]*(hb[:,1:]+hb[:,:-1])+c.va[:,1:]*(hn[:,1:]+hn[:,:-1]))
   qx=.5*(c.utb+u);qy=.5*(c.vtb+v)
   fx=torch.zeros_like(c.el);fy=torch.zeros_like(c.el)
   fx[1:,:]=qx[1:,:]*.5*(c.dy[1:,:]+c.dy[:-1,:])
   fy[:,1:]=qy[:,1:]*.5*(c.dx[:,1:]+c.dx[:,:-1])
   div=(fx[2:,1:-1]-fx[1:-1,1:-1]+fy[1:-1,2:]-fy[1:-1,1:-1])/(c.dx[1:-1,1:-1]*c.dy[1:-1,1:-1])
   flux=.5*(c.vfluxb[1:-1,1:-1]+c.vfluxf[1:-1,1:-1])
   c.etf[1:-1,1:-1]=c.etb[1:-1,1:-1]-c.dti2*(div+flux)
   c.etf.copy_(eta_ghost(c,c.etf))
   res=(c.etf[1:-1,1:-1]-c.etb[1:-1,1:-1])/c.dti2+div+flux
   c._sea_body_continuity_max_ms=float(res.abs().max())
   c._sea_body_eta_baro_difference_m=float((c.etf-c.el).abs().max())
   c._sea_body_boundary_transport_m3s=float((fx[-1,1:-1]-fx[1,1:-1]).sum())
   c._sea_coupling_ready=False
  return u,v

def predictor(c):
 c._sea_old_depth=c.d.clone();c._sea_old_eta_b=c.elb.clone()
 u,v,du,dv=c._compute_barotropic_predictor()
 # Incident part only: unknown radiating eta enters the Robin operator once.
 u[1,1:-1]=c.ramp*c.uabw[1:-1];u[0,1:-1]=u[1,1:-1];u[-1,1:-1]=c.ramp*c.uabe[1:-1]
 v[0,:]=0;v[-1,:]=0
 return u*c.dum,v*c.dvm,du,dv

def solve(c,rhs,d_2d,max_iter=300,tol=1e-12,print_cg_residual=False,initial_guess=None):
 free=c.fsm.bool().clone();free[[0,-1],:]=False;free[:,[0,-1]]=False;area=c.art*free
 x=c.el.clone().masked_fill(~free,0);beta_w=torch.sqrt(c.grav/c.h[1,1:-1]);beta_e=torch.sqrt(c.grav/c.h[-2,1:-1])
 dy_w=.5*(c.dy[1,1:-1]+c.dy[0,1:-1]);dy_e=.5*(c.dy[-1,1:-1]+c.dy[-2,1:-1])
 # Boundary flux H_new * (u_inc +/- beta*eta_new) is nonlinear in eta.
 # Picard iterations retain the true new-depth flux, not a mismatched ghost.
 for nonlinear in range(6):
  guess=eta_ghost(c,x.clone());new_depth=c.h+guess
  hw=.5*(new_depth[1,1:-1]+new_depth[0,1:-1]);he=.5*(new_depth[-1,1:-1]+new_depth[-2,1:-1])
  rw=2*c.dti*hw*dy_w*beta_w/c.art[1,1:-1];re=2*c.dti*he*dy_e*beta_e/c.art[-2,1:-1]
  def op(v):
   z=eta_ghost(c,v.clone());out=c.apply_elliptic_operator(z,d_2d)
   out[1,1:-1]+=rw*z[1,1:-1];out[-2,1:-1]+=re*z[-2,1:-1]
   return out.masked_fill(~free,0)
  b=rhs.masked_fill(~free,0).clone()
  # The new-depth incident flux uses this same Picard geometry.
  # Here u_inc=0, but keep the general exterior eta term explicit in RHS.
  b[1,1:-1]+=2*c.dti*hw*dy_w*(c.ramp*c.uabw[1:-1]+beta_w*c.elw[1:-1])/c.art[1,1:-1]
  b[-2,1:-1]-=2*c.dti*he*dy_e*(c.ramp*c.uabe[1:-1]-beta_e*c.ele[1:-1])/c.art[-2,1:-1]
  # predictor already contains old-depth incident flux; subtract that term
  old=c._sea_old_depth
  b[1,1:-1]-=2*c.dti*.5*(old[1,1:-1]+old[0,1:-1])*dy_w*c.ramp*c.uabw[1:-1]/c.art[1,1:-1]
  b[-2,1:-1]+=2*c.dti*.5*(old[-1,1:-1]+old[-2,1:-1])*dy_e*c.ramp*c.uabe[1:-1]/c.art[-2,1:-1]
  r=b-op(x);p=r.clone();rr=(area*r*r).sum();rr0=(area*b*b).sum().clamp_min(1e-40)
  for it in range(max(max_iter,300)):
   if float(rr)<=max(1e-28,float(rr0)*1e-22):break
   ap=op(p);den=(area*p*ap).sum()
   if not torch.isfinite(den) or float(den)<=0:raise RuntimeError('Nonpositive/nonfinite Robin CG pAp')
   a=rr/den;x=x+a*p;r=r-a*ap;new=(area*r*r).sum();p=r+(new/rr.clamp_min(1e-40))*p;rr=new
  c._sea_cg_iterations=it+1;c._sea_cg_relative_residual=float(torch.sqrt(rr/rr0))
  true_residual=b-op(x);c._sea_cg_true_relative_residual=float(torch.sqrt((area*true_residual.square()).sum()/rr0));c._sea_cg_true_max_m=float(true_residual.abs().max())
  change=float((eta_ghost(c,x.clone())-guess).abs().max())
  if change<1e-11:break
 c._sea_picard_iterations=nonlinear+1
 c._sea_picard_final_change_m=change;c._sea_picard_converged=change<1e-11
 if not c._sea_picard_converged:raise RuntimeError('Robin boundary Picard did not converge')
 return eta_ghost(c,x)

def corrector(c,u,v,du,dv):
 old=c._sea_old_depth;new=c.h+c.elf
 # Frozen interior face-depth exactly matches elliptic L; division by the
 # new face depth makes the resulting actual transport equal that solved.
 hu=.5*(old[1:,:]+old[:-1,:]);hv=.5*(old[:,1:]+old[:,:-1])
 hun=.5*(new[1:,:]+new[:-1,:]);hvn=.5*(new[:,1:]+new[:,:-1])
 dx=.5*(c.dx[1:,:]+c.dx[:-1,:]);dy=.5*(c.dy[:,1:]+c.dy[:,:-1])
 uf=torch.zeros_like(u);vf=torch.zeros_like(v)
 uf[1:,:]=(hu*u[1:,:]-2*c.dti*c.grav*c.alpha*hu*(c.elf[1:,:]-c.elf[:-1,:])/dx)/hun
 vf[:,1:]=(hv*v[:,1:]-2*c.dti*c.grav*c.alpha*hv*(c.elf[:,1:]-c.elf[:,:-1])/dy)/hvn
 uf[1,1:-1]=c.ramp*c.uabw[1:-1]-torch.sqrt(c.grav/c.h[1,1:-1])*(c.elf[1,1:-1]-c.elw[1:-1]);uf[0,1:-1]=uf[1,1:-1]
 uf[-1,1:-1]=c.ramp*c.uabe[1:-1]+torch.sqrt(c.grav/c.h[-2,1:-1])*(c.elf[-2,1:-1]-c.ele[1:-1])
 uf*=c.dum;vf*=c.dvm
 fx=torch.zeros_like(c.el);fy=torch.zeros_like(c.el)
 fx[1:,:]=hun*.5*(c.dy[1:,:]+c.dy[:-1,:])*uf[1:,:]
 fy[:,1:]=hvn*.5*(c.dx[:,1:]+c.dx[:,:-1])*vf[:,1:]
 div=fx[2:,1:-1]-fx[1:-1,1:-1]+fy[1:-1,2:]-fy[1:-1,1:-1]
 residual=c.elf[1:-1,1:-1]-c._sea_old_eta_b[1:-1,1:-1]+2*c.dti*(div/c.art[1:-1,1:-1]+c.vfluxf[1:-1,1:-1])
 c._sea_continuity_max_m=float(residual.abs().max());c._sea_continuity_volume_residual_m3=float((residual*c.art[1:-1,1:-1]*c.fsm[1:-1,1:-1]).sum())
 c._sea_coupling_ready=True
 return uf,vf

def install(c,family):
 c._case_adapter_version=VERSION
 if family=='explicit':c.bcond=types.MethodType(bcond,c)
 else:
  c._implicit_boundary_policy=Boundary()
  c._barotropic_predictor_with_bc=types.MethodType(predictor,c)
  c.solve_implicit_eta_cg=types.MethodType(solve,c)
  c._barotropic_corrector_with_bc=types.MethodType(corrector,c)
