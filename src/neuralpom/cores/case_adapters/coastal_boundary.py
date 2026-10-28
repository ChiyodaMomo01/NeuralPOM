"""Independent POM08 tide boundary adapter; not installed in production.
Explicit elevation/2-D velocity equations translated from archived bcond.
Implicit extends its elliptic solve with a fixed-node lift and area-weighted CG.
"""
import torch,numpy as np,types
from .kernels.coastal_explicit_baroclinic import BaroclinicMixin
from neuralpom.cores.implicit_p0_fixed.boundary.closed_basin import ClosedBasinBoundary

def tide(c,seconds):
 return c._tide_amp*torch.cos((29.03225806451612*3.1416/180)*seconds/3600-c._tide_phase*3.1416/180)
def bcond(c,idx,elf,uaf,vaf,uf,vf,w):
 if idx==1:
  if c._validation_family=='explicit':
   c._tide_substep+=1;sec=c.time*86400-c.dti+c.dte*(c._tide_substep-1)
  else:sec=c.time*86400-c.dte
  target=tide(c,sec)
  if c._validation_family=="explicit" and hasattr(c,"_coastal_tide_overwrite_sum_m3"):
   delta=((target-elf[c._tide_i,c._tide_j])*c.art[c._tide_i,c._tide_j]*c.fsm[c._tide_i,c._tide_j]).sum()
   c._coastal_tide_overwrite_last_m3.copy_(delta)
   c._coastal_tide_overwrite_sum_m3.add_(delta)
  elf[c._tide_i,c._tide_j]=target;elf=elf*c.fsm
  return elf,uaf,vaf,uf,vf,w
 if idx==2:
  j=slice(1,-1);i=slice(1,-1)
  uaf[-1,j]=uaf[-2,j]+c.rfe*torch.sqrt(c.grav/c.d[-2,j])*(c.el[-2,j]-c.el[-1,j]);vaf[-1,j]=0
  uaf[1,j]-=c.rfw*torch.sqrt(c.grav/c.d[1,j])*(c.el[1,j]-c.el[0,j]);uaf[0,j]=uaf[1,j];vaf[0,j]=0
  vaf[i,-1]=vaf[i,-2]+c.rfn*torch.sqrt(c.grav/c.d[i,-2])*(c.el[i,-2]-c.el[i,-1]);uaf[i,-1]=0
  vaf[i,1]-=c.rfs*torch.sqrt(c.grav/c.d[i,1])*(c.el[i,1]-c.el[i,0]);vaf[i,0]=vaf[i,1];uaf[i,0]=0
  return elf,uaf*c.dum,vaf*c.dvm,uf,vf,w
 return BaroclinicMixin.bcond(c,idx,elf,uaf,vaf,uf,vf,w)
class Boundary(ClosedBasinBoundary):
 name='independent_original_pom08_tide_adapter'
 def apply_bcond(self,c,*args):return bcond(c,*args)
 def enforce_eta(self,c,x):
  x[c._fixed]=0;x[c._tide_i,c._tide_j]=tide(c,c.time*86400-c.dte)
 def prepare_eta_rhs(self,c,rhs):return None

def solve(c,rhs,d_2d,max_iter=200,tol=1e-12,print_cg_residual=False,initial_guess=None):
 # Boundary nodes are not unknowns. Subtract the tidal lift before solving
 # the homogeneous problem; weight by cell area for the flux-form operator.
 lift=torch.zeros_like(rhs)
 if hasattr(c,'_tide_i'):lift[c._tide_i,c._tide_j]=tide(c,c.time*86400-c.dte)
 free=~c._fixed;weight=c.art*free
 def op(v):
  v=v.masked_fill(c._fixed,0);return c.apply_elliptic_operator(v,d_2d).masked_fill(c._fixed,0)
 b=(rhs-c.apply_elliptic_operator(lift,d_2d)).masked_fill(c._fixed,0)
 x=torch.zeros_like(rhs);r=b.clone();p=r.clone();rr=(weight*r*r).sum();rr0=rr
 it=0
 for it in range(min(max_iter,500)):
  if float(rr)<=max(1e-30,float(rr0)*1e-24):break
  Ap=op(p);den=(weight*p*Ap).sum();a=rr/(den+1e-40);x=x+a*p;r=r-a*Ap;new=(weight*r*r).sum();p=r+(new/(rr+1e-40))*p;rr=new
 c._validation_eta_iterations=it+1;c._validation_eta_relative_residual=float(torch.sqrt(rr/(rr0+1e-40)))
 return x+lift

def install(c,family,path):
 data=np.loadtxt(path,max_rows=99)
 c._validation_family=family;c._tide_substep=0
 c._tide_i=torch.as_tensor(data[:,0]-1,dtype=torch.long,device=c.device);c._tide_j=torch.as_tensor(data[:,1]-1,dtype=torch.long,device=c.device)
 c._tide_amp=torch.as_tensor(data[:,2],dtype=c.el.dtype,device=c.device);c._tide_phase=torch.as_tensor(data[:,3],dtype=c.el.dtype,device=c.device)
 c.rfe=c.rfw=c.rfn=c.rfs=1.
 c._fixed=torch.ones_like(c.fsm,dtype=torch.bool);c._fixed[1:-1,1:-1]=False;c._fixed |= c.fsm==0;c._fixed[c._tide_i,c._tide_j]=True
 if family=='explicit':c.bcond=types.MethodType(bcond,c)
 else:
  c._implicit_boundary_policy=Boundary();c.solve_implicit_eta_cg=types.MethodType(solve,c)

class SeamountBoundary(ClosedBasinBoundary):
 name='original_pom08_open_seamount_adapter'
 def apply_bcond(self,c,idx,elf,uaf,vaf,uf,vf,w):
  if idx==1:return elf*c.fsm,uaf,vaf,uf,vf,w
  if idx==2:return bcond(c,idx,elf,uaf,vaf,uf,vf,w)
  return BaroclinicMixin.bcond(c,idx,elf,uaf,vaf,uf,vf,w)
