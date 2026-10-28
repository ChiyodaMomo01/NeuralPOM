"""Finite-step dissipative midpoint solve for a frozen native stress operator.

Used to investigate an energy-auditable common B2/B3 integration. Not wired
into production or calibrated runs until the full coupled protocol passes.
"""
import torch


def dissipative_midpoint(operator,u,v,b4u,b4v,seconds,*,rtol=1e-10,max_iterations=80):
    if seconds<=0:raise ValueError('Positive physical duration required')
    u=u.detach().double();v=v.detach().double()
    b4u=b4u.detach().double();b4v=b4v.detach().double()
    wu=operator.wu.double();wv=operator.wv.double()
    def dot(a,b):return (wu*a[0]*b[0]).sum()+(wv*a[1]*b[1]).sum()
    def add(a,b,scale=1.):return (a[0]+scale*b[0],a[1]+scale*b[1])
    def force(a):return operator.dissipative_fourth(a[0],a[1],b4u,b4v)[:2]
    def system(a):return add(a,force(a),-.5*seconds)
    old=(u,v);rhs=add(old,force(old),.5*seconds);x=old
    residual=add(rhs,system(x),-1.);direction=residual
    rr=dot(residual,residual);target=rtol**2*dot(rhs,rhs).clamp_min(1e-60)
    iterations=0
    while bool(rr>target):
        if iterations>=max_iterations:raise RuntimeError('Midpoint CG failed convergence')
        ap=system(direction);denominator=dot(direction,ap)
        if not bool(denominator>0):raise RuntimeError('Nonpositive midpoint quadratic form')
        alpha=rr/denominator;x=add(x,direction,alpha);residual=add(residual,ap,-alpha)
        rr_next=dot(residual,residual);direction=add(residual,direction,rr_next/rr)
        rr=rr_next;iterations+=1
    mid=((old[0]+x[0])/2,(old[1]+x[1])/2)
    _,_,density_u,density_v=operator.dissipative_fourth(mid[0],mid[1],b4u,b4v)
    dissipated_u=seconds*wu*density_u;dissipated_v=seconds*wv*density_v
    before=.5*dot(old,old);after=.5*dot(x,x)
    expected=dissipated_u.sum()+dissipated_v.sum()
    # Stable finite-increment expression avoids subtracting two large energies.
    delta=(x[0]-old[0],x[1]-old[1])
    actual_loss=-dot(mid,delta)
    residual_energy=actual_loss-expected
    tolerance=max(float(before)*rtol*8,1e-12)
    if abs(float(residual_energy))>tolerance or float(actual_loss)<-tolerance:
        raise RuntimeError('Finite-step dissipation identity failed')
    return x[0],x[1],dissipated_u,dissipated_v,dict(iterations=iterations,
        kinetic_before=float(before),kinetic_after=float(after),actual_loss=float(actual_loss),
        dissipation_integral=float(expected),energy_residual=float(residual_energy),
        relative_linear_residual=float((rr/dot(rhs,rhs).clamp_min(1e-60)).sqrt()))
