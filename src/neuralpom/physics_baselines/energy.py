"""Finite-increment, supply-limited pair exchange for B3 budget integration.

All energies below are per unit reference density (m^5/s^2); weights are
physical control volumes (m^3). This is a budget primitive, not yet a POM
backscatter parameterization. Caller must map the column reservoir to edges
conservatively and audit overlapping-edge ordering and split coupling.
"""
import torch


def return_energy_pair(left, right, volume_left, volume_right, reservoir_left,
                       reservoir_right, requested_transport):
    """Anti-diffusive velocity exchange with exact finite-step energy payment.

    A positive transport t (m^3) applies increments
        du_left=t*(u_left-u_right)/V_left,
        du_right=-t*(u_left-u_right)/V_right.
    The gain includes the quadratic increment term. Reduce t to the positive
    quadratic root affordable by the two reservoirs, and debit each reservoir
    proportionally to its available energy. Inputs must be wet, nonnegative
    reservoir energies and transport. Independent pairs can be batched.
    """
    for value in (volume_left,volume_right):
        if bool((value<=0).any()):raise ValueError('Pair volumes must be positive')
    if bool((reservoir_left<0).any() or (reservoir_right<0).any() or (requested_transport<0).any()):
        raise ValueError('Negative reservoir or transport')
    difference=left-right
    linear=difference.square()
    quadratic=.5*linear*(1/volume_left+1/volume_right)
    available=reservoir_left+reservoir_right
    # Stable root avoids subtracting two nearly equal terms. Degenerate
    # equal velocities yield no transfer or energy debit.
    denominator=linear+torch.sqrt(linear.square()+4*quadratic*available)
    affordable=torch.where(denominator>0,2*available/torch.where(denominator>0,denominator,1.),0.)
    transport=torch.minimum(requested_transport,affordable)
    requested_gain=linear*transport+quadratic*transport.square()
    fraction=torch.where(available>0,requested_gain/torch.where(available>0,available,1.),0.)
    # Up to roundoff at the root. This bounds the debit fraction, not the
    # reservoir after an overdraw; velocities use a correspondingly adjusted t.
    fraction=torch.minimum(fraction,torch.ones_like(fraction))
    paid=available*fraction
    denom_paid=linear+torch.sqrt(linear.square()+4*quadratic*paid)
    transport=torch.where(denom_paid>0,2*paid/torch.where(denom_paid>0,denom_paid,1.),0.)
    impulse=transport*difference
    return (left+impulse/volume_left,right-impulse/volume_right,
            reservoir_left*(1-fraction),reservoir_right*(1-fraction),
            dict(paid=paid,transport=transport,limited=affordable<requested_transport*(1-1e-12)))
