"""Read the closure budget from the same complete state used to commit a day.

Current and historical reservoirs are separate augmented-state accounts. Their
sum audits bookkeeping, not the kinetic energy of one physical time level.
"""
import math
import torch

LAYERS=('current','history')


def committed_budget(state):
    if state.get('complete') is not True:raise ValueError('Incomplete physical state')
    scalars=state['scalars'];tensors=state['tensors']
    accounts={}
    for layer,key in zip(LAYERS,('closure_energy','closure_energy_b')):
        values=tensors[key].double()
        if not bool(torch.isfinite(values).all()) or float(values.min())<0:
            raise ValueError('Invalid reservoir state')
        ledger={k.removeprefix('closure_'+layer+'_'):float(v) for k,v in scalars.items() if k.startswith('closure_'+layer+'_')}
        if not all(math.isfinite(v) for v in ledger.values()):raise ValueError('Nonfinite budget ledger')
        ledger.update(reservoir=float(values.sum()),minimum_reservoir=float(values.min()))
        ledger['cap_fraction']=ledger['cap_cells']/max(ledger['eligible_cells'],1)
        ledger['limited_pair_fraction']=ledger['limited_pairs']/max(ledger['active_pairs'],1)
        accounts[layer]=ledger
    credit=sum(a['reservoir_credit'] for a in accounts.values())
    returned=sum(a['backscatter'] for a in accounts.values())
    stored=sum(a['reservoir'] for a in accounts.values())
    residual=stored-credit+returned
    if abs(residual)>max(abs(credit)*1e-6,1e-8):raise ValueError('Augmented reservoir budget does not close')
    return dict(time=float(scalars['time']),accounts=accounts,
                augmented_reservoir=stored,augmented_credit=credit,augmented_returned=returned,
                augmented_residual=residual,returned_fraction_of_credit=returned/credit if credit else None,
                units='m5/s2 per reference density',
                scope='Closure-only; historical account is not extra physical kinetic energy')
