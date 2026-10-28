"""Read-only equality audit of saved reference inputs; not rollout equivalence."""
import hashlib
from pathlib import Path
import numpy as np


def compare_reference_arrays(left, right, days):
    """Compare initial state, geometry and explicitly requested reference days.

    Callers must first validate manifests and committed-day bounds. This checks
    saved inputs only; it cannot establish equal numerical advancement, hidden
    restart state, independence of origins, or correctness of the reference.
    """
    days=list(days)
    if days!=sorted(set(days)) or any(not isinstance(d,int) or d<1 for d in days):
        raise ValueError('Reference days must be distinct positive increasing integers')
    left,right=Path(left),Path(right)
    fields=('u','v','t','s','el','time')
    geometry=('fsm','dum','dvm','h','art','dx','dy','dz','zz','time')
    files=[('initial.npz',fields),('geometry.npz',geometry)]
    files += [(f'day{day:03d}_reference.npz',fields) for day in days]
    evidence=[]
    for name,keys in files:
        with np.load(left/name,allow_pickle=False) as a,np.load(right/name,allow_pickle=False) as b:
            for key in keys:
                if a[key].dtype!=b[key].dtype or not np.array_equal(a[key],b[key],equal_nan=True):
                    raise ValueError(f'Reference input differs: {name}:{key}')
            evidence.append(dict(file=name,arrays=list(keys),
                                 left_sha256=hashlib.sha256((left/name).read_bytes()).hexdigest(),
                                 right_sha256=hashlib.sha256((right/name).read_bytes()).hexdigest()))
    return dict(left=str(left),right=str(right),reference_days=days,files=evidence,
                scope='Saved initial fields, geometry and specified reference days only; not hidden-state or rollout equivalence.')
