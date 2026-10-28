"""Recompute published formal comparisons from their exported daily CSV."""
from pathlib import Path
from collections import defaultdict
import csv
import json
import math


def verify_comparison(directory, protocol):
    root=Path(directory)/'formal_comparison'
    summary=json.loads((root/'summary.json').read_text())
    table=defaultdict(dict)
    fields=('u','v','t','s','el')
    def close(a,b):
        if a is None or b is None:
            if a is not None or b is not None:raise ValueError('Missing-value mismatch')
        elif not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-14):
            raise ValueError(f'Exported metric mismatch: {a}, {b}')
    row_count=0
    for row in csv.DictReader((root/'daily_mse.csv').open()):
        key=(row['variant'],int(row['member']),int(row['start_day']));day=int(row['lead_day'])
        if day in table[key]:raise ValueError('Duplicate daily row')
        values={f:float(row['mse_'+f]) for f in fields}
        if not all(math.isfinite(v) and v>=0 for v in values.values()):raise ValueError('Invalid MSE')
        table[key][day]=values;row_count+=1
    for days in table.values():
        if sorted(days)!=list(range(1,len(days)+1)):raise ValueError('Discontinuous CSV')
    if row_count!=sum(g['completed_case_days'] for g in summary['groups'].values()):raise ValueError('Summary row count differs')
    seen=set()
    for row in csv.DictReader((root/'paired_mse.csv').open()):
        key=tuple(row[k] for k in ('group','baseline','model','window','field'))
        if key in seen:raise ValueError('Duplicate paired row')
        seen.add(key)
        end=int(row['window'].split('_')[0]);start=end-49 if row['window'].endswith('last50') else 1
        comparator=row['group']+'_physics_'+('physics' if row['baseline']=='B0' else row['baseline'])
        left=[];right=[]
        for case in protocol['cases']:
            a=table[(row['model'],case['member'],case['start_day'])]
            b=table[(comparator,case['member'],case['start_day'])]
            if all(d in a and d in b for d in range(start,end+1)):
                left.append(sum(a[d][row['field']] for d in range(start,end+1))/(end-start+1))
                right.append(sum(b[d][row['field']] for d in range(start,end+1))/(end-start+1))
        pair=summary['physics_pairs'][row['group']][row['baseline']][row['model']][row['window']]
        if len(left)!=int(row['paired_cases']) or len(left)!=pair['paired_cases']:raise ValueError('Paired coverage differs')
        x=sum(left)/len(left) if left else None;y=sum(right)/len(right) if right else None
        expected={'model_mse':x,'comparator_mse':y,'relative_mse_change':x/y-1 if y else None}
        for name,value in expected.items():
            close(float(row[name]) if row[name] else None,value)
            close(pair['metrics'][row['field']][name],value)
    expected={(gid,family,model,window,field) for gid,baselines in summary['physics_pairs'].items()
        for family,models in baselines.items() for model,windows in models.items() for window in windows for field in fields}
    if seen!=expected:raise ValueError('Missing or extra paired rows')
    return dict(daily_rows=row_count,paired_metric_rows=len(seen),scope='Exported daily CSV to paired CSV and summary; not raw-field recomputation')
