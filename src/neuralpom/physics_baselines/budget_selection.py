"""Complete-cohort B2/B3 validation against an explicitly registered B0 run."""
import hashlib,json,math
from pathlib import Path
from neuralpom.evaluation.protocol import case_root
from .selection import FIELDS,normalized_score


def read_window(root,candidate,cases,last,first=1,*,budget_identity=None):
    rows=[];states=[];unresolved=False;failed=False;limits=[]
    for case in cases:
        directory=case_root(root,candidate,case);path=directory/'status.json'
        status=json.loads(path.read_text()) if path.exists() else dict(status='pending',completed_days=0)
        states.append(dict(case=case,**status))
        if status['status']=='infrastructure_error':unresolved=True;continue
        if status['status']=='numerical_failure':failed=True;continue
        if status['completed_days']<last:unresolved=True;continue
        daily=[json.loads(x) for x in (directory/'daily.jsonl').read_text().splitlines() if x.strip()]
        daily=[r for r in daily if first<=r['lead_day']<=last]
        if [r['lead_day'] for r in daily]!=list(range(first,last+1)):raise ValueError('Invalid committed window')
        for r in daily:
            expected=case['start_day']+r['lead_day']
            if abs(r['time']-expected)>2e-6 or abs(r['reference_time']-expected)>2e-6:raise ValueError('Reference or forecast clock differs')
            if any(not math.isfinite(r['metrics']['mse/'+f]) or r['metrics']['mse/'+f]<0 for f in FIELDS):raise ValueError('Invalid physical MSE')
            if budget_identity is not None:
                b=json.loads((directory/'budgets'/f"day{r['lead_day']:03d}.json").read_text())
                identity=dict(variant=candidate,**case,**budget_identity)
                if b['identity']!=identity or b['lead_day']!=r['lead_day'] or abs(b['time']-expected)>2e-6:raise ValueError('Budget commit provenance differs')
                values=[b[k] for k in ('augmented_reservoir','augmented_credit','augmented_returned','augmented_residual')]
                if not all(math.isfinite(v) for v in values) or any(v<0 for v in values[:3]):raise ValueError('Invalid augmented budget')
                residual=values[0]-values[1]+values[2]
                tolerance=max(abs(values[1])*1e-6,1e-8)
                if abs(residual)>tolerance or abs(residual-values[3])>tolerance:raise ValueError('Budget fails closure')
                for layer in ('current','history'):
                    account=b['accounts'][layer]
                    if account['minimum_reservoir']<0 or not all(math.isfinite(x) for x in account.values()):raise ValueError('Invalid account')
                for key,reported in zip(('reservoir','reservoir_credit','backscatter'),values[:3]):
                    recomputed=sum(b['accounts'][layer][key] for layer in ('current','history'))
                    if not math.isclose(recomputed,reported,rel_tol=1e-10,abs_tol=1e-8):raise ValueError('Account sums differ from augmented budget')
                limits.append(dict(case=case,day=r['lead_day'],current_cap_fraction=b['accounts']['current']['cap_fraction'],history_cap_fraction=b['accounts']['history']['cap_fraction']))
        rows.append({f:sum(r['metrics']['mse/'+f] for r in daily)/len(daily) for f in FIELDS})
    complete=len(rows)==len(cases) and bool(cases)
    return dict(complete=complete,unresolved=unresolved,failed=failed,statuses=states,limiter_observations=limits,mse={f:sum(r[f] for r in rows)/len(rows) for f in FIELDS} if complete else None)


def summarize(protocol,last,first=1):
    if protocol['family'] not in ('B2','B3') or protocol['selection_split']!='validation':raise ValueError('Budget validation only')
    if last not in protocol['horizons_days'] or not 1<=first<=last:raise ValueError('Unregistered window')
    reference=protocol['baseline_reference'];path=Path(reference['protocol_path'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=reference['protocol_sha256']:raise ValueError('Baseline protocol changed')
    original=json.loads(path.read_text())
    if original['core']!=protocol['core'] or original['cases']!=protocol['cases'] or original['family']!='B1' or original['selection_split']!='validation':raise ValueError('Baseline support differs')
    baseline=next(c for c in original['candidates'] if c['id']==reference['candidate'])
    if baseline['cm']!=.05 or baseline['cc']!=.01:raise ValueError('Reference is not B0')
    def manifest(root,candidate,digest):
        value=json.loads((Path(root)/'runs'/candidate/'manifest.json').read_text())
        if value['protocol_sha256']!=digest or value['weight_sha256'] is not None or value['variant']['id']!=candidate or value['variant']['core']!=protocol['core'] or value['variant']['frequency']!='1h':raise ValueError('Candidate or baseline manifest differs')
        return value['training_contract']['reference_data_identity']
    reference_identity=manifest(original['output_root'],baseline['id'],reference['protocol_sha256'])
    base=read_window(original['output_root'],baseline['id'],protocol['cases'],last,first)
    identity=dict(target_days=max(protocol['horizons_days']),weight_sha256=None,protocol_sha256=protocol['_sha256'])
    result=dict(family=protocol['family'],core=protocol['core'],first=first,last=last,baseline_reference=reference,baseline=base,candidates={},ranking=[],selection_ready=False,selected_for_next_horizon=[])
    for c in protocol['candidates']:
        if (Path(protocol['output_root'])/'runs'/c['id']/'manifest.json').exists():
            if manifest(protocol['output_root'],c['id'],protocol['_sha256'])!=reference_identity:raise ValueError('Reference dataset identity differs')
        data=read_window(protocol['output_root'],c['id'],protocol['cases'],last,first,budget_identity=identity)
        if data['complete'] and not (Path(protocol['output_root'])/'runs'/c['id']/'manifest.json').exists():raise ValueError('Complete result missing manifest')
        result['candidates'][c['id']]=data
        if base['complete'] and data['complete']:
            score,ratios=normalized_score(data['mse'],base['mse'])
            if not math.isfinite(score):raise ValueError('Nonfinite score')
            result['ranking'].append(dict(candidate=c,score=score,ratios=ratios,mse=data['mse'],passes_variable_gate=all(x<=1.1 for x in ratios.values())))
    result['ranking'].sort(key=lambda x:(x['score'],x['candidate']['id']))
    result['selection_ready']=base['complete'] and not any(c['unresolved'] for c in result['candidates'].values())
    # This selects only ten-day candidates; longer windows require spectral review.
    if result['selection_ready'] and last==10 and first==1:result['selected_for_next_horizon']=[r['candidate']['id'] for r in result['ranking']][:2]
    return result
