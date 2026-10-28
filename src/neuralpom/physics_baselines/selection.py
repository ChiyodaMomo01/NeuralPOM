"""Pre-registered complete-window validation selection; failures stay visible."""
import argparse,json,math
from pathlib import Path
from neuralpom.v1.checkpoint import atomic_json

FIELDS=('u','v','t','s','el')


def normalized_score(candidate,baseline):
    ratios={f:candidate[f]/baseline[f] if baseline[f]>0 else (1. if candidate[f]==0 else math.inf) for f in FIELDS}
    return sum(ratios.values())/len(FIELDS),ratios


def summarize(protocol,days):
    if days not in protocol['horizons_days']:raise ValueError('Unregistered horizon')
    root=Path(protocol['output_root']);records={};state={}
    for candidate in protocol['candidates']:
        cid=candidate['id'];rows=[];state[cid]=[]
        for case in protocol['cases']:
            # Import the established directory convention rather than guessing.
            from neuralpom.evaluation.protocol import case_root
            path=case_root(root,cid,case);status= json.loads((path/'status.json').read_text()) if (path/'status.json').exists() else dict(status='pending',completed_days=0)
            state[cid].append(dict(case=case,status=status['status'],completed_days=status['completed_days'],error=status.get('error')))
            if status['status'] in ('numerical_failure','infrastructure_error') or status['completed_days']<days:continue
            daily=[json.loads(line) for line in (path/'daily.jsonl').read_text().splitlines() if line.strip()]
            daily=[r for r in daily if r['lead_day']<=days]
            if [r['lead_day'] for r in daily]!=list(range(1,days+1)):raise ValueError('Incomplete or duplicate committed window')
            for row in daily:
                if abs(row['time']-case['start_day']-row['lead_day'])>2e-6:raise ValueError('Misaligned calibration time')
            rows.append({f:sum(r['metrics']['mse/'+f] for r in daily)/days for f in FIELDS})
        if len(rows)==len(protocol['cases']):records[cid]={f:sum(r[f] for r in rows)/len(rows) for f in FIELDS}
    base=next(c['id'] for c in protocol['candidates'] if c['cm']==.05 and c['cc']==.01)
    result=dict(horizon=days,selection_split='validation',case_support=protocol['cases'],statuses=state,baseline=base,ranking=[],selection_ready=False,selected_for_next_horizon=[])
    if base not in records:return result
    unresolved=[]
    for candidate in protocol['candidates']:
        cid=candidate['id'];states=state[cid]
        if any(s['status']=='infrastructure_error' for s in states):unresolved.append(cid);continue
        if cid not in records:
            if not any(s['status']=='numerical_failure' for s in states):unresolved.append(cid)
            continue
        score,ratios=normalized_score(records[cid],records[base])
        if not math.isfinite(score):raise ValueError('Nonfinite selection score')
        result['ranking'].append(dict(candidate=candidate,score=score,ratios=ratios,mse=records[cid],passes_variable_gate=all(x<=1.1 for x in ratios.values())))
    result['ranking'].sort(key=lambda x:(x['score'],x['candidate']['id']))
    result['selection_ready']=not unresolved
    result['unresolved']=unresolved
    if result['selection_ready']:
        result['selected_for_next_horizon']=[x['candidate']['id'] for x in result['ranking'] if x['passes_variable_gate']][:2]
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--days',type=int,required=True);p.add_argument('--out',required=True);a=p.parse_args()
    result=summarize(json.loads(Path(a.protocol).read_text()),a.days);atomic_json(a.out,result)
    print(json.dumps(dict(selection_ready=result['selection_ready'],ranked=len(result['ranking']),selected=result['selected_for_next_horizon'])))
