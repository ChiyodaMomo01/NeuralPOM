"""Complete, matched validation-window spectral diagnostics for physics candidates."""
import numpy as np
from neuralpom.evaluation.protocol import case_root,read_json
from neuralpom.evaluation.analysis import records


def spectral_errors(prediction,reference):
    p=np.asarray(prediction,dtype=float);r=np.asarray(reference,dtype=float)
    if p.shape!=r.shape or not np.isfinite(p).all() or not np.isfinite(r).all() or (p<0).any() or (r<0).any():raise ValueError('Invalid energy spectrum')
    ps,rs=float(p.sum()),float(r.sum())
    if rs<=0 or ps<=0:return dict(energy_ratio=ps/rs if rs>0 else None,amplitude_error=None,shape_l1=None,energy_l1=None,defined=False)
    return dict(energy_ratio=ps/rs,amplitude_error=abs(ps/rs-1),shape_l1=float(np.abs(p/ps-r/rs).sum()),energy_l1=float(np.abs(p-r).sum()/rs),defined=True)


def compare_window(protocol,candidate,baseline,days,lower=1,*,baseline_root=None):
    pairs=[];unavailable=[]
    for case in protocol['cases']:
        series=[]
        for index,name in enumerate((candidate,baseline)):
            source=baseline_root if index==1 and baseline_root is not None else protocol['output_root']
            root=case_root(source,name,case)
            if not (root/'status.json').exists():series.append(None);continue
            s=read_json(root/'status.json')
            if s['completed_days']<days or s['status'] in ('numerical_failure','infrastructure_error'):series.append(None);continue
            rows=[x for x in records(root/'spectra.jsonl',s['completed_days']) if lower<=x['lead_day']<=days]
            if [x['lead_day'] for x in rows]!=list(range(lower,days+1)):raise ValueError('Invalid spectral day coverage')
            series.append(rows)
        if any(x is None for x in series):unavailable.append(case);continue
        left,right=series
        for a,b in zip(left,right):
            if a['wavenumber']!=b['wavenumber'] or a['sigma_weights']!=b['sigma_weights']:raise ValueError('Spectral geometry differs')
            np.testing.assert_allclose(a['reference'],b['reference'],rtol=1e-12,atol=1e-18,err_msg='Reference spectra differ')
        weights=np.asarray(left[0]['sigma_weights']);representations={}
        for label in ('depth_weighted','depthmean'):
            def pooled(rows,key):
                e=np.mean([x[key] for x in rows],axis=0)
                return (e[:len(weights)]*weights[:,None]).sum(0) if label=='depth_weighted' else e[len(weights)]
            p=pooled(left,'prediction');b=pooled(right,'prediction');ref=pooled(left,'reference')
            representations[label]=dict(candidate=p.tolist(),baseline=b.tolist(),reference=ref.tolist(),candidate_errors=spectral_errors(p,ref),baseline_errors=spectral_errors(b,ref))
        pairs.append(dict(case=case,representations=representations))
    result=dict(candidate=candidate,baseline=baseline,first=lower,last=days,paired_cases=len(pairs),requested_cases=len(protocol['cases']),unavailable=unavailable,complete=not unavailable,per_case=pairs,pooled={})
    if pairs:
        for label in ('depth_weighted','depthmean'):
            arrays={k:np.mean([p['representations'][label][k] for p in pairs],axis=0) for k in ('candidate','baseline','reference')}
            c=spectral_errors(arrays['candidate'],arrays['reference']);b=spectral_errors(arrays['baseline'],arrays['reference'])
            # Numerical comparison slack only; not a physical degradation allowance.
            passes=c['defined'] and b['defined'] and all(c[k]<=b[k]+1e-12 for k in ('amplitude_error','shape_l1'))
            result['pooled'][label]=dict(candidate=c,baseline=b,passes_non_degradation=passes)
    result['passes_spectral_gate']=result['complete'] and bool(pairs) and all(x['passes_non_degradation'] for x in result['pooled'].values())
    return result
