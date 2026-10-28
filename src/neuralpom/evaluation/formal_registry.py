"""Read frozen hourly baseline tests alongside the original evaluation.

Daily aliases remain explicit pending slots until a separate final-parameter
frequency-reuse contract is available. Calibration paths are never searched.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np
from .analysis import collect


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_contract_checks(formal, original):
    settings = original['physics_settings']
    checks = dict(data_identity=formal['reference_data_identity']==original['data_identity'],
        core=formal['closure']['reference_core']==original['core'],
        numerics=formal['closure']['reference_numerics_version']==original['numerics_version'],
        grid=formal['grid']==original['grid'],
        internal_step=formal['physical_step_seconds']==settings['dti']==original['physical_step_seconds'],
        external_step=formal['external_step_seconds']==settings['dte'],
        alpha=formal['alpha']==settings['alpha'],smoth=formal['smoth']==settings['smoth'],
        interval=formal['interval_seconds']==original['interval_seconds'],
        split=formal['split']==original['split'])
    if not all(checks.values()):
        raise ValueError('Formal/reference contract differs: '+str(checks))
    return checks


def compare_saved(left, right, allowed_differences=()):
    """Record known grid differences; never describe these as exact equality."""
    mismatches = {}
    with np.load(left,allow_pickle=False) as a, np.load(right,allow_pickle=False) as b:
        if set(a.files)!=set(b.files):raise ValueError('Saved field sets differ')
        for key in a.files:
            x,y=a[key],b[key]
            if x.shape!=y.shape or x.dtype!=y.dtype:raise ValueError('Shape/dtype differs: '+key)
            if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite saved input')
            if not np.array_equal(x,y):
                if key not in allowed_differences:raise ValueError('Saved input differs: '+key)
                delta=float(np.max(np.abs(x.astype('float64')-y.astype('float64'))))
                mismatches[key]=dict(max_absolute_difference=delta,
                    relative_to_maximum=delta/max(float(np.max(np.abs(y))),1e-30))
    return dict(formal=str(left),reference=str(right),formal_sha256=digest(left),
                reference_sha256=digest(right),differences=mismatches,exact=not mismatches)


def collect_with_formal(protocol, base):
    data=collect(protocol);sources={};provenance={};base=Path(base)
    for group in protocol['groups']:
        gid=group['id'];short='implicit' if group['core']=='implicit_p0_fixed' else group['core']
        for family in ('B1','B2','B3'):
            name=f'{gid}_physics_{family}'
            variant=dict(group,id=name,group_id=gid,stage='physics',policy='physics',baseline_family=family)
            root=base/f'{family.lower()}_{short}_test_v1';path=root/'protocol.json'
            reason='reuse_pending' if path.exists() and group['frequency']!='1h' else 'not_registered'
            if not path.exists() or group['frequency']!='1h':
                data[name]=dict(variant=variant,evaluation_split='test',availability=reason,
                    cases=[dict(case=c,status=dict(status='pending',completed_days=0),rows=[]) for c in protocol['cases']])
                provenance[name]=dict(availability=reason,reason='Formal test absent or daily reuse not yet certified')
                continue
            p=json.loads(path.read_text());launcher=root/'physics_frozen_test.py'
            if digest(launcher)!=p['launcher_sha256']:raise ValueError('Formal launcher changed')
            spec=importlib.util.spec_from_file_location('_registered_'+short+family,launcher)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            model=module.collect_records(path)
            if model['evaluation_split']!='test' or model['variant']['id']!=name:
                raise ValueError('Formal test identity differs')
            if [c['case'] for c in model['cases']]!=protocol['cases']:
                raise ValueError('Formal test origins differ')
            metadata=[]
            for old in data.values():
                if (old['variant']['group_id']!=gid or old['variant'].get('model_family')=='ai'
                    or old['variant'].get('baseline_family') in ('B1','B2','B3')):
                    continue
                manifest=Path(protocol['output_root'])/'runs'/old['variant']['id']/'manifest.json'
                m=json.loads(manifest.read_text())
                metadata.append(dict(path=str(manifest),sha256=digest(manifest),
                    checks=reference_contract_checks(model['reference_contract'],m['training_contract'])))
            saved=[];baseline=data[gid+'_physics_physics']
            for case,control in zip(model['cases'],baseline['cases']):
                if not case['status']['completed_days']:continue
                label=f"m{case['case']['member']}_day{case['case']['start_day']:04d}"
                a=Path(model['result_root'])/'runs'/name/label
                b=Path(protocol['output_root'])/'runs'/baseline['variant']['id']/label
                # Existing CPU/CUDA audits establish the source of dx/art
                # differences. Retain exact magnitudes here, not an equality flag.
                grid=compare_saved(a/'geometry.npz',b/'geometry.npz',('dx','art'))
                saved.append(dict(case=case['case'],initial=compare_saved(a/'initial.npz',b/'initial.npz'),geometry=grid))
                common=min(case['status']['completed_days'],control['status']['completed_days'])
                saved[-1]['reference_snapshots']=[dict(day=day,**compare_saved(a/f'day{day:03d}_reference.npz',b/f'day{day:03d}_reference.npz')) for day in sorted({1,common}) if day>0]
            model['availability']='registered';data[name]=model
            sources[name]=str(Path(model['result_root'])/'runs'/name)
            provenance[name]=dict(availability='registered',protocol_sha256=model['protocol_sha256'],
                manifest_comparisons=metadata,saved_inputs=saved,
                limitation='Hourly comparison only; first/latest common reference snapshots checked. Recorded scores use each run geometry. Not all-day recomputation, exact grid equality, or daily reuse certification.')
    return data,sources,provenance
