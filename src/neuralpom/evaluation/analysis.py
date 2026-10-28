"""MSE first; retain numerical failures, partial coverage, and paired support."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .protocol import FIELDS, variants, case_root, read_json, atomic_json


def records(path, committed=None):
    if not Path(path).exists():
        return []
    rows = []
    lines = Path(path).read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # Only a concurrently appended final line may be incomplete.
            if index != len(lines) - 1:
                raise
            break
        if committed is None or row['lead_day'] <= committed:
            rows.append(row)
    return rows


def collect(protocol):
    result = {}
    contracts, weights, references = {}, {}, {}
    for variant in variants(protocol):
        manifest_path=Path(protocol['output_root'])/'runs'/variant['id']/'manifest.json'
        if manifest_path.exists():
            manifest=read_json(manifest_path)
            if manifest['variant'] != variant:
                raise ValueError('Variant manifest differs from the analysis protocol')
            group=variant['group_id']
            contract=manifest['training_contract']
            if variant.get('model_family') == 'ai':
                references.setdefault(group, []).append(contract)
            else:
                if group in contracts and contracts[group] != contract:
                    raise ValueError('Paired variants have different data/calibration/core contracts')
                contracts[group]=contract
            if variant['stage']!='physics':
                key=(group,variant['stage'],variant.get('model_family', 'hybrid'),
                     variant['id'] if variant.get('model_family') == 'ai' else '')
                digest=manifest['weight_sha256']
                if key in weights and weights[key]!=digest:
                    raise ValueError('Paired inference policies use different checkpoints')
                weights[key]=digest
        cases = []
        for case in protocol['cases']:
            root = case_root(protocol['output_root'], variant['id'], case)
            status = read_json(root / 'status.json') if (root / 'status.json').exists() else dict(
                status='pending', completed_days=0)
            rows = records(root / 'daily.jsonl', status['completed_days'])
            if [r['lead_day'] for r in rows] != list(range(1, status['completed_days'] + 1)):
                raise ValueError(f'Missing or duplicated committed daily records: {root}')
            for row in rows:
                expected = case['start_day'] + row['lead_day']
                if abs(row['time'] - expected) > 2e-6 or abs(row['reference_time'] - expected) > 2e-6:
                    raise ValueError(f'Misaligned time in {root}')
            cases.append(dict(case=case, status=status, rows=rows))
        result[variant['id']] = dict(variant=variant, cases=cases)
    for group, entries in references.items():
        if group not in contracts:
            raise ValueError('AI comparison lacks paired hybrid/physics reference contract')
        other = contracts[group]
        for contract in entries:
            if (contract['reference_core'] != other['core'] or
                any(contract[key] != other[key] for key in ('data_identity', 'split')) or
                contract['codec']['grid'] != [other['grid']['im'], other['grid']['jm'], 16]):
                raise ValueError('AI/hybrid reference data, split, or grid differs')
    return result


def paired_period(model, baseline, days, lower=1):
    """Only whole requested windows; expose survivors instead of hiding attrition."""
    pairs = []
    for m, b in zip(model['cases'], baseline['cases']):
        if m['case'] != b['case']:
            raise ValueError('Paired initial conditions differ')
        mr = [r for r in m['rows'] if lower <= r['lead_day'] <= days]
        br = [r for r in b['rows'] if lower <= r['lead_day'] <= days]
        if len(mr) != days - lower + 1 or len(br) != days - lower + 1:
            continue
        values = {}
        for f in FIELDS:
            x = float(np.mean([r['metrics'][f'mse/{f}'] for r in mr]))
            y = float(np.mean([r['metrics'][f'mse/{f}'] for r in br]))
            values[f] = dict(mse=x, physics_mse=y)
        pairs.append(dict(**m['case'], values=values))
    metrics = {}
    for f in FIELDS:
        x = float(np.mean([p['values'][f]['mse'] for p in pairs])) if pairs else None
        y = float(np.mean([p['values'][f]['physics_mse'] for p in pairs])) if pairs else None
        metrics[f] = dict(mse=x, physics_mse=y,
                          relative_mse_change=(x / y - 1) if y and x is not None else None)
    return dict(start_lead=lower, end_lead=days, paired_cases=len(pairs), per_case=pairs, metrics=metrics,
                conditional_on_full_window=True)


def policy_pair(raw, constrained, days):
    paired = paired_period(raw, constrained, days)
    # Reuse strict matching, but expose neutral names for this comparison.
    return dict(paired_cases=paired['paired_cases'], metrics={f: dict(
        raw_mse=v['mse'], constrained_mse=v['physics_mse'],
        raw_relative_to_constrained=v['relative_mse_change']) for f, v in paired['metrics'].items()})


def model_comparison(left, right, days, lower=1):
    result=paired_period(left,right,days,lower)
    result.update(model=left['variant']['id'],comparator=right['variant']['id'])
    for values in result['metrics'].values():
        values['model_mse']=values.pop('mse')
        values['comparator_mse']=values.pop('physics_mse')
    for case in result['per_case']:
        for values in case['values'].values():
            values['model_mse']=values.pop('mse')
            values['comparator_mse']=values.pop('physics_mse')
    return result


def fixed_curve_windows(protocol, data):
    """Same origins for every lead and all variants within one core/frequency."""
    result = {}
    for group in protocol['groups']:
        models = {name: model for name, model in data.items()
                  if model['variant']['group_id'] == group['id']}
        result[group['id']] = {}
        for horizon in sorted(set([10, *protocol['horizons_days']])):
            eligible = []
            lookups = {name: {(c['case']['member'], c['case']['start_day']): c
                             for c in model['cases']} for name, model in models.items()}
            for case in protocol['cases']:
                key = (case['member'], case['start_day'])
                if models and all(key in lookup and len(lookup[key]['rows']) >= horizon
                                  for lookup in lookups.values()):
                    eligible.append(case)
            daily = {}
            for name, lookup in lookups.items():
                daily[name] = [dict(lead_day=lead, n=len(eligible), mse={f: float(np.mean([
                    lookup[(c['member'], c['start_day'])]['rows'][lead-1]['metrics'][f'mse/{f}']
                    for c in eligible])) for f in FIELDS})
                    for lead in range(1, horizon+1)] if eligible else []
            result[group['id']][str(horizon)] = dict(horizon=horizon, n=len(eligible),
                cases=eligible, daily=daily, conditional_on_full_window=True)
    return result


def summarize(protocol, data=None):
    data = collect(protocol) if data is None else data
    result = dict(schema='neuralpom-evaluation-summary-v2', primary_metric='spatially weighted physical-unit MSE',
                  n_requested=len(protocol['cases']), groups={}, correction_pairs={}, optimized_pairs={}, stage_pairs={})
    scheduler = Path(protocol['output_root']) / 'scheduler.json'
    result['worker_errors'] = [w for w in read_json(scheduler)['workers'] if w['returncode'] not in (None, 0)] if scheduler.exists() else []
    if protocol.get('reuse'):
        parent_scheduler=Path(protocol['reuse']['root'])/'scheduler.json'
        if parent_scheduler.exists():
            result['worker_errors'].extend(w for w in read_json(parent_scheduler)['workers'] if w['returncode'] not in (None,0))
    for name, model in data.items():
        variant = model['variant']
        baseline = data[f"{variant['group_id']}_physics_physics"]
        status_counts = {key: sum(c['status']['status'] == key for c in model['cases'])
                         for key in ('pending', 'running', 'complete', 'numerical_failure', 'infrastructure_error')}
        periods = {str(d): paired_period(model, baseline, d) for d in protocol['horizons_days']}
        periods.update({f'{d}_last50':paired_period(model,baseline,d,max(1,d-49))
                        for d in protocol['horizons_days'] if d>50})
        # A fixed early horizon is explicitly labelled provisional.
        periods['preview10'] = paired_period(model, baseline, 10)
        daily = []
        for lead in range(1, max(protocol['horizons_days']) + 1):
            rows = [c['rows'][lead - 1] for c in model['cases'] if len(c['rows']) >= lead]
            if rows:
                daily.append(dict(lead_day=lead, n=len(rows), cases=[c['case'] for c in model['cases'] if len(c['rows']) >= lead], mse={f: float(np.mean([
                    r['metrics'][f'mse/{f}'] for r in rows])) for f in FIELDS}))
        result['groups'][name] = dict(variant=variant, statuses=status_counts, periods=periods, daily=daily,
            completed_case_days=sum(c['status']['completed_days'] for c in model['cases']),
            failures=[dict(**c['case'], **{k: c['status'].get(k) for k in
                ('status', 'completed_days', 'first_anomaly_day', 'first_anomaly_time', 'error')})
                for c in model['cases'] if c['status']['status'] in ('numerical_failure', 'infrastructure_error')])
    for group in protocol['groups']:
        for policy in protocol['correction_policies']:
            key=f"{group['id']}_{policy}"
            result['stage_pairs'][key]={str(d):model_comparison(
                data[f"{group['id']}_posttrain_{policy}"],data[f"{group['id']}_midtrain4_{policy}"],d)
                for d in [10,*protocol['horizons_days']]}
        for stage in ('pretrain', 'midtrain4', 'posttrain'):
            key = f"{group['id']}_{stage}"
            result['correction_pairs'][key] = {str(d): policy_pair(data[key + '_raw'], data[key + '_constrained'], d)
                for d in [10, *protocol['horizons_days']]}
            if key + '_density75' in data:
                result['optimized_pairs'][key] = {str(d): dict(
                    raw_vs_optimized=model_comparison(data[key+'_raw'],data[key+'_density75'],d),
                    optimized_vs_standard=model_comparison(data[key+'_density75'],data[key+'_constrained'],d))
                    for d in [10,*protocol['horizons_days']]}
    result['ai_pairs'] = {}
    for name, model in data.items():
        if model['variant'].get('model_family') != 'ai':
            continue
        group = model['variant']['group_id']
        comparator = f'{group}_posttrain_density75'
        if comparator in data:
            result['ai_pairs'][name] = {str(d): model_comparison(model, data[comparator], d)
                for d in protocol['horizons_days']}
    result['curve_windows'] = fixed_curve_windows(protocol, data)
    return result


def export(protocol, destination):
    import csv
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    data = collect(protocol)
    result = summarize(protocol, data)
    atomic_json(destination / 'summary.json', result)
    with (destination / 'daily_mse.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['variant', 'member', 'start_day', 'lead_day', *[f'mse_{f}' for f in FIELDS]])
        for name, group in data.items():
            for case in group['cases']:
                for row in case['rows']:
                    writer.writerow([name, case['case']['member'], case['case']['start_day'], row['lead_day'],
                                     *[row['metrics'][f'mse/{f}'] for f in FIELDS]])
    return result
