"""Common-window presentation after frozen-test provenance has been verified.

This layer does not load calibration directories or certify runtime contracts.
The caller must verify source, parameters, reference data and time advancement
before passing formal test records. Missing formal runs remain explicit slots.
"""
from .analysis import summarize, model_comparison, fixed_curve_windows


def registered_curve_windows(protocol, data):
    """Provisional curves may omit only explicitly unavailable formal slots.

    A registered failure or a short trajectory remains in the cohort gate.
    The strict all-method curves in summarize remain unchanged.
    """
    unavailable = {}
    included = {}
    for name, model in data.items():
        reason = model.get('availability')
        if reason in ('not_registered', 'reuse_pending'):
            if model['variant']['stage'] != 'physics' or model['variant'].get('baseline_family') not in ('B1','B2','B3'):
                raise ValueError('Only unavailable formal baseline slots may be omitted')
            if any(c['rows'] or c['status']['status']!='pending' or c['status']['completed_days'] for c in model['cases']):
                raise ValueError('Unavailable slot contains results or failures')
            unavailable[name] = reason
        else:
            included[name] = model
    windows = fixed_curve_windows(protocol, included)
    for group in protocol['groups']:
        gid = group['id']
        requested = [name for name,m in data.items() if m['variant']['group_id']==gid]
        active = [name for name in requested if name in included]
        omitted = {name:unavailable[name] for name in requested if name in unavailable}
        for window in windows[gid].values():
            window.update(requested_variants=requested,included_variants=active,
                          unavailable_variants=omitted,provisional=bool(omitted))
            for name in omitted:window['daily'][name]=[]
    return windows


def unified_summary(protocol, data):
    """Retain all nine NN arms and B0/B1/B2/B3 for each core/frequency."""
    expected_cases = protocol['cases']
    expected = set()
    families = {}
    for group in protocol['groups']:
        gid = group['id']
        expected.update(f'{gid}_{stage}_{policy}'
                        for stage in ('pretrain', 'midtrain4', 'posttrain')
                        for policy in ('raw', 'constrained', 'density75'))
        names = {'B0': f'{gid}_physics_physics'}
        names.update({family: f'{gid}_physics_{family}' for family in ('B1', 'B2', 'B3')})
        families[gid] = names
        expected.update(names.values())
        for family, name in names.items():
            if name not in data:
                raise ValueError('Missing formal baseline slot: ' + name)
            variant = data[name]['variant']
            if variant.get('baseline_family', 'B0') != family or variant['stage'] != 'physics':
                raise ValueError('Baseline family identity differs')
            if family != 'B0' and data[name].get('evaluation_split') != 'test':
                raise ValueError('Calibration results cannot enter frozen-test comparisons')
    from .protocol import variants
    expected.update(v['id'] for v in variants(protocol) if v.get('model_family') == 'ai')
    if set(data) != expected:
        raise ValueError('Unified comparison must contain every NN arm and baseline, exactly once')
    for name, model in data.items():
        if [c['case'] for c in model['cases']] != expected_cases:
            raise ValueError('Different initial conditions or ordering: ' + name)
        for case in model['cases']:
            n = case['status']['completed_days']
            if [r['lead_day'] for r in case['rows']] != list(range(1, n+1)):
                raise ValueError('Noncontinuous committed window: ' + name)
    result = summarize(protocol, data)
    # The original summarizer reads only the original campaign schedulers.
    # Formal physics campaigns have separate schedulers. Reuse across output
    # frequencies must neither hide their failures nor double-count them.
    formal_errors = {}
    for name, model in data.items():
        for worker in model.get('worker_errors', []):
            if worker['returncode'] in (None, 0):
                raise ValueError('Nonfailure in formal worker error records')
            source = model['result_root']
            digest = model['protocol_sha256']
            key = (source, digest, worker['pid'], worker['days'])
            if key not in formal_errors:
                formal_errors[key] = dict(worker=worker, source_root=source,
                                          protocol_sha256=digest, affected_variants=[])
            elif formal_errors[key]['worker'] != worker:
                raise ValueError('Conflicting reused worker failure records')
            formal_errors[key]['affected_variants'].append(name)
    result['formal_worker_errors'] = list(formal_errors.values())
    result['worker_errors'].extend(dict(item['worker'], source_root=item['source_root'],
                                       protocol_sha256=item['protocol_sha256'],
                                       variant=' / '.join(item['affected_variants']),
                                       affected_variants=item['affected_variants'])
                                   for item in formal_errors.values())
    result['physics_pairs'] = {}
    windows = [(str(d), 1, d) for d in (50, 200, 300)]
    windows += [(f'{d}_last50', d-49, d) for d in (200, 300)]
    for gid, names in families.items():
        members = [m for m in data.values() if m['variant']['group_id'] == gid]
        result['physics_pairs'][gid] = {
            family: {model['variant']['id']: {
                key: model_comparison(model, data[baseline], last, first)
                for key, first, last in windows}
                for model in members}
            for family, baseline in names.items()}
    result['comparison_scope'] = 'nine NN arms plus B0/B1/B2/B3; same origins and complete windows'
    result['registered_curve_windows'] = registered_curve_windows(protocol, data)
    result['provenance_note'] = 'Runtime/source/data/selection verification is required upstream; not certified by this presentation layer.'
    return result
