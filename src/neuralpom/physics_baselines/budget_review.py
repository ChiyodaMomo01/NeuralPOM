"""Read-only complete-window decisions for registered B2/B3 validation cohorts."""
import hashlib
import json
from pathlib import Path
from .budget_selection import summarize
from .spectrum_validation import compare_window


def review(protocol, previous, horizon):
    """Use only the preceding frozen selection; never rank unpromoted survivors."""
    if horizon not in (50, 200):
        raise ValueError('Review requires 50 or 200 days')
    if previous.get('protocol_sha256') != protocol['_sha256'] or previous.get('case_support') != protocol['cases']:
        raise ValueError('Previous selection identity differs')
    if not previous.get('selection_ready'):
        raise ValueError('Previous selection incomplete')
    if horizon == 50:
        if previous.get('last') != 10 or previous.get('first') != 1:
            raise ValueError('Expected complete ten-day screen')
        cohort = previous['selected_for_next_horizon']
        if not 1 <= len(cohort) <= 2:
            raise ValueError('No eligible or invalid promoted cohort')
    else:
        if previous.get('last') != 50 or not previous.get('representative_for_200d'):
            raise ValueError('No complete within-family 50-day representative')
        cohort = [previous['representative_for_200d']]
    if len(set(cohort)) != len(cohort):
        raise ValueError('Duplicate promoted candidate')
    registered = {c['id']: c for c in protocol['candidates']}
    if not set(cohort) <= set(registered):
        raise ValueError('Unregistered candidate')
    filtered = dict(protocol, candidates=[registered[c] for c in cohort])
    # summarize independently verifies the baseline protocol and data contract.
    whole = summarize(filtered, horizon)
    reference = protocol['baseline_reference']
    baseline_protocol = json.loads(Path(reference['protocol_path']).read_text())
    spectra = {c: compare_window(filtered, c, reference['candidate'], horizon,
                                baseline_root=baseline_protocol['output_root']) for c in cohort}
    result = dict(whole, protocol_sha256=protocol['_sha256'], case_support=protocol['cases'],
                  cohort=list(cohort), spectral_reviews=spectra,
                  recommended_for_200d=None, representative_for_200d=None, confirmation_200d=None,
                  selection_policy='within_family_validation_mse_v1',
                  performance_gate_is_diagnostic=True, eligible_for_frozen_test=False)
    result.pop('selected_for_next_horizon', None)
    if horizon == 200:
        last50 = summarize(filtered, 200, 151)
        last_spectra = {c: compare_window(filtered, c, reference['candidate'], 200, 151,
                                        baseline_root=baseline_protocol['output_root']) for c in cohort}
        result['last50'] = dict(last50, spectral_reviews=last_spectra)
        result['selection_ready'] = whole['selection_ready'] and last50['selection_ready']
    if result['selection_ready']:
        eligible = [r for r in whole['ranking'] if r['passes_variable_gate'] and
                    spectra[r['candidate']['id']]['passes_spectral_gate']]
        if horizon == 50:
            result['recommended_for_200d'] = eligible[0]['candidate']['id'] if eligible else None
            result['representative_for_200d'] = whole['ranking'][0]['candidate']['id'] if whole['ranking'] else None
            result['decision'] = 'family_representative_selected' if whole['ranking'] else 'no_complete_numerically_valid_candidate'
        else:
            cid = cohort[0]
            result['representative_for_200d'] = cid
            tail = next((r for r in last50['ranking'] if r['candidate']['id'] == cid), None)
            passed = bool(eligible and tail and tail['passes_variable_gate'] and
                          last_spectra[cid]['passes_spectral_gate'])
            complete = bool(whole['ranking'] and tail)
            result['confirmation_200d'] = 'completed' if complete else 'numerical_failure'
            result['improvement_diagnostics_200d'] = 'passed' if passed else 'failed'
            result['eligible_for_frozen_test'] = complete
            result['decision'] = 'retain_family_for_frozen_test' if complete else 'retain_numerical_failure_evidence'
    else:
        result['decision'] = 'incomplete'
    # Cap observations remain evidence, never automatically a mechanism pass.
    result['mechanism_adequacy'] = ('requires_limiter_and_energy_transfer_review'
                                   if protocol['family'] == 'B3' else 'not_backscatter')
    return result


def main():
    import argparse
    from neuralpom.v1.checkpoint import atomic_json
    parser = argparse.ArgumentParser()
    for key in ('protocol', 'out', 'rules'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--previous')
    parser.add_argument('--role-policy')
    parser.add_argument('--horizon', type=int, choices=(10, 50, 200), required=True)
    args = parser.parse_args()
    path = Path(args.protocol)
    protocol = json.loads(path.read_text())
    protocol['_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    if args.horizon == 10:
        result = summarize(protocol, 10)
        result.update(protocol_sha256=protocol['_sha256'], case_support=protocol['cases'])
    else:
        if not args.previous:
            parser.error('--previous is required for long-window review')
        if not args.role_policy:
            parser.error('--role-policy is required for long-window selection')
        prior = Path(args.previous)
        previous = json.loads(prior.read_text())
        if previous.get('spectral_rule_sha256') != hashlib.sha256(Path(args.rules).read_bytes()).hexdigest():
            raise ValueError('Selection rules changed between phases')
        policy_hash = hashlib.sha256(Path(args.role_policy).read_bytes()).hexdigest()
        if args.horizon == 200 and previous.get('role_policy_sha256') != policy_hash:
            raise ValueError('Baseline role policy changed after representative selection')
        result = review(protocol, previous, args.horizon)
        result['role_policy_sha256'] = policy_hash
        result['previous_selection_sha256'] = hashlib.sha256(prior.read_bytes()).hexdigest()
    result['spectral_rule_sha256'] = hashlib.sha256(Path(args.rules).read_bytes()).hexdigest()
    out = Path(args.out)
    if out.exists():
        raise ValueError('Never overwrite a frozen review; use a new snapshot path')
    atomic_json(out, result)
    print(json.dumps({k: result[k] for k in ('selection_ready', 'first', 'last')}))


if __name__ == '__main__':
    main()
