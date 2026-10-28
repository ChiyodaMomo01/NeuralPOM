"""Read budget diagnostics for provenance-verified formal physics records only."""
import json
import math
from pathlib import Path
from .budget_windows import summarize_budget_window


def collect_formal_budgets(model):
    """Preserve every origin/status; never reinterpret calibration as testing.

    A running trajectory may publish its last budget just after physical status.
    That single pending commit is disclosed and cannot enter a complete window.
    Historical or terminal missing budgets are errors, not zero-valued accounts.
    This checks accounts; source/reference equivalence remains the caller's gate.
    """
    if model.get('evaluation_split') != 'test':
        raise ValueError('Formal test records required')
    variant = model['variant']
    family = variant.get('baseline_family')
    if family not in ('B1', 'B2', 'B3') or variant['stage'] != 'physics':
        raise ValueError('Unknown formal physics family')
    output = dict(variant=variant['id'], family=family, cases=[],
                  applicable=family in ('B2', 'B3'), protocol_sha256=model['protocol_sha256'])
    for item in model['cases']:
        case, status = item['case'], item['status']
        n = status['completed_days']
        entry = dict(case=case, status=status, committed_physical_days=n,
                     budget_days=0, pending_budget_day=None, windows={})
        output['cases'].append(entry)
        if family == 'B1':
            entry['note'] = 'This closure has no backscatter reservoir ledger; not a zero-energy measurement.'
            continue
        identity = dict(variant=variant['id'], **case, target_days=300,
                        weight_sha256=None, protocol_sha256=model['protocol_sha256'])
        directory = (Path(model['result_root'])/'runs'/variant['id']/
                     f"m{case['member']}_day{case['start_day']:04d}"/'budgets')
        records = []
        for day in range(1, n+1):
            path = directory/f'day{day:03d}.json'
            if not path.exists():
                if day == n and status['status'] == 'running':
                    entry['pending_budget_day'] = day
                    continue
                raise ValueError('Missing historical or terminal budget: '+str(path))
            row = json.loads(path.read_text())
            if row['lead_day'] != day:
                raise ValueError('Budget filename and day differ')
            for key, total in [('reservoir', 'augmented_reservoir'),
                               ('reservoir_credit', 'augmented_credit'),
                               ('backscatter', 'augmented_returned')]:
                expected = sum(row['accounts'][slot][key] for slot in ('current', 'history'))
                if not math.isclose(expected, row[total], rel_tol=1e-10, abs_tol=1e-8):
                    raise ValueError('Augmented budget and slot accounts differ')
            residual = row['augmented_reservoir']-row['augmented_credit']+row['augmented_returned']
            if not abs(residual-row['augmented_residual']) <= max(row['augmented_credit']*1e-6, 1e-8):
                raise ValueError('Stored augmented residual differs')
            records.append(row)
        entry['budget_days'] = len(records)
        # Validate every committed day, including days outside the fixed windows.
        if records:
            summarize_budget_window(records, 1, len(records), identity)
        windows = [(str(d), 1, d) for d in (10, 50, 200, 300)]
        windows += [(f'{d}_last50', d-49, d) for d in (200, 300)]
        entry['windows'] = {key:summarize_budget_window(records, first, last, identity)
                            for key, first, last in windows}
    output['scope'] = 'Per-origin closure accounts; no survivor averaging or inference of physical statistical correctness.'
    return output
