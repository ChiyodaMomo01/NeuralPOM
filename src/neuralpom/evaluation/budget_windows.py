"""Closure-window diagnostics from cumulative daily accounts, without simulation.

Counters are differenced at window boundaries. Averaging their cumulative
fractions would incorrectly include the days preceding a last-50-day window.
Current and history layers are reported separately; their sum is bookkeeping,
not the physical kinetic energy of an additional ocean state.
"""
import math

COUNTERS = ('dissipation', 'reservoir_credit', 'backscatter', 'heat',
            'cap_cells', 'eligible_cells', 'limited_pairs', 'active_pairs')


def summarize_budget_window(records, first, last, identity):
    """One trajectory, committed daily budget dictionaries through ``last``.

Return None for incomplete windows. The caller must preserve failure/status
coverage and use identical cases for inter-model comparisons.
"""
    if not 1 <= first <= last:
        raise ValueError('Invalid window')
    by_day = {}
    for row in records:
        day = row['lead_day']
        if day in by_day:
            raise ValueError('Duplicate budget commit')
        if row['identity'] != identity or not abs(row['time']-identity['start_day']-day) <= 2e-6:
            raise ValueError('Budget identity or clock differs')
        by_day[day] = row
    needed = list(range(max(1, first-1), last+1))
    if not all(d in by_day for d in needed):
        return None
    # Endpoint differences alone can hide a transient accounting error which
    # cancels before the end of the window. Verify every committed total.
    for day in needed:
        accounts = by_day[day]['accounts']
        totals = {key:sum(accounts[layer][key] for layer in ('current', 'history'))
                  for key in ('reservoir', 'reservoir_credit', 'backscatter')}
        if any(not math.isfinite(value) or value < 0 for value in totals.values()):
            raise ValueError('Invalid daily augmented account')
        residual = totals['reservoir']-totals['reservoir_credit']+totals['backscatter']
        if abs(residual) > max(totals['reservoir_credit']*1e-6, 1e-8):
            raise ValueError('Daily augmented reservoir account does not close')
    layers = {}
    total_tolerance = 0.
    for layer in ('current', 'history'):
        previous = None
        for day in needed:
            account = by_day[day]['accounts'][layer]
            values = [account[k] for k in (*COUNTERS, 'reservoir', 'minimum_reservoir')]
            if any(not math.isfinite(v) or v < 0 for v in values):
                raise ValueError('Invalid nonnegative account')
            if previous is not None and any(account[k] < previous[k] for k in COUNTERS):
                raise ValueError('Cumulative counter decreased inside window')
            previous = account
        end = by_day[last]['accounts'][layer]
        start = by_day[first-1]['accounts'][layer] if first > 1 else {k:0. for k in (*COUNTERS, 'reservoir')}
        increments = {k:end[k]-start[k] for k in COUNTERS}
        credit, returned = increments['reservoir_credit'], increments['backscatter']
        delta = end['reservoir']-start['reservoir']
        residual = delta-credit+returned
        # Difference of two valid cumulative balances can accumulate two errors.
        tolerance = max((end['reservoir_credit']+start['reservoir_credit'])*1e-6, 2e-8)
        # The two reservoirs rotate with leapfrog state, whereas cumulative
        # counters are labelled by the slot where closure was applied. Only
        # their sum closes without an explicit inter-slot transfer counter.
        total_tolerance += tolerance
        if increments['cap_cells'] > increments['eligible_cells'] or increments['limited_pairs'] > increments['active_pairs']:
            raise ValueError('Limiter counts exceed support')
        layers[layer] = dict(increments= increments,
                             reservoir_start=start['reservoir'], reservoir_end=end['reservoir'],
                             reservoir_delta=delta, residual_including_slot_exchange=residual,
                             cap_fraction=(increments['cap_cells']/increments['eligible_cells'] if increments['eligible_cells'] else None),
                             limited_pair_fraction=(increments['limited_pairs']/increments['active_pairs'] if increments['active_pairs'] else None),
                             returned_fraction_of_credit=returned/credit if credit else None,
                             minimum_reservoir_at_daily_commits=min(by_day[d]['accounts'][layer]['minimum_reservoir'] for d in range(first,last+1)))
    residual = sum(a['residual_including_slot_exchange'] for a in layers.values())
    if abs(residual) > total_tolerance:
        raise ValueError('Augmented window reservoir account does not close')
    return dict(first=first,last=last,identity=identity,layers=layers,
                augmented_reservoir_balance_residual=residual,
                energy_units='m5/s2 per reference density',
                scope='Closure accounts only; reservoirs rotate between slots, so individual slot residuals include exchange. History is not extra physical kinetic energy.',
                ratio_note='Window ratios are ratios of counter increments; energy returned can draw on the starting reservoir.')
