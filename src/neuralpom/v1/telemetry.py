"""Aggregation rules for correction telemetry, independent of training graphs."""
from __future__ import annotations

import math


class CorrectionAccumulator:
    """Budgets add; amplitudes retain maxima or explicit per-correction means."""
    def __init__(self):
        self.clear()

    def clear(self):
        self.count = 0
        self.sums, self.minima, self.maxima = {}, {}, {}

    def add(self, diagnostics):
        self.count += 1
        for key, value in diagnostics.items():
            value = float(value)
            if not math.isfinite(value):
                raise FloatingPointError(f'Nonfinite correction diagnostic: {key}')
            if not (key in ('t_budget', 's_budget') or key.endswith(('_mean_delta', '_inventory_delta',
                    '_max_abs', '_rms', '_limit_factor'))):
                raise ValueError(f'Correction diagnostic lacks an aggregation rule: {key}')
            self.sums[key] = self.sums.get(key, 0.) + value
            self.minima[key] = min(self.minima.get(key, value), value)
            self.maxima[key] = max(self.maxima.get(key, value), value)

    def summary(self):
        result = {'correction_count': self.count}
        for key, value in self.sums.items():
            if key.endswith('_max_abs'):
                result[key] = self.maxima[key]
            elif key.endswith('_rms'):
                result[key + '_mean'] = value / self.count
            elif key.endswith('_limit_factor'):
                result[key + '_min'] = self.minima[key]
                result[key + '_mean'] = value / self.count
            else:
                result[key + '_sum'] = value
        return result
