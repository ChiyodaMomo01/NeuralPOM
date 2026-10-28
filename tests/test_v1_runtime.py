from types import SimpleNamespace

import torch
import pytest

from neuralpom.v1.config import resolve_config
from neuralpom.v1.runtime import Runtime, frozen_rollout


def test_runtime_matches_production_filter_and_solver_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv('NEURALPOM_IMPLICIT_SPATIAL_SHAPIRO', '1')
    monkeypatch.setenv('NEURALPOM_ETA_SOLVER_REL_TOL', '.1')
    monkeypatch.setenv('NEURALPOM_ETA_SOLVER_PRECONDITIONER', 'jacobi')
    monkeypatch.setenv('NEURALPOM_HORCON', '.2')
    monkeypatch.setattr('neuralpom.v1.runtime.FixedCoreDataset', lambda *args: SimpleNamespace(grid=(10, 10, 16)))
    config = resolve_config({'data_root': str(tmp_path), 'run_root': str(tmp_path),
                             'grid': {'im': 10, 'jm': 10}, 'world_size': 1})
    runtime = Runtime(config, 'cpu', assets=False)
    assert runtime.core.spatial_shapiro_enabled is False
    assert all(getattr(runtime.core, key) == 0 for key in ('barotropic_shapiro', 'velocity_shapiro', 'tracer_shapiro'))
    assert runtime.core.eta_solver_rel_tol == 1e-5
    assert runtime.core.eta_solver_preconditioner == 'spectral'
    assert runtime.core.horcon == .05 and runtime.core.alpha == .333 and runtime.core.smoth == .15
    state = {key: value.detach().clone() for key, value in vars(runtime.core).items() if torch.is_tensor(value)}
    state['time'] = torch.tensor(1000., dtype=torch.float64)
    state['el'].fill_(.2)
    state['et'].fill_(.3)
    runtime.data.load_initial = lambda *args, **kwargs: state
    runtime.reset(0, 1000)
    torch.testing.assert_close(runtime.core.dt, runtime.core.h + .3)
    torch.testing.assert_close(runtime.core.d, runtime.core.h + .2)


def test_frozen_hourly_intraday_hook_does_not_reset_prediction_or_weights():
    fields = ('u', 'v', 't', 's', 'el')

    class FrozenRuntime:
        interval_seconds = 3600

        def __init__(self):
            self.core = SimpleNamespace()
            self.resets = 0
            self.labels = []
            self.score_calls = []
            self.data = SimpleNamespace(load_target=self.target)

        def reset(self, member, day):
            self.resets += 1
            self.core.time = float(day)
            for key in fields:
                setattr(self.core, key, torch.zeros(2, 2))

        def advance(self, differentiable):
            assert not differentiable and not torch.is_grad_enabled()
            self.core.time += 1 / 24
            for key in fields:
                setattr(self.core, key, getattr(self.core, key) + 1)

        def correct(self, model):
            assert not model.training and not torch.is_grad_enabled()
            for key in fields:
                setattr(self.core, key, getattr(self.core, key) + model.weight[0, 0])
            return {'t_budget': torch.tensor(.1)}

        def target(self, member, day, lead, map_location='cpu'):
            self.labels.append(lead)
            return {**{key: torch.zeros(2, 2) for key in fields}, 'time': torch.tensor(day + lead / 86400)}

        def score(self, member, day, lead):
            assert lead % 86400 == 0 or 0 < lead % 86400 <= 14400
            self.score_calls.append((lead, float(self.core.t[0, 0])))
            value = self.core.t.square().mean()
            return value, {'mse': value}

    runtime = FrozenRuntime()
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(.5)
    before = model.weight.detach().clone()
    intraday = []
    daily = list(frozen_rollout(runtime, model, 7, 8000, 2, on_intraday=intraday.append))
    assert runtime.resets == 1
    assert len(runtime.score_calls) == 10
    assert all(value == pytest.approx(lead / 3600 * 1.5) for lead, value in runtime.score_calls)
    assert all('pre' not in row and all(key.startswith('post/') for key in row['metrics'])
               for row in daily + intraday)
    assert [row['within_day_hour'] for row in intraday] == [1, 2, 3, 4] * 2
    assert [row['lead_day'] for row in daily] == [1, 2]
    torch.testing.assert_close(daily[-1]['post']['t'], torch.full((2, 2), 72.))
    torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
    assert model.training


def test_actual_objective_does_not_write_reference_into_core(tmp_path, monkeypatch):
    from neuralpom.training.daily_objective import daily_objective
    monkeypatch.setattr('neuralpom.v1.runtime.FixedCoreDataset', lambda *args: SimpleNamespace(grid=(10, 10, 16)))
    config = resolve_config({'data_root': str(tmp_path), 'run_root': str(tmp_path),
                             'grid': {'im': 10, 'jm': 10}, 'world_size': 1})
    core = Runtime(config, 'cpu', assets=False).core
    before = {key: value.clone() for key, value in vars(core).items() if torch.is_tensor(value)}
    reference = {key: getattr(core, key).clone() + .01 for key in ('u', 'ub', 'v', 'vb', 't', 's', 'el')}
    scales = {key: 1. for key in (*reference, 'pressure')}
    loss, _ = daily_objective(core, reference, scales, torch.ones(4))
    assert torch.isfinite(loss)
    for key, expected in before.items():
        torch.testing.assert_close(getattr(core, key), expected, rtol=0, atol=0, msg=key)


def test_training_diagnostics_preserves_actual_state_and_gradient_graph(tmp_path, monkeypatch):
    monkeypatch.setattr('neuralpom.v1.runtime.FixedCoreDataset', lambda *args: SimpleNamespace(grid=(10, 10, 16)))
    config = resolve_config({'data_root': str(tmp_path), 'run_root': str(tmp_path),
                             'grid': {'im': 10, 'jm': 10}, 'world_size': 1})
    runtime = Runtime(config, 'cpu', assets=False)
    core = runtime.core
    core.t = core.t.detach().requires_grad_(True)
    probe = (core.t * 2).sum()
    before = {key: value.detach().clone() for key, value in vars(core).items() if torch.is_tensor(value)}
    reference = {key: getattr(core, key).detach().clone() for key in ('u', 'v', 't', 's', 'el', 'et')}
    reference['time'] = core.time
    runtime.data.load_target = lambda *args, **kwargs: reference
    result = runtime.training_diagnostics(0, 0)
    assert result['metrics']['density/mse'] == 0
    assert len(result['velocity_spectra']['prediction']) == 16
    for key, expected in before.items():
        torch.testing.assert_close(getattr(core, key), expected, rtol=0, atol=0, msg=key)
    torch.testing.assert_close(torch.autograd.grad(probe, core.t)[0], torch.full_like(core.t, 2.))


def test_correction_diagnostics_do_not_change_corrector_gradient(tmp_path, monkeypatch):
    from neuralpom.hybrid.sparse_coupling import apply_joint_correction
    monkeypatch.setattr('neuralpom.v1.runtime.FixedCoreDataset', lambda *args: SimpleNamespace(grid=(10, 10, 16)))
    config = resolve_config({'data_root': str(tmp_path), 'run_root': str(tmp_path),
                             'grid': {'im': 10, 'jm': 10}, 'world_size': 1})
    runtime = Runtime(config, 'cpu', assets=False)
    runtime.correction_scales = dict(u=.001, v=.001, t=.001, s=.001)
    before = runtime.snapshot()
    raw = torch.randn(1, 60, 10, 10, requires_grad=True)
    diagnostics = runtime.correct(lambda packed: raw, torch.zeros(1))
    loss = runtime.core.t.square().mean() + runtime.core.u.square().mean()
    gradient = torch.autograd.grad(loss, raw)[0]
    runtime.restore(before)
    independent_raw = raw.detach().clone().requires_grad_(True)
    apply_joint_correction(runtime.core, independent_raw[0], runtime.correction_scales)
    expected_loss = runtime.core.t.square().mean() + runtime.core.u.square().mean()
    expected_gradient = torch.autograd.grad(expected_loss, independent_raw)[0]
    torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)
    assert diagnostics['t_correction_rms'] >= 0
    assert 's_actual_inventory_delta' in diagnostics


def test_daily_correction_diagnostics_aggregate_by_quantity():
    from neuralpom.v1.telemetry import CorrectionAccumulator
    accumulator = CorrectionAccumulator()
    accumulator.add({'u_correction_max_abs': 1., 'u_correction_rms': .2,
                     't_actual_inventory_delta': 3., 'ts_limit_factor': .8})
    accumulator.add({'u_correction_max_abs': 2., 'u_correction_rms': .4,
                     't_actual_inventory_delta': -1., 'ts_limit_factor': .3})
    result = accumulator.summary()
    assert result['u_correction_max_abs'] == 2.
    assert result['u_correction_rms_mean'] == pytest.approx(.3)
    assert result['t_actual_inventory_delta_sum'] == 2.
    assert result['ts_limit_factor_min'] == .3
    assert result['ts_limit_factor_mean'] == pytest.approx(.55)
    assert result['correction_count'] == 2
