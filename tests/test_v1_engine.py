from types import SimpleNamespace
import random

import numpy as np
import pytest
import torch

from neuralpom.v1.checkpoint import RestartStore, capture, restore
from neuralpom.v1.engine import full_bptt, post_window, validate_cursor, gradient_norm, clip_gradients
from neuralpom.v1.config import resolve_config


class ToyRuntime:
    def __init__(self, interval=86400, multiplier=2.):
        self.interval_seconds = interval
        self.multiplier = multiplier
        self.core = SimpleNamespace(time=1000 + interval / 86400, x=torch.tensor(1.))
        self.corrections = []
        self.advances = []
        self.scores = []

    def load_cached(self, member, day):
        self.core.time = day + self.interval_seconds / 86400
        self.core.x = torch.tensor(1.)
        return {}

    def advance(self, differentiable=False):
        self.advances.append(differentiable)
        self.core.x = self.core.x * self.multiplier
        self.core.time += self.interval_seconds / 86400

    def correct(self, model, packed=None):
        self.corrections.append(round((self.core.time - 1000) * 86400))
        self.core.x = self.core.x + model

    def detach(self):
        self.core.x = self.core.x.detach()

    def score(self, member, day, lead, include_eta=True):
        self.scores.append(lead)
        value = self.core.x.square()
        return value, {'mse': value}


@pytest.mark.parametrize('interval', [3600, 86400])
def test_full_bptt_preserves_physical_jacobian_and_endpoint_order(interval):
    runtime = ToyRuntime(interval)
    theta = torch.tensor(.3, requires_grad=True)
    loss, metrics = full_bptt(runtime, theta, 0, 1000, 4)
    loss.backward()
    # Closed-form states are a + b*theta; b includes every physical Jacobian.
    coefficients = [(1, 1), (2, 3), (4, 7), (8, 15)]
    expected_loss = sum((a + b * .3)**2 for a, b in coefficients) / 4
    expected_gradient = sum(2 * b * (a + b * .3) for a, b in coefficients) / 4
    torch.testing.assert_close(loss, torch.tensor(expected_loss))
    torch.testing.assert_close(theta.grad, torch.tensor(expected_gradient))
    assert all('/pre/' not in key for key in metrics)
    assert runtime.scores == [interval * n for n in range(1, 5)]
    assert runtime.advances == [True, True, True]
    assert runtime.corrections == [interval * n for n in range(1, 5)]


def test_daily_24h_tbptt_scores_only_corrected_day_and_ends_after_correction():
    runtime = ToyRuntime()
    theta = torch.tensor(.3, requires_grad=True)
    loss, metrics = post_window(runtime, theta, 0, 1000, window=0, rollout_days=10)
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(1.3**2))
    torch.testing.assert_close(theta.grad, torch.tensor(2 * 1.3))
    assert runtime.advances == []
    assert runtime.scores == [86400]
    assert runtime.corrections == [86400] and runtime.core.time == 1001
    assert all('pre/' not in key for key in metrics)


def test_daily_48h_tbptt_keeps_physical_gradient_between_corrected_daily_losses():
    runtime = ToyRuntime()
    theta = torch.tensor(.3, requires_grad=True)
    loss, metrics = post_window(runtime, theta, 0, 1000, 0, 10, tbptt_hours=48)
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor((1.3**2 + 2.9**2) / 2))
    torch.testing.assert_close(theta.grad, torch.tensor(1.3 + 3 * 2.9))
    assert runtime.advances == [True]
    assert runtime.scores == [86400, 172800]
    assert runtime.corrections == [86400, 172800]
    assert all('pre/' not in key for key in metrics)


def test_hourly_post_trains_only_last_correction_without_physical_graph():
    class TracedRuntime(ToyRuntime):
        def __init__(self):
            super().__init__(interval=3600, multiplier=1.01)
            self.forward_grad_modes = []
            self.correction_grad_modes = []
            self.correction_outputs = []

        def advance(self, differentiable=False):
            self.forward_grad_modes.append(torch.is_grad_enabled())
            super().advance(differentiable)

        def correct(self, model, packed=None):
            self.correction_grad_modes.append(torch.is_grad_enabled())
            super().correct(model[len(self.corrections)], packed)
            self.correction_outputs.append(self.core.x)

    runtime = TracedRuntime()
    initial = runtime.core.x.requires_grad_()
    theta = torch.full((24,), .01, requires_grad=True)
    expected = 1.
    for hour in range(24):
        expected = (expected * 1.01 if hour else expected) + .01
    loss, metrics = post_window(runtime, theta, 0, 1000, 0, 10)
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(expected**2))
    torch.testing.assert_close(theta.grad[:-1], torch.zeros(23), rtol=0, atol=0)
    torch.testing.assert_close(theta.grad[-1], torch.tensor(2 * expected))
    assert initial.grad is None
    assert runtime.advances == [False] * 23
    assert runtime.forward_grad_modes == [False] * 23
    assert runtime.correction_grad_modes == [False] * 23 + [True]
    assert all(value.grad_fn is None for value in runtime.correction_outputs[:-1])
    assert runtime.correction_outputs[-1].grad_fn is not None
    assert len([key for key in metrics if key.startswith('intraday/')]) == 4


@pytest.mark.parametrize('interval,expected', [(86400, 10), (3600, 240)])
def test_post_terminal_boundary_once_and_no_future_label(interval, expected):
    runtime = ToyRuntime(interval, multiplier=1.)
    theta = torch.tensor(.01, requires_grad=True)
    for window in range(10):
        loss, _ = post_window(runtime, theta, 0, 1000, window, 10)
        loss.backward()
        theta.grad = None
        runtime.core.x = runtime.core.x.detach()
    assert len(runtime.corrections) == len(set(runtime.corrections)) == expected
    assert runtime.corrections[-1] == 10 * 86400
    assert max(runtime.scores) == 10 * 86400
    assert all(lead % 86400 == 0 or (interval == 3600 and lead % 86400 <= 14400) for lead in runtime.scores)
    assert abs(runtime.core.time - 1010) < 5e-8


def test_complete_state_roundtrip_preserves_warm_start_and_clock():
    core = SimpleNamespace(device=torch.device('cpu'), x=torch.tensor([1.]),
        _eta_prev_solution=torch.tensor([2.]), time=1010., iint=144,
        _settings_overrides={'test': 1}, _cg_work={'scratch': torch.ones(1)})
    state = capture(core)
    core.x += 100
    core._eta_prev_solution += 100
    core.extra_tensor = torch.zeros(1)
    restore(core, state)
    assert not hasattr(core, 'extra_tensor')
    assert core.x.item() == 1 and core._eta_prev_solution.item() == 2
    assert core.time == 1010 and core.iint == 144 and core._cg_work == {}


def test_restart_replays_optimizer_sampler_cursor_and_rng(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    store = RestartStore(tmp_path, {'stage': 'posttrain'})
    optimizer.zero_grad()
    model(torch.ones(2)).sum().backward()
    optimizer.step()
    store.save(model, optimizer, {'epoch': 2, 'batch': 5, 'window': 4}, 41,
               physical={'complete': True, 'clock': 1004}, extra={'best': 3.})
    expected_random = (random.random(), np.random.rand(), torch.rand(3))
    optimizer.zero_grad()
    model(torch.tensor([2., 3.])).sum().backward()
    optimizer.step()
    expected_weights = {k: v.clone() for k, v in model.state_dict().items()}
    restored = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.1)
    value, physical = store.load(restored, restored_optimizer)
    assert value['cursor'] == {'epoch': 2, 'batch': 5, 'window': 4}
    assert physical['clock'] == 1004
    assert random.random() == expected_random[0]
    assert np.random.rand() == expected_random[1]
    torch.testing.assert_close(torch.rand(3), expected_random[2])
    restored_optimizer.zero_grad()
    restored(torch.tensor([2., 3.])).sum().backward()
    restored_optimizer.step()
    for key, tensor in restored.state_dict().items():
        torch.testing.assert_close(tensor, expected_weights[key], rtol=0, atol=0)


def test_restart_retains_only_two_complete_generations(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    store = RestartStore(tmp_path, {'stage': 'pretrain'})
    for step in range(4):
        if step == 3:
            (tmp_path / 'restart' / 'step_00000003_interrupted').mkdir()
        store.save(model, optimizer, {'batch': step}, step)
    assert len(list((tmp_path / 'restart').glob('step_*'))) == 2
    value, _ = store.load(model, optimizer)
    assert value['step'] == 3


def test_optimizer_step_must_match_epoch_sampler_and_window():
    plan = {'epochs': 20, 'trajectory_batches_per_epoch': 88}
    cursor = {'epoch': 2, 'batch': 4, 'window': 3}
    validate_cursor(cursor, (2 * 88 + 4) * 10 + 3, plan, 10)
    with pytest.raises(ValueError, match='disagree'):
        validate_cursor(cursor, 0, plan, 10)


def test_tbptt_rejects_restart_at_wrong_physical_boundary():
    runtime = ToyRuntime()
    with pytest.raises(ValueError, match='state/cursor'):
        post_window(runtime, torch.tensor(.1, requires_grad=True), 0, 1000, 3, 10)


def test_gradient_clipping_does_not_overflow_for_finite_large_gradients():
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([3e20, 4e20])
    norm = gradient_norm([parameter])
    assert torch.isfinite(norm)
    torch.testing.assert_close(norm, torch.tensor(5e20, dtype=torch.float64))
    clip_gradients([parameter], norm, 10.)
    torch.testing.assert_close(parameter.grad, torch.tensor([6., 8.]))


def test_pretrain_early_stop_best_weights_and_completed_epoch_resume(tmp_path, monkeypatch):
    from neuralpom.v1.engine import Training

    class TinyModel(torch.nn.Module):
        def __init__(self, *args):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

    class TrainRuntime(ToyRuntime):
        device = torch.device('cpu')

        def __init__(self):
            super().__init__()
            self.validation_scores = 0

        def contract(self):
            return {'calibration_id': 'test', 'interval_seconds': 86400}

        def require_cache(self, *args):
            pass

        def correct(self, model, packed=None):
            self.core.x = self.core.x + model.weight

        def score(self, member, day, lead, include_eta=True):
            value = self.core.x.square()
            if member == 7:
                value = value + 100 * self.validation_scores
                self.validation_scores += 1
            return value, {'mse': value}

        def detach(self):
            self.core.x = self.core.x.detach()

    class Logger:
        def log(self, *args):
            pass

        def artifact(self, *args):
            pass

    monkeypatch.setattr('neuralpom.v1.engine.CorrectorNet', TinyModel)
    config = resolve_config({'data_root': str(tmp_path / 'data'), 'run_root': str(tmp_path),
        'world_size': 1, 'split': {'train_members': [0], 'train_start_day': 1000,
                                  'train_end_day': 1008, 'validation_end_day': 7002},
        'stages': {'pretrain': {'max_epochs': 4, 'patience': 1}}})
    training = Training(config, 'pretrain', TrainRuntime(), Logger())
    assert training.train() == 4
    assert training.stopped and training.cursor == {'epoch': 2, 'batch': 0, 'window': 0}
    best = torch.load(tmp_path / 'frozen/pretrain.pt', weights_only=False)
    assert best['step'] == 2
    assert best['selection'] == 'validation_best'
    resumed = Training(config, 'pretrain', TrainRuntime(), Logger())
    assert resumed.train() == 4
    assert resumed.stopped
    torch.testing.assert_close(torch.load(tmp_path / 'frozen/pretrain.pt', weights_only=False)['model']['weight'],
                               best['model']['weight'], rtol=0, atol=0)


@pytest.mark.parametrize('frequency,interval', [('1h', 3600), ('1day', 86400)])
def test_mid4_loads_pretrain_directly_and_completes_without_short_courses(tmp_path, monkeypatch, frequency, interval):
    from neuralpom.v1.engine import Training

    class TinyModel(torch.nn.Module):
        def __init__(self, *args):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

    class Runtime(ToyRuntime):
        device = torch.device('cpu')

        def __init__(self):
            super().__init__(interval, multiplier=1.01)

        def contract(self):
            return {'calibration_id': 'direct4-test', 'interval_seconds': interval}

        def require_cache(self, *args):
            pass

        def correct(self, model, packed=None):
            return super().correct(model.weight, packed)

        def detach(self):
            self.core.x = self.core.x.detach()

    class Logger:
        def log(self, *args):
            pass

        def artifact(self, *args):
            pass

    monkeypatch.setattr('neuralpom.v1.engine.CorrectorNet', TinyModel)
    config = resolve_config({'data_root': str(tmp_path / 'data'), 'run_root': str(tmp_path),
        'frequency': frequency, 'world_size': 1,
        'split': {'train_members': [0], 'train_start_day': 1000, 'train_end_day': 1006,
                  'validation_end_day': 7005},
        'stages': {'pretrain': {'max_epochs': 1}, 'midtrain': {'epochs': 1}}})
    assert Training(config, 'pretrain', Runtime(), Logger()).train() == 1
    best = torch.load(tmp_path / 'frozen/pretrain.pt', weights_only=False)
    runtime = Runtime()
    training = Training(config, 'midtrain4', runtime, Logger())
    assert training.parent['stage'] == 'pretrain'
    torch.testing.assert_close(training.model.weight, best['model']['weight'], rtol=0, atol=0)
    assert training.train() == 1
    assert runtime.advances == [True, True, True]
    assert not (tmp_path / 'midtrain2').exists() and not (tmp_path / 'midtrain3').exists()
    saved = torch.load(tmp_path / 'frozen/midtrain4.pt', weights_only=False)
    assert saved['parent']['stage'] == 'pretrain' and saved['selection'] == 'final'
    assert torch.isfinite(saved['model']['weight'])
    legacy = dict(best, stage='midtrain3')
    legacy_path = tmp_path / 'legacy_parent.pt'
    torch.save(legacy, legacy_path)
    with pytest.raises(ValueError, match='Parent weights'):
        Training({**config, 'run_root': str(tmp_path / 'fresh')}, 'midtrain4', Runtime(), Logger(),
                 parent=legacy_path, resume=False)


@pytest.mark.parametrize('interval, expected_saved_steps', [(1, [1, 2, 3, 4, 5]), (2, [2, 4, 5]), (100, [5])])
def test_checkpoint_interval_epoch_tail_and_exact_replay(tmp_path, monkeypatch, interval, expected_saved_steps):
    import json
    from neuralpom.v1.engine import Training

    class TinyModel(torch.nn.Module):
        def __init__(self, *args):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

    class Runtime(ToyRuntime):
        device = torch.device('cpu')

        def contract(self):
            return {'calibration_id': 'interval-test'}

        def require_cache(self, *args):
            pass

        def load_cached(self, member, day):
            result = super().load_cached(member, day)
            self.core.x += (day - 1000) / 100
            return result

        def correct(self, model, packed=None):
            self.core.x = self.core.x + model.weight

        def detach(self):
            self.core.x = self.core.x.detach()

    class Logger:
        def __init__(self):
            self.rows = []

        def log(self, values, *args):
            self.rows.append(values)

        def artifact(self, *args):
            pass

    class TracedTraining(Training):
        def save(self, physical=False):
            if not hasattr(self, 'saved_steps'):
                self.saved_steps = []
            self.saved_steps.append((self.step, dict(self.cursor)))
            super().save(physical)

    class InterruptedTraining(TracedTraining):
        def record_timing(self, *args):
            super().record_timing(*args)
            if self.step == 3:
                raise RuntimeError('simulated interrupted update')

    monkeypatch.setattr('neuralpom.v1.engine.CorrectorNet', TinyModel)
    def config(root):
        return resolve_config({'data_root': str(tmp_path / 'data'), 'run_root': str(root),
            'world_size': 1, 'split': {'train_members': [0], 'train_start_day': 1000,
                'train_end_day': 1026, 'validation_end_day': 7002},
            'stages': {'pretrain': {'max_epochs': 1, 'checkpoint_every_updates': interval}}})

    baseline = TracedTraining(config(tmp_path / 'baseline'), 'pretrain', Runtime(), Logger())
    assert baseline.train() == 5
    assert [step for step, _ in baseline.saved_steps] == expected_saved_steps
    assert baseline.saved_steps[-1][1] == {'epoch': 1, 'batch': 0, 'window': 0}
    if interval == 100:
        return  # No checkpoint exists before the simulated interruption.
    interrupted_config = config(tmp_path / 'interrupted')
    interrupted = InterruptedTraining(interrupted_config, 'pretrain', Runtime(), Logger())
    with pytest.raises(RuntimeError, match='simulated interrupted update'):
        interrupted.train()
    logger = Logger()
    resumed = TracedTraining(interrupted_config, 'pretrain', Runtime(), logger)
    expected_step = 3 if interval == 1 else 2
    assert resumed.step == expected_step
    rows = [json.loads(line) for line in (resumed.output / 'train.jsonl').read_text().splitlines()]
    assert max(row['step'] for row in rows) == expected_step
    assert logger.rows[-1]['restart/replayed_updates'] == 3 - expected_step
    assert logger.rows[-1]['restart/attempt'] == 2
    assert resumed.train() == 5
    torch.testing.assert_close(resumed.model.weight, baseline.model.weight, rtol=0, atol=0)
    for key in ('exp_avg', 'exp_avg_sq', 'step'):
        torch.testing.assert_close(resumed.optimizer.state[resumed.model.weight][key],
                                   baseline.optimizer.state[baseline.model.weight][key], rtol=0, atol=0)
    rows = [json.loads(line) for line in (resumed.output / 'train.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rows] == [1, 2, 3, 4, 5]
    timing = [json.loads(line) for line in (resumed.output / 'timing.jsonl').read_text().splitlines()]
    assert len(timing) == 5 and all(row['iteration_seconds'] >= row['checkpoint_seconds'] for row in timing)
    resources = [json.loads(line) for line in (resumed.output / 'resource_usage.jsonl').read_text().splitlines()]
    assert len(resources) == (5 if interval == 1 else 6)
    assert sum(row['step'] == 3 for row in resources) == (1 if interval == 1 else 2)


def test_correction_statistics_keeps_budgets_and_limiter_extrema():
    from neuralpom.v1.engine import correction_statistics
    records = [{'t_budget': torch.tensor(.1, requires_grad=True), 'ts_limit_factor': torch.tensor(.8)},
               {'t_budget': torch.tensor(-.04, requires_grad=True), 'ts_limit_factor': torch.tensor(.3)}]
    metrics = correction_statistics(records)
    assert metrics['correction/count'] == 2
    torch.testing.assert_close(metrics['correction/t_budget/sum'], torch.tensor(.06))
    torch.testing.assert_close(metrics['correction/ts_limit_factor/min'], torch.tensor(.3))
    assert not metrics['correction/t_budget/sum'].requires_grad


def test_resource_journal_preserves_complete_replayed_entries_on_truncated_append(tmp_path):
    import json
    from neuralpom.v1.engine import read_records, repair_incomplete_tail
    path = tmp_path / 'resource_usage.jsonl'
    completed = [{'attempt': 1, 'step': 3, 'allocated_gpu_hours': .01},
                 {'attempt': 2, 'step': 3, 'allocated_gpu_hours': .02}]
    prefix = ''.join(json.dumps(row) + '\n' for row in completed)
    path.write_text(prefix + '{"attempt": 3')
    assert read_records(path) == completed
    repair_incomplete_tail(path)
    assert path.read_text() == prefix
    with path.open('a') as stream:
        stream.write(json.dumps({'attempt': 3, 'step': 4}) + '\n')
    assert len(read_records(path)) == 3
    path.write_text(prefix + '{broken}\n' + json.dumps({'attempt': 3}) + '\n')
    with pytest.raises(json.JSONDecodeError):
        read_records(path)


@pytest.mark.parametrize('resource_first', [True, False])
def test_resource_ledger_recovers_a_crash_between_the_two_appends(tmp_path, resource_first):
    import json
    from neuralpom.v1.engine import read_records, reconcile_resource_ledger
    first = {'attempt': 1, 'step': 1, 'allocated_gpu_hours': .01}
    second = {'attempt': 1, 'step': 2, 'allocated_gpu_hours': .02}
    complete = json.dumps(first) + '\n' + json.dumps(second) + '\n'
    interrupted = json.dumps(first) + '\n' + '{"attempt": 1, "step":'
    (tmp_path / 'resource_usage.jsonl').write_text(complete if resource_first else interrupted)
    (tmp_path / 'timing.jsonl').write_text(interrupted if resource_first else complete)
    reconcile_resource_ledger(tmp_path)
    reconcile_resource_ledger(tmp_path)
    assert read_records(tmp_path / 'resource_usage.jsonl') == [first, second]
    assert read_records(tmp_path / 'timing.jsonl') == [first, second]


def test_old_timing_is_migrated_once_even_after_new_resource_events(tmp_path):
    import json
    from neuralpom.v1.engine import read_records, reconcile_resource_ledger
    first = {'step': 1, 'allocated_gpu_hours': .01}
    second = {'attempt': 2, 'step': 2, 'allocated_gpu_hours': .02}
    (tmp_path / 'timing.jsonl').write_text(json.dumps(first) + '\n' + json.dumps(second) + '\n')
    (tmp_path / 'resource_usage.jsonl').write_text(json.dumps(second) + '\n')
    reconcile_resource_ledger(tmp_path)
    reconcile_resource_ledger(tmp_path)
    rows = read_records(tmp_path / 'resource_usage.jsonl')
    assert len(rows) == 2
    assert sum(row['allocated_gpu_hours'] for row in rows) == pytest.approx(.03)
    legacy = next(row for row in rows if row['step'] == 1)
    assert legacy['attempt'] == 0 and legacy['origin'] == 'migrated_legacy_timing'


@pytest.mark.parametrize('has_complete_resource_event', [False, True])
def test_legacy_replay_cost_is_preserved_without_double_count(tmp_path, has_complete_resource_event):
    import json
    from neuralpom.v1.engine import read_records, reconcile_resource_ledger
    (tmp_path / 'timing.jsonl').write_text(json.dumps({'step': 1, 'allocated_gpu_hours': .01}) + '\n')
    attempt = {'attempt': 2, 'resume_from_step': 1,
               'discarded_metrics_gpu_hours': .02, 'discarded_metrics_physics_steps': 144}
    (tmp_path / 'attempts.jsonl').write_text(json.dumps(attempt) + '\n')
    if has_complete_resource_event:
        event = {'attempt': 1, 'step': 2, 'allocated_gpu_hours': .02,
                 'online_forward_physics_steps_all_ranks': 144}
        (tmp_path / 'resource_usage.jsonl').write_text(json.dumps(event) + '\n')
    reconcile_resource_ledger(tmp_path, progress_step=1)
    reconcile_resource_ledger(tmp_path, progress_step=1)
    ledger = read_records(tmp_path / 'resource_usage.jsonl')
    assert sum(row.get('allocated_gpu_hours', 0) for row in ledger) == pytest.approx(.03)
    assert sum(row.get('online_forward_physics_steps_all_ranks', 0) for row in ledger) == 144
    assert [row['step'] for row in read_records(tmp_path / 'timing.jsonl')] == [1]
