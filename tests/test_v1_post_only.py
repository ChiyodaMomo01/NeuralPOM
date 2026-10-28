"""End-to-end post-only supervision and corrected-boundary restart contracts."""
from types import SimpleNamespace
import json

import pytest
import torch

from neuralpom.v1.checkpoint import capture, cpu_tree, restore
from neuralpom.v1.config import resolve_config
from neuralpom.v1.engine import Training, read_records


class TinyModel(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.002))
        self.bias = torch.nn.Parameter(torch.tensor(-0.003))

    def forward(self, value):
        return self.weight * value + self.bias


class ToyRuntime:
    """A differentiable physical map that rejects uncorrected supervision."""

    device = torch.device("cpu")

    def __init__(self, interval):
        self.interval_seconds = interval
        self.core = SimpleNamespace(device=self.device, time=0., x=torch.tensor(0.),
                                    corrected=False, last_corrected_time=-1.)
        for name in ("_accel_cache_ready", "runtime_cuda_graph_fast_path",
                     "runtime_no_grad_fast_path", "runtime_ramp_saturated_fast_path",
                     "runtime_vamax_safe_fast_path"):
            setattr(self.core, name, False)
        self.physical_forward_steps = 0
        self.events = []
        self.restored = []

    def contract(self):
        return {"calibration_id": "post-only-toy", "interval_seconds": self.interval_seconds}

    def require_cache(self, *args):
        pass

    def load_cached(self, member, day):
        self.core.time = day + self.interval_seconds / 86400
        self.core.x = torch.tensor(1. + member * .01)
        self.core.corrected = False
        self.core.last_corrected_time = -1.
        self.events.append(("load", member, day))
        return {}

    def advance(self, differentiable=False):
        assert self.core.corrected, "Physics must continue from the corrected endpoint"
        self.core.x = self.core.x * 1.001 + .0001
        self.core.time += self.interval_seconds / 86400
        self.core.corrected = False
        self.physical_forward_steps += self.interval_seconds // 600
        self.events.append(("advance", self.core.time, differentiable))

    def correct(self, model, packed=None):
        assert not self.core.corrected, "The same endpoint was corrected twice"
        delta = model(self.core.x)
        self.core.x = self.core.x + delta
        self.core.corrected = True
        self.core.last_corrected_time = self.core.time
        self.events.append(("correct", self.core.time, self.core.x.detach().clone(), torch.is_grad_enabled()))
        return {"actual_delta": delta.detach()}

    def score(self, member, day, lead, include_eta=True):
        assert self.core.corrected, "A pre-correction state reached the objective"
        assert self.core.last_corrected_time == self.core.time
        assert abs(self.core.time - day - lead / 86400) < 2e-7
        target = .6 + .005 * lead / 86400
        loss = (self.core.x - target).square()
        self.events.append(("score", lead, torch.is_grad_enabled(), loss.detach().clone()))
        return loss, {"mse": loss}

    def detach(self):
        self.core.x = self.core.x.detach()

    def snapshot(self):
        return capture(self.core)

    def restore(self, state):
        restore(self.core, state)
        assert self.core.corrected, "A post-only restart must contain the corrected boundary"
        self.restored.append(self.snapshot())


class Logger:
    def __init__(self):
        self.rows = []

    def log(self, values, *args):
        self.rows.append(values)

    def artifact(self, *args):
        pass


class Interrupted(Exception):
    pass


class BoundaryTraining(Training):
    stop_after_first = False

    def record_timing(self, *args):
        super().record_timing(*args)
        if self.stage == "posttrain" and self.step == 1:
            self.first_boundary = {
                "physical": self.runtime.snapshot(),
                "weights": cpu_tree(self.model.state_dict()),
                "optimizer": cpu_tree(self.optimizer.state_dict()),
                "cursor": dict(self.cursor),
            }
            if self.stop_after_first:
                raise Interrupted("saved corrected boundary")


def configuration(root, frequency, tbptt_hours):
    return resolve_config({
        "data_root": str(root / "unused-data"), "run_root": str(root),
        "frequency": frequency, "world_size": 1,
        "split": {"train_members": [0], "train_start_day": 1000,
                  "train_end_day": 1011, "validation_end_day": 7005},
        "stages": {"pretrain": {"max_epochs": 1, "checkpoint_every_updates": 1},
                   "midtrain": {"epochs": 1, "checkpoint_every_updates": 1},
                   "posttrain": {"epochs": 1, "rollout_days": 10,
                                 "tbptt_hours": tbptt_hours, "checkpoint_every_updates": 1}},
    })


def prepare_parents(config, interval):
    for stage, updates in (("pretrain", 2), ("midtrain4", 1)):
        runtime, logger = ToyRuntime(interval), Logger()
        training = Training(config, stage, runtime, logger)
        assert training.train() == updates
        scores = [event for event in runtime.events if event[0] == "score"]
        assert scores and all("/pre/" not in key for row in logger.rows for key in row)
        if stage == "midtrain4":
            assert [event[1] for event in scores] == [interval * i for i in range(1, 5)]
            assert len([event for event in runtime.events if event[0] == "correct"]) == 4
            assert all(event[2] for event in scores)
            assert training.parent["stage"] == "pretrain"


def assert_state_exact(actual, expected):
    assert actual["scalars"] == expected["scalars"]
    torch.testing.assert_close(actual["tensors"], expected["tensors"], rtol=0, atol=0)


@pytest.mark.parametrize("frequency,interval,tbptt_hours", [("1h", 3600, 24), ("1day", 86400, 24), ("1day", 86400, 48)])
def test_real_training_post_only_chain_and_corrected_boundary_resume(
        tmp_path, monkeypatch, frequency, interval, tbptt_hours):
    monkeypatch.setattr("neuralpom.v1.engine.CorrectorNet", TinyModel)
    baseline_config = configuration(tmp_path / "baseline", frequency, tbptt_hours)
    resumed_config = configuration(tmp_path / "resumed", frequency, tbptt_hours)
    for config in (baseline_config, resumed_config):
        prepare_parents(config, interval)

    expected_updates = 240 // tbptt_hours
    baseline_runtime = ToyRuntime(interval)
    baseline = BoundaryTraining(baseline_config, "posttrain", baseline_runtime, Logger())
    assert baseline.train() == expected_updates

    prefix_runtime, prefix_logger = ToyRuntime(interval), Logger()
    prefix = BoundaryTraining(resumed_config, "posttrain", prefix_runtime, prefix_logger)
    prefix.stop_after_first = True
    with pytest.raises(Interrupted, match="saved corrected boundary"):
        prefix.train()
    boundary = prefix.first_boundary
    assert boundary["cursor"] == {"epoch": 0, "batch": 0, "window": 1}
    assert boundary["physical"]["scalars"]["corrected"]
    assert boundary["physical"]["scalars"]["time"] == pytest.approx(1000 + tbptt_hours / 24)
    assert_state_exact(boundary["physical"], baseline.first_boundary["physical"])

    continuation_runtime, continuation_logger = ToyRuntime(interval), Logger()
    continuation = Training(resumed_config, "posttrain", continuation_runtime, continuation_logger)
    assert continuation.step == 1 and continuation.cursor == boundary["cursor"]
    assert_state_exact(continuation.physical, boundary["physical"])
    torch.testing.assert_close(continuation.model.state_dict(), boundary["weights"], rtol=0, atol=0)
    torch.testing.assert_close(continuation.optimizer.state_dict(), boundary["optimizer"], rtol=0, atol=0)
    assert continuation.train() == expected_updates
    assert len(continuation_runtime.restored) == 1
    assert_state_exact(continuation_runtime.restored[0], boundary["physical"])
    assert continuation.cursor == {"epoch": 1, "batch": 0, "window": 0}
    assert_state_exact(continuation_runtime.snapshot(), baseline_runtime.snapshot())
    torch.testing.assert_close(continuation.model.state_dict(), baseline.model.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(continuation.optimizer.state_dict(), baseline.optimizer.state_dict(), rtol=0, atol=0)

    events = prefix_runtime.events + continuation_runtime.events
    corrected = [round((event[1] - 1000) * 86400) for event in events if event[0] == "correct"]
    assert corrected == list(range(interval, 10 * 86400 + 1, interval))
    if frequency == '1h':
        assert [event[3] for event in events if event[0] == 'correct'] == ([False] * 23 + [True]) * 10
        assert all(not event[2] for event in events if event[0] == 'advance')
    supervised = [event[1] for event in events if event[0] == "score" and event[2]]
    assert supervised == [day * 86400 for day in range(1, 11)]
    assert all("/pre/" not in key for logger in (prefix_logger, continuation_logger)
               for row in logger.rows for key in row)
    records = read_records(continuation.output / "train.jsonl")
    assert len(records) == expected_updates
    assert all("/pre/" not in key for row in records for key in row)
    first_steps = (tbptt_hours * 3600 - interval) // 600
    remaining_steps = tbptt_hours * 3600 // 600
    assert [row["online_forward_physics_steps"] for row in records] == [first_steps] + [remaining_steps] * (expected_updates - 1)
    frozen = torch.load(continuation.output.parent / "frozen/posttrain.pt", weights_only=False)
    assert frozen["selection"] == "final" and frozen["step"] == expected_updates
    assert frozen["supervision"] == resumed_config["supervision"]


@pytest.mark.parametrize("frequency,interval", [("1h", 3600), ("1day", 86400)])
def test_restart_rejects_checkpoint_missing_post_only_supervision(tmp_path, monkeypatch, frequency, interval):
    monkeypatch.setattr("neuralpom.v1.engine.CorrectorNet", TinyModel)
    config = configuration(tmp_path, frequency, 24)
    prepare_parents(config, interval)
    training = BoundaryTraining(config, "posttrain", ToyRuntime(interval), Logger())
    training.stop_after_first = True
    with pytest.raises(Interrupted):
        training.train()
    pointer = json.loads((training.output / "resume.json").read_text())
    path = training.output / pointer["checkpoint"]
    payload = torch.load(path, weights_only=False)
    assert payload["contract"]["training_settings"].pop("supervision") == "post-correction-only-v1"
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="Restart contract or world size mismatch"):
        Training(config, "posttrain", ToyRuntime(interval), Logger())


def test_restart_rejects_previous_hourly_post_gradient_policy(tmp_path, monkeypatch):
    monkeypatch.setattr("neuralpom.v1.engine.CorrectorNet", TinyModel)
    config = configuration(tmp_path, '1h', 24)
    prepare_parents(config, 3600)
    training = BoundaryTraining(config, "posttrain", ToyRuntime(3600), Logger())
    training.stop_after_first = True
    with pytest.raises(Interrupted):
        training.train()
    pointer = json.loads((training.output / 'resume.json').read_text())
    path = training.output / pointer['checkpoint']
    payload = torch.load(path, weights_only=False)
    payload['contract']['training_settings']['stages']['posttrain']['gradient_mode'] = 'full_window'
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match='Restart contract or world size mismatch'):
        Training(config, 'posttrain', ToyRuntime(3600), Logger())
