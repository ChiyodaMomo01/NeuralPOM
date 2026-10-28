"""Portable score-function, credit-assignment and black-box rollout checks."""
from types import SimpleNamespace
import unittest
import tempfile
from pathlib import Path
from neuralpom.v1.posttrain.rl import RLPosttrain
import torch
from neuralpom.v1.posttrain.policy import GaussianPolicy, returns_to_go, leave_one_out, batch_contexts
from neuralpom.v1.posttrain.rollout import collect

class DummyRuntime:
    interval_seconds = 86400
    device = torch.device('cpu')
    physical_forward_steps = 0

    def __init__(self, fail_on_day=None):
        self.core = SimpleNamespace(**{name: torch.ones(2) for name in ('u', 'v', 't', 's', 'el', 'rho')})
        self.core.time = 1
        self.fail_on_day = fail_on_day

    def load_cached(self, member, day):
        self.core.time = day + 1

    def advance(self, differentiable):
        assert not differentiable and (not torch.is_grad_enabled())
        self.core.time += 1
        self.physical_forward_steps += 144

    def observation(self):
        return torch.zeros(1, 1, 2, 2)

    def apply_raw(self, raw, density75=False):
        assert not raw.requires_grad and (not torch.is_grad_enabled())

    def score(self, member, day, lead, include_eta):
        assert not torch.is_grad_enabled() and include_eta
        if lead // 86400 == self.fail_on_day:
            return (torch.tensor(float('nan')), {})
        return (torch.tensor(float(lead // 86400)), {})

    def detach(self):
        pass

class Checks(unittest.TestCase):

    def test_score_is_advantage_times_noise_jacobian(self):
        weight = torch.tensor([[0.2, 0.4], [-0.1, 0.3]], dtype=torch.float64, requires_grad=True)
        obs = torch.tensor([0.5, -0.7], dtype=torch.float64)
        mean = weight @ obs
        epsilon = torch.tensor([1.2, -0.3], dtype=torch.float64)
        action = mean.detach() + 0.03 * epsilon
        advantage = torch.tensor(0.7, dtype=torch.float64)
        score = GaussianPolicy(0.03).log_prob(mean, action) * advantage
        (gradient,) = torch.autograd.grad(score, weight)
        torch.testing.assert_close(gradient, advantage / 0.03 * epsilon[:, None] * obs[None, :])
        self.assertFalse(action.requires_grad)

    def test_return_to_go_and_hourly_credit(self):
        reward = torch.tensor([-1.0, -2.0, -3.0])
        torch.testing.assert_close(returns_to_go(reward, 86400), torch.tensor([-6.0, -5.0, -3.0]))
        hour = returns_to_go(reward, 3600)
        self.assertEqual(len(hour), 72)
        torch.testing.assert_close(hour, torch.tensor([-6.0, -5.0, -3.0]).repeat_interleave(24))

    def test_rloo_has_no_own_sample_in_baseline(self):
        values = torch.arange(24.0).reshape(8, 3)
        advantage = leave_one_out(values, 4)
        for i in range(8):
            other = [j for j in range(i // 4 * 4, (i // 4 + 1) * 4) if j != i]
            torch.testing.assert_close(advantage[i], values[i] - values[other].mean(0))
        torch.testing.assert_close(advantage.reshape(2, 4, 3).sum(1), torch.zeros(2, 3))

    def test_sampler_uses_every_context_without_padding(self):
        cases = list(range(700))
        seen = [case for step in range(350) for case in batch_contexts(cases, step, 2, 0)]
        self.assertEqual(sorted(seen), cases)
        self.assertNotEqual(batch_contexts(cases, 0, 2, 0), batch_contexts(cases, 350, 2, 0))

    def test_collection_detaches_and_uses_1296_online_steps(self):
        model = torch.nn.Conv2d(1, 1, 1)
        episode = collect(DummyRuntime(), model, 0, 1000, policy=GaussianPolicy(0.03))
        self.assertEqual(episode.valid_actions, 10)
        self.assertEqual(episode.physical_steps, 1296)
        self.assertFalse(episode.daily_losses.requires_grad)
        self.assertTrue(all((not x.requires_grad for x in episode.observations + episode.actions)))
        self.assertTrue(all((p.grad is None for p in model.parameters())))

    def test_failure_keeps_remaining_rewards(self):
        episode = collect(DummyRuntime(fail_on_day=5), torch.nn.Conv2d(1, 1, 1), 0, 1000, policy=GaussianPolicy(0.03), failure_daily_loss=1000.0)
        self.assertTrue(episode.failed)
        self.assertEqual(episode.failure_lead, 5 * 86400)
        self.assertEqual(episode.valid_actions, 5)
        torch.testing.assert_close(episode.daily_losses, torch.tensor([1, 2, 3, 4] + [1000] * 6, dtype=torch.float64))


class WeightSelectionChecks(unittest.TestCase):
    def test_candidate_own_validation_and_last_unvalidated_are_distinct(self):
        trainer = RLPosttrain.__new__(RLPosttrain)
        trainer.model = torch.nn.Linear(1, 1)
        trainer.runtime = SimpleNamespace(contract=lambda: {'test': 'metadata'})
        trainer.config = {'supervision': 'post-correction-only-v1'}
        trainer.options = {'noise_std': .03}
        trainer.step, trainer.best, trainer.parent = 16, .4, {'step': 222}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'candidate.pt'
            trainer.weights(path, 'validation_candidate', validation_loss=.7)
            value = torch.load(path, weights_only=False)
            self.assertEqual(value['validation_loss'], .7)
            self.assertEqual(value['best_validation_loss'], .4)
            trainer.weights(path, 'last_completed_update')
            value = torch.load(path, weights_only=False)
            self.assertIsNone(value['validation_loss'])
            self.assertEqual(value['best_validation_loss'], .4)
