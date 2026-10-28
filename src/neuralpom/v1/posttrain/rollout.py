"""Black-box physical rollouts with a fixed policy for all ten days."""
from dataclasses import dataclass
import torch
from .statistics import correction_statistics


@dataclass
class Trajectory:
    observations: list
    actions: list
    daily_losses: torch.Tensor
    failed: bool
    failure_reason: str | None
    failure_lead: int | None
    physical_steps: int
    diagnostics: dict

    @property
    def valid_actions(self):
        return len(self.actions)


def assert_finite(runtime):
    for field in ('u', 'v', 't', 's', 'el', 'rho'):
        if not torch.isfinite(getattr(runtime.core, field)).all():
            raise FloatingPointError(f'Nonfinite physical state: {field}')


@torch.no_grad()
def collect(runtime, model, member, day, *, days=10, policy=None,
            failure_daily_loss=1000., density75=False, store_actions=True):
    """Cached first endpoint; physical propagation and rewards have no graph.

    A nonfinite state/action/loss is an absorbing numerical failure. Remaining
    daily losses are the registered penalty. Missing data, OOM, programming and
    solver exceptions propagate as runtime errors, never as successful episodes.
    """
    runtime.load_cached(member, day)
    interval = runtime.interval_seconds
    start_steps = runtime.physical_forward_steps
    observations, actions, daily, corrections = [], [], [], []
    failed, failure_lead, failure_reason = False, None, None
    for index in range(1, days * 86400 // interval + 1):
        lead = index * interval
        try:
            if index > 1:
                runtime.advance(differentiable=False)
            assert_finite(runtime)
            observation = runtime.observation()
            mean = model(observation)[0]
            if not torch.isfinite(mean).all():
                raise FloatingPointError('Nonfinite policy mean')
            action = mean if policy is None else policy.sample(mean)
            if not torch.isfinite(action).all():
                raise FloatingPointError('Nonfinite raw action')
            if store_actions:
                observations.append(observation.detach().cpu())
                actions.append(action.detach().cpu())
            corrections.append(runtime.apply_raw(action, density75=density75) or {})
            assert_finite(runtime)
            if lead % 86400 == 0:
                loss, _ = runtime.score(member, day, lead, include_eta=True)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite daily loss')
                daily.append(float(loss))
            runtime.detach()
        except FloatingPointError as error:
            failed, failure_lead, failure_reason = True, lead, str(error)
            break
    daily.extend([failure_daily_loss] * (days - len(daily)))
    runtime.detach()
    return Trajectory(observations, actions,
                      torch.tensor(daily, dtype=torch.float64, device=runtime.device),
                      failed, failure_reason, failure_lead, runtime.physical_forward_steps - start_steps,
                      {key: float(value) for key, value in correction_statistics(corrections).items()})
