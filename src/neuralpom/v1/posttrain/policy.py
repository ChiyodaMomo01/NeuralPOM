"""Fixed-variance, full-field Gaussian policy and grouped return estimator."""
from dataclasses import dataclass
import math
import torch


def validate_settings(settings, world, days):
    for key in ('group_size', 'contexts_per_batch', 'max_trajectories',
                'validation_every_trajectories'):
        if type(settings.get(key)) is not int or settings[key] <= 0:
            raise ValueError(f'RL {key} must be a positive integer')
    patience = settings.get('patience')
    if patience is not None and (type(patience) is not int or patience <= 0):
        raise ValueError('RL patience must be null (disabled) or a positive integer')
    if settings['group_size'] < 2 or settings['group_size'] * settings['contexts_per_batch'] != world:
        raise ValueError('One trajectory per rank; world = groups times contexts')
    for key in ('max_trajectories', 'validation_every_trajectories'):
        if settings[key] % world:
            raise ValueError(f'RL {key} must contain whole distributed batches')
    if days != 10:
        raise ValueError('This RL protocol requires exactly ten daily rewards')
    for key in ('noise_std', 'failure_daily_loss'):
        if not math.isfinite(settings.get(key, 0)) or settings[key] <= 0:
            raise ValueError(f'RL {key} must be finite and positive')
    if settings.get('min_delta', 0) < 0 or not math.isfinite(settings.get('min_delta', 0)):
        raise ValueError('RL min_delta must be finite and nonnegative')


def early_stop_reached(settings, bad_checks):
    """Validation remains diagnostic when patience is unset or null."""
    patience = settings.get('patience')
    return patience is not None and bad_checks >= patience


@dataclass(frozen=True)
class GaussianPolicy:
    std: float

    @torch.no_grad()
    def sample(self, mean):
        return mean.detach() + self.std * torch.randn_like(mean)

    def log_prob(self, mean, action):
        """Joint probability of the detached RAW action, before any projection.

        Sum over action dimensions, never average them: averaging changes the
        estimator by the dimension of the field. Float64 reduction avoids overflow.
        """
        error = (action.detach() - mean) / self.std
        return -.5 * error.square().sum(dtype=torch.float64) - error.numel() * math.log(self.std * math.sqrt(2 * math.pi))


def returns_to_go(daily_rewards, interval_seconds):
    """Undiscounted future daily reward at every correction endpoint."""
    if interval_seconds <= 0 or 86400 % interval_seconds:
        raise ValueError('Correction intervals must divide a day')
    daily_returns = daily_rewards.flip(-1).cumsum(-1).flip(-1)
    return daily_returns.repeat_interleave(86400 // interval_seconds, dim=-1)


def leave_one_out(returns, group_size):
    """Rows are independent trajectories, grouped by identical initial context."""
    if group_size < 2 or returns.shape[0] % group_size:
        raise ValueError('Incomplete leave-one-out groups')
    groups = returns.reshape(-1, group_size, returns.shape[-1])
    baseline = (groups.sum(1, keepdim=True) - groups) / (group_size - 1)
    return (groups - baseline).reshape_as(returns).detach()


def batch_contexts(cases, step, contexts, seed):
    """Deterministic permutation per context pass; no distributed padding."""
    if len(cases) % contexts:
        raise ValueError('Context count must divide the dataset without padding')
    batches = len(cases) // contexts
    epoch, batch = divmod(step, batches)
    order = torch.randperm(len(cases), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
    return [cases[index] for index in order[batch * contexts:(batch + 1) * contexts]]
