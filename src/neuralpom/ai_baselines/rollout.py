"""One-day gradients on self-generated states, never teacher-forced in Posttrain."""
from __future__ import annotations
import torch
from .state import standardized_loss
from .distributed import phase


def local_daily_update(runtime, model, current, reference, optimizer, clip_norm=10.):
    # Detaching at entry removes every previous day's parameter path.
    optimizer.zero_grad(set_to_none=True)
    prediction = phase(lambda: runtime.predict(model, current.detach()), current.device)
    target = phase(lambda: runtime.normalizer.encode(runtime.codec.pack(reference)), current.device)
    loss = phase(lambda: standardized_loss(prediction, target, runtime.codec.weights(runtime.core, reference)), current.device)
    loss.backward()
    norm = phase(lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True), current.device)
    optimizer.step()
    # Carry this actual prediction, without recomputing it after the update.
    return prediction.detach(), loss.detach(), norm.detach()


@torch.no_grad()
def forecast(runtime, model, initial, days):
    current = initial.detach()
    for day in range(1, days+1):
        current = runtime.predict(model, current).detach()
        yield day, current
