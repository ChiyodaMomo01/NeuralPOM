"""Existing daily truncated update; preserved without arithmetic changes."""
import torch
from .statistics import correction_statistics


def post_window(runtime, model, member, day, window, rollout_days, tbptt_hours=24):
    """Hourly: train only the final correction. Daily: retain the window graph."""
    interval = runtime.interval_seconds
    last_only = interval == 3600
    if last_only and tbptt_hours != 24:
        raise ValueError('Hourly Post-train updates once per day with only the final correction gradient')
    width = tbptt_hours * 3600
    if width % 86400 or rollout_days * 86400 % width:
        raise ValueError('TBPTT windows must contain whole days and divide the rollout')
    start, end = window * width, (window + 1) * width
    if window < 0 or end > rollout_days * 86400:
        raise ValueError('TBPTT window exceeds the registered rollout')
    lead = round((float(runtime.core.time) - day) * 86400)
    expected_start = interval if window == 0 else start
    if lead != expected_start:
        raise ValueError(f'TBPTT state/cursor mismatch: {lead} != {expected_start}')
    losses, metrics, corrections = [], {}, []
    for target in range(start + interval, end + 1, interval):
        if target > lead:
            with torch.set_grad_enabled(not last_only):
                runtime.advance(differentiable=not last_only)
            lead += interval
        if lead != target:
            raise ValueError('TBPTT correction is not at its registered endpoint')
        if last_only and target == end:
            runtime.detach()
        with torch.set_grad_enabled(not last_only or target == end):
            corrections.append(runtime.correct(model) or {})
        hour = (lead % 86400) // 3600
        if interval == 3600 and 1 <= hour <= 4:
            with torch.no_grad():
                _, details = runtime.score(member, day, lead)
            metrics.update({f'intraday/day{lead // 86400 + 1}/{hour}h/post/' + key: value.detach() for key, value in details.items()})
        if lead % 86400 == 0:
            post, details = runtime.score(member, day, lead)
            losses.append(post)
            day_index = lead // 86400
            metrics.update({f'day{day_index}/post/' + key: value.detach() for key, value in details.items()})
    if not losses:
        raise ValueError('TBPTT window has no daily supervision')
    metrics.update(correction_statistics(corrections))
    return torch.stack(losses).mean(), metrics
