"""Distributed error propagation for training and initialization."""
import torch.distributed as dist
from .checkpoint import distributed


def rank_zero_action(action):
    """Propagate rank-zero tracking/I/O failures before the next collective."""
    error = None
    if not distributed() or dist.get_rank() == 0:
        try:
            action()
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    if distributed():
        status = [error]
        dist.broadcast_object_list(status, src=0)
        error = status[0]
    if error is not None:
        raise RuntimeError(error)


def collective_call(action):
    """Collect every rank's failure before proceeding to the next collective."""
    result, error = None, None
    try:
        result = action()
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    errors = [error]
    if distributed():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError(f'Collective operation failed: {errors}')
    return result
