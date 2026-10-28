"""Gradient measurement and clipping shared by both posttrain methods."""
import torch


def gradient_norm(parameters):
    """Accumulate in float64 so finite long-window gradients do not overflow."""
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        raise ValueError('The objective has no model gradient path')
    norms = torch.stack([torch.linalg.vector_norm(value.detach(), dtype=torch.float64) for value in gradients])
    return torch.linalg.vector_norm(norms)


@torch.no_grad()
def clip_gradients(parameters, norm, maximum):
    factor = (maximum / (norm + 1e-6)).clamp(max=1.)
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(factor.to(parameter.grad.dtype))
