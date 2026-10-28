"""Derivative conventions for exact physical forward equations."""
import torch


class _StrainMagnitude(torch.autograd.Function):
    @staticmethod
    def forward(ctx, squared_invariant):
        magnitude = torch.sqrt(squared_invariant)
        ctx.save_for_backward(magnitude)
        return magnitude

    @staticmethod
    def backward(ctx, grad_output):
        magnitude, = ctx.saved_tensors
        positive = magnitude > 0
        denominator = torch.where(positive, magnitude, torch.ones_like(magnitude))
        derivative = torch.where(positive, 0.5 / denominator, torch.zeros_like(magnitude))
        return grad_output * derivative


def strain_magnitude(squared_invariant):
    """Exact sqrt forward, zero subgradient of the strain norm at zero strain.

    The invariant is a sum of squared velocity derivatives. Direct composition
    of sqrt and square yields 0 * inf at zero; the norm itself admits the zero
    subgradient there. Positive-strain derivatives and all forward values are
    unchanged. This introduces no viscosity floor or gradient truncation.
    """
    return _StrainMagnitude.apply(squared_invariant)
