from __future__ import annotations

import torch


SAFE_DIV_EPS = 1e-20
SAFE_SQRT_EPS = 1e-8
SAFE_LOG_EPS = 1e-12


def safe_clamp_min(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.clamp(x, min=eps)


def safe_divide(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = SAFE_DIV_EPS) -> torch.Tensor:
    return numerator / (denominator + eps)


def safe_reciprocal(denominator: torch.Tensor, eps: float = SAFE_DIV_EPS) -> torch.Tensor:
    return 1.0 / (denominator + eps)


def safe_sqrt(x: torch.Tensor, eps: float = SAFE_SQRT_EPS) -> torch.Tensor:
    return torch.sqrt(torch.clamp(x, min=eps))


def safe_abs_pow(x: torch.Tensor, exponent: float, eps: float = SAFE_SQRT_EPS) -> torch.Tensor:
    return torch.pow(torch.clamp(torch.abs(x), min=eps), exponent)


def safe_log(x: torch.Tensor, eps: float = SAFE_LOG_EPS) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=eps))
