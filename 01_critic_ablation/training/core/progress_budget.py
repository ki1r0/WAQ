"""Calibrated Progress Budget (CPB) scheduler math.

The functions in this module are intentionally independent of the inference
server so they can be tested without loading Wan, a critic, or V-JEPA.
"""

from __future__ import annotations

import torch


def update_ema(
    previous: torch.Tensor | None,
    current: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("EMA alpha must be in (0, 1]")
    current = current.float()
    if previous is None:
        return current
    return alpha * current + (1.0 - alpha) * previous.to(current)


def calibrated_progress(
    initial_distance: torch.Tensor,
    smoothed_distance: torch.Tensor,
    success_reference: float,
    eps: float,
) -> torch.Tensor:
    initial = initial_distance.float()
    current = smoothed_distance.to(initial)
    reference = torch.as_tensor(success_reference, device=initial.device, dtype=initial.dtype)
    denominator = (initial - reference).clamp_min(eps)
    return ((initial - current) / denominator).clamp(0.0, 1.0)


def distance_confidence(
    distance: torch.Tensor,
    low: float,
    high: float,
    margin: float,
    eps: float,
) -> torch.Tensor:
    if high <= low:
        raise ValueError("CPB distance high bound must exceed low bound")
    if margin < 0.0:
        raise ValueError("CPB OOD margin must be non-negative")
    value = distance.float()
    width = max(float(margin), float(eps))
    below = ((value - (low - width)) / width).clamp(0.0, 1.0)
    above = (((high + width) - value) / width).clamp(0.0, 1.0)
    return torch.minimum(below, above)


def blended_task_budget(
    progress: torch.Tensor,
    confidence: torch.Tensor,
    *,
    fixed_budget: float,
    minimum_budget: float,
    maximum_budget: float,
    kappa: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 <= minimum_budget <= maximum_budget:
        raise ValueError("CPB budgets must satisfy 0 <= minimum <= maximum")
    if fixed_budget < 0.0:
        raise ValueError("CPB fixed budget must be non-negative")
    if kappa <= 0.0:
        raise ValueError("CPB kappa must be positive")
    progress = progress.float().clamp(0.0, 1.0)
    confidence = confidence.to(progress).clamp(0.0, 1.0)
    distance_budget = minimum_budget + (
        maximum_budget - minimum_budget
    ) * (1.0 - progress).pow(kappa)
    budget = (1.0 - confidence) * fixed_budget + confidence * distance_budget
    return budget, distance_budget


def beta_envelope(
    sigmas: torch.Tensor,
    *,
    peak: float,
    concentration: float,
    eps: float,
) -> torch.Tensor:
    if not 0.0 < peak < 1.0:
        raise ValueError("CPB sigma peak must be in (0, 1)")
    if concentration <= 0.0:
        raise ValueError("CPB sigma concentration must be positive")
    sigma = sigmas.float().clamp(0.0, 1.0)
    exponent_sigma = peak * concentration
    exponent_one_minus = (1.0 - peak) * concentration
    shape = sigma.pow(exponent_sigma) * (1.0 - sigma).pow(exponent_one_minus)
    total = shape.sum()
    if not torch.isfinite(total) or float(total) <= eps:
        return torch.full_like(shape, 1.0 / max(1, shape.numel()))
    return shape / total
