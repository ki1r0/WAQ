from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class IQLLosses:
    total: torch.Tensor
    q: torch.Tensor
    value: torch.Tensor
    target: torch.Tensor


@dataclass(frozen=True)
class MCFlowLosses:
    total: torch.Tensor
    q: torch.Tensor
    value: torch.Tensor
    target: torch.Tensor
    q_prediction: torch.Tensor
    value_prediction: torch.Tensor
    timestep: torch.Tensor


@dataclass(frozen=True)
class MCFlowSample:
    x_t: torch.Tensor
    timestep: torch.Tensor
    target_velocity: torch.Tensor


@dataclass(frozen=True)
class MCReturnLosses:
    total: torch.Tensor
    q: torch.Tensor
    value: torch.Tensor
    target: torch.Tensor


def expectile_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    expectile: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if not 0.0 < expectile < 1.0:
        raise ValueError("expectile must be in (0, 1)")
    difference = target - prediction
    weight = torch.where(
        difference >= 0,
        torch.as_tensor(expectile, device=difference.device),
        torch.as_tensor(1.0 - expectile, device=difference.device),
    )
    loss = weight * difference.square()
    if mask is None:
        return loss.mean()
    valid = mask.to(device=loss.device, dtype=loss.dtype)
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


def mc_q_loss(
    q1: torch.Tensor,
    q2: torch.Tensor,
    returns: torch.Tensor,
) -> torch.Tensor:
    target = returns.detach().to(q1.dtype)
    return F.mse_loss(q1, target) + F.mse_loss(q2, target)


def _regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_type: str,
    huber_delta: float,
) -> torch.Tensor:
    if loss_type == "huber":
        return F.smooth_l1_loss(prediction, target, beta=huber_delta)
    if loss_type == "mse":
        return F.mse_loss(prediction, target)
    raise ValueError("loss_type must be 'huber' or 'mse'")


def mc_return_losses(
    q1: torch.Tensor,
    q2: torch.Tensor,
    value: torch.Tensor,
    returns: torch.Tensor,
    value_weight: float = 1.0,
    value_mask: torch.Tensor | None = None,
    loss_type: str = "huber",
    huber_delta: float = 1.0,
) -> MCReturnLosses:
    target = returns.detach().view(-1)
    target_q1 = target.to(q1.dtype)
    target_q2 = target.to(q2.dtype)
    q_loss = _regression_loss(
        q1.view_as(target_q1),
        target_q1,
        loss_type=loss_type,
        huber_delta=huber_delta,
    ) + _regression_loss(
        q2.view_as(target_q2),
        target_q2,
        loss_type=loss_type,
        huber_delta=huber_delta,
    )

    value_prediction = value.view_as(target)
    value_target = target.to(value_prediction.dtype)
    value_loss_unmasked = (
        F.smooth_l1_loss(
            value_prediction,
            value_target,
            beta=huber_delta,
            reduction="none",
        )
        if loss_type == "huber"
        else (value_prediction - value_target).square()
    )
    if value_mask is None:
        value_loss = value_loss_unmasked.mean()
    else:
        valid = value_mask.to(
            device=value_loss_unmasked.device,
            dtype=value_loss_unmasked.dtype,
        ).view_as(value_loss_unmasked)
        value_loss = (
            value_loss_unmasked * valid
        ).sum() / valid.sum().clamp_min(1.0)

    total = q_loss + value_weight * value_loss
    return MCReturnLosses(
        total=total,
        q=q_loss,
        value=value_loss,
        target=target,
    )


def sample_mc_flow_target(
    target: torch.Tensor,
    timestep_range: tuple[float, float] = (0.0, 1.0),
    target_noise: torch.Tensor | None = None,
) -> MCFlowSample:
    start, end = timestep_range
    if not (0.0 <= start <= end <= 1.0):
        raise ValueError("timestep_range must satisfy 0 <= start <= end <= 1")
    target = target.detach().view(-1)
    if target_noise is None:
        target_noise = torch.randn_like(target)
    else:
        target_noise = target_noise.to(target.device, target.dtype).view_as(target)
    timestep = torch.empty_like(target).uniform_(float(start), float(end))
    x_t = (1.0 - timestep) * target_noise + timestep * target
    target_velocity = target - target_noise
    return MCFlowSample(
        x_t=x_t,
        timestep=timestep,
        target_velocity=target_velocity,
    )


def mc_flow_losses(
    q_predictions: list[torch.Tensor],
    value_prediction: torch.Tensor,
    target_velocity: torch.Tensor,
    target: torch.Tensor,
    timestep: torch.Tensor,
    value_weight: float = 1.0,
    loss_type: str = "huber",
    huber_delta: float = 1.0,
    value_mask: torch.Tensor | None = None,
) -> MCFlowLosses:
    if not q_predictions:
        raise ValueError("q_predictions must contain at least one Q head")
    target = target.detach().view(-1)
    target_velocity = target_velocity.detach().view_as(target)

    q_losses = [
        _regression_loss(
            prediction.view_as(target),
            target_velocity.to(prediction.dtype),
            loss_type=loss_type,
            huber_delta=huber_delta,
        )
        for prediction in q_predictions
    ]
    q_loss = torch.stack(q_losses).mean()

    value_prediction = value_prediction.view_as(target)
    value_loss_unmasked = F.smooth_l1_loss(
        value_prediction,
        target_velocity.to(value_prediction.dtype),
        beta=huber_delta,
        reduction="none",
    ) if loss_type == "huber" else (
        value_prediction - target_velocity.to(value_prediction.dtype)
    ).square()
    if value_mask is None:
        value_loss = value_loss_unmasked.mean()
    else:
        valid = value_mask.to(
            device=value_loss_unmasked.device,
            dtype=value_loss_unmasked.dtype,
        ).view_as(value_loss_unmasked)
        value_loss = (
            value_loss_unmasked * valid
        ).sum() / valid.sum().clamp_min(1.0)

    total = q_loss + value_weight * value_loss
    q_stack = torch.stack([prediction.detach().view_as(target) for prediction in q_predictions])
    return MCFlowLosses(
        total=total,
        q=q_loss,
        value=value_loss,
        target=target,
        q_prediction=q_stack.mean(dim=0),
        value_prediction=value_prediction.detach(),
        timestep=timestep.detach(),
    )


def iql_losses(
    q1: torch.Tensor,
    q2: torch.Tensor,
    value: torch.Tensor,
    next_target_value: torch.Tensor,
    reward: torch.Tensor,
    discount: torch.Tensor,
    expectile: float,
    value_weight: float = 1.0,
    value_mask: torch.Tensor | None = None,
    q_loss_type: str = "huber",
    q_huber_delta: float = 1.0,
) -> IQLLosses:
    q_min = torch.minimum(q1, q2).detach()
    value_loss = expectile_loss(
        value, q_min, expectile, mask=value_mask
    )
    target = (
        reward.to(next_target_value.dtype)
        + discount.to(next_target_value.dtype)
        * next_target_value.detach()
    )
    target_q1 = target.to(q1.dtype)
    target_q2 = target.to(q2.dtype)
    if q_loss_type == "huber":
        q_loss = (
            F.smooth_l1_loss(q1, target_q1, beta=q_huber_delta)
            + F.smooth_l1_loss(q2, target_q2, beta=q_huber_delta)
        )
    elif q_loss_type == "mse":
        q_loss = F.mse_loss(q1, target_q1) + F.mse_loss(q2, target_q2)
    else:
        raise ValueError("q_loss_type must be 'huber' or 'mse'")
    total = q_loss + value_weight * value_loss
    return IQLLosses(
        total=total,
        q=q_loss,
        value=value_loss,
        target=target,
    )
