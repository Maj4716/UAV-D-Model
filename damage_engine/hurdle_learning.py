"""Losses for R-independent hurdle targets and interval-censored labels."""

from __future__ import annotations

from typing import Any

from .errors import InputValidationError


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise InputValidationError("PyTorch is required for hurdle losses") from exc
    return torch


def _common_shape(*values: Any):
    torch = _torch()
    if not values or any(not torch.is_tensor(value) for value in values):
        raise InputValidationError("hurdle loss inputs must be PyTorch tensors")
    shape = values[0].shape
    if len(shape) < 1 or any(value.shape != shape for value in values[1:]):
        raise InputValidationError("hurdle loss tensors must share a nonempty shape")
    return torch, shape


def _weights(raw, mask, *, design_weight=None, task_weight=None, precision_weight=None):
    torch = _torch()
    if not torch.is_tensor(mask) or mask.shape != raw.shape:
        raise InputValidationError("hurdle loss mask must match the target shape")
    weight = mask.to(dtype=raw.dtype)
    if design_weight is not None:
        if not torch.is_tensor(design_weight) or design_weight.shape != raw.shape[:-1]:
            raise InputValidationError(
                "design_weight must span case axes and remain separate from task weights"
            )
        if not torch.isfinite(design_weight).all() or torch.any(design_weight <= 0.0):
            raise InputValidationError("design_weight must be finite and positive")
        weight = weight * design_weight.to(dtype=raw.dtype).unsqueeze(-1)
    if task_weight is not None:
        if not torch.is_tensor(task_weight) or task_weight.shape != (raw.shape[-1],):
            raise InputValidationError("task_weight must span the final task axis")
        if not torch.isfinite(task_weight).all() or torch.any(task_weight <= 0.0):
            raise InputValidationError("task_weight must be finite and positive")
        weight = weight * task_weight.to(dtype=raw.dtype)
    if precision_weight is not None:
        if not torch.is_tensor(precision_weight) or precision_weight.shape != raw.shape:
            raise InputValidationError("precision_weight must match the target shape")
        observed = precision_weight[mask.to(dtype=torch.bool)]
        if not torch.isfinite(observed).all() or torch.any(observed <= 0.0):
            raise InputValidationError("available precision weights must be finite and positive")
        safe = torch.where(
            mask.to(dtype=torch.bool),
            precision_weight.to(dtype=raw.dtype),
            torch.zeros((), dtype=raw.dtype, device=raw.device),
        )
        weight = weight * safe
    denominator = weight.sum()
    if not torch.isfinite(denominator) or float(denominator.detach().cpu()) <= 0.0:
        raise InputValidationError("hurdle loss has zero effective weight")
    return weight, denominator


def masked_binomial_nll_from_logits(
    logits,
    successes,
    trials,
    mask,
    *,
    design_weight=None,
    task_weight=None,
):
    """Binomial NLL for hit counts; changing R changes precision, not the target."""

    torch, _ = _common_shape(logits, successes, trials, mask)
    available = mask.to(dtype=torch.bool)
    if not torch.isfinite(logits).all():
        raise InputValidationError("binomial logits must be finite")
    observed_successes = successes[available]
    observed_trials = trials[available]
    if (
        not torch.isfinite(observed_successes).all()
        or not torch.isfinite(observed_trials).all()
        or torch.any(observed_trials < 1.0)
        or torch.any(observed_successes < 0.0)
        or torch.any(observed_successes > observed_trials)
    ):
        raise InputValidationError("binomial counts must satisfy 0 <= successes <= trials")
    safe_successes = torch.where(available, successes.to(dtype=logits.dtype), 0.0)
    safe_trials = torch.where(available, trials.to(dtype=logits.dtype), 1.0)
    raw = safe_trials * torch.nn.functional.softplus(logits) - safe_successes * logits
    weight, denominator = _weights(
        raw, mask, design_weight=design_weight, task_weight=task_weight
    )
    return (raw * weight).sum() / denominator


def masked_interval_squared_loss(
    prediction,
    lower,
    upper,
    mask,
    *,
    design_weight=None,
    task_weight=None,
):
    """Squared distance to an admissible identification interval ``[l,u]``."""

    torch, _ = _common_shape(prediction, lower, upper, mask)
    available = mask.to(dtype=torch.bool)
    if not torch.isfinite(prediction).all():
        raise InputValidationError("interval predictions must be finite")
    lo = lower[available]
    hi = upper[available]
    if not torch.isfinite(lo).all() or not torch.isfinite(hi).all() or torch.any(lo > hi):
        raise InputValidationError("available identification intervals must be finite and ordered")
    safe_lower = torch.where(available, lower.to(dtype=prediction.dtype), prediction)
    safe_upper = torch.where(available, upper.to(dtype=prediction.dtype), prediction)
    raw = torch.relu(safe_lower - prediction) ** 2 + torch.relu(prediction - safe_upper) ** 2
    weight, denominator = _weights(
        raw, mask, design_weight=design_weight, task_weight=task_weight
    )
    return (raw * weight).sum() / denominator


def masked_heteroscedastic_gaussian_nll(
    mean,
    raw_variance,
    target,
    mask,
    *,
    observation_variance=None,
    design_weight=None,
    task_weight=None,
    precision_weight=None,
    variance_floor: float = 1.0e-8,
):
    """Gaussian NLL with predicted variance plus known Monte Carlo variance."""

    torch, _ = _common_shape(mean, raw_variance, target, mask)
    available = mask.to(dtype=torch.bool)
    if variance_floor <= 0.0:
        raise InputValidationError("variance_floor must be positive")
    if not torch.isfinite(mean).all() or not torch.isfinite(raw_variance).all():
        raise InputValidationError("Gaussian predictions must be finite")
    observed_target = target[available]
    if not torch.isfinite(observed_target).all():
        raise InputValidationError("available Gaussian targets must be finite")
    variance = torch.nn.functional.softplus(raw_variance) + variance_floor
    if observation_variance is not None:
        if not torch.is_tensor(observation_variance) or observation_variance.shape != mean.shape:
            raise InputValidationError("observation_variance must match the target shape")
        observed = observation_variance[available]
        if not torch.isfinite(observed).all() or torch.any(observed < 0.0):
            raise InputValidationError("observation_variance must be finite and nonnegative")
        variance = variance + torch.where(
            available,
            observation_variance.to(dtype=mean.dtype),
            torch.zeros((), dtype=mean.dtype, device=mean.device),
        )
    safe_target = torch.where(available, target.to(dtype=mean.dtype), mean)
    raw = 0.5 * ((safe_target - mean) ** 2 / variance + torch.log(variance))
    weight, denominator = _weights(
        raw,
        mask,
        design_weight=design_weight,
        task_weight=task_weight,
        precision_weight=precision_weight,
    )
    return (raw * weight).sum() / denominator


def ordered_state_probabilities(degraded_logits, conditional_failed_logits):
    """Parameterize ``P(failed) <= P(at least degraded)`` by construction."""

    torch, _ = _common_shape(degraded_logits, conditional_failed_logits)
    if not torch.isfinite(degraded_logits).all() or not torch.isfinite(conditional_failed_logits).all():
        raise InputValidationError("ordered state logits must be finite")
    degraded = torch.sigmoid(degraded_logits)
    failed = degraded * torch.sigmoid(conditional_failed_logits)
    return degraded, failed


def monte_carlo_precision_weight(
    sample_variance,
    sample_count,
    mask,
    *,
    variance_floor: float,
):
    """Return task-internal precision weights, distinct from design weights."""

    torch, _ = _common_shape(sample_variance, sample_count, mask)
    if variance_floor <= 0.0:
        raise InputValidationError("variance_floor must be positive")
    available = mask.to(dtype=torch.bool)
    variance = sample_variance[available]
    count = sample_count[available]
    if (
        not torch.isfinite(variance).all()
        or not torch.isfinite(count).all()
        or torch.any(variance < 0.0)
        or torch.any(count < 1.0)
    ):
        raise InputValidationError("Monte Carlo variance/count inputs are invalid")
    weight = torch.zeros_like(sample_variance)
    weight[available] = 1.0 / (variance / count + variance_floor)
    return weight


__all__ = [
    "masked_binomial_nll_from_logits",
    "masked_heteroscedastic_gaussian_nll",
    "masked_interval_squared_loss",
    "monte_carlo_precision_weight",
    "ordered_state_probabilities",
]
