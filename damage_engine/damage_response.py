from __future__ import annotations

import math

from .errors import InputValidationError


def effective_fragment_damage_threshold(state_model: dict) -> tuple[float, str]:
    """Return the threshold used by the fragment damage implementation.

    ``damage_energy_50_j`` is the canonical field for the current absorbed-plus-
    residual damage-energy model.  ``residual_energy_50_j`` remains an explicit
    compatibility fallback for older registries.  Keeping this choice in one
    deep module prevents validation and runtime from drifting apart again.
    """

    field = (
        "damage_energy_50_j"
        if state_model.get("damage_energy_50_j") is not None
        else "residual_energy_50_j"
    )
    try:
        value = float(state_model[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise InputValidationError(
            "fragment damage response is missing a numeric energy threshold",
            context={"threshold_field": field},
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise InputValidationError(
            "fragment damage response energy threshold must be finite and positive",
            context={"threshold_field": field, "value": value},
        )
    return value, field


def effective_fragment_damage_thresholds(
    fragment_model: dict,
    ordered_states: list[str],
) -> tuple[list[float], list[str]]:
    values: list[float] = []
    fields: list[str] = []
    for state in ordered_states:
        value, field = effective_fragment_damage_threshold(
            fragment_model["damage_states"][state]
        )
        values.append(value)
        fields.append(field)
    return values, fields
