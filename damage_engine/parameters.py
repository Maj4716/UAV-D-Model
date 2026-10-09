from __future__ import annotations

import math

import numpy as np

from .errors import InputValidationError
from .utils import (
    duration_to_s,
    impulse_to_pa_s,
    length_to_m,
    pressure_to_pa,
    ratio_value,
)


def _is_epistemic(parameter: dict) -> bool:
    return bool((parameter.get("uncertainty") or {}).get("epistemic", False))


def _positive(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise InputValidationError(f"{name} must be finite and positive")
    return value


def build_epistemic_realization(
    target: dict,
    rng: np.random.Generator,
) -> dict[str, dict[str, float | None]]:
    """Sample case-level epistemic parameters exactly once.

    Non-epistemic parameters are represented by ``None`` and sampled per
    aleatory replicate by :func:`sample_target_realization`.
    """

    output: dict[str, dict[str, float | None]] = {}
    for item in target["armor"] + target["components"]:
        response = item["response"]
        blast_response = response["blast_response"]
        threshold = blast_response.get("peak_overpressure_threshold")
        impulse_threshold = blast_response.get("impulse_threshold")
        duration_threshold = blast_response.get("duration_threshold")
        output[str(item["id"])] = {
            "equivalent_thickness_m": (
                _positive(
                    length_to_m(response["equivalent_thickness"], rng),
                    name="equivalent thickness",
                )
                if _is_epistemic(response["equivalent_thickness"])
                else None
            ),
            "vulnerable_area_ratio": (
                ratio_value(response["vulnerable_area_ratio"], rng)
                if _is_epistemic(response["vulnerable_area_ratio"])
                else None
            ),
            "blast_peak_threshold_pa": (
                _positive(
                    pressure_to_pa(threshold, rng),
                    name="blast peak threshold",
                )
                if threshold and _is_epistemic(threshold)
                else None
            ),
            "blast_impulse_threshold_pa_s": (
                _positive(
                    impulse_to_pa_s(impulse_threshold, rng),
                    name="blast impulse threshold",
                )
                if impulse_threshold and _is_epistemic(impulse_threshold)
                else None
            ),
            "blast_duration_threshold_s": (
                _positive(
                    duration_to_s(duration_threshold, rng),
                    name="blast duration threshold",
                )
                if duration_threshold and _is_epistemic(duration_threshold)
                else None
            ),
        }
    return output


def sample_target_realization(
    target: dict,
    rng: np.random.Generator,
    epistemic: dict[str, dict[str, float | None]],
) -> dict[str, dict[str, float | None]]:
    output: dict[str, dict[str, float | None]] = {}
    for item in target["armor"] + target["components"]:
        identifier = str(item["id"])
        response = item["response"]
        fixed = epistemic[identifier]
        blast_response = response["blast_response"]
        threshold = blast_response.get("peak_overpressure_threshold")
        impulse_threshold = blast_response.get("impulse_threshold")
        duration_threshold = blast_response.get("duration_threshold")
        output[identifier] = {
            "equivalent_thickness_m": (
                fixed["equivalent_thickness_m"]
                if fixed["equivalent_thickness_m"] is not None
                else _positive(
                    length_to_m(response["equivalent_thickness"], rng),
                    name="equivalent thickness",
                )
            ),
            "vulnerable_area_ratio": (
                fixed["vulnerable_area_ratio"]
                if fixed["vulnerable_area_ratio"] is not None
                else ratio_value(response["vulnerable_area_ratio"], rng)
            ),
            "blast_peak_threshold_pa": (
                fixed.get("blast_peak_threshold_pa")
                if fixed.get("blast_peak_threshold_pa") is not None
                else (
                    _positive(
                        pressure_to_pa(threshold, rng),
                        name="blast peak threshold",
                    )
                    if threshold
                    else None
                )
            ),
            "blast_impulse_threshold_pa_s": (
                fixed.get("blast_impulse_threshold_pa_s")
                if fixed.get("blast_impulse_threshold_pa_s") is not None
                else (
                    _positive(
                        impulse_to_pa_s(impulse_threshold, rng),
                        name="blast impulse threshold",
                    )
                    if impulse_threshold
                    else None
                )
            ),
            "blast_duration_threshold_s": (
                fixed.get("blast_duration_threshold_s")
                if fixed.get("blast_duration_threshold_s") is not None
                else (
                    _positive(
                        duration_to_s(duration_threshold, rng),
                        name="blast duration threshold",
                    )
                    if duration_threshold
                    else None
                )
            ),
        }
    return output


def nominal_target_parameters(target: dict) -> dict[str, dict[str, float | None]]:
    output: dict[str, dict[str, float | None]] = {}
    for item in target["armor"] + target["components"]:
        response = item["response"]
        blast_response = response["blast_response"]
        threshold = blast_response.get("peak_overpressure_threshold")
        impulse_threshold = blast_response.get("impulse_threshold")
        duration_threshold = blast_response.get("duration_threshold")
        output[str(item["id"])] = {
            "equivalent_thickness_m": _positive(
                length_to_m(response["equivalent_thickness"]),
                name="equivalent thickness",
            ),
            "vulnerable_area_ratio": ratio_value(response["vulnerable_area_ratio"]),
            "blast_peak_threshold_pa": (
                _positive(pressure_to_pa(threshold), name="blast peak threshold")
                if threshold
                else None
            ),
            "blast_impulse_threshold_pa_s": (
                _positive(
                    impulse_to_pa_s(impulse_threshold),
                    name="blast impulse threshold",
                )
                if impulse_threshold
                else None
            ),
            "blast_duration_threshold_s": (
                _positive(
                    duration_to_s(duration_threshold),
                    name="blast duration threshold",
                )
                if duration_threshold
                else None
            ),
        }
    return output
