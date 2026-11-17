"""Frozen pre-outcome risk features for coverage-aware sampling.

The model predicts two selection-time quantities from deterministic proxy rows:
teacher-domain completeness and teacher-tree saturation conditional on complete
domain.  It never consumes teacher labels at prediction time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np

from .errors import InputValidationError


SAMPLING_RISK_MODEL_VERSION = "sampling-risk-logistic@1.0.0"
RISK_PROFILES = ("small_front", "medium_front", "large_front")
RISK_APPROACHES = (
    "bottom_up_pos_z",
    "front_to_rear_neg_y",
    "left_to_right_pos_x",
    "rear_to_front_pos_y",
    "right_to_left_neg_x",
    "top_down_neg_z",
)
RISK_BANDS = ("audit_zero", "low", "transition", "high")


def sampling_risk_feature_names() -> tuple[str, ...]:
    base = (
        "log_u_penetration",
        "u_penetration",
        "u_hit",
        "damage_severity",
        "normalized_utilization",
        "envelope_consistency_excess",
        "supported_component_fraction",
        "tree_max",
        "tree_mean",
        "tree_information",
        "log1p_expected_component_hits",
        "log1p_expected_component_penetrations",
        "log1p_expected_damage_energy_j",
        "terminal_packet_fraction",
        "log1p_source_fragment_count",
        "u_times_tree_max",
        "support_times_tree_max",
        "utilization_times_tree_max",
        "u_times_support",
        "log_u_squared",
        "tree_max_squared",
        "support_fraction_squared",
    )
    categorical = tuple(f"profile={value}" for value in RISK_PROFILES) + tuple(
        f"approach={value}" for value in RISK_APPROACHES
    ) + tuple(f"band={value}" for value in RISK_BANDS)
    profile_interactions = tuple(
        f"profile={profile}*{feature}"
        for profile in RISK_PROFILES
        for feature in ("u_penetration", "tree_max", "supported_component_fraction")
    )
    return base + categorical + profile_interactions


def _finite(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputValidationError(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise InputValidationError(f"{field} must be finite")
    return result


def sampling_risk_features(proxy_row: Mapping[str, Any]) -> np.ndarray:
    """Return the frozen numeric feature vector for one proxy row."""

    envelope = proxy_row.get("idea1_envelope")
    tree = proxy_row.get("tree_outputs")
    if not isinstance(envelope, Mapping) or not isinstance(tree, Mapping):
        raise InputValidationError(
            "sampling-risk features require idea1_envelope and tree_outputs"
        )
    profile = str(proxy_row.get("profile_id"))
    cell = str(proxy_row.get("approach_profile_cell"))
    approach = cell.split("|", 1)[0]
    band = str(proxy_row.get("idea1_envelope_band"))
    if profile not in RISK_PROFILES:
        raise InputValidationError(f"unsupported sampling-risk profile: {profile}")
    if approach not in RISK_APPROACHES:
        raise InputValidationError(f"unsupported sampling-risk approach: {approach}")
    if band not in RISK_BANDS:
        raise InputValidationError(f"unsupported sampling-risk band: {band}")

    u_penetration = _finite(
        envelope.get("u_penetration"), field="u_penetration"
    )
    u_hit = _finite(envelope.get("u_hit"), field="u_hit")
    severity = _finite(
        envelope.get("damage_severity"), field="damage_severity"
    )
    utilization = _finite(
        envelope.get("normalized_utilization"), field="normalized_utilization"
    )
    consistency = _finite(
        envelope.get("envelope_consistency_excess"),
        field="envelope_consistency_excess",
    )
    component_count = int(envelope.get("component_count", 0))
    if component_count <= 0:
        raise InputValidationError("component_count must be positive")
    supported_fraction = _finite(
        envelope.get("supported_component_count"),
        field="supported_component_count",
    ) / component_count
    tree_values = np.asarray(
        [_finite(value, field=f"tree_outputs.{name}") for name, value in tree.items()],
        dtype=np.float64,
    )
    tree_max = float(np.max(tree_values)) if tree_values.size else 0.0
    tree_mean = float(np.mean(tree_values)) if tree_values.size else 0.0
    tree_information = _finite(
        envelope.get("tree_information"), field="tree_information"
    )
    expected_hits = max(
        0.0,
        _finite(
            proxy_row.get("expected_total_component_hits"),
            field="expected_total_component_hits",
        ),
    )
    expected_penetrations = max(
        0.0,
        _finite(
            proxy_row.get("expected_total_component_penetrations"),
            field="expected_total_component_penetrations",
        ),
    )
    expected_energy = max(
        0.0,
        _finite(
            proxy_row.get("expected_total_component_damage_energy_j"),
            field="expected_total_component_damage_energy_j",
        ),
    )
    terminal_weight = max(
        0.0,
        _finite(
            proxy_row.get("terminal_packet_weight"),
            field="terminal_packet_weight",
        ),
    )
    represented_count = max(
        1.0,
        _finite(
            proxy_row.get("represented_fragment_count"),
            field="represented_fragment_count",
        ),
    )
    source_count = max(
        0.0,
        _finite(
            proxy_row.get("source_fragment_count"),
            field="source_fragment_count",
        ),
    )
    log_u = math.log(max(u_penetration, 1.0e-12))
    base = [
        log_u,
        u_penetration,
        u_hit,
        severity,
        utilization,
        consistency,
        supported_fraction,
        tree_max,
        tree_mean,
        tree_information,
        math.log1p(expected_hits),
        math.log1p(expected_penetrations),
        math.log1p(expected_energy),
        terminal_weight / represented_count,
        math.log1p(source_count),
        u_penetration * tree_max,
        supported_fraction * tree_max,
        utilization * tree_max,
        u_penetration * supported_fraction,
        log_u * log_u,
        tree_max * tree_max,
        supported_fraction * supported_fraction,
    ]
    profile_one_hot = [float(profile == value) for value in RISK_PROFILES]
    categorical = (
        profile_one_hot
        + [float(approach == value) for value in RISK_APPROACHES]
        + [float(band == value) for value in RISK_BANDS]
    )
    interactions = [
        indicator * feature
        for indicator in profile_one_hot
        for feature in (u_penetration, tree_max, supported_fraction)
    ]
    result = np.asarray(base + categorical + interactions, dtype=np.float64)
    if result.size != len(sampling_risk_feature_names()) or not np.all(
        np.isfinite(result)
    ):
        raise RuntimeError("sampling-risk feature construction is invalid")
    return result


def _predict_model(features: np.ndarray, model: Mapping[str, Any]) -> np.ndarray:
    names = tuple(str(value) for value in model.get("feature_names", []))
    if names != sampling_risk_feature_names():
        raise InputValidationError("sampling-risk model feature schema mismatch")
    mean = np.asarray(model.get("feature_mean"), dtype=np.float64)
    scale = np.asarray(model.get("feature_scale"), dtype=np.float64)
    coefficients = np.asarray(model.get("coefficients"), dtype=np.float64)
    intercept = _finite(model.get("intercept"), field="model.intercept")
    expected = len(names)
    if (
        mean.shape != (expected,)
        or scale.shape != (expected,)
        or coefficients.shape != (expected,)
        or np.any(scale <= 0.0)
        or not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(scale))
        or not np.all(np.isfinite(coefficients))
    ):
        raise InputValidationError("sampling-risk model coefficients are invalid")
    standardized = (features - mean) / scale
    logits = standardized @ coefficients + intercept
    probabilities = np.empty_like(logits, dtype=np.float64)
    nonnegative = logits >= 0.0
    probabilities[nonnegative] = 1.0 / (
        1.0 + np.exp(-logits[nonnegative])
    )
    negative_exp = np.exp(logits[~nonnegative])
    probabilities[~nonnegative] = negative_exp / (1.0 + negative_exp)
    return probabilities


def predict_sampling_risks(
    proxy_rows: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    """Predict domain completeness and conditional tree saturation."""

    if artifact.get("schema_version") != SAMPLING_RISK_MODEL_VERSION:
        raise InputValidationError("unsupported sampling-risk model version")
    if not proxy_rows:
        raise InputValidationError("sampling-risk prediction requires proxy rows")
    features = np.vstack([sampling_risk_features(row) for row in proxy_rows])
    domain = artifact.get("domain_complete_model")
    tree = artifact.get("tree_saturation_given_complete_model")
    if not isinstance(domain, Mapping) or not isinstance(tree, Mapping):
        raise InputValidationError("sampling-risk artifact requires both models")
    return {
        "domain_complete_probability": _predict_model(features, domain),
        "tree_saturation_given_complete_probability": _predict_model(
            features, tree
        ),
    }
