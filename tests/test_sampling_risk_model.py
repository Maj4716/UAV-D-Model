from __future__ import annotations

import math

import numpy as np
import pytest

from damage_engine.errors import InputValidationError
from damage_engine.sampling_risk_model import (
    SAMPLING_RISK_MODEL_VERSION,
    predict_sampling_risks,
    sampling_risk_feature_names,
    sampling_risk_features,
)


def _proxy_row() -> dict:
    return {
        "profile_id": "small_front",
        "approach_profile_cell": "top_down_neg_z|small_front",
        "idea1_envelope_band": "transition",
        "idea1_envelope": {
            "component_count": 2,
            "supported_component_count": 1,
            "u_penetration": 0.1,
            "u_hit": 0.2,
            "damage_severity": 0.05,
            "normalized_utilization": 0.5,
            "envelope_consistency_excess": 0.0,
            "tree_information": 0.6,
        },
        "tree_outputs": {"effect-a": 0.2, "effect-b": 0.9},
        "expected_total_component_hits": 3.0,
        "expected_total_component_penetrations": 2.0,
        "expected_total_component_damage_energy_j": 100.0,
        "terminal_packet_weight": 5.0,
        "represented_fragment_count": 20.0,
        "source_fragment_count": 20,
    }


def test_sampling_risk_features_are_finite_and_schema_aligned() -> None:
    features = sampling_risk_features(_proxy_row())

    assert features.shape == (len(sampling_risk_feature_names()),)
    assert np.all(np.isfinite(features))
    assert features[0] == pytest.approx(math.log(0.1))


def test_sampling_risk_prediction_uses_frozen_json_coefficients() -> None:
    count = len(sampling_risk_feature_names())
    model = {
        "feature_names": list(sampling_risk_feature_names()),
        "feature_mean": [0.0] * count,
        "feature_scale": [1.0] * count,
        "coefficients": [0.0] * count,
        "intercept": 0.0,
    }
    artifact = {
        "schema_version": SAMPLING_RISK_MODEL_VERSION,
        "domain_complete_model": model,
        "tree_saturation_given_complete_model": {**model, "intercept": math.log(3.0)},
    }

    result = predict_sampling_risks([_proxy_row()], artifact)

    assert result["domain_complete_probability"][0] == pytest.approx(0.5)
    assert result["tree_saturation_given_complete_probability"][0] == pytest.approx(0.75)


def test_sampling_risk_prediction_is_stable_for_extreme_logits() -> None:
    count = len(sampling_risk_feature_names())
    base = {
        "feature_names": list(sampling_risk_feature_names()),
        "feature_mean": [0.0] * count,
        "feature_scale": [1.0] * count,
        "coefficients": [0.0] * count,
    }
    artifact = {
        "schema_version": SAMPLING_RISK_MODEL_VERSION,
        "domain_complete_model": {**base, "intercept": 1000.0},
        "tree_saturation_given_complete_model": {**base, "intercept": -1000.0},
    }

    with np.errstate(over="raise"):
        result = predict_sampling_risks([_proxy_row()], artifact)

    assert result["domain_complete_probability"][0] == 1.0
    assert result["tree_saturation_given_complete_probability"][0] == 0.0


def test_sampling_risk_prediction_rejects_feature_schema_drift() -> None:
    count = len(sampling_risk_feature_names())
    bad_model = {
        "feature_names": ["wrong"] * count,
        "feature_mean": [0.0] * count,
        "feature_scale": [1.0] * count,
        "coefficients": [0.0] * count,
        "intercept": 0.0,
    }

    with pytest.raises(InputValidationError, match="schema mismatch"):
        predict_sampling_risks(
            [_proxy_row()],
            {
                "schema_version": SAMPLING_RISK_MODEL_VERSION,
                "domain_complete_model": bad_model,
                "tree_saturation_given_complete_model": bad_model,
            },
        )
