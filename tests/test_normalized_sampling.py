from __future__ import annotations

import math

import numpy as np
import pytest

from damage_engine.errors import InputValidationError
from damage_engine.normalized_sampling import (
    calibrate_envelope_policy,
    derive_normalized_envelope,
    design_weight_quality,
    envelope_band,
    fixed_size_inclusion_probabilities,
    global_marginal_coverage_probabilities,
    idea1_selection_score,
    pivotal_sample,
)


def _proxy_result() -> dict:
    return {
        "component_support_threshold": 0.05,
        "component_metrics": {
            "component-001": {
                "expected_hits": 2.0,
                "expected_penetrations": 1.0,
            },
            "component-002": {
                "expected_hits": 0.0,
                "expected_penetrations": 0.0,
            },
        },
        "component_state_probabilities": {
            "component-001": {"degraded": 0.4, "failed": 0.1},
            "component-002": {"degraded": 0.0, "failed": 0.0},
        },
        "tree_outputs": {"effect-a": 0.2, "effect-b": 0.8},
    }


def test_normalized_envelope_uses_case_level_support_not_sample_maximum() -> None:
    result = derive_normalized_envelope(_proxy_result())

    assert result["u_hit"] == pytest.approx((1.0 - math.exp(-2.0)) / 2.0)
    assert result["u_penetration"] == pytest.approx((1.0 - math.exp(-1.0)) / 2.0)
    assert result["damage_severity"] == pytest.approx(0.2)
    assert result["normalized_utilization"] == pytest.approx(
        0.2 / result["u_penetration"]
    )
    assert result["supported_component_count"] == 1
    assert result["component_support_indicator"] == {
        "component-001": 1,
        "component-002": 0,
    }
    assert "not_sample_maximum" in result["normalization_denominator_source"]


def test_calibration_freezes_bands_and_bounded_idea1_score() -> None:
    values = [0.0, 0.1, 0.2, 0.4, 0.8]
    envelopes = []
    for index, value in enumerate(values):
        envelopes.append(
            {
                "component_ids": ["component-001", "component-002"],
                "u_penetration": value,
                "component_support_indicator": {
                    "component-001": int(index >= 1),
                    "component-002": int(index >= 3),
                },
            }
        )
    calibration = calibrate_envelope_policy(envelopes)
    edges = calibration["positive_band_edges"]

    assert calibration["zero_mass"] == pytest.approx(0.2)
    assert calibration["target_band_fractions"]["audit_zero"] == pytest.approx(0.05)
    assert sum(calibration["target_band_fractions"].values()) == pytest.approx(1.0)
    assert 0.1 <= edges["low_to_transition"] < edges["transition_to_high"] <= 0.8
    assert envelope_band({"u_penetration": 0.0}, calibration) == "audit_zero"
    assert envelope_band({"u_penetration": 1.0}, calibration) == "high"

    scored = {
        "component_hit_support": {
            "component-001": 0.8,
            "component-002": 0.3,
        },
        "utilization_information": 0.9,
        "tree_information": 0.5,
    }
    components = idea1_selection_score(scored, calibration)
    assert 0.0 < components["total"] <= 1.0
    assert components["rarity_coverage"] > 0.0


def test_zero_free_calibration_does_not_claim_an_audit_zero_quota() -> None:
    envelopes = []
    for value in (0.1, 0.2, 0.3, 0.4):
        envelopes.append(
            {
                "component_ids": ["component-001"],
                "u_penetration": value,
                "component_support_indicator": {"component-001": 1},
            }
        )

    calibration = calibrate_envelope_policy(envelopes)

    assert calibration["zero_mass"] == 0.0
    assert calibration["target_band_fractions"]["audit_zero"] == 0.0
    assert sum(calibration["target_band_fractions"].values()) == pytest.approx(1.0)


def test_fixed_size_pivotal_sampling_is_exact_deterministic_and_weighted() -> None:
    scores = np.asarray([0.0, 0.1, 0.2, 0.7, 1.0, 2.0], dtype=np.float64)
    probabilities = fixed_size_inclusion_probabilities(
        scores,
        sample_size=3,
        rho_min=0.05,
    )
    first = pivotal_sample(probabilities, seed=123)
    second = pivotal_sample(probabilities, seed=123)

    assert float(np.sum(probabilities)) == pytest.approx(3.0, abs=1.0e-9)
    assert np.all(probabilities > 0.0)
    assert np.array_equal(first, second)
    assert int(np.count_nonzero(first)) == 3
    quality = design_weight_quality(probabilities, first)
    assert quality["selected_count"] == 3
    assert 0.0 < quality["ess_fraction"] <= 1.0
    assert quality["weight_cv"] >= 0.0


def test_pivotal_sampling_preserves_first_order_probabilities() -> None:
    probabilities = np.asarray([0.1, 0.3, 0.5, 0.6, 0.5], dtype=np.float64)
    observed = np.zeros_like(probabilities)
    repetitions = 6000
    for seed in range(repetitions):
        observed += pivotal_sample(probabilities, seed=seed)
    observed /= repetitions

    assert np.max(np.abs(observed - probabilities)) < 0.025


def test_global_marginal_coverage_improves_rare_expected_counts() -> None:
    support = np.asarray(
        [
            [1, 1, 0],
            [1, 0, 1],
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [1, 0, 1],
            [1, 0, 0],
            [1, 0, 0],
        ],
        dtype=np.float64,
    )
    strata = [np.arange(0, 4), np.arange(4, 8)]
    quotas = [2, 2]
    uniform = np.full(8, 0.5, dtype=np.float64)

    first = global_marginal_coverage_probabilities(
        support,
        strata=strata,
        quotas=quotas,
    )
    second = global_marginal_coverage_probabilities(
        support,
        strata=strata,
        quotas=quotas,
    )

    assert np.array_equal(first, second)
    assert float(np.sum(first[:4])) == pytest.approx(2.0, abs=1.0e-9)
    assert float(np.sum(first[4:])) == pytest.approx(2.0, abs=1.0e-9)
    assert np.all((first >= 0.0) & (first <= 1.0))
    expected_uniform = support.T @ uniform
    expected_balanced = support.T @ first
    assert expected_balanced[1] > expected_uniform[1]
    assert expected_balanced[2] > expected_uniform[2]


def test_global_marginal_coverage_requires_partitioned_strata() -> None:
    support = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.float64)

    with pytest.raises(InputValidationError, match="partition"):
        global_marginal_coverage_probabilities(
            support,
            strata=[[0, 1], [1, 2]],
            quotas=[1, 1],
        )


def test_global_marginal_coverage_accepts_stratum_standardized_auxiliary_features() -> None:
    support = np.ones((6, 1), dtype=np.float64)
    auxiliary = np.asarray(
        [
            [0.0, 1.0],
            [0.5, 0.5],
            [1.0, 0.0],
            [10.0, 1.0],
            [15.0, 0.5],
            [20.0, 0.0],
        ],
        dtype=np.float64,
    )

    probabilities = global_marginal_coverage_probabilities(
        support,
        strata=[np.arange(0, 3), np.arange(3, 6)],
        quotas=[1, 1],
        auxiliary_features=auxiliary,
        auxiliary_weights=[1.0, -0.5],
    )

    assert float(np.sum(probabilities[:3])) == pytest.approx(1.0, abs=1.0e-9)
    assert float(np.sum(probabilities[3:])) == pytest.approx(1.0, abs=1.0e-9)
    assert probabilities[2] > probabilities[0]
    assert probabilities[5] > probabilities[3]


def test_global_marginal_coverage_rejects_misaligned_auxiliary_features() -> None:
    support = np.ones((3, 1), dtype=np.float64)

    with pytest.raises(InputValidationError, match="candidate-aligned"):
        global_marginal_coverage_probabilities(
            support,
            strata=[np.arange(3)],
            quotas=[1],
            auxiliary_features=np.ones((2, 1), dtype=np.float64),
            auxiliary_weights=[1.0],
        )
