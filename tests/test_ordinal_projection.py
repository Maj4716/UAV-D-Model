from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pytest

from damage_engine.damage_tree import compile_damage_tree
from damage_engine.errors import InputValidationError
from damage_engine.ordinal_projection import (
    aggregate_ordinal_projection,
    compile_ordinal_projection,
    hard_ordinal_gate,
    nested_cumulative_probabilities,
    ordinal_ranked_probability_score,
    positive_weight_power_mean,
    project_ordinal_replicates,
    wilson_interval,
    zero_anchored_hill,
)


def config_document(*, input_count: int = 1, tau1: float = 0.3, tau2: float = 0.7):
    return {
        "schema_version": "1.0.0",
        "projection_id": "toy-ordinal-projection-v0",
        "category_id": "toy-category",
        "status": "provisional",
        "execution_scope": "software_only",
        "scientific_claim_scope": "project_specific_simulator_emulation",
        "physical_validity_status": "not_established",
        "threshold_approval_status": "not_approved_for_real_world_kmfc",
        "threshold_source_id": "software-fixture-only",
        "input_names": [f"input-{index}" for index in range(input_count)],
        "evidence_scales": [1.0] * input_count,
        "evidence_exponents": [1.0] * input_count,
        "maximum_slopes": [1.0] * input_count,
        "numeric_tolerance": 1.0e-12,
        "aggregate_weights": [1.0] * input_count,
        "aggregate_power": 1.0,
        "tau1": tau1,
        "tau2": tau2,
        "minimum_threshold_gap": 0.1,
        "calibration_artifact_hash": None,
    }


def test_configuration_is_explicitly_provisional_software_only_and_serializable():
    compiled = compile_ordinal_projection(config_document(input_count=2))
    payload = compiled.to_dict()
    frozen = json.loads(json.dumps(payload, allow_nan=False))
    rebuilt = compile_ordinal_projection(frozen)
    assert rebuilt == compiled
    assert rebuilt.hash == compiled.hash
    assert payload["physical_validity_status"] == "not_established"
    assert payload["threshold_approval_status"] == "not_approved_for_real_world_kmfc"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "approved"),
        ("execution_scope", "research"),
        ("scientific_claim_scope", "real_world_effectiveness"),
        ("physical_validity_status", "validated"),
        ("threshold_approval_status", "approved_for_real_world_kmfc"),
    ],
)
def test_configuration_rejects_real_world_or_approved_claims(field, value):
    document = config_document()
    document[field] = value
    with pytest.raises(InputValidationError, match="software-only ordinal projection"):
        compile_ordinal_projection(document)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("evidence_scales", [0.0]),
        ("evidence_exponents", [-1.0]),
        ("evidence_exponents", [0.5]),
        ("evidence_exponents", [2.0]),
        ("maximum_slopes", [0.5]),
        ("numeric_tolerance", -1.0),
        ("numeric_tolerance", 1.0e-3),
        ("aggregate_weights", [0.0]),
        ("aggregate_power", 0.0),
        ("aggregate_power", 3.0),
        ("tau1", 0.0),
        ("tau2", 1.1),
        ("minimum_threshold_gap", 0.5),
    ],
)
def test_configuration_rejects_invalid_numeric_contracts(field, value):
    document = config_document()
    document[field] = value
    with pytest.raises(InputValidationError):
        compile_ordinal_projection(document)


def test_configuration_fails_closed_on_unknown_fields_or_bad_hash():
    unknown = config_document()
    unknown["python_callable"] = "lambda x: 1"
    with pytest.raises(InputValidationError, match="unknown"):
        compile_ordinal_projection(unknown)

    bad_hash = config_document()
    bad_hash["calibration_artifact_hash"] = "sha256:1234"
    with pytest.raises(InputValidationError, match="sha256"):
        compile_ordinal_projection(bad_hash)


def test_zero_anchored_hill_is_exact_at_zero_bounded_and_monotone():
    evidence = np.asarray([[0.0], [0.25], [1.0], [4.0], [1.0e300]])
    transformed = zero_anchored_hill(
        evidence,
        [1.0],
        [1.0],
        maximum_slopes=[1.0],
        numeric_tolerance=1.0e-12,
    )[:, 0]
    assert transformed[0] == 0.0
    assert transformed[1:4] == pytest.approx([0.2, 0.5, 0.8])
    assert np.all(np.diff(transformed) >= 0.0)
    assert np.all((0.0 <= transformed) & (transformed <= 1.0))


def test_hill_slope_cap_and_numeric_tolerance_are_frozen_and_enforced():
    accepted = config_document()
    accepted["evidence_scales"] = [2.0]
    accepted["maximum_slopes"] = [0.5]
    compiled = compile_ordinal_projection(accepted)
    assert compiled.maximum_slopes == (0.5,)
    assert compiled.numeric_tolerance == 1.0e-12

    refused = deepcopy(accepted)
    refused["maximum_slopes"] = [0.49]
    with pytest.raises(InputValidationError, match="maximum_slope"):
        compile_ordinal_projection(refused)


def test_positive_weight_power_mean_is_zero_preserving_bounded_and_monotone():
    lower = np.asarray([[0.0, 0.0], [0.1, 0.4]])
    upper = np.asarray([[0.0, 0.0], [0.3, 0.8]])
    low_score = positive_weight_power_mean(lower, [1.0, 2.0], 2.0)
    high_score = positive_weight_power_mean(upper, [1.0, 2.0], 2.0)
    assert low_score[0] == 0.0
    assert np.all(high_score >= low_score)
    assert np.all((0.0 <= high_score) & (high_score <= 1.0))


def test_power_mean_is_finite_and_invariant_to_extreme_common_weight_scale():
    values = np.asarray([[0.2, 0.8]])
    ordinary = positive_weight_power_mean(values, [1.0, 1.0], 1.0)
    huge = positive_weight_power_mean(values, [1.0e308, 1.0e308], 1.0)
    tiny = positive_weight_power_mean(values, [1.0e-308, 1.0e-308], 1.0)
    assert np.isfinite(huge).all()
    assert huge == pytest.approx(ordinary)
    assert tiny == pytest.approx(ordinary)


def test_hard_gate_is_inside_each_replicate_and_catches_aggregate_after_gate_bug():
    scores = np.asarray([0.2, 0.8])
    level1, level2 = hard_ordinal_gate(scores, tau1=0.6, tau2=0.9)
    assert level1.tolist() == [0.0, 1.0]
    assert level2.tolist() == [0.0, 0.0]
    assert float(np.mean(level1)) == 0.5
    # The forbidden alternative gates the mean and would incorrectly return 0.
    assert float(np.mean(scores) >= 0.6) == 0.0


def test_projected_levels_are_nested_and_necessary_conditions_are_per_replicate():
    compiled = compile_ordinal_projection(config_document())
    result = project_ordinal_replicates(
        [[4.0], [4.0], [4.0]],
        compiled,
        necessary_level1=np.asarray([True, False, True]),
        necessary_level2=np.asarray([False, True, True]),
    )
    assert result.severity_score == pytest.approx([0.8, 0.8, 0.8])
    assert result.level_ge_1.tolist() == [1.0, 0.0, 1.0]
    assert result.level_ge_2.tolist() == [0.0, 0.0, 1.0]
    assert np.all(result.level_ge_2 <= result.level_ge_1)


def test_threshold_equality_is_deterministic_and_inclusive():
    level1, level2 = hard_ordinal_gate(
        np.asarray([0.3, 0.7]), tau1=0.3, tau2=0.7
    )
    assert level1.tolist() == [1.0, 1.0]
    assert level2.tolist() == [0.0, 1.0]


def test_missing_evidence_or_condition_propagates_unavailable_instead_of_zero():
    compiled = compile_ordinal_projection(config_document(input_count=2))
    result = project_ordinal_replicates(
        [[1.0, 1.0], [1.0, np.nan], [4.0, 4.0]],
        compiled,
        evidence_availability=np.asarray(
            [[True, True], [True, False], [True, True]], dtype=bool
        ),
        necessary_level2_availability=np.asarray([True, True, False]),
    )
    assert result.availability.tolist() == [True, False, False]
    assert np.isnan(result.severity_score[1:]).all()
    assert np.isnan(result.level_ge_1[1:]).all()
    assert np.isnan(result.level_ge_2[1:]).all()
    payload = result.to_dict()
    assert payload["severity_score"][1:] == [None, None]
    assert payload["necessary_level2"][2] is None
    json.dumps(payload, allow_nan=False)


def test_exact_counts_classes_rates_and_wilson_intervals_are_consistent():
    compiled = compile_ordinal_projection(config_document())
    # Hill(scale=1,rho=1) maps 0,1,4 exactly to severity 0,.5,.8.
    result = project_ordinal_replicates([[0.0], [1.0], [4.0]], compiled)
    report = aggregate_ordinal_projection(result)
    assert report["trials"] == 3
    assert report["successes_level_ge_1"] == 2
    assert report["successes_level_ge_2"] == 1
    assert (report["n0"], report["n1"], report["n2"]) == (1, 1, 1)
    assert report["q1"] == pytest.approx(2.0 / 3.0)
    assert report["q2"] == pytest.approx(1.0 / 3.0)
    assert sum(report["class_probabilities"]) == pytest.approx(1.0)
    assert report["level_ge_1_interval"]["lower"] < report["q1"]
    assert report["level_ge_1_interval"]["upper"] > report["q1"]
    json.dumps(report, allow_nan=False)


def test_wilson_does_not_report_zero_uncertainty_for_zero_successes():
    interval = wilson_interval(0, 32)
    assert interval["lower"] == 0.0
    assert interval["upper"] > 0.0
    with pytest.raises(InputValidationError):
        wilson_interval(4, 3)


def test_float32_and_float64_give_identical_hard_levels_and_monotone_scores():
    compiled = compile_ordinal_projection(config_document(input_count=3))
    rng = np.random.default_rng(20261016)
    lower = rng.uniform(0.0, 10.0, size=(128, 3))
    upper = lower + rng.uniform(0.0, 2.0, size=(128, 3))
    low32 = project_ordinal_replicates(lower.astype(np.float32), compiled)
    low64 = project_ordinal_replicates(lower.astype(np.float64), compiled)
    high64 = project_ordinal_replicates(upper.astype(np.float64), compiled)
    assert np.allclose(low32.severity_score, low64.severity_score, atol=2.0e-8, rtol=2.0e-7)
    assert np.array_equal(low32.level_ge_1, low64.level_ge_1)
    assert np.array_equal(low32.level_ge_2, low64.level_ge_2)
    assert np.all(high64.severity_score >= low64.severity_score)
    assert np.all(high64.level_ge_1 >= low64.level_ge_1)
    assert np.all(high64.level_ge_2 >= low64.level_ge_2)


def test_nested_cumulative_logits_are_ordered_and_form_three_probabilities():
    logits1 = np.asarray([-1000.0, -2.0, 0.0, 2.0, 1000.0])
    logits2 = np.asarray([1000.0, 2.0, 0.0, -2.0, -1000.0])
    result = nested_cumulative_probabilities(logits1, logits2)
    assert np.all(result["q2"] <= result["q1"])
    assert np.all(result["class_probabilities"] >= 0.0)
    assert np.sum(result["class_probabilities"], axis=-1) == pytest.approx(np.ones(5))


def test_ranked_probability_score_is_zero_for_exact_prediction_and_masked_weighted():
    target = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert ordinal_ranked_probability_score(target, target) == 0.0
    predicted = np.asarray([[0.8, 0.2, 0.0], [0.2, 0.6, 0.2], [0.0, 0.3, 0.7]])
    full = ordinal_ranked_probability_score(predicted, target)
    selected = ordinal_ranked_probability_score(
        predicted,
        target,
        mask=np.asarray([True, False, True]),
        sample_weight=np.asarray([1.0, 99.0, 2.0]),
    )
    assert 0.0 < full <= 1.0
    assert 0.0 < selected <= 1.0


def test_ranked_probability_score_is_stable_for_extreme_weight_scales():
    target = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    predicted = np.asarray([[0.8, 0.2, 0.0], [0.0, 0.2, 0.8]])
    ordinary = ordinal_ranked_probability_score(
        predicted, target, sample_weight=np.asarray([1.0, 1.0])
    )
    huge = ordinal_ranked_probability_score(
        predicted, target, sample_weight=np.asarray([1.0e308, 1.0e308])
    )
    tiny = ordinal_ranked_probability_score(
        predicted, target, sample_weight=np.asarray([1.0e-308, 1.0e-308])
    )
    assert np.isfinite(huge)
    assert huge == pytest.approx(ordinary)
    assert tiny == pytest.approx(ordinary)


def test_legacy_damage_tree_is_unchanged_and_projection_remains_opt_in(target_data):
    target = deepcopy(target_data)
    tree = compile_damage_tree(target)
    states = {
        "fragment": {"flight-control": {"degraded": 0.2, "failed": 0.1}},
        "blast": {"flight-control": {"degraded": 0.4, "failed": 0.2}},
        "combined": {"flight-control": {"degraded": 0.52, "failed": 0.28}},
    }
    before = tree.evaluate(states)
    project_ordinal_replicates([[1.0]], compile_ordinal_projection(config_document()))
    after = tree.evaluate(states)
    assert after == before
    assert after["outputs"] == {"mission_loss": 0.28, "mission_degraded": 0.4}
