from __future__ import annotations

from copy import deepcopy
import math

import pytest

from damage_engine.engine import DamageEngine
from damage_engine.errors import InputValidationError
from damage_engine.hurdle_targets import (
    CONDITIONAL_METRIC_CONDITIONING,
    HIT_PROBABILITY_ESTIMAND,
    HURDLE_TARGET_SPEC_ID,
    extract_component_hurdle_targets,
)
from damage_engine.utils import statistics_dict


def _artificial_result(hit_sequence: list[float]) -> dict:
    count = len(hit_sequence)
    degraded = [0.6 * value for value in hit_sequence]
    failed = [0.2 * value for value in hit_sequence]
    return {
        "case_id": f"artificial-r{count}",
        "target_id": "artificial-target",
        "replicate_count": count,
        "scientific_completion_status": (
            "observed_domain_complete_not_statistically_demonstrated"
        ),
        "penetration_domain_coverage": {"affected_replicate_fraction": 0.0},
        "components": {
            "component-001": {
                "fragment_metrics": {
                    "hit_indicator": statistics_dict(
                        hit_sequence,
                        probability=True,
                    ),
                    "hits": statistics_dict(hit_sequence),
                    "deposited_energy_j": statistics_dict(
                        [100.0 * value for value in hit_sequence]
                    ),
                },
                "damage_states": {
                    "fragment": {
                        "degraded": statistics_dict(degraded, probability=True),
                        "failed": statistics_dict(failed, probability=True),
                    }
                },
            }
        },
    }


def test_repeated_sequence_preserves_r_independent_estimand_semantics() -> None:
    sequence_r4 = [0.0, 0.0, 1.0, 1.0]
    r4 = extract_component_hurdle_targets(_artificial_result(sequence_r4))
    r16 = extract_component_hurdle_targets(
        _artificial_result(sequence_r4 * 4)
    )
    target_r4 = r4["components"]["component-001"]["p_hit"]
    target_r16 = r16["components"]["component-001"]["p_hit"]

    assert r4["target_spec_id"] == r16["target_spec_id"] == HURDLE_TARGET_SPEC_ID
    assert target_r4["estimand"] == target_r16["estimand"] == HIT_PROBABILITY_ESTIMAND
    assert target_r4["value"] == target_r16["value"] == pytest.approx(0.5)
    assert target_r4["sample_count"] == 4
    assert target_r16["sample_count"] == 16
    assert target_r16["standard_error"] < target_r4["standard_error"]
    assert target_r4["binomial_observation"] == {
        "success_count": 2,
        "trial_count": 4,
        "estimand_is_R_independent": True,
    }
    assert target_r4["identification_interval"]["width"] == 0.0

    support_r4 = r4["components"]["component-001"]["derived_support"]
    support_r16 = r16["components"]["component-001"]["derived_support"]
    assert support_r4["status"] == "ambiguous_ci_crosses_threshold"
    assert support_r4["mask"] is False
    assert support_r16["status"] == "positive"
    assert support_r16["value"] == 1
    assert support_r16["evidence_ci95"] == target_r16["ci95"]
    assert support_r4["decision_rule"] == support_r16["decision_rule"]


def test_unconditional_aggregates_are_not_substituted_for_hit_conditionals() -> None:
    extracted = extract_component_hurdle_targets(
        _artificial_result([0.0, 1.0, 0.0, 1.0])
    )
    conditionals = extracted["components"]["component-001"][
        "conditional_positive_targets"
    ]

    assert conditionals["log1p_deposited_energy_j"]["mask"] is False
    assert conditionals["log1p_hit_count"]["mask"] is False
    assert "unconditional totals" in conditionals["log1p_deposited_energy_j"]["reason"]


def test_explicit_conditioned_statistic_requires_enough_positive_replicates() -> None:
    result = _artificial_result([0.0, 1.0, 0.0, 1.0])
    metrics = result["components"]["component-001"]["fragment_metrics"]
    energy = statistics_dict([math.log1p(100.0), math.log1p(200.0)])
    energy["conditioning"] = CONDITIONAL_METRIC_CONDITIONING
    energy["estimand"] = (
        "conditional_mean_log1p_deposited_energy_j_given_component_hit"
    )
    hits = statistics_dict([math.log1p(2.0)])
    hits["conditioning"] = CONDITIONAL_METRIC_CONDITIONING
    hits["estimand"] = "conditional_mean_log1p_hit_count_given_component_hit"
    metrics["hit_conditioned_log1p_deposited_energy_j"] = energy
    metrics["hit_conditioned_log1p_hit_count"] = hits

    extracted = extract_component_hurdle_targets(result)
    conditionals = extracted["components"]["component-001"][
        "conditional_positive_targets"
    ]

    assert conditionals["log1p_deposited_energy_j"]["mask"] is True
    assert conditionals["log1p_deposited_energy_j"]["sample_count"] == 2
    assert conditionals["log1p_hit_count"]["mask"] is False
    assert conditionals["log1p_hit_count"]["status"] == "unavailable"
    assert conditionals["log1p_hit_count"]["sample_count"] == 1


def test_explicit_zero_count_conditioned_statistic_is_masked() -> None:
    result = _artificial_result([0.0, 0.0, 0.0, 0.0])
    empty = statistics_dict([])
    empty.update(
        {
            "conditioning": CONDITIONAL_METRIC_CONDITIONING,
            "estimand": "conditional_mean_log1p_deposited_energy_j_given_component_hit",
        }
    )
    result["components"]["component-001"]["fragment_metrics"][
        "hit_conditioned_log1p_deposited_energy_j"
    ] = empty

    conditional = extract_component_hurdle_targets(result)["components"][
        "component-001"
    ]["conditional_positive_targets"]["log1p_deposited_energy_j"]

    assert conditional["value"] is None
    assert conditional["sample_count"] == 0
    assert conditional["standard_error"] is None
    assert conditional["mask"] is False
    assert conditional["status"] == "unavailable"
    assert "zero hit-positive replicates" in conditional["reason"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"sample_count": -1}, "non-negative integer"),
        ({"conditioning": "wrong-conditioning"}, "incompatible conditioning"),
        ({"estimand": "wrong-estimand"}, "incompatible estimand"),
        ({"mean": 0.0}, "value and uncertainty fields must be null"),
    ],
)
def test_invalid_zero_count_conditioned_statistic_is_rejected(
    mutation: dict,
    message: str,
) -> None:
    result = _artificial_result([0.0, 0.0, 0.0, 0.0])
    empty = statistics_dict([])
    empty.update(
        {
            "conditioning": CONDITIONAL_METRIC_CONDITIONING,
            "estimand": "conditional_mean_log1p_deposited_energy_j_given_component_hit",
            **mutation,
        }
    )
    result["components"]["component-001"]["fragment_metrics"][
        "hit_conditioned_log1p_deposited_energy_j"
    ] = empty

    with pytest.raises(InputValidationError, match=message):
        extract_component_hurdle_targets(result)


def test_mislabeled_conditional_statistic_is_rejected() -> None:
    result = _artificial_result([0.0, 1.0, 0.0, 1.0])
    metric = statistics_dict([math.log1p(100.0), math.log1p(200.0)])
    result["components"]["component-001"]["fragment_metrics"][
        "hit_conditioned_log1p_deposited_energy_j"
    ] = metric

    with pytest.raises(InputValidationError, match="incompatible conditioning"):
        extract_component_hurdle_targets(result)


def test_invalid_probability_aggregate_is_rejected() -> None:
    result = _artificial_result([0.0, 1.0, 0.0, 1.0])
    result["components"]["component-001"]["fragment_metrics"][
        "hit_indicator"
    ]["mean"] = 1.2

    with pytest.raises(InputValidationError, match="must lie in"):
        extract_component_hurdle_targets(result)


def test_domain_censoring_interval_is_separate_from_monte_carlo_ci() -> None:
    result = _artificial_result([0.0, 1.0, 0.0, 1.0])
    result["scientific_completion_status"] = "transition_or_indeterminate"
    result["penetration_domain_coverage"]["affected_replicate_fraction"] = 0.25

    extracted = extract_component_hurdle_targets(result, domain_epsilon=0.1)
    target = extracted["components"]["component-001"]["p_hit"]

    assert target["mask"] is False
    assert target["ci95"] is not None
    assert target["identification_interval"]["lower"] == pytest.approx(0.5)
    assert target["identification_interval"]["upper"] == pytest.approx(0.75)
    assert extracted["source"]["epsilon_complete"] is False
    assert extracted["source"]["censoring_supervision"] == "identification_interval_only"
    assert extracted["uncertainty_separation"][
        "must_not_be_combined_as_one_confidence_interval"
    ] is True


def test_component_specific_bounds_enable_only_the_identified_probability_heads() -> None:
    result = _artificial_result([0.0, 1.0, 0.0, 1.0])
    result["scientific_completion_status"] = "partial_domain_censored"
    result["penetration_domain_coverage"] = {
        "affected_replicate_fraction": 1.0,
        "probability_bounds": {
            "components": {
                "component-001": {
                    "fragment_metrics": {
                        "hit_indicator": {
                            "lower": 0.5,
                            "upper": 0.5,
                            "affected_replicate_fraction": 0.0,
                            "method": "component_specific_causal_worst_case",
                        }
                    },
                    "fragment": {
                        "degraded": {
                            "lower": 0.3,
                            "upper": 0.305,
                            "affected_replicate_fraction": 0.25,
                            "method": "component_specific_causal_worst_case",
                        },
                        "failed": {
                            "lower": 0.1,
                            "upper": 0.35,
                            "affected_replicate_fraction": 0.25,
                            "method": "component_specific_causal_worst_case",
                        },
                    },
                }
            }
        },
    }

    extracted = extract_component_hurdle_targets(result, domain_epsilon=0.01)
    component = extracted["components"]["component-001"]

    assert component["p_hit"]["mask"] is True
    assert component["p_hit"]["identification_interval"]["width"] == 0.0
    assert component["state_probabilities"]["p_degraded"]["mask"] is True
    assert component["state_probabilities"]["p_failed"]["mask"] is False
    assert extracted["source"]["epsilon_complete"] is False
    assert extracted["source"]["probability_target_point_supervision_fraction"] == pytest.approx(
        2.0 / 3.0
    )
    assert extracted["source"]["censoring_supervision"] == (
        "partial_point_and_identification_interval"
    )


def test_missing_fragment_state_is_an_explicit_masked_target() -> None:
    result = _artificial_result([0.0, 1.0, 0.0, 1.0])
    del result["components"]["component-001"]["damage_states"]["fragment"][
        "failed"
    ]

    extracted = extract_component_hurdle_targets(result)
    failed = extracted["components"]["component-001"]["state_probabilities"][
        "p_failed"
    ]

    assert failed["value"] is None
    assert failed["mask"] is False
    assert failed["status"] == "unavailable"
    assert failed["identification_interval"]["mask"] is False
    assert failed["identification_interval"]["width"] == 1.0
    assert extracted["source"]["epsilon_complete"] is False


def test_real_engine_result_exposes_true_hit_conditioned_positive_targets(
    target_data,
    case_data,
    registry,
) -> None:
    engine = DamageEngine(target_data, registry)
    case = deepcopy(case_data)
    case["analysis"]["replicates"] = 4
    case["analysis"]["adaptive_ci_half_width"] = None
    result = engine.run(case).to_dict()

    extracted = extract_component_hurdle_targets(result)

    assert extracted["source"]["replicate_count"] == 4
    assert set(extracted["components"]) == set(result["components"])
    available = 0
    for component_id, target in extracted["components"].items():
        expected = result["components"][component_id]["fragment_metrics"][
            "hit_indicator"
        ]["mean"]
        assert target["p_hit"]["value"] == pytest.approx(expected)
        conditional = target["conditional_positive_targets"][
            "log1p_deposited_energy_j"
        ]
        source = result["components"][component_id]["fragment_metrics"][
            "hit_conditioned_log1p_deposited_energy_j"
        ]
        assert conditional["sample_count"] == source["sample_count"]
        assert conditional["value"] == source["mean"]
        assert conditional["mask"] is (source["sample_count"] >= 2)
        available += int(conditional["mask"])
    assert available >= 1
