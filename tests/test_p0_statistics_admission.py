from __future__ import annotations

from copy import deepcopy
import math

import pytest

from damage_engine.errors import InputValidationError
from damage_engine.online_statistics import (
    OnlineStatisticsStore,
    is_probability_statistic_path,
)
from damage_engine.utils import student_t_critical_95
from tools.evaluate_fast_sampling_proxy import (
    _convergence_report,
    _gate_report,
    _run_teacher,
    _teacher_metrics,
    proxy_domain_coverage,
)
from tools.evaluate_multicondition_penetration import (
    evaluate_condition_gates,
    materialize_condition,
    validate_replicate_budgets,
)


def test_online_statistics_rejects_shape_drift_atomically() -> None:
    store = OnlineStatisticsStore()
    store.observe_replicate({"a": 1.0})

    with pytest.raises(InputValidationError, match="paths changed"):
        store.observe_replicate({"a": 9.0, "b": 2.0})

    assert store.replicate_count == 1
    assert store.statistics(("a",))["sample_count"] == 1
    assert store.statistics(("a",))["mean"] == 1.0
    with pytest.raises(InputValidationError, match="was not observed"):
        store.statistics(("b",))


def test_online_statistics_rejects_nonfinite_replicate_atomically() -> None:
    store = OnlineStatisticsStore()
    store.observe_replicate({"a": 1.0, "b": 2.0})

    with pytest.raises(InputValidationError, match="non-finite"):
        store.observe_replicate({"a": 9.0, "b": math.nan})

    assert store.replicate_count == 1
    assert store.statistics(("a",))["mean"] == 1.0


@pytest.mark.parametrize(
    "name",
    [
        "hit_indicator",
        "damage_positive_indicator",
        "saturated_damage_indicator",
        "penetration_affected_replicate",
        "failure_probability",
        "integrity",
        "blast_failure_realized",
        "blast_failure_sampling_applied",
        "pre_pulse_shielding_survival",
        "same_pulse_failure_feedback_applied",
    ],
)
def test_indicator_and_probability_metrics_use_bounded_intervals(name: str) -> None:
    assert is_probability_statistic_path(("component", name)) is True


def test_student_t_critical_value_has_no_df30_to_df31_discontinuity() -> None:
    at_30 = student_t_critical_95(30)
    at_31 = student_t_critical_95(31)
    at_1000 = student_t_critical_95(1000)

    assert at_30 > at_31 > at_1000 > 1.959
    assert at_30 - at_31 < 0.004


def _minimal_teacher_result(*, replicates: int, support_rate: float, complete: bool = True):
    return {
        "case_id": "case",
        "replicate_count": replicates,
        "scientific_completion_status": (
            "complete_model_domain" if complete else "partial_domain_censored"
        ),
        "penetration_domain_coverage": {
            "affected_replicate_fraction": 0.0 if complete else 0.25,
        },
        "summary": {
            "total_component_hits": {"mean": support_rate},
            "total_component_penetrations": {"mean": 0.0},
        },
        "components": {
            "c": {
                "fragment_metrics": {
                    "hits": {"mean": support_rate},
                    "penetrations": {"mean": 0.0},
                    "deposited_energy_j": {"mean": 0.0},
                    "hit_indicator": {"mean": support_rate},
                }
            }
        },
        "damage_tree": {"outputs": {"mission_kill": {"mean": 0.0}}},
    }


def test_teacher_support_label_targets_rate_not_any_event_over_replicates() -> None:
    low_r = _teacher_metrics(
        _minimal_teacher_result(replicates=4, support_rate=0.10)
    )
    high_r = _teacher_metrics(
        _minimal_teacher_result(replicates=64, support_rate=0.10)
    )

    assert low_r["components"]["c"]["support_rate"] == 0.10
    assert low_r["components"]["c"]["positive_support"] == 1
    assert low_r["any_component_support"] == high_r["any_component_support"] == 1
    assert low_r["support_estimand"] == "component_hit_probability_at_least_threshold"


def test_teacher_metrics_expose_domain_censoring_for_admission_filtering() -> None:
    metrics = _teacher_metrics(
        _minimal_teacher_result(replicates=4, support_rate=0.10, complete=False)
    )
    assert metrics["domain_complete"] is False
    assert metrics["scientific_completion_status"] == "partial_domain_censored"
    assert metrics["penetration_affected_replicate_fraction"] == 0.25


def test_teacher_runner_marks_successful_but_domain_censored_result_nonadmissible() -> None:
    result = _minimal_teacher_result(
        replicates=4, support_rate=0.10, complete=False
    )

    class _Result:
        def to_dict(self):
            return deepcopy(result)

    class _Engine:
        def run(self, case, *, fail_closed):
            assert fail_closed is True
            assert case["analysis"]["replicates"] == 4
            return _Result()

    rows, full, _ = _run_teacher(
        _Engine(),
        {
            "candidate": {
                "case_id": "case",
                "analysis": {},
                "munition": {"profile_id": "p"},
                "metadata": {
                    "contact_manifold_sampling": {"approach_id": "a"}
                },
            }
        },
        ["candidate"],
        replicates=4,
    )
    assert rows[0]["status"] == "domain_censored"
    assert rows[0]["domain_complete"] is False
    assert full[0]["candidate_id"] == "candidate"


def test_proxy_domain_mass_is_a_hard_gate() -> None:
    rows = [
        {
            "represented_fragment_count": 10.0,
            "out_of_domain_packet_weight": 0.5,
        }
    ]
    domain = proxy_domain_coverage(rows)
    assert domain["out_of_domain_packet_weight_fraction"] == pytest.approx(0.05)

    accuracy = {
        "case_any_hit_auc": 1.0,
        "case_any_hit_auc_ci95": [0.9, 1.0],
        "case_hit_spearman": 1.0,
        "case_hit_spearman_ci95": [0.9, 1.0],
        "case_energy_spearman": 1.0,
        "case_energy_spearman_ci95": [0.9, 1.0],
        "component_support_auc": 1.0,
        "component_support_auc_ci95": [0.9, 1.0],
        "component_support_recall": 1.0,
        "component_support_recall_ci95": [0.95, 1.0],
        "top20_energy_overlap": 1.0,
        "top20_energy_overlap_ci95": [1.0, 1.0],
    }
    convergence = {
        "sampling_score_spearman": 1.0,
        "aggregate_component_exposure_relative_l1": 0.0,
    }
    report = _gate_report(accuracy, convergence, 100.0, domain)
    assert report["checks"]["proxy_domain_complete"] is False
    assert report["overall_pass"] is False


def test_proxy_convergence_names_the_two_resolutions_being_compared() -> None:
    def row(score: float):
        return {
            "sampling_score": score,
            "component_metrics": {"c": {"expected_hits": score}},
        }

    lower = {str(index): row(float(index)) for index in range(1, 4)}
    higher = {str(index): row(float(index) + 0.1) for index in range(1, 4)}
    report = _convergence_report(
        lower,
        higher,
        lower_resolution=4,
        higher_resolution=8,
    )
    assert report["comparison"] == "azimuths_per_ring_4_vs_8"
    assert report["higher_resolution_is_reference"] is True


def test_holdout_materialization_requires_independent_holdout_seed(case_data) -> None:
    condition = {
        "id": "h",
        "role": "holdout",
        "profile_id": "medium_front",
        "latent": [1 / 12, 0.5, 0.5, 0.5, 0.25, 0.5],
        "confirmation_seed": 29,
    }
    manifold = {
        "variables": [
            {"name": "approach_azimuth_rad", "minimum": -math.pi, "maximum": math.pi},
            {"name": "approach_elevation_rad", "minimum": -0.5, "maximum": 0.5},
            {"name": "impact_u", "minimum": 0.0, "maximum": 1.0},
            {"name": "impact_v", "minimum": 0.0, "maximum": 1.0},
            {"name": "roll_rad", "minimum": -math.pi, "maximum": math.pi},
            {"name": "standoff_m", "minimum": 0.0, "maximum": 1.0},
        ]
    }

    with pytest.raises(ValueError, match="holdout_seed"):
        materialize_condition(
            deepcopy(case_data), manifold, condition, phase="holdout"
        )


def test_phase_replicate_budgets_enforce_predeclared_minima() -> None:
    assert validate_replicate_budgets(
        32, 64, 128, mode="diagnostic"
    ) == (32, 64, 128)
    with pytest.raises(ValueError, match="selection"):
        validate_replicate_budgets(31, 64, 128, mode="diagnostic")
    with pytest.raises(ValueError, match="confirmation"):
        validate_replicate_budgets(32, 63, 128, mode="diagnostic")
    with pytest.raises(ValueError, match="holdout"):
        validate_replicate_budgets(32, 64, 127, mode="diagnostic")


def test_zero_observed_domain_events_need_enough_replicates_for_one_percent_gate() -> None:
    metrics = {
        "armor_penetration_rate": 0.5,
        "component_penetration_rate": 0.5,
        "damage_positive_given_hit": 0.5,
        "saturated_damage_fraction": 0.1,
        "ordinal_projection_max_correction": 0.0,
        "energy_closure_max_abs_j": 0.0,
        "expected_hit_component_observations": 40.0,
        "penetration_affected_replicate_fraction": 0.0,
        "scientific_completion_status": "complete_model_domain",
    }
    gates = {
        "armor_penetration_rate_range": [0.05, 0.95],
        "component_penetration_rate_range": [0.05, 0.95],
        "damage_positive_given_hit_range": [0.05, 0.95],
        "saturated_damage_fraction_max": 0.5,
        "ordinal_projection_max_correction_max": 1.0e-12,
        "energy_closure_max_abs_j": 1.0e-8,
        "expected_hit_component_observations_min": 30.0,
        "penetration_affected_replicate_upper95_max": 0.01,
    }

    at_32 = evaluate_condition_gates(metrics, gates, replicate_count=32)
    at_299 = evaluate_condition_gates(metrics, gates, replicate_count=299)
    assert at_32["passed"] is False
    assert at_32["penetration_affected_replicate_upper95"] > 0.08
    assert at_299["passed"] is True
    assert at_299["penetration_affected_replicate_upper95"] < 0.01
