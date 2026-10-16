import numpy as np
import pytest

from damage_engine.errors import InputValidationError
from damage_engine.evaluation_planning import plan_reference_evaluation
from damage_engine.replicate_convergence import replicate_convergence_report
from tools.run_replicate_convergence_study import (
    DEFAULT_LEVELS,
    DEFAULT_MINIMUM_CASES,
    _build_case_selection_disclosure,
    _build_parser,
    _validate_plan_configuration,
)


def test_reference_plan_reproduces_report_order_of_magnitude():
    report = plan_reference_evaluation(0.0384)
    events = {row["target_expected_events"]: row for row in report["event_count_plans"]}
    precision = {
        row["target_absolute_half_width"]: row for row in report["precision_plans"]
    }
    assert events[100]["iid_case_count"] == 2605
    assert 1410 <= precision[0.01]["iid_case_count"] <= 1425
    assert 5660 <= precision[0.005]["iid_case_count"] <= 5690


def test_reference_plan_inflates_for_predeclared_effective_fraction():
    report = plan_reference_evaluation(
        0.01, target_event_counts=[50], absolute_half_widths=[0.01],
        effective_sample_fraction=0.5,
    )
    row = report["event_count_plans"][0]
    assert row["iid_case_count"] == 5000
    assert row["design_adjusted_case_count"] == 10000


@pytest.mark.parametrize("value", [0.0, 1.0, float("nan")])
def test_reference_plan_rejects_invalid_prevalence(value):
    with pytest.raises(InputValidationError):
        plan_reference_evaluation(value)


def test_replicate_convergence_uses_nested_fixed_prefixes():
    rng = np.random.default_rng(20261013)
    traces = rng.binomial(1, 0.2, size=(72, 128, 2)).astype(float)
    report = replicate_convergence_report(
        traces,
        levels=(4, 8, 16, 32, 64, 128),
        ci_half_width_tolerance=[0.3, 0.3],
    )
    assert report["schema_version"] == "1.1.0"
    assert report["design"] == "nested_fixed_prefix_numerical_diagnostic"
    assert report["external_preregistration_status"] == (
        "not_established_by_this_report"
    )
    assert report["levels"] == [4, 8, 16, 32, 64, 128]
    assert report["reference_prefix_replicates"] == 128
    assert report["results"][-1]["relative_compute_cost"] == 32.0
    assert report["results"][-1]["ci_critical_value"] == pytest.approx(
        1.978819534, abs=1.0e-7
    )
    assert report["results"][-1]["ci_critical_value_method"].startswith(
        "exact_table_for_df_1_through_31"
    )
    assert report["results"][-1]["absolute_error_to_final_p95_by_output"] == [
        0.0,
        0.0,
    ]
    assert report["results"][-1]["absolute_error_to_final_p99_by_output"] == [
        0.0,
        0.0,
    ]
    assert len(report["results"][-1]["fraction_exceeding_tolerance_by_output"]) == 2
    assert "mean_shift_from_previous_max_by_output" not in report["results"][0]
    assert report["comparison_definition"]["absolute_error_quantiles"] == [0.95, 0.99]


def test_replicate_convergence_reports_threshold_neighborhood_flips():
    traces = np.asarray(
        [
            [[1], [1], [1], [1], [0], [0], [0], [0]],
            [[1], [1], [1], [0], [0], [0], [0], [0]],
            [[0], [0], [1], [1], [1], [1], [1], [0]],
            [[0], [0], [0], [0], [0], [0], [0], [0]],
        ],
        dtype=float,
    )
    report = replicate_convergence_report(
        traces,
        levels=(2, 4, 8),
        decision_threshold=0.5,
        threshold_neighborhood_half_width=0.2,
    )
    first = report["results"][0]
    expected_error = np.abs(traces[:, :2, 0].mean(axis=1) - traces[:, :, 0].mean(axis=1))
    assert first["absolute_error_to_final_p95_by_output"] == pytest.approx(
        [np.quantile(expected_error, 0.95)]
    )
    assert first["absolute_error_to_final_p99_by_output"] == pytest.approx(
        [np.quantile(expected_error, 0.99)]
    )
    assert first["threshold_label_flip_rate_by_output"] == pytest.approx([0.5])
    assert first["threshold_neighborhood_case_count_by_output"] == [3]
    assert first["threshold_neighborhood_flip_rate_by_output"] == pytest.approx(
        [2.0 / 3.0]
    )
    assert report["results"][-1]["threshold_neighborhood_flip_rate_by_output"] == [
        0.0
    ]


def test_replicate_convergence_defaults_to_r128_and_72_case_execution_floor():
    parser = _build_parser()
    arguments = parser.parse_args(
        [
            "--target", "target.json", "--physics", "physics.json",
            "--cases", "cases.json", "--output", "report.json",
        ]
    )
    assert tuple(arguments.levels) == DEFAULT_LEVELS
    assert arguments.replicates == 128
    assert arguments.max_cases == DEFAULT_MINIMUM_CASES == 72
    assert arguments.minimum_cases == 72


def test_convergence_cli_case_budget_requires_explicit_lower_minimum():
    with pytest.raises(InputValidationError, match="requested minimum"):
        _validate_plan_configuration(
            max_cases=24,
            minimum_cases=72,
            replicates=128,
            levels=DEFAULT_LEVELS,
        )
    assert _validate_plan_configuration(
        max_cases=24,
        minimum_cases=24,
        replicates=128,
        levels=DEFAULT_LEVELS,
    ) == DEFAULT_LEVELS


def test_convergence_selection_disclosure_does_not_claim_formal_design():
    disclosure = _build_case_selection_disclosure(
        cases_mode="sampling_run_reference_case_stream",
        sampling_role="reference_train",
        sampling_stratum=None,
        minimum_cases=72,
        selected_case_count=72,
    )

    assert disclosure["new_simulation_outcomes_read_for_selection"] is False
    assert disclosure["upstream_outcome_blindness"] == "not_verified_by_this_report"
    assert disclosure["external_preregistration"] == "not_verified_by_this_report"
    assert disclosure["required_stratum_quotas"] == (
        "not_supplied_or_verified_by_this_report"
    )
    assert disclosure["formal_design_admission"]["status"] == (
        "not_established_by_this_report"
    )
    assert disclosure["requested_minimum_case_count_met"] is True
    assert "outcome_blind" not in disclosure
    assert "pre_registered" not in disclosure


def test_convergence_selection_disclosure_marks_low_requested_floor_exploratory():
    disclosure = _build_case_selection_disclosure(
        cases_mode="json_or_jsonl",
        sampling_role=None,
        sampling_stratum=None,
        minimum_cases=24,
        selected_case_count=24,
    )

    assert disclosure["requested_minimum_below_recommendation"] is True
    assert disclosure["recommended_minimum_case_count_met"] is False
    assert disclosure["formal_design_admission"]["status"] == (
        "not_established_by_this_report"
    )


def test_replicate_convergence_rejects_outcome_dependent_or_missing_prefix():
    with pytest.raises(InputValidationError):
        replicate_convergence_report(np.ones((2, 4, 1)), levels=(4, 8))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"decision_threshold": [-0.1]},
        {"decision_threshold": [0.5, 0.5]},
        {"threshold_neighborhood_half_width": 0.0},
    ],
)
def test_replicate_convergence_rejects_invalid_threshold_contract(kwargs):
    with pytest.raises(InputValidationError):
        replicate_convergence_report(
            np.ones((2, 4, 1)), levels=(2, 4), **kwargs
        )
