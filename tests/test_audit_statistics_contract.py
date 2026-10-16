from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from damage_engine.audit_statistics import (
    stable_effective_sample_size,
    standardized_mean_difference,
    validate_design_weight_contract,
)
from damage_engine.errors import InputValidationError


def test_smd_reports_same_constant_as_explicit_estimable_zero() -> None:
    result = standardized_mean_difference(np.asarray([4.0, 4.0]), np.asarray([4.0, 4.0]))

    assert result["status"] == "same_constant"
    assert result["raw_mean_difference"] == 0.0
    assert result["pooled_standard_deviation"] == 0.0
    assert result["pooled_standard_deviation_status"] == "finite_zero"
    assert result["absolute_standardized_mean_difference"] == 0.0


def test_smd_does_not_encode_different_constants_as_zero() -> None:
    result = standardized_mean_difference(np.asarray([1.0, 1.0]), np.asarray([3.0, 3.0]))

    assert result["status"] == "not_estimable_zero_scale_different_constants"
    assert result["raw_mean_difference"] == -2.0
    assert result["pooled_standard_deviation"] == 0.0
    assert result["absolute_standardized_mean_difference"] is None


def test_smd_reports_total_finite_and_empty_group_denominators() -> None:
    result = standardized_mean_difference(np.asarray([np.nan, np.inf]), np.asarray([2.0]))

    assert result["status"] == "not_estimable_empty_group"
    assert result["left_total_count"] == 2
    assert result["left_finite_count"] == 0
    assert result["left_nonfinite_count"] == 2
    assert result["left_missing_count"] == 1
    assert result["left_invalid_count"] == 1
    assert result["right_total_count"] == 1
    assert result["right_finite_count"] == 1
    assert result["absolute_standardized_mean_difference"] is None


def test_smd_uses_stable_scale_for_extreme_finite_values() -> None:
    extreme = np.asarray([1.0e308, -1.0e308])
    result = standardized_mean_difference(extreme, extreme)

    assert result["status"] == "estimated"
    assert result["pooled_standard_deviation"] == pytest.approx(1.0e308)
    assert result["pooled_standard_deviation_status"] == "finite_positive"
    assert result["absolute_standardized_mean_difference"] == 0.0


def test_extreme_different_constants_keep_scaled_raw_difference() -> None:
    result = standardized_mean_difference(
        np.asarray([1.0e308, 1.0e308]), np.asarray([-1.0e308, -1.0e308])
    )

    assert result["status"] == "not_estimable_zero_scale_different_constants"
    assert result["raw_mean_difference"] is None
    assert result["raw_mean_difference_status"] == "nonfinite_unrepresentable"
    assert result["raw_mean_difference_scaled"] == 2.0


def test_stable_ess_is_scale_invariant_and_rejects_bad_shape() -> None:
    assert stable_effective_sample_size(np.asarray([1.0, 2.0])) == pytest.approx(1.8)
    assert stable_effective_sample_size(np.asarray([1.0e300, 2.0e300])) == pytest.approx(1.8)
    assert stable_effective_sample_size(np.asarray([], dtype=np.float64)) == 0.0
    with pytest.raises(InputValidationError, match="one-dimensional"):
        stable_effective_sample_size(np.ones((2, 1), dtype=np.float64))


@pytest.mark.parametrize(
    ("weights", "mask", "probabilities", "message"),
    [
        ([1.0, np.nan], [True, True], [0.5, 0.5], "finite and positive"),
        ([1.0, 0.0], [True, True], [0.5, 0.5], "finite and positive"),
        ([1.0, 2.0], [True, True], [0.5, 0.0], r"lie in \(0, 1\]"),
        ([1.0, 2.0], [True, True], [0.5, 1.1], r"lie in \(0, 1\]"),
        ([1.0, 2.0], [True, True], [0.5], "share the design-weight axis"),
        ([1.0, 2.0], [1, 1], [0.5, 0.5], "one-dimensional bool"),
    ],
)
def test_design_weight_contract_fails_closed_for_eligible_invalid_values(
    weights: list[float],
    mask: list[bool] | list[int],
    probabilities: list[float],
    message: str,
) -> None:
    with pytest.raises(InputValidationError, match=message):
        validate_design_weight_contract(
            np.asarray(weights), np.asarray(mask), np.asarray(probabilities)
        )


def test_design_weight_contract_allows_ineligible_missing_values() -> None:
    weights, mask, probabilities = validate_design_weight_contract(
        np.asarray([1.0, np.nan]),
        np.asarray([True, False]),
        np.asarray([0.5, np.nan]),
    )

    assert mask.tolist() == [True, False]
    assert weights[0] == 1.0
    assert np.isnan(weights[1])
    assert probabilities is not None and np.isnan(probabilities[1])


def test_distribution_split_drift_exposes_constant_and_empty_states() -> None:
    pytest.importorskip("scipy")
    from tools.audit_dataset_distribution import _split_drift

    features = np.asarray([[1.0], [1.0], [3.0], [3.0]], dtype=np.float64)
    mask = np.ones(features.shape, dtype=bool)
    splits = np.asarray(["train", "train", "validation", "validation"])
    report = _split_drift(features, mask, ["constant_by_split"], splits)
    records = report[0]["comparisons"]

    assert records[0]["standardized_mean_difference_status"] == (
        "not_estimable_zero_scale_different_constants"
    )
    assert records[0]["raw_mean_difference"] == -2.0
    assert records[1]["standardized_mean_difference_status"] == "not_estimable_empty_group"
    assert records[1]["denominators"]["right_total_count"] == 0
    json.dumps(report, allow_nan=False)


def test_tool_summaries_keep_extreme_finite_values_strict_json_safe() -> None:
    pytest.importorskip("scipy")
    from tools.audit_dataset_distribution import _summary as distribution_summary
    from tools.audit_loitering_impact_dataset import _summary as loitering_summary

    for summary in (
        distribution_summary(np.asarray([1.0e308, 1.0e308])),
        loitering_summary(np.asarray([1.0e308, 1.0e308])),
    ):
        assert summary["mean"] == 1.0e308
        assert summary["std"] == 0.0
        assert summary["numeric_method"] == "max_abs_scaled_float64"
        assert summary["finite_count"] == 2
        json.dumps(summary, allow_nan=False)


def test_distribution_ci_summary_keeps_empty_availability_json_safe() -> None:
    pytest.importorskip("scipy")
    from tools.audit_dataset_distribution import _ci_stats_by_label

    records = _ci_stats_by_label(
        np.zeros((3, 1, 2), dtype=np.float64),
        np.zeros((3, 1), dtype=bool),
        ["unavailable-task"],
    )
    assert records[0]["status"] == "not_estimable_no_finite_values"
    assert records[0]["fraction_status"] == "not_estimable_no_available_rows"
    assert records[0]["nonzero_fraction"] is None
    json.dumps(records, allow_nan=False)


def test_distribution_correlations_report_constant_inputs_as_not_estimable() -> None:
    pytest.importorskip("scipy")
    from tools.audit_dataset_distribution import _pearson_pair, _spearman_pair

    left = np.asarray([1.0, 1.0, 1.0])
    right = np.asarray([0.0, 1.0, 2.0])
    spearman = _spearman_pair(left, right)
    pearson = _pearson_pair(left, right)
    assert spearman["status"] == "not_estimable_constant_input"
    assert spearman["spearman_rho"] is None
    assert pearson["status"] == "not_estimable_constant_input"
    assert pearson["pearson_r"] is None
    json.dumps({"spearman": spearman, "pearson": pearson}, allow_nan=False)


def test_loitering_split_drift_and_weight_audit_share_contract() -> None:
    from tools.audit_loitering_impact_dataset import _split_drift, _weight_audit

    rows = [
        {"origin_y_m": 1.0, "origin_z_m": 9.0, "impact_speed_mps": 100.0},
        {"origin_y_m": 1.0, "origin_z_m": 9.0, "impact_speed_mps": 100.0},
        {"origin_y_m": 2.0, "origin_z_m": 9.0, "impact_speed_mps": 100.0},
        {"origin_y_m": 2.0, "origin_z_m": 9.0, "impact_speed_mps": 100.0},
    ]
    splits = np.asarray(["train", "train", "validation", "validation"])
    drift = _split_drift(rows, np.asarray(["small_front"] * 4), splits)
    origin_y = drift["continuous"][0]["comparisons"][0]
    origin_z = drift["continuous"][1]["comparisons"][0]
    assert origin_y["standardized_mean_difference_status"] == (
        "not_estimable_zero_scale_different_constants"
    )
    assert origin_z["standardized_mean_difference_status"] == "same_constant"

    weights = _weight_audit(
        {
            "sampling_design_weight_raw": np.asarray([1.0, 2.0, np.nan, 4.0]),
            "sampling_weight_mask": np.asarray([True, True, False, True]),
            "sampling_selection_probability": np.asarray([0.5, 0.25, np.nan, 1.0]),
        },
        np.asarray(["train", "train", "validation", "test"]),
    )
    assert weights["contract_status"] == "validated_fail_closed"
    assert weights["total_count"] == 4
    assert weights["finite_weight_count"] == 3
    assert weights["eligible_count"] == 3
    assert weights["all"]["total_count"] == 4
    assert weights["all"]["eligible_count"] == 3
    assert weights["by_split"]["validation"]["ess_status"] == (
        "not_estimable_no_eligible_rows"
    )
    json.dumps({"drift": drift, "weights": weights}, allow_nan=False)


@pytest.mark.parametrize(
    "arrays",
    [
        {
            "sampling_design_weight_raw": np.asarray([1.0, np.nan]),
            "sampling_weight_mask": np.asarray([True, True]),
        },
        {
            "sampling_design_weight_raw": np.asarray([1.0, 2.0]),
            "sampling_weight_mask": np.asarray([True, True]),
            "sampling_selection_probability": np.asarray([0.5, 0.0]),
        },
    ],
)
def test_loitering_weight_audit_fails_closed(arrays: dict[str, np.ndarray]) -> None:
    from tools.audit_loitering_impact_dataset import _weight_audit

    with pytest.raises(InputValidationError):
        _weight_audit(arrays, np.asarray(["train", "test"]))


def test_loitering_loader_uses_validated_committed_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools import audit_loitering_impact_dataset as audit_tool

    shard_path = tmp_path / "committed-tensors.npz"
    np.savez_compressed(
        shard_path,
        case_ids=np.asarray(["case-1"]),
        split_names=np.asarray(["train"]),
    )
    called: list[Path] = []

    def committed(root: Path):
        called.append(root)
        return {"dataset_id": "test"}, (shard_path,)

    monkeypatch.setattr(audit_tool, "validated_committed_shard_tensor_paths", committed)
    arrays, _metadata, paths = audit_tool._load_tensors(tmp_path)

    assert called == [tmp_path]
    assert arrays["case_ids"].tolist() == ["case-1"]
    assert paths == [shard_path]


def test_audit_tools_do_not_rediscover_part_shards_with_glob() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "tools/audit_dataset_distribution.py",
        "tools/audit_loitering_impact_dataset.py",
        "tools/audit_system_learning_contract.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        assert '.glob("part-' not in source
        assert "validated_committed_shard_tensor_paths" in source
        assert "allow_nan=False" in source
