"""Integrity checks for the output/label/tree distribution-gap artifact."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.utils import content_hash, file_hash


HERE = Path(__file__).resolve().parent


def main() -> int:
    metrics_path = HERE / "output_gap_metrics.json"
    summary_path = HERE / "SUMMARY.md"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    report_hash = metrics.pop("report_hash")
    assert content_hash(metrics) == report_hash
    scope = metrics["scope"]
    assert scope["sealed_test_trace_json_lines_decoded"] == 0
    assert (scope["train_case_count"], scope["validation_case_count"]) == (2000, 300)
    trace = metrics["trace_contracts"]
    for field in (
        "blast_nonzero_count",
        "combined_fragment_mismatch_count",
        "mechanism_state_out_of_bounds_count",
        "failed_exceeds_degraded_violation_count",
        "tree_output_out_of_bounds_count",
        "tree_output_mismatch_count",
    ):
        assert trace[field] == 0, field
    label = metrics["label_contracts"]
    assert label["p_hit_success_trial_reconstruction_maximum_absolute_error"] < 1.0e-12
    assert label["p_degraded_trace_aggregation_maximum_absolute_error"] < 1.0e-12
    assert label["p_failed_trace_aggregation_maximum_absolute_error"] < 1.0e-12
    assert label["probability_nesting"]["p_degraded_exceeds_p_hit_count"] == 0
    assert label["probability_nesting"]["p_failed_exceeds_p_degraded_count"] == 0
    for row in label["interval_contracts"].values():
        assert row["ci_contains_point_violation_count"] == 0
        assert row["identification_contains_point_violation_count"] == 0
        assert row["mask_disagrees_with_width_le_epsilon_count"] == 0
    distributions = metrics["actual_label_distributions"]
    p_hit_fixed = distributions["p_hit"]["fixed_67_component_identified_case_mean"]
    assert p_hit_fixed["identified_validation_minus_train_mean_interval"][0] > 0.0
    assert p_hit_fixed["bootstrap_95_outer_envelope"][0] < 0.0
    assert p_hit_fixed["bootstrap_95_outer_envelope"][1] > 0.0
    p_hit_complete = distributions["p_hit"]["complete_67_component_case_analysis"]
    assert (p_hit_complete["train_case_count"], p_hit_complete["validation_case_count"]) == (
        1955,
        283,
    )
    assert "missingness_sensitive_case_mean_analysis" in distributions["p_hit"]
    assert "case_mean_validation_minus_train_bootstrap" not in metrics
    summary = summary_path.read_text(encoding="utf-8")
    for fragment in (
        "3,430,400",
        "5,145,600",
        "153,361",
        "0.091729",
        "0.000740",
        "1,955/283",
        "删失敏感性诊断",
        "0.042257",
        "结果解码数为 0",
    ):
        assert fragment in summary, fragment
    print(
        json.dumps(
            {
                "status": "pass",
                "metrics_sha256": file_hash(metrics_path),
                "summary_sha256": file_hash(summary_path),
                "sealed_test_trace_json_lines_decoded": 0,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
