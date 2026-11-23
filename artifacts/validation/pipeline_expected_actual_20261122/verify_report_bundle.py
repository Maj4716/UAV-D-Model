"""Deterministic integrity checks for the stage-wise assessment bundle."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


OUT = Path(__file__).resolve().parent


def load(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()


def close(left: float, right: float, tolerance: float = 1.0e-9) -> None:
    assert abs(left - right) <= tolerance, (left, right)


def main() -> int:
    config = load("analysis_config.json")
    metrics = load("metrics.json")
    manifest = load("evidence_manifest.json")
    report = (OUT / "REPORT.md").read_text(encoding="utf-8")

    assert config["analysis_id"] == "pipeline-expected-actual-20261122-v2-stagewise"
    assert config["analysis_id"] == metrics["analysis_id"] == manifest["analysis_id"]
    assert config["populations"]["formal_split_counts"] == {
        "train": 2000,
        "validation": 300,
        "test": 300,
    }
    assert config["populations"]["development_outputs"]["teacher_replicate_count"] == 25600
    assert config["sealing"]["test_result_json_lines_decoded"] == 0
    assert config["sealing"]["test_metrics_computed"] is False

    design = metrics["stage_00_03_design_contact_archive"]
    assert (design["candidate_count"], design["selected_count"]) == (4301, 2600)
    close(design["selected_region_tv"], 0.20153846153846153)
    assert design["archive_assertions"]["failed"] == 0

    mechanism = metrics["stage_04_07_aggregate_mechanism"]
    domain = mechanism["domain_censoring"]
    assert domain["proxy"]["affected_case_count"] == 1094
    assert domain["teacher"]["affected_case_count"] == 213
    assert domain["teacher_and_proxy_complete_case_count"] == 1120
    joint = mechanism["joint_complete_stages"]
    assert joint["source_fragments"]["actual_minus_expected_mean"] == 0.0
    close(joint["armor_hits"]["actual_minus_expected_mean"], 1.6192523602832463)
    close(
        joint["component_deposited_energy_j"]["actual_minus_expected_mean"],
        25342.469851294532,
        tolerance=1.0e-6,
    )

    micro = metrics["stage_04_09_micro_audit"]
    assert micro["selection"]["selected_case_count"] == 30
    assert micro["fragment_source_contract"]["fragment_trace_count"] == 40800
    assert micro["component_reachability"]["fragment_component_pair_count"] == 2733600
    assert micro["axis_diagnosis_counts"] == {"front_axis_not_declared_backward": 30}
    assert micro["archive_equivalence"]["summary_max_abs_error"] == 0.0
    assert micro["evidence_boundaries"]["test_results_used"] is False

    outputs = metrics["stage_08_11_teacher_labels_outputs"]
    assert outputs["trace_contracts"]["tree_output_mismatch_count"] == 0
    p_hit = outputs["actual_label_distributions"]["p_hit"]
    fixed = p_hit["fixed_67_component_identified_case_mean"]
    assert fixed["identified_validation_minus_train_mean_interval"][0] > 0.0
    assert fixed["bootstrap_95_outer_envelope"][0] < 0.0
    assert fixed["bootstrap_95_outer_envelope"][1] > 0.0
    complete = p_hit["complete_67_component_case_analysis"]
    assert (complete["train_case_count"], complete["validation_case_count"]) == (1955, 283)

    proxy_tree = metrics["stage_09_10_proxy_states_and_tree"]
    close(
        proxy_tree["joint_complete_probability_heads"]["p_hit"][
            "actual_minus_expected_mean"
        ],
        0.0413551862656257,
    )
    assert proxy_tree["trace_aggregation_consistency"][
        "trace_record_count_mismatch_case_count"
    ] == 0
    for row in proxy_tree["joint_complete_tree_outputs"].values():
        assert row["actual_minus_expected_mean"] > 0.0

    promotion = metrics["stage_12_13_proxy_admission_and_sealing"][
        "r8_pilot_promotion_gate"
    ]
    assert promotion["gate_passed"] is False
    assert promotion["wilson_upper"] > promotion["gate_upper_max"]
    assert metrics["external_physical_distribution_gap"]["status"] == "not_estimable"
    assert metrics["mechanism_verification"]["status"] == "pass"

    required_report_fragments = [
        "0.201538",
        "47.565%",
        "9.261%",
        "1,120",
        "25,342 J",
        "40,800",
        "2,733,600",
        "front_axis_not_declared_backward",
        "0.003015",
        "87.279%",
        "0.000740",
        "0.041355",
        "8.599%",
        "64/67",
        "62/67",
        "不可估计：缺少外部参照",
        "解码 0",
    ]
    missing = [value for value in required_report_fragments if value not in report]
    assert not missing, f"report missing values: {missing}"

    assert manifest["sealed_test_result_json_lines_decoded"] == 0
    assert manifest["evidence_count"] == len(manifest["files"])
    for row in manifest["files"]:
        path = Path(row["path"])
        assert path.is_file(), path
        assert path.stat().st_size == row["size_bytes"], path
        assert digest(path) == row["sha256"], path

    print(
        json.dumps(
            {
                "analysis_id": config["analysis_id"],
                "evidence_files_verified": len(manifest["files"]),
                "report_key_values_verified": len(required_report_fragments),
                "sealed_test_result_json_lines_decoded": 0,
                "status": "pass",
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
