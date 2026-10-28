"""Machine-check the final sparsity-causality report against frozen evidence.

The validator is intentionally read-only except for its own JSON result.  It
does not rerun the stochastic diagnostic; it verifies that the report cites the
validated result, current engine identity, contact-manifold smoke evidence,
zero-anchor bound, provisional-tree admission status, and final JUnit suite.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPORT = ROOT / "49_毁伤分布稀疏性根因裁决与数据生成整改研究报告_2026-10-23.md"
CONTROL_VALIDATION = HERE / "controlled_factorial_validation.json"
CONTROL_RESULT = HERE / "controlled_factorial_diagnostic.json"
ZERO_ANCHOR = HERE / "zero_anchor_balance_feasibility.json"
POSE_SUMMARY = ROOT / "review_evidence/contact_pose_sampling_20261023/verification_summary.json"
TREE_AUDIT = HERE / "damage_tree_validity_and_software_prototype_audit.json"
JUNIT = HERE / "pytest_full_final_20261023.xml"
OUTPUT = HERE / "final_research_report_validation.json"
sys.path.insert(0, str(ROOT))


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def close(actual: float, expected: float, *, atol: float = 1e-12) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=atol)


checks: dict[str, dict[str, Any]] = {}


def record(name: str, passed: bool, actual: Any, expected: Any) -> None:
    checks[name] = {
        "pass": bool(passed),
        "actual": actual,
        "expected": expected,
    }


required = [
    REPORT,
    CONTROL_VALIDATION,
    CONTROL_RESULT,
    ZERO_ANCHOR,
    POSE_SUMMARY,
    TREE_AUDIT,
    JUNIT,
]
missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
record("required_files_exist", not missing, missing, [])
if missing:
    result = {
        "schema_version": "1.0.0",
        "overall_pass": False,
        "checks": checks,
        "claim_boundary": "validation failed before evidence could be cross-checked",
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(1)


report = REPORT.read_text(encoding="utf-8")
control_validation = load_json(CONTROL_VALIDATION)
control = load_json(CONTROL_RESULT)
zero_anchor = load_json(ZERO_ANCHOR)
pose = load_json(POSE_SUMMARY)
tree = load_json(TREE_AUDIT)

from damage_engine.dataset import ENGINE_CODE_HASH  # noqa: E402


record(
    "report_has_no_placeholder",
    "FINAL_FULL_TEST_RESULT" not in report,
    "FINAL_FULL_TEST_RESULT" in report,
    False,
)
record(
    "controlled_diagnostic_validation_passed",
    control_validation.get("overall_pass") is True,
    control_validation.get("overall_pass"),
    True,
)

evidence_hashes = {
    "current": ENGINE_CODE_HASH,
    "controlled_diagnostic": control["software_identity"]["engine_code_hash"],
    "contact_pose_smoke": pose["engine_code_hash"],
    "contact_pose_manifest": pose["manifest_engine_code_hash"],
}
record(
    "engine_identity_is_current_and_consistent",
    len(set(evidence_hashes.values())) == 1 and ENGINE_CODE_HASH in report,
    evidence_hashes,
    {"all_equal": True, "hash_appears_in_report": True},
)

design = control["design"]
record(
    "controlled_design_counts",
    design["attempted_case_condition_count"] == 324
    and design["accepted_case_condition_count"] == 324
    and design["unique_candidate_count"] == 162
    and control["runtime"]["simulated_replicates"] == 2592,
    {
        "attempted": design["attempted_case_condition_count"],
        "accepted": design["accepted_case_condition_count"],
        "unique_candidates": design["unique_candidate_count"],
        "replicates": control["runtime"]["simulated_replicates"],
    },
    {"attempted": 324, "accepted": 324, "unique_candidates": 162, "replicates": 2592},
)
contract = control["contract_checks"]
record(
    "controlled_contact_and_seed_contract",
    contract["all_accepted_use_first_contact_model"] is True
    and contract["all_accepted_target_facing"] is True
    and contract["all_accepted_body_forward_aligned"] is True
    and contract["unique_aleatory_seed_count"] == 162
    and contract["each_seed_maps_to_one_candidate_key_and_two_axis_conditions"] is True,
    {
        "first_contact": contract["all_accepted_use_first_contact_model"],
        "target_facing": contract["all_accepted_target_facing"],
        "body_forward_aligned": contract["all_accepted_body_forward_aligned"],
        "unique_seeds": contract["unique_aleatory_seed_count"],
        "paired_axes": contract["each_seed_maps_to_one_candidate_key_and_two_axis_conditions"],
    },
    {
        "first_contact": True,
        "target_facing": True,
        "body_forward_aligned": True,
        "unique_seeds": 162,
        "paired_axes": True,
    },
)

r8 = control["aggregates"]["by_axis_and_r"]["8"]
categories = ("K", "M", "F", "C")
expected_rates = {
    "archived_minus_body_x": {"K": 14 / 162, "M": 18 / 162, "F": 24 / 162, "C": 12 / 162},
    "counterfactual_plus_body_x": {"K": 117 / 162, "M": 1.0, "F": 134 / 162, "C": 161 / 162},
}
actual_rates = {
    axis: {cat: r8[axis]["category_ever_positive_rate"][cat] for cat in categories}
    for axis in expected_rates
}
record(
    "r8_support_rates",
    all(close(actual_rates[axis][cat], expected_rates[axis][cat]) for axis in expected_rates for cat in categories),
    actual_rates,
    expected_rates,
)

paired = control["paired_axis_effects"]["levels"]["8"]["category"]
expected_differences = {"K": 103 / 162, "M": 144 / 162, "F": 110 / 162, "C": 149 / 162}
actual_differences = {
    cat: paired[cat]["ever_positive_rate_difference_plus_minus_minus"]["mean"]
    for cat in categories
}
record(
    "paired_axis_risk_differences",
    all(close(actual_differences[cat], expected_differences[cat]) for cat in categories),
    actual_differences,
    expected_differences,
)

zero_r8 = zero_anchor["controlled_pilot_by_axis_and_r"]["8"]["archived_minus_body_x"]
expected_tv = {"K": 47 / 81, "M": 5 / 9, "F": 14 / 27, "C": 16 / 27}
actual_tv = {
    cat: zero_r8[cat]["minimum_total_variation_distance_from_uniform_lower_bound"]
    for cat in categories
}
record(
    "zero_anchor_total_variation_lower_bounds",
    all(close(actual_tv[cat], expected_tv[cat]) for cat in categories)
    and all(zero_r8[cat]["uniform_three_class_support_necessary_condition"] is False for cat in categories),
    actual_tv,
    expected_tv,
)

record(
    "contact_pose_smoke_contract",
    pose["all_s_contact_admitted"] is True
    and pose["approach_counts"] == {
        "bottom_up_pos_z": 3,
        "front_to_rear_neg_y": 3,
        "left_to_right_pos_x": 3,
        "rear_to_front_pos_y": 3,
        "right_to_left_neg_x": 3,
        "top_down_neg_z": 3,
    }
    and pose["archive_model_input_shape"] == [18, 15]
    and pose["minimum_body_x_velocity_cosine"] == 1.0
    and pose["rejected_noncontact_count"] == 4
    and pose["rejected_teacher_executed_values"] == [False]
    and pose["targeted_test_count"] == 47,
    {
        "all_s_contact_admitted": pose["all_s_contact_admitted"],
        "approach_counts": pose["approach_counts"],
        "archive_model_input_shape": pose["archive_model_input_shape"],
        "minimum_body_x_velocity_cosine": pose["minimum_body_x_velocity_cosine"],
        "rejected_noncontact_count": pose["rejected_noncontact_count"],
        "rejected_teacher_executed_values": pose["rejected_teacher_executed_values"],
        "targeted_test_count": pose["targeted_test_count"],
    },
    {
        "six_approaches_with_three_cases_each": True,
        "archive_model_input_shape": [18, 15],
        "minimum_body_x_velocity_cosine": 1.0,
        "rejected_noncontact_count": 4,
        "rejected_teacher_executed_values": [False],
        "targeted_test_count": 47,
    },
)

tree_verdict = tree["verdict"]
record(
    "provisional_tree_not_admitted",
    tree_verdict["research_training_admission"] is False
    and tree_verdict["physical_calibration"] == "not_performed"
    and tree_verdict["software_prototype"] == "software_properties_passed_not_admitted",
    tree_verdict,
    {
        "research_training_admission": False,
        "physical_calibration": "not_performed",
        "software_prototype": "software_properties_passed_not_admitted",
    },
)

suite_root = ET.parse(JUNIT).getroot()
suites = suite_root.findall("testsuite") if suite_root.tag == "testsuites" else [suite_root]
junit_totals = {
    key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
    for key in ("tests", "failures", "errors", "skipped")
}
record(
    "final_junit_suite",
    junit_totals == {"tests": 892, "failures": 0, "errors": 0, "skipped": 0},
    junit_totals,
    {"tests": 892, "failures": 0, "errors": 0, "skipped": 0},
)

required_report_fragments = [
    "research_admission=false",
    "47 passed",
    "892 passed",
    "324/324",
    "2592",
    "8.6%/11.1%/14.8%/7.4%",
    "72.2%/100.0%/82.7%/99.4%",
    "0.580/0.556/0.519/0.593",
    "最终稀疏性的占比目前不可辨识",
    "尚不能把历史 `-body-X` 判为人为错误",
]
missing_fragments = [fragment for fragment in required_report_fragments if fragment not in report]
record("report_contains_frozen_results", not missing_fragments, missing_fragments, [])

overall_pass = all(item["pass"] for item in checks.values())
result = {
    "schema_version": "1.0.0",
    "overall_pass": overall_pass,
    "report": str(REPORT.relative_to(ROOT)),
    "report_sha256": "sha256:" + hashlib.sha256(REPORT.read_bytes()).hexdigest(),
    "current_engine_code_hash": ENGINE_CODE_HASH,
    "checks": checks,
    "claim_boundary": (
        "This verifies internal consistency and frozen software evidence only. "
        "It does not establish physical source-axis correctness, parameter calibration, "
        "operational prevalence, or formal research-training admission."
    ),
}
OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"overall_pass": overall_pass, "failed": [k for k, v in checks.items() if not v["pass"]]}, ensure_ascii=False))
raise SystemExit(0 if overall_pass else 1)
