"""Deterministic integrity checks for the two mechanism-analysis JSON files."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.utils import content_hash


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise AssertionError(f"non-finite value at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite(child, f"{path}[{index}]")


def _verify_hash(report: dict[str, Any]) -> None:
    payload = deepcopy(report)
    observed = payload.pop("report_hash")
    assert content_hash(payload) == observed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--micro", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    aggregate = _load(args.aggregate)
    micro = _load(args.micro)
    _assert_finite(aggregate, "aggregate")
    _assert_finite(micro, "micro")
    _verify_hash(aggregate)
    _verify_hash(micro)

    assert aggregate["source"]["test_result_json_lines_decoded"] == 0
    assert aggregate["source"]["test_metrics_computed"] is False
    assert aggregate["population"]["case_count"] == 2300
    assert aggregate["population"]["split_counts"] == {
        "all_mixed_R8_R32": 2300,
        "train_R8": 2000,
        "validation_R32": 300,
    }
    storage = aggregate["storage_and_observability"]
    assert storage["teacher_domain_complete_case_count"] == 2087
    assert storage["proxy_domain_complete_case_count"] == 1206
    assert storage["teacher_and_proxy_domain_complete_case_count"] == 1120
    expected_scope_counts = {
        "all_mixed_R8_R32": 1120,
        "train_R8": 977,
        "validation_R32": 143,
    }
    for stage in aggregate["stage_comparisons"].values():
        selected = stage["teacher_and_proxy_domain_complete_cases_by_protocol"]
        assert {
            scope: values["paired_case_count"] for scope, values in selected.items()
        } == expected_scope_counts
    assert sum(
        item["teacher_and_proxy_domain_complete_case_count"]
        for item in aggregate["profile_region_strata"].values()
    ) == 1120
    for scope, case_count in expected_scope_counts.items():
        for head in ("p_hit", "p_degraded", "p_failed"):
            values = aggregate[
                "component_probability_heads_teacher_and_proxy_complete67_by_protocol"
            ][scope][head]
            assert values["shape"] == [case_count, 67]
            assert values["identified_case_component_count"] == case_count * 67

    selection = micro["selection"]
    assert selection["cell_count"] == 15
    assert selection["selected_case_count"] == 30
    assert selection["train_result_json_lines_decoded"] == 30
    assert selection["test_result_json_lines_decoded"] == 0
    assert len(set(selection["selected_case_ids"])) == 30
    assert micro["source"]["replicate_count"] == 8
    assert micro["source"]["audit_wrapper_count_values"] == [8]
    assert len(micro["source"]["aleatory_seed_values"]) == 30
    equivalence = micro["archive_equivalence"]
    assert equivalence["summary_max_abs_error"] == 0.0
    assert equivalence["component_max_abs_error"] == 0.0
    assert equivalence["tree_max_abs_error"] == 0.0
    assert equivalence["metric_missingness_mismatch_count"] == 0
    source = micro["fragment_source_contract"]
    assert source["fragment_trace_count"] == source["expected_fragment_trace_count"] == 40800
    assert source["fragment_count_mismatch_case_count"] == 0
    reach = micro["component_reachability"]
    assert sum(item["count"] for item in reach["categories"].values()) == reach[
        "fragment_component_pair_count"
    ]
    assert reach["fragment_component_pair_count"] == 40800 * 67
    for component in reach["by_component"].values():
        assert component["fragment_component_pair_count"] == 40800
        assert sum(item["count"] for item in component["categories"].values()) == 40800
    replicate_rows = [
        row
        for case in micro["per_case"]
        for row in case["replicate_sufficient_statistics"]
    ]
    assert len(replicate_rows) == 30 * 8
    assert all(
        row["fragment_trace_count"] == row["expected_fragment_trace_count"]
        for row in replicate_rows
    )
    assert len(micro["profile_region_strata"]) == 15
    assert all(item["case_count"] == 2 for item in micro["profile_region_strata"].values())

    checks = {
        "strict_json_and_finite_values": "pass",
        "content_hashes": "pass",
        "sealed_test_result_json_lines_decoded": 0,
        "aggregate_development_case_count": 2300,
        "aggregate_joint_domain_complete_case_count": 1120,
        "micro_selected_train_case_count": 30,
        "micro_replicate_rows": len(replicate_rows),
        "micro_fragment_trace_count": 40800,
        "micro_fragment_component_pair_count": 40800 * 67,
        "archive_equivalence": "exact",
    }
    result = {
        "schema_version": "mechanism-output-verification@1.0.0",
        "status": "pass",
        "aggregate_report_hash": aggregate["report_hash"],
        "micro_report_hash": micro["report_hash"],
        "checks": checks,
    }
    result["report_hash"] = content_hash(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
