from __future__ import annotations

from copy import deepcopy
import json

from damage_engine.engine import DamageEngine
from damage_engine.utils import content_hash
from tools.recompute_tree_outputs import build_report, recompute_tree_outputs


def _result_with_component_state_trace(target_data, case_data, registry):
    engine = DamageEngine(target_data, registry)
    case = deepcopy(case_data)
    case["analysis"]["replicates"] = 4
    case["analysis"]["adaptive_ci_half_width"] = None
    case["analysis"]["capture_component_state_trace"] = True
    return engine.run(case).to_dict()


def test_current_tree_offline_recompute_matches_online_mean_within_one_e_minus_nine(
    target_data,
    case_data,
    registry,
):
    result = _result_with_component_state_trace(
        target_data,
        case_data,
        registry,
    )

    recomputed = recompute_tree_outputs(result, target_data)

    assert recomputed["recomputed_tree"]["definition_hash"] == content_hash(
        target_data["damage_tree"]
    )
    assert recomputed["replicate_count"] == 4
    comparison = recomputed["online_same_output_comparison"]
    assert comparison is not None
    assert comparison["maximum_absolute_error"] <= 1.0e-9
    for output_name, mean in recomputed["output_means"].items():
        assert abs(mean - result["damage_tree"]["outputs"][output_name]["mean"]) <= 1.0e-9


def test_report_reads_json_and_jsonl_results(
    tmp_path,
    target_data,
    case_data,
    registry,
):
    result = _result_with_component_state_trace(
        target_data,
        case_data,
        registry,
    )
    json_path = tmp_path / "result.json"
    jsonl_path = tmp_path / "results.jsonl"
    json_path.write_text(json.dumps(result), encoding="utf-8")
    jsonl_path.write_text(json.dumps(result) + "\n", encoding="utf-8")

    report = build_report([json_path, jsonl_path], target_data)

    assert report["case_count"] == 2
    assert report["tree_definition_hash"] == content_hash(
        target_data["damage_tree"]
    )
    assert all(
        case["online_same_output_comparison"]["maximum_absolute_error"]
        <= 1.0e-9
        for case in report["cases"]
    )
