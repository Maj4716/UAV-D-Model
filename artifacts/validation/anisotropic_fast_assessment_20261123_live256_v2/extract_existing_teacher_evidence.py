"""Extract paired train/validation teacher rows without decoding sealed test results."""

from __future__ import annotations

import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.exposure_audit import teacher_component_hit_probabilities
from damage_engine.utils import content_hash, file_hash
from tools.evaluate_proxy_teacher_distribution import load_development_evidence


ASSESSMENT = HERE / "assessment.json"
PILOT = ROOT / "artifacts/datasets/recovery_role_r8_r32_train2000_eval300_20261122"
OUTPUT = HERE / "existing_teacher_rows.json"


def main() -> None:
    assessment = json.loads(ASSESSMENT.read_text(encoding="utf-8"))
    wanted = list(assessment["candidate_ids"])
    wanted_set = set(wanted)
    evidence, source = load_development_evidence(PILOT)
    if source["test_result_json_lines_decoded"] != 0:
        raise RuntimeError("sealed test result row was decoded")
    selected = {
        str(item.case["case_id"]): item
        for item in evidence
        if str(item.case["case_id"]) in wanted_set
    }
    if set(selected) != wanted_set:
        raise RuntimeError("quick assessment IDs do not align with development evidence")
    rows = [
        {
            "candidate_id": candidate_id,
            "split_role": selected[candidate_id].split,
            "component_hit_probability": teacher_component_hit_probabilities(
                selected[candidate_id].result
            ),
        }
        for candidate_id in wanted
    ]
    live = assessment["provenance"]["live_proxy_identity"]
    target_hashes = set(source["source_contracts"].get("target_hashes") or [])
    report = {
        "schema_version": "existing-engine-evidence-adapter@1.0.0",
        "teacher_rows": rows,
        "source_audit": source,
        "identity_comparison": {
            "target_hash_match": live["target_hash"] in target_hashes,
            "physics_registry_hash_match": (
                live["physics_registry_hash"]
                == source["source_engine"].get("configuration_hash")
            ),
            "engine_code_hash_match": (
                live["engine_code_hash"] == source["source_engine"].get("code_hash")
            ),
            "live_proxy_identity": live,
            "teacher_engine_identity": source["source_engine"],
        },
        "provenance": {
            "assessment_sha256": file_hash(ASSESSMENT),
            "pilot": str(PILOT),
            "selected_case_count": len(rows),
            "test_result_json_lines_decoded": source[
                "test_result_json_lines_decoded"
            ],
            "test_metrics_computed": source["test_metrics_computed"],
            "test_open_count_increment": source["test_open_count_increment"],
        },
        "evidence_boundary": (
            "Existing train/validation software-teacher evidence only; no new "
            "simulation and no external physical validation."
        ),
    }
    report["report_hash"] = content_hash(report)
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "selected_case_count": len(rows),
                "test_result_json_lines_decoded": source[
                    "test_result_json_lines_decoded"
                ],
                "identity_comparison": report["identity_comparison"],
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
