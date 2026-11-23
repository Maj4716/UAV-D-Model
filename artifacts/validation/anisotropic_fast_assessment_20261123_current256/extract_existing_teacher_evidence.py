"""Extract role-labelled development teacher rows without decoding sealed test results."""

from __future__ import annotations

from collections import Counter
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


def _report(
    *,
    rows: list[dict[str, object]],
    source: dict[str, object],
    live: dict[str, object],
    role_scope: str,
) -> dict[str, object]:
    target_hashes = set(source["source_contracts"].get("target_hashes") or [])
    role_counts = Counter(str(row["split_role"]) for row in rows)
    report: dict[str, object] = {
        "schema_version": "existing-engine-evidence-adapter@1.1.0",
        "role_scope": role_scope,
        "teacher_role_counts": dict(sorted(role_counts.items())),
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
            "pilot": PILOT.relative_to(ROOT).as_posix(),
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
    return report


def main() -> None:
    assessment = json.loads(ASSESSMENT.read_text(encoding="utf-8"))
    wanted = list(assessment["candidate_ids"])
    wanted_set = set(wanted)
    evidence, source = load_development_evidence(PILOT)
    if source["test_result_json_lines_decoded"] != 0:
        raise RuntimeError("sealed test result row was decoded")
    if source["test_metrics_computed"] or source["test_open_count_increment"] != 0:
        raise RuntimeError("sealed test evidence was opened or evaluated")
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
    outputs = {
        "existing_teacher_rows.json": (rows, "development_train_validation"),
        "train_teacher_rows.json": (
            [row for row in rows if row["split_role"] == "train"],
            "train",
        ),
        "validation_teacher_rows.json": (
            [row for row in rows if row["split_role"] == "validation"],
            "validation",
        ),
    }
    summary: dict[str, object] = {}
    for name, (subset, role_scope) in outputs.items():
        report = _report(
            rows=subset,
            source=source,
            live=live,
            role_scope=role_scope,
        )
        path = HERE / name
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        summary[name] = {
            "row_count": len(subset),
            "teacher_role_counts": report["teacher_role_counts"],
            "report_hash": report["report_hash"],
        }
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
