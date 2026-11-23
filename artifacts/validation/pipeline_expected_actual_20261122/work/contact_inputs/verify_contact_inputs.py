"""Integrity checks for the pre-outcome contact-input analysis."""

from __future__ import annotations

import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.utils import content_hash, file_hash  # noqa: E402


def main() -> int:
    metrics_path = HERE / "contact_input_metrics.json"
    summary_path = HERE / "SUMMARY.md"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    report_hash = metrics.pop("report_hash")
    assert content_hash(metrics) == report_hash
    scope = metrics["scope"]
    assert scope["formal_input_case_count"] == 2600
    assert scope["split_counts"] == {"train": 2000, "validation": 300, "test": 300}
    assert scope["teacher_result_files_opened"] == 0
    assert scope["teacher_result_json_lines_decoded"] == 0
    checks = metrics["deterministic_contract_checks"]
    assert checks["eligibility_check_false_or_missing_counts"] == {}
    assert checks["nonpositive_forward_normal_cosine_count"] == 0
    assert checks["velocity_magnitude_minus_declared_speed_max_abs_mps"] < 1.0e-12
    assert checks["velocity_unit_direction_minus_declared_direction_max_l2"] < 1.0e-12
    assert checks["collision_time_minus_distance_over_speed_max_abs_s"] < 1.0e-12
    assert checks["detonation_point_minus_first_contact_point_max_l2_m"] < 1.0e-12
    assert checks["detonation_time_minus_first_contact_time_max_abs_s"] < 1.0e-12
    assert metrics["actual_distributions"]["speed_mps"]["out_of_declared_range_count"] == 0
    assert len(metrics["actual_distributions"]["profile_region_strata"]) == 15
    summary = summary_path.read_text(encoding="utf-8")
    for fragment in ("2,600", "结果 JSON 解码数均为 0", "S-CONTACT", "KS="):
        assert fragment in summary, fragment
    print(
        json.dumps(
            {
                "status": "pass",
                "metrics_sha256": file_hash(metrics_path),
                "summary_sha256": file_hash(summary_path),
                "teacher_result_json_lines_decoded": 0,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
