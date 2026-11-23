"""Cache teacher damage-tree blocks (train/validation only) for analyze_sweep.py."""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DATASET, ROOT, WORK  # noqa: E402

sys.path.insert(0, str(ROOT))
from tools.evaluate_proxy_teacher_distribution import load_development_evidence  # noqa: E402


def main() -> int:
    evidence, audit = load_development_evidence(DATASET)
    assert audit["test_result_json_lines_decoded"] == 0
    raw = [
        {"case_id": item.case["case_id"], "damage_tree": item.result["damage_tree"]}
        for item in evidence
    ]
    with gzip.open(WORK / "teacher_damage_tree_raw.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(raw, stream)
    print("cached", len(raw))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
