"""Compare the frozen 5000-case archive with the pre-change inventory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.utils import content_hash


DATASET = ROOT / "raw_data" / "dataset_5000_loitering_impact_fragment_only_v1"
EVIDENCE = Path(__file__).resolve().parent


def inventory() -> dict:
    files = []
    for path in sorted(item for item in DATASET.rglob("*") if item.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append(
            {
                "path": path.relative_to(DATASET).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": f"sha256:{digest}",
            }
        )
    report = {
        "schema_version": "1.0.0",
        "scope": str(DATASET),
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }
    report["inventory_hash"] = content_hash(report)
    return report


def main() -> int:
    before_path = EVIDENCE / "dataset_preservation_before.json"
    after_path = EVIDENCE / "dataset_preservation_after.json"
    comparison_path = EVIDENCE / "dataset_preservation_comparison.json"
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = inventory()
    before_map = {item["path"]: item for item in before["files"]}
    after_map = {item["path"]: item for item in after["files"]}
    added = sorted(set(after_map) - set(before_map))
    removed = sorted(set(before_map) - set(after_map))
    changed = sorted(
        path
        for path in set(before_map) & set(after_map)
        if before_map[path] != after_map[path]
    )
    identical = not added and not removed and not changed and (
        before.get("inventory_hash") == after.get("inventory_hash")
    )
    comparison = {
        "schema_version": "1.0.0",
        "identical": identical,
        "before_inventory_hash": before.get("inventory_hash"),
        "after_inventory_hash": after.get("inventory_hash"),
        "before_file_count": before.get("file_count"),
        "after_file_count": after.get("file_count"),
        "before_total_bytes": before.get("total_bytes"),
        "after_total_bytes": after.get("total_bytes"),
        "added": added,
        "removed": removed,
        "changed": changed,
    }
    after_path.write_text(
        json.dumps(after, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    comparison_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, ensure_ascii=False, allow_nan=False))
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
