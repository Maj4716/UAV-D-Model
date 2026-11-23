#!/usr/bin/env python3
"""Verify the workspace-change reporting bundle and write machine evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
TZ = ZoneInfo("Asia/Shanghai")
REQUIRED = [
    "WORKSPACE_CHANGE_REPORT.md",
    "PEER_BRIEFING.md",
    "EVIDENCE_INDEX.md",
    "commit_summary.csv",
    "historical_commit_changes.csv",
    "tracked_worktree_diff.csv",
    "current_change_inventory.csv",
    "directory_summary.csv",
    "inventory_summary.json",
    "git_status_snapshot.txt",
    "AUDIT_LOG.txt",
    "verify_bundle.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_numstat(value: str) -> int:
    return int(value) if value.isdigit() else 0


def local_links(markdown: str) -> list[str]:
    targets = re.findall(r"\[[^\]]*\]\(([^)]+)\)", markdown)
    return [
        target.strip("<>")
        for target in targets
        if not target.startswith(("http://", "https://", "#", "mailto:"))
    ]


def main() -> int:
    checks: list[dict] = []

    def check(name: str, passed: bool, details: object = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "details": details})

    for name in REQUIRED:
        check(f"required_file:{name}", (ROOT / name).is_file())

    summary = json.loads((ROOT / "inventory_summary.json").read_text(encoding="utf-8"))
    commits = read_csv("commit_summary.csv")
    history = read_csv("historical_commit_changes.csv")
    tracked = read_csv("tracked_worktree_diff.csv")
    current = read_csv("current_change_inventory.csv")
    directories = read_csv("directory_summary.csv")

    check("commit_count_matches", len(commits) == summary["commit_count"], len(commits))
    check(
        "historical_path_count_matches",
        len(history) == summary["historical_commit_change_rows"],
        len(history),
    )
    check(
        "tracked_diff_count_matches",
        len(tracked) == summary["tracked_worktree_diff_files"],
        len(tracked),
    )
    check(
        "tracked_added_lines_match",
        sum(parse_numstat(row["added_lines"]) for row in tracked)
        == summary["tracked_worktree_added_lines"],
    )
    check(
        "tracked_deleted_lines_match",
        sum(parse_numstat(row["deleted_lines"]) for row in tracked)
        == summary["tracked_worktree_deleted_lines"],
    )
    check(
        "current_inventory_count_matches",
        len(current) == summary["current_inventory_files"],
        len(current),
    )
    check(
        "current_inventory_bytes_match",
        sum(int(row["size_bytes"]) for row in current) == summary["current_inventory_bytes"],
    )
    check("directory_summary_nonempty", bool(directories), len(directories))
    check("scan_errors_zero", summary["scan_errors"] == [], summary["scan_errors"])
    check("expected_branch", summary["branch"] == "main", summary["branch"])
    check(
        "expected_head",
        summary["head"] == "ff2ef1f21c7d6bf73630bdb11331289278fb69dd",
        summary["head"],
    )
    check(
        "audit_tool_in_inventory",
        any(row["path"] == "tools/build_workspace_change_inventory.py" for row in current),
    )
    check(
        "self_output_excluded",
        not any("workspace_changes_since_20261104_20261123" in row["path"] for row in current),
    )

    for markdown_name in ("WORKSPACE_CHANGE_REPORT.md", "PEER_BRIEFING.md", "EVIDENCE_INDEX.md"):
        markdown = (ROOT / markdown_name).read_text(encoding="utf-8")
        links = local_links(markdown)
        missing = [link for link in links if not (ROOT / link).resolve().exists()]
        check(f"local_links_resolve:{markdown_name}", not missing, missing)

    report = (ROOT / "WORKSPACE_CHANGE_REPORT.md").read_text(encoding="utf-8")
    for token in (
        "43 个 tracked 文件",
        "135 个研究功能文件",
        "v4 物理与 SamplingSpec v2",
        "外部物理标定",
        "Python 3.12.13/NumPy 2.4.6",
    ):
        check(f"report_contains:{token}", token in report)

    manifest_entries = []
    for name in REQUIRED:
        path = ROOT / name
        if path.exists():
            manifest_entries.append({
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    manifest = {
        "schema_version": "workspace-change-report-manifest/1.0",
        "generated_at": datetime.now(TZ).isoformat(timespec="seconds"),
        "files": manifest_entries,
    }
    (ROOT / "deliverable_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for entry in manifest_entries:
        check(
            f"manifest_hash:{entry['path']}",
            sha256(ROOT / entry["path"]) == entry["sha256"],
        )

    passed = sum(item["passed"] for item in checks)
    verification = {
        "schema_version": "workspace-change-report-verification/1.0",
        "verified_at": datetime.now(TZ).isoformat(timespec="seconds"),
        "passed": passed,
        "total": len(checks),
        "all_passed": passed == len(checks),
        "checks": checks,
    }
    (ROOT / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    return 0 if verification["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
