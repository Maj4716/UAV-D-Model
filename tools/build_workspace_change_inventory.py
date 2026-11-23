#!/usr/bin/env python3
"""Build a reproducible inventory of workspace changes since a cutoff date.

The inventory separates committed history, tracked working-tree differences,
untracked source files, and ignored/unindexed local artifacts.  It deliberately
excludes Git internals and ephemeral Python/pytest caches.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Iterable
from zoneinfo import ZoneInfo


LOCAL_TZ = ZoneInfo("Asia/Shanghai")
DEFAULT_CUTOFF = "2026-11-04T00:00:00+08:00"
EXCLUDED_DIRS = {".git", ".pytest_cache", "__pycache__"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


@dataclass(frozen=True)
class CommitRow:
    commit: str
    authored_at: str
    subject: str
    files_changed: int
    insertions: int
    deletions: int


def run_git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.stdout


def parse_int(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 0


def normalize_path(value: str) -> str:
    return value.replace("\\", "/").lstrip("./")


def theme_for(path: str) -> str:
    p = path.lower()
    name = PurePosixPath(p).name
    if p.startswith("artifacts/refactor_20261117/"):
        return "repository_refactor"
    if "anisotropic_fast_assessment" in p:
        return "anisotropic_fast_assessment"
    if "proxy_convergence" in p or "fast_sampling_proxy" in p:
        return "fast_proxy_convergence"
    if "pipeline_expected_actual" in p or "data_generation_fixes" in p:
        return "pipeline_audit_and_fixes"
    if "dataset_recovery" in p or "recovery_role" in p:
        return "dataset_recovery"
    if "dataset_v2" in p or "hurdle" in p or "target_encoding" in p:
        return "dataset_v2_and_learning"
    if any(token in p for token in (
        "sphere_entry", "continuous_direction", "sampling_measure",
        "sampling_support", "sampling_inference", "design_sequences",
        "reference_audit", "distribution_metrics",
    )):
        return "reference_and_sphere_sampling"
    if any(token in p for token in (
        "exposure_", "multifidelity", "fast_assessment",
    )):
        return "exposure_and_multifidelity"
    if "idea2" in p or any(token in p for token in (
        "stage2_mechanism", "mechanism_diagnostics", "sensitivity",
        "credibility", "warhead_factor",
    )):
        return "idea2_mechanism_and_credibility"
    if "tree_candidate" in p or "offline_tree" in p:
        return "offline_tree_candidates"
    if any(token in p for token in (
        "fragment", "penetration", "munition_profiles", "taylor_shapiro",
    )):
        return "fragment_and_penetration_physics"
    if p.startswith("tests/") or name.startswith("test_"):
        return "tests_other"
    if p.startswith("docs/") or p.endswith(".md"):
        return "documentation_other"
    if p.startswith("configs/") or p.startswith("schemas/"):
        return "configuration_and_schema_other"
    if p.startswith("tools/"):
        return "tooling_other"
    if p.startswith("artifacts/datasets/"):
        return "datasets_other"
    if p.startswith("artifacts/experiments/"):
        return "experiments_other"
    if p.startswith("artifacts/validation/"):
        return "validation_other"
    return "core_or_project_other"


def logical_kind(path: str) -> str:
    p = PurePosixPath(path)
    suffix = p.suffix.lower()
    if suffix in {".py"}:
        return "python"
    if suffix in {".json", ".jsonl", ".yaml", ".yml", ".toml"}:
        return "config_or_structured_data"
    if suffix in {".md", ".txt", ".rst"}:
        return "documentation_or_log"
    if suffix in {".csv", ".tsv", ".npz", ".npy", ".parquet", ".gz"}:
        return "data_or_result"
    if suffix in {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".html"}:
        return "figure_or_report"
    return "other"


def iter_files(root: Path, output_dir: Path) -> Iterable[Path]:
    output_abs = output_dir.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        dirnames[:] = [
            d for d in dirnames
            if d not in EXCLUDED_DIRS
            and (current / d).resolve() != output_abs
        ]
        for filename in filenames:
            path = current / filename
            if path.suffix.lower() in EXCLUDED_SUFFIXES:
                continue
            yield path


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def get_commit_rows(root: Path, cutoff: str) -> tuple[list[CommitRow], list[dict], set[str]]:
    raw = run_git(root, "log", f"--since={cutoff}", "--format=%H%x1f%aI%x1f%s")
    commits: list[CommitRow] = []
    path_rows: list[dict] = []
    current_candidate_paths: set[str] = set()
    for line in raw.splitlines():
        if not line.strip():
            continue
        commit, authored_at, subject = line.split("\x1f", 2)
        shortstat = run_git(root, "show", "--format=", "--shortstat", commit)
        files_match = re.search(r"(\d+) files? changed", shortstat)
        insertions_match = re.search(r"(\d+) insertions?\(\+\)", shortstat)
        deletions_match = re.search(r"(\d+) deletions?\(-\)", shortstat)
        files_changed = int(files_match.group(1)) if files_match else 0
        insertions = int(insertions_match.group(1)) if insertions_match else 0
        deletions = int(deletions_match.group(1)) if deletions_match else 0
        commits.append(CommitRow(commit, authored_at, subject, files_changed, insertions, deletions))

        changes = run_git(root, "diff-tree", "--root", "--no-commit-id", "--name-status", "-r", "-M", commit)
        for change in changes.splitlines():
            parts = change.split("\t")
            if len(parts) < 2:
                continue
            status = parts[0]
            if status.startswith("R") and len(parts) >= 3:
                old_path = normalize_path(parts[1])
                new_path = normalize_path(parts[2])
                path_rows.append({
                    "commit": commit,
                    "authored_at": authored_at,
                    "subject": subject,
                    "status": status,
                    "old_path": old_path,
                    "path": new_path,
                    "theme": theme_for(new_path),
                })
                current_candidate_paths.add(new_path)
            else:
                changed_path = normalize_path(parts[1])
                path_rows.append({
                    "commit": commit,
                    "authored_at": authored_at,
                    "subject": subject,
                    "status": status,
                    "old_path": "",
                    "path": changed_path,
                    "theme": theme_for(changed_path),
                })
                if not status.startswith("D"):
                    current_candidate_paths.add(changed_path)
    return commits, path_rows, current_candidate_paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/validation/workspace_changes_since_20261104_20261123"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = (root / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(LOCAL_TZ)
    cutoff_dt = datetime.fromisoformat(args.cutoff)
    if cutoff_dt.tzinfo is None:
        cutoff_dt = cutoff_dt.replace(tzinfo=LOCAL_TZ)

    branch = run_git(root, "branch", "--show-current").strip()
    head = run_git(root, "rev-parse", "HEAD").strip()
    commits, commit_path_rows, committed_candidates = get_commit_rows(root, args.cutoff)

    tracked = {normalize_path(p) for p in run_git(root, "ls-files").splitlines() if p.strip()}
    untracked = {
        normalize_path(p)
        for p in run_git(root, "ls-files", "--others", "--exclude-standard").splitlines()
        if p.strip()
    }
    tracked_modified = {
        normalize_path(p)
        for p in run_git(root, "diff", "HEAD", "--name-only").splitlines()
        if p.strip()
    }

    tracked_diff_rows: list[dict] = []
    for line in run_git(root, "diff", "HEAD", "--numstat").splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        changed_path = normalize_path(parts[2])
        tracked_diff_rows.append({
            "path": changed_path,
            "added_lines": parts[0],
            "deleted_lines": parts[1],
            "theme": theme_for(changed_path),
        })

    file_rows: list[dict] = []
    scan_errors: list[dict] = []
    for path in iter_files(root, output_dir):
        try:
            stat = path.stat()
        except OSError as exc:
            scan_errors.append({"path": str(path), "error": repr(exc)})
            continue
        rel = normalize_path(path.relative_to(root).as_posix())
        mtime = datetime.fromtimestamp(stat.st_mtime, LOCAL_TZ)
        if rel in tracked_modified:
            git_state = "tracked_modified"
        elif rel in tracked and rel in committed_candidates:
            git_state = "tracked_committed_since_cutoff"
        elif rel in untracked:
            git_state = "untracked"
        elif rel not in tracked:
            git_state = "ignored_or_unindexed"
        else:
            continue

        if git_state in {"untracked", "ignored_or_unindexed"} and mtime < cutoff_dt:
            continue
        file_rows.append({
            "path": rel,
            "git_state": git_state,
            "theme": theme_for(rel),
            "kind": logical_kind(rel),
            "size_bytes": stat.st_size,
            "mtime_local": mtime.isoformat(timespec="seconds"),
            "top_level": rel.split("/", 1)[0],
            "extension": path.suffix.lower(),
        })

    file_rows.sort(key=lambda row: (row["git_state"], row["theme"], row["path"]))
    commit_path_rows.sort(key=lambda row: (row["authored_at"], row["path"]))
    tracked_diff_rows.sort(key=lambda row: row["path"])

    write_csv(
        output_dir / "commit_summary.csv",
        ["commit", "authored_at", "subject", "files_changed", "insertions", "deletions"],
        [asdict(row) for row in commits],
    )
    write_csv(
        output_dir / "historical_commit_changes.csv",
        ["commit", "authored_at", "subject", "status", "old_path", "path", "theme"],
        commit_path_rows,
    )
    write_csv(
        output_dir / "tracked_worktree_diff.csv",
        ["path", "added_lines", "deleted_lines", "theme"],
        tracked_diff_rows,
    )
    write_csv(
        output_dir / "current_change_inventory.csv",
        ["path", "git_state", "theme", "kind", "size_bytes", "mtime_local", "top_level", "extension"],
        file_rows,
    )

    by_state = defaultdict(lambda: {"files": 0, "bytes": 0})
    by_theme = defaultdict(lambda: {"files": 0, "bytes": 0})
    by_top = defaultdict(lambda: {"files": 0, "bytes": 0})
    by_kind = defaultdict(lambda: {"files": 0, "bytes": 0})
    for row in file_rows:
        size = int(row["size_bytes"])
        for container, key in (
            (by_state, row["git_state"]),
            (by_theme, row["theme"]),
            (by_top, row["top_level"]),
            (by_kind, row["kind"]),
        ):
            container[key]["files"] += 1
            container[key]["bytes"] += size

    directory_rows = []
    for top, values in sorted(by_top.items(), key=lambda item: (-item[1]["bytes"], item[0])):
        directory_rows.append({"top_level": top, **values})
    write_csv(output_dir / "directory_summary.csv", ["top_level", "files", "bytes"], directory_rows)

    deleted_history = sum(1 for row in commit_path_rows if row["status"].startswith("D"))
    current_status = run_git(root, "status", "--short", "--untracked-files=all")
    finished_at = datetime.now(LOCAL_TZ)
    summary = {
        "schema_version": "workspace-change-inventory/1.0",
        "root": str(root),
        "cutoff": cutoff_dt.isoformat(),
        "snapshot_started_at": started_at.isoformat(timespec="seconds"),
        "snapshot_finished_at": finished_at.isoformat(timespec="seconds"),
        "branch": branch,
        "head": head,
        "commit_count": len(commits),
        "commits": [asdict(row) for row in commits],
        "historical_commit_change_rows": len(commit_path_rows),
        "historical_deleted_rows": deleted_history,
        "tracked_worktree_diff_files": len(tracked_diff_rows),
        "tracked_worktree_added_lines": sum(parse_int(row["added_lines"]) for row in tracked_diff_rows),
        "tracked_worktree_deleted_lines": sum(parse_int(row["deleted_lines"]) for row in tracked_diff_rows),
        "current_inventory_files": len(file_rows),
        "current_inventory_bytes": sum(int(row["size_bytes"]) for row in file_rows),
        "by_git_state": dict(sorted(by_state.items())),
        "by_theme": dict(sorted(by_theme.items(), key=lambda item: (-item[1]["bytes"], item[0]))),
        "by_top_level": dict(sorted(by_top.items(), key=lambda item: (-item[1]["bytes"], item[0]))),
        "by_kind": dict(sorted(by_kind.items(), key=lambda item: (-item[1]["bytes"], item[0]))),
        "scan_errors": scan_errors,
        "exclusions": {
            "directories": sorted(EXCLUDED_DIRS),
            "suffixes": sorted(EXCLUDED_SUFFIXES),
            "self_output_directory": normalize_path(output_dir.relative_to(root).as_posix()),
        },
        "interpretation": {
            "committed_history": "Authoritative Git path history since cutoff, including files later deleted.",
            "current_inventory": "Current files attributable to post-cutoff commits, tracked worktree diffs, or post-cutoff untracked/ignored mtimes.",
            "mtime_boundary": "Filesystem mtimes classify untracked and ignored files; they are operational timestamps, not authorship proof.",
        },
    }
    (output_dir / "inventory_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "git_status_snapshot.txt").write_text(current_status, encoding="utf-8")
    log_lines = [
        f"snapshot_started_at={summary['snapshot_started_at']}",
        f"snapshot_finished_at={summary['snapshot_finished_at']}",
        f"branch={branch}",
        f"head={head}",
        f"cutoff={summary['cutoff']}",
        f"commit_count={len(commits)}",
        f"historical_commit_change_rows={len(commit_path_rows)}",
        f"tracked_worktree_diff_files={len(tracked_diff_rows)}",
        f"current_inventory_files={len(file_rows)}",
        f"current_inventory_bytes={summary['current_inventory_bytes']}",
        f"scan_errors={len(scan_errors)}",
    ]
    (output_dir / "AUDIT_LOG.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
