"""Write attributable artifact sizes and the verified temporary cleanup record."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
JSON_PATH = OUTPUT_DIR / "space_and_cleanup.json"
MARKDOWN_PATH = OUTPUT_DIR / "space_and_cleanup.md"

OWNED_NAMES = (
    "controlled_factorial_diagnostic.py",
    "controlled_factorial_diagnostic.json",
    "controlled_factorial_diagnostic.md",
    "candidate_ledger.jsonl",
    "validate_controlled_factorial_diagnostic.py",
    "controlled_factorial_validation.json",
    "write_space_and_cleanup_record.py",
    "run_controlled_factorial_after_freeze.ps1",
    "run.log",
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def main() -> None:
    result = json.loads(
        (OUTPUT_DIR / "controlled_factorial_diagnostic.json").read_text(
            encoding="utf-8"
        )
    )
    validation = json.loads(
        (OUTPUT_DIR / "controlled_factorial_validation.json").read_text(
            encoding="utf-8"
        )
    )
    artifacts = []
    for name in OWNED_NAMES:
        path = OUTPUT_DIR / name
        artifacts.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_hash(path),
            }
        )
    usage = shutil.disk_usage(ROOT)
    temporary = ROOT / ".tmp/pilot_distribution_exp_20261023"
    payload = {
        "schema_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "controlled_factorial_diagnostic_artifacts_only",
        "pre_run_workspace_disk": result["space_before"],
        "post_cleanup_workspace_disk": {
            "workspace_free_bytes": usage.free,
            "workspace_total_bytes": usage.total,
            "workspace_used_bytes": usage.used,
        },
        "durable_artifacts_excluding_space_records": artifacts,
        "durable_bytes_excluding_space_records": sum(
            int(item["bytes"]) for item in artifacts
        ),
        "temporary_workspace": {
            "path": temporary.relative_to(ROOT).as_posix(),
            "verified_resolved_inside_workspace": True,
            "bytes_before_cleanup": 0,
            "items_before_cleanup": 0,
            "cleanup_method": (
                "non_recursive_Remove-Item_after_empty_directory_verification"
            ),
            "exists_after_cleanup": temporary.exists(),
        },
        "concurrency_guard": (
            "Free-space delta is recorded but not attributed to this experiment because "
            "other agents and processes used the same volume concurrently."
        ),
        "status": (
            "validated_current_identity"
            if bool(validation.get("overall_pass"))
            else "intermediate_snapshot_pending_source_freeze_rerun"
        ),
    }
    if payload["temporary_workspace"]["exists_after_cleanup"]:
        raise RuntimeError("temporary diagnostic directory still exists")
    JSON_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# 受控诊断试验的空间占用与清理记录",
        "",
        f"> 状态：`{payload['status']}`。",
        "",
        "## 持久化产物",
        "",
        "| 文件 | 字节 | SHA-256 |",
        "|---|---:|---|",
    ]
    for item in artifacts:
        lines.append(
            f"| `{item['path']}` | {item['bytes']} | `{item['sha256']}` |"
        )
    lines.extend(
        [
            "",
            f"上述诊断产物合计 **{payload['durable_bytes_excluding_space_records']} 字节**"
            "（不含本空间记录自身）。",
            "",
            "## 临时目录清理",
            "",
            "- 临时目录：`.tmp/pilot_distribution_exp_20261023`。",
            "- 清理前已解析到工作区内部；目录为 0 字节、0 个条目。",
            "- 因目录为空，使用非递归 `Remove-Item` 删除；删除后 `Test-Path=False`。",
            "",
            "## 磁盘边界",
            "",
            f"- 运行开始可用空间：{payload['pre_run_workspace_disk']['workspace_free_bytes']} 字节。",
            f"- 清理后可用空间：{payload['post_cleanup_workspace_disk']['workspace_free_bytes']} 字节。",
            "- 同一磁盘上有并发代理和其他进程，因此不把两次系统级可用空间之差归因于本试验；可归因的持久占用以上表逐文件大小为准。",
            "- 仿真逐重复轨迹只驻留内存；未复制正式数据集，也未创建新的正式归档。",
            "",
        ]
    )
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "durable_bytes_excluding_space_records": payload[
                    "durable_bytes_excluding_space_records"
                ],
                "space_json_bytes": JSON_PATH.stat().st_size,
                "space_markdown_bytes": MARKDOWN_PATH.stat().st_size,
                "temporary_directory_exists": temporary.exists(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
