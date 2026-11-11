from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from build_p0_r06_diagnostic_bundles import finalize_staging_directory
from generate_p0_r06_diagnostics import strict_json
from validate_p0_r06_diagnostics import job_base_files, sha256, validate_batch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = ROOT / "manual_upload_20261029_r06h_timestep_confirmation"
DEFAULT_OUTPUT = ROOT / "artifacts" / "r06h_tssfac01_submission_20261029"
DEFAULT_AUDIT = (
    ROOT
    / "analysis"
    / "72639935_r06g_0800_tssfac02_20261029"
    / "r06g_tssfac02_return_audit.json"
)
REVISION = "R06_timestep_confirmation_20261029"
JOB_ID = "r06h_0800_tssfac01"
JOB_DIR = "job_01_v0800_tssfac01"
UPLOAD_DIR = "uav_hf_r06h_0800_tssfac01"
HEAD_DECK = "r06h_0800_tssfac01.k"
PARENT_DECK_SHA256 = "sha256:b5f4da367e9ef6eaae7098e47a84610eb55c51760cc9bf8dbb4721693b7a732a"


def _verify_authorization(audit: dict) -> None:
    decision = audit["decision"]
    if (
        audit["source_evidence"]["platform_job_id"] != "72639935"
        or audit["source_evidence"]["deck_sha256"] != PARENT_DECK_SHA256
        or decision["solver_parser_runtime"] != "pass"
        or decision["resource_identity"] != "pass"
        or decision["measurement_channels"] != "pass"
        or decision["timestep_refinement_gate"] != "pass"
        or decision["release_r06h_0800_tssfac01"] is not True
    ):
        raise AssertionError("Job 72639935 audit does not authorize R06h")


def stage_confirmation(batch_dir: Path, output_dir: Path, audit_path: Path) -> dict:
    batch_dir = batch_dir.resolve()
    output_dir = output_dir.resolve()
    audit_path = audit_path.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite direct-folder delivery: {output_dir}")
    staging = output_dir.parent / f".{output_dir.name}.staging"
    if staging.exists():
        raise FileExistsError(f"delivery staging directory already exists: {staging}")

    audit = strict_json(audit_path)
    _verify_authorization(audit)
    validation = validate_batch(batch_dir, write_reports=False)
    if validation["overall_status"] != "pass" or validation["job_count"] != 1:
        raise AssertionError("R06h batch is not a single statically valid job")
    source_job = batch_dir / JOB_DIR
    source_manifest = strict_json(source_job / "manifest.json")
    if source_manifest["package_revision"] != REVISION or source_manifest["job_id"] != JOB_ID:
        raise AssertionError("R06h batch identity differs")
    expected_source_files = job_base_files(source_manifest) | {HEAD_DECK, "static_validation.json"}
    actual_source_files = {path.name for path in source_job.iterdir() if path.is_file()}
    if actual_source_files != expected_source_files:
        raise AssertionError("R06h source job does not contain the exact validated whitelist")
    if any(path.is_symlink() or not path.is_file() for path in source_job.iterdir()):
        raise AssertionError("R06h source job must contain ordinary files only")

    staging.mkdir(parents=True)
    upload = staging / UPLOAD_DIR
    upload.mkdir()
    for name in sorted(expected_source_files):
        shutil.copy2(source_job / name, upload / name)
    if any(path.suffix.lower() == ".zip" for path in staging.rglob("*")):
        raise AssertionError("direct-folder delivery must not contain ZIP archives")
    copied_files = [
        {
            "path": f"{UPLOAD_DIR}/{name}",
            "size_bytes": (upload / name).stat().st_size,
            "sha256": sha256(upload / name),
        }
        for name in sorted(expected_source_files)
    ]
    manifest = {
        "schema_version": "hf-p0-r06h-direct-folder-delivery@1.0.0",
        "package_revision": REVISION,
        "job_id": JOB_ID,
        "status": "ready_for_user_submission_not_submitted_not_solver_validated",
        "upload_folder": UPLOAD_DIR,
        "head_deck": HEAD_DECK,
        "archive_policy": "no_archive_direct_folder_only",
        "parent_control": {
            "platform_job_id": "72639935",
            "deck_sha256": PARENT_DECK_SHA256,
            "audit_path": audit_path.as_posix(),
            "audit_sha256": sha256(audit_path),
        },
        "single_factor": {
            "field": "CONTROL_TIMESTEP.TSSFAC",
            "control": 0.2,
            "treatment": 0.1,
            "dt2ms": 0.0,
        },
        "resource_contract": {
            "software": "ls-dyna_R15_mmp_d",
            "nodes": 1,
            "ranks": 32,
            "walltime": "04:00:00",
            "head_memory_mb": 512,
            "child_memory_mb": 256,
            "command_line_parameters": "",
            "fresh_work_directory": "/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06h_0800_ts01",
        },
        "files": copied_files,
        "next_gate_if_return_passes": "select_local_production_timestep_before_any_2000_control",
        "prohibited": [
            "creating or submitting a ZIP archive",
            "submitting any other job in parallel with this confirmation diagnostic",
            "changing the explosion-instant/post-explosion initial state or any factor other than TSSFAC",
            "using nodes/ranks other than one node and 32 total MPP ranks",
            "enabling mass scaling",
            "treating static PASS as solver, physical, engine, damage-tree or dataset acceptance",
        ],
    }
    manifest_path = staging / "DELIVERY_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    instructions = f"""# R06h 直接文件夹提交说明

不要压缩。只上传整个 `{UPLOAD_DIR}` 文件夹，并在平台选择 `{HEAD_DECK}` 作为 head/input deck。

资源参数：1 个节点、32 MPP ranks、`ls-dyna_R15_mmp_d`、命令行参数留空。使用全新工作目录
`/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06h_0800_ts01`。

本轮唯一变量是 `CONTROL_TIMESTEP.TSSFAC: 0.2 -> 0.1`，`DT2MS=0`。当前模型已经从爆炸后的破片初始状态开始，
不再添加爆炸前飞行或起爆阶段，也不允许在这次时间步对照中改变初始状态。

完成后下载完整回传：`.k`、`d3hsp/o=`、`mes*`、`glstat`、`binout*`、完整 `d3plot*`、完整 `d3part*`、
stdout/stderr、Slurm 脚本和资源记录。如果平台返回一个路径，之后只需把那个路径发给我。
"""
    instruction_path = staging / "SUBMIT_ONLY_THIS_FOLDER.md"
    instruction_path.write_text(instructions, encoding="utf-8", newline="\n")
    expected_root = {UPLOAD_DIR, manifest_path.name, instruction_path.name}
    if {path.name for path in staging.iterdir()} != expected_root:
        raise AssertionError("direct-folder delivery root differs from the exact whitelist")
    if {path.name for path in upload.iterdir()} != expected_source_files:
        raise AssertionError("direct upload folder differs from the exact source whitelist")
    for item in copied_files:
        copied = staging / item["path"]
        if copied.stat().st_size != item["size_bytes"] or sha256(copied) != item["sha256"]:
            raise AssertionError(f"copied file identity differs: {item['path']}")
    finalize_staging_directory(staging, output_dir)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage R06h as a direct folder without ZIP")
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = stage_confirmation(args.batch_dir, args.output_dir, args.audit)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output": str(args.output_dir.resolve()),
                "upload_folder": manifest["upload_folder"],
                "head_deck": manifest["head_deck"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
