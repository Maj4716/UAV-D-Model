from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from build_p0_r06_diagnostic_bundles import finalize_staging_directory
from generate_p0_r06_diagnostics import strict_json
from validate_p0_r06_diagnostics import job_base_files, sha256, validate_batch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = ROOT / "manual_upload_20261030_r06i_terminal_timestep_refinement"
DEFAULT_OUTPUT = ROOT / "artifacts" / "r06i_tssfac005_submission_20261030"
DEFAULT_AUDIT = (
    ROOT
    / "analysis"
    / "72701744_r06h_0800_tssfac01_20261030"
    / "r06h_tssfac01_return_audit.json"
)
REVISION = "R06_timestep_terminal_refinement_20261030"
JOB_ID = "r06i_0800_tssfac005"
JOB_DIR = "job_01_v0800_tssfac005"
UPLOAD_DIR = "uav_hf_r06i_0800_tssfac005"
HEAD_DECK = "r06i_0800_tssfac005.k"
PARENT_DECK_SHA256 = "sha256:6e4fb3806722df91df8500ea3e2a6758ef6a07c36e913f4e8eccc7a9083374c3"


def _verify_authorization(audit: dict) -> None:
    decision = audit["decision"]
    if (
        audit["source_evidence"]["platform_job_id"] != "72701744"
        or audit["source_evidence"]["deck_sha256"] != PARENT_DECK_SHA256
        or decision["solver_parser_runtime"] != "pass"
        or decision["resource_identity"] != "pass"
        or decision["measurement_channels"] != "pass"
        or decision["timestep_confirmation_gate"] != "fail"
        or decision["failure_scope"] != "deleted_upstream_divergence_and_response_unavailability"
        or decision["release_r06i_0800_tssfac005"] is not True
        or decision["release_any_further_timestep_halving_after_r06i"] is not False
    ):
        raise AssertionError("Job 72701744 audit does not authorize R06i")


def stage_terminal_refinement(batch_dir: Path, output_dir: Path, audit_path: Path) -> dict:
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
        raise AssertionError("R06i batch is not a single statically valid job")
    source_job = batch_dir / JOB_DIR
    source_manifest = strict_json(source_job / "manifest.json")
    if source_manifest["package_revision"] != REVISION or source_manifest["job_id"] != JOB_ID:
        raise AssertionError("R06i batch identity differs")
    expected_source_files = job_base_files(source_manifest) | {HEAD_DECK, "static_validation.json"}
    actual_source_files = {path.name for path in source_job.iterdir() if path.is_file()}
    if actual_source_files != expected_source_files:
        raise AssertionError("R06i source job does not contain the exact validated whitelist")
    if any(path.is_symlink() or not path.is_file() for path in source_job.iterdir()):
        raise AssertionError("R06i source job must contain ordinary files only")

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
        "schema_version": "hf-p0-r06i-direct-folder-delivery@1.0.0",
        "package_revision": REVISION,
        "job_id": JOB_ID,
        "status": "ready_for_user_submission_not_submitted_not_solver_validated",
        "upload_folder": UPLOAD_DIR,
        "head_deck": HEAD_DECK,
        "archive_policy": "no_archive_direct_folder_only",
        "parent_control": {
            "platform_job_id": "72701744",
            "deck_sha256": PARENT_DECK_SHA256,
            "audit_path": audit_path.as_posix(),
            "audit_sha256": sha256(audit_path),
        },
        "single_factor": {
            "field": "CONTROL_TIMESTEP.TSSFAC",
            "control": 0.1,
            "treatment": 0.05,
            "dt2ms": 0.0,
        },
        "terminal_stop_rule": {
            "no_automatic_tssfac_below": 0.05,
            "next_action_after_return": "audit_then_select_timestep_or_pivot_to_deletion_material_contact_diagnostics",
        },
        "resource_contract": {
            "software": "ls-dyna_R15_mmp_d",
            "nodes": 1,
            "ranks": 32,
            "walltime": "04:00:00",
            "head_memory_mb": 512,
            "child_memory_mb": 256,
            "command_line_parameters": "",
            "fresh_work_directory": "/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06i_0800_ts005",
        },
        "files": copied_files,
        "prohibited": [
            "creating or submitting a ZIP archive",
            "submitting any other job in parallel with this terminal refinement",
            "changing the post-explosion initial state or any factor other than TSSFAC",
            "using nodes/ranks other than one node and 32 total MPP ranks",
            "enabling mass scaling",
            "automatically continuing to TSSFAC=0.025 after this return",
            "treating static PASS as solver, physical, engine, damage-tree or dataset acceptance",
        ],
    }
    manifest_path = staging / "DELIVERY_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    instructions = f"""# R06i 终止式时间步细化提交说明

不要压缩。只上传整个 `{UPLOAD_DIR}` 文件夹，并在平台选择 `{HEAD_DECK}` 作为 head/input deck。

资源参数：1 个节点、32 MPP ranks、`ls-dyna_R15_mmp_d`、命令行参数留空。使用全新工作目录
`/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06i_0800_ts005`。

本轮唯一变量是 `CONTROL_TIMESTEP.TSSFAC: 0.1 -> 0.05`，`DT2MS=0`。当前模型继续从爆炸后破片初始状态开始，
不改变初始几何、速度、材料、接触、失效或输出设置。

这是自动减半的终止算例。回传后不会自动生成 TSSFAC=0.025；将根据结果选定时间步，或转入破片删除/材料失效/接触诊断。

完成后下载完整回传：`.k`、`d3hsp/o=`、`mes*`、`glstat`、`binout*`、完整 `d3plot*`、完整 `d3part*`、
stdout/stderr、Slurm 脚本和资源记录。之后只需把回传路径发给我。
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
    parser = argparse.ArgumentParser(description="Stage R06i as a direct folder without ZIP")
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = stage_terminal_refinement(args.batch_dir, args.output_dir, args.audit)
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
