from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from build_p0_r06_diagnostic_bundles import finalize_staging_directory
from generate_p0_r06_diagnostics import strict_json
from validate_p0_r06_diagnostics import job_base_files, sha256, validate_batch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = ROOT / "manual_upload_20261030_r06j_fragment_erosion_isolation"
DEFAULT_OUTPUT = ROOT / "artifacts" / "r06j_fragment_erod1_submission_20261030"
DEFAULT_AUDIT = (
    ROOT
    / "analysis"
    / "72740022_r06i_0800_tssfac005_20261030"
    / "r06i_tssfac005_return_audit.json"
)
REVISION = "R06_fragment_erosion_isolation_20261030"
JOB_ID = "r06j_0800_fragment_erod1"
JOB_DIR = "job_01_v0800_fragment_erod1"
UPLOAD_DIR = "uav_hf_r06j_0800_fragment_erod1"
HEAD_DECK = "r06j_0800_fragment_erod1.k"
PARENT_DECK_SHA256 = "sha256:e2e8da8e55304b7fe2deb2c1790e00bf7fea9c64145c6c09a954c50492f3dd88"


def _verify_authorization(audit: dict) -> None:
    decision = audit["decision"]
    if (
        audit["source_evidence"]["platform_job_id"] != "72740022"
        or audit["source_evidence"]["deck_sha256"] != PARENT_DECK_SHA256
        or decision["solver_parser_runtime"] != "pass"
        or decision["resource_identity"] != "pass"
        or decision["measurement_channels"] != "pass"
        or decision["timestep_terminal_consistency"] != "pass"
        or decision["response_availability"] != "fail"
        or decision["release_r06j_fragment_erod1_diagnostic"] is not True
        or decision["release_any_further_timestep_halving"] is not False
    ):
        raise AssertionError("Job 72740022 audit does not authorize R06j")


def stage_fragment_erosion_diagnostic(batch_dir: Path, output_dir: Path, audit_path: Path) -> dict:
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
        raise AssertionError("R06j batch is not a single statically valid job")
    source_job = batch_dir / JOB_DIR
    source_manifest = strict_json(source_job / "manifest.json")
    if source_manifest["package_revision"] != REVISION or source_manifest["job_id"] != JOB_ID:
        raise AssertionError("R06j batch identity differs")
    expected_source_files = job_base_files(source_manifest) | {HEAD_DECK, "static_validation.json"}
    actual_source_files = {path.name for path in source_job.iterdir() if path.is_file()}
    if actual_source_files != expected_source_files:
        raise AssertionError("R06j source job does not contain the exact validated whitelist")
    if any(path.is_symlink() or not path.is_file() for path in source_job.iterdir()):
        raise AssertionError("R06j source job must contain ordinary files only")

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
        "schema_version": "hf-p0-r06j-direct-folder-delivery@1.0.0",
        "package_revision": REVISION,
        "job_id": JOB_ID,
        "status": "ready_for_user_submission_not_submitted_not_solver_validated",
        "upload_folder": UPLOAD_DIR,
        "head_deck": HEAD_DECK,
        "archive_policy": "no_archive_direct_folder_only",
        "parent_control": {
            "platform_job_id": "72740022",
            "deck_sha256": PARENT_DECK_SHA256,
            "audit_path": audit_path.as_posix(),
            "audit_sha256": sha256(audit_path),
            "timestep_terminal_consistency": "pass",
            "response_availability": "fail",
        },
        "single_factor": {
            "field": "fragment MAT_015 Card 4 field 3 EROD",
            "control": 0,
            "treatment": 1,
            "plate_erod": 0,
            "tssfac": 0.05,
            "dt2ms": 0.0,
            "meaning": (
                "failed fragment elements are retained and their deviatoric stresses are set to zero"
            ),
        },
        "diagnostic_boundary": {
            "purpose": "isolate whether fragment failure erosion causes upstream deleted mass",
            "erod1_qoi_use": "mechanism_evidence_only_not_physical_label_or_calibration_target",
            "no_automatic_successor_release": True,
        },
        "resource_contract": {
            "software": "ls-dyna_R15_mmp_d",
            "nodes": 1,
            "ranks": 32,
            "walltime": "04:00:00",
            "head_memory_mb": 512,
            "child_memory_mb": 256,
            "command_line_parameters": "",
            "fresh_work_directory": "/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06j_0800_ferod1",
        },
        "files": copied_files,
        "prohibited": [
            "creating or submitting a ZIP archive",
            "submitting any other job in parallel with this diagnostic",
            "changing the post-explosion initial state or any factor other than fragment MAT_015 EROD",
            "changing plate MAT_015 EROD from zero",
            "using nodes/ranks other than one node and 32 total MPP ranks",
            "enabling mass scaling or changing TSSFAC from 0.05",
            "treating an EROD=1 response as physical validation, engine/damage-tree calibration or a dataset label",
        ],
    }
    manifest_path = staging / "DELIVERY_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    instructions = f"""# R06j 碎片侵蚀机制隔离提交说明

不要压缩。只上传整个 `{UPLOAD_DIR}` 文件夹，并在平台选择 `{HEAD_DECK}` 作为 head/input deck。

资源参数：1 个节点、32 MPP ranks、`ls-dyna_R15_mmp_d`、命令行参数留空。使用全新工作目录
`/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06j_0800_ferod1`。

本轮固定 `TSSFAC=0.05`、`DT2MS=0`，唯一变量是碎片材料 MAT_015 Card 4 的 `EROD: 0 -> 1`；
板材 EROD 仍为 0。初始几何、爆炸后破片速度、材料强度/失效参数、接触、网格、终止时间和输出设置不变。

这只是失效后碎片删除机制诊断。EROD=1 会保留失效单元并清零其偏应力，因此所得响应不得作为物理标签、
仿真引擎标定目标或毁伤树标定目标。即使求解正常结束，也必须先审计 HVAR6、删除质量、负体积、能量和接触证据。

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage R06j as a direct folder without ZIP")
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    manifest = stage_fragment_erosion_diagnostic(args.batch_dir, args.output_dir, args.audit)
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
