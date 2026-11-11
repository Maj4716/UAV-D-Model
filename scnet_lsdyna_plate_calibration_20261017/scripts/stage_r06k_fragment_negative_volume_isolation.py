from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from build_p0_r06_diagnostic_bundles import finalize_staging_directory
from generate_p0_r06_diagnostics import strict_json
from validate_p0_r06_diagnostics import job_base_files, sha256, validate_batch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = ROOT / "manual_upload_20261030_r06k_fragment_negative_volume_isolation"
DEFAULT_OUTPUT = ROOT / "artifacts" / "r06k_fragment_psfail_isolation_submission_20261030"
DEFAULT_AUDIT = (
    ROOT
    / "analysis"
    / "72751725_r06j_0800_fragment_erod1_20261030"
    / "r06j_fragment_erod1_return_audit.json"
)
REVISION = "R06_fragment_negative_volume_isolation_20261030"
JOB_ID = "r06k_0800_fragment_psfail_plate"
JOB_DIR = "job_01_v0800_fragment_psfail_plate"
UPLOAD_DIR = "uav_hf_r06k_0800_fragment_psfail_plate"
HEAD_DECK = "r06k_0800_fragment_psfail_plate.k"
PARENT_DECK_SHA256 = "sha256:8ec1613324159b3ebc00217871ad29321564423b0a7f73a267207e4952bada77"


def _verify_authorization(audit: dict) -> None:
    decision = audit["decision"]
    if (
        audit["source_evidence"]["platform_job_id"] != "72751725"
        or audit["source_evidence"]["deck_sha256"] != PARENT_DECK_SHA256
        or decision["solver_parser_runtime"] != "pass"
        or decision["resource_identity"] != "pass"
        or decision["measurement_channels"] != "pass"
        or decision["fragment_failure_erosion_hypothesis"] != "partial_support"
        or decision["remaining_deletion_mechanism"] != "eroding_contact_automatic_negative_volume_failure"
        or decision["response_availability"] != "fail"
        or decision["release_r06k_fragment_negative_volume_psfail_isolation"] is not True
    ):
        raise AssertionError("Job 72751725 audit does not authorize R06k")


def stage_fragment_negative_volume_diagnostic(batch_dir: Path, output_dir: Path, audit_path: Path) -> dict:
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
        raise AssertionError("R06k batch is not a single statically valid job")
    source_job = batch_dir / JOB_DIR
    source_manifest = strict_json(source_job / "manifest.json")
    if source_manifest["package_revision"] != REVISION or source_manifest["job_id"] != JOB_ID:
        raise AssertionError("R06k batch identity differs")
    expected_source_files = job_base_files(source_manifest) | {HEAD_DECK, "static_validation.json"}
    actual_source_files = {path.name for path in source_job.iterdir() if path.is_file()}
    if actual_source_files != expected_source_files:
        raise AssertionError("R06k source job does not contain the exact validated whitelist")
    if any(path.is_symlink() or not path.is_file() for path in source_job.iterdir()):
        raise AssertionError("R06k source job must contain ordinary files only")

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
        "schema_version": "hf-p0-r06k-direct-folder-delivery@1.0.0",
        "package_revision": REVISION,
        "job_id": JOB_ID,
        "status": "ready_for_user_submission_not_submitted_not_solver_validated",
        "upload_folder": UPLOAD_DIR,
        "head_deck": HEAD_DECK,
        "archive_policy": "no_archive_direct_folder_only",
        "parent_control": {
            "platform_job_id": "72751725",
            "deck_sha256": PARENT_DECK_SHA256,
            "audit_path": audit_path.as_posix(),
            "audit_sha256": sha256(audit_path),
            "fragment_deleted_element_count": 35,
            "fragment_negative_volume_element_count": 35,
            "response_availability": "fail",
        },
        "single_factor": {
            "field": "CONTROL_SOLID Card 1 field 5 PSFAIL",
            "control": 0,
            "treatment": 910001,
            "treatment_part_set": [1002],
            "excluded_fragment_part": 2002,
            "fragment_mat015_erod": 1,
            "plate_mat015_erod": 0,
            "tssfac": 0.05,
            "dt2ms": 0.0,
        },
        "diagnostic_boundary": {
            "purpose": "isolate ERODING-contact automatic negative-volume deletion of fragment elements",
            "possible_outcomes": [
                "normal termination with fragment deletion suppressed",
                "severe distortion or timestep collapse",
                "solver termination",
            ],
            "all_outcomes_are_mechanism_evidence_only": True,
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
            "fresh_work_directory": "/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06k_0800_fpsfail",
        },
        "files": copied_files,
        "prohibited": [
            "creating or submitting a ZIP archive",
            "submitting any other job in parallel with this diagnostic",
            "changing the post-explosion initial state or any factor other than CONTROL_SOLID PSFAIL",
            "changing fragment MAT_015 EROD from one or changing plate MAT_015 EROD from zero",
            "changing the ERODING_SINGLE_SURFACE contact formulation",
            "using nodes/ranks other than one node and 32 total MPP ranks",
            "enabling mass scaling or changing TSSFAC from 0.05",
            "treating a PSFAIL-isolation response as physical validation, engine/damage-tree calibration or a dataset label",
        ],
    }
    manifest_path = staging / "DELIVERY_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    instructions = f"""# R06k 碎片负体积删除机制隔离提交说明

不要压缩。只上传整个 `{UPLOAD_DIR}` 文件夹，并在平台选择 `{HEAD_DECK}` 作为 head/input deck。

资源参数：1 个节点、32 MPP ranks、`ls-dyna_R15_mmp_d`、命令行参数留空。使用全新工作目录
`/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06k_0800_fpsfail`。

本轮保持 `TSSFAC=0.05`、`DT2MS=0`、碎片 MAT_015 `EROD=1`、板材 `EROD=0` 和 ERODING 接触不变。
唯一变量是 `CONTROL_SOLID.PSFAIL: 0 -> 910001`。集合 910001 只包含板材 part 1002，因此碎片 part 2002
被排除在 ERODING 接触自动启用的负体积删除准则之外。

这是机制隔离，不保证正常结束。时间步坍缩、严重畸变或求解器终止同样是需要完整保留的诊断证据，
不得修改模型以追求完成，也不得把所得响应作为物理标签、仿真引擎标定目标或毁伤树标定目标。

完成或中止后都下载完整回传：`.k`、`d3hsp/o=`、`mes*`、`glstat`、`binout*`、所有 `d3plot*`、所有
`d3part*`、stdout/stderr、Slurm 脚本和资源记录。之后只需把回传路径发给我。
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
    parser = argparse.ArgumentParser(description="Stage R06k as a direct folder without ZIP")
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    manifest = stage_fragment_negative_volume_diagnostic(args.batch_dir, args.output_dir, args.audit)
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
