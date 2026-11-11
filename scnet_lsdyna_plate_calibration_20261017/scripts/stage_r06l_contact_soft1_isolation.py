from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from build_p0_r06_diagnostic_bundles import finalize_staging_directory
from generate_p0_r06_diagnostics import strict_json
from validate_p0_r06_diagnostics import job_base_files, sha256, validate_batch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = ROOT / "manual_upload_20261031_r06l_contact_soft1_isolation"
DEFAULT_OUTPUT = ROOT / "artifacts" / "r06l_contact_soft1_submission_20261031"
DEFAULT_AUDIT = (
    ROOT
    / "analysis"
    / "72764387_r06k_0800_fragment_psfail_plate_20261031"
    / "r06k_fragment_psfail_return_audit.json"
)
REVISION = "R06_contact_soft1_isolation_20261031"
JOB_ID = "r06l_0800_contact_soft1"
JOB_DIR = "job_01_v0800_contact_soft1"
UPLOAD_DIR = "uav_hf_r06l_0800_contact_soft1"
HEAD_DECK = "r06l_0800_contact_soft1.k"
PARENT_DECK_SHA256 = "sha256:f78396a9fd15114fe411507a7774d68cecb10a25460413474772f2b99e8a17ca"


def _verify_authorization(audit: dict) -> None:
    decision = audit["decision"]
    if (
        audit["source_evidence"]["platform_job_id"] != "72764387"
        or audit["source_evidence"]["deck_sha256"] != PARENT_DECK_SHA256
        or decision["solver_parser_runtime"] != "pass"
        or decision["resource_identity"] != "pass"
        or decision["measurement_channels"] != "pass"
        or decision["fragment_negative_volume_hypothesis"] != "supported"
        or decision["fragment_deletion_suppression"] != "pass"
        or decision["numerical_stability"] != "fail_contact_energy_and_timestep_collapse"
        or decision["release_r06l_contact_soft1_isolation"] is not True
    ):
        raise AssertionError("Job 72764387 audit does not authorize R06l")


def stage_contact_soft1_diagnostic(batch_dir: Path, output_dir: Path, audit_path: Path) -> dict:
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
        raise AssertionError("R06l batch is not a single statically valid job")
    source_job = batch_dir / JOB_DIR
    source_manifest = strict_json(source_job / "manifest.json")
    if source_manifest["package_revision"] != REVISION or source_manifest["job_id"] != JOB_ID:
        raise AssertionError("R06l batch identity differs")
    expected_source_files = job_base_files(source_manifest) | {HEAD_DECK, "static_validation.json"}
    actual_source_files = {path.name for path in source_job.iterdir() if path.is_file()}
    if actual_source_files != expected_source_files:
        raise AssertionError("R06l source job does not contain the exact validated whitelist")
    if any(path.is_symlink() or not path.is_file() for path in source_job.iterdir()):
        raise AssertionError("R06l source job must contain ordinary files only")

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
    parent_stability = audit["formal_fragment_negative_volume_comparison_to_job_72751725"]["performance_and_numerical_stability"]
    manifest = {
        "schema_version": "hf-p0-r06l-direct-folder-delivery@1.0.0",
        "package_revision": REVISION,
        "job_id": JOB_ID,
        "status": "ready_for_user_submission_not_submitted_not_solver_validated",
        "upload_folder": UPLOAD_DIR,
        "head_deck": HEAD_DECK,
        "archive_policy": "no_archive_direct_folder_only",
        "parent_control": {
            "platform_job_id": "72764387",
            "deck_sha256": PARENT_DECK_SHA256,
            "audit_path": audit_path.as_posix(),
            "audit_sha256": sha256(audit_path),
            "fragment_deleted_element_count": 0,
            "fragment_negative_volume_element_count": 0,
            "cycle_count": parent_stability["treatment_cycle_count"],
            "minimum_time_step_ms": parent_stability["minimum_time_step_ms"],
            "final_sliding_energy_fraction_of_initial": audit["returned_outputs"]["glstat"]["final_sliding_energy_fraction_of_initial"],
            "numerical_stability": "fail",
        },
        "single_factor": {
            "field": "CONTACT_ERODING_SINGLE_SURFACE Optional Card A field 1 SOFT",
            "control": 2,
            "treatment": 1,
            "psfail_part_set_id": 910001,
            "fragment_mat015_erod": 1,
            "plate_mat015_erod": 0,
            "tssfac": 0.05,
            "dt2ms": 0.0,
        },
        "diagnostic_boundary": {
            "purpose": "isolate whether SOFT=1 removes the R06k contact-energy and timestep instability",
            "candidate_gates": {
                "fragment_deleted_element_count": 0,
                "fragment_negative_volume_element_count": 0,
                "total_energy_ratio_range": [0.95, 1.05],
                "absolute_final_sliding_energy_fraction_of_initial_max": 0.05,
                "cycle_count_max": int(parent_stability["treatment_cycle_count"] // 10),
                "minimum_time_step_ms_min": parent_stability["minimum_time_step_ms"] * 10.0,
            },
            "all_outcomes_are_numerical_mechanism_evidence_only": True,
            "no_automatic_successor_release": True,
        },
        "resource_contract": {
            "software": "ls-dyna_R15_mmp_d",
            "nodes": 1,
            "ranks": 32,
            "head_memory_mb": 512,
            "child_memory_mb": 256,
            "command_line_parameters": "",
            "fresh_work_directory": "/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06l_0800_soft1",
        },
        "files": copied_files,
        "prohibited": [
            "creating or submitting a ZIP archive",
            "submitting another job in parallel with this diagnostic",
            "changing any factor other than contact Optional Card A SOFT",
            "changing PSFAIL=910001, fragment EROD=1, plate EROD=0, TSSFAC=0.05 or DT2MS=0",
            "using nodes/ranks other than one node and 32 total MPP ranks",
            "treating a SOFT=1 response as physical validation, engine/damage-tree calibration or a dataset label",
        ],
    }
    manifest_path = staging / "DELIVERY_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    instructions = f"""# R06l 接触 SOFT=1 单因素诊断提交说明

不要压缩。只上传整个 `{UPLOAD_DIR}` 文件夹，并在平台选择 `{HEAD_DECK}` 作为 head/input deck。

资源参数：1 个节点、32 MPP ranks、`ls-dyna_R15_mmp_d`、head memory 512 MB、child memory 256 MB、
命令行参数留空。使用全新工作目录 `/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06l_0800_soft1`。

本轮继续从爆炸瞬间后的同一碎片状态开始。保持 `PSFAIL=910001`、`TSSFAC=0.05`、`DT2MS=0`、
碎片 MAT_015 `EROD=1`、板材 `EROD=0` 不变。唯一变量是 ERODING_SINGLE_SURFACE Optional Card A
的 `SOFT: 2 -> 1`。

这是数值机制诊断，不是物理标定。即使正常结束，也必须检查碎片删除/负体积、总能量比、滑移能量、
最小时间步和循环数；不得把本轮响应直接作为仿真引擎或毁伤树的标定目标。

完成或中止后都下载完整回传：`.k`、`d3hsp/o=`、`mes*`、`glstat`、`binout*`、所有 `d3plot*`、
所有 `d3part*`、stdout/stderr、Slurm 脚本和资源记录。之后只需把回传路径发给我。
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
    parser = argparse.ArgumentParser(description="Stage R06l as a direct folder without ZIP")
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    manifest = stage_contact_soft1_diagnostic(args.batch_dir, args.output_dir, args.audit)
    print(json.dumps({"status": manifest["status"], "output": str(args.output_dir.resolve()), "upload_folder": manifest["upload_folder"], "head_deck": manifest["head_deck"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
