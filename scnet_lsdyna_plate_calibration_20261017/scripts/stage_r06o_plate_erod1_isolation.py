from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from build_p0_r06_diagnostic_bundles import finalize_staging_directory
from build_p0_r06o_plate_erod1_isolation import HEAD_DECK, JOB_DIR
from build_r06g_tssfac02_return_audit import sha256, strict_json
from create_p0_r06o_plate_erod1_isolation_design import JOB_ID, MAXIMUM_CYCLE_COUNT, REVISION, TARGET_PLATE_ELEMENT_ID


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = ROOT / "manual_upload_20261102_r06o_plate_erod1_isolation"
DEFAULT_OUTPUT = ROOT / "artifacts" / "r06o_plate_erod1_isolation_submission_20261102"
UPLOAD_DIR = "uav_hf_r06o_0800_plate_erod1_isolation"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def stage_isolation(batch_dir: Path, output_dir: Path) -> dict:
    batch_dir = batch_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    staging = output_dir.parent / f".{output_dir.name}.staging"
    if staging.exists():
        raise FileExistsError(staging)
    readiness = strict_json(batch_dir / "R06O_SUBMISSION_READINESS.json")
    batch_manifest = strict_json(batch_dir / "batch_manifest.json")
    source_job = batch_dir / JOB_DIR
    manifest = strict_json(source_job / "manifest.json")
    static = strict_json(source_job / "static_validation.json")
    expected_differences = ["TITLE_identifier_line", "plate_MAT_JOHNSON_COOK_EROD_field_0_to_1"]
    if (
        readiness["overall_status"] != "pass"
        or readiness["allowed_deck_differences"] != expected_differences
        or readiness["required_resources"] != {"nodes": 1, "ranks": 32}
        or batch_manifest["package_revision"] != REVISION
        or batch_manifest["job_count"] != 1
        or manifest["job_id"] != JOB_ID
        or manifest["package_revision"] != REVISION
        or static["overall_status"] != "pass"
        or static["all_other_deck_lines_byte_identical"] is not True
        or static["plate_erod_candidate"] != 1
    ):
        raise AssertionError("R06o source batch is not the frozen plate EROD=1 isolation")
    if sha256(source_job / HEAD_DECK) != manifest["deck"]["sha256"]:
        raise AssertionError("R06o head-deck hash differs")
    expected_files = {
        HEAD_DECK,
        "p0_r06o_plate_erod1_isolation_design.json",
        "p0_r06o_plate_erod1_output_contract.json",
        "jc4340_reference.json",
        "p0_case_part_map.csv",
        "job_contract.json",
        "static_validation.json",
        "manifest.json",
        "UPLOAD_README.txt",
    }
    actual_files = {path.name for path in source_job.iterdir() if path.is_file()}
    if actual_files != expected_files or any(path.is_symlink() or not path.is_file() for path in source_job.iterdir()):
        raise AssertionError("R06o source job differs from ordinary-file whitelist")

    staging.mkdir(parents=True)
    upload = staging / UPLOAD_DIR
    upload.mkdir()
    for name in sorted(expected_files):
        shutil.copy2(source_job / name, upload / name)
    if any(path.suffix.lower() == ".zip" for path in staging.rglob("*")):
        raise AssertionError("direct-folder delivery must not contain ZIP archives")
    copied_files = [
        {"path": f"{UPLOAD_DIR}/{name}", "size_bytes": (upload / name).stat().st_size, "sha256": sha256(upload / name)}
        for name in sorted(expected_files)
    ]
    delivery = {
        "schema_version": "hf-p0-r06o-direct-folder-delivery@1.0.0",
        "package_revision": REVISION,
        "job_id": JOB_ID,
        "status": "ready_for_user_submission_not_submitted_not_solver_validated",
        "upload_folder": UPLOAD_DIR,
        "head_deck": HEAD_DECK,
        "head_deck_sha256": manifest["deck"]["sha256"],
        "archive_policy": "no_archive_direct_folder_only",
        "single_factor": "plate_MAT_JOHNSON_COOK_EROD_0_to_1",
        "deck_changes": expected_differences,
        "target_plate_element_id": TARGET_PLATE_ELEMENT_ID,
        "resource_contract": {
            "software": "ls-dyna_R15_mmp_d",
            "nodes": 1,
            "ranks": 32,
            "hard_gate": True,
            "head_memory_mb": 512,
            "child_memory_mb": 256,
            "command_line_parameters": "",
            "walltime": "12:00:00",
            "fresh_work_directory": "/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06o_0800_plate_erod1",
        },
        "solver_controls": {"problem_time_ms": 0.04, "maximum_cycle_count": MAXIMUM_CYCLE_COUNT, "tssfac": 0.05, "dt2ms": 0.0},
        "files": copied_files,
        "return_required": [
            "submitted .k", "d3hsp/o= and mes*", "glstat, matsum, sleout, nodout and ncforc", "binout*",
            "all d3plot* and d3part*", "stdout/stderr, Slurm script and resource record",
        ],
        "prohibited": [
            "creating or submitting a ZIP archive",
            "allocating any rank count other than exactly 32",
            "editing any deck field beyond the delivered plate EROD=1 change",
            "extending the 0.04 ms horizon in this isolation run",
            "restarting or tuning after a timestep-collapse result",
            "using this response for engine fitting, damage-tree fitting or a reference label",
        ],
    }
    _write_json(staging / "DELIVERY_MANIFEST.json", delivery)
    instructions = f"""# R06o 板件 MAT EROD=1 隔离提交说明

不要压缩。只上传整个 `{UPLOAD_DIR}` 文件夹，选择 `{HEAD_DECK}` 作为 head/input deck。

硬性资源参数：1 个节点、恰好 32 个 MPP ranks、`ls-dyna_R15_mmp_d`、head memory 512 MB、
child memory 256 MB、命令行参数留空。提交预览必须包含 `--ntasks-per-node=32`。

使用全新工作目录 `/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06o_0800_plate_erod1`。
相对回传 R06n，本轮仅改变 TITLE 和板件 MAT_JOHNSON_COOK 的 EROD 字段 0→1；碎片自适应 SPH、
接触、TSSFAC=0.05、DT2MS=0、0.04 ms 时长及 {MAXIMUM_CYCLE_COUNT:,} 循环保护不变。

本轮可能因保留失效板件单元而发生时间步塌缩。若提前停止或达到循环上限，不要改参数重启；直接下载完整目录。
无论完成或失败，保留 `.k`、`mes*`、`glstat`、`matsum`、`ncforc`、`binout*`、所有 d3plot/d3part、
stdout/stderr、Slurm 脚本和资源记录。之后只需发送回传路径。

这是数值失效路径隔离，不是物理、仿真引擎或毁伤树标定。
"""
    (staging / "SUBMIT_ONLY_THIS_FOLDER.md").write_text(instructions, encoding="utf-8", newline="\n")
    if {path.name for path in staging.iterdir()} != {UPLOAD_DIR, "DELIVERY_MANIFEST.json", "SUBMIT_ONLY_THIS_FOLDER.md"}:
        raise AssertionError("R06o delivery root differs from whitelist")
    for item in copied_files:
        copied = staging / item["path"]
        if copied.stat().st_size != item["size_bytes"] or sha256(copied) != item["sha256"]:
            raise AssertionError(f"copied file identity differs: {item['path']}")
    finalize_staging_directory(staging, output_dir)
    return delivery


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage R06o as a direct folder without ZIP")
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    delivery = stage_isolation(args.batch_dir, args.output_dir)
    print(json.dumps({
        "status": delivery["status"],
        "output": str(args.output_dir.resolve()),
        "upload_folder": delivery["upload_folder"],
        "head_deck": delivery["head_deck"],
        "head_deck_sha256": delivery["head_deck_sha256"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
