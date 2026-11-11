from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from build_p0_r06_diagnostic_bundles import finalize_staging_directory
from build_p0_r06n_adaptive_sph_32rank_confirmation import HEAD_DECK, JOB_DIR
from build_r06g_tssfac02_return_audit import sha256, strict_json
from create_p0_r06n_adaptive_sph_32rank_confirmation_design import JOB_ID, MAXIMUM_CYCLE_COUNT, REVISION


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = ROOT / "manual_upload_20261101_r06n_adaptive_sph_32rank_confirmation"
DEFAULT_OUTPUT = ROOT / "artifacts" / "r06n_adaptive_sph_32rank_confirmation_submission_20261101"
UPLOAD_DIR = "uav_hf_r06n_0800_adaptive_sph_32rank_confirmation"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def stage_confirmation(batch_dir: Path, output_dir: Path) -> dict:
    batch_dir = batch_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    staging = output_dir.parent / f".{output_dir.name}.staging"
    if staging.exists():
        raise FileExistsError(staging)
    readiness = strict_json(batch_dir / "R06N_SUBMISSION_READINESS.json")
    batch_manifest = strict_json(batch_dir / "batch_manifest.json")
    source_job = batch_dir / JOB_DIR
    job_manifest = strict_json(source_job / "manifest.json")
    static_validation = strict_json(source_job / "static_validation.json")
    if (
        readiness["overall_status"] != "pass"
        or readiness["title_only_deck_change"] is not True
        or readiness["required_resources"] != {"nodes": 1, "ranks": 32}
        or batch_manifest["package_revision"] != REVISION
        or batch_manifest["job_count"] != 1
        or job_manifest["job_id"] != JOB_ID
        or job_manifest["package_revision"] != REVISION
        or static_validation["overall_status"] != "pass"
        or static_validation["solver_model_output_cards_byte_identical"] is not True
    ):
        raise AssertionError("R06n source batch is not the frozen title-only 32-rank confirmation")
    if sha256(source_job / HEAD_DECK) != job_manifest["deck"]["sha256"]:
        raise AssertionError("R06n head-deck hash differs from manifest")
    expected_files = {
        HEAD_DECK,
        "p0_r06n_adaptive_sph_32rank_confirmation_design.json",
        "p0_r06n_adaptive_sph_32rank_output_contract.json",
        "jc4340_reference.json",
        "p0_case_part_map.csv",
        "job_contract.json",
        "static_validation.json",
        "manifest.json",
        "UPLOAD_README.txt",
    }
    actual_files = {path.name for path in source_job.iterdir() if path.is_file()}
    if actual_files != expected_files or any(path.is_symlink() or not path.is_file() for path in source_job.iterdir()):
        raise AssertionError("R06n source job differs from the exact ordinary-file whitelist")

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
        "schema_version": "hf-p0-r06n-direct-folder-delivery@1.0.0",
        "package_revision": REVISION,
        "job_id": JOB_ID,
        "status": "ready_for_user_submission_not_submitted_not_solver_validated",
        "upload_folder": UPLOAD_DIR,
        "head_deck": HEAD_DECK,
        "head_deck_sha256": job_manifest["deck"]["sha256"],
        "archive_policy": "no_archive_direct_folder_only",
        "single_factor": "actual_mpp_rank_count_64_to_exactly_32",
        "deck_change": "TITLE_identifier_line_only",
        "resource_contract": {
            "software": "ls-dyna_R15_mmp_d",
            "nodes": 1,
            "ranks": 32,
            "hard_gate": True,
            "head_memory_mb": 512,
            "child_memory_mb": 256,
            "command_line_parameters": "",
            "walltime": "12:00:00",
            "fresh_work_directory": "/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06n_0800_adaptive_sph_32rank",
        },
        "solver_controls": {
            "problem_time_ms": 0.04,
            "maximum_cycle_count": MAXIMUM_CYCLE_COUNT,
            "tssfac": 0.05,
            "dt2ms": 0.0,
        },
        "files": copied_files,
        "return_required": [
            "submitted .k", "d3hsp/o= and mes*", "glstat, matsum, sleout, nodout and ncforc", "binout*",
            "all d3plot* and d3part*", "stdout/stderr, Slurm script and resource record",
        ],
        "prohibited": [
            "creating or submitting a ZIP archive",
            "allocating any rank count other than exactly 32",
            "editing model, contact, termination or output cards",
            "extending the 0.04 ms horizon in this resource-confirmation run",
            "using this response for engine fitting, damage-tree fitting or a reference label",
        ],
    }
    _write_json(staging / "DELIVERY_MANIFEST.json", delivery)
    instructions = f"""# R06n 精确 32-rank 自适应 SPH 确认提交说明

不要压缩。只上传整个 `{UPLOAD_DIR}` 文件夹，选择 `{HEAD_DECK}` 作为 head/input deck。

硬性资源参数：1 个节点、**恰好 32 个 MPP ranks**、`ls-dyna_R15_mmp_d`、head memory 512 MB、
child memory 256 MB、命令行参数留空。请在提交预览/生成的 Slurm 脚本中确认
`--ntasks-per-node=32`；若平台自动改为 64，不要把该作业视为本轮有效确认。

使用全新工作目录 `/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06n_0800_adaptive_sph_32rank`。
本轮仍从爆炸瞬间后的状态开始，模型、接触、输出、0.04 ms 时长和 {MAXIMUM_CYCLE_COUNT:,} 循环保护均不变；
与回传 R06m 的输入 deck 相比只有 TITLE 标识行改变。本轮不要顺便延长时长，否则会把资源和时长两个因素混在一起。

无论完成、提前停机或平台资源不符，都下载完整作业目录。之后只需把回传路径发给我。
这是资源一致性与工程复现确认，不是物理标定；不得用于仿真引擎或毁伤树拟合。
"""
    (staging / "SUBMIT_ONLY_THIS_FOLDER.md").write_text(instructions, encoding="utf-8", newline="\n")
    if {path.name for path in staging.iterdir()} != {UPLOAD_DIR, "DELIVERY_MANIFEST.json", "SUBMIT_ONLY_THIS_FOLDER.md"}:
        raise AssertionError("R06n delivery root differs from whitelist")
    for item in copied_files:
        copied = staging / item["path"]
        if copied.stat().st_size != item["size_bytes"] or sha256(copied) != item["sha256"]:
            raise AssertionError(f"copied file identity differs: {item['path']}")
    finalize_staging_directory(staging, output_dir)
    return delivery


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage R06n as a direct folder without ZIP")
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    delivery = stage_confirmation(args.batch_dir, args.output_dir)
    print(json.dumps({
        "status": delivery["status"],
        "output": str(args.output_dir.resolve()),
        "upload_folder": delivery["upload_folder"],
        "head_deck": delivery["head_deck"],
        "head_deck_sha256": delivery["head_deck_sha256"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
