from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from build_p0_r06_diagnostic_bundles import finalize_staging_directory
from build_p0_r06u_plate_adaptive_sph_canary import HEAD_DECK, JOB_DIR
from build_r06g_tssfac02_return_audit import sha256, strict_json
from create_p0_r06u_plate_adaptive_sph_canary_design import (
    ASCII_INTERVAL_MS,
    BINARY_INTERVAL_MS,
    JOB_ID,
    MAXIMUM_CYCLE_COUNT,
    PROBLEM_TIME_MS,
    REVISION,
    SPEED_MPS,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = ROOT / "manual_upload_20261104_r06u_plate_adaptive_sph_canary"
DEFAULT_OUTPUT = ROOT / "artifacts" / "r06u_plate_adaptive_sph_canary_submission_20261104"
UPLOAD_DIR = "uav_hf_r06u_1200_plate_adaptive_sph_canary"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def stage_submission(batch_dir: Path, output_dir: Path) -> dict:
    batch_dir = batch_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    staging = output_dir.parent / f".{output_dir.name}.staging"
    if staging.exists():
        raise FileExistsError(staging)
    readiness = strict_json(batch_dir / "R06U_SUBMISSION_READINESS.json")
    batch_manifest = strict_json(batch_dir / "batch_manifest.json")
    source_job = batch_dir / JOB_DIR
    manifest = strict_json(source_job / "manifest.json")
    static = strict_json(source_job / "static_validation.json")
    if (
        readiness["overall_status"] != "pass"
        or readiness["required_resources"] != {"nodes": 1, "ranks": 32}
        or batch_manifest["package_revision"] != REVISION
        or batch_manifest["job_count"] != 1
        or manifest["job_id"] != JOB_ID
        or manifest["package_revision"] != REVISION
        or static["overall_status"] != "pass"
        or static["all_other_keyword_blocks_identical"] is not True
        or static["control_solid_psfail"] != 0
        or static["plate_adaptive_sph"] != {"solid_part_id": 1002, "sph_part_id": 1003, "particles_per_failed_hexahedron": 8, "material_id": 1, "eos_id": 1}
        or static["termination_time_ms"] != PROBLEM_TIME_MS
        or static["maximum_cycle_count"] != MAXIMUM_CYCLE_COUNT
        or static["impact_speed_mps"] != SPEED_MPS
        or static["d3plot_interval_ms"] != BINARY_INTERVAL_MS
        or static["d3part_interval_ms"] != BINARY_INTERVAL_MS
        or static["ascii_interval_ms"] != ASCII_INTERVAL_MS
    ):
        raise AssertionError("R06u source batch is not the frozen plate adaptive-SPH canary")
    if sha256(source_job / HEAD_DECK) != manifest["deck"]["sha256"]:
        raise AssertionError("R06u head-deck hash differs")
    expected_files = {
        HEAD_DECK,
        "p0_r06u_plate_adaptive_sph_canary_design.json",
        "p0_r06u_plate_adaptive_sph_canary_output_contract.json",
        "jc4340_reference.json",
        "p0_case_part_map.csv",
        "job_contract.json",
        "static_validation.json",
        "manifest.json",
        "UPLOAD_README.txt",
    }
    actual_files = {path.name for path in source_job.iterdir() if path.is_file()}
    if actual_files != expected_files or any(path.is_symlink() or not path.is_file() for path in source_job.iterdir()):
        raise AssertionError("R06u source job differs from ordinary-file whitelist")

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
        "schema_version": "hf-p0-r06u-direct-folder-delivery@1.0.0",
        "package_revision": REVISION,
        "job_id": JOB_ID,
        "status": "ready_for_user_submission_not_submitted_not_solver_validated",
        "upload_folder": UPLOAD_DIR,
        "head_deck": HEAD_DECK,
        "head_deck_sha256": manifest["deck"]["sha256"],
        "archive_policy": "no_archive_direct_folder_only",
        "model_form_canary": "failed_plate_solids_part_1002_to_same-material_SPH_part_1003",
        "resource_contract": {
            "software": "ls-dyna_R15_mmp_d",
            "nodes": 1,
            "ranks": 32,
            "hard_gate": True,
            "head_memory_mb": 512,
            "child_memory_mb": 256,
            "command_line_parameters": "",
            "walltime": "12:00:00",
            "fresh_work_directory": "/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06u_1200_plate_adaptive_sph_canary",
        },
        "solver_controls": {"problem_time_ms": PROBLEM_TIME_MS, "maximum_cycle_count": MAXIMUM_CYCLE_COUNT, "tssfac": 0.05, "dt2ms": 0.0},
        "output_intervals_ms": {"d3plot": BINARY_INTERVAL_MS, "d3part": BINARY_INTERVAL_MS, "ascii_databases": ASCII_INTERVAL_MS},
        "planning_estimates": {"projected_output_total_gb_decimal": 6.753, "claim_boundary": "linear horizon scaling only; actual R06u output can differ"},
        "staged_download_policy": {
            "stage1_only_first": [
                "submitted deck", "manifest.json", "job_contract.json", "Slurm script", "platform log",
                "std.out", "std.err", "glstat", "matsum", "sleout", "nodout", "ncforc",
                "sldsph_frag if present", "all mes0000 through mes0031",
            ],
            "expected_stage1_size_mib": 35,
            "do_not_download_initially": ["d3plot*", "d3part*", "binout*"],
        },
        "files": copied_files,
        "prohibited": [
            "creating or submitting a ZIP archive",
            "allocating any rank count other than exactly 32",
            "editing any delivered deck field",
            "downloading heavy binary families before Stage 1 is reviewed",
            "using R06u directly for engine fitting, damage-tree fitting or a reference label",
        ],
    }
    _write_json(staging / "DELIVERY_MANIFEST.json", delivery)
    instructions = f"""# R06u 板件自适应 SPH 能力验证提交说明

不要压缩。只上传整个 `{UPLOAD_DIR}` 文件夹，选择 `{HEAD_DECK}` 作为 head/input deck。

硬性资源参数：1 个节点、恰好 32 个 MPP ranks、`ls-dyna_R15_mmp_d`、head memory 512 MB、
child memory 256 MB、命令行参数留空、walltime 12:00:00。提交预览必须包含 `--ntasks-per-node=32`。

使用全新工作目录 `/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06u_1200_plate_adaptive_sph_canary`。
本算例从已完成的 R06s 分支，PSFAIL 恢复为 0，并增加板件失败单元 part 1002 → 同材料 SPH part 1003 的局部转换。
速度 1200 m/s；短时终点 {PROBLEM_TIME_MS:.2f} ms，超过 R06t 的 0.10872 ms 崩溃点；ENDCYC={MAXIMUM_CYCLE_COUNT:,}。
D3PLOT/D3PART 0.00015 ms、ASCII 0.00025 ms、TSSFAC=0.05、DT2MS=0，其他材料、网格和资源均冻结。

计算完成后不要立即下载大文件。第一阶段只下载：提交 deck、manifest.json、job_contract.json、Slurm 脚本、
平台 log、std.out、std.err、glstat、matsum、sleout、nodout、ncforc、若存在则 sldsph_frag，以及 mes0000–mes0031。
先发送该小文件目录路径；不要下载 d3plot、d3part 或 binout。

本轮只验证板件局部 SPH 替代是否能越过 R06t 崩溃点，不是物理验证，也不是仿真引擎或毁伤树标定。
"""
    (staging / "SUBMIT_ONLY_THIS_FOLDER.md").write_text(instructions, encoding="utf-8", newline="\n")
    if {path.name for path in staging.iterdir()} != {UPLOAD_DIR, "DELIVERY_MANIFEST.json", "SUBMIT_ONLY_THIS_FOLDER.md"}:
        raise AssertionError("R06u delivery root differs from whitelist")
    for item in copied_files:
        copied = staging / item["path"]
        if copied.stat().st_size != item["size_bytes"] or sha256(copied) != item["sha256"]:
            raise AssertionError(f"copied file identity differs: {item['path']}")
    finalize_staging_directory(staging, output_dir)
    return delivery


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage R06u as a direct folder without ZIP")
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    delivery = stage_submission(args.batch_dir, args.output_dir)
    print(json.dumps({
        "status": delivery["status"],
        "output": str(args.output_dir.resolve()),
        "upload_folder": delivery["upload_folder"],
        "head_deck": delivery["head_deck"],
        "head_deck_sha256": delivery["head_deck_sha256"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
