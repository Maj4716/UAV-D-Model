from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from build_p0_r06_diagnostic_bundles import finalize_staging_directory
from build_p0_r06s_1200_speed_stability import HEAD_DECK, JOB_DIR
from build_r06g_tssfac02_return_audit import sha256, strict_json
from create_p0_r06s_1200_speed_stability_design import (
    ASCII_INTERVAL_MS,
    CANDIDATE_BINARY_INTERVAL_MS,
    CANDIDATE_SPEED_MPS,
    JOB_ID,
    MAXIMUM_CYCLE_COUNT,
    PROBLEM_TIME_MS,
    REVISION,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = ROOT / "manual_upload_20261103_r06s_1200_speed_stability_diagnostic"
DEFAULT_OUTPUT = ROOT / "artifacts" / "r06s_1200_speed_stability_submission_20261103"
UPLOAD_DIR = "uav_hf_r06s_1200_speed_stability_diagnostic"


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
    readiness = strict_json(batch_dir / "R06S_SUBMISSION_READINESS.json")
    batch_manifest = strict_json(batch_dir / "batch_manifest.json")
    source_job = batch_dir / JOB_DIR
    manifest = strict_json(source_job / "manifest.json")
    static = strict_json(source_job / "static_validation.json")
    expected_differences = [
        "TITLE_identifier_line",
        "DATABASE_BINARY_D3PLOT_dt_0p0001_to_0p00015",
        "DATABASE_BINARY_D3PART_dt_0p0001_to_0p00015",
        "INITIAL_VELOCITY_GENERATION_part_2002_vz_minus2000_to_minus1200",
    ]
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
        or static["single_physical_factor"] != "fragment_initial_speed_mps_2000_to_1200"
        or static["termination_time_ms"] != PROBLEM_TIME_MS
        or static["maximum_cycle_count"] != MAXIMUM_CYCLE_COUNT
        or static["impact_speed_mps"] != CANDIDATE_SPEED_MPS
        or static["d3plot_interval_ms"] != CANDIDATE_BINARY_INTERVAL_MS
        or static["d3part_interval_ms"] != CANDIDATE_BINARY_INTERVAL_MS
        or static["ascii_interval_ms"] != ASCII_INTERVAL_MS
    ):
        raise AssertionError("R06s source batch is not the frozen 1200 m/s stability diagnostic")
    if sha256(source_job / HEAD_DECK) != manifest["deck"]["sha256"]:
        raise AssertionError("R06s head-deck hash differs")
    expected_files = {
        HEAD_DECK,
        "p0_r06s_1200_speed_stability_design.json",
        "p0_r06s_1200_speed_stability_output_contract.json",
        "jc4340_reference.json",
        "p0_case_part_map.csv",
        "job_contract.json",
        "static_validation.json",
        "manifest.json",
        "UPLOAD_README.txt",
    }
    actual_files = {path.name for path in source_job.iterdir() if path.is_file()}
    if actual_files != expected_files or any(path.is_symlink() or not path.is_file() for path in source_job.iterdir()):
        raise AssertionError("R06s source job differs from ordinary-file whitelist")

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
        "schema_version": "hf-p0-r06s-direct-folder-delivery@1.0.0",
        "package_revision": REVISION,
        "job_id": JOB_ID,
        "status": "ready_for_user_submission_not_submitted_not_solver_validated",
        "upload_folder": UPLOAD_DIR,
        "head_deck": HEAD_DECK,
        "head_deck_sha256": manifest["deck"]["sha256"],
        "archive_policy": "no_archive_direct_folder_only",
        "physical_factor": "fragment_initial_speed_2000_to_1200_mps_stability_diagnostic",
        "measurement_controls": "D3PLOT_and_D3PART_0p0001_to_0p00015_ms",
        "deck_changes": expected_differences,
        "resource_contract": {
            "software": "ls-dyna_R15_mmp_d",
            "nodes": 1,
            "ranks": 32,
            "hard_gate": True,
            "head_memory_mb": 512,
            "child_memory_mb": 256,
            "command_line_parameters": "",
            "walltime": "12:00:00",
            "fresh_work_directory": "/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06s_1200_speed_stability_diagnostic",
        },
        "solver_controls": {"problem_time_ms": PROBLEM_TIME_MS, "maximum_cycle_count": MAXIMUM_CYCLE_COUNT, "tssfac": 0.05, "dt2ms": 0.0},
        "output_intervals_ms": {"d3plot": CANDIDATE_BINARY_INTERVAL_MS, "d3part": CANDIDATE_BINARY_INTERVAL_MS, "ascii_databases": ASCII_INTERVAL_MS},
        "planning_estimates": {
            "projected_d3plot_state_count": 3335,
            "projected_d3part_state_count": 3335,
            "projected_output_total_gb_decimal": 22.51,
            "projected_maximum_downward_step_mm": 0.205736867056,
            "required_maximum_downward_step_mm": 0.25,
            "claim_boundary": "planning estimates only; actual R06s output must pass",
        },
        "staged_download_policy": {
            "stage1_only_first": [
                "submitted deck", "manifest.json", "job_contract.json", "Slurm script", "platform log",
                "std.out", "std.err", "glstat", "all mes0000 through mes0031",
            ],
            "expected_stage1_size_mib": 26,
            "do_not_download_initially": ["d3plot*", "d3part*", "binout*"],
        },
        "files": copied_files,
        "prohibited": [
            "creating or submitting a ZIP archive",
            "allocating any rank count other than exactly 32",
            "editing any delivered deck field",
            "downloading heavy binary families before the Stage-1 gate is reviewed",
            "using R06s directly for engine fitting, damage-tree fitting or a reference label",
        ],
    }
    _write_json(staging / "DELIVERY_MANIFEST.json", delivery)
    instructions = f"""# R06s 1200 m/s 数值稳定性诊断提交说明

不要压缩。只上传整个 `{UPLOAD_DIR}` 文件夹，选择 `{HEAD_DECK}` 作为 head/input deck。

硬性资源参数：1 个节点、恰好 32 个 MPP ranks、`ls-dyna_R15_mmp_d`、head memory 512 MB、
child memory 256 MB、命令行参数留空、walltime 12:00:00。提交预览必须包含 `--ntasks-per-node=32`。

使用全新工作目录 `/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06s_1200_speed_stability_diagnostic`。
唯一物理因子是 part 2002 初速度从 2000 改为 1200 m/s；D3PLOT/D3PART 0.00015 ms 是测量控制。
终止时刻 0.50 ms、ENDCYC={MAXIMUM_CYCLE_COUNT:,}、ASCII 0.00025 ms、TSSFAC=0.05、DT2MS=0，
以及材料、接触、自适应 SPH、网格和 32 ranks 均冻结不变。

计算完成后不要立即下载大文件。第一阶段只下载：提交 deck、manifest.json、job_contract.json、Slurm 脚本、
平台 log、std.out、std.err、glstat、mes0000 至 mes0031。先发送该小文件目录路径；不要下载 d3plot、d3part 或 binout。

本轮用于判断2000 m/s的板件负体积和滑移能失败是否局限于极端速度，不是物理验证，也不是标定。
"""
    (staging / "SUBMIT_ONLY_THIS_FOLDER.md").write_text(instructions, encoding="utf-8", newline="\n")
    if {path.name for path in staging.iterdir()} != {UPLOAD_DIR, "DELIVERY_MANIFEST.json", "SUBMIT_ONLY_THIS_FOLDER.md"}:
        raise AssertionError("R06s delivery root differs from whitelist")
    for item in copied_files:
        copied = staging / item["path"]
        if copied.stat().st_size != item["size_bytes"] or sha256(copied) != item["sha256"]:
            raise AssertionError(f"copied file identity differs: {item['path']}")
    finalize_staging_directory(staging, output_dir)
    return delivery


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage R06s as a direct folder without ZIP")
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
