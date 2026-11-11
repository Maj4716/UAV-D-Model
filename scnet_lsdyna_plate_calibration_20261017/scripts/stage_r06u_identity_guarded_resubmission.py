from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from build_p0_r06_diagnostic_bundles import finalize_staging_directory
from build_r06g_tssfac02_return_audit import sha256, strict_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "artifacts" / "r06u_plate_adaptive_sph_canary_submission_20261104" / "uav_hf_r06u_1200_plate_adaptive_sph_canary"
DEFAULT_OUTPUT = ROOT / "artifacts" / "r06u_plate_adaptive_sph_canary_resubmission_20261104"
UPLOAD_DIR = "uav_hf_r06u_1200_plate_adaptive_sph_canary_retry"
HEAD_DECK = "r06u_1200_plate_adaptive_sph_canary.k"
HEAD_DECK_SHA256 = "sha256:7bfb99a49bb757632237b2e2c625500fef27e811988eac3149df7a3e056647d4"
FRESH_WORK_DIR = "/work/home/acd7dt5yv4/LS_DYNA/uav_hf_r06u_1200_plate_adaptive_sph_canary_retry"
EXPECTED_FILES = {
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


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def stage_resubmission(source_dir: Path, output_dir: Path) -> dict:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    staging = output_dir.parent / f".{output_dir.name}.staging"
    if staging.exists():
        raise FileExistsError(staging)

    manifest = strict_json(source_dir / "manifest.json")
    contract = strict_json(source_dir / "job_contract.json")
    static = strict_json(source_dir / "static_validation.json")
    actual_files = {path.name for path in source_dir.iterdir() if path.is_file()}
    if actual_files != EXPECTED_FILES or any(path.is_symlink() or not path.is_file() for path in source_dir.iterdir()):
        raise AssertionError("frozen R06u source differs from ordinary-file whitelist")
    if (
        manifest["job_id"] != "r06u_1200_plate_adaptive_sph_canary"
        or manifest["deck"]["path"] != HEAD_DECK
        or manifest["deck"]["sha256"] != HEAD_DECK_SHA256
        or sha256(source_dir / HEAD_DECK) != HEAD_DECK_SHA256
        or contract["head_deck"] != HEAD_DECK
        or contract["case"]["termination_time_ms"] != 0.15
        or contract["case"]["control_solid_psfail"] != 0
        or static["overall_status"] != "pass"
        or static["plate_adaptive_sph"]["solid_part_id"] != 1002
        or static["plate_adaptive_sph"]["sph_part_id"] != 1003
        or static["all_other_keyword_blocks_identical"] is not True
    ):
        raise AssertionError("source is not the frozen, validated R06u canary")

    staging.mkdir(parents=True)
    upload = staging / UPLOAD_DIR
    upload.mkdir()
    for name in sorted(EXPECTED_FILES):
        shutil.copy2(source_dir / name, upload / name)
    copied_files = [
        {
            "path": f"{UPLOAD_DIR}/{name}",
            "size_bytes": (upload / name).stat().st_size,
            "sha256": sha256(upload / name),
            "byte_identical_to_frozen_r06u": (upload / name).read_bytes() == (source_dir / name).read_bytes(),
        }
        for name in sorted(EXPECTED_FILES)
    ]
    if not all(item["byte_identical_to_frozen_r06u"] for item in copied_files):
        raise AssertionError("resubmission copy differs from frozen R06u source")
    if any(path.suffix.lower() == ".zip" for path in staging.rglob("*")):
        raise AssertionError("direct-folder delivery must not contain ZIP archives")

    delivery = {
        "schema_version": "hf-p0-r06u-identity-guarded-resubmission@1.0.0",
        "created_date": "2026-11-04",
        "status": "ready_for_user_resubmission_not_submitted_not_solver_validated",
        "reason": "platform job 73171751 was labelled R06u but executed the exact frozen R06t deck",
        "model_change_from_original_r06u": "none_exact_byte_identical_resubmission",
        "upload_folder": UPLOAD_DIR,
        "head_deck": HEAD_DECK,
        "head_deck_sha256": HEAD_DECK_SHA256,
        "fresh_work_directory": FRESH_WORK_DIR,
        "resource_contract": {
            "software": "ls-dyna_R15_mmp_d",
            "nodes": 1,
            "ranks": 32,
            "head_memory_mb": 512,
            "child_memory_mb": 256,
            "command_line_parameters": "",
            "walltime": "12:00:00",
        },
        "pre_submit_identity_gate": [
            f"selected input filename must equal {HEAD_DECK}",
            f"selected input SHA-256 must equal {HEAD_DECK_SHA256}",
            "selected input filename must not contain r06t or psfail",
            f"platform work directory must equal {FRESH_WORK_DIR}",
            "submission preview command input must end in /r06u_1200_plate_adaptive_sph_canary.k",
        ],
        "staged_download_policy": {
            "stage1_only_first": [
                "submitted deck", "manifest.json", "job_contract.json", "Slurm script", "platform log",
                "std.out", "std.err", "glstat", "matsum", "sleout", "nodout", "ncforc",
                "sldsph_frag if present", "all mes0000 through mes0031",
            ],
            "do_not_download_initially": ["d3plot*", "d3part*", "binout*"],
        },
        "files": copied_files,
        "prohibited": [
            "selecting r06t_1200_fragment_only_psfail_diagnostic.k",
            "reusing the previous platform work directory",
            "editing any delivered file",
            "creating or submitting a ZIP archive",
            "using any rank count other than exactly 32",
            "fitting the engine, damage tree or reference labels from this capability canary",
        ],
    }
    _write_json(staging / "RESUBMISSION_MANIFEST.json", delivery)
    instructions = f"""# R06u 原样重提交：提交前身份核对

本文件夹是对正确 R06u 的逐字节原样重交付；没有修改任何模型参数。

1. 不要压缩，只上传整个 `{UPLOAD_DIR}` 文件夹。
2. 必须选择 `{HEAD_DECK}` 作为 head/input deck。
3. 该 deck 的 SHA-256 必须是 `{HEAD_DECK_SHA256}`。
4. 若界面出现 `r06t` 或 `psfail` 文件名，立即取消提交；那是旧输入卡。
5. 使用全新工作目录 `{FRESH_WORK_DIR}`，不要复用上次目录。
6. 提交预览中的 `i=` 必须以 `/{HEAD_DECK}` 结尾。

资源仍为：1 节点、恰好 32 ranks、`ls-dyna_R15_mmp_d`、head memory 512 MB、
child memory 256 MB、命令行参数留空、walltime 12:00:00。

R06u 终止时刻应为 0.15 ms，`CONTROL_SOLID.PSFAIL=0`，板件 part 1002 失败后转入同材料 SPH part 1003。
本轮只验证能否越过 R06t 的 0.10872 ms 崩溃点，不用于物理有效性、仿真引擎标定或毁伤树标定。

回传仍按小文件优先：deck、manifest、job_contract、Slurm、平台 log、stdout/stderr、
glstat/matsum/sleout/nodout/ncforc、若有 sldsph_frag、mes0000--mes0031；先不要下载 d3plot/d3part/binout。
"""
    (staging / "IDENTITY_CHECK_BEFORE_SUBMIT.md").write_text(instructions, encoding="utf-8", newline="\n")
    if {path.name for path in staging.iterdir()} != {UPLOAD_DIR, "RESUBMISSION_MANIFEST.json", "IDENTITY_CHECK_BEFORE_SUBMIT.md"}:
        raise AssertionError("resubmission delivery root differs from whitelist")
    finalize_staging_directory(staging, output_dir)
    return delivery


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage an identity-guarded byte-identical R06u resubmission")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    delivery = stage_resubmission(args.source_dir, args.output_dir)
    print(json.dumps({
        "status": delivery["status"],
        "output": str(args.output_dir.resolve()),
        "upload_folder": delivery["upload_folder"],
        "head_deck": delivery["head_deck"],
        "head_deck_sha256": delivery["head_deck_sha256"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
