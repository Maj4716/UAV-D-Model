from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from build_p0_r06_diagnostic_bundles import finalize_staging_directory
from build_p0_r06m_adaptive_sph_canary import HEAD_DECK, JOB_DIR
from create_p0_r06m_adaptive_sph_canary_design import JOB_ID, MAXIMUM_CYCLE_COUNT, REVISION
from generate_p0_r06_diagnostics import strict_json
from validate_p0_r06_diagnostics import sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = ROOT / "manual_upload_20261101_r06m_adaptive_sph_canary"
DEFAULT_OUTPUT = ROOT / "artifacts" / "r06m_adaptive_sph_canary_submission_20261101"
UPLOAD_DIR = "uav_hf_r06m_0800_adaptive_sph_canary"


def stage_adaptive_sph_canary(batch_dir: Path, output_dir: Path) -> dict:
    batch_dir = batch_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite direct-folder delivery: {output_dir}")
    staging = output_dir.parent / f".{output_dir.name}.staging"
    if staging.exists():
        raise FileExistsError(f"delivery staging directory already exists: {staging}")

    readiness = strict_json(batch_dir / "R06M_SUBMISSION_READINESS.json")
    batch_manifest = strict_json(batch_dir / "batch_manifest.json")
    source_job = batch_dir / JOB_DIR
    job_manifest = strict_json(source_job / "manifest.json")
    static_validation = strict_json(source_job / "static_validation.json")
    if (
        readiness["overall_status"] != "pass"
        or batch_manifest["package_revision"] != REVISION
        or batch_manifest["job_count"] != 1
        or job_manifest["job_id"] != JOB_ID
        or job_manifest["package_revision"] != REVISION
        or static_validation["overall_status"] != "pass"
    ):
        raise AssertionError("R06m source batch is not the frozen statically valid canary")
    if sha256(source_job / HEAD_DECK) != job_manifest["deck"]["sha256"]:
        raise AssertionError("R06m head-deck identity differs from its manifest")

    expected_source_files = {
        HEAD_DECK,
        "p0_r06m_adaptive_sph_canary_design.json",
        "p0_r06m_adaptive_sph_output_contract.json",
        "jc4340_reference.json",
        "p0_case_part_map.csv",
        "job_contract.json",
        "static_validation.json",
        "manifest.json",
        "UPLOAD_README.txt",
    }
    actual_source_files = {path.name for path in source_job.iterdir() if path.is_file()}
    if actual_source_files != expected_source_files:
        raise AssertionError("R06m source job differs from the exact validated whitelist")
    if any(path.is_symlink() or not path.is_file() for path in source_job.iterdir()):
        raise AssertionError("R06m source job must contain ordinary files only")

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
    delivery = {
        "schema_version": "hf-p0-r06m-direct-folder-delivery@1.0.0",
        "package_revision": REVISION,
        "job_id": JOB_ID,
        "status": "ready_for_user_submission_not_submitted_not_solver_validated",
        "upload_folder": UPLOAD_DIR,
        "head_deck": HEAD_DECK,
        "head_deck_sha256": job_manifest["deck"]["sha256"],
        "archive_policy": "no_archive_direct_folder_only",
        "branch_parent": job_manifest["immutable_parent"],
        "capability_bundle": {
            "failed_fragment_solid_part_id": 2002,
            "adaptive_sph_part_id": 2003,
            "particles_per_failed_hexahedron": 8,
            "solid_contact_soft": 2,
            "dedicated_sph_to_solid_contact_soft": 1,
            "solid_contact_part_set_id": 920001,
            "maximum_cycle_count": MAXIMUM_CYCLE_COUNT,
            "required_problem_time_ms": 0.04,
        },
        "resource_contract": {
            "software": "ls-dyna_R15_mmp_d",
            "nodes": 1,
            "ranks": 32,
            "head_memory_mb": 512,
            "child_memory_mb": 256,
            "command_line_parameters": "",
            "fresh_work_directory": "/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06m_0800_adaptive_sph",
        },
        "files": copied_files,
        "return_required": [
            "submitted .k",
            "d3hsp/o= and mes*",
            "glstat, matsum, sleout, nodout and ncforc",
            "binout*",
            "all d3plot* and d3part*",
            "stdout/stderr, Slurm script and resource record",
        ],
        "prohibited": [
            "creating or submitting a ZIP archive",
            "submitting another high-fidelity calibration job in parallel",
            "changing the deck, contact cards, cycle guard, output cards or resource identity",
            "calling a cycle-guard exit a completed response",
            "using this capability-canary response for engine fitting, damage-tree fitting or a reference label",
        ],
    }
    _write_json(staging / "DELIVERY_MANIFEST.json", delivery)
    instructions = f"""# R06m 自适应实体转 SPH 能力验证提交说明

不要压缩。只上传整个 `{UPLOAD_DIR}` 文件夹，在平台选择 `{HEAD_DECK}` 作为 head/input deck。

资源参数：1 个节点、32 MPP ranks、`ls-dyna_R15_mmp_d`、head memory 512 MB、child memory 256 MB，
命令行参数留空。使用全新工作目录 `/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06m_0800_adaptive_sph`。

本轮从与 R06j 相同的爆炸瞬间后状态开始，800 m/s、TSSFAC=0.05、DT2MS=0。碎片实体 2002
失效后转换为 SPH part 2003，每个六面体生成 8 个粒子。实体接触仍为 SOFT=2；SOFT=1 只用于
SPH 对实体的专用非对称节点-面接触，因此不是重复已经失败的 R06l 全局 SOFT=1 路线。

防失控边界：必须在 {MAXIMUM_CYCLE_COUNT:,} 个循环以内到达问题时间 0.04 ms。如果先达到循环上限，
该次只能记为能力验证失败/部分结果，不算正常完成，不要重启或擅自修改参数，直接下载完整回传。

无论完成、提前停机还是解析失败，都下载完整目录，尤其保留 `.k`、`d3hsp/o=`、`mes*`、`glstat`、
`matsum`、`ncforc`、`binout*`、所有 `d3plot*`、所有 `d3part*`、stdout/stderr、Slurm 脚本和资源记录。
之后只需把回传路径发给我。

这是解析/运行/质量-能量/输出能力验证，不是物理标定；不得把响应直接用于仿真引擎或毁伤树标定。
"""
    (staging / "SUBMIT_ONLY_THIS_FOLDER.md").write_text(instructions, encoding="utf-8", newline="\n")

    expected_root = {UPLOAD_DIR, "DELIVERY_MANIFEST.json", "SUBMIT_ONLY_THIS_FOLDER.md"}
    if {path.name for path in staging.iterdir()} != expected_root:
        raise AssertionError("R06m delivery root differs from the exact whitelist")
    for item in copied_files:
        copied = staging / item["path"]
        if copied.stat().st_size != item["size_bytes"] or sha256(copied) != item["sha256"]:
            raise AssertionError(f"copied file identity differs: {item['path']}")
    finalize_staging_directory(staging, output_dir)
    return delivery


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage R06m as a direct folder without ZIP")
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    delivery = stage_adaptive_sph_canary(args.batch_dir, args.output_dir)
    print(
        json.dumps(
            {
                "status": delivery["status"],
                "output": str(args.output_dir.resolve()),
                "upload_folder": delivery["upload_folder"],
                "head_deck": delivery["head_deck"],
                "head_deck_sha256": delivery["head_deck_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
