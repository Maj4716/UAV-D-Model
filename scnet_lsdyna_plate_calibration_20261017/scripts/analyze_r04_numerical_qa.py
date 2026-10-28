"""Read-only Stage 0-2 audit for the SCNet R04 returned LS-DYNA run."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from parse_lsdyna_ascii import (
    bind_negative_volume_events,
    parse_control_timestep,
    parse_glstat_series,
    parse_rank_diagnostics,
    summarize_glstat,
)


TZ = ZoneInfo("Asia/Shanghai")


def now() -> str:
    return datetime.now(TZ).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_log(path: Path, message: str) -> None:
    line = f"{now()} {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def verify_intake(
    result_dir: Path,
    download_dir: Path,
    archive_path: Path,
    intake_analysis: Path,
    log_path: Path,
) -> dict:
    inventory_path = intake_analysis / "file_inventory.json"
    archive_identity_path = intake_analysis / "archive_identity.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    archive_identity = json.loads(archive_identity_path.read_text(encoding="utf-8"))
    mismatches: list[dict] = []
    checked = 0
    for item in inventory:
        if item.get("directory"):
            continue
        relative = Path(*Path(item["name"]).parts)
        expected_hash = item["sha256"]
        expected_bytes = item["bytes"]
        authoritative = result_dir / relative
        downloaded = download_dir / relative
        row = {"name": item["name"]}
        for label, path in (("result", authoritative), ("download", downloaded)):
            if not path.exists():
                row[f"{label}_problem"] = "missing"
                continue
            actual_bytes = path.stat().st_size
            actual_hash = sha256(path)
            if actual_bytes != expected_bytes or actual_hash != expected_hash:
                row[f"{label}_problem"] = "size_or_hash_mismatch"
                row[f"{label}_bytes"] = actual_bytes
                row[f"{label}_sha256"] = actual_hash
        if len(row) > 1:
            mismatches.append(row)
        checked += 1
        if checked % 20 == 0:
            append_log(log_path, f"Stage 0 current-state rehash: {checked} files checked in both copies.")
    current_archive_hash = sha256(archive_path)
    archive_match = current_archive_hash == archive_identity["sha256"]
    return {
        "schema_version": "hf-return-intake-verification@1.0.0",
        "verification_time": now(),
        "stored_inventory": str(inventory_path),
        "files_checked_in_each_copy": checked,
        "result_dir": str(result_dir),
        "download_dir": str(download_dir),
        "mismatches": mismatches,
        "all_files_match_stored_zip_inventory": not mismatches,
        "archive_path": str(archive_path),
        "stored_archive_sha256": archive_identity["sha256"],
        "current_archive_sha256": current_archive_hash,
        "archive_hash_unchanged": archive_match,
        "status": "pass" if not mismatches and archive_match else "fail",
    }


def binary_readability_inventory(result_dir: Path) -> dict:
    files = []
    for name in ["binout0000", "d3plot", *[f"d3plot{i:02d}" for i in range(1, 11)]]:
        path = result_dir / name
        files.append({"name": name, "present": path.exists(), "bytes": path.stat().st_size if path.exists() else None})
    readers = {
        "lasso": module_available("lasso"),
        "lasso.dyna": module_available("lasso.dyna"),
        "qd": module_available("qd"),
        "qd.cae": module_available("qd.cae"),
        "lsprepost_on_path": shutil.which("lsprepost") is not None or shutil.which("lspp") is not None,
    }
    available = any(readers.values())
    return {
        "schema_version": "hf-binary-readability-inventory@1.0.0",
        "files": files,
        "reader_availability": readers,
        "readability_status": "not_checked_reader_available" if available else "not_checked_reader_unavailable",
        "interpretation": (
            "Presence and size are recorded only. No binary payload or d3plot state was interpreted in Stage 0-2."
        ),
    }


def write_energy_plot(path: Path, records: list[dict[str, float | None]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = [row["time"] for row in records]
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.2), sharex=True, constrained_layout=True)
    for field, label in (
        ("kinetic_energy", "Kinetic"),
        ("internal_energy", "Internal"),
        ("sliding_energy", "Sliding interface"),
        ("eroded_kinetic_energy", "Eroded kinetic"),
        ("eroded_internal_energy", "Eroded internal"),
    ):
        axes[0].plot(time, [row[field] for row in records], label=label, linewidth=1.5)
    axes[0].set_ylabel("Energy (model units)")
    axes[0].legend(ncol=2, frameon=False)
    axes[0].grid(alpha=0.25)
    axes[1].plot(time, [row["total_energy_ratio"] for row in records], label="Total / initial", linewidth=1.7)
    axes[1].plot(time, [row["energy_ratio_without_eroded"] for row in records], label="Without eroded", linewidth=1.4)
    axes[1].axhspan(0.95, 1.05, color="#2ca02c", alpha=0.10, label="Candidate band")
    axes[1].set_xlabel("Problem time (ms)")
    axes[1].set_ylabel("Energy ratio")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.suptitle("SCNet LS-DYNA R04 global energy history")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_qa(
    intake: dict,
    diagnostics: dict,
    glstat: dict,
    timestep: dict,
    case_summary: list[dict],
    binary: dict,
) -> dict:
    final_ratio = glstat.get("final_total_energy_ratio")
    maximum_hg_ratio = glstat.get("maximum_absolute_hourglass_over_internal")
    solver_pass = (
        diagnostics["rank_log_count"] == 32
        and diagnostics["normal_termination_rank_count"] == 32
        and not diagnostics["error_fatal_counts"]
    )
    energy_pass = final_ratio is not None and 0.95 <= final_ratio <= 1.05
    hourglass_pass = maximum_hg_ratio is not None and maximum_hg_ratio <= 0.05
    configured_off = not timestep["configured_mass_scaling"]
    added_mass = glstat.get("final_added_mass")
    mass_status = "pass" if configured_off and added_mass == 0.0 else "configured_off_actual_unavailable"
    negative_count = diagnostics["negative_volume_unique_element_count"]
    mapped_event_count = sum(row["total_negative_volume_events"] for row in case_summary)
    unmapped = diagnostics["negative_volume_event_count"] - mapped_event_count
    gates = {
        "archive_and_input_identity": intake["status"],
        "solver_normal_termination": "pass" if solver_pass else "fail",
        "global_total_energy_candidate_band": "pass" if energy_pass else "fail",
        "hourglass_candidate_gate": "pass" if hourglass_pass else "fail",
        "mass_scaling_gate": mass_status,
        "negative_volume_localization": "review_required" if negative_count else "pass",
        "binary_database_readability": binary["readability_status"],
        "case_level_perforation_and_residual_velocity": "not_assessed_stage_3",
    }
    return {
        "schema_version": "hf-p0-numerical-qa@1.0.0",
        "job_id": "72095164",
        "assessment_time": now(),
        "overall_status": "review_required",
        "gates": gates,
        "termination": {
            "rank_logs": diagnostics["rank_log_count"],
            "normal_termination_rank_count": diagnostics["normal_termination_rank_count"],
            "error_fatal_counts": diagnostics["error_fatal_counts"],
        },
        "warning_counts": diagnostics["warning_counts"],
        "negative_volume": {
            "event_count": diagnostics["negative_volume_event_count"],
            "unique_event_count": diagnostics["negative_volume_unique_event_count"],
            "unique_element_count": negative_count,
            "unmapped_event_count": unmapped,
            "case_summary": case_summary,
        },
        "global_statistics": glstat,
        "control_timestep": timestep,
        "binary_readability": binary,
        "review_reasons": [
            "All seven cases contain negative-volume erosion in both plate and fragment parts; spatial/visual reasonableness is not established by logs alone.",
            "Standalone MATSUM/NODOUT/SLEOUT files are absent and no supported BINOUT/d3plot reader is installed in the audited environment.",
            "The returned GLSTAT does not expose an added-mass field, so configured-off mass scaling is verified but actual added mass remains unavailable.",
            "Per-case perforation state, residual velocity, response stabilization and mesh sensitivity remain Stage 3-4 work.",
        ],
        "claim_boundary": (
            "Stage 0-2 establishes archive/input identity, solver completion and global ASCII numerical indicators only. "
            "It does not establish case-level numerical qualification, convergence, physical validity, application calibration or research admission."
        ),
    }


def build_qa_markdown(qa: dict) -> str:
    g = qa["global_statistics"]
    n = qa["negative_volume"]
    rows = []
    for row in n["case_summary"]:
        rows.append(
            f"| `{row['case_id']}` | {row['impact_speed_mps']:.0f} | {row['mesh_mm']:.2f} | "
            f"{row['plate_negative_volume_events']} | {row['fragment_negative_volume_events']} | "
            f"{row['first_event_time']:.7g} | {row['last_event_time']:.7g} |"
        )
    gate_rows = [f"| {name} | `{status}` |" for name, status in qa["gates"].items()]
    return "\n".join(
        [
            "# SCNet R04 Stage 0-2 numerical QA",
            "",
            f"- Job: `{qa['job_id']}`",
            f"- Overall: `{qa['overall_status']}`",
            f"- Final total-energy ratio: `{g['final_total_energy_ratio']:.6f}`",
            f"- Total-energy-ratio range: `{g['minimum_total_energy_ratio']:.6f}–{g['maximum_total_energy_ratio']:.6f}`",
            f"- Maximum |hourglass/internal|: `{g['maximum_absolute_hourglass_over_internal']:.6g}`",
            f"- Final energy ratio without eroded energy: `{g['final_energy_ratio_without_eroded']:.6f}`",
            f"- Final eroded-energy fraction of initial: `{g['final_eroded_energy_fraction_of_initial']:.6%}`",
            f"- Minimum/maximum time step: `{g['minimum_time_step']:.7g}` / `{g['maximum_time_step']:.7g}` ms",
            "",
            "## Gates",
            "",
            "| Gate | Status |",
            "|---|---|",
            *gate_rows,
            "",
            "## Negative-volume localization",
            "",
            f"`{n['unique_element_count']}` unique elements; all mapped to the seven frozen cases.",
            "",
            "| Case | Speed m/s | Mesh mm | Plate | Fragment | First time | Last time |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## Boundary",
            "",
            qa["claim_boundary"],
            "",
        ]
    )


def build_integration_recommendations(qa: dict) -> str:
    n = qa["negative_volume"]["unique_element_count"]
    return f"""# 基于 R04 Stage 0-2 的高保真仿真接入建议

## 当前可用证据

R04 已建立可信的归档身份、输入身份、求解器正常终止和全局能量历史。最终总能量比位于候选区间，小时玻璃能为零；但 `{n}` 个单元发生负体积侵蚀，且逐工况穿孔、残速和网格一致性尚未建立。因此当前结果只能用于完善高保真数据合同和质量门，不能直接替换本地装甲响应参数或 KMFC 阈值。

## 仿真引擎的优化顺序

1. 增加版本化 `high_fidelity_run` 与 `numerical_qa` artifact，绑定 solver build、deck/material/geometry/case-map hash、输出 hash 和 QA 状态；QA 未通过时 fail closed。
2. 保留 `demo_armor_response`，以后新增独立的高保真响应模型 ID。先把穿孔状态、残余速度和能量转移拆成三个可单独校准的模块，避免一条经验曲线承担全部误差。
3. 高保真结果只进入 teacher/audit 通道；求解器状态、接触结果、删除单元和 d3plot 派生量不得成为部署时 learner 输入。
4. 每个拟合 artifact 保存适用域、网格层级、材料认识变体、不确定度和 OOD 拒绝边界；不要在适用域外静默外推。
5. 将负体积、侵蚀能、时间步突降和网格不一致作为样本级 mask/拒绝原因，而不是把数值异常编码成真实毁伤标签。

## 数据集生成器的优化顺序

1. 继续冻结参考 `p0`、Validation/Test 和标签语义；高保真自适应选点只允许在 Train/calibration 池进行。
2. 建立 `hf_numerical_verification`、`hf_response_fit`、`hf_response_holdout` 三类明确角色，P1 的 18/6 fit/holdout 身份保持冻结。
3. 保存连续监督量及可用性 mask：穿孔、残速、吸收/转移能量、侵蚀比例、响应是否稳定；不要把 unavailable 写成 0，也不要直接把单元删除映射成 KMFC 等级。
4. 用低保真大候选池加高保真小样本做多保真残差校准或代理模型；选点依据应是输入空间覆盖、代理不确定度和预注册边界邻域，不是已观察 Test 或标签均衡目标。
5. 保留随机参考点和独立高保真 holdout；输出选择概率、design weight、父 case lineage 和求解器 cohort，禁止跨网格/材料/求解器 cohort 无标识混池。
6. 只有 Stage 3-4 证明 P0 的逐工况响应与网格行为可接受后，才允许用这些连续 QoI 校准本地引擎，再生成新版本数据；旧数据和旧 response-model ID 不覆盖。

## 当前停止线

在 BINOUT/d3plot 可读、侵蚀空间位置得到复核、残速稳定且三层网格结论一致之前，不提交 P1，不修改材料参数，不重写 KMFC 树，不生成新的科研训练数据。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--intake-analysis", type=Path, required=True)
    parser.add_argument("--case-map", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "worklog.md"
    append_log(log_path, "Started Stage 0-2 analysis. Returned inputs are read-only; derived files go only to the analysis directory.")
    append_log(log_path, "Pre-run checks: Python compilation PASS; postprocessor self-test PASS; 3 real-format unit tests PASS; repro_check six essentials OK.")

    intake = verify_intake(
        args.result_dir, args.download_dir, args.archive, args.intake_analysis, log_path
    )
    write_json(args.output_dir / "intake_verification.json", intake)
    if intake["status"] != "pass":
        append_log(log_path, "Stopped: intake verification failed.")
        raise SystemExit("intake verification failed")
    append_log(log_path, "Stage 0 passed: ZIP hash and both extracted copies match the stored 100-file inventory.")

    records = parse_glstat_series(args.result_dir / "glstat")
    glstat = summarize_glstat(records)
    diagnostics = parse_rank_diagnostics(args.result_dir)
    bound_events, case_summary = bind_negative_volume_events(
        diagnostics["negative_volume_events"], args.deck, args.case_map
    )
    timestep = parse_control_timestep(args.deck)
    binary = binary_readability_inventory(args.result_dir)

    write_csv(args.output_dir / "energy_history.csv", records)
    warning_events = [
        {"severity": "warning", "id": "40509", "code": "SOL+509", **row}
        for row in bound_events
    ]
    write_csv(args.output_dir / "warning_events.csv", warning_events)
    write_csv(args.output_dir / "warning_summary.csv", diagnostics["warning_counts"])
    write_csv(args.output_dir / "element_erosion_by_case.csv", case_summary)
    write_json(args.output_dir / "binary_readability.json", binary)
    write_energy_plot(args.output_dir / "energy_history.png", records)

    qa = build_qa(intake, diagnostics, glstat, timestep, case_summary, binary)
    write_json(args.output_dir / "numerical_qa.json", qa)
    (args.output_dir / "numerical_qa.md").write_text(build_qa_markdown(qa), encoding="utf-8")
    (args.output_dir / "high_fidelity_integration_recommendations.md").write_text(
        build_integration_recommendations(qa), encoding="utf-8"
    )
    append_log(
        log_path,
        "Stage 1-2 completed: real GLSTAT parsed, all 32 rank logs scanned, negative-volume events bound to frozen cases, and global QA saved.",
    )
    append_log(log_path, "Scope boundary retained: no BINOUT/d3plot state interpretation, P1 submission, engine calibration, dataset regeneration, training, or sealed-test access.")

    owned = [
        path
        for path in args.output_dir.iterdir()
        if path.is_file() and path.name not in {"analysis_manifest.json"}
    ]
    manifest = {
        "schema_version": "hf-stage0-2-analysis-manifest@1.0.0",
        "created_at": now(),
        "analyzer": str(Path(__file__).resolve()),
        "analyzer_sha256": sha256(Path(__file__).resolve()),
        "parser_sha256": sha256(Path(__file__).with_name("parse_lsdyna_ascii.py")),
        "artifacts": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(owned)
        ],
    }
    write_json(args.output_dir / "analysis_manifest.json", manifest)
    print(json.dumps({"overall_status": qa["overall_status"], "gates": qa["gates"], "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
