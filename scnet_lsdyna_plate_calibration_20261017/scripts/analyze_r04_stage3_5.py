"""Read-only Stage 3-5 post-processing for the SCNet R04 P0 LS-DYNA run."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np


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


def load_cases(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("plate_part_id", "fragment_part_id", "fragment_center_history_node_id"):
            row[key] = int(row[key])
        for key in ("impact_speed_mps", "mesh_mm", "symmetry_scale_factor"):
            row[key] = float(row[key])
        for key in ("plate_elements", "fragment_elements"):
            row[key] = int(row[key])
    return rows


def install_vendor_path(vendor_dir: Path) -> None:
    if not vendor_dir.is_dir():
        raise FileNotFoundError(f"reader vendor directory not found: {vendor_dir}")
    sys.path.insert(0, str(vendor_dir.resolve()))


def distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def relative_difference(value: float, reference: float) -> float | None:
    if not math.isfinite(value) or not math.isfinite(reference) or reference == 0.0:
        return None
    return abs(value - reference) / abs(reference)


def is_monotonic(values: list[float]) -> bool:
    return all(b >= a for a, b in zip(values, values[1:])) or all(
        b <= a for a, b in zip(values, values[1:])
    )


def alive_mask(encoded: np.ndarray) -> np.ndarray:
    """LASSO stores zero for deleted and part-index+1 for live solids."""
    return np.asarray(encoded) != 0.0


def final_window_statistics(time: np.ndarray, values: np.ndarray, start_ms: float = 0.018) -> dict:
    mask = time >= start_ms
    selected_t = time[mask]
    selected = values[mask]
    if selected.size < 2:
        return {
            "window_start_ms": start_ms,
            "sample_count": int(selected.size),
            "range": None,
            "relative_range": None,
            "linear_slope_per_ms": None,
        }
    mean = float(np.mean(selected))
    return {
        "window_start_ms": start_ms,
        "sample_count": int(selected.size),
        "range": float(np.ptp(selected)),
        "relative_range": float(np.ptp(selected) / max(abs(mean), 1e-12)),
        "linear_slope_per_ms": float(np.polyfit(selected_t, selected, 1)[0]),
    }


def element_reference_volumes(coords: np.ndarray, connectivity: np.ndarray) -> np.ndarray:
    p1 = coords[connectivity[:, 0]]
    p2 = coords[connectivity[:, 1]]
    p4 = coords[connectivity[:, 3]]
    p5 = coords[connectivity[:, 4]]
    return np.einsum("ij,ij->i", p2 - p1, np.cross(p4 - p1, p5 - p1))


def resolve_node_state_coordinates(
    reference_coordinates: np.ndarray, node_state_array: np.ndarray
) -> tuple[np.ndarray, str]:
    """Handle d3plot files that store absolute coordinates under LASSO's displacement key."""
    if np.allclose(node_state_array[0], reference_coordinates, rtol=1e-9, atol=1e-9):
        return node_state_array, "absolute_coordinates"
    if np.allclose(node_state_array[0], 0.0, rtol=0.0, atol=1e-9):
        return reference_coordinates[None, :, :] + node_state_array, "displacement_from_reference"
    raise ValueError(
        "node state array is neither absolute coordinates nor displacement from reference"
    )


def nearest_center_mismatch_count(
    final_xy: np.ndarray, own_center: np.ndarray, all_centers: np.ndarray
) -> int:
    distances = np.linalg.norm(final_xy[:, None, :] - all_centers[None, :, :], axis=2)
    own_index = int(np.argmin(np.linalg.norm(all_centers - own_center, axis=1)))
    return int(np.count_nonzero(np.argmin(distances, axis=1) != own_index))


def build_mesh_assessment(case_rows: list[dict]) -> dict:
    rows = sorted(
        (row for row in case_rows if row["impact_speed_mps"] == 2000.0),
        key=lambda row: row["mesh_mm"],
        reverse=True,
    )
    if [row["mesh_mm"] for row in rows] != [1.0, 0.75, 0.5]:
        raise ValueError("expected frozen 1.00/0.75/0.50 mm mesh triplet")
    speeds = [row["survivor_average_residual_speed_mps"] for row in rows]
    masses = [row["fragment_surviving_mass_fraction"] for row in rows]
    center_valid = [row["center_history_node_qoi_available"] for row in rows]
    proxies = [row["perforation_proxy"] for row in rows]
    coarse, medium, fine = rows
    return {
        "schema_version": "hf-p0-mesh-assessment@1.0.0",
        "ordered_mesh_mm": [row["mesh_mm"] for row in rows],
        "perforation_proxy_consistent": len(set(proxies)) == 1,
        "perforation_proxy_values": proxies,
        "center_history_node_qoi_available": center_valid,
        "center_history_node_gate": "unavailable" if not all(center_valid) else "available",
        "survivor_average_residual_speed_mps": speeds,
        "survivor_speed_monotonic_with_refinement": is_monotonic(speeds),
        "relative_difference_coarse_to_medium": relative_difference(
            coarse["survivor_average_residual_speed_mps"], medium["survivor_average_residual_speed_mps"]
        ),
        "relative_difference_medium_to_fine": relative_difference(
            medium["survivor_average_residual_speed_mps"], fine["survivor_average_residual_speed_mps"]
        ),
        "fragment_surviving_mass_fraction": masses,
        "mass_fraction_spread_percentage_points": 100.0 * (max(masses) - min(masses)),
        "pre_registered_numeric_error_band_found": False,
        "response_consistency_gate": "fail",
        "reasons": [
            "The frozen design/report requires a preregistered residual-velocity error band but contains no numeric tolerance.",
            "The center history node has no live supporting solid at the final state for the 1.00 and 0.75 mm cases.",
            "The surviving-fragment part-average residual speed changes strongly across the three meshes.",
            "Fragment surviving mass fraction is strongly mesh dependent, so the part-average speed does not compare like-for-like residue.",
        ],
    }


def build_speed_assessment(case_rows: list[dict]) -> dict:
    rows = sorted(
        (row for row in case_rows if row["mesh_mm"] == 0.75),
        key=lambda row: row["impact_speed_mps"],
    )
    proxies = [row["perforation_proxy"] for row in rows]
    speeds = [row["survivor_average_residual_speed_mps"] for row in rows]
    has_perforated = any(proxies)
    has_unperforated = any(not value for value in proxies)
    return {
        "schema_version": "hf-p0-speed-assessment@1.0.0",
        "impact_speed_mps": [row["impact_speed_mps"] for row in rows],
        "perforation_proxy": proxies,
        "has_perforated_point": has_perforated,
        "has_unperforated_point": has_unperforated,
        "perforation_transition_bracketed": has_perforated and has_unperforated,
        "survivor_average_residual_speed_mps": speeds,
        "survivor_speed_monotonic_with_impact_speed": is_monotonic(speeds),
        "fragment_surviving_mass_fraction": [row["fragment_surviving_mass_fraction"] for row in rows],
        "velocity_response_fit_gate": "fail" if not (has_perforated and has_unperforated) else "review",
        "reasons": [
            "All five medium-mesh cases place surviving fragment material beyond the initial rear face; no non-perforated boundary point is present.",
            "The surviving-fragment part-average speed is non-monotonic because surviving mass changes substantially with impact speed.",
            "A deterministic all-perforated scan cannot identify an experimental V50 and does not bracket even a numerical transition proxy.",
        ],
    }


def write_response_plot(path: Path, case_rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    medium = sorted(
        (row for row in case_rows if row["mesh_mm"] == 0.75),
        key=lambda row: row["impact_speed_mps"],
    )
    mesh = sorted(
        (row for row in case_rows if row["impact_speed_mps"] == 2000.0),
        key=lambda row: row["mesh_mm"],
        reverse=True,
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), layout="constrained")
    axes[0, 0].plot(
        [row["impact_speed_mps"] for row in medium],
        [row["survivor_average_residual_speed_mps"] for row in medium],
        color="#0072B2", marker="o", label="Surviving-part average",
    )
    label_used = False
    for row in medium:
        if row["center_history_node_qoi_available"]:
            axes[0, 0].scatter(
                row["impact_speed_mps"], row["center_history_node_residual_speed_mps"],
                color="#D55E00", marker="s",
                label="Live-supported center node" if not label_used else None,
            )
            label_used = True
    axes[0, 0].plot(
        [row["impact_speed_mps"] for row in medium],
        [row["impact_speed_mps"] for row in medium],
        color="#666666", linestyle="--", label="Incident speed",
    )
    axes[0, 0].set(xlabel="Incident speed (m/s)", ylabel="Residual-speed proxy (m/s)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].plot(
        [row["impact_speed_mps"] for row in medium],
        [100.0 * row["fragment_surviving_mass_fraction"] for row in medium],
        color="#009E73", marker="^",
    )
    axes[0, 1].set(xlabel="Incident speed (m/s)", ylabel="Surviving fragment mass (%)", ylim=(0, 105))
    axes[1, 0].plot(
        [row["mesh_mm"] for row in mesh],
        [row["survivor_average_residual_speed_mps"] for row in mesh],
        color="#0072B2", marker="o", label="Surviving-part average",
    )
    valid = [row for row in mesh if row["center_history_node_qoi_available"]]
    axes[1, 0].scatter(
        [row["mesh_mm"] for row in valid],
        [row["center_history_node_residual_speed_mps"] for row in valid],
        color="#D55E00", marker="s", label="Live-supported center node",
    )
    axes[1, 0].invert_xaxis()
    axes[1, 0].set(xlabel="Mesh size (mm; finer to the right)", ylabel="Residual-speed proxy (m/s)")
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 1].plot(
        [row["mesh_mm"] for row in mesh],
        [100.0 * row["fragment_surviving_mass_fraction"] for row in mesh],
        color="#009E73", marker="^", label="Mass",
    )
    axes[1, 1].plot(
        [row["mesh_mm"] for row in mesh],
        [100.0 * row["fragment_surviving_reference_volume_fraction"] for row in mesh],
        color="#CC79A7", marker="D", linestyle="--", label="Reference volume",
    )
    axes[1, 1].invert_xaxis()
    axes[1, 1].set(xlabel="Mesh size (mm; finer to the right)", ylabel="Surviving fragment (%)", ylim=(0, 105))
    axes[1, 1].legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.25)
    fig.suptitle("R04 case-level response diagnostics (unavailable center-node QoI omitted)")
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def write_spatial_plot(
    path: Path,
    cases: list[dict],
    case_rows: list[dict],
    element_part_indexes: np.ndarray,
    part_ids: np.ndarray,
    final_centroids: np.ndarray,
    final_alive: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_by_case = {row["case_id"]: row for row in case_rows}
    fig, axes = plt.subplots(4, 2, figsize=(11.0, 15.0), layout="constrained")
    for ax, case in zip(axes.flat, cases):
        row = row_by_case[case["case_id"]]
        cx, cy = row["initial_center_x_mm"], row["initial_center_y_mm"]
        radius = np.hypot(final_centroids[:, 0] - cx, final_centroids[:, 1] - cy)
        for part_id, label, color in (
            (case["plate_part_id"], "plate live", "#777777"),
            (case["fragment_part_id"], "fragment live", "#E69F00"),
        ):
            part_index = int(np.where(part_ids == part_id)[0][0])
            part = element_part_indexes == part_index
            live = part & final_alive
            dead = part & ~final_alive
            ax.scatter(radius[live], final_centroids[live, 2], s=2.0, alpha=0.35, color=color, label=label)
            ax.scatter(
                radius[dead], final_centroids[dead, 2], s=7.0, alpha=0.45,
                color="#D55E00", marker="x", label="deleted" if part_id == case["plate_part_id"] else None,
            )
        ax.axhline(0.0, color="#333333", linestyle=":", linewidth=0.9)
        ax.axhline(-10.0, color="#333333", linestyle="--", linewidth=0.9)
        ax.set(xlim=(0, 25), ylim=(-50, 10), xlabel="Radial distance from impact axis (mm)", ylabel="z (mm)")
        ax.set_title(f"{case['case_id']} | proxy={row['perforation_proxy']}", fontsize=9)
        ax.grid(alpha=0.18)
    axes.flat[0].legend(frameon=False, fontsize=7, ncol=2)
    axes.flat[-1].axis("off")
    fig.suptitle("R04 final-state r-z review: live and deleted solid centroids")
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def build_report(assessment: dict, case_rows: list[dict], mesh: dict, speed: dict) -> str:
    case_table = []
    for row in case_rows:
        center_value = (
            f"{row['center_history_node_residual_speed_mps']:.1f}"
            if row["center_history_node_qoi_available"] else "不可用"
        )
        case_table.append(
            f"| `{row['case_id']}` | {row['impact_speed_mps']:.0f} | {row['mesh_mm']:.2f} | "
            f"{row['survivor_average_residual_speed_mps']:.1f} | {center_value} | "
            f"{100*row['fragment_surviving_mass_fraction']:.1f}% | "
            f"{100*row['plate_deleted_reference_volume_fraction']:.3f}% | {row['perforation_proxy']} |"
        )
    gate_rows = [f"| {name} | `{status}` |" for name, status in assessment["gates"].items()]
    return "\n".join(
        [
            "# SCNet R04 阶段 3–5 逐工况响应、网格判定与准入结论",
            "",
            f"- 作业：`{assessment['job_id']}`",
            f"- 总结论：`{assessment['overall_status']}`",
            f"- P0 响应标定准入：`{assessment['calibration_admission']}`",
            f"- P1 提交准入：`{assessment['p1_submission_admission']}`",
            "",
            "## 阶段 3：数据库与逐工况提取",
            "",
            "`binout0000` 与完整 `d3plot*` 文件族均由项目局部 LASSO 2.0.4 成功打开。BINOUT 实际目录为 "
            "`JOBINFO/glstat/matsum/nodout/sleout`；d3plot 识别 184,949 节点、163,730 实体单元和 14 个 part。",
            "",
            "| 工况 | 入速 m/s | 网格 mm | 存活 part 平均残速 m/s | 中心节点残速 | 破片存活质量 | 板删除参考体积 | 穿孔代理 |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
            *case_table,
            "",
            "中心历史节点只有在末态仍有存活实体单元支撑时才作为可用 QoI。其余工况虽然 NODOUT 仍保留数值，"
            "但节点支撑已全部删除，记录可能是冻结值，已按 unavailable 处理。",
            "",
            "穿孔代理定义为：仍存活的破片实体质心越过初始板背面且存活 part 的平均 z 速度继续向后。"
            "它是数值几何代理，不是试验穿孔判据。",
            "",
            "## 阶段 4：速度与网格行为",
            "",
            f"- 速度扫描穿孔转变是否被括住：`{speed['perforation_transition_bracketed']}`；五个中网格点均为穿孔代理。",
            f"- 存活 part 平均残速是否随入速单调：`{speed['survivor_speed_monotonic_with_impact_speed']}`。",
            f"- 三网格存活 part 平均残速：`{mesh['survivor_average_residual_speed_mps']}` m/s（1.00/0.75/0.50 mm）。",
            f"- 粗→中相对差：`{mesh['relative_difference_coarse_to_medium']:.2%}`；中→细相对差：`{mesh['relative_difference_medium_to_fine']:.2%}`。",
            f"- 三网格破片存活质量率：`{[round(100*x, 2) for x in mesh['fragment_surviving_mass_fraction']]}`%；跨网格极差 `{mesh['mass_fraction_spread_percentage_points']:.2f}` 个百分点。",
            f"- 中心节点三网格 QoI 可用性：`{mesh['center_history_node_qoi_available']}`。",
            "- 设计文档要求“进入预注册误差带”，但设计 JSON 与报告中没有发现数值误差带；不能事后选择宽松阈值宣告通过。",
            "",
            "## 阶段 5：门控结论",
            "",
            "| Gate | 状态 |",
            "|---|---|",
            *gate_rows,
            "",
            "P0 的文件身份、求解完成和全局能量门继续成立；但响应层资格失败：速度扫描没有未穿孔点，"
            "三网格残速与存活质量高度敏感，中心历史节点大多因支撑删除而不可用。"
            "因此本次 R04 只能作为读取链、侵蚀和网格问题的数值诊断证据，不能进入装甲响应拟合、V50 标定、KMFC 阈值修改或科研数据生成。",
            "",
            "## 下一轮最小修复设计",
            "",
            "1. 在提交前明确残余速度主 QoI（推荐固定捕获面上的质量—速度联合分布及守恒闭合），不要依赖可能被删除的单个中心节点。",
            "2. 扩展更低速度直到至少得到一个可靠未穿孔点；冻结后再运行，不依据结果临时挑点。",
            "3. 对 2000 m/s 网格组统一输出存活/侵蚀质量、动量和捕获面数据，并预注册数值容差。",
            "4. 先用最小三网格复算闭合响应；通过前继续禁止 P1、材料反调、训练数据再生成和 sealed Test 使用。",
            "",
            "## 结论边界",
            "",
            assessment["claim_boundary"],
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--case-map", type=Path, required=True)
    parser.add_argument("--stage0-2-qa", type=Path, required=True)
    parser.add_argument("--vendor-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "worklog.md"
    append_log(log_path, "Started Stage 3-5. Original returned files are opened read-only; all outputs are derived artifacts.")
    append_log(log_path, "Dependency note: full isolated dependency copying was stopped after filesystem unpacking stalled; no target was created. Minimal local lasso-python 2.0.4 + h5py 3.16.0 then succeeded.")
    install_vendor_path(args.vendor_dir)
    from lasso.dyna import Binout, D3plot
    import lasso

    environment = {
        "schema_version": "hf-reader-environment@1.0.0",
        "created_at": now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "vendor_dir": str(args.vendor_dir.resolve()),
        "lasso_import_path": str(Path(lasso.__file__).resolve()),
        "versions": {
            name: distribution_version(name)
            for name in (
                "lasso-python", "h5py", "numpy", "scipy", "pandas", "plotly",
                "psutil", "rich", "scikit-learn", "matplotlib",
            )
        },
        "note": "LASSO and h5py are project-local. Other numerical packages come from the project interpreter; Plotly/psutil/rich are not used by this extraction path.",
    }
    write_json(args.output_dir / "reader_environment.json", environment)

    cases = load_cases(args.case_map)
    stage0_2 = json.loads(args.stage0_2_qa.read_text(encoding="utf-8"))
    append_log(log_path, "Opening BINOUT and enumerating real database directories.")
    binout = Binout(str(args.result_dir / "binout*"))
    binout_dirs = binout.read()
    binout_inventory = {directory: binout.read(directory) for directory in binout_dirs}

    append_log(log_path, "Opening the complete d3plot family and reading first/last selected response states.")
    d3plot = D3plot(
        str(args.result_dir / "d3plot"),
        state_array_filter=[
            "timesteps", "node_displacement", "element_solid_is_alive", "part_mass",
            "part_velocity", "part_internal_energy", "part_kinetic_energy",
        ],
        state_filter={0, -1},
        buffered_reading=True,
    )

    arrays = d3plot.arrays
    node_ids = np.asarray(arrays["node_ids"], dtype=np.int64)
    node_coords = np.asarray(arrays["node_coordinates"], dtype=float)
    node_displacement = np.asarray(arrays["node_displacement"], dtype=float)
    connectivity = np.asarray(arrays["element_solid_node_indexes"], dtype=np.int64)
    element_part_indexes = np.asarray(arrays["element_solid_part_indexes"], dtype=np.int64)
    part_ids = np.asarray(arrays["part_ids"], dtype=np.int64)
    encoded_alive = np.asarray(arrays["element_solid_is_alive"], dtype=float)
    final_alive = alive_mask(encoded_alive[-1])
    reference_volume = element_reference_volumes(node_coords, connectivity)
    if np.any(reference_volume <= 0.0):
        raise ValueError("non-positive reference volume found in d3plot geometry")
    node_state_coordinates, node_state_mode = resolve_node_state_coordinates(
        node_coords, node_displacement
    )
    final_coords = node_state_coordinates[-1]
    final_centroids = final_coords[connectivity].mean(axis=1)

    nodout_ids = np.asarray(binout.read("nodout", "ids"), dtype=np.int64)
    nodout_time = np.asarray(binout.read("nodout", "time"), dtype=float)
    nodout_x = np.asarray(binout.read("nodout", "x_coordinate"), dtype=float)
    nodout_y = np.asarray(binout.read("nodout", "y_coordinate"), dtype=float)
    nodout_z = np.asarray(binout.read("nodout", "z_coordinate"), dtype=float)
    nodout_vz = np.asarray(binout.read("nodout", "z_velocity"), dtype=float)
    matsum_ids = np.asarray(binout.read("matsum", "ids"), dtype=np.int64)
    matsum_time = np.asarray(binout.read("matsum", "time"), dtype=float)
    matsum_vz = np.asarray(binout.read("matsum", "z_rbvelocity"), dtype=float)
    matsum_ke = np.asarray(binout.read("matsum", "kinetic_energy"), dtype=float)
    matsum_ie = np.asarray(binout.read("matsum", "internal_energy"), dtype=float)
    matsum_eroded_ke = np.asarray(binout.read("matsum", "eroded_kinetic_energy"), dtype=float)
    matsum_eroded_ie = np.asarray(binout.read("matsum", "eroded_internal_energy"), dtype=float)
    matsum_z_momentum = np.asarray(binout.read("matsum", "z_momentum"), dtype=float)
    if not np.allclose(nodout_time, matsum_time):
        raise ValueError("NODOUT and MATSUM time axes differ")

    node_lookup = {int(value): index for index, value in enumerate(node_ids)}
    nodout_lookup = {int(value): index for index, value in enumerate(nodout_ids)}
    matsum_lookup = {int(value): index for index, value in enumerate(matsum_ids)}
    part_lookup = {int(value): index for index, value in enumerate(part_ids)}
    case_centers = np.asarray(
        [
            [
                nodout_x[0, nodout_lookup[case["fragment_center_history_node_id"]]],
                nodout_y[0, nodout_lookup[case["fragment_center_history_node_id"]]],
            ]
            for case in cases
        ], dtype=float,
    )

    part_mass = np.asarray(arrays["part_mass"], dtype=float)
    part_velocity = np.asarray(arrays["part_velocity"], dtype=float)
    part_ie = np.asarray(arrays["part_internal_energy"], dtype=float)
    part_ke = np.asarray(arrays["part_kinetic_energy"], dtype=float)
    matsum_in_d3plot_order = np.asarray([matsum_lookup[int(part_id)] for part_id in part_ids])
    ordered_matsum_vz = matsum_vz[:, matsum_in_d3plot_order]
    ordered_matsum_ke = matsum_ke[:, matsum_in_d3plot_order]
    ordered_matsum_ie = matsum_ie[:, matsum_in_d3plot_order]
    ordered_matsum_momentum = matsum_z_momentum[:, matsum_in_d3plot_order]
    derived_final_mass = np.divide(
        ordered_matsum_momentum[-1],
        ordered_matsum_vz[-1],
        out=np.full(part_ids.shape, np.nan, dtype=float),
        where=np.abs(ordered_matsum_vz[-1]) > 1e-12,
    )
    valid_mass = np.isfinite(derived_final_mass)
    mass_relative_error = np.abs(
        derived_final_mass[valid_mass] - part_mass[-1, valid_mass]
    ) / np.maximum(np.abs(part_mass[-1, valid_mass]), 1e-12)
    cross_database = {
        "final_part_z_velocity_max_abs_difference": float(
            np.max(np.abs(part_velocity[-1, :, 2] - ordered_matsum_vz[-1]))
        ),
        "final_part_kinetic_energy_max_abs_difference": float(
            np.max(np.abs(part_ke[-1] - ordered_matsum_ke[-1]))
        ),
        "final_part_internal_energy_max_abs_difference": float(
            np.max(np.abs(part_ie[-1] - ordered_matsum_ie[-1]))
        ),
        "final_part_mass_from_momentum_max_relative_error": float(np.max(mass_relative_error)),
        "d3plot_binout_part_response_match": bool(
            np.allclose(part_velocity[-1, :, 2], ordered_matsum_vz[-1], rtol=1e-9, atol=1e-9)
            and np.allclose(part_ke[-1], ordered_matsum_ke[-1], rtol=1e-9, atol=1e-9)
            and np.allclose(part_ie[-1], ordered_matsum_ie[-1], rtol=1e-9, atol=1e-9)
            and np.max(mass_relative_error) <= 1e-9
        ),
    }
    case_rows: list[dict] = []
    timeseries_rows: list[dict] = []
    spatial_mismatch_total = 0

    for case, center in zip(cases, case_centers):
        plate_index = part_lookup[case["plate_part_id"]]
        fragment_index = part_lookup[case["fragment_part_id"]]
        plate_elements = element_part_indexes == plate_index
        fragment_elements = element_part_indexes == fragment_index
        case_elements = plate_elements | fragment_elements
        case_node_indexes = np.unique(connectivity[case_elements].ravel())
        mismatch_count = nearest_center_mismatch_count(final_coords[case_node_indexes, :2], center, case_centers)
        spatial_mismatch_total += mismatch_count

        history_id = case["fragment_center_history_node_id"]
        history_node_index = node_lookup[history_id]
        history_output_index = nodout_lookup[history_id]
        support = np.any(connectivity == history_node_index, axis=1) & fragment_elements
        live_support_count = int(np.count_nonzero(support & final_alive))
        center_available = live_support_count > 0
        fragment_matsum_index = matsum_lookup[case["fragment_part_id"]]
        plate_matsum_index = matsum_lookup[case["plate_part_id"]]

        plate_reference_volume = float(reference_volume[plate_elements].sum())
        fragment_reference_volume = float(reference_volume[fragment_elements].sum())
        plate_alive_volume = float(reference_volume[plate_elements & final_alive].sum())
        fragment_alive_volume = float(reference_volume[fragment_elements & final_alive].sum())
        initial_back_face = float(node_coords[np.unique(connectivity[plate_elements].ravel()), 2].min())
        live_fragment = fragment_elements & final_alive
        alive_fragment_behind = live_fragment & (final_centroids[:, 2] < initial_back_face)
        behind_fraction = (
            float(reference_volume[alive_fragment_behind].sum() / fragment_alive_volume)
            if fragment_alive_volume > 0.0 else 0.0
        )
        survivor_average_speed = float(-part_velocity[-1, fragment_index, 2])
        perforation_proxy = bool(behind_fraction > 0.0 and survivor_average_speed > 0.0)

        center_recorded_speed = float(-nodout_vz[-1, history_output_index])
        center_speed = center_recorded_speed if center_available else None
        center_stability = final_window_statistics(nodout_time, -nodout_vz[:, history_output_index])
        part_stability = final_window_statistics(matsum_time, -matsum_vz[:, fragment_matsum_index])
        row = {
            "case_id": case["case_id"], "role": case["role"],
            "impact_speed_mps": case["impact_speed_mps"], "mesh_mm": case["mesh_mm"],
            "plate_part_id": case["plate_part_id"], "fragment_part_id": case["fragment_part_id"],
            "history_node_id": history_id,
            "initial_center_x_mm": float(center[0]), "initial_center_y_mm": float(center[1]),
            "initial_plate_back_face_z_mm": initial_back_face,
            "history_node_final_z_mm": float(nodout_z[-1, history_output_index]),
            "history_node_live_support_count": live_support_count,
            "center_history_node_qoi_available": center_available,
            "center_history_node_recorded_final_speed_mps": center_recorded_speed,
            "center_history_node_residual_speed_mps": center_speed,
            "center_history_node_final_window_relative_range": center_stability["relative_range"],
            "survivor_average_residual_speed_mps": survivor_average_speed,
            "survivor_average_final_window_relative_range": part_stability["relative_range"],
            "fragment_initial_mass_kg": float(part_mass[0, fragment_index]),
            "fragment_final_mass_kg": float(part_mass[-1, fragment_index]),
            "fragment_surviving_mass_fraction": float(part_mass[-1, fragment_index] / part_mass[0, fragment_index]),
            "fragment_initial_element_count": int(np.count_nonzero(fragment_elements)),
            "fragment_final_live_element_count": int(np.count_nonzero(fragment_elements & final_alive)),
            "fragment_surviving_reference_volume_fraction": float(fragment_alive_volume / fragment_reference_volume),
            "plate_initial_element_count": int(np.count_nonzero(plate_elements)),
            "plate_final_live_element_count": int(np.count_nonzero(plate_elements & final_alive)),
            "plate_deleted_reference_volume_fraction": float(1.0 - plate_alive_volume / plate_reference_volume),
            "surviving_fragment_reference_volume_behind_initial_back_face_fraction": behind_fraction,
            "perforation_proxy": perforation_proxy,
            "fragment_final_kinetic_energy": float(part_ke[-1, fragment_index]),
            "fragment_final_internal_energy": float(part_ie[-1, fragment_index]),
            "fragment_eroded_kinetic_energy": float(matsum_eroded_ke[-1, fragment_matsum_index]),
            "fragment_eroded_internal_energy": float(matsum_eroded_ie[-1, fragment_matsum_index]),
            "plate_final_kinetic_energy": float(part_ke[-1, plate_index]),
            "plate_final_internal_energy": float(part_ie[-1, plate_index]),
            "plate_eroded_kinetic_energy": float(matsum_eroded_ke[-1, plate_matsum_index]),
            "plate_eroded_internal_energy": float(matsum_eroded_ie[-1, plate_matsum_index]),
            "cross_case_nearest_center_node_mismatch_count": mismatch_count,
        }
        case_rows.append(row)
        for index, time_value in enumerate(nodout_time):
            timeseries_rows.append(
                {
                    "case_id": case["case_id"], "time_ms": float(time_value),
                    "history_node_z_mm": float(nodout_z[index, history_output_index]),
                    "history_node_recorded_speed_mps": float(-nodout_vz[index, history_output_index]),
                    "survivor_part_average_speed_mps": float(-matsum_vz[index, fragment_matsum_index]),
                    "fragment_kinetic_energy": float(matsum_ke[index, fragment_matsum_index]),
                    "fragment_internal_energy": float(matsum_ie[index, fragment_matsum_index]),
                    "fragment_eroded_kinetic_energy": float(matsum_eroded_ke[index, fragment_matsum_index]),
                    "fragment_eroded_internal_energy": float(matsum_eroded_ie[index, fragment_matsum_index]),
                }
            )

    database = {
        "schema_version": "hf-r04-database-readability@1.0.0", "status": "pass",
        "reader": "lasso-python", "reader_version": distribution_version("lasso-python"),
        "binout_files": [str(Path(path).resolve()) for path in binout.filelist],
        "binout_inventory": binout_inventory, "binout_time_records": int(nodout_time.size),
        "d3plot_family_start": str((args.result_dir / "d3plot").resolve()),
        "d3plot_selected_times_ms": np.asarray(arrays["timesteps"]).tolist(),
        "d3plot_nodes": int(d3plot.header.n_nodes), "d3plot_solids": int(d3plot.header.n_solids),
        "d3plot_part_ids": part_ids.tolist(), "case_ids": [case["case_id"] for case in cases],
        "identity_checks": {
            "case_part_ids_all_present": all(
                case["plate_part_id"] in part_lookup and case["fragment_part_id"] in part_lookup for case in cases
            ),
            "history_node_ids_match": set(nodout_ids.tolist()) == {case["fragment_center_history_node_id"] for case in cases},
            "matsum_part_ids_match": set(matsum_ids.tolist()) == set(part_ids.tolist()),
            "case_spatial_isolation_dynamic": spatial_mismatch_total == 0,
            "d3plot_binout_part_response_match": cross_database[
                "d3plot_binout_part_response_match"
            ],
        },
        "element_alive_encoding": "0=deleted; nonzero=live part-index+1",
        "node_state_coordinate_interpretation": node_state_mode,
        "cross_database_consistency": cross_database,
        "d3plot_minus_binout_final_time_ms": float(
            np.asarray(arrays["timesteps"])[-1] - nodout_time[-1]
        ),
    }
    write_json(args.output_dir / "database_readability.json", database)
    write_csv(args.output_dir / "case_response.csv", case_rows)
    write_csv(args.output_dir / "case_response_timeseries.csv", timeseries_rows)
    append_log(log_path, "Stage 3 complete: database directories, identities, responses, deletion/mass masks, and dynamic isolation extracted.")

    mesh = build_mesh_assessment(case_rows)
    speed = build_speed_assessment(case_rows)
    write_json(args.output_dir / "mesh_assessment.json", mesh)
    write_json(args.output_dir / "speed_assessment.json", speed)
    write_response_plot(args.output_dir / "response_diagnostics.png", case_rows)
    write_spatial_plot(
        args.output_dir / "final_state_spatial_review.png", cases, case_rows,
        element_part_indexes, part_ids, final_centroids, final_alive,
    )
    append_log(log_path, "Stage 4 complete: frozen speed and three-grid sets assessed; unavailable center-node values remain gaps.")

    core_stage0_2_gates = (
        "archive_and_input_identity",
        "solver_normal_termination",
        "global_total_energy_candidate_band",
        "hourglass_candidate_gate",
    )
    stage0_2_core_pass = all(
        stage0_2["gates"].get(name) == "pass" for name in core_stage0_2_gates
    )
    gates = {
        "stage0_2_archive_input_solver_global_energy": "pass" if stage0_2_core_pass else "fail",
        "binary_database_readability": "pass",
        "database_identity_and_dynamic_case_isolation": "pass" if all(database["identity_checks"].values()) else "fail",
        "case_level_qoi_completeness": "partial",
        "mesh_perforation_proxy_consistency": "pass" if mesh["perforation_proxy_consistent"] else "fail",
        "mesh_residual_response_consistency": mesh["response_consistency_gate"],
        "speed_scan_transition_bracketing": "fail" if not speed["perforation_transition_bracketed"] else "pass",
        "p0_response_calibration_admission": "denied",
        "p1_submission_admission": "denied",
    }
    assessment = {
        "schema_version": "hf-p0-stage3-5-assessment@1.0.0", "job_id": "72095164",
        "assessment_time": now(), "overall_status": "completed_reject_for_response_calibration",
        "calibration_admission": "denied", "p1_submission_admission": "denied",
        "gates": gates, "database": database, "case_response_summary": case_rows,
        "mesh_assessment": mesh, "speed_assessment": speed,
        "all_cases_perforation_proxy": all(row["perforation_proxy"] for row in case_rows),
        "primary_rejection_reasons": [
            "The five-point medium-mesh speed scan contains no non-perforated proxy point, so the transition is not bracketed.",
            "The three-grid surviving-fragment average speed differs strongly coarse-to-medium and medium-to-fine.",
            "Fragment surviving mass fraction varies strongly across the three meshes.",
            "Five of seven center history nodes have no live solid support at the final state; frozen NODOUT values are unavailable as residual-speed QoI.",
            "No numeric preregistered residual-velocity error band is present in the frozen design or report.",
        ],
        "allowed_use": [
            "binary-reader and post-processing validation", "numerical erosion/deletion diagnostics",
            "design of a corrected P0 response/output contract",
        ],
        "prohibited_use": [
            "V50 or material calibration", "armor-response parameter fitting", "KMFC threshold or label generation",
            "P1 submission", "research dataset regeneration or model training", "sealed-test access",
        ],
        "claim_boundary": (
            "Stages 3-5 establish binary readability and diagnose case-level numerical response. "
            "The rejection is a P0 numerical-response qualification decision, not proof that LS-DYNA, the platform, or the reference material is universally invalid. "
            "No experimental physical validation or real-world effectiveness claim is established."
        ),
    }
    write_json(args.output_dir / "stage3_5_assessment.json", assessment)
    (args.output_dir / "stage3_5_report.md").write_text(build_report(assessment, case_rows, mesh, speed), encoding="utf-8")
    append_log(log_path, "Stage 5 complete: P0 response calibration and P1 submission denied; engine/data/training/test state untouched.")

    owned = [path for path in args.output_dir.iterdir() if path.is_file() and path.name != "analysis_manifest.json"]
    manifest = {
        "schema_version": "hf-stage3-5-analysis-manifest@1.0.0", "created_at": now(),
        "analyzer": str(Path(__file__).resolve()), "analyzer_sha256": sha256(Path(__file__).resolve()),
        "stage0_2_qa_sha256": sha256(args.stage0_2_qa), "case_map_sha256": sha256(args.case_map),
        "artifacts": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(owned)
        ],
    }
    write_json(args.output_dir / "analysis_manifest.json", manifest)
    print(json.dumps({"overall_status": assessment["overall_status"], "gates": gates, "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
