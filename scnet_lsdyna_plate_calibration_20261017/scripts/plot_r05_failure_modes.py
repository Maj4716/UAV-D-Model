"""Create truthful, provenance-bound figures from the R05 Stage-B CSV artifacts.

The figures are diagnostic views of solver-returned observations.  They do not
infer a physical failure mechanism or promote any P0 response quantity.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_DIR = PROJECT / "analysis" / "72252362_r05_stage_ab_20261023"
SCRIPT_NAME = "plot_r05_failure_modes"
SCRIPT_VERSION = "1.0.0"
MANIFEST_SCHEMA_VERSION = "hf-p0-r05-failure-visualization@1.0.0"
PNG_DPI = 200
FROZEN_DELETED_UPSTREAM_GATE = 0.05
FOCUS_CASES = ("p0_v0800_h0p75", "p0_v2000_h0p75")

PANEL_A_BASENAME = "r05_failure_modes_panel_a_counts"
PANEL_B_BASENAME = "r05_failure_modes_panel_b_time_initial_z"
PANEL_C_BASENAME = "r05_failure_modes_panel_c_deleted_upstream_gate"
FIGURE_SPECS: dict[str, dict[str, float]] = {
    PANEL_A_BASENAME: {"width_in": 7.2, "height_in": 4.5},
    PANEL_B_BASENAME: {"width_in": 7.2, "height_in": 3.8},
    PANEL_C_BASENAME: {"width_in": 7.2, "height_in": 4.5},
}
PNG_OUTPUTS = tuple(f"{name}.png" for name in FIGURE_SPECS)
SVG_OUTPUTS = tuple(f"{name}.svg" for name in FIGURE_SPECS)
SOURCE_DATA_NAMES = {
    "panel_a": "r05_failure_modes_panel_a_source_data.csv",
    "panel_b": "r05_failure_modes_panel_b_source_data.csv",
    "panel_c": "r05_failure_modes_panel_c_source_data.csv",
}
NOTES_NAME = "r05_failure_visualization_notes_zh.md"
MANIFEST_NAME = "r05_failure_visualization_manifest.json"
OWNED_OUTPUTS = (
    *PNG_OUTPUTS,
    *SVG_OUTPUTS,
    *SOURCE_DATA_NAMES.values(),
    NOTES_NAME,
    MANIFEST_NAME,
)

PART_COLORS = {"plate": "#0072B2", "fragment": "#D55E00"}
FAILURE_HATCHES = {"negative_volume": "////", "other_solid_failure": ".."}
FAILURE_MARKERS = {"negative_volume": "o", "other_solid_failure": "X"}
COMBINATIONS = (
    ("plate", "negative_volume"),
    ("plate", "other_solid_failure"),
    ("fragment", "negative_volume"),
    ("fragment", "other_solid_failure"),
)
COUNT_FIELDS = {
    ("plate", "negative_volume"): "plate_negative_volume_count",
    ("plate", "other_solid_failure"): "plate_other_failure_count",
    ("fragment", "negative_volume"): "fragment_negative_volume_count",
    ("fragment", "other_solid_failure"): "fragment_other_failure_count",
}
DISPLAY_FAILURE = {
    "negative_volume": "negative volume",
    "other_solid_failure": "other solid failure",
}

EVENT_REQUIRED_FIELDS = {
    "failure_type",
    "element_id",
    "time_ms",
    "case_id",
    "part_kind",
    "initial_deck_centroid_z_mm",
    "initial_position_semantics",
}
SUMMARY_REQUIRED_FIELDS = {
    "case_id",
    "impact_speed_mps",
    "mesh_mm",
    "failure_event_count",
    "plate_negative_volume_count",
    "plate_other_failure_count",
    "fragment_negative_volume_count",
    "fragment_other_failure_count",
    "deleted_upstream_mass_fraction",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def strict_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _finite_float(value: Any, label: str, *, minimum: float | None = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(parsed) or (minimum is not None and parsed < minimum):
        raise ValueError(f"{label} must be finite and at least {minimum}")
    return parsed


def _nonnegative_int(value: Any, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if str(value).strip() not in {str(parsed), f"{parsed}.0"} or parsed < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return parsed


def _read_csv(path: Path, required_fields: set[str], label: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = sorted(required_fields - set(fieldnames))
        if missing:
            raise ValueError(f"{label} required fields are missing: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{label} must contain at least one data row")
    return rows


def _case_label(speed_mps: float, mesh_mm: float) -> str:
    return f"{speed_mps:g} m/s\nh={mesh_mm:g} mm"


def load_inputs(
    events_path: Path | str,
    summary_path: Path | str,
    design_path: Path | str,
) -> dict[str, Any]:
    """Load and cross-check the two Stage-B tables and the frozen 5% gate."""

    events_path = Path(events_path).resolve()
    summary_path = Path(summary_path).resolve()
    design_path = Path(design_path).resolve()
    event_rows = _read_csv(events_path, EVENT_REQUIRED_FIELDS, "failure event CSV")
    summary_rows = _read_csv(summary_path, SUMMARY_REQUIRED_FIELDS, "case summary CSV")

    summaries: list[dict[str, Any]] = []
    by_case: dict[str, dict[str, Any]] = {}
    for row_index, row in enumerate(summary_rows, start=2):
        case_id = str(row["case_id"]).strip()
        if not case_id or case_id in by_case:
            raise ValueError(f"case summary row {row_index} has an empty or duplicate case_id")
        speed = _finite_float(row["impact_speed_mps"], f"{case_id}.impact_speed_mps", minimum=0.0)
        mesh = _finite_float(row["mesh_mm"], f"{case_id}.mesh_mm", minimum=0.0)
        if speed <= 0.0 or mesh <= 0.0:
            raise ValueError(f"{case_id} speed and mesh must be positive")
        counts = {
            combination: _nonnegative_int(row[field], f"{case_id}.{field}")
            for combination, field in COUNT_FIELDS.items()
        }
        failure_event_count = _nonnegative_int(
            row["failure_event_count"], f"{case_id}.failure_event_count"
        )
        if sum(counts.values()) != failure_event_count:
            raise ValueError(f"{case_id} category counts do not sum to failure_event_count")
        deleted_fraction = _finite_float(
            row["deleted_upstream_mass_fraction"],
            f"{case_id}.deleted_upstream_mass_fraction",
            minimum=0.0,
        )
        if deleted_fraction > 1.0:
            raise ValueError(f"{case_id} deleted_upstream_mass_fraction must not exceed one")
        parsed = {
            "case_id": case_id,
            "case_label": _case_label(speed, mesh),
            "impact_speed_mps": speed,
            "mesh_mm": mesh,
            "failure_event_count": failure_event_count,
            "counts": counts,
            "deleted_upstream_mass_fraction": deleted_fraction,
        }
        summaries.append(parsed)
        by_case[case_id] = parsed

    events: list[dict[str, Any]] = []
    seen_element_ids: set[int] = set()
    observed_counts: Counter[tuple[str, str, str]] = Counter()
    for row_index, row in enumerate(event_rows, start=2):
        case_id = str(row["case_id"]).strip()
        part_kind = str(row["part_kind"]).strip()
        failure_type = str(row["failure_type"]).strip()
        if case_id not in by_case:
            raise ValueError(f"failure event row {row_index} references unknown case {case_id!r}")
        if part_kind not in PART_COLORS:
            raise ValueError(f"failure event row {row_index} has unsupported part_kind")
        if failure_type not in FAILURE_MARKERS:
            raise ValueError(f"failure event row {row_index} has unsupported failure_type")
        element_id = _nonnegative_int(row["element_id"], f"event row {row_index}.element_id")
        if element_id in seen_element_ids:
            raise ValueError(f"failure_events_unique contains duplicate element_id {element_id}")
        seen_element_ids.add(element_id)
        time_ms = _finite_float(row["time_ms"], f"event row {row_index}.time_ms", minimum=0.0)
        initial_z = _finite_float(
            row["initial_deck_centroid_z_mm"],
            f"event row {row_index}.initial_deck_centroid_z_mm",
        )
        semantics = str(row["initial_position_semantics"]).strip()
        if semantics != "deck_reference_geometry_not_failure_time_position":
            raise ValueError(
                "failure event initial position must be explicitly marked as deck reference geometry"
            )
        events.append(
            {
                "case_id": case_id,
                "case_label": by_case[case_id]["case_label"],
                "element_id": element_id,
                "failure_type": failure_type,
                "part_kind": part_kind,
                "time_ms": time_ms,
                "initial_deck_centroid_z_mm": initial_z,
                "initial_position_semantics": semantics,
            }
        )
        observed_counts[(case_id, part_kind, failure_type)] += 1

    for summary in summaries:
        case_id = summary["case_id"]
        if sum(observed_counts[(case_id, *combination)] for combination in COMBINATIONS) != summary[
            "failure_event_count"
        ]:
            raise ValueError(f"{case_id} event counts differ from case summary")
        for combination in COMBINATIONS:
            if observed_counts[(case_id, *combination)] != summary["counts"][combination]:
                raise ValueError(f"{case_id} event counts differ from case summary")

    missing_focus = [case_id for case_id in FOCUS_CASES if case_id not in by_case]
    if missing_focus:
        raise ValueError(f"focus cases are missing from the case summary: {missing_focus}")

    try:
        design = json.loads(
            design_path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
        gate = _finite_float(
            design["qualification_gates"]["capture_ledger"][
                "deleted_upstream_mass_fraction_max"
            ],
            "frozen deleted-upstream gate",
            minimum=0.0,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"cannot read the frozen deleted-upstream gate: {exc}") from exc
    if not math.isclose(gate, FROZEN_DELETED_UPSTREAM_GATE, rel_tol=0.0, abs_tol=0.0):
        raise ValueError(
            f"frozen deleted-upstream gate must be {FROZEN_DELETED_UPSTREAM_GATE}, got {gate}"
        )

    return {
        "events": events,
        "summaries": summaries,
        "summary_by_case": by_case,
        "gate": gate,
        "paths": {
            "events": events_path,
            "summary": summary_path,
            "design": design_path,
        },
        "source_columns": {
            "events": list(event_rows[0]),
            "summary": list(summary_rows[0]),
        },
    }


def build_source_tables(inputs: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    panel_a: list[dict[str, Any]] = []
    for summary in inputs["summaries"]:
        for part_kind, failure_type in COMBINATIONS:
            panel_a.append(
                {
                    "case_id": summary["case_id"],
                    "case_label": summary["case_label"].replace("\n", "; "),
                    "impact_speed_mps": summary["impact_speed_mps"],
                    "mesh_mm": summary["mesh_mm"],
                    "part_kind": part_kind,
                    "failure_type": failure_type,
                    "count": summary["counts"][(part_kind, failure_type)],
                    "structural_zero_retained": summary["counts"][(part_kind, failure_type)] == 0,
                }
            )

    focus_order = {case_id: index for index, case_id in enumerate(FOCUS_CASES)}
    panel_b = [
        dict(event)
        for event in sorted(
            (event for event in inputs["events"] if event["case_id"] in focus_order),
            key=lambda event: (
                focus_order[event["case_id"]],
                event["time_ms"],
                event["element_id"],
            ),
        )
    ]

    panel_c = [
        {
            "case_id": summary["case_id"],
            "case_label": summary["case_label"].replace("\n", "; "),
            "impact_speed_mps": summary["impact_speed_mps"],
            "mesh_mm": summary["mesh_mm"],
            "deleted_upstream_mass_fraction": summary["deleted_upstream_mass_fraction"],
            "frozen_gate_fraction": inputs["gate"],
            "gate_status": (
                "at_or_below_frozen_gate"
                if summary["deleted_upstream_mass_fraction"] <= inputs["gate"]
                else "above_frozen_gate"
            ),
        }
        for summary in inputs["summaries"]
    ]
    return {"panel_a": panel_a, "panel_b": panel_b, "panel_c": panel_c}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _style_context() -> mpl.RcParams:
    return mpl.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "r05-failure-visualization-v1",
        }
    )


def _combo_label(part_kind: str, failure_type: str) -> str:
    return f"{part_kind.capitalize()} - {DISPLAY_FAILURE[failure_type]}"


def _plot_panel_a(rows: Sequence[Mapping[str, Any]]) -> plt.Figure:
    case_ids = list(dict.fromkeys(str(row["case_id"]) for row in rows))
    labels = [
        next(row["case_label"] for row in rows if row["case_id"] == case_id).replace(
            "; ", "\n"
        )
        for case_id in case_ids
    ]
    x = np.arange(len(case_ids), dtype=float)
    width = 0.19
    spec = FIGURE_SPECS[PANEL_A_BASENAME]
    with _style_context():
        fig, ax = plt.subplots(
            figsize=(spec["width_in"], spec["height_in"]), layout="constrained"
        )
        for index, (part_kind, failure_type) in enumerate(COMBINATIONS):
            counts = [
                next(
                    int(row["count"])
                    for row in rows
                    if row["case_id"] == case_id
                    and row["part_kind"] == part_kind
                    and row["failure_type"] == failure_type
                )
                for case_id in case_ids
            ]
            bars = ax.bar(
                x + (index - 1.5) * width,
                counts,
                width=width,
                color=PART_COLORS[part_kind],
                edgecolor="#202020",
                linewidth=0.7,
                hatch=FAILURE_HATCHES[failure_type],
                label=_combo_label(part_kind, failure_type),
                zorder=3,
            )
            ax.bar_label(bars, labels=[str(value) for value in counts], padding=2, fontsize=6.5)
        ax.set(
            xticks=x,
            xticklabels=labels,
            ylabel="Unique failed elements (count)",
            xlabel="Case: impact speed and mesh",
            title="A. MES failed-element counts; structural zeros retained",
        )
        ax.set_ylim(bottom=0.0)
        ax.grid(axis="y", zorder=0)
        ax.legend(ncol=2, frameon=False, loc="upper left")
        fig.text(
            0.995,
            0.005,
            "Observed counts only; no uncertainty interval.",
            ha="right",
            va="bottom",
            fontsize=7,
            color="#333333",
        )
    return fig


def _plot_panel_b(rows: Sequence[Mapping[str, Any]], inputs: Mapping[str, Any]) -> plt.Figure:
    spec = FIGURE_SPECS[PANEL_B_BASENAME]
    with _style_context():
        fig, axes = plt.subplots(
            1,
            2,
            sharex=True,
            sharey=True,
            figsize=(spec["width_in"], spec["height_in"]),
            layout="constrained",
        )
        for axis, case_id in zip(axes, FOCUS_CASES):
            case_rows = [row for row in rows if row["case_id"] == case_id]
            for part_kind, failure_type in COMBINATIONS:
                subset = [
                    row
                    for row in case_rows
                    if row["part_kind"] == part_kind and row["failure_type"] == failure_type
                ]
                if not subset:
                    continue
                axis.scatter(
                    [row["time_ms"] for row in subset],
                    [row["initial_deck_centroid_z_mm"] for row in subset],
                    s=18,
                    marker=FAILURE_MARKERS[failure_type],
                    color=PART_COLORS[part_kind],
                    edgecolors="#202020",
                    linewidths=0.35,
                    alpha=0.72,
                    zorder=3,
                )
            case_label = inputs["summary_by_case"][case_id]["case_label"].replace("\n", ", ")
            axis.set_title(case_label)
            axis.set_xlim(left=0.0)
            axis.grid(zorder=0)
        axes[0].set_ylabel("Initial deck centroid z (mm; reference geometry)")
        for axis in axes:
            axis.set_xlabel("MES failure-event time (ms)")
        handles = [
            Line2D(
                [],
                [],
                linestyle="none",
                marker=FAILURE_MARKERS[failure_type],
                markersize=5.5,
                markerfacecolor=PART_COLORS[part_kind],
                markeredgecolor="#202020",
                label=_combo_label(part_kind, failure_type),
            )
            for part_kind, failure_type in COMBINATIONS
        ]
        fig.legend(handles=handles, ncol=2, frameon=False, loc="outside lower center")
        fig.suptitle("B. Event time versus undeformed deck-reference z; no connecting lines")
    return fig


def _plot_panel_c(rows: Sequence[Mapping[str, Any]], gate: float) -> plt.Figure:
    spec = FIGURE_SPECS[PANEL_C_BASENAME]
    values = [float(row["deleted_upstream_mass_fraction"]) for row in rows]
    labels = [str(row["case_label"]) for row in rows]
    y = np.arange(len(rows), dtype=float)
    xmax = max(gate * 2.0, max(values) * 1.16, 0.1)
    colors = [
        PART_COLORS["plate"] if value <= gate else PART_COLORS["fragment"] for value in values
    ]
    hatches = ["" if value <= gate else "////" for value in values]
    with _style_context():
        fig, ax = plt.subplots(
            figsize=(spec["width_in"], spec["height_in"]), layout="constrained"
        )
        bars = ax.barh(
            y,
            values,
            color=colors,
            edgecolor="#202020",
            linewidth=0.7,
            zorder=3,
        )
        for bar, hatch, value in zip(bars, hatches, values):
            bar.set_hatch(hatch)
            x_text = max(value + 0.012 * xmax, 0.012 * xmax)
            ax.text(
                x_text,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1%}",
                va="center",
                ha="left",
                fontsize=7.5,
            )
        ax.axvline(
            gate,
            color="#000000",
            linestyle="--",
            linewidth=1.4,
            label=f"Frozen gate = {gate:.0%}",
            zorder=4,
        )
        ax.set(
            yticks=y,
            yticklabels=labels,
            xlim=(0.0, xmax),
            xlabel="Deleted-upstream initial reference-mass fraction",
            ylabel="Case: impact speed and mesh",
            title="C. Capture-ledger availability diagnostic against the frozen 5% gate",
        )
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(axis="x", zorder=0)
        handles = [
            Patch(
                facecolor=PART_COLORS["plate"],
                edgecolor="#202020",
                label="At or below gate",
            ),
            Patch(
                facecolor=PART_COLORS["fragment"],
                edgecolor="#202020",
                hatch="////",
                label="Above gate",
            ),
            Line2D([], [], color="#000000", linestyle="--", label=f"Frozen gate = {gate:.0%}"),
        ]
        ax.legend(handles=handles, frameon=False, loc="lower right")
        ax.text(
            0.995,
            0.99,
            "Deterministic ledger fraction; no uncertainty interval.",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="#333333",
        )
    return fig


def _save_figure_pair(
    fig: plt.Figure,
    stage_dir: Path,
    basename: str,
    *,
    title: str,
    description: str,
) -> None:
    png_path = stage_dir / f"{basename}.png"
    svg_path = stage_dir / f"{basename}.svg"
    png_metadata = {"Software": f"Matplotlib {mpl.__version__}", "Title": title}
    svg_metadata = {
        "Title": title,
        "Description": description,
        "Creator": f"{SCRIPT_NAME} {SCRIPT_VERSION}; Matplotlib {mpl.__version__}",
        "Date": None,
    }
    # SVG conversion happens at save time, after the plot-building style
    # context has exited.  Reassert these export settings here so the manifest
    # can truthfully report searchable text and deterministic SVG identifiers.
    with mpl.rc_context(
        {"svg.fonttype": "none", "svg.hashsalt": "r05-failure-visualization-v1"}
    ):
        fig.savefig(
            png_path,
            dpi=PNG_DPI,
            facecolor="white",
            edgecolor="white",
            transparent=False,
            metadata=png_metadata,
        )
        fig.savefig(
            svg_path,
            format="svg",
            facecolor="white",
            edgecolor="white",
            transparent=False,
            metadata=svg_metadata,
        )
    plt.close(fig)

    with Image.open(png_path) as source:
        converted = source.convert("RGB")
        converted.save(
            png_path,
            format="PNG",
            dpi=(PNG_DPI, PNG_DPI),
            compress_level=9,
            optimize=False,
        )


def _inspect_png(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        dpi = image.info.get("dpi")
        metadata = {
            "format": image.format,
            "mode": image.mode,
            "width_px": image.width,
            "height_px": image.height,
            "dpi": [float(value) for value in dpi] if dpi else None,
            "has_alpha_channel": "A" in image.getbands() or "transparency" in image.info,
            "background": "opaque_white",
        }
    if metadata["mode"] != "RGB" or metadata["has_alpha_channel"]:
        raise ValueError(f"PNG export is not opaque RGB: {path}")
    return metadata


def _inspect_svg(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    root = ET.fromstring(text)
    images = [element for element in root.iter() if element.tag.endswith("image")]
    text_elements = [element for element in root.iter() if element.tag.endswith("text")]
    if "#ffffff" not in text.lower():
        raise ValueError(f"SVG export lacks an explicit white background: {path}")
    return {
        "format": "SVG",
        "width": root.attrib.get("width"),
        "height": root.attrib.get("height"),
        "view_box": root.attrib.get("viewBox"),
        "text_element_count": len(text_elements),
        "embedded_raster_count": len(images),
        "background": "opaque_white",
        "container_transparency": False,
    }


def _linearized_channel(value: int) -> float:
    channel = value / 255.0
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def _relative_luminance(hex_color: str) -> float:
    value = hex_color.lstrip("#")
    channels = [int(value[index : index + 2], 16) for index in (0, 2, 4)]
    red, green, blue = (_linearized_channel(channel) for channel in channels)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(foreground: str, background: str = "#FFFFFF") -> float:
    first = _relative_luminance(foreground)
    second = _relative_luminance(background)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def _artifact_record(path: Path, *, kind: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": kind,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "metadata": dict(metadata),
    }


def _build_notes(inputs: Mapping[str, Any], tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    case_a_total = sum(int(row["count"]) for row in tables["panel_a"])
    focus_counts = Counter(str(row["case_id"]) for row in tables["panel_b"])
    above_gate = sum(row["gate_status"] == "above_frozen_gate" for row in tables["panel_c"])
    highest = max(tables["panel_c"], key=lambda row: float(row["deleted_upstream_mass_fraction"]))
    return f"""# R05 阶段 B 失效模式图表说明与替代文本

## 证据边界

这些图只展示 Job 72252362 返回文件中已经记录的数值事件和捕获账本诊断。它们不证明真实裂纹、穿孔或任何物理因果，也不构成 P0、材料模型、仿真引擎或数据集生成器准入。图中未绘制误差条：输入是一次确定性求解的事件计数与质量账本比例，没有重复样本或已定义的统计不确定性。

## 图 A：各 case 的失效单元计数

替代文本：零基线分组柱状图，按七个 case（或当前输入中的全部 case）分别展示 plate/fragment 与 negative-volume/other-solid-failure 四种组合；颜色区分部件、纹理区分失效文本类别，并直接标注计数。当前完整输入共包含 {case_a_total} 个唯一 MES 失效事件，结构零保留为 0，不从图或源数据中删除。

说明：柱高是唯一单元计数，不是发生概率；`other_solid_failure` 仅表示求解器文本没有进一步给出机理。

## 图 B：重点工况的事件时间—初始 z

替代文本：两个并排散点面板，每个点对应一个唯一 MES 失效事件；横轴是事件时间，纵轴是该单元由输入 deck 参考几何计算的初始质心 z。800 m/s、0.75 mm 工况有 {focus_counts[FOCUS_CASES[0]]} 个点，2000 m/s、0.75 mm 工况有 {focus_counts[FOCUS_CASES[1]]} 个点。颜色区分 plate/fragment，圆点与 X 标记区分两类求解器失效文本。点之间不连接、不插值、不平滑，也不做随机抖动或分箱。

重要语义：纵轴是初始参考位置，不代表失效瞬时位置或变形后位置；散点的接近、先后或聚集不能单独解释为物理因果。

## 图 C：捕获前删除质量比例与冻结门

替代文本：从零开始的水平条形图，逐 case 显示 `deleted_upstream_mass_fraction`，并以黑色虚线标出冻结的 5% 可用性门。当前 {above_gate}/{len(tables['panel_c'])} 个 case 超过该门；最高值是 `{highest['case_id']}` 的 {float(highest['deleted_upstream_mass_fraction']):.1%}。颜色、斜线纹理和直接百分比标签共同编码是否超过门限。

说明：门限来自冻结设计 `qualification_gates.capture_ledger.deleted_upstream_mass_fraction_max`。超过门限表示主 QoI 按合同不可用，不是对物理现象的概率判断。

## 可复核数据与导出

- 图 A 源数据：`{SOURCE_DATA_NAMES['panel_a']}`，显式包含所有 case×类别组合和结构零。
- 图 B 源数据：`{SOURCE_DATA_NAMES['panel_b']}`，保留每个点的 case、EID、原始时间和初始 deck z；没有聚合或平滑。
- 图 C 源数据：`{SOURCE_DATA_NAMES['panel_c']}`，保留每个 case 的原始比例、冻结门及判定。
- PNG 为 {PNG_DPI} ppi、RGB、无 alpha 通道、白色不透明背景；SVG 保留矢量文字/图元并使用白色不透明背景。
- 尺寸、像素、哈希、调色板、透明度和全部变换记录见 `{MANIFEST_NAME}`。
"""


def _source_record(path: Path, *, row_count: int, columns: Sequence[str]) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "row_count": row_count,
        "columns": list(columns),
    }


def _publish_stage(stage_dir: Path, output_dir: Path, *, replace: bool) -> None:
    publications = [
        (stage_dir / name, output_dir / name) for name in OWNED_OUTPUTS
    ]
    for source, _destination in publications:
        if not source.is_file():
            raise RuntimeError(f"staged output is missing: {source.name}")

    if replace:
        backup_dir = stage_dir / ".previous-generation"
        backup_dir.mkdir()
        for _source, destination in publications:
            if destination.exists():
                if not destination.is_file():
                    raise RuntimeError(
                        f"existing visualization output is not a regular file: {destination}"
                    )
                shutil.copy2(destination, backup_dir / destination.name)

        published: list[Path] = []
        try:
            for source, destination in publications:
                os.replace(source, destination)
                published.append(destination)
        except BaseException as publish_error:
            rollback_errors: list[str] = []
            for destination in reversed(published):
                backup = backup_dir / destination.name
                try:
                    if backup.is_file():
                        os.replace(backup, destination)
                    else:
                        destination.unlink(missing_ok=True)
                except OSError as rollback_error:
                    rollback_errors.append(f"{destination}: {rollback_error}")
            if rollback_errors:
                raise RuntimeError(
                    "visualization replacement failed and rollback was incomplete: "
                    + "; ".join(rollback_errors)
                ) from publish_error
            raise
        return

    published: list[Path] = []
    try:
        for source, destination in publications:
            try:
                if os.name == "nt":
                    # The stage directory is created inside output_dir, so a
                    # Windows rename stays on one volume, is atomic for this
                    # file, and fails rather than replacing an existing path.
                    os.rename(source, destination)
                else:
                    os.link(source, destination)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"refusing to overwrite existing output: {destination}"
                ) from exc
            published.append(destination)
    except BaseException as publish_error:
        rollback_errors: list[str] = []
        for destination in reversed(published):
            try:
                destination.unlink(missing_ok=True)
            except OSError as rollback_error:
                rollback_errors.append(f"{destination}: {rollback_error}")
        if rollback_errors:
            raise RuntimeError(
                "visualization publication failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from publish_error
        raise


def generate_visualizations(
    *,
    events_path: Path | str,
    summary_path: Path | str,
    design_path: Path | str,
    output_dir: Path | str,
    replace: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.is_symlink():
        raise ValueError("visualization output directory must not be a symbolic link")

    collisions = [name for name in OWNED_OUTPUTS if (output_dir / name).exists()]
    if collisions and not replace:
        raise FileExistsError(
            "refusing to overwrite existing visualization outputs without --replace: "
            + ", ".join(collisions)
        )

    inputs = load_inputs(events_path, summary_path, design_path)
    tables = build_source_tables(inputs)
    for source_path in inputs["paths"].values():
        if source_path.parent == output_dir and source_path.name in OWNED_OUTPUTS:
            raise ValueError("an input path collides with an owned visualization output")

    with tempfile.TemporaryDirectory(prefix=".r05-viz-stage-", dir=output_dir) as temp:
        stage_dir = Path(temp)
        panel_fields = {
            "panel_a": [
                "case_id",
                "case_label",
                "impact_speed_mps",
                "mesh_mm",
                "part_kind",
                "failure_type",
                "count",
                "structural_zero_retained",
            ],
            "panel_b": [
                "case_id",
                "case_label",
                "element_id",
                "failure_type",
                "part_kind",
                "time_ms",
                "initial_deck_centroid_z_mm",
                "initial_position_semantics",
            ],
            "panel_c": [
                "case_id",
                "case_label",
                "impact_speed_mps",
                "mesh_mm",
                "deleted_upstream_mass_fraction",
                "frozen_gate_fraction",
                "gate_status",
            ],
        }
        for panel, rows in tables.items():
            _write_csv(stage_dir / SOURCE_DATA_NAMES[panel], rows, panel_fields[panel])

        _save_figure_pair(
            _plot_panel_a(tables["panel_a"]),
            stage_dir,
            PANEL_A_BASENAME,
            title="R05 panel A: failed-element counts",
            description="Zero-baseline grouped bars with color and hatch redundancy; structural zeros retained.",
        )
        _save_figure_pair(
            _plot_panel_b(tables["panel_b"], inputs),
            stage_dir,
            PANEL_B_BASENAME,
            title="R05 panel B: event time versus initial deck-reference z",
            description="Raw failure-event points only; no connection, smoothing, jitter, or failure-time position claim.",
        )
        _save_figure_pair(
            _plot_panel_c(tables["panel_c"], inputs["gate"]),
            stage_dir,
            PANEL_C_BASENAME,
            title="R05 panel C: deleted-upstream mass fraction versus frozen gate",
            description="Zero-baseline bars with a frozen five-percent gate; no uncertainty interval.",
        )

        notes = _build_notes(inputs, tables)
        (stage_dir / NOTES_NAME).write_text(notes, encoding="utf-8", newline="\n")

        artifacts: dict[str, Any] = {}
        for name in PNG_OUTPUTS:
            artifacts[name] = _artifact_record(
                stage_dir / name, kind="figure_png", metadata=_inspect_png(stage_dir / name)
            )
        for name in SVG_OUTPUTS:
            artifacts[name] = _artifact_record(
                stage_dir / name, kind="figure_svg", metadata=_inspect_svg(stage_dir / name)
            )
        for panel, name in SOURCE_DATA_NAMES.items():
            artifacts[name] = _artifact_record(
                stage_dir / name,
                kind="source_data_csv",
                metadata={"row_count": len(tables[panel]), "columns": panel_fields[panel]},
            )
        artifacts[NOTES_NAME] = _artifact_record(
            stage_dir / NOTES_NAME,
            kind="description_and_alt_text_markdown",
            metadata={"language": "zh-CN", "encoding": "UTF-8"},
        )

        source_data = {
            panel: {
                "artifact": name,
                "row_count": len(tables[panel]),
                "columns": panel_fields[panel],
            }
            for panel, name in SOURCE_DATA_NAMES.items()
        }
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "script_identity": {
                "name": SCRIPT_NAME,
                "version": SCRIPT_VERSION,
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "runtime": {
                "python": platform.python_version(),
                "matplotlib": mpl.__version__,
                "numpy": np.__version__,
                "pillow": Image.__version__,
            },
            "sources": {
                "failure_events_unique": _source_record(
                    inputs["paths"]["events"],
                    row_count=len(inputs["events"]),
                    columns=inputs["source_columns"]["events"],
                ),
                "failure_summary_by_case": _source_record(
                    inputs["paths"]["summary"],
                    row_count=len(inputs["summaries"]),
                    columns=inputs["source_columns"]["summary"],
                ),
                "frozen_design": {
                    "path": str(inputs["paths"]["design"]),
                    "size_bytes": inputs["paths"]["design"].stat().st_size,
                    "sha256": sha256_file(inputs["paths"]["design"]),
                    "json_pointer": (
                        "/qualification_gates/capture_ledger/"
                        "deleted_upstream_mass_fraction_max"
                    ),
                },
            },
            "frozen_gate": {
                "name": "deleted_upstream_mass_fraction_max",
                "value": inputs["gate"],
                "comparison": "at_or_below passes the availability component; above fails",
            },
            "figures": {
                "panel_a": {
                    "png": f"{PANEL_A_BASENAME}.png",
                    "svg": f"{PANEL_A_BASENAME}.svg",
                    "encoding": (
                        "zero-baseline grouped bar position/length; part by color and failure text "
                        "category by hatch; direct count labels"
                    ),
                    "transformation": "exact case/category count lookup; no filtering; structural zeros retained",
                },
                "panel_b": {
                    "png": f"{PANEL_B_BASENAME}.png",
                    "svg": f"{PANEL_B_BASENAME}.svg",
                    "encoding": (
                        "raw event time versus undeformed deck-reference centroid z; part by color "
                        "and failure text category by marker"
                    ),
                    "transformation": (
                        "predeclared focus-case selection only; no binning, jitter, interpolation, "
                        "connection, smoothing, aggregation, or exclusion within selected cases"
                    ),
                    "artist_alpha": 0.72,
                    "overplotting_note": "coincident points may overlap; source-data CSV preserves every event",
                },
                "panel_c": {
                    "png": f"{PANEL_C_BASENAME}.png",
                    "svg": f"{PANEL_C_BASENAME}.svg",
                    "encoding": (
                        "zero-baseline horizontal bar; frozen threshold line; color, hatch, and "
                        "direct percentage label redundantly encode gate status"
                    ),
                    "transformation": "direct case-level fraction lookup; no estimation or normalization",
                },
            },
            "source_data": source_data,
            "uncertainty": {
                "displayed": False,
                "reason": (
                    "single deterministic solver return; no replicate distribution or inferential "
                    "uncertainty definition is available"
                ),
            },
            "missing_data_policy": (
                "required plotting values must be finite; missing/nonfinite values fail generation "
                "instead of being converted to zero or silently omitted"
            ),
            "accessibility": {
                "background": "opaque white",
                "color_is_redundant": True,
                "part_colors": PART_COLORS,
                "failure_hatches": FAILURE_HATCHES,
                "failure_markers": FAILURE_MARKERS,
                "contrast_against_white": {
                    key: contrast_ratio(color) for key, color in PART_COLORS.items()
                },
                "note": (
                    "contrast ratios and redundant encodings are recorded for review; they are not "
                    "an accessibility or publisher-compliance certification"
                ),
            },
            "export": {
                "png_dpi": PNG_DPI,
                "png_mode": "RGB",
                "png_alpha_channel": False,
                "svg_fonttype": "none",
                "background": "opaque_white",
                "bbox_inches": None,
                "figure_specs": FIGURE_SPECS,
                "replace_was_explicit": bool(replace),
            },
            "artifacts": artifacts,
            "notes_and_alt_text": NOTES_NAME,
            "claim_boundary": (
                "Diagnostic visualization of solver-returned observations only; not physical causation, "
                "experimental validation, P0 response admission, engine calibration, or dataset-generator "
                "promotion."
            ),
        }
        (stage_dir / MANIFEST_NAME).write_text(
            strict_json(manifest), encoding="utf-8", newline="\n"
        )
        _publish_stage(stage_dir, output_dir, replace=replace)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create provenance-bound R05 Stage-B diagnostic figures"
    )
    parser.add_argument(
        "--events-csv",
        type=Path,
        default=DEFAULT_ANALYSIS_DIR / "failure_events_unique.csv",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_ANALYSIS_DIR / "failure_summary_by_case.csv",
    )
    parser.add_argument(
        "--design-json", type=Path, default=PROJECT / "05" / "p0_cases_r05.json"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="explicitly replace only this script's owned visualization outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = generate_visualizations(
        events_path=args.events_csv,
        summary_path=args.summary_csv,
        design_path=args.design_json,
        output_dir=args.output_dir,
        replace=args.replace,
    )
    print(
        strict_json(
            {
                "status": "complete",
                "schema_version": manifest["schema_version"],
                "output_dir": str(args.output_dir.resolve()),
                "artifact_count": len(manifest["artifacts"]) + 1,
            }
        ),
        end="",
    )


if __name__ == "__main__":
    main()
