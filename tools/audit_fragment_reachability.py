"""Run an outcome-blind, Train-only fragment reachability diagnostic.

Cases are selected by input geometry only: munition profile, first-contact
armor, and proximity to predeclared trajectory-y centers.  The command never
uses archived labels to select cases and never mutates the source archive.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.engine import ENGINE_CODE_HASH, DamageEngine
from damage_engine.errors import InputValidationError
from damage_engine.fragment_audit import (
    REACHABILITY_CATEGORIES,
    diagnose_front_axis_semantics,
    summarize_component_reachability,
)
from damage_engine.normalization import normalize_target
from damage_engine.registry import PhysicsRegistry
from damage_engine.sampling_checkpoint import iter_reference_formal_cases
from damage_engine.utils import content_hash, file_hash


DEFAULT_Y_CENTERS_M = (-2.5, -0.5, 1.5)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--physics", required=True)
    parser.add_argument("--sampling-run", required=True)
    parser.add_argument("--system-output", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--trace-output",
        help=(
            "optional deterministic .jsonl.gz sidecar containing every raw "
            "fragment-audit record used by the summary"
        ),
    )
    parser.add_argument("--sampling-role", default="reference_train")
    parser.add_argument(
        "--trajectory-y-centers-m",
        nargs="+",
        type=float,
        default=list(DEFAULT_Y_CENTERS_M),
    )
    return parser


@contextmanager
def _deterministic_gzip_jsonl(path: Path):
    """Atomically stream canonical JSONL through gzip with a fixed timestamp."""

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise InputValidationError(
            "fragment-audit trace staging path already exists",
            context={"path": str(partial)},
        )
    try:
        with partial.open("wb") as raw_stream:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_stream,
                compresslevel=9,
                mtime=0,
            ) as compressed:
                def write(record: dict) -> None:
                    payload = json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                    compressed.write(payload + b"\n")

                yield write
        partial.replace(path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _leaf_components_for_outputs(target: dict, output_names: Iterable[str]) -> dict[str, list[str]]:
    tree = target.get("damage_tree", {})
    nodes = tree.get("nodes", ())
    lookup = {node.get("id"): node for node in nodes if isinstance(node, dict)}
    requested = tuple(map(str, output_names))
    if not requested or len(set(requested)) != len(requested):
        raise InputValidationError("system-output values must be unique and nonempty")

    def visit(node_id: str, stack: tuple[str, ...] = ()) -> set[str]:
        if node_id in stack:
            raise InputValidationError(
                "damage-tree cycle encountered while resolving audit components",
                context={"cycle": [*stack, node_id]},
            )
        node = lookup.get(node_id)
        if node is None:
            raise InputValidationError(
                "requested audit output references an unknown damage-tree node",
                context={"node_id": node_id},
            )
        if node.get("node_type") == "leaf":
            component_id = node.get("component_id")
            if not isinstance(component_id, str) or not component_id:
                raise InputValidationError("damage-tree leaf has no component_id")
            return {component_id}
        components: set[str] = set()
        inputs = node.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            raise InputValidationError(
                "non-leaf damage-tree node has no inputs",
                context={"node_id": node_id},
            )
        for child in inputs:
            components.update(visit(str(child), (*stack, node_id)))
        return components

    return {name: sorted(visit(name)) for name in requested}


def _select_cases(
    sampling_run: Path,
    *,
    role: str,
    y_centers: tuple[float, ...],
    y_bounds: tuple[float, float],
) -> tuple[list[tuple[tuple[str, str, float], dict]], dict]:
    if (
        not y_centers
        or len(set(y_centers)) != len(y_centers)
        or any(not math.isfinite(value) for value in y_centers)
    ):
        raise InputValidationError("trajectory-y centers must be unique and nonempty")
    if any(not (y_bounds[0] <= value <= y_bounds[1]) for value in y_centers):
        raise InputValidationError(
            "trajectory-y centers must remain inside the frozen design domain",
            context={"frozen_bounds_m": list(y_bounds)},
        )
    selected: dict[tuple[str, str, float], tuple[float, dict]] = {}
    profiles: set[str] = set()
    armors: set[str] = set()
    eligible_count = 0
    for case in iter_reference_formal_cases(sampling_run):
        sampling = case.get("metadata", {}).get("sampling", {})
        if sampling.get("role") != role:
            continue
        eligible_count += 1
        profile = str(case.get("munition", {}).get("profile_id", ""))
        armor = str(case.get("encounter", {}).get("collision_armor_id", ""))
        position = case.get("encounter", {}).get("trajectory_origin_m")
        if not profile or not armor or not isinstance(position, list) or len(position) != 3:
            raise InputValidationError("eligible audit case is missing profile/armor/trajectory")
        y_value = float(position[1])
        profiles.add(profile)
        armors.add(armor)
        for center in y_centers:
            key = (profile, armor, center)
            distance = abs(y_value - center)
            current = selected.get(key)
            tie_break = str(case.get("case_id", ""))
            if current is None or (distance, tie_break) < (
                current[0],
                str(current[1].get("case_id", "")),
            ):
                selected[key] = (distance, deepcopy(case))
    expected = {
        (profile, armor, center)
        for profile in profiles
        for armor in armors
        for center in y_centers
    }
    missing = sorted(expected - set(selected))
    if eligible_count == 0 or not expected or missing:
        raise InputValidationError(
            "outcome-blind fragment audit could not fill the declared input strata",
            context={"eligible_count": eligible_count, "missing": missing},
        )
    return (
        [(key, selected[key][1]) for key in sorted(expected)],
        {
            "eligible_case_count": eligible_count,
            "profiles": sorted(profiles),
            "collision_armor_ids": sorted(armors),
            "frozen_trajectory_y_bounds_m": list(y_bounds),
            "trajectory_y_centers_m": list(y_centers),
            "selected_case_count": len(expected),
        },
    )


def _trajectory_y_bounds(sampling_run: Path) -> tuple[tuple[float, float], Path]:
    sampling_spec_path = sampling_run / "plan" / "sampling_spec.json"
    document = json.loads(sampling_spec_path.read_text(encoding="utf-8"))
    variables = document.get("reference_distribution", {}).get("variables", ())
    matches = [
        item
        for item in variables
        if isinstance(item, dict)
        and item.get("path") == "/encounter/trajectory_origin_m/1"
        and item.get("distribution") == "uniform"
    ]
    if len(matches) != 1:
        raise InputValidationError(
            "sampling spec must define one uniform trajectory-y reference variable"
        )
    low = float(matches[0].get("low"))
    high = float(matches[0].get("high"))
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise InputValidationError("trajectory-y reference bounds are invalid")
    return (low, high), sampling_spec_path


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    target_path = Path(arguments.target).resolve()
    physics_path = Path(arguments.physics).resolve()
    sampling_run = Path(arguments.sampling_run).resolve()
    target = json.loads(target_path.read_text(encoding="utf-8"))
    registry = PhysicsRegistry.from_file(physics_path)
    engine = DamageEngine(target, registry)
    output_components = _leaf_components_for_outputs(target, arguments.system_output)
    component_ids = sorted({item for values in output_components.values() for item in values})
    y_bounds, sampling_spec_path = _trajectory_y_bounds(sampling_run)
    selected, selection = _select_cases(
        sampling_run,
        role=arguments.sampling_role,
        y_centers=tuple(float(value) for value in arguments.trajectory_y_centers_m),
        y_bounds=y_bounds,
    )

    aggregate = {
        component_id: {category: 0 for category in REACHABILITY_CATEGORIES}
        for component_id in component_ids
    }
    case_reports = []
    axis_reports = {}
    trace_destination = (
        Path(arguments.trace_output).resolve() if arguments.trace_output else None
    )
    summary_destination = Path(arguments.output).resolve()
    if trace_destination == summary_destination:
        raise InputValidationError("trace-output must differ from the summary output")
    trace_record_count = 0
    trace_event_count = 0
    trace_event_type_counts: dict[str, int] = {}
    trace_context = (
        _deterministic_gzip_jsonl(trace_destination)
        if trace_destination is not None
        else nullcontext(None)
    )
    with trace_context as write_trace:
        for (profile, armor, center), original in selected:
            case = deepcopy(original)
            case.setdefault("analysis", {})["capture_fragment_audit"] = True
            result = engine.run(case, fail_closed=True).to_dict()
            audit_records = result["fragment_audit"]
            for replicate_record in audit_records:
                events = replicate_record.get("events", ())
                if not isinstance(events, list):
                    raise InputValidationError(
                        "fragment-audit replicate record has no event list"
                    )
                trace_event_count += len(events)
                for event in events:
                    record_type = str(event.get("record_type", "unknown"))
                    trace_event_type_counts[record_type] = (
                        trace_event_type_counts.get(record_type, 0) + 1
                    )
            if write_trace is not None:
                for record_index, record in enumerate(audit_records):
                    write_trace(
                        {
                            "case_id": case["case_id"],
                            "record_index": record_index,
                            "audit_record": record,
                        }
                    )
                    trace_record_count += 1
            summary = summarize_component_reachability(audit_records, component_ids)
            for component_id in component_ids:
                for category, value in summary["components"][component_id]["counts"].items():
                    aggregate[component_id][category] += int(value)
            if profile not in axis_reports:
                source = registry.fragment_source(
                    case["warhead"]["fragment_source"]["model_id"]
                )
                axis_reports[profile] = diagnose_front_axis_semantics(case, source)
            encounter = case["encounter"]
            case_reports.append(
                {
                    "case_id": case["case_id"],
                    "input_stratum": {
                        "munition_profile": profile,
                        "collision_armor_id": armor,
                        "declared_trajectory_y_center_m": center,
                        "trajectory_origin_y_m": float(
                            encounter["trajectory_origin_m"][1]
                        ),
                        "trajectory_origin_z_m": float(
                            encounter["trajectory_origin_m"][2]
                        ),
                        "impact_speed_mps": float(
                            encounter["velocity_target_mps"][0]
                        ),
                    },
                    "replicate_fragment_counts": summary["replicate_fragment_counts"],
                    "components": summary["components"],
                }
            )

    aggregate_components = {}
    for component_id, counts in aggregate.items():
        total = sum(counts.values())
        aggregate_components[component_id] = {
            "fragment_trace_count": total,
            "counts": counts,
            "fractions": {name: value / total for name, value in counts.items()},
        }
    checkpoint_path = sampling_run / "sampling_checkpoint.json"
    report = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "outcome_blind": True,
            "labels_read_for_selection": False,
            "sampling_role": arguments.sampling_role,
            "stratification": "munition_profile x collision_armor_id x nearest_declared_trajectory_y_center",
            **selection,
        },
        "requested_system_outputs": list(arguments.system_output),
        "system_output_components": output_components,
        "aggregate_components": aggregate_components,
        "front_axis_diagnostics": axis_reports,
        "cases": case_reports,
        "bindings": {
            "target_path": str(target_path),
            "target_file_hash": file_hash(target_path),
            "normalized_target_hash": content_hash(normalize_target(target)),
            "physics_path": str(physics_path),
            "physics_file_hash": file_hash(physics_path),
            "sampling_checkpoint_hash": file_hash(checkpoint_path),
            "sampling_spec_hash": file_hash(sampling_spec_path),
            "engine_code_hash": ENGINE_CODE_HASH,
        },
        "raw_trace_sidecar": (
            {
                "path": str(trace_destination),
                "file_hash": file_hash(trace_destination),
                "record_count": trace_record_count,
                "audit_event_count": trace_event_count,
                "audit_event_type_counts": trace_event_type_counts,
                "format": "canonical_jsonl_gzip_mtime_0",
            }
            if trace_destination is not None
            else {
                "path": None,
                "file_hash": None,
                "record_count": 0,
                "audit_event_count": trace_event_count,
                "audit_event_type_counts": trace_event_type_counts,
                "format": None,
                "status": "not_requested",
            }
        ),
        "source_archive_mutated": False,
        "evidence_boundary": (
            "Train-only synthetic geometry/response diagnostic. It separates executed "
            "reachability failure modes but does not validate the physical fragment source, "
            "component geometry, armor response, damage thresholds, or real-world prevalence."
        ),
    }
    destination = summary_destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(destination),
                "selected_case_count": len(case_reports),
                "component_count": len(component_ids),
                "trace_record_count": trace_record_count,
                "trace_event_count": trace_event_count,
                "trace_event_type_counts": trace_event_type_counts,
                "trace_output": (
                    str(trace_destination) if trace_destination is not None else None
                ),
                "source_archive_mutated": False,
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
