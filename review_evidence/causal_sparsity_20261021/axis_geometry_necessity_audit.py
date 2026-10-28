"""Read-only geometric necessity audit for the 2026-10-21 sparsity study.

The audit does not change the source model, target, cases, or archive.  It asks
whether the observed fragment rays can *possibly* reach the candidate KMFC
components along the x axis under the frozen front-contact design.  An AABB
test is only a necessary reachability condition: passing it does not prove a
hit, while failing it proves that a ray whose x coordinate only decreases
cannot intersect the object.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.geometry import compile_target  # noqa: E402
from damage_engine.engine import DamageEngine, ENGINE_CODE_HASH, ENGINE_VERSION  # noqa: E402
from damage_engine.normalization import normalize_target  # noqa: E402
from damage_engine.registry import PhysicsRegistry  # noqa: E402


TARGET = ROOT / "raw_data" / "target.rebuilt.fragment_only.v1.json"
PHYSICS = ROOT / "examples" / "physics.loitering_impact.json"
CASE_TEMPLATE = ROOT / "raw_data" / "case.loitering_impact.fragment_only.v1.json"
FORMAL_MANIFEST = (
    ROOT
    / "raw_data"
    / "dataset_5000_loitering_impact_fragment_only_v1"
    / "formal_dataset"
    / "manifest.json"
)
CASES = (
    ROOT
    / "raw_data"
    / "dataset_5000_loitering_impact_fragment_only_v1"
    / "rounds"
    / "round-000"
    / "formal_cases.jsonl.gz"
)
TRACE = (
    ROOT
    / "review_evidence"
    / "optimization_20261014"
    / "fragment_reachability_train_trace.jsonl.gz"
)
TRACE_AUDIT = (
    ROOT
    / "review_evidence"
    / "optimization_20261014"
    / "fragment_reachability_train_audit.json"
)
KMFC_SUPPORT = ROOT / "review_evidence" / "kmfc_v0_component_support_audit_20261015.json"
OUTPUT = Path(__file__).with_suffix(".json")


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def rounded_key(values: np.ndarray, digits: int = 12) -> str:
    return json.dumps([round(float(value), digits) for value in values], separators=(",", ":"))


def canonical_quaternion(values: list[float]) -> np.ndarray:
    q = np.asarray(values, dtype=np.float64)
    q = q / np.linalg.norm(q)
    for value in q:
        if abs(float(value)) > 1.0e-15:
            if value < 0.0:
                q = -q
            break
    return q


def contact_probe(compiled) -> dict:
    """Finite diagnostic probe of the declared y-z support, not a proof."""

    direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)

    def summarize(points: np.ndarray) -> dict:
        first_armor = Counter()
        no_contact = 0
        for y_value, z_value in points:
            origin = np.asarray([-10.0, float(y_value), float(z_value)], dtype=np.float64)
            hits = [hit for hit in compiled.ray_hits(origin, direction) if hit.role == "armor"]
            if not hits:
                no_contact += 1
                continue
            first = min(hits, key=lambda hit: (hit.entry_t, hit.object_id))
            first_armor[first.object_id] += 1
        return {
            "point_count": int(points.shape[0]),
            "contact_count": int(points.shape[0] - no_contact),
            "no_contact_count": int(no_contact),
            "first_contact_armor_counts": dict(sorted(first_armor.items())),
        }

    y_grid = np.linspace(-3.0, 2.0, 51)
    z_grid = np.linspace(0.65, 1.8, 51)
    grid = np.asarray([(y, z) for y in y_grid for z in z_grid], dtype=np.float64)
    rng = np.random.default_rng(20261021)
    monte_carlo = np.column_stack(
        (
            rng.uniform(-3.0, 2.0, 20_000),
            rng.uniform(0.65, 1.8, 20_000),
        )
    )
    return {
        "trajectory_origin_x_m": -10.0,
        "direction_target_xyz": direction.tolist(),
        "declared_y_bounds_m": [-3.0, 2.0],
        "declared_z_bounds_m": [0.65, 1.8],
        "grid_probe": summarize(grid),
        "fixed_seed_uniform_probe": {
            "seed": 20261021,
            **summarize(monte_carlo),
        },
        "evidence_boundary": (
            "Finite grid and fixed-seed probes show behavior inside the declared rectangle; "
            "they do not mathematically prove contact for every real-valued point."
        ),
    }


def main() -> None:
    target = json.loads(TARGET.read_text(encoding="utf-8"))
    physics = json.loads(PHYSICS.read_text(encoding="utf-8"))
    case_template = json.loads(CASE_TEMPLATE.read_text(encoding="utf-8"))
    formal_manifest = json.loads(FORMAL_MANIFEST.read_text(encoding="utf-8"))
    trace_audit = json.loads(TRACE_AUDIT.read_text(encoding="utf-8"))
    kmfc = json.loads(KMFC_SUPPORT.read_text(encoding="utf-8"))["kmfc_v0_branches"]
    compiled = compile_target(normalize_target(target))
    objects = {item.object_id: item for item in compiled.components}

    current_case = deepcopy(case_template)
    current_case.setdefault("analysis", {})["capture_fragment_audit"] = True
    current_engine = DamageEngine(target, PhysicsRegistry.from_file(PHYSICS))
    current_result = current_engine.run(current_case, fail_closed=True).to_dict()
    current_direction_x = [
        float(event["direction_target_xyz"][0])
        for replicate in current_result["fragment_audit"]
        for event in replicate["events"]
        if event.get("record_type") == "fragment_trace"
    ]

    case_profile: dict[str, str] = {}
    velocity_directions: set[str] = set()
    orientations: set[str] = set()
    contact_x_by_armor: dict[str, set[float]] = defaultdict(set)
    contact_count_by_armor: Counter[str] = Counter()
    with gzip.open(CASES, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            case = json.loads(line)
            case_profile[case["case_id"]] = case["munition"]["profile_id"]
            encounter = case["encounter"]
            velocity = np.asarray(encounter["velocity_target_mps"], dtype=np.float64)
            velocity_directions.add(rounded_key(velocity / np.linalg.norm(velocity)))
            orientations.add(
                rounded_key(canonical_quaternion(encounter["body_to_target_orientation_xyzw"]))
            )
            contact_x_by_armor[str(encounter["collision_armor_id"])].add(
                float(encounter["detonation_position_m"][0])
            )
            contact_count_by_armor[str(encounter["collision_armor_id"])] += 1

    trace_speed: dict[str, list[float]] = defaultdict(list)
    trace_dx: dict[str, list[float]] = defaultdict(list)
    trace_case_ids: set[str] = set()
    with gzip.open(TRACE, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            case_id = str(record["case_id"])
            trace_case_ids.add(case_id)
            profile = case_profile[case_id]
            for event in record["audit_record"]["events"]:
                if event.get("record_type") != "fragment_trace":
                    continue
                direction = np.asarray(event["direction_target_xyz"], dtype=np.float64)
                trace_dx[profile].append(float(direction[0]))
                trace_speed[profile].append(float(event["initial_speed_mps"]))

    contact_summary = {}
    for armor_id, values in sorted(contact_x_by_armor.items()):
        ordered = sorted(values)
        contact_summary[armor_id] = {
            "case_count": int(contact_count_by_armor[armor_id]),
            "unique_x_count": len(ordered),
            "contact_x_minimum_m": float(ordered[0]),
            "contact_x_maximum_m": float(ordered[-1]),
            "contact_x_span_m": float(ordered[-1] - ordered[0]),
        }

    category_geometry = {}
    for category, branches in kmfc.items():
        component_ids = sorted({identifier for values in branches.values() for identifier in values})
        bounds = {
            identifier: {
                "aabb_min_x_m": float(objects[identifier].geometry.bounds_min[0]),
                "aabb_max_x_m": float(objects[identifier].geometry.bounds_max[0]),
            }
            for identifier in component_ids
        }
        by_armor = {}
        for armor_id, values in sorted(contact_x_by_armor.items()):
            # Extruded-polygon intersection arithmetic can differ by a few ulps.
            # Use the largest observed x0 for a conservative all-cases exclusion:
            # min_x > max(x0) implies min_x > x0 for every case on this armor.
            contact_x_min = min(values)
            contact_x_max = max(values)
            impossible = [
                identifier
                for identifier in component_ids
                if bounds[identifier]["aabb_min_x_m"] > contact_x_max
            ]
            not_ruled_out = [identifier for identifier in component_ids if identifier not in impossible]
            by_armor[armor_id] = {
                "contact_x_minimum_m": contact_x_min,
                "contact_x_maximum_m": contact_x_max,
                "conservative_exclusion_uses_maximum_contact_x": True,
                "component_count": len(component_ids),
                "aabb_necessary_condition_failed_count": len(impossible),
                "aabb_necessary_condition_not_ruled_out_count": len(not_ruled_out),
                "failed_component_ids": impossible,
                "not_ruled_out_component_ids": not_ruled_out,
            }
        category_geometry[category] = {
            "component_count": len(component_ids),
            "component_bounds_x_m": bounds,
            "by_contact_armor": by_armor,
        }

    source_axes = {}
    for model_id, model in sorted(physics["fragment_source_models"].items()):
        axis = np.asarray(model["direction"]["axis_body"], dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        source_axes[model_id] = {
            "axis_body": axis.tolist(),
            "axis_dot_body_plus_x": float(axis[0]),
            "direction_kind": model["direction"]["kind"],
            "source_id": model.get("source_id"),
        }

    trace_summary = {}
    for profile in sorted(trace_dx):
        directions = np.asarray(trace_dx[profile], dtype=np.float64)
        speeds = np.asarray(trace_speed[profile], dtype=np.float64)
        trace_summary[profile] = {
            "fragment_trace_count": int(directions.size),
            "direction_x_minimum": float(directions.min()),
            "direction_x_median": float(np.median(directions)),
            "direction_x_maximum": float(directions.max()),
            "nonnegative_direction_x_count": int(np.count_nonzero(directions >= 0.0)),
            "initial_speed_mps_minimum": float(speeds.min()),
            "initial_speed_mps_median": float(np.median(speeds)),
            "initial_speed_mps_maximum": float(speeds.max()),
        }

    result = {
        "schema_version": "1.0.0",
        "audit_date": "2026-10-21",
        "scope": "read_only_axis_geometry_necessary_condition_diagnostic",
        "inputs": {
            "target": {"path": TARGET.relative_to(ROOT).as_posix(), "sha256": sha256(TARGET)},
            "physics": {"path": PHYSICS.relative_to(ROOT).as_posix(), "sha256": sha256(PHYSICS)},
            "case_template": {
                "path": CASE_TEMPLATE.relative_to(ROOT).as_posix(),
                "sha256": sha256(CASE_TEMPLATE),
            },
            "formal_manifest": {
                "path": FORMAL_MANIFEST.relative_to(ROOT).as_posix(),
                "sha256": sha256(FORMAL_MANIFEST),
            },
            "formal_cases": {"path": CASES.relative_to(ROOT).as_posix(), "sha256": sha256(CASES)},
            "fragment_trace": {"path": TRACE.relative_to(ROOT).as_posix(), "sha256": sha256(TRACE)},
            "fragment_trace_audit": {
                "path": TRACE_AUDIT.relative_to(ROOT).as_posix(),
                "sha256": sha256(TRACE_AUDIT),
            },
            "kmfc_support": {
                "path": KMFC_SUPPORT.relative_to(ROOT).as_posix(),
                "sha256": sha256(KMFC_SUPPORT),
            },
        },
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "software_identity": {
            "current": {
                "engine_version": ENGINE_VERSION,
                "engine_code_hash": ENGINE_CODE_HASH,
            },
            "formal_archive": {
                "engine_version": formal_manifest["engine"]["version"],
                "engine_code_hash": formal_manifest["engine"]["code_hash"],
            },
            "historical_train_only_trace": {
                "engine_code_hash": trace_audit["bindings"]["engine_code_hash"],
            },
            "formal_archive_matches_current_engine": (
                formal_manifest["engine"]["code_hash"] == ENGINE_CODE_HASH
            ),
            "historical_trace_matches_current_engine": (
                trace_audit["bindings"]["engine_code_hash"] == ENGINE_CODE_HASH
            ),
        },
        "formal_case_count": len(case_profile),
        "formal_design": {
            "normalized_velocity_direction_unique_count": len(velocity_directions),
            "normalized_velocity_directions": sorted(velocity_directions),
            "canonical_orientation_unique_count": len(orientations),
            "canonical_orientations_xyzw": sorted(orientations),
            "contact_x_by_armor": contact_summary,
        },
        "declared_support_contact_probe": contact_probe(compiled),
        "declared_fragment_source_axes": source_axes,
        "current_dev3_template_fragment_trace": {
            "case_id": current_case["case_id"],
            "fragment_trace_count": len(current_direction_x),
            "direction_x_minimum": min(current_direction_x),
            "direction_x_maximum": max(current_direction_x),
            "nonnegative_direction_x_count": sum(
                value >= 0.0 for value in current_direction_x
            ),
            "all_direction_x_strictly_negative": all(
                value < 0.0 for value in current_direction_x
            ),
        },
        "observed_train_only_fragment_trace": {
            "selected_case_count": len(trace_case_ids),
            "profile_summary": trace_summary,
            "all_observed_direction_x_strictly_negative": all(
                value < 0.0 for values in trace_dx.values() for value in values
            ),
        },
        "kmfc_aabb_necessary_reachability": category_geometry,
        "logical_interpretation": {
            "lemma": (
                "For a ray starting at x0 with direction_x < 0, x(t)=x0+t*direction_x "
                "strictly decreases for t>0. A component whose AABB minimum x is greater "
                "than x0 therefore cannot be intersected by that ray."
            ),
            "strength": (
                "Failing the AABB condition proves impossibility for the audited ray family; "
                "passing it is only necessary and does not prove geometric intersection, "
                "penetration, response, or damage."
            ),
            "causal_use": (
                "The result localizes a deterministic support restriction jointly induced by "
                "the frozen contact surface and source-axis convention. It does not establish "
                "that the convention is physically wrong or that flipping it is a valid model."
            ),
        },
        "evidence_boundary": (
            "The 6,120 traces are an outcome-blind Train-only diagnostic drawn from 18 frozen "
            "cases, not a complete physical-domain survey. Target geometry and response models "
            "remain unverified. This audit proves only a necessary geometric exclusion under the "
            "observed negative-x ray family and frozen contact coordinates."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
