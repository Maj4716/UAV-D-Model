"""Compare pre-outcome 4D design sequences for the reduced sphere-entry sampler.

The comparison fixes approach and munition profile in every cell.  Only the
four continuous physical coordinates ``aim_u``, ``aim_v``, ``speed`` and
``body_x_roll`` change.  It resolves first contact and audits S-CONTACT, but it
never executes the stochastic damage teacher.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.contact_contract import S_CONTACT_ESTIMAND, audit_s_contact
from damage_engine.contact_manifold_sampling import apply_contact_manifold
from damage_engine.design_sequences import (
    DESIGN_SEQUENCE_SCHEMA_VERSION,
    SUPPORTED_DESIGN_METHODS,
    design_diagnostics,
    generate_unit_design,
)
from damage_engine.engine import ENGINE_CODE_HASH, ENGINE_VERSION, DamageEngine
from damage_engine.errors import DamageEngineError, InputValidationError
from damage_engine.munition_profiles import apply_munition_profile
from damage_engine.registry import PhysicsRegistry
from damage_engine.schema_resources import schema_tree_hash
from damage_engine.sampling_spec import SamplingSpec
from damage_engine.sphere_entry_ray_sampling import (
    REDUCED_PHYSICAL_RAY_5D,
    SPHERE_ENTRY_RAY_IMPLEMENTATION_HASH,
    apply_sphere_entry_reanchor,
    map_sphere_entry_latent,
)
from damage_engine.utils import content_hash, source_tree_hash


TOOL_SCHEMA_VERSION = "sphere-entry-design-comparison@1.0.0"
DESIGN_METHODS = tuple(SUPPORTED_DESIGN_METHODS)
PROFILES = ("small_front", "medium_front", "large_front")
CONTINUOUS_COORDINATES = (
    "aim_u_quantile",
    "aim_v_quantile",
    "speed_quantile",
    "body_x_roll_quantile",
)
INCIDENCE_BIN_EDGES_DEG = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0)
LEXICOGRAPHIC_CRITERIA = (
    {"field": "failed_s_contact_count", "direction": "minimize"},
    {"field": "minimum_cell_contact_rate", "direction": "maximize"},
    {"field": "unique_first_armor_ids", "direction": "maximize"},
    {"field": "incidence_angle_occupied_bin_count", "direction": "maximize"},
    {
        "field": "worst_postscreen_maximum_2d_discrepancy",
        "direction": "minimize",
    },
    {
        "field": "worst_postscreen_minimum_nn_distance",
        "direction": "maximize",
    },
    {"field": "worst_prescreen_global_discrepancy", "direction": "minimize"},
)


def _current_engine_code_hash() -> str:
    """Recompute the engine identity to detect source drift during a run."""

    return content_hash(
        {
            "python_source_tree_hash": source_tree_hash(ROOT / "damage_engine"),
            "schema_tree_hash": schema_tree_hash(),
        }
    )


def _load_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError(f"expected JSON object: {path}")
    return document


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def incidence_angle_deg(
    direction_target_xyz: Sequence[float],
    surface_normal_target_xyz: Sequence[float],
) -> float:
    """Return the acute incidence angle from the surface normal in degrees."""

    direction = np.asarray(direction_target_xyz, dtype=np.float64)
    normal = np.asarray(surface_normal_target_xyz, dtype=np.float64)
    if (
        direction.shape != (3,)
        or normal.shape != (3,)
        or not np.isfinite(direction).all()
        or not np.isfinite(normal).all()
    ):
        raise InputValidationError("incidence angle requires two finite 3-vectors")
    direction_norm = float(np.linalg.norm(direction))
    normal_norm = float(np.linalg.norm(normal))
    if direction_norm <= 0.0 or normal_norm <= 0.0:
        raise InputValidationError("incidence angle vectors must be nonzero")
    cosine = abs(float(np.dot(direction / direction_norm, normal / normal_norm)))
    return math.degrees(math.acos(float(np.clip(cosine, 0.0, 1.0))))


def _legacy_transverse_manifold(
    reduced_manifold: Mapping[str, Any],
) -> dict[str, Any]:
    manifold = deepcopy(dict(reduced_manifold))
    manifold["sampler_id"] = "target-facing-ray-manifold@1.0.0"
    manifold["mode"] = "target_facing_ray_with_body_x_roll"
    manifold.pop("sphere", None)
    manifold.pop("latent_layout", None)
    return manifold


def _categorical_coverage(values: Sequence[str]) -> dict[str, Any]:
    counts = Counter(str(value) for value in values)
    total = sum(counts.values())
    entropy = 0.0
    if total:
        probabilities = np.asarray(list(counts.values()), dtype=np.float64) / total
        entropy = float(-np.sum(probabilities * np.log(probabilities)))
    return {
        "unique_count": len(counts),
        "counts": dict(sorted(counts.items())),
        "shannon_entropy_nats": entropy,
    }


def _normal_coverage(normals: Sequence[Sequence[float]]) -> dict[str, Any]:
    canonical = [
        ",".join(f"{float(component):.6f}" for component in normal)
        for normal in normals
    ]
    coverage = _categorical_coverage(canonical)
    if normals:
        array = np.asarray(normals, dtype=np.float64)
        component_minimum = [float(value) for value in np.min(array, axis=0)]
        component_maximum = [float(value) for value in np.max(array, axis=0)]
    else:
        component_minimum = None
        component_maximum = None
    return {
        **coverage,
        "quantization_decimals": 6,
        "component_minimum": component_minimum,
        "component_maximum": component_maximum,
    }


def _incidence_coverage(angles: Sequence[float]) -> dict[str, Any]:
    if not angles:
        return {
            "count": 0,
            "minimum": None,
            "maximum": None,
            "median": None,
            "range": None,
            "bin_edges": list(INCIDENCE_BIN_EDGES_DEG),
            "bin_counts": [0] * (len(INCIDENCE_BIN_EDGES_DEG) - 1),
            "occupied_bin_count": 0,
        }
    array = np.asarray(angles, dtype=np.float64)
    counts, _edges = np.histogram(array, bins=np.asarray(INCIDENCE_BIN_EDGES_DEG))
    return {
        "count": len(angles),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "median": float(np.median(array)),
        "range": float(np.ptp(array)),
        "bin_edges": list(INCIDENCE_BIN_EDGES_DEG),
        "bin_counts": [int(value) for value in counts],
        "occupied_bin_count": int(np.count_nonzero(counts)),
    }


def evaluate_design_cell(
    *,
    engine: DamageEngine,
    base_case: Mapping[str, Any],
    reduced_specification: Mapping[str, Any],
    approach_index: int,
    profile_id: str,
    method: str,
    sample_count: int,
    seed: int,
    maximin_restarts: int = 16,
) -> dict[str, Any]:
    """Evaluate one fixed approach/profile cell without running the teacher."""

    manifold = reduced_specification.get("contact_manifold")
    if not isinstance(manifold, Mapping):
        raise InputValidationError("reduced SamplingSpec requires contact_manifold")
    if manifold.get("latent_layout") != REDUCED_PHYSICAL_RAY_5D:
        raise InputValidationError("design comparison requires reduced_physical_ray_5d")
    approaches = manifold.get("approaches")
    if not isinstance(approaches, Sequence) or not 0 <= int(approach_index) < len(approaches):
        raise InputValidationError("approach_index is out of range")
    if profile_id not in PROFILES:
        raise InputValidationError("unsupported munition profile")
    sphere = manifold.get("sphere")
    if not isinstance(sphere, Mapping):
        raise InputValidationError("reduced contact manifold requires sphere policy")
    selection = reduced_specification.get("contact_conditioning", {}).get(
        "selection_declaration"
    )
    if not isinstance(selection, Mapping):
        raise InputValidationError("reduced SamplingSpec requires selection declaration")

    design = generate_unit_design(
        sample_count,
        len(CONTINUOUS_COORDINATES),
        seed=seed,
        method=method,
        maximin_restarts=maximin_restarts,
    )
    prescreen = design_diagnostics(design.points)
    legacy_manifold = _legacy_transverse_manifold(manifold)
    selector = (int(approach_index) + 0.5) / len(approaches)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    accepted_points: list[np.ndarray] = []
    for candidate_index, point in enumerate(design.points):
        physical_latent = [selector, *(float(value) for value in point)]
        mapped = map_sphere_entry_latent(
            physical_latent,
            layout=REDUCED_PHYSICAL_RAY_5D,
            legacy_standoff_quantile=0.5,
        )
        case = deepcopy(dict(base_case))
        case.setdefault("target_ref", {})["content_hash"] = engine.target_hash
        case["case_id"] = (
            f"design-{method}-{approaches[int(approach_index)]['approach_id']}-"
            f"{profile_id}-{seed}-{candidate_index}"
        )
        try:
            apply_munition_profile(case, profile_id)
            apply_contact_manifold(
                case,
                legacy_manifold,
                mapped.legacy_contact_manifold_latent,
            )
            contact_record = case["metadata"]["contact_manifold_sampling"]
            sphere_record = apply_sphere_entry_reanchor(
                case,
                engine.compiled_target,
                target_hash=engine.target_hash,
                clearance_m=float(sphere["clearance_m"]),
                object_scope=str(sphere["object_scope"]),
                latent_layout=REDUCED_PHYSICAL_RAY_5D,
                latent=physical_latent,
                line_point_target_m=contact_record["aim_point_target_m"],
                legacy_standoff_quantile=0.5,
            )
            prepared = engine.prepare_case(case)
            audit = audit_s_contact(
                prepared,
                engine.compiled_target,
                data_role="contact_conditional_reference",
                estimand=S_CONTACT_ESTIMAND,
                selection_declaration=selection,
            )
            first = audit["first_contact"]
            direction = prepared["encounter"]["velocity_target_mps"]
            normal = first["collision_normal_target_xyz"]
            accepted.append(
                {
                    "candidate_index": candidate_index,
                    "continuous_latent": [float(value) for value in point],
                    "physical_latent": physical_latent,
                    "physical_ray_id": sphere_record["physical_ray_id"],
                    "collision_armor_id": first["collision_armor_id"],
                    "collision_normal_target_xyz": normal,
                    "incidence_angle_deg": incidence_angle_deg(direction, normal),
                }
            )
            accepted_points.append(np.asarray(point, dtype=np.float64))
        except DamageEngineError as exc:
            rejected.append(
                {
                    "candidate_index": candidate_index,
                    "continuous_latent": [float(value) for value in point],
                    "error_code": exc.code,
                    "message": exc.message,
                }
            )

    postscreen = None
    postscreen_reason = None
    if len(accepted_points) >= 2:
        postscreen = design_diagnostics(np.asarray(accepted_points, dtype=np.float64))
    else:
        postscreen_reason = "fewer_than_two_s_contact_admitted_points"
    armor_ids = [row["collision_armor_id"] for row in accepted]
    normals = [row["collision_normal_target_xyz"] for row in accepted]
    angles = [float(row["incidence_angle_deg"]) for row in accepted]
    return {
        "method": method,
        "approach_id": approaches[int(approach_index)]["approach_id"],
        "profile_id": profile_id,
        "seed": int(seed),
        "candidate_count": int(sample_count),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "contact_rate": len(accepted) / int(sample_count),
        "changed_coordinates": list(CONTINUOUS_COORDINATES),
        "fixed_coordinates": ["approach_selector", "munition_profile"],
        "design_metadata": design.metadata,
        "prescreen_design_diagnostics": prescreen,
        "postscreen_design_diagnostics": postscreen,
        "postscreen_diagnostics_unavailable_reason": postscreen_reason,
        "first_armor_coverage": _categorical_coverage(armor_ids),
        "normal_coverage": _normal_coverage(normals),
        "incidence_angle_coverage_deg": _incidence_coverage(angles),
        "accepted_contacts": accepted,
        "rejected_candidates": rejected,
        "teacher_executed": False,
    }


def _method_summary(cell_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not cell_reports:
        raise InputValidationError("method summary requires cell reports")
    accepted = [
        row
        for cell in cell_reports
        for row in cell.get("accepted_contacts", [])
    ]
    candidate_count = sum(int(cell["candidate_count"]) for cell in cell_reports)
    accepted_count = sum(int(cell["accepted_count"]) for cell in cell_reports)
    rates = [float(cell["contact_rate"]) for cell in cell_reports]
    valid_postscreen = [
        cell["postscreen_design_diagnostics"]
        for cell in cell_reports
        if cell.get("postscreen_design_diagnostics") is not None
    ]
    post_projection = [
        float(item["two_dimensional_projections"]["maximum_centered_discrepancy"])
        for item in valid_postscreen
    ]
    post_nearest = [
        float(item["nearest_neighbor"]["minimum_distance"])
        for item in valid_postscreen
    ]
    prescreen_global = [
        float(cell["prescreen_design_diagnostics"]["global_centered_discrepancy"])
        for cell in cell_reports
    ]
    armors = [row["collision_armor_id"] for row in accepted]
    normals = [row["collision_normal_target_xyz"] for row in accepted]
    angles = [float(row["incidence_angle_deg"]) for row in accepted]
    incidence = _incidence_coverage(angles)
    return {
        "method": cell_reports[0]["method"],
        "cell_count": len(cell_reports),
        "candidate_count": candidate_count,
        "accepted_count": accepted_count,
        "failed_s_contact_count": candidate_count - accepted_count,
        "contact_rate": accepted_count / candidate_count,
        "minimum_cell_contact_rate": min(rates),
        "mean_cell_contact_rate": float(np.mean(rates)),
        "unique_first_armor_ids": len(set(armors)),
        "first_armor_coverage": _categorical_coverage(armors),
        "normal_coverage": _normal_coverage(normals),
        "incidence_angle_coverage_deg": incidence,
        "incidence_angle_occupied_bin_count": incidence["occupied_bin_count"],
        "postscreen_diagnostic_cell_count": len(valid_postscreen),
        "worst_postscreen_maximum_2d_discrepancy": (
            max(post_projection) if post_projection else None
        ),
        "worst_postscreen_minimum_nn_distance": (
            min(post_nearest) if post_nearest else None
        ),
        "worst_prescreen_global_discrepancy": max(prescreen_global),
    }


def _rank_value(summary: Mapping[str, Any], criterion: Mapping[str, str]) -> float:
    value = summary.get(criterion["field"])
    if value is None:
        return math.inf
    number = float(value)
    if not math.isfinite(number):
        return math.inf
    return number if criterion["direction"] == "minimize" else -number


def lexicographic_recommendation(
    method_summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Rank methods by the frozen outcome-blind pre-outcome criteria."""

    missing = [method for method in DESIGN_METHODS if method not in method_summaries]
    if missing:
        raise InputValidationError(
            "lexicographic recommendation requires every design method",
            context={"missing": missing},
        )
    priority = {method: index for index, method in enumerate(DESIGN_METHODS)}

    def key(method: str) -> tuple[float, ...]:
        summary = method_summaries[method]
        return tuple(_rank_value(summary, item) for item in LEXICOGRAPHIC_CRITERIA) + (
            float(priority[method]),
        )

    ranking = sorted(DESIGN_METHODS, key=key)
    return {
        "rule_id": "preoutcome-s-contact-geometry-lexicographic@1.0.0",
        "outcome_blind": True,
        "criterion_order": [dict(item) for item in LEXICOGRAPHIC_CRITERIA],
        "final_tie_breaker": "declared_method_order",
        "declared_method_order": list(DESIGN_METHODS),
        "ranking": ranking,
        "recommended_method": ranking[0],
        "ranking_vectors": {
            method: [float(value) for value in key(method)] for method in ranking
        },
    }


def _cell_seed(base_seed: int, approach_index: int, profile_index: int) -> int:
    generated = np.random.SeedSequence(
        [int(base_seed), int(approach_index), int(profile_index)]
    ).generate_state(1, dtype=np.uint64)
    return int(generated[0])


def _run(args: argparse.Namespace) -> int:
    started = perf_counter()
    engine_code_hash_at_start = _current_engine_code_hash()
    specification = _load_json(args.reduced_spec)
    parsed_spec = SamplingSpec.from_file(args.reduced_spec)
    manifold = parsed_spec.contact_manifold
    if manifold is None or manifold.get("latent_layout") != REDUCED_PHYSICAL_RAY_5D:
        raise ValueError("B3 requires a reduced_physical_ray_5d SamplingSpec")
    count = int(args.sample_count)
    if count < 2 or count & (count - 1):
        raise ValueError("sample_count must be a power of two and at least two")
    engine = DamageEngine(
        _load_json(args.target), PhysicsRegistry.from_file(args.physics)
    )
    base_case = _load_json(args.base_case)
    approaches = manifold["approaches"]
    method_cells: dict[str, list[dict[str, Any]]] = {
        method: [] for method in DESIGN_METHODS
    }
    for method in DESIGN_METHODS:
        for approach_index, _approach in enumerate(approaches):
            for profile_index, profile_id in enumerate(PROFILES):
                cell = evaluate_design_cell(
                    engine=engine,
                    base_case=base_case,
                    reduced_specification=specification,
                    approach_index=approach_index,
                    profile_id=profile_id,
                    method=method,
                    sample_count=count,
                    seed=_cell_seed(args.seed, approach_index, profile_index),
                    maximin_restarts=int(args.maximin_restarts),
                )
                method_cells[method].append(cell)
            if len(method_cells[method]) % 6 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "preoutcome_design_comparison",
                            "method": method,
                            "cells_completed": len(method_cells[method]),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    summaries = {
        method: _method_summary(cells) for method, cells in method_cells.items()
    }
    recommendation = lexicographic_recommendation(summaries)
    expected_cells = len(approaches) * len(PROFILES)
    engine_code_hash_at_end = _current_engine_code_hash()
    checks = {
        "engine_identity_matches_import": engine_code_hash_at_start
        == ENGINE_CODE_HASH,
        "source_stable_during_run": engine_code_hash_at_end
        == engine_code_hash_at_start,
        "reduced_1_6_specification": parsed_spec.schema_version == "1.6.0",
        "all_methods_evaluated": set(method_cells) == set(DESIGN_METHODS),
        "all_approach_profile_cells_evaluated": all(
            len(cells) == expected_cells for cells in method_cells.values()
        ),
        "teacher_never_executed": all(
            not cell["teacher_executed"]
            for cells in method_cells.values()
            for cell in cells
        ),
        "recommendation_available": recommendation["recommended_method"]
        in DESIGN_METHODS,
    }
    report = {
        "schema_version": TOOL_SCHEMA_VERSION,
        "stage": "preoutcome_design_comparison",
        "estimand_contract": S_CONTACT_ESTIMAND,
        "source_identity": {
            "reduced_spec_file_hash": _file_hash(args.reduced_spec),
            "reduced_sampling_spec_content_hash": parsed_spec.hash,
            "target_file_hash": _file_hash(args.target),
            "normalized_target_hash": engine.target_hash,
            "physics_file_hash": _file_hash(args.physics),
            "physics_registry_hash": engine.registry.hash,
            "base_case_hash": _file_hash(args.base_case),
            "engine_version": ENGINE_VERSION,
            "engine_code_hash": engine_code_hash_at_start,
            "engine_code_hash_at_start": engine_code_hash_at_start,
            "engine_code_hash_at_end": engine_code_hash_at_end,
            "source_stable_during_run": engine_code_hash_at_end
            == engine_code_hash_at_start,
            "sphere_entry_implementation_hash": SPHERE_ENTRY_RAY_IMPLEMENTATION_HASH,
            "design_sequence_schema_version": DESIGN_SEQUENCE_SCHEMA_VERSION,
        },
        "configuration": {
            "methods": list(DESIGN_METHODS),
            "sample_count_per_cell": count,
            "continuous_dimension": len(CONTINUOUS_COORDINATES),
            "changed_coordinates": list(CONTINUOUS_COORDINATES),
            "fixed_factors": ["approach_selector", "munition_profile"],
            "approach_count": len(approaches),
            "profile_count": len(PROFILES),
            "profiles": list(PROFILES),
            "seed": int(args.seed),
            "cell_seed_shared_across_methods": True,
            "maximin_restarts": int(args.maximin_restarts),
            "teacher_execution": "prohibited",
        },
        "method_summaries": summaries,
        "method_cells": method_cells,
        "recommendation": recommendation,
        "execution_gate": {"checks": checks, "overall_pass": all(checks.values())},
        "elapsed_seconds": perf_counter() - started,
        "evidence_scope": "preoutcome_design_and_first_contact_geometry_only",
    }
    _write_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "candidate_count": sum(
                    item["candidate_count"] for item in summaries.values()
                ),
                "recommended_method": recommendation["recommended_method"],
                "overall_pass": report["execution_gate"]["overall_pass"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if report["execution_gate"]["overall_pass"] else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reduced-spec",
        type=Path,
        default=Path("configs/sampling/p0_sphere_entry_reduced_5d.v1.json"),
    )
    parser.add_argument(
        "--base-case", type=Path, default=Path("configs/cases/sampling_reference_r4.json")
    )
    parser.add_argument(
        "--target", type=Path, default=Path("configs/targets/target.fragment_only.v1.json")
    )
    parser.add_argument(
        "--physics", type=Path, default=Path("configs/physics/loitering_impact.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/validation/sampling_sphere_v1/design_sequence_comparison.json"
        ),
    )
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026112004)
    parser.add_argument("--maximin-restarts", type=int, default=16)
    return parser


def main() -> int:
    return _run(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
