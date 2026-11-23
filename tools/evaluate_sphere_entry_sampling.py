"""Validate V6-A sphere-entry reanchoring against the frozen 1.4 baseline.

The tool compares the same directed physical ray under two longitudinal origin
parameterizations.  It deliberately keeps the transverse proposal, impact
speed, body roll and software-teacher random realization fixed.
"""

from __future__ import annotations

import argparse
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
from damage_engine.engine import ENGINE_CODE_HASH, ENGINE_VERSION, DamageEngine
from damage_engine.munition_profiles import apply_munition_profile
from damage_engine.registry import PhysicsRegistry
from damage_engine.schema_resources import schema_tree_hash
from damage_engine.sampling_spec import SamplingSpec
from damage_engine.sphere_entry_ray_sampling import (
    PAIRED_LEGACY_STANDOFF_6D,
    REDUCED_PHYSICAL_RAY_5D,
    SPHERE_ENTRY_RAY_IMPLEMENTATION_HASH,
    apply_sphere_entry_reanchor,
    map_sphere_entry_latent,
)
from damage_engine.utils import content_hash, source_tree_hash


TOOL_VERSION = "sphere-entry-equivalence@1.0.0"
LAYOUT_REDUCTION_TOOL_VERSION = "sphere-entry-layout-reduction-equivalence@1.0.0"
CRN_POLICY_ID = "physical-ray-profile-speed-roll-crn@1.0.0"
PROFILES = ("small_front", "medium_front", "large_front")
SCIENCE_ROOTS = (
    "status",
    "scientific_completion_status",
    "penetration_domain_coverage",
    "summary",
    "components",
    "damage_tree",
)
EXCLUDED_NUISANCE_FIELDS = (
    "case_id",
    "provenance",
    "encounter.trajectory_origin_m",
    "encounter.collision_distance_m",
    "encounter.detonation_time_s",
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


def _current_engine_code_hash() -> str:
    """Recompute the engine identity to detect source drift during a run."""

    return content_hash(
        {
            "python_source_tree_hash": source_tree_hash(ROOT / "damage_engine"),
            "schema_tree_hash": schema_tree_hash(),
        }
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _canonical_float(value: float) -> float:
    result = round(float(value), 12)
    return 0.0 if abs(result) < 5.0e-13 else result


def common_random_group(
    *,
    physical_ray_id: str,
    profile_id: str,
    impact_speed_mps: float,
    body_x_roll_rad: float,
    target_hash: str,
    seed_namespace: int,
) -> dict[str, Any]:
    """Derive one teacher random group without parameterization nuisances."""

    speed = float(impact_speed_mps)
    roll = float(body_x_roll_rad)
    if not isinstance(physical_ray_id, str) or not physical_ray_id:
        raise ValueError("physical_ray_id must be nonempty")
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError("profile_id must be nonempty")
    if not math.isfinite(speed) or speed <= 0.0:
        raise ValueError("impact_speed_mps must be finite and positive")
    if not math.isfinite(roll):
        raise ValueError("body_x_roll_rad must be finite")
    if not isinstance(target_hash, str) or not target_hash:
        raise ValueError("target_hash must be nonempty")
    payload = {
        "policy_id": CRN_POLICY_ID,
        "physical_ray_id": physical_ray_id,
        "profile_id": profile_id,
        "impact_speed_mps": _canonical_float(speed),
        "body_x_roll_rad": _canonical_float(roll),
        "target_hash": target_hash,
        "seed_namespace": int(seed_namespace),
    }
    digest = content_hash(payload).split(":", 1)[1]
    return {
        **payload,
        "policy_hash": content_hash(
            {
                "policy_id": CRN_POLICY_ID,
                "included_fields": [
                    "physical_ray_id",
                    "profile_id",
                    "impact_speed_mps",
                    "body_x_roll_rad",
                    "target_hash",
                    "seed_namespace",
                ],
                "excluded_fields": [
                    "standoff_m",
                    "sphere_radius_m",
                    "clearance_m",
                    "trajectory_origin_m",
                    "collision_distance_m",
                    "detonation_time_s",
                ],
            }
        ),
        "epistemic_group_id": "sphere-crn-" + digest[:24],
        "aleatory_seed": int(digest[:16], 16) & ((1 << 63) - 1),
    }


def _science_projection(result: Mapping[str, Any]) -> dict[str, Any]:
    return {name: result.get(name) for name in SCIENCE_ROOTS}


def _compare_values(
    left: Any,
    right: Any,
    *,
    path: str,
    atol: float,
    rtol: float,
    mismatches: list[str],
    maxima: dict[str, float],
    counts: dict[str, int],
) -> None:
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            mismatches.append(path)
            return
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            for key in sorted(left_keys.symmetric_difference(right_keys)):
                mismatches.append(f"{path}/{key}" if path else str(key))
        for key in sorted(left_keys.intersection(right_keys)):
            _compare_values(
                left[key],
                right[key],
                path=f"{path}/{key}" if path else str(key),
                atol=atol,
                rtol=rtol,
                mismatches=mismatches,
                maxima=maxima,
                counts=counts,
            )
        return
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        if not isinstance(right, Sequence) or isinstance(right, (str, bytes)):
            mismatches.append(path)
            return
        if len(left) != len(right):
            mismatches.append(path)
            return
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _compare_values(
                left_item,
                right_item,
                path=f"{path}/{index}",
                atol=atol,
                rtol=rtol,
                mismatches=mismatches,
                maxima=maxima,
                counts=counts,
            )
        return
    if isinstance(left, bool) or isinstance(right, bool):
        counts["categorical"] += 1
        if left is not right:
            mismatches.append(path)
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        left_value = float(left)
        right_value = float(right)
        counts["numeric"] += 1
        if not math.isfinite(left_value) or not math.isfinite(right_value):
            if not (math.isnan(left_value) and math.isnan(right_value)):
                mismatches.append(path)
            return
        absolute = abs(left_value - right_value)
        relative = absolute / max(abs(left_value), abs(right_value), 1.0e-300)
        maxima["absolute"] = max(maxima["absolute"], absolute)
        maxima["relative"] = max(maxima["relative"], relative)
        if not math.isclose(left_value, right_value, abs_tol=atol, rel_tol=rtol):
            mismatches.append(path)
        return
    counts["categorical"] += 1
    if left != right:
        mismatches.append(path)


def compare_science_outputs(
    legacy: Mapping[str, Any],
    sphere: Mapping[str, Any],
    *,
    atol: float = 1.0e-12,
    rtol: float = 1.0e-10,
) -> dict[str, Any]:
    """Compare only post-contact scientific fields, excluding provenance."""

    mismatches: list[str] = []
    maxima = {"absolute": 0.0, "relative": 0.0}
    counts = {"numeric": 0, "categorical": 0}
    left = _science_projection(legacy)
    right = _science_projection(sphere)
    _compare_values(
        left,
        right,
        path="",
        atol=float(atol),
        rtol=float(rtol),
        mismatches=mismatches,
        maxima=maxima,
        counts=counts,
    )
    unique = sorted(set(mismatches))
    return {
        "passed": not unique,
        "atol": float(atol),
        "rtol": float(rtol),
        "numeric_value_count": counts["numeric"],
        "categorical_value_count": counts["categorical"],
        "maximum_absolute_difference": maxima["absolute"],
        "maximum_relative_difference": maxima["relative"],
        "mismatch_count": len(unique),
        "mismatch_paths": unique,
        "compared_roots": list(SCIENCE_ROOTS),
        "excluded_nuisance_fields": list(EXCLUDED_NUISANCE_FIELDS),
    }


def _geometry_report(
    engine: DamageEngine,
    legacy_case: Mapping[str, Any],
    sphere_case: Mapping[str, Any],
    *,
    atol: float = 1.0e-10,
) -> dict[str, Any]:
    legacy_encounter = legacy_case["encounter"]
    sphere_encounter = sphere_case["encounter"]
    legacy_origin = np.asarray(legacy_encounter["trajectory_origin_m"], dtype=np.float64)
    sphere_origin = np.asarray(sphere_encounter["trajectory_origin_m"], dtype=np.float64)
    legacy_velocity = np.asarray(legacy_encounter["velocity_target_mps"], dtype=np.float64)
    sphere_velocity = np.asarray(sphere_encounter["velocity_target_mps"], dtype=np.float64)
    legacy_direction = legacy_velocity / np.linalg.norm(legacy_velocity)
    sphere_direction = sphere_velocity / np.linalg.norm(sphere_velocity)
    legacy_hits = engine.compiled_target.ray_hits(legacy_origin, legacy_direction)
    sphere_hits = engine.compiled_target.ray_hits(sphere_origin, sphere_direction)
    mismatches: list[str] = []
    maximum_point_error = 0.0
    maximum_normal_error = 0.0
    maximum_path_error = 0.0
    if len(legacy_hits) != len(sphere_hits):
        mismatches.append("hit_count")
    for index, (left, right) in enumerate(zip(legacy_hits, sphere_hits)):
        if (left.object_id, left.role) != (right.object_id, right.role):
            mismatches.append(f"hits/{index}/identity")
        point_error = float(np.max(np.abs(left.entry_point - right.entry_point)))
        normal_error = float(np.max(np.abs(left.entry_normal - right.entry_normal)))
        path_error = abs(left.path_length_m - right.path_length_m)
        maximum_point_error = max(maximum_point_error, point_error)
        maximum_normal_error = max(maximum_normal_error, normal_error)
        maximum_path_error = max(maximum_path_error, path_error)
        if point_error > atol:
            mismatches.append(f"hits/{index}/entry_point")
        if normal_error > atol:
            mismatches.append(f"hits/{index}/entry_normal")
        if path_error > atol:
            mismatches.append(f"hits/{index}/path_length_m")
    if not np.allclose(legacy_velocity, sphere_velocity, rtol=0.0, atol=atol):
        mismatches.append("velocity_target_mps")
    if not np.allclose(
        legacy_encounter["body_to_target_orientation_xyzw"],
        sphere_encounter["body_to_target_orientation_xyzw"],
        rtol=0.0,
        atol=atol,
    ):
        mismatches.append("body_to_target_orientation_xyzw")
    return {
        "passed": not mismatches,
        "legacy_hit_count": len(legacy_hits),
        "sphere_hit_count": len(sphere_hits),
        "legacy_hit_ids": [item.object_id for item in legacy_hits],
        "sphere_hit_ids": [item.object_id for item in sphere_hits],
        "maximum_entry_point_error_m": maximum_point_error,
        "maximum_entry_normal_error": maximum_normal_error,
        "maximum_path_length_error_m": maximum_path_error,
        "mismatch_paths": sorted(set(mismatches)),
    }


def evaluate_case_pair(
    engine: DamageEngine,
    legacy_case: Mapping[str, Any],
    sphere_case: Mapping[str, Any],
    *,
    selection_declaration: Mapping[str, Any],
    replicates: int,
    common_group: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one S-CONTACT and teacher-equivalence pair with common randomness."""

    if int(replicates) < 2:
        raise ValueError("paired teacher evaluation requires at least two replicates")
    prepared: list[dict[str, Any]] = []
    contacts: list[dict[str, Any]] = []
    for label, source in (("legacy", legacy_case), ("sphere", sphere_case)):
        case = deepcopy(dict(source))
        analysis = case.setdefault("analysis", {})
        analysis["replicates"] = int(replicates)
        analysis["aleatory_seed"] = int(common_group["aleatory_seed"])
        analysis["epistemic_sample_id"] = str(common_group["epistemic_group_id"])
        analysis["adaptive_ci_half_width"] = None
        analysis["capture_fragment_audit"] = False
        analysis["capture_replicate_trace"] = False
        case["case_id"] = f"sphere-equivalence-{label}-{common_group['epistemic_group_id']}"
        resolved = engine.prepare_case(case)
        contact = audit_s_contact(
            resolved,
            engine.compiled_target,
            data_role="contact_conditional_reference",
            estimand=S_CONTACT_ESTIMAND,
            selection_declaration=selection_declaration,
        )
        prepared.append(resolved)
        contacts.append(contact["first_contact"])
    geometry = _geometry_report(engine, prepared[0], prepared[1])
    legacy_result = engine.run(prepared[0], fail_closed=True).to_dict()
    sphere_result = engine.run(prepared[1], fail_closed=True).to_dict()
    teacher = compare_science_outputs(legacy_result, sphere_result)
    return {
        "physical_ray_id": sphere_case.get("metadata", {})
        .get("sphere_entry_ray_sampling", {})
        .get("physical_ray_id"),
        "common_random_group": dict(common_group),
        "legacy_contact": contacts[0],
        "sphere_contact": contacts[1],
        "geometry": geometry,
        "teacher": teacher,
        "passed": geometry["passed"] and teacher["passed"],
    }


def _build_pair(
    *,
    engine: DamageEngine,
    base_case: Mapping[str, Any],
    legacy_manifold: Mapping[str, Any],
    latent: Sequence[float],
    profile_id: str,
    clearance_m: float,
    object_scope: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    legacy = deepcopy(dict(base_case))
    sphere = deepcopy(dict(base_case))
    for case in (legacy, sphere):
        case.setdefault("target_ref", {})["content_hash"] = engine.target_hash
        apply_munition_profile(case, profile_id)
        apply_contact_manifold(case, legacy_manifold, latent)
    contact_record = sphere["metadata"]["contact_manifold_sampling"]
    sphere_record = apply_sphere_entry_reanchor(
        sphere,
        engine.compiled_target,
        target_hash=engine.target_hash,
        clearance_m=clearance_m,
        object_scope=object_scope,
        latent_layout=PAIRED_LEGACY_STANDOFF_6D,
        latent=latent,
        line_point_target_m=contact_record["aim_point_target_m"],
    )
    return legacy, sphere, {
        "physical_ray_id": sphere_record["physical_ray_id"],
        "speed_mps": float(contact_record["speed_mps"]),
        "body_x_roll_rad": float(contact_record["body_x_roll_rad"]),
        "approach_id": str(contact_record["approach_id"]),
        "sphere_definition_hash": sphere_record["sphere"]["definition_hash"],
    }


def build_layout_reduction_pair(
    *,
    engine: DamageEngine,
    base_case: Mapping[str, Any],
    legacy_manifold: Mapping[str, Any],
    physical_latent: Sequence[float],
    profile_id: str,
    clearance_m: float,
    object_scope: str,
    dummy_standoff_quantile: float = 0.5,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build the same physical sphere-entry ray under the 6D and 5D layouts."""

    mapped = map_sphere_entry_latent(
        physical_latent,
        layout=REDUCED_PHYSICAL_RAY_5D,
        legacy_standoff_quantile=dummy_standoff_quantile,
    )
    paired_latent = list(mapped.legacy_contact_manifold_latent)
    paired = deepcopy(dict(base_case))
    reduced = deepcopy(dict(base_case))
    records: list[dict[str, Any]] = []
    for case, layout, latent in (
        (paired, PAIRED_LEGACY_STANDOFF_6D, paired_latent),
        (reduced, REDUCED_PHYSICAL_RAY_5D, list(mapped.raw_latent)),
    ):
        case.setdefault("target_ref", {})["content_hash"] = engine.target_hash
        apply_munition_profile(case, profile_id)
        # The frozen transverse proposal still consumes its legacy 6D vector.
        # Its longitudinal coordinate is a dummy for both sphere-entry layouts.
        apply_contact_manifold(case, legacy_manifold, paired_latent)
        contact_record = case["metadata"]["contact_manifold_sampling"]
        records.append(
            apply_sphere_entry_reanchor(
                case,
                engine.compiled_target,
                target_hash=engine.target_hash,
                clearance_m=clearance_m,
                object_scope=object_scope,
                latent_layout=layout,
                latent=latent,
                line_point_target_m=contact_record["aim_point_target_m"],
                legacy_standoff_quantile=dummy_standoff_quantile,
            )
        )

    paired_contact = paired["metadata"]["contact_manifold_sampling"]
    return paired, reduced, {
        "paired_latent_layout": PAIRED_LEGACY_STANDOFF_6D,
        "reduced_latent_layout": REDUCED_PHYSICAL_RAY_5D,
        "physical_latent": list(mapped.physical_latent),
        "dummy_standoff_quantile": float(dummy_standoff_quantile),
        "paired_physical_ray_id": records[0]["physical_ray_id"],
        "reduced_physical_ray_id": records[1]["physical_ray_id"],
        "speed_mps": float(paired_contact["speed_mps"]),
        "body_x_roll_rad": float(paired_contact["body_x_roll_rad"]),
        "approach_id": str(paired_contact["approach_id"]),
        "sphere_definition_hash": records[0]["sphere"]["definition_hash"],
    }


def evaluate_layout_reduction_pair(
    engine: DamageEngine,
    paired_case: Mapping[str, Any],
    reduced_case: Mapping[str, Any],
    *,
    selection_declaration: Mapping[str, Any],
    replicates: int,
    common_group: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the B1 6D and B2 5D layouts under common randomness."""

    if int(replicates) < 2:
        raise ValueError("layout-reduction evaluation requires at least two replicates")
    paired_record = (
        paired_case.get("metadata", {}).get("sphere_entry_ray_sampling", {})
    )
    reduced_record = (
        reduced_case.get("metadata", {}).get("sphere_entry_ray_sampling", {})
    )
    prepared: list[dict[str, Any]] = []
    contacts: list[dict[str, Any]] = []
    for label, source in (("paired-6d", paired_case), ("reduced-5d", reduced_case)):
        case = deepcopy(dict(source))
        analysis = case.setdefault("analysis", {})
        analysis["replicates"] = int(replicates)
        analysis["aleatory_seed"] = int(common_group["aleatory_seed"])
        analysis["epistemic_sample_id"] = str(common_group["epistemic_group_id"])
        analysis["adaptive_ci_half_width"] = None
        analysis["capture_fragment_audit"] = False
        analysis["capture_replicate_trace"] = False
        case["case_id"] = (
            f"sphere-layout-equivalence-{label}-"
            f"{common_group['epistemic_group_id']}"
        )
        resolved = engine.prepare_case(case)
        contact = audit_s_contact(
            resolved,
            engine.compiled_target,
            data_role="contact_conditional_reference",
            estimand=S_CONTACT_ESTIMAND,
            selection_declaration=selection_declaration,
        )
        prepared.append(resolved)
        contacts.append(contact["first_contact"])

    geometry = _geometry_report(engine, prepared[0], prepared[1])
    paired_result = engine.run(prepared[0], fail_closed=True).to_dict()
    reduced_result = engine.run(prepared[1], fail_closed=True).to_dict()
    teacher = compare_science_outputs(paired_result, reduced_result)
    checks = {
        "physical_ray_id_equal": paired_record.get("physical_ray_id")
        == reduced_record.get("physical_ray_id"),
        "both_s_contact_admitted": len(contacts) == 2,
        "first_armor_object_equal": contacts[0].get("collision_armor_id")
        == contacts[1].get("collision_armor_id"),
        "geometry_equal": bool(geometry["passed"]),
        "teacher_science_equal": bool(teacher["passed"]),
    }
    return {
        "paired_physical_ray_id": paired_record.get("physical_ray_id"),
        "reduced_physical_ray_id": reduced_record.get("physical_ray_id"),
        "common_random_group": dict(common_group),
        "paired_contact": contacts[0],
        "reduced_contact": contacts[1],
        "geometry": geometry,
        "teacher": teacher,
        "gate": {"checks": checks, "overall_pass": all(checks.values())},
    }


def _run_paired_equivalence(args: argparse.Namespace) -> int:
    started = perf_counter()
    engine_code_hash_at_start = _current_engine_code_hash()
    legacy_document = _load_json(args.legacy_spec)
    sphere_spec = SamplingSpec.from_file(args.sphere_spec)
    sphere_document = sphere_spec.contact_manifold
    if sphere_document is None:
        raise ValueError("sphere-entry SamplingSpec has no contact manifold")
    sphere_policy = sphere_document["sphere"]
    if sphere_document["latent_layout"] != PAIRED_LEGACY_STANDOFF_6D:
        raise ValueError("G1 requires paired_legacy_standoff_6d")
    target = _load_json(args.target)
    engine = DamageEngine(target, PhysicsRegistry.from_file(args.physics))
    base_case = _load_json(args.base_case)
    legacy_manifold = legacy_document["contact_manifold"]
    selection = legacy_document["contact_conditioning"]["selection_declaration"]
    approach_count = len(legacy_manifold["approaches"])

    edge = float(args.geometry_edge_quantile)
    if not 0.0 < edge < 0.5:
        raise ValueError("geometry_edge_quantile must lie in (0, 0.5)")
    geometry_points = (
        ("center", 0.5, 0.5),
        ("lower_lower", edge, edge),
        ("lower_upper", edge, 1.0 - edge),
        ("upper_lower", 1.0 - edge, edge),
        ("upper_upper", 1.0 - edge, 1.0 - edge),
    )
    geometry_rows: list[dict[str, Any]] = []
    physical_ray_ids: list[str] = []
    for approach_index, approach in enumerate(legacy_manifold["approaches"]):
        selector = (approach_index + 0.5) / approach_count
        for position_name, aim_u, aim_v in geometry_points:
            latent = [selector, aim_u, aim_v, 0.5, 0.5, 0.5]
            legacy, sphere, identity = _build_pair(
                engine=engine,
                base_case=base_case,
                legacy_manifold=legacy_manifold,
                latent=latent,
                profile_id="medium_front",
                clearance_m=float(sphere_policy["clearance_m"]),
                object_scope=str(sphere_policy["object_scope"]),
            )
            check = _geometry_report(engine, legacy, sphere)
            physical_ray_ids.append(identity["physical_ray_id"])
            geometry_rows.append(
                {
                    "approach_id": approach["approach_id"],
                    "position": position_name,
                    "physical_ray_id": identity["physical_ray_id"],
                    "sphere_definition_hash": identity["sphere_definition_hash"],
                    **check,
                }
            )

    teacher_rows: list[dict[str, Any]] = []
    cell_counts: dict[str, int] = {}
    for approach_index, approach in enumerate(legacy_manifold["approaches"]):
        selector = (approach_index + 0.5) / approach_count
        for profile_id in PROFILES:
            cell = f"{approach['approach_id']}|{profile_id}"
            cell_counts[cell] = 0
            for seed in args.teacher_seeds:
                rng = np.random.default_rng(int(seed) + 1009 * approach_index)
                standoff, speed, roll = (0.1 + 0.8 * float(value) for value in rng.random(3))
                latent = [selector, 0.5, 0.5, standoff, speed, roll]
                legacy, sphere, identity = _build_pair(
                    engine=engine,
                    base_case=base_case,
                    legacy_manifold=legacy_manifold,
                    latent=latent,
                    profile_id=profile_id,
                    clearance_m=float(sphere_policy["clearance_m"]),
                    object_scope=str(sphere_policy["object_scope"]),
                )
                group = common_random_group(
                    physical_ray_id=identity["physical_ray_id"],
                    profile_id=profile_id,
                    impact_speed_mps=identity["speed_mps"],
                    body_x_roll_rad=identity["body_x_roll_rad"],
                    target_hash=engine.target_hash,
                    seed_namespace=int(seed),
                )
                pair = evaluate_case_pair(
                    engine,
                    legacy,
                    sphere,
                    selection_declaration=selection,
                    replicates=int(args.replicates),
                    common_group=group,
                )
                teacher_rows.append(
                    {
                        "approach_profile_cell": cell,
                        "teacher_seed": int(seed),
                        **pair,
                    }
                )
                cell_counts[cell] += 1
                physical_ray_ids.append(identity["physical_ray_id"])
                if len(teacher_rows) % 9 == 0:
                    print(
                        json.dumps(
                            {
                                "stage": "paired_equivalence",
                                "teacher_pairs_completed": len(teacher_rows),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

    expected_cells = approach_count * len(PROFILES)
    engine_code_hash_at_end = _current_engine_code_hash()
    checks = {
        "engine_identity_matches_import": engine_code_hash_at_start
        == ENGINE_CODE_HASH,
        "source_stable_during_run": engine_code_hash_at_end
        == engine_code_hash_at_start,
        "geometry_all_pass": all(row["passed"] for row in geometry_rows),
        "teacher_all_pass": all(row["passed"] for row in teacher_rows),
        "all_approach_profile_cells_covered": len(cell_counts) == expected_cells
        and all(count == len(args.teacher_seeds) for count in cell_counts.values()),
        "at_least_three_teacher_seeds": len(set(args.teacher_seeds)) >= 3,
        "legacy_1_4_hash_matches_preregistered": SamplingSpec.from_file(
            args.legacy_spec
        ).hash
        == "sha256:4589489789fbb0389f7da8ce1c5db5bd0162b70ba1933838e8edfe76b9c1062b",
    }
    report = {
        "schema_version": TOOL_VERSION,
        "stage": "paired_equivalence",
        "estimand_contract": S_CONTACT_ESTIMAND,
        "source_identity": {
            "legacy_spec_hash": _file_hash(args.legacy_spec),
            "legacy_sampling_spec_content_hash": SamplingSpec.from_file(
                args.legacy_spec
            ).hash,
            "sphere_spec_hash": _file_hash(args.sphere_spec),
            "sphere_sampling_spec_content_hash": sphere_spec.hash,
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
            "sphere_definition_hash": geometry_rows[0]["sphere_definition_hash"],
            "physical_ray_ordered_set_hash": content_hash(physical_ray_ids),
            "crn_policy_hash": teacher_rows[0]["common_random_group"]["policy_hash"],
        },
        "configuration": {
            "replicates": int(args.replicates),
            "teacher_seeds": [int(value) for value in args.teacher_seeds],
            "geometry_edge_quantile": edge,
            "sphere": sphere_policy,
            "latent_layout": sphere_document["latent_layout"],
        },
        "pair_count": len(geometry_rows) + len(teacher_rows),
        "geometry_pair_count": len(geometry_rows),
        "teacher_pair_count": len(teacher_rows),
        "cell_counts": cell_counts,
        "excluded_nuisance_fields": list(EXCLUDED_NUISANCE_FIELDS),
        "geometry_checks": geometry_rows,
        "teacher_checks": teacher_rows,
        "gate": {"checks": checks, "overall_pass": all(checks.values())},
        "elapsed_seconds": perf_counter() - started,
        "evidence_scope": "software_teacher_longitudinal_parameterization_equivalence",
    }
    _write_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "pair_count": report["pair_count"],
                "overall_pass": report["gate"]["overall_pass"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if report["gate"]["overall_pass"] else 2


def _run_layout_reduction_equivalence(args: argparse.Namespace) -> int:
    started = perf_counter()
    engine_code_hash_at_start = _current_engine_code_hash()
    g1_report = _load_json(args.g1_report)
    if not bool(g1_report.get("gate", {}).get("overall_pass")):
        raise ValueError("layout reduction requires a passing G1 report")

    legacy_document = _load_json(args.legacy_spec)
    paired_document = _load_json(args.paired_spec)
    reduced_document = _load_json(args.reduced_spec)
    paired_spec = SamplingSpec.from_file(args.paired_spec)
    reduced_spec = SamplingSpec.from_file(args.reduced_spec)
    paired_manifold = paired_spec.contact_manifold
    reduced_manifold = reduced_spec.contact_manifold
    if paired_manifold is None or reduced_manifold is None:
        raise ValueError("layout-reduction SamplingSpecs require contact manifolds")
    if paired_manifold.get("latent_layout") != PAIRED_LEGACY_STANDOFF_6D:
        raise ValueError("B1 requires paired_legacy_standoff_6d")
    if reduced_manifold.get("latent_layout") != REDUCED_PHYSICAL_RAY_5D:
        raise ValueError("B2 requires reduced_physical_ray_5d")

    target = _load_json(args.target)
    engine = DamageEngine(target, PhysicsRegistry.from_file(args.physics))
    base_case = _load_json(args.base_case)
    legacy_manifold = legacy_document["contact_manifold"]
    selection = paired_document["contact_conditioning"]["selection_declaration"]
    paired_sphere = paired_manifold["sphere"]
    reduced_sphere = reduced_manifold["sphere"]
    approach_count = len(legacy_manifold["approaches"])
    dummy_standoff = float(args.dummy_standoff_quantile)
    if not 0.0 < dummy_standoff < 1.0:
        raise ValueError("dummy_standoff_quantile must lie in (0, 1)")

    rows: list[dict[str, Any]] = []
    cell_counts: dict[str, int] = {}
    physical_ray_ids: list[str] = []
    for approach_index, approach in enumerate(legacy_manifold["approaches"]):
        selector = (approach_index + 0.5) / approach_count
        for profile_index, profile_id in enumerate(PROFILES):
            cell = f"{approach['approach_id']}|{profile_id}"
            cell_counts[cell] = 0
            for seed in args.teacher_seeds:
                rng = np.random.default_rng(
                    int(seed) + 1009 * approach_index + 1_000_003 * profile_index
                )
                aim_u, aim_v, speed, roll = (
                    0.1 + 0.8 * float(value) for value in rng.random(4)
                )
                physical_latent = [selector, aim_u, aim_v, speed, roll]
                paired, reduced, identity = build_layout_reduction_pair(
                    engine=engine,
                    base_case=base_case,
                    legacy_manifold=legacy_manifold,
                    physical_latent=physical_latent,
                    profile_id=profile_id,
                    clearance_m=float(paired_sphere["clearance_m"]),
                    object_scope=str(paired_sphere["object_scope"]),
                    dummy_standoff_quantile=dummy_standoff,
                )
                group = common_random_group(
                    physical_ray_id=identity["paired_physical_ray_id"],
                    profile_id=profile_id,
                    impact_speed_mps=identity["speed_mps"],
                    body_x_roll_rad=identity["body_x_roll_rad"],
                    target_hash=engine.target_hash,
                    seed_namespace=int(seed),
                )
                pair = evaluate_layout_reduction_pair(
                    engine,
                    paired,
                    reduced,
                    selection_declaration=selection,
                    replicates=int(args.replicates),
                    common_group=group,
                )
                rows.append(
                    {
                        "approach_profile_cell": cell,
                        "teacher_seed": int(seed),
                        "physical_latent": physical_latent,
                        "dummy_standoff_quantile": dummy_standoff,
                        **pair,
                    }
                )
                cell_counts[cell] += 1
                physical_ray_ids.append(identity["paired_physical_ray_id"])
                if len(rows) % 9 == 0:
                    print(
                        json.dumps(
                            {
                                "stage": "layout_reduction_equivalence",
                                "teacher_pairs_completed": len(rows),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

    expected_cells = approach_count * len(PROFILES)
    engine_code_hash_at_end = _current_engine_code_hash()
    checks = {
        "engine_identity_matches_import": engine_code_hash_at_start
        == ENGINE_CODE_HASH,
        "source_stable_during_run": engine_code_hash_at_end
        == engine_code_hash_at_start,
        "g1_prerequisite_passed": True,
        "g1_engine_identity_matches_current": g1_report.get(
            "source_identity", {}
        ).get("engine_code_hash")
        == engine_code_hash_at_start,
        "all_layout_pairs_pass": all(
            row["gate"]["overall_pass"] for row in rows
        ),
        "all_approach_profile_cells_covered": len(cell_counts) == expected_cells
        and all(count == len(args.teacher_seeds) for count in cell_counts.values()),
        "at_least_three_teacher_seeds": len(set(args.teacher_seeds)) >= 3,
        "paired_and_reduced_sphere_definitions_equal": paired_sphere
        == reduced_sphere,
        "paired_1_5_hash_matches_g1": paired_spec.hash
        == "sha256:ca7bebc8e5af1b96f756f0b2fdbdfe05d118d87c01e5c04861d149eb52590024",
    }
    report = {
        "schema_version": LAYOUT_REDUCTION_TOOL_VERSION,
        "stage": "layout_reduction_equivalence",
        "estimand_contract": S_CONTACT_ESTIMAND,
        "source_identity": {
            "g1_report_hash": _file_hash(args.g1_report),
            "legacy_spec_hash": _file_hash(args.legacy_spec),
            "paired_spec_hash": _file_hash(args.paired_spec),
            "paired_sampling_spec_content_hash": paired_spec.hash,
            "reduced_spec_hash": _file_hash(args.reduced_spec),
            "reduced_sampling_spec_content_hash": reduced_spec.hash,
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
            "physical_ray_ordered_set_hash": content_hash(physical_ray_ids),
            "crn_policy_hash": rows[0]["common_random_group"]["policy_hash"],
        },
        "configuration": {
            "replicates": int(args.replicates),
            "teacher_seeds": [int(value) for value in args.teacher_seeds],
            "dummy_standoff_quantile": dummy_standoff,
            "paired_latent_layout": paired_manifold["latent_layout"],
            "reduced_latent_layout": reduced_manifold["latent_layout"],
            "sphere": paired_sphere,
        },
        "pair_count": len(rows),
        "cell_counts": cell_counts,
        "pair_checks": rows,
        "gate": {"checks": checks, "overall_pass": all(checks.values())},
        "elapsed_seconds": perf_counter() - started,
        "evidence_scope": "software_teacher_sphere_entry_latent_reduction_equivalence",
    }
    _write_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "pair_count": report["pair_count"],
                "overall_pass": report["gate"]["overall_pass"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if report["gate"]["overall_pass"] else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    paired = subparsers.add_parser(
        "paired-equivalence", help="run the preregistered V6-A G1 equivalence gate"
    )
    paired.add_argument(
        "--legacy-spec", type=Path, default=Path("configs/sampling/p0_contact_reference.json")
    )
    paired.add_argument(
        "--sphere-spec",
        type=Path,
        default=Path("configs/sampling/p0_sphere_entry_reference.v1.json"),
    )
    paired.add_argument(
        "--base-case", type=Path, default=Path("configs/cases/sampling_reference_r4.json")
    )
    paired.add_argument(
        "--target", type=Path, default=Path("configs/targets/target.fragment_only.v1.json")
    )
    paired.add_argument(
        "--physics", type=Path, default=Path("configs/physics/loitering_impact.json")
    )
    paired.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/validation/sampling_sphere_v1/paired_equivalence.json"),
    )
    paired.add_argument("--replicates", type=int, default=4)
    paired.add_argument(
        "--teacher-seeds", type=int, nargs="+", default=[2026112001, 2026112002, 2026112003]
    )
    paired.add_argument("--geometry-edge-quantile", type=float, default=1.0e-6)
    paired.set_defaults(handler=_run_paired_equivalence)
    reduced = subparsers.add_parser(
        "layout-reduction-equivalence",
        help="run the B1 6D versus B2 5D latent-reduction equivalence gate",
    )
    reduced.add_argument(
        "--g1-report",
        type=Path,
        default=Path("artifacts/validation/sampling_sphere_v1/paired_equivalence.json"),
    )
    reduced.add_argument(
        "--legacy-spec", type=Path, default=Path("configs/sampling/p0_contact_reference.json")
    )
    reduced.add_argument(
        "--paired-spec",
        type=Path,
        default=Path("configs/sampling/p0_sphere_entry_reference.v1.json"),
    )
    reduced.add_argument(
        "--reduced-spec",
        type=Path,
        default=Path("configs/sampling/p0_sphere_entry_reduced_5d.v1.json"),
    )
    reduced.add_argument(
        "--base-case", type=Path, default=Path("configs/cases/sampling_reference_r4.json")
    )
    reduced.add_argument(
        "--target", type=Path, default=Path("configs/targets/target.fragment_only.v1.json")
    )
    reduced.add_argument(
        "--physics", type=Path, default=Path("configs/physics/loitering_impact.json")
    )
    reduced.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/validation/sampling_sphere_v1/layout_reduction_equivalence.json"
        ),
    )
    reduced.add_argument("--replicates", type=int, default=4)
    reduced.add_argument(
        "--teacher-seeds", type=int, nargs="+", default=[2026112001, 2026112002, 2026112003]
    )
    reduced.add_argument("--dummy-standoff-quantile", type=float, default=0.5)
    reduced.set_defaults(handler=_run_layout_reduction_equivalence)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
