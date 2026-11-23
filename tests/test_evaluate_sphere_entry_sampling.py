from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from damage_engine.contact_manifold_sampling import apply_contact_manifold
from damage_engine.engine import ENGINE_CODE_HASH, DamageEngine
from damage_engine.munition_profiles import apply_munition_profile
from damage_engine.registry import PhysicsRegistry
from damage_engine.sphere_entry_ray_sampling import (
    PAIRED_LEGACY_STANDOFF_6D,
    REDUCED_PHYSICAL_RAY_5D,
    apply_sphere_entry_reanchor,
)
from tools.evaluate_sphere_entry_sampling import (
    _current_engine_code_hash,
    build_layout_reduction_pair,
    common_random_group,
    compare_science_outputs,
    evaluate_case_pair,
    evaluate_layout_reduction_pair,
)


ROOT = Path(__file__).resolve().parents[1]


def test_current_engine_code_hash_recomputes_import_identity() -> None:
    assert _current_engine_code_hash() == ENGINE_CODE_HASH


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _engine() -> DamageEngine:
    return DamageEngine(
        _load(ROOT / "configs/targets/target.fragment_only.v1.json"),
        PhysicsRegistry.from_file(ROOT / "configs/physics/loitering_impact.json"),
    )


def test_common_random_group_is_replayable_from_physical_state() -> None:
    arguments = {
        "physical_ray_id": "sha256:" + "1" * 64,
        "profile_id": "medium_front",
        "impact_speed_mps": 120.0,
        "body_x_roll_rad": 0.25,
        "target_hash": "sha256:" + "2" * 64,
        "seed_namespace": 73,
    }

    first = common_random_group(**arguments)
    second = common_random_group(**arguments)

    assert first == second
    assert first["aleatory_seed"] >= 0
    assert first["epistemic_group_id"].startswith("sphere-crn-")


def test_science_comparison_ignores_provenance_but_detects_response_drift() -> None:
    legacy = {
        "status": "ok",
        "scientific_completion_status": "complete_model_domain",
        "penetration_domain_coverage": {"affected_replicate_fraction": 0.0},
        "summary": {"x": {"mean": 0.25}},
        "components": {"c": {"fragment_metrics": {"hits": {"mean": 1.0}}}},
        "damage_tree": {"outputs": {"m": {"mean": 0.5}}},
        "provenance": {"case_hash": "legacy"},
    }
    sphere = deepcopy(legacy)
    sphere["provenance"] = {"case_hash": "sphere"}

    report = compare_science_outputs(legacy, sphere)

    assert report["passed"] is True
    sphere["summary"]["x"]["mean"] = 0.3
    report = compare_science_outputs(legacy, sphere)
    assert report["passed"] is False
    assert report["mismatch_paths"] == ["summary/x/mean"]


def test_one_real_pair_preserves_contact_and_teacher_science() -> None:
    engine = _engine()
    base = _load(ROOT / "configs/cases/sampling_reference_r4.json")
    base["target_ref"]["content_hash"] = engine.target_hash
    spec = _load(ROOT / "configs/sampling/p0_contact_reference.json")
    manifold = spec["contact_manifold"]
    latent = [1.0 / 12.0, 0.5, 0.5, 0.2, 0.25, 0.75]

    legacy = deepcopy(base)
    sphere = deepcopy(base)
    for case in (legacy, sphere):
        apply_munition_profile(case, "medium_front")
        apply_contact_manifold(case, manifold, latent)
    contact_record = sphere["metadata"]["contact_manifold_sampling"]
    sphere_record = apply_sphere_entry_reanchor(
        sphere,
        engine.compiled_target,
        target_hash=engine.target_hash,
        clearance_m=0.01,
        object_scope="all",
        latent_layout=PAIRED_LEGACY_STANDOFF_6D,
        latent=latent,
        line_point_target_m=contact_record["aim_point_target_m"],
    )
    group = common_random_group(
        physical_ray_id=sphere_record["physical_ray_id"],
        profile_id="medium_front",
        impact_speed_mps=contact_record["speed_mps"],
        body_x_roll_rad=contact_record["body_x_roll_rad"],
        target_hash=engine.target_hash,
        seed_namespace=91,
    )

    report = evaluate_case_pair(
        engine,
        legacy,
        sphere,
        selection_declaration=spec["contact_conditioning"]["selection_declaration"],
        replicates=2,
        common_group=group,
    )

    assert report["geometry"]["passed"] is True
    assert report["teacher"]["passed"] is True
    assert report["legacy_contact"]["collision_armor_id"] == report[
        "sphere_contact"
    ]["collision_armor_id"]
    assert report["legacy_contact"]["collision_distance_m"] != report[
        "sphere_contact"
    ]["collision_distance_m"]


def test_reduced_5d_pair_matches_paired_6d_physical_ray_and_teacher() -> None:
    engine = _engine()
    base = _load(ROOT / "configs/cases/sampling_reference_r4.json")
    legacy_spec = _load(ROOT / "configs/sampling/p0_contact_reference.json")
    paired_spec = _load(
        ROOT / "configs/sampling/p0_sphere_entry_reference.v1.json"
    )
    physical_latent = [1.0 / 12.0, 0.25, 0.75, 0.4, 0.6]

    paired, reduced, identity = build_layout_reduction_pair(
        engine=engine,
        base_case=base,
        legacy_manifold=legacy_spec["contact_manifold"],
        physical_latent=physical_latent,
        profile_id="medium_front",
        clearance_m=paired_spec["contact_manifold"]["sphere"]["clearance_m"],
        object_scope=paired_spec["contact_manifold"]["sphere"]["object_scope"],
        dummy_standoff_quantile=0.5,
    )

    assert identity["paired_latent_layout"] == PAIRED_LEGACY_STANDOFF_6D
    assert identity["reduced_latent_layout"] == REDUCED_PHYSICAL_RAY_5D
    assert identity["paired_physical_ray_id"] == identity["reduced_physical_ray_id"]
    assert paired["encounter"]["trajectory_origin_m"] == reduced["encounter"][
        "trajectory_origin_m"
    ]

    group = common_random_group(
        physical_ray_id=identity["paired_physical_ray_id"],
        profile_id="medium_front",
        impact_speed_mps=identity["speed_mps"],
        body_x_roll_rad=identity["body_x_roll_rad"],
        target_hash=engine.target_hash,
        seed_namespace=92,
    )
    report = evaluate_layout_reduction_pair(
        engine,
        paired,
        reduced,
        selection_declaration=legacy_spec["contact_conditioning"][
            "selection_declaration"
        ],
        replicates=2,
        common_group=group,
    )

    assert report["gate"]["overall_pass"] is True
    assert report["gate"]["checks"] == {
        "physical_ray_id_equal": True,
        "both_s_contact_admitted": True,
        "first_armor_object_equal": True,
        "geometry_equal": True,
        "teacher_science_equal": True,
    }
