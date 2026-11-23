from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from damage_engine.errors import InputValidationError, OutOfDomainError
from damage_engine.geometry import (
    AABB_ENCLOSING_SPHERE_ALGORITHM_ID,
    RAY_INTERSECTION_POINT_DECIMALS,
    compile_target,
)
from damage_engine.normalization import normalize_target
from damage_engine.sphere_entry_ray_sampling import (
    PAIRED_LEGACY_STANDOFF_6D,
    REDUCED_PHYSICAL_RAY_5D,
    apply_sphere_entry_reanchor,
    map_sphere_entry_latent,
    physical_ray_identity,
    reanchor_ray_to_sphere,
    sphere_entry_latent_declaration,
)


def _box(identifier: str, role: str, center: list[float], size: list[float]) -> dict:
    return {
        "id": identifier,
        "role": role,
        "geometry": {
            "shape": "box",
            "center_m": center,
            "size_lwh_m": size,
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "response": {},
    }


def _compiled_target():
    return compile_target(
        {
            "target_id": "sphere-entry-test-target",
            "armor": [_box("armor", "armor", [0.0, 0.0, 0.0], [2.0, 2.0, 2.0])],
            "components": [
                _box("component", "component", [0.25, 0.0, 0.0], [0.5, 0.5, 0.5])
            ],
        }
    )


def _case() -> dict:
    return {
        "encounter": {
            "trajectory_origin_m": [-8.0, 0.25, 0.1],
            "velocity_target_mps": [100.0, 0.0, 0.0],
            "body_to_target_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "metadata": {
            "contact_manifold_sampling": {
                "status": "constructed_pending_s_contact",
                "aim_point_target_m": [0.0, 0.25, 0.1],
                "standoff_m": 8.0,
            }
        },
    }


def test_public_aabb_enclosing_sphere_has_declared_algorithm_and_scope() -> None:
    target = compile_target(
        {
            "target_id": "asymmetric-target",
            "armor": [_box("armor", "armor", [-1.0, 0.0, 0.0], [2.0, 2.0, 2.0])],
            "components": [
                _box("component", "component", [2.0, 1.0, 0.0], [2.0, 2.0, 2.0])
            ],
        }
    )

    sphere = target.enclosing_sphere(object_scope="all", clearance_m=0.25)
    armor = target.enclosing_sphere(object_scope="armor", clearance_m=0.0)

    assert sphere.algorithm_id == AABB_ENCLOSING_SPHERE_ALGORITHM_ID
    assert sphere.center_target_m == (0.5, 0.5, 0.0)
    assert sphere.radius_m == pytest.approx(math.sqrt(9.5) + 0.25)
    assert sphere.object_scope == "all"
    assert armor.center_target_m == (-1.0, 0.0, 0.0)
    assert armor.radius_m == pytest.approx(math.sqrt(3.0))
    assert sphere.as_document()["target_id"] == "asymmetric-target"

    for item in target.objects:
        lower = item.geometry.bounds_min
        upper = item.geometry.bounds_max
        for x in (lower[0], upper[0]):
            for y in (lower[1], upper[1]):
                for z in (lower[2], upper[2]):
                    assert sphere.contains_point((x, y, z))


def test_enclosing_sphere_definition_hash_is_stable_and_binds_design() -> None:
    target = _compiled_target()
    first = target.enclosing_sphere(object_scope="all", clearance_m=0.1)
    repeated = target.enclosing_sphere(object_scope="all", clearance_m=0.1)
    changed_clearance = target.enclosing_sphere(object_scope="all", clearance_m=0.2)
    changed_scope = target.enclosing_sphere(object_scope="armor", clearance_m=0.1)

    assert first.definition_hash == repeated.definition_hash
    assert first.as_document()["definition_hash"] == first.definition_hash
    assert first.definition_hash != changed_clearance.definition_hash
    assert first.definition_hash != changed_scope.definition_hash


@pytest.mark.parametrize(
    ("scope", "clearance"),
    [("components", 0.0), ("all", -1.0), ("all", float("nan"))],
)
def test_enclosing_sphere_rejects_unsupported_or_nonfinite_inputs(
    scope: str, clearance: float
) -> None:
    with pytest.raises(InputValidationError):
        _compiled_target().enclosing_sphere(
            object_scope=scope,  # type: ignore[arg-type]
            clearance_m=clearance,
        )


def test_ray_reanchor_matches_independent_worked_example() -> None:
    sphere = _compiled_target().enclosing_sphere(
        object_scope="all", clearance_m=5.0 - math.sqrt(3.0)
    )
    ray = reanchor_ray_to_sphere(
        line_point_target_m=[7.0, 3.0, 0.0],
        direction_target_xyz=[1.0, 0.0, 0.0],
        sphere=sphere,
        target_hash="sha256:worked-example",
    )

    assert ray.origin_target_m == pytest.approx((-4.0, 3.0, 0.0))
    assert ray.direction_target_xyz == pytest.approx((1.0, 0.0, 0.0))
    assert ray.transverse_offset_target_m == pytest.approx((0.0, 3.0, 0.0))
    assert ray.transverse_radius_m == pytest.approx(3.0)
    assert np.linalg.norm(np.asarray(ray.origin_target_m)) == pytest.approx(5.0)


def test_physical_ray_identity_is_translation_invariant_and_target_bound() -> None:
    first = physical_ray_identity(
        line_point_target_m=[-8.0, 0.25, 0.1],
        direction_target_xyz=[2.0, 0.0, 0.0],
        target_hash="sha256:target-a",
    )
    shifted = physical_ray_identity(
        line_point_target_m=[17.0, 0.25, 0.1],
        direction_target_xyz=[1.0, 0.0, 0.0],
        target_hash="sha256:target-a",
    )
    other_target = physical_ray_identity(
        line_point_target_m=[17.0, 0.25, 0.1],
        direction_target_xyz=[1.0, 0.0, 0.0],
        target_hash="sha256:target-b",
    )

    assert first["physical_ray_id"] == shifted["physical_ray_id"]
    assert first["moment_target_m"] == shifted["moment_target_m"]
    assert first["physical_ray_id"] != other_target["physical_ray_id"]


def test_ray_reanchor_rejects_a_line_outside_the_sphere() -> None:
    sphere = _compiled_target().enclosing_sphere(object_scope="all")
    with pytest.raises(OutOfDomainError, match="does not intersect"):
        reanchor_ray_to_sphere(
            line_point_target_m=[0.0, 2.0, 0.0],
            direction_target_xyz=[1.0, 0.0, 0.0],
            sphere=sphere,
            target_hash="sha256:target",
        )


def test_latent_layouts_share_the_same_five_physical_coordinates() -> None:
    paired = map_sphere_entry_latent(
        [0.1, 0.2, 0.3, 0.9, 0.4, 0.5],
        layout=PAIRED_LEGACY_STANDOFF_6D,
    )
    reduced = map_sphere_entry_latent(
        [0.1, 0.2, 0.3, 0.4, 0.5],
        layout=REDUCED_PHYSICAL_RAY_5D,
        legacy_standoff_quantile=0.9,
    )

    assert paired.physical_latent == reduced.physical_latent
    assert paired.legacy_contact_manifold_latent == reduced.legacy_contact_manifold_latent
    assert paired.no_op_latent_names == ("standoff_quantile",)
    assert reduced.no_op_latent_names == ()
    assert sphere_entry_latent_declaration(PAIRED_LEGACY_STANDOFF_6D)[
        "latent_dimension"
    ] == 6
    assert sphere_entry_latent_declaration(REDUCED_PHYSICAL_RAY_5D)[
        "latent_dimension"
    ] == 5


@pytest.mark.parametrize(
    ("latent", "layout"),
    [
        ([0.1] * 5, PAIRED_LEGACY_STANDOFF_6D),
        ([0.1] * 6, REDUCED_PHYSICAL_RAY_5D),
        ([0.0] + [0.1] * 5, PAIRED_LEGACY_STANDOFF_6D),
        ([0.1] * 5, "unknown"),
    ],
)
def test_latent_layout_validation_fails_closed(latent: list[float], layout: str) -> None:
    with pytest.raises(InputValidationError):
        map_sphere_entry_latent(latent, layout=layout)


def test_case_reanchor_preserves_physical_ray_and_exact_target_intersections() -> None:
    target = _compiled_target()
    case = _case()
    old_case = deepcopy(case)
    direction = np.array([1.0, 0.0, 0.0])
    old_hits = target.ray_hits(np.asarray(case["encounter"]["trajectory_origin_m"]), direction)

    record = apply_sphere_entry_reanchor(
        case,
        target,
        target_hash="sha256:test-target",
        clearance_m=0.1,
        object_scope="all",
        latent_layout=PAIRED_LEGACY_STANDOFF_6D,
        latent=[0.1, 0.2, 0.3, 0.9, 0.4, 0.5],
        line_point_target_m=[0.0, 0.25, 0.1],
    )
    new_hits = target.ray_hits(np.asarray(case["encounter"]["trajectory_origin_m"]), direction)

    assert case["encounter"]["velocity_target_mps"] == old_case["encounter"][
        "velocity_target_mps"
    ]
    assert case["encounter"]["body_to_target_orientation_xyzw"] == old_case[
        "encounter"
    ]["body_to_target_orientation_xyzw"]
    assert case["encounter"]["trajectory_origin_m"] != old_case["encounter"][
        "trajectory_origin_m"
    ]
    assert record["latent_layout"] == PAIRED_LEGACY_STANDOFF_6D
    assert record["no_op_latent_names"] == ["standoff_quantile"]
    assert case["metadata"]["contact_manifold_sampling"]["status"] == (
        "reanchored_pending_s_contact"
    )
    assert [hit.object_id for hit in new_hits] == [hit.object_id for hit in old_hits]
    for old, new in zip(old_hits, new_hits, strict=True):
        assert new.entry_point == pytest.approx(old.entry_point)
        assert new.entry_normal == pytest.approx(old.entry_normal)
        assert new.path_length_m == pytest.approx(old.path_length_m)


def test_same_directed_line_has_byte_identical_contact_point_after_reanchoring() -> None:
    root = Path(__file__).resolve().parents[1]
    target = compile_target(
        normalize_target(
            json.loads(
                (root / "configs/targets/target.fragment_only.v1.json").read_text(
                    encoding="utf-8"
                )
            )
        )
    )
    direction = np.array([-1.0, 0.0, 0.0])
    far_origin = np.array([9.572186094533954, -0.5, 1.225])
    near_origin = np.array([4.289000850536004, -0.5, 1.225])

    far = next(
        hit
        for hit in target.ray_hits(far_origin, direction)
        if hit.role == "armor"
    )
    near = next(
        hit
        for hit in target.ray_hits(near_origin, direction)
        if hit.role == "armor"
    )

    np.testing.assert_array_equal(far.entry_point, near.entry_point)
    assert far.path_length_m == pytest.approx(near.path_length_m, abs=1.0e-14)
    assert far.entry_normal == pytest.approx(near.entry_normal, abs=1.0e-14)
    origin_translation = float(np.dot(far_origin - near_origin, -direction))
    assert far.entry_t - near.entry_t == pytest.approx(origin_translation, abs=1.0e-14)

    half_grid_m = 0.5 * 10.0 ** (-RAY_INTERSECTION_POINT_DECIMALS)
    raw_reconstruction = far_origin + far.entry_t * direction
    assert np.max(np.abs(far.entry_point - raw_reconstruction)) <= (
        half_grid_m + 8.0 * np.finfo(np.float64).eps
    )

