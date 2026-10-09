from __future__ import annotations

import numpy as np
import pytest

from damage_engine.errors import InputValidationError
from damage_engine.geometry import (
    BoxGeometry,
    CylinderGeometry,
    ExtrudedPolygonGeometry,
    compile_target,
    geometry_from_json,
)


def test_oriented_box_ray_interval():
    box = BoxGeometry(np.zeros(3), np.array([2.0, 2.0, 2.0]), np.eye(3))
    hit = box.intersect(np.array([-3.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    assert hit is not None
    entry, exit_value, normal = hit
    assert entry == 2.0
    assert exit_value == 4.0
    np.testing.assert_allclose(normal, [-1.0, 0.0, 0.0])


def test_cylinder_ray_interval():
    cylinder = CylinderGeometry(np.zeros(3), 1.0, 2.0, np.eye(3))
    hit = cylinder.intersect(np.array([-3.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    assert hit is not None
    assert hit[0] == 2.0
    assert hit[1] == 4.0


def test_arbitrary_extruded_polygon_ray_interval():
    polygon = ExtrudedPolygonGeometry(
        vertices=np.array(
            [
                [-0.5, -1.0, -1.0],
                [-0.5, 1.0, -1.0],
                [-0.5, 1.0, 1.0],
                [-0.5, -1.0, 1.0],
            ]
        ),
        direction=np.array([1.0, 0.0, 0.0]),
        distance=0.02,
    )
    hit = polygon.intersect(np.array([-2.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    assert hit is not None
    assert abs(hit[0] - 1.5) < 1.0e-12
    assert abs(hit[1] - 1.52) < 1.0e-12


def test_compiled_target_orders_armor_before_component(target_data):
    target = compile_target(target_data)
    hits = target.ray_hits(np.array([-2.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    assert [hit.object_id for hit in hits] == ["armor-front", "flight-control"]
    assert hits[0].entry_t < hits[1].entry_t


def test_extruded_polygon_contains_all_boundary_edges():
    polygon = ExtrudedPolygonGeometry(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        direction=np.array([1.0, 0.0, 0.0]),
        distance=0.1,
    )
    for point in (
        np.array([0.05, 0.0, 0.5]),
        np.array([0.05, 1.0, 0.5]),
        np.array([0.05, 0.5, 0.0]),
        np.array([0.05, 0.5, 1.0]),
    ):
        assert polygon.contains(point)


def test_centimetre_aliases_fail_with_explicit_si_message():
    with pytest.raises(InputValidationError, match="required SI fields"):
        geometry_from_json(
            {
                "shape": "box",
                "center_cm": [0.0, 0.0, 0.0],
                "size_lwh_cm": [40.0, 30.0, 20.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        )


def test_extruded_prism_matches_equivalent_oriented_box_for_random_rays():
    axis = np.array([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    helper = np.array([0.0, 0.0, 1.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    center = np.array([0.4, -0.3, 0.2])
    depth, width, height = 0.4, 2.0, 3.0
    base_center = center - 0.5 * depth * axis
    vertices = np.array(
        [
            base_center - 0.5 * width * u - 0.5 * height * v,
            base_center + 0.5 * width * u - 0.5 * height * v,
            base_center + 0.5 * width * u + 0.5 * height * v,
            base_center - 0.5 * width * u + 0.5 * height * v,
        ]
    )
    prism = ExtrudedPolygonGeometry(vertices, axis, depth)
    box = BoxGeometry(
        center,
        np.array([depth, width, height]),
        np.column_stack((axis, u, v)),
    )

    rng = np.random.default_rng(20260926)
    for _ in range(200):
        radial = rng.normal(size=3)
        radial /= np.linalg.norm(radial)
        origin = center + 5.0 * radial
        aim = center + rng.uniform(-0.15, 0.15, size=3)
        direction = aim - origin
        direction /= np.linalg.norm(direction)
        prism_hit = prism.intersect(origin, direction)
        box_hit = box.intersect(origin, direction)
        assert (prism_hit is None) == (box_hit is None)
        if prism_hit is not None and box_hit is not None:
            np.testing.assert_allclose(prism_hit[:2], box_hit[:2], rtol=0.0, atol=1.0e-10)
