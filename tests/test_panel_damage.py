from __future__ import annotations

import math

import numpy as np
import pytest

from damage_engine.panel_damage import PanelDamageState


def test_panel_damage_reuses_existing_hole_and_does_not_double_count_area():
    state = PanelDamageState({"armor-a": 1.0})
    center = np.array([0.0, 0.0, 0.0])
    normal = np.array([1.0, 0.0, 0.0])
    area = math.pi * 0.01**2

    first_increment = state.record_perforation(
        "armor-a", center, normal, hole_area_m2=area
    )
    second_increment = state.record_perforation(
        "armor-a", center, normal, hole_area_m2=area
    )

    assert first_increment == pytest.approx(area)
    assert second_increment == 0.0
    assert state.is_open("armor-a", center, normal)
    assert state.is_open(
        "armor-a", center, normal, clearance_radius_m=0.01
    )
    assert not state.is_open(
        "armor-a",
        np.array([0.0, 0.005, 0.0]),
        normal,
        clearance_radius_m=0.006,
    )
    assert not state.is_open(
        "armor-a", center, normal, query_time_s=0.0
    )
    assert state.is_open(
        "armor-a", center, normal, query_time_s=1.0e-6
    )
    assert state.covered_area_m2("armor-a") == pytest.approx(area)
    assert state.integrity("armor-a") == pytest.approx(1.0 - area)
    summary = state.summary()["armor-a"]
    assert summary["perforation_event_count"] == 2
    assert summary["opening_primitive_count"] == 1
    assert summary["opening_connected_component_count"] == 1


def test_panel_damage_keeps_different_surfaces_spatially_distinct():
    state = PanelDamageState({"armor-a": 2.0})
    area = math.pi * 0.01**2
    state.record_perforation(
        "armor-a",
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        hole_area_m2=area,
    )

    assert not state.is_open(
        "armor-a",
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    )
    assert not state.is_open(
        "armor-a",
        np.array([0.0, 0.1, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    )


def test_partial_overlap_union_estimate_is_bounded_and_deterministic():
    first = PanelDamageState({"armor-a": 1.0})
    second = PanelDamageState({"armor-a": 1.0})
    area = math.pi * 0.02**2
    records = (
        (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
        (np.array([0.0, 0.02, 0.0]), np.array([1.0, 0.0, 0.0])),
    )
    for state in (first, second):
        for center, normal in records:
            state.record_perforation(
                "armor-a", center, normal, hole_area_m2=area
            )

    union = first.covered_area_m2("armor-a")
    assert area < union < 2.0 * area
    assert union == second.covered_area_m2("armor-a")
    summary = first.summary()["armor-a"]
    assert summary["opening_primitive_count"] == 2
    assert summary["opening_connected_component_count"] == 1

    first.record_perforation(
        "armor-a",
        np.array([0.0, 0.2, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        hole_area_m2=area,
    )
    summary = first.summary()["armor-a"]
    assert summary["opening_primitive_count"] == 3
    assert summary["opening_connected_component_count"] == 2


def test_contained_earlier_hole_remains_visible_before_later_large_hole_forms():
    state = PanelDamageState({"armor-a": 1.0})
    center = np.array([0.0, 0.0, 0.0])
    normal = np.array([1.0, 0.0, 0.0])
    large_area = math.pi * 0.02**2
    small_area = math.pi * 0.01**2

    state.record_perforation(
        "armor-a",
        center,
        normal,
        hole_area_m2=large_area,
        formed_time_s=2.0,
    )
    increment = state.record_perforation(
        "armor-a",
        center,
        normal,
        hole_area_m2=small_area,
        formed_time_s=1.0,
    )

    assert increment == 0.0
    assert state.is_open(
        "armor-a",
        center,
        normal,
        clearance_radius_m=0.009,
        query_time_s=1.5,
    )
    assert not state.is_open(
        "armor-a",
        np.array([0.0, 0.015, 0.0]),
        normal,
        query_time_s=1.5,
    )
    assert state.covered_area_m2(
        "armor-a", query_time_s=1.5
    ) == pytest.approx(small_area)
    assert state.integrity(
        "armor-a", query_time_s=1.5
    ) == pytest.approx(1.0 - small_area)
    assert state.covered_area_m2("armor-a") == pytest.approx(large_area)
    summary = state.summary()["armor-a"]
    assert summary["perforation_event_count"] == 2
    assert summary["hole_count"] == 1
    assert summary["opening_primitive_count"] == 1
    assert summary["opening_connected_component_count"] == 1


def test_opening_activation_uses_strict_event_order_without_fixed_time_delay():
    state = PanelDamageState({"armor-a": 1.0})
    center = np.array([0.0, 0.0, 0.0])
    normal = np.array([1.0, 0.0, 0.0])
    state.record_perforation(
        "armor-a",
        center,
        normal,
        hole_area_m2=0.01,
        formed_time_s=1.0e-12,
    )

    assert not state.is_open(
        "armor-a", center, normal, query_time_s=1.0e-12
    )
    assert state.is_open(
        "armor-a", center, normal, query_time_s=2.0e-12
    )
