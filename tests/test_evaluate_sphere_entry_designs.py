from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from damage_engine.engine import ENGINE_CODE_HASH, DamageEngine
from damage_engine.registry import PhysicsRegistry
from tools.evaluate_sphere_entry_designs import (
    _current_engine_code_hash,
    DESIGN_METHODS,
    evaluate_design_cell,
    incidence_angle_deg,
    lexicographic_recommendation,
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


def test_incidence_angle_is_zero_for_head_on_and_ninety_for_grazing() -> None:
    assert incidence_angle_deg([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert incidence_angle_deg([0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]) == pytest.approx(90.0)


def test_lexicographic_recommendation_uses_declared_pre_outcome_order() -> None:
    summaries = {
        "randomized_lhs": {
            "failed_s_contact_count": 0,
            "minimum_cell_contact_rate": 1.0,
            "unique_first_armor_ids": 5,
            "incidence_angle_occupied_bin_count": 3,
            "worst_postscreen_maximum_2d_discrepancy": 0.01,
            "worst_postscreen_minimum_nn_distance": 0.5,
            "worst_prescreen_global_discrepancy": 0.01,
        },
        "scrambled_sobol": {
            "failed_s_contact_count": 0,
            "minimum_cell_contact_rate": 1.0,
            "unique_first_armor_ids": 6,
            "incidence_angle_occupied_bin_count": 1,
            "worst_postscreen_maximum_2d_discrepancy": 0.9,
            "worst_postscreen_minimum_nn_distance": 0.01,
            "worst_prescreen_global_discrepancy": 0.9,
        },
        "maximin_lhs": {
            "failed_s_contact_count": 1,
            "minimum_cell_contact_rate": 0.99,
            "unique_first_armor_ids": 99,
            "incidence_angle_occupied_bin_count": 9,
            "worst_postscreen_maximum_2d_discrepancy": 0.001,
            "worst_postscreen_minimum_nn_distance": 0.9,
            "worst_prescreen_global_discrepancy": 0.001,
        },
    }

    recommendation = lexicographic_recommendation(summaries)

    assert recommendation["recommended_method"] == "scrambled_sobol"
    assert recommendation["outcome_blind"] is True
    assert recommendation["ranking"][0] == "scrambled_sobol"
    assert recommendation["criterion_order"][0]["field"] == "failed_s_contact_count"


def test_real_cell_reports_pre_and_postscreen_geometry_without_teacher() -> None:
    engine = _engine()
    specification = _load(
        ROOT / "configs/sampling/p0_sphere_entry_reduced_5d.v1.json"
    )
    base_case = _load(ROOT / "configs/cases/sampling_reference_r4.json")
    report = evaluate_design_cell(
        engine=engine,
        base_case=base_case,
        reduced_specification=specification,
        approach_index=0,
        profile_id="medium_front",
        method="scrambled_sobol",
        sample_count=4,
        seed=71,
        maximin_restarts=4,
    )

    assert report["method"] in DESIGN_METHODS
    assert report["candidate_count"] == 4
    assert report["prescreen_design_diagnostics"]["dimension"] == 4
    assert report["accepted_count"] == 4
    assert report["contact_rate"] == pytest.approx(1.0)
    assert report["postscreen_design_diagnostics"]["sample_count"] == 4
    assert report["teacher_executed"] is False
    assert report["first_armor_coverage"]["unique_count"] >= 1
    assert report["normal_coverage"]["unique_count"] >= 1
    assert report["incidence_angle_coverage_deg"]["maximum"] <= 90.0
    assert all(
        np.isfinite(row["incidence_angle_deg"])
        for row in report["accepted_contacts"]
    )
