from __future__ import annotations

import pytest

from damage_engine.exposure_audit import build_exposure_teacher_audit
from damage_engine.exposure_field import (
    PROXY_REACHED,
    PROXY_UNREACHED,
    build_exposure_field,
    exposure_footprint,
)
from damage_engine.exposure_sampling import select_exposure_balanced_arm


def _rows() -> list[dict]:
    return [
        {
            "candidate_id": f"c{index}",
            "cell_id": "a" if index < 3 else "b",
            "measure_weight": 1.0,
            "proxy_domain_status": "complete_proxy_domain",
            "component_lambda": {
                "engine": float([2, 1, 0, 0, 0.5, 0.2][index]),
                "sensor": float([0, 0, 0, 1, 2, 0.2][index]),
                "never": 0.0,
            },
            "component_damage_energy_j": {
                "engine": float([20, 10, 0, 0, 5, 2][index]),
                "sensor": float([0, 0, 0, 10, 20, 2][index]),
                "never": 0.0,
            },
        }
        for index in range(6)
    ]


def test_exposure_field_reports_hajek_ess_cells_and_non_structural_zeros():
    report = build_exposure_field(_rows())

    assert report["measure_ess"] == pytest.approx(6.0)
    assert report["components"]["engine"]["reachability_status"] == PROXY_REACHED
    assert report["components"]["never"]["reachability_status"] == PROXY_UNREACHED
    assert report["components"]["never"]["structural_zero_claim"] is False
    assert report["components"]["engine"]["lambda_hajek"] == pytest.approx(3.7 / 6)
    assert set(report["cell_envelopes"]) == {"a", "b"}


def test_exposure_balanced_selection_is_fixed_size_positive_and_train_only():
    result = select_exposure_balanced_arm(
        _rows(), sample_size=4, seed=7, rho_min=0.05, iterations=12
    )

    assert result["selected_count"] == 4
    assert len(result["selected_candidate_ids"]) == 4
    assert result["reference_replacement"] is False
    assert result["selection_role"] == "train_enrichment_exposure_balanced"
    assert result["proxy_unreached_component_ids_excluded_from_optimization"] == [
        "never"
    ]
    assert all(row["inclusion_probability"] > 0.0 for row in result["candidate_ledger"])
    assert result["sampling_footprint"]["design_ess"] > 0.0


def test_exposure_footprint_uses_inverse_probability_hajek_weights():
    footprint = exposure_footprint(
        _rows(),
        ["c0", "c3", "c4"],
        {"c0": 0.5, "c3": 0.25, "c4": 0.75},
    )

    assert footprint["selected_count"] == 3
    assert 0.0 < footprint["design_ess_fraction"] <= 1.0
    assert footprint["components"]["never"]["structural_zero_claim"] is False


def test_teacher_audit_falsifies_proxy_non_reach_without_reclassifying_structural_zero():
    teacher = [
        {
            "candidate_id": "c2",
            "component_hit_probability": {
                "engine": 0.25,
                "sensor": 0.0,
                "never": 0.0,
            },
        },
        {
            "candidate_id": "c3",
            "component_hit_probability": {
                "engine": 0.0,
                "sensor": 0.5,
                "never": 0.0,
            },
        },
    ]

    audit = build_exposure_teacher_audit(_rows(), teacher)

    assert audit["teacher_positive_among_proxy_unreached"] == 1
    assert audit["interpretation"]["structural_zero_claim"] is False
    assert audit["components"]["engine"]["false_unreached_rate"] == pytest.approx(0.5)
