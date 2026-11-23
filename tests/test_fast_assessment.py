from __future__ import annotations

import json
import math

import pytest

from damage_engine.errors import InputValidationError
from damage_engine.exposure_field import PROXY_UNREACHED
from damage_engine.fast_assessment import (
    compare_engine_evidence,
    evaluate_fast_assessment,
)


def _case_rows() -> list[dict]:
    values = (
        ("c0", "a", [0.10, 0.15]),
        ("c1", "a", [0.25, 0.30]),
        ("c2", "b", [0.70, 0.75]),
        ("c3", "b", [0.90, 0.85]),
    )
    return [
        {
            "candidate_id": candidate_id,
            "cell_id": cell_id,
            "measure_weight": 1.0,
            "latent": latent,
            "case": {"case_id": candidate_id},
        }
        for candidate_id, cell_id, latent in values
    ]


def _exposure_rows() -> list[dict]:
    values = {
        "c0": ("a", 0.0, 0.5),
        "c1": ("a", 0.7, 0.2),
        "c2": ("b", 0.2, 1.2),
        "c3": ("b", 1.0, 0.1),
    }
    return [
        {
            "candidate_id": candidate_id,
            "cell_id": cell,
            "measure_weight": 1.0,
            "proxy_domain_status": "complete_proxy_domain",
            "component_lambda": {"engine": engine, "sensor": sensor},
            "component_damage_energy_j": {
                "engine": 10.0 * engine,
                "sensor": 10.0 * sensor,
            },
        }
        for candidate_id, (cell, engine, sensor) in values.items()
    ]


def test_fast_assessment_builds_three_tier_contract_and_ipw_footprints():
    result = evaluate_fast_assessment(
        _case_rows(),
        exposure_rows=_exposure_rows(),
        selected_ids=["c0", "c2"],
        inclusion_probabilities={"c0": 0.5, "c2": 0.5},
    )

    tier_1 = result["tier_1_generator"]
    global_ball = tier_1["reference"]["global"]
    assert global_ball["dimension"] == 2
    assert global_ball["effective_rank"] == 2
    assert set(tier_1["reference"]["cells"]) == {"a", "b"}
    radii = global_ball["mahalanobis_radius"]
    assert radii["q50"] <= radii["q90"] <= radii["q95"] <= radii["q99"]

    selection = tier_1["selection_comparison"]
    assert selection["status"] == "evaluated"
    assert selection["raw_selection"]["cell_mass_total_variation"] == pytest.approx(0.0)
    assert selection["inverse_probability_recovered"][
        "cell_mass_total_variation"
    ] == pytest.approx(0.0)
    assert selection["inverse_probability_recovered"]["design_ess"] == pytest.approx(2.0)

    tier_2 = result["tier_2_geometry_proxy"]
    assert tier_2["exposure_field"]["candidate_count"] == 4
    assert tier_2["sampling_footprint"]["selected_count"] == 2
    assert tier_2["proxy_zero_semantics"] == PROXY_UNREACHED
    assert tier_2["structural_zero_claim"] is False
    assert result["tier_3_engine"]["status"] == "not_evaluated"
    json.dumps(result, allow_nan=False)


def test_l1_extracts_metadata_physical_latent_and_reports_rank_deficiency():
    rows = []
    exposure = []
    for index, value in enumerate((0.2, 0.4, 0.6)):
        candidate_id = f"m{index}"
        rows.append(
            {
                "candidate_id": candidate_id,
                "cell_id": "one-cell",
                "case": {
                    "case_id": candidate_id,
                    "metadata": {
                        "continuous_direction_sampling": {
                            "physical_latent": [value, value]
                        }
                    },
                },
            }
        )
        exposure.append(
            {
                "candidate_id": candidate_id,
                "cell_id": "one-cell",
                "measure_weight": 1.0,
                "proxy_domain_status": "complete_proxy_domain",
                "component_lambda": {"component": value},
                "component_damage_energy_j": {"component": value},
            }
        )

    result = evaluate_fast_assessment(rows, exposure_rows=exposure)
    ball = result["tier_1_generator"]["reference"]["global"]

    assert ball["dimension"] == 2
    assert ball["effective_rank"] == 1
    assert all(math.isfinite(value) for value in ball["eigenvalues_descending"])


def test_proxy_output_scope_falls_back_to_legacy_evaluate_signature():
    class LegacyProxy:
        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, case):
            self.calls += 1
            value = 0.1 + 0.1 * int(case["case_id"][-1])
            return {
                "case_id": case["case_id"],
                "proxy_domain_status": "complete_proxy_domain",
                "component_metrics": {
                    "component": {
                        "expected_hits": value,
                        "expected_damage_energy_j": 2.0 * value,
                    }
                },
            }

    proxy = LegacyProxy()
    result = evaluate_fast_assessment(_case_rows(), proxy=proxy)

    assert proxy.calls == 4
    assert result["tier_2_geometry_proxy"]["exposure_field"][
        "candidate_count"
    ] == 4


def test_engine_comparison_is_fixed_case_weighted_and_reuses_existing_audits():
    assessment = evaluate_fast_assessment(
        _case_rows(), exposure_rows=_exposure_rows()
    )
    teacher = [
        {
            "candidate_id": "c0",
            "component_hit_probability": {"engine": 0.20, "sensor": 0.35},
        },
        {
            "candidate_id": "c1",
            "component_hit_probability": {"engine": 0.45, "sensor": 0.15},
        },
        {
            "candidate_id": "c2",
            "component_hit_probability": {"engine": 0.18, "sensor": 0.72},
        },
    ]

    result = compare_engine_evidence(
        assessment, teacher, split_role="validation"
    )

    assert result["status"] == "evaluated"
    assert result["paired_candidate_ids"] == ["c0", "c1", "c2"]
    assert result["overall"]["pair_count"] == 6
    assert result["overall"]["weighted_probability_mse"] == pytest.approx(
        result["overall"]["weighted_brier"]
    )
    assert result["teacher_role_counts"] == {"unspecified": 3}
    assert result["overall"]["probability_mapping"] == (
        "p_proxy=1-exp(-component_lambda)"
    )
    assert result["reachability_audit"][
        "teacher_positive_among_proxy_unreached"
    ] == 1
    assert result["paired_efficiency"]["status"] == "evaluated"
    assert result["worst_cell"]["cell_id"] in {"a", "b"}
    assert result["proxy_zero_semantics"] == PROXY_UNREACHED
    assert result["structural_zero_claim"] is False
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("split_role", ["test", "sealed_test", "sealed-test"])
def test_engine_comparison_rejects_test_and_sealed_roles(split_role: str):
    assessment = evaluate_fast_assessment(
        _case_rows(), exposure_rows=_exposure_rows()
    )
    teacher = [
        {
            "candidate_id": "c0",
            "component_hit_probability": {"engine": 0.0, "sensor": 0.1},
        }
    ]

    with pytest.raises(InputValidationError, match="test or sealed"):
        compare_engine_evidence(assessment, teacher, split_role=split_role)


def test_engine_comparison_rejects_row_level_sealed_marker():
    assessment = evaluate_fast_assessment(
        _case_rows(), exposure_rows=_exposure_rows()
    )
    teacher = [
        {
            "candidate_id": "c0",
            "split": "sealed_test",
            "component_hit_probability": {"engine": 0.0, "sensor": 0.1},
        }
    ]

    with pytest.raises(InputValidationError, match="test or sealed"):
        compare_engine_evidence(assessment, teacher)


def test_engine_comparison_requires_aggregate_role_to_cover_row_roles():
    assessment = evaluate_fast_assessment(
        _case_rows(), exposure_rows=_exposure_rows()
    )
    teacher = [
        {
            "candidate_id": "c0",
            "split_role": "train",
            "component_hit_probability": {"engine": 0.0, "sensor": 0.1},
        },
        {
            "candidate_id": "c1",
            "split_role": "validation",
            "component_hit_probability": {"engine": 0.2, "sensor": 0.3},
        },
    ]

    with pytest.raises(InputValidationError, match="does not describe"):
        compare_engine_evidence(assessment, teacher, split_role="validation")

    result = compare_engine_evidence(
        assessment,
        teacher,
        split_role="development_train_validation",
    )
    assert result["teacher_role_counts"] == {"train": 1, "validation": 1}
