from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from damage_engine.errors import InputValidationError
from damage_engine.kmfc_ordinal_teacher import (
    KMFC_CATEGORY_ORDER,
    aggregate_kmfc_ordinal_teacher,
    compile_kmfc_ordinal_teacher,
    evaluate_kmfc_ordinal_teacher,
)


def _projection(category: str, names: list[str], *, tau1=0.3, tau2=0.7):
    return {
        "schema_version": "1.0.0",
        "projection_id": f"kmfc-{category.lower()}-software-fixture-v0",
        "category_id": category,
        "status": "provisional",
        "execution_scope": "software_only",
        "scientific_claim_scope": "project_specific_simulator_emulation",
        "physical_validity_status": "not_established",
        "threshold_approval_status": "not_approved_for_real_world_kmfc",
        "threshold_source_id": "software-property-fixture-not-calibration",
        "input_names": names,
        "evidence_scales": [1.0] * len(names),
        "evidence_exponents": [1.0] * len(names),
        "maximum_slopes": [1.0] * len(names),
        "numeric_tolerance": 1.0e-12,
        "aggregate_weights": [1.0] * len(names),
        "aggregate_power": 1.0,
        "tau1": tau1,
        "tau2": tau2,
        "minimum_threshold_gap": 0.1,
        "calibration_artifact_hash": None,
    }


def config_document(
    *,
    model: str = "state_expected_severity",
    tau1: float = 0.3,
    tau2: float = 0.7,
):
    if model == "state_expected_severity":
        quantity, unit = "component_state_expected_severity", "1"
    else:
        quantity, unit = "component_residual_kinetic_energy_j", "J"
    categories = {}
    for category in KMFC_CATEGORY_ORDER:
        name = f"{category}.fixture-input"
        categories[category] = {
            "evidence_model": model,
            "evidence_channels": [
                {
                    "name": name,
                    "quantity": quantity,
                    "unit": unit,
                    "source_field": f"replicate.{category}.fixture",
                    "required": True,
                }
            ],
            "projection": _projection(category, [name], tau1=tau1, tau2=tau2),
        }
    return {
        "schema_version": "1.0.0",
        "teacher_id": "kmfc-project-specific-software-fixture-v0",
        "status": "provisional",
        "execution_scope": "software_only",
        "scientific_claim_scope": "project_specific_simulator_emulation",
        "physical_validity_status": "not_established",
        "deployment_status": "prototype_not_admitted",
        "legacy_tree_policy": "preserve_unchanged",
        "availability_policy": "all_required_evidence_and_gate_conditions",
        "evaluation_order": "per_replicate_smooth_then_hard_gate_before_outer_aggregation",
        "parameter_selection_policy": "tree_design_fit_only_no_validation_or_test",
        "parameter_status": "software_fixture_not_calibrated",
        "category_order": list(KMFC_CATEGORY_ORDER),
        "categories": categories,
    }


def _same_evidence(rows):
    return {category: np.asarray(rows, dtype=np.float64) for category in KMFC_CATEGORY_ORDER}


def test_configuration_freezes_four_categories_and_software_only_claim_boundary():
    compiled = compile_kmfc_ordinal_teacher(config_document())
    payload = compiled.to_dict()
    rebuilt = compile_kmfc_ordinal_teacher(json.loads(json.dumps(payload)))
    assert rebuilt == compiled
    assert rebuilt.hash == compiled.hash
    assert tuple(item.category_id for item in compiled.categories) == KMFC_CATEGORY_ORDER
    assert payload["execution_scope"] == "software_only"
    assert payload["physical_validity_status"] == "not_established"
    assert payload["deployment_status"] == "prototype_not_admitted"
    assert payload["legacy_tree_policy"] == "preserve_unchanged"


@pytest.mark.parametrize("missing", KMFC_CATEGORY_ORDER)
def test_configuration_requires_exact_kmfc_category_inventory(missing):
    document = config_document()
    del document["categories"][missing]
    with pytest.raises(InputValidationError, match="exactly K, M, F and C"):
        compile_kmfc_ordinal_teacher(document)


def test_configuration_rejects_channel_projection_mismatch_and_unknown_scope():
    mismatch = config_document()
    mismatch["categories"]["M"]["projection"]["input_names"] = ["wrong"]
    with pytest.raises(InputValidationError, match="exactly match"):
        compile_kmfc_ordinal_teacher(mismatch)

    widened = config_document()
    widened["physical_validity_status"] = "validated"
    with pytest.raises(InputValidationError, match="software-only KMFC prototype"):
        compile_kmfc_ordinal_teacher(widened)


@pytest.mark.parametrize(
    "quantity",
    [
        "deposited_energy_j",
        "fragment_deposited_energy_j",
        "component_penetration_energy_loss_j",
        "penetration_energy_loss_j",
    ],
)
def test_penetration_energy_loss_is_not_silently_used_as_residual_energy(quantity):
    document = config_document(model="residual_kinetic_energy_hill")
    document["categories"]["K"]["evidence_channels"][0]["quantity"] = quantity
    with pytest.raises(InputValidationError, match="not interchangeable"):
        compile_kmfc_ordinal_teacher(document)


def test_evidence_model_enforces_quantity_unit_and_state_range():
    wrong_unit = config_document(model="residual_kinetic_energy_hill")
    wrong_unit["categories"]["F"]["evidence_channels"][0]["unit"] = "1"
    with pytest.raises(InputValidationError, match="unit"):
        compile_kmfc_ordinal_teacher(wrong_unit)

    compiled = compile_kmfc_ordinal_teacher(config_document())
    invalid = _same_evidence([[1.01]])
    with pytest.raises(InputValidationError, match=r"\[0,1\]"):
        evaluate_kmfc_ordinal_teacher(invalid, compiled)


def test_zero_anchor_nested_levels_and_exclusive_level_are_structural():
    compiled = compile_kmfc_ordinal_teacher(config_document())
    # Hill(scale=1) maps 0,.5,1 to 0,1/3,1/2 respectively.
    result = evaluate_kmfc_ordinal_teacher(
        _same_evidence([[0.0], [0.5], [1.0]]),
        compiled,
        necessary_level2_by_category={
            category: np.asarray([True, True, False])
            for category in KMFC_CATEGORY_ORDER
        },
    )
    for category in KMFC_CATEGORY_ORDER:
        item = result.category(category)
        assert item.projection.severity_score.tolist() == pytest.approx(
            [0.0, 1.0 / 3.0, 0.5]
        )
        assert item.projection.level_ge_1.tolist() == [0.0, 1.0, 1.0]
        assert item.projection.level_ge_2.tolist() == [0.0, 0.0, 0.0]
        assert item.exclusive_level.tolist() == [0.0, 1.0, 1.0]
        assert np.all(item.projection.level_ge_2 <= item.projection.level_ge_1)


def test_gate_is_applied_per_replicate_before_outer_aggregation():
    compiled = compile_kmfc_ordinal_teacher(
        config_document(
            model="residual_kinetic_energy_hill", tau1=0.6, tau2=0.9
        )
    )
    # Hill(scale=1) maps .25 and 4 to .2 and .8.  Per-replicate gating
    # therefore gives one success; gating their mean would incorrectly give 0.
    result = evaluate_kmfc_ordinal_teacher(_same_evidence([[0.25], [4.0]]), compiled)
    report = aggregate_kmfc_ordinal_teacher(result)
    for category in KMFC_CATEGORY_ORDER:
        assert result.category(category).projection.level_ge_1.tolist() == [0.0, 1.0]
        assert report["categories"][category]["successes_level_ge_1"] == 1
        assert report["categories"][category]["q1"] == 0.5


def test_unavailable_is_not_level_zero_and_each_category_keeps_its_denominator():
    compiled = compile_kmfc_ordinal_teacher(config_document())
    evidence = _same_evidence([[0.0], [np.nan], [1.0]])
    masks = {
        category: np.asarray([[True], [False], [True]], dtype=bool)
        for category in KMFC_CATEGORY_ORDER
    }
    # A known-false gate is a valid zero.  An unavailable gate is missing.
    result = evaluate_kmfc_ordinal_teacher(
        evidence,
        compiled,
        evidence_availability_by_category=masks,
        necessary_level1_by_category={"M": np.asarray([True, True, False])},
        necessary_level1_availability_by_category={
            "F": np.asarray([True, True, False])
        },
    )
    assert result.category("K").projection.availability.tolist() == [True, False, True]
    assert result.category("M").exclusive_level.tolist()[:1] == [0.0]
    assert result.category("M").exclusive_level[2] == 0.0
    assert result.category("F").projection.availability.tolist() == [True, False, False]
    assert np.isnan(result.category("F").exclusive_level[1:]).all()
    report = aggregate_kmfc_ordinal_teacher(result)
    assert report["categories"]["K"]["trials"] == 2
    assert report["categories"]["K"]["unavailable_replicates"] == 1
    assert report["categories"]["F"]["trials"] == 1
    assert report["categories"]["F"]["unavailable_replicates"] == 2
    json.dumps(result.to_dict(), allow_nan=False)
    json.dumps(report, allow_nan=False)


def test_residual_energy_model_accepts_nonnegative_joules_with_explicit_semantics():
    compiled = compile_kmfc_ordinal_teacher(
        config_document(model="residual_kinetic_energy_hill")
    )
    result = evaluate_kmfc_ordinal_teacher(
        _same_evidence([[0.0], [2.0]]), compiled
    )
    payload = result.to_dict()
    assert payload["categories"]["K"]["energy_semantics"] == (
        "component_response_residual_kinetic_energy"
    )
    assert payload["categories"]["K"]["evidence_channels"][0]["quantity"] == (
        "component_residual_kinetic_energy_j"
    )


def test_checked_in_unfitted_candidate_is_compilable_but_not_admitted():
    path = (
        Path(__file__).resolve().parents[1]
        / "review_evidence/human_factor_remediation_20261022"
        / "kmfc_project_specific_software_prototype_v0_unfitted.json"
    )
    compiled = compile_kmfc_ordinal_teacher(
        json.loads(path.read_text(encoding="utf-8"))
    )
    payload = compiled.to_dict()
    assert payload["teacher_id"].endswith("-unfitted")
    assert payload["parameter_status"] == "software_fixture_not_calibrated"
    assert payload["deployment_status"] == "prototype_not_admitted"
    assert all(
        category["projection"]["calibration_artifact_hash"] is None
        for category in payload["categories"].values()
    )

