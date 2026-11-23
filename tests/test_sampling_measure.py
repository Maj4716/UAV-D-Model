from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from damage_engine.errors import InputValidationError
from damage_engine.sampling_measure import (
    SAMPLING_MEASURE_BUNDLE_VERSION,
    SAMPLING_MEASURE_CONTRACT_VERSION,
    importance_weight,
    mixture_inclusion_probability,
    validate_sampling_measure_bundle,
    validate_sampling_measure_contract,
)


def _weight_semantics() -> dict:
    return {
        "design": {
            "field": "design_weight_raw",
            "formula": "inverse_inclusion_probability",
            "purpose": "finite_frame_design_inference",
        },
        "evaluation": {
            "field": "evaluation_importance_weight",
            "formula": "target_density_over_proposal_density",
            "purpose": "target_measure_evaluation",
        },
        "training": {
            "field": "training_objective_weight",
            "formula": "declared_training_objective_weight",
            "purpose": "optimization_objective_only",
        },
    }


def _contract(role: str) -> dict:
    kind = {
        "reference": "declared_reference_measure",
        "coverage": "training_coverage_measure",
    }[role]
    density = {
        "reference": "declared_reference_density",
        "coverage": "coverage_design_density",
    }[role]
    return {
        "schema_version": SAMPLING_MEASURE_CONTRACT_VERSION,
        "status": "available",
        "role": role,
        "measure_kind": kind,
        "estimand": {
            "estimand_id": "criterion_exceedance_probability_given_first_armor_contact",
            "definition": {
                "population": "first_armor_contact_candidates",
                "outcome": "criterion_exceedance",
            },
        },
        "measure": {
            "measure_id": f"{role}-measure-v1",
            "definition": {"approach_profile_cells": 18, "normalization": "unit_mass"},
        },
        "proposal": {
            "proposal_id": f"{role}-proposal-v1",
            "definition": {"sequence": "randomized_lhs", "outcome_blind": True},
        },
        "condition_on_s_contact": True,
        "target_density_semantics": density,
        "inclusion_probability_field": "inclusion_probability",
        "weight_semantics": _weight_semantics(),
    }


def _operational_contract() -> dict:
    reference = _contract("reference")
    for field in ("proposal", "inclusion_probability_field", "weight_semantics"):
        reference.pop(field)
    reference.update(
        {
            "role": "operational",
            "measure_kind": "operational_frequency_measure",
            "target_density_semantics": "operational_prevalence_density",
        }
    )
    reference["measure"]["measure_id"] = "mission-frequency-v1"
    reference["measure"]["definition"] = {
        "source": "declared_mission_scenario",
        "normalization": "unit_mass",
    }
    return reference


def test_reference_contract_computes_and_replays_all_hashes() -> None:
    validated = validate_sampling_measure_contract(_contract("reference"))
    replayed = validate_sampling_measure_contract(validated)

    assert validated == replayed
    assert validated["estimand"]["estimand_hash"].startswith("sha256:")
    assert validated["measure"]["measure_hash"].startswith("sha256:")
    assert validated["proposal"]["proposal_hash"].startswith("sha256:")
    assert validated["contract_hash"].startswith("sha256:")


@pytest.mark.parametrize(
    "mutation,match",
    [
        (("role", "reference"), "role"),
        (("measure_kind", "declared_reference_measure"), "measure_kind"),
        (("target_density_semantics", "operational_prevalence_density"), "density"),
    ],
)
def test_coverage_cannot_masquerade_as_another_measure_role(mutation, match) -> None:
    document = _contract("coverage")
    document[mutation[0]] = mutation[1]

    with pytest.raises(InputValidationError, match=match):
        validate_sampling_measure_contract(document)


def test_reference_and_coverage_require_separate_weight_semantics() -> None:
    document = _contract("reference")
    document["weight_semantics"]["evaluation"]["field"] = "design_weight_raw"

    with pytest.raises(InputValidationError, match="distinct fields"):
        validate_sampling_measure_contract(document)


def test_declared_hash_mismatch_fails_closed() -> None:
    document = _contract("reference")
    document["measure"]["measure_hash"] = "sha256:" + "0" * 64

    with pytest.raises(InputValidationError, match="measure_hash"):
        validate_sampling_measure_contract(document)


def test_bundle_requires_explicit_operational_unavailable_state() -> None:
    with pytest.raises(InputValidationError, match="operational"):
        validate_sampling_measure_bundle(
            {
                "schema_version": SAMPLING_MEASURE_BUNDLE_VERSION,
                "reference": _contract("reference"),
                "coverage": _contract("coverage"),
            }
        )

    validated = validate_sampling_measure_bundle(
        {
            "schema_version": SAMPLING_MEASURE_BUNDLE_VERSION,
            "operational": {
                "status": "unavailable",
                "reason": "mission-frequency distribution has not been supplied",
            },
            "reference": _contract("reference"),
            "coverage": _contract("coverage"),
        }
    )

    assert validated["operational"]["status"] == "unavailable"
    assert validated["operational"]["reason"].startswith("mission-frequency")
    assert validated["bundle_hash"].startswith("sha256:")


def test_bundle_accepts_a_versioned_available_operational_measure() -> None:
    validated = validate_sampling_measure_bundle(
        {
            "schema_version": SAMPLING_MEASURE_BUNDLE_VERSION,
            "operational": _operational_contract(),
            "reference": _contract("reference"),
            "coverage": _contract("coverage"),
        }
    )

    assert validated["operational"]["role"] == "operational"
    assert validated["operational"]["contract_hash"].startswith("sha256:")


def test_bundle_requires_comparable_estimand_and_contact_conditioning() -> None:
    coverage = _contract("coverage")
    coverage["condition_on_s_contact"] = False
    document = {
        "schema_version": SAMPLING_MEASURE_BUNDLE_VERSION,
        "operational": {"status": "unavailable", "reason": "not supplied"},
        "reference": _contract("reference"),
        "coverage": coverage,
    }

    with pytest.raises(InputValidationError, match="condition_on_s_contact"):
        validate_sampling_measure_bundle(document)


def test_importance_weight_supports_scalars_and_arrays() -> None:
    assert importance_weight(0.2, 0.1) == pytest.approx(2.0)
    np.testing.assert_allclose(
        importance_weight([0.2, 0.0, 0.3], [0.1, 0.0, 0.6]),
        [2.0, 0.0, 0.5],
    )


def test_importance_weight_detects_positivity_violation() -> None:
    with pytest.raises(InputValidationError, match="positivity"):
        importance_weight([0.2, 0.3], [0.1, 0.0])


def test_mixture_inclusion_probability_uses_union_not_sum() -> None:
    observed = mixture_inclusion_probability(
        [0.6, 0.7], independent_selections=True
    )

    assert observed == pytest.approx(1.0 - (1.0 - 0.6) * (1.0 - 0.7))
    assert observed == pytest.approx(0.88)
    assert observed != pytest.approx(0.6 + 0.7)


def test_mixture_inclusion_requires_independence_or_joint_probability() -> None:
    with pytest.raises(InputValidationError, match="joint inclusion"):
        mixture_inclusion_probability([0.2, 0.4], independent_selections=False)


@pytest.mark.parametrize("probabilities", [[], [-0.1, 0.2], [0.2, 1.1]])
def test_mixture_inclusion_probability_validates_inputs(probabilities) -> None:
    with pytest.raises(InputValidationError, match="probabil"):
        mixture_inclusion_probability(
            probabilities,
            independent_selections=True,
        )


def test_project_sphere_entry_measure_bundle_is_explicit_and_valid() -> None:
    root = Path(__file__).resolve().parents[1]
    source = json.loads(
        (
            root
            / "configs"
            / "sampling"
            / "sphere_entry_measure_bundle.v1.json"
        ).read_text(encoding="utf-8")
    )

    bundle = validate_sampling_measure_bundle(source)

    assert bundle["operational"]["status"] == "unavailable"
    assert bundle["reference"]["proposal"]["definition"]["sequence"] == (
        "scrambled_sobol"
    )
    assert bundle["coverage"]["measure"]["definition"][
        "reference_replacement"
    ] is False
    assert bundle["reference"]["estimand"]["estimand_hash"] == bundle[
        "coverage"
    ]["estimand"]["estimand_hash"]
