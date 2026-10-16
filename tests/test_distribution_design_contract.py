"""Artificial scalar tests for the outcome-blind distribution-design contract.

These tests never construct or execute DamageEngine.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from damage_engine.case_factory import CaseFactory, json_pointer_get
from damage_engine.errors import InputValidationError
from damage_engine.label_spec import LabelSpec
from damage_engine.sampling_spec import SamplingSpec
from damage_engine.utils import content_hash


def _label_spec() -> LabelSpec:
    return LabelSpec(
        {
            "schema_version": "1.0.0",
            "label_spec_id": "design-contract-toy-labels",
            "revision": "1",
            "status": "provisional",
            "source": {
                "damage_tree_id": "toy-tree",
                "damage_tree_revision": "1",
                "damage_tree_definition_hash": "sha256:" + "2" * 64,
            },
            "tasks": {
                "toy": {
                    "type": "multi_label_event_probability",
                    "outputs": ["toy-event"],
                    "loss": "binary_cross_entropy",
                }
            },
            "provenance": {"approval_status": "not_approved", "evidence_ref": None},
        }
    )


def _document() -> dict:
    variables = [
        {"name": "driver", "path": "/design/driver", "distribution": "uniform", "low": 0.0, "high": 1.0},
        {"name": "joint_a", "path": "/design/joint_a", "distribution": "uniform", "low": 0.0, "high": 10.0},
        {"name": "joint_b", "path": "/design/joint_b", "distribution": "uniform", "low": 100.0, "high": 200.0},
        {"name": "model", "path": "/design/model", "distribution": "choice", "values": ["off", "on"]},
        {"name": "conditional", "path": "/design/conditional", "distribution": "uniform", "low": 10.0, "high": 20.0},
    ]
    return {
        "schema_version": "1.1.0",
        "sampling_spec_id": "distribution-design-toy",
        "revision": "1",
        "status": "provisional",
        "reference_distribution": {
            "distribution_id": "toy-p0",
            "interpretation": "design_distribution",
            "variables": variables,
            "constraints": [{"kind": "sum_leq", "paths": ["/design/joint_a"], "limit": 10.0}],
        },
        "budgets": {"train_pilot": 2, "validation": 1, "test": 1, "rare_stress_test": 1},
        "seeds": {"design": 101, "selection": 102, "split": 103, "screening": 104, "formal": 105},
        "formal_policy": {"replicates": 2, "adaptive_ci_half_width": None, "epistemic_policy": "unique_per_lineage"},
        "split_policy": {"registry_id": "toy-design-splits", "freeze_before_outcomes": True, "adaptive_train_only": True},
        "quality_gates": {"positivity_min": 0.1, "min_ess_fraction": 0.5, "max_weight_cv": 2.0},
        "feature_roles": [
            {"feature_id": item["name"], "path": item["path"], "tensor_feature_names": [item["name"]], "role": "learnable_variable", "availability_stage": "raw_input", "deployment_available": True}
            for item in variables
        ],
        "variable_blocks": [
            {"block_id": "coupled-scale", "kind": "joint", "variable_names": ["joint_a", "joint_b"], "joint_mode": "shared_quantile"},
            {"block_id": "model-conditional", "kind": "conditional", "variable_names": ["conditional"], "condition": {"path": "/design/model", "operator": "equals", "value": "on"}, "inactive_policy": "preserve_base"},
        ],
        "geometry_strata": [
            {"stratum_id": "high-driver", "description": "Artificial pre-label upper interval", "pre_label": True, "predicate": {"all": [{"source": "case_input", "path": "/design/driver", "operator": "ge", "value": 0.6}]}}
        ],
        "stream_roles": [
            {"role": "reference_train", "stream": "reference", "split": "train", "train_eligible": True},
            {"role": "reference_validation", "stream": "reference", "split": "validation", "train_eligible": False},
            {"role": "reference_test", "stream": "reference", "split": "test", "train_eligible": False},
            {"role": "geometry_stratified_train", "stream": "geometry_stratified", "split": "train", "train_eligible": True},
            {"role": "rare_stress_test", "stream": "stress", "split": "test", "train_eligible": False},
        ],
        "round_budgets": [
            {"round": 0, "allocations": [
                {"role": "reference_validation", "stratum_id": "p0", "proposal_id": "toy-p0", "count": 1},
                {"role": "reference_test", "stratum_id": "p0", "proposal_id": "toy-p0", "count": 1},
                {"role": "reference_train", "stratum_id": "p0", "proposal_id": "toy-p0", "count": 1},
                {"role": "geometry_stratified_train", "stratum_id": "high-driver", "proposal_id": "toy-geometry-high", "count": 1},
                {"role": "rare_stress_test", "stratum_id": "high-driver", "proposal_id": "toy-stress-high", "count": 1},
            ]},
        ],
        "replicate_policy": {
            "default_replicates": 2,
            "outcome_adaptive": False,
            "overrides": [
                {"role": "geometry_stratified_train", "stratum_id": "high-driver", "replicates": 3},
                {"role": "rare_stress_test", "replicates": 4},
            ],
        },
        "key_projections": [
            {"projection_id": "driver-x-joint", "paths": ["/design/driver", "/design/joint_a"], "purpose": "coverage_audit", "pre_label_only": True}
        ],
        "proxy_scope": {
            "scope": "fixed_target",
            "geometry_boundary": "solver_proxy",
            "target_identity_policy": "fixed_target_hash",
            "deployment_features_match_training": True,
            "evidence_status": "unverified",
        },
        "legal_sampling": {"invalid_candidate_policy": "reject_and_resample", "max_attempts_per_allocation": 500},
    }


def _factory(case_data: dict, spec: SamplingSpec) -> CaseFactory:
    case = deepcopy(case_data)
    case["design"] = {"driver": 0.0, "joint_a": 0.0, "joint_b": 100.0, "model": "off", "conditional": -1.0}
    return CaseFactory(
        case,
        spec,
        _label_spec(),
        target_hash="sha256:" + "3" * 64,
        damage_tree_definition_hash="sha256:" + "2" * 64,
        physics_registry_hash="sha256:" + "4" * 64,
        engine_code_hash="sha256:" + "5" * 64,
    )


def test_extended_contract_generates_legal_role_aware_plan_without_engine(case_data) -> None:
    spec = SamplingSpec(_document())
    plan = _factory(case_data, spec).build_reference_plan()

    assert spec.latent_dimension == 4
    assert plan.latent_design.shape == (5, 4)
    assert [row["role"] for row in plan.candidate_rows] == [
        "reference_validation", "reference_test", "rare_stress_test",
        "reference_train", "geometry_stratified_train",
    ]
    assert [row["round"] for row in plan.candidate_rows] == [0, 0, 0, 0, 0]
    assert all(
        row["split"] == "test" and not row["weight_mask"]
        for row in plan.candidate_rows if row["role"] == "rare_stress_test"
    )
    assert all(
        row["split"] == "train"
        for row in plan.candidate_rows if row["role"] == "geometry_stratified_train"
    )
    assert all(
        json_pointer_get(case, "/design/driver") >= 0.6
        for case, row in zip(plan.cases, plan.candidate_rows)
        if row["stratum"] == "high-driver"
    )
    for case, row in zip(plan.cases, plan.candidate_rows):
        a = json_pointer_get(case, "/design/joint_a") / 10.0
        b = (json_pointer_get(case, "/design/joint_b") - 100.0) / 100.0
        assert a == pytest.approx(b)
        conditional = json_pointer_get(case, "/design/conditional")
        if json_pointer_get(case, "/design/model") == "on":
            assert 10.0 <= conditional <= 20.0
        else:
            assert conditional == -1.0
        expected_replicates = 4 if row["role"] == "rare_stress_test" else 3 if row["role"] == "geometry_stratified_train" else 2
        assert case["analysis"]["replicates"] == row["formal_count"] == expected_replicates
        sampling = case["metadata"]["sampling"]
        assert sampling["schema_version"] == "1.1.0"
        assert sampling["sampling_spec_document"] == spec.data
        assert sampling["training_design_document"] == spec.training_design_document
        assert sampling["training_design_hash"] == content_hash(sampling["training_design_document"])
        assert sampling["role"] == row["role"] and sampling["round"] == row["round"]


def test_extended_plan_is_deterministic_and_freezes_all_assignments(case_data) -> None:
    spec = SamplingSpec(_document())
    first = _factory(case_data, spec).build_reference_plan()
    second = _factory(case_data, spec).build_reference_plan()

    np.testing.assert_array_equal(first.latent_design, second.latent_design)
    assert first.cases == second.cases
    assert first.candidate_rows == second.candidate_rows
    assert first.split_registry.definition_document() == second.split_registry.definition_document()
    assert content_hash(
        {
            "cases": first.cases,
            "candidate_rows": first.candidate_rows,
            "latent_design": first.latent_design.tolist(),
            "split_registry": first.split_registry.definition_document(),
        }
    ) == "sha256:b890f13aae26c31e65541805828ed38bc3e83a7110fcf519d64e0ce1b5d26503"


def test_impossible_prelabel_stratum_exhausts_frozen_attempt_budget(case_data) -> None:
    document = _document()
    document["geometry_strata"][0]["predicate"]["all"][0]["value"] = 2.0
    document["legal_sampling"]["max_attempts_per_allocation"] = 7

    with pytest.raises(InputValidationError, match="unable to fill") as error:
        _factory(case_data, SamplingSpec(document)).build_reference_plan()

    assert error.value.context["accepted"] == 0
    assert error.value.context["attempts"] == 7


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["stream_roles"][-1].update(stream="reference", split="train", train_eligible=True), "fixed split"),
        (lambda value: value["round_budgets"][0]["allocations"][-1].update(proposal_id="toy-p0"), "non-p0 proposal"),
        (lambda value: value["feature_roles"][0].update(role="fixed_context"), "learnable_variable"),
        (lambda value: value["proxy_scope"].update(target_identity_policy="target_family_registry"), "identity policy"),
        (lambda value: value["budgets"].update(train_pilot=99), "do not reconcile"),
    ],
)
def test_semantically_unsafe_design_contracts_fail_closed(mutation, message) -> None:
    document = _document()
    mutation(document)
    with pytest.raises(InputValidationError, match=message):
        SamplingSpec(document)


def test_unsupported_joint_and_unknown_prelabeled_geometry_fail_closed() -> None:
    document = _document()
    document["variable_blocks"][0]["joint_mode"] = "gaussian_copula"
    with pytest.raises(InputValidationError, match="failed schema validation"):
        SamplingSpec(document)

    document = _document()
    clause = document["geometry_strata"][0]["predicate"]["all"][0]
    clause["source"] = "prelabel_screening"
    clause["path"] = "/unknown_geometry_feature"
    with pytest.raises(InputValidationError, match="unknown geometry feature"):
        SamplingSpec(document)


def test_tensor_feature_role_names_are_globally_unique() -> None:
    document = _document()
    document["feature_roles"][1]["tensor_feature_names"] = list(
        document["feature_roles"][0]["tensor_feature_names"]
    )
    with pytest.raises(InputValidationError, match="globally unique"):
        SamplingSpec(document)

    missing_mapping = _document()
    missing_mapping["feature_roles"][0].pop("tensor_feature_names")
    with pytest.raises(InputValidationError, match="failed schema validation"):
        SamplingSpec(missing_mapping)


def test_reference_plan_rejects_predeclared_future_rounds() -> None:
    document = _document()
    document["round_budgets"].append({
        "round": 1,
        "allocations": [{
            "role": "geometry_stratified_train",
            "stratum_id": "high-driver",
            "proposal_id": "toy-geometry-high",
            "count": 1,
        }],
    })
    document["budgets"]["train_pilot"] += 1
    with pytest.raises(InputValidationError, match="round zero only"):
        SamplingSpec(document)


def test_legacy_10_contract_remains_valid_but_cannot_smuggle_11_fields() -> None:
    document = _document()
    for field in (
        "feature_roles", "variable_blocks", "geometry_strata", "stream_roles",
        "round_budgets", "replicate_policy", "key_projections", "proxy_scope", "legal_sampling",
    ):
        document.pop(field)
    document["schema_version"] = "1.0.0"
    document["budgets"]["rare_stress_test"] = 0
    document["budgets"]["train_pilot"] = 1
    spec = SamplingSpec(document)
    assert not spec.is_extended
    assert spec.latent_dimension == 5
    assert spec.role_allocations()[0]["role"] == "reference_validation"

    document["feature_roles"] = _document()["feature_roles"]
    with pytest.raises(InputValidationError, match="require schema_version 1.1.0"):
        SamplingSpec(document)
