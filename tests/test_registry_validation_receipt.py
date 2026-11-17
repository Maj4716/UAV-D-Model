from __future__ import annotations

from copy import deepcopy

import pytest

from damage_engine.engine import DamageEngine
from damage_engine.errors import InputValidationError, OutOfDomainError
from damage_engine.registry import PhysicsRegistry, physics_model_payload_hash
from damage_engine.utils import content_hash
from damage_engine.validation import clear_validation_caches, validate_inputs


def _scientific_validation_record():
    return {
        "schema_version": "1.0.0",
        "independent_data_sources": [
            {
                "source_id": "withheld",
                "role": "validation",
                "content_hash": "sha256:" + "1" * 64,
                "used_for_calibration": False,
            }
        ],
        "dimensional_error_metrics": [
            {
                "metric_id": "velocity",
                "mechanism": "fragment",
                "unit": "m/s",
                "observed_error": 1.0,
                "acceptance_limit": 2.0,
                "passed": True,
            }
        ],
        "interval_coverage": [
            {
                "metric_id": "coverage",
                "mechanism": "fragment",
                "nominal": 0.95,
                "observed": 0.95,
                "sample_count": 20,
                "passed": True,
            }
        ],
        "mechanism_stratified_errors": [
            {
                "stratum_id": "near",
                "metric_id": "velocity",
                "value": 1.0,
                "acceptance_limit": 2.0,
                "passed": True,
            }
        ],
        "extrapolation_failure_rules": [
            {
                "rule_id": "outside",
                "condition": "outside frozen domain",
                "action": "refuse",
            }
        ],
        "uncertainty_ledger": {
            "parameter_epistemic": "assessed",
            "model_form": "assessed",
            "numerical_error": "assessed",
        },
        "approval": {
            "approver_id": "reviewer",
            "approved_at": "2026-11-16T00:00:00Z",
            "protocol_version": "1",
        },
    }


def _validated_registry_data(registry):
    data = deepcopy(registry.data)
    data["validation_status"] = "validated"
    data["validation_receipt"] = {
        "receipt_id": "validation-receipt-1",
        "validated_at": "2026-11-16T00:00:00Z",
        "evidence_hash": "sha256:" + "2" * 64,
        "validated_model_payload_hash": physics_model_payload_hash(data),
        "domain": {
            "mechanisms": ["fragment"],
            "operating_conditions": {
                "case_constraints": [
                    {
                        "path": "/environment/temperature_k",
                        "minimum": 250.0,
                        "maximum": 330.0,
                    },
                    {
                        "path": "/warhead/fragment_source/direction_model_id",
                        "allowed_values": ["cone"],
                    },
                ]
            },
            "extrapolation_policy": "refuse",
        },
        "holdout_metrics": [
            {
                "metric_id": "velocity-rmse",
                "mechanism": "fragment",
                "unit": "m/s",
                "value": 1.0,
                "acceptance_range": [0.0, 2.0],
                "sample_count": 20,
                "passed": True,
            }
        ],
        "scientific_validation": _scientific_validation_record(),
    }
    return data


def _validated_target_data(target_data):
    target = deepcopy(target_data)

    def visit(value):
        if isinstance(value, dict):
            if {"value", "unit", "source_id", "validation_status"}.issubset(value):
                value["validation_status"] = "validated"
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(target)
    for item in target["armor"] + target["components"]:
        item["provenance"]["status"] = "approved"
    return target


def test_validated_registry_requires_content_bound_validation_receipt(registry):
    missing = deepcopy(registry.data)
    missing["validation_status"] = "validated"
    with pytest.raises(InputValidationError, match="validation receipt"):
        PhysicsRegistry(missing)

    valid = PhysicsRegistry(_validated_registry_data(registry))
    valid.require_validated()


def test_validation_receipt_cannot_be_reused_after_model_parameter_change(registry):
    data = _validated_registry_data(registry)
    data["fragment_response_models"]["demo_armor_response"]["ballistic_limit"][
        "coefficient"
    ] += 1.0

    with pytest.raises(InputValidationError, match="payload hash"):
        PhysicsRegistry(data)


def test_validation_receipt_requires_timezone_and_passing_holdout(registry):
    naive = _validated_registry_data(registry)
    naive["validation_receipt"]["validated_at"] = "2026-11-16T00:00:00"
    with pytest.raises(InputValidationError, match="timezone"):
        PhysicsRegistry(naive)

    failed = _validated_registry_data(registry)
    failed["validation_receipt"]["holdout_metrics"][0]["passed"] = False
    with pytest.raises(InputValidationError, match="holdout"):
        PhysicsRegistry(failed)


def test_validated_registry_executes_case_domain_on_every_case(registry, case_data):
    validated = PhysicsRegistry(_validated_registry_data(registry))
    accepted = deepcopy(case_data)
    accepted["analysis"]["mechanisms"] = ["fragment"]
    accepted["analysis"]["fidelity_level"] = "L3"
    validated.require_case_in_validated_domain(accepted)

    too_hot = deepcopy(accepted)
    too_hot["environment"]["temperature_k"] = 350.0
    with pytest.raises(OutOfDomainError) as exc_info:
        validated.require_case_in_validated_domain(too_hot)
    assert exc_info.value.context["reason_code"] == (
        "validation_receipt_condition_out_of_domain"
    )
    assert exc_info.value.context["path"] == "/environment/temperature_k"

    unsupported = deepcopy(accepted)
    unsupported["analysis"]["mechanisms"] = ["fragment", "blast"]
    with pytest.raises(OutOfDomainError) as exc_info:
        validated.require_case_in_validated_domain(unsupported)
    assert exc_info.value.context["reason_code"] == (
        "validation_receipt_mechanism_out_of_domain"
    )


def test_validation_receipt_requires_holdout_coverage_for_every_mechanism(registry):
    data = _validated_registry_data(registry)
    data["validation_receipt"]["domain"]["mechanisms"].append("blast")
    with pytest.raises(InputValidationError, match="holdout coverage"):
        PhysicsRegistry(data)


def test_l3_domain_check_is_not_bypassed_by_static_validation_cache(
    registry, target_data, case_data
):
    clear_validation_caches()
    validated_registry = PhysicsRegistry(_validated_registry_data(registry))
    target = _validated_target_data(target_data)
    accepted = deepcopy(case_data)
    accepted["analysis"]["mechanisms"] = ["fragment"]
    accepted["analysis"]["fidelity_level"] = "L3"
    accepted["target_ref"]["content_hash"] = content_hash(target)
    validate_inputs(target, accepted, validated_registry)

    too_hot = deepcopy(accepted)
    too_hot["environment"]["temperature_k"] = 350.0
    with pytest.raises(OutOfDomainError) as exc_info:
        validate_inputs(target, too_hot, validated_registry)
    assert exc_info.value.context["reason_code"] == (
        "validation_receipt_condition_out_of_domain"
    )


def test_l3_result_provenance_contains_content_bound_validation_receipt(
    registry, target_data, case_data
):
    validated_registry = PhysicsRegistry(_validated_registry_data(registry))
    target = _validated_target_data(target_data)
    case = deepcopy(case_data)
    case["analysis"]["mechanisms"] = ["fragment"]
    case["analysis"]["fidelity_level"] = "L3"
    case["analysis"]["replicates"] = 1
    case["analysis"]["adaptive_ci_half_width"] = None
    case["target_ref"]["content_hash"] = content_hash(target)

    result = DamageEngine(target, validated_registry).run(case).to_dict()

    receipt = result["provenance"]["physics_validation_receipt"]
    assert receipt["receipt_id"] == "validation-receipt-1"
    assert receipt["validated_model_payload_hash"] == physics_model_payload_hash(
        validated_registry.data
    )
    assert receipt["domain"] == validated_registry.data["validation_receipt"][
        "domain"
    ]
