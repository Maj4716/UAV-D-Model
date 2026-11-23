"""Versioned statistical-measure contracts for sampling and evaluation.

The contract keeps the population estimand, target measure, candidate proposal,
selection probability and three downstream weight meanings as separate
identities.  This prevents a coverage design from inheriting operational or
reference interpretation through a shared generic ``weight`` field.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .errors import InputValidationError
from .utils import content_hash


SAMPLING_MEASURE_CONTRACT_VERSION = "sampling-measure-contract@1.0.0"
SAMPLING_MEASURE_BUNDLE_VERSION = "sampling-measure-bundle@1.0.0"

_ROLE_IDENTITIES = {
    "operational": (
        "operational_frequency_measure",
        "operational_prevalence_density",
    ),
    "reference": (
        "declared_reference_measure",
        "declared_reference_density",
    ),
    "coverage": (
        "training_coverage_measure",
        "coverage_design_density",
    ),
}
_WEIGHT_CONTRACTS = {
    "design": (
        "inverse_inclusion_probability",
        "finite_frame_design_inference",
    ),
    "evaluation": (
        "target_density_over_proposal_density",
        "target_measure_evaluation",
    ),
    "training": (
        "declared_training_objective_weight",
        "optimization_objective_only",
    ),
}


def _detached_json(document: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(document, Mapping):
        raise InputValidationError(f"{name} must be an object")
    try:
        value = json.loads(
            json.dumps(
                document,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"{name} must be finite JSON data") from exc
    if not isinstance(value, dict):  # pragma: no cover - guarded by Mapping
        raise InputValidationError(f"{name} must be an object")
    return value


def _nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"{field} must be a nonempty string")
    return value


def _exact_keys(
    document: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str],
    field: str,
) -> None:
    missing = sorted(required - set(document))
    unexpected = sorted(set(document) - required - optional)
    if missing or unexpected:
        raise InputValidationError(
            f"{field} fields are invalid",
            context={"missing": missing, "unexpected": unexpected},
        )


def _verified_hash(
    declared: Any,
    expected: str,
    *,
    field: str,
) -> str:
    if declared is not None and declared != expected:
        raise InputValidationError(
            f"{field} is inconsistent with its canonical definition",
            context={"expected": expected, "observed": declared},
        )
    return expected


def _validate_estimand(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InputValidationError("estimand must be an object")
    result = _detached_json(value, name="estimand")
    _exact_keys(
        result,
        required={"estimand_id", "definition"},
        optional={"estimand_hash"},
        field="estimand",
    )
    identifier = _nonempty_string(result["estimand_id"], field="estimand_id")
    if not isinstance(result["definition"], dict) or not result["definition"]:
        raise InputValidationError("estimand.definition must be a nonempty object")
    expected = content_hash(
        {"estimand_id": identifier, "definition": result["definition"]}
    )
    result["estimand_hash"] = _verified_hash(
        result.get("estimand_hash"),
        expected,
        field="estimand_hash",
    )
    return result


def _validate_measure(
    value: Any,
    *,
    measure_kind: str,
    condition_on_s_contact: bool,
    target_density_semantics: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InputValidationError("measure must be an object")
    result = _detached_json(value, name="measure")
    _exact_keys(
        result,
        required={"measure_id", "definition"},
        optional={"measure_hash"},
        field="measure",
    )
    identifier = _nonempty_string(result["measure_id"], field="measure_id")
    if not isinstance(result["definition"], dict) or not result["definition"]:
        raise InputValidationError("measure.definition must be a nonempty object")
    expected = content_hash(
        {
            "measure_id": identifier,
            "measure_kind": measure_kind,
            "definition": result["definition"],
            "condition_on_s_contact": condition_on_s_contact,
            "target_density_semantics": target_density_semantics,
        }
    )
    result["measure_hash"] = _verified_hash(
        result.get("measure_hash"),
        expected,
        field="measure_hash",
    )
    return result


def _validate_proposal(value: Any, *, measure_hash: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InputValidationError("proposal must be an object")
    result = _detached_json(value, name="proposal")
    _exact_keys(
        result,
        required={"proposal_id", "definition"},
        optional={"proposal_hash"},
        field="proposal",
    )
    identifier = _nonempty_string(result["proposal_id"], field="proposal_id")
    if not isinstance(result["definition"], dict) or not result["definition"]:
        raise InputValidationError("proposal.definition must be a nonempty object")
    expected = content_hash(
        {
            "proposal_id": identifier,
            "definition": result["definition"],
            "target_measure_hash": measure_hash,
        }
    )
    result["proposal_hash"] = _verified_hash(
        result.get("proposal_hash"),
        expected,
        field="proposal_hash",
    )
    return result


def _validate_weight_semantics(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InputValidationError("weight_semantics must be an object")
    result = _detached_json(value, name="weight_semantics")
    _exact_keys(
        result,
        required=set(_WEIGHT_CONTRACTS),
        optional=set(),
        field="weight_semantics",
    )
    fields: list[str] = []
    for role, (formula, purpose) in _WEIGHT_CONTRACTS.items():
        record = result[role]
        if not isinstance(record, dict):
            raise InputValidationError(f"weight_semantics.{role} must be an object")
        _exact_keys(
            record,
            required={"field", "formula", "purpose"},
            optional=set(),
            field=f"weight_semantics.{role}",
        )
        fields.append(
            _nonempty_string(record["field"], field=f"weight_semantics.{role}.field")
        )
        if record["formula"] != formula or record["purpose"] != purpose:
            raise InputValidationError(
                f"weight_semantics.{role} formula or purpose is unsupported",
                context={
                    "expected_formula": formula,
                    "expected_purpose": purpose,
                    "observed": record,
                },
            )
    if len(set(fields)) != len(fields):
        raise InputValidationError(
            "design, evaluation and training weights require distinct fields"
        )
    return result


def validate_sampling_measure_contract(
    document: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and hash one available Operational, Reference or Coverage measure."""

    result = _detached_json(document, name="sampling measure contract")
    role = result.get("role")
    if role not in _ROLE_IDENTITIES:
        raise InputValidationError(
            "sampling measure role is unsupported",
            context={"role": role, "supported": sorted(_ROLE_IDENTITIES)},
        )
    is_selected_design = role in {"reference", "coverage"}
    common_required = {
        "schema_version",
        "status",
        "role",
        "measure_kind",
        "estimand",
        "measure",
        "condition_on_s_contact",
        "target_density_semantics",
    }
    selected_required = {
        "proposal",
        "inclusion_probability_field",
        "weight_semantics",
    }
    _exact_keys(
        result,
        required=common_required | (selected_required if is_selected_design else set()),
        optional={"contract_hash"},
        field="sampling measure contract",
    )
    if result["schema_version"] != SAMPLING_MEASURE_CONTRACT_VERSION:
        raise InputValidationError("sampling measure schema_version is unsupported")
    if result["status"] != "available":
        raise InputValidationError("available sampling measure status must be 'available'")
    expected_kind, expected_density = _ROLE_IDENTITIES[role]
    if result["measure_kind"] != expected_kind:
        raise InputValidationError(
            "measure_kind is inconsistent with the declared sampling measure role",
            context={"role": role, "expected": expected_kind},
        )
    if result["target_density_semantics"] != expected_density:
        raise InputValidationError(
            "target density semantics are inconsistent with the declared role",
            context={"role": role, "expected": expected_density},
        )
    if not isinstance(result["condition_on_s_contact"], bool):
        raise InputValidationError("condition_on_s_contact must be boolean")

    result["estimand"] = _validate_estimand(result["estimand"])
    result["measure"] = _validate_measure(
        result["measure"],
        measure_kind=expected_kind,
        condition_on_s_contact=result["condition_on_s_contact"],
        target_density_semantics=expected_density,
    )
    if is_selected_design:
        result["proposal"] = _validate_proposal(
            result["proposal"],
            measure_hash=result["measure"]["measure_hash"],
        )
        _nonempty_string(
            result["inclusion_probability_field"],
            field="inclusion_probability_field",
        )
        result["weight_semantics"] = _validate_weight_semantics(
            result["weight_semantics"]
        )

    declared_contract_hash = result.pop("contract_hash", None)
    expected_contract_hash = content_hash(result)
    result["contract_hash"] = _verified_hash(
        declared_contract_hash,
        expected_contract_hash,
        field="contract_hash",
    )
    return result


def _validate_operational_entry(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InputValidationError("operational entry must be an object")
    marker = _detached_json(value, name="operational entry")
    if marker.get("status") == "unavailable":
        _exact_keys(
            marker,
            required={"status", "reason"},
            optional=set(),
            field="unavailable operational entry",
        )
        _nonempty_string(marker["reason"], field="operational.reason")
        return marker
    validated = validate_sampling_measure_contract(marker)
    if validated["role"] != "operational":
        raise InputValidationError(
            "operational bundle entry must declare the operational role"
        )
    return validated


def validate_sampling_measure_bundle(
    document: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an explicit Operational/Reference/Coverage measure bundle."""

    result = _detached_json(document, name="sampling measure bundle")
    if "operational" not in result:
        raise InputValidationError(
            "sampling measure bundle requires an explicit operational contract or unavailable marker"
        )
    _exact_keys(
        result,
        required={"schema_version", "operational", "reference", "coverage"},
        optional={"bundle_hash"},
        field="sampling measure bundle",
    )
    if result["schema_version"] != SAMPLING_MEASURE_BUNDLE_VERSION:
        raise InputValidationError("sampling measure bundle schema_version is unsupported")
    result["operational"] = _validate_operational_entry(result["operational"])
    result["reference"] = validate_sampling_measure_contract(result["reference"])
    result["coverage"] = validate_sampling_measure_contract(result["coverage"])
    if result["reference"]["role"] != "reference":
        raise InputValidationError("reference bundle entry must declare the reference role")
    if result["coverage"]["role"] != "coverage":
        raise InputValidationError("coverage bundle entry must declare the coverage role")
    if (
        result["reference"]["estimand"]["estimand_hash"]
        != result["coverage"]["estimand"]["estimand_hash"]
    ):
        raise InputValidationError(
            "reference and coverage contracts must share one estimand"
        )
    if (
        result["reference"]["condition_on_s_contact"]
        != result["coverage"]["condition_on_s_contact"]
    ):
        raise InputValidationError(
            "reference and coverage condition_on_s_contact must match"
        )
    declared_bundle_hash = result.pop("bundle_hash", None)
    expected_bundle_hash = content_hash(result)
    result["bundle_hash"] = _verified_hash(
        declared_bundle_hash,
        expected_bundle_hash,
        field="bundle_hash",
    )
    return result


def importance_weight(target_density: Any, proposal_density: Any) -> Any:
    """Compute ``p / q`` and fail on positive target mass outside proposal support."""

    try:
        target, proposal = np.broadcast_arrays(
            np.asarray(target_density, dtype=np.float64),
            np.asarray(proposal_density, dtype=np.float64),
        )
    except (TypeError, ValueError) as exc:
        raise InputValidationError(
            "target and proposal densities must be broadcastable numeric values"
        ) from exc
    if (
        not np.isfinite(target).all()
        or not np.isfinite(proposal).all()
        or np.any(target < 0.0)
        or np.any(proposal < 0.0)
    ):
        raise InputValidationError(
            "target and proposal densities must be finite and nonnegative"
        )
    unsupported = (target > 0.0) & (proposal == 0.0)
    if np.any(unsupported):
        raise InputValidationError(
            "positivity violation: target density is positive where proposal density is zero",
            context={"unsupported_count": int(np.count_nonzero(unsupported))},
        )
    result = np.zeros_like(target, dtype=np.float64)
    supported = proposal > 0.0
    result[supported] = target[supported] / proposal[supported]
    return float(result) if result.ndim == 0 else result


def mixture_inclusion_probability(
    probabilities: Sequence[float],
    *,
    independent_selections: bool,
) -> float:
    """Return the union inclusion probability for independent source arms.

    Correlated source-arm selections require their joint inclusion probability;
    marginal probabilities alone do not identify the union probability.
    """

    if independent_selections is not True:
        raise InputValidationError(
            "correlated selections require a declared joint inclusion probability"
        )
    try:
        values = np.asarray(tuple(probabilities), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(
            "mixture inclusion probabilities must be numeric"
        ) from exc
    if values.ndim != 1 or values.size == 0:
        raise InputValidationError(
            "mixture inclusion probabilities must be a nonempty sequence"
        )
    if (
        not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise InputValidationError(
            "mixture inclusion probabilities must lie in [0, 1]"
        )
    if np.any(values == 1.0):
        return 1.0
    log_exclusion = math.fsum(math.log1p(-float(value)) for value in values)
    probability = -math.expm1(log_exclusion)
    return 0.0 if probability == 0.0 else float(probability)
