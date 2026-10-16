"""Fail-closed structure checks for independent scientific validation evidence.

This verifies that an approval record carries the evidence categories required
for governance.  It cannot judge whether an experiment or reviewer is
scientifically adequate; that remains an external responsibility.
"""

from __future__ import annotations

from datetime import datetime
import math
from typing import Any, Mapping

from .content_contract import is_sha256_identity
from .errors import InputValidationError


_UNCERTAINTY_CATEGORIES = {
    "parameter_epistemic", "model_form", "numerical_error"
}


def _nonempty_list(value: Any, name: str) -> list:
    if not isinstance(value, list) or not value:
        raise InputValidationError(f"scientific validation {name} must be nonempty")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputValidationError(f"scientific validation {name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise InputValidationError(f"scientific validation {name} must be finite")
    return number


def validate_scientific_validation_record(value: Mapping[str, Any] | None) -> None:
    """Reject incomplete or failed research-admission evidence categories."""

    if not isinstance(value, Mapping):
        raise InputValidationError(
            "research admission requires a structured scientific_validation record"
        )
    required = {
        "schema_version", "independent_data_sources", "dimensional_error_metrics",
        "interval_coverage", "mechanism_stratified_errors",
        "extrapolation_failure_rules", "uncertainty_ledger", "approval",
    }
    if set(value) != required or value.get("schema_version") != "1.0.0":
        raise InputValidationError(
            "scientific_validation fields or schema version are invalid",
            context={"missing": sorted(required - set(value)), "unknown": sorted(set(value) - required)},
        )

    sources = _nonempty_list(value["independent_data_sources"], "independent_data_sources")
    has_validation_source = False
    for source in sources:
        if not isinstance(source, Mapping) or set(source) != {
            "source_id", "role", "content_hash", "used_for_calibration"
        }:
            raise InputValidationError("scientific validation data-source record is malformed")
        if (
            not isinstance(source["source_id"], str) or not source["source_id"]
            or source["role"] not in {"calibration", "validation", "benchmark"}
            # Preserve the historical string coercion for version-1 evidence
            # records.  A future schema version may require a JSON string.
            or not is_sha256_identity(str(source["content_hash"]))
            or type(source["used_for_calibration"]) is not bool
        ):
            raise InputValidationError("scientific validation data-source identity is invalid")
        has_validation_source |= (
            source["role"] == "validation" and not source["used_for_calibration"]
        )
    if not has_validation_source:
        raise InputValidationError(
            "scientific validation requires data withheld from calibration"
        )

    for metric in _nonempty_list(value["dimensional_error_metrics"], "dimensional_error_metrics"):
        if not isinstance(metric, Mapping) or set(metric) != {
            "metric_id", "mechanism", "unit", "observed_error", "acceptance_limit", "passed"
        }:
            raise InputValidationError("dimensional error metric is malformed")
        observed = _finite(metric["observed_error"], "observed_error")
        limit = _finite(metric["acceptance_limit"], "acceptance_limit")
        if (
            not all(isinstance(metric[name], str) and metric[name] for name in ("metric_id", "mechanism", "unit"))
            or limit < 0.0 or observed < 0.0 or metric["passed"] is not True
            or observed > limit
        ):
            raise InputValidationError("dimensional error metric does not pass its declared limit")

    for coverage in _nonempty_list(value["interval_coverage"], "interval_coverage"):
        if not isinstance(coverage, Mapping) or set(coverage) != {
            "metric_id", "mechanism", "nominal", "observed", "sample_count", "passed"
        }:
            raise InputValidationError("interval coverage record is malformed")
        nominal = _finite(coverage["nominal"], "nominal coverage")
        observed = _finite(coverage["observed"], "observed coverage")
        if (
            not 0.0 < nominal < 1.0 or not 0.0 <= observed <= 1.0
            or not isinstance(coverage["sample_count"], int) or coverage["sample_count"] < 1
            or coverage["passed"] is not True
        ):
            raise InputValidationError("interval coverage record is not an accepted result")

    for metric in _nonempty_list(
        value["mechanism_stratified_errors"], "mechanism_stratified_errors"
    ):
        if not isinstance(metric, Mapping) or set(metric) != {
            "stratum_id", "metric_id", "value", "acceptance_limit", "passed"
        }:
            raise InputValidationError("mechanism-stratified error record is malformed")
        observed = _finite(metric["value"], "stratified error")
        limit = _finite(metric["acceptance_limit"], "stratified acceptance limit")
        if observed < 0.0 or limit < 0.0 or observed > limit or metric["passed"] is not True:
            raise InputValidationError("mechanism-stratified error does not pass")

    for rule in _nonempty_list(
        value["extrapolation_failure_rules"], "extrapolation_failure_rules"
    ):
        if not isinstance(rule, Mapping) or set(rule) != {"rule_id", "condition", "action"}:
            raise InputValidationError("extrapolation failure rule is malformed")
        if (
            not isinstance(rule["rule_id"], str) or not rule["rule_id"]
            or not isinstance(rule["condition"], str) or not rule["condition"]
            or rule["action"] != "refuse"
        ):
            raise InputValidationError("extrapolation failure rules must explicitly refuse")

    ledger = value["uncertainty_ledger"]
    if not isinstance(ledger, Mapping) or set(ledger) != _UNCERTAINTY_CATEGORIES:
        raise InputValidationError("uncertainty ledger must cover parameter, model-form and numerical error")
    if any(ledger[name] not in {"assessed", "bounded"} for name in _UNCERTAINTY_CATEGORIES):
        raise InputValidationError("all uncertainty categories must be assessed or bounded")

    approval = value["approval"]
    if not isinstance(approval, Mapping) or set(approval) != {
        "approver_id", "approved_at", "protocol_version"
    }:
        raise InputValidationError("scientific validation approval record is malformed")
    try:
        timestamp = datetime.fromisoformat(str(approval["approved_at"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise InputValidationError("scientific validation approval time is invalid") from exc
    if (
        timestamp.tzinfo is None
        or not isinstance(approval["approver_id"], str) or not approval["approver_id"]
        or not isinstance(approval["protocol_version"], str) or not approval["protocol_version"]
    ):
        raise InputValidationError("scientific validation approval must be identified and timezone-aware")
