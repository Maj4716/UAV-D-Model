from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .content_contract import is_sha256_identity, validate_json_schema
from .errors import (
    InputValidationError,
    MissingModelError,
    OutOfDomainError,
    UnverifiedModelError,
)
from .model_snapshot import FrozenSnapshotDict, freeze_snapshot
from .scientific_validation import validate_scientific_validation_record
from .utils import content_hash, load_json


_MODEL_PAYLOAD_SECTIONS = (
    "fragment_source_models",
    "fragment_response_models",
    "layer_stack_models",
    "blast_field_models",
    "blast_response_models",
    "transmission_models",
)


def physics_model_payload_hash(data: Mapping[str, Any]) -> str:
    """Hash only the model payload covered by a validation receipt.

    Status, provenance and the receipt itself are excluded to avoid a
    self-referential hash.  Config identity and every executable model section
    are included, so a receipt cannot be copied onto changed parameters.
    """

    return content_hash(
        {
            "schema_version": data.get("schema_version"),
            "config_id": data.get("config_id"),
            **{section: data.get(section, {}) for section in _MODEL_PAYLOAD_SECTIONS},
        }
    )


def _validate_validation_receipt(data: Mapping[str, Any]) -> None:
    receipt = data.get("validation_receipt")
    if not isinstance(receipt, Mapping):
        raise InputValidationError(
            "validated physics registry requires a structured validation receipt"
        )
    required = {
        "receipt_id",
        "validated_at",
        "evidence_hash",
        "validated_model_payload_hash",
        "domain",
        "holdout_metrics",
        "scientific_validation",
    }
    if set(receipt) != required:
        raise InputValidationError(
            "physics validation receipt fields are invalid",
            context={
                "missing": sorted(required - set(receipt)),
                "unknown": sorted(set(receipt) - required),
            },
        )
    if not isinstance(receipt["receipt_id"], str) or not receipt["receipt_id"]:
        raise InputValidationError("physics validation receipt_id must be nonempty")
    try:
        timestamp = datetime.fromisoformat(
            str(receipt["validated_at"]).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise InputValidationError(
            "physics validation receipt timestamp is invalid"
        ) from exc
    if timestamp.tzinfo is None:
        raise InputValidationError(
            "physics validation receipt timestamp must be timezone-aware"
        )
    for key in ("evidence_hash", "validated_model_payload_hash"):
        if not is_sha256_identity(str(receipt[key])):
            raise InputValidationError(
                "physics validation receipt hashes must be canonical SHA-256 identities",
                context={"field": key},
            )
    expected_payload_hash = physics_model_payload_hash(data)
    if receipt["validated_model_payload_hash"] != expected_payload_hash:
        raise InputValidationError(
            "physics validation receipt payload hash does not match executable models",
            context={
                "expected": expected_payload_hash,
                "observed": receipt["validated_model_payload_hash"],
            },
        )
    domain = receipt["domain"]
    if not isinstance(domain, Mapping):
        raise InputValidationError("physics validation receipt domain must be structured")
    mechanisms = domain.get("mechanisms")
    operating_conditions = domain.get("operating_conditions")
    if (
        not isinstance(mechanisms, list)
        or not mechanisms
        or any(not isinstance(item, str) or not item for item in mechanisms)
        or not isinstance(operating_conditions, Mapping)
        or not operating_conditions
        or domain.get("extrapolation_policy") != "refuse"
    ):
        raise InputValidationError(
            "physics validation receipt domain must declare mechanisms, operating conditions and refusal outside domain"
        )
    case_constraints = operating_conditions.get("case_constraints")
    if (
        set(operating_conditions) != {"case_constraints"}
        or not isinstance(case_constraints, list)
        or not case_constraints
    ):
        raise InputValidationError(
            "physics validation operating conditions require nonempty executable case constraints"
        )
    constraint_fields = {"path", "minimum", "maximum", "allowed_values"}
    for constraint in case_constraints:
        if (
            not isinstance(constraint, Mapping)
            or not set(constraint).issubset(constraint_fields)
            or not isinstance(constraint.get("path"), str)
            or not constraint["path"].startswith("/")
            or constraint["path"] == "/"
        ):
            raise InputValidationError(
                "physics validation case constraint is malformed"
            )
        has_numeric_bound = "minimum" in constraint or "maximum" in constraint
        has_allowed_values = "allowed_values" in constraint
        if not has_numeric_bound and not has_allowed_values:
            raise InputValidationError(
                "physics validation case constraint must declare a numeric bound or allowed values",
                context={"path": constraint["path"]},
            )
        for bound_name in ("minimum", "maximum"):
            if bound_name in constraint:
                bound = constraint[bound_name]
                if (
                    isinstance(bound, bool)
                    or not isinstance(bound, (int, float))
                    or not math.isfinite(float(bound))
                ):
                    raise InputValidationError(
                        "physics validation case constraint bounds must be finite numbers",
                        context={"path": constraint["path"], "field": bound_name},
                    )
        if (
            "minimum" in constraint
            and "maximum" in constraint
            and float(constraint["minimum"]) > float(constraint["maximum"])
        ):
            raise InputValidationError(
                "physics validation case constraint bounds are reversed",
                context={"path": constraint["path"]},
            )
        if has_allowed_values:
            allowed_values = constraint["allowed_values"]
            if not isinstance(allowed_values, list) or not allowed_values:
                raise InputValidationError(
                    "physics validation case constraint allowed_values must be nonempty",
                    context={"path": constraint["path"]},
                )
    metrics = receipt["holdout_metrics"]
    if not isinstance(metrics, list) or not metrics:
        raise InputValidationError(
            "physics validation receipt requires nonempty holdout metrics"
        )
    metric_fields = {
        "metric_id",
        "mechanism",
        "unit",
        "value",
        "acceptance_range",
        "sample_count",
        "passed",
    }
    for metric in metrics:
        if not isinstance(metric, Mapping) or set(metric) != metric_fields:
            raise InputValidationError(
                "physics validation holdout metric is malformed"
            )
        try:
            value = float(metric["value"])
            acceptance = [float(item) for item in metric["acceptance_range"]]
        except (TypeError, ValueError, OverflowError) as exc:
            raise InputValidationError(
                "physics validation holdout metric values must be numeric"
            ) from exc
        if (
            len(acceptance) != 2
            or not all(math.isfinite(item) for item in (value, *acceptance))
            or acceptance[0] > acceptance[1]
            or not acceptance[0] <= value <= acceptance[1]
            or not isinstance(metric["sample_count"], int)
            or isinstance(metric["sample_count"], bool)
            or metric["sample_count"] < 1
            or metric["passed"] is not True
            or any(
                not isinstance(metric[name], str) or not metric[name]
                for name in ("metric_id", "mechanism", "unit")
            )
        ):
            raise InputValidationError(
                "physics validation holdout metric did not pass its declared acceptance range"
            )
        if metric["mechanism"] not in mechanisms:
            raise InputValidationError(
                "physics validation holdout metric mechanism is outside the declared domain",
                context={
                    "metric_id": metric["metric_id"],
                    "mechanism": metric["mechanism"],
                    "domain_mechanisms": mechanisms,
                },
            )
    covered_mechanisms = {str(metric["mechanism"]) for metric in metrics}
    uncovered_mechanisms = sorted(set(mechanisms) - covered_mechanisms)
    if uncovered_mechanisms:
        raise InputValidationError(
            "physics validation receipt lacks holdout coverage for declared mechanisms",
            context={"uncovered_mechanisms": uncovered_mechanisms},
        )
    validate_scientific_validation_record(receipt["scientific_validation"])


def _resolve_json_pointer(document: object, pointer: str) -> object:
    """Resolve an RFC 6901 JSON pointer without accepting implicit coercions."""

    current = document
    for raw_token in pointer.split("/")[1:]:
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            if token not in current:
                raise KeyError(token)
            current = current[token]
        elif isinstance(current, list):
            if not token.isdecimal():
                raise KeyError(token)
            index = int(token)
            if index >= len(current):
                raise KeyError(token)
            current = current[index]
        else:
            raise KeyError(token)
    return current


def _require_finite_numbers(value: object, *, path: str = "$.") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite_numbers(child, path=f"{path}{key}.")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _require_finite_numbers(child, path=f"{path}{index}.")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise InputValidationError(
                "physics registry contains a non-finite numeric value",
                context={"path": path[:-1]},
            )


def _require_positive(value: object, *, name: str, model_id: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise InputValidationError(
            f"{name} must be finite and positive",
            context={"model_id": model_id, "value": number},
        )
    return number


@dataclass(frozen=True, init=False)
class PhysicsRegistry:
    _data: dict
    _frozen_data: FrozenSnapshotDict
    _hash: str

    def __init__(self, data: Mapping[str, Any]):
        # Own a detached snapshot before validation.  Neither later mutation of
        # the caller's document nor mutation of an exported copy can split the
        # registry hash from the models used by the solver.
        object.__setattr__(self, "_data", deepcopy(dict(data)))
        self.__post_init__()

    @property
    def data(self) -> dict:
        """Return a mutable export detached from the executable snapshot."""

        return deepcopy(self._data)

    @classmethod
    def from_file(cls, path: str | Path) -> "PhysicsRegistry":
        return cls(load_json(path))

    def __post_init__(self):
        validate_json_schema(
            self._data,
            "physics_registry.schema.json",
            failure_message="physics registry schema validation failed",
            max_errors=20,
            compatibility="legacy_records",
        )
        _require_finite_numbers(self._data)
        required = {
            "schema_version",
            "config_id",
            "validation_status",
            "fragment_response_models",
            "fragment_source_models",
            "blast_field_models",
            "blast_response_models",
            "transmission_models",
        }
        missing = sorted(required - set(self._data))
        if missing:
            raise InputValidationError(
                "physics registry is missing required sections",
                context={"missing": missing},
            )
        if self._data["validation_status"] == "validated":
            _validate_validation_receipt(self._data)
        elif "validation_receipt" in self._data:
            raise InputValidationError(
                "validation receipt is reserved for a validated physics registry",
                context={"validation_status": self._data["validation_status"]},
            )
        for model_id, model in self._data["fragment_source_models"].items():
            initial = model["initial_velocity"]
            if initial["kind"] == "fixed" and "velocity_mps" not in initial:
                raise InputValidationError(
                    "fixed fragment source requires velocity_mps",
                    context={"model_id": model_id},
                )
            if initial["kind"] == "fixed":
                _require_positive(
                    initial["velocity_mps"],
                    name="fixed fragment velocity",
                    model_id=model_id,
                )
            direction = model["direction"]
            if direction["kind"] == "cone" and "half_angle_deg" not in direction:
                raise InputValidationError(
                    "cone fragment source requires half_angle_deg",
                    context={"model_id": model_id},
                )
            axis = np.asarray(direction.get("axis_body", [1.0, 0.0, 0.0]), dtype=float)
            if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) <= 1.0e-12:
                raise InputValidationError(
                    "fragment source axis_body must be a finite non-zero 3-vector",
                    context={"model_id": model_id},
                )
            valid_speed = model.get("valid_speed_mps")
            if valid_speed is not None:
                lower, upper = map(float, valid_speed)
                if lower <= 0 or lower >= upper:
                    raise InputValidationError(
                        "fragment valid_speed_mps must be positive and strictly ordered",
                        context={"model_id": model_id, "valid_speed_mps": valid_speed},
                    )
        for model_id, model in self._data["blast_field_models"].items():
            table = model["scaled_distance_table"]
            scaled_distance = np.asarray([row["z_m_kg13"] for row in table], dtype=float)
            if np.any(np.diff(scaled_distance) <= 0):
                raise InputValidationError(
                    "blast scaled-distance tables must be strictly increasing",
                    context={"model_id": model_id},
                )
            peak = np.asarray([row["peak_pressure_pa"] for row in table], dtype=float)
            impulse = np.asarray(
                [row["scaled_impulse_pa_s_kgm13"] for row in table], dtype=float
            )
            duration = np.asarray(
                [row["scaled_duration_s_kgm13"] for row in table], dtype=float
            )
            if np.any(np.diff(peak) > 0) or np.any(np.diff(impulse) > 0):
                raise InputValidationError(
                    "blast peak pressure and impulse must not increase with scaled distance",
                    context={"model_id": model_id},
                )
            if np.any(np.diff(duration) < 0):
                raise InputValidationError(
                    "blast positive duration must not decrease with scaled distance",
                    context={"model_id": model_id},
                )
            reflection = model.get("reflection") or {}
            minimum = float(reflection.get("minimum_factor", 1.0))
            maximum = float(reflection.get("maximum_factor", 2.0))
            pressure_scale = float(reflection.get("pressure_scale_pa", 1.0))
            angular_exponent = float(reflection.get("angular_exponent", 1.0))
            if minimum <= 0 or maximum < minimum or pressure_scale <= 0 or angular_exponent < 0:
                raise InputValidationError(
                    "blast reflection parameters are physically inconsistent",
                    context={"model_id": model_id},
                )
            for exposure, amplification in (model.get("compartment_amplification") or {}).items():
                _require_positive(
                    amplification,
                    name=f"compartment amplification for {exposure}",
                    model_id=model_id,
                )
        for model_id, model in self._data["transmission_models"].items():
            _require_positive(
                model["peak_half_thickness_m"],
                name="peak half-thickness",
                model_id=model_id,
            )
            _require_positive(
                model["impulse_half_thickness_m"],
                name="impulse half-thickness",
                model_id=model_id,
            )
        for stack_id, stack in self._data.get("layer_stack_models", {}).items():
            fractions = [float(layer["thickness_fraction"]) for layer in stack["layers"]]
            if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1.0e-9):
                raise InputValidationError(
                    "layer stack thickness fractions must sum to one",
                    context={"layer_stack_id": stack_id, "fractions": fractions},
                )
            missing_models = sorted(
                {
                    str(layer["fragment_response_model_id"])
                    for layer in stack["layers"]
                    if str(layer["fragment_response_model_id"])
                    not in self._data["fragment_response_models"]
                }
            )
            if missing_models:
                raise InputValidationError(
                    "layer stack references unknown fragment response models",
                    context={
                        "layer_stack_id": stack_id,
                        "missing_models": missing_models,
                    },
                )
        object.__setattr__(self, "_frozen_data", freeze_snapshot(self._data))
        object.__setattr__(self, "_hash", content_hash(self._data))

    @property
    def hash(self) -> str:
        return self._hash

    @property
    def validation_status(self) -> str:
        return str(self._data["validation_status"])

    def require_validated(self) -> None:
        if self.validation_status != "validated":
            raise UnverifiedModelError(
                "L3 execution requires a validated physics registry",
                context={"validation_status": self.validation_status},
            )
        _validate_validation_receipt(self._data)

    def require_case_in_validated_domain(self, case: Mapping[str, Any]) -> None:
        """Refuse an L3 case outside the receipt's executable validation domain."""

        self.require_validated()
        domain = self._data["validation_receipt"]["domain"]
        declared_mechanisms = set(domain["mechanisms"])
        requested_mechanisms = set(case["analysis"]["mechanisms"])
        unsupported_mechanisms = sorted(requested_mechanisms - declared_mechanisms)
        if unsupported_mechanisms:
            raise OutOfDomainError(
                "L3 case requests mechanisms outside the validated domain",
                context={
                    "reason_code": "validation_receipt_mechanism_out_of_domain",
                    "requested_mechanisms": sorted(requested_mechanisms),
                    "validated_mechanisms": sorted(declared_mechanisms),
                    "unsupported_mechanisms": unsupported_mechanisms,
                    "receipt_id": self._data["validation_receipt"]["receipt_id"],
                },
            )

        for constraint in domain["operating_conditions"]["case_constraints"]:
            path = str(constraint["path"])
            try:
                observed = _resolve_json_pointer(case, path)
            except (KeyError, IndexError, TypeError) as exc:
                raise OutOfDomainError(
                    "L3 case lacks an operating condition required by the validation receipt",
                    context={
                        "reason_code": "validation_receipt_condition_missing",
                        "path": path,
                        "receipt_id": self._data["validation_receipt"]["receipt_id"],
                    },
                ) from exc

            violation: str | None = None
            if "allowed_values" in constraint and observed not in constraint["allowed_values"]:
                violation = "not_in_allowed_values"
            if "minimum" in constraint or "maximum" in constraint:
                if (
                    isinstance(observed, bool)
                    or not isinstance(observed, (int, float))
                    or not math.isfinite(float(observed))
                ):
                    violation = "not_finite_numeric"
                elif "minimum" in constraint and float(observed) < float(constraint["minimum"]):
                    violation = "below_minimum"
                elif "maximum" in constraint and float(observed) > float(constraint["maximum"]):
                    violation = "above_maximum"
            if violation is not None:
                raise OutOfDomainError(
                    "L3 case operating condition is outside the validated domain",
                    context={
                        "reason_code": "validation_receipt_condition_out_of_domain",
                        "path": path,
                        "observed": observed,
                        "constraint": dict(constraint),
                        "violation": violation,
                        "receipt_id": self._data["validation_receipt"]["receipt_id"],
                    },
                )

    def fragment_response(self, model_id: str) -> dict:
        return self._get("fragment_response_models", model_id)

    def fragment_source(self, model_id: str) -> dict:
        return self._get("fragment_source_models", model_id)

    def layer_stack(self, model_id: str) -> dict:
        return self._get("layer_stack_models", model_id)

    def blast_field(self, model_id: str) -> dict:
        return self._get("blast_field_models", model_id)

    def blast_response(self, model_id: str) -> dict:
        return self._get("blast_response_models", model_id)

    def transmission(self, model_id: str) -> dict:
        return self._get("transmission_models", model_id)

    def _get(self, section: str, model_id: str) -> dict:
        try:
            model = self._frozen_data[section][model_id]
        except KeyError as exc:
            raise MissingModelError(
                f"model {model_id!r} is not registered in {section}",
                context={"section": section, "model_id": model_id},
            ) from exc
        return model
