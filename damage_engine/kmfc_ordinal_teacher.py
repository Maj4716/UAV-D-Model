"""Software-only KMFC ordinal-teacher prototype.

This module adds a narrow, auditable KMFC contract around
``ordinal_projection``.  It is deliberately opt-in and is not connected to
the legacy production damage tree.  The implementation only establishes
software and mathematical properties; it does not establish physical
validity or approve any threshold for real-world use.

Three evidence models are admitted:

``state_expected_severity``
    A dimensionless, per-replicate component/branch severity in ``[0, 1]``
    derived from approved cumulative component states.

``residual_kinetic_energy_hill``
    Per-replicate residual kinetic energy at the component response stage in
    joules.  This is the energy quantity used by the current fragment response
    law.

``approved_branch_severity``
    A dimensionless, per-replicate branch score in ``[0, 1]`` produced by a
    separately versioned and reviewed lower-level aggregation.  This mode lets
    categories weight functional branches equally instead of accidentally
    weighting branches by their component counts.

The archived ``deposited_energy_j`` quantity is a penetration energy loss
(``pre - post``), not residual kinetic energy.  It is therefore rejected as a
teacher input by this prototype instead of being silently treated as an
interchangeable energy measurement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from .errors import InputValidationError
from .ordinal_projection import (
    OrdinalProjectionConfig,
    OrdinalProjectionResult,
    aggregate_ordinal_projection,
    compile_ordinal_projection,
    project_ordinal_replicates,
)
from .utils import content_hash


KMFC_ORDINAL_TEACHER_SCHEMA_VERSION = "1.0.0"
KMFC_CATEGORY_ORDER = ("K", "M", "F", "C")

_STATUS = "provisional"
_EXECUTION_SCOPE = "software_only"
_CLAIM_SCOPE = "project_specific_simulator_emulation"
_PHYSICAL_STATUS = "not_established"
_DEPLOYMENT_STATUS = "prototype_not_admitted"
_LEGACY_TREE_POLICY = "preserve_unchanged"
_AVAILABILITY_POLICY = "all_required_evidence_and_gate_conditions"
_EVALUATION_ORDER = "per_replicate_smooth_then_hard_gate_before_outer_aggregation"
_PARAMETER_SELECTION_POLICY = "tree_design_fit_only_no_validation_or_test"
_PARAMETER_STATUS = "software_fixture_not_calibrated"

_STATE_MODEL = "state_expected_severity"
_RESIDUAL_ENERGY_MODEL = "residual_kinetic_energy_hill"
_BRANCH_MODEL = "approved_branch_severity"
_MODEL_QUANTITY_AND_UNIT = {
    _STATE_MODEL: ("component_state_expected_severity", "1"),
    _RESIDUAL_ENERGY_MODEL: ("component_residual_kinetic_energy_j", "J"),
    _BRANCH_MODEL: ("approved_branch_severity_score", "1"),
}
_NONINTERCHANGEABLE_ENERGY_QUANTITIES = frozenset(
    {
        "deposited_energy_j",
        "fragment_deposited_energy_j",
        "component_penetration_energy_loss_j",
        "penetration_energy_loss_j",
    }
)

_REQUIRED_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "teacher_id",
        "status",
        "execution_scope",
        "scientific_claim_scope",
        "physical_validity_status",
        "deployment_status",
        "legacy_tree_policy",
        "availability_policy",
        "evaluation_order",
        "parameter_selection_policy",
        "parameter_status",
        "category_order",
        "categories",
    }
)
_REQUIRED_CATEGORY_FIELDS = frozenset(
    {"evidence_model", "evidence_channels", "projection"}
)
_REQUIRED_CHANNEL_FIELDS = frozenset(
    {"name", "quantity", "unit", "source_field", "required"}
)


def _exact(value: Any, expected: str, *, name: str) -> str:
    if value != expected:
        raise InputValidationError(
            f"{name} must be {expected!r} for the software-only KMFC prototype"
        )
    return expected


def _closed_fields(
    value: Mapping[str, Any],
    required: frozenset[str],
    *,
    name: str,
) -> None:
    fields = set(value)
    missing = sorted(required - fields)
    unknown = sorted(fields - required)
    if missing or unknown:
        raise InputValidationError(
            f"{name} fields are incomplete or unknown",
            context={"missing": missing, "unknown": unknown},
        )


def _nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class EvidenceChannelSpec:
    """A required, semantically typed input to one category projection."""

    name: str
    quantity: str
    unit: str
    source_field: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "quantity": self.quantity,
            "unit": self.unit,
            "source_field": self.source_field,
            "required": True,
        }


@dataclass(frozen=True)
class KMFCCategoryTeacherConfig:
    """One category configuration sharing an ordinal latent severity axis."""

    category_id: str
    evidence_model: str
    evidence_channels: tuple[EvidenceChannelSpec, ...]
    projection: OrdinalProjectionConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_model": self.evidence_model,
            "evidence_channels": [item.to_dict() for item in self.evidence_channels],
            "projection": self.projection.to_dict(),
        }


@dataclass(frozen=True)
class KMFCOrdinalTeacherConfig:
    """Frozen four-category software prototype configuration."""

    teacher_id: str
    categories: tuple[KMFCCategoryTeacherConfig, ...]

    def category(self, category_id: str) -> KMFCCategoryTeacherConfig:
        for item in self.categories:
            if item.category_id == category_id:
                return item
        raise InputValidationError(f"unknown KMFC category: {category_id!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": KMFC_ORDINAL_TEACHER_SCHEMA_VERSION,
            "teacher_id": self.teacher_id,
            "status": _STATUS,
            "execution_scope": _EXECUTION_SCOPE,
            "scientific_claim_scope": _CLAIM_SCOPE,
            "physical_validity_status": _PHYSICAL_STATUS,
            "deployment_status": _DEPLOYMENT_STATUS,
            "legacy_tree_policy": _LEGACY_TREE_POLICY,
            "availability_policy": _AVAILABILITY_POLICY,
            "evaluation_order": _EVALUATION_ORDER,
            "parameter_selection_policy": _PARAMETER_SELECTION_POLICY,
            "parameter_status": _PARAMETER_STATUS,
            "category_order": list(KMFC_CATEGORY_ORDER),
            "categories": {
                item.category_id: item.to_dict() for item in self.categories
            },
        }

    @property
    def hash(self) -> str:
        return content_hash(self.to_dict())


def _compile_channel(
    document: Mapping[str, Any],
    *,
    category_id: str,
    index: int,
    expected_quantity: str,
    expected_unit: str,
) -> EvidenceChannelSpec:
    if not isinstance(document, Mapping):
        raise InputValidationError(
            f"categories.{category_id}.evidence_channels[{index}] must be an object"
        )
    _closed_fields(
        document,
        _REQUIRED_CHANNEL_FIELDS,
        name=f"categories.{category_id}.evidence_channels[{index}]",
    )
    name = _nonempty_string(
        document["name"],
        name=f"categories.{category_id}.evidence_channels[{index}].name",
    )
    quantity = _nonempty_string(
        document["quantity"],
        name=f"categories.{category_id}.evidence_channels[{index}].quantity",
    )
    if quantity in _NONINTERCHANGEABLE_ENERGY_QUANTITIES:
        raise InputValidationError(
            "penetration energy loss/deposited_energy_j is not interchangeable with "
            "component residual kinetic energy and is not admitted as teacher evidence",
            context={"category_id": category_id, "channel": name, "quantity": quantity},
        )
    if quantity != expected_quantity:
        raise InputValidationError(
            "evidence channel quantity does not match its declared evidence_model",
            context={
                "category_id": category_id,
                "channel": name,
                "expected_quantity": expected_quantity,
                "observed_quantity": quantity,
            },
        )
    unit = _nonempty_string(
        document["unit"],
        name=f"categories.{category_id}.evidence_channels[{index}].unit",
    )
    if unit != expected_unit:
        raise InputValidationError(
            "evidence channel unit does not match its declared evidence_model",
            context={
                "category_id": category_id,
                "channel": name,
                "expected_unit": expected_unit,
                "observed_unit": unit,
            },
        )
    source_field = _nonempty_string(
        document["source_field"],
        name=f"categories.{category_id}.evidence_channels[{index}].source_field",
    )
    if document["required"] is not True:
        raise InputValidationError(
            "KMFC-v0 only admits required evidence channels; optional-input semantics are not defined"
        )
    return EvidenceChannelSpec(
        name=name,
        quantity=quantity,
        unit=unit,
        source_field=source_field,
    )


def compile_kmfc_ordinal_teacher(
    config: Mapping[str, Any],
) -> KMFCOrdinalTeacherConfig:
    """Validate and freeze a four-category project-specific prototype.

    Exact status and scope strings prevent the executable prototype from being
    promoted to a physically validated tree by changing descriptive metadata.
    """

    if not isinstance(config, Mapping):
        raise InputValidationError("KMFC ordinal teacher configuration must be an object")
    _closed_fields(config, _REQUIRED_TOP_LEVEL_FIELDS, name="KMFC ordinal teacher")
    if config["schema_version"] != KMFC_ORDINAL_TEACHER_SCHEMA_VERSION:
        raise InputValidationError("unsupported KMFC ordinal teacher schema version")
    teacher_id = _nonempty_string(config["teacher_id"], name="teacher_id")
    _exact(config["status"], _STATUS, name="status")
    _exact(config["execution_scope"], _EXECUTION_SCOPE, name="execution_scope")
    _exact(
        config["scientific_claim_scope"],
        _CLAIM_SCOPE,
        name="scientific_claim_scope",
    )
    _exact(
        config["physical_validity_status"],
        _PHYSICAL_STATUS,
        name="physical_validity_status",
    )
    _exact(config["deployment_status"], _DEPLOYMENT_STATUS, name="deployment_status")
    _exact(config["legacy_tree_policy"], _LEGACY_TREE_POLICY, name="legacy_tree_policy")
    _exact(
        config["availability_policy"],
        _AVAILABILITY_POLICY,
        name="availability_policy",
    )
    _exact(config["evaluation_order"], _EVALUATION_ORDER, name="evaluation_order")
    _exact(
        config["parameter_selection_policy"],
        _PARAMETER_SELECTION_POLICY,
        name="parameter_selection_policy",
    )
    _exact(config["parameter_status"], _PARAMETER_STATUS, name="parameter_status")

    order = config["category_order"]
    if not isinstance(order, Sequence) or isinstance(order, (str, bytes)):
        raise InputValidationError("category_order must be the ordered KMFC sequence")
    if tuple(order) != KMFC_CATEGORY_ORDER:
        raise InputValidationError("category_order must be exactly ['K', 'M', 'F', 'C']")
    categories_document = config["categories"]
    if not isinstance(categories_document, Mapping):
        raise InputValidationError("categories must be an object keyed by K, M, F and C")
    category_keys = set(categories_document)
    expected_keys = set(KMFC_CATEGORY_ORDER)
    if category_keys != expected_keys:
        raise InputValidationError(
            "categories must contain exactly K, M, F and C",
            context={
                "missing": sorted(expected_keys - category_keys),
                "unknown": sorted(category_keys - expected_keys),
            },
        )

    compiled_categories: list[KMFCCategoryTeacherConfig] = []
    for category_id in KMFC_CATEGORY_ORDER:
        document = categories_document[category_id]
        if not isinstance(document, Mapping):
            raise InputValidationError(f"categories.{category_id} must be an object")
        _closed_fields(
            document,
            _REQUIRED_CATEGORY_FIELDS,
            name=f"categories.{category_id}",
        )
        evidence_model = _nonempty_string(
            document["evidence_model"],
            name=f"categories.{category_id}.evidence_model",
        )
        if evidence_model not in _MODEL_QUANTITY_AND_UNIT:
            raise InputValidationError(
                "unsupported KMFC evidence_model",
                context={
                    "category_id": category_id,
                    "allowed": sorted(_MODEL_QUANTITY_AND_UNIT),
                },
            )
        expected_quantity, expected_unit = _MODEL_QUANTITY_AND_UNIT[evidence_model]
        channel_documents = document["evidence_channels"]
        if (
            isinstance(channel_documents, (str, bytes))
            or not isinstance(channel_documents, Sequence)
            or not channel_documents
        ):
            raise InputValidationError(
                f"categories.{category_id}.evidence_channels must be nonempty"
            )
        channels = tuple(
            _compile_channel(
                item,
                category_id=category_id,
                index=index,
                expected_quantity=expected_quantity,
                expected_unit=expected_unit,
            )
            for index, item in enumerate(channel_documents)
        )
        names = tuple(item.name for item in channels)
        if len(set(names)) != len(names):
            raise InputValidationError(
                f"categories.{category_id}.evidence channel names must be unique"
            )
        projection_document = document["projection"]
        projection = compile_ordinal_projection(projection_document)
        if projection.category_id != category_id:
            raise InputValidationError(
                f"categories.{category_id}.projection.category_id must equal {category_id!r}"
            )
        if projection.input_names != names:
            raise InputValidationError(
                f"categories.{category_id} evidence channel order must exactly match projection.input_names"
            )
        compiled_categories.append(
            KMFCCategoryTeacherConfig(
                category_id=category_id,
                evidence_model=evidence_model,
                evidence_channels=channels,
                projection=projection,
            )
        )
    return KMFCOrdinalTeacherConfig(
        teacher_id=teacher_id,
        categories=tuple(compiled_categories),
    )


def _validate_category_mapping(
    value: Mapping[str, Any] | None,
    *,
    name: str,
) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise InputValidationError(f"{name} must be an object keyed by KMFC category")
    unknown = sorted(set(value) - set(KMFC_CATEGORY_ORDER))
    if unknown:
        raise InputValidationError(
            f"{name} contains unknown KMFC categories", context={"unknown": unknown}
        )
    return value


@dataclass(frozen=True)
class KMFCCategoryTeacherResult:
    """One category result with an explicit exclusive 0/1/2 level."""

    category: KMFCCategoryTeacherConfig
    projection: OrdinalProjectionResult
    exclusive_level: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        payload = self.projection.to_dict()
        payload.update(
            {
                "evidence_model": self.category.evidence_model,
                "evidence_channels": [
                    item.to_dict() for item in self.category.evidence_channels
                ],
                "exclusive_level": [
                    None if not isfinite(float(value)) else int(value)
                    for value in self.exclusive_level
                ],
                "energy_semantics": (
                    "component_response_residual_kinetic_energy"
                    if self.category.evidence_model == _RESIDUAL_ENERGY_MODEL
                    else (
                        "downstream_component_state_expected_severity"
                        if self.category.evidence_model == _STATE_MODEL
                        else "upstream_versioned_approved_branch_severity"
                    )
                ),
            }
        )
        return payload


@dataclass(frozen=True)
class KMFCOrdinalTeacherResult:
    """Aligned per-replicate K/M/F/C prototype outputs."""

    configuration: KMFCOrdinalTeacherConfig
    categories: tuple[KMFCCategoryTeacherResult, ...]
    replicate_count: int

    def category(self, category_id: str) -> KMFCCategoryTeacherResult:
        for item in self.categories:
            if item.category.category_id == category_id:
                return item
        raise InputValidationError(f"unknown KMFC result category: {category_id!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": KMFC_ORDINAL_TEACHER_SCHEMA_VERSION,
            "teacher_id": self.configuration.teacher_id,
            "configuration_hash": self.configuration.hash,
            "status": _STATUS,
            "execution_scope": _EXECUTION_SCOPE,
            "scientific_claim_scope": _CLAIM_SCOPE,
            "physical_validity_status": _PHYSICAL_STATUS,
            "deployment_status": _DEPLOYMENT_STATUS,
            "replicate_count": self.replicate_count,
            "evaluation_order": _EVALUATION_ORDER,
            "categories": {
                item.category.category_id: item.to_dict() for item in self.categories
            },
        }


def evaluate_kmfc_ordinal_teacher(
    evidence_by_category: Mapping[str, Any],
    configuration: KMFCOrdinalTeacherConfig,
    *,
    evidence_availability_by_category: Mapping[str, Any] | None = None,
    necessary_level1_by_category: Mapping[str, Any] | None = None,
    necessary_level2_by_category: Mapping[str, Any] | None = None,
    necessary_level1_availability_by_category: Mapping[str, Any] | None = None,
    necessary_level2_availability_by_category: Mapping[str, Any] | None = None,
) -> KMFCOrdinalTeacherResult:
    """Evaluate smooth evidence and final hard gates inside each replicate.

    Missing evidence remains unavailable.  A known-false necessary condition
    is a valid level-zero result; an unavailable condition makes the category
    unavailable for that replicate.
    """

    if not isinstance(configuration, KMFCOrdinalTeacherConfig):
        raise InputValidationError("KMFC ordinal teacher configuration must be compiled")
    evidence_map = _validate_category_mapping(
        evidence_by_category, name="evidence_by_category"
    )
    if set(evidence_map) != set(KMFC_CATEGORY_ORDER):
        raise InputValidationError("evidence_by_category must contain exactly K, M, F and C")
    availability_map = _validate_category_mapping(
        evidence_availability_by_category,
        name="evidence_availability_by_category",
    )
    condition1_map = _validate_category_mapping(
        necessary_level1_by_category,
        name="necessary_level1_by_category",
    )
    condition2_map = _validate_category_mapping(
        necessary_level2_by_category,
        name="necessary_level2_by_category",
    )
    condition1_availability_map = _validate_category_mapping(
        necessary_level1_availability_by_category,
        name="necessary_level1_availability_by_category",
    )
    condition2_availability_map = _validate_category_mapping(
        necessary_level2_availability_by_category,
        name="necessary_level2_availability_by_category",
    )

    results: list[KMFCCategoryTeacherResult] = []
    replicate_count: int | None = None
    for category in configuration.categories:
        raw = np.asarray(evidence_map[category.category_id])
        if raw.ndim != 2:
            raise InputValidationError(
                f"evidence_by_category.{category.category_id} must be [replicate,input]"
            )
        if replicate_count is None:
            replicate_count = int(raw.shape[0])
        elif int(raw.shape[0]) != replicate_count:
            raise InputValidationError("all KMFC categories must share a replicate axis")
        availability = availability_map.get(category.category_id)
        values = raw.astype(np.float64, copy=False) if raw.dtype.kind in "fiu" else raw
        if category.evidence_model in {_STATE_MODEL, _BRANCH_MODEL}:
            if availability is None:
                selected = np.ones(raw.shape, dtype=bool)
            else:
                selected = np.asarray(availability)
                if selected.dtype != np.dtype(bool) or selected.shape != raw.shape:
                    raise InputValidationError(
                        f"evidence availability for {category.category_id} must be bool and aligned"
                    )
            if raw.dtype.kind not in "fiu":
                raise InputValidationError(
                    f"state expected severity for {category.category_id} must be numeric"
                )
            if np.any(~np.isfinite(values[selected])) or np.any(
                (values[selected] < 0.0) | (values[selected] > 1.0)
            ):
                raise InputValidationError(
                    "available dimensionless state/branch severity evidence must lie in [0,1]"
                )
        projection = project_ordinal_replicates(
            raw,
            category.projection,
            evidence_availability=availability,
            necessary_level1=condition1_map.get(category.category_id),
            necessary_level2=condition2_map.get(category.category_id),
            necessary_level1_availability=condition1_availability_map.get(
                category.category_id
            ),
            necessary_level2_availability=condition2_availability_map.get(
                category.category_id
            ),
        )
        exclusive = np.full(projection.availability.shape, np.nan, dtype=np.float64)
        available = projection.availability
        exclusive[available] = (
            projection.level_ge_1[available] + projection.level_ge_2[available]
        )
        results.append(
            KMFCCategoryTeacherResult(
                category=category,
                projection=projection,
                exclusive_level=exclusive,
            )
        )
    assert replicate_count is not None
    return KMFCOrdinalTeacherResult(
        configuration=configuration,
        categories=tuple(results),
        replicate_count=replicate_count,
    )


def aggregate_kmfc_ordinal_teacher(
    result: KMFCOrdinalTeacherResult,
) -> dict[str, Any]:
    """Aggregate per-replicate gates without changing their denominators."""

    if not isinstance(result, KMFCOrdinalTeacherResult):
        raise InputValidationError("KMFC aggregation requires a KMFC teacher result")
    categories: dict[str, Any] = {}
    for item in result.categories:
        report = aggregate_ordinal_projection(item.projection)
        report["unavailable_replicates"] = int(
            result.replicate_count - report["trials"]
        )
        report["evidence_model"] = item.category.evidence_model
        categories[item.category.category_id] = report
    return {
        "schema_version": KMFC_ORDINAL_TEACHER_SCHEMA_VERSION,
        "teacher_id": result.configuration.teacher_id,
        "configuration_hash": result.configuration.hash,
        "status": _STATUS,
        "execution_scope": _EXECUTION_SCOPE,
        "scientific_claim_scope": _CLAIM_SCOPE,
        "physical_validity_status": _PHYSICAL_STATUS,
        "deployment_status": _DEPLOYMENT_STATUS,
        "replicate_count": result.replicate_count,
        "evaluation_order": _EVALUATION_ORDER,
        "categories": categories,
        "interpretation": (
            "project-specific criterion-exceedance frequencies; not real-world "
            "KMFC probabilities"
        ),
    }

