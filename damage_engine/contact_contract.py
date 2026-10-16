"""Versioned admission contract for contact-conditioned damage datasets.

S-CONTACT is an input/admission contract.  It proves that one persisted case
can be recomputed as a forward first-armour contact and that the case-selection
declaration is outcome blind.  It does not estimate a hit probability and does
not admit any unconditional mission or damage claim.

The module deliberately does not construct or repair cases.  Generation code
may use the returned record as auditable evidence, while any failed invariant
raises :class:`InputValidationError` with a stable ``reason_code``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

import numpy as np

from .errors import DamageEngineError, InputValidationError
from .geometry import CompiledTarget
from .impact import (
    IMPACT_VALIDATION_HASH,
    IMPACT_MODEL_ID,
    IMPACT_TRIGGER,
    validate_impact_detonation,
)
from .utils import EPS, content_hash, quaternion_matrix_xyzw


S_CONTACT_SCHEMA_VERSION = "1.0.0"
S_CONTACT_CONTRACT_ID = "s-contact-first-armor@1.0.0"
S_CONTACT_MODE = "contact_conditioned_first_armor"
S_CONTACT_ESTIMAND = "criterion_exceedance_probability_given_first_armor_contact"
SELECTION_DECLARATION_VERSION = "1.0.0"

# These are information roles, not interchangeable train/validation/test labels.
# Every role here is conditional on S-CONTACT.  Invalid geometry has a separate
# diagnostic role and never receives a formal damage label.
CONTACT_CONDITIONED_DATA_ROLES = frozenset(
    {
        "tree_design_fit",
        "tree_design_holdout",
        "model_train_reference",
        "contact_conditional_reference",
        "contact_enriched_train",
        "model_validation",
        "sealed_reference_test",
        "geometry_ordinal_challenge",
        "contact_challenge",
    }
)
INVALID_GEOMETRY_DIAGNOSTIC_ROLE = "invalid_geometry_diagnostic"

PROHIBITED_UNCONDITIONAL_ESTIMANDS = frozenset(
    {
        "hit_probability",
        "guidance_success_probability",
        "mission_success_probability",
        "unconditional_damage_probability",
        "unconditional_kmfc_probability",
    }
)

_SELECTION_KEYS = frozenset(
    {
        "schema_version",
        "selection_rule_id",
        "selection_stage",
        "outcome_blind",
        "dependency_paths",
    }
)
_OUTCOME_DEPENDENCY_MARKERS = (
    "outcome",
    "result",
    "response",
    "damage_tree",
    "tree_output",
    "label",
    "teacher",
    "system_y",
    "component_y",
    "component_aux_y",
    "armor_y",
    "mechanism_state",
    "fragment_metric",
    "blast_metric",
    "deposited_energy",
    "penetration",
    "damage_state",
    "severity",
    "successes",
    "level_count",
    "score_rarity",
    "score_uncertainty",
)
# Version 1.0 declarations may depend only on these schema-defined inputs.
# Keeping an exact allow-list prevents outcome aliases such as ``/kmfc/q1`` or
# ``/predictions/criterion_probability`` from becoming valid merely because a
# blacklist did not anticipate their spelling.  New input fields require an
# explicit contract revision and therefore a new S-CONTACT identity.
_PRE_OUTCOME_DEPENDENCY_PATHS = frozenset(
    {
        "/target_ref/target_id",
        "/target_ref/target_revision",
        "/target_ref/content_hash",
        "/munition/profile_id",
        "/munition/profile_revision",
        "/munition/validation_status",
        "/munition/reference_source_id",
        "/munition/reference_source_location",
        "/munition/display_name_zh",
        "/munition/trajectory_reference_point",
        "/munition/detonation_point",
        "/munition/fuze_type",
        "/munition/projectile_diameter_m",
        "/munition/projectile_length_m",
        "/munition/warhead_diameter_m",
        "/munition/warhead_length_m",
        "/munition/wall_thickness_m",
        "/munition/tnt_equivalence",
        "/munition/fragment_ring_count",
        "/munition/fragments_per_ring",
        "/munition/fragment_spread_sigma_deg",
        "/munition/fragment_axial_coverage_ratio",
        "/encounter/trajectory_origin_m",
        "/encounter/velocity_target_mps",
        "/encounter/body_to_target_orientation_xyzw",
        "/encounter/trajectory_reference_point",
        "/encounter/detonation_trigger",
        "/encounter/detonation_delay_s",
        "/encounter/impact_model_id",
        "/encounter/detonation_position_m",
        "/encounter/detonation_time_s",
        "/encounter/collision_distance_m",
        "/encounter/collision_armor_id",
        "/encounter/collision_normal_target_xyz",
        "/environment/air_density_kg_m3",
        "/environment/ambient_pressure_pa",
        "/environment/temperature_k",
        "/analysis/mechanisms",
        "/analysis/coupling_model",
        "/analysis/fidelity_level",
        "/analysis/replicates",
        "/analysis/aleatory_seed",
        "/analysis/epistemic_sample_id",
        "/analysis/adaptive_ci_half_width",
        "/analysis/adaptive_physical_ci",
        "/analysis/evaluation_standard_ids",
        "/analysis/capture_replicate_trace",
        "/analysis/capture_fragment_audit",
        "/parameter_set_id",
    }
)
_ALIGNMENT_COSINE_MINIMUM = 1.0 - 1.0e-6
_FORWARD_DISTANCE_MINIMUM_M = 1.0e-8

# This document is the complete executable identity of S-CONTACT rather than a
# hash of the human-readable contract ID alone.  Generation/checkpoint code can
# therefore prove which mode, estimand, geometry tolerances and information
# boundary admitted a formal case.
S_CONTACT_CONTRACT_DOCUMENT = {
    "schema_version": S_CONTACT_SCHEMA_VERSION,
    "contract_id": S_CONTACT_CONTRACT_ID,
    "contact_mode": S_CONTACT_MODE,
    "estimand": S_CONTACT_ESTIMAND,
    "impact_model_id": IMPACT_MODEL_ID,
    "impact_validation_hash": IMPACT_VALIDATION_HASH,
    "geometry_thresholds": {
        "alignment_cosine_minimum": _ALIGNMENT_COSINE_MINIMUM,
        "forward_contact_distance_minimum_m": _FORWARD_DISTANCE_MINIMUM_M,
    },
    "contact_conditioned_data_roles": sorted(CONTACT_CONDITIONED_DATA_ROLES),
    "prohibited_unconditional_estimands": sorted(
        PROHIBITED_UNCONDITIONAL_ESTIMANDS
    ),
    "outcome_dependency_markers": sorted(_OUTCOME_DEPENDENCY_MARKERS),
    "allowed_pre_outcome_dependency_paths": sorted(
        _PRE_OUTCOME_DEPENDENCY_PATHS
    ),
    "selection_declaration_version": SELECTION_DECLARATION_VERSION,
}
S_CONTACT_CONTRACT_HASH = content_hash(S_CONTACT_CONTRACT_DOCUMENT)


def _refuse(reason_code: str, message: str, **context: Any) -> None:
    raise InputValidationError(
        message,
        context={
            "contract_id": S_CONTACT_CONTRACT_ID,
            "reason_code": reason_code,
            **context,
        },
    )


def _finite_vector3(value: Any, field: str, reason_code: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        _refuse(reason_code, f"{field} must be a finite 3-vector")
    if array.shape != (3,) or not np.isfinite(array).all():
        _refuse(reason_code, f"{field} must be a finite 3-vector")
    return array


def _dependency_is_outcome_aware(path: str) -> bool:
    # Decode the two JSON Pointer escapes before checking path segments.  The
    # declaration is intentionally conservative: a response/model/label path
    # is not a legal admission dependency even when available in the case file.
    segments = [
        segment.replace("~1", "/").replace("~0", "~").lower().replace("-", "_")
        for segment in path.lstrip("/").split("/")
    ]
    return any(
        marker in segment
        for segment in segments
        for marker in _OUTCOME_DEPENDENCY_MARKERS
    )


def _is_nonroot_json_pointer(value: Any) -> bool:
    """Return whether *value* is a non-root RFC 6901 JSON Pointer."""

    return (
        isinstance(value, str)
        and value.startswith("/")
        and value != "/"
        and re.search(r"~(?:[^01]|$)", value) is None
    )


def validate_outcome_blind_selection(declaration: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical selection declaration or fail closed.

    This validates the declared selection dependencies; it does not infer them
    from arbitrary caller code.  A generator must bind the resulting hash into
    its sampling/manifest provenance for the declaration to be auditable.
    """

    if not isinstance(declaration, Mapping):
        _refuse(
            "selection_declaration_invalid",
            "S-CONTACT selection declaration must be an object",
        )
    observed_keys = set(declaration)
    if observed_keys != _SELECTION_KEYS:
        _refuse(
            "selection_declaration_invalid",
            "S-CONTACT selection declaration has missing or unknown fields",
            missing=sorted(_SELECTION_KEYS - observed_keys),
            unknown=sorted(observed_keys - _SELECTION_KEYS),
        )
    if declaration["schema_version"] != SELECTION_DECLARATION_VERSION:
        _refuse(
            "selection_declaration_version_unsupported",
            "S-CONTACT selection declaration version is unsupported",
            observed=declaration["schema_version"],
        )
    rule_id = declaration["selection_rule_id"]
    if not isinstance(rule_id, str) or not rule_id:
        _refuse(
            "selection_declaration_invalid",
            "selection_rule_id must be a nonempty string",
        )
    if declaration["selection_stage"] != "pre_outcome":
        _refuse(
            "selection_not_pre_outcome",
            "S-CONTACT admission selection must run before damage outcomes",
            observed=declaration["selection_stage"],
        )
    if declaration["outcome_blind"] is not True:
        _refuse(
            "selection_not_outcome_blind",
            "S-CONTACT admission selection must explicitly be outcome blind",
        )
    dependencies = declaration["dependency_paths"]
    if not isinstance(dependencies, Sequence) or isinstance(
        dependencies, (str, bytes)
    ):
        _refuse(
            "selection_declaration_invalid",
            "dependency_paths must contain unique non-root JSON Pointer paths",
        )
    # Validate element types before hashing/deduplicating.  Calling set() first
    # leaked a raw TypeError for declarations such as dependency_paths=[{}].
    if any(not _is_nonroot_json_pointer(path) for path in dependencies):
        _refuse(
            "selection_declaration_invalid",
            "dependency_paths must contain unique non-root JSON Pointer paths",
        )
    if len(dependencies) != len(set(dependencies)):
        _refuse(
            "selection_declaration_invalid",
            "dependency_paths must contain unique non-root JSON Pointer paths",
        )
    forbidden = sorted(
        path
        for path in dependencies
        if path not in _PRE_OUTCOME_DEPENDENCY_PATHS
        or _dependency_is_outcome_aware(path)
    )
    if forbidden:
        _refuse(
            "selection_outcome_dependency",
            "S-CONTACT admission selection declares an outcome-aware or unapproved dependency",
            forbidden_dependency_paths=forbidden,
            allowed_dependency_paths=sorted(_PRE_OUTCOME_DEPENDENCY_PATHS),
        )
    canonical = {
        "schema_version": SELECTION_DECLARATION_VERSION,
        "selection_rule_id": rule_id,
        "selection_stage": "pre_outcome",
        "outcome_blind": True,
        "dependency_paths": list(dependencies),
    }
    return {**canonical, "content_hash": content_hash(canonical)}


def assert_contact_conditioned_estimand(estimand: str) -> None:
    """Reject unconditional claims for an S-CONTACT dataset."""

    if estimand in PROHIBITED_UNCONDITIONAL_ESTIMANDS:
        _refuse(
            "unconditional_estimand_forbidden",
            "contact-conditioned data cannot identify an unconditional estimand",
            estimand=estimand,
        )
    if estimand != S_CONTACT_ESTIMAND:
        _refuse(
            "estimand_not_declared_by_contract",
            "estimand is not the versioned S-CONTACT estimand",
            estimand=estimand,
            allowed=S_CONTACT_ESTIMAND,
        )


def _geometry_solution(case: Mapping[str, Any], target: CompiledTarget) -> dict[str, Any]:
    if not isinstance(case, Mapping):
        _refuse("case_not_mapping", "S-CONTACT case must be an object")
    encounter = case.get("encounter")
    if not isinstance(encounter, Mapping):
        _refuse("encounter_missing", "S-CONTACT case requires an encounter object")
    if encounter.get("detonation_trigger") != IMPACT_TRIGGER:
        _refuse(
            "contact_trigger_invalid",
            "S-CONTACT requires the first_armor_contact detonation trigger",
        )

    origin = _finite_vector3(
        encounter.get("trajectory_origin_m"),
        "encounter.trajectory_origin_m",
        "trajectory_origin_invalid",
    )
    velocity = _finite_vector3(
        encounter.get("velocity_target_mps"),
        "encounter.velocity_target_mps",
        "velocity_invalid",
    )
    speed = float(np.linalg.norm(velocity))
    if not np.isfinite(speed) or speed <= EPS:
        _refuse("velocity_zero", "S-CONTACT velocity must be finite and nonzero")
    direction = velocity / speed

    containing = sorted(
        item.object_id for item in target.armor if item.geometry.contains(origin)
    )
    if containing:
        _refuse(
            "trajectory_origin_inside_armor",
            "S-CONTACT trajectory origin must be outside all target armor",
            containing_armor_ids=containing,
        )

    try:
        rotation = quaternion_matrix_xyzw(
            encounter.get("body_to_target_orientation_xyzw")
        )
    except (DamageEngineError, TypeError, ValueError) as exc:
        _refuse(
            "orientation_invalid",
            "S-CONTACT body orientation must be a valid finite quaternion",
            detail=str(exc),
        )
    body_forward = rotation @ np.asarray([1.0, 0.0, 0.0])
    alignment = float(np.dot(body_forward, direction))
    if not np.isfinite(alignment) or alignment < _ALIGNMENT_COSINE_MINIMUM:
        _refuse(
            "orientation_velocity_misaligned",
            "S-CONTACT body +X axis must align with velocity",
            alignment_cosine=alignment,
            minimum=_ALIGNMENT_COSINE_MINIMUM,
        )

    forward_hits = [
        hit for hit in target.ray_hits(origin, direction) if hit.role == "armor"
    ]
    if not forward_hits:
        backward_hits = [
            hit for hit in target.ray_hits(origin, -direction) if hit.role == "armor"
        ]
        if backward_hits:
            _refuse(
                "armor_intersection_only_behind_origin",
                "S-CONTACT requires an armor intersection on the positive velocity ray",
                backward_armor_ids=sorted({hit.object_id for hit in backward_hits}),
            )
        _refuse(
            "no_forward_armor_intersection",
            "S-CONTACT trajectory does not intersect armor on the positive velocity ray",
        )
    hit = min(forward_hits, key=lambda item: (item.entry_t, item.object_id))
    distance = float(hit.entry_t)
    if not np.isfinite(distance) or distance <= _FORWARD_DISTANCE_MINIMUM_M:
        _refuse(
            "forward_contact_distance_not_positive",
            "S-CONTACT first armor intersection must be strictly ahead of the origin",
            collision_distance_m=distance,
        )
    return {
        "origin": origin,
        "velocity": velocity,
        "speed": speed,
        "direction": direction,
        "alignment": alignment,
        "hit": hit,
        "distance": distance,
    }
def audit_s_contact(
    case: Mapping[str, Any],
    target: CompiledTarget,
    *,
    data_role: str,
    estimand: str,
    selection_declaration: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a machine-readable S-CONTACT admission record.

    All checks are read-only.  Geometry is recomputed from the frozen target,
    and the persisted first-contact fields must match the existing
    ``continuous-ray-first-armor-v1`` implementation exactly within its own
    declared tolerances.
    """

    if data_role == INVALID_GEOMETRY_DIAGNOSTIC_ROLE:
        _refuse(
            "diagnostic_role_has_no_formal_label",
            "invalid_geometry_diagnostic cases cannot enter a formal labelled stream",
            data_role=data_role,
        )
    if data_role not in CONTACT_CONDITIONED_DATA_ROLES:
        _refuse(
            "data_role_not_contact_conditioned",
            "data role is not admitted by the S-CONTACT contract",
            data_role=data_role,
        )
    assert_contact_conditioned_estimand(estimand)
    selection = validate_outcome_blind_selection(selection_declaration)
    solution = _geometry_solution(case, target)
    try:
        validate_impact_detonation(case, target)
    except DamageEngineError as exc:
        _refuse(
            "persisted_first_contact_mismatch",
            "persisted first-contact fields do not match frozen-geometry recomputation",
            source_error_code=exc.code,
            detail=exc.message,
        )

    encounter = case["encounter"]
    hit = solution["hit"]
    return {
        "schema_version": S_CONTACT_SCHEMA_VERSION,
        "contract_id": S_CONTACT_CONTRACT_ID,
        "contract_hash": S_CONTACT_CONTRACT_HASH,
        "contact_mode": S_CONTACT_MODE,
        "impact_model_id": IMPACT_MODEL_ID,
        "status": "eligible_contact_conditioned",
        "data_role": data_role,
        "formal_label_allowed": True,
        "estimand": {
            "id": S_CONTACT_ESTIMAND,
            "condition": "first_armor_contact=true",
            "scope": "contact_conditioned_only",
            "prohibited_claims": sorted(PROHIBITED_UNCONDITIONAL_ESTIMANDS),
        },
        "selection": selection,
        "checks": {
            "origin_outside_armor": True,
            "velocity_finite_nonzero": True,
            "body_axis_velocity_aligned": True,
            "positive_forward_armor_intersection": True,
            "persisted_first_contact_recomputed_match": True,
            "selection_pre_outcome": True,
            "selection_outcome_blind": True,
        },
        "first_contact": {
            "collision_armor_id": hit.object_id,
            "collision_distance_m": solution["distance"],
            "collision_time_s": solution["distance"] / solution["speed"],
            "collision_point_target_m": [float(value) for value in hit.entry_point],
            "collision_normal_target_xyz": [float(value) for value in hit.entry_normal],
            "persisted_collision_armor_id": encounter["collision_armor_id"],
            "persisted_impact_model_id": encounter["impact_model_id"],
        },
        "evidence_boundary": (
            "This record admits only first-armor-contact-conditioned KMFC response. "
            "It does not estimate hit probability, guidance success, mission success, "
            "or unconditional damage probability."
        ),
    }
