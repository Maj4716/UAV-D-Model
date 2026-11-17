from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .geometry import CompiledTarget
from .registry import PhysicsRegistry


def _walk_parameter_records(value: Any):
    if isinstance(value, Mapping):
        if "uncertainty" in value and "value" in value:
            yield value
        for child in value.values():
            yield from _walk_parameter_records(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            yield from _walk_parameter_records(child)


def _aabb_overlap(first, second) -> bool:
    lower = np.maximum(first.geometry.bounds_min, second.geometry.bounds_min)
    upper = np.minimum(first.geometry.bounds_max, second.geometry.bounds_max)
    return bool(np.all(upper - lower > 1.0e-12))


def build_model_coverage_report(
    target: Mapping[str, Any],
    compiled_target: CompiledTarget,
    registry: PhysicsRegistry,
) -> dict[str, Any]:
    """Audit static coverage without upgrading it to physical validation."""

    component_ids = {item.object_id for item in compiled_target.components}
    leaf_component_ids = {
        str(node["component_id"])
        for node in target["damage_tree"]["nodes"]
        if node["node_type"] == "leaf"
    }
    unreferenced = sorted(component_ids - leaf_component_ids)
    not_inside_armor_material = sorted(
        component.object_id
        for component in compiled_target.components
        if not any(
            armor.geometry.contains(component.geometry.center)
            for armor in compiled_target.armor
        )
    )
    aabb_overlap_pairs = sorted(
        [first.object_id, second.object_id]
        for index, first in enumerate(compiled_target.components)
        for second in compiled_target.components[index + 1 :]
        if _aabb_overlap(first, second)
    )
    center_containment_pairs = sorted(
        [first.object_id, second.object_id]
        for index, first in enumerate(compiled_target.components)
        for second in compiled_target.components[index + 1 :]
        if first.geometry.contains(second.geometry.center)
        or second.geometry.contains(first.geometry.center)
    )
    unresolved_pi_surface_ids = sorted(
        {
            str(identifier)
            for component in compiled_target.components
            if (
                identifier := component.response["blast_response"].get(
                    "pressure_impulse_surface_id"
                )
            )
            is not None
        }
    )
    missing_peak_threshold_ids: list[str] = []
    for component in compiled_target.components:
        blast_response = component.response["blast_response"]
        target_override = blast_response.get("peak_overpressure_threshold")
        registered = registry.blast_response(str(blast_response["model_id"]))
        registered_states = registered.get("damage_states", {})
        has_registered_peak = any(
            isinstance(state, Mapping)
            and state.get("peak_50_pa") is not None
            and float(state["peak_50_pa"]) > 0.0
            for state in registered_states.values()
        )
        if target_override is None and not has_registered_peak:
            missing_peak_threshold_ids.append(component.object_id)
    missing_peak_threshold_ids.sort()
    parameters = list(_walk_parameter_records(target))
    uncertainty_kinds = Counter(
        str(record["uncertainty"].get("kind", "missing"))
        for record in parameters
    )
    epistemic_count = sum(
        bool(record["uncertainty"].get("epistemic", False))
        for record in parameters
    )
    validation_statuses = Counter(
        str(record.get("validation_status", "missing"))
        for record in parameters
    )
    fragment_model_reuse = Counter(
        str(component.response["fragment_response_model_id"])
        for component in compiled_target.components
    )
    blast_model_reuse = Counter(
        str(component.response["blast_response"]["model_id"])
        for component in compiled_target.components
    )

    static_blockers: list[str] = []
    if registry.validation_status != "validated":
        static_blockers.append("physics_registry_not_validated")
    if unreferenced:
        static_blockers.append("damage_tree_component_coverage_incomplete")
    if unresolved_pi_surface_ids:
        static_blockers.append("pressure_impulse_surface_ids_not_resolved_by_registry")
    if missing_peak_threshold_ids:
        static_blockers.append("component_blast_peak_thresholds_missing")
    if uncertainty_kinds.get("none", 0) > 0:
        static_blockers.append("parameter_uncertainty_unquantified")
    if any(status != "validated" for status in validation_statuses):
        static_blockers.append("target_response_parameters_not_fully_validated")

    # Static structure checks cannot establish empirical credibility.  An
    # external calibration/validation receipt is deliberately not inferred
    # from target or registry labels and must be bound by a higher-level
    # research-admission workflow.
    external_physical_validation_receipt_bound = False
    research_blockers = list(static_blockers)
    research_blockers.append("external_physical_validation_evidence_not_bound")
    diagnostic_flags: list[str] = []
    if not_inside_armor_material:
        diagnostic_flags.append("component_centers_not_inside_armor_material")
    if aabb_overlap_pairs:
        diagnostic_flags.append("component_aabb_overlap_candidates_present")

    return {
        "schema_version": "1.0.0",
        "component_count": len(component_ids),
        "damage_tree_referenced_component_count": len(
            component_ids & leaf_component_ids
        ),
        "damage_tree_component_coverage_fraction": (
            len(component_ids & leaf_component_ids) / len(component_ids)
            if component_ids
            else 1.0
        ),
        "damage_tree_unreferenced_component_ids": unreferenced,
        "component_centers_not_inside_armor_material_ids": (
            not_inside_armor_material
        ),
        # Compatibility alias retained for consumers of the 0.10 report.  It
        # is diagnostic only: solid-membership is not an enclosure test.
        "component_centers_outside_all_armor_ids": not_inside_armor_material,
        "component_aabb_overlap_candidate_pairs": aabb_overlap_pairs,
        "component_center_containment_pairs": center_containment_pairs,
        "unresolved_pressure_impulse_surface_ids": unresolved_pi_surface_ids,
        "components_missing_blast_peak_threshold_ids": missing_peak_threshold_ids,
        "target_parameter_count": len(parameters),
        "target_parameter_uncertainty_kinds": dict(
            sorted(uncertainty_kinds.items())
        ),
        "target_epistemic_parameter_count": epistemic_count,
        "target_parameter_validation_statuses": dict(
            sorted(validation_statuses.items())
        ),
        "component_fragment_response_model_reuse": dict(
            sorted(fragment_model_reuse.items())
        ),
        "component_blast_response_model_reuse": dict(
            sorted(blast_model_reuse.items())
        ),
        "static_coverage_gate_passed": not static_blockers,
        "static_coverage_blockers": static_blockers,
        "diagnostic_review_flags": diagnostic_flags,
        "external_physical_validation_receipt_bound": (
            external_physical_validation_receipt_bound
        ),
        "research_admission_ready": False,
        "research_admission_blockers": research_blockers,
        "scope": "static_model_coverage_audit_not_physical_validation",
    }
