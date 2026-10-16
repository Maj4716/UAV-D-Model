"""Read-only diagnostics for opt-in fragment reachability traces.

The helpers in this module classify records emitted when
``analysis.capture_fragment_audit`` is enabled.  They do not alter fragment
generation, ray traversal, penetration draws, or damage aggregation.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
import math
from typing import Any

import numpy as np

from .errors import InputValidationError
from .utils import normalize, quaternion_matrix_xyzw


REACHABILITY_CATEGORIES = (
    "no_geometric_intersection",
    "blocked_before_component",
    "hit_no_penetration",
    "penetrated_no_damage",
    "response_possible",
)


def _records_by_replicate(
    fragment_audit: Iterable[Mapping[str, Any]],
) -> list[tuple[int, list[dict]]]:
    """Normalize an engine-level or one-replicate fragment audit payload."""

    raw = list(fragment_audit)
    if not raw:
        raise InputValidationError("fragment audit must contain at least one record")
    wrapped = [item for item in raw if isinstance(item, Mapping) and "events" in item]
    if wrapped:
        if len(wrapped) != len(raw):
            raise InputValidationError(
                "fragment audit cannot mix replicate wrappers and event records"
            )
        result: list[tuple[int, list[dict]]] = []
        seen: set[int] = set()
        for fallback, wrapper in enumerate(wrapped):
            replicate = wrapper.get("replicate", fallback)
            if not isinstance(replicate, int) or isinstance(replicate, bool) or replicate < 0:
                raise InputValidationError("fragment audit replicate index is invalid")
            if replicate in seen:
                raise InputValidationError("fragment audit replicate indices must be unique")
            events = wrapper.get("events")
            if not isinstance(events, list) or any(
                not isinstance(item, Mapping) for item in events
            ):
                raise InputValidationError("fragment audit events must be an array of objects")
            seen.add(replicate)
            result.append((replicate, [dict(item) for item in events]))
        return result
    if any(not isinstance(item, Mapping) for item in raw):
        raise InputValidationError("fragment audit records must be objects")
    return [(0, [dict(item) for item in raw])]


def _fragment_id(record: Mapping[str, Any]) -> int:
    value = record.get("fragment_id")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise InputValidationError("fragment audit record has an invalid fragment_id")
    return value


def summarize_component_reachability(
    fragment_audit: Iterable[Mapping[str, Any]],
    component_ids: Iterable[str],
) -> dict[str, Any]:
    """Classify every traced fragment for each requested component.

    Categories are mutually exclusive on the fragment/component axis.  A
    component absent from the ray's complete geometry path is
    ``no_geometric_intersection``.  A component present on that path but never
    processed because traversal stopped earlier is ``blocked_before_component``.
    Processed component intersections are separated by penetration and by
    whether any archived per-state response probability was positive.
    """

    requested = tuple(component_ids)
    if (
        not requested
        or len(set(requested)) != len(requested)
        or any(not isinstance(value, str) or not value for value in requested)
    ):
        raise InputValidationError("component_ids must be unique nonempty strings")

    counts = {
        component_id: {category: 0 for category in REACHABILITY_CATEGORIES}
        for component_id in requested
    }
    replicate_counts: dict[int, int] = {}
    total_traces = 0
    for replicate, records in _records_by_replicate(fragment_audit):
        traces: dict[int, dict] = {}
        intersections: dict[tuple[int, str], list[dict]] = defaultdict(list)
        for record in records:
            record_type = record.get("record_type")
            if record_type == "fragment_trace":
                identifier = _fragment_id(record)
                if identifier in traces:
                    raise InputValidationError(
                        "fragment audit contains duplicate fragment_trace records"
                    )
                candidate_path = record.get("geometry_candidate_path")
                processed_path = record.get("processed_path")
                if (
                    not isinstance(candidate_path, list)
                    or not isinstance(processed_path, list)
                    or any(not isinstance(item, Mapping) for item in candidate_path)
                    or any(not isinstance(item, Mapping) for item in processed_path)
                ):
                    raise InputValidationError(
                        "fragment_trace requires geometry_candidate_path and processed_path arrays"
                    )
                for path_name, path in (
                    ("geometry_candidate_path", candidate_path),
                    ("processed_path", processed_path),
                ):
                    if any(
                        not isinstance(item.get("object_id"), str)
                        or not item.get("object_id")
                        or not isinstance(item.get("role"), str)
                        or not item.get("role")
                        for item in path
                    ):
                        raise InputValidationError(
                            f"fragment_trace {path_name} contains an invalid path item"
                        )
                traces[identifier] = record
            elif record_type == "intersection":
                identifier = _fragment_id(record)
                object_id = record.get("object_id")
                if not isinstance(object_id, str) or not object_id:
                    raise InputValidationError(
                        "fragment intersection record requires a nonempty object_id"
                    )
                intersections[(identifier, object_id)].append(record)

        if not traces:
            raise InputValidationError(
                "fragment audit has no fragment_trace records; legacy hit-only audits "
                "cannot prove reachability"
            )
        unknown_fragment_ids = sorted(
            {fragment_id for fragment_id, _ in intersections} - set(traces)
        )
        if unknown_fragment_ids:
            raise InputValidationError(
                "fragment intersections are missing their fragment_trace records",
                context={"fragment_ids": unknown_fragment_ids},
            )
        replicate_counts[replicate] = len(traces)
        total_traces += len(traces)
        for identifier, trace in traces.items():
            candidate_ids = [
                item.get("object_id") for item in trace["geometry_candidate_path"]
            ]
            for component_id in requested:
                component_events = intersections.get((identifier, component_id), ())
                if component_id not in candidate_ids:
                    category = "no_geometric_intersection"
                elif component_events:
                    if len(component_events) != 1:
                        raise InputValidationError(
                            "fragment audit contains duplicate component intersections",
                            context={
                                "fragment_id": identifier,
                                "component_id": component_id,
                            },
                        )
                    event = component_events[-1]
                    if event.get("role") != "component":
                        raise InputValidationError(
                            "requested component_id resolved to a non-component intersection",
                            context={"component_id": component_id},
                        )
                    penetrated = event.get("penetrated")
                    if not isinstance(penetrated, bool):
                        raise InputValidationError(
                            "component intersection requires a boolean penetrated field"
                        )
                    if not penetrated:
                        category = "hit_no_penetration"
                    else:
                        responses = event.get("damage_state_response_probabilities")
                        if not isinstance(responses, Mapping):
                            raise InputValidationError(
                                "component intersection is missing state response probabilities"
                            )
                        if any(
                            not isinstance(value, (int, float))
                            or isinstance(value, bool)
                            or not math.isfinite(float(value))
                            or not 0.0 <= float(value) <= 1.0
                            for value in responses.values()
                        ):
                            raise InputValidationError(
                                "state response probabilities must be finite values in [0,1]"
                            )
                        positive = any(float(value) > 0.0 for value in responses.values())
                        category = "response_possible" if positive else "penetrated_no_damage"
                else:
                    category = "blocked_before_component"
                counts[component_id][category] += 1

    components = {}
    for component_id, component_counts in counts.items():
        observed = sum(component_counts.values())
        if observed != total_traces:
            raise InputValidationError("fragment reachability categories do not exhaust traces")
        components[component_id] = {
            "fragment_trace_count": observed,
            "counts": component_counts,
            "fractions": {
                name: value / observed for name, value in component_counts.items()
            },
        }
    return {
        "schema_version": "1.0.0",
        "semantics": "mutually_exclusive_fragment_component_reachability",
        "replicate_fragment_counts": {
            str(index): count for index, count in sorted(replicate_counts.items())
        },
        "fragment_trace_count": total_traces,
        "components": components,
        "evidence_boundary": (
            "Geometry and synthetic-response audit only. Categories diagnose the executed "
            "ray/response pipeline and do not establish physical validity."
        ),
    }


def diagnose_front_axis_semantics(
    case: Mapping[str, Any],
    fragment_source_model: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe the FRONT source axis relative to body +X and target velocity.

    This is deliberately diagnostic: it records whether the declared source
    axis is backward in the body frame and whether body +X is aligned with the
    encounter velocity.  It never rewrites the source axis or encounter.
    """

    try:
        encounter = case["encounter"]
        munition = case["munition"]
        direction = fragment_source_model["direction"]
        orientation = encounter["body_to_target_orientation_xyzw"]
        velocity = np.asarray(encounter["velocity_target_mps"], dtype=np.float64)
        axis_body = normalize(
            np.asarray(direction["axis_body"], dtype=np.float64),
            name="fragment source body axis",
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise InputValidationError(
            "front-axis diagnostic requires encounter, munition and source-axis fields"
        ) from exc
    if velocity.shape != (3,) or not np.isfinite(velocity).all():
        raise InputValidationError("front-axis diagnostic velocity must be a finite 3-vector")
    speed = float(np.linalg.norm(velocity))
    if speed <= 0.0:
        raise InputValidationError("front-axis diagnostic requires nonzero encounter velocity")
    rotation = quaternion_matrix_xyzw(orientation)
    body_forward_target = normalize(rotation @ np.asarray([1.0, 0.0, 0.0]))
    source_axis_target = normalize(rotation @ axis_body)
    velocity_direction = velocity / speed
    body_alignment = float(np.dot(body_forward_target, velocity_direction))
    source_alignment = float(np.dot(source_axis_target, velocity_direction))
    source_body_alignment = float(np.dot(axis_body, np.asarray([1.0, 0.0, 0.0])))
    detonation_point = str(munition.get("detonation_point", "unknown"))
    front_backward = detonation_point == "front" and source_body_alignment < -0.999999
    velocity_aligned = body_alignment > 0.999999
    if detonation_point != "front":
        diagnosis = "non_front_profile"
    elif front_backward and velocity_aligned and source_alignment < -0.999999:
        diagnosis = "front_backward_axis_opposes_velocity"
    elif front_backward:
        diagnosis = "front_backward_axis_with_noncanonical_body_velocity_alignment"
    else:
        diagnosis = "front_axis_not_declared_backward"
    return {
        "schema_version": "1.0.0",
        "detonation_point": detonation_point,
        "body_forward_target_xyz": body_forward_target.tolist(),
        "source_axis_body_xyz": axis_body.tolist(),
        "source_axis_target_xyz": source_axis_target.tolist(),
        "velocity_direction_target_xyz": velocity_direction.tolist(),
        "body_forward_dot_velocity": body_alignment,
        "source_axis_dot_velocity": source_alignment,
        "source_axis_dot_body_forward": source_body_alignment,
        "reference_front_backward_rule_satisfied": front_backward,
        "projectile_velocity_aligned_with_body_forward": velocity_aligned,
        "primary_fragment_axis_opposes_projectile_velocity": source_alignment < 0.0,
        "diagnosis": diagnosis,
        "evidence_boundary": (
            "Coordinate-semantic diagnostic only. It does not decide whether the declared "
            "FRONT backward source is an adequate physical model for impact damage."
        ),
    }
