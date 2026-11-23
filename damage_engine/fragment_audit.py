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


def extract_warhead_target_matching_observations(
    fragment_audit: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Extract E7 armor, component-response and post-armor reach margins.

    Log margins are emitted only when their numerator is strictly positive.
    Zero-energy component events remain visible through explicit counters and
    through the below-threshold fraction instead of being replaced by an
    arbitrary epsilon.
    """

    armor_log_margin: list[float] = []
    armor_event_count = 0
    armor_invalid_margin_count = 0
    armor_below_v50_count = 0
    component_event_count = 0
    component_states: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "eligible_event_count": 0,
            "below_threshold_count": 0,
            "nonpositive_energy_count": 0,
            "log_damage_energy_over_e50": [],
        }
    )
    armor_penetrating_fragment_count = 0
    component_reached_after_armor_count = 0

    for _replicate, records in _records_by_replicate(fragment_audit):
        intersections_by_fragment: dict[int, list[tuple[int, dict]]] = defaultdict(list)
        for fallback_index, record in enumerate(records):
            if record.get("record_type") != "intersection":
                continue
            fragment_id = _fragment_id(record)
            order = record.get("global_event_index", fallback_index)
            if not isinstance(order, int) or isinstance(order, bool) or order < 0:
                raise InputValidationError(
                    "fragment intersection has an invalid global_event_index"
                )
            intersections_by_fragment[fragment_id].append((order, record))
            role = record.get("role")
            if role == "armor":
                armor_event_count += 1
                speed = record.get("arrival_speed_mps")
                v50 = record.get("v50_mps")
                if (
                    not isinstance(speed, (int, float))
                    or isinstance(speed, bool)
                    or not math.isfinite(float(speed))
                    or float(speed) <= 0.0
                    or not isinstance(v50, (int, float))
                    or isinstance(v50, bool)
                    or not math.isfinite(float(v50))
                    or float(v50) <= 0.0
                ):
                    armor_invalid_margin_count += 1
                else:
                    ratio = float(speed) / float(v50)
                    armor_log_margin.append(math.log(ratio))
                    armor_below_v50_count += int(ratio < 1.0)
            elif role == "component":
                component_event_count += 1
                energy = record.get("damage_energy_j")
                parameters = record.get("damage_state_response_parameters")
                if not isinstance(parameters, Mapping):
                    continue
                if (
                    not isinstance(energy, (int, float))
                    or isinstance(energy, bool)
                    or not math.isfinite(float(energy))
                    or float(energy) < 0.0
                ):
                    continue
                energy_value = float(energy)
                for state, raw_parameters in parameters.items():
                    if not isinstance(state, str) or not state or not isinstance(
                        raw_parameters, Mapping
                    ):
                        raise InputValidationError(
                            "component audit response parameters are malformed"
                        )
                    threshold = raw_parameters.get("damage_energy_50_j")
                    if (
                        not isinstance(threshold, (int, float))
                        or isinstance(threshold, bool)
                        or not math.isfinite(float(threshold))
                        or float(threshold) <= 0.0
                    ):
                        raise InputValidationError(
                            "component audit damage_energy_50_j must be finite and positive"
                        )
                    state_record = component_states[state]
                    state_record["eligible_event_count"] += 1
                    state_record["below_threshold_count"] += int(
                        energy_value < float(threshold)
                    )
                    if energy_value <= 0.0:
                        state_record["nonpositive_energy_count"] += 1
                    else:
                        state_record["log_damage_energy_over_e50"].append(
                            math.log(energy_value / float(threshold))
                        )

        for ordered in intersections_by_fragment.values():
            armor_was_penetrated = False
            component_reached = False
            for _order, record in sorted(ordered, key=lambda item: item[0]):
                if record.get("role") == "armor" and record.get("penetrated") is True:
                    armor_was_penetrated = True
                elif record.get("role") == "component" and armor_was_penetrated:
                    component_reached = True
            if armor_was_penetrated:
                armor_penetrating_fragment_count += 1
                component_reached_after_armor_count += int(component_reached)

    return {
        "schema_version": "warhead-target-matching-observations@1.0.0",
        "armor_event_count": armor_event_count,
        "armor_invalid_margin_count": armor_invalid_margin_count,
        "armor_below_v50_count": armor_below_v50_count,
        "armor_log_v_over_v50": armor_log_margin,
        "component_event_count": component_event_count,
        "component_states": dict(component_states),
        "armor_penetrating_fragment_count": armor_penetrating_fragment_count,
        "component_reached_after_armor_count": component_reached_after_armor_count,
    }


def combine_warhead_target_matching_observations(
    observations: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    combined: dict[str, Any] = {
        "schema_version": "warhead-target-matching-observations@1.0.0",
        "armor_event_count": 0,
        "armor_invalid_margin_count": 0,
        "armor_below_v50_count": 0,
        "armor_log_v_over_v50": [],
        "component_event_count": 0,
        "component_states": {},
        "armor_penetrating_fragment_count": 0,
        "component_reached_after_armor_count": 0,
    }
    for observation in observations:
        if observation.get("schema_version") != combined["schema_version"]:
            raise InputValidationError("matching-observation schema version is unsupported")
        for field in (
            "armor_event_count",
            "armor_invalid_margin_count",
            "armor_below_v50_count",
            "component_event_count",
            "armor_penetrating_fragment_count",
            "component_reached_after_armor_count",
        ):
            combined[field] += int(observation[field])
        combined["armor_log_v_over_v50"].extend(
            float(value) for value in observation["armor_log_v_over_v50"]
        )
        for state, source in observation["component_states"].items():
            target = combined["component_states"].setdefault(
                state,
                {
                    "eligible_event_count": 0,
                    "below_threshold_count": 0,
                    "nonpositive_energy_count": 0,
                    "log_damage_energy_over_e50": [],
                },
            )
            for field in (
                "eligible_event_count",
                "below_threshold_count",
                "nonpositive_energy_count",
            ):
                target[field] += int(source[field])
            target["log_damage_energy_over_e50"].extend(
                float(value) for value in source["log_damage_energy_over_e50"]
            )
    return combined


def _finite_distribution_summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise InputValidationError("matching margins must be a finite vector")
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "quantiles": {"q10": None, "q50": None, "q90": None},
            "minimum": None,
            "maximum": None,
        }
    quantiles = np.quantile(array, [0.1, 0.5, 0.9])
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "quantiles": {
            "q10": float(quantiles[0]),
            "q50": float(quantiles[1]),
            "q90": float(quantiles[2]),
        },
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def summarize_warhead_target_matching_observations(
    observations: Mapping[str, Any],
) -> dict[str, Any]:
    if observations.get("schema_version") != (
        "warhead-target-matching-observations@1.0.0"
    ):
        raise InputValidationError("matching-observation schema version is unsupported")
    armor_valid = len(observations["armor_log_v_over_v50"])
    states = {}
    for state, values in observations["component_states"].items():
        eligible = int(values["eligible_event_count"])
        states[state] = {
            "eligible_event_count": eligible,
            "nonpositive_energy_count": int(values["nonpositive_energy_count"]),
            "below_threshold_fraction": (
                int(values["below_threshold_count"]) / eligible
                if eligible
                else None
            ),
            "log_damage_energy_over_e50": _finite_distribution_summary(
                values["log_damage_energy_over_e50"]
            ),
        }
    penetrating = int(observations["armor_penetrating_fragment_count"])
    reached = int(observations["component_reached_after_armor_count"])
    return {
        "schema_version": "warhead-target-matching-summary@1.0.0",
        "M_armor": {
            "event_count": int(observations["armor_event_count"]),
            "valid_margin_count": armor_valid,
            "invalid_margin_count": int(observations["armor_invalid_margin_count"]),
            "below_v50_fraction": (
                int(observations["armor_below_v50_count"]) / armor_valid
                if armor_valid
                else None
            ),
            "log_arrival_speed_over_v50": _finite_distribution_summary(
                observations["armor_log_v_over_v50"]
            ),
        },
        "M_comp": {
            "component_hit_event_count": int(observations["component_event_count"]),
            "energy_semantics": "damage_energy_j_over_state_damage_energy_50_j",
            "states": states,
        },
        "M_reach": {
            "armor_penetrating_fragment_count": penetrating,
            "component_reached_after_armor_count": reached,
            "fraction": reached / penetrating if penetrating else None,
        },
        "evidence_boundary": (
            "Executed software-teacher matching diagnostic. Thresholds, armor and "
            "Taylor-Shapiro parameters remain externally unvalidated."
        ),
    }


def summarize_warhead_target_matching(
    fragment_audit: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    return summarize_warhead_target_matching_observations(
        extract_warhead_target_matching_observations(fragment_audit)
    )


def diagnose_front_axis_semantics(
    case: Mapping[str, Any],
    fragment_source_model: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe the resolved FRONT source axis in target coordinates.

    A body-vector source is rotated into the target frame.  An impact-inward
    source is resolved from the negative physical contact normal.  The
    diagnostic never rewrites the source axis or encounter.
    """

    try:
        encounter = case["encounter"]
        munition = case["munition"]
        direction = fragment_source_model["direction"]
        orientation = encounter["body_to_target_orientation_xyzw"]
        velocity = np.asarray(encounter["velocity_target_mps"], dtype=np.float64)
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
    axis_mode = str(direction.get("axis_mode", "body_vector"))
    if axis_mode == "impact_inward":
        try:
            collision_normal = normalize(
                np.asarray(
                    encounter["collision_normal_target_xyz"],
                    dtype=np.float64,
                ),
                name="collision normal",
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise InputValidationError(
                "impact-inward front-axis diagnostic requires a collision normal"
            ) from exc
        source_axis_target = -collision_normal
        axis_body = normalize(rotation.T @ source_axis_target)
        impact_inward = float(
            np.dot(source_axis_target, -collision_normal)
        ) > 0.999999
    else:
        try:
            axis_body = normalize(
                np.asarray(direction["axis_body"], dtype=np.float64),
                name="fragment source body axis",
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise InputValidationError(
                "body-vector front-axis diagnostic requires axis_body"
            ) from exc
        source_axis_target = normalize(rotation @ axis_body)
        impact_inward = False
    velocity_direction = velocity / speed
    body_alignment = float(np.dot(body_forward_target, velocity_direction))
    source_alignment = float(np.dot(source_axis_target, velocity_direction))
    source_body_alignment = float(np.dot(axis_body, np.asarray([1.0, 0.0, 0.0])))
    detonation_point = str(munition.get("detonation_point", "unknown"))
    front_backward = detonation_point == "front" and source_body_alignment < -0.999999
    velocity_aligned = body_alignment > 0.999999
    if axis_mode == "impact_inward" and impact_inward:
        diagnosis = "impact_inward_axis_follows_contact_normal"
    elif axis_mode == "impact_inward":
        diagnosis = "impact_inward_axis_resolution_failed"
    elif detonation_point != "front":
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
        "direction_axis_mode": axis_mode,
        "body_forward_target_xyz": body_forward_target.tolist(),
        "source_axis_body_xyz": axis_body.tolist(),
        "source_axis_target_xyz": source_axis_target.tolist(),
        "velocity_direction_target_xyz": velocity_direction.tolist(),
        "body_forward_dot_velocity": body_alignment,
        "source_axis_dot_velocity": source_alignment,
        "source_axis_dot_body_forward": source_body_alignment,
        "reference_front_backward_rule_satisfied": front_backward,
        "impact_inward_rule_satisfied": impact_inward,
        "projectile_velocity_aligned_with_body_forward": velocity_aligned,
        "primary_fragment_axis_opposes_projectile_velocity": source_alignment < 0.0,
        "diagnosis": diagnosis,
        "evidence_boundary": (
            "Coordinate-semantic diagnostic only. It does not decide whether the declared "
            "FRONT backward source is an adequate physical model for impact damage."
        ),
    }
