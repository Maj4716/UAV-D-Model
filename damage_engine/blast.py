from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .errors import InputValidationError, OutOfDomainError
from .geometry import CompiledTarget
from .registry import PhysicsRegistry
from .utils import EPS, loglog_interpolate, sigmoid


@dataclass(frozen=True)
class BlastLoad:
    peak_pressure_pa: float
    impulse_pa_s: float
    positive_duration_s: float
    arrival_time_s: float


@dataclass
class BlastSimulation:
    component_state_probabilities: dict[str, dict[str, float]]
    component_metrics: dict[str, dict[str, float]]
    armor_integrity: dict[str, float]
    armor_metrics: dict[str, dict[str, float]]


def _free_field_load(
    distance_m: float,
    equivalent_mass_kg: float,
    model: dict,
    ambient_pressure_pa: float,
    temperature_k: float | None = None,
) -> BlastLoad:
    if (
        not math.isfinite(distance_m)
        or not math.isfinite(equivalent_mass_kg)
        or distance_m <= 0
        or equivalent_mass_kg <= 0
    ):
        raise OutOfDomainError(
            "blast distance and equivalent mass must be positive",
            context={"distance_m": distance_m, "equivalent_mass_kg": equivalent_mass_kg},
        )
    if not math.isfinite(ambient_pressure_pa) or ambient_pressure_pa <= 0:
        raise InputValidationError("ambient pressure must be finite and positive")
    scaled_distance = distance_m / equivalent_mass_kg ** (1.0 / 3.0)
    points = model["scaled_distance_table"]
    x = np.asarray([row["z_m_kg13"] for row in points], dtype=np.float64)
    peak = loglog_interpolate(
        scaled_distance,
        x,
        np.asarray([row["peak_pressure_pa"] for row in points]),
        quantity="blast peak pressure",
    )
    scaled_impulse = loglog_interpolate(
        scaled_distance,
        x,
        np.asarray([row["scaled_impulse_pa_s_kgm13"] for row in points]),
        quantity="blast scaled impulse",
    )
    scaled_duration = loglog_interpolate(
        scaled_distance,
        x,
        np.asarray([row["scaled_duration_s_kgm13"] for row in points]),
        quantity="blast scaled duration",
    )
    mass_scale = equivalent_mass_kg ** (1.0 / 3.0)
    gas_temperature = float(
        model.get("reference_temperature_k", 288.15)
        if temperature_k is None
        else temperature_k
    )
    if not math.isfinite(gas_temperature) or gas_temperature <= 0:
        raise InputValidationError("blast propagation temperature must be finite and positive")
    sound_speed = math.sqrt(
        float(model.get("heat_capacity_ratio", 1.4))
        * float(model.get("specific_gas_constant_j_kgk", 287.05))
        * gas_temperature
    )
    shock_speed_factor = math.sqrt(max(1.0, 1.0 + peak / max(ambient_pressure_pa, EPS)))
    return BlastLoad(
        peak_pressure_pa=peak,
        impulse_pa_s=scaled_impulse * mass_scale,
        positive_duration_s=scaled_duration * mass_scale,
        arrival_time_s=distance_m / (sound_speed * shock_speed_factor),
    )


def _reflection_factor(
    incident_peak_pa: float,
    ambient_pressure_pa: float,
    incidence_cosine: float,
    model: dict,
) -> float:
    if (
        not math.isfinite(incident_peak_pa)
        or incident_peak_pa < 0
        or not math.isfinite(ambient_pressure_pa)
        or ambient_pressure_pa <= 0
        or not math.isfinite(incidence_cosine)
    ):
        raise InputValidationError("blast reflection inputs must be finite and physically valid")
    reflection = model.get("reflection") or {}
    minimum = float(reflection.get("minimum_factor", 1.0))
    maximum = float(reflection.get("maximum_factor", 2.0))
    pressure_scale = float(reflection.get("pressure_scale_pa", ambient_pressure_pa))
    angular_exponent = float(reflection.get("angular_exponent", 1.0))
    strength = incident_peak_pa / max(incident_peak_pa + pressure_scale, EPS)
    angle = float(np.clip(incidence_cosine, 0.0, 1.0)) ** angular_exponent
    return float(minimum + (maximum - minimum) * strength * angle)


def _transmit(
    load: BlastLoad,
    target: CompiledTarget,
    origin: np.ndarray,
    destination: np.ndarray,
    registry: PhysicsRegistry,
    armor_integrity: dict[str, float],
    parameter_realization: dict[str, dict[str, float | None]],
) -> tuple[BlastLoad, float, list[str]]:
    peak_factor = 1.0
    impulse_factor = 1.0
    total_effective_thickness = 0.0
    panel_ids: list[str] = []
    segment = np.asarray(destination, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    segment_length = float(np.linalg.norm(segment))
    if not math.isfinite(segment_length) or segment_length <= EPS:
        raise InputValidationError("blast transmission segment must have positive finite length")
    direction = segment / segment_length
    for hit in target.armor_on_segment(origin, destination):
        material_id = str(hit.response["material_id"])
        model = registry.transmission(material_id)
        thickness = float(parameter_realization[hit.object_id]["equivalent_thickness_m"])
        if not math.isfinite(thickness) or thickness <= 0:
            raise InputValidationError(
                "armor equivalent thickness must be finite and positive",
                context={"object_id": hit.object_id},
            )
        cosine = max(abs(float(np.dot(direction, hit.entry_normal))), 0.1)
        path_thickness = thickness / cosine
        integrity = float(np.clip(armor_integrity.get(hit.object_id, 1.0), 0.0, 1.0))
        if not math.isfinite(float(armor_integrity.get(hit.object_id, 1.0))):
            raise InputValidationError(
                "armor integrity must be finite",
                context={"object_id": hit.object_id},
            )
        total_effective_thickness += path_thickness * integrity
        peak_half = float(model["peak_half_thickness_m"])
        impulse_half = float(model["impulse_half_thickness_m"])
        floor = float(model.get("minimum_transmission", 0.0))
        intact_peak = max(
            floor,
            math.exp(-math.log(2.0) * path_thickness / peak_half),
        )
        intact_impulse = max(
            floor,
            math.exp(-math.log(2.0) * path_thickness / impulse_half),
        )
        # ``armor_integrity`` is the intact projected-area fraction.  Average
        # transmitted load over intact material and open perforations instead
        # of treating a hole as a uniform reduction in panel thickness.
        peak_factor *= (1.0 - integrity) + integrity * intact_peak
        impulse_factor *= (1.0 - integrity) + integrity * intact_impulse
        panel_ids.append(hit.object_id)
    return (
        BlastLoad(
            peak_pressure_pa=load.peak_pressure_pa * peak_factor,
            impulse_pa_s=load.impulse_pa_s * impulse_factor,
            positive_duration_s=load.positive_duration_s,
            arrival_time_s=load.arrival_time_s,
        ),
        total_effective_thickness,
        panel_ids,
    )


def _state_probability(
    peak_pa: float,
    impulse_pa_s: float,
    duration_s: float,
    state_model: dict,
    vulnerable_ratio: float,
) -> float:
    peak_50 = float(state_model["peak_50_pa"])
    impulse_50 = float(state_model["impulse_50_pa_s"])
    if (
        not math.isfinite(peak_pa)
        or not math.isfinite(impulse_pa_s)
        or not math.isfinite(duration_s)
        or peak_pa < 0
        or impulse_pa_s < 0
        or duration_s < 0
    ):
        raise InputValidationError("blast load values must be finite and non-negative")
    if not math.isfinite(peak_50) or not math.isfinite(impulse_50) or peak_50 <= 0 or impulse_50 <= 0:
        raise InputValidationError("blast response P-I scales must be positive")
    if not math.isfinite(vulnerable_ratio) or not 0.0 <= vulnerable_ratio <= 1.0:
        raise InputValidationError("vulnerable area ratio must lie in [0,1]")
    peak_exponent = float(state_model.get("peak_exponent", 1.0))
    impulse_exponent = float(state_model.get("impulse_exponent", 1.0))
    log_peak_term = (
        peak_exponent * (math.log(peak_pa) - math.log(peak_50))
        if peak_pa > 0
        else -math.inf
    )
    log_impulse_term = (
        impulse_exponent * (math.log(impulse_pa_s) - math.log(impulse_50))
        if impulse_pa_s > 0
        else -math.inf
    )
    log_interaction = float(np.logaddexp(log_peak_term, log_impulse_term))
    duration_50 = state_model.get("duration_50_s")
    if duration_50 is not None:
        duration_scale = float(duration_50)
        duration_exponent = float(state_model.get("duration_exponent", 0.0))
        if not math.isfinite(duration_scale) or duration_scale <= 0:
            raise InputValidationError("blast response duration scale must be finite and positive")
        if duration_s <= 0 and duration_exponent != 0:
            log_interaction = -math.inf if duration_exponent > 0 else math.inf
        elif duration_s > 0:
            log_interaction += duration_exponent * (
                math.log(duration_s) - math.log(duration_scale)
            )
    if log_interaction == -math.inf:
        return 0.0
    slope = float(state_model.get("log_slope", 4.0))
    if not math.isfinite(slope) or slope <= 0:
        raise InputValidationError("blast response log slope must be finite and positive")
    probability = sigmoid(slope * log_interaction)
    return float(np.clip(probability * vulnerable_ratio, 0.0, 1.0))


def simulate_blast(
    case: dict,
    target: CompiledTarget,
    registry: PhysicsRegistry,
    rng: np.random.Generator,
    *,
    parameter_realization: dict[str, dict[str, float | None]],
    armor_integrity: dict[str, float] | None = None,
) -> BlastSimulation:
    blast_source = case["warhead"]["blast_source"]
    field_model = registry.blast_field(blast_source["model_id"])
    equivalent_mass = float(blast_source["equivalent_mass_kg"])
    ambient = float(case["environment"]["ambient_pressure_pa"])
    temperature = float(case["environment"]["temperature_k"])
    origin = np.asarray(case["encounter"]["detonation_position_m"], dtype=np.float64)
    integrity = (
        {item.object_id: 1.0 for item in target.armor}
        if armor_integrity is None
        else armor_integrity
    )
    post_blast_integrity: dict[str, float] = {}
    armor_metrics: dict[str, dict[str, float]] = {}
    for armor in target.armor:
        center = armor.geometry.center
        ray = center - origin
        distance = float(np.linalg.norm(ray))
        incident = _free_field_load(
            distance,
            equivalent_mass,
            field_model,
            ambient,
            temperature,
        )
        direction = ray / distance
        same_hit = next(
            (
                hit
                for hit in target.ray_hits(
                    origin,
                    direction,
                    maximum_distance_m=distance + 1.0e-6,
                )
                if hit.object_id == armor.object_id
            ),
            None,
        )
        incidence_cosine = (
            abs(float(np.dot(direction, same_hit.entry_normal))) if same_hit else 0.5
        )
        reflection = _reflection_factor(
            incident.peak_pressure_pa,
            ambient,
            incidence_cosine,
            field_model,
        )
        reflected = BlastLoad(
            peak_pressure_pa=incident.peak_pressure_pa * reflection,
            impulse_pa_s=incident.impulse_pa_s * math.sqrt(reflection),
            positive_duration_s=incident.positive_duration_s,
            arrival_time_s=incident.arrival_time_s,
        )
        response = armor.source["response"]
        response_model = registry.blast_response(response["blast_response"]["model_id"])
        failed_model = response_model.get("damage_states", {}).get("failed")
        realized_threshold = parameter_realization[armor.object_id][
            "blast_peak_threshold_pa"
        ]
        realized_impulse_threshold = parameter_realization[armor.object_id].get(
            "blast_impulse_threshold_pa_s"
        )
        realized_duration_threshold = parameter_realization[armor.object_id].get(
            "blast_duration_threshold_s"
        )
        if (
            realized_threshold is not None
            or realized_impulse_threshold is not None
            or realized_duration_threshold is not None
        ) and failed_model is None:
            raise InputValidationError(
                "armor blast thresholds require a registered 'failed' damage state",
                context={"object_id": armor.object_id},
            )
        failure_probability = 0.0
        if failed_model is not None:
            state_model = dict(failed_model)
            if realized_threshold is not None:
                state_model["peak_50_pa"] = float(realized_threshold)
            if realized_impulse_threshold is not None:
                state_model["impulse_50_pa_s"] = float(realized_impulse_threshold)
            if realized_duration_threshold is not None:
                state_model["duration_50_s"] = float(realized_duration_threshold)
            failure_probability = _state_probability(
                reflected.peak_pressure_pa,
                reflected.impulse_pa_s,
                reflected.positive_duration_s,
                state_model,
                float(parameter_realization[armor.object_id]["vulnerable_area_ratio"]),
            )
        incoming_integrity = float(integrity.get(armor.object_id, 1.0))
        if not math.isfinite(incoming_integrity) or not 0.0 <= incoming_integrity <= 1.0:
            raise InputValidationError(
                "armor integrity must lie in [0,1]",
                context={"object_id": armor.object_id},
            )
        remaining_integrity = incoming_integrity * (1.0 - failure_probability)
        post_blast_integrity[armor.object_id] = float(np.clip(remaining_integrity, 0.0, 1.0))
        armor_metrics[armor.object_id] = {
            "distance_m": distance,
            "incident_peak_pressure_pa": incident.peak_pressure_pa,
            "incident_impulse_pa_s": incident.impulse_pa_s,
            "reflected_peak_pressure_pa": reflected.peak_pressure_pa,
            "reflected_impulse_pa_s": reflected.impulse_pa_s,
            "positive_duration_s": reflected.positive_duration_s,
            "arrival_time_s": reflected.arrival_time_s,
            "incidence_cosine": incidence_cosine,
            "blast_failure_probability": failure_probability,
            "pre_blast_integrity": incoming_integrity,
            "post_blast_integrity": post_blast_integrity[armor.object_id],
        }
    probabilities: dict[str, dict[str, float]] = {}
    metrics: dict[str, dict[str, float]] = {}
    for component in target.components:
        center = component.geometry.center
        ray = center - origin
        distance = float(np.linalg.norm(ray))
        incident = _free_field_load(
            distance,
            equivalent_mass,
            field_model,
            ambient,
            temperature,
        )
        exposure = component.source.get("exposure_class", "internal")
        if exposure == "external":
            direction = ray / max(distance, EPS)
            same_hit = next(
                (
                    hit
                    for hit in target.ray_hits(origin, direction, maximum_distance_m=distance + 1.0e-6)
                    if hit.object_id == component.object_id
                ),
                None,
            )
            incidence_cosine = (
                abs(float(np.dot(direction, same_hit.entry_normal))) if same_hit else 0.5
            )
            factor = _reflection_factor(incident.peak_pressure_pa, ambient, incidence_cosine, field_model)
            effective = BlastLoad(
                peak_pressure_pa=incident.peak_pressure_pa * factor,
                impulse_pa_s=incident.impulse_pa_s * math.sqrt(factor),
                positive_duration_s=incident.positive_duration_s,
                arrival_time_s=incident.arrival_time_s,
            )
            shielding = 0.0
            panel_ids: list[str] = []
        else:
            effective, shielding, panel_ids = _transmit(
                incident,
                target,
                origin,
                center,
                registry,
                post_blast_integrity,
                parameter_realization,
            )
            amplification = float(field_model.get("compartment_amplification", {}).get(exposure, 1.0))
            effective = BlastLoad(
                peak_pressure_pa=effective.peak_pressure_pa * amplification,
                impulse_pa_s=effective.impulse_pa_s * amplification,
                positive_duration_s=effective.positive_duration_s,
                arrival_time_s=effective.arrival_time_s,
            )
        response = component.source["response"]
        response_model = registry.blast_response(response["blast_response"]["model_id"])
        vulnerable = float(parameter_realization[component.object_id]["vulnerable_area_ratio"])
        states = [state for state in component.source["damage_states"] if state != "intact"]
        raw: dict[str, float] = {}
        for state in states:
            state_model = dict(response_model["damage_states"].get(state, {}))
            if not state_model:
                raw[state] = 0.0
                continue
            realized_threshold = parameter_realization[component.object_id]["blast_peak_threshold_pa"]
            if state == states[0] and realized_threshold is not None:
                state_model["peak_50_pa"] = float(realized_threshold)
            if state == states[0]:
                realized_impulse_threshold = parameter_realization[component.object_id].get(
                    "blast_impulse_threshold_pa_s"
                )
                realized_duration_threshold = parameter_realization[component.object_id].get(
                    "blast_duration_threshold_s"
                )
                if realized_impulse_threshold is not None:
                    state_model["impulse_50_pa_s"] = float(realized_impulse_threshold)
                if realized_duration_threshold is not None:
                    state_model["duration_50_s"] = float(realized_duration_threshold)
            raw[state] = _state_probability(
                effective.peak_pressure_pa,
                effective.impulse_pa_s,
                effective.positive_duration_s,
                state_model,
                vulnerable,
            )
        previous = 1.0
        probabilities[component.object_id] = {}
        for state in states:
            value = min(previous, raw[state])
            probabilities[component.object_id][state] = value
            previous = value
        metrics[component.object_id] = {
            "distance_m": distance,
            "incident_peak_pressure_pa": incident.peak_pressure_pa,
            "incident_impulse_pa_s": incident.impulse_pa_s,
            "effective_peak_pressure_pa": effective.peak_pressure_pa,
            "effective_impulse_pa_s": effective.impulse_pa_s,
            "positive_duration_s": effective.positive_duration_s,
            "arrival_time_s": effective.arrival_time_s,
            "shielding_effective_thickness_m": shielding,
            "shielding_panel_count": float(len(panel_ids)),
        }
    return BlastSimulation(
        component_state_probabilities=probabilities,
        component_metrics=metrics,
        armor_integrity=post_blast_integrity,
        armor_metrics=armor_metrics,
    )
