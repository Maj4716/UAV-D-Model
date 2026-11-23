"""Executable parameter variants for the idea2 C3 sensitivity study.

The interval set is an exploratory scenario envelope, not a probability model.
This module only builds in-memory target/registry/case variants and never writes
parameters back to the authoritative configuration files.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .engine import DamageEngine
from .errors import InputValidationError
from .fast_sampling_proxy import FastSamplingProxy
from .munition_profiles import apply_stage2_mechanism_variant
from .registry import PhysicsRegistry
from .utils import content_hash


C3_PARAMETER_IDS = (
    "armor_thickness_scale",
    "component_threshold_scale",
    "v50_coefficient_scale",
    "gurney_speed_scale",
    "direction_spread_sigma_deg",
)


def parameter_point(
    parameter_ids: Sequence[str], values: Sequence[float]
) -> dict[str, float]:
    """Validate and name one physical C3 design row."""

    names = tuple(map(str, parameter_ids))
    array = np.asarray(values, dtype=np.float64)
    if names != C3_PARAMETER_IDS or array.shape != (len(C3_PARAMETER_IDS),):
        raise InputValidationError(
            "C3 parameter order must match the registered five-parameter contract",
            context={"expected": list(C3_PARAMETER_IDS), "actual": list(names)},
        )
    if not np.isfinite(array).all() or np.any(array <= 0.0):
        raise InputValidationError("C3 parameter values must be finite and positive")
    return {name: float(value) for name, value in zip(names, array, strict=True)}


def _component_response_model_ids(target: Mapping[str, Any]) -> set[str]:
    identifiers = {
        str(component["response"]["fragment_response_model_id"])
        for component in target["components"]
    }
    if not identifiers:
        raise InputValidationError("C3 target has no component response models")
    return identifiers


def build_c3_variant(
    baseline_target: Mapping[str, Any],
    baseline_registry: Mapping[str, Any],
    point: Mapping[str, float],
) -> tuple[dict[str, Any], PhysicsRegistry, str]:
    """Build one in-memory target and registry variant for the five C3 factors."""

    if set(point) != set(C3_PARAMETER_IDS):
        raise InputValidationError(
            "C3 parameter point has a missing or unknown field",
            context={
                "expected": sorted(C3_PARAMETER_IDS),
                "actual": sorted(map(str, point)),
            },
        )
    values = {name: float(point[name]) for name in C3_PARAMETER_IDS}
    if not all(math.isfinite(value) and value > 0.0 for value in values.values()):
        raise InputValidationError("C3 parameter values must be finite and positive")

    fingerprint = content_hash(values).split(":", 1)[1][:16]
    target = deepcopy(dict(baseline_target))
    target["target_revision"] = (
        f"{baseline_target['target_revision']}-c3-{fingerprint}"
    )
    armor_scale = values["armor_thickness_scale"]
    for armor in target["armor"]:
        thickness = armor["response"]["equivalent_thickness"]
        thickness["value"] = float(thickness["value"]) * armor_scale
        thickness["source_id"] = "C3-EXPLORATORY-INTERVAL-NOT-CALIBRATED"
        thickness["validation_status"] = "unverified"

    registry_data = deepcopy(dict(baseline_registry))
    component_models = _component_response_model_ids(target)
    threshold_scale = values["component_threshold_scale"]
    v50_scale = values["v50_coefficient_scale"]
    for model_id, model in registry_data["fragment_response_models"].items():
        ballistic_limit = model["ballistic_limit"]
        ballistic_limit["coefficient"] = (
            float(ballistic_limit["coefficient"]) * v50_scale
        )
        if model_id in component_models:
            for state in model.get("damage_states", {}).values():
                for field in ("residual_energy_50_j", "damage_energy_50_j"):
                    if field in state:
                        state[field] = float(state[field]) * threshold_scale

    sigma_deg = values["direction_spread_sigma_deg"]
    touched_sources = 0
    for source in registry_data["fragment_source_models"].values():
        direction = source["direction"]
        if "spread_sigma_deg" not in direction:
            raise InputValidationError(
                "C3 direction-spread factor requires an explicit spread_sigma_deg"
            )
        direction["spread_sigma_deg"] = sigma_deg
        touched_sources += 1
    if touched_sources == 0:
        raise InputValidationError("C3 registry has no fragment source models")

    registry_data["config_id"] = f"idea2-c3-{fingerprint}"
    registry_data.setdefault("provenance", {})["idea2_c3_sensitivity"] = {
        "status": "exploratory_sensitivity_only_not_physical_probability",
        "parameter_point": values,
        "definition_hash": content_hash(values),
        "writeback_allowed": False,
    }
    return target, PhysicsRegistry(registry_data), fingerprint


def bind_legacy_case_to_c3_variant(
    base_case: Mapping[str, Any],
    *,
    engine: DamageEngine,
    point: Mapping[str, float],
    variant_id: str,
    replicates: int,
) -> dict[str, Any]:
    """Bind a legacy-v1 loitering case while preserving its CRN seed.

    The registered C3 interval set was defined around the legacy Taylor-ring
    baseline.  Taylor-Shapiro is kept as a separate categorical model family.
    """

    if int(replicates) < 1:
        raise InputValidationError("C3 teacher replicates must be positive")
    case = deepcopy(dict(base_case))
    original_seed = int(case["analysis"]["aleatory_seed"])
    original_id = str(case["case_id"])
    apply_stage2_mechanism_variant(
        case,
        gurney_constant_factor=float(point["gurney_speed_scale"]),
        direction_model_kind="taylor_ring",
        experiment_id=f"C3-{variant_id}",
    )
    case["case_id"] = f"{original_id}::C3::{variant_id}"
    case["target_ref"] = {
        "target_id": str(engine.target_data["target_id"]),
        "target_revision": str(engine.target_data["target_revision"]),
        "content_hash": engine.target_hash,
    }
    case["parameter_set_id"] = engine.registry.data["config_id"]
    analysis = case["analysis"]
    analysis["replicates"] = int(replicates)
    analysis["aleatory_seed"] = original_seed
    analysis["adaptive_ci_half_width"] = None
    analysis.pop("adaptive_physical_ci", None)
    analysis["capture_fragment_audit"] = False
    analysis["capture_replicate_trace"] = False
    analysis["capture_component_state_trace"] = False
    metadata = case.setdefault("metadata", {})
    metadata["idea2_c3_sensitivity"] = {
        "variant_id": str(variant_id),
        "parameter_point": {name: float(point[name]) for name in C3_PARAMETER_IDS},
        "common_random_number_seed": original_seed,
        "writeback_allowed": False,
    }
    return case


def proxy_metrics(result: Mapping[str, Any]) -> np.ndarray:
    states = result["component_state_probabilities"]
    degraded = [
        float(values.get("degraded", 0.0)) for values in states.values()
    ]
    return np.asarray(
        [
            float(result["expected_total_component_hits"]),
            float(result["expected_total_component_damage_energy_j"]),
            float(result["supported_component_count"]) / max(1, len(states)),
            float(np.mean(degraded)) if degraded else 0.0,
            float(
                result["proxy_domain_coverage"]["out_of_domain_packet_fraction"]
            ),
        ],
        dtype=np.float64,
    )


PROXY_OUTPUT_NAMES = (
    "expected_total_component_hits",
    "expected_total_component_damage_energy_j",
    "supported_component_fraction",
    "mean_component_p_degraded",
    "proxy_out_of_domain_packet_fraction",
)


def teacher_metrics(result: Mapping[str, Any]) -> np.ndarray:
    components = result["components"]
    hit = [
        float(value["fragment_metrics"]["hit_indicator"]["mean"])
        for value in components.values()
    ]
    deposited = sum(
        float(value["fragment_metrics"]["deposited_energy_j"]["mean"])
        for value in components.values()
    )
    degraded = [
        float(value["damage_states"]["combined"]["degraded"]["mean"])
        for value in components.values()
    ]
    return np.asarray(
        [
            float(result["summary"]["total_component_hits"]["mean"]),
            float(deposited),
            float(np.mean(hit)) if hit else 0.0,
            float(np.mean(degraded)) if degraded else 0.0,
            float(result["penetration_domain_coverage"]["affected_replicate_fraction"]),
        ],
        dtype=np.float64,
    )


TEACHER_OUTPUT_NAMES = (
    "total_component_hits_mean",
    "total_component_deposited_energy_j",
    "mean_component_p_hit",
    "mean_component_p_degraded",
    "domain_affected_replicate_fraction",
)


def evaluate_proxy_point(
    baseline_target: Mapping[str, Any],
    baseline_registry: Mapping[str, Any],
    base_cases: Sequence[Mapping[str, Any]],
    point: Mapping[str, float],
    *,
    source_seed: int,
    azimuths_per_ring: int,
) -> np.ndarray:
    target, registry, variant_id = build_c3_variant(
        baseline_target, baseline_registry, point
    )
    engine = DamageEngine(target, registry)
    proxy = FastSamplingProxy(
        engine,
        source_seed=int(source_seed),
        azimuths_per_ring=int(azimuths_per_ring),
    )
    rows = []
    for case in base_cases:
        bound = bind_legacy_case_to_c3_variant(
            case,
            engine=engine,
            point=point,
            variant_id=variant_id,
            replicates=1,
        )
        rows.append(proxy_metrics(proxy.evaluate(bound)))
    if not rows:
        raise InputValidationError("C3 proxy evaluation requires non-test cases")
    return np.mean(np.vstack(rows), axis=0)


def evaluate_teacher_point(
    baseline_target: Mapping[str, Any],
    baseline_registry: Mapping[str, Any],
    base_cases: Sequence[Mapping[str, Any]],
    point: Mapping[str, float],
    *,
    replicates: int,
) -> np.ndarray:
    target, registry, variant_id = build_c3_variant(
        baseline_target, baseline_registry, point
    )
    engine = DamageEngine(target, registry)
    rows = []
    for case in base_cases:
        bound = bind_legacy_case_to_c3_variant(
            case,
            engine=engine,
            point=point,
            variant_id=variant_id,
            replicates=replicates,
        )
        rows.append(teacher_metrics(engine.run(bound).to_dict()))
    if not rows:
        raise InputValidationError("C3 teacher evaluation requires non-test cases")
    return np.mean(np.vstack(rows), axis=0)


__all__ = [
    "C3_PARAMETER_IDS",
    "PROXY_OUTPUT_NAMES",
    "TEACHER_OUTPUT_NAMES",
    "bind_legacy_case_to_c3_variant",
    "build_c3_variant",
    "evaluate_proxy_point",
    "evaluate_teacher_point",
    "parameter_point",
    "proxy_metrics",
    "teacher_metrics",
]
