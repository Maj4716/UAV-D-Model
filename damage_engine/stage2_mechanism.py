"""Versioned Stage-2 mechanism experiments from idea2 E3, E5 and C2.

The helpers in this module build explicit in-memory target/registry variants,
preserve common aleatory seeds across variants, and keep the Taylor-Shapiro v2
candidate outside the legacy hollow-ring versus solid-cone factorial.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations, product
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .damage_tree import compile_damage_tree
from .errors import InputValidationError
from .fragments import _sample_directions, generate_fragments
from .munition_profiles import (
    MUNITION_PROFILES,
    PROFILE_REVISION,
    apply_stage2_mechanism_variant,
    profile_document,
    warhead_document,
)
from .penetration import FragmentImpact, resolve_fragment_impact
from .registry import PhysicsRegistry
from .utils import content_hash


E3_ARMOR_MM = (10.0, 1.0)
E3_THRESHOLD_FACTORS = (1.0, 0.1)
E3_DIRECTION_LEVELS = ("legacy_hollow", "solid_cone")
E3_GURNEY_FACTORS = (1.0, 1.3)
E5_SIGMA_DEG = (0.5, 2.0, 5.0)
E5_TAU = (0.04, 0.08, 0.16)
DIRECTION_NOISE_DISTRIBUTIONS = ("normal", "uniform", "laplace")
STAGE2_CONTRACT_VERSION = "stage2-mechanism-experiments@1.0.0"


def _slug_number(value: float) -> str:
    return format(float(value), ".12g").replace(".", "p")


def e3_factorial_design() -> list[dict[str, Any]]:
    """Return the deterministic 2^4 E3 physical-factor design."""

    rows: list[dict[str, Any]] = []
    for armor_mm, threshold, direction, gurney in product(
        E3_ARMOR_MM,
        E3_THRESHOLD_FACTORS,
        E3_DIRECTION_LEVELS,
        E3_GURNEY_FACTORS,
    ):
        factors = {
            "A_armor_mm": armor_mm,
            "T_component_threshold_factor": threshold,
            "F_direction_model": direction,
            "W_gurney_factor": gurney,
        }
        variant_id = (
            f"e3-a{_slug_number(armor_mm)}"
            f"-t{_slug_number(threshold)}"
            f"-f{'hollow' if direction == 'legacy_hollow' else 'solid'}"
            f"-w{_slug_number(gurney)}"
        )
        rows.append({"variant_id": variant_id, "factors": factors})
    return rows


def e5_noise_design(
    distributions: Sequence[str] = ("normal",),
) -> list[dict[str, Any]]:
    """Return sigma x tau rows, optionally crossed with noise shape."""

    names = tuple(str(value) for value in distributions)
    if not names or any(name not in DIRECTION_NOISE_DISTRIBUTIONS for name in names):
        raise InputValidationError(
            "E5 direction distributions must be normal, uniform or laplace"
        )
    rows: list[dict[str, Any]] = []
    for distribution, sigma_deg, tau in product(names, E5_SIGMA_DEG, E5_TAU):
        factors = {
            "direction_noise_distribution": distribution,
            "sigma_deg": sigma_deg,
            "tau": tau,
        }
        rows.append(
            {
                "variant_id": (
                    f"e5-{distribution}-s{_slug_number(sigma_deg)}"
                    f"-tau{_slug_number(tau)}"
                ),
                "factors": factors,
            }
        )
    return rows


def _legacy_source_profile_map() -> dict[str, Any]:
    return {profile.source_model_id: profile for profile in MUNITION_PROFILES.values()}


def gurney_speed_for_profile(profile, *, factor: float = 1.0) -> float:
    """Independent Gurney-cylinder gold using the declared mass basis."""

    effective_casing = profile.casing_mass_kg + profile.fragment_mass_total_kg
    return float(
        profile.gurney_constant_mps
        * float(factor)
        / math.sqrt(effective_casing / profile.charge_mass_kg + 0.5)
    )


def _legacy_outer_angle_deg(
    source_model: Mapping[str, Any],
    profile,
    *,
    gurney_factor: float,
) -> float:
    direction = source_model["direction"]
    if direction.get("kind") != "taylor_ring":
        raise InputValidationError(
            "E3 requires the legacy taylor_ring registry as its baseline; "
            "Taylor-Shapiro remains a separate candidate"
        )
    speed = gurney_speed_for_profile(profile, factor=gurney_factor)
    base = math.atan2(2.0 * speed, profile.detonation_velocity_mps)
    correction = math.radians(float(direction.get("axial_angle_correction_deg", 0.0)))
    return float(math.degrees(base + correction))


def _component_response_model_ids(target: Mapping[str, Any]) -> set[str]:
    result = {
        str(component["response"]["fragment_response_model_id"])
        for component in target["components"]
    }
    if not result:
        raise InputValidationError("Stage-2 target has no component response models")
    return result


def _scaled_component_thresholds(
    registry_data: dict[str, Any],
    target: Mapping[str, Any],
    factor: float,
) -> None:
    scale = float(factor)
    if not math.isfinite(scale) or scale <= 0.0:
        raise InputValidationError("component threshold factor must be positive")
    for model_id in _component_response_model_ids(target):
        try:
            states = registry_data["fragment_response_models"][model_id][
                "damage_states"
            ]
        except KeyError as exc:
            raise InputValidationError(
                "component response model is missing from the registry",
                context={"model_id": model_id},
            ) from exc
        for state in states.values():
            for field in ("residual_energy_50_j", "damage_energy_50_j"):
                if field in state:
                    state[field] = float(state[field]) * scale


def build_e3_variant(
    baseline_target: Mapping[str, Any],
    baseline_registry: Mapping[str, Any],
    factors: Mapping[str, Any],
) -> tuple[dict[str, Any], PhysicsRegistry]:
    """Build one validated E3 target/physics corner without touching disk."""

    armor_mm = float(factors["A_armor_mm"])
    threshold_factor = float(factors["T_component_threshold_factor"])
    direction_level = str(factors["F_direction_model"])
    gurney_factor = float(factors["W_gurney_factor"])
    if armor_mm not in E3_ARMOR_MM:
        raise InputValidationError("E3 armor level must be 10 mm or 1 mm")
    if threshold_factor not in E3_THRESHOLD_FACTORS:
        raise InputValidationError("E3 threshold factor must be 1 or 0.1")
    if direction_level not in E3_DIRECTION_LEVELS:
        raise InputValidationError("E3 direction level is invalid")
    if gurney_factor not in E3_GURNEY_FACTORS:
        raise InputValidationError("E3 Gurney factor must be 1 or 1.3")

    target = deepcopy(dict(baseline_target))
    target["target_revision"] = (
        f"{baseline_target['target_revision']}-stage2-e3-a{_slug_number(armor_mm)}"
    )
    for armor in target["armor"]:
        thickness = armor["response"]["equivalent_thickness"]
        thickness["value"] = armor_mm
        thickness["unit"] = "mm"
        thickness["source_id"] = (
            f"STAGE2-E3-ARMOR-{_slug_number(armor_mm)}MM-UNVERIFIED"
        )

    registry_data = deepcopy(dict(baseline_registry))
    _scaled_component_thresholds(registry_data, target, threshold_factor)
    profile_by_source = _legacy_source_profile_map()
    if set(profile_by_source) - set(registry_data["fragment_source_models"]):
        raise InputValidationError(
            "E3 baseline registry does not contain all three legacy sources"
        )
    for source_id, profile in profile_by_source.items():
        source_model = registry_data["fragment_source_models"][source_id]
        if source_model["direction"].get("kind") != "taylor_ring":
            raise InputValidationError(
                "E3 baseline direction must be legacy taylor_ring; "
                "Taylor-Shapiro is not an E3 factor level"
            )
        if direction_level == "solid_cone":
            half_angle = _legacy_outer_angle_deg(
                source_model,
                profile,
                gurney_factor=gurney_factor,
            )
            source_model["direction"] = {
                "kind": "cone",
                "axis_mode": "impact_inward",
                "half_angle_deg": half_angle,
            }

    variant_fingerprint = content_hash(dict(factors)).split(":", 1)[1][:12]
    registry_data["config_id"] = f"stage2-e3-{variant_fingerprint}"
    provenance = registry_data.setdefault("provenance", {})
    provenance["stage2_mechanism_variant"] = {
        "contract_version": STAGE2_CONTRACT_VERSION,
        "experiment": "E3",
        "factors": dict(factors),
        "taylor_shapiro_policy": "independent_candidate_not_factor_level",
    }
    return target, PhysicsRegistry(registry_data)


def build_e5_variant(
    baseline_registry: Mapping[str, Any],
    factors: Mapping[str, Any],
) -> PhysicsRegistry:
    """Build one validated E5 sigma/tau/noise-shape registry."""

    sigma_deg = float(factors["sigma_deg"])
    tau = float(factors["tau"])
    distribution = str(factors["direction_noise_distribution"])
    if sigma_deg not in E5_SIGMA_DEG or tau not in E5_TAU:
        raise InputValidationError("E5 sigma or tau is outside the preregistered grid")
    if distribution not in DIRECTION_NOISE_DISTRIBUTIONS:
        raise InputValidationError("E5 direction distribution is invalid")
    data = deepcopy(dict(baseline_registry))
    profile_by_source = _legacy_source_profile_map()
    for source_id in profile_by_source:
        try:
            direction = data["fragment_source_models"][source_id]["direction"]
        except KeyError as exc:
            raise InputValidationError(
                "E5 baseline registry is missing a legacy fragment source"
            ) from exc
        if direction.get("kind") != "taylor_ring":
            raise InputValidationError(
                "E5 requires legacy taylor_ring; Taylor-Shapiro is analyzed "
                "as an independent model candidate"
            )
        direction["spread_sigma_deg"] = sigma_deg
        direction["spread_distribution"] = distribution
    for model in data["fragment_response_models"].values():
        model["ballistic_limit"]["transition_width_ratio"] = tau
    fingerprint = content_hash(dict(factors)).split(":", 1)[1][:12]
    data["config_id"] = f"stage2-e5-{fingerprint}"
    provenance = data.setdefault("provenance", {})
    provenance["stage2_mechanism_variant"] = {
        "contract_version": STAGE2_CONTRACT_VERSION,
        "experiment": "E5",
        "factors": dict(factors),
        "equal_variance_noise_shapes": True,
        "taylor_shapiro_policy": "independent_candidate_not_factor_level",
    }
    return PhysicsRegistry(data)


def prepare_stage2_case(
    base_case: Mapping[str, Any],
    *,
    target: Mapping[str, Any],
    target_hash: str,
    registry: PhysicsRegistry,
    variant_id: str,
    replicates: int,
    gurney_factor: float = 1.0,
    direction_level: str = "legacy_hollow",
    factors: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind one base case to an E3/E5 variant while preserving its RNG seed."""

    if int(replicates) < 1:
        raise InputValidationError("Stage-2 replicates must be positive")
    if direction_level not in E3_DIRECTION_LEVELS:
        raise InputValidationError("Stage-2 direction level is invalid")
    case = deepcopy(dict(base_case))
    base_case_id = str(case["case_id"])
    apply_stage2_mechanism_variant(
        case,
        gurney_constant_factor=float(gurney_factor),
        direction_model_kind=(
            "cone" if direction_level == "solid_cone" else "taylor_ring"
        ),
        experiment_id=str(variant_id),
    )
    case["case_id"] = f"{base_case_id}::{variant_id}"
    case["target_ref"] = {
        "target_id": str(target["target_id"]),
        "target_revision": str(target["target_revision"]),
        "content_hash": str(target_hash),
    }
    case["parameter_set_id"] = str(registry.data["config_id"])
    analysis = case["analysis"]
    analysis["replicates"] = int(replicates)
    analysis["adaptive_ci_half_width"] = None
    analysis["capture_component_state_trace"] = True
    metadata = case.setdefault("metadata", {})
    metadata["stage2_base_case_id"] = base_case_id
    metadata["stage2_variant_id"] = str(variant_id)
    metadata["stage2_factors"] = dict(factors or {})
    metadata["common_random_number_contract"] = (
        "same_base_case_aleatory_seed_across_all_variants"
    )
    return case


def _probability_metrics(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "zero_rate": None,
            "saturation_rate": None,
            "intermediate_rate": None,
            "mean_binary_entropy_nats": None,
        }
    array = np.clip(array, 0.0, 1.0)
    clipped = np.clip(array, 1.0e-15, 1.0 - 1.0e-15)
    entropy = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))
    return {
        "count": int(array.size),
        "zero_rate": float(np.mean(array <= 1.0e-6)),
        "saturation_rate": float(np.mean(array >= 0.99)),
        "intermediate_rate": float(np.mean((array > 0.01) & (array < 0.99))),
        "mean_binary_entropy_nats": float(np.mean(entropy)),
    }


def hill_tail_report(
    values: Iterable[float],
    fractions: Sequence[float] = (0.05, 0.10, 0.20),
) -> dict[str, Any]:
    positive = np.sort(
        np.asarray([value for value in values if float(value) > 0.0], dtype=np.float64)
    )
    estimates = []
    for fraction in fractions:
        if not 0.0 < float(fraction) < 1.0:
            raise InputValidationError("Hill fractions must lie in (0,1)")
        k = min(max(1, int(math.floor(len(positive) * float(fraction)))), max(0, len(positive) - 1))
        alpha = None
        xi = None
        if len(positive) >= 10 and k >= 2:
            threshold = positive[-k - 1]
            if threshold > 0.0:
                xi_value = float(np.mean(np.log(positive[-k:] / threshold)))
                if math.isfinite(xi_value) and xi_value > 0.0:
                    xi = xi_value
                    alpha = float(1.0 / xi_value)
        estimates.append(
            {
                "tail_fraction": float(fraction),
                "k": int(k),
                "xi": xi,
                "alpha": alpha,
            }
        )
    primary = next(
        (item for item in estimates if math.isclose(item["tail_fraction"], 0.10)),
        estimates[0],
    )
    return {
        "positive_count": int(len(positive)),
        "primary_tail_fraction": primary["tail_fraction"],
        "primary_alpha": primary["alpha"],
        "estimates": estimates,
    }


def _offline_tree_case_views(
    result: Mapping[str, Any],
    target: Mapping[str, Any],
) -> tuple[list[float], list[float], float]:
    trace = result.get("component_state_trace")
    if not isinstance(trace, Mapping) or not trace.get("records"):
        raise InputValidationError(
            "Stage-2 result requires a nonempty component_state_trace"
        )
    tree = compile_damage_tree(dict(target))
    current_rows: list[list[float]] = []
    identity_rows: list[list[float]] = []
    maximum_error = 0.0
    identity_keys: list[tuple[str, str]] | None = None
    for record in trace["records"]:
        states = record["component_mechanism_states"]
        outputs = tree.evaluate(states)["outputs"]
        online = record.get("system_outputs", {})
        current_rows.append([float(outputs[name]) for name in tree.outputs])
        maximum_error = max(
            maximum_error,
            max(
                (
                    abs(float(outputs[name]) - float(online[name]))
                    for name in tree.outputs
                ),
                default=0.0,
            ),
        )
        combined = states["combined"]
        keys = sorted(
            (component_id, state_name)
            for component_id, state_values in combined.items()
            for state_name in state_values
        )
        if identity_keys is None:
            identity_keys = keys
        elif keys != identity_keys:
            raise InputValidationError(
                "identity tree component/state vocabulary changed across replicates"
            )
        identity_rows.append(
            [float(combined[component][state]) for component, state in keys]
        )
    return (
        np.mean(np.asarray(current_rows), axis=0).tolist(),
        np.mean(np.asarray(identity_rows), axis=0).tolist(),
        float(maximum_error),
    )


def extract_stage2_case_outcome(
    result: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    base_case_id: str,
) -> dict[str, Any]:
    """Extract the compact case record needed for E3/E5 analysis."""

    hit_probabilities: list[float] = []
    positive_energy: list[float] = []
    for component in result["components"].values():
        metrics = component["fragment_metrics"]
        hit = float(metrics["hit_indicator"]["mean"])
        energy = float(metrics["deposited_energy_j"]["mean"])
        hit_probabilities.append(hit)
        if energy > 0.0:
            positive_energy.append(energy)
    current_tree, identity_tree, offline_error = _offline_tree_case_views(
        result,
        target,
    )
    coverage = result["penetration_domain_coverage"]
    return {
        "base_case_id": str(base_case_id),
        "status": str(result["status"]),
        "component_zero_rate": float(
            np.mean(np.asarray(hit_probabilities) <= 1.0e-12)
        ),
        "any_component_support": float(any(value > 1.0e-12 for value in hit_probabilities)),
        "positive_component_energy_j": positive_energy,
        "domain_complete": float(coverage["affected_replicate_count"] == 0),
        "affected_replicate_fraction": float(
            coverage["affected_replicate_fraction"]
        ),
        "current_tree_case_values": current_tree,
        "identity_tree_case_values": identity_tree,
        "offline_tree_max_abs_error": offline_error,
    }


def summarize_stage2_outcomes(outcomes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not outcomes:
        raise InputValidationError("Stage-2 summary requires at least one outcome")
    energies = [
        float(value)
        for outcome in outcomes
        for value in outcome["positive_component_energy_j"]
    ]
    current_values = [
        float(value)
        for outcome in outcomes
        for value in outcome["current_tree_case_values"]
    ]
    identity_values = [
        float(value)
        for outcome in outcomes
        for value in outcome["identity_tree_case_values"]
    ]
    return {
        "case_count": len(outcomes),
        "component_zero_rate": float(
            np.mean([outcome["component_zero_rate"] for outcome in outcomes])
        ),
        "case_any_support_rate": float(
            np.mean([outcome["any_component_support"] for outcome in outcomes])
        ),
        "domain_complete_rate": float(
            np.mean([outcome["domain_complete"] for outcome in outcomes])
        ),
        "mean_affected_replicate_fraction": float(
            np.mean(
                [outcome["affected_replicate_fraction"] for outcome in outcomes]
            )
        ),
        "positive_component_energy_tail": hill_tail_report(energies),
        "tree_views": {
            "current": _probability_metrics(current_values),
            "identity_component_states": _probability_metrics(identity_values),
            "maximum_offline_recompute_abs_error": max(
                float(outcome["offline_tree_max_abs_error"])
                for outcome in outcomes
            ),
        },
        "status_counts": {
            status: sum(outcome["status"] == status for outcome in outcomes)
            for status in sorted({str(item["status"]) for item in outcomes})
        },
    }


def _scalar_metrics_from_summary(
    summary: Mapping[str, Any],
    *,
    tree_view: str,
) -> dict[str, float | None]:
    tree = summary["tree_views"][tree_view]
    return {
        "component_zero_rate": float(summary["component_zero_rate"]),
        "case_any_support_rate": float(summary["case_any_support_rate"]),
        "domain_complete_rate": float(summary["domain_complete_rate"]),
        "hill_tail_alpha": summary["positive_component_energy_tail"][
            "primary_alpha"
        ],
        "tree_zero_rate": tree["zero_rate"],
        "tree_saturation_rate": tree["saturation_rate"],
        "tree_intermediate_rate": tree["intermediate_rate"],
        "tree_entropy_nats": tree["mean_binary_entropy_nats"],
    }


_E3_FACTOR_CODES = {
    "A": lambda row: -1.0 if float(row["A_armor_mm"]) == 10.0 else 1.0,
    "T": lambda row: -1.0 if float(row["T_component_threshold_factor"]) == 1.0 else 1.0,
    "F": lambda row: -1.0 if row["F_direction_model"] == "legacy_hollow" else 1.0,
    "W": lambda row: -1.0 if float(row["W_gurney_factor"]) == 1.0 else 1.0,
    "Tr": lambda row: -1.0 if row["tree_view"] == "current" else 1.0,
}


def _effect_table(expanded_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    factor_names = tuple(_E3_FACTOR_CODES)
    terms = [(name,) for name in factor_names] + list(combinations(factor_names, 2))
    metric_names = tuple(expanded_rows[0]["metrics"])
    effects: dict[str, dict[str, float | None]] = {}
    for term in terms:
        term_id = ":".join(term)
        coded = np.asarray(
            [
                math.prod(_E3_FACTOR_CODES[name](row) for name in term)
                for row in expanded_rows
            ],
            dtype=np.float64,
        )
        effects[term_id] = {}
        for metric in metric_names:
            values = [row["metrics"][metric] for row in expanded_rows]
            if any(value is None or not math.isfinite(float(value)) for value in values):
                effects[term_id][metric] = None
            else:
                effects[term_id][metric] = float(
                    2.0 * np.mean(coded * np.asarray(values, dtype=np.float64))
                )
    return effects


def analyze_e3_factorial(
    variant_records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int = 1000,
    bootstrap_seed: int = 2026112003,
) -> dict[str, Any]:
    """Compute balanced effects and paired-case bootstrap intervals."""

    if len(variant_records) != 16:
        raise InputValidationError("E3 analysis requires all 16 physical variants")
    if int(bootstrap_replicates) < 0:
        raise InputValidationError("bootstrap_replicates must be non-negative")
    base_ids = [
        [str(item["base_case_id"]) for item in record["outcomes"]]
        for record in variant_records
    ]
    if not base_ids[0] or any(ids != base_ids[0] for ids in base_ids[1:]):
        raise InputValidationError(
            "E3 variants must contain the same ordered base cases for CRN analysis"
        )

    def expanded(index: np.ndarray | None = None) -> tuple[list[dict], list[dict]]:
        physical: list[dict] = []
        rows: list[dict] = []
        for record in variant_records:
            selected = (
                list(record["outcomes"])
                if index is None
                else [record["outcomes"][int(i)] for i in index]
            )
            summary = summarize_stage2_outcomes(selected)
            physical.append(
                {
                    "variant_id": record["variant_id"],
                    "factors": dict(record["factors"]),
                    "summary": summary,
                }
            )
            for tree_view in ("current", "identity_component_states"):
                rows.append(
                    {
                        **dict(record["factors"]),
                        "tree_view": tree_view,
                        "metrics": _scalar_metrics_from_summary(
                            summary,
                            tree_view=tree_view,
                        ),
                    }
                )
        return physical, rows

    physical, expanded_rows = expanded()
    point = _effect_table(expanded_rows)
    samples: dict[str, dict[str, list[float]]] = {
        term: {metric: [] for metric in metrics}
        for term, metrics in point.items()
    }
    rng = np.random.default_rng(int(bootstrap_seed))
    count = len(base_ids[0])
    for _ in range(int(bootstrap_replicates)):
        indices = rng.integers(0, count, size=count)
        _, rows = expanded(indices)
        draw = _effect_table(rows)
        for term, metrics in draw.items():
            for metric, value in metrics.items():
                if value is not None and math.isfinite(float(value)):
                    samples[term][metric].append(float(value))
    effect_report: dict[str, Any] = {}
    for term, metrics in point.items():
        effect_report[term] = {}
        for metric, estimate in metrics.items():
            draws = samples[term][metric]
            effect_report[term][metric] = {
                "effect_high_minus_low": estimate,
                "bootstrap_ci95": (
                    [
                        float(np.quantile(draws, 0.025)),
                        float(np.quantile(draws, 0.975)),
                    ]
                    if draws
                    else None
                ),
                "bootstrap_draw_count": len(draws),
            }
    return {
        "schema_version": "1.0.0",
        "experiment": "E3",
        "contract_version": STAGE2_CONTRACT_VERSION,
        "common_random_numbers": True,
        "physical_variants": physical,
        "tree_factor": {
            "low": "current_tree_offline_recompute",
            "high": "identity_component_state_outputs",
        },
        "effects": effect_report,
        "bootstrap": {
            "replicates": int(bootstrap_replicates),
            "seed": int(bootstrap_seed),
            "paired_resampling_unit": "base_case",
        },
    }


def summarize_e5_variants(
    variant_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    variants = []
    for record in variant_records:
        variants.append(
            {
                "variant_id": str(record["variant_id"]),
                "factors": dict(record["factors"]),
                "summary": summarize_stage2_outcomes(record["outcomes"]),
            }
        )
    return {
        "schema_version": "1.0.0",
        "experiment": "E5",
        "contract_version": STAGE2_CONTRACT_VERSION,
        "common_random_numbers": True,
        "equal_variance_noise_shapes": True,
        "variants": variants,
    }


def _normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _clipped_normal_moments(
    mean: float,
    sigma: float,
    lower: float = 0.0,
    upper: float = math.pi,
) -> tuple[float, float]:
    if sigma <= 0.0:
        clipped = float(np.clip(mean, lower, upper))
        return clipped, 0.0
    alpha = (lower - mean) / sigma
    beta = (upper - mean) / sigma
    cdf_alpha = _normal_cdf(alpha)
    cdf_beta = _normal_cdf(beta)
    interior_probability = cdf_beta - cdf_alpha
    pdf_alpha = _normal_pdf(alpha)
    pdf_beta = _normal_pdf(beta)
    clipped_mean = (
        lower * cdf_alpha
        + mean * interior_probability
        + sigma * (pdf_alpha - pdf_beta)
        + upper * (1.0 - cdf_beta)
    )
    clipped_second_moment = (
        lower * lower * cdf_alpha
        + (mean * mean + sigma * sigma) * interior_probability
        + 2.0 * mean * sigma * (pdf_alpha - pdf_beta)
        + sigma
        * sigma
        * (alpha * pdf_alpha - beta * pdf_beta)
        + upper * upper * (1.0 - cdf_beta)
    )
    variance = clipped_second_moment - clipped_mean * clipped_mean
    return float(clipped_mean), float(max(0.0, variance))


def _legacy_angle_gold(
    profile,
    source_model: Mapping[str, Any],
    speed: float,
) -> dict[str, Any]:
    direction = source_model["direction"]
    if direction.get("kind") != "taylor_ring":
        raise InputValidationError(
            "C2 legacy angle gold requires taylor_ring; run Taylor-Shapiro "
            "as its separate candidate report"
        )
    if direction.get("spread_distribution", "normal") != "normal":
        raise InputValidationError(
            "C2 clipped-normal angle gold requires normal direction noise"
        )
    ring_count = int(direction["ring_count"])
    base_count, remainder = divmod(profile.fragment_count, ring_count)
    ring_sizes = [base_count] * ring_count
    ring_sizes[-1] += remainder
    base = math.atan2(2.0 * speed, profile.detonation_velocity_mps)
    correction = math.radians(float(direction.get("axial_angle_correction_deg", 0.0)))
    sigma = math.radians(float(direction.get("spread_sigma_deg", 0.0)))
    means: list[float] = []
    variances: list[float] = []
    weights: list[float] = []
    nominal_deg: list[float] = []
    for index, size in enumerate(ring_sizes):
        normalized = index / max(1, ring_count - 1) if ring_count > 1 else 0.5
        nominal = base + correction * normalized
        mean, variance = _clipped_normal_moments(nominal, sigma)
        means.append(mean)
        variances.append(variance)
        weights.append(size / profile.fragment_count)
        nominal_deg.append(math.degrees(nominal))
    weights_array = np.asarray(weights)
    means_array = np.asarray(means)
    expected_mean = float(np.sum(weights_array * means_array))
    expected_variance = float(
        np.sum(
            weights_array
            * (np.asarray(variances) + (means_array - expected_mean) ** 2)
        )
    )
    return {
        "base_angle_deg": math.degrees(base),
        "nominal_ring_angles_deg": nominal_deg,
        "expected_mean_angle_deg": math.degrees(expected_mean),
        "expected_std_angle_deg": math.degrees(math.sqrt(expected_variance)),
        "spread_distribution": str(direction.get("spread_distribution", "normal")),
        "spread_sigma_deg": math.degrees(sigma),
        "gold_definition": "exact_clipped_normal_ring_mixture_moments",
    }


def _empirical_legacy_angles(
    profile,
    source_model: Mapping[str, Any],
    speed: float,
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    if int(repeats) < 1:
        raise InputValidationError("C2 direction repeats must be positive")
    rng = np.random.default_rng(int(seed))
    axis = np.array([1.0, 0.0, 0.0])
    angles: list[np.ndarray] = []
    for _ in range(int(repeats)):
        directions = _sample_directions(
            profile.fragment_count,
            dict(source_model),
            np.eye(3),
            rng,
            gurney_speed=speed,
            detonation_velocity_mps=profile.detonation_velocity_mps,
            impact_inward_axis_target=axis,
        )
        angles.append(
            np.arccos(np.clip(np.asarray(directions) @ axis, -1.0, 1.0))
        )
    array = np.concatenate(angles)
    return {
        "sample_count": int(array.size),
        "mean_angle_deg": float(np.degrees(np.mean(array))),
        "std_angle_deg": float(np.degrees(np.std(array))),
        "quantiles_deg": {
            str(q): float(np.degrees(np.quantile(array, q)))
            for q in (0.01, 0.10, 0.50, 0.90, 0.99)
        },
    }


def _ballistic_v50(
    model: Mapping[str, Any],
    *,
    mass_kg: float,
    thickness_m: float,
    incidence_cosine: float,
) -> float:
    ballistic = model["ballistic_limit"]
    return float(
        float(ballistic["coefficient"])
        * (thickness_m * 1000.0) ** float(ballistic["thickness_exponent"])
        * (mass_kg * 1000.0) ** float(ballistic["mass_exponent"])
        * (1.0 / incidence_cosine) ** float(ballistic["obliquity_exponent"])
    )


def v50_transition_report(
    model: Mapping[str, Any],
    *,
    trials_per_speed: int = 4096,
    seed: int = 2026112005,
    mass_kg: float = 0.005,
    thickness_m: float = 0.010,
    incidence_cosine: float = 1.0,
) -> dict[str, Any]:
    """Check p(V50)=0.5 and the declared log-speed logistic width."""

    if int(trials_per_speed) < 1:
        raise InputValidationError("V50 trials_per_speed must be positive")
    if not 0.0 < float(incidence_cosine) <= 1.0:
        raise InputValidationError("incidence_cosine must lie in (0,1]")
    tau = float(model["ballistic_limit"].get("transition_width_ratio", 0.08))
    v50 = _ballistic_v50(
        model,
        mass_kg=mass_kg,
        thickness_m=thickness_m,
        incidence_cosine=incidence_cosine,
    )
    direction = np.array(
        [-incidence_cosine, math.sqrt(max(0.0, 1.0 - incidence_cosine**2)), 0.0]
    )
    normal = np.array([1.0, 0.0, 0.0])
    rng = np.random.default_rng(int(seed))
    points = []
    for log_speed_units in (-2.0, -1.0, 0.0, 1.0, 2.0):
        speed = v50 * math.exp(tau * log_speed_units)
        outcomes = [
            resolve_fragment_impact(
                FragmentImpact(
                    speed_mps=speed,
                    mass_kg=mass_kg,
                    direction=direction,
                    surface_normal=normal,
                    equivalent_thickness_m=thickness_m,
                ),
                dict(model),
                rng,
            )
            for _ in range(int(trials_per_speed))
        ]
        points.append(
            {
                "log_speed_units": log_speed_units,
                "speed_mps": speed,
                "analytic_probability": float(outcomes[0].penetration_probability),
                "empirical_perforation_rate": float(
                    np.mean([outcome.perforated for outcome in outcomes])
                ),
            }
        )
    at_v50 = next(item for item in points if item["log_speed_units"] == 0.0)
    return {
        "v50_mps": v50,
        "transition_width_ratio_tau": tau,
        "analytic_probability_at_v50": at_v50["analytic_probability"],
        "empirical_probability_at_v50": at_v50["empirical_perforation_rate"],
        "ten_to_ninety_speed_ratio": math.exp(2.0 * tau * math.log(9.0)),
        "trials_per_speed": int(trials_per_speed),
        "speed_scan": points,
        "maximum_empirical_minus_analytic_abs_error": max(
            abs(
                item["empirical_perforation_rate"]
                - item["analytic_probability"]
            )
            for item in points
        ),
    }


def build_c2_report(
    registry: PhysicsRegistry,
    *,
    direction_repeats: int = 256,
    v50_trials_per_speed: int = 4096,
    seed: int = 2026112007,
) -> dict[str, Any]:
    """Build the three-profile Gurney/angle and V50 analytic audit."""

    profile_reports = []
    for index, profile in enumerate(MUNITION_PROFILES.values()):
        source = registry.fragment_source(profile.source_model_id)
        expected_speed = gurney_speed_for_profile(profile)
        generation_case = {
            "encounter": {
                "detonation_position_m": [0.0, 0.0, 0.0],
                "body_to_target_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                "velocity_target_mps": [0.0, 0.0, 0.0],
                "collision_normal_target_xyz": [-1.0, 0.0, 0.0],
            },
            "munition": profile_document(profile),
            "warhead": warhead_document(profile),
        }
        generated = generate_fragments(
            generation_case,
            registry,
            np.random.default_rng(int(seed) + 1000 + index),
        )
        observed_speeds = np.asarray(
            [fragment.initial_speed_mps for fragment in generated],
            dtype=np.float64,
        )
        angle_gold = _legacy_angle_gold(profile, source, expected_speed)
        empirical = _empirical_legacy_angles(
            profile,
            source,
            expected_speed,
            repeats=direction_repeats,
            seed=int(seed) + index,
        )
        profile_reports.append(
            {
                "profile_id": profile.profile_id,
                "mass_basis": "casing_mass_plus_fragment_mass_for_excludes_fragments",
                "expected_gurney_speed_mps": expected_speed,
                "engine_generated_gurney_speed_mps": float(
                    np.mean(observed_speeds)
                ),
                "maximum_gurney_speed_abs_error_mps": float(
                    np.max(np.abs(observed_speeds - expected_speed))
                ),
                "angle_gold": angle_gold,
                "angle_empirical": empirical,
                "mean_angle_error_deg": (
                    empirical["mean_angle_deg"]
                    - angle_gold["expected_mean_angle_deg"]
                ),
                "std_angle_error_deg": (
                    empirical["std_angle_deg"]
                    - angle_gold["expected_std_angle_deg"]
                ),
            }
        )
    response_reports = {}
    for index, (model_id, model) in enumerate(
        registry.data["fragment_response_models"].items()
    ):
        response_reports[model_id] = v50_transition_report(
            model,
            trials_per_speed=v50_trials_per_speed,
            seed=int(seed) + 100 + index,
        )
    return {
        "schema_version": "1.0.0",
        "experiment": "C2",
        "contract_version": STAGE2_CONTRACT_VERSION,
        "physics_registry_hash": registry.hash,
        "profile_reports": profile_reports,
        "fragment_response_reports": response_reports,
        "taylor_shapiro_policy": (
            "independent_candidate; this C2 report audits the legacy taylor_ring baseline"
        ),
    }
