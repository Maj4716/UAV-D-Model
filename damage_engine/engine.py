from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .blast import BlastSimulation, simulate_blast
from .damage_tree import CompiledDamageTree, compile_damage_tree
from .errors import DamageEngineError, InputValidationError
from .fragments import (
    FragmentListData,
    FragmentSimulation,
    load_fragment_list,
    simulate_fragments,
)
from .geometry import CompiledTarget, compile_target
from .input_identity import DEPENDENCY_LOGICAL_ID, FrozenCase, case_identity_hash
from .normalization import normalize_target
from .parameters import (
    build_epistemic_realization,
    nominal_target_parameters,
    sample_target_realization,
)
from .registry import PhysicsRegistry
from .schema_resources import schema_tree_hash
from .utils import content_hash, source_tree_hash, statistics_dict
from .validation import validate_inputs
from .impact import resolve_impact_detonation


ENGINE_VERSION = "0.7.0.dev3"
SCHEMA_TREE_HASH = schema_tree_hash()
ENGINE_CODE_HASH = content_hash(
    {
        "python_source_tree_hash": source_tree_hash(Path(__file__).resolve().parent),
        "schema_tree_hash": SCHEMA_TREE_HASH,
    }
)


_PHYSICAL_METRIC_UNITS = {
    "component.fragment.hits": "count",
    "component.fragment.penetrations": "count",
    "component.fragment.mean_arrival_speed_mps": "m/s",
    "component.fragment.mean_penetration_margin": "1",
    "component.fragment.deposited_energy_j": "J",
    "component.blast.distance_m": "m",
    "component.blast.incident_peak_pressure_pa": "Pa",
    "component.blast.incident_impulse_pa_s": "Pa*s",
    "component.blast.effective_peak_pressure_pa": "Pa",
    "component.blast.effective_impulse_pa_s": "Pa*s",
    "component.blast.positive_duration_s": "s",
    "component.blast.arrival_time_s": "s",
    "component.blast.shielding_effective_thickness_m": "m",
    "component.blast.shielding_panel_count": "count",
    "armor.post_fragment_integrity": "1",
    "armor.post_combined_integrity": "1",
    "armor.blast.distance_m": "m",
    "armor.blast.incident_peak_pressure_pa": "Pa",
    "armor.blast.incident_impulse_pa_s": "Pa*s",
    "armor.blast.reflected_peak_pressure_pa": "Pa",
    "armor.blast.reflected_impulse_pa_s": "Pa*s",
    "armor.blast.positive_duration_s": "s",
    "armor.blast.arrival_time_s": "s",
    "armor.blast.incidence_cosine": "1",
    "armor.blast.blast_failure_probability": "1",
    "armor.blast.pre_blast_integrity": "1",
    "armor.blast.post_blast_integrity": "1",
    "summary.total_fragments": "count",
    "summary.total_armor_hits": "count",
    "summary.total_armor_penetrations": "count",
    "summary.total_component_hits": "count",
    "summary.total_component_penetrations": "count",
}


def build_replicate_trace(replicates: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the joint per-replicate records needed for convergence studies.

    This opt-in trace preserves the order in which the tree and mechanism
    outputs were evaluated.  It deliberately avoids recomputing a tree from
    aggregate leaves, because nonlinear gates generally do not commute with
    averaging.
    """
    if not replicates:
        raise InputValidationError("replicate trace requires at least one completed replicate")
    records = []
    for index, replicate in enumerate(replicates):
        required = {
            "mechanism_states", "fragment_metrics", "blast_metrics", "tree", "summary",
            "parameter_realization_hash",
        }
        missing = sorted(required - set(replicate))
        if missing:
            raise InputValidationError(
                "replicate trace source is incomplete", context={"replicate": index, "missing": missing}
            )
        tree = replicate["tree"]
        if not isinstance(tree, dict) or not {"outputs", "node_probabilities"}.issubset(tree):
            raise InputValidationError("replicate trace tree record is incomplete")
        records.append(
            {
                "replicate_index": index,
                "parameter_realization_hash": replicate["parameter_realization_hash"],
                "system_outputs": tree["outputs"],
                "tree_node_probabilities": tree["node_probabilities"],
                "component_mechanism_states": replicate["mechanism_states"],
                "fragment_metrics": replicate["fragment_metrics"],
                "blast_metrics": replicate["blast_metrics"],
                "summary": replicate["summary"],
            }
        )
    return {
        "schema_version": "1.0.0",
        "semantics": "joint_per_replicate_outputs_before_aggregation",
        "replicate_count": len(records),
        "records": records,
        "scientific_status": "raw_simulator_repetitions_not_independent_physical_validation",
    }


@dataclass
class EngineResult:
    payload: dict[str, Any]

    @property
    def status(self) -> str:
        return str(self.payload["status"])

    def to_dict(self) -> dict[str, Any]:
        return self.payload

    def write_json(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as stream:
            json.dump(self.payload, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _zero_fragment(target: CompiledTarget) -> FragmentSimulation:
    return FragmentSimulation(
        component_state_probabilities={
            item.object_id: {
                state: 0.0
                for state in item.source["damage_states"]
                if state != "intact"
            }
            for item in target.components
        },
        component_metrics={
            item.object_id: {
                "hits": 0.0,
                "penetrations": 0.0,
                "mean_arrival_speed_mps": 0.0,
                "mean_penetration_margin": 0.0,
                "deposited_energy_j": 0.0,
            }
            for item in target.components
        },
        armor_integrity={item.object_id: 1.0 for item in target.armor},
        total_fragments=0,
        total_armor_hits=0,
        total_armor_penetrations=0,
        total_component_hits=0,
        total_component_penetrations=0,
    )


def _zero_blast(target: CompiledTarget) -> BlastSimulation:
    return BlastSimulation(
        component_state_probabilities={
            item.object_id: {
                state: 0.0
                for state in item.source["damage_states"]
                if state != "intact"
            }
            for item in target.components
        },
        component_metrics={
            item.object_id: {
                "distance_m": 0.0,
                "incident_peak_pressure_pa": 0.0,
                "incident_impulse_pa_s": 0.0,
                "effective_peak_pressure_pa": 0.0,
                "effective_impulse_pa_s": 0.0,
                "positive_duration_s": 0.0,
                "arrival_time_s": 0.0,
                "shielding_effective_thickness_m": 0.0,
                "shielding_panel_count": 0.0,
            }
            for item in target.components
        },
        armor_integrity={item.object_id: 1.0 for item in target.armor},
        armor_metrics={
            item.object_id: {
                "distance_m": 0.0,
                "incident_peak_pressure_pa": 0.0,
                "incident_impulse_pa_s": 0.0,
                "reflected_peak_pressure_pa": 0.0,
                "reflected_impulse_pa_s": 0.0,
                "positive_duration_s": 0.0,
                "arrival_time_s": 0.0,
                "incidence_cosine": 0.0,
                "blast_failure_probability": 0.0,
                "pre_blast_integrity": 1.0,
                "post_blast_integrity": 1.0,
            }
            for item in target.armor
        },
    )


def _combine_states(
    target: CompiledTarget,
    fragment: dict[str, dict[str, float]],
    blast: dict[str, dict[str, float]],
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, dict[str, float]]],
]:
    combined: dict[str, dict[str, float]] = {}
    frechet_bounds: dict[str, dict[str, dict[str, float]]] = {}
    for item in target.components:
        identifier = item.object_id
        combined[identifier] = {}
        frechet_bounds[identifier] = {}
        for state in item.source["damage_states"]:
            if state == "intact":
                continue
            fragment_probability = fragment[identifier][state]
            blast_probability = blast[identifier][state]
            combined[identifier][state] = float(
                np.clip(
                    1.0 - (1.0 - fragment_probability) * (1.0 - blast_probability),
                    0.0,
                    1.0,
                )
            )
            frechet_bounds[identifier][state] = {
                "lower": float(max(fragment_probability, blast_probability)),
                "upper": float(min(1.0, fragment_probability + blast_probability)),
            }
    return combined, frechet_bounds


class DamageEngine:
    """Validated orchestration layer for fragment/blast Monte Carlo simulation."""

    def __init__(self, target: dict, registry: PhysicsRegistry):
        self.input_target_hash = content_hash(target)
        self.target_data = normalize_target(target)
        self.registry = registry
        self.compiled_target = compile_target(self.target_data)
        self.damage_tree: CompiledDamageTree = compile_damage_tree(self.target_data)
        self.target_hash = content_hash(self.target_data)

    def run(self, case: dict, *, fail_closed: bool = True) -> EngineResult:
        try:
            return EngineResult(self._run(self.prepare_case(case)))
        except DamageEngineError as error:
            if fail_closed:
                raise
            return EngineResult(
                {
                    "status": "refused",
                    "case_id": case.get("case_id"),
                    "refusal": error.to_dict(),
                    "provenance": self._provenance(case, replicate_count=0),
                }
            )

    def prepare_case(self, case: dict) -> dict:
        """Materialize geometry-derived fields before hashing or simulation."""

        return resolve_impact_detonation(case, self.compiled_target)

    def _run(self, case: dict) -> dict:
        validate_inputs(
            self.target_data,
            case,
            self.registry,
            compiled_target=self.compiled_target,
        )
        fragment_source = case["warhead"]["fragment_source"]
        fragment_list_data: FragmentListData | None = (
            load_fragment_list(fragment_source)
            if fragment_source.get("fragment_list_ref") is not None
            else None
        )
        mechanisms = set(case["analysis"]["mechanisms"])
        coupling = case["analysis"]["coupling_model"]
        maximum_replicates = int(case["analysis"]["replicates"])
        adaptive_target = case["analysis"].get("adaptive_ci_half_width")
        physical_tolerances = case["analysis"].get("adaptive_physical_ci") or {}
        self._validate_physical_tolerances(physical_tolerances, mechanisms)
        adaptive_enabled = adaptive_target is not None or bool(physical_tolerances)
        capture_audit = bool(case["analysis"].get("capture_fragment_audit", False))
        capture_replicate_trace = bool(case["analysis"].get("capture_replicate_trace", False))
        root_seed = np.random.SeedSequence(int(case["analysis"]["aleatory_seed"]))
        replicate_seeds = root_seed.spawn(maximum_replicates)
        epistemic_seed = int(
            content_hash(
                {
                    "target_hash": self.target_hash,
                    "epistemic_sample_id": case["analysis"]["epistemic_sample_id"],
                }
            ).split(":", 1)[1][:16],
            16,
        )
        epistemic = build_epistemic_realization(
            self.target_data,
            np.random.default_rng(epistemic_seed),
        )
        parameter_features = nominal_target_parameters(self.target_data)
        for object_id, values in epistemic.items():
            for key, value in values.items():
                if value is not None:
                    parameter_features[object_id][key] = value
        replicates: list[dict] = []
        audit: list[dict] = []
        for replicate_index, seed in enumerate(replicate_seeds):
            parameter_seed, fragment_seed, blast_seed = seed.spawn(3)
            parameter_rng = np.random.default_rng(parameter_seed)
            fragment_rng = np.random.default_rng(fragment_seed)
            blast_rng = np.random.default_rng(blast_seed)
            parameter_realization = sample_target_realization(
                self.target_data,
                parameter_rng,
                epistemic,
            )
            fragment = (
                simulate_fragments(
                    case,
                    self.compiled_target,
                    self.registry,
                    fragment_rng,
                    parameter_realization=parameter_realization,
                    capture_audit=capture_audit,
                    fragment_list_data=fragment_list_data,
                )
                if "fragment" in mechanisms
                else _zero_fragment(self.compiled_target)
            )
            blast_integrity = (
                fragment.armor_integrity
                if coupling == "sequential" and "fragment" in mechanisms
                else {item.object_id: 1.0 for item in self.compiled_target.armor}
            )
            blast = (
                simulate_blast(
                    case,
                    self.compiled_target,
                    self.registry,
                    blast_rng,
                    parameter_realization=parameter_realization,
                    armor_integrity=blast_integrity,
                )
                if "blast" in mechanisms
                else _zero_blast(self.compiled_target)
            )
            combined, combined_frechet_bounds = _combine_states(
                self.compiled_target,
                fragment.component_state_probabilities,
                blast.component_state_probabilities,
            )
            if "blast" not in mechanisms:
                combined_armor_integrity = fragment.armor_integrity
            elif coupling == "sequential" and "fragment" in mechanisms:
                combined_armor_integrity = blast.armor_integrity
            elif "fragment" in mechanisms:
                combined_armor_integrity = {
                    item.object_id: (
                        fragment.armor_integrity[item.object_id]
                        * blast.armor_integrity[item.object_id]
                    )
                    for item in self.compiled_target.armor
                }
            else:
                combined_armor_integrity = blast.armor_integrity
            mechanism_states = {
                "fragment": fragment.component_state_probabilities,
                "blast": blast.component_state_probabilities,
                "combined": combined,
            }
            tree = self.damage_tree.evaluate(mechanism_states)
            replicates.append(
                {
                    "mechanism_states": mechanism_states,
                    "combined_frechet_bounds": combined_frechet_bounds,
                    "fragment_metrics": fragment.component_metrics,
                    "blast_metrics": blast.component_metrics,
                    "fragment_armor_integrity": fragment.armor_integrity,
                    "combined_armor_integrity": combined_armor_integrity,
                    "armor_blast_metrics": blast.armor_metrics,
                    "tree": tree,
                    "parameter_realization_hash": content_hash(parameter_realization),
                    "summary": {
                        "total_fragments": float(fragment.total_fragments),
                        "total_armor_hits": float(fragment.total_armor_hits),
                        "total_armor_penetrations": float(fragment.total_armor_penetrations),
                        "total_component_hits": float(fragment.total_component_hits),
                        "total_component_penetrations": float(fragment.total_component_penetrations),
                    },
                }
            )
            if capture_audit:
                audit.append(
                    {
                        "replicate": replicate_index,
                        "events": fragment.audit,
                    }
                )
            if adaptive_enabled and len(replicates) >= 32 and self._has_converged(
                replicates,
                probability_target=(
                    float(adaptive_target) if adaptive_target is not None else None
                ),
                physical_tolerances=physical_tolerances,
            ):
                break
        components = self._aggregate_components(replicates)
        armor = self._aggregate_armor(replicates)
        requested_standards = tuple(case["analysis"]["evaluation_standard_ids"])
        tree = self._aggregate_tree(replicates, requested_standards)
        summary = self._aggregate_summary(replicates)
        warnings = []
        if self.registry.validation_status != "validated":
            warnings.append("Physics registry is not validated; outputs are not real-world certified.")
        if coupling == "none" and mechanisms == {"fragment", "blast"}:
            warnings.append("Fragment/blast coupling is disabled; combined probabilities use conditional independence.")
        elif coupling == "sequential" and mechanisms == {"fragment", "blast"}:
            warnings.append(
                "Sequential armor coupling is applied, but fragment/blast state probabilities "
                "are still combined using conditional independence within each replicate."
            )
        if adaptive_target is not None and not physical_tolerances:
            warnings.append(
                "Adaptive convergence covers probability labels only; physical metrics require "
                "separate unit-specific convergence criteria."
            )
        achieved_half_width = self._maximum_probability_half_width(replicates)
        physical_convergence = self._physical_convergence_report(
            replicates, physical_tolerances
        )
        probability_target_met = (
            achieved_half_width is not None and achieved_half_width <= float(adaptive_target)
            if adaptive_target is not None
            else None
        )
        all_targets_met = (
            len(replicates) >= 32
            and (probability_target_met is not False)
            and physical_convergence["target_met"]
            if adaptive_enabled
            else None
        )
        convergence = {
            "mode": "adaptive" if adaptive_enabled else "fixed_replicates",
            "probability_scope": [
                "damage_tree.outputs",
                "component.damage_states.fragment",
                "component.damage_states.blast",
                "component.damage_states.combined",
                "component.damage_states.combined.frechet_union_bounds",
            ],
            "physical_metrics_in_scope": bool(physical_tolerances),
            "physical_metric_tolerances": physical_tolerances,
            "physical_metric_results": physical_convergence["selectors"],
            "physical_metric_unmet_series_count": physical_convergence[
                "unmet_series_count"
            ],
            "physical_metric_worst_series": physical_convergence["worst_series"],
            "physical_target_met": (
                physical_convergence["target_met"] if physical_tolerances else None
            ),
            "target_half_width": (
                float(adaptive_target) if adaptive_target is not None else None
            ),
            "minimum_required_replicates": 32 if adaptive_enabled else None,
            "achieved_max_half_width": achieved_half_width,
            "probability_target_met": probability_target_met,
            "target_met": all_targets_met,
            "reached_replicate_limit": len(replicates) == maximum_replicates,
        }
        payload = {
            "status": "ok",
            "case_id": case["case_id"],
            "target_id": self.target_data["target_id"],
            "component_order": list(self.compiled_target.component_order),
            "replicate_count": len(replicates),
            "mechanisms": sorted(mechanisms),
            "coupling_model": coupling,
            "combined_dependency_assumption": "conditional_independence_given_replicate",
            "combined_dependency_bounds": "frechet_union",
            "convergence": convergence,
            "parameter_realization": parameter_features,
            "armor": armor,
            "components": components,
            "damage_tree": tree,
            "summary": summary,
            "warnings": warnings,
            "provenance": self._provenance(
                case,
                replicate_count=len(replicates),
                fragment_list_provenance=(
                    fragment_list_data.provenance
                    if fragment_list_data is not None
                    else None
                ),
                epistemic_realization=epistemic,
                parameter_realization=parameter_features,
            ),
        }
        if capture_audit:
            payload["fragment_audit"] = audit
        if capture_replicate_trace:
            payload["replicate_trace"] = build_replicate_trace(replicates)
        return payload

    def _has_converged(
        self,
        replicates: list[dict],
        *,
        probability_target: float | None,
        physical_tolerances: dict,
    ) -> bool:
        half_width = self._maximum_probability_half_width(replicates)
        probability_met = (half_width is not None and half_width <= probability_target
                           if probability_target is not None else True)
        return probability_met and self._physical_convergence_report(
            replicates, physical_tolerances
        )["target_met"]

    @staticmethod
    def _validate_physical_tolerances(
        tolerances: dict,
        mechanisms: set[str],
    ) -> None:
        unknown = sorted(set(tolerances) - set(_PHYSICAL_METRIC_UNITS))
        if unknown:
            raise InputValidationError(
                "adaptive physical convergence contains unsupported metric selectors",
                context={
                    "unsupported": unknown,
                    "supported": sorted(_PHYSICAL_METRIC_UNITS),
                },
            )
        unavailable = []
        for selector in tolerances:
            if (
                selector.startswith("component.fragment.")
                or selector == "armor.post_fragment_integrity"
                or selector.startswith("summary.total_")
            ) and "fragment" not in mechanisms:
                unavailable.append(selector)
            if (
                selector.startswith("component.blast.")
                or selector.startswith("armor.blast.")
            ) and "blast" not in mechanisms:
                unavailable.append(selector)
        if unavailable:
            raise InputValidationError(
                "adaptive physical convergence requests metrics for a disabled mechanism",
                context={"unavailable": sorted(set(unavailable)), "mechanisms": sorted(mechanisms)},
            )

    def _physical_metric_series(
        self,
        replicates: list[dict],
        selector: str,
    ) -> list[tuple[str, np.ndarray, bool]]:
        probability_metric = selector.endswith("probability") or selector.endswith("integrity")
        if selector.startswith("component.fragment."):
            key = selector.removeprefix("component.fragment.")
            return [
                (
                    component.object_id,
                    np.asarray(
                        [item["fragment_metrics"][component.object_id][key] for item in replicates],
                        dtype=np.float64,
                    ),
                    probability_metric,
                )
                for component in self.compiled_target.components
            ]
        if selector.startswith("component.blast."):
            key = selector.removeprefix("component.blast.")
            return [
                (
                    component.object_id,
                    np.asarray(
                        [item["blast_metrics"][component.object_id][key] for item in replicates],
                        dtype=np.float64,
                    ),
                    probability_metric,
                )
                for component in self.compiled_target.components
            ]
        if selector == "armor.post_fragment_integrity":
            return [
                (
                    armor.object_id,
                    np.asarray(
                        [item["fragment_armor_integrity"][armor.object_id] for item in replicates],
                        dtype=np.float64,
                    ),
                    True,
                )
                for armor in self.compiled_target.armor
            ]
        if selector == "armor.post_combined_integrity":
            return [
                (
                    armor.object_id,
                    np.asarray(
                        [item["combined_armor_integrity"][armor.object_id] for item in replicates],
                        dtype=np.float64,
                    ),
                    True,
                )
                for armor in self.compiled_target.armor
            ]
        if selector.startswith("armor.blast."):
            key = selector.removeprefix("armor.blast.")
            return [
                (
                    armor.object_id,
                    np.asarray(
                        [item["armor_blast_metrics"][armor.object_id][key] for item in replicates],
                        dtype=np.float64,
                    ),
                    probability_metric,
                )
                for armor in self.compiled_target.armor
            ]
        if selector.startswith("summary."):
            key = selector.removeprefix("summary.")
            return [
                (
                    "target",
                    np.asarray([item["summary"][key] for item in replicates], dtype=np.float64),
                    False,
                )
            ]
        raise InputValidationError(
            "unsupported adaptive physical metric selector",
            context={"selector": selector},
        )

    def _physical_convergence_report(
        self,
        replicates: list[dict],
        tolerances: dict,
    ) -> dict:
        if not tolerances:
            return {
                "target_met": True,
                "unmet_series_count": 0,
                "worst_series": None,
                "selectors": {},
            }
        selector_results: dict[str, dict] = {}
        unmet_series_count = 0
        worst_series: dict | None = None
        worst_ratio = -1.0
        for selector, tolerance in tolerances.items():
            absolute = tolerance.get("absolute_half_width")
            relative = tolerance.get("relative_half_width")
            relative_floor = float(tolerance.get("relative_floor", 0.0))
            selector_worst: dict | None = None
            selector_worst_ratio = -1.0
            selector_unmet = 0
            series_values = self._physical_metric_series(replicates, selector)
            for series_id, values, probability in series_values:
                stats = statistics_dict(values, probability=probability)
                estimable = stats["ci95_upper"] is not None and stats["ci95_lower"] is not None
                half_width = (0.5 * (stats["ci95_upper"] - stats["ci95_lower"])
                              if estimable else None)
                allowed_candidates = []
                if absolute is not None:
                    allowed_candidates.append(float(absolute))
                if relative is not None:
                    allowed_candidates.append(
                        float(relative) * max(abs(float(stats["mean"])), relative_floor)
                    )
                allowed = max(allowed_candidates, default=0.0)
                met = estimable and half_width <= allowed
                if not met:
                    selector_unmet += 1
                    unmet_series_count += 1
                ratio = (
                    half_width / allowed
                    if estimable and allowed > 0.0
                    else (0.0 if estimable and half_width == 0.0 else float("inf"))
                )
                result = {
                    "series_id": series_id,
                    "unit": _PHYSICAL_METRIC_UNITS[selector],
                    "mean": float(stats["mean"]),
                    "ci95_half_width": float(half_width) if estimable else None,
                    "uncertainty_estimable": bool(estimable),
                    "sample_count": int(stats.get("sample_count", len(values))),
                    "allowed_half_width": float(allowed),
                    "tolerance_ratio": (
                        float(ratio) if np.isfinite(ratio) else None
                    ),
                    "target_met": bool(met),
                }
                if ratio > selector_worst_ratio:
                    selector_worst_ratio = ratio
                    selector_worst = result
                if ratio > worst_ratio:
                    worst_ratio = ratio
                    worst_series = {"selector": selector, **result}
            selector_results[selector] = {
                "unit": _PHYSICAL_METRIC_UNITS[selector],
                "series_count": len(series_values),
                "unmet_series_count": selector_unmet,
                "target_met": selector_unmet == 0,
                "worst_series": selector_worst,
            }
        return {
            "target_met": unmet_series_count == 0,
            "unmet_series_count": unmet_series_count,
            "worst_series": worst_series,
            "selectors": selector_results,
        }

    def _probability_series(self, replicates: list[dict]) -> list[np.ndarray]:
        values: list[np.ndarray] = []
        for output in self.damage_tree.outputs:
            values.append(
                np.asarray([item["tree"]["outputs"][output] for item in replicates])
            )
        for component in self.compiled_target.components:
            for mechanism in ("fragment", "blast", "combined"):
                for state in component.source["damage_states"]:
                    if state == "intact":
                        continue
                    values.append(
                        np.asarray(
                            [
                                item["mechanism_states"][mechanism][component.object_id][state]
                                for item in replicates
                            ]
                        )
                    )
                    if mechanism == "combined":
                        for bound in ("lower", "upper"):
                            values.append(
                                np.asarray(
                                    [
                                        item["combined_frechet_bounds"][component.object_id][
                                            state
                                        ][bound]
                                        for item in replicates
                                    ]
                                )
                            )
        return values

    def _maximum_probability_half_width(self, replicates: list[dict]) -> float | None:
        half_widths = []
        for series in self._probability_series(replicates):
            stats = statistics_dict(series, probability=True)
            if stats["ci95_upper"] is None or stats["ci95_lower"] is None:
                return None
            half_widths.append(0.5 * (stats["ci95_upper"] - stats["ci95_lower"]))
        return max(half_widths, default=0.0)

    def _aggregate_components(self, replicates: list[dict]) -> dict:
        output: dict[str, dict] = {}
        for component in self.compiled_target.components:
            identifier = component.object_id
            state_result: dict[str, dict] = {}
            for mechanism in ("fragment", "blast", "combined"):
                state_result[mechanism] = {}
                for state in component.source["damage_states"]:
                    if state == "intact":
                        continue
                    values = [
                        item["mechanism_states"][mechanism][identifier][state]
                        for item in replicates
                    ]
                    statistics = statistics_dict(values, probability=True)
                    if mechanism == "combined":
                        statistics["frechet_union_lower_bound"] = statistics_dict(
                            [
                                item["combined_frechet_bounds"][identifier][state]["lower"]
                                for item in replicates
                            ],
                            probability=True,
                        )
                        statistics["frechet_union_upper_bound"] = statistics_dict(
                            [
                                item["combined_frechet_bounds"][identifier][state]["upper"]
                                for item in replicates
                            ],
                            probability=True,
                        )
                    state_result[mechanism][state] = statistics
            fragment_keys = replicates[0]["fragment_metrics"][identifier].keys()
            fragment_metrics = {
                key: statistics_dict(
                    [item["fragment_metrics"][identifier][key] for item in replicates]
                )
                for key in fragment_keys
            }
            blast_keys = replicates[0]["blast_metrics"][identifier].keys()
            blast_metrics = {
                key: statistics_dict(
                    [item["blast_metrics"][identifier][key] for item in replicates]
                )
                for key in blast_keys
            }
            output[identifier] = {
                "name": component.source["name"],
                "component_type": component.source["component_type"],
                "damage_states": state_result,
                "fragment_metrics": fragment_metrics,
                "blast_metrics": blast_metrics,
            }
        return output

    def _aggregate_armor(self, replicates: list[dict]) -> dict:
        output: dict[str, dict] = {}
        for armor in self.compiled_target.armor:
            identifier = armor.object_id
            metric_keys = replicates[0]["armor_blast_metrics"][identifier].keys()
            output[identifier] = {
                "name": armor.source["name"],
                "post_fragment_integrity": statistics_dict(
                    [item["fragment_armor_integrity"][identifier] for item in replicates],
                    probability=True,
                ),
                "post_combined_integrity": statistics_dict(
                    [item["combined_armor_integrity"][identifier] for item in replicates],
                    probability=True,
                ),
                "blast_metrics": {
                    key: statistics_dict(
                        [item["armor_blast_metrics"][identifier][key] for item in replicates],
                        probability=(key == "blast_failure_probability" or key.endswith("integrity")),
                    )
                    for key in metric_keys
                },
            }
        return output

    def _aggregate_tree(
        self,
        replicates: list[dict],
        requested_standards: tuple[str, ...],
    ) -> dict:
        node_ids = self.damage_tree.order
        outputs = {
            node_id: statistics_dict(
                [item["tree"]["outputs"][node_id] for item in replicates],
                probability=True,
            )
            for node_id in self.damage_tree.outputs
        }
        return {
            "version": self.damage_tree.version,
            "dependency_assumption": "explicit_per_gate",
            "dependency_models": dict(self.damage_tree.dependency_models),
            "node_probabilities": {
                node_id: statistics_dict(
                    [item["tree"]["node_probabilities"][node_id] for item in replicates],
                    probability=True,
                )
                for node_id in node_ids
            },
            "outputs": outputs,
            "evaluation_standards": {
                standard_id: {
                    output_id: outputs[output_id]
                    for output_id in self.damage_tree.standards[standard_id]
                }
                for standard_id in requested_standards
            },
        }

    @staticmethod
    def _aggregate_summary(replicates: list[dict]) -> dict:
        keys = replicates[0]["summary"].keys()
        return {
            key: statistics_dict([item["summary"][key] for item in replicates])
            for key in keys
        }

    def _provenance(
        self,
        case: dict,
        *,
        replicate_count: int,
        fragment_list_provenance: dict[str, Any] | None = None,
        epistemic_realization: dict | None = None,
        parameter_realization: dict | None = None,
    ) -> dict:
        fragment_list_reference = case.get("warhead", {}).get("fragment_source", {}).get(
            "fragment_list_ref"
        )
        fragment_list = fragment_list_provenance
        if fragment_list is None and fragment_list_reference is not None:
            fragment_list = {
                "reference": fragment_list_reference,
                "status": "not_loaded_or_refused",
            }
        if isinstance(case, FrozenCase) and fragment_list is not None:
            location = next(
                (item for item in case.input_identity.get("locations", [])
                 if item.get("logical_source_id") == DEPENDENCY_LOGICAL_ID),
                None,
            )
            if location is None:
                raise InputValidationError("frozen external input is missing its original source location")
            # Preserve the logical source location rather than publishing an
            # ephemeral snapshot path as the archive's sole replay reference.
            fragment_list = dict(fragment_list)
            fragment_list["reference"] = location["reference"]
            fragment_list["resolved_path"] = location["resolved_path"]
            fragment_list["snapshot_used"] = True
        return {
            "engine_version": ENGINE_VERSION,
            "engine_code_hash": ENGINE_CODE_HASH,
            "schema_version": "1.0.0",
            "schema_tree_hash": SCHEMA_TREE_HASH,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target_hash": self.target_hash,
            "input_target_hash": self.input_target_hash,
            "case_hash": (
                case_identity_hash(case, loaded_dependency=fragment_list)
                if fragment_list is not None and fragment_list.get("content_hash")
                else case_identity_hash(case)
                if fragment_list_reference is None or hasattr(case, "input_identity")
                else content_hash(case)
            ),
            "physics_registry_hash": self.registry.hash,
            "physics_validation_status": self.registry.validation_status,
            "fragment_list": fragment_list,
            "damage_tree_version": self.damage_tree.version,
            "evaluation_standard_ids": list(
                case.get("analysis", {}).get("evaluation_standard_ids", [])
            ),
            "aleatory_seed": case.get("analysis", {}).get("aleatory_seed"),
            "epistemic_sample_id": case.get("analysis", {}).get("epistemic_sample_id"),
            "epistemic_realization_key": content_hash(
                {
                    "target_hash": self.target_hash,
                    "epistemic_sample_id": case.get("analysis", {}).get("epistemic_sample_id"),
                }
            ),
            "epistemic_realization_hash": (
                content_hash(epistemic_realization) if epistemic_realization is not None else None
            ),
            "parameter_realization_hash": (
                content_hash(parameter_realization) if parameter_realization is not None else None
            ),
            "parameter_realization_hash_scope": "nominal_feature_realization_not_all_replicate_parameters",
            "parameter_randomness_scope": "epistemic_parameters_fixed_per_case_non_epistemic_parameters_may_vary_per_replicate",
            "epistemic_sampler_version": "target-parameter-sampler@1.0.0",
            "replicate_count": replicate_count,
        }
