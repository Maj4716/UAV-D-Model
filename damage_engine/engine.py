from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .blast import BlastSimulation, simulate_blast
from .convergence import finite_horizon_probability_bound
from .damage_tree import CompiledDamageTree, compile_damage_tree
from .errors import DamageEngineError, InputValidationError, OutOfDomainError
from .fragments import (
    FragmentListData,
    FragmentSimulation,
    load_fragment_list,
    simulate_fragments,
)
from .geometry import CompiledTarget, compile_target
from .input_identity import DEPENDENCY_LOGICAL_ID, FrozenCase, case_identity_hash
from .normalization import normalize_target
from .online_statistics import (
    OnlineStatisticsStore,
    is_probability_statistic_path,
)
from .parameters import (
    build_epistemic_realization,
    nominal_target_parameters,
    sample_target_realization,
)
from .registry import PhysicsRegistry
from .mechanism_coupling import combine_mechanism_states
from .model_coverage import build_model_coverage_report
from .schema_resources import schema_tree_hash
from .utils import content_hash, source_tree_hash
from .validation import validate_inputs
from .impact import resolve_impact_detonation


ENGINE_VERSION = "0.11.0.dev0"
DOMAIN_ABSENCE_ALPHA = 0.05
DOMAIN_ABSENCE_RATE_THRESHOLD = 0.01
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
    "component.fragment.damage_energy_j": "J",
    "component.fragment.ordinal_projection_max_correction": "1",
    "component.fragment.spatial_damage_cluster_count": "count",
    "component.fragment.spatial_damage_cluster_max_diameter_m": "m",
    "component.fragment.hit_indicator": "1",
    "component.fragment.damage_positive_indicator": "1",
    "component.fragment.saturated_damage_indicator": "1",
    "component.fragment.hit_conditioned_degraded_probability": "1",
    "component.fragment.hit_conditioned_failed_probability": "1",
    "component.blast.distance_m": "m",
    "component.blast.incident_peak_pressure_pa": "Pa",
    "component.blast.incident_impulse_pa_s": "Pa*s",
    "component.blast.effective_peak_pressure_pa": "Pa",
    "component.blast.effective_impulse_pa_s": "Pa*s",
    "component.blast.positive_duration_s": "s",
    "component.blast.positive_phase_shape_exponent": "1",
    "component.blast.arrival_time_s": "s",
    "component.blast.shielding_effective_thickness_m": "m",
    "component.blast.shielding_panel_count": "count",
    "component.blast.ordinal_projection_max_correction": "1",
    "armor.post_fragment_integrity": "1",
    "armor.post_combined_integrity": "1",
    "armor.blast.distance_m": "m",
    "armor.blast.incident_peak_pressure_pa": "Pa",
    "armor.blast.incident_impulse_pa_s": "Pa*s",
    "armor.blast.reflected_peak_pressure_pa": "Pa",
    "armor.blast.reflected_impulse_pa_s": "Pa*s",
    "armor.blast.positive_duration_s": "s",
    "armor.blast.positive_phase_shape_exponent": "1",
    "armor.blast.arrival_time_s": "s",
    "armor.blast.incidence_cosine": "1",
    "armor.blast.blast_failure_probability": "1",
    "armor.blast.blast_failure_realized": "1",
    "armor.blast.blast_failure_sampling_applied": "1",
    "armor.blast.pre_blast_integrity": "1",
    "armor.blast.post_blast_integrity": "1",
    "armor.blast.pre_pulse_shielding_survival": "1",
    "armor.blast.blast_failure_effective_time_s": "s",
    "armor.blast.same_pulse_failure_feedback_applied": "1",
    "summary.total_fragments": "count",
    "summary.total_armor_hits": "count",
    "summary.total_armor_penetrations": "count",
    "summary.total_component_hits": "count",
    "summary.total_component_penetrations": "count",
    "summary.total_existing_hole_reuses": "count",
    "summary.total_out_of_domain_impacts": "count",
    "summary.total_unique_armor_holes": "count",
    "summary.total_perforation_events": "count",
    "summary.total_opening_primitives": "count",
}


def _zero_event_upper_bound(sample_count: int, *, alpha: float = 0.05) -> float:
    """Exact one-sided binomial upper bound after zero observed events."""

    if sample_count <= 0:
        raise InputValidationError(
            "zero-event confidence bound requires a positive sample count"
        )
    if not 0.0 < alpha < 1.0:
        raise InputValidationError("zero-event confidence alpha must lie in (0,1)")
    return float(1.0 - alpha ** (1.0 / sample_count))


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
            "mechanism_states",
            "combined_frechet_bounds",
            "fragment_metrics",
            "blast_metrics",
            "fragment_armor_integrity",
            "fragment_panel_damage_metrics",
            "combined_armor_integrity",
            "armor_blast_metrics",
            "tree",
            "tree_dependency_scenarios",
            "tree_gate_dependency_scenarios",
            "domain_censoring",
            "summary",
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
            deepcopy({
                "replicate_index": index,
                "parameter_realization_hash": replicate["parameter_realization_hash"],
                "system_outputs": tree["outputs"],
                "tree_node_probabilities": tree["node_probabilities"],
                "component_mechanism_states": replicate["mechanism_states"],
                "fragment_metrics": replicate["fragment_metrics"],
                "blast_metrics": replicate["blast_metrics"],
                "combined_frechet_bounds": replicate[
                    "combined_frechet_bounds"
                ],
                "armor": {
                    "fragment_integrity": replicate[
                        "fragment_armor_integrity"
                    ],
                    "combined_integrity": replicate[
                        "combined_armor_integrity"
                    ],
                    "fragment_panel_damage": replicate[
                        "fragment_panel_damage_metrics"
                    ],
                    "blast_metrics": replicate["armor_blast_metrics"],
                },
                "mechanism_dependency_scenarios": replicate[
                    "tree_dependency_scenarios"
                ],
                "gate_dependency_scenarios": replicate[
                    "tree_gate_dependency_scenarios"
                ],
                "domain_censoring": replicate["domain_censoring"],
                "summary": replicate["summary"],
            })
        )
    return {
        "schema_version": "2.0.0",
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
                "damage_energy_j": 0.0,
                "ordinal_projection_max_correction": 0.0,
                "spatial_damage_cluster_count": 0.0,
                "spatial_damage_cluster_max_diameter_m": 0.0,
                "hit_indicator": 0.0,
                "damage_positive_indicator": 0.0,
                "saturated_damage_indicator": 0.0,
                "hit_conditioned_degraded_probability": 0.0,
                "hit_conditioned_failed_probability": 0.0,
            }
            for item in target.components
        },
        armor_integrity={item.object_id: 1.0 for item in target.armor},
        total_fragments=0,
        total_armor_hits=0,
        total_armor_penetrations=0,
        total_component_hits=0,
        total_component_penetrations=0,
        total_existing_hole_reuses=0,
        total_out_of_domain_impacts=0,
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
                "positive_phase_shape_exponent": 0.0,
                "arrival_time_s": 0.0,
                "shielding_effective_thickness_m": 0.0,
                "shielding_panel_count": 0.0,
                "ordinal_projection_max_correction": 0.0,
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
                "positive_phase_shape_exponent": 0.0,
                "arrival_time_s": 0.0,
                "incidence_cosine": 0.0,
                "blast_failure_probability": 0.0,
                "blast_failure_realized": 0.0,
                "blast_failure_sampling_applied": 0.0,
                "pre_blast_integrity": 1.0,
                "post_blast_integrity": 1.0,
                "pre_pulse_shielding_survival": 1.0,
                "blast_failure_effective_time_s": 0.0,
                "same_pulse_failure_feedback_applied": 0.0,
            }
            for item in target.armor
        },
    )


class DamageEngine:
    """Validated orchestration layer for fragment/blast Monte Carlo simulation."""

    def __init__(self, target: dict, registry: PhysicsRegistry):
        self.input_target_hash = content_hash(target)
        self._target_data = normalize_target(deepcopy(target))
        # Reconstruct the registry so the engine owns one coherent model
        # snapshot even if the caller later replaces or mutates its document.
        self.registry = PhysicsRegistry(deepcopy(registry.data))
        self.compiled_target = compile_target(self._target_data)
        self.damage_tree: CompiledDamageTree = compile_damage_tree(
            self._target_data
        )
        self.target_hash = content_hash(self._target_data)
        self._model_coverage = build_model_coverage_report(
            self._target_data,
            self.compiled_target,
            self.registry,
        )

    @property
    def target_data(self) -> dict:
        """Return an owned export of the normalized target snapshot."""

        return deepcopy(self._target_data)

    @property
    def model_coverage(self) -> dict:
        """Return the static coverage audit bound to this model snapshot."""

        return deepcopy(self._model_coverage)

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

        return resolve_impact_detonation(deepcopy(case), self.compiled_target)

    def _run(self, case: dict) -> dict:
        validate_inputs(
            self._target_data,
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
            self._target_data,
            np.random.default_rng(epistemic_seed),
        )
        parameter_features = nominal_target_parameters(self._target_data)
        for object_id, values in epistemic.items():
            for key, value in values.items():
                if value is not None:
                    parameter_features[object_id][key] = value
        statistics_store = OnlineStatisticsStore()
        retained_replicates: list[dict[str, Any]] | None = (
            [] if capture_replicate_trace else None
        )
        parameter_hash_chain = content_hash(
            {"schema_version": "1.0.0", "semantics": "empty_replicate_parameter_chain"}
        )
        audit: list[dict] = []
        unresolved_replicate_count = 0
        for replicate_index, seed in enumerate(replicate_seeds):
            parameter_seed, fragment_seed, blast_seed = seed.spawn(3)
            parameter_rng = np.random.default_rng(parameter_seed)
            fragment_rng = np.random.default_rng(fragment_seed)
            blast_rng = np.random.default_rng(blast_seed)
            armor_failure_quantiles = {
                item.object_id: float(blast_rng.random())
                for item in self.compiled_target.armor
            }
            parameter_realization = sample_target_realization(
                self._target_data,
                parameter_rng,
                epistemic,
            )
            blast_timeline_probe = (
                simulate_blast(
                    case,
                    self.compiled_target,
                    self.registry,
                    parameter_realization=parameter_realization,
                    armor_integrity={
                        item.object_id: 1.0
                        for item in self.compiled_target.armor
                    },
                    armor_failure_quantiles=armor_failure_quantiles,
                )
                if coupling == "sequential"
                and mechanisms == {"fragment", "blast"}
                else None
            )
            armor_blast_damage_events = (
                {
                    armor_id: {
                        "effective_time_s": float(
                            metrics["blast_failure_effective_time_s"]
                        ),
                        "survival_factor": float(
                            1.0 - metrics["blast_failure_realized"]
                        ),
                    }
                    for armor_id, metrics in blast_timeline_probe.armor_metrics.items()
                }
                if blast_timeline_probe is not None
                else None
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
                    armor_blast_damage_events=armor_blast_damage_events,
                )
                if "fragment" in mechanisms
                else _zero_fragment(self.compiled_target)
            )
            if (
                case["analysis"]["fidelity_level"] == "L3"
                and fragment.total_out_of_domain_impacts > 0
            ):
                raise OutOfDomainError(
                    "L3 execution encountered a fragment contact outside the validated penetration domain",
                    context={
                        "reason_code": "l3_unresolved_penetration_event",
                        "replicate_index": replicate_index,
                        "out_of_domain_impact_count": (
                            fragment.total_out_of_domain_impacts
                        ),
                    },
                )
            replicate_has_unresolved_penetration = (
                fragment.total_out_of_domain_impacts > 0
            )
            unresolved_replicate_count += int(
                replicate_has_unresolved_penetration
            )
            blast_integrity = (
                {
                    armor.object_id: fragment.panel_damage_state.integrity(
                        armor.object_id,
                        query_time_s=float(
                            blast_timeline_probe.armor_metrics[armor.object_id][
                                "arrival_time_s"
                            ]
                        ),
                    )
                    for armor in self.compiled_target.armor
                }
                if coupling == "sequential"
                and mechanisms == {"fragment", "blast"}
                and blast_timeline_probe is not None
                and fragment.panel_damage_state is not None
                else {item.object_id: 1.0 for item in self.compiled_target.armor}
            )
            blast = (
                simulate_blast(
                    case,
                    self.compiled_target,
                    self.registry,
                    parameter_realization=parameter_realization,
                    armor_integrity=blast_integrity,
                    panel_damage_state=(
                        fragment.panel_damage_state
                        if coupling == "sequential" and "fragment" in mechanisms
                        else None
                    ),
                    armor_failure_quantiles=armor_failure_quantiles,
                )
                if "blast" in mechanisms
                else _zero_blast(self.compiled_target)
            )
            component_states = {
                item.object_id: tuple(
                    state for state in item.source["damage_states"] if state != "intact"
                )
                for item in self.compiled_target.components
            }
            combination = combine_mechanism_states(
                component_states,
                fragment.component_state_probabilities,
                blast.component_state_probabilities,
                shared_vulnerable_ratio={
                    item.object_id: float(
                        parameter_realization[item.object_id][
                            "vulnerable_area_ratio"
                        ]
                    )
                    for item in self.compiled_target.components
                },
            )
            combined = combination.point_estimate
            combined_frechet_bounds = combination.frechet_bounds
            if "blast" not in mechanisms:
                combined_armor_integrity = fragment.armor_integrity
            elif coupling == "sequential" and "fragment" in mechanisms:
                combined_armor_integrity = {
                    item.object_id: float(
                        np.clip(
                            fragment.armor_integrity[item.object_id]
                            * (1.0 - blast.armor_metrics[item.object_id][
                                "blast_failure_realized"
                            ]),
                            0.0,
                            1.0,
                        )
                    )
                    for item in self.compiled_target.armor
                }
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
            tree_dependency_scenarios = {
                scenario_name: self.damage_tree.evaluate(
                    {
                        "fragment": fragment.component_state_probabilities,
                        "blast": blast.component_state_probabilities,
                        "combined": scenario_states,
                    }
                )
                for scenario_name, scenario_states in combination.scenarios.items()
            }
            tree_gate_dependency_scenarios = (
                self.damage_tree.evaluate_dependency_scenarios(mechanism_states)
            )
            tree = tree_gate_dependency_scenarios["declared"]
            panel_damage_summary = (
                fragment.panel_damage_state.summary()
                if fragment.panel_damage_state is not None
                else {
                    item.object_id: {
                        "hole_count": 0,
                        "perforation_event_count": 0,
                        "opening_primitive_count": 0,
                        "opening_connected_component_count": 0,
                        "covered_area_m2": 0.0,
                        "integrity": 1.0,
                        "union_method": "not_applicable",
                    }
                    for item in self.compiled_target.armor
                }
            )
            replicate_record = {
                    "mechanism_states": mechanism_states,
                    "combined_frechet_bounds": combined_frechet_bounds,
                    "fragment_metrics": fragment.component_metrics,
                    "blast_metrics": blast.component_metrics,
                    "fragment_armor_integrity": fragment.armor_integrity,
                    "fragment_panel_damage_metrics": {
                        armor_id: {
                            "hole_count": float(values["hole_count"]),
                            "perforation_event_count": float(
                                values["perforation_event_count"]
                            ),
                            "opening_primitive_count": float(
                                values["opening_primitive_count"]
                            ),
                            "opening_connected_component_count": float(
                                values["opening_connected_component_count"]
                            ),
                            "covered_area_m2": float(values["covered_area_m2"]),
                        }
                        for armor_id, values in panel_damage_summary.items()
                    },
                    "combined_armor_integrity": combined_armor_integrity,
                    "armor_blast_metrics": blast.armor_metrics,
                    "tree": tree,
                    "tree_dependency_scenarios": tree_dependency_scenarios,
                    "tree_gate_dependency_scenarios": tree_gate_dependency_scenarios,
                    "domain_censoring": {
                        "penetration_affected_replicate": float(
                            replicate_has_unresolved_penetration
                        )
                    },
                    "parameter_realization_hash": content_hash(parameter_realization),
                    "summary": {
                        "total_fragments": float(fragment.total_fragments),
                        "total_armor_hits": float(fragment.total_armor_hits),
                        "total_armor_penetrations": float(fragment.total_armor_penetrations),
                        "total_component_hits": float(fragment.total_component_hits),
                        "total_component_penetrations": float(fragment.total_component_penetrations),
                        "total_existing_hole_reuses": float(
                            fragment.total_existing_hole_reuses
                        ),
                        "total_out_of_domain_impacts": float(
                            fragment.total_out_of_domain_impacts
                        ),
                        "total_unique_armor_holes": float(
                            sum(
                                int(values["opening_connected_component_count"])
                                for values in panel_damage_summary.values()
                            )
                        ),
                        "total_perforation_events": float(
                            sum(
                                int(values["perforation_event_count"])
                                for values in panel_damage_summary.values()
                            )
                        ),
                        "total_opening_primitives": float(
                            sum(
                                int(values["opening_primitive_count"])
                                for values in panel_damage_summary.values()
                            )
                        ),
                    },
                }
            statistics_store.observe_replicate(replicate_record)
            if retained_replicates is not None:
                retained_replicates.append(replicate_record)
            parameter_hash_chain = content_hash(
                {
                    "schema_version": "1.0.0",
                    "previous": parameter_hash_chain,
                    "replicate_index": replicate_index,
                    "parameter_realization_hash": replicate_record[
                        "parameter_realization_hash"
                    ],
                }
            )
            if capture_audit:
                audit.append(
                    {
                        "replicate": replicate_index,
                        "events": fragment.audit,
                    }
                )
            early_stopping_enabled = (
                adaptive_target is not None
                and not physical_tolerances
                and unresolved_replicate_count == 0
                and (
                    "fragment" not in mechanisms
                    or _zero_event_upper_bound(
                        statistics_store.replicate_count,
                        alpha=DOMAIN_ABSENCE_ALPHA,
                    )
                    <= DOMAIN_ABSENCE_RATE_THRESHOLD
                )
            )
            if (
                early_stopping_enabled
                and statistics_store.replicate_count >= 32
                and self._has_converged(
                    sample_count=statistics_store.replicate_count,
                    probability_target=float(adaptive_target),
                    maximum_replicates=maximum_replicates,
                )
            ):
                break
        components = self._aggregate_components(statistics_store)
        armor = self._aggregate_armor(statistics_store)
        requested_standards = tuple(case["analysis"]["evaluation_standard_ids"])
        tree = self._aggregate_tree(statistics_store, requested_standards)
        summary = self._aggregate_summary(statistics_store)
        affected_replicate_fraction = float(
            statistics_store.statistics(
                ("domain_censoring", "penetration_affected_replicate"),
                probability=True,
            )["mean"]
        )
        penetration_domain_probability_bounds = None
        if affected_replicate_fraction > 0.0:
            penetration_domain_probability_bounds = (
                self._build_penetration_domain_censoring_bounds(
                components,
                armor,
                tree,
                affected_replicate_fraction=affected_replicate_fraction,
            )
            )
        unresolved_mean = float(
            summary.get("total_out_of_domain_impacts", {"mean": 0.0})["mean"]
        )
        resolved_mean = float(
            summary.get("total_armor_hits", {"mean": 0.0})["mean"]
            + summary.get("total_component_hits", {"mean": 0.0})["mean"]
        )
        penetration_domain_applicable = "fragment" in mechanisms
        observed_domain_complete = (
            penetration_domain_applicable
            and affected_replicate_fraction == 0.0
        )
        zero_event_upper_confidence_bound = (
            _zero_event_upper_bound(
                statistics_store.replicate_count,
                alpha=DOMAIN_ABSENCE_ALPHA,
            )
            if observed_domain_complete
            else None
        )
        penetration_domain_statistically_complete = (
            observed_domain_complete
            and zero_event_upper_confidence_bound is not None
            and zero_event_upper_confidence_bound
            <= DOMAIN_ABSENCE_RATE_THRESHOLD
        )
        penetration_domain_coverage = {
            "status": (
                "not_applicable"
                if not penetration_domain_applicable
                else "partial_unresolved"
                if affected_replicate_fraction > 0.0
                else "complete"
                if penetration_domain_statistically_complete
                else "observed_no_censoring_not_statistically_complete"
            ),
            "mean_resolved_impact_count": resolved_mean,
            "mean_unresolved_impact_count": unresolved_mean,
            "affected_replicate_count": unresolved_replicate_count,
            "affected_replicate_fraction": affected_replicate_fraction,
            "zero_event_upper_confidence_bound": (
                zero_event_upper_confidence_bound
            ),
            "zero_event_confidence_level": (
                1.0 - DOMAIN_ABSENCE_ALPHA
                if zero_event_upper_confidence_bound is not None
                else None
            ),
            "domain_event_rate_acceptance_threshold": (
                DOMAIN_ABSENCE_RATE_THRESHOLD
                if penetration_domain_applicable
                else None
            ),
            "statistically_demonstrated_complete": (
                bool(penetration_domain_statistically_complete)
                if penetration_domain_applicable
                else None
            ),
            "unresolved_impact_fraction": (
                unresolved_mean / (resolved_mean + unresolved_mean)
                if resolved_mean + unresolved_mean > 0.0
                else 0.0
            ),
            "unresolved_event_handling": (
                "terminate_fragment_path; publish bounded probability intervals over affected replicates"
            ),
            "point_estimate_scope": (
                "observed-path estimate; component failure values are lower endpoints and arbitrary tree outputs receive two-sided worst-case bounds"
            ),
            "probability_bound_method": (
                "affected-replicate worst-case bounded-outcome interval"
                if affected_replicate_fraction > 0.0
                else "not_required"
            ),
            "probability_bounds": penetration_domain_probability_bounds,
        }
        warnings = []
        if self.registry.validation_status != "validated":
            warnings.append("Physics registry is not validated; outputs are not real-world certified.")
        if "fragment" in mechanisms:
            warnings.append(
                "Armor openings use an analytic coplanar circular-union model. A fragment may "
                "reuse only a geometrically clear opening, and blast leakage is evaluated at "
                "the local ray/panel intersection rather than smeared across the whole panel. "
                "Opening reuse requires a strictly earlier formation event. Instantaneous "
                "hole formation, crack growth, "
                "plugging, spall, ricochet, and fragment breakup are outside this model."
            )
            if float(summary.get("total_out_of_domain_impacts", {"mean": 0.0})["mean"]) > 0.0:
                warnings.append(
                    "Some fragment contacts lay outside the registered penetration-model "
                    "domain. Those fragment paths were terminated as unresolved events; "
                    "probability outputs include affected-replicate worst-case bounds and "
                    "the run is not admitted as converged."
                )
            elif not penetration_domain_statistically_complete:
                warnings.append(
                    "No unresolved penetration event was observed, but the exact one-sided "
                    "zero-event confidence bound remains above the preregistered 1% domain-event "
                    "rate threshold; this run is observed-domain-complete, not statistically "
                    "demonstrated domain-complete."
                )
        if any(
            item.response.get("layer_stack_id") is not None
            for item in (*self.compiled_target.armor, *self.compiled_target.components)
        ):
            warnings.append(
                "Layer stacks are resolved sequentially with declared thickness fractions; "
                "layer-interface delamination and material interaction require external validation."
            )
        if coupling == "none" and mechanisms == {"fragment", "blast"}:
            warnings.append("Fragment/blast coupling is disabled; combined probabilities use conditional independence.")
        elif coupling == "sequential" and mechanisms == {"fragment", "blast"}:
            warnings.append(
                "Time-ordered two-way armor coupling is applied: fragment openings affect the "
                "blast front and positive phase at their formation times, while armor blast "
                "failure changes later fragment resistance after the pulse ends. Component "
                "fragment/blast state probabilities use conditional independence within the "
                "minimum shared vulnerable support as an explicit sensitivity point."
            )
        if adaptive_target is not None and not physical_tolerances:
            warnings.append(
                "Adaptive convergence uses a finite-horizon simultaneous Hoeffding bound for "
                "bounded probability labels; continuous physical metrics are not included."
            )
        if physical_tolerances:
            warnings.append(
                "Continuous physical-metric intervals are fixed-horizon pointwise approximations; "
                "they are assessed only at the preregistered replicate limit and do not authorize "
                "adaptive early stopping."
            )
        probability_bound = (
            finite_horizon_probability_bound(
                sample_count=statistics_store.replicate_count,
                series_count=self._probability_series_count(),
                maximum_looks=maximum_replicates,
            )
            if adaptive_target is not None
            else None
        )
        achieved_half_width = (
            probability_bound.half_width if probability_bound is not None else None
        )
        physical_convergence = self._physical_convergence_report(
            statistics_store, physical_tolerances
        )
        probability_target_met = (
            achieved_half_width is not None and achieved_half_width <= float(adaptive_target)
            if adaptive_target is not None
            else None
        )
        if adaptive_target is not None and affected_replicate_fraction > 0.0:
            probability_target_met = False
        all_targets_met = (
            statistics_store.replicate_count >= 32
            and (probability_target_met is not False)
            and physical_convergence["target_met"]
            and (
                not penetration_domain_applicable
                or penetration_domain_statistically_complete
            )
            if adaptive_enabled
            else None
        )
        convergence = {
            "mode": (
                "fixed_horizon_physical_assessment"
                if physical_tolerances
                else "adaptive_probability"
                if adaptive_target is not None
                else "fixed_replicates"
            ),
            "early_stopping_enabled": bool(
                adaptive_target is not None
                and not physical_tolerances
                and (
                    not penetration_domain_applicable
                    or penetration_domain_statistically_complete
                )
            ),
            "probability_scope": [
                "damage_tree.outputs",
                "component.damage_states.fragment",
                "component.damage_states.blast",
                "component.damage_states.combined",
                "component.damage_states.combined.frechet_union_bounds",
                "damage_tree.mechanism_dependency_scenarios",
                "damage_tree.gate_dependency_scenarios",
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
            "interval_method": (
                probability_bound.interval_method
                if probability_bound is not None
                else None
            ),
            "coverage_status": (
                "domain_censored_worst_case_interval"
                if affected_replicate_fraction > 0.0
                else probability_bound.coverage_status
                if probability_bound is not None
                else "fixed_horizon_only"
            ),
            "penetration_domain_complete": (
                bool(penetration_domain_statistically_complete)
                if penetration_domain_applicable
                else None
            ),
            "penetration_domain_observed_complete": (
                bool(observed_domain_complete)
                if penetration_domain_applicable
                else None
            ),
            "domain_censoring_blocks_convergence": (
                affected_replicate_fraction > 0.0
            ),
            "domain_evidence_blocks_scientific_completion": (
                penetration_domain_applicable
                and not penetration_domain_statistically_complete
            ),
            "familywise_alpha": (
                probability_bound.familywise_alpha
                if probability_bound is not None
                else None
            ),
            "monitored_probability_series_count": self._probability_series_count(),
            "maximum_planned_looks": maximum_replicates,
            "probability_target_met": probability_target_met,
            "target_met": all_targets_met,
            "reached_replicate_limit": (
                statistics_store.replicate_count == maximum_replicates
            ),
        }
        payload = {
            "status": "ok",
            "scientific_completion_status": (
                "not_applicable_no_fragment"
                if not penetration_domain_applicable
                else "partial_domain_censored"
                if affected_replicate_fraction > 0.0
                else "complete_model_domain"
                if penetration_domain_statistically_complete
                else "observed_domain_complete_not_statistically_demonstrated"
            ),
            "case_id": case["case_id"],
            "target_id": self.target_data["target_id"],
            "component_order": list(self.compiled_target.component_order),
            "replicate_count": statistics_store.replicate_count,
            "mechanisms": sorted(mechanisms),
            "coupling_model": coupling,
            "coupling_semantics": (
                "time_ordered_two_way_armor_state_with_post_pulse_blast_failure"
                if coupling == "sequential"
                and mechanisms == {"fragment", "blast"}
                else "uncoupled_mechanisms"
                if coupling == "none"
                else "single_mechanism_not_applicable"
            ),
            "combined_dependency_assumption": (
                "conditional_independence_within_minimum_shared_support_sensitivity"
            ),
            "combined_dependency_identification": "not_identified",
            "combined_dependency_bounds": "frechet_union",
            "combined_dependency_scenarios": [
                "frechet_lower",
                "conditional_independence",
                "frechet_upper",
            ],
            "combined_dependency_scenario_semantics": {
                "frechet_lower": "assumption_free_union_lower_bound",
                "conditional_independence": (
                    "conditional_independence_within_minimum_shared_support_sensitivity"
                ),
                "frechet_upper": "assumption_free_union_upper_bound",
            },
            "blast_load_semantics": {
                "incident_positive_duration_s": "registered_free_field_positive_phase",
                "transmitted_positive_duration_s": (
                    "equivalent_duration_preserving_peak_impulse_representability"
                ),
                "transmitted_peak_pressure_pa": (
                    "maximum_of_event_partitioned_local_transmitted_pressure_history"
                ),
            },
            "convergence": convergence,
            "parameter_realization": parameter_features,
            "armor": armor,
            "components": components,
            "damage_tree": tree,
            "model_coverage": self.model_coverage,
            "summary": summary,
            "penetration_domain_coverage": penetration_domain_coverage,
            "replicate_statistics_storage": {
                "mode": (
                    "online_sufficient_moments_plus_raw_trace"
                    if retained_replicates is not None
                    else "online_sufficient_moments"
                ),
                "raw_trace_retained": retained_replicates is not None,
                "parameter_realization_hash_chain": parameter_hash_chain,
                "hash_chain_schema_version": "1.0.0",
            },
            "warnings": warnings,
            "provenance": self._provenance(
                case,
                replicate_count=statistics_store.replicate_count,
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
            if retained_replicates is None:
                raise RuntimeError("replicate trace retention invariant failed")
            payload["replicate_trace"] = build_replicate_trace(retained_replicates)
        return payload

    def _has_converged(
        self,
        *,
        sample_count: int,
        probability_target: float,
        maximum_replicates: int,
    ) -> bool:
        if sample_count < 32:
            return False
        bound = finite_horizon_probability_bound(
            sample_count=sample_count,
            series_count=self._probability_series_count(),
            maximum_looks=maximum_replicates,
        )
        return bound.half_width <= probability_target

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

    def _physical_metric_statistics(
        self,
        statistics_store: OnlineStatisticsStore,
        selector: str,
    ) -> list[tuple[str, dict[str, Any]]]:
        probability_metric = is_probability_statistic_path((selector,))
        if selector.startswith("component.fragment."):
            key = selector.removeprefix("component.fragment.")
            return [
                (
                    component.object_id,
                    statistics_store.statistics(
                        ("fragment_metrics", component.object_id, key),
                        probability=probability_metric,
                    ),
                )
                for component in self.compiled_target.components
            ]
        if selector.startswith("component.blast."):
            key = selector.removeprefix("component.blast.")
            return [
                (
                    component.object_id,
                    statistics_store.statistics(
                        ("blast_metrics", component.object_id, key),
                        probability=probability_metric,
                    ),
                )
                for component in self.compiled_target.components
            ]
        if selector == "armor.post_fragment_integrity":
            return [
                (
                    armor.object_id,
                    statistics_store.statistics(
                        ("fragment_armor_integrity", armor.object_id),
                        probability=True,
                    ),
                )
                for armor in self.compiled_target.armor
            ]
        if selector == "armor.post_combined_integrity":
            return [
                (
                    armor.object_id,
                    statistics_store.statistics(
                        ("combined_armor_integrity", armor.object_id),
                        probability=True,
                    ),
                )
                for armor in self.compiled_target.armor
            ]
        if selector.startswith("armor.blast."):
            key = selector.removeprefix("armor.blast.")
            return [
                (
                    armor.object_id,
                    statistics_store.statistics(
                        ("armor_blast_metrics", armor.object_id, key),
                        probability=probability_metric,
                    ),
                )
                for armor in self.compiled_target.armor
            ]
        if selector.startswith("summary."):
            key = selector.removeprefix("summary.")
            return [
                (
                    "target",
                    statistics_store.statistics(("summary", key)),
                )
            ]
        raise InputValidationError(
            "unsupported adaptive physical metric selector",
            context={"selector": selector},
        )

    def _physical_convergence_report(
        self,
        statistics_store: OnlineStatisticsStore,
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
            series_values = self._physical_metric_statistics(
                statistics_store, selector
            )
            for series_id, stats in series_values:
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
                    "sample_count": int(stats["sample_count"]),
                    "allowed_half_width": float(allowed),
                    "tolerance_ratio": (
                        float(ratio) if np.isfinite(ratio) else None
                    ),
                    "target_met": bool(met),
                    "coverage_status": (
                        "fixed_horizon_pointwise_approximation_not_simultaneous"
                    ),
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
                "coverage_status": (
                    "fixed_horizon_pointwise_approximation_not_simultaneous"
                ),
            }
        return {
            "target_met": unmet_series_count == 0,
            "unmet_series_count": unmet_series_count,
            "worst_series": worst_series,
            "selectors": selector_results,
        }

    def _probability_series_count(self) -> int:
        component_state_count = sum(
            len([state for state in component.source["damage_states"] if state != "intact"])
            for component in self.compiled_target.components
        )
        # Five component series per state: fragment, blast, independence point,
        # and the two Fréchet union limits.  Three mechanism-dependence tree
        # scenarios are always monitored.  Gate-dependence alternatives enter
        # the family only when their global override is structurally feasible.
        gate_scenario_count = 1 + sum(
            self.damage_tree.dependency_override_compatibility(model)[
                "compatible"
            ]
            for model in ("independent", "fully_dependent")
        )
        return 5 * component_state_count + (
            3 + gate_scenario_count
        ) * len(self.damage_tree.outputs)

    @staticmethod
    def _build_penetration_domain_censoring_bounds(
        components: dict,
        armor: dict,
        tree: dict,
        *,
        affected_replicate_fraction: float,
    ) -> dict:
        """Build separate finite bounds without changing legacy statistic records."""

        fraction = float(np.clip(affected_replicate_fraction, 0.0, 1.0))

        def bound(statistics: dict, *, direction: str) -> dict | None:
            if not isinstance(statistics, dict):
                return None
            mean = statistics.get("mean")
            if not isinstance(mean, (int, float)) or isinstance(mean, bool):
                return None
            point = float(mean)
            if direction == "failure_monotone_increasing":
                lower, upper = point, min(1.0, point + fraction)
            elif direction == "integrity_monotone_decreasing":
                lower, upper = max(0.0, point - fraction), point
            else:
                lower = max(0.0, point - fraction)
                upper = min(1.0, point + fraction)
            return {
                "point": point,
                "lower": lower,
                "upper": upper,
                "affected_replicate_fraction": fraction,
                "method": "affected_replicate_worst_case_bounded_outcome",
                "monotonicity": direction,
            }

        component_bounds: dict[str, dict] = {}
        for component_id, component in components.items():
            state_results = component.get("damage_states", {})
            component_bounds[component_id] = {}
            for mechanism in ("fragment", "combined"):
                mechanism_bounds: dict[str, dict] = {}
                for state, statistics in state_results.get(mechanism, {}).items():
                    state_bound = bound(
                        statistics,
                        direction="failure_monotone_increasing",
                    )
                    if state_bound is not None:
                        mechanism_bounds[state] = state_bound
                component_bounds[component_id][mechanism] = mechanism_bounds

        armor_bounds = {
            armor_id: {
                "post_fragment_integrity": bound(
                    armor_result.get("post_fragment_integrity", {}),
                    direction="integrity_monotone_decreasing",
                ),
                "post_combined_integrity": bound(
                    armor_result.get("post_combined_integrity", {}),
                    direction="integrity_monotone_decreasing",
                ),
            }
            for armor_id, armor_result in armor.items()
        }
        tree_output_bounds = {
            output_id: bound(
                statistics,
                direction="arbitrary_bounded_tree_output",
            )
            for output_id, statistics in tree.get("outputs", {}).items()
        }
        mechanism_scenario_bounds = {
            scenario_name: {
                output_id: bound(
                    statistics,
                    direction="arbitrary_bounded_tree_output",
                )
                for output_id, statistics in scenario.items()
            }
            for scenario_name, scenario in tree.get(
                "mechanism_dependency_sensitivity", {}
            ).get("scenarios", {}).items()
        }
        gate_scenario_bounds: dict[str, dict] = {}
        for scenario_name, scenario in tree.get(
            "gate_dependency_sensitivity", {}
        ).get("scenarios", {}).items():
            gate_scenario_bounds[scenario_name] = {
                "status": scenario.get("status"),
                "outputs": {
                    output_id: (
                        bound(
                            statistics,
                            direction="arbitrary_bounded_tree_output",
                        )
                        if statistics is not None
                        else None
                    )
                    for output_id, statistics in scenario.get("outputs", {}).items()
                },
            }
        return {
            "method": "affected_replicate_worst_case_bounded_outcome",
            "affected_replicate_fraction": fraction,
            "components": component_bounds,
            "armor_integrity": armor_bounds,
            "damage_tree_outputs": tree_output_bounds,
            "damage_tree_mechanism_scenarios": mechanism_scenario_bounds,
            "damage_tree_gate_scenarios": gate_scenario_bounds,
        }

    def _aggregate_components(
        self,
        statistics_store: OnlineStatisticsStore,
    ) -> dict:
        output: dict[str, dict] = {}
        for component in self.compiled_target.components:
            identifier = component.object_id
            state_result: dict[str, dict] = {}
            for mechanism in ("fragment", "blast", "combined"):
                state_result[mechanism] = {}
                for state in component.source["damage_states"]:
                    if state == "intact":
                        continue
                    statistics = statistics_store.statistics(
                        ("mechanism_states", mechanism, identifier, state),
                        probability=True,
                    )
                    if mechanism == "combined":
                        statistics["frechet_union_lower_bound"] = statistics_store.statistics(
                            ("combined_frechet_bounds", identifier, state, "lower"),
                            probability=True,
                        )
                        statistics["frechet_union_upper_bound"] = statistics_store.statistics(
                            ("combined_frechet_bounds", identifier, state, "upper"),
                            probability=True,
                        )
                    state_result[mechanism][state] = statistics
            fragment_keys = statistics_store.child_keys(
                ("fragment_metrics", identifier)
            )
            fragment_metrics = {
                key: statistics_store.statistics(
                    ("fragment_metrics", identifier, key),
                    probability=is_probability_statistic_path(
                        ("fragment_metrics", identifier, key)
                    ),
                )
                for key in fragment_keys
            }
            blast_keys = statistics_store.child_keys(("blast_metrics", identifier))
            blast_metrics = {
                key: statistics_store.statistics(
                    ("blast_metrics", identifier, key),
                    probability=is_probability_statistic_path(
                        ("blast_metrics", identifier, key)
                    ),
                )
                for key in blast_keys
            }
            output[identifier] = {
                "name": component.source["name"],
                "component_type": component.source["component_type"],
                "fragment_damage_accumulation_model": str(
                    self.registry.fragment_response(
                        component.response["fragment_response_model_id"]
                    ).get("damage_accumulation_model", "independent_hits")
                ),
                "layer_stack_id": component.response.get("layer_stack_id"),
                "pressure_impulse_surface_id": component.response[
                    "blast_response"
                ].get("pressure_impulse_surface_id"),
                "pressure_impulse_surface_status": (
                    "not_declared"
                    if component.response["blast_response"].get(
                        "pressure_impulse_surface_id"
                    )
                    is None
                    else "declared_but_not_resolved_by_registry"
                ),
                "damage_states": state_result,
                "fragment_metrics": fragment_metrics,
                "blast_metrics": blast_metrics,
            }
        return output

    def _aggregate_armor(
        self,
        statistics_store: OnlineStatisticsStore,
    ) -> dict:
        output: dict[str, dict] = {}
        for armor in self.compiled_target.armor:
            identifier = armor.object_id
            metric_keys = statistics_store.child_keys(
                ("armor_blast_metrics", identifier)
            )
            output[identifier] = {
                "name": armor.source["name"],
                "layer_stack_id": armor.response.get("layer_stack_id"),
                "post_fragment_integrity": statistics_store.statistics(
                    ("fragment_armor_integrity", identifier),
                    probability=True,
                ),
                "post_combined_integrity": statistics_store.statistics(
                    ("combined_armor_integrity", identifier),
                    probability=True,
                ),
                "fragment_panel_damage": {
                    "hole_count": statistics_store.statistics(
                        (
                            "fragment_panel_damage_metrics",
                            identifier,
                            "hole_count",
                        )
                    ),
                    "covered_area_m2": statistics_store.statistics(
                        (
                            "fragment_panel_damage_metrics",
                            identifier,
                            "covered_area_m2",
                        )
                    ),
                    "perforation_event_count": statistics_store.statistics(
                        (
                            "fragment_panel_damage_metrics",
                            identifier,
                            "perforation_event_count",
                        )
                    ),
                    "opening_primitive_count": statistics_store.statistics(
                        (
                            "fragment_panel_damage_metrics",
                            identifier,
                            "opening_primitive_count",
                        )
                    ),
                    "opening_connected_component_count": statistics_store.statistics(
                        (
                            "fragment_panel_damage_metrics",
                            identifier,
                            "opening_connected_component_count",
                        )
                    ),
                    "union_method": (
                        "analytic_coplanar_circle_union"
                    ),
                },
                "blast_metrics": {
                    key: statistics_store.statistics(
                        ("armor_blast_metrics", identifier, key),
                        probability=is_probability_statistic_path(
                            ("armor_blast_metrics", identifier, key)
                        ),
                    )
                    for key in metric_keys
                },
            }
        return output

    def _aggregate_tree(
        self,
        statistics_store: OnlineStatisticsStore,
        requested_standards: tuple[str, ...],
    ) -> dict:
        node_ids = self.damage_tree.order
        outputs = {
            node_id: statistics_store.statistics(
                ("tree", "outputs", node_id),
                probability=True,
            )
            for node_id in self.damage_tree.outputs
        }
        scenario_names = (
            "frechet_lower",
            "conditional_independence",
            "frechet_upper",
        )
        scenario_outputs = {
            scenario_name: {
                node_id: statistics_store.statistics(
                    (
                        "tree_dependency_scenarios",
                        scenario_name,
                        "outputs",
                        node_id,
                    ),
                    probability=True,
                )
                for node_id in self.damage_tree.outputs
            }
            for scenario_name in scenario_names
        }
        gate_scenario_reports: dict[str, dict] = {}
        for scenario_name in ("declared", "independent", "fully_dependent"):
            compatibility = (
                {"compatible": True, "conflicts": []}
                if scenario_name == "declared"
                else self.damage_tree.dependency_override_compatibility(
                    scenario_name
                )
            )
            scenario_models = (
                dict(self.damage_tree.dependency_models)
                if scenario_name == "declared"
                else {
                    node_id: scenario_name
                    for node_id in self.damage_tree.dependency_models
                }
            )
            if compatibility["compatible"]:
                gate_scenario_reports[scenario_name] = {
                    "status": "available",
                    "outputs": {
                        node_id: statistics_store.statistics(
                            (
                                "tree_gate_dependency_scenarios",
                                scenario_name,
                                "outputs",
                                node_id,
                            ),
                            probability=True,
                        )
                        for node_id in self.damage_tree.outputs
                    },
                    "dependency_models": scenario_models,
                }
            else:
                gate_scenario_reports[scenario_name] = {
                    "status": "unavailable",
                    "reason_code": (
                        "dependency_override_structurally_infeasible"
                    ),
                    "reason": (
                        "the global dependency model conflicts with known "
                        "leaf-event nesting, shared causes, or polarity"
                    ),
                    "outputs": {
                        node_id: None for node_id in self.damage_tree.outputs
                    },
                    "dependency_models": scenario_models,
                    "structural_conflicts": compatibility["conflicts"],
                }
        tree_is_monotone = all(
            node["node_type"] != "not" for node in self.damage_tree.nodes.values()
        )
        return {
            "version": self.damage_tree.version,
            "dependency_assumption": "explicit_per_gate",
            "dependency_models": dict(self.damage_tree.dependency_models),
            "node_probabilities": {
                node_id: statistics_store.statistics(
                    ("tree", "node_probabilities", node_id),
                    probability=True,
                )
                for node_id in node_ids
            },
            "outputs": outputs,
            "mechanism_dependency_sensitivity": {
                "identification_status": "not_identified",
                "point_estimate_assumption": (
                    "conditional_independence_within_minimum_shared_support_sensitivity"
                ),
                "scenarios": scenario_outputs,
                "tree_is_monotone_in_leaf_probabilities": tree_is_monotone,
                "frechet_scenarios_are_output_bounds": tree_is_monotone,
            },
            "gate_dependency_sensitivity": {
                "identification_status": "not_identified",
                "point_estimate_assumption": "declared",
                "scenarios": gate_scenario_reports,
                "available_scenarios": [
                    name
                    for name, report in gate_scenario_reports.items()
                    if report["status"] == "available"
                ],
            },
            "evaluation_standards": {
                standard_id: {
                    output_id: outputs[output_id]
                    for output_id in self.damage_tree.standards[standard_id]
                }
                for standard_id in requested_standards
            },
        }

    @staticmethod
    def _aggregate_summary(statistics_store: OnlineStatisticsStore) -> dict:
        keys = statistics_store.child_keys(("summary",))
        return {
            key: statistics_store.statistics(("summary", key))
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
        validation_receipt = self.registry.data.get("validation_receipt")
        validation_receipt_summary = (
            {
                "receipt_id": validation_receipt["receipt_id"],
                "validated_at": validation_receipt["validated_at"],
                "evidence_hash": validation_receipt["evidence_hash"],
                "validated_model_payload_hash": validation_receipt[
                    "validated_model_payload_hash"
                ],
                "domain": validation_receipt["domain"],
            }
            if validation_receipt is not None
            else None
        )
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
            "physics_validation_receipt": validation_receipt_summary,
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
