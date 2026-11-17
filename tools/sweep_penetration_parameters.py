"""Exploratory repeated-sampling sweep for fragment-response parameters.

This tool is intentionally a software-identifiability aid, not a physical
calibration procedure.  It evaluates one fixed encounter repeatedly, ranks
explicit candidate parameter sets with a disclosed heuristic, and preserves
all candidate results so that the selected configuration is reviewable.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from damage_engine.engine import ENGINE_CODE_HASH, DamageEngine
from damage_engine.registry import PhysicsRegistry
from damage_engine.schema_resources import schema_tree_hash
from damage_engine.utils import (
    content_hash,
    file_hash,
    load_json,
    source_tree_hash,
)


def source_identity_snapshot(paths: dict[str, Path]) -> dict[str, str]:
    """Capture executable and input identities before or after a sweep."""

    identity = {
        "engine_code_hash": content_hash(
            {
                "python_source_tree_hash": source_tree_hash(
                    REPOSITORY_ROOT / "damage_engine"
                ),
                "schema_tree_hash": schema_tree_hash(),
            }
        ),
        "sweep_tool_hash": file_hash(Path(__file__).resolve()),
    }
    identity.update(
        {
            f"{name}_hash": file_hash(path)
            for name, path in sorted(paths.items())
        }
    )
    return identity


def admission_gate_report(
    metrics: dict[str, Any],
    declared_gates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate hard domain/energy gates plus declared distribution gates."""

    checks: dict[str, bool] = {
        "scientific_completion_status": (
            metrics.get("scientific_completion_status") == "complete_model_domain"
        ),
        "penetration_affected_replicate_fraction": (
            float(metrics.get("penetration_affected_replicate_fraction", 1.0))
            == 0.0
        ),
        "energy_closure_max_abs_j": (
            math.isfinite(
                float(metrics.get("energy_closure_max_abs_j", math.inf))
            )
            and float(metrics.get("energy_closure_max_abs_j", math.inf)) <= 1.0e-8
        ),
    }
    gates = declared_gates or {}
    for gate_name, metric_name in (
        ("armor_penetration_rate_range", "armor_penetration_rate"),
        ("component_penetration_rate_range", "component_penetration_rate"),
        ("damage_positive_given_hit_range", "damage_positive_given_hit"),
    ):
        if gate_name in gates:
            lower, upper = (float(value) for value in gates[gate_name])
            value = float(metrics.get(metric_name, math.nan))
            checks[gate_name] = (
                0.0 <= lower <= upper <= 1.0
                and math.isfinite(value)
                and lower <= value <= upper
            )
    for gate_name, metric_name in (
        ("saturated_damage_fraction_max", "saturated_damage_fraction"),
        (
            "ordinal_projection_max_correction_max",
            "ordinal_projection_max_correction",
        ),
        ("out_of_domain_impact_fraction_max", "out_of_domain_impact_fraction"),
    ):
        if gate_name in gates:
            value = float(metrics.get(metric_name, math.inf))
            checks[gate_name] = math.isfinite(value) and value <= float(
                gates[gate_name]
            )
    if "energy_closure_max_abs_j" in gates:
        value = float(metrics.get("energy_closure_max_abs_j", math.inf))
        checks["energy_closure_max_abs_j"] = (
            math.isfinite(value)
            and value <= float(gates["energy_closure_max_abs_j"])
        )
    if "expected_hit_component_observations_min" in gates:
        value = float(metrics.get("expected_hit_component_observations", -math.inf))
        checks["expected_hit_component_observations_min"] = (
            math.isfinite(value)
            and value >= float(gates["expected_hit_component_observations_min"])
        )
    failed = [name for name, passed in checks.items() if not passed]
    return {"passed": not failed, "checks": checks, "failed_gates": failed}


def is_formally_admissible(
    metrics: dict[str, Any],
    declared_gates: dict[str, Any] | None = None,
) -> bool:
    return bool(admission_gate_report(metrics, declared_gates)["passed"])


def _recursive_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _recursive_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_candidate_plan(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    document = load_json(path)
    objective = dict(document["objective"])
    defaults = dict(document.get("candidate_defaults", {}))
    candidates: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for raw in document["candidates"]:
        candidate_id = str(raw["id"])
        if candidate_id in identifiers:
            raise ValueError(f"duplicate candidate id: {candidate_id}")
        identifiers.add(candidate_id)
        parameters = dict(raw.get("parameters", {}))
        if bool(raw.get("inherit_defaults", True)):
            parameters = _recursive_merge(defaults, parameters)
        candidates.append({"id": candidate_id, "parameters": parameters})
    if not candidates:
        raise ValueError("candidate plan must contain at least one candidate")
    return objective, candidates


def apply_candidate(
    registry_data: dict[str, Any],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    updated = deepcopy(registry_data)
    response_models = updated["fragment_response_models"]
    armor = response_models["demo_armor_response"]
    component = response_models["demo_component_response"]
    armor["ballistic_limit"].update(parameters.get("armor_ballistic", {}))
    component["ballistic_limit"].update(parameters.get("component_ballistic", {}))
    if "armor_hole_area_multiplier" in parameters:
        armor["hole_area_multiplier"] = float(
            parameters["armor_hole_area_multiplier"]
        )
    if "component_damage_energy_model" in parameters:
        component["damage_energy_model"] = deepcopy(
            parameters["component_damage_energy_model"]
        )
    if "component_damage_accumulation_model" in parameters:
        component["damage_accumulation_model"] = str(
            parameters["component_damage_accumulation_model"]
        )
    if "component_damage_correlation_length_m" in parameters:
        component["damage_correlation_length_m"] = float(
            parameters["component_damage_correlation_length_m"]
        )
    for state, threshold in parameters.get("component_damage_thresholds_j", {}).items():
        component["damage_states"][state]["damage_energy_50_j"] = float(threshold)
    return updated


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    summary = result["summary"]
    replicate_count = int(result.get("replicate_count", 1))
    domain_coverage = result["penetration_domain_coverage"]
    affected_replicate_fraction = float(
        domain_coverage["affected_replicate_fraction"]
    )
    scientific_completion_status = str(result["scientific_completion_status"])
    armor_hits = float(summary["total_armor_hits"]["mean"])
    armor_penetrations = float(summary["total_armor_penetrations"]["mean"])
    component_hits = float(summary["total_component_hits"]["mean"])
    component_penetrations = float(
        summary["total_component_penetrations"]["mean"]
    )
    existing_hole_reuses = float(
        summary.get("total_existing_hole_reuses", {"mean": 0.0})["mean"]
    )
    out_of_domain_impacts = float(
        summary.get("total_out_of_domain_impacts", {"mean": 0.0})["mean"]
    )

    hit_components: list[dict[str, Any]] = []
    ordinal_violations = 0
    ordinal_projection_max_correction = 0.0
    hit_probability_mass = 0.0
    damage_positive_mass = 0.0
    saturated_mass = 0.0
    degraded_given_hit_numerator = 0.0
    failed_given_hit_numerator = 0.0
    for component_id, component in result["components"].items():
        fragment_metrics = component["fragment_metrics"]
        hits = float(component["fragment_metrics"]["hits"]["mean"])
        degraded = float(
            component["damage_states"]["fragment"]["degraded"]["mean"]
        )
        failed = float(component["damage_states"]["fragment"]["failed"]["mean"])
        correction = float(
            fragment_metrics.get(
                "ordinal_projection_max_correction", {"mean": 0.0}
            )["mean"]
        )
        ordinal_projection_max_correction = max(
            ordinal_projection_max_correction,
            correction,
        )
        if correction > 1.0e-12:
            ordinal_violations += 1
        hit_mass = float(
            fragment_metrics.get(
                "hit_indicator", {"mean": float(hits > 0.0)}
            )["mean"]
        )
        positive_mass = float(
            fragment_metrics.get(
                "damage_positive_indicator",
                {"mean": float(hits > 0.0 and degraded > 0.01)},
            )["mean"]
        )
        saturation_mass = float(
            fragment_metrics.get(
                "saturated_damage_indicator",
                {
                    "mean": float(
                        hits > 0.0 and (degraded > 0.95 or failed > 0.95)
                    )
                },
            )["mean"]
        )
        degraded_numerator = float(
            fragment_metrics.get(
                "hit_conditioned_degraded_probability",
                {"mean": degraded if hits > 0.0 else 0.0},
            )["mean"]
        )
        failed_numerator = float(
            fragment_metrics.get(
                "hit_conditioned_failed_probability",
                {"mean": failed if hits > 0.0 else 0.0},
            )["mean"]
        )
        hit_probability_mass += hit_mass
        damage_positive_mass += positive_mass
        saturated_mass += saturation_mass
        degraded_given_hit_numerator += degraded_numerator
        failed_given_hit_numerator += failed_numerator
        if hits > 0.0:
            hit_components.append(
                {
                    "component_id": component_id,
                    "hits": hits,
                    "degraded": degraded,
                    "failed": failed,
                }
            )

    hit_count = len(hit_components)
    positive_count = sum(item["degraded"] > 0.01 for item in hit_components)
    component_count = len(result["components"])
    return {
        "mean_armor_hits": armor_hits,
        "mean_armor_penetrations": armor_penetrations,
        "armor_penetration_rate": _safe_rate(armor_penetrations, armor_hits),
        "mean_existing_hole_reuses": existing_hole_reuses,
        "existing_hole_reuse_per_armor_hit": _safe_rate(
            existing_hole_reuses,
            armor_hits,
        ),
        "mean_out_of_domain_impacts": out_of_domain_impacts,
        "out_of_domain_impact_fraction": _safe_rate(
            out_of_domain_impacts,
            armor_hits + component_hits + out_of_domain_impacts,
        ),
        "penetration_affected_replicate_fraction": affected_replicate_fraction,
        "scientific_completion_status": scientific_completion_status,
        "mean_component_hits": component_hits,
        "mean_component_penetrations": component_penetrations,
        "component_penetration_rate": _safe_rate(
            component_penetrations,
            component_hits,
        ),
        "component_count": component_count,
        "hit_component_count": hit_count,
        "hit_component_fraction": _safe_rate(hit_count, component_count),
        "damage_positive_component_count": positive_count,
        "damage_positive_given_hit": _safe_rate(
            damage_positive_mass,
            hit_probability_mass,
        ),
        "zero_damage_given_hit_fraction": _safe_rate(
            hit_probability_mass - damage_positive_mass,
            hit_probability_mass,
        ),
        "saturated_damage_fraction": _safe_rate(
            saturated_mass,
            hit_probability_mass,
        ),
        "mean_degraded_given_hit": _safe_rate(
            degraded_given_hit_numerator,
            hit_probability_mass,
        ),
        "mean_failed_given_hit": _safe_rate(
            failed_given_hit_numerator,
            hit_probability_mass,
        ),
        "expected_hit_components_per_replicate": float(hit_probability_mass),
        "expected_hit_component_observations": float(
            replicate_count * hit_probability_mass
        ),
        "ordinal_violation_count": ordinal_violations,
        "ordinal_projection_max_correction": float(
            ordinal_projection_max_correction
        ),
    }


def audit_energy_closure(result: dict[str, Any]) -> dict[str, float | int]:
    maximum = 0.0
    event_count = 0
    for replicate in result.get("fragment_audit", []):
        for event in replicate.get("events", []):
            if event.get("record_type") != "intersection":
                continue
            event_count += 1
            error = abs(
                float(event["incident_energy_j"])
                - float(event["residual_energy_j"])
                - float(event["absorbed_energy_j"])
            )
            maximum = max(maximum, error)
    return {
        "energy_closure_event_count": event_count,
        "energy_closure_max_abs_j": maximum,
    }


def score_metrics(metrics: dict[str, Any], objective: dict[str, Any]) -> float:
    """Return a lower-is-better software-identifiability score.

    The targets are not empirical truth.  They keep the fixed-point experiment
    away from all-zero/all-one regimes so later high-fidelity data can identify
    response parameters rather than encounter a saturated simulator.
    """

    terms = (
        ("armor_penetration_rate", "armor_penetration_rate_target", 0.25, 2.0),
        (
            "component_penetration_rate",
            "component_penetration_rate_target",
            0.25,
            1.5,
        ),
        (
            "damage_positive_given_hit",
            "damage_positive_given_hit_target",
            0.35,
            1.0,
        ),
        (
            "mean_degraded_given_hit",
            "mean_degraded_given_hit_target",
            0.35,
            1.0,
        ),
        (
            "mean_failed_given_hit",
            "mean_failed_given_hit_target",
            0.25,
            1.0,
        ),
    )
    score = 0.0
    for metric_name, target_name, scale, weight in terms:
        delta = (float(metrics[metric_name]) - float(objective[target_name])) / scale
        score += weight * delta * delta
    score += 8.0 * float(metrics["saturated_damage_fraction"])
    score += 50.0 * int(metrics["ordinal_violation_count"])
    score += 100.0 * float(
        metrics.get("ordinal_projection_max_correction", 0.0)
    )
    closure_error = float(metrics.get("energy_closure_max_abs_j", 0.0))
    if not math.isfinite(closure_error) or closure_error > 1.0e-8:
        score += 1000.0
    return float(score)


def run_candidate(
    *,
    target_data: dict[str, Any],
    case_data: dict[str, Any],
    registry_data: dict[str, Any],
    candidate: dict[str, Any],
    objective: dict[str, Any],
    replicates: int,
) -> dict[str, Any]:
    started = perf_counter()
    candidate_registry = apply_candidate(registry_data, candidate["parameters"])
    engine = DamageEngine(target_data, PhysicsRegistry(candidate_registry))

    repeated_case = deepcopy(case_data)
    repeated_case["analysis"]["replicates"] = int(replicates)
    repeated_case["analysis"]["adaptive_ci_half_width"] = None
    repeated_case["analysis"]["capture_fragment_audit"] = False
    result = engine.run(repeated_case).to_dict()
    metrics = summarize_result(result)

    audit_case = deepcopy(case_data)
    audit_case["analysis"]["replicates"] = 1
    audit_case["analysis"]["adaptive_ci_half_width"] = None
    audit_case["analysis"]["capture_fragment_audit"] = True
    audit = engine.run(audit_case).to_dict()
    metrics.update(audit_energy_closure(audit))

    return {
        "candidate_id": candidate["id"],
        "parameters": candidate["parameters"],
        "replicates": int(replicates),
        "metrics": metrics,
        "score": score_metrics(metrics, objective),
        "elapsed_seconds": perf_counter() - started,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--physics", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--replicates", type=int, default=64)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    arguments = _build_parser().parse_args()
    if arguments.replicates < 2:
        raise ValueError("replicates must be at least two")

    target_path = Path(arguments.target).resolve()
    case_path = Path(arguments.case).resolve()
    physics_path = Path(arguments.physics).resolve()
    candidates_path = Path(arguments.candidates).resolve()
    output_path = Path(arguments.output).resolve()
    input_paths = {
        "target": target_path,
        "case": case_path,
        "physics": physics_path,
        "candidate_plan": candidates_path,
    }
    source_identity_at_start = source_identity_snapshot(input_paths)
    plan_document = load_json(candidates_path)
    objective, candidates = load_candidate_plan(candidates_path)
    admission_gates = plan_document.get("admission_gates")
    if not isinstance(admission_gates, dict) or not admission_gates:
        raise ValueError(
            "candidate plan must declare non-empty admission_gates for formal selection"
        )
    target_data = load_json(target_path)
    case_data = load_json(case_path)
    registry_data = load_json(physics_path)

    results = []
    for index, candidate in enumerate(candidates, 1):
        evaluated = run_candidate(
            target_data=target_data,
            case_data=case_data,
            registry_data=registry_data,
            candidate=candidate,
            objective=objective,
            replicates=arguments.replicates,
        )
        evaluated["admission_gate_report"] = admission_gate_report(
            evaluated["metrics"], admission_gates
        )
        results.append(evaluated)
        metrics = evaluated["metrics"]
        print(
            f"[{index:02d}/{len(candidates):02d}] {candidate['id']}: "
            f"score={evaluated['score']:.4f}, "
            f"armor={metrics['armor_penetration_rate']:.3f}, "
            f"component={metrics['component_penetration_rate']:.3f}, "
            f"damage+|hit={metrics['damage_positive_given_hit']:.3f}, "
            f"saturated={metrics['saturated_damage_fraction']:.3f}",
            flush=True,
        )

    ranked = sorted(results, key=lambda item: (item["score"], item["candidate_id"]))
    diagnostic_ranking_leader_id = ranked[0]["candidate_id"]
    admissible = [
        item for item in ranked if item["admission_gate_report"]["passed"]
    ]
    selected_candidate_id = (
        admissible[0]["candidate_id"] if admissible else None
    )
    source_identity_at_end = source_identity_snapshot(input_paths)
    changed_source_identities = sorted(
        key
        for key in set(source_identity_at_start) | set(source_identity_at_end)
        if source_identity_at_start.get(key) != source_identity_at_end.get(key)
    )
    source_stable_during_run = (
        not changed_source_identities
        and source_identity_at_start["engine_code_hash"] == ENGINE_CODE_HASH
    )
    if not source_stable_during_run:
        selected_candidate_id = None
    payload = {
        "schema_version": "1.2.0",
        "study_kind": "exploratory_single_point_repeated_sampling",
        "interpretation_status": "software_tuning_only_not_physical_calibration",
        "study_status": (
            "invalid_source_drift"
            if not source_stable_during_run
            else "formal_software_candidate_available"
            if selected_candidate_id is not None
            else "diagnostic_only_no_domain_complete_candidate"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "engine_code_hash": ENGINE_CODE_HASH,
        "inputs": {
            "target_path": str(target_path),
            "target_file_hash": file_hash(target_path),
            "case_path": str(case_path),
            "case_file_hash": file_hash(case_path),
            "physics_path": str(physics_path),
            "physics_file_hash": file_hash(physics_path),
            "candidate_plan_path": str(candidates_path),
            "candidate_plan_file_hash": file_hash(candidates_path),
            "replicates_per_candidate": int(arguments.replicates),
        },
        "objective": objective,
        "admission_gates": admission_gates,
        "source_identity_at_start": source_identity_at_start,
        "source_identity_at_end": source_identity_at_end,
        "source_stable_during_run": source_stable_during_run,
        "changed_source_identities": changed_source_identities,
        "selection_rule": (
            "formal selection requires complete_model_domain, zero affected "
            "replicates, energy closure, every declared distribution gate, "
            "and stable source identities; the "
            "diagnostic leader minimizes the disclosed software-identifiability "
            "score and is not a physical calibration"
        ),
        "diagnostic_ranking_leader_id": diagnostic_ranking_leader_id,
        "formally_admissible_candidate_ids": [
            item["candidate_id"] for item in admissible
        ],
        "selected_candidate_id": selected_candidate_id,
        "ranking": [item["candidate_id"] for item in ranked],
        "results": ranked,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"diagnostic_ranking_leader={diagnostic_ranking_leader_id}")
    print(f"selected={selected_candidate_id}")
    print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
