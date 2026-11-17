"""Predeclared multi-condition robustness selection for penetration parameters.

The study separates candidate selection from confirmation/holdout evaluation.
Its objective is software identifiability and non-degenerate behaviour across
geometry, munition and random-seed conditions.  It is not a substitute for
physical calibration data.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from damage_engine.contact_manifold_sampling import apply_contact_manifold
from damage_engine.engine import (
    DOMAIN_ABSENCE_ALPHA,
    DOMAIN_ABSENCE_RATE_THRESHOLD,
    ENGINE_CODE_HASH,
)
from damage_engine.munition_profiles import apply_munition_profile
from damage_engine.schema_resources import schema_tree_hash
from damage_engine.utils import content_hash, file_hash, load_json, source_tree_hash
from tools.sweep_penetration_parameters import run_candidate, score_metrics


EVALUATION_TOOL_PATH = Path(__file__).resolve()
SWEEP_TOOL_PATH = REPOSITORY_ROOT / "tools" / "sweep_penetration_parameters.py"


_MEAN_METRICS = (
    "armor_penetration_rate",
    "component_penetration_rate",
    "damage_positive_given_hit",
    "zero_damage_given_hit_fraction",
    "saturated_damage_fraction",
    "mean_degraded_given_hit",
    "mean_failed_given_hit",
    "expected_hit_components_per_replicate",
    "expected_hit_component_observations",
    "out_of_domain_impact_fraction",
    "penetration_affected_replicate_fraction",
)

DIAGNOSTIC_MIN_SELECTION_REPLICATES = 32
DIAGNOSTIC_MIN_CONFIRMATION_REPLICATES = 64
DIAGNOSTIC_MIN_HOLDOUT_REPLICATES = 128


def minimum_zero_event_replicates(
    *,
    alpha: float = DOMAIN_ABSENCE_ALPHA,
    rate_threshold: float = DOMAIN_ABSENCE_RATE_THRESHOLD,
) -> int:
    """Return the minimum R whose exact zero-event upper bound meets the gate."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    if not 0.0 < rate_threshold < 1.0:
        raise ValueError("rate_threshold must lie in (0,1)")
    count = max(
        1,
        math.ceil(math.log(alpha) / math.log(1.0 - rate_threshold)),
    )
    while 1.0 - alpha ** (1.0 / count) > rate_threshold:
        count += 1
    return count


FORMAL_MIN_REPLICATES = minimum_zero_event_replicates()


def study_mode_contract(mode: str) -> dict[str, Any]:
    """Describe the admission authority carried by a study mode."""

    normalized = str(mode).strip().lower()
    if normalized == "formal":
        return {
            "mode": "formal",
            "promotion_permitted": True,
            "interpretation_status": "formal_software_admission",
        }
    if normalized == "diagnostic":
        return {
            "mode": "diagnostic",
            "promotion_permitted": False,
            "interpretation_status": "diagnostic_only_non_promotable",
        }
    raise ValueError("mode must be 'formal' or 'diagnostic'")


def source_identity_snapshot(
    input_paths: Path | dict[str, Path],
) -> dict[str, str]:
    """Bind the complete executable selection chain and frozen study plan."""

    paths = (
        {"plan": input_paths}
        if isinstance(input_paths, Path)
        else dict(input_paths)
    )
    identity = {
        "engine_code_hash": content_hash(
            {
                "python_source_tree_hash": source_tree_hash(
                    REPOSITORY_ROOT / "damage_engine"
                ),
                "schema_tree_hash": schema_tree_hash(),
            }
        ),
        "evaluation_tool_hash": file_hash(EVALUATION_TOOL_PATH),
        "sweep_tool_hash": file_hash(SWEEP_TOOL_PATH),
    }
    identity.update(
        {
            f"{name}_hash": file_hash(path)
            for name, path in sorted(paths.items())
        }
    )
    return identity


def source_identity_drift(
    start: dict[str, str], end: dict[str, str]
) -> list[str]:
    """Return stable, machine-readable identities changed during execution."""

    return sorted(
        key for key in set(start) | set(end) if start.get(key) != end.get(key)
    )


def materialize_condition(
    base_case: dict[str, Any],
    contact_manifold: dict[str, Any],
    condition: dict[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    """Build a deterministic pre-outcome case for selection or confirmation."""

    if phase not in {"selection", "confirmation", "holdout"}:
        raise ValueError("phase must be 'selection', 'confirmation', or 'holdout'")
    seed_key = f"{phase}_seed"
    if seed_key not in condition:
        raise ValueError(f"condition {condition.get('id')} is missing {seed_key}")
    case = deepcopy(base_case)
    apply_munition_profile(case, str(condition["profile_id"]))
    apply_contact_manifold(case, contact_manifold, condition["latent"])
    condition_id = str(condition["id"])
    case["case_id"] = f"penetration-{phase}-{condition_id}"
    analysis = case.setdefault("analysis", {})
    analysis["aleatory_seed"] = int(condition[seed_key])
    analysis["epistemic_sample_id"] = f"penetration-{phase}-{condition_id}"
    analysis["adaptive_ci_half_width"] = None
    analysis["capture_fragment_audit"] = False
    analysis["capture_replicate_trace"] = False
    case.setdefault("metadata", {})["multicondition_study"] = {
        "condition_id": condition_id,
        "role": str(condition["role"]),
        "phase": phase,
        "outcome_blind": True,
    }
    return case


def _finite_metric(metrics: dict[str, Any], key: str) -> float:
    value = float(metrics[key])
    if not math.isfinite(value):
        raise ValueError(f"metric {key} must be finite")
    return value


def evaluate_condition_gates(
    metrics: dict[str, Any],
    gates: dict[str, Any],
    *,
    replicate_count: int | None = None,
) -> dict[str, Any]:
    """Evaluate predeclared non-degeneracy and numerical-validity gates."""

    checks: dict[str, bool] = {}
    for gate_name, metric_name in (
        ("armor_penetration_rate_range", "armor_penetration_rate"),
        ("component_penetration_rate_range", "component_penetration_rate"),
        ("damage_positive_given_hit_range", "damage_positive_given_hit"),
    ):
        lower, upper = (float(value) for value in gates[gate_name])
        if not 0.0 <= lower <= upper <= 1.0:
            raise ValueError(f"{gate_name} must be an ordered probability range")
        checks[gate_name] = lower <= _finite_metric(metrics, metric_name) <= upper

    for gate_name, metric_name in (
        ("saturated_damage_fraction_max", "saturated_damage_fraction"),
        (
            "ordinal_projection_max_correction_max",
            "ordinal_projection_max_correction",
        ),
        ("energy_closure_max_abs_j", "energy_closure_max_abs_j"),
    ):
        checks[gate_name] = _finite_metric(metrics, metric_name) <= float(
            gates[gate_name]
        )
    checks["expected_hit_component_observations_min"] = _finite_metric(
        metrics, "expected_hit_component_observations"
    ) >= float(gates["expected_hit_component_observations_min"])
    if "out_of_domain_impact_fraction_max" in gates:
        checks["out_of_domain_impact_fraction_max"] = _finite_metric(
            metrics, "out_of_domain_impact_fraction"
        ) <= float(gates["out_of_domain_impact_fraction_max"])
    checks["penetration_domain_complete"] = (
        _finite_metric(metrics, "penetration_affected_replicate_fraction") == 0.0
        and metrics.get("scientific_completion_status") == "complete_model_domain"
    )
    zero_event_upper95 = None
    if replicate_count is not None:
        if replicate_count < 1:
            raise ValueError("replicate_count must be positive")
        if _finite_metric(metrics, "penetration_affected_replicate_fraction") == 0.0:
            zero_event_upper95 = 1.0 - 0.05 ** (1.0 / replicate_count)
        else:
            zero_event_upper95 = 1.0
    if "penetration_affected_replicate_upper95_max" in gates:
        if zero_event_upper95 is None:
            raise ValueError(
                "replicate_count is required for the penetration-domain confidence gate"
            )
        checks["penetration_affected_replicate_upper95_max"] = (
            zero_event_upper95
            <= float(gates["penetration_affected_replicate_upper95_max"])
        )
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failed,
        "checks": checks,
        "failed_gates": failed,
        "penetration_affected_replicate_upper95": zero_event_upper95,
    }


def aggregate_equal_weight_conditions(
    rows: list[dict[str, Any]],
    objective: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate condition-level estimands with one vote per condition."""

    if not rows:
        raise ValueError("at least one condition result is required")
    equal_weight_metrics = {
        key: sum(_finite_metric(row["metrics"], key) for row in rows) / len(rows)
        for key in _MEAN_METRICS
        if all(key in row["metrics"] for row in rows)
    }
    condition_scores = [
        float(row.get("score", score_metrics(row["metrics"], objective)))
        for row in rows
    ]
    passed_count = sum(bool(row.get("gates", {}).get("passed", True)) for row in rows)
    return {
        "condition_count": len(rows),
        "equal_condition_weight": 1.0 / len(rows),
        "equal_weight_metrics": equal_weight_metrics,
        "mean_condition_score": sum(condition_scores) / len(condition_scores),
        "worst_condition_score": max(condition_scores),
        "robust_score": (
            sum(condition_scores) / len(condition_scores)
            + 0.25 * max(condition_scores)
        ),
        "passed_condition_count": passed_count,
        "all_condition_gates_passed": passed_count == len(rows),
    }


def _flatten_leaves(value: Any, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    if isinstance(value, dict):
        output: dict[tuple[str, ...], Any] = {}
        for key in sorted(value):
            output.update(_flatten_leaves(value[key], (*prefix, str(key))))
        return output
    return {prefix: value}


def _parameter_change_count(parameters: dict[str, Any], anchor: dict[str, Any]) -> int:
    left = _flatten_leaves(parameters)
    right = _flatten_leaves(anchor)
    return sum(left.get(key) != right.get(key) for key in set(left) | set(right))


def _monotonic_checks(
    candidate_results: dict[str, dict[str, Any]],
    checks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for check in checks:
        lower_id = str(check["lower_candidate_id"])
        higher_id = str(check["higher_candidate_id"])
        unavailable_candidate_ids = [
            candidate_id
            for candidate_id in (lower_id, higher_id)
            if not all(
                condition["gates"]["checks"]["penetration_domain_complete"]
                for condition in candidate_results[candidate_id]["conditions"]
            )
        ]
        if unavailable_candidate_ids:
            output.append(
                {
                    **check,
                    "status": "unavailable",
                    "passed": None,
                    "reason": "domain_censored",
                    "unavailable_candidate_ids": unavailable_candidate_ids,
                    "lower_value": None,
                    "higher_value": None,
                }
            )
            continue
        metric = str(check["metric"])
        relation = str(check["expected_relation"])
        tolerance = float(check.get("tolerance", 0.0))
        left = float(
            candidate_results[lower_id]["aggregate"]["equal_weight_metrics"][metric]
        )
        right = float(
            candidate_results[higher_id]["aggregate"]["equal_weight_metrics"][metric]
        )
        if relation == "lower_ge_higher":
            passed = left + tolerance >= right
        elif relation == "lower_le_higher":
            passed = left <= right + tolerance
        else:
            raise ValueError(f"unsupported monotonic relation: {relation}")
        output.append(
            {
                **check,
                "status": "available",
                "lower_value": left,
                "higher_value": right,
                "passed": passed,
            }
        )
    return output


def resolve_condition_sets(
    plan: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve and validate the sealed selection/confirmation/holdout phases."""

    holdout_evidence_status = str(plan.get("holdout_evidence_status", ""))
    if holdout_evidence_status not in {
        "sealed_unobserved",
        "consumed_by_prior_soft_gate_run",
    }:
        raise ValueError(
            "holdout_evidence_status must declare sealed_unobserved or "
            "consumed_by_prior_soft_gate_run"
        )
    condition_map = {str(row["id"]): row for row in plan["conditions"]}
    phase_ids = {
        "selection": [str(value) for value in plan["selection_condition_ids"]],
        "confirmation": [
            str(value) for value in plan["confirmation_condition_ids"]
        ],
        "holdout": [str(value) for value in plan["holdout_condition_ids"]],
    }
    for phase, identifiers in phase_ids.items():
        if not identifiers:
            raise ValueError(f"{phase}_condition_ids must not be empty")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"{phase}_condition_ids must not contain duplicates")
        missing = sorted(set(identifiers) - set(condition_map))
        if missing:
            raise ValueError(f"unknown {phase} condition ids: {missing}")
    if set(phase_ids["confirmation"]) & set(phase_ids["holdout"]):
        raise ValueError("confirmation and holdout condition ids must be disjoint")
    if set(phase_ids["selection"]) & set(phase_ids["holdout"]):
        raise ValueError("selection and holdout condition ids must be disjoint")
    for identifier in phase_ids["holdout"]:
        if str(condition_map[identifier]["role"]) != "holdout":
            raise ValueError(f"holdout condition {identifier} must have role=holdout")
    for phase in ("selection", "confirmation"):
        for identifier in phase_ids[phase]:
            if str(condition_map[identifier]["role"]) == "holdout":
                raise ValueError(
                    f"{phase} condition {identifier} must not have role=holdout"
                )
    return tuple(
        [condition_map[identifier] for identifier in phase_ids[phase]]
        for phase in ("selection", "confirmation", "holdout")
    )


def validate_formal_study_plan(plan: dict[str, Any]) -> None:
    """Reject a formal study whose holdout or random streams are not sealed."""

    if str(plan.get("holdout_evidence_status", "")) != "sealed_unobserved":
        raise ValueError(
            "formal mode requires holdout_evidence_status=sealed_unobserved"
        )
    condition_sets = resolve_condition_sets(plan)
    phase_seeds: list[tuple[str, str, int]] = []
    for phase, conditions in zip(
        ("selection", "confirmation", "holdout"),
        condition_sets,
        strict=True,
    ):
        seed_key = f"{phase}_seed"
        for condition in conditions:
            condition_id = str(condition["id"])
            if seed_key not in condition:
                raise ValueError(
                    f"formal {phase} condition {condition_id} is missing {seed_key}"
                )
            phase_seeds.append(
                (phase, condition_id, int(condition[seed_key]))
            )

    seed_owners: dict[int, list[str]] = {}
    for phase, condition_id, seed in phase_seeds:
        seed_owners.setdefault(seed, []).append(f"{phase}:{condition_id}")
    reused = {
        seed: owners for seed, owners in seed_owners.items() if len(owners) > 1
    }
    if reused:
        details = ", ".join(
            f"{seed}=>{'/'.join(owners)}"
            for seed, owners in sorted(reused.items())
        )
        raise ValueError(
            "formal mode requires independent phase seeds; reused seeds: "
            f"{details}"
        )


def _evaluate_one(
    *,
    target_data: dict[str, Any],
    base_case: dict[str, Any],
    registry_data: dict[str, Any],
    contact_manifold: dict[str, Any],
    condition: dict[str, Any],
    candidate: dict[str, Any],
    objective: dict[str, Any],
    gates: dict[str, Any],
    phase: str,
    replicates: int,
) -> dict[str, Any]:
    condition_case = materialize_condition(
        base_case, contact_manifold, condition, phase=phase
    )
    evaluated = run_candidate(
        target_data=target_data,
        case_data=condition_case,
        registry_data=registry_data,
        candidate=candidate,
        objective=objective,
        replicates=replicates,
    )
    evaluated["condition_id"] = str(condition["id"])
    evaluated["condition_role"] = str(condition["role"])
    evaluated["profile_id"] = str(condition["profile_id"])
    evaluated["phase"] = phase
    evaluated["replicates"] = int(replicates)
    evaluated["aleatory_seed"] = int(
        condition_case["analysis"]["aleatory_seed"]
    )
    evaluated["input_case_hash"] = content_hash(condition_case)
    evaluated["gates"] = evaluate_condition_gates(
        evaluated["metrics"], gates, replicate_count=replicates
    )
    return evaluated


def select_admissible_candidate(
    ranking: list[str],
    candidate_results: dict[str, dict[str, Any]],
) -> str | None:
    """Return the highest-ranked candidate that passed every selection gate."""

    return next(
        (
            candidate_id
            for candidate_id in ranking
            if candidate_results[candidate_id]["aggregate"][
                "all_condition_gates_passed"
            ]
        ),
        None,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("formal", "diagnostic"),
        default="formal",
        help=(
            "formal permits software admission and requires the full zero-event "
            "evidence contract; diagnostic is explicitly non-promotable"
        ),
    )
    parser.add_argument("--target", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--physics", required=True)
    parser.add_argument("--sampling-spec", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--selection-replicates", type=int)
    parser.add_argument("--confirmation-replicates", type=int)
    parser.add_argument("--holdout-replicates", type=int)
    parser.add_argument("--output", required=True)
    return parser


def validate_replicate_budgets(
    selection: int,
    confirmation: int,
    holdout: int,
    *,
    mode: str = "formal",
) -> tuple[int, int, int]:
    """Enforce the phase budgets authorized by the selected study mode."""

    normalized_mode = str(study_mode_contract(mode)["mode"])
    budgets = (int(selection), int(confirmation), int(holdout))
    minimums = (
        (FORMAL_MIN_REPLICATES,) * 3
        if normalized_mode == "formal"
        else (
            DIAGNOSTIC_MIN_SELECTION_REPLICATES,
            DIAGNOSTIC_MIN_CONFIRMATION_REPLICATES,
            DIAGNOSTIC_MIN_HOLDOUT_REPLICATES,
        )
    )
    for phase, value, minimum in zip(
        ("selection", "confirmation", "holdout"),
        budgets,
        minimums,
        strict=True,
    ):
        if value < minimum:
            raise ValueError(
                f"{normalized_mode} {phase} replicate count must be at least "
                f"{minimum}"
            )
    return budgets


def resolve_replicate_budgets(
    mode: str,
    selection: int | None,
    confirmation: int | None,
    holdout: int | None,
) -> tuple[int, int, int]:
    """Fill mode-specific defaults, then enforce the same explicit contract."""

    normalized_mode = str(study_mode_contract(mode)["mode"])
    defaults = (
        (FORMAL_MIN_REPLICATES,) * 3
        if normalized_mode == "formal"
        else (
            DIAGNOSTIC_MIN_SELECTION_REPLICATES,
            DIAGNOSTIC_MIN_CONFIRMATION_REPLICATES,
            DIAGNOSTIC_MIN_HOLDOUT_REPLICATES,
        )
    )
    requested = (selection, confirmation, holdout)
    resolved = tuple(
        default if value is None else int(value)
        for value, default in zip(requested, defaults, strict=True)
    )
    return validate_replicate_budgets(
        *resolved,
        mode=normalized_mode,
    )


def main() -> int:
    args = _parser().parse_args()
    mode_contract = study_mode_contract(args.mode)
    (
        args.selection_replicates,
        args.confirmation_replicates,
        args.holdout_replicates,
    ) = resolve_replicate_budgets(
        args.mode,
        args.selection_replicates,
        args.confirmation_replicates,
        args.holdout_replicates,
    )
    print(
        f"study_mode={args.mode}; "
        f"promotion_permitted={str(mode_contract['promotion_permitted']).lower()}; "
        f"interpretation={mode_contract['interpretation_status']}",
        flush=True,
    )
    if args.mode == "diagnostic":
        print(
            "DIAGNOSTIC MODE: rankings and gate results are non-promotable; "
            "no candidate will enter confirmation or holdout.",
            flush=True,
        )
    paths = {
        name: Path(getattr(args, name.replace("-", "_"))).resolve()
        for name in ("target", "case", "physics", "sampling_spec", "plan")
    }
    source_identity_at_start = source_identity_snapshot(paths)
    target_data = load_json(paths["target"])
    base_case = load_json(paths["case"])
    registry_data = load_json(paths["physics"])
    sampling_spec = load_json(paths["sampling_spec"])
    plan = load_json(paths["plan"])
    if args.mode == "formal":
        validate_formal_study_plan(plan)
    objective = dict(plan["objective"])
    gates = dict(plan["admission_gates"])
    defaults = deepcopy(plan.get("candidate_defaults", {}))
    candidates = []
    for raw in plan["candidates"]:
        parameters = deepcopy(defaults)
        for key, value in raw.get("parameters", {}).items():
            if isinstance(value, dict) and isinstance(parameters.get(key), dict):
                parameters[key].update(deepcopy(value))
            else:
                parameters[key] = deepcopy(value)
        candidates.append({"id": str(raw["id"]), "parameters": parameters})
    (
        selection_conditions,
        confirmation_conditions,
        holdout_conditions,
    ) = resolve_condition_sets(plan)
    holdout_evidence_status = str(plan["holdout_evidence_status"])
    holdout_is_sealed = holdout_evidence_status == "sealed_unobserved"
    manifold = sampling_spec["contact_manifold"]

    candidate_results: dict[str, dict[str, Any]] = {}
    for candidate_index, candidate in enumerate(candidates, 1):
        rows = []
        for condition_index, condition in enumerate(selection_conditions, 1):
            row = _evaluate_one(
                target_data=target_data,
                base_case=base_case,
                registry_data=registry_data,
                contact_manifold=manifold,
                condition=condition,
                candidate=candidate,
                objective=objective,
                gates=gates,
                phase="selection",
                replicates=args.selection_replicates,
            )
            rows.append(row)
            print(
                f"selection {candidate_index}/{len(candidates)} "
                f"condition {condition_index}/{len(selection_conditions)} "
                f"{candidate['id']}:{condition['id']} score={row['score']:.4f}",
                flush=True,
            )
        candidate_results[candidate["id"]] = {
            "candidate_id": candidate["id"],
            "parameters": candidate["parameters"],
            "conditions": rows,
            "aggregate": aggregate_equal_weight_conditions(rows, objective),
        }

    anchor_id = str(plan["anchor_candidate_id"])
    anchor_parameters = candidate_results[anchor_id]["parameters"]
    for result in candidate_results.values():
        result["parameter_change_count_from_anchor"] = _parameter_change_count(
            result["parameters"], anchor_parameters
        )
    ranking = sorted(
        candidate_results,
        key=lambda identifier: (
            -candidate_results[identifier]["aggregate"]["passed_condition_count"],
            candidate_results[identifier]["aggregate"]["robust_score"],
            candidate_results[identifier]["parameter_change_count_from_anchor"],
            identifier,
        ),
    )
    diagnostic_ranking_leader_id = ranking[0]
    admissible_selection_candidate_ids = [
        identifier
        for identifier in ranking
        if candidate_results[identifier]["aggregate"][
            "all_condition_gates_passed"
        ]
    ]
    gate_candidate_id = select_admissible_candidate(ranking, candidate_results)
    selected_id = (
        gate_candidate_id
        if bool(mode_contract["promotion_permitted"])
        else None
    )
    monotonicity = _monotonic_checks(
        candidate_results, list(plan.get("monotonic_checks", []))
    )

    confirmation_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    confirmation_passed: bool | None = None
    holdout_passed: bool | None = None
    if selected_id is not None:
        selected = next(item for item in candidates if item["id"] == selected_id)
        for condition_index, condition in enumerate(
            confirmation_conditions, 1
        ):
            row = _evaluate_one(
                target_data=target_data,
                base_case=base_case,
                registry_data=registry_data,
                contact_manifold=manifold,
                condition=condition,
                candidate=selected,
                objective=objective,
                gates=gates,
                phase="confirmation",
                replicates=args.confirmation_replicates,
            )
            confirmation_rows.append(row)
            print(
                f"confirmation {condition_index}/"
                f"{len(confirmation_conditions)} "
                f"{selected_id}:{condition['id']} passed={row['gates']['passed']}",
                flush=True,
            )
        confirmation_passed = bool(confirmation_rows) and all(
            row["gates"]["passed"] for row in confirmation_rows
        )
        if confirmation_passed and holdout_is_sealed:
            for condition_index, condition in enumerate(holdout_conditions, 1):
                row = _evaluate_one(
                    target_data=target_data,
                    base_case=base_case,
                    registry_data=registry_data,
                    contact_manifold=manifold,
                    condition=condition,
                    candidate=selected,
                    objective=objective,
                    gates=gates,
                    phase="holdout",
                    replicates=args.holdout_replicates,
                )
                holdout_rows.append(row)
                print(
                    f"holdout {condition_index}/{len(holdout_conditions)} "
                    f"{selected_id}:{condition['id']} passed={row['gates']['passed']}",
                    flush=True,
                )
            holdout_passed = bool(holdout_rows) and all(
                row["gates"]["passed"] for row in holdout_rows
            )
    monotonicity_passed = all(
        row["status"] == "available" and row["passed"] is True
        for row in monotonicity
    )
    all_selection_domain_complete = all(
        condition["gates"]["checks"]["penetration_domain_complete"]
        for result in candidate_results.values()
        for condition in result["conditions"]
    )
    source_identity_at_end = source_identity_snapshot(paths)
    changed_source_identities = source_identity_drift(
        source_identity_at_start, source_identity_at_end
    )
    source_stable_during_run = (
        not changed_source_identities
        and source_identity_at_start["engine_code_hash"] == ENGINE_CODE_HASH
    )
    promotion_eligible = (
        bool(mode_contract["promotion_permitted"])
        and selected_id is not None
        and confirmation_passed is True
        and holdout_passed is True
        and holdout_is_sealed
        and monotonicity_passed
        and source_stable_during_run
    )
    if not source_stable_during_run:
        study_status = "invalid_source_drift"
    elif args.mode == "diagnostic":
        study_status = "diagnostic_complete_non_promotable"
    elif selected_id is None:
        study_status = "selection_no_admissible_candidate"
    elif confirmation_passed is not True:
        study_status = "confirmation_failed"
    elif not holdout_is_sealed:
        study_status = "holdout_unavailable_consumed"
    elif holdout_passed is not True:
        study_status = "holdout_failed"
    elif not monotonicity_passed:
        study_status = "monotonicity_failed"
    else:
        study_status = "complete"

    ledger_entries: list[dict[str, Any]] = []
    for result in candidate_results.values():
        for row in result["conditions"]:
            ledger_entries.append(
                {
                    "phase": "selection",
                    "candidate_id": result["candidate_id"],
                    "condition_id": row["condition_id"],
                    "aleatory_seed": row["aleatory_seed"],
                    "replicates": row["replicates"],
                    "input_case_hash": row["input_case_hash"],
                    "result_hash": content_hash(row),
                }
            )
    for phase, rows in (
        ("confirmation", confirmation_rows),
        ("holdout", holdout_rows),
    ):
        for row in rows:
            ledger_entries.append(
                {
                    "phase": phase,
                    "candidate_id": str(selected_id),
                    "condition_id": row["condition_id"],
                    "aleatory_seed": row["aleatory_seed"],
                    "replicates": row["replicates"],
                    "input_case_hash": row["input_case_hash"],
                    "result_hash": content_hash(row),
                }
            )
    ledger_entries = [
        {"sequence": index, **entry}
        for index, entry in enumerate(ledger_entries, start=1)
    ]

    payload = {
        "schema_version": "1.2.0",
        "study_kind": "exploratory_multicondition_robustness_selection",
        "interpretation_status": "software_tuning_only_not_physical_calibration",
        "study_mode_contract": mode_contract,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "engine_code_hash": ENGINE_CODE_HASH,
        "engine_code_hash_at_start": source_identity_at_start["engine_code_hash"],
        "engine_code_hash_at_end": source_identity_at_end["engine_code_hash"],
        "source_identity_at_start": source_identity_at_start,
        "source_identity_at_end": source_identity_at_end,
        "changed_source_identities": changed_source_identities,
        "source_stable_during_run": source_stable_during_run,
        "status": study_status,
        "inputs": {
            key: {
                "path": str(path),
                "file_hash": source_identity_at_start[f"{key}_hash"],
            }
            for key, path in paths.items()
        },
        "design": {
            "condition_weighting": "equal_weight_after_condition_level_aggregation",
            "common_random_numbers_within_condition": True,
            "selection_replicates": args.selection_replicates,
            "confirmation_replicates": args.confirmation_replicates,
            "holdout_replicates": args.holdout_replicates,
            "holdout_used_for_reselection": False,
            "holdout_evidence_status": holdout_evidence_status,
            "holdout_sealed_until_confirmation_passes": holdout_is_sealed,
            "replicate_contract": {
                "domain_absence_alpha": DOMAIN_ABSENCE_ALPHA,
                "domain_event_rate_threshold": DOMAIN_ABSENCE_RATE_THRESHOLD,
                "formal_minimum_per_phase": FORMAL_MIN_REPLICATES,
                "diagnostic_minimums": {
                    "selection": DIAGNOSTIC_MIN_SELECTION_REPLICATES,
                    "confirmation": DIAGNOSTIC_MIN_CONFIRMATION_REPLICATES,
                    "holdout": DIAGNOSTIC_MIN_HOLDOUT_REPLICATES,
                },
            },
        },
        "execution_ledger": {
            "semantics": (
                "ordered_tamper_evident_phase_ledger_bound_to_input_and_result_hashes"
            ),
            "entries": ledger_entries,
            "ledger_hash": content_hash(ledger_entries),
        },
        "objective": objective,
        "admission_gates": gates,
        "hard_admission_invariants": {
            "penetration_domain_complete": (
                "scientific_completion_status must be complete_model_domain and "
                "penetration affected_replicate_fraction must equal zero"
            )
        },
        "diagnostic_ranking": ranking,
        "ranking_interpretation_status": (
            "diagnostic_only_non_promotable"
            if args.mode == "diagnostic"
            else "admission_eligible"
            if all_selection_domain_complete
            else "diagnostic_only_domain_censored"
        ),
        "diagnostic_ranking_leader_id": diagnostic_ranking_leader_id,
        "diagnostic_gate_candidate_id": gate_candidate_id,
        "admissible_selection_candidate_ids": admissible_selection_candidate_ids,
        "selected_candidate_id": selected_id,
        "selection_results": [candidate_results[item] for item in ranking],
        "monotonic_checks": monotonicity,
        "monotonic_checks_all_available_and_passed": monotonicity_passed,
        "monotonic_checks_interpretation_status": (
            "diagnostic_only_non_promotable"
            if args.mode == "diagnostic"
            else "admission_eligible"
            if all_selection_domain_complete
            else "diagnostic_only_domain_censored"
        ),
        "confirmation_results": confirmation_rows,
        "confirmation_status": (
            "not_evaluated"
            if confirmation_passed is None
            else "passed" if confirmation_passed else "failed"
        ),
        "confirmation_all_conditions_passed": confirmation_passed,
        "holdout_results": holdout_rows,
        "holdout_status": (
            "unavailable_consumed"
            if confirmation_passed is True and not holdout_is_sealed
            else "not_evaluated"
            if holdout_passed is None
            else "passed"
            if holdout_passed
            else "failed"
        ),
        "holdout_passed": holdout_passed,
        "promotion_eligible": promotion_eligible,
        "promotion_eligibility_status": (
            "ineligible_diagnostic_mode"
            if args.mode == "diagnostic"
            else "eligible"
            if promotion_eligible
            else "formal_gates_not_satisfied"
        ),
        "promotion_rule": (
            "a candidate must first pass every selection gate; non-holdout "
            "confirmation is evaluated next and holdout remains sealed until all "
            "confirmation conditions pass; holdout and predeclared monotonic "
            "checks must then pass; every evaluated condition must have complete "
            "penetration-domain coverage with zero affected replicates"
        ),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"diagnostic_ranking_leader={diagnostic_ranking_leader_id}")
    print(f"diagnostic_gate_candidate={gate_candidate_id}")
    print(f"selected={selected_id}")
    print(f"promotion_eligible={promotion_eligible}")
    print(
        "promotion_eligibility_status="
        f"{'ineligible_diagnostic_mode' if args.mode == 'diagnostic' else 'formal_evaluation'}"
    )
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
