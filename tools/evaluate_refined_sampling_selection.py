"""Freeze and validate an 8-ray-per-ring refined sampling selection.

This follow-up reuses explicit teacher results from the first proxy experiment,
but the refined selection is frozen from input-only proxy scores before any
missing teacher cases are executed.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.engine import ENGINE_CODE_HASH, ENGINE_VERSION, DamageEngine  # noqa: E402
from damage_engine.fast_sampling_proxy import FastSamplingProxy  # noqa: E402
from damage_engine.registry import PhysicsRegistry  # noqa: E402
from damage_engine.schema_resources import schema_tree_hash  # noqa: E402
from damage_engine.utils import content_hash, source_tree_hash  # noqa: E402
from evaluate_fast_sampling_proxy import (  # noqa: E402
    _accuracy_report,
    _cell_stratified_arm_difference_bootstrap,
    _arm_report,
    _file_hash,
    _greedy_coverage_selection,
    _load_json,
    _run_teacher,
    _screen,
    _write_json,
    _write_jsonl_gz,
)


def _read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-directory", type=Path, required=True)
    parser.add_argument(
        "--proposal-input",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("configs/targets/target.fragment_only.v1.json"),
    )
    parser.add_argument(
        "--physics",
        type=Path,
        default=Path("configs/physics/loitering_impact.json"),
    )
    parser.add_argument("--azimuths-per-ring", type=int, default=8)
    parser.add_argument("--budget-per-cell", type=int, default=4)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026111420)
    return parser


def main() -> int:
    args = _parser().parse_args()
    evidence = args.evidence_directory
    base_evaluation = _load_json(evidence / "evaluation.json")
    recorded_engine_hash = str(base_evaluation["inputs"]["engine_code_hash"])
    if recorded_engine_hash != ENGINE_CODE_HASH:
        raise RuntimeError(
            "refusing to mix teacher/proxy evidence across engine revisions: "
            f"base={recorded_engine_hash}, current={ENGINE_CODE_HASH}"
        )
    outputs = [
        evidence / "refined_008_proxy_rows.jsonl.gz",
        evidence / "refined_008_selection_manifest.json",
        evidence / "refined_008_teacher_additions.jsonl.gz",
        evidence / "refined_008_teacher_results_additions.jsonl.gz",
        evidence / "refined_008_evaluation.json",
    ]
    existing_outputs = [str(path) for path in outputs if path.exists()]
    if existing_outputs:
        raise FileExistsError(f"refusing to overwrite: {existing_outputs}")

    proposal = _load_json(args.proposal_input)
    candidate_ids = [str(value) for value in proposal["candidate_ids"]]
    cases = proposal["candidate_cases"]
    cases_by_id = {
        identifier: case
        for identifier, case in zip(candidate_ids, cases, strict=True)
    }
    engine = DamageEngine(
        _load_json(args.target),
        PhysicsRegistry(_load_json(args.physics)),
    )
    proxy = FastSamplingProxy(
        engine,
        azimuths_per_ring=args.azimuths_per_ring,
    )
    proxy_rows, proxy_seconds = _screen(cases, candidate_ids, proxy)
    _write_jsonl_gz(outputs[0], proxy_rows)
    proxy_by_id = {row["candidate_id"]: row for row in proxy_rows}
    refined_selected = _greedy_coverage_selection(
        proxy_rows,
        per_cell=args.budget_per_cell,
    )
    original_selection = _load_json(evidence / "selection_manifest.json")
    random_selected = [
        str(value)
        for value in original_selection["stratified_random_candidate_ids"]
    ]
    validation_ids = sorted(set(refined_selected) | set(random_selected))

    existing_teacher = _read_jsonl_gz(evidence / "teacher_metrics.jsonl.gz")
    existing_teacher_by_id = {
        row["candidate_id"]: row for row in existing_teacher
    }
    missing_ids = [
        identifier for identifier in validation_ids if identifier not in existing_teacher_by_id
    ]
    selection_manifest = {
        "schema_version": "refined-fast-sampling-selection@1.0.0",
        "status": "frozen_before_missing_teacher_execution",
        "proposal_input_hash": _file_hash(args.proposal_input),
        "engine_version": ENGINE_VERSION,
        "engine_code_hash": ENGINE_CODE_HASH,
        "azimuths_per_ring": args.azimuths_per_ring,
        "budget_per_cell": args.budget_per_cell,
        "candidate_count": len(candidate_ids),
        "refined_proxy_candidate_ids": refined_selected,
        "stratified_random_candidate_ids": random_selected,
        "validation_union_candidate_ids": validation_ids,
        "teacher_results_reused_candidate_ids": sorted(
            set(validation_ids) & set(existing_teacher_by_id)
        ),
        "missing_teacher_candidate_ids": missing_ids,
        "teacher_outcomes_used_in_selection": False,
    }
    _write_json(outputs[1], selection_manifest)

    additions, full_results, teacher_addition_seconds = _run_teacher(
        engine,
        cases_by_id,
        missing_ids,
        replicates=args.replicates,
    )
    _write_jsonl_gz(outputs[2], additions)
    _write_jsonl_gz(outputs[3], full_results)
    combined_teacher_by_id = dict(existing_teacher_by_id)
    combined_teacher_by_id.update({row["candidate_id"]: row for row in additions})
    combined_teacher = [combined_teacher_by_id[identifier] for identifier in validation_ids]

    accuracy = _accuracy_report(
        proxy_by_id,
        combined_teacher,
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
    )
    proxy_arm = _arm_report(refined_selected, combined_teacher_by_id)
    random_arm = _arm_report(random_selected, combined_teacher_by_id)
    difference = _cell_stratified_arm_difference_bootstrap(
        refined_selected,
        random_selected,
        combined_teacher_by_id,
        draws=args.bootstrap_draws,
        seed=args.bootstrap_seed + 100,
    )
    base = _load_json(evidence / "evaluation.json")
    refinement = _load_json(evidence / "resolution_refinement_summary.json")
    teacher_seconds_per_case = float(base["timing"]["teacher_seconds_per_case"])
    proxy_seconds_per_case = proxy_seconds / len(candidate_ids)
    speedup = teacher_seconds_per_case / proxy_seconds_per_case
    convergence_8_to_16 = refinement["resolutions"]["16"][
        "convergence_from_previous"
    ]
    coverage_not_worse = (
        proxy_arm["component_coverage"]["ever_positive_count"]
        >= random_arm["component_coverage"]["ever_positive_count"]
        and proxy_arm["component_coverage"]["at_least_5_cases"]
        >= random_arm["component_coverage"]["at_least_5_cases"]
    )
    hits_difference_positive = (
        difference["mean_total_component_hits_difference"]["ci95"][0] > 0.0
    )
    energy_difference_positive = (
        difference["mean_total_component_energy_difference_j"]["ci95"][0] > 0.0
    )
    plan_level_checks = {
        "case_any_hit_auc_at_least_0p85": accuracy["case_any_hit_auc"] >= 0.85,
        "case_hit_spearman_at_least_0p80": accuracy["case_hit_spearman"] >= 0.80,
        "case_energy_spearman_at_least_0p80": accuracy["case_energy_spearman"] >= 0.80,
        "component_support_auc_at_least_0p85": accuracy["component_support_auc"] >= 0.85,
        "top20_overlap_at_least_0p70": accuracy["top20_energy_overlap"] >= 0.70,
        "speedup_at_least_10": speedup >= 10.0,
        "eight_to_sixteen_rank_at_least_0p90": convergence_8_to_16[
            "sampling_score_spearman"
        ]
        >= 0.90,
        "eight_to_sixteen_component_l1_at_most_0p15": convergence_8_to_16[
            "aggregate_component_exposure_relative_l1"
        ]
        <= 0.15,
        "teacher_component_coverage_not_worse": coverage_not_worse,
        "teacher_hit_difference_ci_positive": hits_difference_positive,
        "teacher_energy_difference_ci_positive": energy_difference_positive,
    }
    plan_level_pass = all(plan_level_checks.values())
    stand_alone_component_pass = accuracy["component_support_recall"] >= 0.90
    if plan_level_pass and not stand_alone_component_pass:
        recommendation = (
            "conditional_pass_for_plan_level_prescreening_with_mandatory_small_teacher_pilot"
        )
    elif plan_level_pass:
        recommendation = "pass_for_plan_level_prescreening"
    else:
        recommendation = "do_not_use_for_sampling_approval"

    live_engine_code_hash = content_hash(
        {
            "python_source_tree_hash": source_tree_hash(ROOT / "damage_engine"),
            "schema_tree_hash": schema_tree_hash(),
        }
    )
    source_stable_during_run = live_engine_code_hash == ENGINE_CODE_HASH
    if not source_stable_during_run:
        plan_level_pass = False
        recommendation = "discard_evidence_and_rerun_after_engine_source_is_frozen"

    report = {
        "schema_version": "refined-fast-sampling-evaluation@1.0.0",
        "status": "complete" if source_stable_during_run else "invalid_source_drift",
        "evidence_boundary": base["evidence_boundary"],
        "engine_code_hash": ENGINE_CODE_HASH,
        "engine_code_hash_at_end": live_engine_code_hash,
        "source_stable_during_run": source_stable_during_run,
        "selection_frozen_before_missing_teacher_execution": True,
        "candidate_count": len(candidate_ids),
        "azimuths_per_ring": args.azimuths_per_ring,
        "proxy_seconds_all_candidates": proxy_seconds,
        "proxy_seconds_per_case": proxy_seconds_per_case,
        "teacher_seconds_per_case": teacher_seconds_per_case,
        "speedup_vs_teacher": speedup,
        "refined_arm_count": len(refined_selected),
        "random_arm_count": len(random_selected),
        "validation_union_count": len(validation_ids),
        "reused_teacher_count": len(validation_ids) - len(missing_ids),
        "new_teacher_count": len(missing_ids),
        "new_teacher_seconds": teacher_addition_seconds,
        "accuracy": accuracy,
        "numerical_convergence_8_to_16": convergence_8_to_16,
        "arms": {
            "refined_proxy_greedy": proxy_arm,
            "stratified_random": random_arm,
            "differences": difference,
        },
        "plan_level_checks": plan_level_checks,
        "plan_level_pass": plan_level_pass,
        "stand_alone_component_recall_gate": {
            "threshold": 0.90,
            "observed": accuracy["component_support_recall"],
            "pass": stand_alone_component_pass,
        },
        "recommendation": recommendation,
        "interpretation": (
            "The proxy may rank and compare sampling plans, but it must not be used "
            "as a stand-alone per-component label or as physical validation."
        ),
    }
    _write_json(outputs[4], report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
