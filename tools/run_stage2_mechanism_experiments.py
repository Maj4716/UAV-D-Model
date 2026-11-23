"""Run idea2 Stage-2 E3, E5 and C2 mechanism experiments.

The documented defaults are E3 16 x 300 x R4 and E5 9 x 300 x R4.  Noise
shape is optional in the document, so E5 defaults to the nine normal-noise
sigma/tau cells; pass all three distributions to obtain 27 cells.  ``--dry-run``
writes the bound design without executing the teacher.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.case_factory import CaseFactory, load_case_template
from damage_engine.engine import ENGINE_CODE_HASH, DamageEngine
from damage_engine.errors import InputValidationError
from damage_engine.label_spec import LabelSpec
from damage_engine.registry import PhysicsRegistry
from damage_engine.sampling_spec import SamplingSpec
from damage_engine.stage2_mechanism import (
    DIRECTION_NOISE_DISTRIBUTIONS,
    STAGE2_CONTRACT_VERSION,
    analyze_e3_factorial,
    build_c2_report,
    build_e3_variant,
    build_e5_variant,
    e3_factorial_design,
    e5_noise_design,
    extract_stage2_case_outcome,
    prepare_stage2_case,
    summarize_e5_variants,
    summarize_stage2_outcomes,
)
from damage_engine.utils import content_hash, file_hash, load_json


DEFAULT_TARGET = ROOT / "configs" / "targets" / "target.fragment_only.v1.json"
DEFAULT_PHYSICS = ROOT / "configs" / "physics" / "loitering_impact.json"
DEFAULT_BASE_CASE = (
    ROOT / "configs" / "cases" / "loitering_impact.fragment_only.v1.json"
)
DEFAULT_SAMPLING_SPEC = ROOT / "configs" / "sampling" / "p0_contact_reference.json"
DEFAULT_LABEL_SPEC = (
    ROOT / "configs" / "labels" / "loitering_impact.fragment_only.v1.json"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _append_log(path: Path, event: str, **fields: Any) -> None:
    record = {"timestamp": _utc_now(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def _iter_cases(path: Path) -> Iterable[dict[str, Any]]:
    name = path.name.lower()
    opener = gzip.open if name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        if name.endswith((".jsonl", ".jsonl.gz")):
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise InputValidationError(
                        "Stage-2 JSONL case must be an object",
                        context={"path": str(path), "line": line_number},
                    )
                yield value
            return
        value = json.load(stream)
    records = value if isinstance(value, list) else [value]
    for record in records:
        if not isinstance(record, dict):
            raise InputValidationError("Stage-2 case input must contain objects")
        yield record


def _load_base_cases(
    args: argparse.Namespace,
    baseline_engine: DamageEngine,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if args.max_cases < 1:
        raise InputValidationError("--max-cases must be positive")
    if args.cases is not None:
        path = Path(args.cases).resolve()
        cases = []
        for record in _iter_cases(path):
            cases.append(record)
            if len(cases) >= args.max_cases:
                break
        source = {
            "mode": "json_or_jsonl",
            "path": str(path),
            "file_hash": file_hash(path),
        }
    else:
        base_case_path = Path(args.base_case).resolve()
        sampling_path = Path(args.sampling_spec).resolve()
        label_path = Path(args.label_spec).resolve()
        sampling_spec = SamplingSpec.from_file(sampling_path)
        label_spec = LabelSpec.from_file(label_path)
        label_spec.validate_output_vocabulary(baseline_engine.damage_tree.outputs)
        plan = CaseFactory(
            load_case_template(base_case_path),
            sampling_spec,
            label_spec,
            target_hash=baseline_engine.target_hash,
            damage_tree_definition_hash=content_hash(
                baseline_engine.target_data["damage_tree"]
            ),
            physics_registry_hash=baseline_engine.registry.hash,
            physics_registry=baseline_engine.registry,
            engine_code_hash=ENGINE_CODE_HASH,
            compiled_target=baseline_engine.compiled_target,
        ).build_reference_plan()
        cases = []
        for record in plan.cases:
            role = record.get("metadata", {}).get("sampling", {}).get("role")
            if not args.include_reference_test and role == "reference_test":
                continue
            cases.append(record)
            if len(cases) >= args.max_cases:
                break
        source = {
            "mode": "generated_reference_plan",
            "base_case": str(base_case_path),
            "base_case_file_hash": file_hash(base_case_path),
            "sampling_spec": str(sampling_path),
            "sampling_spec_file_hash": file_hash(sampling_path),
            "sampling_spec_hash": sampling_spec.hash,
            "label_spec": str(label_path),
            "label_spec_file_hash": file_hash(label_path),
            "label_spec_hash": label_spec.hash,
            "reference_test_included": bool(args.include_reference_test),
        }
    if len(cases) != args.max_cases:
        raise InputValidationError(
            "Stage-2 case source did not provide the requested case count",
            context={"requested": args.max_cases, "available": len(cases)},
        )
    if len({str(case["case_id"]) for case in cases}) != len(cases):
        raise InputValidationError("Stage-2 base case IDs must be unique")
    return cases, source


def _common_configuration(
    args: argparse.Namespace,
    *,
    experiment: str,
    target_path: Path,
    physics_path: Path,
    case_source: dict[str, Any],
    design: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "contract_version": STAGE2_CONTRACT_VERSION,
        "experiment": experiment,
        "created_at": _utc_now(),
        "engine_code_hash": ENGINE_CODE_HASH,
        "target_path": str(target_path),
        "target_file_hash": file_hash(target_path),
        "physics_path": str(physics_path),
        "physics_file_hash": file_hash(physics_path),
        "case_source": case_source,
        "case_count": int(args.max_cases),
        "replicates": int(args.replicates),
        "common_random_numbers": {
            "enabled": True,
            "unit": "same ordered base cases and same aleatory_seed across variants",
        },
        "design": design,
        "dry_run": bool(args.dry_run),
        "scientific_status": (
            "software_mechanism_experiment_on_unvalidated_physics_not_external_validation"
        ),
    }


def _minimal_engine_record(
    result: dict[str, Any],
    outcome: dict[str, Any],
    *,
    variant_id: str,
    factors: dict[str, Any],
) -> dict[str, Any]:
    return {
        "case_id": result["case_id"],
        "target_id": result["target_id"],
        "variant_id": variant_id,
        "factors": factors,
        "status": result["status"],
        "damage_tree": result["damage_tree"],
        "component_state_trace": result["component_state_trace"],
        "stage2_outcome": outcome,
        "provenance": {
            "engine_code_hash": result["provenance"]["engine_code_hash"],
            "physics_registry_hash": result["provenance"]["physics_registry_hash"],
            "target_hash": result["provenance"]["target_hash"],
            "aleatory_seed": result["provenance"].get("aleatory_seed"),
        },
    }


def _execute_variant(
    *,
    output_root: Path,
    variant_id: str,
    factors: dict[str, Any],
    target: dict[str, Any],
    registry: PhysicsRegistry,
    base_cases: list[dict[str, Any]],
    replicates: int,
    direction_level: str,
    gurney_factor: float,
    log_path: Path,
) -> dict[str, Any]:
    variant_root = output_root / "variants" / variant_id
    variant_root.mkdir(parents=True, exist_ok=True)
    engine = DamageEngine(target, registry)
    outcomes: list[dict[str, Any]] = []
    started = perf_counter()
    trace_path = variant_root / "engine_results.minimal.jsonl.gz"
    with gzip.open(trace_path, "wt", encoding="utf-8") as stream:
        for index, base_case in enumerate(base_cases):
            case = prepare_stage2_case(
                base_case,
                target=target,
                target_hash=engine.target_hash,
                registry=registry,
                variant_id=variant_id,
                replicates=replicates,
                gurney_factor=gurney_factor,
                direction_level=direction_level,
                factors=factors,
            )
            result = engine.run(case).to_dict()
            outcome = extract_stage2_case_outcome(
                result,
                target,
                base_case_id=str(base_case["case_id"]),
            )
            outcomes.append(outcome)
            stream.write(
                json.dumps(
                    _minimal_engine_record(
                        result,
                        outcome,
                        variant_id=variant_id,
                        factors=factors,
                    ),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
            _append_log(
                log_path,
                "case_complete",
                variant_id=variant_id,
                case_index=index,
                base_case_id=base_case["case_id"],
                status=result["status"],
            )
    summary = summarize_stage2_outcomes(outcomes)
    record = {
        "variant_id": variant_id,
        "factors": factors,
        "target_hash": engine.target_hash,
        "physics_registry_hash": registry.hash,
        "trace_path": str(trace_path),
        "elapsed_seconds": perf_counter() - started,
        "summary": summary,
    }
    _write_json(variant_root / "summary.json", record)
    _append_log(
        log_path,
        "variant_complete",
        variant_id=variant_id,
        case_count=len(outcomes),
        elapsed_seconds=record["elapsed_seconds"],
    )
    return {"variant_id": variant_id, "factors": factors, "outcomes": outcomes}


def _run_e3(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "run.log"
    target_path = Path(args.target).resolve()
    physics_path = Path(args.physics).resolve()
    target = load_json(target_path)
    baseline_registry = PhysicsRegistry.from_file(physics_path)
    baseline_engine = DamageEngine(target, baseline_registry)
    cases, case_source = _load_base_cases(args, baseline_engine)
    design = e3_factorial_design()
    config = _common_configuration(
        args,
        experiment="E3",
        target_path=target_path,
        physics_path=physics_path,
        case_source=case_source,
        design=design,
    )
    config["bootstrap_replicates"] = int(args.bootstrap_replicates)
    config["bootstrap_seed"] = int(args.bootstrap_seed)
    config["tree_factor"] = (
        "offline current tree versus identity component-state outputs"
    )
    config["taylor_shapiro_policy"] = "independent_candidate_not_factor_level"
    _write_json(output / "config.json", config)
    _append_log(log_path, "experiment_start", experiment="E3", dry_run=args.dry_run)
    if args.dry_run:
        _write_json(
            output / "report.json",
            {
                "schema_version": "1.0.0",
                "experiment": "E3",
                "status": "dry_run_design_only",
                "physical_variant_count": len(design),
                "planned_teacher_case_runs": len(design) * len(cases),
            },
        )
        return 0

    variant_records = []
    for row in design:
        factors = row["factors"]
        variant_target, registry = build_e3_variant(
            target,
            baseline_registry.data,
            factors,
        )
        variant_records.append(
            _execute_variant(
                output_root=output,
                variant_id=row["variant_id"],
                factors=factors,
                target=variant_target,
                registry=registry,
                base_cases=cases,
                replicates=args.replicates,
                direction_level=factors["F_direction_model"],
                gurney_factor=factors["W_gurney_factor"],
                log_path=log_path,
            )
        )
    report = analyze_e3_factorial(
        variant_records,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    report["config_hash"] = content_hash(config)
    _write_json(output / "report.json", report)
    _append_log(log_path, "experiment_complete", experiment="E3")
    return 0


def _run_e5(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "run.log"
    target_path = Path(args.target).resolve()
    physics_path = Path(args.physics).resolve()
    target = load_json(target_path)
    baseline_registry = PhysicsRegistry.from_file(physics_path)
    baseline_engine = DamageEngine(target, baseline_registry)
    cases, case_source = _load_base_cases(args, baseline_engine)
    design = e5_noise_design(args.distributions)
    config = _common_configuration(
        args,
        experiment="E5",
        target_path=target_path,
        physics_path=physics_path,
        case_source=case_source,
        design=design,
    )
    config["direction_noise_distributions"] = list(args.distributions)
    config["equal_variance_noise_shapes"] = True
    config["taylor_shapiro_policy"] = "independent_candidate_not_factor_level"
    _write_json(output / "config.json", config)
    _append_log(log_path, "experiment_start", experiment="E5", dry_run=args.dry_run)
    if args.dry_run:
        _write_json(
            output / "report.json",
            {
                "schema_version": "1.0.0",
                "experiment": "E5",
                "status": "dry_run_design_only",
                "variant_count": len(design),
                "planned_teacher_case_runs": len(design) * len(cases),
            },
        )
        return 0

    variant_records = []
    for row in design:
        registry = build_e5_variant(baseline_registry.data, row["factors"])
        variant_records.append(
            _execute_variant(
                output_root=output,
                variant_id=row["variant_id"],
                factors=row["factors"],
                target=target,
                registry=registry,
                base_cases=cases,
                replicates=args.replicates,
                direction_level="legacy_hollow",
                gurney_factor=1.0,
                log_path=log_path,
            )
        )
    report = summarize_e5_variants(variant_records)
    report["config_hash"] = content_hash(config)
    _write_json(output / "report.json", report)
    _append_log(log_path, "experiment_complete", experiment="E5")
    return 0


def _run_c2(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    physics_path = Path(args.physics).resolve()
    registry = PhysicsRegistry.from_file(physics_path)
    config = {
        "schema_version": "1.0.0",
        "contract_version": STAGE2_CONTRACT_VERSION,
        "experiment": "C2",
        "created_at": _utc_now(),
        "engine_code_hash": ENGINE_CODE_HASH,
        "physics_path": str(physics_path),
        "physics_file_hash": file_hash(physics_path),
        "physics_registry_hash": registry.hash,
        "direction_repeats": int(args.direction_repeats),
        "v50_trials_per_speed": int(args.v50_trials_per_speed),
        "seed": int(args.seed),
    }
    _write_json(output / "config.json", config)
    report = build_c2_report(
        registry,
        direction_repeats=args.direction_repeats,
        v50_trials_per_speed=args.v50_trials_per_speed,
        seed=args.seed,
    )
    report["config_hash"] = content_hash(config)
    _write_json(output / "report.json", report)
    _append_log(output / "run.log", "experiment_complete", experiment="C2")
    return 0


def _add_case_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--physics", type=Path, default=DEFAULT_PHYSICS)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--base-case", type=Path, default=DEFAULT_BASE_CASE)
    parser.add_argument("--sampling-spec", type=Path, default=DEFAULT_SAMPLING_SPEC)
    parser.add_argument("--label-spec", type=Path, default=DEFAULT_LABEL_SPEC)
    parser.add_argument("--max-cases", type=int, default=300)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--include-reference-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run idea2 Stage-2 E3/E5 mechanism experiments or C2 gold checks."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    e3 = subparsers.add_parser("e3", help="2^4 physical factorial plus offline tree factor")
    _add_case_source_arguments(e3)
    e3.add_argument("--bootstrap-replicates", type=int, default=1000)
    e3.add_argument("--bootstrap-seed", type=int, default=2026112003)
    e3.set_defaults(handler=_run_e3)

    e5 = subparsers.add_parser("e5", help="sigma x tau and optional noise-shape grid")
    _add_case_source_arguments(e5)
    e5.add_argument(
        "--distributions",
        nargs="+",
        choices=DIRECTION_NOISE_DISTRIBUTIONS,
        default=["normal"],
        help=(
            "normal gives the documented 9-cell grid; add uniform/laplace for "
            "equal-variance noise-shape comparisons"
        ),
    )
    e5.set_defaults(handler=_run_e5)

    c2 = subparsers.add_parser("c2", help="Gurney, direction-angle and V50 gold checks")
    c2.add_argument("--physics", type=Path, default=DEFAULT_PHYSICS)
    c2.add_argument("--direction-repeats", type=int, default=256)
    c2.add_argument("--v50-trials-per-speed", type=int, default=4096)
    c2.add_argument("--seed", type=int, default=2026112007)
    c2.add_argument("--output", type=Path, required=True)
    c2.set_defaults(handler=_run_c2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
