"""Run fixed-prefix R=4/8/16/32/64/128 numerical convergence diagnostics.

The command changes only the fixed replicate count supplied for the current
invocation and the trace-capture
flag.  It never chooses a stopping point after inspecting outcomes.  The
report deliberately does not claim that the input cases were selected by a
pre-registered, outcome-blind or scientifically adequate stratified design;
those properties require a separately reviewed selection-plan artifact.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from damage_engine.case_factory import CaseFactory, load_case_template
from damage_engine.engine import ENGINE_CODE_HASH, DamageEngine
from damage_engine.errors import InputValidationError
from damage_engine.input_identity import freeze_case_inputs
from damage_engine.label_spec import LabelSpec
from damage_engine.registry import PhysicsRegistry
from damage_engine.replicate_convergence import replicate_convergence_report
from damage_engine.sampling_spec import SamplingSpec
from damage_engine.utils import content_hash, file_hash, load_json


DEFAULT_LEVELS = (4, 8, 16, 32, 64, 128)
DEFAULT_MINIMUM_CASES = 72


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--physics", required=True)
    cases_source = parser.add_mutually_exclusive_group(required=True)
    cases_source.add_argument("--cases")
    cases_source.add_argument("--sampling-run")
    cases_source.add_argument("--base-case")
    parser.add_argument("--sampling-spec")
    parser.add_argument("--label-spec")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-cases", type=int, default=DEFAULT_MINIMUM_CASES)
    parser.add_argument(
        "--minimum-cases",
        type=int,
        default=DEFAULT_MINIMUM_CASES,
        help=(
            "requested execution floor; values below the recommended 72 must "
            "be supplied explicitly and remain exploratory"
        ),
    )
    parser.add_argument("--replicates", type=int, default=DEFAULT_LEVELS[-1])
    parser.add_argument("--levels", nargs="+", type=int, default=list(DEFAULT_LEVELS))
    parser.add_argument("--ci-half-width-tolerance", type=float, default=0.05)
    parser.add_argument("--decision-threshold", nargs="+", type=float, default=[0.5])
    parser.add_argument(
        "--threshold-neighborhood-half-width",
        nargs="+",
        type=float,
        default=[0.1],
    )
    parser.add_argument("--sampling-role")
    parser.add_argument("--sampling-stratum")
    return parser


def _validate_plan_configuration(
    *, max_cases: int, minimum_cases: int, replicates: int, levels
) -> tuple[int, ...]:
    if max_cases < 1 or minimum_cases < 1:
        raise InputValidationError("max-cases and minimum-cases must be positive")
    if max_cases < minimum_cases:
        raise InputValidationError(
            "max-cases cannot be smaller than the requested minimum-cases"
        )
    prefixes = tuple(int(level) for level in levels)
    if not prefixes or tuple(sorted(set(prefixes))) != prefixes or prefixes[0] < 2:
        raise InputValidationError(
            "levels must be unique increasing fixed prefixes of at least two"
        )
    if replicates != prefixes[-1]:
        raise InputValidationError("replicates must equal the largest fixed prefix")
    return prefixes


def _build_case_selection_disclosure(
    *,
    cases_mode: str,
    sampling_role: str | None,
    sampling_stratum: str | None,
    minimum_cases: int,
    selected_case_count: int,
) -> dict:
    """Describe only selection properties established by this invocation.

    The tool can establish that it does not read newly simulated outcomes when
    applying its optional metadata filters and stream-order limit.  It cannot
    establish how an external case file or sampling run was created, whether a
    design was registered before any outcomes were inspected, or whether the
    selected prefix satisfies the intended crossed-stratum quotas.
    """

    return {
        "source_mode": cases_mode,
        "algorithm_in_this_invocation": (
            "stream_order_prefix_after_optional_sampling_metadata_filters"
        ),
        "selection_fields_read_in_this_invocation": [
            name
            for name, value in (
                ("metadata.sampling.role", sampling_role),
                ("metadata.sampling.stratum", sampling_stratum),
            )
            if value is not None
        ],
        "new_simulation_outcomes_read_for_selection": False,
        "upstream_outcome_blindness": "not_verified_by_this_report",
        "external_preregistration": "not_verified_by_this_report",
        "required_stratum_quotas": "not_supplied_or_verified_by_this_report",
        "formal_design_admission": {
            "status": "not_established_by_this_report",
            "reason_codes": [
                "no_bound_preregistered_selection_plan",
                "no_verified_crossed_stratum_quota_table",
                "upstream_case_selection_provenance_not_established",
            ],
        },
        "sampling_role_filter": sampling_role,
        "sampling_stratum_filter": sampling_stratum,
        "requested_minimum_case_count": minimum_cases,
        "selected_case_count": selected_case_count,
        "requested_minimum_case_count_met": selected_case_count >= minimum_cases,
        "recommended_minimum_case_count": DEFAULT_MINIMUM_CASES,
        "recommended_minimum_case_count_met": (
            selected_case_count >= DEFAULT_MINIMUM_CASES
        ),
        "requested_minimum_below_recommendation": (
            minimum_cases < DEFAULT_MINIMUM_CASES
        ),
        "interpretation": (
            "The case count is an execution floor only. Meeting it does not "
            "establish preregistration, outcome-blind upstream construction, "
            "stratum balance, or formal scientific admission."
        ),
    }


def _iter_cases(path: Path):
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise InputValidationError(
                        "invalid convergence-study case JSONL",
                        context={"line": line_number, "path": str(path)},
                    ) from exc
                if not isinstance(value, dict):
                    raise InputValidationError("convergence-study cases must be objects")
                yield value
        return
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    records = value if isinstance(value, list) else [value]
    for record in records:
        if not isinstance(record, dict):
            raise InputValidationError("convergence-study cases must be objects")
        yield record


def main() -> int:
    arguments = _build_parser().parse_args()
    arguments.levels = list(
        _validate_plan_configuration(
            max_cases=arguments.max_cases,
            minimum_cases=arguments.minimum_cases,
            replicates=arguments.replicates,
            levels=arguments.levels,
        )
    )

    target_path = Path(arguments.target).resolve()
    physics_path = Path(arguments.physics).resolve()
    target = load_json(target_path)
    engine = DamageEngine(target, PhysicsRegistry.from_file(physics_path))
    if arguments.base_case:
        if not arguments.sampling_spec or not arguments.label_spec:
            raise InputValidationError(
                "base-case mode requires sampling-spec and label-spec"
            )
        base_case_path = Path(arguments.base_case).resolve()
        sampling_spec_path = Path(arguments.sampling_spec).resolve()
        label_spec_path = Path(arguments.label_spec).resolve()
        sampling_spec = SamplingSpec.from_file(sampling_spec_path)
        label_spec = LabelSpec.from_file(label_spec_path)
        label_spec.validate_output_vocabulary(engine.damage_tree.outputs)
        plan = CaseFactory(
            load_case_template(base_case_path),
            sampling_spec,
            label_spec,
            target_hash=engine.target_hash,
            damage_tree_definition_hash=content_hash(engine.target_data["damage_tree"]),
            physics_registry_hash=engine.registry.hash,
            physics_registry=engine.registry,
            engine_code_hash=ENGINE_CODE_HASH,
        ).build_reference_plan()
        sampling_run = None
        cases_path = base_case_path
        case_iterator = iter(plan.cases)
        cases_mode = "generated_reference_plan_from_base_case_and_specs"
        design_inputs = {
            "sampling_spec_path": str(sampling_spec_path),
            "sampling_spec_file_hash": file_hash(sampling_spec_path),
            "sampling_spec_hash": sampling_spec.hash,
            "label_spec_path": str(label_spec_path),
            "label_spec_file_hash": file_hash(label_spec_path),
            "label_spec_hash": label_spec.hash,
        }
    elif arguments.sampling_run:
        sampling_run = Path(arguments.sampling_run).resolve()
        from damage_engine.sampling_checkpoint import (
            iter_reference_formal_cases,
            reference_formal_cases_path,
        )

        cases_path = reference_formal_cases_path(sampling_run)
        case_iterator = iter_reference_formal_cases(sampling_run)
        cases_mode = "sampling_run_reference_case_stream"
        design_inputs = {}
    else:
        sampling_run = None
        cases_path = Path(arguments.cases).resolve()
        case_iterator = _iter_cases(cases_path)
        cases_mode = "json_or_jsonl"
        design_inputs = {}
    cases = []
    for original in case_iterator:
        sampling = original.get("metadata", {}).get("sampling", {})
        if (
            arguments.sampling_role
            and sampling.get("role") != arguments.sampling_role
        ):
            continue
        if (
            arguments.sampling_stratum
            and sampling.get("stratum") != arguments.sampling_stratum
        ):
            continue
        case = deepcopy(original)
        analysis = case.setdefault("analysis", {})
        analysis["replicates"] = arguments.replicates
        analysis["adaptive_ci_half_width"] = None
        analysis.pop("adaptive_physical_ci", None)
        analysis["capture_replicate_trace"] = True
        cases.append(case)
        if len(cases) >= arguments.max_cases:
            break
    if not cases:
        raise InputValidationError("convergence-study case input is empty")
    if len(cases) < arguments.minimum_cases:
        raise InputValidationError(
            "convergence-study case count is below the requested minimum",
            context={
                "selected_case_count": len(cases),
                "minimum_cases": arguments.minimum_cases,
            },
        )

    traces = []
    case_ids = []
    output_names = None
    started = perf_counter()
    with freeze_case_inputs(cases, cases_base_directory=cases_path.parent) as frozen:
        for case in frozen:
            result = engine.run(case, fail_closed=True).to_dict()
            records = result.get("replicate_trace", {}).get("records")
            if not isinstance(records, list) or len(records) != arguments.replicates:
                raise InputValidationError("engine did not return the requested replicate trace")
            observed_names = list(records[0]["system_outputs"])
            if output_names is None:
                output_names = observed_names
            elif observed_names != output_names:
                raise InputValidationError("system output schema changed within convergence study")
            traces.append(
                [
                    [float(record["system_outputs"][name]) for name in output_names]
                    for record in records
                ]
            )
            case_ids.append(str(result["case_id"]))
    elapsed = perf_counter() - started
    assert output_names is not None
    report = replicate_convergence_report(
        np.asarray(traces, dtype=np.float64),
        levels=arguments.levels,
        ci_half_width_tolerance=arguments.ci_half_width_tolerance,
        decision_threshold=arguments.decision_threshold,
        threshold_neighborhood_half_width=(
            arguments.threshold_neighborhood_half_width
        ),
    )
    report.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "case_ids": case_ids,
            "system_output_names": output_names,
            "requested_replicates": arguments.replicates,
            "claim_status": {
                "numerical_fixed_prefix_convergence": "computed",
                "formal_preregistered_stratified_study": (
                    "not_established_by_this_report"
                ),
                "scientific_admission": "not_established_by_this_report",
            },
            "case_selection": _build_case_selection_disclosure(
                cases_mode=cases_mode,
                sampling_role=arguments.sampling_role,
                sampling_stratum=arguments.sampling_stratum,
                minimum_cases=arguments.minimum_cases,
                selected_case_count=len(case_ids),
            ),
            "elapsed_seconds": elapsed,
            "cases_per_second": len(case_ids) / elapsed,
            "engine": {
                "version": result["provenance"]["engine_version"],
                "code_hash": result["provenance"]["engine_code_hash"],
                "target_hash": engine.target_hash,
                "physics_registry_hash": engine.registry.hash,
            },
            "inputs": {
                "target_path": str(target_path),
                "target_file_hash": file_hash(target_path),
                "physics_path": str(physics_path),
                "physics_file_hash": file_hash(physics_path),
                "cases_path": str(cases_path),
                "cases_file_hash": file_hash(cases_path),
                "cases_mode": cases_mode,
                "sampling_checkpoint_hash": (
                    file_hash(sampling_run / "sampling_checkpoint.json")
                    if sampling_run is not None
                    else None
                ),
                **design_inputs,
            },
        }
    )
    destination = Path(arguments.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
