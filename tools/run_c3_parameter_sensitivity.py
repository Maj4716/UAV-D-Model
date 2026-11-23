#!/usr/bin/env python
"""Run idea2 C3 Morris-proxy and Sobol-teacher sensitivity experiments."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.engine import ENGINE_CODE_HASH, ENGINE_VERSION  # noqa: E402
from damage_engine.sensitivity import (  # noqa: E402
    analyze_morris,
    analyze_sobol,
    intervals_from_document,
    morris_design,
    sobol_saltelli_design,
)
from damage_engine.sensitivity_execution import (  # noqa: E402
    C3_PARAMETER_IDS,
    PROXY_OUTPUT_NAMES,
    TEACHER_OUTPUT_NAMES,
    evaluate_proxy_point,
    evaluate_teacher_point,
    parameter_point,
)
from damage_engine.utils import content_hash, file_hash  # noqa: E402


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_non_test_cases(path: Path, count: int) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    cases: list[dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            case = json.loads(line)
            role = (case.get("metadata") or {}).get("sampling", {}).get("role")
            if role == "reference_test":
                continue
            if role not in {None, "reference_train", "reference_validation"}:
                continue
            cases.append(case)
            if len(cases) >= count:
                break
    if len(cases) != count:
        raise ValueError(f"requested {count} non-test cases, found {len(cases)}")
    return cases


def _jsonable_design(design: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in design.items()
    }


def _morris_ranking(
    analysis: dict[str, Any],
    intervals,
    outputs: np.ndarray,
    output_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    scales = np.ptp(outputs, axis=0)
    fallback = np.maximum(np.max(np.abs(outputs), axis=0), 1.0)
    scales = np.where(scales > 1.0e-12, scales, fallback)
    span = {item.parameter_id: item.upper - item.lower for item in intervals}
    rows = []
    for row in analysis["parameters"]:
        normalized = (
            np.asarray(row["mu_star"], dtype=np.float64)
            * span[row["parameter_id"]]
            / scales
        )
        rows.append(
            {
                "parameter_id": row["parameter_id"],
                "aggregate_normalized_mu_star": float(np.mean(normalized)),
                "normalized_mu_star_by_output": {
                    name: float(value)
                    for name, value in zip(output_names, normalized, strict=True)
                },
            }
        )
    rows.sort(key=lambda item: item["aggregate_normalized_mu_star"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _sobol_ranking(
    analysis: dict[str, Any], output_names: tuple[str, ...]
) -> list[dict[str, Any]]:
    rows = []
    for row in analysis["parameters"]:
        total = np.asarray(row["total_order"], dtype=np.float64)
        rows.append(
            {
                "parameter_id": row["parameter_id"],
                "aggregate_total_order": float(np.mean(total)),
                "total_order_by_output": {
                    name: float(value)
                    for name, value in zip(output_names, total, strict=True)
                },
            }
        )
    rows.sort(key=lambda item: item["aggregate_total_order"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(
            "artifacts/validation/sampling_sphere_v1/no_bottom_impact_state_pilot/"
            "rounds/round-000/formal_cases.jsonl.gz"
        ),
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
    parser.add_argument(
        "--intervals",
        type=Path,
        default=Path("configs/uncertainty/idea2_parameter_intervals.v1.json"),
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--case-count", type=int, default=2)
    parser.add_argument("--morris-trajectories", type=int, default=4)
    parser.add_argument("--morris-levels", type=int, default=8)
    parser.add_argument("--morris-seed", type=int, default=2026112022)
    parser.add_argument("--proxy-source-seed", type=int, default=2026112023)
    parser.add_argument("--proxy-azimuths-per-ring", type=int, default=4)
    parser.add_argument("--sobol-base-size", type=int, default=8)
    parser.add_argument("--sobol-seed", type=int, default=2026112024)
    parser.add_argument("--teacher-replicates", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.case_count < 1 or args.teacher_replicates < 1:
        raise ValueError("case count and teacher replicates must be positive")
    output = args.output_directory
    result_path = output / "results.json"
    if result_path.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {result_path}")

    target = _load_json(args.target)
    registry = _load_json(args.physics)
    interval_document = _load_json(args.intervals)
    intervals = intervals_from_document(interval_document)
    if tuple(item.parameter_id for item in intervals) != C3_PARAMETER_IDS:
        raise ValueError("interval document does not match the C3 parameter contract")
    cases = _load_non_test_cases(args.cases, args.case_count)

    morris = morris_design(
        intervals,
        trajectories=args.morris_trajectories,
        levels=args.morris_levels,
        seed=args.morris_seed,
    )
    morris_outputs = np.vstack(
        [
            evaluate_proxy_point(
                target,
                registry,
                cases,
                parameter_point(morris["parameter_ids"], row),
                source_seed=args.proxy_source_seed,
                azimuths_per_ring=args.proxy_azimuths_per_ring,
            )
            for row in morris["samples"]
        ]
    )
    morris_analysis = analyze_morris(morris, morris_outputs)

    sobol = sobol_saltelli_design(
        intervals, base_size=args.sobol_base_size, seed=args.sobol_seed
    )
    sobol_outputs = np.vstack(
        [
            evaluate_teacher_point(
                target,
                registry,
                cases,
                parameter_point(sobol["parameter_ids"], row),
                replicates=args.teacher_replicates,
            )
            for row in sobol["samples"]
        ]
    )
    sobol_analysis = analyze_sobol(sobol, sobol_outputs)

    config = {
        "schema_version": "idea2-c3-experiment-config@1.0.0",
        "cases": str(args.cases),
        "case_count": args.case_count,
        "selected_case_ids": [str(case["case_id"]) for case in cases],
        "selected_case_roles": [
            (case.get("metadata") or {}).get("sampling", {}).get("role")
            for case in cases
        ],
        "reference_test_excluded": True,
        "target": str(args.target),
        "physics": str(args.physics),
        "intervals": str(args.intervals),
        "morris_trajectories": args.morris_trajectories,
        "morris_levels": args.morris_levels,
        "morris_seed": args.morris_seed,
        "proxy_source_seed": args.proxy_source_seed,
        "proxy_azimuths_per_ring": args.proxy_azimuths_per_ring,
        "sobol_base_size": args.sobol_base_size,
        "sobol_seed": args.sobol_seed,
        "teacher_replicates": args.teacher_replicates,
        "engine_version": ENGINE_VERSION,
        "engine_code_hash": ENGINE_CODE_HASH,
        "input_hashes": {
            "cases_file": file_hash(args.cases),
            "target_file": file_hash(args.target),
            "physics_file": file_hash(args.physics),
            "interval_file": file_hash(args.intervals),
        },
        "formal_reference_budget": {
            "morris": "screen with a larger proxy trajectory set",
            "sobol": "teacher base size selected after Morris, with convergence checks",
        },
        "scaled_run_deviation": (
            "Configured CLI budgets may be smaller than the formal study. Rankings from "
            "a scaled run are software evidence only."
        ),
    }
    config["config_hash"] = content_hash(config)
    document = {
        "schema_version": "idea2-c3-sensitivity-results@1.0.0",
        "config": config,
        "interval_set": interval_document,
        "morris_proxy": {
            "output_names": list(PROXY_OUTPUT_NAMES),
            "design": _jsonable_design(morris),
            "outputs": morris_outputs.tolist(),
            "analysis": morris_analysis,
            "ranking": _morris_ranking(
                morris_analysis, intervals, morris_outputs, PROXY_OUTPUT_NAMES
            ),
        },
        "sobol_teacher": {
            "output_names": list(TEACHER_OUTPUT_NAMES),
            "design": _jsonable_design(sobol),
            "outputs": sobol_outputs.tolist(),
            "analysis": sobol_analysis,
            "ranking": _sobol_ranking(sobol_analysis, TEACHER_OUTPUT_NAMES),
        },
        "evidence_boundary": {
            "interval_semantics": "exploratory scenario envelope, not calibrated probability",
            "morris_role": "FastSamplingProxy screening only; never a formal label",
            "sobol_role": "software-teacher sensitivity under the unverified registry",
            "categorical_direction_family": (
                "legacy Taylor, solid cone, and Taylor-Shapiro are handled by factorial "
                "experiments, not by continuous Morris/Sobol coordinates"
            ),
            "common_random_numbers": (
                "Each selected physical case retains the same aleatory_seed across all "
                "parameter samples."
            ),
            "writeback": "No parameter is written back to the target or physics registry.",
            "physical_validation": False,
        },
    }
    document["result_hash"] = content_hash(document)
    output.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "run.log").write_text(
        "C3 sensitivity completed\n"
        f"morris_proxy_evaluations={len(morris_outputs)}\n"
        f"sobol_teacher_evaluations={len(sobol_outputs)}\n"
        f"reference_test_excluded=true\nresult_hash={document['result_hash']}\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# C3 parameter sensitivity\n\n"
        "Morris uses the outcome-blind FastSamplingProxy; Sobol uses the real software "
        "teacher with common random numbers. The configured intervals are exploratory "
        "scenario bounds, and this run does not update the physics registry. See "
        "`results.json` for hashes, rankings, and the evidence boundary.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"output": str(result_path), "result_hash": document["result_hash"]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
