"""Reuse frozen proxy-validation cases to test higher angular resolutions."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.engine import ENGINE_CODE_HASH, DamageEngine  # noqa: E402
from damage_engine.fast_sampling_proxy import FastSamplingProxy  # noqa: E402
from damage_engine.registry import PhysicsRegistry  # noqa: E402
from damage_engine.schema_resources import schema_tree_hash  # noqa: E402
from damage_engine.utils import content_hash, source_tree_hash  # noqa: E402
from evaluate_fast_sampling_proxy import (  # noqa: E402
    _accuracy_report,
    _convergence_report,
    _load_json,
    _write_json,
    _write_jsonl_gz,
)


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-directory", type=Path, required=True)
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
    parser.add_argument("--azimuths-per-ring", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026111410)
    return parser


def main() -> int:
    args = _parser().parse_args()
    evidence = args.evidence_directory
    base_evaluation = _load_json(evidence / "evaluation.json")
    recorded_engine_hash = str(base_evaluation["inputs"]["engine_code_hash"])
    if recorded_engine_hash != ENGINE_CODE_HASH:
        raise RuntimeError(
            "refusing to compare proxy resolutions across engine revisions: "
            f"base={recorded_engine_hash}, current={ENGINE_CODE_HASH}"
        )
    validation_rows = _read_jsonl_gz(evidence / "validation_cases_frozen.jsonl.gz")
    teacher_rows = _read_jsonl_gz(evidence / "teacher_metrics.jsonl.gz")
    coarse_rows = _read_jsonl_gz(evidence / "proxy_rows.jsonl.gz")
    case_ids = [str(row["candidate_id"]) for row in validation_rows]
    cases = [row["case"] for row in validation_rows]
    coarse_by_id = {
        row["candidate_id"]: row for row in coarse_rows if row["candidate_id"] in set(case_ids)
    }

    engine = DamageEngine(
        _load_json(args.target),
        PhysicsRegistry(_load_json(args.physics)),
    )
    teacher_seconds_per_case = float(
        base_evaluation["timing"]["teacher_seconds_per_case"]
    )
    summary: dict[str, Any] = {
        "schema_version": "fast-sampling-resolution-refinement@1.0.0",
        "status": "complete",
        "case_count": len(cases),
        "teacher_results_reused": True,
        "teacher_seconds_per_case": teacher_seconds_per_case,
        "resolutions": {},
        "evidence_boundary": base_evaluation["evidence_boundary"],
    }
    previous_by_id = coarse_by_id
    previous_resolution = int(base_evaluation["design"].get("coarse_azimuths_per_ring", 2))
    for offset, resolution in enumerate(args.azimuths_per_ring):
        output_rows = evidence / f"resolution_{resolution:03d}_proxy_rows.jsonl.gz"
        output_summary = evidence / f"resolution_{resolution:03d}_evaluation.json"
        if output_rows.exists() or output_summary.exists():
            raise FileExistsError(f"refusing to overwrite resolution {resolution} evidence")
        proxy = FastSamplingProxy(engine, azimuths_per_ring=resolution)
        started = perf_counter()
        rows: list[dict[str, Any]] = []
        for candidate_id, case in zip(case_ids, cases, strict=True):
            result = proxy.evaluate(case)
            result["candidate_id"] = candidate_id
            rows.append(result)
        elapsed = perf_counter() - started
        by_id = {row["candidate_id"]: row for row in rows}
        accuracy = _accuracy_report(
            by_id,
            teacher_rows,
            bootstrap_draws=args.bootstrap_draws,
            bootstrap_seed=args.bootstrap_seed + offset * 100,
        )
        convergence = _convergence_report(previous_by_id, by_id)
        seconds_per_case = elapsed / len(rows)
        current = {
            "azimuths_per_ring": resolution,
            "elapsed_seconds": elapsed,
            "seconds_per_case": seconds_per_case,
            "speedup_vs_teacher": teacher_seconds_per_case / seconds_per_case,
            "accuracy": accuracy,
            "convergence_from_previous": {
                "previous_azimuths_per_ring": previous_resolution,
                **convergence,
            },
        }
        _write_jsonl_gz(output_rows, rows)
        _write_json(output_summary, current)
        summary["resolutions"][str(resolution)] = current
        previous_by_id = by_id
        previous_resolution = resolution
    live_engine_code_hash = content_hash(
        {
            "python_source_tree_hash": source_tree_hash(ROOT / "damage_engine"),
            "schema_tree_hash": schema_tree_hash(),
        }
    )
    summary["engine_code_hash"] = ENGINE_CODE_HASH
    summary["engine_code_hash_at_end"] = live_engine_code_hash
    summary["source_stable_during_run"] = live_engine_code_hash == ENGINE_CODE_HASH
    if not summary["source_stable_during_run"]:
        summary["status"] = "invalid_source_drift"
    _write_json(evidence / "resolution_refinement_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
