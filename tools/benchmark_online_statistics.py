"""Measure default online aggregation against explicit raw-trace retention."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import sys
from time import perf_counter
import tracemalloc
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from damage_engine.engine import ENGINE_CODE_HASH, DamageEngine
from damage_engine.registry import PhysicsRegistry
from damage_engine.schema_resources import schema_tree_hash
from damage_engine.utils import content_hash, file_hash, load_json, source_tree_hash


BENCHMARK_TOOL_PATH = Path(__file__).resolve()


def _quantile(values: list[float], probability: float) -> float:
    """Linear-interpolated sample quantile for small benchmark repetitions."""

    if not values:
        raise ValueError("at least one value is required")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution_summary(values: list[float]) -> dict[str, float | int]:
    """Summarize repeated timings or Python-allocation peaks robustly."""

    q1 = _quantile(values, 0.25)
    median = _quantile(values, 0.50)
    q3 = _quantile(values, 0.75)
    return {
        "count": len(values),
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "minimum": min(values),
        "maximum": max(values),
    }


def execution_order(round_index: int) -> tuple[str, str]:
    """Alternate mode order to reduce deterministic first/second-run bias."""

    return (
        ("online", "raw_trace")
        if round_index % 2 == 0
        else ("raw_trace", "online")
    )


def _source_identity_snapshot(paths: dict[str, Path]) -> dict[str, str]:
    return {
        "engine_code_hash": content_hash(
            {
                "python_source_tree_hash": source_tree_hash(
                    REPOSITORY_ROOT / "damage_engine"
                ),
                "schema_tree_hash": schema_tree_hash(),
            }
        ),
        "benchmark_tool_hash": file_hash(BENCHMARK_TOOL_PATH),
        **{f"{name}_hash": file_hash(path) for name, path in paths.items()},
    }


def _source_identity_drift(
    start: dict[str, str], end: dict[str, str]
) -> list[str]:
    return sorted(
        key for key in set(start) | set(end) if start.get(key) != end.get(key)
    )


def _run(engine: DamageEngine, case: dict, *, capture_trace: bool) -> dict:
    configured = deepcopy(case)
    configured["analysis"]["capture_replicate_trace"] = capture_trace
    configured["analysis"]["capture_fragment_audit"] = False
    configured["analysis"]["adaptive_ci_half_width"] = None
    gc.collect()
    tracemalloc.start()
    started = perf_counter()
    result = engine.run(configured).to_dict()
    elapsed = perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    scientific_digest = content_hash(
        {
            "armor": result["armor"],
            "components": result["components"],
            "damage_tree": result["damage_tree"],
            "summary": result["summary"],
        }
    )
    return {
        "elapsed_seconds": elapsed,
        "tracemalloc_peak_bytes": peak_bytes,
        "scientific_output_digest": scientific_digest,
        "storage": result["replicate_statistics_storage"],
        "replicate_trace_present": "replicate_trace" in result,
    }


def _summarize_mode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    digests = sorted({str(row["scientific_output_digest"]) for row in rows})
    storage = [row["storage"] for row in rows]
    trace_presence = [bool(row["replicate_trace_present"]) for row in rows]
    return {
        "run_count": len(rows),
        "elapsed_seconds": _distribution_summary(
            [float(row["elapsed_seconds"]) for row in rows]
        ),
        "tracemalloc_peak_bytes": _distribution_summary(
            [float(row["tracemalloc_peak_bytes"]) for row in rows]
        ),
        "scientific_output_digest": digests[0] if len(digests) == 1 else None,
        "scientific_output_consistent": len(digests) == 1,
        "storage_consistent": all(value == storage[0] for value in storage),
        "storage": storage[0],
        "replicate_trace_present_consistent": all(
            value == trace_presence[0] for value in trace_presence
        ),
        "replicate_trace_present": trace_presence[0],
        "runs": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--physics", required=True)
    parser.add_argument("--replicates", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    if arguments.replicates < 2:
        raise ValueError("replicates must be at least two")
    if arguments.rounds < 3:
        raise ValueError("rounds must be at least three")
    if arguments.warmup_runs < 1:
        raise ValueError("warmup-runs must be at least one")

    target_path = Path(arguments.target).resolve()
    case_path = Path(arguments.case).resolve()
    physics_path = Path(arguments.physics).resolve()
    output_path = Path(arguments.output).resolve()
    source_paths = {
        "target": target_path,
        "case": case_path,
        "physics": physics_path,
    }
    source_identity_at_start = _source_identity_snapshot(source_paths)
    target = load_json(target_path)
    case = load_json(case_path)
    case["analysis"]["replicates"] = arguments.replicates
    engine = DamageEngine(target, PhysicsRegistry.from_file(physics_path))

    mode_configuration = {"online": False, "raw_trace": True}
    for warmup_index in range(arguments.warmup_runs):
        for mode in execution_order(warmup_index):
            _run(engine, case, capture_trace=mode_configuration[mode])

    run_rows: dict[str, list[dict[str, Any]]] = {"online": [], "raw_trace": []}
    execution_schedule = []
    for round_index in range(arguments.rounds):
        order = execution_order(round_index)
        execution_schedule.append(list(order))
        for mode in order:
            row = _run(engine, case, capture_trace=mode_configuration[mode])
            row["round_index"] = round_index
            row["execution_position"] = order.index(mode)
            run_rows[mode].append(row)

    online = _summarize_mode(run_rows["online"])
    trace = _summarize_mode(run_rows["raw_trace"])
    equivalent = (
        online["scientific_output_consistent"]
        and trace["scientific_output_consistent"]
        and online["scientific_output_digest"] == trace["scientific_output_digest"]
    )
    source_identity_at_end = _source_identity_snapshot(source_paths)
    changed_source_identities = _source_identity_drift(
        source_identity_at_start, source_identity_at_end
    )
    source_stable_during_run = (
        not changed_source_identities
        and source_identity_at_start["engine_code_hash"] == ENGINE_CODE_HASH
    )
    payload = {
        "schema_version": "1.1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "engine_code_hash": ENGINE_CODE_HASH,
        "source_identity_at_start": source_identity_at_start,
        "source_identity_at_end": source_identity_at_end,
        "changed_source_identities": changed_source_identities,
        "source_stable_during_run": source_stable_during_run,
        "status": "complete" if source_stable_during_run else "invalid_source_drift",
        "interpretation_status": "software_performance_measurement_not_physical_validation",
        "inputs": {
            "target": str(target_path),
            "target_hash": source_identity_at_start["target_hash"],
            "case": str(case_path),
            "case_hash": source_identity_at_start["case_hash"],
            "physics": str(physics_path),
            "physics_hash": source_identity_at_start["physics_hash"],
            "replicates": arguments.replicates,
            "rounds": arguments.rounds,
            "warmup_runs": arguments.warmup_runs,
        },
        "execution_schedule": execution_schedule,
        "online": online,
        "raw_trace": trace,
        "scientific_outputs_equal": equivalent,
        "median_peak_memory_ratio_raw_trace_over_online": (
            trace["tracemalloc_peak_bytes"]["median"]
            / max(1.0, online["tracemalloc_peak_bytes"]["median"])
        ),
        "notes": [
            "tracemalloc measures incremental Python allocations, not total process RSS",
            "timing and allocation summaries report median and IQR over repeated alternating-order runs after warm-up",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if equivalent and source_stable_during_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
