from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
import statistics
import time

from damage_engine import DamageEngine, PhysicsRegistry
from damage_engine.utils import load_json


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build_case(template: dict, index: int, replicates: int) -> dict:
    case = deepcopy(template)
    case["case_id"] = f"benchmark-{index:06d}"
    case["analysis"]["aleatory_seed"] = 10_000_000 + index
    case["analysis"]["replicates"] = replicates
    case["analysis"]["adaptive_ci_half_width"] = None
    return case


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure deterministic single-process engine throughput."
    )
    parser.add_argument("--target", default="examples/target.demo.json")
    parser.add_argument("--case", default="examples/case.demo.json")
    parser.add_argument("--physics", default="examples/physics.demo.json")
    parser.add_argument("--cases", type=int, default=40)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    arguments = parser.parse_args()
    if arguments.cases < 1 or arguments.replicates < 1 or arguments.warmup < 0:
        parser.error("cases and replicates must be positive; warmup cannot be negative")

    target = load_json(Path(arguments.target))
    template = load_json(Path(arguments.case))
    registry = PhysicsRegistry.from_file(Path(arguments.physics))
    engine = DamageEngine(target, registry)

    for index in range(arguments.warmup):
        engine.run(build_case(template, -(index + 1), arguments.replicates))

    durations: list[float] = []
    start = time.perf_counter()
    for index in range(arguments.cases):
        case_start = time.perf_counter()
        engine.run(build_case(template, index, arguments.replicates))
        durations.append(time.perf_counter() - case_start)
    elapsed = time.perf_counter() - start

    report = {
        "benchmark": "single_process_engine",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "case_count": arguments.cases,
        "replicates_per_case": arguments.replicates,
        "elapsed_s": elapsed,
        "cases_per_second": arguments.cases / elapsed,
        "replicates_per_second": arguments.cases * arguments.replicates / elapsed,
        "latency_s": {
            "median": statistics.median(durations),
            "p90": percentile(durations, 0.90),
            "maximum": max(durations),
        },
        "target_hash": engine.target_hash,
        "registry_hash": registry.hash,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
