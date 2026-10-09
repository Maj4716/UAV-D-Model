from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
import tempfile
import time

import numpy as np

from damage_engine import DamageEngine, PhysicsRegistry
from damage_engine.dataset import DatasetWriter
from damage_engine.utils import load_json


def build_cases(template: dict, count: int, replicates: int) -> list[dict]:
    cases: list[dict] = []
    for index in range(count):
        case = deepcopy(template)
        case["case_id"] = f"dataset-benchmark-{index:06d}"
        case["analysis"]["aleatory_seed"] = 20_000_000 + index
        case["analysis"]["replicates"] = replicates
        case["analysis"]["adaptive_ci_half_width"] = None
        case.setdefault("metadata", {})["split_group"] = f"condition-{index // 2:06d}"
        cases.append(case)
    return cases


def run_once(
    engine: DamageEngine,
    cases: list[dict],
    directory: Path,
    *,
    workers: int,
    compression: str,
) -> tuple[dict, float]:
    started = time.perf_counter()
    manifest = DatasetWriter(
        directory,
        split_seed=17,
        tensor_compression=compression,
    ).run_cases(engine, cases, workers=workers)
    return manifest, time.perf_counter() - started


def arrays_equal(first: Path, second: Path) -> bool:
    with np.load(first, allow_pickle=False) as left, np.load(second, allow_pickle=False) as right:
        if left.files != right.files:
            return False
        return all(np.array_equal(left[name], right[name]) for name in left.files)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark deterministic dataset generation.")
    parser.add_argument("--target", default="examples/target.demo.json")
    parser.add_argument("--case", default="examples/case.demo.json")
    parser.add_argument("--physics", default="examples/physics.demo.json")
    parser.add_argument("--cases", type=int, default=48)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--parallel-workers", type=int, default=2)
    arguments = parser.parse_args()
    if arguments.cases < 1 or arguments.replicates < 1 or arguments.parallel_workers < 1:
        parser.error("cases, replicates, and parallel-workers must be positive")

    engine = DamageEngine(
        load_json(arguments.target),
        PhysicsRegistry.from_file(arguments.physics),
    )
    cases = build_cases(load_json(arguments.case), arguments.cases, arguments.replicates)
    with tempfile.TemporaryDirectory(prefix="damage-dataset-benchmark-") as temporary:
        root = Path(temporary)
        serial_manifest, serial_seconds = run_once(
            engine, cases, root / "serial", workers=1, compression="compressed"
        )
        parallel_manifest, parallel_seconds = run_once(
            engine,
            cases,
            root / "parallel",
            workers=arguments.parallel_workers,
            compression="compressed",
        )
        uncompressed_manifest, uncompressed_seconds = run_once(
            engine,
            cases,
            root / "parallel-uncompressed",
            workers=arguments.parallel_workers,
            compression="none",
        )
        serial_tensor = root / "serial" / "tensors.npz"
        parallel_tensor = root / "parallel" / "tensors.npz"
        uncompressed_tensor = root / "parallel-uncompressed" / "tensors.npz"
        report = {
            "benchmark": "dataset_generation",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "case_count": arguments.cases,
            "replicates_per_case": arguments.replicates,
            "parallel_workers": arguments.parallel_workers,
            "serial_compressed": {
                "elapsed_s": serial_seconds,
                "cases_per_second": arguments.cases / serial_seconds,
                "tensor_bytes": serial_tensor.stat().st_size,
            },
            "parallel_compressed": {
                "elapsed_s": parallel_seconds,
                "cases_per_second": arguments.cases / parallel_seconds,
                "tensor_bytes": parallel_tensor.stat().st_size,
            },
            "parallel_uncompressed": {
                "elapsed_s": uncompressed_seconds,
                "cases_per_second": arguments.cases / uncompressed_seconds,
                "tensor_bytes": uncompressed_tensor.stat().st_size,
            },
            "parallel_speedup": serial_seconds / parallel_seconds,
            "serial_parallel_arrays_identical": arrays_equal(serial_tensor, parallel_tensor),
            "dataset_ids_identical": len(
                {
                    serial_manifest["dataset_id"],
                    parallel_manifest["dataset_id"],
                    uncompressed_manifest["dataset_id"],
                }
            )
            == 1,
        }
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
