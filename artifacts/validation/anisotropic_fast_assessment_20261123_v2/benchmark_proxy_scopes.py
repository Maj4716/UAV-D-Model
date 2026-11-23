"""Benchmark full versus exposure-only FastSamplingProxy on development inputs."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.engine import ENGINE_CODE_HASH, DamageEngine
from damage_engine.fast_sampling_proxy import FastSamplingProxy
from damage_engine.registry import PhysicsRegistry
from damage_engine.utils import file_hash, source_tree_hash


TARGET = ROOT / "configs/targets/target.fragment_only.v1.json"
PHYSICS = ROOT / "configs/physics/loitering_impact.taylor_shapiro.obliquity_bound.v3.json"
CASES = (
    ROOT
    / "artifacts/datasets/recovery_role_r8_r32_train2000_eval300_20261122"
    / "rounds/round-000/formal_cases.jsonl.gz"
)
FIELDS = (
    "expected_armor_hits",
    "expected_armor_penetrations",
    "expected_total_component_hits",
    "expected_total_component_penetrations",
    "expected_total_component_absorbed_energy_j",
    "expected_total_component_damage_energy_j",
    "out_of_domain_packet_weight",
    "proxy_domain_status",
    "component_metrics",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("performance_benchmark.json"),
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--case-count", type=int, default=48)
    args = parser.parse_args()
    target = json.loads(TARGET.read_text(encoding="utf-8"))
    engine = DamageEngine(target, PhysicsRegistry.from_file(PHYSICS))
    cases = []
    test_inputs_seen = 0
    with gzip.open(CASES, "rt", encoding="utf-8") as stream:
        for line in stream:
            case = json.loads(line)
            split = case.get("metadata", {}).get("frozen_split")
            if split == "test":
                test_inputs_seen += 1
                continue
            if split in {"train", "validation"}:
                cases.append(case)
            if len(cases) == args.case_count:
                break
    proxy = FastSamplingProxy(
        engine,
        source_seed=2026112001,
        azimuths_per_ring=4,
        layer_quadrature_samples=16,
    )
    exposure_reference = proxy.evaluate(cases[0], output_scope="exposure")
    full_reference = proxy.evaluate(cases[0], output_scope="full")
    fields_equal = all(exposure_reference[field] == full_reference[field] for field in FIELDS)
    for case in cases[: min(8, len(cases))]:
        proxy.evaluate(case, output_scope="full")
        proxy.evaluate(case, output_scope="exposure")
    times = {"exposure": [], "full": []}
    orders = [
        ("full", "exposure") if index % 2 == 0 else ("exposure", "full")
        for index in range(args.repeats)
    ]
    paired_speedups = []
    for order in orders:
        local = {}
        for scope in order:
            started = perf_counter()
            for case in cases:
                proxy.evaluate(case, output_scope=scope)
            elapsed = perf_counter() - started
            times[scope].append(elapsed)
            local[scope] = elapsed
        paired_speedups.append(local["full"] / local["exposure"])
    median_exposure = float(np.median(times["exposure"]))
    median_full = float(np.median(times["full"]))
    report = {
        "schema_version": "fast-proxy-scope-benchmark@1.0.0",
        "development_case_count": len(cases),
        "repeats": args.repeats,
        "execution_order": [list(order) for order in orders],
        "times_seconds": times,
        "median_exposure_seconds": median_exposure,
        "median_full_seconds": median_full,
        "exposure_only_speedup": median_full / median_exposure,
        "paired_speedups": paired_speedups,
        "median_paired_speedup": float(np.median(paired_speedups)),
        "exposure_cases_per_second": len(cases) / median_exposure,
        "full_cases_per_second": len(cases) / median_full,
        "shared_exposure_fields_exactly_equal": fields_equal,
        "test_case_inputs_seen": test_inputs_seen,
        "teacher_result_rows_read": 0,
        "sealed_test_result_rows_decoded": 0,
        "inputs": {
            "target_sha256": file_hash(TARGET),
            "physics_sha256": file_hash(PHYSICS),
            "formal_cases_sha256": file_hash(CASES),
        },
        "live_identity": {
            "target_hash": engine.target_hash,
            "physics_registry_hash": engine.registry.hash,
            "engine_code_hash": ENGINE_CODE_HASH,
            "damage_engine_source_tree_hash": source_tree_hash(ROOT / "damage_engine"),
        },
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
