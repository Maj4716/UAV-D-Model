"""Clean per-case runtime of selected proxy configurations.

Each configuration runs in its own single-threaded process (at most 10 at once
on a 16-core host, so cores are not oversubscribed) over the same stratified
subset of development cases (first N per profile, evidence order).  The engine
is constructed once per process and excluded from the timing.

Usage: python benchmark_runtime.py [N_PER_PROFILE]   (default 20)
Writes results/runtime_benchmark.json.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from copy import deepcopy
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import PHYSICS, ROOT, STUDY, TARGET, load_cases  # noqa: E402
from run_sweep import CONFIGS  # noqa: E402

NAMES = [
    "az4", "az8", "az16", "azall", "azall_reg_dom",
    "azall_k4", "azall_k4_reg_dom", "azall_k8_reg", "azall_k8_reg_dom", "azall_k16_reg_dom",
]


def _time_config(task):
    name, cases = task
    sys.path.insert(0, str(ROOT))
    from damage_engine.engine import DamageEngine
    from damage_engine.fast_sampling_proxy import FastSamplingProxy
    from damage_engine.registry import PhysicsRegistry

    engine = DamageEngine(
        json.loads(TARGET.read_text(encoding="utf-8-sig")),
        PhysicsRegistry.from_file(PHYSICS),
    )
    proxy = FastSamplingProxy(engine, **CONFIGS[name])
    proxy.evaluate(deepcopy(cases[0]))  # warm-up (imports, caches)
    seconds = []
    for case in cases:
        start = time.perf_counter()
        proxy.evaluate(deepcopy(case))
        seconds.append(time.perf_counter() - start)
    return name, seconds


def main(argv: list[str]) -> int:
    per_profile = int(argv[0]) if argv else 20
    cases, seen = [], {}
    for case in load_cases():
        profile = case["munition"]["profile_id"]
        if seen.get(profile, 0) < per_profile:
            seen[profile] = seen.get(profile, 0) + 1
            cases.append(case)
    with Pool(min(10, len(NAMES))) as pool:
        rows = pool.map(_time_config, [(name, cases) for name in NAMES])
    report = {
        "host": {"platform": platform.platform(), "cpu_count": os.cpu_count()},
        "cases": len(cases),
        "per_profile": per_profile,
        "concurrent_processes": min(10, len(NAMES)),
        "configs": {
            name: {
                "kwargs": CONFIGS[name],
                "mean_seconds_per_case": float(np.mean(s)),
                "median_seconds_per_case": float(np.median(s)),
                "p95_seconds_per_case": float(np.quantile(s, 0.95)),
            }
            for name, s in rows
        },
    }
    base = report["configs"]["az4"]["mean_seconds_per_case"]
    for row in report["configs"].values():
        row["relative_to_az4"] = row["mean_seconds_per_case"] / base
    (STUDY / "results" / "runtime_benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, row in report["configs"].items():
        print(f"{name:16s} {row['mean_seconds_per_case']:.3f} s/case  x{row['relative_to_az4']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
