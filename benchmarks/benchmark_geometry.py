from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from damage_engine.engine import DamageEngine
from damage_engine.raw_import import import_raw_target
from damage_engine.registry import PhysicsRegistry
from damage_engine.utils import load_json, normalize


ROOT = Path(__file__).resolve().parents[1]


def _compiled_target():
    target, _ = import_raw_target(
        ROOT / "raw_data" / "List.csv",
        ROOT / "raw_data" / "armor.csv",
        load_json(ROOT / "raw_data" / "import_config.json"),
    )
    registry = PhysicsRegistry.from_file(ROOT / "examples" / "physics.demo.json")
    return DamageEngine(target, registry).compiled_target


def run(ray_count: int, seed: int) -> dict:
    if ray_count < 1:
        raise ValueError("ray_count must be positive")
    target = _compiled_target()
    rng = np.random.default_rng(seed)
    lower = np.min(target._bounds_min, axis=0) - 5.0
    upper = np.max(target._bounds_max, axis=0) + 5.0
    origins = rng.uniform(lower, upper, size=(ray_count, 3))
    directions = np.asarray([normalize(value) for value in rng.normal(size=(ray_count, 3))])

    start = time.perf_counter()
    bvh = [target._candidate_indices(origin, direction) for origin, direction in zip(origins, directions)]
    bvh_seconds = time.perf_counter() - start
    start = time.perf_counter()
    reference = [
        target._candidate_indices_bruteforce(origin, direction)
        for origin, direction in zip(origins, directions)
    ]
    reference_seconds = time.perf_counter() - start
    equivalent = all(np.array_equal(left, right) for left, right in zip(bvh, reference))
    return {
        "objects": len(target.objects),
        "rays": ray_count,
        "seed": seed,
        "candidate_sets_equal": equivalent,
        "bvh_seconds": bvh_seconds,
        "vectorized_reference_seconds": reference_seconds,
        "speedup": reference_seconds / bvh_seconds if bvh_seconds else float("inf"),
        "mean_candidates": float(np.mean([len(value) for value in bvh])),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rays", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260928)
    arguments = parser.parse_args(argv)
    import json

    print(json.dumps(run(arguments.rays, arguments.seed), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

