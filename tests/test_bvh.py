from __future__ import annotations

from pathlib import Path

import numpy as np

from damage_engine.engine import DamageEngine
from damage_engine.raw_import import import_raw_target
from damage_engine.registry import PhysicsRegistry
from damage_engine.utils import load_json, normalize


ROOT = Path(__file__).resolve().parents[1]


def _raw_compiled_target():
    target, _ = import_raw_target(
        ROOT / "raw_data" / "List.csv",
        ROOT / "raw_data" / "armor.csv",
        load_json(ROOT / "raw_data" / "import_config.json"),
    )
    registry = PhysicsRegistry.from_file(ROOT / "examples" / "physics.demo.json")
    return DamageEngine(target, registry).compiled_target


def test_bvh_broad_phase_matches_vectorized_reference_on_raw_target():
    target = _raw_compiled_target()
    assert len(target.objects) == 69
    rng = np.random.default_rng(20260928)
    lower = np.min(target._bounds_min, axis=0) - 5.0
    upper = np.max(target._bounds_max, axis=0) + 5.0

    directions = [
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
    ]
    directions.extend(normalize(rng.normal(size=3)) for _ in range(997))
    for direction in directions:
        origin = rng.uniform(lower, upper)
        maximum_distance = float(rng.choice([np.inf, rng.uniform(0.0, 20.0)]))
        observed = target._candidate_indices(origin, direction, maximum_distance)
        expected = target._candidate_indices_bruteforce(
            origin, direction, maximum_distance
        )
        np.testing.assert_array_equal(observed, expected)


def test_bvh_preserves_sorted_physical_hits():
    target = _raw_compiled_target()
    origins_and_directions = [
        (np.asarray([-10.0, 0.0, 1.0]), np.asarray([1.0, 0.0, 0.0])),
        (np.asarray([0.0, -10.0, 1.0]), np.asarray([0.0, 1.0, 0.0])),
        (np.asarray([0.0, 0.0, 10.0]), np.asarray([0.0, 0.0, -1.0])),
    ]
    for origin, direction in origins_and_directions:
        hits = target.ray_hits(origin, direction)
        ordering = [(hit.entry_t, 0 if hit.role == "armor" else 1, hit.object_id) for hit in hits]
        assert ordering == sorted(ordering)
        for hit in hits:
            assert hit.exit_t >= hit.entry_t >= 0.0

