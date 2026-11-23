"""Run FastSamplingProxy configurations over the 2,300 development cases.

Proxy-side only: reads the pre-outcome case cache (work/development_cases.json.gz)
and writes one results/runs/<name>.npz per configuration plus
results/runs/<name>.json with provenance.  No teacher quantity is read here.

Usage: python run_sweep.py NAME [NAME ...]   (names from CONFIGS below)
       python run_sweep.py --list
Environment: SWEEP_WORKERS (default 14), SWEEP_FORCE=1 to overwrite.
"""
from __future__ import annotations

import hashlib
import json
import os
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

RUNS = STUDY / "results" / "runs"
ALL = 1_000_000  # >= every ring size: one ray per fragment
DEFAULT_SEED = 2026111401

CONFIGS: dict[str, dict] = {
    # azimuthal ray sweep (default seed, one realization, historical accumulation)
    "az2": dict(azimuths_per_ring=2),
    "az4": dict(azimuths_per_ring=4),  # historical default (review v0)
    "az8": dict(azimuths_per_ring=8),
    "az16": dict(azimuths_per_ring=16),
    "azall": dict(azimuths_per_ring=ALL),  # review v2; declared gate configuration
    # layer quadrature sweep
    "azall_q4": dict(azimuths_per_ring=ALL, layer_quadrature_samples=4),
    "azall_q64": dict(azimuths_per_ring=ALL, layer_quadrature_samples=64),
    # reach cutoff
    "azall_cut0": dict(azimuths_per_ring=ALL, reach_probability_cutoff=0.0),
    # source seed sweep (single realization)
    "az4_s7777777": dict(azimuths_per_ring=4, source_seed=7_777_777),  # review v1
    "az4_s2026112301": dict(azimuths_per_ring=4, source_seed=2026112301),
    "az4_s2026112302": dict(azimuths_per_ring=4, source_seed=2026112302),
    "azall_s7777777": dict(azimuths_per_ring=ALL, source_seed=7_777_777),
    "azall_s2026112301": dict(azimuths_per_ring=ALL, source_seed=2026112301),
    "azall_s2026112302": dict(azimuths_per_ring=ALL, source_seed=2026112302),
    # source-realization averaging
    "azall_k2": dict(azimuths_per_ring=ALL, source_realizations=2),
    "azall_k4": dict(azimuths_per_ring=ALL, source_realizations=4),
    "azall_k8": dict(azimuths_per_ring=ALL, source_realizations=8),
    "azall_k16": dict(azimuths_per_ring=ALL, source_realizations=16),
    "az4_k8": dict(azimuths_per_ring=4, source_realizations=8),
    # registry (spatial cumulative) damage accumulation
    "az4_reg": dict(azimuths_per_ring=4, damage_accumulation="registry"),
    "azall_reg": dict(azimuths_per_ring=ALL, damage_accumulation="registry"),
    "azall_k4_reg": dict(
        azimuths_per_ring=ALL, source_realizations=4, damage_accumulation="registry"
    ),
    "azall_k8_reg": dict(
        azimuths_per_ring=ALL, source_realizations=8, damage_accumulation="registry"
    ),
    "azall_k16_reg": dict(
        azimuths_per_ring=ALL, source_realizations=16, damage_accumulation="registry"
    ),
    # teacher-consistent domain censoring (added after the first sweep)
    "azall_dom": dict(azimuths_per_ring=ALL, domain_censoring="branch_probability"),
    "azall_reg_dom": dict(
        azimuths_per_ring=ALL,
        damage_accumulation="registry",
        domain_censoring="branch_probability",
    ),
    "azall_k4_reg_dom": dict(
        azimuths_per_ring=ALL,
        source_realizations=4,
        damage_accumulation="registry",
        domain_censoring="branch_probability",
    ),
    "azall_k8_reg_dom": dict(  # == TEACHER_PREDICTOR_OPTIONS
        azimuths_per_ring=ALL,
        source_realizations=8,
        damage_accumulation="registry",
        domain_censoring="branch_probability",
    ),
    "azall_k16_reg_dom": dict(
        azimuths_per_ring=ALL,
        source_realizations=16,
        damage_accumulation="registry",
        domain_censoring="branch_probability",
    ),
}

_ENGINE = None
_COMPS: list[str] = []


def _init() -> None:
    global _ENGINE, _COMPS
    sys.path.insert(0, str(ROOT))
    from damage_engine.engine import DamageEngine
    from damage_engine.registry import PhysicsRegistry

    _ENGINE = DamageEngine(
        json.loads(TARGET.read_text(encoding="utf-8-sig")),
        PhysicsRegistry.from_file(PHYSICS),
    )
    _COMPS = list(_ENGINE.compiled_target.component_order)


def _evaluate(task):
    from damage_engine.fast_sampling_proxy import FastSamplingProxy

    index, case, kwargs = task
    proxy = FastSamplingProxy(_ENGINE, **kwargs)
    start = time.perf_counter()
    r = proxy.evaluate(deepcopy(case))
    elapsed = time.perf_counter() - start
    states = r["component_state_probabilities"]
    coverage = r["proxy_domain_coverage"]
    return {
        "index": index,
        "seconds": elapsed,
        "lambda": [r["component_metrics"][c]["expected_hits"] for c in _COMPS],
        "energy": [r["component_metrics"][c]["expected_damage_energy_j"] for c in _COMPS],
        "absorbed": [r["component_metrics"][c]["expected_absorbed_energy_j"] for c in _COMPS],
        "phit_prod": [r["component_hit_probabilities"][c] for c in _COMPS],
        "p_deg": [states[c].get("degraded", 0.0) for c in _COMPS],
        "p_fail": [states[c].get("failed", 0.0) for c in _COMPS],
        "tree": r["tree_outputs"],
        "ood_packet_frac": coverage["out_of_domain_packet_fraction"],
        "ood_mass_frac": r["out_of_domain_mass_fraction"],
        "ood_armor_w": coverage["out_of_domain_packet_weight_by_role"]["armor"],
        "ood_comp_w": coverage["out_of_domain_packet_weight_by_role"]["component"],
        "represented": r["represented_fragment_count"],
        "complete": r["proxy_domain_status"] == "complete_proxy_domain",
        "comp_hits": r["expected_total_component_hits"],
        "armor_hits": r["expected_armor_hits"],
        "layered": r["layered_interaction_count"],
        "rays": r["representative_ray_count"],
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(name: str, cases: list[dict], workers: int) -> None:
    kwargs = CONFIGS[name]
    RUNS.mkdir(parents=True, exist_ok=True)
    module = ROOT / "damage_engine/fast_sampling_proxy.py"
    fragments = ROOT / "damage_engine/fragments.py"
    before = (_sha(module), _sha(fragments))
    t0 = time.time()
    with Pool(workers, initializer=_init) as pool:
        rows = pool.map(
            _evaluate, [(i, c, kwargs) for i, c in enumerate(cases)], chunksize=8
        )
    wall = time.time() - t0
    after = (_sha(module), _sha(fragments))
    if before != after:
        raise RuntimeError(f"{name}: engine sources changed during the run; rerun")
    rows.sort(key=lambda row: row["index"])
    tree_names = sorted(rows[0]["tree"])
    arr = {
        key: np.asarray([row[key] for row in rows])
        for key in (
            "seconds", "lambda", "energy", "absorbed", "phit_prod", "p_deg", "p_fail",
            "ood_packet_frac", "ood_mass_frac", "ood_armor_w", "ood_comp_w",
            "represented", "complete", "comp_hits", "armor_hits", "layered", "rays",
        )
    }
    arr["tree"] = np.asarray([[row["tree"][t] for t in tree_names] for row in rows])
    arr["tree_names"] = np.asarray(tree_names)
    arr["case_id"] = np.asarray([c["case_id"] for c in cases])
    np.savez_compressed(RUNS / f"{name}.npz", **arr)
    sys.path.insert(0, str(ROOT))
    from damage_engine.engine import ENGINE_CODE_HASH

    meta = {
        "name": name,
        "kwargs": dict(kwargs),
        "case_count": len(cases),
        "workers": workers,
        "wall_seconds": wall,
        "mean_case_seconds_in_worker": float(arr["seconds"].mean()),
        "proxy_module_sha256": before[0],
        "fragments_module_sha256": before[1],
        "engine_code_hash_at_run": ENGINE_CODE_HASH,
        "finished_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (RUNS / f"{name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"{name}: {wall:.0f}s wall, {meta['mean_case_seconds_in_worker']:.3f}s/case",
        flush=True,
    )


def main(argv: list[str]) -> int:
    if not argv or argv == ["--list"]:
        print("\n".join(CONFIGS))
        return 0
    workers = int(os.environ.get("SWEEP_WORKERS", "14"))
    cases = load_cases()
    for name in argv:
        if (RUNS / f"{name}.npz").exists() and not os.environ.get("SWEEP_FORCE"):
            print(f"{name}: exists, skipped")
            continue
        run(name, cases, workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
