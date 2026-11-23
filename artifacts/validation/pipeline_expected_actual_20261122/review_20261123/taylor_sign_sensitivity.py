"""Review check: how much would the Taylor-Shapiro tilt sign change the hit pattern?

Counterfactual with the deterministic proxy (all fragments as rays, same seed, same
scatter draws): baseline vs. Shapiro angle sign flipped (fragments lean toward the
detonation-wave travel direction, as Taylor 1941 predicts for nose initiation).
This is a sensitivity estimate for prioritisation, not a corrected dataset.
Development (train/validation) inputs only; uses a stratified subset of cases.
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
import damage_engine.fragments as fragments_module  # noqa: E402
from damage_engine.engine import DamageEngine  # noqa: E402
from damage_engine.fast_sampling_proxy import FastSamplingProxy  # noqa: E402
from damage_engine.registry import PhysicsRegistry  # noqa: E402
from tools.evaluate_proxy_teacher_distribution import load_development_evidence  # noqa: E402

DATASET = ROOT / "artifacts/datasets/recovery_role_r8_r32_train2000_eval300_20261122"
TARGET = ROOT / "configs/targets/target.fragment_only.v1.json"
PHYSICS = ROOT / "configs/physics/loitering_impact.taylor_shapiro.obliquity_bound.v3.json"
PER_CELL = 20

original = fragments_module.taylor_shapiro_angles


def flipped(*args, **kwargs):
    taylor, shapiro = original(*args, **kwargs)
    return taylor, -shapiro


def run(proxy, cases):
    lam = []
    for case in cases:
        r = proxy.evaluate(deepcopy(case))
        lam.append([r["component_metrics"][c]["expected_hits"] for c in proxy.engine.compiled_target.component_order])
    return -np.expm1(-np.asarray(lam))


def main() -> int:
    evidence, audit = load_development_evidence(DATASET)
    assert audit.get("test_result_json_lines_decoded", 0) == 0
    cells: dict[str, list] = {}
    for item in evidence:
        key = f'{item.case["metadata"]["contact_manifold_sampling"]["region_id"]}::{item.case["munition"]["profile_id"]}'
        cells.setdefault(key, []).append(item)
    chosen = []
    for key in sorted(cells):
        ordered = sorted(cells[key], key=lambda it: it.case["case_id"])
        chosen.extend(ordered[:PER_CELL])
    cases = [it.case for it in chosen]
    engine = DamageEngine(json.loads(TARGET.read_text(encoding="utf-8-sig")), PhysicsRegistry.from_file(PHYSICS))
    proxy = FastSamplingProxy(engine, azimuths_per_ring=1_000_000)
    base = run(proxy, cases)
    fragments_module.taylor_shapiro_angles = flipped
    try:
        flip = run(proxy, cases)
    finally:
        fragments_module.taylor_shapiro_angles = original
    hit_b, hit_f = base > 0.5, flip > 0.5
    out = {
        "cases": len(cases),
        "pairs": int(base.size),
        "mean_p_hit_baseline": float(base.mean()),
        "mean_p_hit_sign_flipped": float(flip.mean()),
        "mean_abs_change": float(np.mean(np.abs(flip - base))),
        "share_pairs_hit_status_changes_threshold_0p5": float(np.mean(hit_b != hit_f)),
        "share_baseline_hit_pairs_that_change": float(np.mean(hit_b[hit_b] != hit_f[hit_b])) if hit_b.any() else None,
        "share_cases_with_any_status_change": float(np.mean((hit_b != hit_f).any(axis=1))),
        "mean_components_hit_per_case_baseline": float(hit_b.sum(1).mean()),
        "mean_components_hit_per_case_flipped": float(hit_f.sum(1).mean()),
    }
    (HERE / "taylor_sign_sensitivity.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
