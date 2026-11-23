"""Review check: re-run FastSamplingProxy variants on the 2,300 development cases.

Purpose: test root-cause hypotheses for the proxy-teacher gap reported in
REPORT.md section 8.1 without touching sealed-test results.
  V0  default proxy (azimuths_per_ring=4, source_seed default)   -> should reproduce REPORT numbers
  V1  same, different source_seed (scatter realisation)          -> seed sensitivity of a "deterministic" prediction
  V2  azimuths_per_ring = all fragments (no azimuthal packet compression)
Outputs a compressed npz consumed by analyze_review_checks.py.
"""
from __future__ import annotations

import json, math, sys, time
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from damage_engine.engine import DamageEngine
from damage_engine.fast_sampling_proxy import FastSamplingProxy
from damage_engine.hurdle_targets import extract_component_hurdle_targets
from damage_engine.registry import PhysicsRegistry
from tools.evaluate_proxy_teacher_distribution import load_development_evidence

DATASET = ROOT / "artifacts/datasets/recovery_role_r8_r32_train2000_eval300_20261122"
TARGET = ROOT / "configs/targets/target.fragment_only.v1.json"
PHYSICS = ROOT / "configs/physics/loitering_impact.taylor_shapiro.obliquity_bound.v3.json"
OUT = Path(__file__).resolve().parent / "proxy_variants.npz"


def mean_or_zero(rec):
    v = rec.get("mean")
    return 0.0 if v is None else float(v)


def main() -> int:
    t0 = time.time()
    evidence, audit = load_development_evidence(DATASET)
    assert audit.get("test_result_json_lines_decoded", 0) == 0
    target = json.loads(TARGET.read_text(encoding="utf-8-sig"))
    engine = DamageEngine(target, PhysicsRegistry.from_file(PHYSICS))
    comps = list(engine.compiled_target.component_order)
    n, k = len(evidence), len(comps)
    variants = {
        "v0": FastSamplingProxy(engine),
        "v1": FastSamplingProxy(engine, source_seed=7_777_777),
        "v2": FastSamplingProxy(engine, azimuths_per_ring=1_000_000),
    }
    arr = {
        "split": np.empty(n, dtype="U16"), "profile": np.empty(n, dtype="U32"),
        "region": np.empty(n, dtype="U32"), "R": np.zeros(n, np.int64),
        "speed": np.zeros(n), "teacher_affected": np.zeros(n),
        "t_comp_hits": np.zeros(n), "t_armor_hits": np.zeros(n),
        "t_p_hit": np.zeros((n, k)), "t_p_deg": np.zeros((n, k)), "t_p_fail": np.zeros((n, k)),
        "m_p_hit": np.zeros((n, k), bool), "m_p_deg": np.zeros((n, k), bool), "m_p_fail": np.zeros((n, k), bool),
    }
    for v in variants:
        arr[f"{v}_complete"] = np.zeros(n, bool)
        arr[f"{v}_ood_frac"] = np.zeros(n)
        arr[f"{v}_lambda"] = np.zeros((n, k))
        arr[f"{v}_p_deg"] = np.zeros((n, k))
        arr[f"{v}_p_fail"] = np.zeros((n, k))
        arr[f"{v}_comp_hits"] = np.zeros(n)
        arr[f"{v}_armor_hits"] = np.zeros(n)
    for i, item in enumerate(evidence):
        case, result = item.case, item.result
        arr["split"][i] = item.split
        arr["profile"][i] = case["munition"]["profile_id"]
        arr["region"][i] = case["metadata"]["contact_manifold_sampling"]["region_id"]
        arr["R"][i] = int(case["analysis"]["replicates"])
        arr["speed"][i] = float(np.linalg.norm(case["encounter"]["velocity_target_mps"]))
        arr["teacher_affected"][i] = float(result["penetration_domain_coverage"]["affected_replicate_fraction"])
        arr["t_comp_hits"][i] = mean_or_zero(result["summary"]["total_component_hits"])
        arr["t_armor_hits"][i] = mean_or_zero(result["summary"]["total_armor_hits"])
        tg = extract_component_hurdle_targets(result)["components"]
        for j, c in enumerate(comps):
            t = tg[c]
            arr["t_p_hit"][i, j] = float(t["p_hit"]["value"]); arr["m_p_hit"][i, j] = bool(t["p_hit"]["mask"])
            sp = t["state_probabilities"]
            arr["t_p_deg"][i, j] = float(sp["p_degraded"]["value"]); arr["m_p_deg"][i, j] = bool(sp["p_degraded"]["mask"])
            arr["t_p_fail"][i, j] = float(sp["p_failed"]["value"]); arr["m_p_fail"][i, j] = bool(sp["p_failed"]["mask"])
        for v, px in variants.items():
            r = px.evaluate(deepcopy(case))
            arr[f"{v}_complete"][i] = r["proxy_domain_status"] == "complete_proxy_domain"
            arr[f"{v}_ood_frac"][i] = float(r["proxy_domain_coverage"]["out_of_domain_packet_fraction"])
            arr[f"{v}_comp_hits"][i] = float(r["expected_total_component_hits"])
            arr[f"{v}_armor_hits"][i] = float(r["expected_armor_hits"])
            for j, c in enumerate(comps):
                arr[f"{v}_lambda"][i, j] = float(r["component_metrics"][c]["expected_hits"])
                st = r["component_state_probabilities"][c]
                arr[f"{v}_p_deg"][i, j] = float(st.get("degraded", 0.0))
                arr[f"{v}_p_fail"][i, j] = float(st.get("failed", 0.0))
        if (i + 1) % 100 == 0:
            print(f"{i+1}/{n} {time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(OUT, component_ids=np.asarray(comps), **arr)
    print("saved", OUT, f"{time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
