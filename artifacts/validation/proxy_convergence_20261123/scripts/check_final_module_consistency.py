"""Runs produced before the domain-censoring patch must match the final module.

The first sweep ran on an intermediate module (sha recorded per run).  This
re-evaluates the first N development cases with the final module for runs whose
recorded module hash differs from the current one and compares every stored
array bit-for-bit.  Writes results/final_module_consistency.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import PHYSICS, ROOT, STUDY, TARGET, load_cases  # noqa: E402
from run_sweep import CONFIGS  # noqa: E402

sys.path.insert(0, str(ROOT))
from damage_engine.engine import DamageEngine  # noqa: E402
from damage_engine.fast_sampling_proxy import FastSamplingProxy  # noqa: E402
from damage_engine.registry import PhysicsRegistry  # noqa: E402

RUNS = STUDY / "results" / "runs"
CHECK = ["az4", "azall", "azall_reg", "azall_k4_reg", "az4_s7777777"]
KEYS = ["lambda", "phit_prod", "p_deg", "p_fail", "energy", "absorbed", "ood_mass_frac", "ood_packet_frac"]


def main(argv: list[str]) -> int:
    count = int(argv[0]) if argv else 60
    current = hashlib.sha256((ROOT / "damage_engine/fast_sampling_proxy.py").read_bytes()).hexdigest()
    engine = DamageEngine(json.loads(TARGET.read_text(encoding="utf-8-sig")), PhysicsRegistry.from_file(PHYSICS))
    comps = list(engine.compiled_target.component_order)
    cases = load_cases()[:count]
    report = {"final_module_sha256": current, "cases": count, "runs": {}}
    ok = True
    for name in CHECK:
        meta = json.loads((RUNS / f"{name}.json").read_text(encoding="utf-8"))
        stored = np.load(RUNS / f"{name}.npz")
        proxy = FastSamplingProxy(engine, **CONFIGS[name])
        diffs = {key: 0.0 for key in KEYS}
        for i, case in enumerate(cases):
            r = proxy.evaluate(deepcopy(case))
            fresh = {
                "lambda": [r["component_metrics"][c]["expected_hits"] for c in comps],
                "phit_prod": [r["component_hit_probabilities"][c] for c in comps],
                "p_deg": [r["component_state_probabilities"][c].get("degraded", 0.0) for c in comps],
                "p_fail": [r["component_state_probabilities"][c].get("failed", 0.0) for c in comps],
                "energy": [r["component_metrics"][c]["expected_damage_energy_j"] for c in comps],
                "absorbed": [r["component_metrics"][c]["expected_absorbed_energy_j"] for c in comps],
                "ood_mass_frac": r["out_of_domain_mass_fraction"],
                "ood_packet_frac": r["proxy_domain_coverage"]["out_of_domain_packet_fraction"],
            }
            for key in KEYS:
                diffs[key] = max(diffs[key], float(np.max(np.abs(np.asarray(fresh[key]) - stored[key][i]))))
        same = all(value == 0.0 for value in diffs.values())
        ok &= same
        report["runs"][name] = {
            "run_module_sha256": meta["proxy_module_sha256"],
            "max_abs_diff": diffs,
            "bitwise_identical": same,
        }
        print(name, same, diffs, flush=True)
    report["all_bitwise_identical"] = bool(ok)
    (STUDY / "results" / "final_module_consistency.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
