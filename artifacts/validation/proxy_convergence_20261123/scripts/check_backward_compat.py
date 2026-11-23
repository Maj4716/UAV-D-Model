"""Check that the edited proxy reproduces every pre-edit output key bit-for-bit.

Reads baseline/pre_edit_outputs.json (written by snapshot_baseline.py from the
unmodified module) and re-evaluates the same 40 cases under the same four
configurations with default options.  Every pre-existing key except
``schema_version`` must be identical; added keys are listed.
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from damage_engine.engine import DamageEngine  # noqa: E402
from damage_engine.fast_sampling_proxy import FastSamplingProxy  # noqa: E402
from damage_engine.registry import PhysicsRegistry  # noqa: E402

sys.path.insert(0, str(HERE))
from common import load_cases  # noqa: E402

TARGET = ROOT / "configs/targets/target.fragment_only.v1.json"
PHYSICS = ROOT / "configs/physics/loitering_impact.taylor_shapiro.obliquity_bound.v3.json"
BASE = HERE.parent / "baseline" / "pre_edit_outputs.json"
OUT = HERE.parent / "results" / "backward_compat_check.json"


def main() -> int:
    base = json.loads(BASE.read_text(encoding="utf-8"))
    cases = {case["case_id"]: case for case in load_cases()}
    engine = DamageEngine(
        json.loads(TARGET.read_text(encoding="utf-8-sig")),
        PhysicsRegistry.from_file(PHYSICS),
    )
    configs = {
        "v0": dict(),
        "v1": dict(source_seed=7_777_777),
        "v2": dict(azimuths_per_ring=1_000_000),
    }
    mismatches = []
    added: set[str] = set()
    for case_id, row in base["cases"].items():
        case = cases[case_id]
        new = {name: FastSamplingProxy(engine, **kw).evaluate(deepcopy(case)) for name, kw in configs.items()}
        new["v0_exposure"] = FastSamplingProxy(engine).evaluate(deepcopy(case), output_scope="exposure")
        for name, old in row.items():
            fresh = json.loads(json.dumps(new[name], sort_keys=True))
            for key, value in old.items():
                if key == "schema_version":
                    continue
                if key == "proxy_domain_coverage":
                    value = dict(value)
                    fresh_value = {k: fresh[key][k] for k in value}
                else:
                    fresh_value = fresh.get(key)
                if fresh_value != value:
                    mismatches.append({"case_id": case_id, "config": name, "key": key})
            added.update(set(fresh) - set(old))
    report = {
        "cases": len(base["cases"]),
        "configs": sorted(next(iter(base["cases"].values()))),
        "pre_edit_module_sha256": base["module_sha256"],
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:50],
        "added_top_level_keys": sorted(added),
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
