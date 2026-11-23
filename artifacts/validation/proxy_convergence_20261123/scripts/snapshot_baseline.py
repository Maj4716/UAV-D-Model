"""Snapshot proxy outputs before any module edit (backward-compatibility oracle).

Writes baseline/pre_edit_outputs.json: full evaluate() outputs of the
unmodified proxy for the first 40 development cases (inputs only) under
v0 (4 rays/ring), v1 (other seed), v2 (all fragments) and exposure scope.
"""
from __future__ import annotations

import hashlib
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
from tools.evaluate_proxy_teacher_distribution import load_development_evidence  # noqa: E402

DATASET = ROOT / "artifacts/datasets/recovery_role_r8_r32_train2000_eval300_20261122"
TARGET = ROOT / "configs/targets/target.fragment_only.v1.json"
PHYSICS = ROOT / "configs/physics/loitering_impact.taylor_shapiro.obliquity_bound.v3.json"
OUT = HERE.parent / "baseline" / "pre_edit_outputs.json"


def main() -> int:
    evidence, audit = load_development_evidence(DATASET)
    assert audit["test_result_json_lines_decoded"] == 0
    engine = DamageEngine(
        json.loads(TARGET.read_text(encoding="utf-8-sig")),
        PhysicsRegistry.from_file(PHYSICS),
    )
    configs = {
        "v0": dict(),
        "v1": dict(source_seed=7_777_777),
        "v2": dict(azimuths_per_ring=1_000_000),
    }
    out = {
        "module_sha256": hashlib.sha256(
            (ROOT / "damage_engine/fast_sampling_proxy.py").read_bytes()
        ).hexdigest(),
        "cases": {},
    }
    for item in evidence[:40]:
        case = item.case
        row = {}
        for name, kwargs in configs.items():
            proxy = FastSamplingProxy(engine, **kwargs)
            row[name] = proxy.evaluate(deepcopy(case))
        row["v0_exposure"] = FastSamplingProxy(engine).evaluate(
            deepcopy(case), output_scope="exposure"
        )
        out["cases"][case["case_id"]] = row
    OUT.write_text(json.dumps(out, sort_keys=True), encoding="utf-8")
    print("wrote", OUT, len(out["cases"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
