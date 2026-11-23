"""Review check: split the proxy-teacher damage-tree gap (REPORT 8.2) into two parts.

  proxy gap = [tree(E_proxy states) - tree(E_teacher states)]      input-state bias
            + [tree(E_teacher states) - E_r tree(teacher states_r)] aggregation-order (Jensen) effect

Both terms are computable from archived per-replicate component_state_trace plus
the proxy state probabilities stored in proxy_variants.npz.  Train/validation only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from damage_engine.engine import DamageEngine  # noqa: E402
from damage_engine.registry import PhysicsRegistry  # noqa: E402
from tools.evaluate_proxy_teacher_distribution import load_development_evidence  # noqa: E402

DATASET = ROOT / "artifacts/datasets/recovery_role_r8_r32_train2000_eval300_20261122"
TARGET = ROOT / "configs/targets/target.fragment_only.v1.json"
PHYSICS = ROOT / "configs/physics/loitering_impact.taylor_shapiro.obliquity_bound.v3.json"


def main() -> int:
    npz = dict(np.load(HERE / "proxy_variants.npz", allow_pickle=False))
    comps = npz["component_ids"].tolist()
    joint = (npz["teacher_affected"] == 0.0) & npz["v0_complete"]
    evidence, audit = load_development_evidence(DATASET)
    assert audit.get("test_result_json_lines_decoded", 0) == 0
    engine = DamageEngine(
        json.loads(TARGET.read_text(encoding="utf-8-sig")),
        PhysicsRegistry.from_file(PHYSICS),
    )
    state_names = {
        item.object_id: [s for s in item.source["damage_states"] if s != "intact"]
        for item in engine.compiled_target.components
    }

    def tree(states: dict) -> dict:
        zeros = {c: {s: 0.0 for s in state_names[c]} for c in state_names}
        return engine.damage_tree.evaluate({"fragment": states, "blast": zeros, "combined": states})["outputs"]

    rows = []
    for i, item in enumerate(evidence):
        if not joint[i]:
            continue
        assert item.case["case_id"] == item.result.get("case_id", item.case["case_id"])
        records = item.result["component_state_trace"]["records"]
        names = sorted(records[0]["system_outputs"])
        teacher_tree = {n: float(np.mean([r["system_outputs"][n] for r in records])) for n in names}
        mean_states = {
            c: {
                s: float(np.mean([r["component_mechanism_states"]["fragment"][c][s] for r in records]))
                for s in state_names[c]
            }
            for c in state_names
        }
        tree_of_mean = tree(mean_states)
        proxy_states = {}
        for j, c in enumerate(comps):
            proxy_states[c] = {}
            for s in state_names[c]:
                key = {"degraded": "v0_p_deg", "failed": "v0_p_fail"}[s]
                proxy_states[c][s] = float(npz[key][i, j])
        proxy_tree = tree(proxy_states)
        rows.append({"teacher": teacher_tree, "tree_of_mean": tree_of_mean, "proxy": proxy_tree})

    names = sorted(rows[0]["teacher"])
    out = {"joint_complete_cases": len(rows), "outputs": {}}
    for n in names:
        t = np.array([r["teacher"][n] for r in rows])
        m = np.array([r["tree_of_mean"][n] for r in rows])
        p = np.array([r["proxy"][n] for r in rows])
        total = float(t.mean() - p.mean())
        out["outputs"][n] = {
            "proxy_mean": float(p.mean()),
            "tree_of_mean_teacher_states": float(m.mean()),
            "teacher_mean_of_replicate_trees": float(t.mean()),
            "teacher_minus_proxy": total,
            "input_state_bias_part": float(m.mean() - p.mean()),
            "aggregation_order_part": float(t.mean() - m.mean()),
            "order_share_of_gap": float((t.mean() - m.mean()) / total) if total else None,
        }
    (HERE / "tree_gap_decomposition.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
