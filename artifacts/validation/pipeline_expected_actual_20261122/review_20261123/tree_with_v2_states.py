"""Review check: damage-tree means when the proxy uses all-fragment rays (v2 states).

Same 1,120 joint-complete cases as REPORT 8.2 (teacher complete & default-proxy complete).
Teacher means are taken from tree_gap_decomposition.json (same case set).
"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from damage_engine.engine import DamageEngine
from damage_engine.registry import PhysicsRegistry
npz = dict(np.load(HERE / "proxy_variants.npz"))
dec = json.loads((HERE / "tree_gap_decomposition.json").read_text())
engine = DamageEngine(json.loads((ROOT / "configs/targets/target.fragment_only.v1.json").read_text(encoding="utf-8-sig")),
                      PhysicsRegistry.from_file(ROOT / "configs/physics/loitering_impact.taylor_shapiro.obliquity_bound.v3.json"))
names = {c.object_id: [s for s in c.source["damage_states"] if s != "intact"] for c in engine.compiled_target.components}
comps = npz["component_ids"].tolist()
joint = (npz["teacher_affected"] == 0) & npz["v0_complete"]
acc = {}
for i in np.where(joint)[0]:
    st = {c: {s: float(npz["v2_p_deg" if s == "degraded" else "v2_p_fail"][i, j]) for s in names[c]} for j, c in enumerate(comps)}
    z = {c: {s: 0.0 for s in names[c]} for c in names}
    for k, v in engine.damage_tree.evaluate({"fragment": st, "blast": z, "combined": st})["outputs"].items():
        acc.setdefault(k, []).append(float(v))
out = {"cases": int(joint.sum()), "outputs": {}}
for k, vals in acc.items():
    t = dec["outputs"][k]["teacher_mean_of_replicate_trees"]; p0 = dec["outputs"][k]["proxy_mean"]; p2 = float(np.mean(vals))
    out["outputs"][k] = {"proxy_v0": p0, "proxy_v2": p2, "teacher": t, "gap_v0": t - p0, "gap_v2": t - p2,
                         "gap_closed_fraction": (p2 - p0) / (t - p0) if t != p0 else None}
(HERE / "tree_with_v2_states.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
