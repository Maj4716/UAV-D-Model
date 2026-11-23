"""Review check: v0 vs v2 proxy agreement over the full development scope.

Reads proxy_variants.npz only (no dataset decoding).  Writes proxy_v2_full_scope.json.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
d = dict(np.load(HERE / "proxy_variants.npz"))
T, M = d["t_p_hit"], d["m_p_hit"]
ph = {v: -np.expm1(-np.maximum(d[f"{v}_lambda"], 0)) for v in ("v0", "v1", "v2")}
ok = d["teacher_affected"] == 0
out = {}
for v in ("v0", "v2"):
    for name, cm in (("all_dev", np.ones(len(T), bool)), ("teacher_complete", ok), ("validation_all", d["split"] == "validation")):
        s = cm[:, None] & M
        p, t = ph[v][s], T[s]
        out[f"{v}|{name}"] = dict(pairs=int(s.sum()), proxy_mean=float(p.mean()), teacher_mean=float(t.mean()), brier=float(np.mean((p - t) ** 2)))
    f = d[f"{v}_ood_frac"]
    out[f"{v}|ood_frac"] = dict(partial=int((~d[f"{v}_complete"]).sum()), mean=float(f.mean()), median=float(np.median(f)), p90=float(np.quantile(f, 0.9)))
for v in ("v0", "v2"):
    s = (d["split"] == "validation")[:, None] & M
    z = ph[v][s] == 0
    out[f"{v}|false_unreached_val_R32"] = float((T[s][z] > 0).mean())
for v in ("v0", "v2"):
    out[f"{v}|comp_hits_mean_teacher_complete"] = float(d[f"{v}_comp_hits"][ok].mean())
    out[f"{v}|armor_hits_mean_teacher_complete"] = float(d[f"{v}_armor_hits"][ok].mean())
out["teacher|comp_hits_mean_teacher_complete"] = float(d["t_comp_hits"][ok].mean())
out["teacher|armor_hits_mean_teacher_complete"] = float(d["t_armor_hits"][ok].mean())
out["v0|distinct_lambda_gt0_teacher_complete"] = float((d["v0_lambda"][ok] > 0).sum(1).mean())
out["v2|distinct_lambda_gt0_teacher_complete"] = float((d["v2_lambda"][ok] > 0).sum(1).mean())
out["teacher|distinct_p_hit_gt0_teacher_complete"] = float(((T > 0) & M)[ok].sum(1).mean())
(HERE / "proxy_v2_full_scope.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=1))
