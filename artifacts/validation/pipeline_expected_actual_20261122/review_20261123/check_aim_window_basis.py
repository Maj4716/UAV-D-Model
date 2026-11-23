"""Review check: is the region-dependent S-CONTACT admission explained by the aim-window basis?

Symmetry prediction: if the target is (near) mirror-symmetric, opposite sectors
(pos_x/neg_x, pos_y/neg_y) should have similar S-CONTACT admission rates.  The
sampler builds the aim window with dynamic_tangent_basis(direction):
u = projection of the least-aligned target axis, v = cross(direction, u).
Because v flips with the direction, the declared aim_v_range [0.75, 1.7] m can
land above the target centre for one sector and below it for the opposite one.
This script maps every candidate latent (pre-outcome inputs only) back to its
aim point and tabulates admission by region and by the vertical side of the aim band.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from damage_engine.continuous_direction_sampling import (  # noqa: E402
    _horizontal_direction,
    _top_direction,
    dynamic_tangent_basis,
)

DATASET = ROOT / "artifacts/datasets/recovery_role_r8_r32_train2000_eval300_20261122"
spec = json.loads((DATASET / "plan/sampling_spec.json").read_text(encoding="utf-8"))
regions = spec["contact_manifold"]["regions"]
weights = np.array([float(r["selection_weight"]) for r in regions])
cum = np.cumsum(weights / weights.sum())
table = pq.read_table(DATASET / "rounds/round-000/candidates.parquet", columns=["latent", "selected"])
rows = table.to_pylist()

stats = defaultdict(lambda: {"candidates": 0, "admitted": 0})
by_region = defaultdict(lambda: {"candidates": 0, "admitted": 0, "v_z_positive": 0, "aim_z": [], "aim_z_admitted": []})
for row in rows:
    lat = np.asarray(row["latent"], float)
    phys = lat[1:]
    idx = min(int(np.searchsorted(cum, phys[0], side="right")), len(regions) - 1)
    reg = regions[idx]
    if reg["kind"] == "horizontal_sector":
        direction, _ = _horizontal_direction(reg, float(phys[1]), float(phys[2]))
    else:
        direction, _ = _top_direction(reg, float(phys[1]), float(phys[2]))
    u, v = dynamic_tangent_basis(direction)
    lo_u, hi_u = reg["aim_u_range_m"]
    lo_v, hi_v = reg["aim_v_range_m"]
    aim_u = lo_u + phys[3] * (hi_u - lo_u)
    aim_v = lo_v + phys[4] * (hi_v - lo_v)
    aim = np.asarray(reg["aim_center_target_m"], float) + aim_u * u + aim_v * v
    side = "aim_band_above_centre" if aim[2] >= 0 else "aim_band_below_centre"
    key = (reg["region_id"], side)
    stats[key]["candidates"] += 1
    stats[key]["admitted"] += int(bool(row["selected"]))
    r = by_region[reg["region_id"]]
    r["candidates"] += 1
    r["admitted"] += int(bool(row["selected"]))
    r["v_z_positive"] += int(v[2] > 0)
    r["aim_z"].append(float(aim[2]))
    if row["selected"]:
        r["aim_z_admitted"].append(float(aim[2]))

out = {"candidate_count": len(rows), "by_region": {}, "by_region_and_aim_side": {}}
for name, r in by_region.items():
    z = np.asarray(r["aim_z"])
    out["by_region"][name] = {
        "candidates": r["candidates"],
        "admission_rate": r["admitted"] / r["candidates"],
        "share_tangent_v_pointing_up": r["v_z_positive"] / r["candidates"],
        "aim_point_z_quantiles_p05_p50_p95": np.quantile(z, [0.05, 0.5, 0.95]).tolist(),
    }
for (name, side), s in sorted(stats.items()):
    out["by_region_and_aim_side"][f"{name}|{side}"] = {
        "candidates": s["candidates"],
        "admission_rate": s["admitted"] / s["candidates"],
    }
tot = defaultdict(lambda: [0, 0])
for (name, side), s in stats.items():
    tot[side][0] += s["candidates"]
    tot[side][1] += s["admitted"]
out["pooled_by_aim_side"] = {side: {"candidates": c, "admission_rate": a / c} for side, (c, a) in tot.items()}

# target geometry: armour/component vertical extent and y-mirror symmetry of armour boxes (if box primitives)
target = json.loads((ROOT / "configs/targets/target.fragment_only.v1.json").read_text(encoding="utf-8-sig"))


def collect_points(obj, acc):
    if isinstance(obj, dict):
        for key, val in obj.items():
            if key in ("min_m", "max_m", "center_m", "vertices_m", "position_m", "origin_m") and isinstance(val, list):
                arr = np.asarray(val, float).reshape(-1, 3) if np.size(val) % 3 == 0 else None
                if arr is not None:
                    acc.append(arr)
            else:
                collect_points(val, acc)
    elif isinstance(obj, list):
        for val in obj:
            collect_points(val, acc)


for part in ("armor", "components"):
    acc: list = []
    collect_points(target.get(part), acc)
    if acc:
        pts = np.vstack(acc)
        out[f"{part}_declared_point_extent_min_max_xyz"] = [pts.min(0).tolist(), pts.max(0).tolist()]
(HERE / "aim_window_basis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=1))
