"""Review check: does the aim-window side explain the reported region x profile label heterogeneity?

Uses pre-outcome aim points stored in each development case and the teacher
p_hit arrays in proxy_variants.npz (same loader order; verified below).
Train/validation only; sealed-test results are never decoded.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from tools.evaluate_proxy_teacher_distribution import load_development_evidence  # noqa: E402

DATASET = ROOT / "artifacts/datasets/recovery_role_r8_r32_train2000_eval300_20261122"
npz = dict(np.load(HERE / "proxy_variants.npz", allow_pickle=False))
evidence, audit = load_development_evidence(DATASET)
assert audit.get("test_result_json_lines_decoded", 0) == 0
assert len(evidence) == len(npz["region"])

side = []
for i, item in enumerate(evidence):
    meta = item.case["metadata"]["contact_manifold_sampling"]
    assert meta["region_id"] == npz["region"][i]
    assert item.case["munition"]["profile_id"] == npz["profile"][i]
    aim = item.case["metadata"]["continuous_direction_sampling"]["aim_point_target_m"]
    side.append("above" if aim[2] >= 0 else "below")
side = np.asarray(side)

T, M = npz["t_p_hit"], npz["m_p_hit"]
case_mean = np.nanmean(np.where(M, T, np.nan), axis=1)
out: dict = {"by_region": {}, "by_region_side": {}, "by_profile_region_side": {}}
for reg in sorted(set(npz["region"].tolist())):
    rm = npz["region"] == reg
    out["by_region"][reg] = {
        "cases": int(rm.sum()),
        "share_above": float(np.mean(side[rm] == "above")),
        "mean_case_p_hit": float(case_mean[rm].mean()),
    }
    for s in ("above", "below"):
        m = rm & (side == s)
        if m.any():
            out["by_region_side"][f"{reg}|{s}"] = {"cases": int(m.sum()), "mean_case_p_hit": float(case_mean[m].mean())}
for prof in sorted(set(npz["profile"].tolist())):
    for reg in sorted(set(npz["region"].tolist())):
        for s in ("above", "below"):
            m = (npz["profile"] == prof) & (npz["region"] == reg) & (side == s)
            if m.sum() >= 5:
                out["by_profile_region_side"][f"{reg}::{prof}|{s}"] = {
                    "cases": int(m.sum()), "mean_case_p_hit": float(case_mean[m].mean())}
# variance decomposition of case-mean p_hit: region x profile cells vs (region x profile x side) cells
def r2(groups):
    grand = case_mean.mean()
    ss_tot = float(((case_mean - grand) ** 2).sum())
    ss_between = 0.0
    for g in set(groups):
        m = groups == g
        ss_between += m.sum() * (case_mean[m].mean() - grand) ** 2
    return ss_between / ss_tot
cell = np.char.add(np.char.add(npz["region"], "::"), npz["profile"])
out["r2_case_p_hit_region_x_profile"] = float(r2(cell))
out["r2_case_p_hit_region_x_profile_x_side"] = float(r2(np.char.add(np.char.add(cell, "|"), side)))
out["r2_case_p_hit_profile_x_side"] = float(r2(np.char.add(np.char.add(npz["profile"], "|"), side)))
(HERE / "label_heterogeneity_by_aim_side.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=1))
