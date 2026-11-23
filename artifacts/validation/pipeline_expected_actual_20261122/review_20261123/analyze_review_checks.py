"""Review checks on top of proxy_variants.npz and the original bundle metrics.

All quantities are train/validation only (sealed test never decoded).
Writes review_checks.json next to this script.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
d = dict(np.load(HERE / "proxy_variants.npz", allow_pickle=False))
n, k = d["t_p_hit"].shape
out: dict = {"case_count": int(n), "component_count": int(k)}

teacher_ok = d["teacher_affected"] == 0.0
ph = {v: -np.expm1(-np.maximum(d[f"{v}_lambda"], 0.0)) for v in ("v0", "v1", "v2")}
T = d["t_p_hit"]
M = d["m_p_hit"]
R = d["R"]
STATE_KEY = {"p_degraded": "p_deg", "p_failed": "p_fail"}


def pair_stats(case_mask, proxy, teacher=T, mask=M):
    sel = case_mask[:, None] & mask
    p, t = proxy[sel], teacher[sel]
    zero = p == 0.0
    return {
        "cases": int(case_mask.sum()),
        "pairs": int(sel.sum()),
        "proxy_mean": float(p.mean()),
        "teacher_mean": float(t.mean()),
        "ratio_proxy_to_teacher": float(p.mean() / t.mean()),
        "brier": float(np.mean((p - t) ** 2)),
        "proxy_zero_fraction": float(zero.mean()),
        "teacher_positive_among_proxy_zero": float((t[zero] > 0).mean()) if zero.any() else None,
        "teacher_mass_share_where_proxy_below_0p01": float(t[p < 0.01].sum() / t.sum()),
    }


# 1. reproduction of REPORT 8.1 (joint complete under default proxy)
joint0 = teacher_ok & d["v0_complete"]
out["reproduce_report_8_1_p_hit"] = pair_stats(joint0, ph["v0"])
out["reproduce_domain_counts"] = {
    "proxy_partial_cases_v0": int((~d["v0_complete"]).sum()),
    "teacher_affected_cases": int((~teacher_ok).sum()),
    "joint_complete_cases_v0": int(joint0.sum()),
}

# 2. composition of the joint-complete subset vs all development cases
comp: dict = {}
for key in ("profile", "region"):
    vals = sorted(set(d[key].tolist()))
    comp[key] = {
        v: {
            "all_share": float(np.mean(d[key] == v)),
            "joint_share": float(np.mean(d[key][joint0] == v)),
            "proxy_partial_rate": float(np.mean(~d["v0_complete"][d[key] == v])),
            "teacher_affected_rate": float(np.mean(~teacher_ok[d[key] == v])),
        }
        for v in vals
    }
comp["speed_mean_all"] = float(d["speed"].mean())
comp["speed_mean_joint"] = float(d["speed"][joint0].mean())
comp["speed_mean_proxy_partial"] = float(d["speed"][~d["v0_complete"]].mean())
comp["teacher_mean_p_hit_all_identified"] = float(T[M].mean())
comp["teacher_mean_p_hit_joint_identified"] = float(T[joint0[:, None] & M].mean())
out["joint_subset_composition"] = comp

# 3. proxy OOD fraction among partial cases
part = ~d["v0_complete"]
f = d["v0_ood_frac"][part]
out["proxy_ood_fraction_among_partial_v0"] = {
    "count": int(part.sum()),
    "mean": float(f.mean()),
    "quantiles_p10_p50_p90": np.quantile(f, [0.1, 0.5, 0.9]).tolist(),
    "share_below_0p01": float(np.mean(f < 0.01)),
    "share_below_0p05": float(np.mean(f < 0.05)),
}

# 4. root cause A: azimuthal packet compression (v0 4 rays/ring vs v2 every fragment)
common = teacher_ok & d["v0_complete"] & d["v2_complete"]
rc: dict = {
    "v0_4_rays_per_ring": pair_stats(common, ph["v0"]),
    "v2_all_fragments": pair_stats(common, ph["v2"]),
}
for v in ("v0", "v2"):
    lam = d[f"{v}_lambda"][common]
    rc[f"{v}_mean_total_expected_component_hits_per_case"] = float(lam.sum(axis=1).mean())
    rc[f"{v}_mean_distinct_components_with_lambda_gt0"] = float((lam > 0).sum(axis=1).mean())
rc["teacher_mean_total_component_hits_per_case"] = float(d["t_comp_hits"][common].mean())
rc["teacher_mean_distinct_components_with_p_hit_gt0"] = float(((T > 0) & M)[common].sum(axis=1).mean())
rc["v2_partial_cases"] = int((~d["v2_complete"]).sum())
by_prof = {}
for p in sorted(set(d["profile"].tolist())):
    cm = common & (d["profile"] == p)
    by_prof[p] = {
        "cases": int(cm.sum()),
        "v0_ratio": pair_stats(cm, ph["v0"])["ratio_proxy_to_teacher"],
        "v2_ratio": pair_stats(cm, ph["v2"])["ratio_proxy_to_teacher"],
        "v0_teacher_positive_among_proxy_zero": pair_stats(cm, ph["v0"])["teacher_positive_among_proxy_zero"],
        "v2_teacher_positive_among_proxy_zero": pair_stats(cm, ph["v2"])["teacher_positive_among_proxy_zero"],
    }
rc["by_profile"] = by_prof
out["root_cause_A_azimuth_packets"] = rc

# 5. root cause B: damage accumulation (per-hit independent in proxy vs spatial cumulative in teacher)
rb: dict = {}
for v in ("v0", "v2"):
    row: dict = {}
    for head, tk, mk in (
        ("p_hit", "t_p_hit", "m_p_hit"),
        ("p_degraded", "t_p_deg", "m_p_deg"),
        ("p_failed", "t_p_fail", "m_p_fail"),
    ):
        sel = common[:, None] & d[mk]
        prox = ph[v] if head == "p_hit" else d[f"{v}_{STATE_KEY[head]}"]
        row[head] = {
            "proxy_mean": float(prox[sel].mean()),
            "teacher_mean": float(d[tk][sel].mean()),
            "ratio": float(prox[sel].mean() / d[tk][sel].mean()),
        }
    both = common[:, None] & M & d["m_p_fail"] & d["m_p_deg"] & (T > 0) & (ph[v] > 0)
    row["conditional_on_hit_failed_over_hit"] = {
        "proxy": float((d[f"{v}_p_fail"][both] / ph[v][both]).mean()),
        "teacher": float((d["t_p_fail"][both] / T[both]).mean()),
        "pairs": int(both.sum()),
    }
    row["conditional_on_hit_degraded_over_hit"] = {
        "proxy": float((d[f"{v}_p_deg"][both] / ph[v][both]).mean()),
        "teacher": float((d["t_p_deg"][both] / T[both]).mean()),
    }
    rb[v] = row
out["root_cause_B_damage_accumulation"] = rb

# 6. seed sensitivity of the 'deterministic' proxy (v0 vs v1: only source_seed differs)
both_ok = teacher_ok & d["v0_complete"] & d["v1_complete"]
sel = both_ok[:, None] & M
a, b = ph["v0"][sel], ph["v1"][sel]
out["seed_sensitivity_v0_vs_v1"] = {
    "domain_status_disagreement_cases": int((d["v0_complete"] != d["v1_complete"]).sum()),
    "pairs": int(sel.sum()),
    "mean_abs_p_hit_change": float(np.mean(np.abs(a - b))),
    "share_pairs_abs_change_gt_0p5": float(np.mean(np.abs(a - b) > 0.5)),
    "zero_status_flip_fraction": float(np.mean((a == 0) != (b == 0))),
    "brier_v0": float(np.mean((a - T[sel]) ** 2)),
    "brier_v1": float(np.mean((b - T[sel]) ** 2)),
    "mean_v0": float(a.mean()),
    "mean_v1": float(b.mean()),
}

# 7. teacher Monte-Carlo noise floor inside the soft-label Brier (joint, default proxy)
sel = joint0[:, None] & M
Rm = np.broadcast_to(R[:, None], T.shape)[sel]
t = T[sel]
noise = t * (1 - t) / np.maximum(Rm - 1, 1)  # unbiased estimate of Var(p_hat) per pair
brier = float(np.mean((ph["v0"][sel] - t) ** 2))
out["brier_teacher_noise_floor_joint_v0"] = {
    "brier": brier,
    "estimated_teacher_variance_term": float(noise.mean()),
    "noise_share_of_brier": float(noise.mean() / brier),
    "brier_minus_noise": float(brier - noise.mean()),
}

# 8. zero / fractional mass by replicate protocol (R8 train vs R32 validation)
zr: dict = {}
for split in ("train", "validation"):
    s = (d["split"] == split)[:, None] & M
    x = T[s]
    zr[split] = {
        "R": int(np.unique(R[d["split"] == split])[0]),
        "pairs": int(s.sum()),
        "zero_mass": float(np.mean(x == 0)),
        "one_mass": float(np.mean(x == 1)),
        "fractional_mass": float(np.mean((x > 0) & (x < 1))),
        "mean": float(x.mean()),
    }
s = (d["split"] == "validation")[:, None] & M
q = T[s]
zr["validation_predicted_zero_mass_if_R8"] = float(np.mean((1 - q) ** 8))
zr["validation_predicted_one_mass_if_R8"] = float(np.mean(q ** 8))
out["zero_mass_by_protocol"] = zr

# 9. false-unreached on the full R8 train split (instead of the 45-case pilot)
fu: dict = {}
tr_idx = np.where(d["split"] == "train")[0]
for v in ("v0", "v2"):
    num = np.array([((ph[v][i] == 0) & M[i] & (T[i] > 0)).sum() for i in tr_idx], float)
    den = np.array([((ph[v][i] == 0) & M[i]).sum() for i in tr_idx], float)
    rng = np.random.default_rng(20261123)
    idx = rng.integers(0, tr_idx.size, size=(4000, tr_idx.size))
    boot = num[idx].sum(1) / den[idx].sum(1)
    fu[v] = {
        "proxy_zero_pairs": int(den.sum()),
        "teacher_positive": int(num.sum()),
        "rate": float(num.sum() / den.sum()),
        "case_cluster_bootstrap_95": np.quantile(boot, [0.025, 0.975]).tolist(),
    }
out["false_unreached_train_R8"] = fu

# 10. micro-audit calibration-in-large: model-based SE vs reported stratified bootstrap (n_h = 2)
metrics = json.loads((BUNDLE / "metrics.json").read_text(encoding="utf-8"))
cal = metrics["stage_04_09_micro_audit"]["penetration_probability_calibration"]
mc: dict = {}
for role in ("armor", "component"):
    c = cal[role]
    se = math.sqrt(c["mean_model_bernoulli_variance_p_times_one_minus_p"] / c["event_count"])
    lo, hi = c["calibration_in_large_case_cluster_bootstrap_95_interval"]
    gap = c["calibration_in_large_observed_minus_predicted"]
    mc[role] = {
        "events": c["event_count"],
        "observed_minus_predicted": gap,
        "model_based_se": se,
        "z": gap / se,
        "model_based_95": [gap - 1.96 * se, gap + 1.96 * se],
        "reported_stratified_bootstrap_95": [lo, hi],
        "implied_bootstrap_se": (hi - lo) / 3.92,
        "se_ratio_model_over_bootstrap": se / ((hi - lo) / 3.92),
    }
out["micro_calibration_in_large_se_check"] = mc

# 11. zero-parameter Lambert baseline for first-contact incidence angle: F(theta) = sin^2(theta)
contact = json.loads((BUNDLE / "work/contact_inputs/contact_input_metrics.json").read_text(encoding="utf-8"))
inc = contact["actual_distributions"]["incidence_angle_from_normal_deg"]
lam_q = {
    name: math.degrees(math.asin(math.sqrt(p)))
    for name, p in (("p05", 0.05), ("p25", 0.25), ("median", 0.5), ("p75", 0.75), ("p95", 0.95))
}
lam_q["mean"] = 45.0
lam_q["population_standard_deviation"] = math.degrees(
    math.sqrt(math.pi ** 2 / 8 - 0.5 - (math.pi / 4) ** 2)
)
out["incidence_angle_lambert_baseline"] = {
    name: {"predicted_deg": lam_q[name], "observed_deg": inc[name], "diff_deg": inc[name] - lam_q[name]}
    for name in lam_q
}

# 12. train-validation minimum detectable difference for p_hit (observed-component case means)
case_mean = np.nanmean(np.where(M, T, np.nan), axis=1)
tr, va = case_mean[d["split"] == "train"], case_mean[d["split"] == "validation"]
se = math.sqrt(np.var(tr, ddof=1) / tr.size + np.var(va, ddof=1) / va.size)
out["split_drift_power_p_hit"] = {
    "train_mean": float(tr.mean()),
    "validation_mean": float(va.mean()),
    "se_difference": se,
    "mde_80pct_power_alpha05": 2.8 * se,
    "mde_relative_to_mean": 2.8 * se / float(case_mean.mean()),
}

# 13. predictive skill of the proxy against naive lookup baselines (fit on train, score on validation)
val_joint = joint0 & (d["split"] == "validation")
train_ok = (d["split"] == "train")[:, None] & M
comp_mean = np.nanmean(np.where(train_ok, T, np.nan), axis=0)
cells = np.char.add(np.char.add(d["region"], "::"), d["profile"])
cell_comp_mean = {}
for c in set(cells.tolist()):
    m = (cells == c) & (d["split"] == "train")
    cell_comp_mean[c] = np.nanmean(np.where(M[m], T[m], np.nan), axis=0)
base2 = np.vstack([cell_comp_mean[c] for c in cells])
sel = val_joint[:, None] & M
tv = T[sel]
skill = {
    "validation_joint_pairs": int(sel.sum()),
    "brier_constant_train_mean": float(np.mean((np.nanmean(T[train_ok]) - tv) ** 2)),
    "brier_component_mean_lookup": float(np.mean((np.broadcast_to(comp_mean, T.shape)[sel] - tv) ** 2)),
    "brier_component_x_region_x_profile_lookup": float(np.mean((np.nan_to_num(base2)[sel] - tv) ** 2)),
    "brier_proxy_v0": float(np.mean((ph["v0"][sel] - tv) ** 2)),
    "brier_proxy_v2": float(np.mean((ph["v2"][sel] - tv) ** 2)),
}
Rv = np.broadcast_to(R[:, None], T.shape)[sel]
skill["teacher_noise_term_validation"] = float(np.mean(tv * (1 - tv) / (Rv - 1)))
out["proxy_skill_vs_lookup_baselines_validation"] = skill

(HERE / "review_checks.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, indent=1, ensure_ascii=False))
