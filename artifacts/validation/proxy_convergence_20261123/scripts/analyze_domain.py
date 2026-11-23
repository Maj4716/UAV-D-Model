"""Proxy OOD packet-mass fraction as a continuous covariate (replaces the any-OOD flag).

For selected runs: agreement metrics by OOD mass-fraction band over all cases,
and the relation between the proxy OOD mass fraction and the teacher's
affected-replicate fraction.  Writes results/domain_metrics.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import STUDY, TEACHER  # noqa: E402

RUNS = STUDY / "results" / "runs"
BANDS = [(0.0, 0.0), (0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.0)]
NAMES = ["az4", "azall", "azall_k8_reg", "azall_dom", "azall_k8_reg_dom"]


def band_mask(x, lo, hi):
    return (x == 0.0) if hi == 0.0 else ((x > lo) & (x <= hi))


def main() -> int:
    t = dict(np.load(TEACHER, allow_pickle=False))
    out = {}
    for name in NAMES:
        path = RUNS / f"{name}.npz"
        if not path.exists():
            continue
        d = dict(np.load(path, allow_pickle=False))
        x = d["ood_mass_frac"]
        rows = []
        for lo, hi in BANDS:
            cases = band_mask(x, lo, hi)
            sel = cases[:, None] & t["m_p_hit"]
            self_f = cases[:, None] & t["m_p_fail"]
            rows.append(
                {
                    "band": "== 0" if hi == 0.0 else f"({lo}, {hi}]",
                    "cases": int(cases.sum()),
                    "p_hit_ratio_union": float(d["phit_prod"][sel].sum() / max(t["t_p_hit"][sel].sum(), 1e-300)),
                    "p_hit_brier_union": float(np.mean((d["phit_prod"][sel] - t["t_p_hit"][sel]) ** 2)) if sel.any() else None,
                    "p_fail_ratio": float(d["p_fail"][self_f].sum() / max(t["t_p_fail"][self_f].sum(), 1e-300)),
                    "teacher_affected_case_share": float(np.mean(t["teacher_affected"][cases] > 0)) if cases.any() else None,
                }
            )
        affected = t["teacher_affected"]
        out[name] = {
            "bands": rows,
            "any_ood_flag_cases": int((x > 0).sum()),
            "ood_mass_fraction_mean": float(x.mean()),
            "spearman_ood_mass_vs_teacher_affected_fraction": float(
                spearmanr(x, affected).statistic
            ),
            "teacher_affected_fraction_mean": float(affected.mean()),
            "proxy_expected_refused_contacts_per_case_mean": float((d["ood_armor_w"] + d["ood_comp_w"]).mean()),
            "teacher_unresolved_impacts_per_replicate_mean": float(t["t_unresolved_impacts"].mean()),
            "proxy_cases_with_refused_mass_ge_0p01_fragments": int(((d["ood_armor_w"] + d["ood_comp_w"]) >= 0.01).sum()),
            "teacher_cases_with_any_unresolved_impact": int((t["t_unresolved_impacts"] > 0).sum()),
            "spearman_proxy_refused_vs_teacher_unresolved": float(
                spearmanr(d["ood_armor_w"] + d["ood_comp_w"], t["t_unresolved_impacts"]).statistic
            ),
        }
    (STUDY / "results" / "domain_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for name, row in out.items():
        print(name, {k: v for k, v in row.items() if k != "bands"})
        for b in row["bands"]:
            print("   ", b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
