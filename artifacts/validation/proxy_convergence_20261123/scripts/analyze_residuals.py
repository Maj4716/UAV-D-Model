"""Residual analysis for the recommended configuration.

1. Brier(K) = a + b/K fit over the source-realization series (union p_hit),
   compared with the teacher Monte-Carlo noise floor.
2. Per-component worst p_hit / p_failed ratios (components with teacher mass).
3. Tree-output leaf components with the largest failed-state shortfall.
Writes results/residual_metrics.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import STUDY, TEACHER  # noqa: E402

RUNS = STUDY / "results" / "runs"
SERIES = {
    "historical_accumulation": [("azall", 1), ("azall_k2", 2), ("azall_k4", 4), ("azall_k8", 8), ("azall_k16", 16)],
    "teacher_predictor_options": [("azall_reg_dom", 1), ("azall_k4_reg_dom", 4), ("azall_k8_reg_dom", 8), ("azall_k16_reg_dom", 16)],
}
PRESET = "azall_k8_reg_dom"


def brier(proxy, teacher, mask):
    return float(np.mean((proxy[mask] - teacher[mask]) ** 2))


def main() -> int:
    t = dict(np.load(TEACHER, allow_pickle=False))
    M, T = t["m_p_hit"], t["t_p_hit"]
    noise = T * (1 - T) / np.maximum(t["R"][:, None] - 1, 1)
    floor = float(noise[M].mean())
    out: dict = {"teacher_noise_floor_p_hit_brier": floor, "brier_vs_K": {}}
    for label, series in SERIES.items():
        ks, ys, yf = [], [], []
        for name, k in series:
            d = np.load(RUNS / f"{name}.npz")
            ks.append(k)
            ys.append(brier(d["phit_prod"], T, M))
            yf.append(brier(d["p_fail"], t["t_p_fail"], t["m_p_fail"]))
        X = np.stack([np.ones(len(ks)), 1.0 / np.asarray(ks, float)], axis=1)
        a, b = np.linalg.lstsq(X, np.asarray(ys), rcond=None)[0]
        af, bf = np.linalg.lstsq(X, np.asarray(yf), rcond=None)[0]
        out["brier_vs_K"][label] = {
            "K": ks,
            "p_hit_brier": ys,
            "fit_p_hit_a_plus_b_over_K": {"a_K_infinity": float(a), "b": float(b)},
            "p_hit_a_minus_teacher_noise_floor": float(a - floor),
            "p_failed_brier": yf,
            "fit_p_failed_a_plus_b_over_K": {"a_K_infinity": float(af), "b": float(bf)},
        }
    hist, reg = np.load(RUNS / "azall.npz"), np.load(RUNS / "azall_reg.npz")
    mf = t["m_p_fail"]
    bands = []
    for lo, hi in ((0.0, 0.5), (0.5, 2.0), (2.0, 5.0), (5.0, np.inf)):
        sel = mf & (hist["lambda"] > lo) & (hist["lambda"] <= hi)
        bands.append(
            {
                "lambda_band": f"({lo}, {hi}]",
                "pairs": int(sel.sum()),
                "teacher_p_failed_mean": float(t["t_p_fail"][sel].mean()),
                "azall_historical_mean": float(hist["p_fail"][sel].mean()),
                "azall_registry_mean": float(reg["p_fail"][sel].mean()),
            }
        )
    zero = mf & (hist["lambda"] == 0.0)
    out["p_failed_by_lambda_band"] = {
        "bands": bands,
        "lambda_zero_pairs": int(zero.sum()),
        "teacher_p_failed_mass_share_at_lambda_zero": float(
            t["t_p_fail"][zero].sum() / t["t_p_fail"][mf].sum()
        ),
    }
    d = np.load(RUNS / f"{PRESET}.npz")
    comps = [str(c) for c in t["component_ids"]]
    rows = []
    for j, c in enumerate(comps):
        mh, mf = M[:, j], t["m_p_fail"][:, j]
        th, tf = T[mh, j].sum(), t["t_p_fail"][mf, j].sum()
        rows.append(
            {
                "component": c,
                "teacher_p_hit_mass": float(th),
                "p_hit_ratio": float(d["phit_prod"][mh, j].sum() / th) if th > 0 else None,
                "teacher_p_failed_mass": float(tf),
                "p_failed_ratio": float(d["p_fail"][mf, j].sum() / tf) if tf > 0 else None,
            }
        )
    massive = [r for r in rows if r["teacher_p_hit_mass"] >= 20.0]
    out["preset"] = PRESET
    out["components_with_teacher_p_hit_mass_ge_20"] = len(massive)
    out["worst_p_hit_ratio"] = sorted(massive, key=lambda r: r["p_hit_ratio"])[:5]
    failing = [r for r in rows if r["teacher_p_failed_mass"] >= 10.0]
    out["components_with_teacher_p_failed_mass_ge_10"] = len(failing)
    out["worst_p_failed_ratio"] = sorted(failing, key=lambda r: r["p_failed_ratio"])[:5]
    out["p_failed_ratio_quantiles_over_components"] = np.quantile(
        [r["p_failed_ratio"] for r in failing], [0.1, 0.5, 0.9]
    ).tolist()
    out["p_hit_ratio_quantiles_over_components"] = np.quantile(
        [r["p_hit_ratio"] for r in massive], [0.1, 0.5, 0.9]
    ).tolist()
    (STUDY / "results" / "residual_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
