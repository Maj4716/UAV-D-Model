"""Convergence and teacher-agreement metrics for every sweep run.

Inputs : results/runs/*.npz (proxy, outcome-blind) and work/teacher_arrays.npz
         (train/validation teacher targets; sealed test never decoded).
Outputs: results/convergence_metrics.json, results/convergence_tables.md

All teacher comparisons use every development case: no filtering on the proxy
"any OOD packet" flag.  Pairs are restricted only by the teacher hurdle masks
(point-identified targets), exactly as the earlier reports did.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import STUDY, TEACHER, WORK  # noqa: E402
import gzip  # noqa: E402

RUNS = STUDY / "results" / "runs"
OUT_JSON = STUDY / "results" / "convergence_metrics.json"
OUT_MD = STUDY / "results" / "convergence_tables.md"
Z95 = 1.959963984540054
BOOT = 2000
BOOT_SEED = 2026112311


def wilson_upper(successes: int, trials: int) -> float | None:
    if trials == 0:
        return None
    p = successes / trials
    denominator = 1.0 + Z95 * Z95 / trials
    centre = (p + Z95 * Z95 / (2.0 * trials)) / denominator
    half = Z95 * math.sqrt(p * (1 - p) / trials + Z95 * Z95 / (4 * trials * trials)) / denominator
    return min(1.0, centre + half)


def ratio_and_brier(proxy, teacher, mask, cases, rng, draws=BOOT):
    """Pooled ratio of means and Brier with case-cluster bootstrap intervals."""
    sel = mask & cases[:, None]
    p_sum = np.where(sel, proxy, 0.0).sum(1)
    t_sum = np.where(sel, teacher, 0.0).sum(1)
    se_sum = np.where(sel, (proxy - teacher) ** 2, 0.0).sum(1)
    count = sel.sum(1).astype(float)
    idx = np.where(cases)[0]
    point = {
        "pairs": int(count.sum()),
        "proxy_mean": float(p_sum.sum() / count.sum()),
        "teacher_mean": float(t_sum.sum() / count.sum()),
        "ratio": float(p_sum.sum() / t_sum.sum()),
        "brier": float(se_sum.sum() / count.sum()),
    }
    if draws:
        b = rng.integers(0, idx.size, size=(draws, idx.size))
        chosen = idx[b]
        ratio = p_sum[chosen].sum(1) / t_sum[chosen].sum(1)
        brier = se_sum[chosen].sum(1) / count[chosen].sum(1)
        point["ratio_ci95"] = np.quantile(ratio, [0.025, 0.975]).tolist()
        point["brier_ci95"] = np.quantile(brier, [0.025, 0.975]).tolist()
    return point


def false_unreached(lam, hit_indicator, mask, cases, rng, draws=BOOT):
    sel = mask & cases[:, None] & (lam == 0.0)
    false = sel & (hit_indicator > 0.0)
    trials, successes = int(sel.sum()), int(false.sum())
    out = {
        "proxy_unreached_pairs": trials,
        "teacher_positive": successes,
        "rate": None if trials == 0 else successes / trials,
        "wilson_upper_95": wilson_upper(successes, trials),
    }
    if draws and trials:
        num, den = false.sum(1).astype(float), sel.sum(1).astype(float)
        idx = np.where(cases)[0]
        chosen = idx[rng.integers(0, idx.size, size=(draws, idx.size))]
        out["case_cluster_bootstrap_95"] = np.quantile(
            num[chosen].sum(1) / den[chosen].sum(1), [0.025, 0.975]
        ).tolist()
    return out


def conditional_on_hit(proxy_state, proxy_hit, teacher_state, teacher_hit, mask):
    both = mask & (teacher_hit > 0) & (proxy_hit > 0)
    return {
        "pairs": int(both.sum()),
        "proxy": float((proxy_state[both] / proxy_hit[both]).mean()),
        "teacher": float((teacher_state[both] / teacher_hit[both]).mean()),
    }


def analyze(name: str, d, t, rng, reference: dict) -> dict:
    meta = json.loads((RUNS / f"{name}.json").read_text(encoding="utf-8"))
    lam = d["lambda"]
    poisson = -np.expm1(-np.maximum(lam, 0.0))
    union = d["phit_prod"]
    split = t["split"]
    train, val = split == "train", split == "validation"
    every = np.ones(lam.shape[0], bool)
    M, T, H = t["m_p_hit"], t["t_p_hit"], t["hit_indicator"]
    out: dict = {"kwargs": meta["kwargs"], "rays_per_case_mean": float(d["rays"].mean())}
    out["runtime"] = {
        "mean_seconds_per_case": float(d["seconds"].mean()),
        "p95_seconds_per_case": float(np.quantile(d["seconds"], 0.95)),
        "wall_seconds_14_workers": meta["wall_seconds"],
    }
    out["p_hit"] = {}
    for label, proxy in (("poisson_1_minus_exp_lambda", poisson), ("arrival_union", union)):
        row = {
            "all": ratio_and_brier(proxy, T, M, every, rng),
            "train": ratio_and_brier(proxy, T, M, train, rng, draws=0),
            "validation": ratio_and_brier(proxy, T, M, val, rng, draws=0),
        }
        sel = M
        row["teacher_mass_share_where_proxy_below_0p01"] = float(
            T[sel & (proxy < 0.01)].sum() / T[sel].sum()
        )
        out["p_hit"][label] = row
    out["false_unreached"] = {
        "train_R8": false_unreached(lam, H, M, train, rng),
        "validation_R32": false_unreached(lam, H, M, val, rng, draws=0),
    }
    out["p_degraded"] = ratio_and_brier(d["p_deg"], t["t_p_deg"], t["m_p_deg"], every, rng)
    out["p_failed"] = ratio_and_brier(d["p_fail"], t["t_p_fail"], t["m_p_fail"], every, rng)
    mask_all = M & t["m_p_deg"] & t["m_p_fail"]
    out["conditional_failed_over_hit"] = conditional_on_hit(d["p_fail"], union, t["t_p_fail"], T, mask_all)
    out["conditional_degraded_over_hit"] = conditional_on_hit(d["p_deg"], union, t["t_p_deg"], T, mask_all)
    out["domain"] = {
        "ood_mass_fraction_mean": float(d["ood_mass_frac"].mean()),
        "ood_mass_fraction_median": float(np.median(d["ood_mass_frac"])),
        "ood_mass_fraction_p90": float(np.quantile(d["ood_mass_frac"], 0.9)),
        "any_ood_packet_flag_cases": int((~d["complete"]).sum()),
        "any_ood_packet_flag_share": float((~d["complete"]).mean()),
        "ood_weight_share_at_armor": float(
            d["ood_armor_w"].sum() / max(d["ood_armor_w"].sum() + d["ood_comp_w"].sum(), 1e-300)
        ),
        "layered_interactions_total": int(d["layered"].sum()),
        "proxy_expected_refused_contacts_per_case": float((d["ood_armor_w"] + d["ood_comp_w"]).mean()),
        "teacher_mean_unresolved_impacts_per_case": float(t["t_unresolved_impacts"].mean()),
    }
    out["armor_hits"] = {
        "proxy_mean": float(d["armor_hits"].mean()),
        "teacher_mean": float(t["t_armor_hits"].mean()),
        "ratio": float(d["armor_hits"].mean() / t["t_armor_hits"].mean()),
    }
    out["component_hits"] = {
        "proxy_mean": float(d["comp_hits"].mean()),
        "teacher_mean": float(t["t_comp_hits"].mean()),
        "ratio": float(d["comp_hits"].mean() / t["t_comp_hits"].mean()),
    }
    sel = t["m_p_hit"]
    out["absorbed_energy_ratio_identified_pairs"] = float(d["absorbed"][sel].sum() / t["deposited_energy"][sel].sum())
    if TREE is not None:
        names = [str(x) for x in d["tree_names"]]
        teacher_tree = np.stack([TREE[n] for n in names], axis=1)
        ok = t["teacher_affected"] == 0.0
        out["tree_outputs_teacher_unaffected_cases"] = {
            "cases": int(ok.sum()),
            **{
                n: {
                    "proxy_mean": float(d["tree"][ok, j].mean()),
                    "teacher_mean": float(teacher_tree[ok, j].mean()),
                    "ratio": float(d["tree"][ok, j].mean() / teacher_tree[ok, j].mean()),
                    "brier": float(np.mean((d["tree"][ok, j] - teacher_tree[ok, j]) ** 2)),
                }
                for j, n in enumerate(names)
            },
        }
    # outcome-blind self-convergence against the finest run of the same accumulation
    for key, ref in reference.items():
        if ref is None or ref is d:
            continue
        ref_union = ref["phit_prod"]
        out.setdefault("self_convergence", {})[key] = {
            "rms_p_hit_union": float(np.sqrt(np.mean((union - ref_union) ** 2))),
            "rms_lambda": float(np.sqrt(np.mean((lam - ref["lambda"]) ** 2))),
            "unreached_status_disagreement_share": float(np.mean((lam == 0) != (ref["lambda"] == 0))),
            "rms_p_failed": float(np.sqrt(np.mean((d["p_fail"] - ref["p_fail"]) ** 2))),
            "rms_p_degraded": float(np.sqrt(np.mean((d["p_deg"] - ref["p_deg"]) ** 2))),
        }
    return out


def review_subset(d_by_name: dict, t) -> dict:
    """Continuity with REVIEW §3.4: three-way complete 173-case subset."""
    teacher_ok = t["teacher_affected"] == 0.0
    common = teacher_ok & d_by_name["az4"]["complete"] & d_by_name["azall"]["complete"]
    out = {"cases": int(common.sum())}
    for name, d in d_by_name.items():
        union = d["phit_prod"]
        poisson = -np.expm1(-np.maximum(d["lambda"], 0.0))
        row = {}
        for head, proxy, tk, mk in (
            ("p_hit_poisson", poisson, "t_p_hit", "m_p_hit"),
            ("p_hit_union", union, "t_p_hit", "m_p_hit"),
            ("p_degraded", d["p_deg"], "t_p_deg", "m_p_deg"),
            ("p_failed", d["p_fail"], "t_p_fail", "m_p_fail"),
        ):
            sel = common[:, None] & t[mk]
            row[head] = float(proxy[sel].mean() / t[tk][sel].mean())
        both = common[:, None] & t["m_p_hit"] & t["m_p_deg"] & t["m_p_fail"] & (t["t_p_hit"] > 0) & (poisson > 0)
        row["failed_over_hit_poisson_denominator"] = {
            "proxy": float((d["p_fail"][both] / poisson[both]).mean()),
            "teacher": float((t["t_p_fail"][both] / t["t_p_hit"][both]).mean()),
            "pairs": int(both.sum()),
        }
        out[name] = row
    return out


def by_profile(d, t, rng) -> dict:
    out = {}
    for profile in sorted(set(t["profile"].tolist())):
        cases = t["profile"] == profile
        out[profile] = {
            "cases": int(cases.sum()),
            "p_hit_union": ratio_and_brier(d["phit_prod"], t["t_p_hit"], t["m_p_hit"], cases, rng, draws=0),
            "p_failed": ratio_and_brier(d["p_fail"], t["t_p_fail"], t["m_p_fail"], cases, rng, draws=0),
            "false_unreached_train": false_unreached(
                d["lambda"], t["hit_indicator"], t["m_p_hit"], cases & (t["split"] == "train"), rng, draws=0
            ),
        }
    return out


def fmt(value, digits=4):
    if value is None:
        return "—"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(fmt(v, digits) for v in value) + "]"
    return f"{value:.{digits}f}"


def pct(value, digits=2):
    return "—" if value is None else f"{100 * value:.{digits}f}%"


TREE: dict | None = None


def load_tree(case_ids) -> dict | None:
    path = WORK / "teacher_damage_tree_raw.json.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        raw = json.load(stream)
    assert [row["case_id"] for row in raw] == list(case_ids)
    names = sorted(raw[0]["damage_tree"]["outputs"])
    return {n: np.asarray([row["damage_tree"]["outputs"][n]["mean"] for row in raw], float) for n in names}


def main() -> int:
    global TREE
    t = dict(np.load(TEACHER, allow_pickle=False))
    TREE = load_tree(t["case_id"])
    names = sorted(p.stem for p in RUNS.glob("*.npz"))
    runs = {name: dict(np.load(RUNS / f"{name}.npz", allow_pickle=False)) for name in names}
    for d in runs.values():
        assert list(d["case_id"]) == list(t["case_id"])
    # hurdle p_hit equals hit_indicator on identified pairs (sanity)
    sanity = float(np.max(np.abs(t["t_p_hit"] - t["hit_indicator"])[t["m_p_hit"]]))
    noise = t["t_p_hit"] * (1 - t["t_p_hit"]) / np.maximum(t["R"][:, None] - 1, 1)
    reference = {
        "vs_azall_k16": runs.get("azall_k16"),
        "vs_azall_k16_reg": runs.get("azall_k16_reg"),
        "vs_azall_k16_reg_dom": runs.get("azall_k16_reg_dom"),
    }
    rng = np.random.default_rng(BOOT_SEED)
    metrics = {
        "teacher_sanity_max_abs_hurdle_p_hit_minus_hit_indicator_on_identified": sanity,
        "teacher_noise_floor_brier_p_hit": {
            split: float(noise[(t["split"] == split)[:, None] & t["m_p_hit"]].mean())
            for split in ("train", "validation")
        } | {"all": float(noise[t["m_p_hit"]].mean())},
        "runs": {name: analyze(name, runs[name], t, rng, reference) for name in names},
    }
    if "az4" in runs and "azall" in runs:
        metrics["review_173_case_subset"] = review_subset(
            {n: runs[n] for n in ("az4", "azall", "azall_reg", "azall_k16", "azall_k16_reg", "azall_k8_reg_dom") if n in runs}, t
        )
    for name in ("azall", "azall_reg", "azall_k8_reg", "azall_k8_reg_dom"):
        if name in runs:
            metrics.setdefault("by_profile", {})[name] = by_profile(runs[name], t, rng)
    OUT_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    lines = ["# Convergence tables (generated by analyze_sweep.py)", ""]
    lines += [
        "All 2,300 development cases; identified pairs only (teacher hurdle masks); no proxy domain-flag filtering.",
        "",
        "| run | rays/case | s/case | p_hit ratio (Poisson) | p_hit ratio (union) | Brier p_hit (Poisson) | Brier p_hit (union) | train FU rate | train FU Wilson up | val FU rate | OOD mass mean | any-OOD cases | p_deg ratio | p_fail ratio | Brier p_fail | failed/hit proxy vs teacher |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        m = metrics["runs"][name]
        ph = m["p_hit"]
        fu = m["false_unreached"]
        lines.append(
            f"| {name} | {m['rays_per_case_mean']:.0f} | {m['runtime']['mean_seconds_per_case']:.3f} "
            f"| {fmt(ph['poisson_1_minus_exp_lambda']['all']['ratio'], 3)} | {fmt(ph['arrival_union']['all']['ratio'], 3)} "
            f"| {fmt(ph['poisson_1_minus_exp_lambda']['all']['brier'])} | {fmt(ph['arrival_union']['all']['brier'])} "
            f"| {pct(fu['train_R8']['rate'])} | {pct(fu['train_R8']['wilson_upper_95'])} | {pct(fu['validation_R32']['rate'])} "
            f"| {fmt(m['domain']['ood_mass_fraction_mean'])} | {m['domain']['any_ood_packet_flag_cases']} "
            f"| {fmt(m['p_degraded']['ratio'], 3)} | {fmt(m['p_failed']['ratio'], 3)} | {fmt(m['p_failed']['brier'])} "
            f"| {fmt(m['conditional_failed_over_hit']['proxy'], 3)} / {fmt(m['conditional_failed_over_hit']['teacher'], 3)} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
