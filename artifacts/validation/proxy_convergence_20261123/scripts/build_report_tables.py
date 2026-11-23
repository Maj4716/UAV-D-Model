"""Assemble the REPORT.md tables from the analysis outputs.

Inputs : results/convergence_metrics.json, results/gate_evaluation.json,
         results/runtime_benchmark.json, results/runs/*.npz (seed spread only)
Output : results/report_tables.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import STUDY  # noqa: E402

RES = STUDY / "results"
M = json.loads((RES / "convergence_metrics.json").read_text(encoding="utf-8"))
RUNS = M["runs"]
BENCH = json.loads((RES / "runtime_benchmark.json").read_text(encoding="utf-8")) if (RES / "runtime_benchmark.json").exists() else None
GATE = json.loads((RES / "gate_evaluation.json").read_text(encoding="utf-8"))


def f(x, d=4):
    return "—" if x is None else f"{x:.{d}f}"


def p(x, d=2):
    return "—" if x is None else f"{100 * x:.{d}f}%"


def ci(pair, d=4, as_pct=False):
    if not pair:
        return ""
    if as_pct:
        return f" [{100 * pair[0]:.2f}, {100 * pair[1]:.2f}]"
    return f" [{pair[0]:.{d}f}, {pair[1]:.{d}f}]"


def runtime(name):
    if BENCH and name in BENCH["configs"]:
        return f(BENCH["configs"][name]["mean_seconds_per_case"], 3)
    return f(RUNS[name]["runtime"]["mean_seconds_per_case"], 3) + "*"


def selfconv(name, key="vs_azall_k16"):
    row = RUNS[name].get("self_convergence", {}).get(key)
    return row


lines: list[str] = []
add = lines.append

# ---- table 1: azimuthal rays ---------------------------------------------
add("### T1. Azimuthal ray count (default seed, 1 realization, historical accumulation)")
add("")
add("| config | rays/case | s/case | p_hit ratio Poisson | p_hit ratio union | Brier Poisson | Brier union | train FU (Wilson up) | val FU | RMS Δp_hit vs ref | unreached-status flips vs ref | OOD mass mean | any-OOD cases |")
add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for name in ("az2", "az4", "az8", "az16", "azall"):
    r = RUNS[name]
    sc = selfconv(name) or {}
    add(
        f"| `{name}` | {r['rays_per_case_mean']:.0f} | {runtime(name)} "
        f"| {f(r['p_hit']['poisson_1_minus_exp_lambda']['all']['ratio'], 3)} | {f(r['p_hit']['arrival_union']['all']['ratio'], 3)} "
        f"| {f(r['p_hit']['poisson_1_minus_exp_lambda']['all']['brier'])} | {f(r['p_hit']['arrival_union']['all']['brier'])} "
        f"| {p(r['false_unreached']['train_R8']['rate'])} ({p(r['false_unreached']['train_R8']['wilson_upper_95'])}) "
        f"| {p(r['false_unreached']['validation_R32']['rate'])} "
        f"| {f(sc.get('rms_p_hit_union'))} | {p(sc.get('unreached_status_disagreement_share'))} "
        f"| {f(r['domain']['ood_mass_fraction_mean'])} | {r['domain']['any_ood_packet_flag_cases']} |"
    )
add("")

# ---- table 2: layer quadrature ---------------------------------------------
add("### T2. Layer quadrature (all-fragment rays)")
add("")
add("| config | layer_quadrature_samples | layered interactions (all cases) | Brier union | train FU | p_fail ratio |")
add("|---|---:|---:|---:|---:|---:|")
for name, q in (("azall_q4", 4), ("azall", 16), ("azall_q64", 64)):
    r = RUNS[name]
    add(f"| `{name}` | {q} | {r['domain']['layered_interactions_total']} | {f(r['p_hit']['arrival_union']['all']['brier'])} | {p(r['false_unreached']['train_R8']['rate'])} | {f(r['p_failed']['ratio'], 3)} |")
add("")

# ---- table 3: seeds -------------------------------------------------------
add("### T3. Source seed (single realization)")
add("")
seed_sets = {
    "az4": ["az4", "az4_s7777777", "az4_s2026112301", "az4_s2026112302"],
    "azall": ["azall", "azall_s7777777", "azall_s2026112301", "azall_s2026112302"],
}
add("| rays | seeds | p_hit ratio union (min–max) | Brier union (min–max) | train FU (min–max) | any-OOD cases (min–max) | OOD mass mean (min–max) | mean pairwise RMS Δp_hit between seeds | mean pairwise unreached flips |")
add("|---|---:|---|---|---|---|---|---:|---:|")
for label, names in seed_sets.items():
    names = [n for n in names if n in RUNS]
    arrays = {n: np.load(RES / "runs" / f"{n}.npz") for n in names}
    rms, flips = [], []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rms.append(float(np.sqrt(np.mean((arrays[a]["phit_prod"] - arrays[b]["phit_prod"]) ** 2))))
            flips.append(float(np.mean((arrays[a]["lambda"] == 0) != (arrays[b]["lambda"] == 0))))

    def span(getter, fmt=f):
        vals = [getter(RUNS[n]) for n in names]
        return f"{fmt(min(vals))}–{fmt(max(vals))}"

    add(
        f"| {label} | {len(names)} "
        f"| {span(lambda r: r['p_hit']['arrival_union']['all']['ratio'], lambda x: f(x, 3))} "
        f"| {span(lambda r: r['p_hit']['arrival_union']['all']['brier'])} "
        f"| {span(lambda r: r['false_unreached']['train_R8']['rate'], p)} "
        f"| {span(lambda r: r['domain']['any_ood_packet_flag_cases'], lambda x: str(x))} "
        f"| {span(lambda r: r['domain']['ood_mass_fraction_mean'])} "
        f"| {f(float(np.mean(rms)))} | {p(float(np.mean(flips)))} |"
    )
add("")

# ---- table 4: realizations --------------------------------------------------
add("### T4. Source-realization averaging (all-fragment rays)")
add("")
add("| config | K | s/case | p_hit ratio union | Brier union [95% CI] | Brier Poisson | train FU (Wilson up) | val FU | RMS Δp_hit vs K16 | RMS Δp_fail vs K16 (same options) | any-OOD cases | OOD mass mean |")
add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for name, k in (("azall", 1), ("azall_k2", 2), ("azall_k4", 4), ("azall_k8", 8), ("azall_k16", 16), ("az4_k8", 8), ("azall_reg_dom", 1), ("azall_k4_reg_dom", 4), ("azall_k8_reg_dom", 8), ("azall_k16_reg_dom", 16)):
    if name not in RUNS:
        continue
    r = RUNS[name]
    sc = selfconv(name, "vs_azall_k16_reg_dom" if name.endswith("_dom") else "vs_azall_k16") or {}
    add(
        f"| `{name}` | {k} | {runtime(name)} | {f(r['p_hit']['arrival_union']['all']['ratio'], 3)} "
        f"| {f(r['p_hit']['arrival_union']['all']['brier'])}{ci(r['p_hit']['arrival_union']['all'].get('brier_ci95'))} "
        f"| {f(r['p_hit']['poisson_1_minus_exp_lambda']['all']['brier'])} "
        f"| {p(r['false_unreached']['train_R8']['rate'])} ({p(r['false_unreached']['train_R8']['wilson_upper_95'])}) "
        f"| {p(r['false_unreached']['validation_R32']['rate'])} "
        f"| {f(sc.get('rms_p_hit_union'))} | {f(sc.get('rms_p_failed'))} "
        f"| {r['domain']['any_ood_packet_flag_cases']} | {f(r['domain']['ood_mass_fraction_mean'])} |"
    )
add("")
add(f"Teacher Monte-Carlo noise floor of the p_hit Brier, E[q̂(1−q̂)/(R−1)]: all {f(M['teacher_noise_floor_brier_p_hit']['all'])}, train R8 {f(M['teacher_noise_floor_brier_p_hit']['train'])}, validation R32 {f(M['teacher_noise_floor_brier_p_hit']['validation'])}.")
add("")

# ---- table 5: accumulation ----------------------------------------------------
add("### T5. Damage accumulation: historical per-hit product vs registry spatial clusters")
add("")
add("| config | p_deg ratio | p_fail ratio [95% CI] | Brier p_deg | Brier p_fail [95% CI] | failed/hit proxy | failed/hit teacher | degraded/hit proxy | degraded/hit teacher | RMS Δp_fail vs K16 (same accumulation and censoring) |")
add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for name in ("az4", "az4_reg", "azall", "azall_reg", "azall_k4", "azall_k4_reg", "azall_k8", "azall_k8_reg", "azall_k16", "azall_k16_reg", "azall_reg_dom", "azall_k8_reg_dom", "azall_k16_reg_dom"):
    if name not in RUNS:
        continue
    r = RUNS[name]
    sc = selfconv(name, "vs_azall_k16_reg_dom" if name.endswith("_dom") else "vs_azall_k16_reg") or {}
    add(
        f"| `{name}` | {f(r['p_degraded']['ratio'], 3)} | {f(r['p_failed']['ratio'], 3)}{ci(r['p_failed'].get('ratio_ci95'), 3)} "
        f"| {f(r['p_degraded']['brier'])} | {f(r['p_failed']['brier'])}{ci(r['p_failed'].get('brier_ci95'))} "
        f"| {f(r['conditional_failed_over_hit']['proxy'], 3)} | {f(r['conditional_failed_over_hit']['teacher'], 3)} "
        f"| {f(r['conditional_degraded_over_hit']['proxy'], 3)} | {f(r['conditional_degraded_over_hit']['teacher'], 3)} "
        f"| {f(sc.get('rms_p_failed'))} |"
    )
add("")

add("### T5b. Domain censoring: whole packet on any refused branch vs refused quantile mass only")
add("")
add("| config | OOD mass fraction mean | any-OOD flag cases | proxy expected refused contacts / case | teacher unresolved impacts / replicate | armor hits proxy/teacher | component hits proxy/teacher | absorbed energy proxy/teacher | p_hit ratio union | Brier union | train FU | p_fail ratio |")
add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for name in ("azall", "azall_dom", "azall_reg", "azall_reg_dom", "azall_k8_reg", "azall_k8_reg_dom"):
    if name not in RUNS:
        continue
    r = RUNS[name]
    dm = r["domain"]
    add(
        f"| `{name}` | {f(dm['ood_mass_fraction_mean'])} | {dm['any_ood_packet_flag_cases']} | {f(dm['proxy_expected_refused_contacts_per_case'], 3)} "
        f"| {f(dm['teacher_mean_unresolved_impacts_per_case'], 3)} | {f(r['armor_hits']['ratio'], 3)} | {f(r['component_hits']['ratio'], 3)} "
        f"| {f(r['absorbed_energy_ratio_identified_pairs'], 3)} | {f(r['p_hit']['arrival_union']['all']['ratio'], 3)} "
        f"| {f(r['p_hit']['arrival_union']['all']['brier'])} | {p(r['false_unreached']['train_R8']['rate'])} | {f(r['p_failed']['ratio'], 3)} |"
    )
add("")

sub = M.get("review_173_case_subset", {})
if sub:
    add(f"### T6. Continuity with REVIEW §3.4 ({sub['cases']}-case three-way-complete subset)")
    add("")
    add("| config | p_hit ratio (Poisson) | p_hit ratio (union) | p_deg ratio | p_fail ratio | failed/hit proxy (Poisson denominator) | failed/hit teacher |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    for name, row in sub.items():
        if name == "cases":
            continue
        fh = row["failed_over_hit_poisson_denominator"]
        add(f"| `{name}` | {f(row['p_hit_poisson'], 3)} | {f(row['p_hit_union'], 3)} | {f(row['p_degraded'], 3)} | {f(row['p_failed'], 3)} | {f(fh['proxy'], 3)} | {f(fh['teacher'], 3)} |")
    add("")

# ---- table 7: tree outputs ----------------------------------------------------
tree_names = None
add("### T7. Damage-tree outputs, proxy/teacher mean ratio (teacher-unaffected cases)")
add("")
cols = ("az4", "azall", "azall_reg", "azall_reg_dom", "azall_k8_reg_dom", "azall_k16_reg_dom")
cols = [c for c in cols if c in RUNS and "tree_outputs_teacher_unaffected_cases" in RUNS[c]]
if cols:
    tree_names = [k for k in RUNS[cols[0]]["tree_outputs_teacher_unaffected_cases"] if k != "cases"]
    add("| output | teacher mean | " + " | ".join(f"`{c}`" for c in cols) + " |")
    add("|---|---:|" + "---:|" * len(cols))
    for t in tree_names:
        teacher = RUNS[cols[0]]["tree_outputs_teacher_unaffected_cases"][t]["teacher_mean"]
        add(f"| {t} | {f(teacher)} | " + " | ".join(f(RUNS[c]["tree_outputs_teacher_unaffected_cases"][t]["ratio"], 3) for c in cols) + " |")
    add(f"\nCases: {RUNS[cols[0]]['tree_outputs_teacher_unaffected_cases']['cases']} (teacher affected-replicate fraction = 0; selection uses the teacher side only).")
    add("")

# ---- table 8: gate ------------------------------------------------------------
add("### T8. Promotion gate on train R8 (2,000 cases)")
add("")
add("| status | config | identified pairs | proxy-unreached pairs | teacher-positive | FU rate | Wilson 95% upper | case-cluster bootstrap 95% | gate ≤ 3% |")
add("|---|---|---:|---:|---:|---:|---:|---|---|")
rows = [("pre-declared", GATE["primary"])] + [("post-hoc", v) for v in GATE["posthoc"].values()]
for status, row in rows:
    g = row["train_R8"]
    add(
        f"| {status} | `{row['run']}` | {g['identified_pairs']} | {g['proxy_unreached_pairs']} | {g['teacher_positive_among_unreached']} "
        f"| {p(g['false_unreached_rate'], 3)} | {p(g['wilson_upper_95'], 3)} | {ci(g.get('case_cluster_bootstrap_95'), as_pct=True)} "
        f"| {'PASS' if g['passes_gate'] else 'FAIL'} |"
    )
add("")
sec = GATE["primary"]["train_R8_unmasked_exposure_audit_secondary"]
add(f"Secondary (unmasked `exposure_audit`, declared config): {sec['teacher_positive_among_proxy_unreached']}/{sec['proxy_unreached_pair_count']} = {p(sec['false_unreached_rate'], 3)}, Wilson upper {p(sec['false_unreached_wilson_upper'], 3)}.")
val = GATE["primary"]["validation_R32_supplementary"]
add(f"Validation R32 (supplementary, declared config): {p(val['false_unreached_rate'], 3)}, Wilson upper {p(val['wilson_upper_95'], 3)}.")
add("")
add("Declared config by profile (train): " + "; ".join(
    f"{k} {p(v['false_unreached_rate'])} (Wilson {p(v['wilson_upper_95'])})" for k, v in GATE["primary"]["train_by_profile"].items()))
add("")
add("Declared config by region (train): " + "; ".join(
    f"{k} {p(v['false_unreached_rate'])} (Wilson {p(v['wilson_upper_95'])})" for k, v in GATE["primary"]["train_by_region"].items()))
add("")

(RES / "report_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
