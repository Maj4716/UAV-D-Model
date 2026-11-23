"""Re-evaluate the pre-declared proxy promotion gate on the 2,000-case train R8 split.

Primary (declared in GATE_DECLARATION.json before any run of this study):
  configuration ``azall`` (one ray per fragment, default seed, one realization);
  identified pairs (teacher hurdle p_hit mask); proxy unreached = expected_hits == 0;
  false = teacher fragment hit_indicator mean > 0; Wilson 95% upper <= 3%.
Secondary: unmasked damage_engine.exposure_audit result, case-cluster bootstrap,
validation R32, per-profile/per-region breakdown.
Post-hoc (not declared, reported separately): any extra run names given on the
command line, e.g. the recommended default chosen from the sweep.

Usage: python evaluate_gate.py [POSTHOC_RUN ...]
Writes results/gate_evaluation.json.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import AUDIT, ROOT, STUDY, TEACHER  # noqa: E402

sys.path.insert(0, str(ROOT))
from damage_engine.exposure_audit import build_exposure_teacher_audit  # noqa: E402

DECLARATION = STUDY / "GATE_DECLARATION.json"
RUNS = STUDY / "results" / "runs"
OUT = STUDY / "results" / "gate_evaluation.json"
DECLARED_RUN = "azall"
DECLARED_KWARGS = {"azimuths_per_ring": 1_000_000}
GATE_MAX = 0.03
Z95 = 1.959963984540054
BOOT = 4000
BOOT_SEED = 2026112312


def wilson_upper(successes: int, trials: int) -> float | None:
    if trials == 0:
        return None
    p = successes / trials
    denominator = 1.0 + Z95 * Z95 / trials
    centre = (p + Z95 * Z95 / (2.0 * trials)) / denominator
    half = Z95 * math.sqrt(p * (1 - p) / trials + Z95 * Z95 / (4 * trials * trials)) / denominator
    return min(1.0, centre + half)


def gate_numbers(lam, t, cases, rng, *, bootstrap=True) -> dict:
    mask = t["m_p_hit"] & cases[:, None]
    unreached = mask & (-np.expm1(-np.maximum(lam, 0.0)) == 0.0)
    false = unreached & (t["hit_indicator"] > 0.0)
    trials, successes = int(unreached.sum()), int(false.sum())
    out = {
        "cases": int(cases.sum()),
        "identified_pairs": int(mask.sum()),
        "proxy_unreached_pairs": trials,
        "teacher_positive_among_unreached": successes,
        "false_unreached_rate": None if trials == 0 else successes / trials,
        "wilson_upper_95": wilson_upper(successes, trials),
    }
    out["passes_gate"] = out["wilson_upper_95"] is not None and out["wilson_upper_95"] <= GATE_MAX
    if bootstrap and trials:
        num, den = false.sum(1).astype(float), unreached.sum(1).astype(float)
        idx = np.where(cases)[0]
        chosen = idx[rng.integers(0, idx.size, size=(BOOT, idx.size))]
        rates = num[chosen].sum(1) / den[chosen].sum(1)
        out["case_cluster_bootstrap_95"] = np.quantile(rates, [0.025, 0.975]).tolist()
        out["case_cluster_bootstrap_upper_975_below_gate"] = bool(np.quantile(rates, 0.975) <= GATE_MAX)
    return out


def unmasked_exposure_audit(lam, t, cases) -> dict:
    comps = [str(c) for c in t["component_ids"]]
    exposure, teacher = [], []
    for i in np.where(cases)[0]:
        cid = str(t["case_id"][i])
        exposure.append({"candidate_id": cid, "cell_id": "all",
                         "component_lambda": {c: float(lam[i, j]) for j, c in enumerate(comps)}})
        teacher.append({"candidate_id": cid,
                        "component_hit_probability": {c: float(t["hit_indicator"][i, j]) for j, c in enumerate(comps)}})
    audit = build_exposure_teacher_audit(exposure, teacher)
    return {k: audit[k] for k in (
        "audited_candidate_count", "proxy_unreached_pair_count",
        "teacher_positive_among_proxy_unreached", "false_unreached_rate",
        "false_unreached_wilson_upper", "confidence")}


def evaluate_run(name: str, t, rng, *, declared: bool) -> dict:
    d = np.load(RUNS / f"{name}.npz", allow_pickle=False)
    meta = json.loads((RUNS / f"{name}.json").read_text(encoding="utf-8"))
    assert list(d["case_id"]) == list(t["case_id"])
    lam = d["lambda"]
    train = t["split"] == "train"
    assert set(t["R"][train].tolist()) == {8}
    result = {
        "status": "pre-declared primary" if declared else "post-hoc secondary (not declared)",
        "run": name,
        "kwargs": meta["kwargs"],
        "proxy_module_sha256_at_run": meta["proxy_module_sha256"],
        "train_R8": gate_numbers(lam, t, train, rng),
        "validation_R32_supplementary": gate_numbers(lam, t, t["split"] == "validation", rng, bootstrap=False),
        "train_R8_unmasked_exposure_audit_secondary": unmasked_exposure_audit(lam, t, train),
        "train_by_profile": {
            p: gate_numbers(lam, t, train & (t["profile"] == p), rng, bootstrap=False)
            for p in sorted(set(t["profile"].tolist()))
        },
        "train_by_region": {
            r: gate_numbers(lam, t, train & (t["region"] == r), rng, bootstrap=False)
            for r in sorted(set(t["region"].tolist()))
        },
    }
    return result


def main(argv: list[str]) -> int:
    declaration_text = DECLARATION.read_text(encoding="utf-8")
    declaration = json.loads(declaration_text)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert audit["test_result_json_lines_decoded"] == 0
    teacher_hash = audit["source_engine"]["code_hash"]
    assert teacher_hash == declaration["gate"]["teacher_engine_code_hash"], teacher_hash
    meta = json.loads((RUNS / f"{DECLARED_RUN}.json").read_text(encoding="utf-8"))
    assert meta["kwargs"] == DECLARED_KWARGS, meta["kwargs"]
    reproduction = json.loads((STUDY / "results/reproduction_check.json").read_text(encoding="utf-8"))
    oracle_ok = reproduction[DECLARED_RUN]["max_abs_diff"]["lambda"] == 0.0
    # both timestamps are local (+08:00) ISO strings; compare YYYY-MM-DDTHH:MM:SS
    run_after_declaration = meta["finished_at_local"][:19] > declaration["declared_at_local"][:19]
    assert run_after_declaration, "declared configuration must be run after the declaration"
    t = dict(np.load(TEACHER, allow_pickle=False))
    rng = np.random.default_rng(BOOT_SEED)
    report = {
        "declaration_sha256": hashlib.sha256(declaration_text.encode("utf-8")).hexdigest(),
        "declared_at_local": declaration["declared_at_local"],
        "declared_run_finished_at_local": meta["finished_at_local"],
        "teacher_engine_code_hash": teacher_hash,
        "gate_criterion": f"false-unreached Wilson 95% upper <= {GATE_MAX}",
        "reproduction_oracle_v2_lambda_bitwise_identical": bool(oracle_ok),
        "primary": evaluate_run(DECLARED_RUN, t, rng, declared=True),
        "posthoc": {name: evaluate_run(name, t, rng, declared=False) for name in argv},
    }
    report["primary"]["gate_verdict"] = (
        "PASS" if (oracle_ok and report["primary"]["train_R8"]["passes_gate"]) else "FAIL"
    )
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    p = report["primary"]["train_R8"]
    print(json.dumps({"verdict": report["primary"]["gate_verdict"], **{k: p[k] for k in (
        "identified_pairs", "proxy_unreached_pairs", "teacher_positive_among_unreached",
        "false_unreached_rate", "wilson_upper_95", "case_cluster_bootstrap_95")}}, indent=2))
    for name, row in report["posthoc"].items():
        q = row["train_R8"]
        print(name, q["false_unreached_rate"], q["wilson_upper_95"], q.get("case_cluster_bootstrap_95"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
