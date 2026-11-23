"""Is the declared gate configuration sensitive to the Python/numpy environment?

The archive was generated with Python 3.12.13 / numpy 2.4.6; the study ran with
Python 3.13.5 / numpy 2.1.3.  This re-evaluates the declared configuration
(``azall``) under the *current* interpreter on all 2,300 development cases,
compares with the stored run, and recomputes the train-R8 gate.

Usage (from this folder): <python> check_environment_sensitivity.py [RUN]
Writes results/environment_sensitivity_<RUN>_py<major><minor>.json.
"""
from __future__ import annotations

import json
import os
import platform
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import STUDY, TEACHER, load_cases  # noqa: E402
from evaluate_gate import gate_numbers  # noqa: E402
from run_sweep import CONFIGS, _evaluate, _init  # noqa: E402

KEYS = ["lambda", "phit_prod", "p_deg", "p_fail", "energy", "ood_mass_frac"]


def main(argv: list[str]) -> int:
    name = argv[0] if argv else "azall"
    stored = dict(np.load(STUDY / "results" / "runs" / f"{name}.npz", allow_pickle=False))
    stored_meta = json.loads((STUDY / "results" / "runs" / f"{name}.json").read_text(encoding="utf-8"))
    cases = load_cases()
    workers = int(os.environ.get("SWEEP_WORKERS", "14"))
    with Pool(workers, initializer=_init) as pool:
        rows = pool.map(_evaluate, [(i, c, CONFIGS[name]) for i, c in enumerate(cases)], chunksize=8)
    rows.sort(key=lambda row: row["index"])
    fresh = {key: np.asarray([row[key] for row in rows]) for key in KEYS}
    diffs = {key: float(np.max(np.abs(fresh[key] - stored[key]))) for key in KEYS}
    flips = int(np.sum((fresh["lambda"] == 0.0) != (stored["lambda"] == 0.0)))
    t = dict(np.load(TEACHER, allow_pickle=False))
    train = t["split"] == "train"
    rng = np.random.default_rng(2026112312)
    report = {
        "run": name,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "stored_run_python_numpy": "3.13.5 / 2.1.3 (sweep environment)",
        "stored_proxy_module_sha256": stored_meta["proxy_module_sha256"],
        "cases": len(cases),
        "max_abs_diff_vs_stored": diffs,
        "bitwise_identical": all(value == 0.0 for value in diffs.values()),
        "unreached_status_flips_vs_stored": flips,
        "gate_train_R8_this_environment": gate_numbers(fresh["lambda"], t, train, rng),
    }
    tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    out = STUDY / "results" / f"environment_sensitivity_{name}_{tag}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    g = report["gate_train_R8_this_environment"]
    print(json.dumps({k: report[k] for k in ("python", "numpy", "bitwise_identical", "max_abs_diff_vs_stored", "unreached_status_flips_vs_stored")}, indent=1))
    print("gate", g["false_unreached_rate"], g["wilson_upper_95"], g["passes_gate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
