"""Oracle: sweep runs must reproduce the review's proxy_variants.npz bit-for-bit.

az4 == review v0, az4_s7777777 == review v1, azall == review v2 on all 2,300
development cases (expected_hits, p_degraded, p_failed, OOD packet fraction,
domain flag).  Writes results/reproduction_check.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import ROOT, STUDY  # noqa: E402

REVIEW = ROOT / "artifacts/validation/pipeline_expected_actual_20261122/review_20261123/proxy_variants.npz"
PAIRS = {"az4": "v0", "az4_s7777777": "v1", "azall": "v2"}


def main() -> int:
    ref = np.load(REVIEW)
    order = json.loads((STUDY / "baseline/development_case_order.json").read_text())
    out = {}
    ok = True
    for run, variant in PAIRS.items():
        path = STUDY / "results/runs" / f"{run}.npz"
        if not path.exists():
            out[run] = "not run"
            continue
        d = np.load(path)
        assert list(d["case_id"]) == order
        diffs = {
            "lambda": float(np.max(np.abs(d["lambda"] - ref[f"{variant}_lambda"]))),
            "p_deg": float(np.max(np.abs(d["p_deg"] - ref[f"{variant}_p_deg"]))),
            "p_fail": float(np.max(np.abs(d["p_fail"] - ref[f"{variant}_p_fail"]))),
            "ood_packet_frac": float(np.max(np.abs(d["ood_packet_frac"] - ref[f"{variant}_ood_frac"]))),
            "complete_flag_disagreements": int(np.sum(d["complete"] != ref[f"{variant}_complete"])),
        }
        out[run] = {"review_variant": variant, "max_abs_diff": diffs}
        ok &= all(v == 0 for v in diffs.values())
    out["all_bitwise_identical"] = bool(ok)
    (STUDY / "results/reproduction_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
