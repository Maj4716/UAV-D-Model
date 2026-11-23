"""Shared helpers: frozen development inputs and teacher arrays.

``build_cache`` decodes the 2,300 train/validation rows once through
``tools.evaluate_proxy_teacher_distribution.load_development_evidence`` (sealed
test rows are skipped before JSON decoding) and stores
  work/development_cases.json.gz   pre-outcome case inputs, evidence order
  work/teacher_arrays.npz          teacher targets aligned to that order
Teacher arrays are only read by the analysis scripts.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = HERE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET = ROOT / "artifacts/datasets/recovery_role_r8_r32_train2000_eval300_20261122"
TARGET = ROOT / "configs/targets/target.fragment_only.v1.json"
PHYSICS = ROOT / "configs/physics/loitering_impact.taylor_shapiro.obliquity_bound.v3.json"
WORK = STUDY / "work"
CASES = WORK / "development_cases.json.gz"
TEACHER = WORK / "teacher_arrays.npz"
AUDIT = WORK / "source_audit.json"


def build_cache() -> None:
    from damage_engine.hurdle_targets import extract_component_hurdle_targets
    from tools.evaluate_proxy_teacher_distribution import load_development_evidence

    WORK.mkdir(parents=True, exist_ok=True)
    evidence, audit = load_development_evidence(DATASET)
    assert audit["test_result_json_lines_decoded"] == 0
    AUDIT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    with gzip.open(CASES, "wt", encoding="utf-8") as stream:
        json.dump([item.case for item in evidence], stream)
    comps = list(evidence[0].result["component_order"])
    n, k = len(evidence), len(comps)
    arr = {
        "case_id": np.asarray([item.case["case_id"] for item in evidence]),
        "component_ids": np.asarray(comps),
        "split": np.asarray([item.split for item in evidence]),
        "profile": np.asarray([item.case["munition"]["profile_id"] for item in evidence]),
        "region": np.asarray(
            [item.case["metadata"]["contact_manifold_sampling"]["region_id"] for item in evidence]
        ),
        "R": np.asarray([int(item.case["analysis"]["replicates"]) for item in evidence]),
        "teacher_affected": np.zeros(n),
        "hit_indicator": np.zeros((n, k)),
        "deposited_energy": np.zeros((n, k)),
        "t_p_hit": np.zeros((n, k)), "m_p_hit": np.zeros((n, k), bool),
        "t_p_deg": np.zeros((n, k)), "m_p_deg": np.zeros((n, k), bool),
        "t_p_fail": np.zeros((n, k)), "m_p_fail": np.zeros((n, k), bool),
        "t_comp_hits": np.zeros(n),
        "t_armor_hits": np.zeros(n),
        "t_resolved_impacts": np.zeros(n),
        "t_unresolved_impacts": np.zeros(n),
        "t_unresolved_armor_impacts": np.zeros(n),
    }
    for i, item in enumerate(evidence):
        result = item.result
        arr["teacher_affected"][i] = float(
            result["penetration_domain_coverage"]["affected_replicate_fraction"]
        )
        value = result["summary"]["total_component_hits"].get("mean")
        arr["t_comp_hits"][i] = 0.0 if value is None else float(value)
        value = result["summary"]["total_armor_hits"].get("mean")
        arr["t_armor_hits"][i] = 0.0 if value is None else float(value)
        coverage = result["penetration_domain_coverage"]
        arr["t_resolved_impacts"][i] = float(coverage["mean_resolved_impact_count"])
        arr["t_unresolved_impacts"][i] = float(coverage["mean_unresolved_impact_count"])
        arr["t_unresolved_armor_impacts"][i] = float(coverage["mean_unresolved_armor_impact_count"])
        targets = extract_component_hurdle_targets(result)["components"]
        for j, c in enumerate(comps):
            fragment = result["components"][c]["fragment_metrics"]
            arr["hit_indicator"][i, j] = float(fragment["hit_indicator"]["mean"])
            arr["deposited_energy"][i, j] = float(fragment["deposited_energy_j"]["mean"])
            t = targets[c]
            arr["t_p_hit"][i, j] = float(t["p_hit"]["value"])
            arr["m_p_hit"][i, j] = bool(t["p_hit"]["mask"])
            sp = t["state_probabilities"]
            arr["t_p_deg"][i, j] = float(sp["p_degraded"]["value"])
            arr["m_p_deg"][i, j] = bool(sp["p_degraded"]["mask"])
            arr["t_p_fail"][i, j] = float(sp["p_failed"]["value"])
            arr["m_p_fail"][i, j] = bool(sp["p_failed"]["mask"])
    np.savez_compressed(TEACHER, **arr)


def load_cases() -> list[dict]:
    with gzip.open(CASES, "rt", encoding="utf-8") as stream:
        return json.load(stream)


if __name__ == "__main__":
    build_cache()
    print("cached", CASES, TEACHER)
