"""Verification of the Taylor-Shapiro tilt-sign fix.

1. Direction check (pre-outcome): for every profile, compare each fragment's
   nominal lean with the convention-free Taylor statement "the deviation from
   the casing normal follows the detonation wave's tangential component".
   v3 (legacy convention) is expected to fail it everywhere; v4 to pass it
   everywhere.
2. Legacy end-to-end reproduction: re-run a few archived train cases with the
   v3 registry and require summary, component metrics and damage-tree outputs
   to equal the archived teacher results exactly.  Run this with the Python
   environment that produced the archive (numpy rounding differs across builds).

Only train results are decoded; sealed-test rows are skipped by the loader.
"""
from __future__ import annotations

import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from damage_engine.engine import DamageEngine  # noqa: E402
from damage_engine.fragments import (  # noqa: E402
    _sample_taylor_shapiro_directions,
    build_taylor_shapiro_ring_layout,
)
from damage_engine.munition_profiles import apply_munition_profile  # noqa: E402
from damage_engine.registry import PhysicsRegistry  # noqa: E402
from tools.evaluate_proxy_teacher_distribution import load_development_evidence  # noqa: E402

DATASET = ROOT / "artifacts/datasets/recovery_role_r8_r32_train2000_eval300_20261122"
TARGET = ROOT / "configs/targets/target.fragment_only.v1.json"
CONFIGS = {
    "v3_legacy": (
        ROOT / "configs/physics/loitering_impact.taylor_shapiro.obliquity_bound.v3.json",
        ROOT / "configs/cases/loitering_impact.fragment_only.taylor_shapiro.obliquity_bound.v3.json",
    ),
    "v4_wave_directed": (
        ROOT / "configs/physics/loitering_impact.taylor_shapiro.wave_directed.obliquity_bound.v4.json",
        ROOT / "configs/cases/loitering_impact.fragment_only.taylor_shapiro.wave_directed.obliquity_bound.v4.json",
    ),
}
REPRODUCED_CASES = 3


def direction_check() -> dict:
    out: dict = {}
    for tag, (physics, case_path) in CONFIGS.items():
        registry = PhysicsRegistry.from_file(physics)
        for profile in ("small_front", "medium_front", "large_front"):
            case = json.loads(case_path.read_text(encoding="utf-8-sig"))
            apply_munition_profile(case, profile)
            source_doc = case["warhead"]["fragment_source"]
            source = registry.fragment_source(source_doc["model_id"])
            layout = build_taylor_shapiro_ring_layout(case, source, source_doc["total_count"])
            speed = 2000.0
            directions = _sample_taylor_shapiro_directions(
                layout,
                np.eye(3),
                np.random.default_rng(0),
                gurney_speed=speed,
                detonation_velocity_mps=float(source_doc["detonation_velocity_mps"]),
                spread_sigma_deg=0.0,
            )
            n, w = layout.local_normals_body, layout.wave_directions_body
            tangential = w - np.einsum("ij,ij->i", w, n)[:, None] * n
            deviation = directions - np.einsum("ij,ij->i", directions, n)[:, None] * n
            consistent = np.einsum("ij,ij->i", deviation, tangential) > 0.0
            tilt = np.degrees(np.arcsin(directions @ layout.body_axis))
            out[f"{tag}|{profile}"] = {
                "angle_convention": layout.angle_convention,
                "fragments": int(len(directions)),
                "share_consistent_with_taylor": float(consistent.mean()),
                "axial_tilt_deg_mean": float(tilt.mean()),
                "axial_tilt_deg_range": [float(tilt.min()), float(tilt.max())],
                "taylor_bound_deg": math.degrees(
                    math.asin(speed / (2.0 * float(source_doc["detonation_velocity_mps"])))
                ),
            }
    return out


def legacy_reproduction() -> dict:
    evidence, audit = load_development_evidence(DATASET)
    assert audit.get("test_result_json_lines_decoded", 0) == 0
    engine = DamageEngine(
        json.loads(TARGET.read_text(encoding="utf-8-sig")),
        PhysicsRegistry.from_file(CONFIGS["v3_legacy"][0]),
    )
    train = sorted((item for item in evidence if item.split == "train"), key=lambda it: it.case["case_id"])
    rows = []
    for item in train[:REPRODUCED_CASES]:
        rerun = engine.run(deepcopy(item.case)).to_dict()
        archived = item.result
        rows.append(
            {
                "case_id": item.case["case_id"],
                "profile": item.case["munition"]["profile_id"],
                "summary_equal": rerun["summary"] == archived["summary"],
                "components_equal": rerun["components"] == archived["components"],
                "damage_tree_outputs_equal": rerun["damage_tree"]["outputs"] == archived["damage_tree"]["outputs"],
            }
        )
    return {
        "cases": rows,
        "all_equal": all(r["summary_equal"] and r["components_equal"] and r["damage_tree_outputs_equal"] for r in rows),
    }


def main() -> int:
    out = {"python": sys.version.split()[0], "direction_check": direction_check()}
    out["legacy_end_to_end_reproduction"] = legacy_reproduction()
    (HERE / "taylor_fix_verification.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
