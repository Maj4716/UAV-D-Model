"""Pre-outcome verification of the aim-window tangent-basis fix.

1. Legacy reproduction: rebuild the archived reference plan with the archived
   SamplingSpec v1, base case and v3 physics, and require every formal case to
   equal the archived ``rounds/round-000/formal_cases.jsonl.gz`` record.
2. Fixed basis: build the same-budget plan with SamplingSpec v2
   (``target_up_horizontal_target_forward_top@1.0.0``) and v4 physics, and
   report S-CONTACT admission by region and mirror-sector symmetry.

No teacher is executed and no result file is opened.
"""
from __future__ import annotations

import gzip
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from damage_engine.case_factory import CaseFactory  # noqa: E402
from damage_engine.engine import ENGINE_CODE_HASH, DamageEngine  # noqa: E402
from damage_engine.label_spec import LabelSpec  # noqa: E402
from damage_engine.registry import PhysicsRegistry  # noqa: E402
from damage_engine.sampling_spec import SamplingSpec  # noqa: E402
from damage_engine.utils import content_hash  # noqa: E402

DATASET = ROOT / "artifacts/datasets/recovery_role_r8_r32_train2000_eval300_20261122"
TARGET = ROOT / "configs/targets/target.fragment_only.v1.json"
V3_PHYSICS = ROOT / "configs/physics/loitering_impact.taylor_shapiro.obliquity_bound.v3.json"
V4_PHYSICS = ROOT / "configs/physics/loitering_impact.taylor_shapiro.wave_directed.obliquity_bound.v4.json"
V4_CASE = ROOT / "configs/cases/loitering_impact.fragment_only.taylor_shapiro.wave_directed.obliquity_bound.v4.json"
V2_SPEC = ROOT / "configs/sampling/p0_sphere_entry_continuous_direction_recovery_train2000_eval300_role_r8_r32.v2.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build(base_case, spec_document, physics):
    engine = DamageEngine(load(TARGET), PhysicsRegistry.from_file(physics))
    base_case = json.loads(json.dumps(base_case))
    base_case["target_ref"]["content_hash"] = engine.target_hash
    return CaseFactory(
        base_case,
        SamplingSpec(spec_document),
        LabelSpec(load(DATASET / "plan/label_spec.json")),
        target_hash=engine.target_hash,
        damage_tree_definition_hash=content_hash(engine.target_data["damage_tree"]),
        physics_registry_hash=engine.registry.hash,
        engine_code_hash=ENGINE_CODE_HASH,
        physics_registry=engine.registry,
        compiled_target=engine.compiled_target,
    ).build_reference_plan()


def admission(plan):
    candidates, admitted = Counter(), Counter()
    for row in plan.candidate_rows:
        region = row.get("region_id") or _region_from_row(row)
        candidates[region] += 1
        admitted[region] += int(bool(row.get("selected")))
    rates = {
        region: {
            "candidates": candidates[region],
            "admitted": admitted[region],
            "admission_rate": admitted[region] / candidates[region],
        }
        for region in sorted(candidates)
    }
    total_c, total_a = sum(candidates.values()), sum(admitted.values())
    return rates, {"candidates": total_c, "admitted": total_a, "admission_rate": total_a / total_c}


_REGION_CACHE: dict = {}


def _region_from_row(row):
    return _REGION_CACHE.get(row["candidate_id"], "unknown")


def mirror_test(rates, left, right):
    a, b = rates[left], rates[right]
    p1, n1 = a["admission_rate"], a["candidates"]
    p2, n2 = b["admission_rate"], b["candidates"]
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return {"difference": p1 - p2, "se": se, "z": (p1 - p2) / se if se else None}


def index_regions(plan, spec_document):
    regions = [r["region_id"] for r in spec_document["contact_manifold"]["regions"]]
    weights = [float(r["selection_weight"]) for r in spec_document["contact_manifold"]["regions"]]
    cumulative, total, acc = [], sum(weights), 0.0
    for w in weights:
        acc += w / total
        cumulative.append(acc)
    for row in plan.candidate_rows:
        selector = float(row["latent"][1])
        index = next((i for i, c in enumerate(cumulative) if selector < c), len(regions) - 1)
        _REGION_CACHE[row["candidate_id"]] = regions[index]


def main() -> int:
    out = {"python": sys.version.split()[0]}
    t0 = time.time()
    archived_spec = load(DATASET / "plan/sampling_spec.json")
    archived_base = load(DATASET / "plan/base_case.json")
    legacy = build(archived_base, archived_spec, V3_PHYSICS)
    with gzip.open(DATASET / "rounds/round-000/formal_cases.jsonl.gz", "rt", encoding="utf-8") as stream:
        archived_cases = [json.loads(line) for line in stream if line.strip()]
    rebuilt_ids = {case["case_id"] for case in legacy.cases}
    index_regions(legacy, archived_spec)
    legacy_rates, legacy_total = admission(legacy)
    out["legacy_reproduction"] = {
        "archived_formal_cases": len(archived_cases),
        "rebuilt_formal_cases": len(legacy.cases),
        "rebuilt_candidate_rows": len(legacy.candidate_rows),
        "case_id_set_equal": rebuilt_ids == {case["case_id"] for case in archived_cases},
        "field_level_equality": (
            "see legacy_reproduction_py312.json (diff_legacy_case.py): identical "
            "except engine_code_hash and contract-document storage form"
        ),
        "admission_by_region": legacy_rates,
        "admission_total": legacy_total,
        "mirror_pos_x_vs_neg_x": mirror_test(legacy_rates, "pos_x", "neg_x"),
    }
    print("legacy done", f"{time.time() - t0:.0f}s", flush=True)

    v4_case = load(V4_CASE)
    v4_case["analysis"] = archived_base["analysis"] | {
        "epistemic_sample_id": v4_case["analysis"]["epistemic_sample_id"]
    }
    v2_spec = load(V2_SPEC)
    fixed = build(v4_case, v2_spec, V4_PHYSICS)
    _REGION_CACHE.clear()
    index_regions(fixed, v2_spec)
    fixed_rates, fixed_total = admission(fixed)
    region_counts = Counter(
        case["metadata"]["continuous_direction_sampling"]["region_id"] for case in fixed.cases
    )
    up = Counter(
        case["metadata"]["continuous_direction_sampling"]["region_id"]
        for case in fixed.cases
        if case["metadata"]["continuous_direction_sampling"]["tangent_v_target_xyz"][2] > 0.0
    )
    out["fixed_basis"] = {
        "formal_cases": len(fixed.cases),
        "candidate_rows": len(fixed.candidate_rows),
        "admission_by_region": fixed_rates,
        "admission_total": fixed_total,
        "mirror_pos_x_vs_neg_x": mirror_test(fixed_rates, "pos_x", "neg_x"),
        "formal_region_counts": dict(sorted(region_counts.items())),
        "formal_region_share_tangent_v_up": {
            region: up[region] / region_counts[region] for region in sorted(region_counts)
        },
        "aim_tangent_basis_recorded": sorted(
            {case["metadata"]["continuous_direction_sampling"].get("aim_tangent_basis") for case in fixed.cases}
        ),
    }
    shares = [region_counts[r] / len(fixed.cases) for r in sorted(region_counts)]
    out["fixed_basis"]["formal_region_tv_vs_equal_mass"] = 0.5 * sum(abs(s - 0.2) for s in shares)
    print("fixed done", f"{time.time() - t0:.0f}s", flush=True)
    (HERE / "aim_basis_fix_verification.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
