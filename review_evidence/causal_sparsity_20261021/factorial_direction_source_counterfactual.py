"""Diagnostic counterfactual for causal localization of sparse KMFC support.

This script never mutates production inputs.  It holds the target, munition
profiles, impact speed, stochastic seeds and damage tree fixed while varying:

1. the cardinal approach face (all trajectories still point at the target and
   body +X is aligned with velocity); and
2. the Taylor-ring body-frame axis, comparing the archived ``-body-X`` setting
   with a deliberately counterfactual ``+body-X`` setting.

The axis flip is a sensitivity probe, not a proposed or validated physical
correction.  It is useful because the current impact contract aligns body +X
with inward velocity, so the archived -body-X axis is outward at first contact.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import damage_engine  # noqa: E402
from damage_engine.engine import ENGINE_CODE_HASH, DamageEngine  # noqa: E402
from damage_engine.errors import DamageEngineError  # noqa: E402
from damage_engine.munition_profiles import apply_munition_profile  # noqa: E402
from damage_engine.registry import PhysicsRegistry  # noqa: E402


TARGET_PATH = ROOT / "raw_data/target.rebuilt.fragment_only.v1.json"
REGISTRY_PATH = ROOT / "examples/physics.loitering_impact.json"
BASE_CASE_PATH = (
    ROOT
    / "raw_data/dataset_5000_loitering_impact_fragment_only_v1/plan/base_case.json"
)
OUTPUT_PATH = Path(__file__).with_suffix(".json")
SCRIPT_PATH = Path(__file__).resolve()

PROFILES = ("small_front", "medium_front", "large_front")
SPEED_MPS = 120.0
REPLICATES = 2
ENERGY_THRESHOLDS_J = (100.0, 1_000.0, 10_000.0)

# Project-specific KMFC-v0 component grouping from report 35.  It is used only
# to aggregate continuous reachability signals, not to fit a formal tree.
BRANCHES: dict[str, dict[str, list[str]]] = {
    "K": {
        "fuel_store": ["component-003"],
        "ammunition_store": ["component-046", "component-053", "component-054"],
    },
    "M": {
        "powertrain": ["component-001", "component-002", "component-003"],
        "drive_and_idler": ["component-004", "component-005", "component-030", "component-031"],
        "right_carrying": [
            *[f"component-{i:03d}" for i in range(6, 12)],
            *[f"component-{i:03d}" for i in range(18, 21)],
        ],
        "left_carrying": [
            *[f"component-{i:03d}" for i in range(12, 18)],
            *[f"component-{i:03d}" for i in range(21, 24)],
        ],
        "suspension": [f"component-{i:03d}" for i in range(24, 30)],
        "right_track": [f"component-{i:03d}" for i in range(32, 36)],
        "left_track": [f"component-{i:03d}" for i in range(36, 40)],
        "mobility_operator": ["component-058"],
    },
    "F": {
        "sensing": ["component-040", "component-041", "component-042"],
        "fire_control": ["component-043", "component-044"],
        "weapon": ["component-045", "component-050", "component-051", "component-052"],
        "feed_and_ammunition": ["component-046", "component-049", "component-053", "component-054"],
        "countermeasure": ["component-047", "component-048"],
    },
    "C": {
        "essential_crew": ["component-058", "component-059", "component-060"],
        "embarked_personnel": [f"component-{i:03d}" for i in range(61, 68)],
    },
}

# Exact unit quaternions (xyzw) mapping body +X to the named target-frame
# velocity direction.  Origins are outside the target and rays point inward.
SQRT_HALF = math.sqrt(0.5)
FACE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "left_to_right_pos_x": {
        "origin_axis": 0,
        "origin_value": -10.0,
        "direction": [1.0, 0.0, 0.0],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "grid": [(-2.5, 0.75), (0.0, 0.75), (1.5, 0.75), (-2.5, 1.25), (0.0, 1.25), (1.5, 1.25), (-2.5, 1.70), (0.0, 1.70), (1.5, 1.70)],
        "transverse_axes": (1, 2),
    },
    "right_to_left_neg_x": {
        "origin_axis": 0,
        "origin_value": 10.0,
        "direction": [-1.0, 0.0, 0.0],
        "orientation_xyzw": [0.0, 0.0, 1.0, 0.0],
        "grid": [(-2.5, 0.75), (0.0, 0.75), (1.5, 0.75), (-2.5, 1.25), (0.0, 1.25), (1.5, 1.25), (-2.5, 1.70), (0.0, 1.70), (1.5, 1.70)],
        "transverse_axes": (1, 2),
    },
    "rear_to_front_pos_y": {
        "origin_axis": 1,
        "origin_value": -10.0,
        "direction": [0.0, 1.0, 0.0],
        "orientation_xyzw": [0.0, 0.0, SQRT_HALF, SQRT_HALF],
        "grid": [(-1.25, 0.75), (0.0, 0.75), (1.25, 0.75), (-1.25, 1.25), (0.0, 1.25), (1.25, 1.25), (-1.25, 1.70), (0.0, 1.70), (1.25, 1.70)],
        "transverse_axes": (0, 2),
    },
    "front_to_rear_neg_y": {
        "origin_axis": 1,
        "origin_value": 10.0,
        "direction": [0.0, -1.0, 0.0],
        "orientation_xyzw": [0.0, 0.0, -SQRT_HALF, SQRT_HALF],
        "grid": [(-1.25, 0.75), (0.0, 0.75), (1.25, 0.75), (-1.25, 1.25), (0.0, 1.25), (1.25, 1.25), (-1.25, 1.70), (0.0, 1.70), (1.25, 1.70)],
        "transverse_axes": (0, 2),
    },
    "top_down_neg_z": {
        "origin_axis": 2,
        "origin_value": 10.0,
        "direction": [0.0, 0.0, -1.0],
        "orientation_xyzw": [0.0, SQRT_HALF, 0.0, SQRT_HALF],
        "grid": [(-1.25, -2.5), (0.0, -2.5), (1.25, -2.5), (-1.25, 0.0), (0.0, 0.0), (1.25, 0.0), (-1.25, 2.0), (0.0, 2.0), (1.25, 2.0)],
        "transverse_axes": (0, 1),
    },
    "bottom_up_pos_z": {
        "origin_axis": 2,
        "origin_value": -10.0,
        "direction": [0.0, 0.0, 1.0],
        "orientation_xyzw": [0.0, -SQRT_HALF, 0.0, SQRT_HALF],
        "grid": [(-1.25, -2.5), (0.0, -2.5), (1.25, -2.5), (-1.25, 0.0), (0.0, 0.0), (1.25, 0.0), (-1.25, 2.0), (0.0, 2.0), (1.25, 2.0)],
        "transverse_axes": (0, 1),
    },
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object in {path}")
    return value


def make_registry(axis_sign: int) -> PhysicsRegistry:
    data = deepcopy(load_json(REGISTRY_PATH))
    for model in data["fragment_source_models"].values():
        model["direction"]["axis_body"] = [float(axis_sign), 0.0, 0.0]
    return PhysicsRegistry(data)


def make_case(face: str, grid_index: int, profile: str) -> dict[str, Any]:
    definition = FACE_DEFINITIONS[face]
    origin = [0.0, 0.0, 1.2]
    origin[int(definition["origin_axis"])] = float(definition["origin_value"])
    a, b = definition["grid"][grid_index]
    axis_a, axis_b = definition["transverse_axes"]
    origin[axis_a] = float(a)
    origin[axis_b] = float(b)
    direction = np.asarray(definition["direction"], dtype=np.float64)

    case = deepcopy(load_json(BASE_CASE_PATH))
    case_id = f"cf-{face}-{grid_index:02d}-{profile}"
    case["case_id"] = case_id
    case["encounter"]["trajectory_origin_m"] = origin
    case["encounter"]["velocity_target_mps"] = (SPEED_MPS * direction).tolist()
    case["encounter"]["body_to_target_orientation_xyzw"] = list(
        definition["orientation_xyzw"]
    )
    case["analysis"]["aleatory_seed"] = int(
        hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:8], 16
    )
    case["analysis"]["epistemic_sample_id"] = case_id
    case["analysis"]["replicates"] = REPLICATES
    case["analysis"]["adaptive_ci_half_width"] = None
    case["analysis"]["capture_replicate_trace"] = True
    case["analysis"]["capture_fragment_audit"] = False
    apply_munition_profile(case, profile)
    return case


def summarize_result(
    result: dict[str, Any], *, condition: str, face: str, profile: str
) -> dict[str, Any]:
    records = result["replicate_trace"]["records"]
    component_ids = sorted(records[0]["fragment_metrics"])
    positive_components: set[str] = set()
    state_positive_components: set[str] = set()
    component_energy: dict[str, float] = {}
    component_hit: dict[str, float] = {}
    component_penetration: dict[str, float] = {}
    for component_id in component_ids:
        energies = [
            float(record["fragment_metrics"][component_id]["deposited_energy_j"])
            for record in records
        ]
        hits = [
            float(record["fragment_metrics"][component_id]["hits"])
            for record in records
        ]
        penetrations = [
            float(record["fragment_metrics"][component_id]["penetrations"])
            for record in records
        ]
        component_energy[component_id] = float(np.mean(energies))
        component_hit[component_id] = float(np.mean(hits))
        component_penetration[component_id] = float(np.mean(penetrations))
        if any(value > 0.0 for value in energies):
            positive_components.add(component_id)
        if any(
            float(probability) > 0.0
            for record in records
            for probability in record["component_mechanism_states"]["combined"][component_id].values()
        ):
            state_positive_components.add(component_id)

    category_positive: dict[str, bool] = {}
    smooth_scores: dict[str, dict[str, float]] = {}
    for category, branches in BRANCHES.items():
        branch_energy = {
            branch: max(component_energy[component_id] for component_id in ids)
            for branch, ids in branches.items()
        }
        category_positive[category] = any(value > 0.0 for value in branch_energy.values())
        smooth_scores[category] = {
            str(int(threshold)): float(
                np.mean(
                    [
                        value / (value + threshold) if value > 0.0 else 0.0
                        for value in branch_energy.values()
                    ]
                )
            )
            for threshold in ENERGY_THRESHOLDS_J
        }

    tree_values = {
        output: float(result["damage_tree"]["outputs"][output]["mean"])
        for output in sorted(result["damage_tree"]["outputs"])
    }
    summaries = [record["summary"] for record in records]
    encounter = result["provenance"]["input_summary"]["encounter"] if "input_summary" in result["provenance"] else None
    return {
        "case_id": result["case_id"],
        "condition": condition,
        "face": face,
        "profile": profile,
        "replicate_count": len(records),
        "collision_armor_id": result.get("provenance", {}).get("collision_armor_id"),
        "total_armor_hits_mean": float(np.mean([item["total_armor_hits"] for item in summaries])),
        "total_armor_penetrations_mean": float(np.mean([item["total_armor_penetrations"] for item in summaries])),
        "total_component_hits_mean": float(np.mean([item["total_component_hits"] for item in summaries])),
        "total_component_penetrations_mean": float(np.mean([item["total_component_penetrations"] for item in summaries])),
        "any_component_hit": any(value > 0.0 for value in component_hit.values()),
        "any_component_penetration": any(value > 0.0 for value in component_penetration.values()),
        "any_positive_deposited_energy": bool(positive_components),
        "positive_energy_component_ids": sorted(positive_components),
        "any_positive_component_state": bool(state_positive_components),
        "positive_state_component_ids": sorted(state_positive_components),
        "category_positive_energy": category_positive,
        "zero_anchored_smooth_category_scores": smooth_scores,
        "tree_outputs": tree_values,
        "tree_any_positive": any(value > 0.0 for value in tree_values.values()),
        "encounter_from_provenance": encounter,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    profile_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['condition']}|{row['face']}"] .append(row)
        profile_grouped[f"{row['condition']}|{row['profile']}"] .append(row)

    def one(group: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(group)
        categories = sorted(BRANCHES)
        outputs = sorted(group[0]["tree_outputs"]) if group else []
        return {
            "case_count": count,
            "profile_counts": dict(sorted(Counter(row["profile"] for row in group).items())),
            "case_stage_survival": {
                "fragment_armor_hit": sum(row["total_armor_hits_mean"] > 0 for row in group),
                "fragment_armor_penetration": sum(row["total_armor_penetrations_mean"] > 0 for row in group),
                "component_hit": sum(row["any_component_hit"] for row in group),
                "component_penetration": sum(row["any_component_penetration"] for row in group),
                "positive_deposited_energy": sum(row["any_positive_deposited_energy"] for row in group),
                "positive_component_state": sum(row["any_positive_component_state"] for row in group),
                "positive_existing_tree_output": sum(row["tree_any_positive"] for row in group),
            },
            "mean_counts": {
                "armor_hits": float(np.mean([row["total_armor_hits_mean"] for row in group])),
                "armor_penetrations": float(np.mean([row["total_armor_penetrations_mean"] for row in group])),
                "component_hits": float(np.mean([row["total_component_hits_mean"] for row in group])),
                "component_penetrations": float(np.mean([row["total_component_penetrations_mean"] for row in group])),
            },
            "category_positive_energy_case_count": {
                category: sum(row["category_positive_energy"][category] for row in group)
                for category in categories
            },
            "existing_tree_positive_case_count": {
                output: sum(row["tree_outputs"][output] > 0.0 for row in group)
                for output in outputs
            },
            "zero_anchored_smooth_positive_case_count": {
                category: {
                    threshold: sum(
                        row["zero_anchored_smooth_category_scores"][category][threshold] > 0.0
                        for row in group
                    )
                    for threshold in map(lambda value: str(int(value)), ENERGY_THRESHOLDS_J)
                }
                for category in categories
            },
        }

    by_condition_face = {key: one(value) for key, value in sorted(grouped.items())}
    by_condition_profile = {
        key: one(value) for key, value in sorted(profile_grouped.items())
    }
    by_condition: dict[str, Any] = {}
    for condition in sorted({row["condition"] for row in rows}):
        by_condition[condition] = one([row for row in rows if row["condition"] == condition])
    return {
        "by_condition": by_condition,
        "by_condition_and_face": by_condition_face,
        "by_condition_and_profile": by_condition_profile,
    }


def contrasts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the two requested single-factor contrasts and paired axis effects."""

    current_name = "archived_axis_minus_body_x"
    flip_name = "counterfactual_axis_plus_body_x"
    baseline_face = "left_to_right_pos_x"

    def support_rates(group: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(group)
        return {
            "case_count": count,
            "component_hit_rate": sum(row["any_component_hit"] for row in group) / count,
            "tree_any_positive_rate": sum(row["tree_any_positive"] for row in group) / count,
            "category_positive_energy_rate": {
                category: sum(row["category_positive_energy"][category] for row in group) / count
                for category in sorted(BRANCHES)
            },
        }

    current_rows = [row for row in rows if row["condition"] == current_name]
    direction_only = {
        "interpretation": (
            "The source axis, target, profiles, speed and tree are unchanged. "
            "Only the inward cardinal approach direction, aligned pose and compatible transverse grid vary."
        ),
        "archived_p0_face_proxy": support_rates(
            [row for row in current_rows if row["face"] == baseline_face]
        ),
        "other_five_faces_pooled": support_rates(
            [row for row in current_rows if row["face"] != baseline_face]
        ),
        "all_six_faces": support_rates(current_rows),
    }

    by_condition_case = {
        (row["condition"], row["case_id"]): row for row in rows
    }
    pair_ids = sorted(
        {
            case_id
            for condition, case_id in by_condition_case
            if condition == current_name
            and (flip_name, case_id) in by_condition_case
        }
    )

    def paired_effect(case_ids: list[str]) -> dict[str, Any]:
        category_transitions: dict[str, Counter[str]] = {
            category: Counter() for category in sorted(BRANCHES)
        }
        component_transitions: Counter[str] = Counter()
        tree_transitions: Counter[str] = Counter()
        for case_id in case_ids:
            before = by_condition_case[(current_name, case_id)]
            after = by_condition_case[(flip_name, case_id)]
            component_transitions[
                f"{int(before['any_component_hit'])}->{int(after['any_component_hit'])}"
            ] += 1
            tree_transitions[
                f"{int(before['tree_any_positive'])}->{int(after['tree_any_positive'])}"
            ] += 1
            for category in sorted(BRANCHES):
                category_transitions[category][
                    f"{int(before['category_positive_energy'][category])}->"
                    f"{int(after['category_positive_energy'][category])}"
                ] += 1
        return {
            "paired_case_count": len(case_ids),
            "component_hit_transitions": dict(sorted(component_transitions.items())),
            "tree_any_positive_transitions": dict(sorted(tree_transitions.items())),
            "category_positive_energy_transitions": {
                category: dict(sorted(counter.items()))
                for category, counter in category_transitions.items()
            },
        }

    baseline_pair_ids = [
        case_id
        for case_id in pair_ids
        if by_condition_case[(current_name, case_id)]["face"] == baseline_face
    ]
    axis_only = {
        "interpretation": (
            "Each pair has identical approach face, contact-ray origin, velocity, pose, profile, "
            "speed and seeds. Only Taylor direction.axis_body changes from [-1,0,0] to [+1,0,0]."
        ),
        "archived_p0_face_proxy_paired": paired_effect(baseline_pair_ids),
        "all_faces_paired": paired_effect(pair_ids),
        "scientific_guard": (
            "This diagnoses sensitivity to a coordinate/sign convention. It is not evidence that "
            "+body-X is the physically correct Taylor axis."
        ),
    }
    return {"direction_pose_only": direction_only, "source_axis_only": axis_only}


def main() -> None:
    target = load_json(TARGET_PATH)
    conditions = {
        "archived_axis_minus_body_x": make_registry(-1),
        "counterfactual_axis_plus_body_x": make_registry(+1),
    }
    engines = {
        name: DamageEngine(deepcopy(target), registry)
        for name, registry in conditions.items()
    }
    rows: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []
    attempted = 0
    for condition, engine in engines.items():
        for face, definition in FACE_DEFINITIONS.items():
            for grid_index in range(len(definition["grid"])):
                for profile in PROFILES:
                    attempted += 1
                    case = make_case(face, grid_index, profile)
                    try:
                        result = engine.run(case).to_dict()
                    except DamageEngineError as error:
                        refusals.append(
                            {
                                "condition": condition,
                                "face": face,
                                "grid_index": grid_index,
                                "profile": profile,
                                "case_id": case["case_id"],
                                "error": error.to_dict(),
                            }
                        )
                        continue
                    rows.append(
                        summarize_result(
                            result,
                            condition=condition,
                            face=face,
                            profile=profile,
                        )
                    )

    payload = {
        "schema_version": "1.0.0",
        "audit_id": "factorial-direction-source-axis-counterfactual-20261021",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "diagnostic_counterfactual_not_physical_calibration",
        "research_question": (
            "Does KMFC sparsity persist necessarily under the fixed projectile/target, "
            "or does approach support and source-axis convention materially alter reachability?"
        ),
        "controlled_factors": {
            "held_fixed": [
                "target geometry and component response",
                "three munition profiles",
                f"impact speed={SPEED_MPS} m/s",
                f"replicates={REPLICATES}",
                "damage-tree definition",
                "deterministic case-specific aleatory and epistemic seeds",
            ],
            "varied": {
                "approach_face": list(FACE_DEFINITIONS),
                "fragment_source_axis_body": {
                    "archived_axis_minus_body_x": [-1.0, 0.0, 0.0],
                    "counterfactual_axis_plus_body_x": [1.0, 0.0, 0.0],
                },
            },
            "important_semantics": (
                "Every accepted trajectory points toward the target and body +X is aligned "
                "with velocity. Therefore -body-X is outward and +body-X is inward at first contact."
            ),
        },
        "design": {
            "profiles": list(PROFILES),
            "speed_mps": SPEED_MPS,
            "replicates": REPLICATES,
            "grid_points_per_face": {
                face: len(value["grid"]) for face, value in FACE_DEFINITIONS.items()
            },
            "attempted_case_condition_combinations": attempted,
            "accepted_case_condition_combinations": len(rows),
            "refused_case_condition_combinations": len(refusals),
            "energy_thresholds_j_for_sensitivity": list(ENERGY_THRESHOLDS_J),
        },
        "software_identity": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "damage_engine_version": damage_engine.__version__,
            "engine_code_hash": ENGINE_CODE_HASH,
            "script_path": str(SCRIPT_PATH.relative_to(ROOT)).replace("\\", "/"),
            "script_sha256": file_hash(SCRIPT_PATH),
            "target_sha256": file_hash(TARGET_PATH),
            "registry_sha256": file_hash(REGISTRY_PATH),
            "base_case_sha256": file_hash(BASE_CASE_PATH),
            "counterfactual_registry_hash": conditions["counterfactual_axis_plus_body_x"].hash,
        },
        "aggregate": aggregate(rows),
        "single_factor_contrasts": contrasts(rows),
        "refusals": refusals,
        "case_results": rows,
        "interpretation_guards": {
            "axis_flip_status": (
                "Sensitivity intervention only; it does not establish that +body-X is physically correct."
            ),
            "tree_gate_fact": (
                "The zero-anchored smooth score is exactly zero whenever every underlying "
                "component energy is zero. Threshold changes alter magnitude but cannot create support."
            ),
            "sampling_boundary": (
                "The deterministic cardinal grid is an intervention design, not an operational "
                "reference distribution and not a balanced training distribution."
            ),
            "validation_boundary": (
                "All physics models and munition parameters remain unverified; results identify "
                "software-mechanism sensitivity, not real-world lethality."
            ),
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["design"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["aggregate"]["by_condition"], ensure_ascii=False, indent=2))
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
