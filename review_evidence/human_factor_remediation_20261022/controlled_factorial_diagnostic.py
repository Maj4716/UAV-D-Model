"""Controlled diagnostic of KMFC support under geometry/source/profile/R factors.

This is a read-only software-mechanism experiment.  It reuses the cardinal
contact design and KMFC-v0 component grouping from the 2026-10-21 causal
diagnostic, but adds roll pose, common-random-number pairing, actual first
contact bookkeeping, three munition profiles, and nested R=1/2/4/8 summaries.

It does not mutate the production target, registry, damage tree, sampling
specification, or any formal dataset archive.  The +body-X source axis is a
counterfactual sensitivity intervention, not a physically validated repair.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import shutil
import sys
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
OUTPUT_DIR = SCRIPT_PATH.parent
RESULT_PATH = OUTPUT_DIR / "controlled_factorial_diagnostic.json"
REPORT_PATH = OUTPUT_DIR / "controlled_factorial_diagnostic.md"
LEDGER_PATH = OUTPUT_DIR / "candidate_ledger.jsonl"
LEGACY_SCRIPT_PATH = (
    ROOT
    / "review_evidence/causal_sparsity_20261021/"
    "factorial_direction_source_counterfactual.py"
)

sys.path.insert(0, str(ROOT))

from damage_engine.engine import ENGINE_CODE_HASH, DamageEngine  # noqa: E402
from damage_engine.errors import DamageEngineError  # noqa: E402
from damage_engine.impact import (  # noqa: E402
    IMPACT_ALIGNMENT_COSINE_MINIMUM,
    IMPACT_MODEL_ID,
    validate_impact_detonation,
)
from damage_engine.munition_profiles import apply_munition_profile  # noqa: E402
from damage_engine.utils import quaternion_matrix_xyzw  # noqa: E402
import damage_engine  # noqa: E402


def _load_legacy_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "factorial_direction_source_counterfactual_20261021",
        LEGACY_SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load diagnostic dependency: {LEGACY_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LEGACY = _load_legacy_module()
TARGET_PATH: Path = LEGACY.TARGET_PATH
REGISTRY_PATH: Path = LEGACY.REGISTRY_PATH
BASE_CASE_PATH: Path = LEGACY.BASE_CASE_PATH
FACE_DEFINITIONS: dict[str, dict[str, Any]] = LEGACY.FACE_DEFINITIONS
BRANCHES: dict[str, dict[str, list[str]]] = LEGACY.BRANCHES

PROFILES = ("small_front", "medium_front", "large_front")
AXIS_CONDITIONS = {
    "archived_minus_body_x": -1,
    "counterfactual_plus_body_x": +1,
}
SELECTED_GRID_INDICES = (3, 4, 5)
ROLL_DEGREES = (0.0, 45.0, 90.0)
PREFIX_LEVELS = (1, 2, 4, 8)
MAX_REPLICATES = max(PREFIX_LEVELS)
SPEED_MPS = 120.0
TARGET_REFERENCE_POINT_M = np.asarray([0.0, 0.0, 1.2], dtype=np.float64)
EPISTEMIC_SAMPLE_ID = "hfdiag-fixed-target-realization-20261022-v1"
BOOTSTRAP_SEED = 20261022
BOOTSTRAP_DRAWS = 4000


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return value


def quaternion_multiply_xyzw(left: Iterable[float], right: Iterable[float]) -> list[float]:
    """Return left ⊗ right using the repository's xyzw convention."""

    x1, y1, z1, w1 = map(float, left)
    x2, y2, z2, w2 = map(float, right)
    value = np.asarray(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )
    value /= np.linalg.norm(value)
    # q and -q encode the same rotation.  Canonicalization makes identity and
    # uniqueness checks stable, including the w=0 half-turn cases.
    first_nonzero = next((item for item in value if abs(item) > 1.0e-15), 1.0)
    if value[3] < -1.0e-15 or (abs(value[3]) <= 1.0e-15 and first_nonzero < 0.0):
        value *= -1.0
    return [float(item) for item in value]


def rolled_orientation(face: str, roll_deg: float) -> list[float]:
    align = FACE_DEFINITIONS[face]["orientation_xyzw"]
    half = math.radians(roll_deg) / 2.0
    body_roll = [math.sin(half), 0.0, 0.0, math.cos(half)]
    return quaternion_multiply_xyzw(align, body_roll)


def candidate_key(face: str, grid_index: int, roll_deg: float, profile: str) -> str:
    return f"{face}|g{grid_index:02d}|roll{roll_deg:05.1f}|{profile}"


def paired_seed(face: str, grid_index: int, roll_deg: float, profile: str) -> int:
    # Every complete candidate gets its own stochastic stream.  Only the two
    # source-axis interventions for that exact candidate share the stream, which
    # preserves the intended paired contrast without collapsing 162 candidates
    # onto three grid-index streams.
    token = "hfdiag-axis-pair|" + candidate_key(face, grid_index, roll_deg, profile)
    return int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)


def make_case(face: str, grid_index: int, roll_deg: float, profile: str) -> dict[str, Any]:
    definition = FACE_DEFINITIONS[face]
    origin = [0.0, 0.0, 1.2]
    origin[int(definition["origin_axis"])] = float(definition["origin_value"])
    a, b = definition["grid"][grid_index]
    axis_a, axis_b = definition["transverse_axes"]
    origin[int(axis_a)] = float(a)
    origin[int(axis_b)] = float(b)
    direction = np.asarray(definition["direction"], dtype=np.float64)
    orientation = rolled_orientation(face, roll_deg)

    body_forward = quaternion_matrix_xyzw(orientation) @ np.asarray(
        [1.0, 0.0, 0.0], dtype=np.float64
    )
    alignment = float(np.dot(body_forward, direction))
    if alignment < IMPACT_ALIGNMENT_COSINE_MINIMUM:
        raise RuntimeError(
            f"constructed pose is not velocity-aligned: face={face}, roll={roll_deg}, "
            f"cosine={alignment}"
        )
    toward_dot = float(np.dot(TARGET_REFERENCE_POINT_M - np.asarray(origin), direction))
    if toward_dot <= 0.0:
        raise RuntimeError(f"constructed velocity is not target-facing: face={face}")

    case = deepcopy(load_json(BASE_CASE_PATH))
    key = candidate_key(face, grid_index, roll_deg, profile)
    case["case_id"] = "hfdiag-" + key.replace("|", "-").replace(".", "p")
    case["encounter"]["trajectory_origin_m"] = origin
    case["encounter"]["velocity_target_mps"] = (SPEED_MPS * direction).tolist()
    case["encounter"]["body_to_target_orientation_xyzw"] = orientation
    case["analysis"]["aleatory_seed"] = paired_seed(
        face, grid_index, roll_deg, profile
    )
    case["analysis"]["epistemic_sample_id"] = EPISTEMIC_SAMPLE_ID
    case["analysis"]["replicates"] = MAX_REPLICATES
    case["analysis"]["adaptive_ci_half_width"] = None
    case["analysis"].pop("adaptive_physical_ci", None)
    case["analysis"]["capture_replicate_trace"] = True
    case["analysis"]["capture_fragment_audit"] = False
    case.setdefault("metadata", {})["diagnostic_design"] = {
        "audit_id": "controlled-factorial-kmfc-support-20261022",
        "candidate_key": key,
        "face": face,
        "grid_index": grid_index,
        "roll_deg": roll_deg,
        "profile": profile,
        "training_eligible": False,
    }
    apply_munition_profile(case, profile)
    return case


def _category_components(category: str) -> list[str]:
    return sorted(
        {
            component_id
            for identifiers in BRANCHES[category].values()
            for component_id in identifiers
        }
    )


CATEGORY_COMPONENTS = {
    category: _category_components(category) for category in sorted(BRANCHES)
}


def summarize_prefix(records: list[dict[str, Any]]) -> dict[str, Any]:
    category_replicate_positive: dict[str, list[bool]] = {
        category: [] for category in CATEGORY_COMPONENTS
    }
    category_replicate_max_energy: dict[str, list[float]] = {
        category: [] for category in CATEGORY_COMPONENTS
    }
    positive_component_ids: dict[str, set[str]] = {
        category: set() for category in CATEGORY_COMPONENTS
    }
    for record in records:
        metrics = record["fragment_metrics"]
        for category, component_ids in CATEGORY_COMPONENTS.items():
            component_energies = {
                component_id: float(metrics[component_id]["deposited_energy_j"])
                for component_id in component_ids
            }
            positive_component_ids[category].update(
                component_id
                for component_id, energy in component_energies.items()
                if energy > 0.0
            )
            maximum = max(component_energies.values(), default=0.0)
            category_replicate_max_energy[category].append(maximum)
            category_replicate_positive[category].append(maximum > 0.0)

    output_names = sorted(records[0]["system_outputs"])
    tree_output_mean = {
        name: float(np.mean([float(record["system_outputs"][name]) for record in records]))
        for name in output_names
    }
    summary_names = (
        "total_armor_hits",
        "total_armor_penetrations",
        "total_component_hits",
        "total_component_penetrations",
    )
    summary_mean = {
        name: float(np.mean([float(record["summary"][name]) for record in records]))
        for name in summary_names
    }
    return {
        "replicate_count": len(records),
        "category_ever_positive": {
            category: any(values)
            for category, values in category_replicate_positive.items()
        },
        "category_replicate_positive_rate": {
            category: float(np.mean(values))
            for category, values in category_replicate_positive.items()
        },
        "category_mean_max_component_energy_j": {
            category: float(np.mean(category_replicate_max_energy[category]))
            for category in CATEGORY_COMPONENTS
        },
        "category_positive_component_ids": {
            category: sorted(values) for category, values in positive_component_ids.items()
        },
        "tree_output_mean": tree_output_mean,
        "tree_output_positive": {
            name: value > 0.0 for name, value in tree_output_mean.items()
        },
        "tree_any_positive": any(value > 0.0 for value in tree_output_mean.values()),
        "summary_mean": summary_mean,
        "any_component_hit": summary_mean["total_component_hits"] > 0.0,
        "any_component_penetration": summary_mean["total_component_penetrations"] > 0.0,
    }


def summarize_result(
    result: dict[str, Any],
    *,
    axis_condition: str,
    face: str,
    grid_index: int,
    roll_deg: float,
    profile: str,
    prepared_case: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    records = result["replicate_trace"]["records"]
    if len(records) != MAX_REPLICATES:
        raise RuntimeError(
            f"unexpected replicate count {len(records)} for {result['case_id']}"
        )
    encounter = prepared_case["encounter"]
    direction = np.asarray(encounter["velocity_target_mps"], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    orientation = encounter["body_to_target_orientation_xyzw"]
    body_forward = quaternion_matrix_xyzw(orientation) @ np.asarray(
        [1.0, 0.0, 0.0], dtype=np.float64
    )
    alignment = float(np.dot(body_forward, direction))
    toward_dot = float(
        np.dot(
            TARGET_REFERENCE_POINT_M - np.asarray(encounter["trajectory_origin_m"]),
            direction,
        )
    )
    if alignment < IMPACT_ALIGNMENT_COSINE_MINIMUM or toward_dot <= 0.0:
        raise RuntimeError("accepted result violated target-facing pose contract")
    validate_impact_detonation(prepared_case, ENGINES[axis_condition].compiled_target)

    prefixes = {
        str(level): summarize_prefix(records[:level]) for level in PREFIX_LEVELS
    }
    contact_normal = [
        0.0 if abs(float(value)) < 1.0e-12 else float(value)
        for value in encounter["collision_normal_target_xyz"]
    ]
    surface_key = (
        f"{encounter['collision_armor_id']}|normal="
        + ",".join(f"{value:+.6f}" for value in contact_normal)
    )
    return {
        "candidate_key": candidate_key(face, grid_index, roll_deg, profile),
        "case_id": result["case_id"],
        "axis_condition": axis_condition,
        "face": face,
        "grid_index": grid_index,
        "roll_deg": roll_deg,
        "profile": profile,
        "aleatory_seed": int(prepared_case["analysis"]["aleatory_seed"]),
        "epistemic_sample_id": prepared_case["analysis"]["epistemic_sample_id"],
        "trajectory_origin_m": [float(value) for value in encounter["trajectory_origin_m"]],
        "velocity_direction_target_xyz": [float(value) for value in direction],
        "orientation_xyzw": [float(value) for value in orientation],
        "body_forward_alignment_cosine": alignment,
        "target_facing_dot_m": toward_dot,
        "impact_model_id": encounter["impact_model_id"],
        "collision_armor_id": encounter["collision_armor_id"],
        "collision_normal_target_xyz": contact_normal,
        "contact_surface_key": surface_key,
        "detonation_position_m": [float(value) for value in encounter["detonation_position_m"]],
        "collision_distance_m": float(encounter["collision_distance_m"]),
        "elapsed_seconds": elapsed_seconds,
        "prefix": prefixes,
    }


def quantiles(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "q0": None,
            "q25": None,
            "q50": None,
            "q75": None,
            "q90": None,
            "q95": None,
            "q100": None,
        }
    result: dict[str, float | int | None] = {
        "count": int(array.size),
        "mean": float(np.mean(array)),
    }
    for name, q in (
        ("q0", 0.0),
        ("q25", 0.25),
        ("q50", 0.50),
        ("q75", 0.75),
        ("q90", 0.90),
        ("q95", 0.95),
        ("q100", 1.0),
    ):
        result[name] = float(np.quantile(array, q))
    return result


def aggregate_group(rows: list[dict[str, Any]], level: int) -> dict[str, Any]:
    key = str(level)
    if not rows:
        return {"accepted_case_count": 0, "replicate_count": 0}
    output_names = sorted(rows[0]["prefix"][key]["tree_output_mean"])
    category_support = {
        category: sum(
            bool(row["prefix"][key]["category_ever_positive"][category])
            for row in rows
        )
        for category in CATEGORY_COMPONENTS
    }
    tree_support = {
        output: sum(
            bool(row["prefix"][key]["tree_output_positive"][output]) for row in rows
        )
        for output in output_names
    }
    return {
        "accepted_case_count": len(rows),
        "replicate_count": len(rows) * level,
        "category_ever_positive_count": category_support,
        "category_ever_positive_rate": {
            category: count / len(rows) for category, count in category_support.items()
        },
        "category_mean_replicate_positive_rate": {
            category: float(
                np.mean(
                    [
                        row["prefix"][key]["category_replicate_positive_rate"][category]
                        for row in rows
                    ]
                )
            )
            for category in CATEGORY_COMPONENTS
        },
        "category_mean_max_component_energy_j": {
            category: quantiles(
                row["prefix"][key]["category_mean_max_component_energy_j"][category]
                for row in rows
            )
            for category in CATEGORY_COMPONENTS
        },
        "tree_any_positive_count": sum(
            bool(row["prefix"][key]["tree_any_positive"]) for row in rows
        ),
        "tree_any_positive_rate": sum(
            bool(row["prefix"][key]["tree_any_positive"]) for row in rows
        )
        / len(rows),
        "tree_output_positive_count": tree_support,
        "tree_output_positive_rate": {
            output: count / len(rows) for output, count in tree_support.items()
        },
        "tree_output_distribution": {
            output: {
                **quantiles(
                    row["prefix"][key]["tree_output_mean"][output] for row in rows
                ),
                "positive_only": quantiles(
                    row["prefix"][key]["tree_output_mean"][output]
                    for row in rows
                    if row["prefix"][key]["tree_output_mean"][output] > 0.0
                ),
            }
            for output in output_names
        },
        "mean_counts": {
            summary_name: float(
                np.mean(
                    [
                        row["prefix"][key]["summary_mean"][summary_name]
                        for row in rows
                    ]
                )
            )
            for summary_name in rows[0]["prefix"][key]["summary_mean"]
        },
    }


def grouped_aggregate(
    rows: list[dict[str, Any]], fields: tuple[str, ...], level: int
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        name = "|".join(str(row[field]) for field in fields)
        groups[name].append(row)
    return {
        name: aggregate_group(group, level) for name, group in sorted(groups.items())
    }


def bootstrap_mean_ci(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, array.size, size=(BOOTSTRAP_DRAWS, array.size))
    means = np.mean(array[indices], axis=1)
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "method": "fixed_diagnostic_grid_paired_stability_bootstrap",
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "population_inference": False,
    }


def paired_axis_effects(rows: list[dict[str, Any]]) -> dict[str, Any]:
    minus = "archived_minus_body_x"
    plus = "counterfactual_plus_body_x"
    indexed = {(row["axis_condition"], row["candidate_key"]): row for row in rows}
    keys = sorted(
        key
        for condition, key in indexed
        if condition == minus and (plus, key) in indexed
    )
    output_names = sorted(rows[0]["prefix"][str(MAX_REPLICATES)]["tree_output_mean"])
    levels: dict[str, Any] = {}
    for level in PREFIX_LEVELS:
        prefix = str(level)
        category: dict[str, Any] = {}
        for name in CATEGORY_COMPONENTS:
            transitions: Counter[str] = Counter()
            binary_differences: list[float] = []
            replicate_rate_differences: list[float] = []
            energy_differences: list[float] = []
            for key in keys:
                before = indexed[(minus, key)]["prefix"][prefix]
                after = indexed[(plus, key)]["prefix"][prefix]
                b = bool(before["category_ever_positive"][name])
                a = bool(after["category_ever_positive"][name])
                transitions[f"{int(b)}->{int(a)}"] += 1
                binary_differences.append(float(a) - float(b))
                replicate_rate_differences.append(
                    float(after["category_replicate_positive_rate"][name])
                    - float(before["category_replicate_positive_rate"][name])
                )
                energy_differences.append(
                    float(after["category_mean_max_component_energy_j"][name])
                    - float(before["category_mean_max_component_energy_j"][name])
                )
            category[name] = {
                "ever_positive_transitions": dict(sorted(transitions.items())),
                "ever_positive_rate_difference_plus_minus_minus": bootstrap_mean_ci(
                    binary_differences
                ),
                "replicate_positive_rate_difference_plus_minus_minus": bootstrap_mean_ci(
                    replicate_rate_differences
                ),
                "mean_max_component_energy_difference_j": bootstrap_mean_ci(
                    energy_differences
                ),
            }

        tree_outputs: dict[str, Any] = {}
        for output in output_names:
            transitions = Counter()
            binary_differences = []
            mean_differences = []
            for key in keys:
                before = indexed[(minus, key)]["prefix"][prefix]
                after = indexed[(plus, key)]["prefix"][prefix]
                b = bool(before["tree_output_positive"][output])
                a = bool(after["tree_output_positive"][output])
                transitions[f"{int(b)}->{int(a)}"] += 1
                binary_differences.append(float(a) - float(b))
                mean_differences.append(
                    float(after["tree_output_mean"][output])
                    - float(before["tree_output_mean"][output])
                )
            tree_outputs[output] = {
                "positive_transitions": dict(sorted(transitions.items())),
                "positive_rate_difference_plus_minus_minus": bootstrap_mean_ci(
                    binary_differences
                ),
                "mean_probability_difference_plus_minus_minus": bootstrap_mean_ci(
                    mean_differences
                ),
            }
        levels[prefix] = {
            "paired_candidate_count": len(keys),
            "category": category,
            "old_tree_outputs": tree_outputs,
        }
    return {
        "pairing": (
            "Same candidate key, target realization, aleatory seed, face, point, roll, "
            "profile, speed and nested replicate prefix; source axis sign only."
        ),
        "matched_candidate_count": len(keys),
        "levels": levels,
        "bootstrap_scope": (
            "Intervals are fixed-diagnostic-grid paired stability intervals. "
            "They are not confidence intervals for an operational population."
        ),
    }


def nested_r_discovery(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for axis in AXIS_CONDITIONS:
        subset = [row for row in rows if row["axis_condition"] == axis]
        transitions: dict[str, Any] = {}
        comparisons = list(zip(PREFIX_LEVELS, PREFIX_LEVELS[1:])) + [
            (PREFIX_LEVELS[0], PREFIX_LEVELS[-1])
        ]
        for previous, current in comparisons:
            before_key, after_key = str(previous), str(current)
            categories = {}
            for category in CATEGORY_COMPONENTS:
                counts = Counter()
                for row in subset:
                    before = bool(
                        row["prefix"][before_key]["category_ever_positive"][category]
                    )
                    after = bool(
                        row["prefix"][after_key]["category_ever_positive"][category]
                    )
                    counts[f"{int(before)}->{int(after)}"] += 1
                categories[category] = dict(sorted(counts.items()))
            tree_any = Counter()
            for row in subset:
                before = bool(row["prefix"][before_key]["tree_any_positive"])
                after = bool(row["prefix"][after_key]["tree_any_positive"])
                tree_any[f"{int(before)}->{int(after)}"] += 1
            transitions[f"R{previous}_to_R{current}"] = {
                "category": categories,
                "tree_any_positive": dict(sorted(tree_any.items())),
            }
        result[axis] = transitions
    return result


def axis_by_roll_interaction(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe axis risk differences at each roll and their design-grid spread."""

    minus = "archived_minus_body_x"
    plus = "counterfactual_plus_body_x"
    indexed = {(row["axis_condition"], row["candidate_key"]): row for row in rows}
    by_roll: dict[str, Any] = {}
    for roll_deg in ROLL_DEGREES:
        keys = sorted(
            row["candidate_key"]
            for row in rows
            if row["axis_condition"] == minus
            and float(row["roll_deg"]) == roll_deg
            and (plus, row["candidate_key"]) in indexed
        )
        category = {}
        for name in CATEGORY_COMPONENTS:
            differences = []
            replicate_rate_differences = []
            for key in keys:
                before = indexed[(minus, key)]["prefix"][str(MAX_REPLICATES)]
                after = indexed[(plus, key)]["prefix"][str(MAX_REPLICATES)]
                differences.append(
                    float(after["category_ever_positive"][name])
                    - float(before["category_ever_positive"][name])
                )
                replicate_rate_differences.append(
                    float(after["category_replicate_positive_rate"][name])
                    - float(before["category_replicate_positive_rate"][name])
                )
            category[name] = {
                "ever_positive_rate_difference_plus_minus_minus": float(
                    np.mean(differences)
                ),
                "replicate_positive_rate_difference_plus_minus_minus": float(
                    np.mean(replicate_rate_differences)
                ),
            }
        tree_differences = []
        for key in keys:
            before = indexed[(minus, key)]["prefix"][str(MAX_REPLICATES)]
            after = indexed[(plus, key)]["prefix"][str(MAX_REPLICATES)]
            tree_differences.append(
                float(after["tree_any_positive"])
                - float(before["tree_any_positive"])
            )
        by_roll[str(roll_deg)] = {
            "paired_candidate_count": len(keys),
            "category": category,
            "tree_any_positive_rate_difference_plus_minus_minus": float(
                np.mean(tree_differences)
            ),
        }

    interaction_spread = {
        "category_ever_positive_axis_effect_range_across_roll": {
            category: {
                "minimum": min(
                    item["category"][category][
                        "ever_positive_rate_difference_plus_minus_minus"
                    ]
                    for item in by_roll.values()
                ),
                "maximum": max(
                    item["category"][category][
                        "ever_positive_rate_difference_plus_minus_minus"
                    ]
                    for item in by_roll.values()
                ),
                "spread": max(
                    item["category"][category][
                        "ever_positive_rate_difference_plus_minus_minus"
                    ]
                    for item in by_roll.values()
                )
                - min(
                    item["category"][category][
                        "ever_positive_rate_difference_plus_minus_minus"
                    ]
                    for item in by_roll.values()
                ),
            }
            for category in CATEGORY_COMPONENTS
        },
        "tree_any_positive_axis_effect_range_across_roll": {
            "minimum": min(
                item["tree_any_positive_rate_difference_plus_minus_minus"]
                for item in by_roll.values()
            ),
            "maximum": max(
                item["tree_any_positive_rate_difference_plus_minus_minus"]
                for item in by_roll.values()
            ),
            "spread": max(
                item["tree_any_positive_rate_difference_plus_minus_minus"]
                for item in by_roll.values()
            )
            - min(
                item["tree_any_positive_rate_difference_plus_minus_minus"]
                for item in by_roll.values()
            ),
        },
        "interpretation": (
            "The spread is a descriptive source-axis-by-roll interaction on the fixed grid. "
            "It is not an operational-population interaction coefficient."
        ),
    }
    return {"by_roll": by_roll, "interaction_spread": interaction_spread}


def roll_sensitivity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row["axis_condition"],
                row["face"],
                int(row["grid_index"]),
                row["profile"],
            )
        ].append(row)
    categories = {name: 0 for name in CATEGORY_COMPONENTS}
    tree_any = 0
    output_names = sorted(rows[0]["prefix"][str(MAX_REPLICATES)]["tree_output_mean"])
    tree_outputs = {name: 0 for name in output_names}
    complete = 0
    for group in groups.values():
        if sorted(float(row["roll_deg"]) for row in group) != list(ROLL_DEGREES):
            continue
        complete += 1
        prefix_rows = [row["prefix"][str(MAX_REPLICATES)] for row in group]
        for category in CATEGORY_COMPONENTS:
            if len(
                {
                    bool(item["category_ever_positive"][category])
                    for item in prefix_rows
                }
            ) > 1:
                categories[category] += 1
        if len({bool(item["tree_any_positive"]) for item in prefix_rows}) > 1:
            tree_any += 1
        for output in output_names:
            if len(
                {bool(item["tree_output_positive"][output]) for item in prefix_rows}
            ) > 1:
                tree_outputs[output] += 1
    return {
        "complete_roll_triplet_count": complete,
        "category_triplets_with_binary_support_disagreement": categories,
        "tree_any_positive_triplets_with_disagreement": tree_any,
        "old_tree_output_triplets_with_binary_support_disagreement": tree_outputs,
        "interpretation": (
            "Roll is about the velocity/body +X axis.  Any observed effect is sensitivity "
            "of the finite fragment realization and target geometry, not a change in aim."
        ),
    }


def factor_spreads(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for axis in AXIS_CONDITIONS:
        subset = [row for row in rows if row["axis_condition"] == axis]
        axis_result: dict[str, Any] = {}
        for factor in ("face", "roll_deg", "profile", "contact_surface_key"):
            grouped = grouped_aggregate(subset, (factor,), MAX_REPLICATES)
            axis_result[factor] = {
                "level_count": len(grouped),
                "category_rate_range": {
                    category: {
                        "minimum": min(
                            summary["category_ever_positive_rate"][category]
                            for summary in grouped.values()
                        ),
                        "maximum": max(
                            summary["category_ever_positive_rate"][category]
                            for summary in grouped.values()
                        ),
                        "spread": max(
                            summary["category_ever_positive_rate"][category]
                            for summary in grouped.values()
                        )
                        - min(
                            summary["category_ever_positive_rate"][category]
                            for summary in grouped.values()
                        ),
                    }
                    for category in CATEGORY_COMPONENTS
                },
                "tree_any_positive_rate_range": {
                    "minimum": min(
                        summary["tree_any_positive_rate"] for summary in grouped.values()
                    ),
                    "maximum": max(
                        summary["tree_any_positive_rate"] for summary in grouped.values()
                    ),
                    "spread": max(
                        summary["tree_any_positive_rate"] for summary in grouped.values()
                    )
                    - min(
                        summary["tree_any_positive_rate"] for summary in grouped.values()
                    ),
                },
            }
        result[axis] = axis_result
    return result


def _fmt_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _fmt_num(value: float | None) -> str:
    if value is None:
        return "NA"
    if value == 0.0:
        return "0"
    if abs(value) < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def build_markdown(payload: dict[str, Any]) -> str:
    accepted = payload["design"]["accepted_case_condition_count"]
    attempted = payload["design"]["attempted_case_condition_count"]
    refused = payload["design"]["refused_case_condition_count"]
    axis_r8 = payload["aggregates"]["by_axis_and_r"][str(MAX_REPLICATES)]
    paired = payload["paired_axis_effects"]["levels"][str(MAX_REPLICATES)]
    lines = [
        "# KMFC 上游支持与旧毁伤树输出的受控多因子诊断",
        "",
        "> 日期：2026-10-23（Asia/Shanghai；UTC 开始时间为 2026-10-22）  ",
        "> 性质：诊断性软件机制试验；不是物理标定、武器效能评估或可用于训练的正式数据。  ",
        f"> 引擎：`{payload['software_identity']['damage_engine_version']}` / "
        f"`{payload['software_identity']['engine_code_hash']}`。",
        "",
        "## 1. 问题、假设与设计",
        "",
        "试验检验如下机制假设：当弹型和目标不变时，KMFC 稀疏性是否主要由入射面/接触面、绕速度轴滚转姿态、Taylor 破片源轴符号、弹型和 Monte Carlo 重复数 `R` 所控制。",
        "",
        "- 固定项：目标、旧毁伤树、物理参数表、120 m/s 速度、相同目标参数实现和公共随机数。",
        "- 变化项：6 个正交来向 × 3 个横向接触点 × 3 个滚转角 × 3 个弹型 × 2 个源轴候选。",
        "- `R` 设计：每个有效工况只运行一次 `Rmax=8`，从同一轨迹提取 `R=1/2/4/8` 嵌套前缀，不把不同 R 当独立样本。",
        "- 姿态合同：机体 `+X` 与速度严格对齐，滚转只绕该轴旋转；引擎重新求解并验证首次装甲接触。",
        "- 配对：162 个完整候选（来向、接触点、roll、弹型）各自使用唯一 aleatory seed；仅同一候选的两个源轴干预共享 seed，以形成严格轴对。",
        "- 目标不确定性：全试验只固定一个 `epistemic_sample_id`，因此不能估计目标参数的 epistemic 方差或其与各因素的交互。",
        "",
        "## 2. 接触合同和分母",
        "",
        f"- 候选条件数：{attempted}；通过且完成仿真：{accepted}；拒绝：{refused}。",
        f"- 唯一飞行/弹型候选数（不计源轴复制）：{payload['design']['unique_candidate_count']}。",
        "- 只有 `status=accepted` 且符合 `continuous-ray-first-armor-v1` 的结果进入标签分母。所有拒绝都保留在 `candidate_ledger.jsonl`，但不生成标签。",
        f"- 已验收方向 {payload['contract_checks']['unique_velocity_direction_count']} 个、姿态 {payload['contract_checks']['unique_orientation_count']} 个；最小机体/速度对齐余弦为 {payload['contract_checks']['minimum_alignment_cosine']:.12f}。",
        f"- Aleatory seed 唯一值 {payload['contract_checks']['unique_aleatory_seed_count']} 个；每个 seed 必须恰好对应同一 `candidate_key` 的两个源轴条件。全试验 epistemic 实现数为 {payload['contract_checks']['unique_epistemic_sample_id_count']}。",
        "",
        "## 3. R=8 总体结果",
        "",
        "| 源轴 | 有效工况 | K 上游支持 | M 上游支持 | F 上游支持 | C 上游支持 | 旧树任一输出非零 | 平均部件命中 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for axis in AXIS_CONDITIONS:
        item = axis_r8[axis]
        lines.append(
            f"| `{axis}` | {item['accepted_case_count']} | "
            + " | ".join(
                _fmt_pct(item["category_ever_positive_rate"][category])
                for category in ("K", "M", "F", "C")
            )
            + f" | {_fmt_pct(item['tree_any_positive_rate'])} | "
            + f"{item['mean_counts']['total_component_hits']:.2f} |"
        )

    lines.extend(
        [
            "",
            "### 3.1 严格配对的源轴差异",
            "",
            "下表的风险差为 `+body-X 支持率 - -body-X 支持率`。区间是“固定诊断网格配对稳定性区间”，只表示本规则网格对设计单元重采样的稳定性；它不是目标任务或作战总体的置信区间。",
            "",
            "| 类别 | `0→1` | `1→0` | 支持率差 | 95% 设计单元区间 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for category in ("K", "M", "F", "C"):
        item = paired["category"][category]
        transitions = item["ever_positive_transitions"]
        effect = item["ever_positive_rate_difference_plus_minus_minus"]
        lines.append(
            f"| {category} | {transitions.get('0->1', 0)} | {transitions.get('1->0', 0)} | "
            f"{effect['mean']:+.3f} | [{effect['ci95_low']:+.3f}, {effect['ci95_high']:+.3f}] |"
        )

    lines.extend(
        [
            "",
            "### 3.2 嵌套 R 前缀",
            "",
            "| 源轴 | R | K | M | F | C | 旧树任一非零 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for axis in AXIS_CONDITIONS:
        for level in PREFIX_LEVELS:
            item = payload["aggregates"]["by_axis_and_r"][str(level)][axis]
            lines.append(
                f"| `{axis}` | {level} | "
                + " | ".join(
                    _fmt_pct(item["category_ever_positive_rate"][category])
                    for category in ("K", "M", "F", "C")
                )
                + f" | {_fmt_pct(item['tree_any_positive_rate'])} |"
            )

    lines.extend(
        [
            "",
            "### 3.3 R 前缀的支持发现与标签翻转",
            "",
            "单元格为 `0→1 / 1→0`。对于非负的“前缀内至少一次非零”定义，理论上只能由 0 变 1；任何 `1→0` 都表明实现或账本错误。",
            "",
            "| 源轴 | R 比较 | K | M | F | C | 旧树任一非零 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for axis in AXIS_CONDITIONS:
        for comparison, item in payload["nested_r_discovery"][axis].items():
            cells = []
            for category in ("K", "M", "F", "C"):
                counts = item["category"][category]
                cells.append(f"{counts.get('0->1', 0)} / {counts.get('1->0', 0)}")
            tree_counts = item["tree_any_positive"]
            lines.append(
                f"| `{axis}` | `{comparison}` | "
                + " | ".join(cells)
                + f" | {tree_counts.get('0->1', 0)} / {tree_counts.get('1->0', 0)} |"
            )

    lines.extend(
        [
            "",
            "## 4. 因素分层（R=8）",
            "",
            "### 4.1 弹型",
            "",
            "| 源轴 | 弹型 | n | K | M | F | C | 旧树任一非零 | 平均部件命中 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, item in payload["aggregates"]["by_axis_profile_r8"].items():
        axis, profile = key.split("|", 1)
        lines.append(
            f"| `{axis}` | `{profile}` | {item['accepted_case_count']} | "
            + " | ".join(
                _fmt_pct(item["category_ever_positive_rate"][category])
                for category in ("K", "M", "F", "C")
            )
            + f" | {_fmt_pct(item['tree_any_positive_rate'])} | "
            + f"{item['mean_counts']['total_component_hits']:.2f} |"
        )

    lines.extend(
        [
            "",
            "### 4.2 入射面",
            "",
            "| 源轴 | 入射面 | n | K | M | F | C | 旧树任一非零 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, item in payload["aggregates"]["by_axis_face_r8"].items():
        axis, face = key.split("|", 1)
        lines.append(
            f"| `{axis}` | `{face}` | {item['accepted_case_count']} | "
            + " | ".join(
                _fmt_pct(item["category_ever_positive_rate"][category])
                for category in ("K", "M", "F", "C")
            )
            + f" | {_fmt_pct(item['tree_any_positive_rate'])} |"
        )

    lines.extend(
        [
            "",
            "### 4.3 绕速度轴滚转",
            "",
            "| 源轴 | 滚转 | n | K | M | F | C | 旧树任一非零 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, item in payload["aggregates"]["by_axis_roll_r8"].items():
        axis, roll = key.split("|", 1)
        lines.append(
            f"| `{axis}` | {float(roll):.0f}° | {item['accepted_case_count']} | "
            + " | ".join(
                _fmt_pct(item["category_ever_positive_rate"][category])
                for category in ("K", "M", "F", "C")
            )
            + f" | {_fmt_pct(item['tree_any_positive_rate'])} |"
        )

    roll = payload["roll_sensitivity_r8"]
    lines.extend(
        [
            "",
            "**Roll 边际主效应（汇合两个源轴）**",
            "",
            "| 滚转 | n | K | M | F | C | 旧树任一非零 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for roll_key, item in payload["aggregates"]["by_roll_marginal_r8"].items():
        lines.append(
            f"| {float(roll_key):.0f}° | {item['accepted_case_count']} | "
            + " | ".join(
                _fmt_pct(item["category_ever_positive_rate"][category])
                for category in ("K", "M", "F", "C")
            )
            + f" | {_fmt_pct(item['tree_any_positive_rate'])} |"
        )

    interaction = payload["axis_by_roll_interaction_r8"]
    lines.extend(
        [
            "",
            "**`source-axis × roll` 描述性交互**",
            "",
            "单元格为各滚转角下 `+body-X - -body-X` 的支持率差。",
            "",
            "| 滚转 | 配对 n | K | M | F | C | 旧树任一非零 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for roll_key, item in interaction["by_roll"].items():
        lines.append(
            f"| {float(roll_key):.0f}° | {item['paired_candidate_count']} | "
            + " | ".join(
                f"{item['category'][category]['ever_positive_rate_difference_plus_minus_minus']:+.3f}"
                for category in ("K", "M", "F", "C")
            )
            + f" | {item['tree_any_positive_rate_difference_plus_minus_minus']:+.3f} |"
        )
    interaction_spreads = interaction["interaction_spread"][
        "category_ever_positive_axis_effect_range_across_roll"
    ]
    lines.extend(
        [
            "",
            f"共有 {roll['complete_roll_triplet_count']} 组完整滚转三联组。出现二值支持分歧的组数为："
            + "、".join(
                f"{category}={roll['category_triplets_with_binary_support_disagreement'][category]}"
                for category in ("K", "M", "F", "C")
            )
            + f"，旧树任一输出={roll['tree_any_positive_triplets_with_disagreement']}。"
            + " 源轴效应在不同 roll 之间的支持率差跨度为 "
            + "、".join(
                f"{category}={interaction_spreads[category]['spread']:.3f}"
                for category in ("K", "M", "F", "C")
            )
            + f"，旧树任一输出={interaction['interaction_spread']['tree_any_positive_axis_effect_range_across_roll']['spread']:.3f}。",
            "",
            "### 4.4 实际首次接触面",
            "",
            "| 源轴 | 装甲/法向 | n | K | M | F | C | 旧树任一非零 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, item in payload["aggregates"]["by_axis_contact_surface_r8"].items():
        axis, surface = key.split("|", 1)
        lines.append(
            f"| `{axis}` | `{surface}` | {item['accepted_case_count']} | "
            + " | ".join(
                _fmt_pct(item["category_ever_positive_rate"][category])
                for category in ("K", "M", "F", "C")
            )
            + f" | {_fmt_pct(item['tree_any_positive_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## 5. 旧毁伤树输出分布（R=8）",
            "",
            "| 源轴 | 旧树输出 | 非零率 | 全样本均值 | 中位数 | P95 | 最大值 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for axis in AXIS_CONDITIONS:
        item = axis_r8[axis]
        for output, distribution in item["tree_output_distribution"].items():
            lines.append(
                f"| `{axis}` | `{output}` | "
                f"{_fmt_pct(item['tree_output_positive_rate'][output])} | "
                f"{_fmt_num(distribution['mean'])} | {_fmt_num(distribution['q50'])} | "
                f"{_fmt_num(distribution['q95'])} | {_fmt_num(distribution['q100'])} |"
            )

    minus = axis_r8["archived_minus_body_x"]
    plus = axis_r8["counterfactual_plus_body_x"]
    r1_minus = payload["aggregates"]["by_axis_and_r"]["1"]["archived_minus_body_x"]
    r8_minus = minus
    lines.extend(
        [
            "",
            "## 6. 证据结论",
            "",
            "1. **源轴是一阶敏感因素。** 严格配对下，从当前 `-body-X` 切换到诊断用 `+body-X`，K/M/F/C 支持率分别由 "
            + "/".join(_fmt_pct(minus["category_ever_positive_rate"][c]) for c in ("K", "M", "F", "C"))
            + " 变为 "
            + "/".join(_fmt_pct(plus["category_ever_positive_rate"][c]) for c in ("K", "M", "F", "C"))
            + "。这反证了“固定弹型与目标必然造成同等程度稀疏”，但不能证明 `+body-X` 物理正确。",
            "2. **入射面和实际接触面不可被单一方向代替。** 分面结果的差异说明当前只有 `+X` 来向的正式设计无法识别其他接触面的支持结构。",
            "3. **弹型同时影响二值可达性与剂量，但不能代替几何校正。** 表 4.1 给出了在相同几何/随机数下的弹型差异；若源轴使部件不可达，增加破片数不能系统性修复严格零。",
            "4. **R 改变“至少一次观测到支持”的概率。** 在当前源轴下，R=1 到 R=8 的 K/M/F/C 支持率从 "
            + "/".join(_fmt_pct(r1_minus["category_ever_positive_rate"][c]) for c in ("K", "M", "F", "C"))
            + " 变为 "
            + "/".join(_fmt_pct(r8_minus["category_ever_positive_rate"][c]) for c in ("K", "M", "F", "C"))
            + "。因此不同 R 下的 `any-positive` 不可直接比较；后续必须保存 `successes/trials` 或完整重复轨迹。",
            "5. **滚转敏感性是有限破片实现与非对称目标的联合效应。** 它不改变速度朝向，却可改变有限破片束绕主轴的方位；三联组分歧计数已在 4.3 中报告。",
            "6. **旧树输出仍然是上游仿真与旧门控的合成结果。** 本试验可定位支持在哪一阶段流失，不能独立标定新 KMFC 等级阈值。",
            "",
            "## 7. 与深度学习衔接的执行建议",
            "",
            "1. 先完成战斗部坐标/源轴的实物定义核对，再决定生产参数；本反事实不能直接替换正式注册表。",
            "2. 将“来向×接触面×滚转×弹型”定义为训练获取的机制分层，但必须与代表作战分布的 `p0 Reference` 分开。该平衡网格可以用于机制覆盖/表示学习，不得用于估计真实频率或总体校准。",
            "3. 学习目标保存每类别的连续剂量、`successes/trials`、等级互斥计数和树版本哈希。KMFC 四类可重叠，应用独立 sigmoid/有序累积目标，不用 softmax。",
            "4. 拆分必须按外层候选/轨迹组进行；同一工况的 R 个重复不得跨 Train/Validation/Test，也不得当成 R 个独立样本。",
            "5. 对于结构性零，先修复上游可达性和任务定义；对于可达但希有的类别，再使用分层获取、重要性权重和软标签。类别权重或 focal loss 不能修复上游零支持。",
            "",
            "## 8. 限制和不可外推项",
            "",
            "- 全部数值来自当前未经外部物理标定的软件模型，不能证明实际毁伤概率。",
            "- 网格为因果定位而平衡，不是真实来袭分布；统计比率只适用于本固定设计。",
            "- 速度固定为 120 m/s，未识别速度与其他因素的交互。",
            "- 只考察了 `-body-X` 与 `+body-X` 两个源轴符号，没有检验偏心起爆、多点起爆、破片列表或高保真破片场。",
            "- R=8 仅用于中等规模的支持诊断，不是已通过预注册收敛阈值的正式重复数。",
            "- 全试验仅使用一个固定的 epistemic 目标参数实现；轴对内比较可控，但结果未覆盖目标参数不确定性。",
            "",
            "## 9. 复现与完整性",
            "",
            "```powershell",
            "& 'D:\\ProgramData\\Anaconda\\envs\\pytorch312-lmdt\\python.exe' `",
            "  review_evidence\\human_factor_remediation_20261022\\controlled_factorial_diagnostic.py",
            "```",
            "",
            "主 JSON 保存设计、软件/输入哈希、分层汇总、配对效应与每工况的嵌套前缀结果。`candidate_ledger.jsonl` 保留所有候选的接受/拒绝状态。空间与清理证据见 `space_and_cleanup.json` 和 `space_and_cleanup.md`。",
            "",
        ]
    )
    return "\n".join(lines)


def make_space_snapshot() -> dict[str, Any]:
    usage = shutil.disk_usage(ROOT)
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_free_bytes": usage.free,
        "workspace_total_bytes": usage.total,
        "workspace_used_bytes": usage.used,
    }


ENGINES: dict[str, DamageEngine] = {}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    wall_start = perf_counter()
    space_before = make_space_snapshot()
    target = load_json(TARGET_PATH)
    registries = {
        name: LEGACY.make_registry(sign) for name, sign in AXIS_CONDITIONS.items()
    }
    global ENGINES
    ENGINES = {
        name: DamageEngine(deepcopy(target), registry)
        for name, registry in registries.items()
    }

    rows: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    attempted = 0
    for axis_condition, engine in ENGINES.items():
        for face in FACE_DEFINITIONS:
            for grid_index in SELECTED_GRID_INDICES:
                for roll_deg in ROLL_DEGREES:
                    for profile in PROFILES:
                        attempted += 1
                        case = make_case(face, grid_index, roll_deg, profile)
                        key = candidate_key(face, grid_index, roll_deg, profile)
                        ledger_base = {
                            "attempt_index": attempted - 1,
                            "axis_condition": axis_condition,
                            "candidate_key": key,
                            "case_id": case["case_id"],
                            "face": face,
                            "grid_index": grid_index,
                            "roll_deg": roll_deg,
                            "profile": profile,
                            "velocity_target_mps": case["encounter"]["velocity_target_mps"],
                            "orientation_xyzw": case["encounter"][
                                "body_to_target_orientation_xyzw"
                            ],
                            "included_in_label_denominator": False,
                        }
                        run_start = perf_counter()
                        try:
                            prepared = engine.prepare_case(case)
                            validate_impact_detonation(prepared, engine.compiled_target)
                            result = engine.run(prepared, fail_closed=True).to_dict()
                            elapsed = perf_counter() - run_start
                            row = summarize_result(
                                result,
                                axis_condition=axis_condition,
                                face=face,
                                grid_index=grid_index,
                                roll_deg=roll_deg,
                                profile=profile,
                                prepared_case=prepared,
                                elapsed_seconds=elapsed,
                            )
                        except DamageEngineError as error:
                            elapsed = perf_counter() - run_start
                            ledger.append(
                                {
                                    **ledger_base,
                                    "status": "refused",
                                    "elapsed_seconds": elapsed,
                                    "refusal": error.to_dict(),
                                }
                            )
                            continue
                        except Exception as error:
                            elapsed = perf_counter() - run_start
                            ledger.append(
                                {
                                    **ledger_base,
                                    "status": "diagnostic_error",
                                    "elapsed_seconds": elapsed,
                                    "error_type": type(error).__name__,
                                    "message": str(error),
                                }
                            )
                            continue
                        rows.append(row)
                        ledger.append(
                            {
                                **ledger_base,
                                "status": "accepted",
                                "included_in_label_denominator": True,
                                "elapsed_seconds": elapsed,
                                "impact_model_id": row["impact_model_id"],
                                "collision_armor_id": row["collision_armor_id"],
                                "collision_normal_target_xyz": row[
                                    "collision_normal_target_xyz"
                                ],
                                "detonation_position_m": row["detonation_position_m"],
                                "collision_distance_m": row["collision_distance_m"],
                                "body_forward_alignment_cosine": row[
                                    "body_forward_alignment_cosine"
                                ],
                                "target_facing_dot_m": row["target_facing_dot_m"],
                            }
                        )

    accepted_keys_by_axis = {
        axis: sorted(
            row["candidate_key"] for row in rows if row["axis_condition"] == axis
        )
        for axis in AXIS_CONDITIONS
    }
    directions = {
        tuple(round(value, 12) for value in row["velocity_direction_target_xyz"])
        for row in rows
    }
    orientations = {
        tuple(round(value, 12) for value in row["orientation_xyzw"]) for row in rows
    }
    seed_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        seed_groups[int(row["aleatory_seed"])].append(row)
    required_axes = set(AXIS_CONDITIONS)
    seed_pairing_violations = [
        {
            "aleatory_seed": seed,
            "row_count": len(group),
            "candidate_keys": sorted({row["candidate_key"] for row in group}),
            "axis_conditions": sorted({row["axis_condition"] for row in group}),
        }
        for seed, group in sorted(seed_groups.items())
        if len(group) != 2
        or len({row["candidate_key"] for row in group}) != 1
        or {row["axis_condition"] for row in group} != required_axes
    ]
    contact_counts = Counter(row["contact_surface_key"] for row in rows)
    statuses = Counter(item["status"] for item in ledger)
    wall_elapsed = perf_counter() - wall_start

    by_axis_and_r = {
        str(level): grouped_aggregate(rows, ("axis_condition",), level)
        for level in PREFIX_LEVELS
    }
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "audit_id": "controlled-factorial-kmfc-support-20261022",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "diagnostic_software_mechanism_sensitivity_not_physical_calibration",
        "research_question": (
            "How much do target-facing approach/contact surface, roll pose, Taylor source-axis "
            "candidate, munition profile and nested replicate count R change KMFC upstream "
            "support and legacy damage-tree output distributions?"
        ),
        "design": {
            "faces": list(FACE_DEFINITIONS),
            "selected_grid_indices_per_face": list(SELECTED_GRID_INDICES),
            "roll_degrees": list(ROLL_DEGREES),
            "profiles": list(PROFILES),
            "axis_conditions": {
                name: [float(sign), 0.0, 0.0]
                for name, sign in AXIS_CONDITIONS.items()
            },
            "speed_mps": SPEED_MPS,
            "nested_prefix_levels": list(PREFIX_LEVELS),
            "maximum_replicates": MAX_REPLICATES,
            "common_random_number_rule": (
                "each complete candidate_key(face, grid_index, roll_deg, profile) has one "
                "unique aleatory seed; only its two source-axis interventions share that seed"
            ),
            "expected_unique_aleatory_seed_count": 162,
            "epistemic_realization_count": 1,
            "fixed_epistemic_sample_id": EPISTEMIC_SAMPLE_ID,
            "epistemic_limitation": (
                "A single fixed target-parameter realization isolates paired software effects "
                "but cannot quantify epistemic variance or factor-by-epistemic interactions."
            ),
            "attempted_case_condition_count": attempted,
            "accepted_case_condition_count": len(rows),
            "refused_case_condition_count": int(statuses.get("refused", 0)),
            "diagnostic_error_count": int(statuses.get("diagnostic_error", 0)),
            "unique_candidate_count": len({row["candidate_key"] for row in rows}),
            "label_denominator_rule": (
                "accepted first-contact simulations only; refused/diagnostic-error candidates "
                "remain in the ledger and are excluded from every label denominator"
            ),
        },
        "contract_checks": {
            "all_accepted_use_first_contact_model": all(
                row["impact_model_id"] == IMPACT_MODEL_ID for row in rows
            ),
            "all_accepted_target_facing": all(row["target_facing_dot_m"] > 0.0 for row in rows),
            "all_accepted_body_forward_aligned": all(
                row["body_forward_alignment_cosine"]
                >= IMPACT_ALIGNMENT_COSINE_MINIMUM
                for row in rows
            ),
            "minimum_alignment_cosine": min(
                row["body_forward_alignment_cosine"] for row in rows
            ),
            "minimum_target_facing_dot_m": min(row["target_facing_dot_m"] for row in rows),
            "all_accepted_have_eight_replicates": all(
                row["prefix"][str(MAX_REPLICATES)]["replicate_count"] == MAX_REPLICATES
                for row in rows
            ),
            "accepted_candidate_sets_equal_across_axis_conditions": len(
                {tuple(keys) for keys in accepted_keys_by_axis.values()}
            )
            == 1,
            "unique_aleatory_seed_count": len(seed_groups),
            "expected_unique_aleatory_seed_count": 162,
            "each_seed_maps_to_one_candidate_key_and_two_axis_conditions": not seed_pairing_violations,
            "seed_pairing_violations": seed_pairing_violations,
            "unique_epistemic_sample_id_count": len(
                {row["epistemic_sample_id"] for row in rows}
            ),
            "unique_velocity_direction_count": len(directions),
            "unique_orientation_count": len(orientations),
            "contact_surface_counts": dict(sorted(contact_counts.items())),
            "ledger_status_counts": dict(sorted(statuses.items())),
        },
        "software_identity": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "damage_engine_version": damage_engine.__version__,
            "engine_code_hash": ENGINE_CODE_HASH,
            "script_path": SCRIPT_PATH.relative_to(ROOT).as_posix(),
            "script_sha256": None,
            "legacy_factorial_script_path": LEGACY_SCRIPT_PATH.relative_to(ROOT).as_posix(),
            "legacy_factorial_script_sha256": file_hash(LEGACY_SCRIPT_PATH),
            "target_path": TARGET_PATH.relative_to(ROOT).as_posix(),
            "target_sha256": file_hash(TARGET_PATH),
            "registry_path": REGISTRY_PATH.relative_to(ROOT).as_posix(),
            "registry_sha256": file_hash(REGISTRY_PATH),
            "base_case_path": BASE_CASE_PATH.relative_to(ROOT).as_posix(),
            "base_case_sha256": file_hash(BASE_CASE_PATH),
            "registry_hashes": {
                name: registry.hash for name, registry in registries.items()
            },
        },
        "runtime": {
            "started_at_utc": started_at.isoformat(),
            "elapsed_seconds": wall_elapsed,
            "accepted_case_conditions_per_second": len(rows) / wall_elapsed,
            "simulated_replicates": len(rows) * MAX_REPLICATES,
        },
        "aggregates": {
            "by_axis_and_r": by_axis_and_r,
            "by_axis_face_r8": grouped_aggregate(
                rows, ("axis_condition", "face"), MAX_REPLICATES
            ),
            "by_axis_roll_r8": grouped_aggregate(
                rows, ("axis_condition", "roll_deg"), MAX_REPLICATES
            ),
            "by_roll_marginal_r8": grouped_aggregate(
                rows, ("roll_deg",), MAX_REPLICATES
            ),
            "by_axis_profile_r8": grouped_aggregate(
                rows, ("axis_condition", "profile"), MAX_REPLICATES
            ),
            "by_axis_contact_surface_r8": grouped_aggregate(
                rows, ("axis_condition", "contact_surface_key"), MAX_REPLICATES
            ),
            "factor_rate_spreads_r8": factor_spreads(rows),
        },
        "paired_axis_effects": paired_axis_effects(rows),
        "nested_r_discovery": nested_r_discovery(rows),
        "roll_sensitivity_r8": roll_sensitivity(rows),
        "axis_by_roll_interaction_r8": axis_by_roll_interaction(rows),
        "case_results": rows,
        "claim_status": {
            "software_mechanism_sensitivity": "computed",
            "contact_conditioning": "enforced_for_all_accepted_cases",
            "physical_axis_correctness": "not_established",
            "physical_parameter_validation": "not_established",
            "operational_frequency_estimation": "not_established",
            "formal_training_dataset_admission": "not_established",
        },
        "interpretation_guards": {
            "axis": (
                "+body-X is a counterfactual intervention only; it is not a validated production fix"
            ),
            "sampling": (
                "balanced diagnostic grid is not an operational reference distribution"
            ),
            "R": (
                "nested any-positive support changes mechanically with R; store successes/trials "
                "and never compare different R values as if their labels had identical semantics"
            ),
            "training": (
                "these cases are mechanism-diagnostic evidence and must remain separate from p0 "
                "reference validation/test and from formal research training until admission gates pass"
            ),
        },
        "space_before": space_before,
    }

    # Hash the executed source only after the run; the source is not modified by main().
    payload["software_identity"]["script_sha256"] = file_hash(SCRIPT_PATH)
    LEDGER_PATH.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for item in ledger
        ),
        encoding="utf-8",
    )
    RESULT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    REPORT_PATH.write_text(build_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "attempted": attempted,
                "accepted": len(rows),
                "statuses": dict(sorted(statuses.items())),
                "elapsed_seconds": wall_elapsed,
                "result_path": RESULT_PATH.relative_to(ROOT).as_posix(),
                "report_path": REPORT_PATH.relative_to(ROOT).as_posix(),
                "ledger_path": LEDGER_PATH.relative_to(ROOT).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
