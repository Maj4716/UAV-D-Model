"""Reproduce the pre-outcome sampling/design audit without opening teacher results.

The script reads only the seven declared pre-outcome/configuration artifacts in
``SOURCE_PATHS``.  In particular it never walks the formal-dataset shards and
never opens a ``results.jsonl*`` file.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[5]
sys.path.insert(0, str(REPO_ROOT))

from damage_engine.design_sequences import design_diagnostics  # noqa: E402


DATASET_ROOT = (
    REPO_ROOT
    / "artifacts/datasets/recovery_role_r8_r32_train2000_eval300_20261122"
)
OUTPUT_PATH = SCRIPT_PATH.with_name("sampling_metrics.json")

SOURCE_PATHS = {
    "sampling_spec": REPO_ROOT
    / "configs/sampling/p0_sphere_entry_continuous_direction_recovery_train2000_eval300_role_r8_r32.v1.json",
    "candidates": DATASET_ROOT / "rounds/round-000/candidates.parquet",
    "formal_cases": DATASET_ROOT / "rounds/round-000/formal_cases.jsonl.gz",
    "split_registry": DATASET_ROOT / "plan/split_registry.json",
    "sampling_quality": DATASET_ROOT / "audits/sampling_quality.json",
    "sampling_checkpoint": DATASET_ROOT / "sampling_checkpoint.json",
    "writer_checkpoint": DATASET_ROOT / "formal_dataset/checkpoint.json",
}

LATENT_NAMES = [
    "munition_profile_selector",
    "direction_region_selector",
    "direction_coordinate_1",
    "direction_coordinate_2",
    "aim_u_quantile",
    "aim_v_quantile",
    "speed_quantile",
    "body_x_roll_quantile",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def uniform_ks(values: np.ndarray) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    n = ordered.size
    if n == 0:
        return math.nan
    upper = np.arange(1, n + 1, dtype=np.float64) / n - ordered
    lower = ordered - np.arange(0, n, dtype=np.float64) / n
    return float(max(np.max(upper), np.max(lower)))


def two_sample_ks(first: np.ndarray, second: np.ndarray) -> float:
    first = np.sort(np.asarray(first, dtype=np.float64))
    second = np.sort(np.asarray(second, dtype=np.float64))
    support = np.unique(np.concatenate((first, second)))
    first_cdf = np.searchsorted(first, support, side="right") / first.size
    second_cdf = np.searchsorted(second, support, side="right") / second.size
    return float(np.max(np.abs(first_cdf - second_cdf)))


def wilson_interval(successes: int, total: int) -> dict[str, float | int | None]:
    if total == 0:
        return {
            "successes": successes,
            "total": total,
            "rate": None,
            "wilson_95_low": None,
            "wilson_95_high": None,
        }
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return {
        "successes": successes,
        "total": total,
        "rate": rate,
        "wilson_95_low": center - half,
        "wilson_95_high": center + half,
    }


def selector_label(value: float, labels: list[str]) -> str:
    index = min(int(math.floor(float(value) * len(labels))), len(labels) - 1)
    return labels[index]


def category_metrics(
    counts: Counter[str], expected_masses: dict[str, float]
) -> dict[str, Any]:
    labels = list(expected_masses)
    total = int(sum(counts.get(label, 0) for label in labels))
    rows = []
    for label in labels:
        count = int(counts.get(label, 0))
        mass = count / total if total else None
        expected_mass = float(expected_masses[label])
        rows.append(
            {
                "label": label,
                "count": count,
                "expected_count": total * expected_mass,
                "mass": mass,
                "expected_mass": expected_mass,
                "mass_deviation": None if mass is None else mass - expected_mass,
            }
        )
    deviations = [abs(row["mass_deviation"]) for row in rows if row["mass_deviation"] is not None]
    maximum_row = max(rows, key=lambda row: abs(row["mass_deviation"] or 0.0))
    return {
        "denominator": total,
        "rows": rows,
        "total_variation_distance": 0.5 * float(sum(deviations)),
        "maximum_absolute_mass_deviation": float(max(deviations, default=0.0)),
        "maximum_deviation_label": maximum_row["label"],
    }


def marginal_uniform_metrics(points: np.ndarray) -> list[dict[str, Any]]:
    expected_variance = 1.0 / 12.0
    records = []
    for axis, name in enumerate(LATENT_NAMES):
        values = points[:, axis]
        mean = float(np.mean(values))
        variance = float(np.var(values, ddof=0))
        records.append(
            {
                "axis": axis,
                "name": name,
                "denominator": int(values.size),
                "expected_distribution": "Uniform(0,1)",
                "expected_mean": 0.5,
                "actual_mean": mean,
                "mean_bias": mean - 0.5,
                "expected_population_variance": expected_variance,
                "actual_population_variance": variance,
                "variance_ratio": variance / expected_variance,
                "ks_distance": uniform_ks(values),
            }
        )
    return records


def compact_design_diagnostics(points: np.ndarray) -> dict[str, Any]:
    diagnostics = design_diagnostics(points)
    projections = diagnostics["two_dimensional_projections"]["projections"]
    maximum_projection = max(projections, key=lambda item: item["centered_discrepancy"])
    correlation = np.asarray(spearmanr(points, axis=0).statistic, dtype=np.float64)
    absolute = np.abs(correlation)
    np.fill_diagonal(absolute, -np.inf)
    first, second = np.unravel_index(int(np.argmax(absolute)), absolute.shape)
    if first > second:
        first, second = second, first
    return {
        "denominator": int(points.shape[0]),
        "dimension": int(points.shape[1]),
        "points_hash": diagnostics["points_hash"],
        "global_centered_discrepancy": diagnostics["global_centered_discrepancy"],
        "marginal_uniformity": diagnostics["marginal_uniformity"],
        "two_dimensional_projections": {
            "projection_count": diagnostics["two_dimensional_projections"]["projection_count"],
            "maximum_centered_discrepancy": diagnostics["two_dimensional_projections"]["maximum_centered_discrepancy"],
            "maximum_axes": maximum_projection["axes"],
            "maximum_axis_names": [
                LATENT_NAMES[maximum_projection["axes"][0]],
                LATENT_NAMES[maximum_projection["axes"][1]],
            ],
            "median_centered_discrepancy": diagnostics["two_dimensional_projections"]["median_centered_discrepancy"],
        },
        "nearest_neighbor_euclidean_8d": diagnostics["nearest_neighbor"],
        "rank_dependence": {
            "maximum_absolute_spearman_rho": float(absolute[first, second]),
            "signed_spearman_rho": float(correlation[first, second]),
            "axes": [int(first), int(second)],
            "axis_names": [LATENT_NAMES[first], LATENT_NAMES[second]],
        },
    }


def rate_breakdown(
    records: list[dict[str, Any]], labels: Iterable[str], key: str
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for label in labels:
        subset = [record for record in records if record[key] == label]
        selected = sum(bool(record["selected"]) for record in subset)
        output[label] = wilson_interval(selected, len(subset))
        output[label]["rejected"] = len(subset) - selected
    return output


def tri_state_counts(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    output = Counter()
    for record in records:
        value = record[key]
        output["null" if value is None else str(bool(value)).lower()] += 1
    return dict(sorted(output.items()))


def add_assertion(
    assertions: list[dict[str, Any]], name: str, passed: bool, evidence: Any
) -> None:
    assertions.append({"name": name, "passed": bool(passed), "evidence": evidence})


def main() -> None:
    spec = read_json(SOURCE_PATHS["sampling_spec"])
    registry = read_json(SOURCE_PATHS["split_registry"])
    quality = read_json(SOURCE_PATHS["sampling_quality"])
    sampling_checkpoint = read_json(SOURCE_PATHS["sampling_checkpoint"])
    writer_checkpoint = read_json(SOURCE_PATHS["writer_checkpoint"])

    profile_values = list(spec["reference_distribution"]["variables"][0]["values"])
    profile_masses = {profile: 1.0 / len(profile_values) for profile in profile_values}
    region_entries = spec["contact_manifold"]["regions"]
    region_values = [entry["region_id"] for entry in region_entries]
    region_weight_sum = float(sum(entry["selection_weight"] for entry in region_entries))
    region_masses = {
        entry["region_id"]: float(entry["selection_weight"]) / region_weight_sum
        for entry in region_entries
    }
    joint_values = [
        f"{profile}|{region}" for profile in profile_values for region in region_values
    ]
    joint_masses = {
        f"{profile}|{region}": profile_masses[profile] * region_masses[region]
        for profile in profile_values
        for region in region_values
    }

    quota_by_role: dict[str, int] = {}
    for round_budget in spec["round_budgets"]:
        for allocation in round_budget["allocations"]:
            quota_by_role[allocation["role"]] = quota_by_role.get(allocation["role"], 0) + int(
                allocation["count"]
            )
    roles = [entry["role"] for entry in spec["stream_roles"]]
    role_to_split = {entry["role"]: entry["split"] for entry in spec["stream_roles"]}

    candidate_columns = [
        "candidate_id",
        "design_group_id",
        "epistemic_group_id",
        "lineage_id",
        "role",
        "split",
        "split_group",
        "selected",
        "design_weight_raw",
        "inclusion_probability",
        "weight_mask",
        "latent",
        "executed",
        "successful",
        "admitted",
        "refusal",
        "admission_purpose",
    ]
    table = pq.read_table(SOURCE_PATHS["candidates"], columns=candidate_columns)
    candidates: list[dict[str, Any]] = table.to_pylist()
    for record in candidates:
        latent = [float(value) for value in record["latent"]]
        record["latent"] = latent
        record["profile"] = selector_label(latent[0], profile_values)
        record["region"] = selector_label(latent[1], region_values)
        record["profile_region"] = f"{record['profile']}|{record['region']}"

    selected_candidates = [record for record in candidates if record["selected"]]
    rejected_candidates = [record for record in candidates if not record["selected"]]
    raw_points = np.asarray([record["latent"] for record in candidates], dtype=np.float64)
    selected_points = np.asarray(
        [record["latent"] for record in selected_candidates], dtype=np.float64
    )

    candidate_profile_counts = Counter(record["profile"] for record in candidates)
    selected_profile_counts = Counter(record["profile"] for record in selected_candidates)
    candidate_region_counts = Counter(record["region"] for record in candidates)
    selected_region_counts = Counter(record["region"] for record in selected_candidates)
    candidate_joint_counts = Counter(record["profile_region"] for record in candidates)
    selected_joint_counts = Counter(record["profile_region"] for record in selected_candidates)
    selected_role_counts = Counter(record["role"] for record in selected_candidates)
    candidate_role_counts = Counter(record["role"] for record in candidates)

    expected_role_total = sum(quota_by_role.values())
    role_masses = {
        role: quota_by_role[role] / expected_role_total for role in quota_by_role
    }

    design_sets: dict[str, Any] = {}
    for set_name, subset in (
        ("candidate_frame_all_roles", candidates),
        ("selected_formal_frame_all_roles", selected_candidates),
    ):
        points = np.asarray([record["latent"] for record in subset], dtype=np.float64)
        design_sets[set_name] = {
            "marginals": marginal_uniform_metrics(points),
            "joint_diagnostics": compact_design_diagnostics(points),
        }
    for role in roles:
        for prefix, source in (
            ("candidate_frame", candidates),
            ("selected_formal_frame", selected_candidates),
        ):
            subset = [record for record in source if record["role"] == role]
            points = np.asarray([record["latent"] for record in subset], dtype=np.float64)
            design_sets[f"{prefix}:{role}"] = {
                "marginals": marginal_uniform_metrics(points),
                "joint_diagnostics": compact_design_diagnostics(points),
            }

    selection_shift = []
    for axis, name in enumerate(LATENT_NAMES):
        raw_values = raw_points[:, axis]
        selected_values = selected_points[:, axis]
        raw_std = float(np.std(raw_values, ddof=0))
        mean_difference = float(np.mean(selected_values) - np.mean(raw_values))
        selection_shift.append(
            {
                "axis": axis,
                "name": name,
                "candidate_denominator": int(raw_values.size),
                "selected_denominator": int(selected_values.size),
                "candidate_mean": float(np.mean(raw_values)),
                "selected_mean": float(np.mean(selected_values)),
                "selected_minus_candidate_mean": mean_difference,
                "difference_in_candidate_standard_deviation_units": mean_difference / raw_std,
                "two_sample_ks_distance_descriptive": two_sample_ks(
                    raw_values, selected_values
                ),
            }
        )

    refusal_counts = Counter()
    for record in rejected_candidates:
        refusal = record["refusal"] or {}
        refusal_counts[refusal.get("code") or "missing_code"] += 1
    selected_refusal_count = sum(
        bool((record["refusal"] or {}).get("code")) for record in selected_candidates
    )

    weights = np.asarray(
        [record["design_weight_raw"] for record in selected_candidates], dtype=np.float64
    )
    inclusion = np.asarray(
        [record["inclusion_probability"] for record in selected_candidates],
        dtype=np.float64,
    )
    recomputed_ess = float(np.sum(weights) ** 2 / np.sum(weights * weights))
    recomputed_cv = float(np.std(weights, ddof=0) / np.mean(weights))
    quality_thresholds = spec["quality_gates"]
    recomputed_quality = {
        "denominator": len(selected_candidates),
        "positivity_min": float(np.min(inclusion)),
        "positivity_max": float(np.max(inclusion)),
        "effective_sample_size": recomputed_ess,
        "ess_fraction": recomputed_ess / len(selected_candidates),
        "design_weight_cv_population": recomputed_cv,
        "thresholds": quality_thresholds,
    }
    recomputed_quality["gates_passed"] = bool(
        recomputed_quality["positivity_min"] >= quality_thresholds["positivity_min"]
        and recomputed_quality["ess_fraction"] >= quality_thresholds["min_ess_fraction"]
        and recomputed_quality["design_weight_cv_population"]
        <= quality_thresholds["max_weight_cv"]
    )

    registry_assignments = registry["assignments"]
    registry_by_lineage: dict[str, dict[str, Any]] = {}
    duplicate_registry_lineages = 0
    for assignment in registry_assignments:
        lineage = assignment["lineage_id"]
        if lineage in registry_by_lineage:
            duplicate_registry_lineages += 1
        registry_by_lineage[lineage] = assignment
    registry_missing = 0
    registry_mismatch_counts = Counter()
    for record in candidates:
        assignment = registry_by_lineage.get(record["lineage_id"])
        if assignment is None:
            registry_missing += 1
            continue
        for field in ("role", "split", "epistemic_group_id"):
            if assignment.get(field) != record.get(field):
                registry_mismatch_counts[field] += 1
    group_roles: dict[str, set[str]] = defaultdict(set)
    group_splits: dict[str, set[str]] = defaultdict(set)
    for assignment in registry_assignments:
        group = assignment["epistemic_group_id"]
        group_roles[group].add(assignment["role"])
        group_splits[group].add(assignment["split"])
    role_leaking_groups = sum(len(values) > 1 for values in group_roles.values())
    split_leaking_groups = sum(len(values) > 1 for values in group_splits.values())

    candidate_by_id = {record["candidate_id"]: record for record in candidates}
    formal_records: list[dict[str, Any]] = []
    formal_status_counts = Counter()
    eligibility_status_counts = Counter()
    formal_role_counts = Counter()
    formal_split_counts = Counter()
    formal_profile_counts = Counter()
    formal_region_counts = Counter()
    formal_joint_counts = Counter()
    formal_check_true_counts = Counter()
    formal_check_false_or_missing_counts = Counter()
    formal_selection_stage_counts = Counter()
    formal_outcome_blind_counts = Counter()
    formal_label_allowed_counts = Counter()
    formal_candidate_missing = 0
    formal_profile_mapping_mismatch = 0
    formal_region_mapping_mismatch = 0
    formal_latent_mismatch = 0
    formal_weight_mismatch = 0
    formal_registry_mismatch = 0
    with gzip.open(SOURCE_PATHS["formal_cases"], "rt", encoding="utf-8") as handle:
        for line in handle:
            case = json.loads(line)
            metadata = case["metadata"]
            sampling = metadata["sampling"]
            contact = metadata["contact_manifold_sampling"]
            continuous = metadata["continuous_direction_sampling"]
            eligibility = sampling["contact_conditioning"]["eligibility_record"]
            selection = eligibility["selection"]
            role = sampling["role"]
            split = sampling["frozen_split"]
            profile = case["munition"]["profile_id"]
            region = contact["region_id"]
            candidate_id = sampling["candidate_id"]
            record = {
                "case_id": case["case_id"],
                "candidate_id": candidate_id,
                "lineage_id": sampling["lineage_id"],
                "epistemic_group_id": sampling["epistemic_group_id"],
                "role": role,
                "split": split,
                "profile": profile,
                "region": region,
                "profile_region": f"{profile}|{region}",
            }
            formal_records.append(record)
            formal_status_counts[contact.get("status", "missing")] += 1
            eligibility_status_counts[eligibility.get("status", "missing")] += 1
            formal_role_counts[role] += 1
            formal_split_counts[split] += 1
            formal_profile_counts[profile] += 1
            formal_region_counts[region] += 1
            formal_joint_counts[record["profile_region"]] += 1
            formal_selection_stage_counts[selection.get("selection_stage", "missing")] += 1
            formal_outcome_blind_counts[str(selection.get("outcome_blind")).lower()] += 1
            formal_label_allowed_counts[str(eligibility.get("formal_label_allowed")).lower()] += 1
            for check_name, check_value in eligibility.get("checks", {}).items():
                if check_value is True:
                    formal_check_true_counts[check_name] += 1
                else:
                    formal_check_false_or_missing_counts[check_name] += 1
            candidate = candidate_by_id.get(candidate_id)
            if candidate is None:
                formal_candidate_missing += 1
                continue
            if candidate["profile"] != profile:
                formal_profile_mapping_mismatch += 1
            if candidate["region"] != region:
                formal_region_mapping_mismatch += 1
            if not np.array_equal(
                np.asarray(candidate["latent"][1:], dtype=np.float64),
                np.asarray(continuous["latent"], dtype=np.float64),
            ):
                formal_latent_mismatch += 1
            if not (
                candidate["design_weight_raw"] == sampling["design_weight_raw"]
                and candidate["inclusion_probability"] == sampling["selection_probability"]
                and candidate["weight_mask"] == sampling["weight_mask"]
            ):
                formal_weight_mismatch += 1
            assignment = registry_by_lineage.get(sampling["lineage_id"])
            if assignment is None or any(
                assignment.get(field) != record[field]
                for field in ("role", "split", "epistemic_group_id")
            ):
                formal_registry_mismatch += 1

    formal_candidate_ids = [record["candidate_id"] for record in formal_records]
    formal_case_ids = [record["case_id"] for record in formal_records]
    selected_candidate_ids = [record["candidate_id"] for record in selected_candidates]

    completed_shards = writer_checkpoint["completed_shards"]
    writer_case_ids = [
        case_id for shard in completed_shards for case_id in shard.get("case_ids", [])
    ]
    writer_candidate_ids = [
        candidate_id
        for shard in completed_shards
        for candidate_id in shard.get("candidate_ids", [])
    ]
    shard_indices = [int(shard["index"]) for shard in completed_shards]

    assertions: list[dict[str, Any]] = []
    add_assertion(
        assertions,
        "selected_total_matches_declared_budget",
        len(selected_candidates) == expected_role_total,
        {"actual": len(selected_candidates), "expected": expected_role_total},
    )
    add_assertion(
        assertions,
        "selected_role_quotas_exact",
        all(selected_role_counts[role] == quota_by_role[role] for role in quota_by_role),
        {"actual": dict(selected_role_counts), "expected": quota_by_role},
    )
    add_assertion(
        assertions,
        "formal_roles_and_splits_match_declared",
        all(formal_role_counts[role] == quota_by_role[role] for role in quota_by_role)
        and all(formal_split_counts[role_to_split[role]] == quota_by_role[role] for role in quota_by_role),
        {"roles": dict(formal_role_counts), "splits": dict(formal_split_counts)},
    )
    add_assertion(
        assertions,
        "formal_candidate_set_equals_selected_candidate_set",
        set(formal_candidate_ids) == set(selected_candidate_ids)
        and len(formal_candidate_ids) == len(selected_candidate_ids),
        {
            "formal_count": len(formal_candidate_ids),
            "selected_count": len(selected_candidate_ids),
            "formal_unique": len(set(formal_candidate_ids)),
            "selected_unique": len(set(selected_candidate_ids)),
        },
    )
    add_assertion(
        assertions,
        "formal_cases_have_unique_case_and_candidate_ids",
        len(set(formal_case_ids)) == len(formal_case_ids)
        and len(set(formal_candidate_ids)) == len(formal_candidate_ids),
        {
            "case_duplicates": len(formal_case_ids) - len(set(formal_case_ids)),
            "candidate_duplicates": len(formal_candidate_ids) - len(set(formal_candidate_ids)),
        },
    )
    add_assertion(
        assertions,
        "all_formal_cases_admitted_s_contact",
        formal_status_counts == {"admitted_s_contact": len(formal_records)},
        dict(formal_status_counts),
    )
    add_assertion(
        assertions,
        "all_formal_eligibility_checks_true",
        not formal_check_false_or_missing_counts
        and all(count == len(formal_records) for count in formal_check_true_counts.values())
        and len(formal_check_true_counts) == 7,
        {
            "true_counts": dict(formal_check_true_counts),
            "false_or_missing_counts": dict(formal_check_false_or_missing_counts),
        },
    )
    add_assertion(
        assertions,
        "formal_selection_is_pre_outcome_and_outcome_blind",
        formal_selection_stage_counts == {"pre_outcome": len(formal_records)}
        and formal_outcome_blind_counts == {"true": len(formal_records)},
        {
            "selection_stage": dict(formal_selection_stage_counts),
            "outcome_blind": dict(formal_outcome_blind_counts),
        },
    )
    add_assertion(
        assertions,
        "candidate_to_formal_mapping_exact",
        formal_candidate_missing == 0
        and formal_profile_mapping_mismatch == 0
        and formal_region_mapping_mismatch == 0
        and formal_latent_mismatch == 0
        and formal_weight_mismatch == 0,
        {
            "candidate_missing": formal_candidate_missing,
            "profile_mapping_mismatch": formal_profile_mapping_mismatch,
            "region_mapping_mismatch": formal_region_mapping_mismatch,
            "latent_mismatch": formal_latent_mismatch,
            "weight_mismatch": formal_weight_mismatch,
        },
    )
    add_assertion(
        assertions,
        "split_registry_exact_for_all_candidates_and_formal_cases",
        registry_missing == 0
        and not registry_mismatch_counts
        and formal_registry_mismatch == 0
        and duplicate_registry_lineages == 0,
        {
            "candidate_missing": registry_missing,
            "candidate_mismatches": dict(registry_mismatch_counts),
            "formal_mismatches": formal_registry_mismatch,
            "duplicate_registry_lineages": duplicate_registry_lineages,
        },
    )
    add_assertion(
        assertions,
        "epistemic_groups_do_not_cross_role_or_split",
        role_leaking_groups == 0 and split_leaking_groups == 0,
        {
            "group_denominator": len(group_roles),
            "role_leaking_groups": role_leaking_groups,
            "split_leaking_groups": split_leaking_groups,
        },
    )
    add_assertion(
        assertions,
        "all_rejections_have_declared_contact_reason",
        refusal_counts == {"no_forward_armor_intersection": len(rejected_candidates)}
        and selected_refusal_count == 0,
        {
            "rejected_denominator": len(rejected_candidates),
            "refusal_counts": dict(refusal_counts),
            "selected_with_refusal_code": selected_refusal_count,
        },
    )
    add_assertion(
        assertions,
        "recomputed_weight_quality_gates_pass",
        recomputed_quality["gates_passed"],
        recomputed_quality,
    )
    add_assertion(
        assertions,
        "writer_checkpoint_commits_all_formal_inputs",
        writer_checkpoint.get("status") == "complete"
        and len(writer_case_ids) == len(formal_case_ids)
        and set(writer_case_ids) == set(formal_case_ids)
        and set(writer_candidate_ids) == set(formal_candidate_ids)
        and shard_indices == list(range(len(completed_shards))),
        {
            "status": writer_checkpoint.get("status"),
            "completed_shards": len(completed_shards),
            "committed_cases": len(writer_case_ids),
            "formal_cases": len(formal_case_ids),
            "shard_indices_contiguous": shard_indices == list(range(len(completed_shards))),
        },
    )

    reported_weight = quality.get("design_weight", {})
    quality_report_consistency = {
        "candidate_count_matches": quality.get("candidate_count") == len(candidates),
        "selected_count_matches": quality.get("selected_count") == len(selected_candidates),
        "positivity_min_matches": quality.get("positivity_min")
        == recomputed_quality["positivity_min"],
        "positivity_max_matches": quality.get("positivity_max")
        == recomputed_quality["positivity_max"],
        "ess_matches": quality.get("effective_sample_size")
        == recomputed_quality["effective_sample_size"],
        "ess_fraction_matches": quality.get("ess_fraction")
        == recomputed_quality["ess_fraction"],
        "weight_cv_matches": reported_weight.get("cv")
        == recomputed_quality["design_weight_cv_population"],
        "reported_quality_gate_passed": quality.get("quality_gate_passed"),
    }
    add_assertion(
        assertions,
        "sampling_quality_report_matches_recomputation",
        all(quality_report_consistency.values()),
        quality_report_consistency,
    )

    metrics = {
        "schema_version": "pipeline-sampling-design-metrics@1.0.0",
        "analysis_id": "pipeline-expected-actual-20261122-v1:pre-outcome-sampling",
        "scope": {
            "population": "pre-outcome design and S-CONTACT formal-input population",
            "candidate_frame_denominator": len(candidates),
            "formal_input_denominator": len(formal_records),
            "sealed_test_policy": "test inputs may be audited; teacher results remain sealed",
            "teacher_result_files_opened": 0,
            "teacher_result_json_lines_decoded": 0,
            "training_runs_started": 0,
            "a40_runs_started": 0,
            "inferential_rule": "KS, discrepancy, and rank correlation are randomized-QMC design diagnostics; no iid p-values are reported",
        },
        "evidence_sources": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in SOURCE_PATHS.items()
        },
        "declared_design": {
            "sampling_spec_id": spec["sampling_spec_id"],
            "sampling_spec_status": spec["status"],
            "design_method": spec["design_sequence"],
            "contact_estimand": spec["contact_conditioning"],
            "role_quotas": quota_by_role,
            "role_to_split": role_to_split,
            "profile_masses_pre_contact": profile_masses,
            "region_masses_pre_contact": region_masses,
            "joint_profile_region_masses_pre_contact": joint_masses,
        },
        "candidate_generation_and_selection": {
            "candidate_count": len(candidates),
            "selected_count": len(selected_candidates),
            "rejected_count": len(rejected_candidates),
            "overall_selection": wilson_interval(len(selected_candidates), len(candidates)),
            "candidate_counts_by_role": dict(candidate_role_counts),
            "selected_counts_by_role": dict(selected_role_counts),
            "role_quota_comparison": category_metrics(selected_role_counts, role_masses),
            "profile_pre_contact_candidate_vs_declared": category_metrics(
                candidate_profile_counts, profile_masses
            ),
            "profile_formal_selected_vs_pre_contact_declared": category_metrics(
                selected_profile_counts, profile_masses
            ),
            "region_pre_contact_candidate_vs_declared": category_metrics(
                candidate_region_counts, region_masses
            ),
            "region_formal_selected_vs_pre_contact_declared": category_metrics(
                selected_region_counts, region_masses
            ),
            "profile_region_pre_contact_candidate_vs_declared": category_metrics(
                candidate_joint_counts, joint_masses
            ),
            "profile_region_formal_selected_vs_pre_contact_declared": category_metrics(
                selected_joint_counts, joint_masses
            ),
            "selection_rate_by_role": rate_breakdown(candidates, roles, "role"),
            "selection_rate_by_profile": rate_breakdown(
                candidates, profile_values, "profile"
            ),
            "selection_rate_by_region": rate_breakdown(
                candidates, region_values, "region"
            ),
            "selection_rate_by_profile_region": rate_breakdown(
                candidates, joint_values, "profile_region"
            ),
            "refusal_counts": dict(refusal_counts),
            "selected_with_refusal_code": selected_refusal_count,
            "candidate_execution_fields": {
                "executed": tri_state_counts(candidates, "executed"),
                "successful": tri_state_counts(candidates, "successful"),
                "admitted": tri_state_counts(candidates, "admitted"),
                "interpretation": "candidate ledger records pre-outcome contact admission; formal teacher completion is represented only by checkpoint status",
            },
            "design_diagnostics": design_sets,
            "selection_induced_latent_shift": selection_shift,
        },
        "weight_quality": {
            "recomputed": recomputed_quality,
            "reported": {
                "candidate_count": quality.get("candidate_count"),
                "selected_count": quality.get("selected_count"),
                "weighted_selected_count": quality.get("weighted_selected_count"),
                "positivity_min": quality.get("positivity_min"),
                "positivity_max": quality.get("positivity_max"),
                "effective_sample_size": quality.get("effective_sample_size"),
                "ess_fraction": quality.get("ess_fraction"),
                "design_weight": reported_weight,
                "quality_gate_passed": quality.get("quality_gate_passed"),
                "quality_gates": quality.get("quality_gates"),
                "coverage_status": quality.get("coverage_status"),
                "coverage_guaranteed": quality.get("coverage_guaranteed"),
                "registry_assignment_status": quality.get("registry_assignment_status"),
                "evidence_boundary": quality.get("evidence_boundary"),
            },
            "report_consistency": quality_report_consistency,
        },
        "formal_input_and_contact_admission": {
            "formal_case_count": len(formal_records),
            "unique_case_count": len(set(formal_case_ids)),
            "unique_candidate_count": len(set(formal_candidate_ids)),
            "role_counts": dict(formal_role_counts),
            "split_counts": dict(formal_split_counts),
            "profile_counts": dict(formal_profile_counts),
            "region_counts": dict(formal_region_counts),
            "profile_region_counts": dict(formal_joint_counts),
            "contact_status_counts": dict(formal_status_counts),
            "eligibility_status_counts": dict(eligibility_status_counts),
            "eligibility_check_true_counts": dict(formal_check_true_counts),
            "eligibility_check_false_or_missing_counts": dict(
                formal_check_false_or_missing_counts
            ),
            "selection_stage_counts": dict(formal_selection_stage_counts),
            "outcome_blind_counts": dict(formal_outcome_blind_counts),
            "formal_label_allowed_counts": dict(formal_label_allowed_counts),
            "candidate_mapping_mismatch_counts": {
                "candidate_missing": formal_candidate_missing,
                "profile": formal_profile_mapping_mismatch,
                "region": formal_region_mapping_mismatch,
                "latent": formal_latent_mismatch,
                "weight": formal_weight_mismatch,
            },
        },
        "split_registry": {
            "registry_id": registry.get("registry_id"),
            "frozen_at": registry.get("frozen_at"),
            "assignment_phase": registry.get("assignment_phase"),
            "definition_hash": registry.get("definition_hash"),
            "assignment_count": len(registry_assignments),
            "unique_lineage_count": len(registry_by_lineage),
            "candidate_missing_count": registry_missing,
            "candidate_mismatch_counts": dict(registry_mismatch_counts),
            "formal_mismatch_count": formal_registry_mismatch,
            "epistemic_group_count": len(group_roles),
            "role_leaking_group_count": role_leaking_groups,
            "split_leaking_group_count": split_leaking_groups,
        },
        "checkpoints": {
            "sampling": {
                "schema_version": sampling_checkpoint.get("schema_version"),
                "status": sampling_checkpoint.get("status"),
                "rounds_committed_count": len(
                    sampling_checkpoint.get("rounds_committed", [])
                ),
                "split_registry_hash": sampling_checkpoint.get("split_registry_hash"),
                "completed_at": sampling_checkpoint.get("completed_at"),
                "archive_summary": {
                    key: value
                    for key, value in sampling_checkpoint.get("archive_summary", {}).items()
                    if key != "candidate_sets"
                }
                | {
                    "candidate_sets": {
                        name: {
                            "count": record.get("count"),
                            "status": record.get("status"),
                        }
                        for name, record in sampling_checkpoint.get(
                            "archive_summary", {}
                        )
                        .get("candidate_sets", {})
                        .items()
                    }
                },
            },
            "formal_writer": {
                "schema_version": writer_checkpoint.get("schema_version"),
                "status": writer_checkpoint.get("status"),
                "dataset_id": writer_checkpoint.get("dataset_id"),
                "root_manifest_hash": writer_checkpoint.get("root_manifest_hash"),
                "completed_shard_count": len(completed_shards),
                "committed_case_count": len(writer_case_ids),
                "unique_committed_case_count": len(set(writer_case_ids)),
                "unique_committed_candidate_count": len(set(writer_candidate_ids)),
                "shard_indices_contiguous_from_zero": shard_indices
                == list(range(len(completed_shards))),
                "input_snapshot": writer_checkpoint.get("identity", {}).get(
                    "input_snapshot"
                ),
            },
        },
        "assertions": assertions,
        "assertion_summary": {
            "total": len(assertions),
            "passed": sum(assertion["passed"] for assertion in assertions),
            "failed": sum(not assertion["passed"] for assertion in assertions),
            "failed_names": [
                assertion["name"] for assertion in assertions if not assertion["passed"]
            ],
        },
        "reproduction": {
            "python_executable": r"D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe",
            "command_from_repo_root": r"& 'D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe' 'artifacts\validation\pipeline_expected_actual_20261122\work\sampling\analyze_sampling_design.py'",
            "script_path": str(SCRIPT_PATH),
            "output_path": str(OUTPUT_PATH),
        },
    }

    OUTPUT_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT_PATH),
                "candidate_count": len(candidates),
                "selected_count": len(selected_candidates),
                "formal_case_count": len(formal_records),
                "assertion_summary": metrics["assertion_summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if metrics["assertion_summary"]["failed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
