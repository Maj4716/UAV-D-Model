"""Deep, read-only distribution audit for a completed sharded damage dataset.

The report is descriptive.  It does not establish physical validity or research
admission, and it deliberately keeps declared sampling gates separate from
exploratory diagnostics.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import ks_2samp, spearmanr

# Keep direct ``python tools/audit_dataset_distribution.py`` execution bound to
# this checkout rather than an unrelated installed package.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from damage_engine.normalization import load_training_normalization_artifact
from damage_engine.audit_statistics import (
    stable_effective_sample_size,
    stable_finite_summary_statistics,
    standardized_mean_difference,
    validate_design_weight_contract,
)
from damage_engine.torch_dataset import validated_committed_shard_tensor_paths
from damage_engine.utils import file_hash


def _json_number(value: Any) -> float | int | None:
    value = value.item() if isinstance(value, np.generic) else value
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _summary(values: np.ndarray) -> dict[str, Any]:
    raw_values = np.asarray(values, dtype=np.float64)
    values = raw_values[np.isfinite(raw_values)]
    denominators = {
        "total_count": int(raw_values.size),
        "finite_count": int(values.size),
        "missing_count": int(np.count_nonzero(np.isnan(raw_values))),
        "invalid_count": int(np.count_nonzero(np.isinf(raw_values))),
    }
    if values.size == 0:
        return {"count": 0, **denominators, "status": "not_estimable_no_finite_values"}
    unique, counts = np.unique(values, return_counts=True)
    mode_i = int(np.argmax(counts))
    stable = stable_finite_summary_statistics(
        values, np.asarray([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    )
    quantiles = stable["quantiles"]
    mean = stable["mean"]
    std = stable["standard_deviation"]
    return {
        "count": int(values.size),
        **denominators,
        "status": "estimated",
        "unique": int(unique.size),
        "min": float(quantiles[0]),
        "p01": float(quantiles[1]),
        "p05": float(quantiles[2]),
        "p25": float(quantiles[3]),
        "median": float(quantiles[4]),
        "p75": float(quantiles[5]),
        "p95": float(quantiles[6]),
        "p99": float(quantiles[7]),
        "max": float(quantiles[8]),
        "mean": mean,
        "std": std,
        "coefficient_of_variation": stable["coefficient_of_variation"],
        "coefficient_of_variation_status": stable["coefficient_of_variation_status"],
        "numeric_method": stable["numeric_method"],
        "numeric_scale": stable["numeric_scale"],
        "mode": float(unique[mode_i]),
        "mode_count": int(counts[mode_i]),
        "mode_fraction": float(counts[mode_i] / values.size),
        "exact_zero_fraction": float(np.mean(values == 0.0)),
        "exact_one_fraction": float(np.mean(values == 1.0)),
    }


def _flatten_label_block(
    values: np.ndarray,
    mask: np.ndarray,
    names: list[str],
) -> list[dict[str, Any]]:
    flat_values = values.reshape(values.shape[0], -1)
    flat_mask = mask.reshape(mask.shape[0], -1).astype(bool)
    result: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        available = flat_mask[:, index]
        stats = _summary(flat_values[available, index])
        stats.update(
            {
                "name": name,
                "available": int(np.sum(available)),
                "availability": float(np.mean(available)),
            }
        )
        result.append(stats)
    return result


def _split_drift(
    features: np.ndarray,
    feature_mask: np.ndarray,
    names: list[str],
    splits: np.ndarray,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    comparisons = [("train", "validation"), ("train", "test"), ("validation", "test")]
    for j, name in enumerate(names):
        valid = feature_mask[:, j].astype(bool)
        records = []
        for left, right in comparisons:
            a = features[(splits == left) & valid, j].astype(np.float64)
            b = features[(splits == right) & valid, j].astype(np.float64)
            smd = standardized_mean_difference(a, b)
            a_finite = a[np.isfinite(a)]
            b_finite = b[np.isfinite(b)]
            ks = (
                ks_2samp(a_finite, b_finite, alternative="two-sided", method="auto")
                if a_finite.size and b_finite.size
                else None
            )
            records.append(
                {
                    "comparison": f"{left}_vs_{right}",
                    "standardized_mean_difference": smd[
                        "absolute_standardized_mean_difference"
                    ],
                    "standardized_mean_difference_status": smd["status"],
                    "raw_mean_difference": smd["raw_mean_difference"],
                    "pooled_standard_deviation": smd["pooled_standard_deviation"],
                    "pooled_standard_deviation_status": smd[
                        "pooled_standard_deviation_status"
                    ],
                    "denominators": {
                        key: smd[key]
                        for key in (
                            "left_total_count",
                            "right_total_count",
                            "left_finite_count",
                            "right_finite_count",
                            "left_nonfinite_count",
                            "right_nonfinite_count",
                            "left_missing_count",
                            "right_missing_count",
                            "left_invalid_count",
                            "right_invalid_count",
                        )
                    },
                    "ks_status": "estimated" if ks is not None else "not_estimable_empty_group",
                    "ks_statistic": float(ks.statistic) if ks is not None else None,
                    "ks_pvalue_exploratory_unadjusted": (
                        float(ks.pvalue) if ks is not None else None
                    ),
                }
            )
        result.append({"name": name, "comparisons": records})
    return result


def _nearest_reference_coverage(
    features: np.ndarray,
    splits: np.ndarray,
    query_split: str,
) -> dict[str, Any]:
    reference = features[splits == "train"]
    query = features[splits == query_split]
    ref_position = reference[:, :3].astype(np.float64)
    qry_position = query[:, :3].astype(np.float64)
    ref_position /= np.linalg.norm(ref_position, axis=1, keepdims=True)
    qry_position /= np.linalg.norm(qry_position, axis=1, keepdims=True)
    ref_quaternion = reference[:, 4:8].astype(np.float64)
    qry_quaternion = query[:, 4:8].astype(np.float64)
    ref_quaternion /= np.linalg.norm(ref_quaternion, axis=1, keepdims=True)
    qry_quaternion /= np.linalg.norm(qry_quaternion, axis=1, keepdims=True)

    position_dot = np.clip(qry_position @ ref_position.T, -1.0, 1.0)
    quaternion_dot = np.clip(np.abs(qry_quaternion @ ref_quaternion.T), 0.0, 1.0)
    position_angle = np.arccos(position_dot)
    rotation_angle = 2.0 * np.arccos(quaternion_dot)
    combined = np.sqrt(position_angle**2 + rotation_angle**2)
    nearest = np.argmin(combined, axis=1)
    rows = np.arange(query.shape[0])
    return {
        "query_split": query_split,
        "reference_split": "train",
        "count": int(query.shape[0]),
        "metric": "sqrt(position_geodesic_rad^2 + rotation_geodesic_rad^2)",
        "combined_distance_rad": _summary(combined[rows, nearest]),
        "position_angle_deg_at_combined_nearest": _summary(
            np.degrees(position_angle[rows, nearest])
        ),
        "rotation_angle_deg_at_combined_nearest": _summary(
            np.degrees(rotation_angle[rows, nearest])
        ),
    }


def _sphere_coverage(features: np.ndarray) -> dict[str, Any]:
    position = features[:, :3].astype(np.float64)
    unit = position / np.linalg.norm(position, axis=1, keepdims=True)
    azimuth = (np.arctan2(unit[:, 1], unit[:, 0]) + 2.0 * np.pi) % (2.0 * np.pi)
    z_equal_area = (unit[:, 2] + 1.0) / 2.0
    az_bin = np.minimum((azimuth / (2.0 * np.pi) * 20).astype(int), 19)
    z_bin = np.minimum((z_equal_area * 10).astype(int), 9)
    counts = np.bincount(z_bin * 20 + az_bin, minlength=200)
    octant = (
        (unit[:, 0] >= 0).astype(int) * 4
        + (unit[:, 1] >= 0).astype(int) * 2
        + (unit[:, 2] >= 0).astype(int)
    )
    return {
        "radius_m": _summary(np.linalg.norm(position, axis=1)),
        "equal_area_grid": {
            "azimuth_bins": 20,
            "z_bins": 10,
            "occupied_cells": int(np.sum(counts > 0)),
            "total_cells": 200,
            "occupancy_fraction": float(np.mean(counts > 0)),
            "min_count": int(np.min(counts)),
            "median_count": float(np.median(counts)),
            "max_count": int(np.max(counts)),
        },
        "octant_counts": np.bincount(octant, minlength=8).astype(int).tolist(),
    }


def _interval_fractions(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    intervals = [
        ("[0.0,0.1)", 0.0, 0.1),
        ("[0.1,0.4)", 0.1, 0.4),
        ("[0.4,0.6)", 0.4, 0.6),
        ("[0.6,0.9)", 0.6, 0.9),
        ("[0.9,1.0]", 0.9, math.nextafter(1.0, math.inf)),
    ]
    return {
        name: float(np.mean((values >= low) & (values < high)))
        for name, low, high in intervals
    }


def _spearman_matrix(
    features: np.ndarray,
    feature_names: list[str],
    labels: np.ndarray,
    label_names: list[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for i, feature_name in enumerate(feature_names):
        for j, label_name in enumerate(label_names):
            record = _spearman_pair(features[:, i], labels[:, j])
            records.append({"feature": feature_name, "label": label_name, **record})
    records.sort(key=lambda item: abs(item["spearman_rho"] or 0.0), reverse=True)
    return records


def _spearman_pair(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    """Return an explicit diagnostic for a possibly degenerate pair."""

    left_values = np.asarray(left, dtype=np.float64).reshape(-1)
    right_values = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_values.shape != right_values.shape:
        raise ValueError("Spearman inputs must share one axis")
    valid = np.isfinite(left_values) & np.isfinite(right_values)
    paired_count = int(np.count_nonzero(valid))
    result = {
        "paired_count": paired_count,
        "spearman_rho": None,
        "pvalue_exploratory_unadjusted": None,
    }
    if paired_count < 2:
        return {**result, "status": "not_estimable_fewer_than_two_finite_pairs"}
    left_valid = left_values[valid]
    right_valid = right_values[valid]
    if np.unique(left_valid).size <= 1 or np.unique(right_valid).size <= 1:
        return {**result, "status": "not_estimable_constant_input"}
    rho, pvalue = spearmanr(left_valid, right_valid)
    rho_value = _json_number(float(rho))
    pvalue_value = _json_number(float(pvalue))
    if rho_value is None or pvalue_value is None:
        return {**result, "status": "not_estimable_nonfinite_result"}
    return {
        **result,
        "status": "estimated",
        "spearman_rho": rho_value,
        "pvalue_exploratory_unadjusted": pvalue_value,
    }


def _pearson_pair(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    """Return a finite Pearson diagnostic without constant-input warnings."""

    left_values = np.asarray(left, dtype=np.float64).reshape(-1)
    right_values = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_values.shape != right_values.shape:
        raise ValueError("Pearson inputs must share one axis")
    valid = np.isfinite(left_values) & np.isfinite(right_values)
    paired_count = int(np.count_nonzero(valid))
    result = {"paired_count": paired_count, "pearson_r": None}
    if paired_count < 2:
        return {**result, "status": "not_estimable_fewer_than_two_finite_pairs"}
    left_valid = left_values[valid]
    right_valid = right_values[valid]
    if np.unique(left_valid).size <= 1 or np.unique(right_valid).size <= 1:
        return {**result, "status": "not_estimable_constant_input"}
    coefficient = _json_number(float(np.corrcoef(left_valid, right_valid)[0, 1]))
    if coefficient is None:
        return {**result, "status": "not_estimable_nonfinite_result"}
    return {**result, "status": "estimated", "pearson_r": coefficient}


def _ci_stats_by_label(
    ci: np.ndarray,
    mask: np.ndarray,
    names: list[str],
) -> list[dict[str, Any]]:
    flat_ci = ci.reshape(ci.shape[0], -1, 2)
    flat_mask = mask.reshape(mask.shape[0], -1).astype(bool)
    result: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        available = flat_mask[:, index]
        half_width = (flat_ci[available, index, 1] - flat_ci[available, index, 0]) / 2.0
        stats = _summary(half_width)
        fraction_status = "estimated" if half_width.size else "not_estimable_no_available_rows"
        stats.update(
            {
                "name": name,
                "fraction_status": fraction_status,
                "nonzero_fraction": (
                    float(np.mean(half_width > 0.0)) if half_width.size else None
                ),
                "half_width_ge_0_10_fraction": (
                    float(np.mean(half_width >= 0.10)) if half_width.size else None
                ),
                "half_width_ge_0_25_fraction": (
                    float(np.mean(half_width >= 0.25)) if half_width.size else None
                ),
            }
        )
        result.append(stats)
    return result


def _alignment_strata(values: np.ndarray) -> np.ndarray:
    """Descriptive bins frozen by this audit; not universal physics thresholds."""

    result = np.empty(len(values), dtype="U24")
    result[values < 0.80] = "alignment_lt_0.80"
    result[(values >= 0.80) & (values < 0.90)] = "alignment_0.80_0.90"
    result[(values >= 0.90) & (values < 0.917)] = "alignment_0.90_0.917"
    result[(values >= 0.917) & (values < 0.95)] = "alignment_0.917_0.95"
    result[values >= 0.95] = "alignment_ge_0.95"
    return result


def _stratum_effective_count(
    selected: np.ndarray,
    weights: np.ndarray,
    weight_mask: np.ndarray,
    selection_probability: np.ndarray | None = None,
) -> dict[str, Any]:
    selected = np.asarray(selected, dtype=bool)
    values, mask, probabilities = validate_design_weight_contract(
        weights, weight_mask, selection_probability
    )
    if selected.ndim != 1 or selected.shape != values.shape:
        raise ValueError("selected stratum mask must share the design-weight axis")
    eligible = selected & mask
    active = values[eligible]
    ess = stable_effective_sample_size(active)
    selected_count = int(np.count_nonzero(selected))
    finite_weight_count = int(np.count_nonzero(selected & np.isfinite(values)))
    eligible_count = int(np.count_nonzero(eligible))
    result = {
        "case_count": selected_count,
        "finite_design_weight_count": finite_weight_count,
        "design_weight_eligible_count": eligible_count,
        "eligible_finite_design_weight_count": int(
            np.count_nonzero(eligible & np.isfinite(values))
        ),
        "design_weight_ess": ess,
        "design_weight_ess_status": "estimated" if active.size else "not_estimable_no_eligible_rows",
        "eligible_invalid_design_weight_count": 0,
        "note": "Design-weight ESS is not an event count, class weight, or minibatch probability.",
    }
    if probabilities is not None:
        result.update(
            {
                "finite_selection_probability_count": int(
                    np.count_nonzero(selected & np.isfinite(probabilities))
                ),
                "eligible_finite_selection_probability_count": int(
                    np.count_nonzero(eligible & np.isfinite(probabilities))
                ),
                "eligible_invalid_selection_probability_count": 0,
            }
        )
    return result


def _mechanism_effective_counts(
    *,
    splits: np.ndarray,
    fragment_hit: np.ndarray,
    fragment_transition: np.ndarray,
    shielding: np.ndarray,
    weights: np.ndarray,
    weight_mask: np.ndarray,
    selection_probability: np.ndarray | None = None,
) -> dict[str, Any]:
    layers = {
        "all_cases": np.ones(len(splits), dtype=bool),
        "fragment_hit": fragment_hit,
        "fragment_hit_without_equal_penetration": fragment_transition,
        "shielding_path": shielding,
        "shielding_and_fragment_hit": shielding & fragment_hit,
        "no_shielding_no_fragment_hit": ~shielding & ~fragment_hit,
    }
    result: dict[str, Any] = {}
    split_values = ["all", "train", "validation", "test"]
    for layer_name, layer_mask in layers.items():
        result[layer_name] = {}
        for split in split_values:
            split_mask = np.ones(len(splits), dtype=bool) if split == "all" else splits == split
            result[layer_name][split] = _stratum_effective_count(
                layer_mask & split_mask,
                weights,
                weight_mask,
                selection_probability,
            )
    return result


def _geometry_joint_projection(
    *,
    splits: np.ndarray,
    position: np.ndarray,
    alignment_strata: np.ndarray,
    shielding: np.ndarray,
) -> list[dict[str, Any]]:
    unit = position / np.linalg.norm(position, axis=1, keepdims=True)
    octants = np.asarray(
        [
            f"x{'+' if x >= 0 else '-'}y{'+' if y >= 0 else '-'}z{'+' if z >= 0 else '-'}"
            for x, y, z in unit
        ]
    )
    counts = Counter(
        (str(split), str(octant), str(alignment), bool(panel))
        for split, octant, alignment, panel in zip(
            splits, octants, alignment_strata, shielding
        )
    )
    return [
        {
            "split": split,
            "position_octant": octant,
            "alignment_stratum": alignment,
            "shielding_path": panel,
            "count": count,
        }
        for (split, octant, alignment, panel), count in sorted(counts.items())
    ]


def audit(
    dataset_root: Path,
    *,
    normalization_artifact: Path | None = None,
) -> dict[str, Any]:
    manifest, shard_paths = validated_committed_shard_tensor_paths(dataset_root)
    keys = [
        "case_physics_x",
        "case_physics_feature_mask",
        "component_x",
        "component_feature_mask",
        "armor_x",
        "armor_feature_mask",
        "component_armor_path_x",
        "component_armor_path_mask",
        "component_y",
        "component_label_mask",
        "component_y_ci95",
        "component_aux_y",
        "component_aux_label_mask",
        "component_aux_y_ci95",
        "armor_y",
        "armor_label_mask",
        "armor_y_ci95",
        "system_y",
        "system_label_mask",
        "system_y_ci95",
        "split_names",
        "replicate_counts",
        "sampling_design_weight_raw",
        "sampling_weight_mask",
        "sampling_selection_probability",
    ]
    blocks: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    metadata: dict[str, list[str]] = {}
    for shard_index, path in enumerate(shard_paths):
        with np.load(path, allow_pickle=False) as archive:
            for key in keys:
                blocks[key].append(np.asarray(archive[key]))
            if shard_index == 0:
                for key in [
                    "case_physics_feature_names",
                    "component_feature_names",
                    "armor_feature_names",
                    "armor_path_feature_names",
                    "armor_path_feature_units",
                    "mechanism_names",
                    "state_names",
                    "component_aux_label_names",
                    "armor_label_names",
                    "system_output_names",
                    "component_ids",
                    "armor_ids",
                    "graph_node_ids",
                ]:
                    metadata[key] = [str(value) for value in archive[key].tolist()]
    data = {key: np.concatenate(value, axis=0) for key, value in blocks.items()}
    design_weights, design_weight_mask, selection_probability = (
        validate_design_weight_contract(
            data["sampling_design_weight_raw"],
            data["sampling_weight_mask"],
            data["sampling_selection_probability"],
        )
    )
    case_names = metadata["case_physics_feature_names"]
    system_names = metadata["system_output_names"]
    mechanism_names = metadata["mechanism_names"]
    state_names = metadata["state_names"]
    component_names = [f"{mechanism}.{state}" for mechanism in mechanism_names for state in state_names]

    feature_stats = []
    for index, name in enumerate(case_names):
        available = data["case_physics_feature_mask"][:, index].astype(bool)
        stats = _summary(data["case_physics_x"][available, index])
        stats.update(
            {
                "name": name,
                "available": int(np.sum(available)),
                "availability": float(np.mean(available)),
                "constant_among_available": stats.get("unique", 0) <= 1,
            }
        )
        feature_stats.append(stats)

    inferred_feature_roles = {
        "varying_learnable": [
            item["name"]
            for item in feature_stats
            if item["availability"] == 1.0 and not item["constant_among_available"]
        ],
        "context_fixed": [
            item["name"]
            for item in feature_stats
            if item["availability"] == 1.0 and item["constant_among_available"]
        ],
        "optional_masked": [
            item["name"] for item in feature_stats if item["availability"] < 1.0
        ],
        "derived": [],
    }
    feature_role_audit: dict[str, Any] = {
        "basis": "inferred_from_raw_archive; representation transforms are not applied here",
        "roles": inferred_feature_roles,
        "raw_all_missing_features": [
            item["name"] for item in feature_stats if item["available"] == 0
        ],
        "raw_constant_among_available_features": [
            item["name"]
            for item in feature_stats
            if item["available"] > 0 and item["constant_among_available"]
        ],
        "declared_varying_learnable_violations": None,
    }
    if normalization_artifact is not None:
        normalizer = load_training_normalization_artifact(normalization_artifact)
        fitted = {item["name"]: item for item in normalizer["features"]}
        declared = normalizer["feature_roles"]["varying_learnable"]
        violations = [
            name
            for name in declared
            if fitted[name]["available_count"] != normalizer["fit"]["case_count"]
            or fitted[name]["constant_among_available"] is not False
        ]
        feature_role_audit.update(
            {
                "basis": "validated_train_only_normalization_artifact",
                "normalization_id": normalizer["normalization_id"],
                "roles": normalizer["feature_roles"],
                "declared_varying_learnable_violations": violations,
                "artifact_manifest_hash_matches": (
                    normalizer["source_contracts"]["manifest_hash"]
                    == file_hash(dataset_root / "manifest.json")
                ),
            }
        )

    component_label_stats = _flatten_label_block(
        data["component_y"], data["component_label_mask"], component_names
    )
    component_aux_stats = _flatten_label_block(
        data["component_aux_y"],
        data["component_aux_label_mask"],
        metadata["component_aux_label_names"],
    )
    armor_stats = _flatten_label_block(
        data["armor_y"], data["armor_label_mask"], metadata["armor_label_names"]
    )
    system_stats = _flatten_label_block(
        data["system_y"], data["system_label_mask"], system_names
    )
    armor_path_stats = _flatten_label_block(
        data["component_armor_path_x"],
        data["component_armor_path_mask"],
        metadata["armor_path_feature_names"],
    )

    aux_names = metadata["component_aux_label_names"]
    hit_index = aux_names.index("fragment_hits")
    penetration_index = aux_names.index("fragment_penetrations")
    panel_index = aux_names.index("blast_shielding_panel_count")
    hits = data["component_aux_y"][:, 0, hit_index]
    penetrations = data["component_aux_y"][:, 0, penetration_index]
    shielding_panels = data["component_aux_y"][:, 0, panel_index]
    zero_hits = hits == 0.0
    no_shielding_panel = shielding_panels == 0.0
    qx, qy, qz, qw = [
        data["case_physics_x"][:, index].astype(np.float64)
        for index in range(4, 8)
    ]
    body_x_in_target = np.column_stack(
        [
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy + qz * qw),
            2.0 * (qx * qz - qy * qw),
        ]
    )
    line_of_sight = -data["case_physics_x"][:, :3].astype(np.float64)
    line_of_sight /= np.linalg.norm(line_of_sight, axis=1, keepdims=True)
    cone_axis_los_cosine = np.sum(body_x_in_target * line_of_sight, axis=1)
    alignment_strata = _alignment_strata(cone_axis_los_cosine)
    path_feature_index = {
        name: index for index, name in enumerate(metadata["armor_path_feature_names"])
    }
    if "intersects_path" not in path_feature_index:
        raise ValueError("armor path metadata is missing intersects_path")
    intersects_index = path_feature_index["intersects_path"]
    shielding_path = np.any(
        data["component_armor_path_mask"][..., intersects_index].astype(bool)
        & (data["component_armor_path_x"][..., intersects_index] > 0.5),
        axis=(1, 2),
    )
    detonation_position = data["case_physics_x"][:, :3].astype(np.float64)
    detonation_distance = np.linalg.norm(detonation_position, axis=1)
    system_mode_vector = []
    for j in range(data["system_y"].shape[1]):
        unique, counts = np.unique(data["system_y"][:, j], return_counts=True)
        system_mode_vector.append(float(unique[int(np.argmax(counts))]))
    system_mode_vector_array = np.asarray(system_mode_vector, dtype=np.float32)
    system_at_joint_mode = np.all(data["system_y"] == system_mode_vector_array, axis=1)
    mechanism_index = {name: index for index, name in enumerate(mechanism_names)}
    state_index = {name: index for index, name in enumerate(state_names)}
    component_probabilities = data["component_y"][:, 0]
    legacy_tree_fields_available = all(
        name in system_names for name in ("mission_loss", "mission_degraded")
    ) and all(name in mechanism_index for name in ("combined", "blast")) and all(
        name in state_index for name in ("failed", "degraded")
    )
    aggregate_tree_difference = None
    if legacy_tree_fields_available:
        naive_degraded_from_aggregate_leaves = np.maximum(
            component_probabilities[
                :, mechanism_index["combined"], state_index["failed"]
            ],
            component_probabilities[
                :, mechanism_index["blast"], state_index["degraded"]
            ],
        )
        aggregate_tree_difference = np.abs(
            data["system_y"][:, system_names.index("mission_degraded")]
            - naive_degraded_from_aggregate_leaves
        )
    armor_label_index = {name: index for index, name in enumerate(metadata["armor_label_names"])}
    armor_integrity_names = [
        "post_fragment_integrity",
        "post_combined_integrity",
        "blast_pre_integrity",
        "blast_post_integrity",
    ]
    armor_integrity = data["armor_y"][:, 0, [armor_label_index[name] for name in armor_integrity_names]]

    ci_label_names = {
        "component": component_names,
        "component_aux": metadata["component_aux_label_names"],
        "armor": metadata["armor_label_names"],
        "system": system_names,
    }
    ci_blocks = {}
    for prefix in ["component", "component_aux", "armor", "system"]:
        ci = data[f"{prefix}_y_ci95"]
        mask = data[f"{prefix}_label_mask"].astype(bool)
        half_width = (ci[..., 1] - ci[..., 0]) / 2.0
        values = half_width[mask]
        comparable = prefix in {"component", "system"}
        aggregate = None
        if comparable:
            aggregate = {
                **_summary(values),
                "nonzero_fraction": float(np.mean(values > 0.0)),
                "half_width_ge_0_10_fraction": float(np.mean(values >= 0.10)),
                "half_width_ge_0_25_fraction": float(np.mean(values >= 0.25)),
            }
        ci_blocks[prefix] = {
            "cross_label_aggregation_valid": comparable,
            "aggregate": aggregate,
            "per_label": _ci_stats_by_label(ci, mask, ci_label_names[prefix]),
            "note": (
                "All labels in this block are dimensionless probabilities."
                if comparable
                else "Labels use mixed physical units; cross-label CI aggregation is intentionally omitted."
            ),
        }

    splits = data["split_names"].astype(str)
    mechanism_effective_counts = _mechanism_effective_counts(
        splits=splits,
        fragment_hit=hits > 0.0,
        fragment_transition=(hits > 0.0) & (hits != penetrations),
        shielding=shielding_path,
        weights=design_weights,
        weight_mask=design_weight_mask,
        selection_probability=selection_probability,
    )
    geometry_strata = {
        "definitions": {
            "alignment": "dot(rotated body +X, normalized component-center minus detonation-position); current single component is centered at origin",
            "alignment_threshold_status": "descriptive_for_current_demo_only",
            "shielding": "any available component_armor_path.intersects_path > 0.5",
            "distance": "norm(detonation_position_m)",
        },
        "alignment_counts": dict(sorted(Counter(map(str, alignment_strata)).items())),
        "alignment_counts_by_split": {
            split: dict(
                sorted(Counter(map(str, alignment_strata[splits == split])).items())
            )
            for split in ("train", "validation", "test")
        },
        "shielding_path_count": int(np.sum(shielding_path)),
        "shielding_path_count_by_split": {
            split: int(np.sum(shielding_path & (splits == split)))
            for split in ("train", "validation", "test")
        },
        "detonation_distance_m": _summary(detonation_distance),
        "joint_projection": _geometry_joint_projection(
            splits=splits,
            position=detonation_position,
            alignment_strata=alignment_strata,
            shielding=shielding_path,
        ),
    }
    system_pair_spearman = (
        _spearman_pair(data["system_y"][:, 0], data["system_y"][:, 1])
        if data["system_y"].shape[1] >= 2
        else {
            "status": "not_estimable_fewer_than_two_system_outputs",
            "paired_count": 0,
            "spearman_rho": None,
            "pvalue_exploratory_unadjusted": None,
        }
    )
    fragment_hit_penetration_pearson = _pearson_pair(hits, penetrations)
    report = {
        "schema_version": "1.0.0",
        "scope": {
            "dataset_root": str(dataset_root.resolve()),
            "dataset_id": manifest.get("dataset_id"),
            "case_count": int(data["case_physics_x"].shape[0]),
            "shard_count": len(shard_paths),
            "splits": {
                name: int(np.sum(splits == name))
                for name in ["train", "validation", "test"]
            },
            "component_count": len(metadata["component_ids"]),
            "armor_count": len(metadata["armor_ids"]),
            "graph_node_count": len(metadata["graph_node_ids"]),
            "replicate_counts": _summary(data["replicate_counts"]),
        },
        "input_space": {
            "case_physics_feature_count": len(case_names),
            "available_feature_count": int(
                sum(item["availability"] > 0.0 for item in feature_stats)
            ),
            "variable_raw_coordinate_count": int(
                sum(not item["constant_among_available"] for item in feature_stats)
            ),
            "declared_intrinsic_design_dimension": 5,
            "intrinsic_dimension_basis": "2D sphere-shell direction plus 3D SO(3) orientation",
            "features": feature_stats,
            "feature_role_audit": feature_role_audit,
            "sphere_shell": _sphere_coverage(data["case_physics_x"]),
            "geometry_strata_and_joint_projection": geometry_strata,
            "quaternion_representation": {
                "qw_nonnegative_fraction": float(np.mean(data["case_physics_x"][:, 7] >= 0.0)),
                "abs_qw_lt_0_05_fraction": float(np.mean(np.abs(data["case_physics_x"][:, 7]) < 0.05)),
                "abs_qw_lt_0_10_fraction": float(np.mean(np.abs(data["case_physics_x"][:, 7]) < 0.10)),
                "note": "The archive uses a qw>=0 hemisphere convention; the representation seam remains near 180-degree rotations.",
            },
            "split_drift_exploratory": _split_drift(
                data["case_physics_x"],
                data["case_physics_feature_mask"],
                case_names,
                splits,
            ),
            "nearest_train_coverage": [
                _nearest_reference_coverage(data["case_physics_x"], splits, "validation"),
                _nearest_reference_coverage(data["case_physics_x"], splits, "test"),
            ],
            "target_feature_variation": {
                "component_feature_coordinates": int(data["component_x"].shape[-1]),
                "component_variable_coordinates": int(
                    np.sum(np.ptp(data["component_x"].reshape(data["component_x"].shape[0], -1), axis=0) > 0)
                ),
                "armor_feature_coordinates": int(data["armor_x"].shape[-1]),
                "armor_variable_coordinates": int(
                    np.sum(np.ptp(data["armor_x"].reshape(data["armor_x"].shape[0], -1), axis=0) > 0)
                ),
                "armor_path_feature_coordinates": int(data["component_armor_path_x"].shape[-1]),
                "armor_path_variable_coordinates": int(
                    sum(item.get("unique", 0) > 1 for item in armor_path_stats)
                ),
                "armor_path_features": armor_path_stats,
                "component_ids": metadata["component_ids"],
                "armor_ids": metadata["armor_ids"],
            },
        },
        "labels": {
            "system": system_stats,
            "component_probability": component_label_stats,
            "component_auxiliary": component_aux_stats,
            "armor": armor_stats,
            "system_joint_unique_rows": int(np.unique(data["system_y"], axis=0).shape[0]),
            "system_joint_mode_vector": system_mode_vector,
            "system_joint_mode_fraction": float(np.mean(system_at_joint_mode)),
            "system_interval_fractions": {
                name: _interval_fractions(data["system_y"][:, index])
                for index, name in enumerate(system_names)
            },
            "system_label_spearman": system_pair_spearman["spearman_rho"],
            "system_label_spearman_status": system_pair_spearman["status"],
            "system_label_spearman_paired_count": system_pair_spearman[
                "paired_count"
            ],
            "input_to_system_spearman_exploratory": _spearman_matrix(
                data["case_physics_x"], case_names, data["system_y"], system_names
            ),
            "plateau_trace": {
                "zero_fragment_hits_fraction": float(np.mean(zero_hits)),
                "positive_fragment_hits_count": int(np.sum(~zero_hits)),
                "zero_fragment_penetrations_fraction": float(np.mean(penetrations == 0.0)),
                "no_shielding_panel_fraction": float(np.mean(no_shielding_panel)),
                "joint_system_mode_fraction": float(np.mean(system_at_joint_mode)),
                "joint_mode_given_zero_hits_fraction": (
                    float(np.mean(system_at_joint_mode[zero_hits]))
                    if np.any(zero_hits)
                    else None
                ),
                "zero_hits_given_joint_mode_fraction": (
                    float(np.mean(zero_hits[system_at_joint_mode]))
                    if np.any(system_at_joint_mode)
                    else None
                ),
                "joint_mode_given_no_shielding_panel_fraction": (
                    float(np.mean(system_at_joint_mode[no_shielding_panel]))
                    if np.any(no_shielding_panel)
                    else None
                ),
                "no_shielding_panel_given_joint_mode_fraction": (
                    float(np.mean(no_shielding_panel[system_at_joint_mode]))
                    if np.any(system_at_joint_mode)
                    else None
                ),
                "joint_mode_is_exactly_no_panel_and_zero_hits": bool(
                    np.array_equal(system_at_joint_mode, no_shielding_panel & zero_hits)
                ),
                "cone_axis_los_cosine": {
                    "all_cases": _summary(cone_axis_los_cosine),
                    "positive_fragment_hits": _summary(cone_axis_los_cosine[~zero_hits]),
                    "zero_fragment_hits": _summary(cone_axis_los_cosine[zero_hits]),
                    "definition": "dot(rotated body +X fragment-cone axis, detonation-to-component line of sight)",
                },
                "system_summary_zero_hits": [
                    {"name": name, **_summary(data["system_y"][zero_hits, j])}
                    for j, name in enumerate(system_names)
                ],
                "system_summary_positive_hits": [
                    {"name": name, **_summary(data["system_y"][~zero_hits, j])}
                    for j, name in enumerate(system_names)
                ],
                "event_counts_by_split": {
                    split_name: {
                        "count": int(np.sum(splits == split_name)),
                        "no_panel_no_hit": int(
                            np.sum((splits == split_name) & no_shielding_panel & zero_hits)
                        ),
                        "no_panel_positive_hit": int(
                            np.sum((splits == split_name) & no_shielding_panel & ~zero_hits)
                        ),
                        "panel_no_hit": int(
                            np.sum((splits == split_name) & ~no_shielding_panel & zero_hits)
                        ),
                        "panel_positive_hit": int(
                            np.sum((splits == split_name) & ~no_shielding_panel & ~zero_hits)
                        ),
                    }
                    for split_name in ["train", "validation", "test"]
                },
            },
            "structural_relations": {
                "legacy_mission_relation_status": (
                    "estimated"
                    if legacy_tree_fields_available
                    else "not_estimable_required_output_vocabulary_missing"
                ),
                "legacy_mission_required_outputs": [
                    "mission_loss",
                    "mission_degraded",
                ],
                "mission_loss_equals_component_combined_failed_fraction": (
                    float(
                        np.mean(
                            data["system_y"][:, system_names.index("mission_loss")]
                            == component_probabilities[
                                :, mechanism_index["combined"], state_index["failed"]
                            ]
                        )
                    )
                    if legacy_tree_fields_available
                    else None
                ),
                "mission_loss_leq_mission_degraded_fraction": (
                    float(
                        np.mean(
                            data["system_y"][:, system_names.index("mission_loss")]
                            <= data["system_y"][:, system_names.index("mission_degraded")]
                        )
                    )
                    if legacy_tree_fields_available
                    else None
                ),
                "component_failed_leq_degraded_fraction_by_mechanism": {
                    name: float(
                        np.mean(
                            component_probabilities[:, index, state_index["failed"]]
                            <= component_probabilities[:, index, state_index["degraded"]]
                        )
                    )
                    for index, name in enumerate(mechanism_names)
                },
                "fragment_hits_equal_penetrations_fraction": float(np.mean(hits == penetrations)),
                "fragment_hits_penetrations_pearson": (
                    fragment_hit_penetration_pearson["pearson_r"]
                ),
                "fragment_hits_penetrations_pearson_status": (
                    fragment_hit_penetration_pearson["status"]
                ),
                "fragment_hits_penetrations_pearson_paired_count": (
                    fragment_hit_penetration_pearson["paired_count"]
                ),
                "armor_integrity_channels": armor_integrity_names,
                "armor_integrity_channels_equal_per_case_fraction": float(
                    np.mean(np.all(armor_integrity == armor_integrity[:, :1], axis=1))
                ),
                "naive_tree_on_aggregate_leaf_means": {
                    "status": (
                        "estimated"
                        if aggregate_tree_difference is not None
                        else "not_estimable_required_output_vocabulary_missing"
                    ),
                    "difference_count": (
                        int(np.sum(aggregate_tree_difference > 1e-12))
                        if aggregate_tree_difference is not None
                        else None
                    ),
                    "maximum_absolute_difference": (
                        float(np.max(aggregate_tree_difference))
                        if aggregate_tree_difference is not None
                        else None
                    ),
                    "note": "Compares mean(per-replicate fully-dependent OR) with max(aggregate combined.failed, aggregate blast.degraded).",
                },
            },
            "ci_half_width": ci_blocks,
            "mechanism_layer_effective_counts": mechanism_effective_counts,
            "design_weight_contract": _stratum_effective_count(
                np.ones(design_weights.shape, dtype=bool),
                design_weights,
                design_weight_mask,
                selection_probability,
            ),
        },
        "interpretation_guards": {
            "confirmatory_status": "exploratory_descriptive_audit",
            "multiple_comparison_note": "Reported Spearman and KS p-values are unadjusted exploratory diagnostics; do not use them as admission gates.",
            "scientific_boundary": "Archive integrity and distribution diagnostics do not establish physical validity, operational representativeness, or research-training admission.",
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a completed sharded dataset distribution")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--normalization-artifact",
        type=Path,
        help="optional validated Train-only normalization artifact for feature-role gates",
    )
    args = parser.parse_args()
    report = audit(
        args.dataset_root,
        normalization_artifact=args.normalization_artifact,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "case_count": report["scope"]["case_count"]}))


if __name__ == "__main__":
    main()
