from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SPLITS = ("train", "validation", "test")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: Any) -> float | int | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return int(value) if value.is_integer() else value


def _quantiles(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {key: None for key in ("min", "p05", "median", "p95", "max", "mean", "std")}
    q = np.quantile(values, [0.0, 0.05, 0.5, 0.95, 1.0])
    return {
        "min": _number(q[0]), "p05": _number(q[1]), "median": _number(q[2]),
        "p95": _number(q[3]), "max": _number(q[4]),
        "mean": _number(np.mean(values)), "std": _number(np.std(values)),
    }


def _histogram(values: np.ndarray, bins: int = 20) -> dict[str, list[float | int]]:
    counts, edges = np.histogram(np.asarray(values, dtype=np.float64), bins=np.linspace(0.0, 1.0, bins + 1))
    return {"counts": counts.astype(int).tolist(), "edges": edges.astype(float).tolist()}


def _max_split_smd(values: np.ndarray, valid: np.ndarray, splits: np.ndarray) -> float | None:
    maximum = 0.0
    found = False
    for left, right in combinations(SPLITS, 2):
        a = values[valid & (splits == left)]
        b = values[valid & (splits == right)]
        if not len(a) or not len(b):
            continue
        scale = math.sqrt((float(np.var(a)) + float(np.var(b))) / 2.0)
        smd = abs(float(np.mean(a)) - float(np.mean(b))) / scale if scale > 0 else 0.0
        maximum = max(maximum, smd)
        found = True
    return maximum if found else None


def _model_input_binding(manifest: dict[str, Any], tensors: Any) -> dict[str, Any]:
    """Choose the learner-visible tensor, preferring an explicit 1.8 contract."""

    declared = manifest.get("model_input_contract")
    if isinstance(declared, dict):
        tensor_field = declared.get("tensor_field")
        mask_field = declared.get("mask_field")
        if (
            isinstance(tensor_field, str)
            and isinstance(mask_field, str)
            and tensor_field in tensors.files
            and mask_field in tensors.files
        ):
            features = declared.get("features")
            if not isinstance(features, list) or not features:
                raise ValueError("declared model_input_contract has no feature definitions")
            names = [str(item.get("name")) for item in features]
            units = [str(item.get("unit")) for item in features]
            if len(names) != np.asarray(tensors[tensor_field]).shape[1]:
                raise ValueError("model_input_contract feature axis differs from archived tensor")
            return {
                "kind": "uav_control" if tensor_field == "uav_state_x" else "declared",
                "contract": dict(declared),
                "tensor_field": tensor_field,
                "mask_field": mask_field,
                "names": names,
                "units": units,
                "features": [dict(item) for item in features],
            }

    # Tensor contracts <=1.7 did not declare a model-facing field. Preserve the
    # previous report semantics instead of silently reinterpreting old archives.
    for tensor_field, mask_field, names_field, units_field in (
        (
            "case_physics_x",
            "case_physics_feature_mask",
            "case_physics_feature_names",
            "case_physics_feature_units",
        ),
        ("case_x", "case_feature_mask", "case_feature_names", "case_feature_units"),
    ):
        if all(name in tensors.files for name in (tensor_field, mask_field, names_field, units_field)):
            names = np.asarray(tensors[names_field]).astype(str).tolist()
            units = np.asarray(tensors[units_field]).astype(str).tolist()
            return {
                "kind": "legacy",
                "contract": None,
                "tensor_field": tensor_field,
                "mask_field": mask_field,
                "names": names,
                "units": units,
                "features": [
                    {
                        "name": name,
                        "unit": unit,
                        "semantic": "Legacy archive feature; consult its tensor contract.",
                        "source_path": None,
                        "transformation": "legacy_archive_value",
                    }
                    for name, unit in zip(names, units, strict=True)
                ],
            }
    raise ValueError("archive has no supported learner-visible input tensor")


def _load_case_level_arrays(
    shards: list[Path], manifest: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    if not shards:
        raise ValueError("completed dataset has no shards")
    with np.load(shards[0] / "tensors.npz", allow_pickle=False) as first:
        binding = _model_input_binding(manifest, first)
        available = set(first.files)

    required_fields = (
        binding["tensor_field"],
        binding["mask_field"],
        "split_names",
        "system_y",
        "system_label_mask",
        "system_source_available_mask",
        "system_uncertainty_mask",
        "system_y_ci95",
        "replicate_counts",
        "label_task_available_mask",
        "sampling_selection_probability",
        "sampling_design_weight_raw",
        "sampling_weight_mask",
    )
    optional_fields = (
        "sampling_round",
        "sampling_role_index",
        "sampling_stratum_index",
        "component_aux_y",
        "component_aux_label_mask",
        "component_armor_path_x",
        "component_armor_path_mask",
    )
    fields = required_fields + tuple(name for name in optional_fields if name in available)
    missing = sorted(set(required_fields) - available)
    if missing:
        raise ValueError(f"archive lacks required report tensors: {missing}")
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in fields}
    static: dict[str, np.ndarray] = {}
    static_names = tuple(
        name
        for name in (
            "system_output_names",
            "label_task_names",
            "sampling_role_names",
            "sampling_stratum_names",
            "component_aux_label_names",
            "armor_path_feature_names",
        )
        if name in available
    )
    for index, shard in enumerate(shards):
        with np.load(shard / "tensors.npz", allow_pickle=False) as tensors:
            absent = sorted(set(fields) - set(tensors.files))
            if absent:
                raise ValueError(f"shard {shard.name} lacks report tensors: {absent}")
            for name in fields:
                chunks[name].append(np.asarray(tensors[name]))
            if index == 0:
                for name in static_names:
                    static[name] = np.asarray(tensors[name]).copy()
    arrays = {name: np.concatenate(values, axis=0) for name, values in chunks.items()}
    return arrays, static, binding


def _feature_rows(
    x: np.ndarray,
    mask: np.ndarray,
    splits: np.ndarray,
    names: list[str],
    units: list[str],
    definitions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    features = []
    for index, (name, unit) in enumerate(zip(names, units)):
        valid = mask[:, index] & np.isfinite(x[:, index])
        values = x[valid, index]
        unique = int(len(np.unique(values))) if len(values) else 0
        definition = definitions[index] if definitions and index < len(definitions) else {}
        features.append({
            "name": str(name), "unit": str(unit), "available": int(np.sum(valid)),
            "availability": float(np.mean(valid)), "unique": unique,
            "unique_fraction": float(unique / len(values)) if len(values) else 0.0,
            "constant": unique <= 1, "stats": _quantiles(values),
            "max_abs_split_smd": _number(_max_split_smd(x[:, index], valid, splits)),
            "semantic": definition.get("semantic"),
            "source_path": definition.get("source_path"),
            "transformation": definition.get("transformation"),
        })
    return features


def _vector_checks(
    x: np.ndarray, mask: np.ndarray, binding: dict[str, Any]
) -> list[dict[str, Any]]:
    names = list(binding["names"])
    checks: list[dict[str, Any]] = []

    def add(identifier: str, label: str, indices: list[int], unit: str) -> None:
        valid = np.all(mask[:, indices], axis=1) & np.all(np.isfinite(x[:, indices]), axis=1)
        values = np.linalg.norm(x[valid][:, indices], axis=1)
        checks.append({"id": identifier, "label": label, "unit": unit, "stats": _quantiles(values)})

    if binding["kind"] == "uav_control" and len(names) == 10:
        add("relative_position_norm", "UAV 与固定威胁的相对距离", [0, 1, 2], "m")
        add("relative_speed_norm", "UAV 相对固定威胁速度模", [3, 4, 5], "m/s")
        add("quaternion_norm", "姿态四元数范数", [6, 7, 8, 9], "1")
        return checks

    lookup = {name: index for index, name in enumerate(names)}
    detonation_names = ("detonation_x_m", "detonation_y_m", "detonation_z_m")
    quaternion_names = ("orientation_qx", "orientation_qy", "orientation_qz", "orientation_qw")
    if all(name in lookup for name in detonation_names):
        add("detonation_radius", "起爆位置半径", [lookup[name] for name in detonation_names], "m")
    if all(name in lookup for name in quaternion_names):
        add("quaternion_norm", "姿态四元数范数", [lookup[name] for name in quaternion_names], "1")
    return checks


def _training_view_report(
    arrays: dict[str, np.ndarray], binding: dict[str, Any]
) -> dict[str, Any]:
    from damage_engine.training_view import transform_case_physics, transform_uav_state

    x = arrays[binding["tensor_field"]]
    mask = arrays[binding["mask_field"]]
    splits = arrays["split_names"].astype(str)
    transformer = transform_uav_state if binding["kind"] == "uav_control" else transform_case_physics
    transformed: list[np.ndarray] = []
    transformed_masks: list[np.ndarray] = []
    output_names: list[str] | None = None
    output_units: list[str] | None = None
    transforms: list[dict[str, Any]] | None = None
    try:
        for values, available in zip(x, mask, strict=True):
            row, row_mask, names, units, row_transforms = transformer(
                values, available, binding["names"], binding["units"]
            )
            if output_names is None:
                output_names = list(names)
                output_units = list(units)
                transforms = list(row_transforms)
            elif list(names) != output_names or list(units) != output_units:
                raise ValueError("training transform schema changed between rows")
            transformed.append(np.asarray(row, dtype=np.float64))
            transformed_masks.append(np.asarray(row_mask, dtype=bool))
    except Exception as exc:
        if binding["kind"] == "uav_control":
            raise
        return {
            "available": False,
            "reason": f"legacy training transform unavailable: {exc}",
            "source_tensor_field": binding["tensor_field"],
        }
    values = np.stack(transformed)
    available = np.stack(transformed_masks)
    assert output_names is not None and output_units is not None
    definitions = [
        {
            "semantic": "Learner-facing transformed feature.",
            "source_path": None,
            "transformation": item.get("kind") if isinstance(item, dict) else str(item),
        }
        for item in (transforms or [{} for _ in output_names])
    ]
    return {
        "available": True,
        "tensor_field": "training_x",
        "source_tensor_field": binding["tensor_field"],
        "feature_count": len(output_names),
        "feature_names": output_names,
        "feature_units": output_units,
        "features": _feature_rows(
            values, available, splits, output_names, output_units, definitions
        ),
        "transforms": transforms or [],
        "normalization_boundary": "fit on Train only; validation/test apply the frozen artifact",
    }


def _feature_report(
    arrays: dict[str, np.ndarray], binding: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    x = arrays[binding["tensor_field"]]
    mask = arrays[binding["mask_field"]]
    splits = arrays["split_names"].astype(str)
    features = _feature_rows(
        x,
        mask,
        splits,
        list(binding["names"]),
        list(binding["units"]),
        list(binding["features"]),
    )
    derived = {
        "variable_feature_count": sum(not item["constant"] for item in features),
        "constant_feature_count": sum(item["constant"] for item in features),
        "feature_count": len(features),
        "variable_feature_names": [item["name"] for item in features if not item["constant"]],
        "vector_checks": _vector_checks(x, mask, binding),
    }
    return features, derived, _training_view_report(arrays, binding)


def _label_report(arrays: dict[str, np.ndarray], static: dict[str, np.ndarray]) -> list[dict]:
    outputs = static["system_output_names"].astype(str)
    values = arrays["system_y"]
    masks = arrays["system_label_mask"]
    uncertainty = arrays["system_uncertainty_mask"]
    intervals = arrays["system_y_ci95"]
    splits = arrays["split_names"].astype(str)
    rows = []
    for index, name in enumerate(outputs):
        valid = masks[:, index] & np.isfinite(values[:, index])
        selected = values[valid, index].astype(np.float64)
        unique, counts = np.unique(selected, return_counts=True)
        mode_index = int(np.argmax(counts)) if len(counts) else 0
        clipped = np.clip(selected, 1.0e-12, 1.0 - 1.0e-12)
        entropy = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))
        usable_ci = valid & uncertainty[:, index]
        half_width = (intervals[usable_ci, index, 1] - intervals[usable_ci, index, 0]) / 2.0
        split_stats = {}
        for split in SPLITS:
            subset = valid & (splits == split)
            split_stats[split] = {"count": int(np.sum(subset)), **_quantiles(values[subset, index])}
        rows.append({
            "name": str(name), "available": int(np.sum(valid)), "availability": float(np.mean(valid)),
            "stats": _quantiles(selected), "unique": int(len(unique)),
            "mode_value": _number(unique[mode_index]) if len(unique) else None,
            "mode_count": int(counts[mode_index]) if len(counts) else 0,
            "mode_fraction": float(counts[mode_index] / len(selected)) if len(selected) else 0.0,
            "exact_zero_fraction": float(np.mean(selected == 0.0)) if len(selected) else 0.0,
            "exact_one_fraction": float(np.mean(selected == 1.0)) if len(selected) else 0.0,
            "mean_bernoulli_entropy_nats": _number(np.mean(entropy)) if len(entropy) else None,
            "histogram": _histogram(selected), "split": split_stats,
            "ci_available": int(np.sum(usable_ci)), "ci_availability": float(np.mean(usable_ci)),
            "ci_half_width": _quantiles(half_width),
            "ci_nonzero_fraction": float(np.mean(half_width > 0)) if len(half_width) else 0.0,
            "ci_half_width_ge_025_fraction": float(np.mean(half_width >= 0.25)) if len(half_width) else 0.0,
        })
    return rows


def _duplicate_report(
    arrays: dict[str, np.ndarray], binding: dict[str, Any]
) -> dict[str, int]:
    registry: dict[str, str] = {}
    within = 0
    cross = 0
    for values, mask, split in zip(
        arrays[binding["tensor_field"]],
        arrays[binding["mask_field"]],
        arrays["split_names"].astype(str),
    ):
        digest = hashlib.sha256(values.tobytes() + mask.tobytes()).hexdigest()
        previous = registry.get(digest)
        if previous is None:
            registry[digest] = str(split)
        elif previous == split:
            within += 1
        else:
            cross += 1
    return {"unique_input_fingerprints": len(registry), "within_split_duplicates": within, "cross_split_duplicates": cross}


def _weight_report(weights: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    values = np.asarray(weights, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if not len(values):
        return {
            "eligible_n": 0,
            "ineligible_n": int(len(weights)),
            "ess": 0.0,
            "ess_fraction": None,
            "min": None,
            "median": None,
            "p95": None,
            "max": None,
            "cv": None,
        }
    total = float(np.sum(values))
    ess = total * total / float(np.sum(values * values))
    quantiles = np.quantile(values, [0.0, 0.5, 0.95, 1.0])
    mean = float(np.mean(values))
    return {
        "eligible_n": int(len(values)),
        "ineligible_n": int(len(weights) - len(values)),
        "ess": ess,
        "ess_fraction": ess / len(values),
        "min": _number(quantiles[0]),
        "median": _number(quantiles[1]),
        "p95": _number(quantiles[2]),
        "max": _number(quantiles[3]),
        "cv": _number(np.std(values) / mean) if mean else None,
    }


def _event_arrays(
    arrays: dict[str, np.ndarray], static: dict[str, np.ndarray]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    count = len(arrays["split_names"])
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    hits = np.zeros(count, dtype=np.float64)
    penetrations = np.zeros(count, dtype=np.float64)
    hit_available = np.zeros(count, dtype=bool)
    penetration_available = np.zeros(count, dtype=bool)
    if all(
        name in arrays for name in ("component_aux_y", "component_aux_label_mask")
    ) and "component_aux_label_names" in static:
        names = static["component_aux_label_names"].astype(str).tolist()
        values = np.asarray(arrays["component_aux_y"], dtype=np.float64)
        mask = np.asarray(arrays["component_aux_label_mask"], dtype=bool)
        if "fragment_hits" in names:
            index = names.index("fragment_hits")
            hit_available = np.all(mask[:, :, index], axis=1)
            hits = np.sum(np.where(mask[:, :, index], values[:, :, index], 0.0), axis=1)
        if "fragment_penetrations" in names:
            index = names.index("fragment_penetrations")
            penetration_available = np.all(mask[:, :, index], axis=1)
            penetrations = np.sum(
                np.where(mask[:, :, index], values[:, :, index], 0.0), axis=1
            )
    tolerance = 1.0e-9 * np.maximum(1.0, np.abs(hits))
    any_hit = hits > tolerance
    any_penetration = penetrations > tolerance
    not_all = hits - penetrations > tolerance
    both = hit_available & penetration_available
    result["fragment_hit"] = (any_hit, hit_available)
    result["hit_but_not_all_penetrated"] = (not_all, both)
    result["partial_penetration"] = (any_penetration & not_all, both)
    result["full_penetration"] = (any_hit & any_penetration & ~not_all, both)

    occluded = np.zeros(count, dtype=bool)
    occlusion_available = np.zeros(count, dtype=bool)
    if all(
        name in arrays
        for name in ("component_armor_path_x", "component_armor_path_mask")
    ) and "armor_path_feature_names" in static:
        names = static["armor_path_feature_names"].astype(str).tolist()
        if "intersects_path" in names:
            index = names.index("intersects_path")
            values = np.asarray(arrays["component_armor_path_x"], dtype=np.float64)[
                ..., index
            ]
            mask = np.asarray(arrays["component_armor_path_mask"], dtype=bool)[
                ..., index
            ]
            occlusion_available = np.all(mask, axis=(1, 2))
            occluded = np.any(mask & (values > 0.5), axis=(1, 2))
    result["occluded_and_fragment_hit"] = (
        occluded & any_hit,
        occlusion_available & hit_available,
    )
    return result


def _event_report(
    event: np.ndarray,
    available: np.ndarray,
    weights: np.ndarray,
    weight_mask: np.ndarray,
) -> dict[str, Any]:
    available_n = int(np.sum(available))
    if not available_n:
        return {
            "available": False,
            "available_n": 0,
            "unavailable_n": int(len(event)),
            "count": None,
            "rate": None,
            "weighted_rate": None,
        }
    count = int(np.sum(event & available))
    selected = available & weight_mask
    denominator = float(np.sum(weights[selected]))
    numerator = float(np.sum(weights[selected & event]))
    return {
        "available": True,
        "available_n": available_n,
        "unavailable_n": int(len(event) - available_n),
        "count": count,
        "rate": count / available_n,
        "weighted_rate": numerator / denominator if denominator > 0.0 else None,
    }


def _role_category(role: str) -> str:
    if role.startswith("reference_"):
        return "reference"
    if any(token in role for token in ("target", "stratified", "rare", "adaptive")):
        return "targeted"
    return "other"


def _role_report(
    arrays: dict[str, np.ndarray], static: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    count = len(arrays["split_names"])
    if "sampling_role_index" in arrays and "sampling_role_names" in static:
        roles = static["sampling_role_names"].astype(str)[
            np.asarray(arrays["sampling_role_index"], dtype=np.int64)
        ]
    else:
        roles = np.asarray(["all"] * count)
    if "sampling_stratum_index" in arrays and "sampling_stratum_names" in static:
        strata = static["sampling_stratum_names"].astype(str)[
            np.asarray(arrays["sampling_stratum_index"], dtype=np.int64)
        ]
    else:
        strata = np.asarray(["not_declared"] * count)
    splits = arrays["split_names"].astype(str)
    weights = np.asarray(arrays["sampling_design_weight_raw"], dtype=np.float64)
    weight_mask = np.asarray(arrays["sampling_weight_mask"], dtype=bool)
    events = _event_arrays(arrays, static)
    output_names = static["system_output_names"].astype(str)
    outputs = np.asarray(arrays["system_y"], dtype=np.float64)
    output_mask = np.asarray(arrays["system_label_mask"], dtype=bool)
    uncertainty = np.asarray(arrays["system_uncertainty_mask"], dtype=bool)
    intervals = np.asarray(arrays["system_y_ci95"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for role in sorted(set(map(str, roles))):
        selected = roles == role
        role_events = {
            name: _event_report(
                event[selected],
                available[selected],
                weights[selected],
                weight_mask[selected],
            )
            for name, (event, available) in events.items()
        }
        labels = []
        for index, name in enumerate(output_names):
            valid = selected & output_mask[:, index] & np.isfinite(outputs[:, index])
            ci_valid = valid & uncertainty[:, index]
            half_width = (
                intervals[ci_valid, index, 1] - intervals[ci_valid, index, 0]
            ) / 2.0
            labels.append(
                {
                    "name": str(name),
                    "available": int(np.sum(valid)),
                    "mean_probability": _number(np.mean(outputs[valid, index]))
                    if np.any(valid)
                    else None,
                    "expected_event_mass": _number(np.sum(outputs[valid, index]))
                    if np.any(valid)
                    else None,
                    "ci_half_width": _quantiles(half_width),
                }
            )
        rows.append(
            {
                "role": role,
                "category": _role_category(role),
                "n": int(np.sum(selected)),
                "splits": {
                    split: int(np.sum(selected & (splits == split))) for split in SPLITS
                },
                "strata": {
                    stratum: int(np.sum(selected & (strata == stratum)))
                    for stratum in sorted(set(map(str, strata[selected])))
                },
                "weights": _weight_report(weights[selected], weight_mask[selected]),
                "events": role_events,
                "system_labels": labels,
            }
        )
    return rows


def _timing_report(timing: dict[str, Any] | None) -> dict[str, Any]:
    if timing is None:
        return {"provided": False, "metrics": [], "raw": None}
    metrics: list[dict[str, Any]] = []

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, (*path, str(key)))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            name = ".".join(path)
            lowered = name.lower()
            if any(
                token in lowered
                for token in (
                    "second",
                    "minute",
                    "hour",
                    "cases_per_second",
                    "bytes",
                    "mib",
                    "gib",
                )
            ):
                metrics.append({"name": name, "value": _number(value)})

    visit(timing, ())
    return {"provided": True, "metrics": metrics, "raw": timing}


def build_report(dataset_root: Path, timing_path: Path | None) -> dict[str, Any]:
    from damage_engine.torch_dataset import TorchShardedDamageDataset

    verified = TorchShardedDamageDataset(dataset_root, purpose="software_development", max_cached_shards=2)
    manifest = verified.manifest
    checkpoint = verified.checkpoint
    shards = list(verified.shard_paths)
    arrays, static, binding = _load_case_level_arrays(shards, manifest)
    features, derived, training_view = _feature_report(arrays, binding)
    labels = _label_report(arrays, static)
    sampling_root = dataset_root.parent
    sampling_checkpoint = _json(sampling_root / "sampling_checkpoint.json")
    sampling_audit = _json(sampling_root / "audits" / "sampling_quality.json")
    sampling_spec = _json(sampling_root / "plan" / "sampling_spec.json")
    timing_raw = _json(timing_path) if timing_path and timing_path.exists() else None
    disk_files = [path for path in sampling_root.rglob("*") if path.is_file()]
    disk_bytes = sum(path.stat().st_size for path in disk_files)
    splits = arrays["split_names"].astype(str)
    split_counts = {name: int(np.sum(splits == name)) for name in SPLITS}
    eligible = arrays["sampling_weight_mask"].astype(bool)
    weights = arrays["sampling_design_weight_raw"][eligible]
    rho = arrays["sampling_selection_probability"][eligible]
    overall_weights = _weight_report(
        arrays["sampling_design_weight_raw"], arrays["sampling_weight_mask"]
    )
    masks = {
        "model_input_feature_availability": float(
            np.mean(arrays[binding["mask_field"]])
        ),
        "system_label_availability": float(np.mean(arrays["system_label_mask"])),
        "system_source_availability": float(np.mean(arrays["system_source_available_mask"])),
        "system_uncertainty_availability": float(np.mean(arrays["system_uncertainty_mask"])),
        "label_task_availability": float(np.mean(arrays["label_task_available_mask"])),
        "sampling_weight_eligibility": float(np.mean(eligible)),
    }
    source_contracts = manifest.get("source_contracts", {})
    label_document = source_contracts.get("label_spec_document", {})
    declared_contract = binding.get("contract") or {}
    fixed_scenario_hash = declared_contract.get("fixed_scenario_hash") or source_contracts.get(
        "fixed_scenario_hash"
    )
    fixed_scenario = {
        "declared": bool(fixed_scenario_hash),
        "policy": declared_contract.get(
            "fixed_scenario_policy", "not_declared_by_legacy_tensor_contract"
        ),
        "hash": fixed_scenario_hash,
        "evidence_boundary": declared_contract.get(
            "evidence_boundary",
            "Legacy archive: no separate learner-input/fixed-scenario contract was declared.",
        ),
        "excluded_tensor_fields": declared_contract.get("excluded_tensor_fields", []),
    }
    selected_weighted = sampling_audit.get("selected_weighted") or {}
    archive_summary = sampling_checkpoint.get("archive_summary") or {}
    roles = _role_report(arrays, static)
    research_count = int(
        (manifest.get("quality_summary", {}).get("purpose_admission_counts") or {}).get(
            "research_training", 0
        )
    )
    return {
        "generated_at": manifest.get("created_at") or "not_recorded",
        "dataset": {
            "dataset_id": manifest["dataset_id"], "manifest_version": manifest["schema_version"],
            "tensor_contract_version": manifest["tensor_contract"]["version"],
            "checkpoint_status": checkpoint["status"], "sampling_checkpoint_status": sampling_checkpoint["status"],
            "n_total": int(manifest["quality_summary"]["n_total"]),
            "n_valid": int(manifest["quality_summary"]["n_valid"]),
            "n_refused": int(manifest["quality_summary"]["n_refused"]),
            "split_counts": split_counts, "shard_count": len(shards),
            "disk_bytes": disk_bytes, "disk_mib": disk_bytes / 2**20,
            "archive_integrity_validation": "passed", "audit_overrides": list(verified.audit_overrides),
            "label_admission": manifest.get("label_admission", {}),
            "purpose_admission_counts": manifest["quality_summary"].get("purpose_admission_counts", {}),
            "unverified_parameter_fraction": manifest["quality_summary"].get("unverified_parameter_fraction"),
            "split_policy": manifest.get("split_policy", {}),
        },
        "timing": _timing_report(timing_raw),
        "model_input": {
            "kind": binding["kind"],
            "contract_version": declared_contract.get("version"),
            "tensor_field": binding["tensor_field"],
            "mask_field": binding["mask_field"],
            "raw_feature_count": len(binding["names"]),
            "raw_feature_names": list(binding["names"]),
            "raw_feature_units": list(binding["units"]),
            "learner_visible_only": binding["kind"] == "uav_control",
            "fixed_scenario": fixed_scenario,
        },
        "training_view": training_view,
        "features": features, "derived_inputs": derived,
        "labels": labels,
        "label_tasks": label_document.get("tasks", {}),
        "label_task_names": static["label_task_names"].astype(str).tolist(),
        "masks": masks,
        "replicate_counts": {
            str(int(value)): int(count)
            for value, count in zip(*np.unique(arrays["replicate_counts"], return_counts=True))
        },
        "duplicates": _duplicate_report(arrays, binding),
        "roles": roles,
        "sampling": {
            "candidate_count": int(sampling_audit.get("candidate_count", 0)),
            "selected_count": int(sampling_audit.get("selected_count", 0)),
            "positivity_min": _number(np.min(rho)) if len(rho) else None,
            "positivity_max": _number(np.max(rho)) if len(rho) else None,
            "weight_min": _number(np.min(weights)) if len(weights) else None,
            "weight_max": _number(np.max(weights)) if len(weights) else None,
            "weight_cv": _number(np.std(weights) / np.mean(weights)) if len(weights) else None,
            "effective_sample_size": overall_weights["ess"],
            "ess_fraction": overall_weights["ess_fraction"],
            "weight_eligible": int(len(weights)),
            "weight_diagnostics": overall_weights,
            "quality_gates": sampling_spec.get("quality_gates", {}),
            "quality_gate_results": sampling_audit.get("quality_gates", {}),
            "quality_gate_passed": sampling_audit.get("quality_gate_passed"),
            "coverage_status": sampling_audit.get("coverage_status"),
            "coverage_guaranteed": sampling_audit.get("coverage_guaranteed", False),
            "latent_ks": selected_weighted.get("ks_by_axis", []),
            "latent_occupancy": selected_weighted.get(
                "occupancy_fraction_by_axis", []
            ),
            "histogram_bins": selected_weighted.get("histogram_bins"),
            "archive_summary": {
                key: archive_summary.get(key)
                for key in (
                    "planned_candidate_count", "selected_candidate_count", "formal_input_count",
                    "executed_candidate_count", "execution_success_count", "refusal_count",
                    "successful_candidate_count", "purpose_admission_counts",
                    "planned_design_quality_passed", "realized_sampling_quality_status",
                    "scientific_validation_status",
                )
            },
        },
        "boundaries": {
            "software": {
                "archive_integrity": "passed",
                "checkpoint_status": checkpoint["status"],
                "tensor_contract": manifest["tensor_contract"]["version"],
                "statement": (
                    "Archive, hashes, tensor shapes, masks and declared split identities were "
                    "validated by the loader. This is software-integrity evidence only."
                ),
            },
            "physical": {
                "research_training_admitted": research_count > 0,
                "research_training_count": research_count,
                "label_admission": manifest.get("label_admission", {}),
                "statement": (
                    "Synthetic-teacher labels and passing software checks do not establish "
                    "physical calibration, real-scene representativeness, safety, or research admission."
                ),
            },
        },
        "interpretation": {
            "hard_contract": "归档完整性、张量语义、输入映射和声明分组泄漏检查。",
            "declared_gate": "只采用 SamplingSpec 预先声明的 positivity、ESS fraction 和 weight CV 门槛。",
            "descriptive": "特征覆盖、split 差异、标签平台、CI 宽度等只作描述，不虚构通用放行线。",
            "scientific_boundary": "当前标签是 synthetic_teacher；软件通过不证明物理有效、现实代表性或研究训练准入。",
        },
    }




HTML_V2 = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark"><title>数据集健康诊断</title>
<style>
:root{color-scheme:light dark;--bg:#f4f7f8;--surface:#fff;--text:#17232c;--muted:#5b6872;--line:#d8e0e5;--primary:#155e75;--primaryText:#fff;--soft:#e8f1f4;--good:#1f6a43;--goodBg:#e4f4ea;--warn:#8a5a15;--warnBg:#fff3dc;--bad:#a23838;--badBg:#fde9e9;--s1:#18759b;--s2:#bd7220;--s3:#4e7c57}
@media(prefers-color-scheme:dark){:root{--bg:#111920;--surface:#1a252d;--text:#e8eef2;--muted:#acbac5;--line:#3a4853;--primary:#add9e9;--primaryText:#102630;--soft:#263842;--good:#b6e2c4;--goodBg:#1b3525;--warn:#efcc87;--warnBg:#382b17;--bad:#f0b0b0;--badBg:#3c2224;--s1:#72bcdc;--s2:#e5a357;--s3:#91c397}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;line-height:1.55}main{max-width:1380px;margin:auto;padding:28px clamp(15px,4vw,46px) 48px}h1,h2,h3,p{margin-top:0}h1{font-size:clamp(25px,3vw,36px);margin-bottom:6px}h2{font-size:21px}h3{font-size:16px}code{font-family:ui-monospace,"Cascadia Code",Consolas,monospace;font-size:.9em;overflow-wrap:anywhere}.muted{color:var(--muted)}.small{font-size:13px}.tabs{display:flex;gap:6px;flex-wrap:wrap;border-bottom:1px solid var(--line);margin:22px 0;padding-bottom:12px}button{font:inherit;color:inherit;background:var(--surface);border:1px solid var(--line);border-radius:7px;padding:8px 12px}button:hover{background:var(--soft)}button[aria-selected=true]{background:var(--primary);border-color:var(--primary);color:var(--primaryText)}button:focus-visible{outline:3px solid var(--primary);outline-offset:2px}[hidden]{display:none!important}.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:13px;margin:18px 0}.stat,.panel{background:var(--surface);border-radius:10px;padding:17px;min-width:0}.stat .label{color:var(--muted);font-size:13px}.stat .value{font-size:23px;font-weight:600;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}.stat .context{font-size:12px;color:var(--muted);margin-top:4px}.grid2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px;align-items:start}.notice{border-left:3px solid var(--warn);background:var(--warnBg);padding:13px 15px;margin:16px 0}.notice.bad{border-color:var(--bad);background:var(--badBg)}.badge{display:inline-block;padding:2px 7px;border-radius:4px;font-size:12px;background:var(--soft)}.badge.good{color:var(--good);background:var(--goodBg)}.badge.warn{color:var(--warn);background:var(--warnBg)}.badge.bad{color:var(--bad);background:var(--badBg)}.tableWrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{color:var(--muted);font-weight:600;white-space:nowrap}td.num{text-align:right;font-variant-numeric:tabular-nums}.split{display:flex;height:24px;overflow:hidden;border-radius:4px;background:var(--soft);margin:10px 0}.split span{color:#fff;text-align:center;min-width:28px;font-size:12px;line-height:24px}.train{background:var(--s1)}.validation{background:var(--s2)}.test{background:var(--s3)}.check{border-top:1px solid var(--line);padding:11px 0}.check:first-child{border-top:0}.role{margin-bottom:18px}.eventGrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}.event{background:var(--soft);padding:10px;border-radius:6px}.hist{height:210px;display:flex;align-items:flex-end;gap:3px;border-bottom:1px solid var(--line);padding:12px 4px 0}.hist span{background:var(--s1);flex:1;min-width:2px}.metric{display:grid;grid-template-columns:minmax(220px,2fr) minmax(100px,1fr);gap:12px;padding:9px 0;border-bottom:1px solid var(--line)}.metric strong{text-align:right;font-variant-numeric:tabular-nums}.boundary{border-top:3px solid var(--good)}.boundary.physical{border-color:var(--bad)}footer,.evidence{margin-top:24px;padding-top:14px;border-top:1px solid var(--line);color:var(--muted);font-size:13px}@media(max-width:760px){.grid2{grid-template-columns:1fr}.metric{grid-template-columns:1fr}.metric strong{text-align:left}}
</style></head><body><main id="health-app"><header><h1 id="title"></h1><p id="subtitle" class="muted"></p></header>
<nav class="tabs" role="tablist" aria-label="数据集报告视图"><button role="tab" aria-selected="true" data-tab="overview">总览</button><button role="tab" aria-selected="false" data-tab="inputs">原始与训练输入</button><button role="tab" aria-selected="false" data-tab="roles">角色·事件·ESS</button><button role="tab" aria-selected="false" data-tab="labels">标签与 CI</button><button role="tab" aria-selected="false" data-tab="performance">生成时间</button><button role="tab" aria-selected="false" data-tab="boundaries">软件/物理边界</button></nav>
<section id="overview" role="tabpanel"></section><section id="inputs" role="tabpanel" hidden></section><section id="roles" role="tabpanel" hidden></section><section id="labels" role="tabpanel" hidden></section><section id="performance" role="tabpanel" hidden></section><section id="boundaries" role="tabpanel" hidden></section><details class="evidence"><summary>证据口径</summary><div id="evidence"></div></details><footer id="footer"></footer></main>
<script>const REPORT=__REPORT_JSON__;
(()=>{"use strict";const root=document.getElementById("health-app"),$=id=>document.getElementById(id);const fmt=(v,d=3)=>v===null||v===undefined?"—":Number(v).toLocaleString("zh-CN",{maximumFractionDigits:d});const pct=v=>v===null||v===undefined?"—":fmt(v*100,2)+"%";const esc=s=>String(s??"—").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));const badge=(v,k="")=>`<span class="badge ${k}">${esc(v)}</span>`;const stat=(l,v,c="")=>`<div class="stat"><div class="label">${esc(l)}</div><div class="value">${esc(v)}</div><div class="context">${esc(c)}</div></div>`;const sum=o=>Object.values(o||{}).reduce((a,b)=>a+Number(b||0),0);const roleTotal=c=>REPORT.roles.filter(r=>r.category===c).reduce((a,r)=>a+r.n,0);
function renderOverview(){const d=REPORT.dataset,m=REPORT.model_input,t=REPORT.training_view,b=REPORT.boundaries.physical,maxCI=Math.max(0,...REPORT.labels.map(x=>Number(x.ci_half_width.p95||0)));const counts=d.split_counts,total=Math.max(1,sum(counts));$("overview").innerHTML=`<div class="stats">${stat("有效样本",fmt(d.n_valid),`总输入 ${fmt(d.n_total)}`)}${stat("原始模型输入",fmt(m.raw_feature_count)+"D",m.tensor_field)}${stat("训练变换",t.available?fmt(t.feature_count)+"D":"不可用",t.available?t.tensor_field:t.reason)}${stat("归档体积",fmt(d.disk_mib,1)+" MiB",`${fmt(d.shard_count)} 分片`)}${stat("参考角色",fmt(roleTotal("reference")),"风险/评估主体")}${stat("定向角色",fmt(roleTotal("targeted")),"只用于声明范围")}</div><div class="grid2"><div class="panel"><h2>切分</h2><div class="split" aria-label="数据切分">${["train","validation","test"].map((s,i)=>`<span class="${s}" style="width:${100*(counts[s]||0)/total}%">${esc(s)} ${fmt(counts[s]||0)}</span>`).join("")}</div><p class="small muted">精确重复检查与声明分组隔离不等于现实研究单位独立。</p><h3>固定场景边界</h3><p>${badge(m.fixed_scenario.declared?"已声明":"旧合同",m.fixed_scenario.declared?"good":"warn")} <code>${esc(m.fixed_scenario.policy)}</code></p><p class="small"><code>${esc(m.fixed_scenario.hash)}</code></p><p class="small muted">${esc(m.fixed_scenario.evidence_boundary)}</p></div><div class="panel"><h2>对学习效果最重要的事实</h2><div class="check"><strong>输入约束</strong><p>${m.kind==="uav_control"?"默认学习输入只是 UAV 相对位置、速度与姿态；弹体、环境与求解器字段仅供教师审计。":"旧数据集未声明 UAV-only 学习输入，报告保留原有 case 特征解释。"}</p></div><div class="check"><strong>变换边界</strong><p>${t.available?`${fmt(m.raw_feature_count)}D 归档值转为 ${fmt(t.feature_count)}D 训练视图；${esc(t.normalization_boundary)}。`:esc(t.reason)}</p></div><div class="check"><strong>MC 不确定性</strong><p>系统输出 CI 半宽 p95 最大为 ${fmt(maxCI,4)}；可计算不代表已收敛。</p></div><div class="check"><strong>科学准入</strong><p>${b.research_training_admitted?badge("已声明准入","good"):badge("未准入","bad")} ${esc(b.statement)}</p></div></div></div>`}
function featureTable(rows){return `<div class="tableWrap"><table><thead><tr><th>名称</th><th>单位</th><th>状态</th><th>中位</th><th>p05–p95</th><th>语义/变换</th></tr></thead><tbody>${rows.map(f=>`<tr><td><code>${esc(f.name)}</code></td><td>${esc(f.unit)}</td><td>${f.constant?badge("固定","warn"):badge("变化","good")}</td><td class="num">${fmt(f.stats.median,5)}</td><td class="num">${fmt(f.stats.p05,5)} – ${fmt(f.stats.p95,5)}</td><td>${esc(f.semantic||f.transformation)}</td></tr>`).join("")}</tbody></table></div>`}
function renderInputs(){const m=REPORT.model_input,t=REPORT.training_view,d=REPORT.derived_inputs;$("inputs").innerHTML=`<div class="stats">${stat("归档原始维度",fmt(m.raw_feature_count),m.tensor_field)}${stat("实际变化",fmt(d.variable_feature_count),pct(d.variable_feature_count/d.feature_count))}${stat("固定分量",fmt(d.constant_feature_count),pct(d.constant_feature_count/d.feature_count))}${stat("训练维度",t.available?fmt(t.feature_count):"—",t.available?t.tensor_field:t.reason)}</div><div class="panel"><h2>原始学习输入</h2>${featureTable(REPORT.features)}</div><div class="grid2" style="margin-top:20px"><div class="panel"><h2>派生向量检查</h2>${d.vector_checks.length?d.vector_checks.map(c=>`<div class="check"><strong>${esc(c.label)}</strong><p>${fmt(c.stats.min,5)} / ${fmt(c.stats.median,5)} / ${fmt(c.stats.max,5)} ${esc(c.unit)} <span class="muted">(min / median / max)</span></p></div>`).join(""):"<p>旧合同未提供可识别的向量组。</p>"}</div><div class="panel"><h2>边界</h2><p>排除的教师张量：${m.fixed_scenario.excluded_tensor_fields.length?m.fixed_scenario.excluded_tensor_fields.map(x=>`<code>${esc(x)}</code>`).join(" "):"旧合同未单独声明"}</p><p class="small muted">固定场景是本数据集的使用边界，不是可供网络学习的变量。</p></div></div><div class="panel" style="margin-top:20px"><h2>训练变换视图</h2>${t.available?featureTable(t.features):`<div class="notice">${esc(t.reason)}</div>`}</div>`}
function eventCards(events){return `<div class="eventGrid">${Object.entries(events).map(([name,e])=>`<div class="event"><code>${esc(name)}</code><br>${e.available?`<strong>${fmt(e.count)}</strong> / ${fmt(e.available_n)} · rate ${pct(e.rate)}${e.weighted_rate===null?"":` · weighted ${pct(e.weighted_rate)}`}`:"字段不可用"}</div>`).join("")}</div>`}
function renderRoles(){const s=REPORT.sampling;$("roles").innerHTML=`<div class="stats">${stat("角色数",fmt(REPORT.roles.length),"从归档索引恢复")}${stat("参考样本",fmt(roleTotal("reference")),"可用设计权重")}${stat("定向样本",fmt(roleTotal("targeted")),"不应混入参考风险")}${stat("整体 ESS",fmt(s.effective_sample_size,1),`fraction ${fmt(s.ess_fraction,3)}`)}</div><div class="notice">事件 count 是已归档机理字段的观测数；系统软标签的 expected event mass 另行展示。设计权重 ESS 不是类别权重、事件数或 minibatch 概率。</div>${REPORT.roles.map(r=>`<div class="panel role"><h2><code>${esc(r.role)}</code> ${badge(r.category,r.category==="reference"?"good":r.category==="targeted"?"warn":"")}</h2><div class="stats">${stat("n",fmt(r.n),Object.entries(r.splits).map(([k,v])=>`${k} ${v}`).join(" "))}${stat("ESS",fmt(r.weights.ess,2),`eligible ${fmt(r.weights.eligible_n)}`)}${stat("ESS fraction",fmt(r.weights.ess_fraction,3),`ineligible ${fmt(r.weights.ineligible_n)}`)}${stat("权重 p95",fmt(r.weights.p95,3),`变异系数 ${fmt(r.weights.cv,3)}`)}</div><p class="small muted">分层：${Object.entries(r.strata).map(([k,v])=>`${esc(k)}=${fmt(v)}`).join(" · ")}</p>${eventCards(r.events)}<div class="tableWrap"><table><thead><tr><th>系统标签</th><th>均值</th><th>expected event mass</th><th>CI 半宽 median / p95 / max</th></tr></thead><tbody>${r.system_labels.map(l=>`<tr><td><code>${esc(l.name)}</code></td><td class="num">${fmt(l.mean_probability,5)}</td><td class="num">${fmt(l.expected_event_mass,2)}</td><td class="num">${fmt(l.ci_half_width.median,4)} / ${fmt(l.ci_half_width.p95,4)} / ${fmt(l.ci_half_width.max,4)}</td></tr>`).join("")}</tbody></table></div></div>`).join("")}`}
function renderLabels(){const masks=REPORT.masks;$("labels").innerHTML=`<div class="stats">${Object.entries(masks).map(([k,v])=>stat(k,pct(v))).join("")}</div><div class="grid2">${REPORT.labels.map(l=>{const max=Math.max(1,...l.histogram.counts);return `<div class="panel"><h2><code>${esc(l.name)}</code></h2><div class="stats">${stat("均值",fmt(l.stats.mean,5))}${stat("标准差",fmt(l.stats.std,5))}${stat("众数占比",pct(l.mode_fraction))}${stat("CI p95",fmt(l.ci_half_width.p95,5))}</div><div class="hist" role="img" aria-label="${esc(l.name)} 软标签直方图">${l.histogram.counts.map((c,i)=>`<span style="height:${100*c/max}%" title="bin ${i}: ${c}"></span>`).join("")}</div><p class="small muted">CI 可用 ${fmt(l.ci_available)} / ${fmt(l.available)}；半宽为 MC 条件均值估计，不是现实预测区间。</p></div>`}).join("")}</div><div class="panel" style="margin-top:20px"><h2>重复数分布</h2><p>${Object.entries(REPORT.replicate_counts).map(([n,c])=>`${fmt(c)} 条 × R=${esc(n)}`).join(" · ")}</p></div>`}
function renderPerformance(){const t=REPORT.timing;$("performance").innerHTML=t.provided?`<div class="stats">${stat("时间/容量指标",fmt(t.metrics.length),"从 timing JSON 动态提取")}${stat("归档体积",fmt(REPORT.dataset.disk_mib,2)+" MiB","当前完整采样目录")}</div><div class="panel"><h2>生成时间与容量</h2>${t.metrics.length?t.metrics.map(m=>`<div class="metric"><code>${esc(m.name)}</code><strong>${fmt(m.value,5)}</strong></div>`).join(""):"<p>timing JSON 已提供，但未发现标准时间/吞吐/容量键。</p>"}<p class="small muted">时间是当前软件、硬件、worker、重复数与分片配置的观测值，不是 SLA 或任意规模的线性保证。</p></div>`:`<div class="notice">未提供 timing JSON；数据完整性不受影响，但本报告不展示生成时间。</div>`}
function renderBoundaries(){const b=REPORT.boundaries,m=REPORT.model_input.fixed_scenario;$("boundaries").innerHTML=`<div class="grid2"><div class="panel boundary"><h2>软件完整性边界</h2><p>${badge(b.software.archive_integrity,"good")} checkpoint=${esc(b.software.checkpoint_status)} · tensor=${esc(b.software.tensor_contract)}</p><p>${esc(b.software.statement)}</p><ul><li>${esc(REPORT.interpretation.hard_contract)}</li><li>${esc(REPORT.interpretation.declared_gate)}</li><li>audit overrides: <code>${esc(JSON.stringify(REPORT.dataset.audit_overrides))}</code></li></ul></div><div class="panel boundary physical"><h2>物理/研究边界</h2><p>${b.physical.research_training_admitted?badge("已准入","good"):badge("未准入","bad")} research_training=${fmt(b.physical.research_training_count)}</p><p>${esc(b.physical.statement)}</p><ul><li>${esc(REPORT.interpretation.scientific_boundary)}</li><li>固定场景 hash：<code>${esc(m.hash)}</code></li><li>在该固定场景外的目标、弹体、环境或坐标约定泛化未被证明。</li></ul></div></div>`}
function show(tab){for(const button of root.querySelectorAll("[data-tab]")){button.setAttribute("aria-selected",String(button.dataset.tab===tab))}for(const id of ["overview","inputs","roles","labels","performance","boundaries"]){$(id).hidden=id!==tab}}root.addEventListener("click",event=>{const button=event.target.closest("[data-tab]");if(button)show(button.dataset.tab)});root.querySelector('[role="tablist"]').addEventListener("keydown",event=>{if(!["ArrowLeft","ArrowRight","Home","End"].includes(event.key))return;const tabs=[...root.querySelectorAll("[data-tab]")],index=tabs.indexOf(event.target);if(index<0)return;const next=event.key==="Home"?0:event.key==="End"?tabs.length-1:(index+(event.key==="ArrowRight"?1:-1)+tabs.length)%tabs.length;event.preventDefault();tabs[next].focus();show(tabs[next].dataset.tab)});$("title").textContent=`${REPORT.dataset.n_valid.toLocaleString("zh-CN")} 条数据·深度学习健康诊断`;$("subtitle").textContent=`${REPORT.dataset.dataset_id} · tensor ${REPORT.dataset.tensor_contract_version} · input ${REPORT.model_input.tensor_field}`;$("evidence").innerHTML=`<ul><li>${esc(REPORT.interpretation.descriptive)}</li><li>${esc(REPORT.interpretation.scientific_boundary)}</li><li>学习输入重复指纹只使用 <code>${esc(REPORT.model_input.tensor_field)}</code> 及其 mask，不使用教师专用张量。</li></ul>`;$("footer").textContent=`${REPORT.generated_at} · manifest ${REPORT.dataset.manifest_version} · 单文件离线报告`;renderOverview();renderInputs();renderRoles();renderLabels();renderPerformance();renderBoundaries();show("overview")})();</script></body></html>'''


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an offline deep-learning dataset health report")
    parser.add_argument("dataset_root", type=Path, help="Completed sharded dataset directory")
    parser.add_argument("--timing-json", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--html-output", type=Path, required=True)
    arguments = parser.parse_args()
    report = build_report(arguments.dataset_root.resolve(), arguments.timing_json.resolve() if arguments.timing_json else None)
    arguments.json_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.html_output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    arguments.json_output.write_text(payload + "\n", encoding="utf-8")
    embedded = json.dumps(report, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    arguments.html_output.write_text(
        HTML_V2.replace("__REPORT_JSON__", embedded), encoding="utf-8"
    )
    print(json.dumps({
        "status": "ok", "dataset_id": report["dataset"]["dataset_id"],
        "json_output": str(arguments.json_output.resolve()),
        "html_output": str(arguments.html_output.resolve()),
        "n_valid": report["dataset"]["n_valid"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
