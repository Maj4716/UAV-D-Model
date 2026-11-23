#!/usr/bin/env python3
"""Run the preregistered Phase-0 multi-task learning diagnostics.

The development package mixes Reference and enriched training arms.  Training
uses the deduplicated mixture without design weights, while every reported
metric is restricted to ``reference_uniform`` validation members and uses the
arm's inverse-inclusion design weight.  The script never reads or creates a
test split.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DATASET_DIRECTORY = ROOT / "artifacts" / "development_dataset"
DEFAULT_OUTPUT_DIRECTORY = ROOT / "artifacts" / "experiments" / "phase0_20261120"

from damage_engine.system_learning import (  # noqa: E402
    evaluate_system_probabilities,
    fit_reference_prevalence_baseline,
    masked_soft_bce_with_logits,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": "phase0-learning-diagnostics-config@1.0.0",
    "experiment_id": "B0-phase0-multitask-baseline",
    "evidence_scope": (
        "Development-only software-teacher diagnostic; no sealed test and no "
        "claim of external physical validity."
    ),
    "data_policy": {
        "expected_dataset_schema": "idea1-hybrid-neural-dataset@1.1.0",
        "train_split_code": 0,
        "validation_split_code": 1,
        "training_rows": "deduplicated_mixture_train",
        "training_design_weights": "none",
        "evaluation_strategy": "reference_uniform",
        "evaluation_split": "validation",
        "evaluation_design_weight": "design_weight_raw",
        "test_split_access": "forbidden",
        "design_normalization": "mean_and_population_std_fit_on_all_train_rows_only",
        "fraction_selection": "nested_outcome_blind_rng_permutation_of_train_rows",
    },
    "model": {
        "architecture": "shared_mlp_four_heads",
        "hidden_widths": [64, 64],
        "activation": "relu",
        "dropout": 0.0,
        "heads": [
            "domain_complete",
            "component_support",
            "component_severity",
            "tree_outputs",
        ],
    },
    "training": {
        "feature_views": ["x_normalized", "x_design_train_standardized"],
        "train_fractions": [0.25, 0.5, 1.0],
        "seeds": [2026112001, 2026112002, 2026112003],
        "epochs": 50,
        "batch_size": 256,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 5.0,
        "optimizer": "adamw",
        "device": "cpu",
        "torch_num_threads": 1,
        "early_stopping": "disabled_fixed_epoch_budget",
        "task_loss_weights": {
            "domain_complete": 1.0,
            "component_support": 1.0,
            "component_severity": 1.0,
            "tree_outputs": 1.0,
        },
    },
    "evaluation": {
        "calibration_bins": 10,
        "component_ap_bottleneck_threshold": 0.20,
        "minimum_validation_positive_count": 20,
        "tree_intermediate_lower": 0.01,
        "tree_intermediate_upper": 0.99,
        "tree_saturation_threshold": 0.99,
        "tree_zero_threshold": 1.0e-6,
        "summary_statistics": "mean_sample_std_across_seeds",
    },
}


REQUIRED_ARRAYS = (
    "x",
    "x_normalized",
    "x_design",
    "x_train_mean",
    "x_train_scale",
    "domain_complete",
    "response_mask",
    "split_code",
    "component_support",
    "component_severity",
    "tree_outputs",
    "candidate_group_id",
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _sample_std(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size <= 1:
        return 0.0
    return float(np.std(array, ddof=1))


def mean_std(values: Sequence[float | None]) -> dict[str, Any]:
    finite = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(value)],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(np.mean(finite)),
        "std": _sample_std(finite.tolist()),
        "n": int(finite.size),
    }


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != DEFAULT_CONFIG["schema_version"]:
        raise RuntimeError("unsupported Phase-0 configuration schema")
    training = config.get("training")
    evaluation = config.get("evaluation")
    if not isinstance(training, Mapping) or not isinstance(evaluation, Mapping):
        raise RuntimeError("training and evaluation configuration blocks are required")
    seeds = training.get("seeds")
    fractions = training.get("train_fractions")
    views = training.get("feature_views")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 3
        or len(set(map(int, seeds))) != len(seeds)
    ):
        raise RuntimeError("at least three unique training seeds are required")
    if (
        not isinstance(fractions, list)
        or not fractions
        or any(not 0.0 < float(value) <= 1.0 for value in fractions)
        or sorted(map(float, fractions)) != list(map(float, fractions))
    ):
        raise RuntimeError("train fractions must be increasing values in (0, 1]")
    if views != ["x_normalized", "x_design_train_standardized"]:
        raise RuntimeError("both preregistered feature views must be evaluated")
    for name in ("epochs", "batch_size", "torch_num_threads"):
        if int(training.get(name, 0)) <= 0:
            raise RuntimeError(f"training.{name} must be positive")
    for name in ("learning_rate", "gradient_clip_norm"):
        if not math.isfinite(float(training.get(name, 0.0))) or float(
            training[name]
        ) <= 0.0:
            raise RuntimeError(f"training.{name} must be finite and positive")
    if int(evaluation.get("calibration_bins", 0)) < 2:
        raise RuntimeError("evaluation.calibration_bins must be at least two")


def load_phase0_dataset(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    manifest = json.loads(
        (directory / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (directory / "feature_schema.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != DEFAULT_CONFIG["data_policy"][
        "expected_dataset_schema"
    ]:
        raise RuntimeError("unexpected development dataset schema")
    if manifest.get("sealed_test_split") is not None:
        raise RuntimeError("Phase-0 development diagnostics refuse sealed test data")
    with np.load(directory / "tensors.npz", allow_pickle=False) as archive:
        missing = sorted(set(REQUIRED_ARRAYS) - set(archive.files))
        if missing:
            raise RuntimeError(f"development tensor archive is incomplete: {missing}")
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    count = int(arrays["split_code"].size)
    if any(arrays[name].shape[0] != count for name in REQUIRED_ARRAYS if name not in {"x_train_mean", "x_train_scale"}):
        raise RuntimeError("development tensors do not share a case axis")
    if not np.array_equal(
        arrays["domain_complete"].astype(bool), arrays["response_mask"].astype(bool)
    ):
        raise RuntimeError("response mask disagrees with domain completeness")
    return {
        "directory": directory,
        "manifest": manifest,
        "schema": schema,
        "arrays": arrays,
    }


def train_standardize_design(
    x_design: np.ndarray, split_code: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(x_design, dtype=np.float64)
    splits = np.asarray(split_code)
    if values.ndim != 2 or splits.shape != (values.shape[0],):
        raise RuntimeError("x_design and split_code are not aligned")
    train = splits == 0
    if not np.any(train):
        raise RuntimeError("training split is empty")
    mean = np.mean(values[train], axis=0, dtype=np.float64)
    scale = np.std(values[train], axis=0, dtype=np.float64)
    scale[scale < 1.0e-12] = 1.0
    standardized = ((values - mean) / scale).astype(np.float32)
    return standardized, mean.astype(np.float32), scale.astype(np.float32)


def feature_views(dataset: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays = dataset["arrays"]
    design, design_mean, design_scale = train_standardize_design(
        arrays["x_design"], arrays["split_code"]
    )
    train = arrays["split_code"] == 0
    expected_x_mean = np.mean(arrays["x"][train], axis=0, dtype=np.float64)
    expected_x_scale = np.std(arrays["x"][train], axis=0, dtype=np.float64)
    expected_x_scale[expected_x_scale < 1.0e-12] = 1.0
    expected_normalized = (arrays["x"] - expected_x_mean) / expected_x_scale
    if not np.allclose(
        arrays["x_normalized"], expected_normalized, rtol=2.0e-6, atol=2.0e-6
    ):
        raise RuntimeError("x_normalized was not fit from Train only")
    return (
        {
            "x_normalized": np.asarray(arrays["x_normalized"], dtype=np.float32),
            "x_design_train_standardized": design,
        },
        {
            "x_normalized": {
                "mean": np.asarray(arrays["x_train_mean"], dtype=float).tolist(),
                "scale": np.asarray(arrays["x_train_scale"], dtype=float).tolist(),
                "fit_split": "train",
            },
            "x_design_train_standardized": {
                "mean": design_mean.astype(float).tolist(),
                "scale": design_scale.astype(float).tolist(),
                "fit_split": "train",
            },
        },
    )


def load_reference_memberships(
    path: Path,
    candidate_group_ids: Sequence[str],
    split_code: np.ndarray,
    *,
    strategy: str = "reference_uniform",
) -> dict[str, np.ndarray]:
    groups = [str(value) for value in candidate_group_ids]
    if len(groups) != len(set(groups)):
        raise RuntimeError("candidate_group_id must be globally unique")
    lookup = {value: index for index, value in enumerate(groups)}
    splits = np.asarray(split_code)
    if splits.shape != (len(groups),) or not set(np.unique(splits)).issubset({0, 1}):
        raise RuntimeError("split_code must contain aligned Train/Validation codes")
    records: list[tuple[int, float, str]] = []
    seen: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("strategy")) != strategy:
                continue
            group = str(row["candidate_group_id"])
            if group not in lookup:
                raise RuntimeError("Reference membership names an unknown candidate")
            if group in seen:
                raise RuntimeError("duplicate Reference membership for one candidate")
            seen.add(group)
            index = lookup[group]
            expected_split = "train" if int(splits[index]) == 0 else "validation"
            if str(row.get("split")) != expected_split:
                raise RuntimeError("Reference membership split disagrees with tensors")
            probability = float(row["inclusion_probability"])
            weight = float(row["design_weight_raw"])
            if (
                not math.isfinite(probability)
                or not 0.0 < probability <= 1.0
                or not math.isfinite(weight)
                or weight <= 0.0
                or not math.isclose(weight, 1.0 / probability, rel_tol=2.0e-10)
            ):
                raise RuntimeError("invalid Reference inclusion probability or weight")
            records.append((index, weight, expected_split))
    if not records:
        raise RuntimeError("Reference membership is empty")
    records.sort(key=lambda value: value[0])
    indices = np.asarray([value[0] for value in records], dtype=np.int64)
    weights = np.asarray([value[1] for value in records], dtype=np.float64)
    names = np.asarray([value[2] for value in records])
    if not np.any(names == "train") or not np.any(names == "validation"):
        raise RuntimeError("Reference membership must cover Train and Validation")
    return {"indices": indices, "weights": weights, "split_names": names}


def nested_train_indices(
    split_code: np.ndarray, *, fraction: float, seed: int
) -> np.ndarray:
    train = np.flatnonzero(np.asarray(split_code) == 0)
    if train.size == 0:
        raise RuntimeError("training split is empty")
    value = float(fraction)
    if not 0.0 < value <= 1.0:
        raise RuntimeError("training fraction must lie in (0, 1]")
    order = np.random.default_rng(int(seed)).permutation(train)
    count = min(train.size, max(1, int(math.ceil(value * train.size))))
    return np.sort(order[:count])


def leakage_checks(
    arrays: Mapping[str, np.ndarray], reference: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    groups = np.asarray(arrays["candidate_group_id"]).astype(str)
    splits = np.asarray(arrays["split_code"])
    train_groups = set(groups[splits == 0].tolist())
    validation_groups = set(groups[splits == 1].tolist())
    ref_indices = np.asarray(reference["indices"], dtype=np.int64)
    ref_splits = np.asarray(reference["split_names"]).astype(str)
    validation_reference = ref_indices[ref_splits == "validation"]
    checks = {
        "candidate_group_ids_globally_unique": len(set(groups.tolist())) == len(groups),
        "train_validation_candidate_overlap_count": len(
            train_groups & validation_groups
        ),
        "reference_validation_count": int(validation_reference.size),
        "reference_validation_indices_all_in_validation_split": bool(
            np.all(splits[validation_reference] == 1)
        ),
        "response_mask_equals_domain_complete": bool(
            np.array_equal(
                np.asarray(arrays["response_mask"], dtype=bool),
                np.asarray(arrays["domain_complete"], dtype=bool),
            )
        ),
        "test_split_accessed": False,
    }
    checks["passed"] = bool(
        checks["candidate_group_ids_globally_unique"]
        and checks["train_validation_candidate_overlap_count"] == 0
        and checks["reference_validation_count"] > 0
        and checks["reference_validation_indices_all_in_validation_split"]
        and checks["response_mask_equals_domain_complete"]
    )
    if not checks["passed"]:
        raise RuntimeError("Phase-0 leakage checks failed")
    return checks


def _weighted_means(
    targets: np.ndarray, masks: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    values = np.asarray(targets, dtype=np.float64)
    available = np.asarray(masks, dtype=bool)
    case_weights = np.asarray(weights, dtype=np.float64)
    if values.shape != available.shape or values.shape[0] != case_weights.size:
        raise RuntimeError("weighted mean inputs are not aligned")
    means = np.empty(values.shape[1], dtype=np.float64)
    for column in range(values.shape[1]):
        use = available[:, column]
        if not np.any(use):
            raise RuntimeError("Reference Train has an empty regression task")
        means[column] = np.average(values[use, column], weights=case_weights[use])
    return means


def _probability_report(
    targets: np.ndarray,
    predictions: np.ndarray,
    masks: np.ndarray,
    task_names: Sequence[str],
    weights: np.ndarray,
    baseline_predictions: Sequence[float],
    *,
    calibration_bins: int,
) -> dict[str, Any]:
    report = evaluate_system_probabilities(
        targets,
        predictions,
        masks,
        task_names,
        evaluated_tasks=task_names,
        design_weights=weights,
        baseline_predictions=baseline_predictions,
        calibration_bins=calibration_bins,
    )
    for row in report["tasks"]:
        denominator = float(row["design_weight_sum_scaled"])
        row["weighted_prevalence"] = float(
            row["weighted_soft_event_mass_scaled"] / denominator
        )
        row["average_precision"] = row["soft_average_precision"]
        row["average_precision_status"] = row["soft_average_precision_status"]
    report["macro_average_precision"] = float(
        np.mean(
            [
                row["average_precision"]
                for row in report["tasks"]
                if row["average_precision"] is not None
            ]
        )
    )
    report["macro_ece"] = float(np.mean([row["ece"] for row in report["tasks"]]))
    return report


def weighted_regression_report(
    targets: np.ndarray,
    predictions: np.ndarray,
    masks: np.ndarray,
    task_names: Sequence[str],
    weights: np.ndarray,
    baseline_predictions: Sequence[float],
) -> dict[str, Any]:
    values = np.asarray(targets, dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    available = np.asarray(masks, dtype=bool)
    case_weights = np.asarray(weights, dtype=np.float64)
    baseline = np.asarray(baseline_predictions, dtype=np.float64)
    names = list(map(str, task_names))
    if (
        values.shape != predicted.shape
        or values.shape != available.shape
        or values.ndim != 2
        or values.shape[0] != case_weights.size
        or values.shape[1] != len(names)
        or baseline.shape != (len(names),)
    ):
        raise RuntimeError("regression evaluation inputs are not aligned")
    rows = []
    for column, name in enumerate(names):
        use = available[:, column]
        if not np.any(use):
            rows.append({"task_name": name, "status": "undefined_no_available_targets"})
            continue
        y = values[use, column]
        p = predicted[use, column]
        w = case_weights[use]
        denominator = float(np.sum(w, dtype=np.float64))
        squared = float(np.sum(w * np.square(p - y), dtype=np.float64) / denominator)
        absolute = float(np.sum(w * np.abs(p - y), dtype=np.float64) / denominator)
        baseline_squared = float(
            np.sum(w * np.square(baseline[column] - y), dtype=np.float64)
            / denominator
        )
        rows.append(
            {
                "task_name": name,
                "status": "defined",
                "available_count": int(np.count_nonzero(use)),
                "weighted_target_mean": float(np.sum(w * y) / denominator),
                "mae": absolute,
                "rmse": math.sqrt(squared),
                "baseline_rmse": math.sqrt(baseline_squared),
                "mse_skill_score": (
                    None
                    if baseline_squared <= 0.0
                    else 1.0 - squared / baseline_squared
                ),
            }
        )
    defined = [row for row in rows if row["status"] == "defined"]
    return {
        "task_names": names,
        "tasks": rows,
        "macro_mae": float(np.mean([row["mae"] for row in defined])),
        "macro_rmse": float(np.mean([row["rmse"] for row in defined])),
        "macro_mse_skill_score": float(
            np.mean(
                [
                    row["mse_skill_score"]
                    for row in defined
                    if row["mse_skill_score"] is not None
                ]
            )
        ),
    }


def _baseline_context(
    arrays: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    component_names: Sequence[str],
    tree_names: Sequence[str],
) -> dict[str, Any]:
    indices = np.asarray(reference["indices"], dtype=np.int64)
    split_names = np.asarray(reference["split_names"]).astype(str)
    weights = np.asarray(reference["weights"], dtype=np.float64)
    roles = np.where(
        split_names == "train", "reference_train", "reference_validation"
    )
    response = np.asarray(arrays["response_mask"][indices], dtype=bool)
    component_mask = np.broadcast_to(
        response[:, None], (indices.size, len(component_names))
    )
    tree_mask = np.broadcast_to(response[:, None], (indices.size, len(tree_names)))
    domain = np.asarray(arrays["domain_complete"][indices], dtype=np.float64)[:, None]
    domain_baseline = fit_reference_prevalence_baseline(
        domain,
        np.ones_like(domain, dtype=bool),
        ("domain_complete",),
        split_names=split_names,
        sampling_roles=roles,
        design_weights=weights,
    )
    support_baseline = fit_reference_prevalence_baseline(
        arrays["component_support"][indices],
        component_mask,
        component_names,
        split_names=split_names,
        sampling_roles=roles,
        design_weights=weights,
    )
    tree_baseline = fit_reference_prevalence_baseline(
        arrays["tree_outputs"][indices],
        tree_mask,
        tree_names,
        split_names=split_names,
        sampling_roles=roles,
        design_weights=weights,
    )
    reference_train = split_names == "train"
    severity_mask = component_mask & reference_train[:, None]
    severity_positive_mask = severity_mask & (
        arrays["component_support"][indices] > 0.0
    )
    severity_baseline = _weighted_means(
        arrays["component_severity"][indices], severity_mask, weights
    )
    severity_positive_baseline = _weighted_means(
        arrays["component_severity"][indices], severity_positive_mask, weights
    )
    return {
        "domain": domain_baseline,
        "support": support_baseline,
        "tree": tree_baseline,
        "severity": severity_baseline,
        "severity_positive": severity_positive_baseline,
    }


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("PyTorch is required for Phase-0 diagnostics") from exc
    return torch


def _make_model(input_dim: int, component_count: int, tree_count: int, config: Mapping[str, Any]):
    torch = _import_torch()
    widths = [int(value) for value in config["model"]["hidden_widths"]]

    class SharedMultiTaskMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[Any] = []
            current = input_dim
            for width in widths:
                layers.extend((torch.nn.Linear(current, width), torch.nn.ReLU()))
                current = width
            self.shared = torch.nn.Sequential(*layers)
            self.domain = torch.nn.Linear(current, 1)
            self.support = torch.nn.Linear(current, component_count)
            self.severity = torch.nn.Linear(current, component_count)
            self.tree = torch.nn.Linear(current, tree_count)

        def forward(self, inputs):
            hidden = self.shared(inputs)
            return {
                "domain": self.domain(hidden),
                "support": self.support(hidden),
                "severity": torch.sigmoid(self.severity(hidden)),
                "tree": self.tree(hidden),
            }

    return SharedMultiTaskMLP()


def _seed_torch(seed: int, *, threads: int) -> Any:
    torch = _import_torch()
    torch.set_num_threads(int(threads))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return torch


def train_model(
    features: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    train_indices: np.ndarray,
    config: Mapping[str, Any],
    *,
    seed: int,
):
    training = config["training"]
    torch = _seed_torch(seed, threads=int(training["torch_num_threads"]))
    device = torch.device(str(training["device"]))
    model = _make_model(
        int(features.shape[1]),
        int(arrays["component_support"].shape[1]),
        int(arrays["tree_outputs"].shape[1]),
        config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    loss_weights = {
        key: float(value) for key, value in training["task_loss_weights"].items()
    }
    batch_size = int(training["batch_size"])
    generator = np.random.default_rng(int(seed) + 1_000_003)
    epoch_losses = []
    model.train()
    for _epoch in range(int(training["epochs"])):
        order = generator.permutation(train_indices)
        running = defaultdict(float)
        batch_count = 0
        for start in range(0, order.size, batch_size):
            batch = order[start : start + batch_size]
            x = torch.as_tensor(features[batch], dtype=torch.float32, device=device)
            domain_target = torch.as_tensor(
                arrays["domain_complete"][batch, None],
                dtype=torch.float32,
                device=device,
            )
            response = torch.as_tensor(
                arrays["response_mask"][batch], dtype=torch.bool, device=device
            )
            output = model(x)
            domain_loss = masked_soft_bce_with_logits(
                output["domain"],
                domain_target,
                torch.ones_like(domain_target, dtype=torch.bool),
            )
            loss = loss_weights["domain_complete"] * domain_loss
            running["domain_complete"] += float(domain_loss.detach().cpu())
            if bool(torch.any(response)):
                support_target = torch.as_tensor(
                    arrays["component_support"][batch],
                    dtype=torch.float32,
                    device=device,
                )
                severity_target = torch.as_tensor(
                    arrays["component_severity"][batch],
                    dtype=torch.float32,
                    device=device,
                )
                tree_target = torch.as_tensor(
                    arrays["tree_outputs"][batch],
                    dtype=torch.float32,
                    device=device,
                )
                support_mask = response[:, None].expand_as(support_target)
                tree_mask = response[:, None].expand_as(tree_target)
                support_loss = masked_soft_bce_with_logits(
                    output["support"], support_target, support_mask
                )
                tree_loss = masked_soft_bce_with_logits(
                    output["tree"], tree_target, tree_mask
                )
                safe_severity = torch.where(
                    support_mask, severity_target, torch.zeros_like(severity_target)
                )
                severity_loss = (
                    torch.square(output["severity"] - safe_severity)
                    * support_mask.to(dtype=torch.float32)
                ).sum() / support_mask.sum()
                loss = (
                    loss
                    + loss_weights["component_support"] * support_loss
                    + loss_weights["component_severity"] * severity_loss
                    + loss_weights["tree_outputs"] * tree_loss
                )
                running["component_support"] += float(support_loss.detach().cpu())
                running["component_severity"] += float(severity_loss.detach().cpu())
                running["tree_outputs"] += float(tree_loss.detach().cpu())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            optimizer.step()
            running["total"] += float(loss.detach().cpu())
            batch_count += 1
        epoch_losses.append(
            {
                key: value / batch_count
                for key, value in sorted(running.items())
            }
        )
    return model, epoch_losses


def _predict(model, features: np.ndarray, indices: np.ndarray, config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    torch = _import_torch()
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        output = model(
            torch.as_tensor(features[indices], dtype=torch.float32, device=device)
        )
    return {
        "domain": torch.sigmoid(output["domain"]).cpu().numpy(),
        "support": torch.sigmoid(output["support"]).cpu().numpy(),
        "severity": output["severity"].cpu().numpy(),
        "tree": torch.sigmoid(output["tree"]).cpu().numpy(),
    }


def evaluate_model(
    model,
    features: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    baselines: Mapping[str, Any],
    component_names: Sequence[str],
    tree_names: Sequence[str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    ref_indices = np.asarray(reference["indices"], dtype=np.int64)
    ref_splits = np.asarray(reference["split_names"]).astype(str)
    use = ref_splits == "validation"
    indices = ref_indices[use]
    weights = np.asarray(reference["weights"], dtype=np.float64)[use]
    prediction = _predict(model, features, indices, config)
    response = np.asarray(arrays["response_mask"][indices], dtype=bool)
    component_mask = np.broadcast_to(
        response[:, None], (indices.size, len(component_names))
    )
    tree_mask = np.broadcast_to(response[:, None], (indices.size, len(tree_names)))
    bins = int(config["evaluation"]["calibration_bins"])
    domain = _probability_report(
        np.asarray(arrays["domain_complete"][indices], dtype=np.float64)[:, None],
        prediction["domain"],
        np.ones((indices.size, 1), dtype=bool),
        ("domain_complete",),
        weights,
        baselines["domain"]["predictions"],
        calibration_bins=bins,
    )
    support = _probability_report(
        arrays["component_support"][indices],
        prediction["support"],
        component_mask,
        component_names,
        weights,
        baselines["support"]["predictions"],
        calibration_bins=bins,
    )
    tree = _probability_report(
        arrays["tree_outputs"][indices],
        prediction["tree"],
        tree_mask,
        tree_names,
        weights,
        baselines["tree"]["predictions"],
        calibration_bins=bins,
    )
    severity = weighted_regression_report(
        arrays["component_severity"][indices],
        prediction["severity"],
        component_mask,
        component_names,
        weights,
        baselines["severity"],
    )
    positive_mask = component_mask & (
        arrays["component_support"][indices] > 0.0
    )
    severity_positive = weighted_regression_report(
        arrays["component_severity"][indices],
        prediction["severity"],
        positive_mask,
        component_names,
        weights,
        baselines["severity_positive"],
    )
    ap_threshold = float(
        config["evaluation"]["component_ap_bottleneck_threshold"]
    )
    min_positive = int(
        config["evaluation"]["minimum_validation_positive_count"]
    )
    bottlenecks = []
    for row in support["tasks"]:
        if (
            row["average_precision"] is not None
            and row["average_precision"] < ap_threshold
        ) or row["soft_event_mass"] < min_positive:
            bottlenecks.append(
                {
                    "task_name": row["task_name"],
                    "average_precision": row["average_precision"],
                    "unweighted_positive_count": int(round(row["soft_event_mass"])),
                    "weighted_prevalence": row["weighted_prevalence"],
                    "ap_below_threshold": bool(
                        row["average_precision"] is not None
                        and row["average_precision"] < ap_threshold
                    ),
                    "positive_count_below_threshold": bool(
                        row["soft_event_mass"] < min_positive
                    ),
                }
            )
    summary = {
        "domain_average_precision": domain["macro_average_precision"],
        "domain_brier": domain["macro_brier"],
        "domain_ece": domain["macro_ece"],
        "support_macro_average_precision": support["macro_average_precision"],
        "support_macro_brier": support["macro_brier"],
        "support_macro_ece": support["macro_ece"],
        "tree_macro_soft_average_precision": tree["macro_average_precision"],
        "tree_macro_brier": tree["macro_brier"],
        "tree_macro_ece": tree["macro_ece"],
        "severity_macro_rmse": severity["macro_rmse"],
        "severity_macro_mae": severity["macro_mae"],
        "severity_positive_macro_rmse": severity_positive["macro_rmse"],
        "severity_positive_macro_mae": severity_positive["macro_mae"],
        "component_bottleneck_count": len(bottlenecks),
    }
    return {
        "evaluation_population": {
            "strategy": "reference_uniform",
            "split": "validation",
            "case_count": int(indices.size),
            "response_complete_count": int(np.count_nonzero(response)),
            "design_weight_sum": float(np.sum(weights, dtype=np.float64)),
        },
        "summary": summary,
        "component_support": support,
        "component_severity": severity,
        "component_severity_positive": severity_positive,
        "tree_outputs": tree,
        "domain_complete": domain,
        "component_bottlenecks": bottlenecks,
    }


def aggregate_runs(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(str(run["feature_view"]), float(run["train_fraction"]))].append(run)
    output: dict[str, Any] = {}
    for (view, fraction), members in sorted(grouped.items()):
        summaries = [member["evaluation"]["summary"] for member in members]
        scalar_names = sorted(
            name
            for name, value in summaries[0].items()
            if isinstance(value, (int, float)) and name != "component_bottleneck_count"
        )
        aggregate = {
            name: mean_std([summary[name] for summary in summaries])
            for name in scalar_names
        }
        aggregate["component_bottleneck_count"] = mean_std(
            [float(summary["component_bottleneck_count"]) for summary in summaries]
        )
        component_rows = []
        task_names = [
            row["task_name"]
            for row in members[0]["evaluation"]["component_support"]["tasks"]
        ]
        for task_name in task_names:
            task_records = [
                next(
                    row
                    for row in member["evaluation"]["component_support"]["tasks"]
                    if row["task_name"] == task_name
                )
                for member in members
            ]
            component_rows.append(
                {
                    "task_name": task_name,
                    "average_precision": mean_std(
                        [row["average_precision"] for row in task_records]
                    ),
                    "brier": mean_std([row["brier"] for row in task_records]),
                    "weighted_prevalence": mean_std(
                        [row["weighted_prevalence"] for row in task_records]
                    ),
                    "unweighted_positive_count": int(
                        round(task_records[0]["soft_event_mass"])
                    ),
                }
            )
        tree_rows = []
        for task_name in [
            row["task_name"]
            for row in members[0]["evaluation"]["tree_outputs"]["tasks"]
        ]:
            task_records = [
                next(
                    row
                    for row in member["evaluation"]["tree_outputs"]["tasks"]
                    if row["task_name"] == task_name
                )
                for member in members
            ]
            tree_rows.append(
                {
                    "task_name": task_name,
                    "soft_average_precision": mean_std(
                        [row["average_precision"] for row in task_records]
                    ),
                    "brier": mean_std([row["brier"] for row in task_records]),
                    "ece": mean_std([row["ece"] for row in task_records]),
                    "calibration_intercept": mean_std(
                        [row["calibration_intercept"] for row in task_records]
                    ),
                    "calibration_slope": mean_std(
                        [row["calibration_slope"] for row in task_records]
                    ),
                }
            )
        output.setdefault(view, {})[f"{fraction:.2f}"] = {
            "seed_count": len(members),
            "summary": aggregate,
            "components": component_rows,
            "tree_outputs": tree_rows,
        }
    return output


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _provenance(
    dataset_directory: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    files = (
        "dataset_manifest.json",
        "feature_schema.json",
        "tensors.npz",
        "selection_observations.jsonl.gz",
        "cases.jsonl.gz",
    )
    code_files = {
        "tools/run_phase0_learning_diagnostics.py": _sha256_file(
            Path(__file__).resolve()
        ),
        "damage_engine/system_learning.py": _sha256_file(
            ROOT / "damage_engine" / "system_learning.py"
        ),
    }
    return {
        "config_sha256": _sha256_bytes(_canonical_json_bytes(config)),
        "code_sha256": _sha256_bytes(_canonical_json_bytes(code_files)),
        "code_files": code_files,
        "git_commit": _git_commit(),
        "data_files": {
            name: _sha256_file(dataset_directory / name) for name in files
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": _import_torch().__version__,
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _freeze_config(output_directory: Path, config: Mapping[str, Any]) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / "config.json"
    encoded = json.dumps(
        config, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise RuntimeError("existing preregistered config differs from requested config")
    if not path.exists():
        path.write_text(encoded, encoding="utf-8")
    return path


def _markdown_summary(report: Mapping[str, Any]) -> str:
    lines = [
        "# Phase-0 multi-task learning diagnostics",
        "",
        "This report is a development-only software-teacher diagnostic. It does not read a sealed test split.",
        "",
        "| feature view | train fraction | support AP | tree soft AP | domain AP | severity RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for view, fractions in report["learning_curve"].items():
        for fraction, record in fractions.items():
            summary = record["summary"]

            def cell(name: str) -> str:
                value = summary[name]
                if value["mean"] is None:
                    return "undefined"
                return f"{value['mean']:.4f} ± {value['std']:.4f}"

            lines.append(
                f"| {view} | {fraction} | "
                f"{cell('support_macro_average_precision')} | "
                f"{cell('tree_macro_soft_average_precision')} | "
                f"{cell('domain_average_precision')} | "
                f"{cell('severity_macro_rmse')} |"
            )
    lines.extend(
        [
            "",
            "Thresholds, training hyperparameters, seeds, hashes, masks, and leakage checks are recorded in `config.json` and `results.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    dataset_directory: Path,
    output_directory: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_config(config)
    if (output_directory / "results.json").exists():
        raise RuntimeError("results.json already exists; use a new experiment directory")
    dataset = load_phase0_dataset(dataset_directory)
    arrays = dataset["arrays"]
    views, normalization = feature_views(dataset)
    reference = load_reference_memberships(
        dataset_directory / "selection_observations.jsonl.gz",
        arrays["candidate_group_id"],
        arrays["split_code"],
        strategy=str(config["data_policy"]["evaluation_strategy"]),
    )
    checks = leakage_checks(arrays, reference)
    component_names = tuple(map(str, dataset["schema"]["component_ids"]))
    tree_names = tuple(map(str, dataset["schema"]["tree_output_ids"]))
    baselines = _baseline_context(
        arrays, reference, component_names, tree_names
    )
    runs = []
    for feature_name in config["training"]["feature_views"]:
        feature = views[str(feature_name)]
        for fraction in map(float, config["training"]["train_fractions"]):
            for seed in map(int, config["training"]["seeds"]):
                selected = nested_train_indices(
                    arrays["split_code"], fraction=fraction, seed=seed
                )
                model, epoch_losses = train_model(
                    feature, arrays, selected, config, seed=seed
                )
                evaluation = evaluate_model(
                    model,
                    feature,
                    arrays,
                    reference,
                    baselines,
                    component_names,
                    tree_names,
                    config,
                )
                runs.append(
                    {
                        "feature_view": feature_name,
                        "train_fraction": fraction,
                        "seed": seed,
                        "train_case_count": int(selected.size),
                        "train_response_complete_count": int(
                            np.count_nonzero(arrays["response_mask"][selected])
                        ),
                        "selection_sha256": _sha256_bytes(
                            np.ascontiguousarray(selected, dtype=np.int64).tobytes()
                        ),
                        "final_epoch_loss": epoch_losses[-1],
                        "evaluation": evaluation,
                    }
                )
    report = {
        "schema_version": "phase0-learning-diagnostics-results@1.0.0",
        "status": "completed",
        "evidence_scope": config["evidence_scope"],
        "config": deepcopy(config),
        "provenance": _provenance(dataset_directory, config),
        "normalization": normalization,
        "leakage_checks": checks,
        "reference_baselines": {
            "domain": baselines["domain"],
            "support": baselines["support"],
            "tree": baselines["tree"],
            "severity": np.asarray(baselines["severity"], dtype=float).tolist(),
            "severity_positive": np.asarray(
                baselines["severity_positive"], dtype=float
            ).tolist(),
        },
        "runs": runs,
        "learning_curve": aggregate_runs(runs),
    }
    _write_json(output_directory / "results.json", report)
    (output_directory / "README.md").write_text(
        _markdown_summary(report), encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run preregistered Phase-0 multi-task learning diagnostics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset_directory",
        type=Path,
        nargs="?",
        default=DEFAULT_DATASET_DIRECTORY,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="freeze config.json and exit before reading labels or training",
    )
    args = parser.parse_args(argv)
    config = (
        deepcopy(DEFAULT_CONFIG)
        if args.config is None
        else json.loads(args.config.read_text(encoding="utf-8"))
    )
    _validate_config(config)
    output_directory = args.output_directory.resolve()
    config_path = _freeze_config(output_directory, config)
    if args.config_only:
        print(
            json.dumps(
                {
                    "status": "config_frozen",
                    "config": str(config_path),
                    "config_sha256": _sha256_bytes(_canonical_json_bytes(config)),
                },
                ensure_ascii=False,
            )
        )
        return 0
    report = run(args.dataset_directory.resolve(), output_directory, config)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output_directory / "results.json"),
                "run_count": len(report["runs"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
