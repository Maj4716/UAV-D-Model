"""Validation-only learning curves for the dataset-v2 hurdle contract."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np

from .baseline_task_admission import CONDITIONAL_TASKS, load_baseline_task_admission
from .dataset_v2 import load_dataset_v2_core, validate_dataset_v2
from .distribution_metrics import weighted_brier, weighted_calibration
from .errors import InputValidationError
from .hurdle_learning import (
    masked_binomial_nll_from_logits,
    masked_heteroscedastic_gaussian_nll,
    masked_interval_squared_loss,
    ordered_state_probabilities,
)
from .target_encoding import (
    fit_target_encoder,
    inverse_target,
    project_encoded_prediction,
    transform_target,
)
from .utils import content_hash, file_hash


BASELINE_CONTRACT_VERSION = "dataset-v2-hurdle-baseline@1.2.0"
TARGET_ENCODER_METHODS = frozenset(
    {"identity", "box_cox", "empirical_quantile_normal"}
)
DEFAULT_ENCODER_ABLATION_BUNDLES: dict[str, dict[str, str]] = {
    "identity": {"energy": "identity", "hit_count": "identity"},
    "box_cox": {"energy": "box_cox", "hit_count": "box_cox"},
    "quantile_normal": {
        "energy": "empirical_quantile_normal",
        "hit_count": "empirical_quantile_normal",
    },
    "section9_hybrid": {
        "energy": "empirical_quantile_normal",
        "hit_count": "box_cox",
    },
}


def _validate_target_encoder_method(method: str, *, task: str) -> str:
    value = str(method)
    if value == "log1p":
        raise InputValidationError(
            "conditional hurdle targets are already defined on a log1p scale; "
            "a second log1p encoder is not admissible",
            context={"task": task},
        )
    if value not in TARGET_ENCODER_METHODS:
        raise InputValidationError(
            "unsupported hurdle target encoder method",
            context={"task": task, "method": value},
        )
    return value


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise InputValidationError("PyTorch is required for the hurdle baseline") from exc
    return torch


def _require_extended_labels(labels: Mapping[str, np.ndarray]) -> None:
    required = {
        "p_hit_success_count",
        "p_hit_trial_count",
        "p_hit_identification_interval",
        "p_degraded_identification_interval",
        "p_failed_identification_interval",
        "conditional_log1p_energy",
        "conditional_log1p_energy_std_error",
        "conditional_log1p_energy_mask",
        "conditional_log1p_hit_count",
        "conditional_log1p_hit_count_std_error",
        "conditional_log1p_hit_count_mask",
        "derived_support",
        "derived_support_mask",
    }
    missing = sorted(required - set(labels))
    if missing:
        raise InputValidationError(
            "dataset-v2 lacks extended hurdle targets",
            context={"missing": missing},
        )


def _fit_encoder_or_identity(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    method: str,
    task_name: str,
) -> tuple[dict[str, Any], bool]:
    if np.any(mask):
        return (
            fit_target_encoder(
                values,
                mask=mask,
                method=method,
                task_name=task_name,
                source_split="reference_train",
            ),
            True,
        )
    contract = fit_target_encoder(
        [0.0],
        method="identity",
        task_name=task_name,
        source_split="reference_train",
    )
    return contract, False


def _encoded_standard_error(
    values: np.ndarray,
    standard_error: np.ndarray,
    contract: Mapping[str, Any],
) -> np.ndarray:
    lower = np.maximum(0.0, values - standard_error)
    upper = values + standard_error
    return 0.5 * np.abs(
        transform_target(upper, contract) - transform_target(lower, contract)
    )


def _weighted_mean(value, mask, case_weight):
    torch = _torch()
    weight = mask.to(dtype=value.dtype) * case_weight[:, None]
    denominator = weight.sum()
    if float(denominator.detach().cpu()) <= 0.0:
        return None
    return (value * weight).sum() / denominator


def _weighted_average_precision(
    score: np.ndarray, outcome: np.ndarray, weight: np.ndarray
) -> float | None:
    if score.size < 2 or not (np.any(outcome == 0) and np.any(outcome == 1)):
        return None
    order = np.argsort(-score, kind="mergesort")
    y = outcome[order].astype(np.float64)
    w = weight[order].astype(np.float64)
    total_positive = float(np.sum(w * y))
    if total_positive <= 0.0:
        return None
    true_positive = np.cumsum(w * y)
    predicted_positive = np.cumsum(w)
    precision = true_positive / predicted_positive
    return float(np.sum(precision * w * y) / total_positive)


def _model_class(feature_count: int, component_count: int, hidden_size: int):
    torch = _torch()

    class HurdleNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = torch.nn.Sequential(
                torch.nn.Linear(feature_count, hidden_size),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.SiLU(),
            )
            self.domain = torch.nn.Linear(hidden_size, 1)
            self.head = torch.nn.Linear(hidden_size, 7 * component_count)

        def forward(self, x):
            hidden = self.trunk(x)
            raw = self.head(hidden).reshape(-1, component_count, 7)
            return {
                "domain_logit": self.domain(hidden).squeeze(-1),
                "p_hit_logit": raw[..., 0],
                "energy_mean": raw[..., 1],
                "energy_raw_variance": raw[..., 2],
                "hit_count_mean": raw[..., 3],
                "hit_count_raw_variance": raw[..., 4],
                "degraded_logit": raw[..., 5],
                "conditional_failed_logit": raw[..., 6],
            }

    return HurdleNetwork


def _tensor_bundle(
    arrays: Mapping[str, np.ndarray], indices: np.ndarray, device: str
) -> dict[str, Any]:
    torch = _torch()
    labels = arrays["labels"]
    result: dict[str, Any] = {
        "x": torch.as_tensor(arrays["x"][indices], dtype=torch.float32, device=device),
        "weight": torch.as_tensor(
            arrays["weight"][indices], dtype=torch.float32, device=device
        ),
    }
    for name, value in labels.items():
        if value.dtype.kind in {"U", "S", "O"} or value.shape[0] != arrays["x"].shape[0]:
            continue
        if value.dtype.kind == "b":
            dtype = torch.bool
        else:
            dtype = torch.float32
        result[name] = torch.as_tensor(value[indices], dtype=dtype, device=device)
    return result


def _loss(prediction, batch, encoders, task_available, task_admission):
    torch = _torch()
    design = batch["weight"]
    legacy_mask = batch["label_mask"]
    raw_p_hit_mask = batch.get("p_hit_mask", legacy_mask)
    raw_degraded_mask = batch.get("p_degraded_mask", legacy_mask)
    raw_failed_mask = batch.get("p_failed_mask", legacy_mask)
    p_hit_active = task_admission["p_hit"][None, :]
    degraded_active = task_admission["p_degraded"][None, :]
    failed_active = task_admission["p_failed"][None, :]
    p_hit_mask = raw_p_hit_mask & p_hit_active
    degraded_mask = raw_degraded_mask & degraded_active
    failed_mask = raw_failed_mask & failed_active
    losses: dict[str, Any] = {}
    losses["p_hit_binomial"] = masked_binomial_nll_from_logits(
        prediction["p_hit_logit"],
        batch["p_hit_success_count"],
        batch["p_hit_trial_count"],
        p_hit_mask,
        design_weight=design,
    )
    probability = torch.sigmoid(prediction["p_hit_logit"])
    interval_mask = (~raw_p_hit_mask) & p_hit_active
    if torch.any(interval_mask):
        losses["p_hit_interval"] = 0.25 * masked_interval_squared_loss(
            probability,
            batch["p_hit_identification_interval"][..., 0],
            batch["p_hit_identification_interval"][..., 1],
            interval_mask,
            design_weight=design,
        )
    degraded, failed = ordered_state_probabilities(
        prediction["degraded_logit"], prediction["conditional_failed_logit"]
    )
    degraded_raw = torch.nn.functional.binary_cross_entropy(
        degraded, batch["p_degraded"], reduction="none"
    )
    failed_raw = torch.nn.functional.binary_cross_entropy(
        failed, batch["p_failed"], reduction="none"
    )
    losses["p_degraded"] = _weighted_mean(
        degraded_raw, degraded_mask, design
    )
    losses["p_failed"] = _weighted_mean(failed_raw, failed_mask, design)
    degraded_interval_mask = (~raw_degraded_mask) & degraded_active
    failed_interval_mask = (~raw_failed_mask) & failed_active
    if torch.any(degraded_interval_mask) or torch.any(failed_interval_mask):
        degraded_interval_loss = (
            masked_interval_squared_loss(
                degraded,
                batch["p_degraded_identification_interval"][..., 0],
                batch["p_degraded_identification_interval"][..., 1],
                degraded_interval_mask,
                design_weight=design,
            )
            if torch.any(degraded_interval_mask)
            else None
        )
        failed_interval_loss = (
            masked_interval_squared_loss(
                failed,
                batch["p_failed_identification_interval"][..., 0],
                batch["p_failed_identification_interval"][..., 1],
                failed_interval_mask,
                design_weight=design,
            )
            if torch.any(failed_interval_mask)
            else None
        )
        interval_terms = [
            value
            for value in (degraded_interval_loss, failed_interval_loss)
            if value is not None
        ]
        losses["ordered_state_intervals"] = 0.25 * sum(interval_terms)
    domain_complete = torch.ones(
        raw_p_hit_mask.shape[0], dtype=torch.bool, device=raw_p_hit_mask.device
    )
    for observed, active in (
        (raw_p_hit_mask, p_hit_active),
        (raw_degraded_mask, degraded_active),
        (raw_failed_mask, failed_active),
    ):
        domain_complete = domain_complete & torch.all(observed | ~active, dim=1)
    domain_target = domain_complete.to(
        dtype=prediction["domain_logit"].dtype
    )
    losses["domain"] = torch.nn.functional.binary_cross_entropy_with_logits(
        prediction["domain_logit"], domain_target, weight=design
    )
    for public, mean_name, variance_name, contract_name in (
        (
            "conditional_log1p_energy",
            "energy_mean",
            "energy_raw_variance",
            "energy",
        ),
        (
            "conditional_log1p_hit_count",
            "hit_count_mean",
            "hit_count_raw_variance",
            "hit_count",
        ),
    ):
        if not task_available[contract_name]:
            continue
        component_active = task_admission.get(public, task_admission["p_hit"])
        task_mask = batch[public + "_mask"]
        if component_active is not None:
            task_mask = task_mask & component_active[None, :]
        if torch.any(task_mask):
            losses[public] = masked_heteroscedastic_gaussian_nll(
                prediction[mean_name],
                prediction[variance_name],
                batch[public + "_encoded"],
                task_mask,
                observation_variance=batch[public + "_encoded_se"] ** 2,
                design_weight=design,
            )
    finite = [value for value in losses.values() if value is not None]
    return sum(finite), losses


def _evaluation(
    model,
    batch,
    labels: Mapping[str, np.ndarray],
    indices: np.ndarray,
    encoders: Mapping[str, Mapping[str, Any]],
    task_admission: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    torch = _torch()
    model.eval()
    with torch.no_grad():
        prediction = model(batch["x"])
        p_hit = torch.sigmoid(prediction["p_hit_logit"]).cpu().numpy()
        degraded, failed = ordered_state_probabilities(
            prediction["degraded_logit"], prediction["conditional_failed_logit"]
        )
        degraded = degraded.cpu().numpy()
        failed = failed.cpu().numpy()
        domain = torch.sigmoid(prediction["domain_logit"]).cpu().numpy()
        energy_encoded = prediction["energy_mean"].cpu().numpy()
        hit_encoded = prediction["hit_count_mean"].cpu().numpy()
    legacy_mask = labels["label_mask"][indices]
    raw_probability_masks = {
        target: labels.get(target + "_mask", labels["label_mask"])[indices]
        for target in ("p_hit", "p_degraded", "p_failed")
    }
    probability_masks = {
        target: raw_probability_masks[target] & task_admission[target][None, :]
        for target in raw_probability_masks
    }
    case_weight = batch["weight"].cpu().numpy()
    flat_weight = np.repeat(case_weight[:, None], legacy_mask.shape[1], axis=1)

    def probability_metrics(
        predicted: np.ndarray, observed: np.ndarray, selected: np.ndarray
    ) -> dict[str, Any]:
        if not np.any(selected):
            return {"observation_count": 0, "brier": None, "ece": None}
        calibration = weighted_calibration(
            predicted[selected],
            observed[selected],
            weight=flat_weight[selected],
        )
        return {
            "observation_count": int(np.sum(selected)),
            "brier": calibration["brier"],
            "ece": calibration["ece"],
            "calibration_bins": calibration["bins"],
        }

    domain_target = np.ones(indices.size, dtype=bool)
    for target, observed in raw_probability_masks.items():
        domain_target &= np.all(
            observed | ~task_admission[target][None, :], axis=1
        )
    domain_target = domain_target.astype(np.float64)
    domain_calibration = weighted_calibration(
        domain, domain_target, weight=case_weight
    )
    result: dict[str, Any] = {
        "case_count": int(indices.size),
        "p_hit": probability_metrics(
            p_hit,
            labels["p_hit"][indices],
            probability_masks["p_hit"],
        ),
        "p_degraded": probability_metrics(
            degraded,
            labels["p_degraded"][indices],
            probability_masks["p_degraded"],
        ),
        "p_failed": probability_metrics(
            failed,
            labels["p_failed"][indices],
            probability_masks["p_failed"],
        ),
        "domain": {
            "brier": domain_calibration["brier"],
            "ece": domain_calibration["ece"],
            "complete_rate": float(np.average(domain_target, weights=case_weight)),
        },
        "ordered_state_max_violation": float(np.max(failed - degraded, initial=0.0)),
        "p_hit_monte_carlo_noise_floor_mse": (
            float(
                np.average(
                    labels["p_hit_std_error"][indices][probability_masks["p_hit"]]
                    ** 2,
                    weights=flat_weight[probability_masks["p_hit"]],
                )
            )
            if np.any(probability_masks["p_hit"])
            else None
        ),
    }
    for public, predicted_encoded, encoder_name in (
        ("conditional_log1p_energy", energy_encoded, "energy"),
        ("conditional_log1p_hit_count", hit_encoded, "hit_count"),
    ):
        task_mask = labels[public + "_mask"][indices]
        component_active = task_admission.get(public, task_admission["p_hit"])
        if component_active is not None:
            task_mask = task_mask & component_active[None, :]
        if np.any(task_mask):
            projected_encoded = project_encoded_prediction(
                predicted_encoded, encoders[encoder_name]
            )
            original_prediction = inverse_target(
                projected_encoded, encoders[encoder_name]
            )
            error = np.abs(original_prediction - labels[public][indices])
            projection_mask = projected_encoded != predicted_encoded
            result[public] = {
                "observation_count": int(np.sum(task_mask)),
                "mae_original_log1p_scale": float(
                    np.average(error[task_mask], weights=flat_weight[task_mask])
                ),
                "encoded_prediction_projection_count": int(
                    np.sum(projection_mask & task_mask)
                ),
                "encoded_prediction_projection_rate": float(
                    np.mean(projection_mask[task_mask])
                ),
                "prediction_projection_policy": (
                    encoders[encoder_name]["parameters"].get(
                        "prediction_projection", "none"
                    )
                ),
            }
        else:
            result[public] = {
                "observation_count": 0,
                "mae_original_log1p_scale": None,
                "encoded_prediction_projection_count": 0,
                "encoded_prediction_projection_rate": None,
                "prediction_projection_policy": (
                    encoders[encoder_name]["parameters"].get(
                        "prediction_projection", "none"
                    )
                ),
            }
    support_mask = labels["derived_support_mask"][indices] & task_admission[
        "p_hit"
    ][None, :]
    per_component = []
    for column in range(legacy_mask.shape[1]):
        selected = support_mask[:, column]
        value = _weighted_average_precision(
            p_hit[selected, column],
            labels["derived_support"][indices][selected, column],
            case_weight[selected],
        )
        per_component.append(value)
    available_ap = [value for value in per_component if value is not None]
    result["derived_support_pr_auc"] = {
        "estimable_component_count": len(available_ap),
        "component_count": len(per_component),
        "macro": float(np.mean(available_ap)) if available_ap else None,
        "values": per_component,
        "role": "derived_diagnostic_not_primary_target",
        "component_task_scope": "inherits_p_hit_admission",
    }
    return result


def run_hurdle_learning_curves(
    dataset_directory: str | Path,
    output_directory: str | Path,
    *,
    fractions: Sequence[float] = (0.25, 0.50, 1.0),
    seeds: Sequence[int] = (17, 29, 43),
    epochs: int = 120,
    learning_rate: float = 3.0e-3,
    hidden_size: int = 96,
    device: str = "cpu",
    energy_encoder_method: str = "empirical_quantile_normal",
    hit_count_encoder_method: str = "box_cox",
    task_admission_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train with train labels and select only on frozen Reference validation."""

    torch = _torch()
    # Small research pilots are dominated by thread-pool overhead at the
    # default core count.  One BLAS thread is faster and deterministic here.
    torch.set_num_threads(1)
    if epochs < 1 or learning_rate <= 0.0 or hidden_size < 4:
        raise InputValidationError("invalid hurdle baseline optimization settings")
    if not fractions or any(not 0.0 < float(value) <= 1.0 for value in fractions):
        raise InputValidationError("learning-curve fractions must lie in (0,1]")
    if len(set(int(seed) for seed in seeds)) != len(seeds):
        raise InputValidationError("hurdle baseline seeds must be unique")
    encoder_methods = {
        "energy": _validate_target_encoder_method(
            energy_encoder_method, task="conditional_log1p_energy"
        ),
        "hit_count": _validate_target_encoder_method(
            hit_count_encoder_method, task="conditional_log1p_hit_count"
        ),
    }
    dataset = Path(dataset_directory)
    validation = validate_dataset_v2(dataset, recompute_views=False)
    seal_path = dataset / "sealed_test.json"
    if not seal_path.is_file():
        raise InputValidationError("dataset-v2 lacks sealed_test.json")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("test_open_count") != 0:
        raise InputValidationError("sealed test has already been opened")
    core = load_dataset_v2_core(dataset)
    inputs = core["inputs"]
    labels = core["component_labels"]
    weights = core["sampling_weights"]
    _require_extended_labels(labels)
    task_admission_masks, task_admission_contract = load_baseline_task_admission(
        task_admission_path,
        dataset_validation=validation,
        component_ids=labels["component_ids"],
    )
    torch_task_admission = {
        task: torch.as_tensor(mask, dtype=torch.bool, device=device)
        for task, mask in task_admission_masks.items()
    }
    splits = weights["split"].astype(str)
    train_indices = np.flatnonzero(splits == "train")
    validation_indices = np.flatnonzero(splits == "validation")
    test_indices = np.flatnonzero(splits == "test")
    if not train_indices.size or not validation_indices.size or not test_indices.size:
        raise InputValidationError("hurdle baseline requires train, validation and sealed test")
    x = inputs["x"].astype(np.float64)
    input_mask = inputs.get("input_mask", np.ones_like(x, dtype=bool))
    if not np.all(input_mask[train_indices]) or not np.all(input_mask[validation_indices]):
        raise InputValidationError("hurdle baseline requires complete model inputs")
    train_mean = np.mean(x[train_indices], axis=0)
    train_scale = np.std(x[train_indices], axis=0)
    train_scale[train_scale < 1.0e-8] = 1.0
    normalized = (x - train_mean) / train_scale
    arrays = {
        "x": normalized,
        "weight": weights["design_weight"].astype(np.float64),
        "labels": labels,
    }
    output = Path(output_directory)
    if output.exists():
        raise InputValidationError(
            "hurdle baseline output directory already exists",
            context={"path": str(output)},
        )
    output.mkdir(parents=True)
    with (output / "normalization.npz").open("wb") as stream:
        np.savez_compressed(
            stream,
            feature_names=inputs["feature_names"],
            train_mean=train_mean,
            train_scale=train_scale,
        )

    runs = []
    best: tuple[float, dict[str, Any], Any, dict[str, Any]] | None = None
    for fraction in fractions:
        subset_count = max(1, int(math.ceil(train_indices.size * float(fraction))))
        for seed in seeds:
            random.seed(int(seed))
            np.random.seed(int(seed))
            torch.manual_seed(int(seed))
            generator = np.random.default_rng(int(seed))
            subset = np.sort(generator.permutation(train_indices)[:subset_count])
            encoders = {}
            task_available = {}
            for encoder_name, public in (
                ("energy", "conditional_log1p_energy"),
                ("hit_count", "conditional_log1p_hit_count"),
            ):
                contract, available = _fit_encoder_or_identity(
                    labels[public][subset],
                    labels[public + "_mask"][subset]
                    & task_admission_masks.get(
                        public,
                        task_admission_masks["p_hit"],
                    )[None, :],
                    method=encoder_methods[encoder_name],
                    task_name=public,
                )
                encoders[encoder_name] = contract
                task_available[encoder_name] = available
            encoded_arrays = {name: np.asarray(value).copy() for name, value in labels.items()}
            for encoder_name, public in (
                ("energy", "conditional_log1p_energy"),
                ("hit_count", "conditional_log1p_hit_count"),
            ):
                encoded_arrays[public + "_encoded"] = transform_target(
                    labels[public], encoders[encoder_name]
                )
                encoded_arrays[public + "_encoded_se"] = _encoded_standard_error(
                    labels[public], labels[public + "_std_error"], encoders[encoder_name]
                )
            run_arrays = {**arrays, "labels": encoded_arrays}
            train_batch = _tensor_bundle(run_arrays, subset, device)
            validation_batch = _tensor_bundle(run_arrays, validation_indices, device)
            model = _model_class(x.shape[1], labels["component_ids"].size, hidden_size)().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            best_epoch = 0
            best_state = None
            best_score = math.inf
            patience = max(20, epochs // 4)
            stale = 0
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                prediction = model(train_batch["x"])
                loss, _ = _loss(
                    prediction,
                    train_batch,
                    encoders,
                    task_available,
                    torch_task_admission,
                )
                loss.backward()
                optimizer.step()
                if epoch % 5 == 0 or epoch == epochs - 1:
                    metrics = _evaluation(
                        model,
                        validation_batch,
                        labels,
                        validation_indices,
                        encoders,
                        task_admission_masks,
                    )
                    score = metrics["p_hit"]["brier"]
                    if score is not None and score < best_score - 1.0e-10:
                        best_score = float(score)
                        best_epoch = epoch
                        best_state = {
                            key: value.detach().cpu().clone()
                            for key, value in model.state_dict().items()
                        }
                        stale = 0
                    else:
                        stale += 5
                    if stale >= patience:
                        break
            if best_state is None:
                raise InputValidationError("validation contains no observable p_hit labels")
            model.load_state_dict(best_state)
            metrics = _evaluation(
                model,
                validation_batch,
                labels,
                validation_indices,
                encoders,
                task_admission_masks,
            )
            run_id = f"fraction-{float(fraction):.2f}-seed-{int(seed)}"
            run_path = output / f"{run_id}.pt"
            torch.save(best_state, run_path)
            run = {
                "run_id": run_id,
                "train_fraction": float(fraction),
                "train_case_count": int(subset.size),
                "seed": int(seed),
                "best_epoch": int(best_epoch),
                "validation_metrics": metrics,
                "target_encoders": encoders,
                "task_available": task_available,
                "model_state_sha256": file_hash(run_path),
            }
            runs.append(run)
            if math.isclose(float(fraction), 1.0) and (
                best is None or metrics["p_hit"]["brier"] < best[0]
            ):
                best = (float(metrics["p_hit"]["brier"]), run, best_state, encoders)
    if best is None:
        raise InputValidationError("learning curves require a 100% training fraction")
    best_score, selected_run, selected_state, selected_encoders = best
    selected_path = output / "frozen_model.pt"
    torch.save(selected_state, selected_path)
    encoder_document = {
        "schema_version": "hurdle-target-encoder-bundle@1.0.0",
        "fit_scope": "selected_reference_train_subset_only",
        "evaluation_scale": "original_physical_label_scale",
        "requested_methods": encoder_methods,
        "label_scale_note": (
            "conditional targets are already physical log1p estimands; "
            "no additional log1p encoder is applied"
        ),
        "encoders": selected_encoders,
    }
    encoder_document["bundle_hash"] = content_hash(encoder_document)
    (output / "target_encoders.json").write_text(
        json.dumps(encoder_document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    frozen = {
        "schema_version": "frozen-hurdle-model@1.0.0",
        "baseline_contract": BASELINE_CONTRACT_VERSION,
        "dataset_id": validation["dataset_id"],
        "dataset_manifest_hash": validation["manifest_hash"],
        "selected_run_id": selected_run["run_id"],
        "selection_metric": "reference_validation.p_hit.weighted_brier",
        "selection_value": best_score,
        "model_state_sha256": file_hash(selected_path),
        "target_encoder_bundle_hash": encoder_document["bundle_hash"],
        "target_encoder_methods": encoder_methods,
        "support_probability_threshold": 0.05,
        "task_admission": task_admission_contract,
        "test_labels_evaluated": False,
        "test_open_count": 0,
    }
    frozen["frozen_contract_hash"] = content_hash(frozen)
    (output / "frozen_model.json").write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    raw_train_masks = {
        target: labels.get(target + "_mask", labels["label_mask"])[train_indices]
        for target in ("p_hit", "p_degraded", "p_failed")
    }
    train_mask = raw_train_masks["p_hit"] & task_admission_masks["p_hit"][None, :]
    train_nonzero = np.sum(
        (labels["p_hit"][train_indices] > 0.0) & train_mask,
        axis=0,
    )
    design_weight = weights["design_weight"][train_indices].astype(np.float64)
    ess = float(design_weight.sum() ** 2 / np.sum(design_weight**2))
    full_runs = [
        run for run in runs if math.isclose(run["train_fraction"], 1.0)
    ]
    full_brier = np.asarray(
        [run["validation_metrics"]["p_hit"]["brier"] for run in full_runs],
        dtype=np.float64,
    )
    train_complete = np.ones(train_indices.size, dtype=bool)
    for target, observed in raw_train_masks.items():
        train_complete &= np.all(
            observed | ~task_admission_masks[target][None, :], axis=1
        )
    admitted_train_nonzero = train_nonzero[task_admission_masks["p_hit"]]
    conditional_component_support: dict[str, Any] = {}
    conditional_support_gates: dict[str, bool] = {}
    for task in CONDITIONAL_TASKS:
        active = task_admission_masks[task]
        train_counts = np.sum(
            labels[task + "_mask"][train_indices] & active[None, :], axis=0
        )[active]
        validation_counts = np.sum(
            labels[task + "_mask"][validation_indices] & active[None, :], axis=0
        )[active]
        conditional_component_support[task] = {
            "admitted_component_count": int(np.sum(active)),
            "train_observation_count_per_admitted_component": {
                "minimum": int(np.min(train_counts)),
                "median": float(np.median(train_counts)),
                "maximum": int(np.max(train_counts)),
            },
            "validation_observation_count_per_admitted_component": {
                "minimum": int(np.min(validation_counts)),
                "median": float(np.median(validation_counts)),
                "maximum": int(np.max(validation_counts)),
            },
        }
        conditional_support_gates[
            f"{task}_minimum_20_train_observations_per_component"
        ] = bool(np.min(train_counts) >= 20)
        conditional_support_gates[
            f"{task}_minimum_5_validation_observations_per_component"
        ] = bool(np.min(validation_counts) >= 5)
    readiness_gates = {
        "minimum_train_cases_2000": bool(train_indices.size >= 2000),
        "minimum_domain_complete_rate_0p90": bool(
            np.mean(train_complete) >= 0.90
        ),
        "minimum_20_nonzero_cases_per_component": bool(
            np.min(admitted_train_nonzero) >= 20
        ),
        "three_seed_full_fraction": bool(len(full_runs) >= 3),
        "finite_validation_p_hit_brier": bool(
            full_brier.size > 0 and np.isfinite(full_brier).all()
        ),
        **conditional_support_gates,
    }
    readiness = {
        "formal_neural_training_ready": bool(all(readiness_gates.values())),
        "recommended_role": (
            "formal_training_candidate"
            if all(readiness_gates.values())
            else "pipeline_and_exploratory_model_development"
        ),
        "gates": readiness_gates,
        "scope": "frozen_per_task_component_admission_train_validation_only",
        "task_admission": task_admission_contract,
        "train_case_count": int(train_indices.size),
        "train_effective_sample_size": ess,
        "train_ess_fraction": float(ess / train_indices.size),
        "train_domain_complete_rate": float(np.mean(train_complete)),
        "p_hit_admitted_component_count": int(
            np.sum(task_admission_masks["p_hit"])
        ),
        "admitted_component_count_by_task": {
            task: int(np.sum(mask))
            for task, mask in task_admission_masks.items()
        },
        "conditional_component_support": conditional_component_support,
        "per_component_nonzero_p_hit_case_count": {
            "minimum": int(np.min(admitted_train_nonzero)),
            "median": float(np.median(admitted_train_nonzero)),
            "maximum": int(np.max(admitted_train_nonzero)),
        },
        "full_fraction_validation_p_hit_brier": {
            "mean": float(np.mean(full_brier)) if full_brier.size else None,
            "std": float(np.std(full_brier)) if full_brier.size else None,
        },
        "threshold_provenance": (
            "idea2 staged route: 2000-case learning-curve checkpoint, "
            "90% domain-complete engineering admission, and 20 observed "
            "nonzero p_hit cases per admitted component; conditional tasks "
            "require at least 20 Train and 5 Reference-validation observations "
            "for every task-specific admitted component"
        ),
    }
    result = {
        "schema_version": "hurdle-learning-curves-report@1.0.0",
        "baseline_contract": BASELINE_CONTRACT_VERSION,
        "dataset_id": validation["dataset_id"],
        "split_counts": {
            "train": int(train_indices.size),
            "validation": int(validation_indices.size),
            "test_sealed_unread": int(test_indices.size),
        },
        "fractions": [float(value) for value in fractions],
        "seeds": [int(value) for value in seeds],
        "target_encoder_methods": encoder_methods,
        "target_encoder_fit_scope": "each training subset only",
        "target_encoder_evaluation_scale": "original_physical_log1p_estimand_scale",
        "task_admission": task_admission_contract,
        "runs": runs,
        "selected_frozen_model": frozen,
        "training_readiness": readiness,
        "test_access_policy": {
            "labels_used_in_loss": False,
            "labels_used_in_metrics": False,
            "labels_used_in_selection": False,
            "open_count": 0,
        },
    }
    (output / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def run_target_encoder_ablation(
    dataset_directory: str | Path,
    output_directory: str | Path,
    *,
    bundles: Mapping[str, Mapping[str, str]] | None = None,
    fractions: Sequence[float] = (0.25, 0.50, 1.0),
    seeds: Sequence[int] = (17, 29, 43),
    epochs: int = 120,
    learning_rate: float = 3.0e-3,
    hidden_size: int = 96,
    device: str = "cpu",
    task_admission_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare train-only target encoders without opening the sealed Test split."""

    output = Path(output_directory)
    if output.exists():
        raise InputValidationError(
            "target encoder ablation output directory already exists",
            context={"path": str(output)},
        )
    configured = DEFAULT_ENCODER_ABLATION_BUNDLES if bundles is None else bundles
    if not configured:
        raise InputValidationError("target encoder ablation requires at least one bundle")
    normalized: dict[str, dict[str, str]] = {}
    for raw_name, raw_bundle in configured.items():
        name = str(raw_name)
        if not name or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in name.lower()):
            raise InputValidationError("target encoder bundle names must be filesystem-safe")
        if set(raw_bundle) != {"energy", "hit_count"}:
            raise InputValidationError(
                "target encoder bundle must define energy and hit_count",
                context={"bundle": name},
            )
        normalized[name] = {
            "energy": _validate_target_encoder_method(
                str(raw_bundle["energy"]), task="conditional_log1p_energy"
            ),
            "hit_count": _validate_target_encoder_method(
                str(raw_bundle["hit_count"]), task="conditional_log1p_hit_count"
            ),
        }

    output.mkdir(parents=True)
    candidates: list[dict[str, Any]] = []
    for name, methods in normalized.items():
        candidate_output = output / name
        try:
            result = run_hurdle_learning_curves(
                dataset_directory,
                candidate_output,
                fractions=fractions,
                seeds=seeds,
                epochs=epochs,
                learning_rate=learning_rate,
                hidden_size=hidden_size,
                device=device,
                energy_encoder_method=methods["energy"],
                hit_count_encoder_method=methods["hit_count"],
                task_admission_path=task_admission_path,
            )
        except InputValidationError as exc:
            candidates.append(
                {
                    "bundle_id": name,
                    "methods": methods,
                    "status": "failed_closed",
                    "error": exc.to_dict(),
                }
            )
            continue
        full_runs = [
            run
            for run in result["runs"]
            if math.isclose(float(run["train_fraction"]), 1.0)
        ]

        def aggregate(path: tuple[str, ...]) -> dict[str, float | None]:
            observed: list[float] = []
            for run in full_runs:
                value: Any = run["validation_metrics"]
                for key in path:
                    value = value[key]
                if value is not None:
                    observed.append(float(value))
            return {
                "mean": float(np.mean(observed)) if observed else None,
                "std": float(np.std(observed)) if observed else None,
            }

        candidates.append(
            {
                "bundle_id": name,
                "methods": methods,
                "status": "completed",
                "dataset_id": result["dataset_id"],
                "dataset_manifest_hash": result["selected_frozen_model"][
                    "dataset_manifest_hash"
                ],
                "full_fraction_seed_count": len(full_runs),
                "validation_p_hit_brier": aggregate(("p_hit", "brier")),
                "validation_energy_mae_original_log1p_scale": aggregate(
                    ("conditional_log1p_energy", "mae_original_log1p_scale")
                ),
                "validation_hit_count_mae_original_log1p_scale": aggregate(
                    ("conditional_log1p_hit_count", "mae_original_log1p_scale")
                ),
                "results_sha256": file_hash(candidate_output / "results.json"),
                "frozen_model_sha256": result["selected_frozen_model"][
                    "model_state_sha256"
                ],
                "target_encoder_bundle_hash": result["selected_frozen_model"][
                    "target_encoder_bundle_hash"
                ],
                "test_open_count": result["test_access_policy"]["open_count"],
            }
        )
    successful = [
        candidate
        for candidate in candidates
        if candidate["status"] == "completed"
        and candidate["validation_p_hit_brier"]["mean"] is not None
    ]
    if not successful:
        raise InputValidationError("all target encoder ablation bundles failed")
    selected = min(
        successful,
        key=lambda candidate: (
            float(candidate["validation_p_hit_brier"]["mean"]),
            str(candidate["bundle_id"]),
        ),
    )
    report: dict[str, Any] = {
        "schema_version": "hurdle-target-encoder-ablation@1.0.0",
        "baseline_contract": BASELINE_CONTRACT_VERSION,
        "selection_scope": "reference_validation_only",
        "selection_metric": "mean_full_fraction_seed_p_hit_weighted_brier",
        "selected_bundle_id": selected["bundle_id"],
        "dataset_id": selected["dataset_id"],
        "dataset_manifest_hash": selected["dataset_manifest_hash"],
        "fractions": [float(value) for value in fractions],
        "seeds": [int(value) for value in seeds],
        "task_admission_path": (
            str(Path(task_admission_path))
            if task_admission_path is not None
            else None
        ),
        "candidate_bundles": candidates,
        "label_scale_note": (
            "both conditional targets are already log1p physical estimands; "
            "reported MAE is on that archived original scale"
        ),
        "test_access_policy": {
            "labels_used_in_loss": False,
            "labels_used_in_metrics": False,
            "labels_used_in_selection": False,
            "open_count": 0,
        },
    }
    report["report_hash"] = content_hash(report)
    (output / "ablation_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = [
    "BASELINE_CONTRACT_VERSION",
    "DEFAULT_ENCODER_ABLATION_BUNDLES",
    "TARGET_ENCODER_METHODS",
    "run_hurdle_learning_curves",
    "run_target_encoder_ablation",
]
