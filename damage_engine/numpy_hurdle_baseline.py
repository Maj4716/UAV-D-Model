"""Pure-NumPy p_hit neural baseline for sealed dataset-v2 development packages.

The module is deliberately narrow: it trains only the component hit-probability
head, uses train-only normalization, selects epochs/runs on Reference validation,
and never evaluates the sealed test split.  Epsilon-complete targets receive a
soft Bernoulli loss; informative wider identification intervals receive a
squared interval-hinge loss.  A vacuous [0, 1] interval contributes no gradient.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .baseline_task_admission import load_baseline_task_admission
from .dataset_v2 import load_dataset_v2_core, validate_dataset_v2
from .distribution_metrics import weighted_calibration
from .errors import InputValidationError
from .utils import content_hash, file_hash


NUMPY_BASELINE_SCHEMA_VERSION = "numpy-p-hit-learning-curves@1.1.0"


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _initialize_parameters(
    feature_count: int,
    hidden_size: int,
    component_count: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    first_scale = math.sqrt(2.0 / max(1, feature_count + hidden_size))
    second_scale = math.sqrt(2.0 / max(1, hidden_size + component_count))
    return {
        "w1": rng.normal(0.0, first_scale, size=(feature_count, hidden_size)),
        "b1": np.zeros(hidden_size, dtype=np.float64),
        "w2": rng.normal(0.0, second_scale, size=(hidden_size, component_count)),
        "b2": np.zeros(component_count, dtype=np.float64),
    }


def _forward(
    parameters: Mapping[str, np.ndarray], x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.tanh(x @ parameters["w1"] + parameters["b1"])
    probability = _sigmoid(hidden @ parameters["w2"] + parameters["b2"])
    return hidden, probability


def _loss_and_gradients(
    parameters: Mapping[str, np.ndarray],
    x: np.ndarray,
    target: np.ndarray,
    point_mask: np.ndarray,
    interval: np.ndarray,
    case_weight: np.ndarray,
    *,
    interval_loss_weight: float,
    l2_penalty: float,
) -> tuple[float, dict[str, np.ndarray], dict[str, float | int]]:
    hidden, probability = _forward(parameters, x)
    weights = case_weight[:, None]
    point_weight = weights * point_mask.astype(np.float64)
    point_mass = float(np.sum(point_weight))
    if point_mass <= 0.0:
        raise InputValidationError("NumPy baseline training subset has no point p_hit labels")

    clipped_probability = np.clip(probability, 1.0e-9, 1.0 - 1.0e-9)
    binary_cross_entropy = -(
        target * np.log(clipped_probability)
        + (1.0 - target) * np.log1p(-clipped_probability)
    )
    point_loss = float(np.sum(point_weight * binary_cross_entropy) / point_mass)
    logit_gradient = point_weight * (probability - target) / point_mass

    lower = interval[..., 0]
    upper = interval[..., 1]
    width = upper - lower
    informative_interval = (~point_mask) & (width < 1.0 - 1.0e-12)
    interval_weight = weights * informative_interval.astype(np.float64)
    interval_mass = float(np.sum(interval_weight))
    interval_loss = 0.0
    if interval_mass > 0.0 and interval_loss_weight > 0.0:
        residual = np.where(
            probability < lower,
            probability - lower,
            np.where(probability > upper, probability - upper, 0.0),
        )
        interval_loss = float(
            np.sum(interval_weight * residual**2) / interval_mass
        )
        probability_gradient = 2.0 * interval_weight * residual / interval_mass
        logit_gradient += (
            interval_loss_weight
            * probability_gradient
            * probability
            * (1.0 - probability)
        )

    l2_loss = 0.5 * l2_penalty * float(
        np.sum(parameters["w1"] ** 2) + np.sum(parameters["w2"] ** 2)
    )
    loss = point_loss + interval_loss_weight * interval_loss + l2_loss

    gradients: dict[str, np.ndarray] = {}
    gradients["w2"] = hidden.T @ logit_gradient + l2_penalty * parameters["w2"]
    gradients["b2"] = np.sum(logit_gradient, axis=0)
    hidden_gradient = (logit_gradient @ parameters["w2"].T) * (1.0 - hidden**2)
    gradients["w1"] = x.T @ hidden_gradient + l2_penalty * parameters["w1"]
    gradients["b1"] = np.sum(hidden_gradient, axis=0)
    diagnostics: dict[str, float | int] = {
        "point_loss": point_loss,
        "interval_loss": interval_loss,
        "l2_loss": l2_loss,
        "point_observation_count": int(np.sum(point_mask)),
        "informative_interval_count": int(np.sum(informative_interval)),
    }
    return loss, gradients, diagnostics


def _adam_update(
    parameters: dict[str, np.ndarray],
    gradients: Mapping[str, np.ndarray],
    first_moment: dict[str, np.ndarray],
    second_moment: dict[str, np.ndarray],
    *,
    step: int,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1.0e-8,
) -> None:
    for name in parameters:
        gradient = gradients[name]
        first_moment[name] = beta1 * first_moment[name] + (1.0 - beta1) * gradient
        second_moment[name] = (
            beta2 * second_moment[name] + (1.0 - beta2) * gradient**2
        )
        corrected_first = first_moment[name] / (1.0 - beta1**step)
        corrected_second = second_moment[name] / (1.0 - beta2**step)
        parameters[name] -= learning_rate * corrected_first / (
            np.sqrt(corrected_second) + epsilon
        )


def _split_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    point_mask: np.ndarray,
    interval: np.ndarray,
    case_weight: np.ndarray,
    component_ids: np.ndarray,
) -> dict[str, Any]:
    flat_weight = np.repeat(case_weight[:, None], target.shape[1], axis=1)
    if not np.any(point_mask):
        return {
            "point_observation_count": 0,
            "brier": None,
            "ece": None,
            "calibration_bins": [],
            "component_diagnostics": [],
            "binary_contrast_estimable_component_count": 0,
            "admission_evaluable_component_count": 0,
            "informative_interval_count": int(
                np.sum((~point_mask) & ((interval[..., 1] - interval[..., 0]) < 1.0 - 1.0e-12))
            ),
            "interval_violation_rate": None,
            "interval_squared_hinge": None,
        }
    calibration = weighted_calibration(
        probability[point_mask],
        target[point_mask],
        weight=flat_weight[point_mask],
    )
    component_rows: list[dict[str, Any]] = []
    binary_contrast_count = 0
    admission_count = 0
    for column, component_id in enumerate(component_ids.astype(str)):
        selected = point_mask[:, column]
        observed = target[selected, column]
        observed_count = int(observed.size)
        positive_count = int(np.sum(observed > 0.0))
        negative_count = observed_count - positive_count
        estimable = positive_count > 0 and negative_count > 0
        admission_evaluable = positive_count >= 5 and negative_count >= 5
        binary_contrast_count += int(estimable)
        admission_count += int(admission_evaluable)
        component_rows.append(
            {
                "component_id": component_id,
                "point_observation_count": observed_count,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "binary_contrast_estimable": estimable,
                "admission_evaluable_positive_and_negative_at_least_5": (
                    admission_evaluable
                ),
                "brier": (
                    float(
                        np.average(
                            (probability[selected, column] - observed) ** 2,
                            weights=case_weight[selected],
                        )
                    )
                    if observed_count
                    else None
                ),
            }
        )

    lower = interval[..., 0]
    upper = interval[..., 1]
    informative = (~point_mask) & ((upper - lower) < 1.0 - 1.0e-12)
    if np.any(informative):
        residual = np.where(
            probability < lower,
            lower - probability,
            np.where(probability > upper, probability - upper, 0.0),
        )
        interval_violation_rate = float(np.mean(residual[informative] > 0.0))
        interval_squared_hinge = float(
            np.average(
                residual[informative] ** 2,
                weights=flat_weight[informative],
            )
        )
    else:
        interval_violation_rate = None
        interval_squared_hinge = None
    return {
        "point_observation_count": int(np.sum(point_mask)),
        "brier": calibration["brier"],
        "ece": calibration["ece"],
        "calibration_bins": calibration["bins"],
        "component_diagnostics": component_rows,
        "binary_contrast_estimable_component_count": binary_contrast_count,
        "admission_evaluable_component_count": admission_count,
        "informative_interval_count": int(np.sum(informative)),
        "interval_violation_rate": interval_violation_rate,
        "interval_squared_hinge": interval_squared_hinge,
    }


def _fit_one(
    x_train: np.ndarray,
    y_train: np.ndarray,
    mask_train: np.ndarray,
    interval_train: np.ndarray,
    weight_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    mask_validation: np.ndarray,
    interval_validation: np.ndarray,
    weight_validation: np.ndarray,
    component_ids: np.ndarray,
    *,
    seed: int,
    hidden_size: int,
    epochs: int,
    learning_rate: float,
    interval_loss_weight: float,
    l2_penalty: float,
) -> dict[str, Any]:
    if not np.any(mask_validation):
        raise InputValidationError(
            "NumPy baseline validation split has no point-identified p_hit labels"
        )
    rng = np.random.default_rng(seed)
    parameters = _initialize_parameters(
        x_train.shape[1], hidden_size, y_train.shape[1], rng
    )
    first_moment = {name: np.zeros_like(value) for name, value in parameters.items()}
    second_moment = {name: np.zeros_like(value) for name, value in parameters.items()}
    best_parameters: dict[str, np.ndarray] | None = None
    best_brier = math.inf
    best_epoch = 0
    history: list[dict[str, Any]] = []
    last_training: dict[str, float | int] = {}
    for epoch in range(1, epochs + 1):
        loss, gradients, last_training = _loss_and_gradients(
            parameters,
            x_train,
            y_train,
            mask_train,
            interval_train,
            weight_train,
            interval_loss_weight=interval_loss_weight,
            l2_penalty=l2_penalty,
        )
        _adam_update(
            parameters,
            gradients,
            first_moment,
            second_moment,
            step=epoch,
            learning_rate=learning_rate,
        )
        _, validation_probability = _forward(parameters, x_validation)
        validation_metrics = _split_metrics(
            validation_probability,
            y_validation,
            mask_validation,
            interval_validation,
            weight_validation,
            component_ids,
        )
        brier = validation_metrics["brier"]
        if brier is not None and float(brier) < best_brier:
            best_brier = float(brier)
            best_epoch = epoch
            best_parameters = {
                name: np.asarray(value).copy() for name, value in parameters.items()
            }
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(loss),
                    "validation_brier": brier,
                    "validation_ece": validation_metrics["ece"],
                }
            )
    if best_parameters is None:
        raise InputValidationError("NumPy baseline could not select a validation epoch")
    _, train_probability = _forward(best_parameters, x_train)
    _, validation_probability = _forward(best_parameters, x_validation)
    return {
        "parameters": best_parameters,
        "best_epoch": best_epoch,
        "train_metrics": _split_metrics(
            train_probability,
            y_train,
            mask_train,
            interval_train,
            weight_train,
            component_ids,
        ),
        "validation_metrics": _split_metrics(
            validation_probability,
            y_validation,
            mask_validation,
            interval_validation,
            weight_validation,
            component_ids,
        ),
        "training_diagnostics": dict(last_training),
        "history": history,
    }


def _aggregate_runs(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in ("brier", "ece", "interval_squared_hinge"):
        values = [
            float(run["validation_metrics"][metric])
            for run in runs
            if run["validation_metrics"][metric] is not None
        ]
        result[metric] = {
            "mean": float(np.mean(values)) if values else None,
            "variance": float(np.var(values, ddof=0)) if values else None,
            "std": float(np.std(values, ddof=0)) if values else None,
            "count": len(values),
        }
    for metric in (
        "binary_contrast_estimable_component_count",
        "admission_evaluable_component_count",
    ):
        values = np.asarray(
            [run["validation_metrics"][metric] for run in runs],
            dtype=np.float64,
        )
        result[metric] = {
            "mean": float(np.mean(values)),
            "variance": float(np.var(values, ddof=0)),
            "std": float(np.std(values, ddof=0)),
            "minimum": int(np.min(values)),
            "maximum": int(np.max(values)),
            "count": int(values.size),
        }
    return result


def run_numpy_p_hit_learning_curves(
    dataset_directory: str | Path,
    output_directory: str | Path,
    *,
    fractions: Sequence[float] = (0.25, 0.50, 1.0),
    seeds: Sequence[int] = (17, 29, 43),
    epochs: int = 200,
    learning_rate: float = 1.0e-2,
    hidden_size: int = 64,
    interval_loss_weight: float = 0.25,
    l2_penalty: float = 1.0e-4,
    task_admission_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run learning curves without reading any sealed-test metric."""

    if epochs < 1 or hidden_size < 2 or learning_rate <= 0.0:
        raise InputValidationError("NumPy baseline optimization settings are invalid")
    if interval_loss_weight < 0.0 or l2_penalty < 0.0:
        raise InputValidationError("NumPy baseline loss weights must be nonnegative")
    normalized_fractions = tuple(float(value) for value in fractions)
    normalized_seeds = tuple(int(value) for value in seeds)
    if (
        not normalized_fractions
        or any(not 0.0 < value <= 1.0 for value in normalized_fractions)
        or len(set(normalized_fractions)) != len(normalized_fractions)
    ):
        raise InputValidationError("learning-curve fractions must be unique and lie in (0,1]")
    if not any(math.isclose(value, 1.0) for value in normalized_fractions):
        raise InputValidationError("learning curves must include the 100% training fraction")
    if not normalized_seeds or len(set(normalized_seeds)) != len(normalized_seeds):
        raise InputValidationError("NumPy baseline seeds must be nonempty and unique")

    dataset = Path(dataset_directory)
    output = Path(output_directory)
    if output.exists():
        raise InputValidationError(
            "NumPy baseline output directory already exists",
            context={"path": str(output)},
        )
    validation = validate_dataset_v2(dataset, recompute_views=False)
    seal_path = dataset / "sealed_test.json"
    if not seal_path.is_file():
        raise InputValidationError("dataset-v2 lacks sealed_test.json")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("test_open_count") != 0:
        raise InputValidationError("sealed test has already been opened")
    if int(seal.get("test_result_json_lines_decoded", 0)) != 0:
        raise InputValidationError("sealed test result rows have already been decoded")
    if seal.get("test_labels_present_in_core") is not False:
        raise InputValidationError(
            "sealed-test contract must declare opaque test labels outside dataset-v2 core"
        )

    core = load_dataset_v2_core(dataset)
    inputs = core["inputs"]
    labels = core["component_labels"]
    weights = core["sampling_weights"]
    splits = weights["split"].astype(str)
    train_indices = np.flatnonzero(splits == "train")
    validation_indices = np.flatnonzero(splits == "validation")
    test_indices = np.flatnonzero(splits == "test")
    if not train_indices.size or not validation_indices.size or not test_indices.size:
        raise InputValidationError(
            "NumPy baseline requires train, validation and sealed test identities"
        )
    x = np.asarray(inputs["x"], dtype=np.float64)
    input_mask = inputs.get("input_mask", np.ones_like(x, dtype=bool)).astype(bool)
    if not np.all(input_mask[train_indices]) or not np.all(
        input_mask[validation_indices]
    ):
        raise InputValidationError("NumPy baseline requires complete train/validation inputs")
    y = np.asarray(labels["p_hit"], dtype=np.float64)
    point_mask = labels.get("p_hit_mask", labels["label_mask"]).astype(bool)
    if np.any(point_mask[test_indices]):
        raise InputValidationError(
            "sealed test p_hit masks must be false before model development"
        )
    if "p_hit_identification_interval" in labels:
        interval = np.asarray(
            labels["p_hit_identification_interval"], dtype=np.float64
        ).copy()
    else:
        development_indices = np.concatenate([train_indices, validation_indices])
        if np.any(~point_mask[development_indices]):
            raise InputValidationError(
                "masked development p_hit targets require identification intervals"
            )
        interval = np.stack([y, y], axis=-1)
    case_weight = np.asarray(weights["design_weight"], dtype=np.float64)
    if np.any(case_weight <= 0.0) or not np.isfinite(case_weight).all():
        raise InputValidationError("NumPy baseline requires finite positive design weights")
    component_ids = np.asarray(labels["component_ids"]).astype(str)
    task_admission_masks, task_admission_contract = load_baseline_task_admission(
        task_admission_path,
        dataset_validation=validation,
        component_ids=component_ids,
    )
    p_hit_admitted = task_admission_masks["p_hit"]
    point_mask &= p_hit_admitted[None, :]
    interval[:, ~p_hit_admitted, 0] = 0.0
    interval[:, ~p_hit_admitted, 1] = 1.0

    if not np.any(point_mask[validation_indices]):
        raise InputValidationError(
            "NumPy baseline validation split has no point-identified p_hit labels"
        )

    runs: list[dict[str, Any]] = []
    selected_parameters: dict[str, np.ndarray] | None = None
    selected_normalization: tuple[np.ndarray, np.ndarray] | None = None
    selected_run: dict[str, Any] | None = None
    maximum_fraction = max(normalized_fractions)
    for fraction in normalized_fractions:
        subset_count = max(1, int(math.ceil(train_indices.size * fraction)))
        for seed in normalized_seeds:
            generator = np.random.default_rng(seed)
            subset = np.sort(generator.permutation(train_indices)[:subset_count])
            mean = np.mean(x[subset], axis=0)
            scale = np.std(x[subset], axis=0)
            scale[scale < 1.0e-8] = 1.0
            fitted = _fit_one(
                (x[subset] - mean) / scale,
                y[subset],
                point_mask[subset],
                interval[subset],
                case_weight[subset],
                (x[validation_indices] - mean) / scale,
                y[validation_indices],
                point_mask[validation_indices],
                interval[validation_indices],
                case_weight[validation_indices],
                component_ids,
                seed=seed,
                hidden_size=hidden_size,
                epochs=epochs,
                learning_rate=learning_rate,
                interval_loss_weight=interval_loss_weight,
                l2_penalty=l2_penalty,
            )
            public_run = {
                "train_fraction": fraction,
                "seed": seed,
                "train_case_count": int(subset.size),
                "best_epoch": fitted["best_epoch"],
                "train_metrics": fitted["train_metrics"],
                "validation_metrics": fitted["validation_metrics"],
                "training_diagnostics": fitted["training_diagnostics"],
                "history": fitted["history"],
            }
            runs.append(public_run)
            if math.isclose(fraction, maximum_fraction):
                candidate_brier = fitted["validation_metrics"]["brier"]
                if candidate_brier is not None and (
                    selected_run is None
                    or float(candidate_brier)
                    < float(selected_run["validation_metrics"]["brier"])
                ):
                    selected_run = deepcopy(public_run)
                    selected_parameters = fitted["parameters"]
                    selected_normalization = (mean.copy(), scale.copy())

    if selected_run is None or selected_parameters is None or selected_normalization is None:
        raise InputValidationError("NumPy baseline could not select a full-fraction model")

    learning_curve = {
        str(fraction): _aggregate_runs(
            [run for run in runs if math.isclose(run["train_fraction"], fraction)]
        )
        for fraction in normalized_fractions
    }
    train_support = _split_metrics(
        np.full_like(y[train_indices], 0.5),
        y[train_indices],
        point_mask[train_indices],
        interval[train_indices],
        case_weight[train_indices],
        component_ids,
    )["component_diagnostics"]
    validation_support = selected_run["validation_metrics"]["component_diagnostics"]
    support_by_component = [
        {
            "component_id": component_id,
            "train": train_row,
            "validation": validation_row,
        }
        for component_id, train_row, validation_row in zip(
            component_ids.tolist(), train_support, validation_support
        )
    ]
    for row, admitted in zip(
        support_by_component, p_hit_admitted.tolist(), strict=True
    ):
        row["admitted_for_p_hit"] = bool(admitted)

    output.mkdir(parents=True)
    mean, scale = selected_normalization
    with (output / "selected_model.npz").open("wb") as stream:
        np.savez_compressed(
            stream,
            **selected_parameters,
            feature_names=inputs["feature_names"],
            component_ids=component_ids,
            p_hit_task_admission_mask=p_hit_admitted,
            train_mean=mean,
            train_scale=scale,
        )
    report: dict[str, Any] = {
        "schema_version": NUMPY_BASELINE_SCHEMA_VERSION,
        "status": "completed",
        "dataset_id": validation["dataset_id"],
        "dataset_manifest_hash": validation["manifest_hash"],
        "dataset_manifest_sha256": file_hash(dataset / "manifest.json"),
        "task_admission": task_admission_contract,
        "model": {
            "kind": "one_hidden_layer_tanh_multioutput_sigmoid_mlp",
            "implementation": "numpy_adam_full_batch",
            "runtime_dependency": "numpy_only",
            "hidden_size": hidden_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "interval_loss_weight": interval_loss_weight,
            "l2_penalty": l2_penalty,
            "point_loss": "design_weighted_soft_bernoulli_cross_entropy",
            "interval_loss": "design_weighted_squared_identification_interval_hinge",
            "vacuous_interval_policy": "[0,1]_contributes_no_gradient",
            "input_normalization": {
                "method": "per_feature_mean_and_population_standard_deviation",
                "fit_scope": "each_run_training_subset_only",
                "validation_statistics_used": False,
                "test_statistics_used": False,
            },
        },
        "requested_fractions": list(normalized_fractions),
        "requested_seeds": list(normalized_seeds),
        "runs": runs,
        "learning_curve": learning_curve,
        "selected_model": {
            "selection_metric": "point_identified_p_hit_weighted_brier",
            "selection_scope": "frozen_admitted_p_hit_components_on_reference_validation",
            "train_fraction": selected_run["train_fraction"],
            "train_case_count": selected_run["train_case_count"],
            "seed": selected_run["seed"],
            "best_epoch": selected_run["best_epoch"],
            "validation_metrics": selected_run["validation_metrics"],
            "model_file": "selected_model.npz",
        },
        "support_by_component": support_by_component,
        "test_access_policy": {
            "test_identity_count": int(test_indices.size),
            "test_labels_present_in_core": False,
            "labels_used_in_loss": False,
            "labels_used_in_metrics": False,
            "labels_used_in_selection": False,
            "test_result_json_lines_decoded": int(
                seal.get("test_result_json_lines_decoded", 0)
            ),
            "open_count": int(seal["test_open_count"]),
        },
    }
    report["selected_model"]["model_sha256"] = file_hash(
        output / "selected_model.npz"
    )
    report["report_hash"] = content_hash(report)
    (output / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = ["NUMPY_BASELINE_SCHEMA_VERSION", "run_numpy_p_hit_learning_curves"]
