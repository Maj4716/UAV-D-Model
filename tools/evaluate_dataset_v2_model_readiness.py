#!/usr/bin/env python3
"""Evaluate admitted p_hit learnability without opening dataset-v2 test labels.

The evaluator is intentionally independent from the training entry point.  It
reconstructs the selected NumPy model on Reference validation, compares it with
per-component train-only climatology, and binds every result to the dataset,
model, baseline report and frozen task-admission identities.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.baseline_task_admission import load_baseline_task_admission  # noqa: E402
from damage_engine.dataset_v2 import load_dataset_v2_core, validate_dataset_v2  # noqa: E402
from damage_engine.errors import InputValidationError  # noqa: E402
from damage_engine.utils import content_hash, file_hash  # noqa: E402


MODEL_READINESS_SCHEMA_VERSION = "dataset-v2-p-hit-model-readiness@1.0.0"
_MODEL_KEYS = frozenset(
    {
        "w1",
        "b1",
        "w2",
        "b2",
        "feature_names",
        "component_ids",
        "p_hit_task_admission_mask",
        "train_mean",
        "train_scale",
    }
)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputValidationError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise InputValidationError(f"{label} must be a JSON object")
    return value


def _validate_self_hash(document: Mapping[str, Any], *, field: str, label: str) -> None:
    actual = document.get(field)
    payload = dict(document)
    payload.pop(field, None)
    expected = content_hash(payload)
    if actual != expected:
        raise InputValidationError(
            f"{label} self-hash mismatch",
            context={"expected": expected, "actual": actual},
        )


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _predict(model: Mapping[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    hidden = np.tanh(x @ model["w1"] + model["b1"])
    return _sigmoid(hidden @ model["w2"] + model["b2"])


def _weighted_average(value: np.ndarray, weight: np.ndarray) -> float:
    mass = float(np.sum(weight))
    if mass <= 0.0 or not math.isfinite(mass):
        raise InputValidationError("model-readiness metric has zero weight mass")
    return float(np.sum(value * weight) / mass)


def _require_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputValidationError(
            "model-readiness baseline report lacks a numeric field",
            context={"field": field},
        )
    result = float(value)
    if not math.isfinite(result):
        raise InputValidationError(
            "model-readiness baseline field must be finite",
            context={"field": field},
        )
    return result


def _learning_curve_evidence(report: Mapping[str, Any]) -> dict[str, Any]:
    rows = report.get("learning_curve")
    if not isinstance(rows, dict):
        raise InputValidationError("baseline report lacks learning_curve")
    mean_by_fraction: dict[str, float] = {}
    for key, row in rows.items():
        try:
            fraction = float(key)
        except (TypeError, ValueError) as exc:
            raise InputValidationError("learning-curve fraction is not numeric") from exc
        if not isinstance(row, dict) or not isinstance(row.get("brier"), dict):
            raise InputValidationError("learning-curve row lacks brier summary")
        mean = _require_number(
            row["brier"].get("mean"), field=f"learning_curve.{key}.brier.mean"
        )
        mean_by_fraction[str(fraction)] = mean
    if "0.25" not in mean_by_fraction or "1.0" not in mean_by_fraction:
        raise InputValidationError(
            "model readiness requires 0.25 and 1.0 learning-curve fractions"
        )
    quarter = mean_by_fraction["0.25"]
    full = mean_by_fraction["1.0"]
    absolute = quarter - full
    ordered = sorted((float(key), value) for key, value in mean_by_fraction.items())
    return {
        "validation_brier_mean_by_train_fraction": {
            str(fraction): value for fraction, value in ordered
        },
        "quarter_fraction_brier_mean": quarter,
        "full_fraction_brier_mean": full,
        "absolute_brier_reduction_0_25_to_1_0": absolute,
        "relative_brier_reduction_0_25_to_1_0": (
            absolute / quarter if quarter > 0.0 else None
        ),
        "full_fraction_improves_over_quarter_fraction": bool(full < quarter),
        "monotone_nonincreasing_over_reported_fractions": bool(
            all(right <= left for (_, left), (_, right) in zip(ordered, ordered[1:]))
        ),
    }


def evaluate_model_readiness(
    dataset_directory: str | Path,
    model_path: str | Path,
    baseline_results_path: str | Path,
    task_admission_path: str | Path,
) -> dict[str, Any]:
    """Return validation-only learnability evidence for admitted p_hit tasks."""

    dataset = Path(dataset_directory)
    model_file = Path(model_path)
    results_file = Path(baseline_results_path)
    admission_file = Path(task_admission_path)
    validation = validate_dataset_v2(dataset, recompute_views=False)

    seal_file = dataset / "sealed_test.json"
    seal = _load_json(seal_file, label="sealed-test record")
    if int(seal.get("test_open_count", -1)) != 0:
        raise InputValidationError("sealed test has already been opened")
    if int(seal.get("test_result_json_lines_decoded", -1)) != 0:
        raise InputValidationError("sealed test result rows have already been decoded")
    if seal.get("test_labels_present_in_core") is not False:
        raise InputValidationError("sealed test labels must remain absent from core")

    baseline = _load_json(results_file, label="NumPy baseline report")
    _validate_self_hash(baseline, field="report_hash", label="NumPy baseline report")
    expected_identity = {
        "dataset_id": validation["dataset_id"],
        "dataset_manifest_hash": validation["manifest_hash"],
    }
    for field, expected in expected_identity.items():
        if baseline.get(field) != expected:
            raise InputValidationError(
                "NumPy baseline report is bound to a different dataset",
                context={"field": field, "expected": expected, "actual": baseline.get(field)},
            )
    if baseline.get("dataset_manifest_sha256") != file_hash(dataset / "manifest.json"):
        raise InputValidationError("NumPy baseline manifest file hash mismatch")
    access = baseline.get("test_access_policy")
    if not isinstance(access, dict) or int(access.get("open_count", -1)) != 0 or int(
        access.get("test_result_json_lines_decoded", -1)
    ) != 0:
        raise InputValidationError("NumPy baseline report does not preserve the test seal")

    core = load_dataset_v2_core(dataset)
    inputs = core["inputs"]
    labels = core["component_labels"]
    weights = core["sampling_weights"]
    component_ids = np.asarray(labels["component_ids"]).astype(str)
    feature_names = np.asarray(inputs["feature_names"]).astype(str)
    admission_masks, admission_contract = load_baseline_task_admission(
        admission_file,
        dataset_validation=validation,
        component_ids=component_ids,
    )
    admitted = np.asarray(admission_masks["p_hit"], dtype=bool)
    if not np.any(admitted):
        raise InputValidationError("p_hit task admission is empty")
    baseline_admission = baseline.get("task_admission")
    if not isinstance(baseline_admission, dict):
        raise InputValidationError("NumPy baseline report lacks task_admission")
    for field in ("artifact_hash", "artifact_sha256"):
        if baseline_admission.get(field) != admission_contract.get(field):
            raise InputValidationError(
                "NumPy baseline used a different task-admission artifact",
                context={
                    "field": field,
                    "expected": admission_contract.get(field),
                    "actual": baseline_admission.get(field),
                },
            )

    selected = baseline.get("selected_model")
    if not isinstance(selected, dict):
        raise InputValidationError("NumPy baseline report lacks selected_model")
    actual_model_hash = file_hash(model_file)
    if selected.get("model_sha256") != actual_model_hash:
        raise InputValidationError("selected NumPy model file hash mismatch")
    try:
        with np.load(model_file, allow_pickle=False) as archive:
            if set(archive.files) != _MODEL_KEYS:
                raise InputValidationError(
                    "selected NumPy model field set is invalid",
                    context={
                        "missing": sorted(_MODEL_KEYS - set(archive.files)),
                        "unknown": sorted(set(archive.files) - _MODEL_KEYS),
                    },
                )
            model = {key: np.asarray(archive[key]) for key in archive.files}
    except (OSError, ValueError) as exc:
        raise InputValidationError("selected NumPy model is unreadable") from exc

    if model["feature_names"].astype(str).tolist() != feature_names.tolist():
        raise InputValidationError("selected model feature order differs from dataset")
    if model["component_ids"].astype(str).tolist() != component_ids.tolist():
        raise InputValidationError("selected model component order differs from dataset")
    if not np.array_equal(
        np.asarray(model["p_hit_task_admission_mask"], dtype=bool), admitted
    ):
        raise InputValidationError("selected model admission mask differs from frozen artifact")

    x = np.asarray(inputs["x"], dtype=np.float64)
    input_mask = np.asarray(inputs.get("input_mask", np.ones_like(x)), dtype=bool)
    split = np.asarray(weights["split"]).astype(str)
    train_indices = np.flatnonzero(split == "train")
    validation_indices = np.flatnonzero(split == "validation")
    test_indices = np.flatnonzero(split == "test")
    if not train_indices.size or not validation_indices.size or not test_indices.size:
        raise InputValidationError("model readiness requires train, validation and test identities")
    if not np.all(input_mask[np.concatenate([train_indices, validation_indices])]):
        raise InputValidationError("model readiness requires complete development inputs")

    y = np.asarray(labels["p_hit"], dtype=np.float64)
    point_mask = np.asarray(labels.get("p_hit_mask", labels["label_mask"]), dtype=bool)
    if np.any(point_mask[test_indices]):
        raise InputValidationError("sealed-test p_hit mask must remain false")
    case_weight = np.asarray(weights["design_weight"], dtype=np.float64)
    if np.any(case_weight <= 0.0) or not np.isfinite(case_weight).all():
        raise InputValidationError("model readiness requires finite positive design weights")

    feature_count = x.shape[1]
    component_count = component_ids.size
    w1 = np.asarray(model["w1"], dtype=np.float64)
    b1 = np.asarray(model["b1"], dtype=np.float64)
    w2 = np.asarray(model["w2"], dtype=np.float64)
    b2 = np.asarray(model["b2"], dtype=np.float64)
    mean = np.asarray(model["train_mean"], dtype=np.float64)
    scale = np.asarray(model["train_scale"], dtype=np.float64)
    hidden_size = b1.size
    expected_shapes = {
        "w1": (feature_count, hidden_size),
        "b1": (hidden_size,),
        "w2": (hidden_size, component_count),
        "b2": (component_count,),
        "train_mean": (feature_count,),
        "train_scale": (feature_count,),
    }
    for field, expected_shape in expected_shapes.items():
        if np.asarray(model[field]).shape != expected_shape:
            raise InputValidationError(
                "selected model tensor shape mismatch",
                context={
                    "field": field,
                    "expected": list(expected_shape),
                    "actual": list(np.asarray(model[field]).shape),
                },
            )
    if not all(np.isfinite(np.asarray(model[field], dtype=np.float64)).all() for field in ("w1", "b1", "w2", "b2", "train_mean", "train_scale")):
        raise InputValidationError("selected model contains non-finite parameters")
    if np.any(scale <= 0.0):
        raise InputValidationError("selected model train_scale must be positive")

    normalized_validation = (x[validation_indices] - mean) / scale
    probability = _predict(
        {"w1": w1, "b1": b1, "w2": w2, "b2": b2}, normalized_validation
    )
    validation_target = y[validation_indices]
    validation_mask = point_mask[validation_indices] & admitted[None, :]
    validation_weight = case_weight[validation_indices]
    flat_weight = np.broadcast_to(
        validation_weight[:, None], validation_target.shape
    )
    if not np.any(validation_mask):
        raise InputValidationError("admitted p_hit validation set has no point targets")

    train_target = y[train_indices]
    train_mask = point_mask[train_indices] & admitted[None, :]
    train_weight = case_weight[train_indices]
    climatology = np.zeros(component_count, dtype=np.float64)
    per_component: list[dict[str, Any]] = []
    model_squared = (probability - validation_target) ** 2
    climatology_squared = np.zeros_like(model_squared)
    for column, component_id in enumerate(component_ids.tolist()):
        if not admitted[column]:
            continue
        train_selected = train_mask[:, column]
        validation_selected = validation_mask[:, column]
        if not np.any(train_selected) or not np.any(validation_selected):
            raise InputValidationError(
                "admitted component lacks train or validation point targets",
                context={"component_id": component_id},
            )
        climatology[column] = _weighted_average(
            train_target[train_selected, column], train_weight[train_selected]
        )
        climatology_squared[:, column] = (
            climatology[column] - validation_target[:, column]
        ) ** 2
        model_brier = _weighted_average(
            model_squared[validation_selected, column],
            validation_weight[validation_selected],
        )
        climatology_brier = _weighted_average(
            climatology_squared[validation_selected, column],
            validation_weight[validation_selected],
        )
        skill = (
            1.0 - model_brier / climatology_brier
            if climatology_brier > 0.0
            else None
        )
        per_component.append(
            {
                "component_id": component_id,
                "train_point_count": int(np.sum(train_selected)),
                "validation_point_count": int(np.sum(validation_selected)),
                "train_only_climatology": float(climatology[column]),
                "model_brier": model_brier,
                "climatology_brier": climatology_brier,
                "brier_skill": skill,
                "model_beats_climatology": bool(skill is not None and skill > 0.0),
            }
        )

    model_micro = _weighted_average(
        model_squared[validation_mask], flat_weight[validation_mask]
    )
    climatology_micro = _weighted_average(
        climatology_squared[validation_mask], flat_weight[validation_mask]
    )
    model_macro = float(np.mean([row["model_brier"] for row in per_component]))
    climatology_macro = float(
        np.mean([row["climatology_brier"] for row in per_component])
    )
    micro_skill = 1.0 - model_micro / climatology_micro
    macro_skill = 1.0 - model_macro / climatology_macro
    selected_metrics = selected.get("validation_metrics")
    if not isinstance(selected_metrics, dict):
        raise InputValidationError("selected model lacks validation_metrics")
    reported_brier = _require_number(
        selected_metrics.get("brier"), field="selected_model.validation_metrics.brier"
    )
    reconstruction_error = abs(model_micro - reported_brier)
    if reconstruction_error > 1.0e-12:
        raise InputValidationError(
            "reconstructed validation Brier differs from baseline report",
            context={
                "reconstructed": model_micro,
                "reported": reported_brier,
                "absolute_error": reconstruction_error,
            },
        )

    positive_skill = [
        row for row in per_component if row["brier_skill"] is not None and row["brier_skill"] > 0.0
    ]
    evaluable_skill = [row for row in per_component if row["brier_skill"] is not None]
    worst = sorted(
        evaluable_skill,
        key=lambda row: (float(row["brier_skill"]), row["component_id"]),
    )[:10]
    learning = _learning_curve_evidence(baseline)
    gates = {
        "sealed_test_untouched": True,
        "dataset_model_admission_identity_consistent": True,
        "all_admitted_components_validation_evaluable": len(per_component)
        == int(np.sum(admitted)),
        "selected_validation_brier_exactly_reconstructed": reconstruction_error
        <= 1.0e-12,
        "micro_brier_skill_positive": micro_skill > 0.0,
        "macro_brier_skill_positive": macro_skill > 0.0,
        "majority_of_components_have_positive_skill": len(positive_skill)
        > len(evaluable_skill) / 2.0,
        "full_fraction_improves_over_quarter_fraction": learning[
            "full_fraction_improves_over_quarter_fraction"
        ],
    }

    report: dict[str, Any] = {
        "schema_version": MODEL_READINESS_SCHEMA_VERSION,
        "scope": "admitted_p_hit_train_validation_only",
        "dataset_identity": {
            "dataset_id": validation["dataset_id"],
            "dataset_manifest_hash": validation["manifest_hash"],
            "dataset_manifest_sha256": file_hash(dataset / "manifest.json"),
        },
        "artifact_identity": {
            "model_sha256": actual_model_hash,
            "baseline_results_sha256": file_hash(results_file),
            "baseline_report_hash": baseline["report_hash"],
            "task_admission_sha256": admission_contract["artifact_sha256"],
            "task_admission_hash": admission_contract["artifact_hash"],
        },
        "sealed_test": {
            "test_case_count": int(test_indices.size),
            "test_open_count": int(seal["test_open_count"]),
            "test_result_json_lines_decoded": int(
                seal["test_result_json_lines_decoded"]
            ),
            "test_labels_present_in_core": False,
            "test_predictions_computed": False,
            "test_metrics_computed": False,
        },
        "admission": {
            "task": "p_hit",
            "admitted_component_count": int(np.sum(admitted)),
            "admitted_component_ids": component_ids[admitted].tolist(),
        },
        "validation_metrics": {
            "point_observation_count": int(np.sum(validation_mask)),
            "model_micro_brier": model_micro,
            "model_macro_brier": model_macro,
            "train_only_climatology_micro_brier": climatology_micro,
            "train_only_climatology_macro_brier": climatology_macro,
            "micro_brier_skill": micro_skill,
            "macro_brier_skill": macro_skill,
            "positive_skill_component_count": len(positive_skill),
            "skill_evaluable_component_count": len(evaluable_skill),
            "positive_skill_component_fraction": len(positive_skill)
            / len(evaluable_skill),
            "reported_selected_model_micro_brier": reported_brier,
            "reconstruction_absolute_error": reconstruction_error,
        },
        "per_component": per_component,
        "worst_components_by_brier_skill": worst,
        "learning_curve": learning,
        "detectable_learnability_gates": gates,
        "detectable_learnability_pass": all(gates.values()),
        "evidence_boundary": (
            "This is validation-only evidence that the admitted synthetic-teacher "
            "p_hit mapping contains learnable signal beyond train-only climatology; "
            "it is not a sealed-test estimate, physical validation, or deployment admission."
        ),
    }
    report["report_hash"] = content_hash(report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct an admitted NumPy p_hit model and evaluate validation-only "
            "Brier skill without reading sealed-test labels."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument("--task-admission", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise InputValidationError(
            "model-readiness output already exists", context={"path": str(output)}
        )
    report = evaluate_model_readiness(
        args.dataset,
        args.model,
        args.baseline_results,
        args.task_admission,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "report_hash": report["report_hash"],
                "admitted_component_count": report["admission"][
                    "admitted_component_count"
                ],
                "model_micro_brier": report["validation_metrics"][
                    "model_micro_brier"
                ],
                "micro_brier_skill": report["validation_metrics"][
                    "micro_brier_skill"
                ],
                "detectable_learnability_pass": report[
                    "detectable_learnability_pass"
                ],
                "test_result_json_lines_decoded": report["sealed_test"][
                    "test_result_json_lines_decoded"
                ],
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

