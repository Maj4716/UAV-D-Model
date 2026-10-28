"""Generic masked numeric API checks; no domain data, simulator or training.

Inputs are two rows and two anonymously named tasks. The script evaluates
functions only, never performs backward/optimization, and never reads formal
target arrays or suggests task/sampling settings.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import warnings

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from damage_engine.system_learning import (
    evaluate_system_probabilities, fit_reference_prevalence_baseline,
    masked_brier_from_logits, masked_soft_bce_with_logits,
)
from damage_engine.utils import file_hash, source_tree_hash


def source_hashes() -> dict:
    return {"package_source_tree": source_tree_hash(ROOT / "damage_engine"),
            "files": {name: file_hash(ROOT / "damage_engine" / name) for name in
                      ("system_learning.py", "archive_validation.py", "multitask_targets.py", "training_view.py")}}


def summarized(value):
    if torch.is_tensor(value):
        return {"kind": "tensor", "value_text": str(value.item()),
                "is_finite": bool(torch.isfinite(value).all()), "dtype": str(value.dtype)}
    if isinstance(value, dict) and "predictions" in value:
        array = np.asarray(value["predictions"])
        return {"kind": "baseline", "predictions_text": [str(item) for item in array],
                "all_finite": bool(np.isfinite(array).all()), "support": value["support"]}
    if isinstance(value, dict) and "tasks" in value:
        return {"kind": "metrics", "tasks": [
            {name: row.get(name) for name in
             ("task_name", "available_count", "brier", "log_loss", "ece", "calibration_status",
              "soft_average_precision", "soft_average_precision_status")}
            for row in value["tasks"]]}
    return value


def run(callable_) -> dict:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value = callable_()
            result = {"accepted": True, "output": summarized(value)}
        except Exception as error:
            result = {"accepted": False, "error_type": type(error).__name__, "message": str(error)}
    result["warnings"] = [str(item.message) for item in caught]
    return result


def main() -> None:
    before = source_hashes()
    report = {"scope": "general two-row, two-task numeric fixtures; no formal label distribution analysis",
              "python": sys.executable, "torch": torch.__version__, "numpy": np.__version__,
              "source_before": before,
              "covered_public_apis": ["masked_soft_bce_with_logits", "masked_brier_from_logits",
                                      "fit_reference_prevalence_baseline", "evaluate_system_probabilities"]}
    mixed = np.asarray([[True, False], [False, True]])
    targets = np.asarray([[.25, 0.], [0., .75]], dtype=np.float64)
    nan_targets = targets.copy()
    nan_targets[~mixed] = np.nan
    cases = {
        "available_true_zero_targets": (np.zeros_like(targets), np.ones_like(mixed), None, None),
        "mixed_mask_finite_placeholders": (targets, mixed, None, None),
        "mixed_mask_nan_placeholders": (nan_targets, mixed, None, None),
        "all_mask_false_finite_placeholders": (targets, np.zeros_like(mixed), None, None),
        "all_mask_false_nan_placeholders": (np.full_like(targets, np.nan), np.zeros_like(mixed), None, None),
        "unmasked_nan": (nan_targets, np.ones_like(mixed), None, None),
        "zero_case_weight": (targets, mixed, np.asarray([0., 1.]), None),
        "zero_task_weight": (targets, mixed, None, np.asarray([0., 1.])),
        "tiny_positive_task_weight": (targets, mixed, None, np.asarray([1.0e-300, 1.])),
    }
    losses = {}
    for name, (values, mask, case_weight, task_weight) in cases.items():
        row = {}
        for function in (masked_soft_bce_with_logits, masked_brier_from_logits):
            kwargs = {"design_weight": None if case_weight is None else torch.from_numpy(case_weight),
                      "task_weight": None if task_weight is None else torch.from_numpy(task_weight)}
            row[function.__name__] = run(lambda fn=function, v=values, m=mask, kw=kwargs:
                fn(torch.zeros((2, 2), dtype=torch.float32), torch.from_numpy(v), torch.from_numpy(m), **kw))
        losses[name] = row
    report["loss_cases"] = losses

    evaluations = {}
    for name, (values, mask, case_weight, _) in cases.items():
        if "task_weight" in name:
            continue
        baseline = lambda v=values, m=mask, w=case_weight: fit_reference_prevalence_baseline(
            v, m, ("toy-a", "toy-b"), split_names=("train", "train"),
            sampling_roles=("reference_train", "reference_train"), design_weights=w)
        metrics = lambda v=values, m=mask, w=case_weight: evaluate_system_probabilities(
            v, np.full((2, 2), .5), m, ("toy-a", "toy-b"), evaluated_tasks=("toy-a", "toy-b"), design_weights=w)
        evaluations[name] = {"baseline": run(baseline), "metrics": run(metrics)}
    report["baseline_and_metrics_cases"] = evaluations

    # Only a generic dtype boundary: positive finite weights can still overflow
    # aggregation. This is not a proposed setting for any real data or model.
    extreme = np.asarray([1.0e308, 1.0e308])
    report["finite_weight_sum_overflow_boundary"] = {
        "baseline": run(lambda: fit_reference_prevalence_baseline(
            targets, np.ones_like(mixed), ("toy-a", "toy-b"), split_names=("train", "train"),
            sampling_roles=("reference_train", "reference_train"), design_weights=extreme)),
        "metrics": run(lambda: evaluate_system_probabilities(
            targets, np.full((2, 2), .5), np.ones_like(mixed), ("toy-a", "toy-b"),
            evaluated_tasks=("toy-a", "toy-b"), design_weights=extreme)),
    }
    after = source_hashes()
    report["source_after"] = after
    report["source_unchanged"] = before == after
    assert before == after
    for name in ("masked_soft_bce_with_logits", "masked_brier_from_logits"):
        assert losses["mixed_mask_finite_placeholders"][name]["output"]["is_finite"]
        assert not losses["mixed_mask_nan_placeholders"][name]["output"]["is_finite"]
        assert not losses["all_mask_false_finite_placeholders"][name]["accepted"]
    # Render any nonfinite diagnostic scalars as strings instead of invalid JSON.
    def canonical(value):
        if isinstance(value, dict):
            return {key: canonical(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [canonical(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return str(value)
        return value
    report = canonical(report)
    destination = Path(__file__).with_suffix(".json")
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"source_unchanged": report["source_unchanged"],
                      "source_hash": before["package_source_tree"],
                      "loss_cases": report["loss_cases"],
                      "baseline_and_metrics_status": {key: {name: item["accepted"] for name, item in row.items()}
                                                      for key, row in evaluations.items()},
                      "finite_weight_sum_overflow_boundary": report["finite_weight_sum_overflow_boundary"]},
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
