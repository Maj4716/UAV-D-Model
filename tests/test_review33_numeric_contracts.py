"""Regression tests for the review-33 numerical and mask contracts.

The tests use artificial arrays only.  They neither execute the simulator nor
claim anything about physical or scientific validity.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math

import numpy as np
import pytest

from damage_engine.errors import InputValidationError
from damage_engine.normalization import apply_training_normalization
from damage_engine.system_learning import (
    build_system_learning_contract,
    evaluate_system_probabilities,
    fit_reference_prevalence_baseline,
    masked_brier_from_logits,
    masked_soft_bce_with_logits,
)
from damage_engine.training_view import TrainingViewDataset, fit_training_view_artifact
from damage_engine.utils import content_hash
from test_training_view_contract import ArtificialTrainingDataset


torch = pytest.importorskip("torch")


LOSS_FUNCTIONS = (masked_soft_bce_with_logits, masked_brier_from_logits)
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("loss_function", LOSS_FUNCTIONS)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("device", DEVICES)
def test_masked_losses_ignore_nan_target_placeholders_and_preserve_gradients(
    loss_function, dtype, device
):
    mask = torch.tensor(
        [[True, False], [True, True]], dtype=torch.bool, device=device
    )
    finite_targets = torch.tensor(
        [[0.25, 0.0], [0.5, 1.0]], dtype=dtype, device=device
    )
    nan_targets = finite_targets.clone()
    nan_targets[~mask] = float("nan")
    design_weight = torch.tensor([2.0, 3.0], dtype=dtype, device=device)
    task_weight = torch.tensor([1.0, 4.0], dtype=dtype, device=device)

    finite_logits = torch.tensor(
        [[-0.5, 0.25], [0.75, -1.0]], dtype=dtype, device=device,
        requires_grad=True,
    )
    nan_logits = finite_logits.detach().clone().requires_grad_(True)
    expected = loss_function(
        finite_logits,
        finite_targets,
        mask,
        design_weight=design_weight,
        task_weight=task_weight,
    )
    actual = loss_function(
        nan_logits,
        nan_targets,
        mask,
        design_weight=design_weight,
        task_weight=task_weight,
    )
    assert torch.isfinite(actual)
    assert actual.detach().cpu().item() == pytest.approx(
        expected.detach().cpu().item(), rel=2.0e-6, abs=1.0e-7
    )

    actual.backward()
    assert torch.isfinite(nan_logits.grad).all()
    assert torch.count_nonzero(nan_logits.grad[~mask]).item() == 0


@pytest.mark.parametrize("loss_function", LOSS_FUNCTIONS)
def test_masked_losses_reject_available_nan_and_empty_supervision(loss_function):
    logits = torch.zeros((1, 2), dtype=torch.float64)
    with pytest.raises(InputValidationError, match="finite"):
        loss_function(
            logits,
            torch.tensor([[float("nan"), 0.0]], dtype=torch.float64),
            torch.tensor([[True, False]]),
        )
    with pytest.raises(InputValidationError, match="no available"):
        loss_function(
            logits,
            torch.full((1, 2), float("nan"), dtype=torch.float64),
            torch.zeros((1, 2), dtype=torch.bool),
        )


def test_label_support_is_independent_from_design_weight_eligibility():
    contract = build_system_learning_contract(
        np.asarray([[0.0], [1.0], [1.0]]),
        np.ones((3, 1), dtype=bool),
        ("task",),
        split_names=("train", "train", "train"),
        sampling_roles=("reference_train",) * 3,
        dataset_id="artificial-review33",
        dataset_manifest_hash="sha256:" + "3" * 64,
        test_status="sealed_unobserved",
        design_weights=np.asarray([2.0, 7.0, 11.0]),
        design_weight_mask=np.asarray([True, False, False]),
        minimum_soft_event_mass={"train": 1.0, "validation": 1.0, "test": 1.0},
    )
    task = contract["tasks"][0]
    support = task["support"]["all_train"]
    assert support["available_count"] == support["label_available_count"] == 3
    assert support["weight_eligible_count"] == 1
    assert support["joint_analysis_count"] == 1
    assert support["soft_event_mass"] == 2.0
    assert task["label_available"] is True
    assert task["status"] == "default_train_eligible"


def test_extreme_finite_weights_keep_scale_invariant_metrics_finite():
    targets = np.asarray([[0.0], [1.0]])
    mask = np.ones_like(targets, dtype=bool)
    weights = np.asarray([1.0e308, 1.0e308])
    baseline = fit_reference_prevalence_baseline(
        targets,
        mask,
        ("task",),
        split_names=("train", "train"),
        sampling_roles=("reference_train", "reference_train"),
        design_weights=weights,
    )
    assert baseline["predictions"] == pytest.approx([0.5])
    assert baseline["support"][0]["design_weight_sum"] is None
    assert baseline["support"][0]["design_weight_sum_status"].startswith(
        "not_representable"
    )

    report = evaluate_system_probabilities(
        targets,
        np.full_like(targets, 0.5),
        mask,
        ("task",),
        evaluated_tasks=("task",),
        design_weights=weights,
        baseline_predictions=baseline["predictions"],
        calibration_bins=2,
    )
    task = report["tasks"][0]
    assert task["brier"] == pytest.approx(0.25)
    assert task["log_loss"] == pytest.approx(math.log(2.0))
    assert task["ece"] == pytest.approx(0.0)
    assert task["brier_skill_score"] == pytest.approx(0.0)
    assert task["design_weight_sum"] is None
    json.dumps(report, allow_nan=False)


def test_effective_sample_size_is_invariant_to_common_weight_scale():
    def support(weights):
        contract = build_system_learning_contract(
            np.asarray([[0.0], [0.5], [1.0]]),
            np.ones((3, 1), dtype=bool),
            ("task",),
            split_names=("train",) * 3,
            sampling_roles=("reference_train",) * 3,
            dataset_id="artificial-review33",
            dataset_manifest_hash="sha256:" + "4" * 64,
            test_status="sealed_unobserved",
            design_weights=np.asarray(weights),
            minimum_soft_event_mass={
                "train": 1.0,
                "validation": 1.0,
                "test": 1.0,
            },
        )
        return contract["tasks"][0]["support"]["all_train"]

    tiny = support([1.0e-200, 2.0e-200, 3.0e-200])
    large = support([1.0e100, 2.0e100, 3.0e100])
    expected = 36.0 / 14.0
    assert tiny["effective_sample_size"] == pytest.approx(expected)
    assert large["effective_sample_size"] == pytest.approx(expected)
    assert tiny["soft_event_effective_sample_size"] == pytest.approx(
        large["soft_event_effective_sample_size"]
    )
    assert 1.0 <= large["effective_sample_size"] <= 3.0


@pytest.mark.parametrize("unrepresentable", (1.0e300, 1.0e-300))
def test_task_weights_are_revalidated_after_float32_conversion(unrepresentable):
    dataset = ArtificialTrainingDataset()
    artifact = fit_training_view_artifact(dataset, proxy_boundary="end_to_end")
    with pytest.raises(InputValidationError, match="after float32 conversion"):
        TrainingViewDataset(
            dataset,
            artifact,
            proxy_boundary="end_to_end",
            task_loss_weights={"system-events": unrepresentable},
        )


def test_normalization_rejects_nonfinite_available_float32_input_with_context():
    dataset = ArtificialTrainingDataset()
    artifact = fit_training_view_artifact(dataset, proxy_boundary="end_to_end")
    values = np.zeros(len(artifact["features"]), dtype=np.float64)
    mask = np.zeros_like(values, dtype=bool)
    index = 0
    values[index] = 1.0e300
    mask[index] = True
    with pytest.raises(InputValidationError, match="after float32 conversion") as raised:
        apply_training_normalization(values, mask, artifact)
    assert raised.value.context["feature_name"] == artifact["features"][index]["name"]
    assert raised.value.context["normalization_id"] == artifact["normalization_id"]


def test_normalization_rejects_nonfinite_available_output_with_context():
    dataset = ArtificialTrainingDataset()
    artifact = fit_training_view_artifact(dataset, proxy_boundary="end_to_end")
    changed = deepcopy(artifact)
    index = next(
        index
        for index, feature in enumerate(changed["features"])
        if feature["normalization"] == "standardize"
    )
    changed["features"][index]["scale"] = 1.0e-300
    payload = deepcopy(changed)
    payload.pop("normalization_id", None)
    changed["normalization_id"] = content_hash(payload)
    values = np.zeros(len(changed["features"]), dtype=np.float32)
    mask = np.zeros_like(values, dtype=bool)
    values[index] = float(changed["features"][index]["mean"]) + 1.0
    mask[index] = True
    with pytest.raises(InputValidationError, match="normalized available") as raised:
        apply_training_normalization(values, mask, changed)
    assert raised.value.context["feature_name"] == changed["features"][index]["name"]
    assert raised.value.context["normalization_id"] == changed["normalization_id"]
