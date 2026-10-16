from __future__ import annotations

import numpy as np
import pytest

from damage_engine.errors import InputValidationError
from damage_engine.system_learning import (
    assert_trainable_tasks,
    build_system_learning_contract,
    evaluate_system_probabilities,
    fit_reference_prevalence_baseline,
    is_sha256_identity,
    masked_brier_from_logits,
    masked_soft_bce_with_logits,
)


def test_system_learning_hash_identities_require_lowercase_hex() -> None:
    malformed = "sha256:" + "z" * 64
    assert not is_sha256_identity(malformed)
    with pytest.raises(InputValidationError, match="dataset_manifest_hash"):
        build_system_learning_contract(
            np.asarray([[0.0], [1.0]]),
            np.ones((2, 1), dtype=bool),
            ("task",),
            split_names=("train", "train"),
            sampling_roles=("reference_train", "reference_train"),
            dataset_id="synthetic",
            dataset_manifest_hash=malformed,
            test_status="sealed_unobserved",
        )
    with pytest.raises(InputValidationError, match="source contract bindings"):
        build_system_learning_contract(
            np.asarray([[0.0], [1.0]]),
            np.ones((2, 1), dtype=bool),
            ("task",),
            split_names=("train", "train"),
            sampling_roles=("reference_train", "reference_train"),
            dataset_id="synthetic",
            dataset_manifest_hash="sha256:" + "a" * 64,
            test_status="sealed_unobserved",
            source_contract_hashes={"target_hash": malformed},
        )


def _contract():
    # Variable/high-mass, structural zero, and variable/rare respectively.
    y_train = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.1], [1.0, 0.0, 0.0]], dtype=float
    )
    repeats = 250
    y = np.concatenate(
        [
            np.tile(y_train, (repeats, 1)),
            np.asarray([[0.5, 0.0, 0.0], [0.5, 0.0, 0.1]]),
            np.asarray([[0.5, 0.0, 0.0], [0.5, 0.0, 0.1]]),
        ],
        axis=0,
    )
    split = np.asarray(
        ["train"] * (len(y_train) * repeats) + ["validation"] * 2 + ["test"] * 2
    )
    role = np.asarray(
        ["reference_train"] * (len(y_train) * repeats)
        + ["reference_validation"] * 2
        + ["reference_test"] * 2
    )
    return build_system_learning_contract(
        y,
        np.ones_like(y, dtype=bool),
        ("learnable", "constant", "rare"),
        split_names=split,
        sampling_roles=role,
        dataset_id="synthetic",
        dataset_manifest_hash="sha256:" + "a" * 64,
        test_status="observed_diagnostic_only",
    )


def test_contract_uses_train_only_and_fails_closed_on_constant_tasks():
    contract = _contract()
    statuses = {item["task_name"]: item["status"] for item in contract["tasks"]}
    assert statuses == {
        "learnable": "default_train_eligible",
        "constant": "excluded_constant_zero_train",
        "rare": "exploratory_low_event_mass",
    }
    assert contract["selection_policy"]["default_task_names"] == ["learnable"]
    assert contract["test_policy"]["requires_new_sealed_reference_test"] is True
    assert_trainable_tasks(contract, ["learnable"])
    with pytest.raises(InputValidationError, match="not eligible"):
        assert_trainable_tasks(contract, ["constant"])
    with pytest.raises(InputValidationError, match="not eligible"):
        assert_trainable_tasks(contract, ["rare"])
    assert_trainable_tasks(contract, ["rare"], allow_exploratory_low_mass=True)


def test_trainable_task_indices_follow_the_bound_output_axis_not_record_order():
    contract = _contract()
    contract["tasks"] = list(reversed(contract["tasks"]))
    assert assert_trainable_tasks(contract, ["learnable"]) == (0,)
    assert assert_trainable_tasks(
        contract, ["rare"], allow_exploratory_low_mass=True
    ) == (2,)


def test_validation_or_test_positive_cannot_enable_a_constant_train_task():
    y = np.asarray([[0.0], [0.0], [1.0], [1.0]])
    contract = build_system_learning_contract(
        y,
        np.ones_like(y, dtype=bool),
        ("task",),
        split_names=("train", "train", "validation", "test"),
        sampling_roles=(
            "reference_train",
            "reference_train",
            "reference_validation",
            "reference_test",
        ),
        dataset_id="synthetic",
        dataset_manifest_hash="sha256:" + "b" * 64,
        test_status="sealed_unobserved",
    )
    assert contract["tasks"][0]["status"] == "excluded_constant_zero_train"


def test_soft_event_ess_uses_design_weights_without_changing_raw_support():
    contract = build_system_learning_contract(
        np.asarray([[0.0], [0.25], [1.0], [0.5]]),
        np.ones((4, 1), dtype=bool),
        ("task",),
        split_names=("train", "train", "train", "validation"),
        sampling_roles=(
            "reference_train",
            "reference_train",
            "adaptive_train",
            "reference_validation",
        ),
        dataset_id="synthetic",
        dataset_manifest_hash="sha256:" + "e" * 64,
        test_status="sealed_unobserved",
        design_weights=np.asarray([1.0, 2.0, 4.0, 1.0]),
    )
    train = contract["tasks"][0]["support"]["all_train"]
    assert train["nonzero_count"] == 2
    assert train["soft_event_mass"] == pytest.approx(1.25)
    soft = np.asarray([0.0, 0.5, 4.0])
    assert train["soft_event_effective_sample_size"] == pytest.approx(
        soft.sum() ** 2 / np.square(soft).sum()
    )
    assert contract["selection_policy"]["system_training_eligibility_mask"] == [False]


def test_unresolved_task_definition_blocks_training_without_changing_label_mask():
    y = np.asarray([[0.0], [1.0], [0.5]])
    mask = np.ones_like(y, dtype=bool)
    contract = build_system_learning_contract(
        y,
        mask,
        ("effect-f1",),
        split_names=("train", "train", "validation"),
        sampling_roles=(
            "reference_train",
            "reference_train",
            "reference_validation",
        ),
        dataset_id="synthetic",
        dataset_manifest_hash="sha256:" + "f" * 64,
        test_status="sealed_unobserved",
        minimum_soft_event_mass={"train": 1, "validation": 1, "test": 1},
        task_definition_blockers={
            "effect-f1": [
                {
                    "component_id": "component-050",
                    "reason": "unresolved_damage_threshold_not_in_provisional_tree",
                }
            ]
        },
    )
    task = contract["tasks"][0]
    assert task["status"] == "blocked_task_definition_incomplete"
    assert task["label_available"] is True
    assert task["train_eligible"] is False
    assert task["task_definition_blockers"][0]["component_id"] == "component-050"
    assert mask.all()  # eligibility never rewrites availability semantics


def test_role_split_mixing_is_rejected():
    with pytest.raises(InputValidationError, match="Validation"):
        build_system_learning_contract(
            np.asarray([[0.0], [1.0]]),
            np.ones((2, 1), dtype=bool),
            ("task",),
            split_names=("train", "validation"),
            sampling_roles=("reference_train", "adaptive_train"),
            dataset_id="synthetic",
            dataset_manifest_hash="sha256:" + "c" * 64,
            test_status="sealed_unobserved",
        )


def test_masked_soft_losses_preserve_masks_and_separate_weight_axes():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[0.0, 0.0], [1.0, -1.0]], dtype=torch.float64)
    targets = torch.tensor([[0.0, 1.0], [0.5, 0.5]], dtype=torch.float64)
    mask = torch.tensor([[True, False], [True, True]])
    design = torch.tensor([1.0, 2.0], dtype=torch.float64)
    task = torch.tensor([1.0, 3.0], dtype=torch.float64)
    bce = masked_soft_bce_with_logits(
        logits, targets, mask, design_weight=design, task_weight=task
    )
    raw_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    weights = mask * design[:, None] * task[None, :]
    assert bce == pytest.approx(float((raw_bce * weights).sum() / weights.sum()))
    brier = masked_brier_from_logits(
        logits, targets, mask, design_weight=design, task_weight=task
    )
    raw_brier = (torch.sigmoid(logits) - targets) ** 2
    assert brier == pytest.approx(float((raw_brier * weights).sum() / weights.sum()))

    with pytest.raises(InputValidationError, match="case axes"):
        masked_soft_bce_with_logits(
            logits, targets, mask, design_weight=torch.ones(2, 2)
        )


def test_contract_rejects_out_of_range_available_soft_labels():
    with pytest.raises(InputValidationError, match=r"\[0, 1\]"):
        build_system_learning_contract(
            np.asarray([[1.2], [0.0]]),
            np.ones((2, 1), dtype=bool),
            ("task",),
            split_names=("train", "train"),
            sampling_roles=("reference_train", "reference_train"),
            dataset_id="synthetic",
            dataset_manifest_hash="sha256:" + "d" * 64,
            test_status="sealed_unobserved",
        )


def test_reference_prevalence_excludes_enriched_train_and_metrics_are_explicit():
    y = np.asarray([[0.0], [1.0], [1.0], [0.5]])
    roles = (
        "reference_train",
        "reference_train",
        "adaptive_train",
        "reference_validation",
    )
    splits = ("train", "train", "train", "validation")
    baseline = fit_reference_prevalence_baseline(
        y,
        np.ones_like(y, dtype=bool),
        ("task",),
        split_names=splits,
        sampling_roles=roles,
    )
    assert baseline["predictions"] == [0.5]
    assert baseline["support"][0]["available_count"] == 2

    report = evaluate_system_probabilities(
        y,
        np.full_like(y, 0.5),
        np.ones_like(y, dtype=bool),
        ("task",),
        evaluated_tasks=("task",),
        baseline_predictions=baseline["predictions"],
        calibration_bins=2,
    )
    assert report["evaluated_tasks"] == ["task"]
    assert report["macro_task_names"] == ["task"]
    assert report["macro_task_count"] == 1
    assert report["macro_weighting"] == "equal_weight_per_explicit_task"
    assert report["case_level_bootstrap"]["status"].startswith("not_computed")
    assert "soft-binomial" in report["metric_semantics"]["calibration"]
    assert "fractional-event-mass" in report["metric_semantics"][
        "soft_average_precision"
    ]
    assert report["tasks"][0]["brier_skill_score"] == pytest.approx(0.0)
    with pytest.raises(InputValidationError, match="explicit"):
        evaluate_system_probabilities(
            y,
            np.full_like(y, 0.5),
            np.ones_like(y, dtype=bool),
            ("task",),
            evaluated_tasks=(),
        )


def test_soft_calibration_recovers_fractional_binomial_intercept_and_slope():
    probabilities = np.linspace(0.05, 0.95, 201)
    logits = np.log(probabilities) - np.log1p(-probabilities)
    expected_intercept = -0.35
    expected_slope = 1.4
    targets = 1.0 / (
        1.0 + np.exp(-(expected_intercept + expected_slope * logits))
    )
    report = evaluate_system_probabilities(
        targets[:, None],
        probabilities[:, None],
        np.ones((len(targets), 1), dtype=bool),
        ("task",),
        evaluated_tasks=("task",),
    )
    task = report["tasks"][0]
    assert task["calibration_status"] == "defined_soft_binomial_mle"
    assert task["calibration_intercept"] == pytest.approx(
        expected_intercept, abs=1.0e-8
    )
    assert task["calibration_slope"] == pytest.approx(expected_slope, abs=1.0e-8)


def test_calibration_refuses_nonidentifiable_boundary_cases():
    constant_prediction = evaluate_system_probabilities(
        np.asarray([[0.0], [0.5], [1.0]]),
        np.full((3, 1), 0.5),
        np.ones((3, 1), dtype=bool),
        ("task",),
        evaluated_tasks=("task",),
    )["tasks"][0]
    assert constant_prediction["calibration_intercept"] is None
    assert constant_prediction["calibration_slope"] is None
    assert constant_prediction["calibration_status"] == "undefined_constant_prediction"

    no_positive = evaluate_system_probabilities(
        np.zeros((3, 1)),
        np.asarray([[0.1], [0.2], [0.3]]),
        np.ones((3, 1), dtype=bool),
        ("task",),
        evaluated_tasks=("task",),
    )["tasks"][0]
    assert no_positive["calibration_status"] == "undefined_no_positive_soft_mass"
    assert no_positive["soft_average_precision"] is None
    assert no_positive["soft_average_precision_status"] == (
        "undefined_no_positive_soft_mass"
    )


def test_soft_average_precision_is_weighted_and_invariant_within_score_ties():
    targets = np.asarray([[1.0], [0.5], [0.0], [1.0]])
    predictions = np.asarray([[0.9], [0.9], [0.5], [0.1]])
    weights = np.asarray([1.0, 2.0, 1.0, 1.0])

    first = evaluate_system_probabilities(
        targets,
        predictions,
        np.ones_like(targets, dtype=bool),
        ("task",),
        evaluated_tasks=("task",),
        design_weights=weights,
    )["tasks"][0]
    permutation = np.asarray([1, 0, 3, 2])
    second = evaluate_system_probabilities(
        targets[permutation],
        predictions[permutation],
        np.ones_like(targets, dtype=bool)[permutation],
        ("task",),
        evaluated_tasks=("task",),
        design_weights=weights[permutation],
    )["tasks"][0]

    expected = (2.0 / 3.0) * (2.0 / 3.0) + (1.0 / 3.0) * (3.0 / 5.0)
    assert first["soft_average_precision"] == pytest.approx(expected)
    assert second["soft_average_precision"] == pytest.approx(expected)
    assert first["soft_average_precision_status"] == (
        "defined_fractional_event_mass"
    )
    assert first["binary_pr_auc"] is None
    assert first["binary_pr_auc_status"].startswith("not_computed")
