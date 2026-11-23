from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from damage_engine.errors import InputValidationError
from tools.run_proxy_learning_ablation import (
    ARMS,
    run_proxy_learning_ablation_arrays,
)


def _synthetic(seed: int = 12):
    rng = np.random.default_rng(seed)
    train_count = 24
    validation_count = 8
    test_count = 5
    feature_count = 6
    component_count = 4
    count = train_count + validation_count + test_count
    x = rng.normal(size=(count, feature_count))
    coefficient = rng.normal(scale=0.5, size=(feature_count, component_count))
    teacher = 1.0 / (1.0 + np.exp(-(x @ coefficient)))
    proxy = np.clip(0.75 * teacher + 0.15, 0.0, 1.0)
    mask = np.ones_like(teacher, dtype=bool)
    splits = np.asarray(
        ["train"] * train_count
        + ["validation"] * validation_count
        + ["test"] * test_count
    )
    test = splits == "test"
    mask[test] = False
    teacher[test] = np.nan
    proxy[test] = np.nan
    weight = rng.uniform(0.8, 1.2, size=count)
    pretrain_x = rng.normal(size=(40, feature_count))
    pretrain_teacher = 1.0 / (1.0 + np.exp(-(pretrain_x @ coefficient)))
    pretrain_proxy = np.clip(0.75 * pretrain_teacher + 0.15, 0.0, 1.0)
    return {
        "x": x,
        "proxy_hit_probability": proxy,
        "teacher_hit_probability": teacher,
        "label_mask": mask,
        "case_weight": weight,
        "splits": splits,
        "proxy_pretrain_x": pretrain_x,
        "proxy_pretrain_hit_probability": pretrain_proxy,
    }


def _run(values):
    return run_proxy_learning_ablation_arrays(
        **values,
        fractions=(0.5, 1.0),
        seeds=(7,),
        epochs=3,
        pretrain_epochs=2,
        hidden_size=8,
        learning_rate=1.0e-2,
        device="cpu",
    )


def test_four_arms_share_teacher_budget_and_leave_test_unopened() -> None:
    report = _run(_synthetic())
    assert report["arms"] == list(ARMS)
    assert len(report["runs"]) == 2 * len(ARMS)
    assert report["test_access_policy"] == {
        "true_labels_present_in_development_package": False,
        "open_count": 0,
        "metrics_computed": False,
        "selection_uses_test": False,
    }
    assert report["split_counts"]["test_sealed_unread"] == 5
    assert report["selected_arm"] in ARMS
    for fraction in (0.5, 1.0):
        rows = [item for item in report["runs"] if item["train_fraction"] == fraction]
        assert len({item["teacher_subset_hash"] for item in rows}) == 1
        assert len({item["teacher_train_case_count"] for item in rows}) == 1
        assert all(np.isfinite(item["validation_metrics"]["brier"]) for item in rows)


def test_test_payload_changes_do_not_change_any_result() -> None:
    first_values = _synthetic()
    second_values = deepcopy(first_values)
    test = second_values["splits"] == "test"
    second_values["teacher_hit_probability"][test] = 999.0
    second_values["proxy_hit_probability"][test] = -999.0
    first = _run(first_values)
    second = _run(second_values)
    assert first == second


def test_observed_test_mask_is_rejected() -> None:
    values = _synthetic()
    values["label_mask"][-1, 0] = True
    with pytest.raises(InputValidationError, match="sealed test label masks"):
        _run(values)


def test_proxy_pretraining_shape_is_validated() -> None:
    values = _synthetic()
    values["proxy_pretrain_hit_probability"] = values[
        "proxy_pretrain_hit_probability"
    ][:, :-1]
    with pytest.raises(InputValidationError, match="invalid or misaligned"):
        _run(values)


def test_masked_development_nan_does_not_enter_training_loss() -> None:
    values = _synthetic()
    values["label_mask"][0, 0] = False
    values["teacher_hit_probability"][0, 0] = np.nan
    report = _run(values)
    assert all(
        np.isfinite(item["validation_metrics"]["brier"])
        for item in report["runs"]
    )
