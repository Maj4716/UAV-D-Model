from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from damage_engine.errors import InputValidationError
from damage_engine.label_spec import (
    LabelSpec,
    ordinal_class_probabilities,
    threshold_ambiguity_mask,
)


def _label_spec_data() -> dict:
    return {
        "schema_version": "1.0.0",
        "label_spec_id": "learning-labels-v1",
        "revision": "1",
        "status": "provisional",
        "source": {
            "damage_tree_id": "demo-tree",
            "damage_tree_revision": "1",
            "damage_tree_definition_hash": "sha256:" + "1" * 64,
        },
        "tasks": {
            "effects": {
                "type": "multi_label_event_probability",
                "outputs": ["effect-a", "effect-b"],
                "loss": "binary_cross_entropy",
                "description": "Independent event probabilities, not a class simplex.",
            }
        },
        "provenance": {
            "approval_status": "not_approved",
            "evidence_ref": None,
        },
    }


def test_multi_label_task_preserves_independent_event_semantics() -> None:
    spec = LabelSpec(_label_spec_data())

    task = spec.task("effects")
    assert task.task_type == "multi_label_event_probability"
    assert task.outputs == ("effect-a", "effect-b")
    assert task.loss == "binary_cross_entropy"
    assert task.is_probability_task is True
    spec.validate_output_vocabulary(["effect-b", "effect-a", "unused-output"])

    single_event = deepcopy(_label_spec_data())
    single_event["tasks"]["effects"]["type"] = "event_probability"
    with pytest.raises(InputValidationError, match="exactly one output"):
        LabelSpec(single_event)


def test_multi_label_task_rejects_softmax_loss() -> None:
    data = _label_spec_data()
    data["tasks"]["effects"]["loss"] = "softmax_cross_entropy"

    with pytest.raises(InputValidationError, match="multi-label.*softmax"):
        LabelSpec(data)


def test_label_spec_rejects_outputs_absent_from_damage_tree_vocabulary() -> None:
    spec = LabelSpec(_label_spec_data())

    with pytest.raises(InputValidationError, match="absent from the damage tree") as error:
        spec.validate_output_vocabulary(["effect-a"])

    assert error.value.context == {"missing_outputs": ["effect-b"]}


def test_ordinal_cumulative_conversion_is_exact_for_valid_monotone_rows() -> None:
    cumulative = np.array([[0.8, 0.5, 0.1], [1.0, 0.25, 0.0]])

    classes = ordinal_class_probabilities(cumulative)

    np.testing.assert_allclose(
        classes,
        [[0.2, 0.3, 0.4, 0.1], [0.0, 0.75, 0.25, 0.0]],
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(classes.sum(axis=-1), 1.0, rtol=0.0, atol=1.0e-15)


def test_ordinal_cumulative_rejects_non_monotone_teacher_outputs() -> None:
    with pytest.raises(InputValidationError, match="monotonically non-increasing"):
        ordinal_class_probabilities([0.7, 0.72, 0.2])


def test_ordinal_cumulative_rejects_invalid_tolerance() -> None:
    with pytest.raises(InputValidationError, match="tolerance"):
        ordinal_class_probabilities([0.5], atol=-1.0)


def test_threshold_ambiguity_is_inclusive_at_ci_boundaries() -> None:
    ambiguity = threshold_ambiguity_mask(
        [[0.10, 0.30], [0.30, 0.40], [0.41, 0.50]],
        [0.30, 0.45],
    )
    np.testing.assert_array_equal(
        ambiguity,
        [[True, False], [True, False], [False, True]],
    )
