from __future__ import annotations

from pathlib import Path

import pytest

from damage_engine.errors import InputValidationError
from damage_engine.tree_candidate_selection import select_and_validate_tree_candidate
from test_dataset_v2 import _candidate, make_dataset


def test_selection_requires_distinct_seeds(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "dataset")
    with pytest.raises(InputValidationError, match="must be distinct"):
        select_and_validate_tree_candidate(
            dataset,
            [_candidate()],
            development_seed=11,
            validation_seed=11,
            preregistration_id="TREE-TEST-1",
        )


def test_unreviewed_candidate_can_pass_numerically_but_cannot_be_promoted(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "dataset")
    report = select_and_validate_tree_candidate(
        dataset,
        [_candidate()],
        development_seed=11,
        validation_seed=12,
        preregistration_id="TREE-TEST-2",
    )
    assert report["selected_candidate_id"] == "identity-c1"
    assert report["validation_result"]["passes_numerical_gates"] is True
    assert report["promotion_eligible"] is False
    assert report["promotion_reason"] == "semantic_authority_not_approved"


def test_approved_candidate_passes_independent_validation_and_selection_is_deterministic(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(tmp_path / "dataset")
    first = select_and_validate_tree_candidate(
        dataset,
        [_candidate("approved")],
        development_seed=11,
        validation_seed=12,
        preregistration_id="TREE-TEST-3",
    )
    second = select_and_validate_tree_candidate(
        dataset,
        [_candidate("approved")],
        development_seed=11,
        validation_seed=12,
        preregistration_id="TREE-TEST-3",
    )
    assert first["report_hash"] == second["report_hash"]
    assert first["promotion_eligible"] is True
    assert first["seed_policy"]["selection_uses_validation_outcomes"] is False
