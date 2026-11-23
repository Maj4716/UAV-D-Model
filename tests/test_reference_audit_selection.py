from __future__ import annotations

import numpy as np
import pytest

from damage_engine.errors import InputValidationError
from damage_engine.reference_audit_selection import (
    draw_reference_audit_sample,
    reference_audit_inclusion_probabilities,
)


def test_reference_audit_keeps_positive_support_for_proxy_partial_candidates() -> None:
    strata = ["a"] * 6 + ["b"] * 6
    proxy_complete = [True, True, True, True, False, False] * 2

    design = reference_audit_inclusion_probabilities(
        strata,
        proxy_complete,
        sample_size=8,
        proxy_partial_budget_fraction=0.25,
        minimum_inclusion_probability=0.05,
    )

    assert design.probabilities.shape == (12,)
    assert np.all(design.probabilities > 0.0)
    assert np.all(design.probabilities <= 1.0)
    assert float(np.sum(design.probabilities)) == pytest.approx(8.0)
    assert design.metadata["proxy_partial_positive_inclusion_count"] == 4
    assert design.metadata["proxy_partial_expected_sample_count"] >= 2
    assert design.metadata["complete_expected_sample_count"] >= 6
    assert len(design.metadata["group_allocations"]) == 4


def test_reference_audit_draw_is_fixed_size_and_replayable() -> None:
    design = reference_audit_inclusion_probabilities(
        ["a"] * 10 + ["b"] * 10,
        [True] * 8 + [False] * 2 + [True] * 6 + [False] * 4,
        sample_size=10,
        proxy_partial_budget_fraction=0.30,
        minimum_inclusion_probability=0.05,
    )

    first = draw_reference_audit_sample(design, seed=107)
    repeated = draw_reference_audit_sample(design, seed=107)

    np.testing.assert_array_equal(first, repeated)
    assert first.dtype == np.bool_
    assert int(np.count_nonzero(first)) == 10
    assert np.count_nonzero(first & np.asarray([False] * 8 + [True] * 2 + [False] * 6 + [True] * 4)) >= 1


def test_reference_audit_without_proxy_partial_candidates_uses_complete_budget() -> None:
    design = reference_audit_inclusion_probabilities(
        ["a"] * 4 + ["b"] * 4,
        [True] * 8,
        sample_size=4,
        proxy_partial_budget_fraction=0.25,
        minimum_inclusion_probability=0.10,
    )

    assert design.metadata["proxy_partial_candidate_count"] == 0
    assert design.metadata["proxy_partial_expected_sample_count"] == 0
    assert float(np.sum(design.probabilities)) == pytest.approx(4.0)


def test_reference_audit_fails_when_minimum_positivity_exceeds_budget() -> None:
    with pytest.raises(InputValidationError, match="positivity"):
        reference_audit_inclusion_probabilities(
            ["a"] * 100 + ["b"] * 100,
            [True] * 100 + [False] * 100,
            sample_size=20,
            proxy_partial_budget_fraction=0.25,
            minimum_inclusion_probability=0.20,
        )


@pytest.mark.parametrize(
    ("strata", "complete", "sample_size", "match"),
    [
        (["a"], [True, False], 1, "same length"),
        ([""], [True], 1, "stratum"),
        (["a"], [False], 1, "complete-domain"),
        (["a"], [True], 2, "sample_size"),
    ],
)
def test_reference_audit_contract_fails_closed(
    strata, complete, sample_size: int, match: str
) -> None:
    with pytest.raises(InputValidationError, match=match):
        reference_audit_inclusion_probabilities(
            strata,
            complete,
            sample_size=sample_size,
            proxy_partial_budget_fraction=0.25,
            minimum_inclusion_probability=0.05,
        )
