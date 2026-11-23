from __future__ import annotations

import numpy as np
import pytest

from damage_engine.design_sequences import (
    DESIGN_SEQUENCE_SCHEMA_VERSION,
    design_diagnostics,
    generate_unit_design,
)
from damage_engine.errors import InputValidationError


@pytest.mark.parametrize(
    "method",
    ["randomized_lhs", "scrambled_sobol", "maximin_lhs"],
)
def test_design_sequences_are_deterministic_and_strictly_open(method: str) -> None:
    first = generate_unit_design(
        16,
        5,
        seed=1729,
        method=method,
        maximin_restarts=8,
    )
    repeated = generate_unit_design(
        16,
        5,
        seed=1729,
        method=method,
        maximin_restarts=8,
    )

    np.testing.assert_array_equal(first.points, repeated.points)
    assert first.points.shape == (16, 5)
    assert np.all(first.points > 0.0)
    assert np.all(first.points < 1.0)
    assert first.metadata["schema_version"] == DESIGN_SEQUENCE_SCHEMA_VERSION
    assert first.metadata["method"] == method
    assert first.metadata["seed"] == 1729


@pytest.mark.parametrize("method", ["randomized_lhs", "maximin_lhs"])
def test_lhs_methods_preserve_one_point_per_marginal_stratum(method: str) -> None:
    result = generate_unit_design(
        24,
        4,
        seed=23,
        method=method,
        maximin_restarts=6,
    )

    occupied = np.floor(result.points * 24).astype(np.int64)
    for axis in range(4):
        assert sorted(occupied[:, axis].tolist()) == list(range(24))

    diagnostics = design_diagnostics(result.points)
    assert diagnostics["marginal_stratification"]["all_dimensions_latin"] is True


def test_scrambled_sobol_requires_power_of_two_sample_count() -> None:
    with pytest.raises(InputValidationError, match="power of two"):
        generate_unit_design(18, 3, seed=7, method="scrambled_sobol")


def test_scrambled_sobol_records_balance_contract() -> None:
    result = generate_unit_design(32, 3, seed=7, method="scrambled_sobol")

    assert result.metadata["sample_count_is_power_of_two"] is True
    assert result.metadata["balance_contract"] == "sobol_base2_balance"
    assert result.metadata["sobol_power"] == 5


def test_maximin_restart_search_never_worsens_seed_matched_lhs() -> None:
    baseline = generate_unit_design(32, 5, seed=101, method="randomized_lhs")
    optimized = generate_unit_design(
        32,
        5,
        seed=101,
        method="maximin_lhs",
        maximin_restarts=12,
    )

    baseline_distance = design_diagnostics(baseline.points)["nearest_neighbor"][
        "minimum_distance"
    ]
    optimized_distance = design_diagnostics(optimized.points)["nearest_neighbor"][
        "minimum_distance"
    ]
    assert optimized_distance >= baseline_distance
    assert optimized.metadata["restart_count"] == 12
    assert optimized.metadata["selection_objective"] == (
        "maximize_minimum_euclidean_nearest_neighbor_distance"
    )


def test_diagnostics_cover_marginals_projections_and_spacing() -> None:
    points = np.asarray(
        [
            [0.125, 0.875],
            [0.375, 0.125],
            [0.625, 0.625],
            [0.875, 0.375],
        ],
        dtype=np.float64,
    )

    report = design_diagnostics(points)

    assert report["sample_count"] == 4
    assert report["dimension"] == 2
    assert report["marginal_stratification"]["all_dimensions_latin"] is True
    assert report["marginal_uniformity"]["maximum_ks_distance"] == pytest.approx(
        0.125
    )
    assert report["two_dimensional_projections"]["projection_count"] == 1
    assert report["two_dimensional_projections"]["maximum_centered_discrepancy"] > 0.0
    assert report["nearest_neighbor"]["minimum_distance"] > 0.0
    assert report["global_centered_discrepancy"] > 0.0


@pytest.mark.parametrize(
    "args,match",
    [
        ((0, 2), "sample_count"),
        ((4, 0), "dimension"),
        ((4, 2, -1), "seed"),
    ],
)
def test_generation_rejects_invalid_shape_or_seed(args, match: str) -> None:
    sample_count, dimension, *seed = args
    with pytest.raises(InputValidationError, match=match):
        generate_unit_design(
            sample_count,
            dimension,
            seed=seed[0] if seed else 0,
            method="randomized_lhs",
        )


def test_generation_and_diagnostics_fail_closed_on_unsupported_inputs() -> None:
    with pytest.raises(InputValidationError, match="method"):
        generate_unit_design(8, 2, seed=1, method="halton")
    with pytest.raises(InputValidationError, match="open unit cube"):
        design_diagnostics(np.asarray([[0.0, 0.25], [0.75, 1.0]]))
    with pytest.raises(InputValidationError, match="at least two"):
        design_diagnostics(np.asarray([[0.25, 0.75]]))
