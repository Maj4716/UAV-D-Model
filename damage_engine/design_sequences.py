"""Deterministic unit-cube designs and pre-outcome coverage diagnostics.

The generators in this module operate only on latent coordinates.  Geometry
admission, proposal densities and teacher outcomes remain separate contracts.
This makes a design-sequence comparison a one-factor experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import math
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

from .errors import InputValidationError


DESIGN_SEQUENCE_SCHEMA_VERSION = "unit-cube-design@1.0.0"
SUPPORTED_DESIGN_METHODS = (
    "randomized_lhs",
    "scrambled_sobol",
    "maximin_lhs",
)


@dataclass(frozen=True)
class UnitDesign:
    """One generated latent design with reproducibility metadata."""

    points: np.ndarray
    metadata: dict[str, Any]


def _positive_integer(value: Any, *, name: str) -> int:
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
        or int(value) < 1
    ):
        raise InputValidationError(f"{name} must be a positive integer")
    return int(value)


def _seed(value: Any) -> int:
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
        or int(value) < 0
    ):
        raise InputValidationError("seed must be a nonnegative integer")
    return int(value)


def _strictly_open(points: np.ndarray) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64)
    return np.clip(
        result,
        np.nextafter(0.0, 1.0),
        np.nextafter(1.0, 0.0),
    )


def _randomized_lhs(sample_count: int, dimension: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    design = np.empty((sample_count, dimension), dtype=np.float64)
    for axis in range(dimension):
        design[:, axis] = (
            rng.permutation(sample_count) + rng.random(sample_count)
        ) / sample_count
    return _strictly_open(design)


def _nearest_neighbor_distances(points: np.ndarray) -> np.ndarray:
    distances, _indices = cKDTree(points).query(points, k=2, workers=1)
    return np.asarray(distances[:, 1], dtype=np.float64)


def _restart_seed(seed: int, restart_index: int) -> int:
    if restart_index == 0:
        return seed
    generated = np.random.SeedSequence([seed, restart_index]).generate_state(
        1, dtype=np.uint64
    )
    return int(generated[0])


def _maximin_lhs(
    sample_count: int,
    dimension: int,
    *,
    seed: int,
    restarts: int,
) -> tuple[np.ndarray, int, float]:
    best_points: np.ndarray | None = None
    best_restart = -1
    best_score: tuple[float, float, int] | None = None
    for restart_index in range(restarts):
        points = _randomized_lhs(
            sample_count,
            dimension,
            seed=_restart_seed(seed, restart_index),
        )
        minimum_distance = float(np.min(_nearest_neighbor_distances(points)))
        centered_discrepancy = float(qmc.discrepancy(points, method="CD"))
        score = (minimum_distance, -centered_discrepancy, -restart_index)
        if best_score is None or score > best_score:
            best_points = points
            best_restart = restart_index
            best_score = score
    if best_points is None or best_score is None:  # pragma: no cover - validated above
        raise RuntimeError("maximin LHS restart search produced no design")
    return best_points, best_restart, best_score[0]


def _points_hash(points: np.ndarray) -> str:
    array = np.ascontiguousarray(points, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return "sha256:" + digest.hexdigest()


def generate_unit_design(
    sample_count: int,
    dimension: int,
    *,
    seed: int,
    method: str,
    maximin_restarts: int = 16,
) -> UnitDesign:
    """Generate a reproducible design in the strict open unit cube.

    Scrambled Sobol uses ``random_base2`` and therefore requires a power-of-two
    sample count.  This fail-closed rule preserves the sequence's declared
    balance property instead of silently returning a weaker arbitrary prefix.
    ``maximin_lhs`` chooses the best finite randomized-LHS restart by minimum
    Euclidean nearest-neighbor distance; it is not a global maximin optimizer.
    """

    count = _positive_integer(sample_count, name="sample_count")
    width = _positive_integer(dimension, name="dimension")
    normalized_seed = _seed(seed)
    if method not in SUPPORTED_DESIGN_METHODS:
        raise InputValidationError(
            "unsupported design method",
            context={"method": method, "supported": list(SUPPORTED_DESIGN_METHODS)},
        )

    metadata: dict[str, Any] = {
        "schema_version": DESIGN_SEQUENCE_SCHEMA_VERSION,
        "method": method,
        "sample_count": count,
        "dimension": width,
        "seed": normalized_seed,
        "strict_open_unit_cube": True,
    }
    if method == "randomized_lhs":
        points = _randomized_lhs(count, width, seed=normalized_seed)
        metadata["balance_contract"] = "one_point_per_one_dimensional_stratum"
    elif method == "scrambled_sobol":
        if count & (count - 1):
            raise InputValidationError(
                "scrambled_sobol sample_count must be a power of two"
            )
        power = count.bit_length() - 1
        points = _strictly_open(
            qmc.Sobol(d=width, scramble=True, seed=normalized_seed).random_base2(
                power
            )
        )
        metadata.update(
            {
                "sample_count_is_power_of_two": True,
                "sobol_power": power,
                "balance_contract": "sobol_base2_balance",
            }
        )
    else:
        restarts = _positive_integer(maximin_restarts, name="maximin_restarts")
        points, selected_restart, minimum_distance = _maximin_lhs(
            count,
            width,
            seed=normalized_seed,
            restarts=restarts,
        )
        metadata.update(
            {
                "balance_contract": "one_point_per_one_dimensional_stratum",
                "restart_count": restarts,
                "selected_restart_index": selected_restart,
                "selected_minimum_distance": minimum_distance,
                "selection_objective": (
                    "maximize_minimum_euclidean_nearest_neighbor_distance"
                ),
                "tie_breaker": "minimize_centered_discrepancy_then_restart_index",
            }
        )

    metadata["points_hash"] = _points_hash(points)
    return UnitDesign(points=points, metadata=metadata)


def _validated_points(points: Any) -> np.ndarray:
    try:
        array = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InputValidationError("design points must be a numeric matrix") from exc
    if array.ndim != 2 or array.shape[1] < 1:
        raise InputValidationError("design points must be a two-dimensional matrix")
    if array.shape[0] < 2:
        raise InputValidationError("design diagnostics require at least two points")
    if not np.isfinite(array).all():
        raise InputValidationError("design points must be finite")
    if np.any(array <= 0.0) or np.any(array >= 1.0):
        raise InputValidationError("design points must lie in the open unit cube")
    return array


def _uniform_ks_distance(values: np.ndarray) -> float:
    ordered = np.sort(values)
    count = len(ordered)
    upper = np.arange(1, count + 1, dtype=np.float64) / count
    lower = np.arange(count, dtype=np.float64) / count
    return float(
        max(
            np.max(upper - ordered),
            np.max(ordered - lower),
        )
    )


def design_diagnostics(points: Any) -> dict[str, Any]:
    """Measure marginal balance, 2-D projections and Euclidean spacing."""

    array = _validated_points(points)
    count, dimension = array.shape
    marginal_records: list[dict[str, Any]] = []
    for axis in range(dimension):
        bins = np.floor(array[:, axis] * count).astype(np.int64)
        occupied = int(np.unique(bins).size)
        marginal_records.append(
            {
                "axis": axis,
                "ks_distance": _uniform_ks_distance(array[:, axis]),
                "occupied_strata": occupied,
                "expected_strata": count,
                "latin_marginal": occupied == count,
            }
        )

    projection_records: list[dict[str, Any]] = []
    for first, second in itertools.combinations(range(dimension), 2):
        discrepancy = float(
            qmc.discrepancy(array[:, [first, second]], method="CD")
        )
        projection_records.append(
            {
                "axes": [first, second],
                "centered_discrepancy": discrepancy,
            }
        )
    projection_values = [
        record["centered_discrepancy"] for record in projection_records
    ]
    nearest = _nearest_neighbor_distances(array)
    marginal_ks = [record["ks_distance"] for record in marginal_records]
    return {
        "schema_version": DESIGN_SEQUENCE_SCHEMA_VERSION,
        "sample_count": count,
        "dimension": dimension,
        "points_hash": _points_hash(array),
        "marginal_stratification": {
            "all_dimensions_latin": all(
                record["latin_marginal"] for record in marginal_records
            ),
            "dimensions": marginal_records,
        },
        "marginal_uniformity": {
            "maximum_ks_distance": float(max(marginal_ks)),
            "median_ks_distance": float(np.median(marginal_ks)),
        },
        "two_dimensional_projections": {
            "projection_count": len(projection_records),
            "maximum_centered_discrepancy": (
                None if not projection_values else float(max(projection_values))
            ),
            "median_centered_discrepancy": (
                None
                if not projection_values
                else float(np.median(projection_values))
            ),
            "projections": projection_records,
        },
        "nearest_neighbor": {
            "minimum_distance": float(np.min(nearest)),
            "median_distance": float(np.median(nearest)),
            "mean_distance": float(np.mean(nearest)),
            "fifth_percentile_distance": float(np.quantile(nearest, 0.05)),
        },
        "global_centered_discrepancy": float(
            qmc.discrepancy(array, method="CD")
        ),
    }
