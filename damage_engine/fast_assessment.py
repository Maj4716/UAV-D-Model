"""Three-tier, outcome-blind fast assessment for sampling designs.

Tier 1 summarizes the declared candidate frame with weighted empirical
anisotropic ellipsoids in latent space.  Tier 2 delegates all target transport
to :class:`FastSamplingProxy` and the existing exposure-field estimands.  Tier
3 is a separate, optional paired comparison against already-produced
train/validation teacher evidence.  The core evaluator never runs a teacher.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from typing import Any

import numpy as np

from .errors import InputValidationError
from .estimands import positive_effective_sample_size
from .exposure_audit import build_exposure_teacher_audit
from .exposure_field import (
    PROXY_UNREACHED,
    build_exposure_field,
    exposure_footprint,
    exposure_row_from_proxy,
)
from .multifidelity import paired_efficiency_report
from .utils import content_hash


FAST_ASSESSMENT_VERSION = "anisotropic-fast-assessment@1.0.0"
ENGINE_COMPARISON_VERSION = "anisotropic-engine-comparison@1.0.0"

_DEFAULT_CONFIG: dict[str, Any] = {
    "rank_relative_tolerance": 1.0e-10,
    "nullspace_tolerance": 1.0e-10,
    "occupancy_bins": None,
    "reach_tolerance": 0.0,
    "ece_bins": 10,
    "teacher_positive_threshold": 0.0,
    "confidence": 0.95,
}


def _configuration(config: Mapping[str, Any] | None) -> dict[str, Any]:
    if config is not None and not isinstance(config, Mapping):
        raise InputValidationError("fast-assessment config must be a mapping")
    result = {**_DEFAULT_CONFIG, **({} if config is None else dict(config))}
    for field in ("rank_relative_tolerance", "nullspace_tolerance"):
        try:
            value = float(result[field])
        except (TypeError, ValueError, OverflowError) as exc:
            raise InputValidationError(f"{field} must be finite and positive") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise InputValidationError(f"{field} must be finite and positive")
        result[field] = value
    try:
        reach_tolerance = float(result["reach_tolerance"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputValidationError(
            "reach_tolerance must be finite and non-negative"
        ) from exc
    if not math.isfinite(reach_tolerance) or reach_tolerance < 0.0:
        raise InputValidationError(
            "reach_tolerance must be finite and non-negative"
        )
    result["reach_tolerance"] = reach_tolerance
    occupancy_bins = result["occupancy_bins"]
    if occupancy_bins is not None and (
        not isinstance(occupancy_bins, int)
        or isinstance(occupancy_bins, bool)
        or occupancy_bins < 2
    ):
        raise InputValidationError("occupancy_bins must be null or an integer >= 2")
    if (
        not isinstance(result["ece_bins"], int)
        or isinstance(result["ece_bins"], bool)
        or result["ece_bins"] < 2
    ):
        raise InputValidationError("ece_bins must be an integer >= 2")
    try:
        threshold = float(result["teacher_positive_threshold"])
        confidence = float(result["confidence"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputValidationError(
            "teacher comparison thresholds must be finite"
        ) from exc
    if not math.isfinite(threshold) or not 0.0 <= threshold < 1.0:
        raise InputValidationError(
            "teacher_positive_threshold must lie in [0, 1)"
        )
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise InputValidationError("confidence must lie in (0, 1)")
    result["teacher_positive_threshold"] = threshold
    result["confidence"] = confidence
    return result


def _positive_float(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputValidationError(f"{field} must be finite and positive") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise InputValidationError(f"{field} must be finite and positive")
    return result


def _latent_from_row(row: Mapping[str, Any], case: Mapping[str, Any]) -> np.ndarray:
    value = row.get("latent")
    if value is None:
        metadata = case.get("metadata", {})
        if isinstance(metadata, Mapping):
            for record_name in (
                "continuous_direction_sampling",
                "sphere_entry_ray_sampling",
            ):
                record = metadata.get(record_name)
                if not isinstance(record, Mapping):
                    continue
                value = record.get("physical_latent")
                if value is None:
                    value = record.get("latent")
                if value is not None:
                    break
    try:
        latent = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputValidationError(
            "each fast-assessment row requires a numeric latent vector"
        ) from exc
    if (
        latent.ndim != 1
        or latent.size == 0
        or not np.isfinite(latent).all()
        or np.any(latent < 0.0)
        or np.any(latent > 1.0)
    ):
        raise InputValidationError(
            "fast-assessment latent vectors must be nonempty, finite and lie in [0, 1]"
        )
    return latent


def _inferred_cell(case: Mapping[str, Any]) -> str:
    metadata = case.get("metadata", {})
    direction: Any = "unclassified_direction"
    if isinstance(metadata, Mapping):
        for name in (
            "continuous_direction_sampling",
            "sphere_entry_ray_sampling",
            "contact_manifold_sampling",
        ):
            record = metadata.get(name)
            if not isinstance(record, Mapping):
                continue
            direction = record.get(
                "region_id", record.get("approach_id", direction)
            )
            if direction != "unclassified_direction":
                break
    munition = case.get("munition", {})
    profile = (
        munition.get("profile_id", "unclassified_profile")
        if isinstance(munition, Mapping)
        else "unclassified_profile"
    )
    return f"{direction}::{profile}"


def _canonical_case_rows(
    case_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    rows = list(case_rows)
    if not rows:
        raise InputValidationError("fast assessment requires at least one case row")
    canonical: list[dict[str, Any]] = []
    latent_rows: list[np.ndarray] = []
    dimension: int | None = None
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise InputValidationError("each fast-assessment case row must be a mapping")
        case = row.get("case", row)
        if not isinstance(case, Mapping):
            raise InputValidationError("each fast-assessment row requires a case mapping")
        candidate_id = row.get("candidate_id", case.get("case_id", f"case-{index}"))
        if not isinstance(candidate_id, str) or not candidate_id:
            raise InputValidationError("candidate_id must be a nonempty string")
        if candidate_id in seen:
            raise InputValidationError("fast-assessment candidate IDs must be unique")
        seen.add(candidate_id)
        latent = _latent_from_row(row, case)
        if dimension is None:
            dimension = int(latent.size)
        elif latent.size != dimension:
            raise InputValidationError(
                "all fast-assessment latent vectors must have the same dimension"
            )
        cell_id = row.get("cell_id") or _inferred_cell(case)
        if not isinstance(cell_id, str) or not cell_id:
            raise InputValidationError("cell_id must be a nonempty string")
        canonical.append(
            {
                "candidate_id": candidate_id,
                "cell_id": cell_id,
                "measure_weight": _positive_float(
                    row.get("measure_weight", 1.0), field="measure_weight"
                ),
                "case": case,
            }
        )
        latent_rows.append(latent)
    return canonical, np.vstack(latent_rows)


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights) / float(np.sum(ordered_weights))
    index = min(
        int(np.searchsorted(cumulative, float(quantile), side="left")),
        values.size - 1,
    )
    return float(ordered_values[index])


def _uniform_ks(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    normalized = weights[order] / float(np.sum(weights))
    cumulative = np.cumsum(normalized)
    previous = np.concatenate(([0.0], cumulative[:-1]))
    return float(
        max(
            np.max(np.abs(cumulative - ordered)),
            np.max(np.abs(previous - ordered)),
        )
    )


def _ball(
    features: np.ndarray,
    weights: np.ndarray,
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    total = float(np.sum(weights))
    mean = np.sum(features * weights[:, None], axis=0) / total
    centered = features - mean
    covariance = (centered * weights[:, None]).T @ centered / total
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues, kind="stable")[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    maximum = float(eigenvalues[0]) if eigenvalues.size else 0.0
    cutoff = float(config["rank_relative_tolerance"]) * max(maximum, 1.0)
    active = eigenvalues > cutoff
    effective_rank = int(np.count_nonzero(active))
    if effective_rank:
        projected = centered @ eigenvectors[:, active]
        squared_radius = np.sum(
            projected * projected / eigenvalues[active][None, :], axis=1
        )
    else:
        squared_radius = np.zeros(features.shape[0], dtype=np.float64)
    squared_radius = np.maximum(squared_radius, 0.0)
    radius = np.sqrt(squared_radius)
    bins = config["occupancy_bins"]
    if bins is None:
        bins = max(2, min(10, int(round(math.sqrt(features.shape[0])))))
    marginal = []
    for axis in range(features.shape[1]):
        histogram = np.histogram(
            features[:, axis], bins=int(bins), range=(0.0, 1.0)
        )[0]
        marginal.append(
            {
                "axis": axis,
                "uniform_ks": _uniform_ks(features[:, axis], weights),
                "occupancy_fraction": float(np.count_nonzero(histogram) / int(bins)),
            }
        )
    return {
        "count": int(features.shape[0]),
        "dimension": int(features.shape[1]),
        "weight_sum": total,
        "measure_ess": positive_effective_sample_size(weights),
        "mean": mean.tolist(),
        "covariance": covariance.tolist(),
        "eigenvalues_descending": eigenvalues.tolist(),
        "eigenvectors_columns_descending": eigenvectors.tolist(),
        "effective_rank": effective_rank,
        "rank_cutoff": cutoff,
        "mahalanobis_radius": {
            f"q{int(100 * quantile):02d}": _weighted_quantile(
                radius, weights, quantile
            )
            for quantile in (0.50, 0.90, 0.95, 0.99)
        },
        "uniform_marginals": marginal,
        "occupancy_bins": int(bins),
        "distribution_semantics": (
            "weighted empirical latent-space ellipsoid; no Gaussian assumption"
        ),
    }


def _ball_collection(
    canonical: Sequence[Mapping[str, Any]],
    features: np.ndarray,
    weights: np.ndarray,
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    cells = [str(row["cell_id"]) for row in canonical]
    return {
        "global": _ball(features, weights, config=config),
        "cells": {
            cell: _ball(
                features[np.asarray([value == cell for value in cells])],
                weights[np.asarray([value == cell for value in cells])],
                config=config,
            )
            for cell in sorted(set(cells))
        },
    }


def _cell_mass(
    cells: Sequence[str], weights: np.ndarray, universe: Sequence[str]
) -> dict[str, float]:
    total = float(np.sum(weights))
    return {
        cell: float(
            np.sum(weights[np.asarray([value == cell for value in cells])]) / total
        )
        for cell in universe
    }


def _declared_cell_comparison(
    canonical: Sequence[Mapping[str, Any]],
    weights: np.ndarray,
    *,
    config: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    cells = [str(row["cell_id"]) for row in canonical]
    universe = sorted(set(cells))
    realized = _cell_mass(cells, weights, universe)
    declared_value = config.get("expected_cell_mass")
    if declared_value is None:
        return realized, {"status": "not_requested"}
    if not isinstance(declared_value, Mapping):
        raise InputValidationError("expected_cell_mass must be a cell-to-mass mapping")
    if set(str(value) for value in declared_value) != set(universe):
        raise InputValidationError(
            "expected_cell_mass identities must equal the realized cell frame"
        )
    declared: dict[str, float] = {}
    for cell in universe:
        try:
            value = float(declared_value[cell])
        except (TypeError, ValueError, OverflowError) as exc:
            raise InputValidationError(
                "expected_cell_mass values must be finite and non-negative"
            ) from exc
        if not math.isfinite(value) or value < 0.0:
            raise InputValidationError(
                "expected_cell_mass values must be finite and non-negative"
            )
        declared[cell] = value
    total = math.fsum(declared.values())
    if total <= 0.0:
        raise InputValidationError("expected_cell_mass must contain positive mass")
    declared = {cell: value / total for cell, value in declared.items()}
    delta = {cell: realized[cell] - declared[cell] for cell in universe}
    return realized, {
        "status": "evaluated",
        "declared_cell_mass": declared,
        "realized_cell_mass": realized,
        "realized_minus_declared": delta,
        "total_variation": 0.5 * math.fsum(abs(value) for value in delta.values()),
        "semantics": config.get(
            "expected_cell_mass_semantics",
            "caller_declared_reference_cell_mass",
        ),
    }


def _outside_reference_q95(
    reference_features: np.ndarray,
    reference_weights: np.ndarray,
    selected_features: np.ndarray,
    selected_weights: np.ndarray,
    *,
    config: Mapping[str, Any],
) -> float:
    total = float(np.sum(reference_weights))
    mean = np.sum(reference_features * reference_weights[:, None], axis=0) / total
    centered = reference_features - mean
    covariance = (centered * reference_weights[:, None]).T @ centered / total
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues, kind="stable")[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    maximum = float(eigenvalues[0]) if eigenvalues.size else 0.0
    cutoff = float(config["rank_relative_tolerance"]) * max(maximum, 1.0)
    active = eigenvalues > cutoff

    def distances(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        delta = values - mean
        if np.any(active):
            projected = delta @ eigenvectors[:, active]
            squared = np.sum(
                projected * projected / eigenvalues[active][None, :], axis=1
            )
        else:
            squared = np.zeros(values.shape[0], dtype=np.float64)
        if np.any(~active):
            null_projection = delta @ eigenvectors[:, ~active]
            off_support = np.linalg.norm(null_projection, axis=1) > float(
                config["nullspace_tolerance"]
            )
        else:
            off_support = np.zeros(values.shape[0], dtype=bool)
        return np.sqrt(np.maximum(squared, 0.0)), off_support

    reference_radius, _ = distances(reference_features)
    q95 = _weighted_quantile(reference_radius, reference_weights, 0.95)
    selected_radius, off_support = distances(selected_features)
    outside = off_support | (selected_radius > q95)
    return float(np.sum(selected_weights[outside]) / np.sum(selected_weights))


def _ball_gap(
    reference_features: np.ndarray,
    reference_weights: np.ndarray,
    selected_features: np.ndarray,
    selected_weights: np.ndarray,
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    reference = _ball(reference_features, reference_weights, config=config)
    selected = _ball(selected_features, selected_weights, config=config)
    reference_mean = np.asarray(reference["mean"], dtype=np.float64)
    selected_mean = np.asarray(selected["mean"], dtype=np.float64)
    reference_covariance = np.asarray(reference["covariance"], dtype=np.float64)
    selected_covariance = np.asarray(selected["covariance"], dtype=np.float64)
    covariance_gap = float(
        np.linalg.norm(selected_covariance - reference_covariance, ord="fro")
    )
    denominator = float(np.linalg.norm(reference_covariance, ord="fro"))
    return {
        "mean_shift_l2": float(np.linalg.norm(selected_mean - reference_mean)),
        "covariance_frobenius_gap": covariance_gap,
        "covariance_relative_frobenius_gap": (
            covariance_gap if denominator <= 0.0 else covariance_gap / denominator
        ),
        "reference_q95_outside_mass": _outside_reference_q95(
            reference_features,
            reference_weights,
            selected_features,
            selected_weights,
            config=config,
        ),
    }


def _selection_report(
    canonical: Sequence[Mapping[str, Any]],
    features: np.ndarray,
    *,
    selected_ids: Sequence[str] | None,
    inclusion_probabilities: Mapping[str, float] | None,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if selected_ids is None and inclusion_probabilities is None:
        return {"status": "not_requested"}
    if selected_ids is None or inclusion_probabilities is None:
        raise InputValidationError(
            "selected_ids and inclusion_probabilities must be supplied together"
        )
    selected = list(selected_ids)
    if not selected or len(selected) != len(set(selected)):
        raise InputValidationError("selected_ids must be nonempty and unique")
    id_to_index = {
        str(row["candidate_id"]): index for index, row in enumerate(canonical)
    }
    try:
        indices = np.asarray([id_to_index[value] for value in selected], dtype=np.int64)
    except KeyError as exc:
        raise InputValidationError(
            "selected candidate is outside the fast-assessment frame"
        ) from exc
    probabilities = np.asarray(
        [
            _positive_float(
                inclusion_probabilities.get(value),
                field=f"inclusion_probability.{value}",
            )
            for value in selected
        ],
        dtype=np.float64,
    )
    if np.any(probabilities > 1.0):
        raise InputValidationError("inclusion probabilities must lie in (0, 1]")
    reference_weights = np.asarray(
        [float(row["measure_weight"]) for row in canonical], dtype=np.float64
    )
    raw_weights = np.ones(indices.size, dtype=np.float64)
    recovered_weights = reference_weights[indices] / probabilities
    reference_cells = [str(row["cell_id"]) for row in canonical]
    selected_cells = [reference_cells[index] for index in indices]
    cell_universe = sorted(set(reference_cells))
    reference_mass = _cell_mass(reference_cells, reference_weights, cell_universe)
    raw_mass = _cell_mass(selected_cells, raw_weights, cell_universe)
    recovered_mass = _cell_mass(selected_cells, recovered_weights, cell_universe)

    def tv(right: Mapping[str, float]) -> float:
        return 0.5 * math.fsum(
            abs(reference_mass[cell] - right[cell]) for cell in cell_universe
        )

    selected_canonical = [canonical[index] for index in indices]
    return {
        "status": "evaluated",
        "selected_count": len(selected),
        "selected_candidate_ids": selected,
        "raw_selection": {
            "balls": _ball_collection(
                selected_canonical,
                features[indices],
                raw_weights,
                config=config,
            ),
            "cell_mass": raw_mass,
            "cell_mass_total_variation": tv(raw_mass),
            "gap_from_reference": _ball_gap(
                features,
                reference_weights,
                features[indices],
                raw_weights,
                config=config,
            ),
        },
        "inverse_probability_recovered": {
            "balls": _ball_collection(
                selected_canonical,
                features[indices],
                recovered_weights,
                config=config,
            ),
            "cell_mass": recovered_mass,
            "cell_mass_total_variation": tv(recovered_mass),
            "design_ess": positive_effective_sample_size(recovered_weights),
            "design_ess_fraction": positive_effective_sample_size(recovered_weights)
            / len(recovered_weights),
            "gap_from_reference": _ball_gap(
                features,
                reference_weights,
                features[indices],
                recovered_weights,
                config=config,
            ),
        },
        "reference_cell_mass": reference_mass,
    }


def _proxy_result(proxy: Any, case: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        result = proxy.evaluate(case, output_scope="exposure")
    except TypeError as exc:
        if "output_scope" not in str(exc):
            raise
        result = proxy.evaluate(case)
    if not isinstance(result, Mapping):
        raise InputValidationError("proxy.evaluate must return a mapping")
    return result


def _aligned_exposure_rows(
    canonical: Sequence[Mapping[str, Any]],
    *,
    proxy: Any | None,
    exposure_rows: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    if (proxy is None) == (exposure_rows is None):
        raise InputValidationError("supply exactly one of proxy or exposure_rows")
    if proxy is not None:
        rows = []
        for source in canonical:
            result = _proxy_result(proxy, source["case"])
            rows.append(
                exposure_row_from_proxy(
                    result,
                    candidate_id=str(source["candidate_id"]),
                    cell_id=str(source["cell_id"]),
                    measure_weight=float(source["measure_weight"]),
                )
            )
    else:
        rows = [dict(row) for row in exposure_rows or []]
    expected = {str(row["candidate_id"]) for row in canonical}
    observed = [str(row.get("candidate_id")) for row in rows]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise InputValidationError(
            "exposure rows must align one-to-one with fast-assessment case rows"
        )
    by_id = {str(row["candidate_id"]): row for row in rows}
    return [dict(by_id[str(source["candidate_id"])]) for source in canonical]


def _domain_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    by_cell: dict[str, dict[str, int]] = {}
    for row in rows:
        status = str(row.get("proxy_domain_status", "not_reported"))
        cell = str(row["cell_id"])
        counts[status] = counts.get(status, 0) + 1
        local = by_cell.setdefault(cell, {})
        local[status] = local.get(status, 0) + 1
    complete = counts.get("complete_proxy_domain", 0)
    return {
        "candidate_count": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "complete_case_fraction": complete / len(rows),
        "cell_status_counts": {
            cell: dict(sorted(values.items())) for cell, values in sorted(by_cell.items())
        },
    }


def _json_finite(value: Mapping[str, Any]) -> None:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(
            "fast-assessment output must be finite strict JSON"
        ) from exc


def evaluate_fast_assessment(
    case_rows: Sequence[Mapping[str, Any]],
    *,
    proxy: Any | None = None,
    exposure_rows: Sequence[Mapping[str, Any]] | None = None,
    selected_ids: Sequence[str] | None = None,
    inclusion_probabilities: Mapping[str, float] | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate generator coverage and proxy exposure without teacher outcomes."""

    resolved_config = _configuration(config)
    canonical, features = _canonical_case_rows(case_rows)
    reference_weights = np.asarray(
        [float(row["measure_weight"]) for row in canonical], dtype=np.float64
    )
    reference_cell_mass, declared_cell_comparison = _declared_cell_comparison(
        canonical,
        reference_weights,
        config=resolved_config,
    )
    tier_1 = {
        "status": "evaluated",
        "feature_semantics": (
            "declared pre-outcome latent coordinates; empirical ellipsoids are "
            "diagnostics, not Gaussian probability models"
        ),
        "reference": _ball_collection(
            canonical, features, reference_weights, config=resolved_config
        ),
        "reference_cell_mass": reference_cell_mass,
        "declared_cell_comparison": declared_cell_comparison,
        "selection_comparison": _selection_report(
            canonical,
            features,
            selected_ids=selected_ids,
            inclusion_probabilities=inclusion_probabilities,
            config=resolved_config,
        ),
    }
    rows = _aligned_exposure_rows(
        canonical, proxy=proxy, exposure_rows=exposure_rows
    )
    field = build_exposure_field(
        rows,
        reach_tolerance=float(resolved_config["reach_tolerance"]),
        include_candidate_rows=True,
    )
    footprint: dict[str, Any]
    if selected_ids is None:
        footprint = {"status": "not_requested"}
    else:
        footprint = exposure_footprint(
            rows, selected_ids, inclusion_probabilities or {}
        )
    tier_2 = {
        "status": "evaluated",
        "exposure_field": field,
        "sampling_footprint": footprint,
        "proxy_domain_coverage": _domain_report(rows),
        "proxy_zero_semantics": PROXY_UNREACHED,
        "structural_zero_claim": False,
    }
    assessment: dict[str, Any] = {
        "schema_version": FAST_ASSESSMENT_VERSION,
        "candidate_count": len(canonical),
        "candidate_ids": [str(row["candidate_id"]) for row in canonical],
        "config": resolved_config,
        "tier_1_generator": tier_1,
        "tier_2_geometry_proxy": tier_2,
        "tier_3_engine": {
            "status": "not_evaluated",
            "reason": "teacher evidence was not requested by the outcome-blind core",
        },
        "separation_contract": {
            "generator_diagnostics": "tier_1_and_tier_2_sampling_footprint",
            "engine_diagnostics": "separate_fixed_case_paired_comparison",
            "formal_label_admissible": False,
        },
        "evidence_boundary": (
            "Pre-outcome generator and fast-proxy assessment. Proxy outputs are "
            "not formal labels, operational probabilities, or physical validation."
        ),
    }
    assessment["assessment_hash"] = content_hash(assessment)
    _json_finite(assessment)
    return assessment


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _weighted_correlation(
    left: np.ndarray, right: np.ndarray, weights: np.ndarray
) -> float | None:
    total = float(np.sum(weights))
    left_mean = float(np.sum(weights * left) / total)
    right_mean = float(np.sum(weights * right) / total)
    left_centered = left - left_mean
    right_centered = right - right_mean
    left_variance = float(np.sum(weights * left_centered * left_centered) / total)
    right_variance = float(np.sum(weights * right_centered * right_centered) / total)
    if left_variance <= 0.0 or right_variance <= 0.0:
        return None
    value = float(
        np.sum(weights * left_centered * right_centered)
        / (total * math.sqrt(left_variance * right_variance))
    )
    return float(np.clip(value, -1.0, 1.0))


def _paired_metrics(
    proxy_probability: np.ndarray,
    teacher_probability: np.ndarray,
    weights: np.ndarray,
    *,
    ece_bins: int,
    case_count: int,
    component_count: int,
) -> dict[str, Any]:
    total = float(np.sum(weights))
    residual = proxy_probability - teacher_probability
    absolute = np.abs(residual)
    squared = residual * residual
    bin_index = np.minimum(
        (proxy_probability * ece_bins).astype(np.int64), ece_bins - 1
    )
    ece = 0.0
    occupied = 0
    for index in range(ece_bins):
        mask = bin_index == index
        if not np.any(mask):
            continue
        occupied += 1
        local_weights = weights[mask]
        local_total = float(np.sum(local_weights))
        predicted = float(
            np.sum(local_weights * proxy_probability[mask]) / local_total
        )
        observed = float(
            np.sum(local_weights * teacher_probability[mask]) / local_total
        )
        ece += local_total / total * abs(predicted - observed)
    return {
        "case_count": int(case_count),
        "component_count": int(component_count),
        "pair_count": int(proxy_probability.size),
        "weight_sum": total,
        "weighted_probability_mse": float(np.sum(weights * squared) / total),
        "weighted_brier": float(np.sum(weights * squared) / total),
        "weighted_ece": float(ece),
        "weighted_mae": float(np.sum(weights * absolute) / total),
        "weighted_rmse": math.sqrt(float(np.sum(weights * squared) / total)),
        "weighted_bias_proxy_minus_teacher": float(
            np.sum(weights * residual) / total
        ),
        "pearson": _weighted_correlation(
            proxy_probability, teacher_probability, weights
        ),
        "spearman": _weighted_correlation(
            _average_ranks(proxy_probability),
            _average_ranks(teacher_probability),
            weights,
        ),
        "ece_bins": ece_bins,
        "ece_occupied_bins": occupied,
        "probability_mapping": "p_proxy=1-exp(-component_lambda)",
    }


def _reject_sealed_role(value: Any, *, field: str) -> None:
    if value is None:
        return
    normalized = str(value).strip().lower().replace("_", "-")
    if "test" in normalized or "sealed" in normalized:
        raise InputValidationError(
            f"{field} cannot be test or sealed for engine comparison"
        )


def _normalized_role(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("_", "-")
    return normalized or None


def _teacher_role(row: Mapping[str, Any]) -> str | None:
    observed = {
        value
        for field_name in ("split_role", "split", "frozen_split", "role")
        if (value := _normalized_role(row.get(field_name))) is not None
    }
    if len(observed) > 1:
        raise InputValidationError("teacher row declares conflicting split roles")
    return next(iter(observed), None)


def _validate_declared_role(
    split_role: str,
    role_counts: Mapping[str, int],
) -> None:
    """Require the aggregate role label to describe explicit row roles."""

    declared = _normalized_role(split_role)
    assert declared is not None
    observed = {role for role in role_counts if role != "unspecified"}
    if not observed:
        return
    missing = sorted(role for role in observed if role not in declared)
    if missing:
        raise InputValidationError(
            "split_role does not describe all explicit teacher row roles"
        )


def compare_engine_evidence(
    assessment: Mapping[str, Any],
    teacher_rows: Sequence[Mapping[str, Any]],
    *,
    split_role: str = "validation",
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare fixed-case proxy exposure with existing non-test teacher evidence."""

    _reject_sealed_role(split_role, field="split_role")
    if not isinstance(assessment, Mapping) or assessment.get(
        "schema_version"
    ) != FAST_ASSESSMENT_VERSION:
        raise InputValidationError("unsupported fast-assessment contract")
    resolved_config = _configuration(config or assessment.get("config"))
    tier_2 = assessment.get("tier_2_geometry_proxy")
    if not isinstance(tier_2, Mapping):
        raise InputValidationError("assessment lacks tier_2_geometry_proxy")
    field = tier_2.get("exposure_field")
    rows = field.get("candidate_rows") if isinstance(field, Mapping) else None
    if not isinstance(rows, list) or not rows:
        raise InputValidationError(
            "assessment exposure field must retain candidate_rows"
        )
    teacher = list(teacher_rows)
    if not teacher:
        raise InputValidationError("engine comparison requires teacher rows")
    role_counts: dict[str, int] = {}
    for row in teacher:
        if not isinstance(row, Mapping):
            raise InputValidationError("each teacher row must be a mapping")
        for field_name in ("split_role", "split", "frozen_split", "role"):
            _reject_sealed_role(row.get(field_name), field=f"teacher.{field_name}")
        role = _teacher_role(row) or "unspecified"
        role_counts[role] = role_counts.get(role, 0) + 1
    _validate_declared_role(split_role, role_counts)

    exposure_by_id = {str(row["candidate_id"]): row for row in rows}
    teacher_by_id: dict[str, Mapping[str, Any]] = {}
    for row in teacher:
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise InputValidationError("teacher candidate_id must be nonempty")
        if candidate_id in teacher_by_id:
            raise InputValidationError("teacher candidate IDs must be unique")
        teacher_by_id[candidate_id] = row
    if not set(teacher_by_id).issubset(exposure_by_id):
        raise InputValidationError("teacher rows must be an assessment subset")

    ordered_ids = [
        str(row["candidate_id"])
        for row in rows
        if str(row["candidate_id"]) in teacher_by_id
    ]
    component_ids = sorted(
        str(value) for value in exposure_by_id[ordered_ids[0]]["component_lambda"]
    )
    proxy_matrix: list[list[float]] = []
    teacher_matrix: list[list[float]] = []
    case_weights: list[float] = []
    case_cells: list[str] = []
    compact_teacher: list[dict[str, Any]] = []
    for candidate_id in ordered_ids:
        exposure = exposure_by_id[candidate_id]
        lambdas = exposure.get("component_lambda")
        probabilities = teacher_by_id[candidate_id].get("component_hit_probability")
        if not isinstance(lambdas, Mapping) or not isinstance(probabilities, Mapping):
            raise InputValidationError(
                "teacher comparison requires aligned component mappings"
            )
        if sorted(str(value) for value in lambdas) != component_ids or sorted(
            str(value) for value in probabilities
        ) != component_ids:
            raise InputValidationError(
                "proxy and teacher component identities must align"
            )
        proxy_values: list[float] = []
        teacher_values: list[float] = []
        for component_id in component_ids:
            lambda_value = float(lambdas[component_id])
            teacher_value = float(probabilities[component_id])
            if (
                not math.isfinite(lambda_value)
                or lambda_value < 0.0
                or not math.isfinite(teacher_value)
                or not 0.0 <= teacher_value <= 1.0
            ):
                raise InputValidationError(
                    "proxy/teacher comparison values are outside their domains"
                )
            proxy_values.append(float(-math.expm1(-lambda_value)))
            teacher_values.append(teacher_value)
        proxy_matrix.append(proxy_values)
        teacher_matrix.append(teacher_values)
        case_weights.append(
            _positive_float(exposure.get("measure_weight"), field="measure_weight")
        )
        case_cells.append(str(exposure["cell_id"]))
        compact_teacher.append(
            {
                "candidate_id": candidate_id,
                "component_hit_probability": {
                    component_id: teacher_values[index]
                    for index, component_id in enumerate(component_ids)
                },
            }
        )

    proxy_array = np.asarray(proxy_matrix, dtype=np.float64)
    teacher_array = np.asarray(teacher_matrix, dtype=np.float64)
    weights_array = np.asarray(case_weights, dtype=np.float64)
    flattened_weights = np.repeat(weights_array, len(component_ids))
    overall = _paired_metrics(
        proxy_array.ravel(),
        teacher_array.ravel(),
        flattened_weights,
        ece_bins=int(resolved_config["ece_bins"]),
        case_count=len(ordered_ids),
        component_count=len(component_ids),
    )
    cells: dict[str, Any] = {}
    for cell in sorted(set(case_cells)):
        indices = np.asarray([value == cell for value in case_cells], dtype=bool)
        local_weights = np.repeat(weights_array[indices], len(component_ids))
        cells[cell] = _paired_metrics(
            proxy_array[indices].ravel(),
            teacher_array[indices].ravel(),
            local_weights,
            ece_bins=int(resolved_config["ece_bins"]),
            case_count=int(np.count_nonzero(indices)),
            component_count=len(component_ids),
        )
    worst_cell = max(
        cells,
        key=lambda name: (float(cells[name]["weighted_rmse"]), name),
    )
    reachability = build_exposure_teacher_audit(
        rows,
        compact_teacher,
        teacher_positive_threshold=float(
            resolved_config["teacher_positive_threshold"]
        ),
        confidence=float(resolved_config["confidence"]),
    )
    efficiency: dict[str, Any]
    if len(ordered_ids) >= 3:
        efficiency = {
            "status": "evaluated",
            **paired_efficiency_report(proxy_array, teacher_array),
        }
    else:
        efficiency = {
            "status": "not_estimable",
            "reason": "paired_efficiency_report requires at least three cases",
        }
    result: dict[str, Any] = {
        "schema_version": ENGINE_COMPARISON_VERSION,
        "assessment_hash": assessment.get("assessment_hash"),
        "status": "evaluated",
        "split_role": split_role,
        "teacher_role_counts": dict(sorted(role_counts.items())),
        "paired_candidate_ids": ordered_ids,
        "component_ids": component_ids,
        "overall": overall,
        "cells": cells,
        "worst_cell": {
            "criterion": "maximum_weighted_rmse",
            "cell_id": worst_cell,
            "weighted_rmse": cells[worst_cell]["weighted_rmse"],
        },
        "reachability_audit": reachability,
        "paired_efficiency": efficiency,
        "proxy_zero_semantics": PROXY_UNREACHED,
        "structural_zero_claim": False,
        "scope": "software_teacher_comparison_not_external_physical_validation",
    }
    result["comparison_hash"] = content_hash(result)
    _json_finite(result)
    return result


__all__ = [
    "ENGINE_COMPARISON_VERSION",
    "FAST_ASSESSMENT_VERSION",
    "compare_engine_evidence",
    "evaluate_fast_assessment",
]
