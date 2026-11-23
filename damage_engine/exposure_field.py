"""Component exposure-field estimands built from ``FastSamplingProxy`` rows.

The field is a finite-frame proxy estimand.  A component with zero proxy reach
is reported as ``proxy_unreached_in_declared_frame``; that status never asserts
that the physical system has a structural zero.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np

from .errors import InputValidationError
from .estimands import (
    nonnegative_effective_sample_size,
    positive_effective_sample_size,
    positive_weighted_mean,
)


EXPOSURE_FIELD_SCHEMA_VERSION = "component-exposure-field@1.0.0"
PROXY_REACHED = "proxy_reached_in_declared_frame"
PROXY_UNREACHED = "proxy_unreached_in_declared_frame"


def _nonnegative(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputValidationError(f"{field} must be finite and non-negative") from exc
    if not math.isfinite(result) or result < 0.0:
        raise InputValidationError(f"{field} must be finite and non-negative")
    return result


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights / float(np.sum(ordered_weights)))
    index = min(int(np.searchsorted(cumulative, quantile, side="left")), values.size - 1)
    return float(ordered_values[index])


def exposure_row_from_proxy(
    proxy_result: Mapping[str, Any],
    *,
    candidate_id: str | None = None,
    cell_id: str = "all",
    measure_weight: float = 1.0,
) -> dict[str, Any]:
    """Convert one proxy result into the compact field-row contract."""

    metrics = proxy_result.get("component_metrics")
    if not isinstance(metrics, Mapping) or not metrics:
        raise InputValidationError("proxy result requires nonempty component_metrics")
    weight = _nonnegative(measure_weight, field="measure_weight")
    if weight <= 0.0:
        raise InputValidationError("measure_weight must be positive")
    identifier = candidate_id or proxy_result.get("case_id")
    if not isinstance(identifier, str) or not identifier:
        raise InputValidationError("exposure row candidate_id must be nonempty")
    if not isinstance(cell_id, str) or not cell_id:
        raise InputValidationError("exposure row cell_id must be nonempty")
    component_lambda: dict[str, float] = {}
    component_energy: dict[str, float] = {}
    for component_id in sorted(str(value) for value in metrics):
        record = metrics[component_id]
        if not isinstance(record, Mapping):
            raise InputValidationError("each proxy component metric must be an object")
        component_lambda[component_id] = _nonnegative(
            record.get("expected_hits"),
            field=f"component_metrics.{component_id}.expected_hits",
        )
        component_energy[component_id] = _nonnegative(
            record.get("expected_damage_energy_j"),
            field=f"component_metrics.{component_id}.expected_damage_energy_j",
        )
    return {
        "candidate_id": identifier,
        "cell_id": cell_id,
        "measure_weight": weight,
        "proxy_domain_status": str(
            proxy_result.get("proxy_domain_status", "not_reported")
        ),
        "component_lambda": component_lambda,
        "component_damage_energy_j": component_energy,
    }


def _validated_arrays(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if not rows:
        raise InputValidationError("exposure field requires at least one row")
    component_ids = sorted(str(value) for value in rows[0].get("component_lambda", {}))
    if not component_ids:
        raise InputValidationError("exposure rows require component_lambda")
    candidate_ids: list[str] = []
    cells: list[str] = []
    weights: list[float] = []
    lambdas: list[list[float]] = []
    energies: list[list[float]] = []
    for row in rows:
        candidate_id = row.get("candidate_id")
        cell_id = row.get("cell_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise InputValidationError("each exposure candidate_id must be nonempty")
        if not isinstance(cell_id, str) or not cell_id:
            raise InputValidationError("each exposure cell_id must be nonempty")
        lambda_record = row.get("component_lambda")
        energy_record = row.get("component_damage_energy_j")
        if not isinstance(lambda_record, Mapping) or not isinstance(energy_record, Mapping):
            raise InputValidationError("exposure rows require lambda and energy mappings")
        if sorted(str(value) for value in lambda_record) != component_ids or sorted(
            str(value) for value in energy_record
        ) != component_ids:
            raise InputValidationError("exposure component identities must align")
        candidate_ids.append(candidate_id)
        cells.append(cell_id)
        weight = _nonnegative(row.get("measure_weight"), field="measure_weight")
        if weight <= 0.0:
            raise InputValidationError("measure_weight must be positive")
        weights.append(weight)
        lambdas.append(
            [
                _nonnegative(lambda_record[item], field=f"lambda.{item}")
                for item in component_ids
            ]
        )
        energies.append(
            [
                _nonnegative(energy_record[item], field=f"energy.{item}")
                for item in component_ids
            ]
        )
    if len(set(candidate_ids)) != len(candidate_ids):
        raise InputValidationError("exposure candidate IDs must be unique")
    return (
        component_ids,
        candidate_ids,
        np.asarray(weights, dtype=np.float64),
        np.asarray(lambdas, dtype=np.float64),
        np.asarray(energies, dtype=np.float64),
        cells,
    )


def build_exposure_field(
    rows: Sequence[Mapping[str, Any]],
    *,
    reach_tolerance: float = 0.0,
    include_candidate_rows: bool = True,
) -> dict[str, Any]:
    """Build component Lambda/E, reachability, cell envelopes and ESS."""

    if not math.isfinite(reach_tolerance) or reach_tolerance < 0.0:
        raise InputValidationError("reach_tolerance must be finite and non-negative")
    component_ids, candidate_ids, weights, lambdas, energies, cells = _validated_arrays(rows)
    mean_lambda = np.asarray(
        positive_weighted_mean(lambdas, weights, field="exposure measure weights")
    )
    mean_energy = np.asarray(
        positive_weighted_mean(energies, weights, field="exposure measure weights")
    )
    components: dict[str, Any] = {}
    for column, component_id in enumerate(component_ids):
        reached = lambdas[:, column] > reach_tolerance
        status = PROXY_REACHED if bool(np.any(reached)) else PROXY_UNREACHED
        components[component_id] = {
            "lambda_hajek": float(mean_lambda[column]),
            "damage_energy_hajek_j": float(mean_energy[column]),
            "reachability_status": status,
            "reached_candidate_count": int(np.sum(reached)),
            "reached_measure_fraction": float(
                positive_weighted_mean(
                    reached.astype(np.float64),
                    weights,
                    field="exposure measure weights",
                )
            ),
            "lambda_mass_ess": nonnegative_effective_sample_size(
                weights * lambdas[:, column]
            ),
            "structural_zero_claim": False,
            "interpretation": (
                "finite declared proxy frame only; teacher audit may falsify proxy non-reach"
            ),
        }

    cell_reports: dict[str, Any] = {}
    for cell_id in sorted(set(cells)):
        indices = np.asarray([value == cell_id for value in cells], dtype=bool)
        local_weights = weights[indices]
        local_lambda = lambdas[indices]
        local_energy = energies[indices]
        local_lambda_mean = np.asarray(
            positive_weighted_mean(local_lambda, local_weights, field="cell measure weights")
        )
        local_energy_mean = np.asarray(
            positive_weighted_mean(local_energy, local_weights, field="cell measure weights")
        )
        per_component: dict[str, Any] = {}
        for column, component_id in enumerate(component_ids):
            values = local_lambda[:, column]
            per_component[component_id] = {
                "lambda_hajek": float(local_lambda_mean[column]),
                "damage_energy_hajek_j": float(local_energy_mean[column]),
                "lambda_q50": _weighted_quantile(values, local_weights, 0.50),
                "lambda_q90": _weighted_quantile(values, local_weights, 0.90),
                "lambda_max": float(np.max(values)),
                "reachability_status": (
                    PROXY_REACHED
                    if bool(np.any(values > reach_tolerance))
                    else PROXY_UNREACHED
                ),
                "structural_zero_claim": False,
            }
        cell_reports[cell_id] = {
            "candidate_count": int(np.sum(indices)),
            "measure_ess": positive_effective_sample_size(local_weights),
            "components": per_component,
        }

    report: dict[str, Any] = {
        "schema_version": EXPOSURE_FIELD_SCHEMA_VERSION,
        "candidate_count": len(candidate_ids),
        "component_ids": component_ids,
        "cell_ids": sorted(set(cells)),
        "reach_tolerance": float(reach_tolerance),
        "estimand": "reference_measure_hajek_mean_of_fast_proxy_expected_exposure",
        "measure_weight_semantics": "positive_weight_relative_to_declared_reference_frame",
        "measure_ess": positive_effective_sample_size(weights),
        "measure_ess_fraction": positive_effective_sample_size(weights) / len(weights),
        "components": components,
        "cell_envelopes": cell_reports,
        "zero_semantics": {
            "status": PROXY_UNREACHED,
            "structural_zero_claim": False,
            "statement": (
                "proxy_unreached means no reach in the declared finite proxy frame; "
                "it is not a physical or structural zero"
            ),
        },
    }
    if include_candidate_rows:
        report["candidate_rows"] = [dict(row) for row in rows]
    return report


def exposure_footprint(
    reference_rows: Sequence[Mapping[str, Any]],
    selected_ids: Sequence[str],
    inclusion_probabilities: Mapping[str, float],
) -> dict[str, Any]:
    """Compare a design-weighted selected field with its reference frame."""

    component_ids, candidate_ids, reference_weights, lambdas, energies, _ = _validated_arrays(
        reference_rows
    )
    id_to_index = {value: index for index, value in enumerate(candidate_ids)}
    if not selected_ids or len(set(selected_ids)) != len(selected_ids):
        raise InputValidationError("selected_ids must be nonempty and unique")
    try:
        selected_indices = np.asarray([id_to_index[value] for value in selected_ids])
    except KeyError as exc:
        raise InputValidationError("selected candidate is outside the reference frame") from exc
    probabilities = np.asarray(
        [
            _nonnegative(
                inclusion_probabilities.get(value),
                field=f"inclusion_probability.{value}",
            )
            for value in selected_ids
        ],
        dtype=np.float64,
    )
    if np.any(probabilities <= 0.0) or np.any(probabilities > 1.0):
        raise InputValidationError("selected inclusion probabilities must lie in (0, 1]")
    design_weights = reference_weights[selected_indices] / probabilities
    reference_lambda = np.asarray(
        positive_weighted_mean(lambdas, reference_weights, field="reference weights")
    )
    selected_lambda = np.asarray(
        positive_weighted_mean(
            lambdas[selected_indices], design_weights, field="selected design weights"
        )
    )
    sampling_arm_lambda = np.mean(lambdas[selected_indices], axis=0)
    reference_energy = np.asarray(
        positive_weighted_mean(energies, reference_weights, field="reference weights")
    )
    selected_energy = np.asarray(
        positive_weighted_mean(
            energies[selected_indices], design_weights, field="selected design weights"
        )
    )
    sampling_arm_energy = np.mean(energies[selected_indices], axis=0)
    component_report: dict[str, Any] = {}
    absolute_lambda_error: list[float] = []
    for column, component_id in enumerate(component_ids):
        delta = float(selected_lambda[column] - reference_lambda[column])
        absolute_lambda_error.append(abs(delta))
        component_report[component_id] = {
            "reference_lambda_hajek": float(reference_lambda[column]),
            "selected_lambda_hajek": float(selected_lambda[column]),
            "lambda_delta": delta,
            "sampling_arm_lambda_mean": float(sampling_arm_lambda[column]),
            "sampling_footprint_lambda_ratio": (
                None
                if reference_lambda[column] <= 0.0
                else float(sampling_arm_lambda[column] / reference_lambda[column])
            ),
            "reference_damage_energy_hajek_j": float(reference_energy[column]),
            "selected_damage_energy_hajek_j": float(selected_energy[column]),
            "sampling_arm_damage_energy_mean_j": float(
                sampling_arm_energy[column]
            ),
            "reference_reachability_status": (
                PROXY_REACHED
                if bool(np.any(lambdas[:, column] > 0.0))
                else PROXY_UNREACHED
            ),
            "selected_reachability_status": (
                PROXY_REACHED
                if bool(np.any(lambdas[selected_indices, column] > 0.0))
                else PROXY_UNREACHED
            ),
            "structural_zero_claim": False,
        }
    scale = float(np.sum(reference_lambda))
    return {
        "schema_version": "exposure-sampling-footprint@1.0.0",
        "selected_count": len(selected_ids),
        "design_ess": positive_effective_sample_size(design_weights),
        "design_ess_fraction": positive_effective_sample_size(design_weights)
        / len(design_weights),
        "mean_absolute_lambda_error": float(np.mean(absolute_lambda_error)),
        "normalized_l1_hajek_recovery_error": (
            None
            if scale <= 0.0
            else float(np.sum(np.abs(selected_lambda - reference_lambda)) / scale)
        ),
        "normalized_l1_lambda_error": (
            None
            if scale <= 0.0
            else float(np.sum(np.abs(selected_lambda - reference_lambda)) / scale)
        ),
        "normalized_l1_sampling_footprint_shift": (
            None
            if scale <= 0.0
            else float(
                np.sum(np.abs(sampling_arm_lambda - reference_lambda)) / scale
            )
        ),
        "components": component_report,
    }


__all__ = [
    "EXPOSURE_FIELD_SCHEMA_VERSION",
    "PROXY_REACHED",
    "PROXY_UNREACHED",
    "build_exposure_field",
    "exposure_footprint",
    "exposure_row_from_proxy",
]
