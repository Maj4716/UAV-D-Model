"""Fixed-size exposure-balanced Train enrichment selection.

The selector consumes only pre-outcome proxy exposures.  It never changes the
Reference estimand and declares that boundary in every selection ledger.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np

from .errors import InputValidationError
from .exposure_field import exposure_footprint
from .normalized_sampling import (
    global_marginal_coverage_probabilities,
    pivotal_sample,
)


EXPOSURE_BALANCED_SELECTION_VERSION = "exposure-balanced-train-arm@1.0.0"


def _largest_remainder_quotas(sizes: np.ndarray, sample_size: int) -> np.ndarray:
    if sample_size < sizes.size:
        raise InputValidationError(
            "sample_size must cover every declared exposure cell to preserve positivity"
        )
    quotas = np.ones(sizes.size, dtype=np.int64)
    remaining = int(sample_size - sizes.size)
    capacity = sizes - 1
    if remaining == 0:
        return quotas
    if int(np.sum(capacity)) < remaining:
        raise InputValidationError("sample_size exceeds total exposure-cell capacity")
    raw = remaining * capacity / float(np.sum(capacity))
    additions = np.floor(raw).astype(np.int64)
    quotas += additions
    remainder = int(remaining - np.sum(additions))
    order = np.lexsort((np.arange(sizes.size), -(raw - additions)))
    for index in order:
        if remainder == 0:
            break
        if quotas[index] < sizes[index]:
            quotas[index] += 1
            remainder -= 1
    if remainder != 0:
        raise RuntimeError("unable to allocate exposure-balanced cell quotas")
    return quotas


def select_exposure_balanced_arm(
    rows: Sequence[Mapping[str, Any]],
    *,
    sample_size: int,
    seed: int,
    rho_min: float = 0.08,
    smoothing: float = 2.0,
    iterations: int = 80,
) -> dict[str, Any]:
    """Select one positive-probability, cell-stratified Train enrichment arm."""

    if not rows:
        raise InputValidationError("exposure-balanced selection requires rows")
    if not isinstance(sample_size, int) or not 0 < sample_size <= len(rows):
        raise InputValidationError("sample_size must lie in (0, candidate_count]")
    if not isinstance(seed, int) or seed < 0:
        raise InputValidationError("seed must be a non-negative integer")
    if not math.isfinite(rho_min) or not 0.0 < rho_min <= 1.0:
        raise InputValidationError("rho_min must lie in (0, 1]")

    component_ids = sorted(str(value) for value in rows[0].get("component_lambda", {}))
    if not component_ids:
        raise InputValidationError("exposure rows require component_lambda")
    candidate_ids: list[str] = []
    cell_ids: list[str] = []
    matrix: list[list[float]] = []
    energy_total: list[float] = []
    for row in rows:
        identifier = row.get("candidate_id")
        cell = row.get("cell_id")
        record = row.get("component_lambda")
        energy = row.get("component_damage_energy_j")
        if not isinstance(identifier, str) or not identifier:
            raise InputValidationError("candidate_id must be nonempty")
        if not isinstance(cell, str) or not cell:
            raise InputValidationError("cell_id must be nonempty")
        if not isinstance(record, Mapping) or not isinstance(energy, Mapping):
            raise InputValidationError("exposure rows require lambda and energy mappings")
        if sorted(str(value) for value in record) != component_ids:
            raise InputValidationError("component identities must align")
        values = np.asarray([float(record[value]) for value in component_ids])
        energy_values = np.asarray([float(energy[value]) for value in component_ids])
        if (
            not np.isfinite(values).all()
            or np.any(values < 0.0)
            or not np.isfinite(energy_values).all()
            or np.any(energy_values < 0.0)
        ):
            raise InputValidationError("proxy exposure values must be finite and non-negative")
        candidate_ids.append(identifier)
        cell_ids.append(cell)
        matrix.append((-np.expm1(-values)).tolist())
        energy_total.append(float(np.sum(np.log1p(energy_values))))
    if len(set(candidate_ids)) != len(candidate_ids):
        raise InputValidationError("candidate IDs must be unique")

    support = np.asarray(matrix, dtype=np.float64)
    active = np.any(support > 0.0, axis=0)
    active_ids = [value for value, keep in zip(component_ids, active, strict=True) if keep]
    excluded_ids = [value for value, keep in zip(component_ids, active, strict=True) if not keep]
    if not active_ids:
        raise InputValidationError(
            "declared proxy frame has no reached component; exposure balancing is undefined"
        )
    active_support = support[:, active]
    unique_cells = sorted(set(cell_ids))
    strata = [
        np.asarray([index for index, value in enumerate(cell_ids) if value == cell])
        for cell in unique_cells
    ]
    sizes = np.asarray([len(value) for value in strata], dtype=np.int64)
    quotas = _largest_remainder_quotas(sizes, sample_size)
    probabilities = global_marginal_coverage_probabilities(
        active_support,
        strata=strata,
        quotas=quotas.tolist(),
        auxiliary_features=np.asarray(energy_total, dtype=np.float64)[:, None],
        auxiliary_weights=np.asarray([0.15], dtype=np.float64),
        smoothing=float(smoothing),
        iterations=int(iterations),
        rho_min=float(rho_min),
    )
    selected = np.zeros(len(rows), dtype=bool)
    for stratum_index, indices in enumerate(strata):
        if quotas[stratum_index] == 0:
            continue
        local = pivotal_sample(
            probabilities[indices], seed=int(seed + 104729 * (stratum_index + 1))
        )
        selected[indices[local]] = True
    if int(np.sum(selected)) != sample_size:
        raise RuntimeError("exposure-balanced pivotal sample has the wrong size")

    selected_ids = [
        candidate_ids[index] for index in np.flatnonzero(selected)
    ]
    probability_by_id = {
        identifier: float(probabilities[index])
        for index, identifier in enumerate(candidate_ids)
    }
    ledger = [
        {
            "candidate_id": identifier,
            "cell_id": cell_ids[index],
            "inclusion_probability": float(probabilities[index]),
            "selected": bool(selected[index]),
            "design_weight": (
                float(1.0 / probabilities[index]) if selected[index] else None
            ),
        }
        for index, identifier in enumerate(candidate_ids)
    ]
    footprint = exposure_footprint(rows, selected_ids, probability_by_id)
    return {
        "schema_version": EXPOSURE_BALANCED_SELECTION_VERSION,
        "selection_role": "train_enrichment_exposure_balanced",
        "train_eligible": True,
        "reference_replacement": False,
        "outcome_blind": True,
        "candidate_count": len(rows),
        "selected_count": sample_size,
        "seed": seed,
        "rho_min_requested": float(rho_min),
        "positivity_min_observed": float(np.min(probabilities)),
        "component_ids": component_ids,
        "active_component_ids": active_ids,
        "proxy_unreached_component_ids_excluded_from_optimization": excluded_ids,
        "cell_quotas": {
            cell: int(quota)
            for cell, quota in zip(unique_cells, quotas, strict=True)
        },
        "candidate_ledger": ledger,
        "selected_candidate_ids": selected_ids,
        "sampling_footprint": footprint,
        "boundary": (
            "Proxy exposure shapes a Train-only enrichment arm. Reference, "
            "Validation and Test estimands remain unchanged."
        ),
    }


__all__ = [
    "EXPOSURE_BALANCED_SELECTION_VERSION",
    "select_exposure_balanced_arm",
]
