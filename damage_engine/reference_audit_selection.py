"""Fixed-size Reference-audit sampling with positive proxy-partial support."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

from .errors import InputValidationError
from .normalized_sampling import pivotal_sample
from .utils import content_hash


REFERENCE_AUDIT_SELECTION_VERSION = "reference-audit-positive-support@1.0.0"


@dataclass(frozen=True)
class ReferenceAuditDesign:
    """First-order inclusion probabilities and fixed group quotas."""

    probabilities: np.ndarray
    group_indices: tuple[tuple[int, ...], ...]
    group_quotas: tuple[int, ...]
    group_keys: tuple[str, ...]
    metadata: dict[str, Any]


def _positive_integer(value: Any, *, field: str) -> int:
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
        or int(value) < 1
    ):
        raise InputValidationError(f"{field} must be a positive integer")
    return int(value)


def _probability(value: Any, *, field: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise InputValidationError(f"{field} must lie in (0, 1)")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"{field} must lie in (0, 1)") from exc
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise InputValidationError(f"{field} must lie in (0, 1)")
    return result


def _allocate_with_minima(
    capacities: dict[str, int],
    minima: dict[str, int],
    *,
    total: int,
) -> dict[str, int]:
    if set(capacities) != set(minima):
        raise RuntimeError("allocation keys are inconsistent")
    allocation = dict(minima)
    if any(
        allocation[key] < 0 or allocation[key] > capacities[key]
        for key in capacities
    ):
        raise InputValidationError("group positivity minimum exceeds capacity")
    if sum(allocation.values()) > total or total > sum(capacities.values()):
        raise InputValidationError(
            "sample budget cannot satisfy the declared group positivity minima"
        )
    while sum(allocation.values()) < total:
        active = [
            key for key in sorted(capacities) if allocation[key] < capacities[key]
        ]
        if not active:  # pragma: no cover - guarded by total/capacity check
            raise RuntimeError("group allocation exhausted all capacity")
        capacity_total = sum(capacities[key] for key in active)
        target_total = total - sum(
            allocation[key] for key in capacities if key not in active
        )
        winner = max(
            active,
            key=lambda key: (
                target_total * capacities[key] / capacity_total - allocation[key],
                capacities[key] - allocation[key],
                key,
            ),
        )
        allocation[winner] += 1
    return allocation


def reference_audit_inclusion_probabilities(
    strata: Sequence[str],
    proxy_domain_complete: Sequence[bool],
    *,
    sample_size: int,
    proxy_partial_budget_fraction: float,
    minimum_inclusion_probability: float,
) -> ReferenceAuditDesign:
    """Build a fixed-size, stratified design with positive support everywhere.

    The design first reserves a declared budget for proxy-partial candidates,
    then allocates integer quotas across ``stratum x proxy-domain`` groups.
    Equal probabilities within each group make the group-wise pivotal draw
    exactly preserve every quota and every first-order inclusion probability.
    """

    stratum_values = tuple(strata)
    domain_values = tuple(proxy_domain_complete)
    if len(stratum_values) != len(domain_values):
        raise InputValidationError(
            "strata and proxy_domain_complete must have the same length"
        )
    if not stratum_values:
        raise InputValidationError("reference-audit candidates cannot be empty")
    for value in stratum_values:
        if not isinstance(value, str) or not value.strip():
            raise InputValidationError("every stratum must be a nonempty string")
    if any(not isinstance(value, (bool, np.bool_)) for value in domain_values):
        raise InputValidationError("proxy_domain_complete must be boolean")
    if not any(bool(value) for value in domain_values):
        raise InputValidationError(
            "reference-audit design requires at least one complete-domain candidate"
        )

    count = len(stratum_values)
    budget = _positive_integer(sample_size, field="sample_size")
    if budget > count:
        raise InputValidationError("sample_size cannot exceed candidate count")
    partial_fraction = _probability(
        proxy_partial_budget_fraction,
        field="proxy_partial_budget_fraction",
    )
    rho_min = _probability(
        minimum_inclusion_probability,
        field="minimum_inclusion_probability",
    )

    groups: dict[str, list[int]] = {}
    for index, (stratum, complete) in enumerate(
        zip(stratum_values, domain_values, strict=True)
    ):
        domain = "complete" if bool(complete) else "proxy_partial"
        groups.setdefault(f"{stratum}|{domain}", []).append(index)

    capacities = {key: len(indices) for key, indices in groups.items()}
    minima = {
        key: max(1, int(math.ceil(rho_min * capacity)))
        for key, capacity in capacities.items()
    }
    if sum(minima.values()) > budget:
        raise InputValidationError(
            "sample budget cannot satisfy positivity in every stratum-domain group"
        )

    partial_keys = [key for key in sorted(groups) if key.endswith("|proxy_partial")]
    complete_keys = [key for key in sorted(groups) if key.endswith("|complete")]
    partial_capacity = sum(capacities[key] for key in partial_keys)
    complete_capacity = sum(capacities[key] for key in complete_keys)
    minimum_partial = sum(minima[key] for key in partial_keys)
    minimum_complete = sum(minima[key] for key in complete_keys)
    if partial_capacity:
        desired_partial = int(math.ceil(partial_fraction * budget))
        partial_budget = max(minimum_partial, desired_partial)
        partial_budget = min(
            partial_budget,
            partial_capacity,
            budget - minimum_complete,
        )
        complete_budget = budget - partial_budget
        if complete_budget > complete_capacity:
            transfer = complete_budget - complete_capacity
            partial_budget += transfer
            complete_budget -= transfer
    else:
        partial_budget = 0
        complete_budget = budget
    if (
        partial_budget < minimum_partial
        or complete_budget < minimum_complete
        or partial_budget > partial_capacity
        or complete_budget > complete_capacity
    ):
        raise InputValidationError(
            "sample budget cannot satisfy positivity and domain budget constraints"
        )

    quotas: dict[str, int] = {}
    if complete_keys:
        quotas.update(
            _allocate_with_minima(
                {key: capacities[key] for key in complete_keys},
                {key: minima[key] for key in complete_keys},
                total=complete_budget,
            )
        )
    if partial_keys:
        quotas.update(
            _allocate_with_minima(
                {key: capacities[key] for key in partial_keys},
                {key: minima[key] for key in partial_keys},
                total=partial_budget,
            )
        )

    probabilities = np.empty(count, dtype=np.float64)
    group_records: list[dict[str, Any]] = []
    ordered_keys = tuple(sorted(groups))
    ordered_indices: list[tuple[int, ...]] = []
    ordered_quotas: list[int] = []
    for key in ordered_keys:
        indices = tuple(groups[key])
        quota = quotas[key]
        probability = quota / len(indices)
        probabilities[list(indices)] = probability
        stratum, domain = key.rsplit("|", 1)
        ordered_indices.append(indices)
        ordered_quotas.append(quota)
        group_records.append(
            {
                "group": key,
                "stratum": stratum,
                "proxy_domain": domain,
                "candidate_count": len(indices),
                "fixed_sample_quota": quota,
                "inclusion_probability": probability,
            }
        )

    if (
        np.any(probabilities < rho_min - 1.0e-15)
        or np.any(probabilities > 1.0)
        or not math.isclose(
            float(np.sum(probabilities)), budget, rel_tol=0.0, abs_tol=1.0e-12
        )
    ):
        raise RuntimeError("constructed Reference-audit probabilities are invalid")

    document = {
        "schema_version": REFERENCE_AUDIT_SELECTION_VERSION,
        "candidate_count": count,
        "sample_size": budget,
        "proxy_partial_budget_fraction_declared": partial_fraction,
        "minimum_inclusion_probability": rho_min,
        "complete_candidate_count": int(
            sum(bool(value) for value in domain_values)
        ),
        "proxy_partial_candidate_count": int(
            sum(not bool(value) for value in domain_values)
        ),
        "complete_expected_sample_count": complete_budget,
        "proxy_partial_expected_sample_count": partial_budget,
        "proxy_partial_positive_inclusion_count": int(
            sum(
                not bool(value) and probabilities[index] > 0.0
                for index, value in enumerate(domain_values)
            )
        ),
        "group_allocations": group_records,
        "fixed_size_groupwise_pivotal": True,
        "uses_teacher_outcomes": False,
    }
    document["design_hash"] = content_hash(document)
    return ReferenceAuditDesign(
        probabilities=probabilities,
        group_indices=tuple(ordered_indices),
        group_quotas=tuple(ordered_quotas),
        group_keys=ordered_keys,
        metadata=document,
    )


def draw_reference_audit_sample(
    design: ReferenceAuditDesign, *, seed: int
) -> np.ndarray:
    """Replay one group-quota-preserving fixed-size pivotal sample."""

    if not isinstance(design, ReferenceAuditDesign):
        raise InputValidationError("design must be a ReferenceAuditDesign")
    if (
        not isinstance(seed, (int, np.integer))
        or isinstance(seed, (bool, np.bool_))
        or int(seed) < 0
    ):
        raise InputValidationError("seed must be a nonnegative integer")
    selected = np.zeros(design.probabilities.shape, dtype=bool)
    for key, indices, quota in zip(
        design.group_keys,
        design.group_indices,
        design.group_quotas,
        strict=True,
    ):
        derived = int(
            content_hash(
                {
                    "selection_version": REFERENCE_AUDIT_SELECTION_VERSION,
                    "design_hash": design.metadata["design_hash"],
                    "seed": int(seed),
                    "group": key,
                }
            ).split(":", 1)[1][:16],
            16,
        ) & ((1 << 63) - 1)
        group_probabilities = design.probabilities[list(indices)]
        group_selected = pivotal_sample(group_probabilities, seed=derived)
        if int(np.count_nonzero(group_selected)) != quota:
            raise RuntimeError("Reference-audit group quota was not preserved")
        selected[np.asarray(indices, dtype=np.int64)] = group_selected
    expected = int(design.metadata["sample_size"])
    if int(np.count_nonzero(selected)) != expected:
        raise RuntimeError("Reference-audit fixed sample size was not preserved")
    return selected
