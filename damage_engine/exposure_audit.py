"""Teacher audits for finite-frame proxy exposure reachability."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from statistics import NormalDist
from typing import Any

import numpy as np

from .errors import InputValidationError
from .exposure_field import PROXY_REACHED, PROXY_UNREACHED


EXPOSURE_TEACHER_AUDIT_VERSION = "exposure-teacher-audit@1.0.0"


def _wilson_upper(successes: int, trials: int, confidence: float) -> float | None:
    if trials == 0:
        return None
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = proportion + z * z / (2.0 * trials)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    )
    return float(min(1.0, (centre + radius) / denominator))


def teacher_component_hit_probabilities(
    engine_result: Mapping[str, Any] | Any,
) -> dict[str, float]:
    """Extract component hit probabilities from one formal teacher result."""

    document = (
        engine_result.to_dict()
        if hasattr(engine_result, "to_dict")
        else engine_result
    )
    if not isinstance(document, Mapping):
        raise InputValidationError("teacher result must be a mapping or EngineResult")
    components = document.get("components")
    if not isinstance(components, Mapping) or not components:
        raise InputValidationError("teacher result requires nonempty components")
    output: dict[str, float] = {}
    for component_id, record in components.items():
        if not isinstance(record, Mapping):
            raise InputValidationError("teacher component record must be an object")
        fragment = record.get("fragment_metrics")
        if not isinstance(fragment, Mapping):
            raise InputValidationError("teacher component lacks fragment_metrics")
        hit = fragment.get("hit_indicator")
        if not isinstance(hit, Mapping):
            raise InputValidationError("teacher component lacks hit_indicator")
        value = float(hit.get("mean"))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise InputValidationError("teacher hit probability must lie in [0, 1]")
        output[str(component_id)] = value
    return output


def select_teacher_audit_ids(
    exposure_rows: Sequence[Mapping[str, Any]],
    *,
    sample_size: int,
    seed: int,
) -> list[str]:
    """Select a cell/status-stratified proxy audit subset without outcomes."""

    if not exposure_rows:
        raise InputValidationError("teacher audit selection requires exposure rows")
    if not isinstance(sample_size, int) or not 0 < sample_size <= len(exposure_rows):
        raise InputValidationError("audit sample_size must lie in (0, candidate_count]")
    if not isinstance(seed, int) or seed < 0:
        raise InputValidationError("audit seed must be non-negative")
    strata: dict[tuple[str, str], list[str]] = {}
    for row in exposure_rows:
        identifier = row.get("candidate_id")
        cell = row.get("cell_id")
        lambdas = row.get("component_lambda")
        if not isinstance(identifier, str) or not isinstance(cell, str) or not isinstance(lambdas, Mapping):
            raise InputValidationError("audit exposure rows are malformed")
        values = np.asarray([float(value) for value in lambdas.values()])
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise InputValidationError("audit proxy lambdas must be finite and non-negative")
        status = "has_proxy_unreached" if bool(np.any(values == 0.0)) else "all_proxy_reached"
        strata.setdefault((cell, status), []).append(identifier)
    rng = np.random.default_rng(seed)
    queues = []
    for key in sorted(strata):
        values = list(strata[key])
        rng.shuffle(values)
        queues.append(values)
    selected: list[str] = []
    while len(selected) < sample_size:
        progressed = False
        for queue in queues:
            if queue and len(selected) < sample_size:
                selected.append(queue.pop())
                progressed = True
        if not progressed:
            break
    if len(selected) != sample_size:
        raise RuntimeError("teacher audit stratification could not fill its sample")
    return selected


def build_exposure_teacher_audit(
    exposure_rows: Sequence[Mapping[str, Any]],
    teacher_rows: Sequence[Mapping[str, Any]],
    *,
    teacher_positive_threshold: float = 0.0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Measure how often teacher hits falsify proxy non-reach."""

    if not 0.0 <= teacher_positive_threshold < 1.0:
        raise InputValidationError("teacher_positive_threshold must lie in [0, 1)")
    if not 0.0 < confidence < 1.0:
        raise InputValidationError("confidence must lie in (0, 1)")
    exposure_by_id = {str(row.get("candidate_id")): row for row in exposure_rows}
    teacher_by_id = {str(row.get("candidate_id")): row for row in teacher_rows}
    if len(exposure_by_id) != len(exposure_rows) or len(teacher_by_id) != len(teacher_rows):
        raise InputValidationError("teacher audit candidate IDs must be unique")
    if not teacher_by_id or not set(teacher_by_id).issubset(exposure_by_id):
        raise InputValidationError("teacher rows must be a nonempty exposure subset")
    component_ids = sorted(
        str(value) for value in next(iter(exposure_by_id.values())).get("component_lambda", {})
    )
    component_counts = {
        value: {
            "proxy_unreached_teacher_audited": 0,
            "proxy_unreached_teacher_positive": 0,
            "proxy_reached_teacher_positive": 0,
            "proxy_reached_teacher_audited": 0,
        }
        for value in component_ids
    }
    pair_count = 0
    false_unreached = 0
    for candidate_id, teacher_row in teacher_by_id.items():
        proxy = exposure_by_id[candidate_id].get("component_lambda")
        teacher = teacher_row.get("component_hit_probability")
        if not isinstance(proxy, Mapping) or not isinstance(teacher, Mapping):
            raise InputValidationError("teacher audit rows require aligned component mappings")
        if sorted(str(value) for value in proxy) != component_ids or sorted(
            str(value) for value in teacher
        ) != component_ids:
            raise InputValidationError("teacher and proxy component identities must align")
        for component_id in component_ids:
            proxy_value = float(proxy[component_id])
            teacher_value = float(teacher[component_id])
            if (
                not math.isfinite(proxy_value)
                or proxy_value < 0.0
                or not math.isfinite(teacher_value)
                or not 0.0 <= teacher_value <= 1.0
            ):
                raise InputValidationError("teacher audit values are outside their domains")
            teacher_positive = teacher_value > teacher_positive_threshold
            record = component_counts[component_id]
            if proxy_value == 0.0:
                pair_count += 1
                record["proxy_unreached_teacher_audited"] += 1
                if teacher_positive:
                    false_unreached += 1
                    record["proxy_unreached_teacher_positive"] += 1
            else:
                record["proxy_reached_teacher_audited"] += 1
                if teacher_positive:
                    record["proxy_reached_teacher_positive"] += 1
    per_component: dict[str, Any] = {}
    for component_id, counts in component_counts.items():
        trials = counts["proxy_unreached_teacher_audited"]
        successes = counts["proxy_unreached_teacher_positive"]
        per_component[component_id] = {
            **counts,
            "false_unreached_rate": None if trials == 0 else successes / trials,
            "false_unreached_wilson_upper": _wilson_upper(successes, trials, confidence),
            "proxy_zero_semantics": PROXY_UNREACHED,
            "structural_zero_claim": False,
        }
    return {
        "schema_version": EXPOSURE_TEACHER_AUDIT_VERSION,
        "audited_candidate_count": len(teacher_rows),
        "proxy_unreached_pair_count": pair_count,
        "teacher_positive_among_proxy_unreached": false_unreached,
        "false_unreached_rate": None if pair_count == 0 else false_unreached / pair_count,
        "false_unreached_wilson_upper": _wilson_upper(
            false_unreached, pair_count, confidence
        ),
        "confidence": confidence,
        "teacher_positive_threshold": teacher_positive_threshold,
        "components": per_component,
        "interpretation": {
            "proxy_positive": PROXY_REACHED,
            "proxy_zero": PROXY_UNREACHED,
            "teacher_positive_at_proxy_zero": "proxy_non_reach_falsified_for_this_case",
            "structural_zero_claim": False,
        },
    }


__all__ = [
    "EXPOSURE_TEACHER_AUDIT_VERSION",
    "build_exposure_teacher_audit",
    "select_teacher_audit_ids",
    "teacher_component_hit_probabilities",
]
