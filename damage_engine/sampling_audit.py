"""Sampling-weight, coverage and split diagnostics."""

from __future__ import annotations

from collections import Counter
import math
from typing import Iterable, Mapping

import numpy as np

from .adaptive_sampling import effective_sample_size, exact_design_weights
from .estimands import (
    nonnegative_effective_sample_size,
    normalized_positive_weights,
    positive_weighted_mean,
)
from .errors import InputValidationError
from .sampling_validation import load_verified_split_registry, validate_candidate_set, validate_registry_assignments


def _uniform_ks(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    x = values[order]
    w = normalized_positive_weights(weights[order], field="coverage weights")
    cumulative = np.cumsum(w)
    previous = np.concatenate(([0.0], cumulative[:-1]))
    return float(max(np.max(np.abs(cumulative - x)), np.max(np.abs(previous - x))))


def audit_sampling(
    candidate_rows: Iterable[dict],
    latent_design: np.ndarray | None,
    *,
    quality_gates: dict,
    registry_document: dict | None = None,
    executed_candidate_ids: Iterable[str] | None = None,
    successful_candidate_ids: Iterable[str] | None = None,
    admitted_candidate_ids: Iterable[str] | None = None,
    refused_candidate_ids: Iterable[str] | None = None,
    soft_strata: Mapping[str, Iterable[float]] | None = None,
) -> dict:
    rows = list(candidate_rows)
    if not rows:
        raise InputValidationError("sampling audit requires nonempty candidates")
    latent: np.ndarray | None
    if latent_design is None:
        latent = None
    else:
        try:
            latent = np.asarray(latent_design, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise InputValidationError("latent design must be numeric") from error
        if latent.ndim != 2 or latent.shape[0] != len(rows) or latent.shape[1] == 0:
            raise InputValidationError("sampling audit requires aligned nonempty candidates and latent design")
        if not np.all(np.isfinite(latent)) or np.any(latent < 0) or np.any(latent > 1):
            raise InputValidationError("latent design must contain finite values in [0,1]")
    validate_candidate_set(rows)
    try:
        probabilities = np.asarray([row["inclusion_probability"] for row in rows], dtype=np.float64)
        weights = np.asarray([row["design_weight_raw"] for row in rows], dtype=np.float64)
        if any(type(row["selected"]) is not bool or type(row["weight_mask"]) is not bool for row in rows):
            raise InputValidationError("sampling selected and weight_mask flags must be booleans")
        if any(not isinstance(row["stream"], str) or not row["stream"] for row in rows):
            raise InputValidationError("sampling stream must be a nonempty string")
        selected = np.asarray([row["selected"] for row in rows], dtype=bool)
        weight_mask = np.asarray([row["weight_mask"] for row in rows], dtype=bool)
        thresholds = {key: float(quality_gates[key]) for key in ("positivity_min", "min_ess_fraction", "max_weight_cv")}
    except (KeyError, TypeError, ValueError) as error:
        raise InputValidationError("sampling audit fields or quality thresholds are incomplete or invalid") from error
    if (not all(math.isfinite(value) for value in thresholds.values())
            or not 0 < thresholds["positivity_min"] <= 1
            or not 0 <= thresholds["min_ess_fraction"] <= 1 or thresholds["max_weight_cv"] < 0):
        raise InputValidationError("sampling audit quality thresholds are outside their valid ranges")
    if (
        not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0.0)
        or np.any(probabilities > 1.0) or not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
    ):
        raise InputValidationError("candidate ledger contains invalid probabilities or design weights")
    density_fields = {
        "p0_density", "proposal_density", "p0_hierarchy_probability",
        "proposal_hierarchy_probability",
    }
    has_density_fields = [bool(density_fields.intersection(row)) for row in rows]
    if any(has_density_fields) and not all(has_density_fields):
        raise InputValidationError("candidate ledger mixes legacy and general proposal weight contracts")
    if all(has_density_fields):
        if any(not density_fields.issubset(row) for row in rows):
            raise InputValidationError("general proposal rows require complete p0/q/hierarchy probabilities")
        expected_weights = exact_design_weights(
            [row["p0_density"] for row in rows],
            [row["proposal_density"] for row in rows],
            probabilities,
            p0_hierarchy_probability=[row["p0_hierarchy_probability"] for row in rows],
            proposal_hierarchy_probability=[row["proposal_hierarchy_probability"] for row in rows],
        )
        weight_meaning = (
            "p0 conditional density times p0 hierarchy probability divided by "
            "proposal conditional density, proposal hierarchy probability and inclusion probability"
        )
        weight_formula = (
            "p0_density*p0_hierarchy_probability/"
            "(proposal_density*proposal_hierarchy_probability*inclusion_probability)"
        )
    else:
        expected_weights = 1.0 / probabilities
        weight_meaning = "inverse inclusion probability for candidates drawn directly from p0"
        weight_formula = "1/inclusion_probability"
    if not np.allclose(weights, expected_weights, rtol=1.0e-12, atol=1.0e-12):
        message = (
            "raw design weights do not equal the declared p0/proposal/hierarchy/inclusion formula"
            if all(has_density_fields)
            else "raw design weights do not equal inverse inclusion probabilities"
        )
        raise InputValidationError(message)

    round_indices = []
    for row in rows:
        value = row.get("round", 0)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise InputValidationError("sampling round indices must be nonnegative integers")
        round_indices.append(value)
        stream = row.get("stream")
        role = row.get("role")
        split = row.get("split", row.get("frozen_split"))
        adaptive = stream == "adaptive_train" or role == "adaptive_train"
        stress = (
            isinstance(stream, str) and "stress" in stream
        ) or (isinstance(role, str) and "stress" in role)
        if adaptive and not (
            stream == "adaptive_train" and role == "adaptive_train" and split == "train" and value >= 1
        ):
            raise InputValidationError("adaptive candidates must be isolated in Train rounds after round zero")
        if stress and split == "train":
            raise InputValidationError("stress candidates must remain isolated from Train")
    registry_status = "not_supplied"
    if registry_document is None and any("split_registry_hash" in row or "split_registry_id" in row for row in rows):
        raise InputValidationError("declared split registry identities require the complete frozen registry snapshot")
    if registry_document is not None:
        registry = load_verified_split_registry(registry_document)
        validate_registry_assignments(registry, rows)
        for row in rows:
            if (row.get("split_registry_hash", registry.definition_hash) != registry.definition_hash
                    or row.get("split_registry_id", registry.registry_id) != registry.registry_id):
                raise InputValidationError("candidate ledger references a different split registry")
        registry_status = "assignments_verified"
    active = selected & weight_mask
    selected_weights = weights[active]
    if selected_weights.size:
        scaled = selected_weights / float(np.max(selected_weights))
        ess = effective_sample_size(scaled)
        mean_weight = float(np.mean(scaled) * float(np.max(selected_weights)))
        weight_cv = float(np.std(scaled, ddof=0) / np.mean(scaled))
    else:
        ess, mean_weight, weight_cv = 0.0, None, None

    normalized_weights = (
        normalized_positive_weights(selected_weights, field="selected design weights")
        if selected_weights.size
        else np.asarray([])
    )

    def soft_ess(membership: np.ndarray) -> float:
        contributions = selected_weights * membership[active]
        return nonnegative_effective_sample_size(contributions)

    soft_memberships: dict[str, np.ndarray] = {}
    if soft_strata is not None:
        if not isinstance(soft_strata, Mapping) or not soft_strata:
            raise InputValidationError("soft_strata must be a nonempty name-to-membership mapping")
        for name, values in soft_strata.items():
            if not isinstance(name, str) or not name:
                raise InputValidationError("soft stratum names must be nonempty strings")
            try:
                membership = np.asarray(list(values), dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise InputValidationError("soft stratum memberships must be numeric") from exc
            if (
                membership.shape != (len(rows),)
                or not np.all(np.isfinite(membership))
                or np.any(membership < 0.0)
                or np.any(membership > 1.0)
            ):
                raise InputValidationError("soft stratum memberships must align and lie in [0,1]")
            soft_memberships[name] = membership

    def describe_set(mask: np.ndarray) -> dict:
        members = [row for row, included in zip(rows, mask) if included]
        return {"status": "evaluated", "count": len(members),
                "candidate_ids": [row["candidate_id"] for row in members],
                **{f"{key}_counts": dict(sorted(Counter(row.get(key, row.get("frozen_split")) for row in members).items()))
                   for key in ("split", "role", "stream")}}

    def coverage(mask: np.ndarray, design_weights: np.ndarray) -> dict:
        if latent is None:
            return {
                "status": "not_evaluated_without_prelabel_latent_design",
                "count": int(np.count_nonzero(mask)),
                "ks_by_axis": None,
                "occupancy_fraction_by_axis": None,
                "histogram_bins": None,
            }
        count = int(np.count_nonzero(mask))
        if not count:
            return {"status": "not_estimable", "count": 0, "ks_by_axis": [None] * latent.shape[1],
                    "occupancy_fraction_by_axis": [None] * latent.shape[1], "histogram_bins": None}
        bins = max(2, min(10, int(round(math.sqrt(count)))))
        values = latent[mask]
        return {"status": "descriptive_only", "count": count,
                "ks_by_axis": [_uniform_ks(values[:, axis], design_weights) for axis in range(latent.shape[1])],
                "occupancy_fraction_by_axis": [float(np.count_nonzero(np.histogram(values[:, axis], bins=bins, range=(0, 1))[0]) / bins)
                                                for axis in range(latent.shape[1])],
                "histogram_bins": bins}

    sets = {"candidate_frame": describe_set(np.ones(len(rows), dtype=bool)),
            "selected": describe_set(selected), "selected_weight_eligible": describe_set(active)}
    preceding = set(sets["selected"]["candidate_ids"])
    audited_identity_sets: dict[str, set[str] | None] = {}
    for name, identities in (("executed", executed_candidate_ids), ("successful", successful_candidate_ids),
                             ("admitted", admitted_candidate_ids)):
        if identities is None:
            sets[name] = {"status": "not_evaluated", "count": None, "candidate_ids": None}
            audited_identity_sets[name] = None
            preceding = None
            continue
        identities = list(identities)
        if (any(not isinstance(value, str) or not value for value in identities)
                or len(set(identities)) != len(identities) or preceding is None
                or not set(identities).issubset(preceding)):
            raise InputValidationError("executed/successful/admitted identities must be unique nested audited subsets")
        identity_set = set(identities)
        sets[name] = describe_set(np.asarray([row["candidate_id"] in identity_set for row in rows]))
        audited_identity_sets[name] = identity_set
        preceding = identity_set
    if refused_candidate_ids is None:
        sets["refused"] = {"status": "not_evaluated", "count": None, "candidate_ids": None}
    else:
        refused = list(refused_candidate_ids)
        executed_set = audited_identity_sets.get("executed")
        successful_set = audited_identity_sets.get("successful") or set()
        if (
            any(not isinstance(value, str) or not value for value in refused)
            or len(set(refused)) != len(refused)
            or executed_set is None
            or not set(refused).issubset(executed_set)
            or set(refused).intersection(successful_set)
        ):
            raise InputValidationError("refused identities must be a unique executed subset disjoint from successful candidates")
        refusal_mask = np.asarray([row["candidate_id"] in set(refused) for row in rows])
        sets["refused"] = describe_set(refusal_mask)
        sets["refused"]["refusal_code_counts"] = dict(sorted(Counter(
            ((row.get("refusal") or {}).get("code") or "unspecified")
            for row, included in zip(rows, refusal_mask) if included
        ).items()))

    hard_stratum_ess = {}
    stratum_names = sorted({str(row.get("stratum", "__unspecified__")) for row in rows})
    for name in stratum_names:
        mask = np.asarray([str(row.get("stratum", "__unspecified__")) == name for row in rows]) & active
        hard_stratum_ess[name] = {
            "selected_weight_eligible_count": int(np.count_nonzero(mask)),
            "effective_sample_size": effective_sample_size(weights[mask]) if np.any(mask) else 0.0,
        }
    soft_stratum_ess = {
        name: {
            "membership_sum_selected_weight_eligible": float(np.sum(values[active])),
            "soft_effective_sample_size": soft_ess(values),
        }
        for name, values in sorted(soft_memberships.items())
    }

    round_array = np.asarray(round_indices, dtype=np.int64)
    round_summaries = []
    previous_candidate_mean = None
    first_candidate_mean = None
    for round_index in sorted(set(round_indices)):
        round_mask = round_array == round_index
        round_active = round_mask & active
        summary = {
            "round": int(round_index),
            "candidate_count": int(np.count_nonzero(round_mask)),
            "selected_weight_eligible_count": int(np.count_nonzero(round_active)),
        }
        if latent is None:
            summary.update({
                "candidate_mean_by_axis": None,
                "selected_weighted_mean_by_axis": None,
                "candidate_mean_l2_from_first_round": None,
                "candidate_mean_l2_from_previous_round": None,
            })
        else:
            candidate_mean = np.mean(latent[round_mask], axis=0)
            selected_mean = None
            if np.any(round_active):
                round_weights = weights[round_active]
                selected_mean = positive_weighted_mean(
                    latent[round_active],
                    round_weights,
                    field="round design weights",
                )
            if first_candidate_mean is None:
                first_candidate_mean = candidate_mean
            summary.update({
                "candidate_mean_by_axis": candidate_mean.tolist(),
                "selected_weighted_mean_by_axis": None if selected_mean is None else selected_mean.tolist(),
                "candidate_mean_l2_from_first_round": float(np.linalg.norm(candidate_mean - first_candidate_mean)),
                "candidate_mean_l2_from_previous_round": (
                    None if previous_candidate_mean is None
                    else float(np.linalg.norm(candidate_mean - previous_candidate_mean))
                ),
            })
            previous_candidate_mean = candidate_mean
        round_summaries.append(summary)
    gates = {
        "candidate_set_integrity": True, "design_weight_numerics": True, "declared_group_isolation": True,
        "stream_role_split_isolation": True,
        "positivity": float(np.min(probabilities)) >= thresholds["positivity_min"],
        "ess_fraction": bool(selected_weights.size and ess / selected_weights.size >= thresholds["min_ess_fraction"]),
        "weight_cv": bool(weight_cv is not None and weight_cv <= thresholds["max_weight_cv"]),
    }
    return {
        "schema_version": "2.0.0",
        "candidate_count": len(rows),
        "selected_count": int(np.count_nonzero(selected)),
        "weighted_selected_count": int(np.count_nonzero(active)),
        "sets": sets,
        "positivity_min": float(np.min(probabilities)),
        "positivity_max": float(np.max(probabilities)),
        "design_weight": {
            "set": "selected_weight_eligible", "weight_meaning": weight_meaning,
            "formula": weight_formula,
            "minimum": float(np.min(selected_weights)) if selected_weights.size else None,
            "maximum": float(np.max(selected_weights)) if selected_weights.size else None,
            "maximum_normalized": float(np.max(normalized_weights)) if normalized_weights.size else None,
            "mean": mean_weight,
            "cv": weight_cv,
            "p90": float(np.quantile(selected_weights, 0.9)) if selected_weights.size else None,
            "p99": float(np.quantile(selected_weights, 0.99)) if selected_weights.size else None,
            "normalized_p90": float(np.quantile(normalized_weights, 0.9)) if normalized_weights.size else None,
            "normalized_p99": float(np.quantile(normalized_weights, 0.99)) if normalized_weights.size else None,
            "quantiles": {
                str(probability): float(value)
                for probability, value in zip(
                    (0.5, 0.9, 0.99), np.quantile(selected_weights, (0.5, 0.9, 0.99))
                )
            } if selected_weights.size else {},
        },
        "effective_sample_size": ess,
        "ess_fraction": ess / selected_weights.size if selected_weights.size else 0.0,
        "hard_stratum_effective_sample_size": hard_stratum_ess,
        "soft_stratum_effective_sample_size": soft_stratum_ess,
        "soft_stratum_ess_formula": "(sum(weight*membership))^2/sum((weight*membership)^2)",
        "soft_stratum_status": (
            "externally_supplied_memberships_descriptive_only"
            if soft_memberships else "not_supplied"
        ),
        "cross_round_drift": {
            "status": (
                "descriptive_only_no_domain_threshold"
                if latent is not None
                else "not_evaluated_without_prelabel_latent_design"
            ),
            "rounds": round_summaries,
        },
        "candidate_frame_unweighted": {"set": "candidate_frame", "weight_meaning": "one per candidate",
                                         **coverage(np.ones(len(rows), dtype=bool), np.ones(len(rows)))},
        "selected_weighted": {"set": "selected_weight_eligible", "weight_meaning": weight_meaning,
                                **coverage(active, selected_weights)},
        "registry_assignment_status": registry_status,
        "coverage_status": "descriptive_only_no_predeclared_coverage_protocol",
        "coverage_guaranteed": False,
        "quality_gates": gates,
        "quality_gate_passed": all(gates.values()),
        "quality_gate_scope": "input set integrity, declared group isolation and eligible selected-weight diagnostics; excludes coverage and scientific admission",
        "evidence_boundary": (
            "These diagnostics validate the sampling design and weights only; "
            "they do not validate damage physics or real-world label accuracy."
        ),
    }
