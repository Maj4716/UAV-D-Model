"""Auditable candidate-pool primitives for future adaptive train rounds.

This module intentionally contains no damage labels or engine calls.  Scores
are converted to strictly positive Bernoulli inclusion probabilities and are
then frozen before :func:`select_bernoulli` is called.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Iterable, Mapping, Sequence

import numpy as np

from .content_contract import require_sha256_identity
from .errors import InputValidationError
from .estimands import positive_effective_sample_size, positive_weighted_mean
from .input_identity import case_identity_hash
from .learning_input_identity import learning_input_fingerprint
from .utils import content_hash


ADAPTIVE_PROPOSAL_SCHEMA_VERSION = "adaptive-proposal@1.1.0"
_SELECTION_ALGORITHM = "numpy.default_rng-pcg64-random-v1"


def _finite_vector(values, *, name: str, positive: bool = False) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"{name} must be numeric") from exc
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise InputValidationError(f"{name} must be a finite nonempty vector")
    if positive and np.any(result <= 0.0):
        raise InputValidationError(f"{name} must be strictly positive")
    return result


def _aligned_vectors(*values: tuple[str, np.ndarray]) -> None:
    lengths = {array.size for _, array in values}
    if len(lengths) != 1:
        raise InputValidationError(
            "proposal probability vectors must be aligned",
            context={name: int(array.size) for name, array in values},
        )


def defensive_mixture_density(
    p0_density: np.ndarray | list[float],
    adaptive_density: np.ndarray | list[float],
    *,
    reference_fraction: float,
) -> np.ndarray:
    """Return ``q = alpha*p0 + (1-alpha)*q_adaptive`` candidate densities.

    The caller supplies densities with respect to the same base measure.  This
    routine deliberately does not infer, fit or score a domain proposal.
    ``reference_fraction`` must remain positive so an adaptive proposal cannot
    silently remove support that exists under ``p0``.
    """

    p0 = _finite_vector(p0_density, name="p0_density", positive=True)
    adaptive = _finite_vector(adaptive_density, name="adaptive_density", positive=True)
    _aligned_vectors(("p0_density", p0), ("adaptive_density", adaptive))
    if not isinstance(reference_fraction, (int, float)) or isinstance(reference_fraction, bool):
        raise InputValidationError("reference_fraction must be numeric")
    alpha = float(reference_fraction)
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise InputValidationError("reference_fraction must lie in (0, 1]")
    mixture = alpha * p0 + (1.0 - alpha) * adaptive
    if not np.all(np.isfinite(mixture)) or np.any(mixture <= 0.0):
        raise InputValidationError("defensive mixture produced invalid proposal densities")
    return mixture


def exact_design_weights(
    p0_density: np.ndarray | list[float],
    proposal_density: np.ndarray | list[float],
    inclusion_probability: np.ndarray | list[float],
    *,
    p0_hierarchy_probability: np.ndarray | list[float] | None = None,
    proposal_hierarchy_probability: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    """Compute the exact raw design weight for a two-stage proposal.

    The implemented estimand is the reference joint design.  For candidate
    ``i`` the weight is

    ``p0(x_i|g_i) p0(g_i) / (q_t(x_i|g_i) q_t(g_i) rho_i)``.

    Keeping the conditional and hierarchy terms separate prevents a target or
    source-family sampling probability from being accidentally omitted.  When
    the candidate proposal is exactly ``p0`` at both levels this reduces to
    the legacy ``1/rho`` formula.
    """

    p0 = _finite_vector(p0_density, name="p0_density", positive=True)
    proposal = _finite_vector(proposal_density, name="proposal_density", positive=True)
    rho = _finite_vector(inclusion_probability, name="inclusion_probability", positive=True)
    if np.any(rho > 1.0):
        raise InputValidationError("inclusion_probability must lie in (0, 1]")
    p0_hierarchy = (
        np.ones_like(p0)
        if p0_hierarchy_probability is None
        else _finite_vector(p0_hierarchy_probability, name="p0_hierarchy_probability", positive=True)
    )
    proposal_hierarchy = (
        np.ones_like(p0)
        if proposal_hierarchy_probability is None
        else _finite_vector(
            proposal_hierarchy_probability,
            name="proposal_hierarchy_probability",
            positive=True,
        )
    )
    _aligned_vectors(
        ("p0_density", p0),
        ("proposal_density", proposal),
        ("inclusion_probability", rho),
        ("p0_hierarchy_probability", p0_hierarchy),
        ("proposal_hierarchy_probability", proposal_hierarchy),
    )
    if np.any(p0_hierarchy > 1.0) or np.any(proposal_hierarchy > 1.0):
        raise InputValidationError("hierarchy probabilities must lie in (0, 1]")
    weights = (p0 * p0_hierarchy) / (proposal * proposal_hierarchy * rho)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise InputValidationError("exact design-weight calculation produced invalid values")
    return weights


def validate_train_only_acquisition(history_rows: Iterable[Mapping]) -> dict:
    """Validate and identify the formal history used to fit acquisition.

    Every contributing record must be an observed, successful formal result
    frozen in Train.  Reference-Train and earlier adaptive-Train results are
    both allowed; stress, validation and test records are rejected even if a
    caller attempts to relabel their split.
    """

    rows = list(history_rows)
    if not rows:
        raise InputValidationError("adaptive acquisition requires nonempty formal Train history")
    canonical: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise InputValidationError("acquisition history rows must be objects")
        identifier = row.get("candidate_id")
        split = row.get("split", row.get("frozen_split"))
        role = row.get("role")
        stream = row.get("stream")
        result_hash = row.get("formal_result_hash")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise InputValidationError("acquisition history candidate identities must be unique")
        if split != "train":
            raise InputValidationError("adaptive acquisition may use Train formal history only")
        if not isinstance(role, str) or not role or "validation" in role or "test" in role or "stress" in role:
            raise InputValidationError("validation, test and stress roles are sealed from acquisition")
        if not isinstance(stream, str) or not stream or "stress" in stream:
            raise InputValidationError("stress streams are sealed from acquisition")
        if row.get("formal_status") != "ok" or row.get("label_observed") is not True:
            raise InputValidationError("acquisition history must contain observed successful formal Train results")
        require_sha256_identity(
            result_hash,
            field="formal_result_hash",
            acceptance="legacy_python_hex",
            failure_message="formal_result_hash must be a sha256 content identity",
        )
        canonical.append(
            {
                "candidate_id": identifier,
                "split": split,
                "role": role,
                "stream": stream,
                "formal_status": "ok",
                "label_observed": True,
                "formal_result_hash": result_hash,
            }
        )
        seen.add(identifier)
    return {
        "policy": "observed_successful_formal_train_only",
        "record_count": len(canonical),
        "candidate_ids": [row["candidate_id"] for row in canonical],
        "records": canonical,
        "history_hash": content_hash(canonical),
    }


def freeze_adaptive_proposal(
    candidate_ids: Sequence[str],
    *,
    candidate_cases: Sequence[Mapping] | None = None,
    round_index: int,
    round_lineage_id: str,
    parent_round_lineage_id: str,
    p0_id: str,
    p0_hash: str,
    adaptive_proposal_hash: str,
    acquisition_model_hash: str,
    acquisition_history: Iterable[Mapping],
    p0_density: np.ndarray | list[float],
    adaptive_density: np.ndarray | list[float],
    p0_hierarchy_probability: np.ndarray | list[float],
    proposal_hierarchy_probability: np.ndarray | list[float],
    rarity_scores: np.ndarray | list[float],
    uncertainty_scores: np.ndarray | list[float],
    coverage_scores: np.ndarray | list[float],
    selection_scores: np.ndarray | list[float],
    strata: Sequence[str] | None = None,
    expected_count: float,
    rho_min: float,
    reference_fraction: float,
    selection_seed: int,
) -> dict:
    """Freeze an auditable adaptive-Train candidate proposal.

    Rarity, uncertainty and coverage are stored independently.  The caller
    also supplies ``selection_scores``: this function neither combines those
    components nor claims that a combination is scientifically appropriate.
    """

    identifiers = list(candidate_ids)
    if (
        not identifiers
        or any(not isinstance(value, str) or not value for value in identifiers)
        or len(set(identifiers)) != len(identifiers)
    ):
        raise InputValidationError("adaptive proposal candidate IDs must be unique nonempty strings")
    cases: list[dict] = []
    if candidate_cases is not None:
        supplied_cases = list(candidate_cases)
        if len(supplied_cases) != len(identifiers):
            raise InputValidationError(
                "adaptive candidate cases must align one-to-one with candidate IDs",
                context={"candidate_ids": len(identifiers), "candidate_cases": len(supplied_cases)},
            )
        observed_case_ids: set[str] = set()
        observed_learning_inputs: set[str] = set()
        for index, raw_case in enumerate(supplied_cases):
            if not isinstance(raw_case, Mapping):
                raise InputValidationError(
                    "adaptive candidate cases must be JSON objects",
                    context={"candidate_index": index},
                )
            case = deepcopy(dict(raw_case))
            case_id = case.get("case_id")
            if not isinstance(case_id, str) or not case_id or case_id in observed_case_ids:
                raise InputValidationError(
                    "adaptive candidate cases require unique nonempty case_id values",
                    context={"candidate_id": identifiers[index]},
                )
            metadata = case.get("metadata")
            if metadata is not None and not isinstance(metadata, dict):
                raise InputValidationError("adaptive candidate case metadata must be an object")
            if isinstance(metadata, dict) and {
                "sampling", "frozen_split", "split_registry_hash"
            }.intersection(metadata):
                raise InputValidationError(
                    "adaptive candidate cases must not predeclare reserved sampling metadata",
                    context={"candidate_id": identifiers[index]},
                )
            analysis = case.get("analysis")
            formal_seed = analysis.get("aleatory_seed") if isinstance(analysis, dict) else None
            if (
                not isinstance(formal_seed, int)
                or isinstance(formal_seed, bool)
                or formal_seed < 0
            ):
                raise InputValidationError(
                    "adaptive candidate cases require a nonnegative analysis.aleatory_seed",
                    context={"candidate_id": identifiers[index]},
                )
            learning_hash = learning_input_fingerprint(case)
            if learning_hash in observed_learning_inputs:
                raise InputValidationError(
                    "adaptive candidate cases contain duplicate declared learning inputs",
                    context={"candidate_id": identifiers[index], "learning_input_hash": learning_hash},
                )
            observed_case_ids.add(case_id)
            observed_learning_inputs.add(learning_hash)
            cases.append(case)
    if not isinstance(round_index, int) or isinstance(round_index, bool) or round_index < 1:
        raise InputValidationError("adaptive proposal round_index must be a positive integer")
    if not isinstance(round_lineage_id, str) or not round_lineage_id:
        raise InputValidationError("adaptive proposal round_lineage_id must be nonempty")
    if not isinstance(parent_round_lineage_id, str) or not parent_round_lineage_id:
        raise InputValidationError("adaptive proposal parent_round_lineage_id must be nonempty")
    if not isinstance(p0_id, str) or not p0_id:
        raise InputValidationError("adaptive proposal p0_id must be nonempty")
    if not isinstance(selection_seed, int) or isinstance(selection_seed, bool) or selection_seed < 0:
        raise InputValidationError("adaptive proposal selection_seed must be a nonnegative integer")
    for name, value in (
        ("p0_hash", p0_hash),
        ("adaptive_proposal_hash", adaptive_proposal_hash),
        ("acquisition_model_hash", acquisition_model_hash),
    ):
        require_sha256_identity(
            value,
            field=name,
            acceptance="legacy_python_hex",
            failure_message=f"{name} must be a sha256 content identity",
        )
    history_identity = validate_train_only_acquisition(acquisition_history)
    p0 = _finite_vector(p0_density, name="p0_density", positive=True)
    adaptive = _finite_vector(adaptive_density, name="adaptive_density", positive=True)
    p0_hierarchy = _finite_vector(
        p0_hierarchy_probability, name="p0_hierarchy_probability", positive=True
    )
    proposal_hierarchy = _finite_vector(
        proposal_hierarchy_probability,
        name="proposal_hierarchy_probability",
        positive=True,
    )
    score_vectors = [
        ("rarity_scores", _finite_vector(rarity_scores, name="rarity_scores")),
        ("uncertainty_scores", _finite_vector(uncertainty_scores, name="uncertainty_scores")),
        ("coverage_scores", _finite_vector(coverage_scores, name="coverage_scores")),
        ("selection_scores", _finite_vector(selection_scores, name="selection_scores")),
    ]
    if any(np.any(array < 0.0) for _, array in score_vectors):
        raise InputValidationError("adaptive proposal scores must be nonnegative")
    _aligned_vectors(
        ("candidate_ids", np.empty(len(identifiers))),
        ("p0_density", p0),
        ("adaptive_density", adaptive),
        ("p0_hierarchy_probability", p0_hierarchy),
        ("proposal_hierarchy_probability", proposal_hierarchy),
        *score_vectors,
    )
    stratum_names = ["unspecified"] * len(identifiers) if strata is None else list(strata)
    if (
        len(stratum_names) != len(identifiers)
        or any(not isinstance(value, str) or not value for value in stratum_names)
    ):
        raise InputValidationError("adaptive proposal strata must align as nonempty strings")
    if np.any(p0_hierarchy > 1.0) or np.any(proposal_hierarchy > 1.0):
        raise InputValidationError("hierarchy probabilities must lie in (0, 1]")
    proposal = defensive_mixture_density(
        p0, adaptive, reference_fraction=reference_fraction
    )
    selection_score = score_vectors[-1][1]
    rho = inclusion_probabilities(
        selection_score, expected_count=expected_count, rho_min=rho_min
    )
    selected, draws = select_bernoulli(rho, seed=selection_seed)
    weights = exact_design_weights(
        p0,
        proposal,
        rho,
        p0_hierarchy_probability=p0_hierarchy,
        proposal_hierarchy_probability=proposal_hierarchy,
    )
    random_core = {
        "algorithm": _SELECTION_ALGORITHM,
        "seed": int(selection_seed),
        "candidate_order_hash": content_hash(identifiers),
        "inclusion_probability_hash": content_hash(rho.tolist()),
        "random_draw_hash": content_hash(draws.tolist()),
    }
    random_identity = {**random_core, "identity_hash": content_hash(random_core)}
    rows = []
    for index, identifier in enumerate(identifiers):
        candidate_case_hash = case_identity_hash(cases[index]) if cases else None
        learning_input_hash = learning_input_fingerprint(cases[index]) if cases else None
        rows.append(
            {
                "candidate_id": identifier,
                "lineage_id": f"{round_lineage_id}:candidate:{identifier}",
                "epistemic_group_id": f"{round_lineage_id}:epistemic:{identifier}",
                "round": round_index,
                "round_lineage_id": round_lineage_id,
                "parent_round_lineage_id": parent_round_lineage_id,
                "split": "train",
                "role": "adaptive_train",
                "stream": "adaptive_train",
                "stratum": stratum_names[index],
                "p0_density": float(p0[index]),
                "adaptive_density": float(adaptive[index]),
                "proposal_density": float(proposal[index]),
                "p0_hierarchy_probability": float(p0_hierarchy[index]),
                "proposal_hierarchy_probability": float(proposal_hierarchy[index]),
                "score_rarity": float(score_vectors[0][1][index]),
                "score_uncertainty": float(score_vectors[1][1][index]),
                "score_coverage": float(score_vectors[2][1][index]),
                "score_total": float(selection_score[index]),
                "inclusion_probability": float(rho[index]),
                "design_weight_raw": float(weights[index]),
                "selected": bool(selected[index]),
                "weight_mask": True,
                "selection_seed": int(selection_seed),
                "random_draw": float(draws[index]),
                "acquisition_model_hash": acquisition_model_hash,
                "p0_id": p0_id,
                "p0_hash": p0_hash,
                "candidate_case_hash": candidate_case_hash,
                "learning_input_hash": learning_input_hash,
            }
        )
    selected_count = int(np.count_nonzero(selected))
    if cases and selected_count == 0:
        raise InputValidationError(
            "adaptive proposal selected no executable candidate cases; choose a different frozen selection design"
        )
    candidate_case_hashes = [row["candidate_case_hash"] for row in rows] if cases else []
    learning_input_hashes = [row["learning_input_hash"] for row in rows] if cases else []
    selected_case_bindings = [
        {
            "candidate_id": row["candidate_id"],
            "candidate_case_hash": row["candidate_case_hash"],
            "learning_input_hash": row["learning_input_hash"],
            "inclusion_probability": row["inclusion_probability"],
            "design_weight_raw": row["design_weight_raw"],
        }
        for row in rows
        if row["selected"]
    ]
    proposal_core = {
        "schema_version": ADAPTIVE_PROPOSAL_SCHEMA_VERSION,
        "round": round_index,
        "round_lineage_id": round_lineage_id,
        "parent_round_lineage_id": parent_round_lineage_id,
        "stream": "adaptive_train",
        "role": "adaptive_train",
        "split": "train",
        "p0_id": p0_id,
        "p0_hash": p0_hash,
        "adaptive_proposal_hash": adaptive_proposal_hash,
        "acquisition_model_hash": acquisition_model_hash,
        "acquisition_history": history_identity,
        "defensive_reference_fraction": float(reference_fraction),
        "expected_selected_count": float(expected_count),
        "rho_min": float(rho_min),
        "candidate_count": len(rows),
        "realized_selected_count": selected_count,
        "candidate_order_hash": content_hash(identifiers),
        "candidate_ledger_hash": content_hash(rows),
        "candidate_binding_status": "content_bound" if cases else "unbound_abstract",
        "candidate_case_order_hash": content_hash(candidate_case_hashes) if cases else None,
        "candidate_case_ledger_hash": content_hash(candidate_case_hashes) if cases else None,
        "learning_input_order_hash": content_hash(learning_input_hashes) if cases else None,
        "selected_case_set_hash": content_hash(selected_case_bindings) if cases else None,
        "selection_random_identity": random_identity,
        "score_policy": "externally_supplied_components_and_selection_score_no_domain_rule_inferred",
        "design_weight_formula": "p0_density*p0_hierarchy_probability/(proposal_density*proposal_hierarchy_probability*inclusion_probability)",
    }
    manifest = {**proposal_core, "proposal_hash": content_hash(proposal_core)}
    return {"manifest": manifest, "rows": rows, "candidate_cases": cases}


def tempered_prevalence_target(
    prevalence: np.ndarray | list[float], *, gamma: float = 0.5, epsilon: float = 1.0e-12
) -> np.ndarray:
    """Experimental-dormant rare-event target transform.

    The executable acquisition path does not call this function.  It remains
    available for a future pre-registered rare-event policy experiment; it may
    be retired after one release cycle with no internal or documented external
    use.  Calling it does not change the active sampling estimand.
    """

    values = np.asarray(prevalence, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise InputValidationError("prevalence must be a finite nonempty vector")
    if np.any(values < 0.0) or not 0.0 < gamma <= 1.0 or epsilon <= 0.0:
        raise InputValidationError("prevalence, gamma or epsilon is outside its valid domain")
    target = np.power(values + epsilon, gamma)
    return target / np.sum(target)


def inclusion_probabilities(
    scores: np.ndarray | list[float],
    *,
    expected_count: float,
    rho_min: float = 0.1,
) -> np.ndarray:
    """Map scores to positive Bernoulli inclusion probabilities.

    The saturating mapping is ``rho_min + (1-rho_min) c*s/(1+c*s)``.
    ``c`` is solved by bisection so the expected selected count matches the
    requested budget.  Equal/zero scores fall back to a uniform probability.
    """

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise InputValidationError("candidate scores must be a finite nonempty vector")
    if np.any(values < 0.0):
        raise InputValidationError("candidate scores must be nonnegative")
    if not math.isfinite(expected_count) or not 0.0 < expected_count <= values.size:
        raise InputValidationError("expected_count must lie in (0, candidate_count]")
    if not math.isfinite(rho_min) or not 0.0 < rho_min <= 1.0:
        raise InputValidationError("rho_min must lie in (0, 1]")
    minimum_count = rho_min * values.size
    if expected_count < minimum_count - 1.0e-12:
        raise InputValidationError(
            "expected_count is smaller than the positivity floor permits",
            context={"minimum_expected_count": minimum_count},
        )
    if expected_count >= values.size - 1.0e-12:
        return np.ones(values.size, dtype=np.float64)
    if np.all(values == values[0]):
        return np.full(values.size, expected_count / values.size, dtype=np.float64)
    scale = float(np.max(values))
    score_floor = max(scale, 1.0) * 1.0e-12
    normalized = (values + score_floor) / (scale + score_floor)
    # Zero-score cases must still be selectable at rho_min.  Positive scores
    # approach one monotonically as c grows.
    def probabilities(multiplier: float) -> np.ndarray:
        weighted = multiplier * normalized
        return rho_min + (1.0 - rho_min) * (1.0 - 1.0 / (1.0 + weighted))

    low, high = 0.0, 1.0
    while float(np.sum(probabilities(high))) < expected_count and high < 1.0e100:
        high *= 2.0
    if float(np.sum(probabilities(high))) < expected_count - 1.0e-9:
        raise RuntimeError("inclusion probability solver could not meet the expected count")
    for _ in range(100):
        midpoint = 0.5 * (low + high)
        if float(np.sum(probabilities(midpoint))) < expected_count:
            low = midpoint
        else:
            high = midpoint
    result = probabilities(0.5 * (low + high))
    if np.any(result <= 0.0) or np.any(result > 1.0) or not np.all(np.isfinite(result)):
        raise RuntimeError("inclusion probability solver produced an invalid result")
    return result


def select_bernoulli(
    probabilities: np.ndarray | list[float], *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise InputValidationError("inclusion probabilities must be a finite vector")
    if np.any(values <= 0.0) or np.any(values > 1.0):
        raise InputValidationError("inclusion probabilities must lie in (0, 1]")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)) or int(seed) < 0:
        raise InputValidationError("selection seed must be nonnegative")
    draws = np.random.default_rng(int(seed)).random(values.size)
    return draws < values, draws


def raw_design_weights(probabilities: np.ndarray | list[float]) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values <= 0.0) or np.any(values > 1.0):
        raise InputValidationError("inclusion probabilities must lie in (0, 1]")
    return 1.0 / values


def effective_sample_size(weights: np.ndarray | list[float]) -> float:
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise InputValidationError("weights must be a finite positive vector")
    return positive_effective_sample_size(values)


def hajek_mean(values: np.ndarray | list[float], weights: np.ndarray | list[float]) -> float:
    observations = np.asarray(values, dtype=np.float64)
    design_weights = np.asarray(weights, dtype=np.float64)
    if observations.shape != design_weights.shape or observations.ndim != 1:
        raise InputValidationError("values and weights must be one-dimensional and aligned")
    if not np.all(np.isfinite(observations)):
        raise InputValidationError("values must be finite")
    effective_sample_size(design_weights)
    return float(
        positive_weighted_mean(
            observations, design_weights, field="Hájek design weights"
        )
    )
