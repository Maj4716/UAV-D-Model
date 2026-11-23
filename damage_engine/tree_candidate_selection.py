"""Seed-separated selection and validation of offline tree candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
from typing import Any

import numpy as np

from .dataset_v2 import load_dataset_v2_core, validate_component_state_trace
from .errors import InputValidationError
from .offline_tree_candidates import compile_offline_tree_candidate
from .utils import content_hash


SELECTION_SCHEMA_VERSION = "offline-tree-selection@1.0.0"
DEFAULT_GATES = {
    "middle_lower_inclusive": 0.01,
    "middle_upper_inclusive": 0.99,
    "minimum_middle_fraction": 0.30,
    "saturation_threshold_inclusive": 0.99,
    "maximum_saturation_fraction": 0.25,
    "zero_threshold_inclusive": 1.0e-6,
    "maximum_zero_fraction": 0.70,
    "minimum_weighted_binary_entropy_nats": 0.0,
}


def _normalized_gates(raw: Mapping[str, Any] | None) -> dict[str, float]:
    gates = {**DEFAULT_GATES, **(dict(raw) if raw is not None else {})}
    if set(gates) != set(DEFAULT_GATES):
        raise InputValidationError("tree selection gates contain unsupported fields")
    parsed = {key: float(value) for key, value in gates.items()}
    if not all(math.isfinite(value) for value in parsed.values()):
        raise InputValidationError("tree selection gates must be finite")
    if not 0.0 <= parsed["middle_lower_inclusive"] < parsed["middle_upper_inclusive"] <= 1.0:
        raise InputValidationError("tree selection middle interval is invalid")
    for key in (
        "minimum_middle_fraction",
        "saturation_threshold_inclusive",
        "maximum_saturation_fraction",
        "zero_threshold_inclusive",
        "maximum_zero_fraction",
        "minimum_weighted_binary_entropy_nats",
    ):
        upper = math.log(2.0) if key == "minimum_weighted_binary_entropy_nats" else 1.0
        if not 0.0 <= parsed[key] <= upper:
            raise InputValidationError(
                f"tree selection gate {key} must lie in [0,{upper}]"
            )
    return parsed


def _weighted_fraction(mask: np.ndarray, weights: np.ndarray) -> float:
    denominator = float(np.sum(weights))
    if denominator <= 0.0:
        raise InputValidationError("tree selection weights must have positive total")
    return float(np.sum(weights * mask.astype(np.float64)) / denominator)


def _weighted_entropy(probabilities: np.ndarray, weights: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1.0e-15, 1.0 - 1.0e-15)
    values = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log1p(-clipped))
    return float(np.sum(weights * values) / np.sum(weights))


def evaluate_candidate_on_seed(
    candidate_document: Mapping[str, Any],
    traces: Sequence[Mapping[str, Any]],
    design_seeds: np.ndarray,
    weights: np.ndarray,
    *,
    seed: int,
    gates: Mapping[str, float],
) -> dict[str, Any]:
    candidate = compile_offline_tree_candidate(candidate_document)
    indices = np.flatnonzero(design_seeds == int(seed))
    if indices.size == 0:
        raise InputValidationError(
            "requested design seed has no dataset cases", context={"seed": int(seed)}
        )
    rows: list[list[float]] = []
    for index in indices.tolist():
        trace = traces[index]["component_state_trace"]
        validate_component_state_trace(trace)
        replicate_rows = []
        for record in trace["records"]:
            result = candidate.evaluate(record["component_mechanism_states"])
            replicate_rows.append(
                [float(result["outputs"][name]) for name in candidate.output_names]
            )
        rows.append(np.mean(np.asarray(replicate_rows, dtype=np.float64), axis=0).tolist())
    values = np.asarray(rows, dtype=np.float64)
    selected_weights = np.asarray(weights[indices], dtype=np.float64)
    output_metrics: dict[str, Any] = {}
    for column, name in enumerate(candidate.output_names):
        probabilities = values[:, column]
        middle = _weighted_fraction(
            (probabilities >= gates["middle_lower_inclusive"])
            & (probabilities <= gates["middle_upper_inclusive"]),
            selected_weights,
        )
        saturation = _weighted_fraction(
            probabilities >= gates["saturation_threshold_inclusive"], selected_weights
        )
        zero = _weighted_fraction(
            probabilities <= gates["zero_threshold_inclusive"], selected_weights
        )
        entropy = _weighted_entropy(probabilities, selected_weights)
        output_metrics[name] = {
            "middle_fraction": middle,
            "saturation_fraction": saturation,
            "zero_fraction": zero,
            "weighted_binary_entropy_nats": entropy,
            "passes": bool(
                middle >= gates["minimum_middle_fraction"]
                and saturation <= gates["maximum_saturation_fraction"]
                and zero <= gates["maximum_zero_fraction"]
                and entropy >= gates["minimum_weighted_binary_entropy_nats"]
            ),
        }
    passes = all(value["passes"] for value in output_metrics.values())
    return {
        "candidate_id": candidate.candidate_id,
        "candidate_hash": candidate.definition_hash,
        "candidate_status": candidate.status,
        "semantic_authority_status": candidate.authority_status,
        "seed": int(seed),
        "case_count": int(indices.size),
        "case_indices": indices.tolist(),
        "output_metrics": output_metrics,
        "minimum_middle_fraction": min(
            value["middle_fraction"] for value in output_metrics.values()
        ),
        "mean_weighted_binary_entropy_nats": float(
            np.mean(
                [value["weighted_binary_entropy_nats"] for value in output_metrics.values()]
            )
        ),
        "passes_numerical_gates": passes,
    }


def select_and_validate_tree_candidate(
    dataset_dir: str | Path,
    candidate_documents: Sequence[Mapping[str, Any]],
    *,
    development_seed: int,
    validation_seed: int,
    preregistration_id: str,
    gates: Mapping[str, Any] | None = None,
    weight_field: str = "design_weight",
) -> dict[str, Any]:
    """Select on one design seed and assess once on a distinct design seed."""

    if not preregistration_id:
        raise InputValidationError("tree selection requires a preregistration_id")
    if int(development_seed) == int(validation_seed):
        raise InputValidationError("development and validation seeds must be distinct")
    if not candidate_documents:
        raise InputValidationError("tree selection requires at least one candidate")
    parsed_gates = _normalized_gates(gates)
    core = load_dataset_v2_core(dataset_dir)
    sampling = core["sampling_weights"]
    if weight_field not in {"design_weight", "target_measure_weight"}:
        raise InputValidationError(
            "tree selection weight_field must be design_weight or target_measure_weight"
        )
    design_seeds = np.asarray(sampling["design_seed"], dtype=np.int64)
    weights = np.asarray(sampling[weight_field], dtype=np.float64)
    traces = core["component_state_traces"]
    compiled = [compile_offline_tree_candidate(item) for item in candidate_documents]
    hashes = [item.definition_hash for item in compiled]
    if len(hashes) != len(set(hashes)):
        raise InputValidationError("tree selection candidate definitions must be unique")

    development = [
        evaluate_candidate_on_seed(
            candidate.document,
            traces,
            design_seeds,
            weights,
            seed=int(development_seed),
            gates=parsed_gates,
        )
        for candidate in compiled
    ]
    eligible = [item for item in development if item["passes_numerical_gates"]]
    selected_development = min(
        eligible,
        key=lambda item: (
            -(1 if item["semantic_authority_status"] == "approved" else 0),
            -item["minimum_middle_fraction"],
            -item["mean_weighted_binary_entropy_nats"],
            item["candidate_hash"],
        ),
        default=None,
    )
    selected = None
    validation = None
    promotion_eligible = False
    promotion_reason = "no_candidate_passed_development_gates"
    if selected_development is not None:
        selected = next(
            item for item in compiled if item.definition_hash == selected_development["candidate_hash"]
        )
        validation = evaluate_candidate_on_seed(
            selected.document,
            traces,
            design_seeds,
            weights,
            seed=int(validation_seed),
            gates=parsed_gates,
        )
        if not validation["passes_numerical_gates"]:
            promotion_reason = "selected_candidate_failed_independent_validation_gates"
        elif selected.authority_status != "approved":
            promotion_reason = "semantic_authority_not_approved"
        else:
            promotion_eligible = True
            promotion_reason = "numerical_gates_passed_and_semantic_authority_approved"

    report: dict[str, Any] = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "preregistration_id": preregistration_id,
        "dataset": str(Path(dataset_dir)),
        "candidate_hashes": hashes,
        "gates": parsed_gates,
        "weight_field": weight_field,
        "seed_policy": {
            "development_seed": int(development_seed),
            "validation_seed": int(validation_seed),
            "distinct": True,
            "selection_uses_validation_outcomes": False,
        },
        "selection_rule": (
            "among development-gate passers: prefer domain-approved candidates, then "
            "maximize minimum middle fraction, then mean weighted binary entropy, "
            "then ascending definition hash"
        ),
        "development_results": development,
        "selected_candidate_id": None if selected is None else selected.candidate_id,
        "selected_candidate_hash": None if selected is None else selected.definition_hash,
        "validation_result": validation,
        "promotion_eligible": promotion_eligible,
        "promotion_reason": promotion_reason,
    }
    report["report_hash"] = content_hash(report)
    return report
