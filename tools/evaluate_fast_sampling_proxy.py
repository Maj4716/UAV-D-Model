"""Evaluate the fast sampling proxy before any full dataset is generated.

The command screens an outcome-blind proposal frame, freezes two balanced
selection arms, and only then runs the current explicit engine on their union:

* proxy-greedy: component-coverage acquisition within approach/profile cells;
* stratified-random: equal-budget reference from the same candidate frame.

The resulting evidence is diagnostic.  It is not a formal dataset archive and
does not modify frozen Reference/Validation/Test streams.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.engine import ENGINE_CODE_HASH, ENGINE_VERSION, DamageEngine
from damage_engine.errors import DamageEngineError
from damage_engine.fast_sampling_proxy import (
    FAST_SAMPLING_PROXY_VERSION,
    FastSamplingProxy,
)
from damage_engine.registry import PhysicsRegistry
from damage_engine.schema_resources import schema_tree_hash
from damage_engine.utils import content_hash, source_tree_hash


DEFAULT_TARGET = Path("configs/targets/target.fragment_only.v1.json")
DEFAULT_PHYSICS = Path("configs/physics/loitering_impact.json")

# Frozen before the current experiment is run.  These are engineering gates,
# not field-wide physical standards.
GATES = {
    "case_any_hit_auc_min": 0.85,
    "case_hit_spearman_min": 0.80,
    "case_energy_spearman_min": 0.80,
    "component_support_auc_min": 0.85,
    "component_support_recall_min": 0.90,
    "top20_overlap_min": 0.70,
    "speedup_min": 10.0,
    "coarse_fine_rank_min": 0.90,
    "coarse_fine_component_l1_max": 0.15,
}
MIN_ADMISSION_TEACHER_REPLICATES = 32
TEACHER_COMPONENT_SUPPORT_RATE_THRESHOLD = 0.05


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            for row in rows:
                line = json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                stream.write((line + "\n").encode("utf-8"))


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    x = _average_ranks(np.asarray(left, dtype=np.float64))
    y = _average_ranks(np.asarray(right, dtype=np.float64))
    if float(np.std(x)) <= 0.0 or float(np.std(y)) <= 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _roc_auc(labels: list[int], scores: list[float]) -> float | None:
    if len(labels) != len(scores) or not labels:
        return None
    y = np.asarray(labels, dtype=np.int8)
    positive = int(np.sum(y == 1))
    negative = int(np.sum(y == 0))
    if positive == 0 or negative == 0:
        return None
    ranks = _average_ranks(np.asarray(scores, dtype=np.float64))
    numerator = float(np.sum(ranks[y == 1])) - positive * (positive + 1) / 2.0
    return numerator / (positive * negative)


def _stratified_bootstrap_interval(
    strata: list[str],
    metric: Callable[[np.ndarray], float | None],
    *,
    seed: int,
    draws: int,
) -> list[float] | None:
    """Bootstrap cases within fixed approach/profile cells.

    The random validation arm contains the same number of cases from every
    approach/profile cell.  Resampling within each cell preserves that design
    and therefore targets the equal-cell mixture used by the point estimate.
    """

    raw_groups: dict[str, list[int]] = {}
    for index, cell in enumerate(strata):
        raw_groups.setdefault(str(cell), []).append(index)
    groups = {
        cell: np.asarray(indices, dtype=np.int64)
        for cell, indices in raw_groups.items()
    }
    if not groups or sum(len(indices) for indices in groups.values()) < 2:
        return None
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(draws):
        sampled = np.concatenate(
            [
                rng.choice(indices, size=len(indices), replace=True)
                for _, indices in sorted(groups.items())
            ]
        )
        value = metric(sampled)
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    if len(values) < max(20, draws // 10):
        return None
    return [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


def _evaluation_source_identity(
    input_paths: dict[str, Path] | None = None,
) -> dict[str, str]:
    identity = {
        "evaluation_tool_file_hash": _file_hash(Path(__file__).resolve()),
        "proxy_implementation_file_hash": _file_hash(
            ROOT / "damage_engine" / "fast_sampling_proxy.py"
        ),
        "gate_configuration_hash": content_hash(GATES),
    }
    for name, path in sorted((input_paths or {}).items()):
        identity[f"{name}_file_hash"] = _file_hash(path.resolve())
    return identity


def _case_cell(case: dict[str, Any]) -> str:
    profile = str(case["munition"]["profile_id"])
    approach = str(
        case.get("metadata", {})
        .get("contact_manifold_sampling", {})
        .get("approach_id", "unknown")
    )
    return f"{approach}|{profile}"


def _screen(
    cases: list[dict[str, Any]],
    candidate_ids: list[str],
    proxy: FastSamplingProxy,
) -> tuple[list[dict[str, Any]], float]:
    started = perf_counter()
    rows: list[dict[str, Any]] = []
    for candidate_id, case in zip(candidate_ids, cases, strict=True):
        result = proxy.evaluate(case)
        result["candidate_id"] = candidate_id
        result["approach_profile_cell"] = _case_cell(case)
        rows.append(result)
    return rows, perf_counter() - started


def _greedy_coverage_selection(
    rows: list[dict[str, Any]],
    *,
    per_cell: int,
) -> list[str]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["approach_profile_cell"], []).append(row)
    selected: list[str] = []
    for cell in sorted(groups):
        available = list(groups[cell])
        if len(available) < per_cell:
            raise ValueError(f"cell {cell} has fewer than {per_cell} candidates")
        component_ids = sorted(available[0]["component_metrics"])
        support = {identifier: 0.0 for identifier in component_ids}
        for _ in range(per_cell):
            scored: list[tuple[float, float, str, dict[str, Any]]] = []
            for row in available:
                gain = 0.0
                for identifier in component_ids:
                    exposure = min(
                        1.0,
                        float(
                            row["component_metrics"][identifier]["expected_hits"]
                        ),
                    )
                    gain += exposure / (0.25 + support[identifier])
                gain += 0.02 * math.log1p(
                    float(row["expected_total_component_damage_energy_j"])
                )
                scored.append(
                    (
                        gain,
                        float(row["sampling_score"]),
                        str(row["candidate_id"]),
                        row,
                    )
                )
            _, _, candidate_id, winner = max(scored, key=lambda item: item[:3])
            selected.append(candidate_id)
            for identifier in component_ids:
                support[identifier] += float(
                    winner["component_metrics"][identifier]["expected_hits"]
                )
            available = [row for row in available if row["candidate_id"] != candidate_id]
    return selected


def _stratified_random_selection(
    rows: list[dict[str, Any]],
    *,
    per_cell: int,
    seed: int,
) -> list[str]:
    groups: dict[str, list[str]] = {}
    for row in rows:
        groups.setdefault(row["approach_profile_cell"], []).append(
            str(row["candidate_id"])
        )
    rng = np.random.default_rng(seed)
    selected: list[str] = []
    for cell in sorted(groups):
        values = np.asarray(sorted(groups[cell]), dtype=object)
        if len(values) < per_cell:
            raise ValueError(f"cell {cell} has fewer than {per_cell} candidates")
        selected.extend(str(value) for value in rng.choice(values, per_cell, replace=False))
    return selected


def _selection_design_ledger(
    rows: list[dict[str, Any]],
    *,
    greedy_ids: list[str],
    random_ids: list[str],
    per_cell: int,
) -> dict[str, Any]:
    """Record finite-frame inclusion probabilities for the mixed design.

    Greedy cases have union inclusion probability one.  Every other case can
    enter only through simple random sampling without replacement inside its
    approach/profile cell, giving probability ``k / N_h``.  The ledger is
    written for the full candidate frame, not only the observed union.
    """

    if per_cell < 1:
        raise ValueError("per_cell must be positive")
    row_by_id: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[str]] = {}
    for row in rows:
        candidate_id = str(row["candidate_id"])
        if candidate_id in row_by_id:
            raise ValueError(f"duplicate candidate id: {candidate_id}")
        row_by_id[candidate_id] = row
        groups.setdefault(str(row["approach_profile_cell"]), []).append(candidate_id)
    greedy = set(map(str, greedy_ids))
    random = set(map(str, random_ids))
    unknown = sorted((greedy | random) - set(row_by_id))
    if unknown:
        raise ValueError(f"selected candidate IDs are outside the frame: {unknown}")
    for cell, identifiers in groups.items():
        if per_cell > len(identifiers):
            raise ValueError(f"cell {cell} has fewer than {per_cell} candidates")
        random_count = sum(identifier in random for identifier in identifiers)
        greedy_count = sum(identifier in greedy for identifier in identifiers)
        if random_count != per_cell or greedy_count != per_cell:
            raise ValueError(
                f"cell {cell} must contain exactly {per_cell} selections per arm"
            )

    ledger: list[dict[str, Any]] = []
    for cell in sorted(groups):
        identifiers = sorted(groups[cell])
        cell_size = len(identifiers)
        random_probability = per_cell / cell_size
        for candidate_id in identifiers:
            in_greedy = candidate_id in greedy
            in_random = candidate_id in random
            union_probability = 1.0 if in_greedy else random_probability
            role = (
                "greedy_and_random"
                if in_greedy and in_random
                else "greedy"
                if in_greedy
                else "random"
                if in_random
                else "not_selected"
            )
            ledger.append(
                {
                    "candidate_id": candidate_id,
                    "approach_profile_cell": cell,
                    "cell_size": cell_size,
                    "random_inclusion_probability": random_probability,
                    "union_inclusion_probability": union_probability,
                    "design_weight_raw": 1.0 / union_probability,
                    "selected": in_greedy or in_random,
                    "selection_role": role,
                }
            )
    selected_weights = [
        float(row["design_weight_raw"]) for row in ledger if row["selected"]
    ]
    weight_sum = float(sum(selected_weights))
    ess = (
        weight_sum * weight_sum
        / max(sum(weight * weight for weight in selected_weights), np.finfo(float).eps)
    )
    return {
        "probability_semantics": (
            "finite_candidate_frame_union_inclusion_probability; greedy membership "
            "is deterministic after proxy screening and the random arm is SRSWOR per cell"
        ),
        "random_exploration_per_cell": per_cell,
        "random_exploration_nominal_fraction": 0.5,
        "candidate_ledger": ledger,
        "candidate_ledger_hash": content_hash(ledger),
        "positivity_min": min(
            float(row["union_inclusion_probability"]) for row in ledger
        ),
        "selected_design_ess": ess,
        "selected_design_ess_fraction": ess / len(selected_weights),
        "selected_count": len(selected_weights),
    }


def _teacher_metrics(result: dict[str, Any]) -> dict[str, Any]:
    components: dict[str, dict[str, float | int]] = {}
    positive: list[str] = []
    total_energy = 0.0
    for identifier, record in result["components"].items():
        metrics = record["fragment_metrics"]
        hits = float(metrics["hits"]["mean"])
        penetrations = float(metrics["penetrations"]["mean"])
        deposited = float(metrics["deposited_energy_j"]["mean"])
        support_rate = float(
            metrics.get("hit_indicator", metrics["hits"])["mean"]
        )
        if not 0.0 <= support_rate <= 1.0:
            raise ValueError(
                f"component hit-indicator mean must lie in [0,1]: {identifier}"
            )
        supported = int(
            support_rate >= TEACHER_COMPONENT_SUPPORT_RATE_THRESHOLD
        )
        if supported:
            positive.append(identifier)
        total_energy += deposited
        components[identifier] = {
            "hits": hits,
            "penetrations": penetrations,
            "deposited_energy_j": deposited,
            "support_rate": support_rate,
            "positive_support": supported,
        }
    tree_outputs = {
        name: float(record["mean"])
        for name, record in result["damage_tree"]["outputs"].items()
    }
    scientific_completion_status = str(
        result.get("scientific_completion_status", "missing")
    )
    penetration_coverage = result.get("penetration_domain_coverage", {})
    affected_replicate_fraction = float(
        penetration_coverage.get("affected_replicate_fraction", 1.0)
    )
    domain_complete = (
        scientific_completion_status == "complete_model_domain"
        and affected_replicate_fraction == 0.0
    )
    return {
        "case_id": str(result["case_id"]),
        "replicate_count": int(result["replicate_count"]),
        "total_component_hits": float(
            result["summary"]["total_component_hits"]["mean"]
        ),
        "total_component_penetrations": float(
            result["summary"]["total_component_penetrations"]["mean"]
        ),
        "total_component_deposited_energy_j": float(total_energy),
        "support_estimand": "component_hit_probability_at_least_threshold",
        "component_support_rate_threshold": (
            TEACHER_COMPONENT_SUPPORT_RATE_THRESHOLD
        ),
        "any_component_support": int(bool(positive)),
        "positive_component_ids": sorted(positive),
        "positive_component_count": len(positive),
        "components": components,
        "tree_outputs": tree_outputs,
        "tree_output_sum": float(sum(tree_outputs.values())),
        "scientific_completion_status": scientific_completion_status,
        "penetration_affected_replicate_fraction": affected_replicate_fraction,
        "domain_complete": domain_complete,
    }


def _run_teacher(
    engine: DamageEngine,
    cases_by_id: dict[str, dict[str, Any]],
    candidate_ids: list[str],
    *,
    replicates: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    started = perf_counter()
    metrics_rows: list[dict[str, Any]] = []
    full_results: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        case = deepcopy(cases_by_id[candidate_id])
        case["analysis"]["replicates"] = int(replicates)
        case["analysis"]["adaptive_ci_half_width"] = None
        case["analysis"]["capture_fragment_audit"] = False
        case["analysis"]["capture_replicate_trace"] = False
        case_started = perf_counter()
        try:
            result = engine.run(case, fail_closed=True).to_dict()
            metrics = _teacher_metrics(result)
            metrics.update(
                {
                    "candidate_id": candidate_id,
                    "approach_profile_cell": _case_cell(case),
                    "status": (
                        "ok" if metrics["domain_complete"] else "domain_censored"
                    ),
                    "elapsed_seconds": perf_counter() - case_started,
                    "result_hash": content_hash(result),
                }
            )
            metrics_rows.append(metrics)
            full_results.append({"candidate_id": candidate_id, "result": result})
        except DamageEngineError as error:
            metrics_rows.append(
                {
                    "candidate_id": candidate_id,
                    "case_id": case.get("case_id"),
                    "approach_profile_cell": _case_cell(case),
                    "status": "refused",
                    "elapsed_seconds": perf_counter() - case_started,
                    "error": error.to_dict(),
                }
            )
    return metrics_rows, full_results, perf_counter() - started


def _accuracy_report(
    proxy_by_id: dict[str, dict[str, Any]],
    teacher_rows: list[dict[str, Any]],
    *,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    rows = [row for row in teacher_rows if row["status"] == "ok"]
    strata = [str(row["approach_profile_cell"]) for row in rows]
    cell_case_counts: dict[str, int] = {}
    for cell in strata:
        cell_case_counts[cell] = cell_case_counts.get(cell, 0) + 1
    case_labels = [int(row["any_component_support"]) for row in rows]
    case_scores = [
        float(proxy_by_id[row["candidate_id"]]["expected_total_component_hits"])
        for row in rows
    ]
    proxy_hits = case_scores
    actual_hits = [float(row["total_component_hits"]) for row in rows]
    proxy_energy = [
        float(
            proxy_by_id[row["candidate_id"]][
                "expected_total_component_absorbed_energy_j"
            ]
        )
        for row in rows
    ]
    actual_energy = [float(row["total_component_deposited_energy_j"]) for row in rows]

    component_labels: list[int] = []
    component_scores: list[float] = []
    component_pairs_by_case: list[tuple[list[int], list[float]]] = []
    component_predictions_by_case: list[tuple[list[int], list[int]]] = []
    true_positive_count = 0
    predicted_positive_count = 0
    true_predicted_positive_count = 0
    for row in rows:
        proxy = proxy_by_id[row["candidate_id"]]
        labels: list[int] = []
        scores: list[float] = []
        for identifier in sorted(row["components"]):
            label = int(row["components"][identifier]["positive_support"])
            score = float(proxy["component_metrics"][identifier]["expected_hits"])
            predicted = int(score >= float(proxy["component_support_threshold"]))
            labels.append(label)
            scores.append(score)
            true_positive_count += label
            predicted_positive_count += predicted
            true_predicted_positive_count += label * predicted
        component_labels.extend(labels)
        component_scores.extend(scores)
        component_pairs_by_case.append((labels, scores))
        component_predictions_by_case.append(
            (
                labels,
                [
                    int(score >= float(proxy["component_support_threshold"]))
                    for score in scores
                ],
            )
        )

    n_cases = len(rows)
    top_count = max(1, int(math.ceil(0.20 * n_cases)))
    proxy_top = {
        rows[index]["candidate_id"]
        for index in np.argsort(np.asarray(proxy_energy))[-top_count:]
    }
    actual_top = {
        rows[index]["candidate_id"]
        for index in np.argsort(np.asarray(actual_energy))[-top_count:]
    }
    top_overlap = len(proxy_top & actual_top) / top_count

    hit_spearman = _spearman(proxy_hits, actual_hits)
    energy_spearman = _spearman(proxy_energy, actual_energy)
    case_auc = _roc_auc(case_labels, case_scores)
    component_auc = _roc_auc(component_labels, component_scores)

    def case_auc_boot(indices: np.ndarray) -> float | None:
        return _roc_auc(
            [case_labels[index] for index in indices],
            [case_scores[index] for index in indices],
        )

    def hit_rank_boot(indices: np.ndarray) -> float | None:
        return _spearman(
            [proxy_hits[index] for index in indices],
            [actual_hits[index] for index in indices],
        )

    def energy_rank_boot(indices: np.ndarray) -> float | None:
        return _spearman(
            [proxy_energy[index] for index in indices],
            [actual_energy[index] for index in indices],
        )

    def component_auc_boot(indices: np.ndarray) -> float | None:
        labels: list[int] = []
        scores: list[float] = []
        for index in indices:
            current_labels, current_scores = component_pairs_by_case[int(index)]
            labels.extend(current_labels)
            scores.extend(current_scores)
        return _roc_auc(labels, scores)

    def component_recall_boot(indices: np.ndarray) -> float | None:
        positives = 0
        true_positives = 0
        for index in indices:
            labels, predictions = component_predictions_by_case[int(index)]
            positives += sum(labels)
            true_positives += sum(
                label * prediction
                for label, prediction in zip(labels, predictions, strict=True)
            )
        return true_positives / positives if positives else None

    def top_overlap_boot(indices: np.ndarray) -> float | None:
        if len(indices) == 0:
            return None
        sampled_proxy = np.asarray(
            [proxy_energy[int(index)] for index in indices], dtype=np.float64
        )
        sampled_actual = np.asarray(
            [actual_energy[int(index)] for index in indices], dtype=np.float64
        )
        count = max(1, int(math.ceil(0.20 * len(indices))))
        proxy_positions = set(np.argsort(sampled_proxy)[-count:].tolist())
        actual_positions = set(np.argsort(sampled_actual)[-count:].tolist())
        return len(proxy_positions & actual_positions) / count

    return {
        "estimand": "equal_approach_profile_cell_mixture",
        "bootstrap_design": "case_resampling_within_fixed_approach_profile_cells",
        "case_count": n_cases,
        "cell_count": len(cell_case_counts),
        "cell_case_counts": dict(sorted(cell_case_counts.items())),
        "cell_balance_complete": len(set(cell_case_counts.values())) <= 1,
        "case_any_hit_prevalence": float(np.mean(case_labels)) if rows else None,
        "case_any_hit_auc": _finite_or_none(case_auc),
        "case_any_hit_auc_ci95": _stratified_bootstrap_interval(
            strata,
            case_auc_boot,
            seed=bootstrap_seed + 1,
            draws=bootstrap_draws,
        ),
        "case_hit_spearman": _finite_or_none(hit_spearman),
        "case_hit_spearman_ci95": _stratified_bootstrap_interval(
            strata,
            hit_rank_boot,
            seed=bootstrap_seed + 2,
            draws=bootstrap_draws,
        ),
        "case_energy_spearman": _finite_or_none(energy_spearman),
        "case_energy_spearman_ci95": _stratified_bootstrap_interval(
            strata,
            energy_rank_boot,
            seed=bootstrap_seed + 3,
            draws=bootstrap_draws,
        ),
        "component_support_auc": _finite_or_none(component_auc),
        "component_support_auc_ci95": _stratified_bootstrap_interval(
            strata,
            component_auc_boot,
            seed=bootstrap_seed + 4,
            draws=bootstrap_draws,
        ),
        "component_support_recall": (
            true_predicted_positive_count / true_positive_count
            if true_positive_count
            else None
        ),
        "component_support_recall_ci95": _stratified_bootstrap_interval(
            strata,
            component_recall_boot,
            seed=bootstrap_seed + 5,
            draws=bootstrap_draws,
        ),
        "component_support_precision": (
            true_predicted_positive_count / predicted_positive_count
            if predicted_positive_count
            else None
        ),
        "component_positive_count": true_positive_count,
        "component_predicted_positive_count": predicted_positive_count,
        "top20_energy_overlap": float(top_overlap),
        "top20_energy_overlap_ci95": _stratified_bootstrap_interval(
            strata,
            top_overlap_boot,
            seed=bootstrap_seed + 6,
            draws=bootstrap_draws,
        ),
    }


def _random_arm_accuracy_report(
    proxy_by_id: dict[str, dict[str, Any]],
    teacher_rows: list[dict[str, Any]],
    random_ids: list[str],
    *,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Evaluate accuracy only on the outcome-blind stratified random arm."""

    random_set = set(map(str, random_ids))
    selected_rows = [
        row for row in teacher_rows if str(row["candidate_id"]) in random_set
    ]
    report = _accuracy_report(
        proxy_by_id,
        selected_rows,
        bootstrap_draws=bootstrap_draws,
        bootstrap_seed=bootstrap_seed,
    )
    report["evaluation_arm"] = "stratified_random_only"
    report["evaluated_candidate_ids"] = sorted(
        str(row["candidate_id"])
        for row in selected_rows
        if row["status"] == "ok"
    )
    return report


def _arm_report(
    candidate_ids: list[str],
    teacher_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = [teacher_by_id[identifier] for identifier in candidate_ids]
    rows = [row for row in rows if row["status"] == "ok"]
    component_counts: dict[str, int] = {}
    for row in rows:
        for identifier in row["positive_component_ids"]:
            component_counts[identifier] = component_counts.get(identifier, 0) + 1
    return {
        "requested_count": len(candidate_ids),
        "accepted_count": len(rows),
        "any_component_support_rate": (
            float(np.mean([row["any_component_support"] for row in rows]))
            if rows
            else None
        ),
        "mean_total_component_hits": (
            float(np.mean([row["total_component_hits"] for row in rows]))
            if rows
            else None
        ),
        "mean_total_component_deposited_energy_j": (
            float(
                np.mean([row["total_component_deposited_energy_j"] for row in rows])
            )
            if rows
            else None
        ),
        "component_coverage": {
            "ever_positive_count": len(component_counts),
            "at_least_5_cases": sum(value >= 5 for value in component_counts.values()),
            "at_least_10_cases": sum(value >= 10 for value in component_counts.values()),
            "counts": dict(sorted(component_counts.items())),
        },
    }


def _cell_stratified_arm_difference_bootstrap(
    proxy_ids: list[str],
    random_ids: list[str],
    teacher_by_id: dict[str, dict[str, Any]],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    left = [teacher_by_id[identifier] for identifier in proxy_ids]
    right = [teacher_by_id[identifier] for identifier in random_ids]
    left = [row for row in left if row["status"] == "ok"]
    right = [row for row in right if row["status"] == "ok"]
    left_by_cell: dict[str, list[dict[str, Any]]] = {}
    right_by_cell: dict[str, list[dict[str, Any]]] = {}
    for row in left:
        left_by_cell.setdefault(str(row["approach_profile_cell"]), []).append(row)
    for row in right:
        right_by_cell.setdefault(str(row["approach_profile_cell"]), []).append(row)
    cells = sorted(set(left_by_cell) & set(right_by_cell))
    if not cells:
        return {
            "estimand": "equal_approach_profile_cell_mean_difference",
            "bootstrap_unit": "approach_profile_cell",
            "status": "unavailable_no_domain_complete_shared_cells",
            "cell_count": 0,
            "shared_cells": [],
        }
    rng = np.random.default_rng(seed)
    output: dict[str, Any] = {
        "estimand": "equal_approach_profile_cell_mean_difference",
        "bootstrap_unit": "approach_profile_cell",
        "overlap_semantics": (
            "greedy/random overlap is retained inside each cell difference before "
            "cell-level resampling"
        ),
        "cell_count": len(cells),
        "shared_cells": cells,
        "status": "available",
    }
    fields = {
        "any_component_support_rate_difference": "any_component_support",
        "mean_total_component_hits_difference": "total_component_hits",
        "mean_total_component_energy_difference_j": "total_component_deposited_energy_j",
    }
    for output_name, field in fields.items():
        cell_differences = {
            cell: float(np.mean([row[field] for row in left_by_cell[cell]]))
            - float(np.mean([row[field] for row in right_by_cell[cell]]))
            for cell in cells
        }
        difference_values = np.asarray(
            [cell_differences[cell] for cell in cells], dtype=np.float64
        )
        point = float(np.mean(difference_values))
        samples: list[float] = []
        for _ in range(draws):
            indices = rng.integers(0, len(difference_values), len(difference_values))
            samples.append(float(np.mean(difference_values[indices])))
        output[output_name] = {
            "point": point,
            "ci95": (
                [
                    float(np.quantile(samples, 0.025)),
                    float(np.quantile(samples, 0.975)),
                ]
                if len(cells) >= 2
                else None
            ),
            "cell_differences": cell_differences,
        }
    return output


def _convergence_report(
    coarse_by_id: dict[str, dict[str, Any]],
    fine_by_id: dict[str, dict[str, Any]],
    *,
    lower_resolution: int | None = None,
    higher_resolution: int | None = None,
) -> dict[str, Any]:
    identifiers = sorted(set(coarse_by_id) & set(fine_by_id))
    coarse_scores = [float(coarse_by_id[item]["sampling_score"]) for item in identifiers]
    fine_scores = [float(fine_by_id[item]["sampling_score"]) for item in identifiers]
    component_ids = sorted(next(iter(coarse_by_id.values()))["component_metrics"])
    coarse_components = np.asarray(
        [
            sum(
                float(coarse_by_id[item]["component_metrics"][component]["expected_hits"])
                for item in identifiers
            )
            for component in component_ids
        ],
        dtype=np.float64,
    )
    fine_components = np.asarray(
        [
            sum(
                float(fine_by_id[item]["component_metrics"][component]["expected_hits"])
                for item in identifiers
            )
            for component in component_ids
        ],
        dtype=np.float64,
    )
    denominator = max(float(np.sum(np.abs(fine_components))), np.finfo(float).eps)
    comparison = (
        f"azimuths_per_ring_{lower_resolution}_vs_{higher_resolution}"
        if lower_resolution is not None and higher_resolution is not None
        else "lower_vs_higher_resolution"
    )
    return {
        "case_count": len(identifiers),
        "comparison": comparison,
        "lower_resolution": lower_resolution,
        "higher_resolution": higher_resolution,
        "higher_resolution_is_reference": higher_resolution is not None,
        "sampling_score_spearman": _spearman(coarse_scores, fine_scores),
        "aggregate_component_exposure_relative_l1": float(
            np.sum(np.abs(coarse_components - fine_components)) / denominator
        ),
    }


def proxy_domain_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize proxy packet mass that left the implemented model domain."""

    represented = sum(float(row["represented_fragment_count"]) for row in rows)
    out_of_domain = sum(float(row["out_of_domain_packet_weight"]) for row in rows)
    affected_cases = sum(
        float(row["out_of_domain_packet_weight"]) > 0.0 for row in rows
    )
    return {
        "case_count": len(rows),
        "affected_case_count": int(affected_cases),
        "represented_fragment_weight": represented,
        "out_of_domain_packet_weight": out_of_domain,
        "out_of_domain_packet_weight_fraction": (
            out_of_domain / represented if represented > 0.0 else None
        ),
        "domain_complete": out_of_domain == 0.0,
    }


def _lower_ci_meets(
    accuracy: dict[str, Any], metric: str, minimum: float
) -> bool:
    interval = accuracy.get(f"{metric}_ci95")
    return bool(
        accuracy.get(metric) is not None
        and interval is not None
        and len(interval) == 2
        and float(interval[0]) >= minimum
    )


def _gate_report(
    accuracy: dict[str, Any],
    convergence: dict[str, Any],
    speedup: float,
    domain_coverage: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "case_any_hit_auc": _lower_ci_meets(
            accuracy, "case_any_hit_auc", GATES["case_any_hit_auc_min"]
        ),
        "case_hit_spearman": _lower_ci_meets(
            accuracy, "case_hit_spearman", GATES["case_hit_spearman_min"]
        ),
        "case_energy_spearman": _lower_ci_meets(
            accuracy,
            "case_energy_spearman",
            GATES["case_energy_spearman_min"],
        ),
        "component_support_auc": _lower_ci_meets(
            accuracy,
            "component_support_auc",
            GATES["component_support_auc_min"],
        ),
        "component_support_recall": _lower_ci_meets(
            accuracy,
            "component_support_recall",
            GATES["component_support_recall_min"],
        ),
        "top20_energy_overlap": _lower_ci_meets(
            accuracy, "top20_energy_overlap", GATES["top20_overlap_min"]
        ),
        "speedup": speedup >= GATES["speedup_min"],
        "coarse_fine_rank": (
            convergence["sampling_score_spearman"] is not None
            and convergence["sampling_score_spearman"]
            >= GATES["coarse_fine_rank_min"]
        ),
        "coarse_fine_component_l1": (
            convergence["aggregate_component_exposure_relative_l1"]
            <= GATES["coarse_fine_component_l1_max"]
        ),
        "proxy_domain_complete": bool(domain_coverage["domain_complete"]),
    }
    return {
        "checks": checks,
        "passed_count": sum(bool(value) for value in checks.values()),
        "total_count": len(checks),
        "overall_pass": all(checks.values()),
        "statistical_gate_basis": "lower_95_percent_bootstrap_bound",
        "proxy_domain_coverage": domain_coverage,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal-input", type=Path, required=True)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--physics", type=Path, default=DEFAULT_PHYSICS)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--budget-per-cell", type=int, default=4)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--azimuths-per-ring", type=int, default=2)
    parser.add_argument("--convergence-azimuths-per-ring", type=int, default=4)
    parser.add_argument("--reference-azimuths-per-ring", type=int, default=8)
    parser.add_argument("--selection-seed", type=int, default=2026111402)
    parser.add_argument("--bootstrap-seed", type=int, default=2026111403)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.output_directory.exists():
        raise FileExistsError(
            f"output directory already exists; refusing to overwrite: {args.output_directory}"
        )
    if args.budget_per_cell < 1 or args.replicates < 1:
        raise ValueError("budgets and replicates must be positive")
    if not (
        0 < args.azimuths_per_ring
        < args.convergence_azimuths_per_ring
        < args.reference_azimuths_per_ring
    ):
        raise ValueError(
            "proxy azimuth resolutions must be positive and strictly increasing"
        )
    input_paths = {
        "proposal_input": args.proposal_input,
        "target": args.target,
        "physics": args.physics,
    }
    source_identity_at_start = _evaluation_source_identity(input_paths)
    args.output_directory.mkdir(parents=True)

    proposal = _load_json(args.proposal_input)
    candidate_ids = [str(value) for value in proposal["candidate_ids"]]
    cases = proposal["candidate_cases"]
    if len(candidate_ids) != len(cases) or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("proposal candidate IDs and cases are not one-to-one")
    target = _load_json(args.target)
    physics = _load_json(args.physics)
    engine = DamageEngine(target, PhysicsRegistry(physics))

    coarse_proxy = FastSamplingProxy(
        engine,
        azimuths_per_ring=args.azimuths_per_ring,
    )
    coarse_rows, coarse_seconds = _screen(cases, candidate_ids, coarse_proxy)
    _write_jsonl_gz(args.output_directory / "proxy_rows.jsonl.gz", coarse_rows)
    coarse_by_id = {row["candidate_id"]: row for row in coarse_rows}

    proxy_selected = _greedy_coverage_selection(
        coarse_rows,
        per_cell=args.budget_per_cell,
    )
    random_selected = _stratified_random_selection(
        coarse_rows,
        per_cell=args.budget_per_cell,
        seed=args.selection_seed,
    )
    selection_design = _selection_design_ledger(
        coarse_rows,
        greedy_ids=proxy_selected,
        random_ids=random_selected,
        per_cell=args.budget_per_cell,
    )
    validation_ids = sorted(set(proxy_selected) | set(random_selected))
    cases_by_id = {
        identifier: case
        for identifier, case in zip(candidate_ids, cases, strict=True)
    }
    selection_manifest = {
        "schema_version": "fast-sampling-selection@1.1.0",
        "status": "frozen_before_teacher_execution",
        "proposal_input_hash": source_identity_at_start[
            "proposal_input_file_hash"
        ],
        "candidate_count": len(candidate_ids),
        "cell_count": len({_case_cell(case) for case in cases}),
        "budget_per_cell": args.budget_per_cell,
        "proxy_greedy_candidate_ids": proxy_selected,
        "stratified_random_candidate_ids": random_selected,
        "validation_union_candidate_ids": validation_ids,
        "selection_design": selection_design,
        "selection_seed": args.selection_seed,
        "proxy_version": FAST_SAMPLING_PROXY_VERSION,
        "azimuths_per_ring": args.azimuths_per_ring,
        "convergence_azimuths_per_ring": args.convergence_azimuths_per_ring,
        "reference_azimuths_per_ring": args.reference_azimuths_per_ring,
        "engine_version": ENGINE_VERSION,
        "engine_code_hash": ENGINE_CODE_HASH,
        "source_identity_at_start": source_identity_at_start,
    }
    _write_json(args.output_directory / "selection_manifest.json", selection_manifest)
    _write_jsonl_gz(
        args.output_directory / "validation_cases_frozen.jsonl.gz",
        [
            {"candidate_id": identifier, "case": cases_by_id[identifier]}
            for identifier in validation_ids
        ],
    )

    fine_proxy = FastSamplingProxy(
        engine,
        azimuths_per_ring=args.convergence_azimuths_per_ring,
    )
    fine_cases = [cases_by_id[identifier] for identifier in validation_ids]
    fine_rows, fine_seconds = _screen(fine_cases, validation_ids, fine_proxy)
    _write_jsonl_gz(args.output_directory / "fine_proxy_rows.jsonl.gz", fine_rows)
    fine_by_id = {row["candidate_id"]: row for row in fine_rows}
    coarse_domain_coverage = proxy_domain_coverage(coarse_rows)
    fine_domain_coverage = proxy_domain_coverage(fine_rows)

    reference_proxy = FastSamplingProxy(
        engine,
        azimuths_per_ring=args.reference_azimuths_per_ring,
    )
    reference_rows, reference_seconds = _screen(
        fine_cases,
        validation_ids,
        reference_proxy,
    )
    _write_jsonl_gz(
        args.output_directory / "reference_proxy_rows.jsonl.gz",
        reference_rows,
    )
    reference_by_id = {row["candidate_id"]: row for row in reference_rows}
    reference_domain_coverage = proxy_domain_coverage(reference_rows)

    teacher_rows, full_results, teacher_seconds = _run_teacher(
        engine,
        cases_by_id,
        validation_ids,
        replicates=args.replicates,
    )
    _write_jsonl_gz(args.output_directory / "teacher_metrics.jsonl.gz", teacher_rows)
    _write_jsonl_gz(args.output_directory / "teacher_results.jsonl.gz", full_results)
    teacher_by_id = {row["candidate_id"]: row for row in teacher_rows}

    coarse_accuracy = _random_arm_accuracy_report(
        coarse_by_id,
        teacher_rows,
        random_selected,
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
    )
    fine_accuracy = _random_arm_accuracy_report(
        fine_by_id,
        teacher_rows,
        random_selected,
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed + 100,
    )
    coarse_fine_convergence = _convergence_report(
        {identifier: coarse_by_id[identifier] for identifier in random_selected},
        {identifier: fine_by_id[identifier] for identifier in random_selected},
        lower_resolution=args.azimuths_per_ring,
        higher_resolution=args.convergence_azimuths_per_ring,
    )
    fine_reference_convergence = _convergence_report(
        {identifier: fine_by_id[identifier] for identifier in random_selected},
        {
            identifier: reference_by_id[identifier]
            for identifier in random_selected
        },
        lower_resolution=args.convergence_azimuths_per_ring,
        higher_resolution=args.reference_azimuths_per_ring,
    )
    proxy_arm = _arm_report(proxy_selected, teacher_by_id)
    random_arm = _arm_report(random_selected, teacher_by_id)
    arm_difference = _cell_stratified_arm_difference_bootstrap(
        proxy_selected,
        random_selected,
        teacher_by_id,
        draws=args.bootstrap_draws,
        seed=args.bootstrap_seed + 200,
    )

    accepted_teacher = sum(row["status"] == "ok" for row in teacher_rows)
    teacher_per_case = teacher_seconds / max(1, len(validation_ids))
    coarse_per_case = coarse_seconds / max(1, len(candidate_ids))
    fine_per_case = fine_seconds / max(1, len(validation_ids))
    reference_per_case = reference_seconds / max(1, len(validation_ids))
    coarse_speedup = teacher_per_case / max(coarse_per_case, np.finfo(float).eps)
    fine_speedup = teacher_per_case / max(fine_per_case, np.finfo(float).eps)
    coarse_admission_domain = {
        **coarse_domain_coverage,
        "domain_complete": all(
            item["domain_complete"]
            for item in (
                coarse_domain_coverage,
                fine_domain_coverage,
                reference_domain_coverage,
            )
        ),
        "required_resolutions": [
            args.azimuths_per_ring,
            args.convergence_azimuths_per_ring,
            args.reference_azimuths_per_ring,
        ],
    }
    fine_admission_domain = {
        **fine_domain_coverage,
        "domain_complete": bool(
            fine_domain_coverage["domain_complete"]
            and reference_domain_coverage["domain_complete"]
        ),
        "required_resolutions": [
            args.convergence_azimuths_per_ring,
            args.reference_azimuths_per_ring,
        ],
    }
    coarse_gates = _gate_report(
        coarse_accuracy,
        coarse_fine_convergence,
        coarse_speedup,
        coarse_admission_domain,
    )
    fine_gates = _gate_report(
        fine_accuracy,
        fine_reference_convergence,
        fine_speedup,
        fine_admission_domain,
    )
    scheme_coverage_not_worse = (
        proxy_arm["component_coverage"]["ever_positive_count"]
        >= random_arm["component_coverage"]["ever_positive_count"]
        and proxy_arm["component_coverage"]["at_least_5_cases"]
        >= random_arm["component_coverage"]["at_least_5_cases"]
    )
    teacher_complete = accepted_teacher == len(validation_ids)
    replicate_admission_met = args.replicates >= MIN_ADMISSION_TEACHER_REPLICATES
    if not teacher_complete:
        recommendation = "diagnostic_only_incomplete_teacher_execution"
    elif not replicate_admission_met:
        recommendation = "diagnostic_only_teacher_replicates_below_admission_minimum"
    elif coarse_gates["overall_pass"] and scheme_coverage_not_worse:
        recommendation = "approve_coarse_proxy_for_train_candidate_screening"
    elif fine_gates["overall_pass"]:
        recommendation = (
            "rerun_full_candidate_frame_at_fine_resolution_before_sampling_approval"
        )
    else:
        recommendation = "restrict_proxy_to_geometry_diagnostic_and_do_not_approve_dataset"

    live_engine_code_hash = content_hash(
        {
            "python_source_tree_hash": source_tree_hash(ROOT / "damage_engine"),
            "schema_tree_hash": schema_tree_hash(),
        }
    )
    source_identity_at_end = _evaluation_source_identity(input_paths)
    source_stability_checks = {
        "engine_and_schema_tree": live_engine_code_hash == ENGINE_CODE_HASH,
        **{
            name: source_identity_at_end[name] == start_hash
            for name, start_hash in source_identity_at_start.items()
        },
    }
    source_stable_during_run = all(source_stability_checks.values())
    if not source_stable_during_run:
        recommendation = "discard_evidence_and_rerun_after_engine_source_is_frozen"

    report = {
        "schema_version": "fast-sampling-proxy-evaluation@1.1.0",
        "status": (
            "invalid_source_drift"
            if not source_stable_during_run
            else (
                "partial"
                if not teacher_complete
                else "complete"
                if replicate_admission_met
                else "complete_diagnostic"
            )
        ),
        "research_question": (
            "Can a weighted conical expected-transport proxy decide whether a "
            "candidate sampling plan deserves formal dataset generation?"
        ),
        "evidence_boundary": (
            f"Software-level comparison against the current unverified "
            f"DamageEngine {ENGINE_VERSION} ({ENGINE_CODE_HASH}). No high-fidelity "
            "or experimental physical admission is claimed."
        ),
        "inputs": {
            "proposal_input": str(args.proposal_input),
            "proposal_input_hash": source_identity_at_start[
                "proposal_input_file_hash"
            ],
            "target": str(args.target),
            "target_file_hash": source_identity_at_start["target_file_hash"],
            "normalized_target_hash": engine.target_hash,
            "physics": str(args.physics),
            "physics_file_hash": source_identity_at_start["physics_file_hash"],
            "registry_hash": engine.registry.hash,
            "engine_version": ENGINE_VERSION,
            "engine_code_hash": ENGINE_CODE_HASH,
            "engine_code_hash_at_end": live_engine_code_hash,
            "source_identity_at_start": source_identity_at_start,
            "source_identity_at_end": source_identity_at_end,
            "source_stability_checks": source_stability_checks,
            "source_stable_during_run": source_stable_during_run,
        },
        "design": {
            "candidate_count": len(candidate_ids),
            "cell_count": len({_case_cell(case) for case in cases}),
            "budget_per_cell": args.budget_per_cell,
            "proxy_arm_count": len(proxy_selected),
            "random_arm_count": len(random_selected),
            "validation_union_count": len(validation_ids),
            "teacher_replicates": args.replicates,
            "teacher_replicate_admission_minimum": (
                MIN_ADMISSION_TEACHER_REPLICATES
            ),
            "teacher_replicate_admission_met": replicate_admission_met,
            "bootstrap_draws": args.bootstrap_draws,
            "selection_frozen_before_teacher": True,
            "accuracy_evaluation_arm": "stratified_random_only",
            "random_exploration_per_cell": selection_design[
                "random_exploration_per_cell"
            ],
            "selection_positivity_min": selection_design["positivity_min"],
            "selection_design_ess": selection_design["selected_design_ess"],
            "selection_design_ess_fraction": selection_design[
                "selected_design_ess_fraction"
            ],
            "selection_ledger_hash": selection_design["candidate_ledger_hash"],
        },
        "timing": {
            "coarse_proxy_all_candidates_seconds": coarse_seconds,
            "fine_proxy_validation_seconds": fine_seconds,
            "reference_proxy_validation_seconds": reference_seconds,
            "teacher_validation_seconds": teacher_seconds,
            "coarse_proxy_seconds_per_case": coarse_per_case,
            "fine_proxy_seconds_per_case": fine_per_case,
            "reference_proxy_seconds_per_case": reference_per_case,
            "teacher_seconds_per_case": teacher_per_case,
            "coarse_speedup_vs_teacher": coarse_speedup,
            "fine_speedup_vs_teacher": fine_speedup,
            "projected_teacher_all_candidates_seconds": teacher_per_case
            * len(candidate_ids),
            "projected_fine_proxy_all_candidates_seconds": fine_per_case
            * len(candidate_ids),
        },
        "numerical_convergence": {
            "coarse_vs_fine": coarse_fine_convergence,
            "fine_vs_reference": fine_reference_convergence,
            "admission_rule": (
                "coarse uses coarse-vs-fine; fine uses fine-vs-reference"
            ),
        },
        "proxy_domain_coverage": {
            "coarse": coarse_domain_coverage,
            "fine": fine_domain_coverage,
            "reference": reference_domain_coverage,
        },
        "coarse_accuracy": coarse_accuracy,
        "fine_accuracy": fine_accuracy,
        "arms": {
            "proxy_greedy": proxy_arm,
            "stratified_random": random_arm,
            "cell_stratified_arm_difference": arm_difference,
            "proxy_coverage_not_worse": scheme_coverage_not_worse,
        },
        "pre_registered_gates": GATES,
        "coarse_gate_result": coarse_gates,
        "fine_gate_result": fine_gates,
        "recommendation": recommendation,
        "limitations": [
            "Candidate inputs are reused from an older outcome-blind LHS frame; labels are rerun on the current engine.",
            (
                f"The explicit DamageEngine {ENGINE_VERSION} is a software reference "
                "and is not physically calibrated."
            ),
            "Only one candidate frame and one fixed target are evaluated.",
            "The proxy omits ricochet, fragment breakup, spall and stateful multi-hit damage.",
            "A formal Reference/Validation/Test dataset was not generated or accessed.",
            *(
                [
                    "Teacher replication is below the predeclared admission minimum; "
                    "this run is diagnostic-only."
                ]
                if not replicate_admission_met
                else []
            ),
            *(
                [
                    "Engine, schema, proxy, evaluation-tool, or gate identity changed "
                    "during execution; the evidence is invalid for admission."
                ]
                if not source_stable_during_run
                else []
            ),
        ],
    }
    _write_json(args.output_directory / "evaluation.json", report)
    _write_json(
        args.output_directory / "completion.json",
        {
            "status": report["status"],
            "recommendation": recommendation,
            "evaluation_hash": _file_hash(args.output_directory / "evaluation.json"),
            "selection_manifest_hash": _file_hash(
                args.output_directory / "selection_manifest.json"
            ),
            "teacher_accepted_count": accepted_teacher,
            "teacher_domain_censored_count": sum(
                row["status"] == "domain_censored" for row in teacher_rows
            ),
            "teacher_requested_count": len(validation_ids),
        },
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "recommendation": recommendation,
                "coarse_speedup": coarse_speedup,
                "fine_speedup": fine_speedup,
                "coarse_gate_pass": coarse_gates["overall_pass"],
                "fine_gate_pass": fine_gates["overall_pass"],
                "proxy_coverage_not_worse": scheme_coverage_not_worse,
                "output_directory": str(args.output_directory),
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
