"""Evaluate the Idea-1 envelope-normalized sampling strategy.

The command has four reproducible stages:

``calibrate``
    Build a 2000-case pre-outcome frame and freeze U-band edges.
``proxy-trial``
    Compare fixed-size 2000-case strategies on one fresh candidate pool.
``teacher``
    Run the explicit software teacher on one frozen 2000-case strategy.
``aggregate``
    Combine proxy trials or two teacher arms into a compact evidence report.

All artifacts are diagnostic and live outside frozen formal datasets.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import kruskal, spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.case_factory import randomized_lhs
from damage_engine.contact_manifold_sampling import apply_contact_manifold
from damage_engine.engine import ENGINE_CODE_HASH, ENGINE_VERSION, DamageEngine
from damage_engine.errors import DamageEngineError
from damage_engine.fast_sampling_proxy import FastSamplingProxy
from damage_engine.munition_profiles import apply_munition_profile
from damage_engine.normalized_sampling import (
    ENVELOPE_BANDS,
    NORMALIZED_SAMPLING_VERSION,
    calibrate_envelope_policy,
    derive_normalized_envelope,
    design_weight_quality,
    envelope_band,
    fixed_size_inclusion_probabilities,
    global_marginal_coverage_probabilities,
    idea1_selection_score,
    pivotal_sample,
)
from damage_engine.registry import PhysicsRegistry
from damage_engine.sampling_risk_model import (
    SAMPLING_RISK_MODEL_VERSION,
    predict_sampling_risks,
)
from damage_engine.schema_resources import schema_tree_hash
from damage_engine.utils import content_hash, source_tree_hash


TOOL_VERSION = "idea1-normalized-sampling-evaluation@1.4.0"
DEFAULT_BASE_CASE = Path(
    "configs/cases/sampling_reference_r4.json"
)
DEFAULT_SPEC = Path(
    "configs/sampling/p0_contact_reference.json"
)
DEFAULT_TARGET = Path("configs/targets/target.fragment_only.v1.json")
DEFAULT_PHYSICS = Path("configs/physics/loitering_impact.json")
PROFILES = ("small_front", "medium_front", "large_front")
STRATEGIES = (
    "reference_uniform",
    "current_response_priority",
    "idea1_envelope_balanced",
    "idea1_global_coverage_balanced",
    "idea1_coverage_information_balanced",
    "idea1_hurdle_coverage_balanced",
    "idea1_hurdle_strong_balanced",
)
IDEA_STRATEGIES = (
    "idea1_global_coverage_balanced",
    "idea1_coverage_information_balanced",
    "idea1_hurdle_coverage_balanced",
    "idea1_hurdle_strong_balanced",
)
V3_AUXILIARY_FEATURES = ("proxy_tree_max", "u_penetration")
V3_AUXILIARY_WEIGHTS = (-0.40, -0.40)
V4_AUXILIARY_FEATURES = (
    "predicted_domain_complete_probability",
    "predicted_tree_saturation_given_complete_probability",
)
V4_AUXILIARY_WEIGHTS = (0.20, -0.20)
V5_AUXILIARY_WEIGHTS = (0.80, -0.80)
TEACHER_COMPONENT_SUPPORT_RATE_THRESHOLD = 0.05


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
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
    temporary.replace(path)


def _iter_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _source_identity(paths: Mapping[str, Path]) -> dict[str, str]:
    result = {
        "tool_hash": _file_hash(Path(__file__).resolve()),
        "normalized_sampling_hash": _file_hash(
            ROOT / "damage_engine/normalized_sampling.py"
        ),
        "engine_code_hash": ENGINE_CODE_HASH,
    }
    for name, path in sorted(paths.items()):
        result[f"{name}_hash"] = _file_hash(path.resolve())
    return result


def _live_engine_code_hash() -> str:
    return content_hash(
        {
            "python_source_tree_hash": source_tree_hash(ROOT / "damage_engine"),
            "schema_tree_hash": schema_tree_hash(),
        }
    )


def _balanced_counts(total: int, cells: Sequence[str]) -> dict[str, int]:
    if total < len(cells):
        raise ValueError("total must cover every cell")
    base, remainder = divmod(total, len(cells))
    return {
        cell: base + int(index < remainder)
        for index, cell in enumerate(sorted(cells))
    }


def _cell_names(manifold: Mapping[str, Any]) -> list[str]:
    return [
        f"{approach['approach_id']}|{profile}"
        for approach in manifold["approaches"]
        for profile in PROFILES
    ]


def _build_candidate_frame(
    *,
    candidate_count: int,
    design_seed: int,
    base_case: Mapping[str, Any],
    manifold: Mapping[str, Any],
    replicates: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    cells = _cell_names(manifold)
    counts = _balanced_counts(candidate_count, cells)
    candidate_ids: list[str] = []
    cases: list[dict[str, Any]] = []
    serial = 0
    for approach_index, approach in enumerate(manifold["approaches"]):
        for profile_index, profile in enumerate(PROFILES):
            cell = f"{approach['approach_id']}|{profile}"
            count = counts[cell]
            cell_seed = int(
                hashlib.sha256(
                    f"{design_seed}|{approach_index}|{profile_index}".encode("utf-8")
                ).hexdigest()[:15],
                16,
            )
            continuous = randomized_lhs(count, 5, seed=cell_seed)
            for local_index, row in enumerate(continuous):
                selector = (approach_index + 0.5) / len(manifold["approaches"])
                latent = np.concatenate([[selector], row])
                token = hashlib.sha256(
                    (
                        f"idea1|seed={design_seed}|cell={cell}|index={local_index}|"
                        f"latent={latent.tolist()}"
                    ).encode("utf-8")
                ).hexdigest()
                candidate_id = f"idea1-{design_seed}-{serial:05d}-{token[:12]}"
                case = deepcopy(base_case)
                case["case_id"] = f"idea1-case-{design_seed}-{serial:05d}-{token[:12]}"
                apply_munition_profile(case, profile)
                record = apply_contact_manifold(case, manifold, latent)
                analysis = case["analysis"]
                analysis["replicates"] = int(replicates)
                analysis["aleatory_seed"] = int(token[:15], 16)
                analysis["adaptive_ci_half_width"] = None
                analysis["capture_replicate_trace"] = False
                analysis["capture_fragment_audit"] = False
                metadata = case.setdefault("metadata", {})
                for reserved in ("sampling", "frozen_split", "split_registry_hash"):
                    metadata.pop(reserved, None)
                metadata["idea1_sampling"] = {
                    "schema_version": TOOL_VERSION,
                    "candidate_id": candidate_id,
                    "design_seed": int(design_seed),
                    "approach_profile_cell": cell,
                    "approach_id": record["approach_id"],
                    "profile_id": profile,
                    "candidate_serial": serial,
                    "pre_outcome": True,
                    "training_eligible": False,
                    "physical_admission": False,
                }
                candidate_ids.append(candidate_id)
                cases.append(case)
                serial += 1
    if len(candidate_ids) != candidate_count or len(set(candidate_ids)) != candidate_count:
        raise RuntimeError("candidate frame identity construction failed")
    return candidate_ids, cases


def _case_cell(case: Mapping[str, Any]) -> str:
    return str(case["metadata"]["idea1_sampling"]["approach_profile_cell"])


def _screen_frame(
    *,
    engine: DamageEngine,
    proxy: FastSamplingProxy,
    candidate_ids: Sequence[str],
    cases: Sequence[dict[str, Any]],
    calibration: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], float]:
    rows: list[dict[str, Any]] = []
    prepared_cases: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []
    started = perf_counter()
    for index, (candidate_id, case) in enumerate(zip(candidate_ids, cases, strict=True)):
        case_started = perf_counter()
        try:
            prepared = engine.prepare_case(case)
            result = proxy.evaluate(prepared)
            envelope = derive_normalized_envelope(result)
            result.update(
                {
                    "candidate_id": candidate_id,
                    "approach_profile_cell": _case_cell(case),
                    "screen_elapsed_seconds": perf_counter() - case_started,
                    "idea1_envelope": envelope,
                    "eligible_model_domain": (
                        result["proxy_domain_status"] == "complete_proxy_domain"
                    ),
                }
            )
            if calibration is not None:
                result["idea1_envelope_band"] = envelope_band(envelope, calibration)
                result["idea1_score"] = idea1_selection_score(envelope, calibration)
            rows.append(result)
            prepared_cases.append(prepared)
        except DamageEngineError as error:
            refusals.append(
                {
                    "candidate_id": candidate_id,
                    "case_id": case.get("case_id"),
                    "approach_profile_cell": _case_cell(case),
                    "status": "refused_before_proxy_label",
                    "error": error.to_dict(),
                    "elapsed_seconds": perf_counter() - case_started,
                }
            )
        if (index + 1) % 250 == 0 or index + 1 == len(candidate_ids):
            print(
                json.dumps(
                    {
                        "stage": "proxy_screen",
                        "completed": index + 1,
                        "total": len(candidate_ids),
                        "accepted": len(rows),
                        "refused": len(refusals),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return rows, prepared_cases, refusals, perf_counter() - started


def _allocate_counts(
    capacities: Mapping[str, int],
    *,
    total: int,
    weights: Mapping[str, float] | None = None,
    minimum_one: bool = True,
) -> dict[str, int]:
    keys = [key for key in sorted(capacities) if capacities[key] > 0]
    if total < 0 or total > sum(capacities[key] for key in keys):
        raise ValueError("allocation total exceeds capacity")
    if minimum_one and total < len(keys):
        raise ValueError("allocation cannot give every nonempty stratum positive support")
    result = {key: 0 for key in capacities}
    if minimum_one:
        for key in keys:
            result[key] = 1
    remaining = total - sum(result.values())
    raw_weights = {
        key: float((weights or {}).get(key, 1.0))
        for key in keys
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in raw_weights.values()):
        raise ValueError("allocation weights must be finite and positive")
    while remaining > 0:
        active = [key for key in keys if result[key] < capacities[key]]
        if not active:
            raise RuntimeError("allocation exhausted capacity")
        weight_sum = sum(raw_weights[key] for key in active)
        ideal = {
            key: remaining * raw_weights[key] / weight_sum for key in active
        }
        additions = {
            key: min(
                capacities[key] - result[key],
                int(math.floor(ideal[key])),
            )
            for key in active
        }
        added = sum(additions.values())
        if added == 0:
            winner = max(
                active,
                key=lambda key: (
                    ideal[key] - math.floor(ideal[key]),
                    capacities[key] - result[key],
                    key,
                ),
            )
            additions[winner] = 1
            added = 1
        for key, value in additions.items():
            result[key] += value
        remaining -= added
    return result


def _select_stratum(
    rows: Sequence[dict[str, Any]],
    *,
    count: int,
    scores: Sequence[float],
    seed: int,
) -> tuple[set[str], dict[str, float]]:
    if count == 0:
        return set(), {}
    probabilities = fixed_size_inclusion_probabilities(
        scores,
        sample_size=count,
        rho_min=0.05,
    )
    selected = pivotal_sample(probabilities, seed=seed)
    identifiers = [str(row["candidate_id"]) for row in rows]
    selected_ids = {
        identifier for identifier, keep in zip(identifiers, selected, strict=True) if keep
    }
    return selected_ids, {
        identifier: float(probability)
        for identifier, probability in zip(identifiers, probabilities, strict=True)
    }


def _strategy_selection(
    rows: Sequence[dict[str, Any]],
    *,
    strategy: str,
    sample_size: int,
    seed: int,
    calibration: Mapping[str, Any],
    risk_model: Mapping[str, Any] | None = None,
) -> tuple[set[str], dict[str, float], dict[str, Any]]:
    if strategy not in STRATEGIES:
        raise ValueError(f"unsupported strategy: {strategy}")
    eligible = [row for row in rows if row["eligible_model_domain"]]
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_cell[str(row["approach_profile_cell"])].append(row)
    cell_allocations = _allocate_counts(
        {cell: len(values) for cell, values in by_cell.items()},
        total=sample_size,
        weights={cell: 1.0 for cell in by_cell},
        minimum_one=True,
    )
    selected_ids: set[str] = set()
    probabilities: dict[str, float] = {}
    band_allocations: dict[str, dict[str, int]] = {}

    if strategy in IDEA_STRATEGIES:
        row_index = {
            str(row["candidate_id"]): index for index, row in enumerate(eligible)
        }
        strata: list[np.ndarray] = []
        quotas: list[int] = []
        stratum_labels: list[str] = []
        for cell in sorted(by_cell):
            cell_rows = sorted(
                by_cell[cell], key=lambda row: str(row["candidate_id"])
            )
            by_band: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in cell_rows:
                by_band[str(row["idea1_envelope_band"])].append(row)
            target_fractions = calibration["target_band_fractions"]
            allocations = _allocate_counts(
                {band: len(by_band.get(band, [])) for band in ENVELOPE_BANDS},
                total=cell_allocations[cell],
                weights={
                    band: float(target_fractions[band])
                    for band in ENVELOPE_BANDS
                },
                minimum_one=True,
            )
            band_allocations[cell] = allocations
            for band in ENVELOPE_BANDS:
                band_rows = sorted(
                    by_band.get(band, []),
                    key=lambda row: str(row["candidate_id"]),
                )
                if not band_rows:
                    continue
                strata.append(
                    np.asarray(
                        [row_index[str(row["candidate_id"])] for row in band_rows],
                        dtype=np.int64,
                    )
                )
                quotas.append(int(allocations.get(band, 0)))
                stratum_labels.append(f"{cell}|{band}")

        component_ids = eligible[0]["idea1_envelope"]["component_ids"]
        support_matrix = np.asarray(
            [
                [
                    row["idea1_envelope"]["component_support_indicator"][
                        identifier
                    ]
                    for identifier in component_ids
                ]
                for row in eligible
            ],
            dtype=np.float64,
        )
        auxiliary_features = None
        auxiliary_weights = None
        if strategy == "idea1_coverage_information_balanced":
            auxiliary_features = np.asarray(
                [
                    [
                        max(
                            map(float, row["tree_outputs"].values()),
                            default=0.0,
                        ),
                        float(row["idea1_envelope"]["u_penetration"]),
                    ]
                    for row in eligible
                ],
                dtype=np.float64,
            )
            auxiliary_weights = np.asarray(
                V3_AUXILIARY_WEIGHTS, dtype=np.float64
            )
        elif strategy in {
            "idea1_hurdle_coverage_balanced",
            "idea1_hurdle_strong_balanced",
        }:
            if risk_model is None:
                raise ValueError(
                    "hurdle-balanced strategies require a frozen risk model"
                )
            risks = predict_sampling_risks(eligible, risk_model)
            auxiliary_features = np.column_stack(
                [
                    risks["domain_complete_probability"],
                    risks["tree_saturation_given_complete_probability"],
                ]
            )
            auxiliary_weights = np.asarray(
                (
                    V5_AUXILIARY_WEIGHTS
                    if strategy == "idea1_hurdle_strong_balanced"
                    else V4_AUXILIARY_WEIGHTS
                ),
                dtype=np.float64,
            )
        probability_array = global_marginal_coverage_probabilities(
            support_matrix,
            strata=strata,
            quotas=quotas,
            auxiliary_features=auxiliary_features,
            auxiliary_weights=auxiliary_weights,
            smoothing=5.0,
            score_exponent=1.0,
            iterations=80,
            damping=0.2,
            rho_min=0.08,
        )
        baseline_probability = np.zeros(len(eligible), dtype=np.float64)
        for group_offset, (indices, quota) in enumerate(
            zip(strata, quotas, strict=True)
        ):
            baseline_probability[indices] = float(quota) / float(len(indices))
            local_selected = pivotal_sample(
                probability_array[indices],
                seed=seed + 1009 * group_offset,
            )
            for index, keep in zip(indices, local_selected, strict=True):
                if keep:
                    selected_ids.add(str(eligible[int(index)]["candidate_id"]))
        probabilities = {
            str(row["candidate_id"]): float(probability_array[index])
            for index, row in enumerate(eligible)
        }
        if len(selected_ids) != sample_size:
            raise RuntimeError(
                f"{strategy} selected {len(selected_ids)} cases instead of {sample_size}"
            )
        baseline_counts = support_matrix.T @ baseline_probability
        optimized_counts = support_matrix.T @ probability_array
        optimizer_summary = {
            "policy": "sum_k_X_ik_over_expected_count_plus_5",
            "smoothing": 5.0,
            "score_exponent": 1.0,
            "iterations": 80,
            "damping": 0.2,
            "rho_min": 0.08,
            "auxiliary_features": (
                []
                if auxiliary_features is None
                else (
                    list(V4_AUXILIARY_FEATURES)
                    if strategy in {
                        "idea1_hurdle_coverage_balanced",
                        "idea1_hurdle_strong_balanced",
                    }
                    else list(V3_AUXILIARY_FEATURES)
                )
            ),
            "auxiliary_weights": (
                []
                if auxiliary_weights is None
                else [float(value) for value in auxiliary_weights]
            ),
            "stratum_count": len(strata),
            "stratum_labels": stratum_labels,
            "baseline_expected_component_count_min": float(
                np.min(baseline_counts)
            ),
            "optimized_expected_component_count_min": float(
                np.min(optimized_counts)
            ),
            "baseline_expected_component_count_gini": _gini(
                baseline_counts
            ),
            "optimized_expected_component_count_gini": _gini(
                optimized_counts
            ),
        }
        if strategy in {
            "idea1_hurdle_coverage_balanced",
            "idea1_hurdle_strong_balanced",
        }:
            selected_mask = np.asarray(
                [
                    str(row["candidate_id"]) in selected_ids
                    for row in eligible
                ],
                dtype=bool,
            )
            domain_probability = auxiliary_features[:, 0]
            conditional_tree_probability = auxiliary_features[:, 1]
            expected_complete = float(np.sum(domain_probability[selected_mask]))
            optimizer_summary.update(
                {
                    "risk_model_schema_version": risk_model.get(
                        "schema_version"
                    ),
                    "selected_predicted_complete_domain_fraction": (
                        expected_complete / float(np.count_nonzero(selected_mask))
                    ),
                    "selected_predicted_tree_saturation_given_complete": float(
                        np.sum(
                            domain_probability[selected_mask]
                            * conditional_tree_probability[selected_mask]
                        )
                        / expected_complete
                    ),
                }
            )
        return selected_ids, probabilities, {
            "cell_allocations": cell_allocations,
            "band_allocations": band_allocations,
            "global_coverage_optimizer": optimizer_summary,
        }

    for cell_offset, cell in enumerate(sorted(by_cell)):
        cell_rows = sorted(by_cell[cell], key=lambda row: str(row["candidate_id"]))
        cell_count = cell_allocations[cell]
        if strategy != "idea1_envelope_balanced":
            scores = (
                np.ones(len(cell_rows), dtype=np.float64)
                if strategy == "reference_uniform"
                else np.asarray(
                    [float(row["sampling_score"]) for row in cell_rows],
                    dtype=np.float64,
                )
            )
            picked, rho = _select_stratum(
                cell_rows,
                count=cell_count,
                scores=scores,
                seed=seed + 1009 * cell_offset,
            )
            selected_ids.update(picked)
            probabilities.update(rho)
            continue

        by_band: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in cell_rows:
            by_band[str(row["idea1_envelope_band"])].append(row)
        target_fractions = calibration["target_band_fractions"]
        allocations = _allocate_counts(
            {band: len(by_band.get(band, [])) for band in ENVELOPE_BANDS},
            total=cell_count,
            weights={band: float(target_fractions[band]) for band in ENVELOPE_BANDS},
            minimum_one=True,
        )
        band_allocations[cell] = allocations
        for band_offset, band in enumerate(ENVELOPE_BANDS):
            band_rows = sorted(
                by_band.get(band, []), key=lambda row: str(row["candidate_id"])
            )
            count = allocations.get(band, 0)
            if not band_rows:
                continue
            scores = [float(row["idea1_score"]["total"]) for row in band_rows]
            picked, rho = _select_stratum(
                band_rows,
                count=count,
                scores=scores,
                seed=seed + 1009 * cell_offset + 97 * (band_offset + 1),
            )
            selected_ids.update(picked)
            probabilities.update(rho)
    if len(selected_ids) != sample_size:
        raise RuntimeError(
            f"{strategy} selected {len(selected_ids)} cases instead of {sample_size}"
        )
    return selected_ids, probabilities, {
        "cell_allocations": cell_allocations,
        "band_allocations": band_allocations,
    }


def _gini(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or float(np.sum(array)) <= 0.0:
        return 0.0
    ordered = np.sort(array)
    index = np.arange(1, ordered.size + 1, dtype=np.float64)
    return float(
        (2.0 * np.sum(index * ordered) / (ordered.size * np.sum(ordered)))
        - (ordered.size + 1.0) / ordered.size
    )


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        str(quantile): float(np.quantile(array, quantile))
        for quantile in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    }


def _learner_features(case: Mapping[str, Any]) -> np.ndarray:
    manifold = case["metadata"]["contact_manifold_sampling"]
    approach = int(manifold["approach_index"])
    profile = str(case["munition"]["profile_id"])
    latent = np.asarray(manifold["latent"], dtype=np.float64)
    roll = float(manifold["body_x_roll_rad"])
    return np.asarray(
        [float(profile == value) for value in PROFILES]
        + [float(approach == value) for value in range(6)]
        + [
            float(latent[1]),
            float(latent[2]),
            float(latent[3]),
            float(latent[4]),
            float(math.sin(roll)),
            float(math.cos(roll)),
        ],
        dtype=np.float64,
    )


def _strategy_summary(
    *,
    strategy: str,
    selected_rows: Sequence[dict[str, Any]],
    selected_cases: Sequence[dict[str, Any]],
    probabilities: Mapping[str, float],
    risk_model: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    count = len(selected_rows)
    if count != len(selected_cases):
        raise RuntimeError("selected proxy rows and cases are not aligned")
    component_ids = selected_rows[0]["idea1_envelope"]["component_ids"]
    support_counts = {
        identifier: int(
            sum(
                row["idea1_envelope"]["component_support_indicator"][identifier]
                for row in selected_rows
            )
        )
        for identifier in component_ids
    }
    tree_names = sorted(selected_rows[0]["tree_outputs"])
    tree_counts = {
        name: int(sum(float(row["tree_outputs"][name]) >= 0.05 for row in selected_rows))
        for name in tree_names
    }
    rho = np.asarray(
        [probabilities[str(row["candidate_id"])] for row in selected_rows],
        dtype=np.float64,
    )
    selected_mask = np.ones(count, dtype=bool)
    weight_quality = design_weight_quality(rho, selected_mask)
    features = np.vstack([_learner_features(case) for case in selected_cases])
    scaled = StandardScaler().fit_transform(features)
    distances = NearestNeighbors(n_neighbors=2).fit(scaled).kneighbors(
        scaled, return_distance=True
    )[0][:, 1]
    u_pen = [float(row["idea1_envelope"]["u_penetration"]) for row in selected_rows]
    severity = [float(row["idea1_envelope"]["damage_severity"]) for row in selected_rows]
    utilization = [
        float(row["idea1_envelope"]["normalized_utilization"])
        for row in selected_rows
    ]
    tree_max = [max(map(float, row["tree_outputs"].values()), default=0.0) for row in selected_rows]
    band_counts = Counter(str(row["idea1_envelope_band"]) for row in selected_rows)
    cell_counts = Counter(str(row["approach_profile_cell"]) for row in selected_rows)
    support_array = np.asarray(list(support_counts.values()), dtype=np.float64)
    support_rates = support_array / count
    entropy = [
        0.0
        if value <= 0.0 or value >= 1.0
        else -value * math.log(value) - (1.0 - value) * math.log(1.0 - value)
        for value in support_rates
    ]
    summary = {
        "schema_version": TOOL_VERSION,
        "strategy": strategy,
        "selected_count": count,
        "unique_case_count": len({case["case_id"] for case in selected_cases}),
        "model_domain_complete_fraction": float(
            np.mean([row["eligible_model_domain"] for row in selected_rows])
        ),
        "approach_profile_cell_counts": dict(sorted(cell_counts.items())),
        "approach_profile_cell_min": min(cell_counts.values()),
        "approach_profile_cell_max": max(cell_counts.values()),
        "envelope_band_counts": {
            band: int(band_counts.get(band, 0)) for band in ENVELOPE_BANDS
        },
        "envelope_band_fractions": {
            band: float(band_counts.get(band, 0) / count) for band in ENVELOPE_BANDS
        },
        "u_penetration_quantiles": _quantiles(u_pen),
        "damage_severity_quantiles": _quantiles(severity),
        "normalized_utilization_quantiles": _quantiles(utilization),
        "zero_u_fraction": float(np.mean(np.asarray(u_pen) <= 1.0e-12)),
        "zero_severity_fraction": float(np.mean(np.asarray(severity) <= 1.0e-12)),
        "utilization_saturation_fraction": float(
            np.mean(np.asarray(utilization) >= 0.95)
        ),
        "tree_saturation_fraction": float(np.mean(np.asarray(tree_max) >= 0.95)),
        "component_support_counts": support_counts,
        "component_coverage": {
            "ever": int(np.count_nonzero(support_array >= 1)),
            "at_least_5": int(np.count_nonzero(support_array >= 5)),
            "at_least_10": int(np.count_nonzero(support_array >= 10)),
            "at_least_20": int(np.count_nonzero(support_array >= 20)),
            "at_least_50": int(np.count_nonzero(support_array >= 50)),
            "minimum": int(np.min(support_array)),
            "median": float(np.median(support_array)),
            "maximum": int(np.max(support_array)),
            "gini": _gini(support_array),
            "mean_binary_entropy_nats": float(np.mean(entropy)),
        },
        "tree_support_counts": tree_counts,
        "any_component_support_fraction": float(
            np.mean(
                [row["idea1_envelope"]["supported_component_count"] > 0 for row in selected_rows]
            )
        ),
        "learner_space_nearest_neighbor_distance": {
            "median": float(np.median(distances)),
            "p10": float(np.quantile(distances, 0.10)),
            "minimum": float(np.min(distances)),
        },
        "design_weight_quality": weight_quality,
    }
    if risk_model is not None:
        risks = predict_sampling_risks(selected_rows, risk_model)
        domain_probability = risks["domain_complete_probability"]
        conditional_tree_probability = risks[
            "tree_saturation_given_complete_probability"
        ]
        expected_complete = float(np.sum(domain_probability))
        summary["sampling_risk_prediction"] = {
            "model_schema_version": risk_model["schema_version"],
            "complete_domain_fraction": float(np.mean(domain_probability)),
            "tree_saturation_given_complete": float(
                np.sum(domain_probability * conditional_tree_probability)
                / expected_complete
            ),
        }
    return summary


def _resolve_inputs(args: argparse.Namespace) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any], DamageEngine]:
    paths = {
        "base_case": args.base_case.resolve(),
        "sampling_spec": args.sampling_spec.resolve(),
        "target": args.target.resolve(),
        "physics": args.physics.resolve(),
    }
    base_case = _load_json(paths["base_case"])
    sampling_spec = _load_json(paths["sampling_spec"])
    target = _load_json(paths["target"])
    physics = _load_json(paths["physics"])
    engine = DamageEngine(target, PhysicsRegistry(physics))
    return paths, base_case, sampling_spec, engine


def _assert_new_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {path}")
    path.mkdir(parents=True)


def _run_calibration(args: argparse.Namespace) -> int:
    _assert_new_directory(args.output_directory)
    paths, base_case, sampling_spec, engine = _resolve_inputs(args)
    source_at_start = _source_identity(paths)
    candidate_ids, cases = _build_candidate_frame(
        candidate_count=args.candidate_count,
        design_seed=args.design_seed,
        base_case=base_case,
        manifold=sampling_spec["contact_manifold"],
        replicates=args.replicates,
    )
    proxy = FastSamplingProxy(
        engine,
        azimuths_per_ring=args.azimuths_per_ring,
    )
    rows, prepared_cases, refusals, seconds = _screen_frame(
        engine=engine,
        proxy=proxy,
        candidate_ids=candidate_ids,
        cases=cases,
        calibration=None,
    )
    calibration_rows = [
        row for row in rows if row["eligible_model_domain"]
    ]
    if len(calibration_rows) < 2:
        raise RuntimeError("calibration frame has insufficient complete-domain rows")
    calibration = calibrate_envelope_policy(
        [row["idea1_envelope"] for row in calibration_rows]
    )
    for row in rows:
        row["idea1_envelope_band"] = envelope_band(
            row["idea1_envelope"], calibration
        )
        row["idea1_score"] = idea1_selection_score(
            row["idea1_envelope"], calibration
        )
    source_at_end = _source_identity(paths)
    live_hash = _live_engine_code_hash()
    stable = source_at_start == source_at_end and live_hash == ENGINE_CODE_HASH
    _write_json(args.output_directory / "calibration.json", calibration)
    _write_jsonl_gz(args.output_directory / "proxy_rows.jsonl.gz", rows)
    _write_jsonl_gz(args.output_directory / "candidate_cases.jsonl.gz", prepared_cases)
    _write_jsonl_gz(args.output_directory / "refusal_ledger.jsonl.gz", refusals)
    manifest = {
        "schema_version": TOOL_VERSION,
        "stage": "calibration",
        "status": "complete" if stable else "invalid_source_drift",
        "candidate_count_requested": args.candidate_count,
        "proxy_result_count": len(rows),
        "complete_domain_calibration_count": len(calibration_rows),
        "refusal_count": len(refusals),
        "design_seed": args.design_seed,
        "azimuths_per_ring": args.azimuths_per_ring,
        "replicates_stored_for_future_teacher": args.replicates,
        "elapsed_seconds": seconds,
        "engine_version": ENGINE_VERSION,
        "engine_code_hash": ENGINE_CODE_HASH,
        "live_engine_code_hash_at_end": live_hash,
        "source_identity_at_start": source_at_start,
        "source_identity_at_end": source_at_end,
        "source_stable_during_run": stable,
        "calibration_hash": content_hash(calibration),
        "evidence_scope": (
            "pre-outcome proxy calibration only; no formal labels or physical "
            "admission"
        ),
    }
    _write_json(args.output_directory / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    return 0 if stable else 2


def _run_proxy_trial(args: argparse.Namespace) -> int:
    _assert_new_directory(args.output_directory)
    paths, base_case, sampling_spec, engine = _resolve_inputs(args)
    calibration_path = args.calibration.resolve()
    calibration = _load_json(calibration_path)
    risk_model = None
    risk_model_path = None
    if args.risk_model is not None:
        risk_model_path = args.risk_model.resolve()
        risk_model = _load_json(risk_model_path)
        if risk_model.get("schema_version") != SAMPLING_RISK_MODEL_VERSION:
            raise ValueError("unsupported frozen sampling-risk model")
    if risk_model is None:
        raise ValueError(
            "proxy-trial requires --risk-model because the V4 strategy is active"
        )
    source_paths = {
        **paths,
        "calibration": calibration_path,
        "risk_model": risk_model_path,
    }
    source_at_start = _source_identity(source_paths)
    candidate_ids, cases = _build_candidate_frame(
        candidate_count=args.candidate_count,
        design_seed=args.design_seed,
        base_case=base_case,
        manifold=sampling_spec["contact_manifold"],
        replicates=args.replicates,
    )
    proxy = FastSamplingProxy(
        engine,
        azimuths_per_ring=args.azimuths_per_ring,
    )
    rows, prepared_cases, refusals, seconds = _screen_frame(
        engine=engine,
        proxy=proxy,
        candidate_ids=candidate_ids,
        cases=cases,
        calibration=calibration,
    )
    eligible = [row for row in rows if row["eligible_model_domain"]]
    if len(eligible) < args.sample_size:
        raise RuntimeError(
            f"only {len(eligible)} complete-domain candidates; expand the frame "
            f"to select {args.sample_size}"
        )
    case_by_id = {
        str(row["candidate_id"]): case
        for row, case in zip(rows, prepared_cases, strict=True)
    }
    row_by_id = {str(row["candidate_id"]): row for row in rows}
    strategy_summaries: dict[str, dict[str, Any]] = {}
    for strategy_index, strategy in enumerate(STRATEGIES):
        selection_seed = args.selection_seed + 100_003 * strategy_index
        selected_ids, probabilities, allocations = _strategy_selection(
            rows,
            strategy=strategy,
            sample_size=args.sample_size,
            seed=selection_seed,
            calibration=calibration,
            risk_model=risk_model,
        )
        selected_rows = [
            row_by_id[identifier] for identifier in sorted(selected_ids)
        ]
        selected_cases: list[dict[str, Any]] = []
        for identifier in sorted(selected_ids):
            case = deepcopy(case_by_id[identifier])
            row = row_by_id[identifier]
            case.setdefault("metadata", {})["idea1_selection"] = {
                "schema_version": TOOL_VERSION,
                "strategy": strategy,
                "selection_seed": selection_seed,
                "candidate_id": identifier,
                "approach_profile_cell": row["approach_profile_cell"],
                "envelope_band": row["idea1_envelope_band"],
                "inclusion_probability": probabilities[identifier],
                "design_weight_raw": 1.0 / probabilities[identifier],
                "proxy_domain_status": row["proxy_domain_status"],
                "training_eligible": False,
                "physical_admission": False,
            }
            if strategy in {
                "idea1_hurdle_coverage_balanced",
                "idea1_hurdle_strong_balanced",
            }:
                predicted = predict_sampling_risks([row], risk_model)
                case["metadata"]["idea1_selection"]["sampling_risk"] = {
                    "schema_version": risk_model["schema_version"],
                    "domain_complete_probability": float(
                        predicted["domain_complete_probability"][0]
                    ),
                    "tree_saturation_given_complete_probability": float(
                        predicted[
                            "tree_saturation_given_complete_probability"
                        ][0]
                    ),
                }
            selected_cases.append(case)
        summary = _strategy_summary(
            strategy=strategy,
            selected_rows=selected_rows,
            selected_cases=selected_cases,
            probabilities=probabilities,
            risk_model=risk_model,
        )
        summary["selection_seed"] = selection_seed
        summary["allocations"] = allocations
        strategy_summaries[strategy] = summary
        strategy_root = args.output_directory / strategy
        _write_json(strategy_root / "summary.json", summary)
        _write_jsonl_gz(
            strategy_root / "selected_proxy_rows.jsonl.gz", selected_rows
        )
        _write_jsonl_gz(
            strategy_root / "selected_cases.jsonl.gz", selected_cases
        )
        ledger = []
        for row in rows:
            identifier = str(row["candidate_id"])
            probability = probabilities.get(identifier, 0.0)
            ledger.append(
                {
                    "candidate_id": identifier,
                    "approach_profile_cell": row["approach_profile_cell"],
                    "envelope_band": row["idea1_envelope_band"],
                    "eligible_model_domain": row["eligible_model_domain"],
                    "inclusion_probability": probability,
                    "design_weight_raw": (
                        None if probability <= 0.0 else 1.0 / probability
                    ),
                    "selected": identifier in selected_ids,
                }
            )
        _write_jsonl_gz(strategy_root / "selection_ledger.jsonl.gz", ledger)

    _write_jsonl_gz(args.output_directory / "proxy_rows.jsonl.gz", rows)
    _write_jsonl_gz(args.output_directory / "refusal_ledger.jsonl.gz", refusals)
    source_at_end = _source_identity(source_paths)
    live_hash = _live_engine_code_hash()
    stable = source_at_start == source_at_end and live_hash == ENGINE_CODE_HASH
    trial_summary = {
        "schema_version": TOOL_VERSION,
        "stage": "proxy_trial",
        "status": "complete" if stable else "invalid_source_drift",
        "design_seed": args.design_seed,
        "selection_seed": args.selection_seed,
        "candidate_count_requested": args.candidate_count,
        "proxy_result_count": len(rows),
        "complete_domain_candidate_count": len(eligible),
        "partial_domain_candidate_count": int(
            sum(not row["eligible_model_domain"] for row in rows)
        ),
        "refusal_count": len(refusals),
        "sample_size_per_strategy": args.sample_size,
        "azimuths_per_ring": args.azimuths_per_ring,
        "elapsed_seconds": seconds,
        "strategies": strategy_summaries,
        "calibration_hash": content_hash(calibration),
        "risk_model_hash": content_hash(risk_model),
        "engine_version": ENGINE_VERSION,
        "engine_code_hash": ENGINE_CODE_HASH,
        "live_engine_code_hash_at_end": live_hash,
        "source_identity_at_start": source_at_start,
        "source_identity_at_end": source_at_end,
        "source_stable_during_run": stable,
        "evidence_scope": (
            "equal-budget pre-outcome proxy sampling comparisons; no formal "
            "teacher labels"
        ),
    }
    _write_json(args.output_directory / "trial_summary.json", trial_summary)
    print(json.dumps(trial_summary, ensure_ascii=False), flush=True)
    return 0 if stable else 2


def _teacher_metrics(
    result: Mapping[str, Any],
    *,
    proxy_row: Mapping[str, Any],
) -> dict[str, Any]:
    components: dict[str, dict[str, float | int]] = {}
    positive: list[str] = []
    total_energy = 0.0
    severity_values: list[float] = []
    for identifier, record in result["components"].items():
        metrics = record["fragment_metrics"]
        hits = float(metrics["hits"]["mean"])
        penetrations = float(metrics["penetrations"]["mean"])
        deposited = float(metrics["deposited_energy_j"]["mean"])
        hit_indicator = metrics.get("hit_indicator")
        support_rate = (
            float(hit_indicator["mean"])
            if isinstance(hit_indicator, Mapping)
            else float(min(1.0, max(0.0, hits)))
        )
        supported = int(
            support_rate >= TEACHER_COMPONENT_SUPPORT_RATE_THRESHOLD
        )
        if supported:
            positive.append(str(identifier))
        fragment_states = record.get("damage_states", {}).get("fragment", {})
        severity = max(
            (
                float(value["mean"])
                for value in fragment_states.values()
                if isinstance(value, Mapping) and "mean" in value
            ),
            default=0.0,
        )
        severity_values.append(severity)
        total_energy += deposited
        components[str(identifier)] = {
            "hits": hits,
            "penetrations": penetrations,
            "deposited_energy_j": deposited,
            "support_rate": support_rate,
            "positive_support": supported,
            "fragment_damage_severity": severity,
        }
    tree_outputs = {
        str(name): float(record["mean"])
        for name, record in result["damage_tree"]["outputs"].items()
    }
    scientific_completion_status = str(
        result.get("scientific_completion_status", "missing")
    )
    penetration_coverage = result.get("penetration_domain_coverage", {})
    affected_fraction = float(
        penetration_coverage.get("affected_replicate_fraction", 1.0)
    )
    observed_domain_complete = (
        affected_fraction == 0.0
        and scientific_completion_status
        in {
            "complete_model_domain",
            "observed_domain_complete_not_statistically_demonstrated",
        }
    )
    statistically_demonstrated_complete = (
        scientific_completion_status == "complete_model_domain"
        and affected_fraction == 0.0
    )
    teacher_severity = float(np.mean(severity_values))
    proxy_u = float(proxy_row["idea1_envelope"]["u_penetration"])
    teacher_utilization = (
        0.0 if proxy_u <= 1.0e-12 else teacher_severity / proxy_u
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
        "any_component_support": int(bool(positive)),
        "positive_component_ids": sorted(positive),
        "positive_component_count": len(positive),
        "components": components,
        "tree_outputs": tree_outputs,
        "tree_output_sum": float(sum(tree_outputs.values())),
        "teacher_damage_severity": teacher_severity,
        "proxy_u_penetration": proxy_u,
        "proxy_normalized_utilization": float(
            proxy_row["idea1_envelope"]["normalized_utilization"]
        ),
        "teacher_normalized_utilization_raw": float(teacher_utilization),
        "scientific_completion_status": scientific_completion_status,
        "penetration_affected_replicate_fraction": affected_fraction,
        "domain_complete": observed_domain_complete,
        "domain_observed_complete": observed_domain_complete,
        "domain_statistically_demonstrated_complete": (
            statistically_demonstrated_complete
        ),
    }


def _run_teacher(args: argparse.Namespace) -> int:
    strategy_root = args.trial_directory.resolve() / args.strategy
    selected_case_path = strategy_root / "selected_cases.jsonl.gz"
    selected_proxy_path = strategy_root / "selected_proxy_rows.jsonl.gz"
    if not selected_case_path.exists() or not selected_proxy_path.exists():
        raise FileNotFoundError(
            f"frozen strategy artifacts are missing under {strategy_root}"
        )
    if args.output_directory.exists() and not args.resume:
        raise FileExistsError(
            f"refusing to overwrite teacher output: {args.output_directory}"
        )
    args.output_directory.mkdir(parents=True, exist_ok=True)
    paths, _, _, engine = _resolve_inputs(args)
    source_paths = {
        **paths,
        "selected_cases": selected_case_path,
        "selected_proxy_rows": selected_proxy_path,
    }
    source_at_start = _source_identity(source_paths)
    manifest_path = args.output_directory / "teacher_manifest.json"
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        if manifest.get("source_identity_at_start") != source_at_start:
            raise RuntimeError("teacher resume source identity has drifted")
        if int(manifest.get("replicates", -1)) != args.replicates:
            raise RuntimeError("teacher resume replicate count has changed")
    else:
        manifest = {
            "schema_version": TOOL_VERSION,
            "stage": "explicit_teacher",
            "status": "running",
            "strategy": args.strategy,
            "replicates": args.replicates,
            "trial_directory": str(args.trial_directory.resolve()),
            "selected_case_hash": source_at_start["selected_cases_hash"],
            "selected_proxy_hash": source_at_start["selected_proxy_rows_hash"],
            "source_identity_at_start": source_at_start,
            "engine_version": ENGINE_VERSION,
            "engine_code_hash": ENGINE_CODE_HASH,
            "evidence_scope": (
                "R=4 diagnostic software teacher confirmation; not formal "
                "physical admission"
            ),
        }
        _write_json(manifest_path, manifest)

    cases = list(_iter_jsonl_gz(selected_case_path))
    proxy_rows = list(_iter_jsonl_gz(selected_proxy_path))
    if len(cases) < 2000 or len(cases) != len(proxy_rows):
        raise RuntimeError("teacher confirmation requires aligned >=2000-case inputs")
    proxy_by_id = {str(row["candidate_id"]): row for row in proxy_rows}
    metrics_path = args.output_directory / "teacher_metrics.jsonl.gz"
    results_path = args.output_directory / "teacher_results.jsonl.gz"
    existing_rows = list(_iter_jsonl_gz(metrics_path)) if metrics_path.exists() else []
    completed_ids = {str(row["candidate_id"]) for row in existing_rows}
    pending = [
        case
        for case in cases
        if str(case["metadata"]["idea1_selection"]["candidate_id"])
        not in completed_ids
    ]
    metrics_mode = "at" if metrics_path.exists() else "wt"
    results_mode = "at" if results_path.exists() else "wt"
    started = perf_counter()
    with gzip.open(metrics_path, metrics_mode, encoding="utf-8") as metrics_stream, gzip.open(
        results_path, results_mode, encoding="utf-8"
    ) as results_stream:
        for index, frozen_case in enumerate(pending):
            case = deepcopy(frozen_case)
            selection = case["metadata"]["idea1_selection"]
            candidate_id = str(selection["candidate_id"])
            case["analysis"]["replicates"] = int(args.replicates)
            case["analysis"]["adaptive_ci_half_width"] = None
            case["analysis"]["capture_fragment_audit"] = False
            case["analysis"]["capture_replicate_trace"] = False
            case_started = perf_counter()
            try:
                result = engine.run(case, fail_closed=True).to_dict()
                metrics = _teacher_metrics(
                    result,
                    proxy_row=proxy_by_id[candidate_id],
                )
                metrics.update(
                    {
                        "candidate_id": candidate_id,
                        "approach_profile_cell": selection[
                            "approach_profile_cell"
                        ],
                        "envelope_band": selection["envelope_band"],
                        "profile_id": case["munition"]["profile_id"],
                        "status": (
                            "ok" if metrics["domain_complete"] else "domain_censored"
                        ),
                        "elapsed_seconds": perf_counter() - case_started,
                        "result_hash": content_hash(result),
                    }
                )
                results_stream.write(
                    json.dumps(
                        {
                            "candidate_id": candidate_id,
                            "proxy_envelope": proxy_by_id[candidate_id][
                                "idea1_envelope"
                            ],
                            "result": result,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
            except DamageEngineError as error:
                metrics = {
                    "candidate_id": candidate_id,
                    "case_id": case.get("case_id"),
                    "approach_profile_cell": selection["approach_profile_cell"],
                    "envelope_band": selection["envelope_band"],
                    "profile_id": case["munition"]["profile_id"],
                    "status": "refused",
                    "domain_complete": False,
                    "elapsed_seconds": perf_counter() - case_started,
                    "error": error.to_dict(),
                }
            metrics_stream.write(
                json.dumps(
                    metrics,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
            if (index + 1) % 10 == 0:
                metrics_stream.flush()
                results_stream.flush()
            if (index + 1) % 50 == 0 or index + 1 == len(pending):
                print(
                    json.dumps(
                        {
                            "stage": "explicit_teacher",
                            "strategy": args.strategy,
                            "completed_this_run": index + 1,
                            "pending_this_run": len(pending),
                            "completed_total": len(completed_ids) + index + 1,
                            "target_total": len(cases),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    final_rows = list(_iter_jsonl_gz(metrics_path))
    source_at_end = _source_identity(source_paths)
    live_hash = _live_engine_code_hash()
    stable = source_at_start == source_at_end and live_hash == ENGINE_CODE_HASH
    complete = len({str(row["candidate_id"]) for row in final_rows}) == len(cases)
    manifest.update(
        {
            "status": (
                "invalid_source_drift"
                if not stable
                else "complete"
                if complete
                else "partial"
            ),
            "selected_case_count": len(cases),
            "teacher_metric_count": len(final_rows),
            "ok_count": int(sum(row.get("status") == "ok" for row in final_rows)),
            "domain_censored_count": int(
                sum(row.get("status") == "domain_censored" for row in final_rows)
            ),
            "refused_count": int(
                sum(row.get("status") == "refused" for row in final_rows)
            ),
            "elapsed_seconds_this_run": perf_counter() - started,
            "source_identity_at_end": source_at_end,
            "live_engine_code_hash_at_end": live_hash,
            "source_stable_during_run": stable,
        }
    )
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    return 0 if stable and complete else 2


def _nested(value: Mapping[str, Any], path: str) -> float:
    current: Any = value
    for key in path.split("."):
        current = current[key]
    return float(current)


def _run_aggregate_proxy(args: argparse.Namespace) -> int:
    trials = [_load_json(path / "trial_summary.json") for path in args.trial_directories]
    if len(trials) < 3:
        raise ValueError("proxy aggregation requires at least three independent trials")
    seeds = [int(trial["design_seed"]) for trial in trials]
    if len(set(seeds)) != len(seeds):
        raise ValueError("proxy trial design seeds must be unique")
    if any(trial.get("status") != "complete" for trial in trials):
        raise RuntimeError("proxy aggregation requires complete source-stable trials")
    metric_paths = [
        "component_coverage.ever",
        "component_coverage.at_least_20",
        "component_coverage.at_least_50",
        "component_coverage.gini",
        "component_coverage.mean_binary_entropy_nats",
        "envelope_band_fractions.transition",
        "zero_u_fraction",
        "utilization_saturation_fraction",
        "tree_saturation_fraction",
        "design_weight_quality.ess_fraction",
        "design_weight_quality.weight_cv",
        "learner_space_nearest_neighbor_distance.median",
    ]
    if all(
        "sampling_risk_prediction" in trial["strategies"][strategy]
        for trial in trials
        for strategy in STRATEGIES
    ):
        metric_paths.extend(
            [
                "sampling_risk_prediction.complete_domain_fraction",
                "sampling_risk_prediction.tree_saturation_given_complete",
            ]
        )
    strategy_report: dict[str, Any] = {}
    for strategy in STRATEGIES:
        metrics: dict[str, Any] = {}
        for path in metric_paths:
            values = [
                _nested(trial["strategies"][strategy], path) for trial in trials
            ]
            metrics[path] = {
                "values": values,
                "mean": float(np.mean(values)),
                "stddev": float(np.std(values, ddof=1)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
        strategy_report[strategy] = metrics

    idea_strategy = str(args.idea_strategy)
    paired_differences: dict[str, Any] = {}
    for comparator in ("reference_uniform", "current_response_priority"):
        key = f"{idea_strategy}_minus_{comparator}"
        paired_differences[key] = {}
        for path in metric_paths:
            values = [
                _nested(trial["strategies"][idea_strategy], path)
                - _nested(trial["strategies"][comparator], path)
                for trial in trials
            ]
            paired_differences[key][path] = {
                "values": values,
                "mean": float(np.mean(values)),
                "stddev": float(np.std(values, ddof=1)),
            }

    per_trial_gates = []
    for trial in trials:
        idea = trial["strategies"][idea_strategy]
        current = trial["strategies"]["current_response_priority"]
        basic = (
            int(idea["selected_count"]) == 2000
            and int(idea["unique_case_count"]) == 2000
            and float(idea["model_domain_complete_fraction"]) == 1.0
            and len(idea["approach_profile_cell_counts"]) == 18
            and _nested(idea, "design_weight_quality.ess_fraction") >= 0.50
            and _nested(idea, "design_weight_quality.weight_cv") <= 2.0
            and _nested(
                idea,
                "design_weight_quality.inclusion_probability_min_selected",
            )
            >= 0.08
        )
        coverage_noninferior = (
            _nested(idea, "component_coverage.ever")
            >= _nested(current, "component_coverage.ever") - 2.0
        )
        wins = {
            "at_least_20": (
                _nested(idea, "component_coverage.at_least_20")
                >= _nested(current, "component_coverage.at_least_20")
            ),
            "support_gini": (
                _nested(idea, "component_coverage.gini")
                < _nested(current, "component_coverage.gini")
            ),
            "transition_target": (
                abs(_nested(idea, "envelope_band_fractions.transition") - 0.45)
                < abs(
                    _nested(current, "envelope_band_fractions.transition")
                    - 0.45
                )
            ),
        }
        if idea_strategy in {
            "idea1_coverage_information_balanced",
            "idea1_hurdle_coverage_balanced",
            "idea1_hurdle_strong_balanced",
        }:
            wins.update(
                {
                    "at_least_50": (
                        _nested(idea, "component_coverage.at_least_50")
                        >= _nested(current, "component_coverage.at_least_50")
                    ),
                    "tree_saturation": (
                        _nested(idea, "tree_saturation_fraction")
                        <= _nested(current, "tree_saturation_fraction")
                    ),
                    "utilization_saturation": (
                        _nested(idea, "utilization_saturation_fraction")
                        <= _nested(current, "utilization_saturation_fraction")
                        + 0.01
                    ),
                }
            )
        if idea_strategy in {
            "idea1_hurdle_coverage_balanced",
            "idea1_hurdle_strong_balanced",
        }:
            coverage_only = trial["strategies"][
                "idea1_global_coverage_balanced"
            ]
            wins.update(
                {
                    "predicted_complete_domain": (
                        _nested(
                            idea,
                            "sampling_risk_prediction.complete_domain_fraction",
                        )
                        >= _nested(
                            coverage_only,
                            "sampling_risk_prediction.complete_domain_fraction",
                        )
                    ),
                    "predicted_conditional_tree_saturation": (
                        _nested(
                            idea,
                            "sampling_risk_prediction.tree_saturation_given_complete",
                        )
                        <= _nested(
                            coverage_only,
                            "sampling_risk_prediction.tree_saturation_given_complete",
                        )
                    ),
                }
            )
        per_trial_gates.append(
            {
                "design_seed": trial["design_seed"],
                "basic_contract": basic,
                "coverage_noninferior": coverage_noninferior,
                "wins": wins,
                "win_count": int(sum(wins.values())),
                "pass": bool(basic and coverage_noninferior and all(wins.values())),
            }
        )
    report = {
        "schema_version": TOOL_VERSION,
        "stage": "proxy_aggregate",
        "trial_count": len(trials),
        "design_seeds": seeds,
        "primary_idea1_strategy": idea_strategy,
        "strategy_metrics": strategy_report,
        "paired_differences": paired_differences,
        "per_trial_preregistered_gates": per_trial_gates,
        "overall_proxy_gate_pass": all(item["pass"] for item in per_trial_gates),
    }
    _write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


def _teacher_arm_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) < 2000:
        raise ValueError("teacher arm must contain at least 2000 metric rows")
    usable = [row for row in rows if row.get("status") == "ok"]
    completed = [
        row for row in rows if row.get("status") in {"ok", "domain_censored"}
    ]
    if not usable:
        raise RuntimeError("teacher arm has no complete-domain results")
    component_ids = sorted(usable[0]["components"])
    support_counts = {
        identifier: int(
            sum(int(row["components"][identifier]["positive_support"]) for row in usable)
        )
        for identifier in component_ids
    }
    support_array = np.asarray(list(support_counts.values()), dtype=np.float64)
    tree_names = sorted(usable[0]["tree_outputs"])
    tree_counts = {
        name: int(sum(float(row["tree_outputs"][name]) >= 0.05 for row in usable))
        for name in tree_names
    }
    tree_max = [max(map(float, row["tree_outputs"].values()), default=0.0) for row in usable]
    correlations: dict[str, Any] = {}
    for name, field in (
        ("positive_component_count", "positive_component_count"),
        ("total_component_hits", "total_component_hits"),
    ):
        statistic, pvalue = spearmanr(
            [float(row["proxy_u_penetration"]) for row in usable],
            [float(row[field]) for row in usable],
        )
        correlations[name] = {
            "spearman": float(statistic),
            "pvalue": float(pvalue),
        }
    profile_groups: dict[str, list[float]] = defaultdict(list)
    for row in usable:
        profile_groups[str(row["profile_id"])].append(
            float(row["teacher_normalized_utilization_raw"])
        )
    profile_summary = {
        profile: {
            "count": len(values),
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
        }
        for profile, values in sorted(profile_groups.items())
    }
    if len(profile_groups) >= 2 and all(len(values) >= 2 for values in profile_groups.values()):
        test = kruskal(*[profile_groups[key] for key in sorted(profile_groups)])
        invariance_test = {"kruskal_h": float(test.statistic), "pvalue": float(test.pvalue)}
    else:
        invariance_test = None
    return {
        "row_count": len(rows),
        "completed_result_count": len(completed),
        "complete_domain_count": len(usable),
        "complete_domain_fraction": len(usable) / len(rows),
        "refused_count": int(sum(row.get("status") == "refused" for row in rows)),
        "any_component_support_fraction": float(
            np.mean([row["any_component_support"] for row in usable])
        ),
        "positive_component_count_mean": float(
            np.mean([row["positive_component_count"] for row in usable])
        ),
        "positive_component_count_median": float(
            np.median([row["positive_component_count"] for row in usable])
        ),
        "total_component_hits_mean": float(
            np.mean([row["total_component_hits"] for row in usable])
        ),
        "total_component_penetrations_mean": float(
            np.mean([row["total_component_penetrations"] for row in usable])
        ),
        "total_component_deposited_energy_j_mean": float(
            np.mean([row["total_component_deposited_energy_j"] for row in usable])
        ),
        "teacher_damage_severity_mean": float(
            np.mean([row["teacher_damage_severity"] for row in usable])
        ),
        "tree_any_support_fraction": float(
            np.mean([max(map(float, row["tree_outputs"].values()), default=0.0) >= 0.05 for row in usable])
        ),
        "tree_saturation_fraction": float(np.mean(np.asarray(tree_max) >= 0.95)),
        "component_support_counts": support_counts,
        "component_coverage": {
            "ever": int(np.count_nonzero(support_array >= 1)),
            "at_least_5": int(np.count_nonzero(support_array >= 5)),
            "at_least_10": int(np.count_nonzero(support_array >= 10)),
            "at_least_20": int(np.count_nonzero(support_array >= 20)),
            "gini": _gini(support_array),
        },
        "tree_support_counts": tree_counts,
        "proxy_u_teacher_correlations": correlations,
        "normalized_utilization_by_profile": profile_summary,
        "normalized_utilization_profile_test": invariance_test,
    }


def _bootstrap_teacher_metric(rows: Sequence[Mapping[str, Any]], rng: np.random.Generator) -> dict[str, float]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["approach_profile_cell"])].append(row)
    sampled: list[Mapping[str, Any]] = []
    for cell in sorted(groups):
        values = groups[cell]
        indices = rng.integers(0, len(values), size=len(values))
        sampled.extend(values[int(index)] for index in indices)
    usable = [row for row in sampled if row.get("status") == "ok"]
    if not usable:
        return {}
    component_ids = sorted(usable[0]["components"])
    support_counts = np.asarray(
        [
            sum(int(row["components"][identifier]["positive_support"]) for row in usable)
            for identifier in component_ids
        ],
        dtype=np.float64,
    )
    return {
        "complete_domain_fraction": len(usable) / len(sampled),
        "any_component_support_fraction": float(np.mean([row["any_component_support"] for row in usable])),
        "positive_component_count_mean": float(np.mean([row["positive_component_count"] for row in usable])),
        "total_component_hits_mean": float(np.mean([row["total_component_hits"] for row in usable])),
        "total_component_deposited_energy_j_mean": float(np.mean([row["total_component_deposited_energy_j"] for row in usable])),
        "tree_saturation_fraction": float(
            np.mean(
                [max(map(float, row["tree_outputs"].values()), default=0.0) >= 0.95 for row in usable]
            )
        ),
        "component_coverage_at_least_10": float(np.count_nonzero(support_counts >= 10)),
        "component_coverage_at_least_20": float(np.count_nonzero(support_counts >= 20)),
    }


def _run_aggregate_teacher(args: argparse.Namespace) -> int:
    reference_rows = list(
        _iter_jsonl_gz(args.reference_directory / "teacher_metrics.jsonl.gz")
    )
    idea_rows = list(
        _iter_jsonl_gz(args.idea1_directory / "teacher_metrics.jsonl.gz")
    )
    reference = _teacher_arm_summary(reference_rows)
    idea = _teacher_arm_summary(idea_rows)
    rng = np.random.default_rng(args.bootstrap_seed)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(args.bootstrap_draws):
        left = _bootstrap_teacher_metric(reference_rows, rng)
        right = _bootstrap_teacher_metric(idea_rows, rng)
        for name in sorted(set(left).intersection(right)):
            difference = float(right[name] - left[name])
            if math.isfinite(difference):
                draws[name].append(difference)
    intervals = {
        name: {
            "idea1_minus_reference": float(
                _nested(idea, name.replace("component_coverage_", "component_coverage."))
                - _nested(reference, name.replace("component_coverage_", "component_coverage."))
            ),
            "bootstrap_95_interval": [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ],
            "draw_count": len(values),
        }
        for name, values in draws.items()
    }
    if args.idea_strategy in {
        "idea1_coverage_information_balanced",
        "idea1_hurdle_coverage_balanced",
        "idea1_hurdle_strong_balanced",
    }:
        domain_interval = intervals["complete_domain_fraction"][
            "bootstrap_95_interval"
        ]
        support_interval = intervals["any_component_support_fraction"][
            "bootstrap_95_interval"
        ]
        positive_count_interval = intervals["positive_component_count_mean"][
            "bootstrap_95_interval"
        ]
        saturation_interval = intervals["tree_saturation_fraction"][
            "bootstrap_95_interval"
        ]
        gate = {
            "complete_domain_point_noninferior_2pp": (
                idea["complete_domain_fraction"]
                >= reference["complete_domain_fraction"] - 0.02
            ),
            "complete_domain_ci_noninferior_3pp": domain_interval[0] > -0.03,
            "any_support_ci_noninferior_2pp": support_interval[0] > -0.02,
            "positive_component_count_ci_improved": (
                positive_count_interval[0] > 0.0
            ),
            "coverage_at_least_10_not_lower": (
                idea["component_coverage"]["at_least_10"]
                >= reference["component_coverage"]["at_least_10"]
            ),
            "coverage_at_least_20_not_lower": (
                idea["component_coverage"]["at_least_20"]
                >= reference["component_coverage"]["at_least_20"]
            ),
            "component_support_gini_lower": (
                idea["component_coverage"]["gini"]
                < reference["component_coverage"]["gini"]
            ),
            "tree_saturation_point_not_higher": (
                idea["tree_saturation_fraction"]
                <= reference["tree_saturation_fraction"]
            ),
            "tree_saturation_ci_noninferior_1pp": (
                saturation_interval[1] <= 0.01
            ),
        }
    else:
        gate = {
            "complete_domain_not_lower": (
                idea["complete_domain_fraction"]
                >= reference["complete_domain_fraction"]
            ),
            "any_support_noninferior_5pp": (
                idea["any_component_support_fraction"]
                >= reference["any_component_support_fraction"] - 0.05
            ),
            "coverage_at_least_10_not_lower": (
                idea["component_coverage"]["at_least_10"]
                >= reference["component_coverage"]["at_least_10"]
            ),
            "tree_saturation_not_higher": (
                idea["tree_saturation_fraction"]
                <= reference["tree_saturation_fraction"]
            ),
        }
    report = {
        "schema_version": TOOL_VERSION,
        "stage": "teacher_aggregate",
        "reference_uniform": reference,
        str(args.idea_strategy): idea,
        "bootstrap": {
            "draws": args.bootstrap_draws,
            "seed": args.bootstrap_seed,
            "stratification": "approach_profile_cell",
            "differences": intervals,
        },
        "preregistered_teacher_gate": {
            **gate,
            "overall_pass": all(gate.values()),
        },
    }
    _write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


def _add_common_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-case", type=Path, default=DEFAULT_BASE_CASE)
    parser.add_argument("--sampling-spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--physics", type=Path, default=DEFAULT_PHYSICS)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibration = subparsers.add_parser("calibrate")
    _add_common_inputs(calibration)
    calibration.add_argument("--output-directory", type=Path, required=True)
    calibration.add_argument("--candidate-count", type=int, default=2000)
    calibration.add_argument("--design-seed", type=int, default=2026111601)
    calibration.add_argument("--replicates", type=int, default=4)
    calibration.add_argument("--azimuths-per-ring", type=int, default=8)
    calibration.set_defaults(function=_run_calibration)

    trial = subparsers.add_parser("proxy-trial")
    _add_common_inputs(trial)
    trial.add_argument("--calibration", type=Path, required=True)
    trial.add_argument("--risk-model", type=Path, required=True)
    trial.add_argument("--output-directory", type=Path, required=True)
    trial.add_argument("--candidate-count", type=int, default=6048)
    trial.add_argument("--sample-size", type=int, default=2000)
    trial.add_argument("--design-seed", type=int, required=True)
    trial.add_argument("--selection-seed", type=int, required=True)
    trial.add_argument("--replicates", type=int, default=4)
    trial.add_argument("--azimuths-per-ring", type=int, default=8)
    trial.set_defaults(function=_run_proxy_trial)

    teacher = subparsers.add_parser("teacher")
    _add_common_inputs(teacher)
    teacher.add_argument("--trial-directory", type=Path, required=True)
    teacher.add_argument("--strategy", choices=STRATEGIES, required=True)
    teacher.add_argument("--output-directory", type=Path, required=True)
    teacher.add_argument("--replicates", type=int, default=4)
    teacher.add_argument("--resume", action="store_true")
    teacher.set_defaults(function=_run_teacher)

    proxy_aggregate = subparsers.add_parser("aggregate-proxy")
    proxy_aggregate.add_argument(
        "--trial-directories", type=Path, nargs="+", required=True
    )
    proxy_aggregate.add_argument(
        "--idea-strategy", choices=IDEA_STRATEGIES, required=True
    )
    proxy_aggregate.add_argument("--output", type=Path, required=True)
    proxy_aggregate.set_defaults(function=_run_aggregate_proxy)

    teacher_aggregate = subparsers.add_parser("aggregate-teacher")
    teacher_aggregate.add_argument(
        "--reference-directory", type=Path, required=True
    )
    teacher_aggregate.add_argument("--idea1-directory", type=Path, required=True)
    teacher_aggregate.add_argument(
        "--idea-strategy", choices=IDEA_STRATEGIES, required=True
    )
    teacher_aggregate.add_argument("--output", type=Path, required=True)
    teacher_aggregate.add_argument("--bootstrap-draws", type=int, default=2000)
    teacher_aggregate.add_argument("--bootstrap-seed", type=int, default=2026111699)
    teacher_aggregate.set_defaults(function=_run_aggregate_teacher)
    return parser


def main() -> int:
    args = _parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
