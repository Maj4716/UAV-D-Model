"""Run the X1 exposure field and X2 exposure-balanced independent confirmation."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import shlex
import sys
from time import perf_counter
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.contact_contract import audit_s_contact  # noqa: E402
from damage_engine.continuous_direction_sampling import (  # noqa: E402
    apply_continuous_direction_manifold,
)
from damage_engine.design_sequences import generate_unit_design  # noqa: E402
from damage_engine.engine import ENGINE_CODE_HASH, DamageEngine  # noqa: E402
from damage_engine.errors import DamageEngineError  # noqa: E402
from damage_engine.estimands import positive_effective_sample_size  # noqa: E402
from damage_engine.exposure_audit import build_exposure_teacher_audit  # noqa: E402
from damage_engine.fast_sampling_proxy import FastSamplingProxy  # noqa: E402
from damage_engine.impact import resolve_impact_detonation  # noqa: E402
from damage_engine.munition_profiles import apply_munition_profile  # noqa: E402
from damage_engine.normalized_sampling import (  # noqa: E402
    global_marginal_coverage_probabilities,
    pivotal_sample,
)
from damage_engine.registry import PhysicsRegistry  # noqa: E402
from damage_engine.sampling_spec import SamplingSpec  # noqa: E402
from damage_engine.utils import content_hash, source_tree_hash  # noqa: E402


_WORKER_ENGINE: DamageEngine | None = None
_WORKER_PROXY: FastSamplingProxy | None = None
_WORKER_SPEC: SamplingSpec | None = None
_WORKER_BASE_CASE: dict[str, Any] | None = None
_WORKER_COMPONENT_IDS: list[str] = []


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _init_worker(
    target_path: str,
    physics_path: str,
    base_case_path: str,
    sampling_spec_path: str,
    proxy_source_seed: int,
    azimuths_per_ring: int,
    layer_quadrature_samples: int,
) -> None:
    global _WORKER_ENGINE, _WORKER_PROXY, _WORKER_SPEC, _WORKER_BASE_CASE
    global _WORKER_COMPONENT_IDS
    target = _load_json(Path(target_path))
    registry = PhysicsRegistry.from_file(Path(physics_path))
    _WORKER_ENGINE = DamageEngine(target, registry)
    _WORKER_PROXY = FastSamplingProxy(
        _WORKER_ENGINE,
        source_seed=int(proxy_source_seed),
        azimuths_per_ring=int(azimuths_per_ring),
        layer_quadrature_samples=int(layer_quadrature_samples),
    )
    _WORKER_SPEC = SamplingSpec.from_file(Path(sampling_spec_path))
    _WORKER_BASE_CASE = _load_json(Path(base_case_path))
    _WORKER_BASE_CASE["target_ref"]["content_hash"] = _WORKER_ENGINE.target_hash
    _WORKER_COMPONENT_IDS = list(_WORKER_ENGINE.compiled_target.component_order)


def _build_case(payload: tuple[int, int, tuple[float, ...]]) -> dict[str, Any]:
    if _WORKER_ENGINE is None or _WORKER_SPEC is None or _WORKER_BASE_CASE is None:
        raise RuntimeError("X1 worker was not initialized")
    attempt_index, scramble_id, latent = payload
    coordinates = np.asarray(latent, dtype=np.float64)
    independent_width = _WORKER_SPEC.independent_latent_dimension
    profile_variable = next(
        value for value in _WORKER_SPEC.variables if value["name"] == "munition_profile"
    )
    profiles = list(profile_variable["values"])
    profile_index = min(int(coordinates[0] * len(profiles)), len(profiles) - 1)
    profile_id = str(profiles[profile_index])
    case = deepcopy(_WORKER_BASE_CASE)
    candidate_id = f"x1-s{scramble_id:02d}-a{attempt_index:06d}"
    case["case_id"] = candidate_id
    case["munition"]["profile_id"] = profile_id
    manifold = _WORKER_SPEC.contact_manifold
    if manifold is None:
        raise RuntimeError("SamplingSpec 1.7 lacks its manifold")
    apply_continuous_direction_manifold(
        case,
        manifold,
        coordinates[independent_width:],
        _WORKER_ENGINE.compiled_target,
        target_hash=_WORKER_ENGINE.target_hash,
    )
    apply_munition_profile(case)
    case = resolve_impact_detonation(case, _WORKER_ENGINE.compiled_target)
    conditioning = _WORKER_SPEC.contact_conditioning
    if conditioning is None:
        raise RuntimeError("SamplingSpec 1.7 lacks S-CONTACT conditioning")
    admission = audit_s_contact(
        case,
        _WORKER_ENGINE.compiled_target,
        data_role="model_train_reference",
        estimand=str(conditioning["estimand"]),
        selection_declaration=conditioning["selection_declaration"],
    )
    case.setdefault("metadata", {})["x1_s_contact"] = admission
    return case


def _worker_admit(
    payload: tuple[int, int, tuple[float, ...]],
) -> tuple[bool, str]:
    try:
        _build_case(payload)
    except DamageEngineError as exc:
        return False, str(exc.context.get("reason_code", exc.code))
    except (KeyError, TypeError, ValueError) as exc:
        return False, type(exc).__name__
    return True, "accepted"


def _direction_contract_violation(record: dict[str, Any]) -> bool:
    direction = np.asarray(record["direction_target_xyz"], dtype=np.float64)
    if record["region_kind"] == "top_cap":
        offset = math.degrees(math.acos(float(np.clip(-direction[2], -1.0, 1.0))))
        return not -1.0e-10 <= offset <= 45.0 + 1.0e-10
    dive = math.degrees(math.asin(float(np.clip(-direction[2], -1.0, 1.0))))
    centres = {"pos_x": 0.0, "pos_y": 90.0, "neg_x": 180.0, "neg_y": 270.0}
    centre = centres[str(record["region_id"])]
    azimuth = math.degrees(math.atan2(direction[1], direction[0])) % 360.0
    difference = abs((azimuth - centre + 180.0) % 360.0 - 180.0)
    return not (
        -1.0e-10 <= dive <= 45.0 + 1.0e-10
        and difference <= 30.0 + 1.0e-10
    )


def _worker_proxy(payload: tuple[int, int, tuple[float, ...]]) -> dict[str, Any]:
    if _WORKER_PROXY is None:
        raise RuntimeError("X1 proxy worker was not initialized")
    case = _build_case(payload)
    proxy = _WORKER_PROXY.evaluate(case)
    metrics = proxy["component_metrics"]
    metadata = case["metadata"]["continuous_direction_sampling"]
    sphere = case["metadata"]["sphere_entry_ray_sampling"]["sphere"]
    direction = np.asarray(metadata["direction_target_xyz"], dtype=np.float64)
    tangent_u = np.asarray(metadata["tangent_u_target_xyz"], dtype=np.float64)
    tangent_v = np.asarray(metadata["tangent_v_target_xyz"], dtype=np.float64)
    frame = np.column_stack((direction, tangent_u, tangent_v))
    origin = np.asarray(case["encounter"]["trajectory_origin_m"], dtype=np.float64)
    centre = np.asarray(sphere["center_target_m"], dtype=np.float64)
    return {
        "candidate_id": case["case_id"],
        "profile_id": case["munition"]["profile_id"],
        "region_id": metadata["region_id"],
        "cell_id": f"{metadata['region_id']}::{case['munition']['profile_id']}",
        "lambda": [float(metrics[value]["expected_hits"]) for value in _WORKER_COMPONENT_IDS],
        "energy": [
            float(metrics[value]["expected_damage_energy_j"])
            for value in _WORKER_COMPONENT_IDS
        ],
        "proxy_domain_status": proxy["proxy_domain_status"],
        "out_of_domain_packet_weight": float(proxy["out_of_domain_packet_weight"]),
        "frame_orthonormality_error": float(
            np.max(np.abs(frame.T @ frame - np.eye(3)))
        ),
        "frame_determinant_error": abs(float(np.linalg.det(frame)) - 1.0),
        "sphere_boundary_residual_m": abs(
            float(np.linalg.norm(origin - centre)) - float(sphere["radius_m"])
        ),
        "direction_contract_violation": _direction_contract_violation(metadata),
    }


def _worker_teacher(
    payload: tuple[int, int, tuple[float, ...], int, int],
) -> dict[str, Any]:
    if _WORKER_ENGINE is None:
        raise RuntimeError("X1 teacher worker was not initialized")
    attempt_index, scramble_id, latent, replicates, seed = payload
    case = _build_case((attempt_index, scramble_id, latent))
    case["analysis"]["replicates"] = int(replicates)
    case["analysis"]["adaptive_ci_half_width"] = None
    case["analysis"]["aleatory_seed"] = int(seed)
    case["analysis"]["epistemic_sample_id"] = f"x1-teacher-{case['case_id']}"
    case["analysis"]["capture_fragment_audit"] = False
    case["analysis"]["capture_replicate_trace"] = False
    result = _WORKER_ENGINE.run(case, fail_closed=True).to_dict()
    components = result["components"]
    return {
        "candidate_id": case["case_id"],
        "hit_probability": [
            float(components[value]["fragment_metrics"]["hit_indicator"]["mean"])
            for value in _WORKER_COMPONENT_IDS
        ],
    }


def _largest_remainder(
    sizes: np.ndarray, total: int, *, require_each: bool
) -> np.ndarray:
    if require_each and total < sizes.size:
        raise ValueError("total is too small to cover every cell")
    base = np.ones(sizes.size, dtype=np.int64) if require_each else np.zeros(sizes.size, dtype=np.int64)
    remaining = int(total - np.sum(base))
    capacity = sizes - base
    if remaining < 0 or remaining > int(np.sum(capacity)):
        raise ValueError("allocation total is outside cell capacity")
    if remaining == 0:
        return base
    raw = remaining * capacity / float(np.sum(capacity))
    additions = np.floor(raw).astype(np.int64)
    output = base + additions
    remainder = int(remaining - np.sum(additions))
    order = np.lexsort((np.arange(sizes.size), -(raw - additions)))
    for index in order:
        if remainder == 0:
            break
        if output[index] < sizes[index]:
            output[index] += 1
            remainder -= 1
    if remainder:
        raise RuntimeError("largest-remainder allocation did not close")
    return output


def _gini(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or np.any(array < 0.0):
        raise ValueError("Gini values must be nonempty and non-negative")
    total = float(np.sum(array))
    if total == 0.0:
        return 0.0
    ordered = np.sort(array)
    indices = np.arange(1, array.size + 1, dtype=np.float64)
    return float(
        (2.0 * np.sum(indices * ordered) / (array.size * total))
        - (array.size + 1.0) / array.size
    )


def _entropy(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    positive = array[array > 0.0]
    if positive.size == 0:
        return 0.0
    probability = positive / float(np.sum(positive))
    return float(-np.sum(probability * np.log(probability)) / math.log(len(array)))


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    normalized = weights / float(np.sum(weights))
    return normalized @ values


def _arm_metrics(
    lambdas: np.ndarray,
    selected: np.ndarray,
    probabilities: np.ndarray,
    reference_mean: np.ndarray,
    component_ids: list[str],
    cells: np.ndarray,
) -> dict[str, Any]:
    indices = np.flatnonzero(selected)
    selected_probability = probabilities[indices]
    weights = 1.0 / selected_probability
    arm_mean = np.mean(lambdas[indices], axis=0)
    hajek = _weighted_mean(lambdas[indices], weights)
    support_counts = np.sum(lambdas[indices] > 0.0, axis=0)
    scale = float(np.sum(reference_mean))
    return {
        "selected_count": int(indices.size),
        "unique_count": int(np.unique(indices).size),
        "covered_cell_count": int(np.unique(cells[indices]).size),
        "design_ess": positive_effective_sample_size(weights),
        "design_ess_fraction": positive_effective_sample_size(weights) / indices.size,
        "design_weight_cv": float(np.std(weights) / np.mean(weights)),
        "minimum_selected_inclusion_probability": float(np.min(selected_probability)),
        "maximum_selected_inclusion_probability": float(np.max(selected_probability)),
        "component_support_count": {
            value: int(support_counts[index]) for index, value in enumerate(component_ids)
        },
        "support_ever": int(np.sum(support_counts > 0)),
        "support_at_least_20": int(np.sum(support_counts >= 20)),
        "support_at_least_50": int(np.sum(support_counts >= 50)),
        "support_count_gini": _gini(support_counts.astype(np.float64)),
        "component_exposure_entropy": _entropy(arm_mean),
        "normalized_l1_sampling_footprint_shift": (
            None
            if scale <= 0.0
            else float(np.sum(np.abs(arm_mean - reference_mean)) / scale)
        ),
        "normalized_l1_hajek_recovery_error": (
            None
            if scale <= 0.0
            else float(np.sum(np.abs(hajek - reference_mean)) / scale)
        ),
        "component_sampling_footprint_ratio": {
            value: (
                None
                if reference_mean[index] <= 0.0
                else float(arm_mean[index] / reference_mean[index])
            )
            for index, value in enumerate(component_ids)
        },
        "component_hajek_lambda": {
            value: float(hajek[index]) for index, value in enumerate(component_ids)
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("artifacts/experiments/X1_X2_20261120/experiment_config.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/experiments/X1_X2_20261120"),
    )
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument("--scramble-count", type=int)
    parser.add_argument("--teacher-count", type=int)
    parser.add_argument("--teacher-replicates", type=int)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    config_path = _resolve(args.config)
    output = _resolve(args.output)
    config = _load_json(config_path)
    x1_config = config["x1"]
    teacher_config = config["teacher_audit"]
    x2_config = config["x2"]
    candidate_count = int(args.candidate_count or x1_config["candidate_count"])
    scramble_count = int(args.scramble_count or x1_config["scramble_count"])
    teacher_count = int(args.teacher_count or teacher_config["executed_candidate_count"])
    teacher_replicates = int(args.teacher_replicates or teacher_config["executed_replicates"])
    sample_size = int(args.sample_size or x2_config["sample_size_per_arm"])
    workers = int(args.workers or config["workers"])
    if candidate_count % scramble_count:
        raise ValueError("candidate_count must be divisible by scramble_count")
    per_scramble = candidate_count // scramble_count
    if sample_size > candidate_count or teacher_count > candidate_count:
        raise ValueError("teacher/sample sizes cannot exceed candidate_count")
    if output.exists() and not args.force:
        existing = [value for value in output.iterdir() if value.name not in {"experiment_config.json", "PREREGISTRATION.md"}]
        if existing:
            raise FileExistsError(f"refusing to overwrite experiment outputs in {output}")
    output.mkdir(parents=True, exist_ok=True)

    target_path = _resolve(config["target"])
    physics_path = _resolve(config["physics"])
    base_case_path = _resolve(config["base_case"])
    sampling_spec_path = _resolve(config["sampling_spec"])
    main_engine = DamageEngine(
        _load_json(target_path), PhysicsRegistry.from_file(physics_path)
    )
    sampling_spec = SamplingSpec.from_file(sampling_spec_path)
    component_ids = list(main_engine.compiled_target.component_order)
    component_count = len(component_ids)

    lambdas = np.empty((candidate_count, component_count), dtype=np.float32)
    energies = np.empty_like(lambdas)
    latents = np.empty((candidate_count, sampling_spec.latent_dimension), dtype=np.float64)
    attempt_indices = np.empty(candidate_count, dtype=np.int32)
    scramble_ids = np.empty(candidate_count, dtype=np.int16)
    candidate_ids = np.empty(candidate_count, dtype="U32")
    profile_ids = np.empty(candidate_count, dtype="U24")
    region_ids = np.empty(candidate_count, dtype="U16")
    cell_ids = np.empty(candidate_count, dtype="U48")
    proxy_domain_status = np.empty(candidate_count, dtype="U40")
    out_of_domain_weight = np.empty(candidate_count, dtype=np.float64)
    frame_error = np.empty(candidate_count, dtype=np.float64)
    determinant_error = np.empty(candidate_count, dtype=np.float64)
    sphere_residual = np.empty(candidate_count, dtype=np.float64)
    direction_violation = np.empty(candidate_count, dtype=bool)
    rejection_counts: Counter[str] = Counter()
    design_records: list[dict[str, Any]] = []
    timings: dict[str, float] = {}

    worker_args = (
        str(target_path),
        str(physics_path),
        str(base_case_path),
        str(sampling_spec_path),
        int(x1_config["proxy_source_seed"]),
        int(x1_config["proxy_azimuths_per_ring_executed"]),
        int(x1_config["proxy_layer_quadrature_samples_executed"]),
    )
    context = mp.get_context("spawn")
    started_total = perf_counter()
    with context.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=worker_args,
    ) as pool:
        offset = 0
        for scramble_id in range(scramble_count):
            required_power = int(x1_config["sobol_attempt_power"])
            if candidate_count != int(x1_config["candidate_count"]):
                required_power = max(
                    6,
                    int(math.ceil(math.log2(max(2, 3 * per_scramble)))),
                )
            while (1 << required_power) < per_scramble:
                required_power += 1
            design = generate_unit_design(
                1 << required_power,
                sampling_spec.latent_dimension,
                seed=int(x1_config["design_seed_base"]) + scramble_id,
                method="scrambled_sobol",
            )
            payloads = [
                (index, scramble_id, tuple(float(value) for value in point))
                for index, point in enumerate(design.points)
            ]
            admission_started = perf_counter()
            admission = pool.map(_worker_admit, payloads, chunksize=64)
            timings[f"scramble_{scramble_id:02d}_admission_seconds"] = (
                perf_counter() - admission_started
            )
            accepted_indices = [
                index for index, (accepted, _reason) in enumerate(admission) if accepted
            ]
            if len(accepted_indices) < per_scramble:
                raise RuntimeError(
                    f"Sobol attempt frame admitted only {len(accepted_indices)} of {per_scramble} required cases"
                )
            selected_indices = accepted_indices[:per_scramble]
            last_attempt = selected_indices[-1]
            rejection_counts.update(
                reason
                for accepted, reason in admission[: last_attempt + 1]
                if not accepted
            )
            selected_payloads = [payloads[index] for index in selected_indices]
            proxy_started = perf_counter()
            proxy_rows = pool.map(_worker_proxy, selected_payloads, chunksize=16)
            timings[f"scramble_{scramble_id:02d}_proxy_seconds"] = (
                perf_counter() - proxy_started
            )
            stop = offset + per_scramble
            lambdas[offset:stop] = np.asarray(
                [value["lambda"] for value in proxy_rows], dtype=np.float32
            )
            energies[offset:stop] = np.asarray(
                [value["energy"] for value in proxy_rows], dtype=np.float32
            )
            latents[offset:stop] = design.points[selected_indices]
            attempt_indices[offset:stop] = np.asarray(selected_indices, dtype=np.int32)
            scramble_ids[offset:stop] = scramble_id
            candidate_ids[offset:stop] = [value["candidate_id"] for value in proxy_rows]
            profile_ids[offset:stop] = [value["profile_id"] for value in proxy_rows]
            region_ids[offset:stop] = [value["region_id"] for value in proxy_rows]
            cell_ids[offset:stop] = [value["cell_id"] for value in proxy_rows]
            proxy_domain_status[offset:stop] = [
                value["proxy_domain_status"] for value in proxy_rows
            ]
            out_of_domain_weight[offset:stop] = [
                value["out_of_domain_packet_weight"] for value in proxy_rows
            ]
            frame_error[offset:stop] = [
                value["frame_orthonormality_error"] for value in proxy_rows
            ]
            determinant_error[offset:stop] = [
                value["frame_determinant_error"] for value in proxy_rows
            ]
            sphere_residual[offset:stop] = [
                value["sphere_boundary_residual_m"] for value in proxy_rows
            ]
            direction_violation[offset:stop] = [
                value["direction_contract_violation"] for value in proxy_rows
            ]
            design_records.append(
                {
                    **design.metadata,
                    "scramble_id": scramble_id,
                    "accepted_count": per_scramble,
                    "attempt_prefix_count": last_attempt + 1,
                    "contact_acceptance_fraction": per_scramble / (last_attempt + 1),
                    "accepted_prefix_hash": content_hash(
                        {
                            "attempt_indices": selected_indices,
                            "latent": design.points[selected_indices].tolist(),
                        }
                    ),
                }
            )
            offset = stop
            print(
                json.dumps(
                    {
                        "stage": "x1_proxy",
                        "scramble": scramble_id + 1,
                        "scramble_count": scramble_count,
                        "accepted_total": offset,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        unique_cells = sorted(set(cell_ids.tolist()))
        teacher_sizes = np.asarray(
            [int(np.sum(cell_ids == value)) for value in unique_cells], dtype=np.int64
        )
        teacher_quotas = _largest_remainder(
            teacher_sizes,
            teacher_count,
            require_each=teacher_count >= len(unique_cells),
        )
        teacher_rng = np.random.default_rng(int(teacher_config["selection_seed"]))
        teacher_indices_list: list[int] = []
        for cell, quota in zip(unique_cells, teacher_quotas, strict=True):
            local = np.flatnonzero(cell_ids == cell)
            teacher_indices_list.extend(
                int(value)
                for value in teacher_rng.choice(local, size=int(quota), replace=False)
            )
        teacher_indices = np.asarray(sorted(teacher_indices_list), dtype=np.int64)
        teacher_payloads = [
            (
                int(attempt_indices[index]),
                int(scramble_ids[index]),
                tuple(float(value) for value in latents[index]),
                teacher_replicates,
                int(teacher_config["aleatory_seed_base"]) + position,
            )
            for position, index in enumerate(teacher_indices)
        ]
        teacher_started = perf_counter()
        teacher_results = pool.map(_worker_teacher, teacher_payloads, chunksize=1)
        timings["teacher_seconds"] = perf_counter() - teacher_started

    timings["total_compute_seconds"] = perf_counter() - started_total
    teacher_hits = np.asarray(
        [value["hit_probability"] for value in teacher_results], dtype=np.float64
    )

    # X1 finite-frame field and structural audits.
    reference_lambda = np.mean(lambdas.astype(np.float64), axis=0)
    reference_energy = np.mean(energies.astype(np.float64), axis=0)
    reached = lambdas > 0.0
    component_summary = {
        value: {
            "lambda_hajek": float(reference_lambda[index]),
            "damage_energy_hajek_j": float(reference_energy[index]),
            "reachability_status": (
                "proxy_reached_in_declared_frame"
                if bool(np.any(reached[:, index]))
                else "proxy_unreached_in_declared_frame"
            ),
            "reached_candidate_count": int(np.sum(reached[:, index])),
            "reached_fraction": float(np.mean(reached[:, index])),
            "structural_zero_claim": False,
        }
        for index, value in enumerate(component_ids)
    }
    cell_envelopes: dict[str, Any] = {}
    cell_reachability: dict[str, Any] = {}
    for cell in unique_cells:
        indices = np.flatnonzero(cell_ids == cell)
        local_lambda = lambdas[indices].astype(np.float64)
        local_energy = energies[indices].astype(np.float64)
        cell_envelopes[cell] = {
            "candidate_count": int(indices.size),
            "components": {
                value: {
                    "lambda_mean": float(np.mean(local_lambda[:, column])),
                    "lambda_q50": float(np.quantile(local_lambda[:, column], 0.50)),
                    "lambda_q90": float(np.quantile(local_lambda[:, column], 0.90)),
                    "lambda_q99": float(np.quantile(local_lambda[:, column], 0.99)),
                    "lambda_max": float(np.max(local_lambda[:, column])),
                    "damage_energy_mean_j": float(np.mean(local_energy[:, column])),
                }
                for column, value in enumerate(component_ids)
            },
        }
        cell_reachability[cell] = {
            value: {
                "status": (
                    "proxy_reached_in_declared_frame"
                    if bool(np.any(local_lambda[:, column] > 0.0))
                    else "proxy_unreached_in_declared_frame"
                ),
                "structural_zero_claim": False,
            }
            for column, value in enumerate(component_ids)
        }
    scramble_means = np.asarray(
        [
            np.mean(lambdas[scramble_ids == value].astype(np.float64), axis=0)
            for value in range(scramble_count)
        ]
    )
    scramble_se = (
        np.zeros(component_count)
        if scramble_count == 1
        else np.std(scramble_means, axis=0, ddof=1) / math.sqrt(scramble_count)
    )
    x1_report = {
        "schema_version": "x1-component-exposure-field@1.0.0",
        "candidate_count": candidate_count,
        "scramble_count": scramble_count,
        "component_count": component_count,
        "component_ids": component_ids,
        "cell_count": len(unique_cells),
        "cell_ids": unique_cells,
        "measure_ess": float(candidate_count),
        "measure_ess_fraction": 1.0,
        "components": component_summary,
        "cell_envelopes": cell_envelopes,
        "proxy_reachability_matrix": cell_reachability,
        "scramble_standard_error_lambda": {
            value: float(scramble_se[index]) for index, value in enumerate(component_ids)
        },
        "design_sequences": design_records,
        "contact_rejection_counts": dict(sorted(rejection_counts.items())),
        "structural_audit": {
            "all_proxy_values_finite_nonnegative": bool(
                np.isfinite(lambdas).all()
                and np.isfinite(energies).all()
                and np.all(lambdas >= 0.0)
                and np.all(energies >= 0.0)
            ),
            "direction_contract_violation_count": int(np.sum(direction_violation)),
            "max_frame_orthonormality_error": float(np.max(frame_error)),
            "max_frame_determinant_error": float(np.max(determinant_error)),
            "max_sphere_boundary_residual_m": float(np.max(sphere_residual)),
            "complete_proxy_domain_fraction": float(
                np.mean(proxy_domain_status == "complete_proxy_domain")
            ),
            "out_of_domain_candidate_count": int(np.sum(out_of_domain_weight > 0.0)),
            "bottom_up_reference_count": 0,
            "pass": bool(
                np.isfinite(lambdas).all()
                and np.isfinite(energies).all()
                and np.all(lambdas >= 0.0)
                and np.all(energies >= 0.0)
                and not np.any(direction_violation)
                and float(np.max(frame_error)) <= 1.0e-10
                and float(np.max(determinant_error)) <= 1.0e-10
                and float(np.max(sphere_residual)) <= 1.0e-9
            ),
        },
        "zero_semantics": {
            "proxy_zero": "proxy_unreached_in_declared_frame",
            "structural_zero_claim": False,
        },
    }

    # Scaled teacher audit.
    proxy_audit_rows = [
        {
            "candidate_id": str(candidate_ids[index]),
            "cell_id": str(cell_ids[index]),
            "measure_weight": 1.0,
            "component_lambda": {
                value: float(lambdas[index, column])
                for column, value in enumerate(component_ids)
            },
            "component_damage_energy_j": {
                value: float(energies[index, column])
                for column, value in enumerate(component_ids)
            },
        }
        for index in teacher_indices
    ]
    teacher_rows = [
        {
            "candidate_id": str(candidate_ids[index]),
            "component_hit_probability": {
                value: float(teacher_hits[position, column])
                for column, value in enumerate(component_ids)
            },
        }
        for position, index in enumerate(teacher_indices)
    ]
    teacher_audit = build_exposure_teacher_audit(proxy_audit_rows, teacher_rows)
    proxy_flat = lambdas[teacher_indices].astype(np.float64).ravel()
    teacher_flat = teacher_hits.ravel()
    teacher_audit["paired_rank_metrics"] = {
        "spearman_proxy_lambda_vs_teacher_hit_probability": float(
            spearmanr(proxy_flat, teacher_flat).statistic
        ),
        "pearson_log1p_proxy_lambda_vs_teacher_hit_probability": float(
            pearsonr(np.log1p(proxy_flat), teacher_flat).statistic
        ),
        "pair_count": int(proxy_flat.size),
    }
    teacher_audit["executed_scale"] = {
        "candidate_count": teacher_count,
        "replicates": teacher_replicates,
        "target_candidate_count": int(teacher_config["preregistered_candidate_count"]),
        "target_replicates": int(teacher_config["preregistered_replicates"]),
        "protocol_deviation": bool(
            teacher_count != int(teacher_config["preregistered_candidate_count"])
            or teacher_replicates != int(teacher_config["preregistered_replicates"])
        ),
        "deviation_reason": teacher_config["deviation_reason"],
    }
    false_rate = teacher_audit["false_unreached_rate"]
    teacher_audit["gates"] = {
        "point_false_unreached_rate_le_3pct": bool(
            false_rate is not None
            and false_rate
            <= float(teacher_config["structural_zero_false_negative_point_gate_max"])
        ),
        "full_preregistered_scale": bool(
            teacher_count == int(teacher_config["preregistered_candidate_count"])
            and teacher_replicates == int(teacher_config["preregistered_replicates"])
        ),
    }
    teacher_audit["scientific_admission_pass"] = all(teacher_audit["gates"].values())

    # X2 independent Reference and exposure-balanced selections.
    support = -np.expm1(-lambdas.astype(np.float64))
    active = np.any(support > 0.0, axis=0)
    active_support = support[:, active]
    strata = [np.flatnonzero(cell_ids == value) for value in unique_cells]
    stratum_sizes = np.asarray([value.size for value in strata], dtype=np.int64)
    quotas = _largest_remainder(stratum_sizes, sample_size, require_each=True)
    exposure_probability = global_marginal_coverage_probabilities(
        active_support,
        strata=strata,
        quotas=quotas.tolist(),
        smoothing=float(x2_config["smoothing"]),
        score_exponent=float(x2_config["score_exponent"]),
        iterations=int(x2_config["iterations"]),
        damping=float(x2_config["damping"]),
        rho_min=float(x2_config["rho_min_requested_v5"]),
    )
    exposure_selected = np.zeros(candidate_count, dtype=bool)
    reference_selected = np.zeros(candidate_count, dtype=bool)
    reference_probability = np.zeros(candidate_count, dtype=np.float64)
    for stratum_index, (indices, quota) in enumerate(zip(strata, quotas, strict=True)):
        exposure_local = pivotal_sample(
            exposure_probability[indices],
            seed=int(x2_config["coverage_selection_seed"]) + stratum_index,
        )
        exposure_selected[indices[exposure_local]] = True
        reference_probability[indices] = float(quota) / indices.size
        reference_local = pivotal_sample(
            reference_probability[indices],
            seed=int(x2_config["reference_selection_seed"]) + stratum_index,
        )
        reference_selected[indices[reference_local]] = True
    exposure_metrics = _arm_metrics(
        lambdas.astype(np.float64),
        exposure_selected,
        exposure_probability,
        reference_lambda,
        component_ids,
        cell_ids,
    )
    reference_metrics = _arm_metrics(
        lambdas.astype(np.float64),
        reference_selected,
        reference_probability,
        reference_lambda,
        component_ids,
        cell_ids,
    )
    common_gates = {
        "exact_2000_or_configured_count": exposure_metrics["selected_count"] == sample_size,
        "unique_selected": exposure_metrics["unique_count"] == sample_size,
        "all_cells_covered": exposure_metrics["covered_cell_count"] == len(unique_cells),
        "ess_fraction": exposure_metrics["design_ess_fraction"]
        >= float(x2_config["ess_fraction_min"]),
        "weight_cv": exposure_metrics["design_weight_cv"]
        <= float(x2_config["weight_cv_max"]),
        "ever": exposure_metrics["support_ever"]
        >= reference_metrics["support_ever"] - 2,
        "at_least_20": exposure_metrics["support_at_least_20"]
        >= reference_metrics["support_at_least_20"],
        "at_least_50": exposure_metrics["support_at_least_50"]
        >= reference_metrics["support_at_least_50"],
        "support_gini": exposure_metrics["support_count_gini"]
        <= reference_metrics["support_count_gini"],
        "exposure_entropy": exposure_metrics["component_exposure_entropy"]
        >= reference_metrics["component_exposure_entropy"],
    }
    original_positivity = exposure_metrics["minimum_selected_inclusion_probability"] >= float(
        x2_config["rho_min_requested_v5"]
    )
    feasible_positivity = exposure_metrics["minimum_selected_inclusion_probability"] >= float(
        x2_config["rho_min_feasible_floor"]
    )
    x2_report = {
        "schema_version": "x2-exposure-balanced-confirmation@1.0.0",
        "selection_role": "train_enrichment_exposure_balanced",
        "reference_replacement": False,
        "outcome_blind": True,
        "candidate_count": candidate_count,
        "sample_size_per_arm": sample_size,
        "cell_quotas": {
            value: int(quotas[index]) for index, value in enumerate(unique_cells)
        },
        "active_component_ids": [
            value for value, keep in zip(component_ids, active, strict=True) if keep
        ],
        "proxy_unreached_component_ids_excluded_from_optimization": [
            value for value, keep in zip(component_ids, active, strict=True) if not keep
        ],
        "reference_arm": reference_metrics,
        "exposure_balanced_arm": exposure_metrics,
        "common_gates": common_gates,
        "original_v5_positivity_gate": original_positivity,
        "feasible_engineering_positivity_gate": feasible_positivity,
        "original_v5_transfer_gate_pass": bool(
            all(common_gates.values()) and original_positivity
        ),
        "revised_engineering_gate_pass": bool(
            all(common_gates.values()) and feasible_positivity
        ),
        "positivity_feasibility_identity": x2_config[
            "positivity_feasibility_identity"
        ],
        "selected_candidate_ids": {
            "reference": candidate_ids[reference_selected].tolist(),
            "exposure_balanced": candidate_ids[exposure_selected].tolist(),
        },
    }

    arrays_path = output / "x1_candidate_exposure.npz"
    np.savez_compressed(
        arrays_path,
        component_ids=np.asarray(component_ids),
        candidate_ids=candidate_ids,
        profile_ids=profile_ids,
        region_ids=region_ids,
        cell_ids=cell_ids,
        scramble_ids=scramble_ids,
        attempt_indices=attempt_indices,
        latent=latents,
        component_lambda=lambdas,
        component_damage_energy_j=energies,
        proxy_domain_status=proxy_domain_status,
        out_of_domain_packet_weight=out_of_domain_weight,
        exposure_inclusion_probability=exposure_probability,
        exposure_selected=exposure_selected,
        reference_inclusion_probability=reference_probability,
        reference_selected=reference_selected,
        teacher_indices=teacher_indices,
        teacher_component_hit_probability=teacher_hits,
    )
    for name, value in (
        ("x1_exposure_field.json", x1_report),
        ("teacher_audit.json", teacher_audit),
        ("x2_selection.json", x2_report),
    ):
        (output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    ledger_path = output / "x2_candidate_ledger.jsonl.gz"
    with gzip.open(ledger_path, "wt", encoding="utf-8", newline="\n") as stream:
        for index in range(candidate_count):
            stream.write(
                json.dumps(
                    {
                        "candidate_id": str(candidate_ids[index]),
                        "cell_id": str(cell_ids[index]),
                        "exposure_inclusion_probability": float(exposure_probability[index]),
                        "exposure_selected": bool(exposure_selected[index]),
                        "reference_inclusion_probability": float(reference_probability[index]),
                        "reference_selected": bool(reference_selected[index]),
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    full_scale = (
        candidate_count == int(x1_config["candidate_count"])
        and scramble_count == int(x1_config["scramble_count"])
        and teacher_count == int(teacher_config["preregistered_candidate_count"])
        and teacher_replicates == int(teacher_config["preregistered_replicates"])
        and sample_size == int(x2_config["sample_size_per_arm"])
        and int(x1_config["proxy_azimuths_per_ring_executed"])
        == int(x1_config["proxy_validated_default_azimuths_per_ring"])
        and int(x1_config["proxy_layer_quadrature_samples_executed"])
        == int(x1_config["proxy_validated_default_layer_quadrature_samples"])
    )
    summary = {
        "schema_version": "x1-x2-experiment-summary@1.0.0",
        "experiment_id": config["experiment_id"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "executed_scale": {
            "candidate_count": candidate_count,
            "scramble_count": scramble_count,
            "teacher_count": teacher_count,
            "teacher_replicates": teacher_replicates,
            "sample_size_per_arm": sample_size,
        },
        "x1_structural_audit_pass": x1_report["structural_audit"]["pass"],
        "teacher_point_gate_pass": teacher_audit["gates"][
            "point_false_unreached_rate_le_3pct"
        ],
        "teacher_scientific_admission_pass": teacher_audit[
            "scientific_admission_pass"
        ],
        "x2_original_v5_transfer_gate_pass": x2_report[
            "original_v5_transfer_gate_pass"
        ],
        "x2_revised_engineering_gate_pass": x2_report[
            "revised_engineering_gate_pass"
        ],
        "full_preregistered_scale_without_deviation": full_scale,
        "overall_preregistered_pass": bool(
            full_scale
            and x1_report["structural_audit"]["pass"]
            and teacher_audit["scientific_admission_pass"]
            and x2_report["original_v5_transfer_gate_pass"]
        ),
        "timings_seconds": timings,
        "evidence_boundary": (
            "Software-teacher and coarse FastSamplingProxy evidence under an "
            "unverified physics registry; proxy non-reach is not structural zero."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    command = " ".join(shlex.quote(value) for value in [sys.executable, *sys.argv])
    (output / "COMMAND.txt").write_text(command + "\n", encoding="utf-8")
    input_paths = {
        "config": config_path,
        "sampling_spec": sampling_spec_path,
        "base_case": base_case_path,
        "target": target_path,
        "physics": physics_path,
        "runner": Path(__file__).resolve(),
    }
    output_paths = {
        "arrays": arrays_path,
        "x1": output / "x1_exposure_field.json",
        "teacher": output / "teacher_audit.json",
        "x2": output / "x2_selection.json",
        "ledger": ledger_path,
        "summary": output / "summary.json",
    }
    hashes = {
        "schema_version": "x1-x2-hash-ledger@1.0.0",
        "engine_code_hash": ENGINE_CODE_HASH,
        "damage_engine_source_tree_hash": source_tree_hash(ROOT / "damage_engine"),
        "sampling_spec_canonical_hash": sampling_spec.hash,
        "target_canonical_hash": main_engine.target_hash,
        "physics_registry_canonical_hash": main_engine.registry.hash,
        "inputs": {name: _raw_sha256(path) for name, path in input_paths.items()},
        "outputs": {name: _raw_sha256(path) for name, path in output_paths.items()},
    }
    (output / "hashes.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_lines = [
        "# X1/X2 execution result",
        "",
        f"- Candidates: {candidate_count:,} across {scramble_count} independent scrambles.",
        f"- Structural audit: {x1_report['structural_audit']['pass']}.",
        f"- Teacher audit: {teacher_count} cases x R{teacher_replicates}; false-unreached rate={teacher_audit['false_unreached_rate']}.",
        f"- Teacher scientific admission: {teacher_audit['scientific_admission_pass']}.",
        f"- X2 original V5-transfer gate: {x2_report['original_v5_transfer_gate_pass']}.",
        f"- X2 revised feasible engineering gate: {x2_report['revised_engineering_gate_pass']}.",
        f"- Overall preregistered pass: {summary['overall_preregistered_pass']}.",
        "",
        "Proxy non-reach remains a finite-frame status, not a structural zero claim.",
    ]
    (output / "RUN_LOG.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
