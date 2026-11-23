#!/usr/bin/env python
"""Run the three-tier anisotropic fast assessment without building a dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.engine import ENGINE_CODE_HASH, DamageEngine  # noqa: E402
from damage_engine.errors import InputValidationError  # noqa: E402
from damage_engine.fast_assessment import (  # noqa: E402
    compare_engine_evidence,
    evaluate_fast_assessment,
)
from damage_engine.fast_sampling_proxy import FastSamplingProxy  # noqa: E402
from damage_engine.registry import PhysicsRegistry  # noqa: E402
from damage_engine.utils import content_hash, file_hash, source_tree_hash  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/sampling/anisotropic_fast_assessment.v1.json"
DEFAULT_TARGET = ROOT / "configs/targets/target.fragment_only.v1.json"
DEFAULT_PHYSICS = ROOT / "configs/physics/loitering_impact.json"


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputValidationError(
            "fast-assessment JSON input is missing or invalid",
            context={"path": str(path)},
        ) from exc


def _json_lines(path: Path) -> list[Any]:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8-sig") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise InputValidationError(
            "fast-assessment JSONL input is missing or invalid",
            context={"path": str(path)},
        ) from exc


def _load_rows(path: Path, *, keys: Sequence[str]) -> list[dict[str, Any]]:
    name = path.name.lower()
    if name.endswith(".jsonl") or name.endswith(".ndjson") or name.endswith(
        ".jsonl.gz"
    ) or name.endswith(".ndjson.gz"):
        value = _json_lines(path)
    else:
        value = _json(path)
        if isinstance(value, Mapping):
            for key in keys:
                if isinstance(value.get(key), list):
                    value = value[key]
                    break
            else:
                value = [value]
    if not isinstance(value, list) or not value or not all(
        isinstance(row, Mapping) for row in value
    ):
        raise InputValidationError("fast-assessment row input must be a nonempty list")
    return [dict(row) for row in value]


def _case_of(row: Mapping[str, Any]) -> Mapping[str, Any]:
    case = row.get("case", row)
    if not isinstance(case, Mapping):
        raise InputValidationError("case row lacks a case mapping")
    return case


def _candidate_id(row: Mapping[str, Any], index: int) -> str:
    case = _case_of(row)
    value = row.get("candidate_id", case.get("case_id", f"case-{index}"))
    if not isinstance(value, str) or not value:
        raise InputValidationError("candidate IDs must be nonempty strings")
    return value


def _case_split(row: Mapping[str, Any]) -> str:
    case = _case_of(row)
    metadata = case.get("metadata", {})
    if isinstance(metadata, Mapping):
        value = metadata.get("frozen_split")
        if isinstance(value, str) and value:
            return value
        sampling = metadata.get("sampling")
        if isinstance(sampling, Mapping):
            value = sampling.get("frozen_split", sampling.get("split"))
            if isinstance(value, str) and value:
                return value
    value = row.get("frozen_split", row.get("split"))
    return value if isinstance(value, str) and value else "unassigned"


def _infer_cell(row: Mapping[str, Any]) -> str:
    value = row.get("cell_id")
    if isinstance(value, str) and value:
        return value
    case = _case_of(row)
    metadata = case.get("metadata", {})
    direction = "unclassified_direction"
    if isinstance(metadata, Mapping):
        for name in (
            "continuous_direction_sampling",
            "sphere_entry_ray_sampling",
            "contact_manifold_sampling",
        ):
            record = metadata.get(name)
            if not isinstance(record, Mapping):
                continue
            candidate = record.get("region_id", record.get("approach_id"))
            if isinstance(candidate, str) and candidate:
                direction = candidate
                break
    munition = case.get("munition", {})
    profile = (
        munition.get("profile_id", "unclassified_profile")
        if isinstance(munition, Mapping)
        else "unclassified_profile"
    )
    return f"{direction}::{profile}"


def _largest_remainder(sizes: np.ndarray, total: int) -> np.ndarray:
    if total <= 0 or sizes.ndim != 1 or sizes.size == 0 or np.any(sizes <= 0):
        raise InputValidationError("invalid stratified sample request")
    if total >= int(np.sum(sizes)):
        return sizes.astype(np.int64, copy=True)
    ideal = total * sizes.astype(np.float64) / float(np.sum(sizes))
    quotas = np.floor(ideal).astype(np.int64)
    if total >= sizes.size:
        quotas = np.maximum(quotas, 1)
    quotas = np.minimum(quotas, sizes)
    while int(np.sum(quotas)) < total:
        available = quotas < sizes
        score = np.where(available, ideal - quotas, -np.inf)
        quotas[int(np.argmax(score))] += 1
    while int(np.sum(quotas)) > total:
        removable = quotas > (1 if total >= sizes.size else 0)
        score = np.where(removable, quotas - ideal, -np.inf)
        quotas[int(np.argmax(score))] -= 1
    return quotas


def _stratified_indices(
    cell_ids: Sequence[str], *, count: int, seed: int
) -> np.ndarray:
    cells = np.asarray([str(value) for value in cell_ids])
    if count >= cells.size:
        return np.arange(cells.size, dtype=np.int64)
    unique = sorted(set(cells.tolist()))
    strata = [np.flatnonzero(cells == value) for value in unique]
    quotas = _largest_remainder(
        np.asarray([indices.size for indices in strata], dtype=np.int64), count
    )
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for indices, quota in zip(strata, quotas, strict=True):
        selected.extend(
            int(value)
            for value in rng.choice(indices, size=int(quota), replace=False)
        )
    return np.asarray(sorted(selected), dtype=np.int64)


def _subset_rows(
    rows: Sequence[Mapping[str, Any]], indices: np.ndarray
) -> list[dict[str, Any]]:
    return [dict(rows[int(index)]) for index in indices]


def _core_config(config: Mapping[str, Any]) -> dict[str, Any]:
    proxy = config.get("proxy", {})
    calibration = config.get("engine_calibration", {})
    generator = config.get("generator", {})
    return {
        "rank_relative_tolerance": float(
            generator.get("rank_relative_tolerance", 1.0e-10)
        ),
        "nullspace_tolerance": float(
            generator.get("nullspace_tolerance", 1.0e-10)
        ),
        "occupancy_bins": generator.get("occupancy_bins"),
        "reach_tolerance": float(proxy.get("reach_tolerance", 0.0)),
        "ece_bins": int(calibration.get("ece_bins", 10)),
        "teacher_positive_threshold": float(
            calibration.get("teacher_positive_threshold", 0.0)
        ),
        "confidence": float(calibration.get("confidence", 0.95)),
        "expected_cell_mass": generator.get("expected_cell_mass"),
        "expected_cell_mass_semantics": generator.get(
            "expected_cell_mass_semantics"
        ),
    }


def _selection(path: Path | None, arm: str | None) -> tuple[Any, Any]:
    if path is None:
        return None, None
    value = _json(path)
    if not isinstance(value, Mapping):
        raise InputValidationError("selection input must be a JSON object")
    selected = value.get("selected_candidate_ids")
    if isinstance(selected, Mapping):
        if arm is None:
            raise InputValidationError(
                "--selection-arm is required when selected_candidate_ids contains arms"
            )
        selected = selected.get(arm)
    probabilities = value.get("inclusion_probabilities")
    if probabilities is None and arm is not None:
        probabilities = value.get(f"{arm}_inclusion_probabilities")
    if not isinstance(selected, list) or not isinstance(probabilities, Mapping):
        raise InputValidationError(
            "selection requires selected_candidate_ids and inclusion_probabilities"
        )
    return [str(item) for item in selected], {
        str(key): float(item) for key, item in probabilities.items()
    }


def _load_npz_frame(
    path: Path,
    *,
    max_cases: int | None,
    seed: int,
    selection: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], Any]:
    try:
        archive = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise InputValidationError(
            "precomputed NPZ is unreadable", context={"path": str(path)}
        ) from exc
    required = {
        "candidate_ids",
        "cell_ids",
        "latent",
        "component_ids",
        "component_lambda",
        "component_damage_energy_j",
        "proxy_domain_status",
    }
    missing = sorted(required.difference(archive.files))
    if missing:
        archive.close()
        raise InputValidationError(
            "precomputed NPZ lacks required arrays", context={"missing": missing}
        )
    candidate_ids = archive["candidate_ids"].astype(str)
    cell_ids = archive["cell_ids"].astype(str)
    latent = np.asarray(archive["latent"], dtype=np.float64)
    component_ids = archive["component_ids"].astype(str)
    lambdas = np.asarray(archive["component_lambda"], dtype=np.float64)
    energies = np.asarray(
        archive["component_damage_energy_j"], dtype=np.float64
    )
    domain = archive["proxy_domain_status"].astype(str)
    count = candidate_ids.size
    if (
        cell_ids.shape != (count,)
        or domain.shape != (count,)
        or latent.ndim != 2
        or latent.shape[0] != count
        or lambdas.shape != (count, component_ids.size)
        or energies.shape != lambdas.shape
        or not np.isfinite(latent).all()
        or not np.isfinite(lambdas).all()
        or not np.isfinite(energies).all()
        or np.any(lambdas < 0.0)
        or np.any(energies < 0.0)
    ):
        archive.close()
        raise InputValidationError("precomputed NPZ arrays are invalid or misaligned")
    if selection == "teacher":
        if "teacher_indices" not in archive.files:
            archive.close()
            raise InputValidationError("precomputed NPZ lacks teacher_indices")
        indices = np.asarray(archive["teacher_indices"], dtype=np.int64)
    elif selection == "all" or max_cases is None or max_cases >= count:
        indices = np.arange(count, dtype=np.int64)
    else:
        indices = _stratified_indices(cell_ids, count=max_cases, seed=seed)
    if indices.ndim != 1 or indices.size == 0 or np.any(indices < 0) or np.any(
        indices >= count
    ):
        archive.close()
        raise InputValidationError("precomputed NPZ selection indices are invalid")
    rows: list[dict[str, Any]] = []
    exposure: list[dict[str, Any]] = []
    for index in indices:
        position = int(index)
        identifier = str(candidate_ids[position])
        cell = str(cell_ids[position])
        rows.append(
            {
                "candidate_id": identifier,
                "cell_id": cell,
                "measure_weight": 1.0,
                "latent": latent[position].tolist(),
                "case": {"case_id": identifier},
            }
        )
        exposure.append(
            {
                "candidate_id": identifier,
                "cell_id": cell,
                "measure_weight": 1.0,
                "proxy_domain_status": str(domain[position]),
                "component_lambda": {
                    component_id: float(lambdas[position, column])
                    for column, component_id in enumerate(component_ids)
                },
                "component_damage_energy_j": {
                    component_id: float(energies[position, column])
                    for column, component_id in enumerate(component_ids)
                },
            }
        )
    audit = {
        "source_candidate_count": int(count),
        "used_candidate_count": int(indices.size),
        "selection": selection,
        "selection_seed": int(seed),
        "selection_indices_hash": content_hash(indices.tolist()),
        "source_cell_counts": dict(sorted(Counter(cell_ids.tolist()).items())),
        "used_cell_counts": dict(
            sorted(Counter(cell_ids[indices].tolist()).items())
        ),
        "cache_identity": _cache_identity(path),
    }
    return rows, exposure, audit, archive


def _cache_identity(path: Path) -> dict[str, Any]:
    ledger_path = path.parent / "hashes.json"
    current_tree = source_tree_hash(ROOT / "damage_engine")
    current_file = file_hash(path)
    if not ledger_path.is_file():
        return {
            "status": "source_ledger_not_available",
            "current_damage_engine_source_tree_hash": current_tree,
            "current_engine_code_hash": ENGINE_CODE_HASH,
            "current_npz_sha256": current_file,
        }
    ledger = _json(ledger_path)
    if not isinstance(ledger, Mapping):
        raise InputValidationError("cache hashes.json must be an object")
    recorded_tree = ledger.get("damage_engine_source_tree_hash")
    recorded_engine = ledger.get("engine_code_hash")
    outputs = ledger.get("outputs", {})
    recorded_file = outputs.get("arrays") if isinstance(outputs, Mapping) else None
    return {
        "status": (
            "exact_execution_identity"
            if recorded_tree == current_tree
            and recorded_engine == ENGINE_CODE_HASH
            and recorded_file == current_file
            else "historical_cache_current_source_differs"
        ),
        "recorded_damage_engine_source_tree_hash": recorded_tree,
        "current_damage_engine_source_tree_hash": current_tree,
        "damage_engine_source_tree_match": recorded_tree == current_tree,
        "recorded_engine_code_hash": recorded_engine,
        "current_engine_code_hash": ENGINE_CODE_HASH,
        "engine_code_hash_match": recorded_engine == ENGINE_CODE_HASH,
        "recorded_npz_sha256": recorded_file,
        "current_npz_sha256": current_file,
        "npz_hash_match": recorded_file == current_file,
        "ledger_path": str(ledger_path),
    }


def _filter_cases(
    rows: Sequence[Mapping[str, Any]], allowed: set[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split_counts = Counter(_case_split(row) for row in rows)
    kept = [dict(row) for row in rows if _case_split(row) in allowed]
    included_counts = Counter(_case_split(row) for row in kept)
    return kept, {
        "source_split_counts": dict(sorted(split_counts.items())),
        "included_split_counts": dict(sorted(included_counts.items())),
        "allowed_splits": sorted(allowed),
        "test_case_count_included": int(included_counts.get("test", 0)),
    }


def _status(assessment: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    selection = assessment["tier_1_generator"]["selection_comparison"]
    tier_1 = assessment["tier_1_generator"]
    generator_config = config.get("generator", {})
    marginal_ks = [
        float(item["uniform_ks"])
        for item in tier_1["reference"]["global"]["uniform_marginals"]
    ]
    maximum_ks = max(marginal_ks)
    maximum_ks_gate = float(generator_config.get("maximum_uniform_marginal_ks", 0.1))
    declared = tier_1.get("declared_cell_comparison", {"status": "not_requested"})
    maximum_declared_tv = float(
        generator_config.get("maximum_expected_cell_total_variation", 0.1)
    )
    declared_gate = (
        True
        if declared.get("status") != "evaluated"
        else float(declared["total_variation"]) <= maximum_declared_tv
    )
    global_ball = tier_1["reference"]["global"]
    cell_balls = tier_1["reference"]["cells"]
    minimum_cell_count = int(generator_config.get("minimum_cell_count", 1))
    require_full_cell_rank = bool(
        generator_config.get("require_full_cell_rank", False)
    )
    latent_dimension = int(global_ball["dimension"])
    undersized_cells = {
        cell: int(ball["count"])
        for cell, ball in cell_balls.items()
        if int(ball["count"]) < minimum_cell_count
    }
    rank_deficient_cells = {
        cell: int(ball["effective_rank"])
        for cell, ball in cell_balls.items()
        if int(ball["effective_rank"]) < latent_dimension
    }
    cell_count_gate = not undersized_cells
    cell_rank_gate = not require_full_cell_rank or not rank_deficient_cells
    generator_checks: dict[str, Any] = {
        "selection_evaluated": False,
        "maximum_uniform_marginal_ks": maximum_ks,
        "maximum_uniform_marginal_ks_threshold": maximum_ks_gate,
        "uniform_marginal_ks_gate_pass": maximum_ks <= maximum_ks_gate,
        "declared_cell_comparison_evaluated": declared.get("status") == "evaluated",
        "declared_cell_total_variation": declared.get("total_variation"),
        "maximum_expected_cell_total_variation": maximum_declared_tv,
        "declared_cell_mass_gate_pass": declared_gate,
        "minimum_cell_count": minimum_cell_count,
        "cell_count_gate_pass": cell_count_gate,
        "undersized_cells": undersized_cells,
        "latent_dimension": latent_dimension,
        "require_full_cell_rank": require_full_cell_rank,
        "cell_rank_gate_pass": cell_rank_gate,
        "rank_deficient_cells": rank_deficient_cells,
    }
    generator_pass = bool(
        maximum_ks <= maximum_ks_gate
        and declared_gate
        and cell_count_gate
        and cell_rank_gate
    )
    if selection.get("status") == "evaluated":
        recovered = selection["inverse_probability_recovered"]
        maximum_tv = float(
            config.get("generator", {}).get("maximum_cell_total_variation", 0.1)
        )
        minimum_ess = float(
            config.get("generator", {}).get("minimum_measure_ess_fraction", 0.5)
        )
        generator_checks.update({
            "selection_evaluated": True,
            "cell_mass_tv_le_threshold": recovered["cell_mass_total_variation"]
            <= maximum_tv,
            "ess_fraction_ge_threshold": recovered["design_ess_fraction"]
            >= minimum_ess,
            "maximum_cell_total_variation": maximum_tv,
            "minimum_measure_ess_fraction": minimum_ess,
        })
        generator_pass = bool(
            generator_pass
            and generator_checks["cell_mass_tv_le_threshold"]
            and generator_checks["ess_fraction_ge_threshold"]
        )
    generator_status = (
        "pass_diagnostic_gates"
        if generator_pass
        else (
            "declared_to_realized_distribution_gap"
            if not (maximum_ks <= maximum_ks_gate and declared_gate)
            else "insufficient_cell_support"
        )
    )
    domain = assessment["tier_2_geometry_proxy"]["proxy_domain_coverage"]
    minimum_complete = float(
        config.get("proxy", {}).get("minimum_complete_domain_fraction", 0.9)
    )
    proxy_pass = domain["complete_case_fraction"] >= minimum_complete
    return {
        "generator_status": generator_status,
        "generator_checks": generator_checks,
        "proxy_status": (
            "pass_diagnostic_domain_gate"
            if proxy_pass
            else "partial_proxy_domain_warning"
        ),
        "proxy_checks": {
            "complete_domain_fraction": domain["complete_case_fraction"],
            "minimum_complete_domain_fraction": minimum_complete,
            "gate_pass": proxy_pass,
        },
        "engine_status": "not_evaluated",
        "combined_status": "tiers_reported_independently",
    }


def _rehash_assessment(assessment: dict[str, Any]) -> None:
    assessment.pop("assessment_hash", None)
    assessment["assessment_hash"] = content_hash(assessment)


def _strict_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _prepare_output(directory: Path) -> None:
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {directory}")
    directory.mkdir(parents=True, exist_ok=False)


def _command() -> str:
    return subprocess.list2cmdline([sys.executable, *sys.argv])


def _quick_report(assessment: Mapping[str, Any], elapsed: float) -> str:
    diagnosis = assessment["diagnosis"]
    tier_1 = assessment["tier_1_generator"]
    global_ball = tier_1["reference"]["global"]
    tier_2 = assessment["tier_2_geometry_proxy"]
    domain = tier_2["proxy_domain_coverage"]
    components = tier_2["exposure_field"]["components"]
    reached = sum(
        value["reachability_status"] == "proxy_reached_in_declared_frame"
        for value in components.values()
    )
    maximum_ks = max(
        item["uniform_ks"] for item in global_ball["uniform_marginals"]
    )
    declared = tier_1.get("declared_cell_comparison", {"status": "not_requested"})
    declared_line = (
        "- 声明单元质量对照：未提供。"
        if declared.get("status") != "evaluated"
        else f"- 声明→实现单元质量 TV：{declared['total_variation']:.6f}。"
    )
    return "\n".join(
        [
            "# 各向异性三级快速评估",
            "",
            f"- 工况数：{assessment['candidate_count']:,}",
            f"- 运行时间：{elapsed:.3f} s",
            f"- L1 生成器状态：`{diagnosis['generator_status']}`",
            f"- L2 代理状态：`{diagnosis['proxy_status']}`",
            "- L3 引擎状态：`not_evaluated`（快速入口未读取教师证据）",
            "",
            "## L1：生成器各向异性场",
            "",
            f"- 潜变量维数：{global_ball['dimension']}；经验椭球有效秩：{global_ball['effective_rank']}。",
            f"- 最大加权均匀边际 KS：{maximum_ks:.6f}。",
            declared_line,
            f"- 测度 ESS fraction：{global_ball['measure_ess'] / assessment['candidate_count']:.6f}。",
            "",
            "## L2：目标暴露场",
            "",
            f"- 代理完整域比例：{domain['complete_case_fraction']:.6%}。",
            f"- 当前有限帧中代理到达部件：{reached}/{len(components)}。",
            "- 代理零值保持 `proxy_unreached_in_declared_frame`，不声明结构零。",
            "",
            "## 证据边界",
            "",
            "L1 只诊断前验生成器；L2 是方向性轻量代理；L3 必须由独立的同工况软件教师证据触发。以上结果不是正式标签或外部物理验证。",
            "",
        ]
    )


def _write_quick_bundle(
    output: Path,
    assessment: dict[str, Any],
    *,
    report: str,
    manifest: dict[str, Any],
    input_paths: Mapping[str, Path],
) -> None:
    assessment_path = output / "assessment.json"
    report_path = output / "REPORT.md"
    manifest_path = output / "manifest.json"
    command_path = output / "COMMAND.txt"
    assessment_path.write_text(_strict_json(assessment), encoding="utf-8")
    report_path.write_text(report, encoding="utf-8")
    manifest_path.write_text(_strict_json(manifest), encoding="utf-8")
    command_path.write_text(_command() + "\n", encoding="utf-8")
    hashes = {
        "schema_version": "anisotropic-fast-assessment-hashes@1.0.0",
        "inputs": {
            name: file_hash(path) for name, path in sorted(input_paths.items())
        },
        "outputs": {
            path.name: file_hash(path)
            for path in (assessment_path, report_path, manifest_path, command_path)
        },
    }
    (output / "hashes.json").write_text(_strict_json(hashes), encoding="utf-8")


def _quick(args: argparse.Namespace) -> int:
    started = perf_counter()
    config = _json(args.config)
    if not isinstance(config, Mapping):
        raise InputValidationError("fast-assessment config must be an object")
    max_cases = int(args.max_cases or config.get("default_max_cases", 4096))
    if max_cases < 1:
        raise InputValidationError("--max-cases must be positive")
    seed = int(config.get("precomputed_selection", {}).get("seed", 2026112301))
    split_audit: dict[str, Any]
    source_audit: dict[str, Any]
    archive = None
    proxy = None
    live_proxy_identity: dict[str, Any] | None = None
    if args.precomputed_npz is not None:
        case_rows, exposure_rows, source_audit, archive = _load_npz_frame(
            args.precomputed_npz,
            max_cases=max_cases,
            seed=seed,
            selection=args.npz_selection,
        )
        split_audit = {
            "source_split_counts": {"precomputed_unassigned": source_audit["source_candidate_count"]},
            "included_split_counts": {"precomputed_unassigned": source_audit["used_candidate_count"]},
            "allowed_splits": ["precomputed_unassigned"],
            "test_case_count_included": 0,
        }
    else:
        source_rows = _load_rows(args.cases, keys=("cases", "case_rows", "rows"))
        allowed = {item.strip() for item in args.splits.split(",") if item.strip()}
        allowed.add("unassigned")
        case_rows, split_audit = _filter_cases(source_rows, allowed)
        if not case_rows:
            raise InputValidationError("no case rows remain after split filtering")
        cells = [_infer_cell(row) for row in case_rows]
        indices = _stratified_indices(
            cells, count=min(max_cases, len(case_rows)), seed=seed
        )
        case_rows = _subset_rows(case_rows, indices)
        source_audit = {
            "source_candidate_count": len(source_rows),
            "used_candidate_count": len(case_rows),
            "selection": "proportional_cell_stratified",
            "selection_seed": seed,
            "selection_indices_hash": content_hash(indices.tolist()),
        }
        if args.exposure_rows is not None:
            all_exposure = _load_rows(
                args.exposure_rows,
                keys=("candidate_rows", "exposure_rows", "rows"),
            )
            wanted = {
                _candidate_id(row, index)
                for index, row in enumerate(case_rows)
            }
            exposure_rows = [
                row for row in all_exposure if str(row.get("candidate_id")) in wanted
            ]
        else:
            target = _json(args.target)
            engine = DamageEngine(target, PhysicsRegistry.from_file(args.physics))
            proxy_config = config.get("proxy", {})
            proxy = FastSamplingProxy(
                engine,
                source_seed=int(proxy_config.get("source_seed", 2026112001)),
                azimuths_per_ring=int(proxy_config.get("azimuths_per_ring", 4)),
                reach_probability_cutoff=float(
                    proxy_config.get("reach_probability_cutoff", 1.0e-5)
                ),
                component_support_threshold=float(
                    proxy_config.get("component_support_threshold", 0.05)
                ),
                layer_quadrature_samples=int(
                    proxy_config.get("layer_quadrature_samples", 16)
                ),
            )
            live_proxy_identity = {
                "target_hash": engine.target_hash,
                "physics_registry_hash": engine.registry.hash,
                "engine_code_hash": ENGINE_CODE_HASH,
                "damage_engine_source_tree_hash": source_tree_hash(
                    ROOT / "damage_engine"
                ),
                "source_seed": proxy.source_seed,
                "azimuths_per_ring": proxy.azimuths_per_ring,
                "layer_quadrature_samples": proxy.layer_quadrature_samples,
                "output_scope": "exposure",
            }
            exposure_rows = None
    selected_ids, probabilities = _selection(args.selection, args.selection_arm)
    try:
        assessment = evaluate_fast_assessment(
            case_rows,
            proxy=proxy,
            exposure_rows=exposure_rows,
            selected_ids=selected_ids,
            inclusion_probabilities=probabilities,
            config=_core_config(config),
        )
    finally:
        if archive is not None:
            archive.close()
    assessment["method_id"] = config.get("method_id")
    assessment["provenance"] = {
        "config_hash": content_hash(config),
        "source_audit": source_audit,
        "split_audit": split_audit,
        "teacher_executed": False,
        "formal_dataset_built": False,
        "sealed_test_result_rows_decoded": 0,
        "live_proxy_identity": live_proxy_identity,
    }
    assessment["diagnosis"] = _status(assessment, config)
    _rehash_assessment(assessment)
    elapsed = perf_counter() - started
    assessment["runtime"] = {
        "wall_seconds": elapsed,
        "cases_per_second": assessment["candidate_count"] / elapsed,
    }
    _rehash_assessment(assessment)
    _prepare_output(args.output_dir)
    input_paths = {"config": args.config}
    if args.precomputed_npz is not None:
        input_paths["precomputed_npz"] = args.precomputed_npz
    else:
        input_paths["cases"] = args.cases
        if args.exposure_rows is not None:
            input_paths["exposure_rows"] = args.exposure_rows
        else:
            input_paths.update({"target": args.target, "physics": args.physics})
    if args.selection is not None:
        input_paths["selection"] = args.selection
    manifest = {
        "schema_version": "anisotropic-fast-assessment-manifest@1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "quick",
        "candidate_count": assessment["candidate_count"],
        "source_audit": source_audit,
        "split_audit": split_audit,
        "assessment_hash": assessment["assessment_hash"],
        "outputs": ["assessment.json", "REPORT.md", "manifest.json", "COMMAND.txt", "hashes.json"],
    }
    _write_quick_bundle(
        args.output_dir,
        assessment,
        report=_quick_report(assessment, elapsed),
        manifest=manifest,
        input_paths=input_paths,
    )
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "candidate_count": assessment["candidate_count"],
                "wall_seconds": elapsed,
                "generator_status": assessment["diagnosis"]["generator_status"],
                "proxy_status": assessment["diagnosis"]["proxy_status"],
                "engine_status": "not_evaluated",
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


def _teacher_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    value = _json(path)
    metadata: dict[str, Any] = {}
    if isinstance(value, Mapping):
        recorded_hash = value.get("report_hash")
        if recorded_hash is not None:
            unhashed = dict(value)
            unhashed.pop("report_hash", None)
            actual_hash = content_hash(unhashed)
            if recorded_hash != actual_hash:
                raise InputValidationError(
                    "teacher evidence report_hash does not match its content"
                )
            metadata["evidence_integrity"] = {
                "status": "verified",
                "report_hash": recorded_hash,
            }
        else:
            metadata["evidence_integrity"] = {
                "status": "not_declared",
                "report_hash": None,
            }
        for key in (
            "source",
            "source_audit",
            "identity_comparison",
            "provenance",
            "evidence_boundary",
        ):
            if key in value:
                metadata[key] = value[key]
    def audit(value: Any, path_parts: tuple[str, ...] = ()) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                normalized = str(key).strip().lower()
                location = ".".join((*path_parts, str(key)))
                if normalized in {
                    "test_result_json_lines_decoded",
                    "sealed_test_result_rows_decoded",
                    "test_open_count_increment",
                } and int(item or 0) != 0:
                    raise InputValidationError(
                        f"teacher evidence reports opened/decoded test data at {location}"
                    )
                if normalized in {
                    "test_metrics_computed",
                    "sealed_test_opened",
                    "test_opened",
                } and bool(item):
                    raise InputValidationError(
                        f"teacher evidence reports test use at {location}"
                    )
                audit(item, (*path_parts, str(key)))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                audit(item, (*path_parts, str(index)))

    audit(metadata)
    return _load_rows(path, keys=("teacher_rows", "rows")), metadata


def _calibration_report(result: Mapping[str, Any], elapsed: float) -> str:
    overall = result["overall"]
    reach = result["reachability_audit"]
    diagnosis = result["diagnosis"]
    return "\n".join(
        [
            "# 各向异性快速评估：L3 引擎校准",
            "",
            f"- 配对工况：{len(result['paired_candidate_ids'])}",
            f"- 工况—部件对：{overall['pair_count']}",
            f"- 运行时间：{elapsed:.3f} s",
            f"- 引擎差距状态：`{diagnosis['engine_status']}`",
            "",
            "## 配对差距",
            "",
            f"- 加权概率 MSE（兼容字段 weighted_brier）：{overall['weighted_probability_mse']:.6f}",
            f"- weighted ECE：{overall['weighted_ece']:.6f}",
            f"- weighted MAE：{overall['weighted_mae']:.6f}",
            f"- weighted RMSE：{overall['weighted_rmse']:.6f}",
            f"- Spearman：{overall['spearman']}",
            f"- false-unreached：{reach['false_unreached_rate']}",
            f"- false-unreached Wilson 上界：{reach['false_unreached_wilson_upper']}",
            f"- 最差单元：{result['worst_cell']['cell_id']}（RMSE={result['worst_cell']['weighted_rmse']:.6f}）",
            "",
            "该层只量化快速代理与当前软件教师的差距，不是外部物理校准。",
            "",
        ]
    )


def _calibrate(args: argparse.Namespace) -> int:
    started = perf_counter()
    config = _json(args.config)
    if not isinstance(config, Mapping):
        raise InputValidationError("fast-assessment config must be an object")
    source_assessment = _json(args.assessment)
    if not isinstance(source_assessment, Mapping):
        raise InputValidationError("assessment must be a JSON object")
    recorded_assessment_hash = source_assessment.get("assessment_hash")
    unhashed_assessment = dict(source_assessment)
    unhashed_assessment.pop("assessment_hash", None)
    if (
        not isinstance(recorded_assessment_hash, str)
        or recorded_assessment_hash != content_hash(unhashed_assessment)
    ):
        raise InputValidationError(
            "assessment_hash does not match the assessment content"
        )
    temporary_assessment = False
    source_audit: dict[str, Any] = {}
    teacher_source_audit: dict[str, Any] = {}
    paired_assessment = dict(source_assessment)
    teachers, teacher_source_audit = _teacher_rows(args.teacher_evidence)
    result = compare_engine_evidence(
        paired_assessment,
        teachers,
        split_role=args.split_role,
        config=_core_config(config),
    )
    thresholds = config.get("engine_calibration", {})
    maximum_false = float(
        thresholds.get("maximum_false_unreached_wilson_upper", 0.03)
    )
    minimum_spearman = float(
        thresholds.get("minimum_proxy_teacher_spearman", 0.8)
    )
    wilson = result["reachability_audit"]["false_unreached_wilson_upper"]
    spearman = result["overall"]["spearman"]
    false_gate = wilson is not None and wilson <= maximum_false
    rank_gate = spearman is not None and spearman >= minimum_spearman
    result["diagnosis"] = {
        "engine_status": (
            "pass_diagnostic_gates"
            if false_gate and rank_gate
            else "proxy_teacher_gap_detected"
        ),
        "false_unreached_gate_pass": false_gate,
        "rank_gate_pass": rank_gate,
        "maximum_false_unreached_wilson_upper": maximum_false,
        "minimum_proxy_teacher_spearman": minimum_spearman,
        "generator_status": source_assessment.get("diagnosis", {}).get(
            "generator_status", "reported_in_source_assessment"
        ),
        "proxy_status": source_assessment.get("diagnosis", {}).get(
            "proxy_status", "reported_in_source_assessment"
        ),
    }
    result["provenance"] = {
        "source_assessment_hash": source_assessment.get("assessment_hash"),
        "paired_assessment_hash": paired_assessment.get("assessment_hash"),
        "temporary_teacher_subset_assessment": temporary_assessment,
        "precomputed_source_audit": source_audit,
        "teacher_source_audit": teacher_source_audit,
        "teacher_executed": False,
        "sealed_test_result_rows_decoded": 0,
    }
    result.pop("comparison_hash", None)
    result["comparison_hash"] = content_hash(result)
    elapsed = perf_counter() - started
    result["runtime"] = {"wall_seconds": elapsed}
    result.pop("comparison_hash", None)
    result["comparison_hash"] = content_hash(result)
    _prepare_output(args.output_dir)
    result_path = args.output_dir / "engine_calibration.json"
    report_path = args.output_dir / "REPORT.md"
    manifest_path = args.output_dir / "manifest.json"
    command_path = args.output_dir / "COMMAND.txt"
    result_path.write_text(_strict_json(result), encoding="utf-8")
    report_path.write_text(_calibration_report(result, elapsed), encoding="utf-8")
    manifest = {
        "schema_version": "anisotropic-engine-calibration-manifest@1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "calibrate",
        "paired_candidate_count": len(result["paired_candidate_ids"]),
        "comparison_hash": result["comparison_hash"],
        "temporary_teacher_subset_assessment": temporary_assessment,
        "outputs": ["engine_calibration.json", "REPORT.md", "manifest.json", "COMMAND.txt", "hashes.json"],
    }
    manifest_path.write_text(_strict_json(manifest), encoding="utf-8")
    command_path.write_text(_command() + "\n", encoding="utf-8")
    input_paths = {"assessment": args.assessment, "config": args.config}
    input_paths["teacher_evidence"] = args.teacher_evidence
    hashes = {
        "schema_version": "anisotropic-engine-calibration-hashes@1.0.0",
        "inputs": {
            name: file_hash(path) for name, path in sorted(input_paths.items())
        },
        "outputs": {
            path.name: file_hash(path)
            for path in (result_path, report_path, manifest_path, command_path)
        },
    }
    (args.output_dir / "hashes.json").write_text(
        _strict_json(hashes), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "paired_candidate_count": len(result["paired_candidate_ids"]),
                "wall_seconds": elapsed,
                "engine_status": result["diagnosis"]["engine_status"],
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "quick:     run_anisotropic_fast_assessment.py quick --cases CASES "
            "--output-dir DIR\n"
            "calibrate: run_anisotropic_fast_assessment.py calibrate "
            "--assessment DIR/assessment.json --teacher-evidence ROWS.json "
            "--output-dir DIR"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    quick = subparsers.add_parser(
        "quick",
        help="run outcome-blind L1 generator and L2 proxy assessment",
        description=(
            "Evaluate generator coverage and anisotropic exposure without "
            "building a formal dataset or running a teacher."
        ),
    )
    source = quick.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--cases", type=Path, help="JSON/JSONL(.gz) case rows; test is excluded"
    )
    source.add_argument(
        "--precomputed-npz",
        type=Path,
        help="outcome-blind cached exposure frame for L1/L2 replay",
    )
    quick.add_argument(
        "--exposure-rows",
        type=Path,
        help="precomputed exposure JSON aligned to --cases",
    )
    quick.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="diagnostic config"
    )
    quick.add_argument(
        "--target", type=Path, default=DEFAULT_TARGET, help="target definition"
    )
    quick.add_argument(
        "--physics", type=Path, default=DEFAULT_PHYSICS, help="physics registry"
    )
    quick.add_argument(
        "--max-cases", type=int, help="stratified maximum; config default if omitted"
    )
    quick.add_argument(
        "--splits",
        default="train,validation",
        help="comma-separated non-test splits; unassigned is always allowed",
    )
    quick.add_argument(
        "--npz-selection", choices=("stratified", "teacher", "all"), default="stratified"
    )
    quick.add_argument(
        "--selection", type=Path, help="selected IDs and inclusion probabilities"
    )
    quick.add_argument("--selection-arm", help="arm name in a multi-arm selection")
    quick.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new output directory; existing paths are rejected",
    )
    quick.set_defaults(handler=_quick)

    calibrate = subparsers.add_parser(
        "calibrate",
        help="compare an assessment with existing role-labelled teacher evidence",
        description=(
            "Compare fixed quick-assessment cases with existing non-test teacher "
            "rows. The command never executes a teacher."
        ),
    )
    calibrate.add_argument(
        "--assessment", type=Path, required=True, help="quick assessment.json"
    )
    calibrate.add_argument(
        "--teacher-evidence",
        type=Path,
        required=True,
        help="JSON teacher rows with explicit non-test role/provenance",
    )
    calibrate.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="diagnostic config"
    )
    calibrate.add_argument(
        "--split-role",
        default="validation",
        help=(
            "aggregate role label; must describe every explicit row role, e.g. "
            "validation or development_train_validation"
        ),
    )
    calibrate.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new output directory; existing paths are rejected",
    )
    calibrate.set_defaults(handler=_calibrate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "quick" and args.precomputed_npz is not None and args.exposure_rows:
        raise InputValidationError(
            "--exposure-rows cannot be combined with --precomputed-npz"
        )
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
