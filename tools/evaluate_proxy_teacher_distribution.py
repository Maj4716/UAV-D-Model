#!/usr/bin/env python
"""Sealed-test-safe proxy/teacher distribution comparisons.

The evaluator deliberately decodes teacher result rows only for frozen train
and validation splits.  Test result lines are skipped before JSON decoding so
the report cannot become an accidental test-set opening.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
import math
import argparse
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.distribution_metrics import (
    jensen_shannon_binned,
    rbf_mmd2,
    wasserstein1_log_positive,
    weighted_calibration,
)
from damage_engine.errors import InputValidationError
from damage_engine.fast_sampling_proxy import FastSamplingProxy
from damage_engine.hurdle_targets import extract_component_hurdle_targets
from damage_engine.utils import content_hash, file_hash
from damage_engine.engine import DamageEngine
from damage_engine.registry import PhysicsRegistry


REPORT_VERSION = "proxy-teacher-hurdle-distribution@1.0.0"
_DEVELOPMENT_SPLITS = frozenset({"train", "validation"})
_P_HIT_JS_EDGES = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.000000000001)


@dataclass(frozen=True)
class DevelopmentEvidence:
    case: dict[str, Any]
    result: dict[str, Any]
    split: str
    design_weight: float


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputValidationError(
            "proxy/teacher report input JSON is missing or unreadable",
            context={"path": str(path)},
        ) from exc
    if not isinstance(value, dict):
        raise InputValidationError("proxy/teacher report input JSON must be an object")
    return value


def _dataset_directory(pilot: Path) -> Path:
    nested = pilot / "formal_dataset"
    if (nested / "manifest.json").is_file() and (nested / "shards").is_dir():
        return nested
    if (pilot / "manifest.json").is_file() and (pilot / "shards").is_dir():
        return pilot
    raise InputValidationError(
        "pilot does not contain a completed formal sharded dataset",
        context={"pilot": str(pilot)},
    )


def _formal_case_paths(pilot: Path, dataset: Path) -> list[Path]:
    roots = [pilot, dataset.parent]
    if dataset.name == "formal_dataset":
        roots.append(dataset.parent)
    paths: list[Path] = []
    for root in roots:
        paths.extend(sorted((root / "rounds").glob("round-*/formal_cases.jsonl.gz")))
    unique = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    if not unique:
        raise InputValidationError("pilot formal-case ledger is missing")
    return unique


def _load_case_map(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for path in paths:
        try:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    case = json.loads(line)
                    case_id = str(case.get("case_id", ""))
                    if not case_id or case_id in cases:
                        raise ValueError("formal case IDs must be nonempty and unique")
                    cases[case_id] = case
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise InputValidationError(
                "formal-case ledger is unreadable or invalid",
                context={"path": str(path)},
            ) from exc
    return cases


def load_development_evidence(
    pilot_directory: str | Path,
) -> tuple[list[DevelopmentEvidence], dict[str, Any]]:
    """Read train/validation evidence without decoding any test result row."""

    pilot = Path(pilot_directory)
    dataset = _dataset_directory(pilot)
    manifest_path = dataset / "manifest.json"
    manifest = _json(manifest_path)
    cases = _load_case_map(_formal_case_paths(pilot, dataset))
    evidence: list[DevelopmentEvidence] = []
    seen_ids: set[str] = set()
    test_count = 0
    decoded_test_count = 0
    split_counts: dict[str, int] = {}
    shard_count = 0
    for shard in sorted((dataset / "shards").glob("part-*")):
        tensor_path = shard / "tensors.npz"
        result_path = shard / "results.jsonl.gz"
        if not tensor_path.is_file() or not result_path.is_file():
            raise InputValidationError(
                "formal dataset shard is incomplete", context={"shard": shard.name}
            )
        try:
            with np.load(tensor_path, allow_pickle=False) as arrays:
                case_ids = np.asarray(arrays["case_ids"]).astype(str)
                splits = np.asarray(arrays["split_names"]).astype(str)
                weights = np.asarray(
                    arrays["sampling_design_weight_raw"], dtype=np.float64
                )
        except (OSError, KeyError, ValueError) as exc:
            raise InputValidationError(
                "formal dataset shard lacks split/weight metadata",
                context={"shard": shard.name},
            ) from exc
        if (
            case_ids.ndim != 1
            or splits.shape != case_ids.shape
            or weights.shape != case_ids.shape
            or not np.isfinite(weights).all()
            or np.any(weights <= 0.0)
        ):
            raise InputValidationError("formal dataset split/weight arrays are invalid")
        try:
            with gzip.open(result_path, "rt", encoding="utf-8") as stream:
                line_count = 0
                for index, line in enumerate(stream):
                    line_count += 1
                    if index >= case_ids.size:
                        raise ValueError("result shard has more rows than its tensor metadata")
                    case_id = str(case_ids[index])
                    split = str(splits[index])
                    split_counts[split] = split_counts.get(split, 0) + 1
                    if split == "test":
                        # Intentionally do not call json.loads on this line.
                        test_count += 1
                        continue
                    if split not in _DEVELOPMENT_SPLITS:
                        raise ValueError(f"unsupported evaluation split {split!r}")
                    result = json.loads(line)
                    if str(result.get("case_id", "")) != case_id:
                        raise ValueError("result row order differs from tensor case IDs")
                    if case_id in seen_ids or case_id not in cases:
                        raise ValueError("result case ID is duplicate or absent from formal ledger")
                    seen_ids.add(case_id)
                    evidence.append(
                        DevelopmentEvidence(
                            case=cases[case_id],
                            result=result,
                            split=split,
                            design_weight=float(weights[index]),
                        )
                    )
                if line_count != case_ids.size:
                    raise ValueError("result shard row count differs from tensor metadata")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise InputValidationError(
                "formal result shard is unreadable or misaligned",
                context={"shard": shard.name},
            ) from exc
        shard_count += 1
    if not evidence:
        raise InputValidationError("no train/validation teacher rows are available")
    audit = {
        "source_manifest_sha256": file_hash(manifest_path),
        "source_dataset_id": manifest.get("dataset_id"),
        "source_engine": manifest.get("engine"),
        "source_contracts": {
            key: manifest.get("source_contracts", {}).get(key)
            for key in (
                "target_hashes",
                "parameter_set_hashes",
                "sampling_spec_id",
                "sampling_design_hash",
            )
        },
        "shard_count": shard_count,
        "split_counts": split_counts,
        "development_row_count": len(evidence),
        "test_row_count": test_count,
        "test_result_json_lines_decoded": decoded_test_count,
        "test_metrics_computed": False,
        "test_open_count_increment": 0,
        "development_case_ids_hash": content_hash(
            [item.case["case_id"] for item in evidence]
        ),
    }
    return evidence, audit


def _safe_correlations(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.size < 2 or right.size != left.size:
        return {
            "paired_count": int(left.size),
            "spearman": None,
            "pearson": None,
            "status": "not_estimable_insufficient_pairs",
        }
    if np.all(left == left[0]) or np.all(right == right[0]):
        return {
            "paired_count": int(left.size),
            "spearman": None,
            "pearson": None,
            "status": "not_estimable_constant_input",
        }
    from scipy.stats import pearsonr, spearmanr

    spearman = float(spearmanr(left, right).statistic)
    pearson = float(pearsonr(left, right).statistic)
    return {
        "paired_count": int(left.size),
        "spearman": spearman if math.isfinite(spearman) else None,
        "pearson": pearson if math.isfinite(pearson) else None,
        "status": "estimated",
    }


def _region_id(case: Mapping[str, Any]) -> str:
    metadata = case.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return "unresolved"
    for name in (
        "continuous_direction_sampling",
        "sphere_entry_ray_sampling",
        "contact_manifold_sampling",
    ):
        record = metadata.get(name)
        if not isinstance(record, Mapping):
            continue
        for field in ("region_id", "approach_id"):
            value = record.get(field)
            if isinstance(value, str) and value:
                return value
    return "unresolved"


def analyze_proxy_teacher_arrays(
    *,
    proxy_hit_probability: Any,
    teacher_hit_probability: Any,
    proxy_expected_energy_j: Any,
    teacher_mean_deposited_energy_j: Any,
    observed_mask: Any,
    case_weights: Any,
    cells: Sequence[str],
) -> dict[str, Any]:
    """Compute the hurdle-aware metrics from aligned proxy/teacher arrays."""

    proxy_hit = np.asarray(proxy_hit_probability, dtype=np.float64)
    teacher_hit = np.asarray(teacher_hit_probability, dtype=np.float64)
    proxy_energy = np.asarray(proxy_expected_energy_j, dtype=np.float64)
    teacher_energy = np.asarray(teacher_mean_deposited_energy_j, dtype=np.float64)
    mask = np.asarray(observed_mask, dtype=bool)
    weights = np.asarray(case_weights, dtype=np.float64)
    if (
        proxy_hit.ndim != 2
        or teacher_hit.shape != proxy_hit.shape
        or proxy_energy.shape != proxy_hit.shape
        or teacher_energy.shape != proxy_hit.shape
        or mask.shape != proxy_hit.shape
        or weights.shape != (proxy_hit.shape[0],)
        or len(cells) != proxy_hit.shape[0]
        or not np.isfinite(proxy_hit).all()
        or not np.isfinite(teacher_hit).all()
        or not np.isfinite(proxy_energy).all()
        or not np.isfinite(teacher_energy).all()
        or not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
        or np.any((proxy_hit < 0.0) | (proxy_hit > 1.0))
        or np.any((teacher_hit < 0.0) | (teacher_hit > 1.0))
        or np.any(proxy_energy < 0.0)
        or np.any(teacher_energy < 0.0)
    ):
        raise InputValidationError("proxy/teacher arrays are invalid or misaligned")
    pair_weights = np.repeat(weights[:, None], proxy_hit.shape[1], axis=1)
    if not np.any(mask):
        raise InputValidationError("proxy/teacher comparison has no identified observations")
    calibration = weighted_calibration(
        proxy_hit[mask], teacher_hit[mask], weight=pair_weights[mask]
    )
    positive = mask & (proxy_energy > 0.0) & (teacher_energy > 0.0)
    positive_report: dict[str, Any] = {
        "paired_positive_count": int(np.sum(positive)),
        "comparison_semantics": (
            "paired case-component teacher replicate-mean deposited_energy_j "
            "versus proxy expected_damage_energy_j; both values strictly positive"
        ),
        "teacher_estimand": "replicate_mean_deposited_energy_j_given_case",
        "proxy_estimand": "deterministic_expected_damage_energy_j_given_case",
    }
    if np.any(positive):
        proxy_positive = proxy_energy[positive]
        teacher_positive = teacher_energy[positive]
        quantiles = (0.10, 0.50, 0.90, 0.99)
        positive_report.update(
            {
                "log1p_wasserstein1": wasserstein1_log_positive(
                    proxy_positive, teacher_positive
                ),
                "log1p_quantiles": {
                    "probabilities": list(quantiles),
                    "proxy": np.quantile(
                        np.log1p(proxy_positive), quantiles
                    ).tolist(),
                    "teacher": np.quantile(
                        np.log1p(teacher_positive), quantiles
                    ).tolist(),
                },
                "correlation": _safe_correlations(
                    np.log1p(proxy_positive), np.log1p(teacher_positive)
                ),
            }
        )
    else:
        positive_report.update(
            {
                "log1p_wasserstein1": None,
                "log1p_quantiles": None,
                "correlation": _safe_correlations(
                    np.asarray([], dtype=np.float64),
                    np.asarray([], dtype=np.float64),
                ),
            }
        )
    complete_cases = np.all(mask, axis=1)
    mmd = (
        rbf_mmd2(proxy_hit[complete_cases], teacher_hit[complete_cases])
        if np.any(complete_cases)
        else {"mmd2": None, "bandwidth": None}
    )
    strata: dict[str, Any] = {}
    cells_array = np.asarray([str(value) for value in cells])
    for cell in sorted(set(cells_array.tolist())):
        case_selected = cells_array == cell
        cell_mask = mask & case_selected[:, None]
        flat_proxy = proxy_hit[cell_mask]
        flat_teacher = teacher_hit[cell_mask]
        flat_weight = pair_weights[cell_mask]
        if flat_proxy.size:
            cell_calibration = weighted_calibration(
                flat_proxy, flat_teacher, weight=flat_weight
            )
            mae = float(
                np.average(np.abs(flat_proxy - flat_teacher), weights=flat_weight)
            )
        else:
            cell_calibration = {"brier": None, "ece": None, "bins": []}
            mae = None
        strata[cell] = {
            "case_count": int(np.sum(case_selected)),
            "identified_case_component_count": int(flat_proxy.size),
            "brier": cell_calibration["brier"],
            "ece": cell_calibration["ece"],
            "mae": mae,
            "correlation": _safe_correlations(flat_proxy, flat_teacher),
        }
    return {
        "identified_case_component_count": int(np.sum(mask)),
        "domain_complete_case_count": int(np.sum(complete_cases)),
        "hit_probability": {
            "proxy_transform": "1-exp(-expected_hits)",
            "teacher_estimand": "replicate_mean_hit_indicator_given_case",
            "brier": calibration["brier"],
            "ece": calibration["ece"],
            "calibration_bins": calibration["bins"],
            "jensen_shannon_fixed_bins": jensen_shannon_binned(
                proxy_hit[mask], teacher_hit[mask], bin_edges=_P_HIT_JS_EDGES
            ),
            "js_bin_edges": list(_P_HIT_JS_EDGES),
            "correlation": _safe_correlations(proxy_hit[mask], teacher_hit[mask]),
        },
        "positive_energy": positive_report,
        "multivariate_hit_probability": {
            **mmd,
            "case_scope": "domain-complete train/validation cases only",
            "feature_axis": "component hit probabilities",
        },
        "strata": strata,
    }


def build_proxy_teacher_distribution_report(
    pilot_directory: str | Path,
    engine: Any,
    *,
    proxy_source_seed: int = 2026111401,
    azimuths_per_ring: int = 4,
    layer_quadrature_samples: int = 16,
) -> dict[str, Any]:
    evidence, source_audit = load_development_evidence(pilot_directory)
    source_engine = source_audit.get("source_engine") or {}
    expected_target = set(
        source_audit.get("source_contracts", {}).get("target_hashes") or []
    )
    expected_registry = str(source_engine.get("configuration_hash", ""))
    if expected_target and engine.target_hash not in expected_target:
        raise InputValidationError("report target differs from the sampler dataset target")
    if expected_registry and engine.registry.hash != expected_registry:
        raise InputValidationError("report physics registry differs from the sampler dataset")
    component_ids = list(engine.compiled_target.component_order)
    count = len(evidence)
    width = len(component_ids)
    teacher_hit = np.zeros((count, width), dtype=np.float64)
    teacher_energy = np.zeros_like(teacher_hit)
    proxy_hit = np.zeros_like(teacher_hit)
    proxy_energy = np.zeros_like(teacher_hit)
    observed = np.zeros_like(teacher_hit, dtype=bool)
    weights = np.asarray([item.design_weight for item in evidence], dtype=np.float64)
    cells: list[str] = []
    proxy = FastSamplingProxy(
        engine,
        source_seed=proxy_source_seed,
        azimuths_per_ring=azimuths_per_ring,
        layer_quadrature_samples=layer_quadrature_samples,
    )
    for row, item in enumerate(evidence):
        result_components = item.result.get("components", {})
        if set(result_components) != set(component_ids):
            raise InputValidationError("teacher component vocabulary differs from target")
        targets = extract_component_hurdle_targets(item.result)
        proxy_result = proxy.evaluate(item.case)
        proxy_components = proxy_result.get("component_metrics", {})
        if set(proxy_components) != set(component_ids):
            raise InputValidationError("proxy component vocabulary differs from target")
        for column, component_id in enumerate(component_ids):
            fragment = result_components[component_id]["fragment_metrics"]
            teacher_hit[row, column] = float(fragment["hit_indicator"]["mean"])
            teacher_energy[row, column] = float(
                fragment["deposited_energy_j"]["mean"]
            )
            expected_hits = float(proxy_components[component_id]["expected_hits"])
            proxy_hit[row, column] = -math.expm1(-max(0.0, expected_hits))
            proxy_energy[row, column] = float(
                proxy_components[component_id]["expected_damage_energy_j"]
            )
            observed[row, column] = bool(
                targets["components"][component_id]["p_hit"]["mask"]
            )
        profile = str(item.case.get("munition", {}).get("profile_id", "unresolved"))
        cells.append(f"{_region_id(item.case)}::{profile}")
    metrics = analyze_proxy_teacher_arrays(
        proxy_hit_probability=proxy_hit,
        teacher_hit_probability=teacher_hit,
        proxy_expected_energy_j=proxy_energy,
        teacher_mean_deposited_energy_j=teacher_energy,
        observed_mask=observed,
        case_weights=weights,
        cells=cells,
    )
    report: dict[str, Any] = {
        "schema_version": REPORT_VERSION,
        "source": source_audit,
        "evaluation_scope": {
            "included_splits": sorted(_DEVELOPMENT_SPLITS),
            "excluded_split": "test",
            "case_count": count,
            "component_count": width,
            "component_ids": component_ids,
            "design_weight_semantics": "sampler raw inverse-inclusion design weight",
        },
        "proxy_contract": {
            "class": "FastSamplingProxy",
            "source_seed": int(proxy_source_seed),
            "azimuths_per_ring": int(azimuths_per_ring),
            "layer_quadrature_samples": int(layer_quadrature_samples),
            "target_hash": engine.target_hash,
            "physics_registry_hash": engine.registry.hash,
        },
        "metrics": metrics,
        "limitations": [
            "This is software-teacher agreement, not validation against physical truth.",
            "Proxy 1-exp(-lambda) uses a Poisson hit-count approximation.",
            "Positive-energy comparison aligns two different conditional expectations: teacher replicate mean and deterministic proxy expectation.",
            "Only point-identified train/validation component targets enter metrics; interval-censored targets are excluded.",
            "The frozen test split is not decoded, scored, or used for selection.",
        ],
    }
    report["report_hash"] = content_hash(report)
    return report



def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--physics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proxy-source-seed", type=int, default=2026111401)
    parser.add_argument("--azimuths-per-ring", type=int, default=4)
    parser.add_argument("--layer-quadrature-samples", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    target = json.loads(args.target.read_text(encoding="utf-8-sig"))
    engine = DamageEngine(target, PhysicsRegistry.from_file(args.physics))
    report = build_proxy_teacher_distribution_report(
        args.pilot,
        engine,
        proxy_source_seed=args.proxy_source_seed,
        azimuths_per_ring=args.azimuths_per_ring,
        layer_quadrature_samples=args.layer_quadrature_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report_hash": report["report_hash"],
                "test_result_json_lines_decoded": report["source"][
                    "test_result_json_lines_decoded"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


