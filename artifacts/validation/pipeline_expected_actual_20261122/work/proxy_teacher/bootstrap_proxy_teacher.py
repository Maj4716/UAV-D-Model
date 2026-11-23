"""Case-cluster bootstrap for the sealed-test-safe proxy/teacher comparison."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.distribution_metrics import (
    jensen_shannon_binned,
    rbf_mmd2,
    wasserstein1_log_positive,
    weighted_brier,
    weighted_calibration,
)
from damage_engine.engine import DamageEngine
from damage_engine.fast_sampling_proxy import FastSamplingProxy
from damage_engine.hurdle_targets import extract_component_hurdle_targets
from damage_engine.registry import PhysicsRegistry
from damage_engine.utils import content_hash, file_hash
from tools.evaluate_proxy_teacher_distribution import (
    _P_HIT_JS_EDGES,
    load_development_evidence,
)


def _arrays(pilot: Path, target: Path, physics: Path):
    evidence, source_audit = load_development_evidence(pilot)
    target_data = json.loads(target.read_text(encoding="utf-8-sig"))
    engine = DamageEngine(target_data, PhysicsRegistry.from_file(physics))
    expected_targets = set(
        source_audit.get("source_contracts", {}).get("target_hashes") or []
    )
    expected_registry = str((source_audit.get("source_engine") or {}).get("configuration_hash", ""))
    if expected_targets and engine.target_hash not in expected_targets:
        raise ValueError("target identity differs from the dataset")
    if expected_registry and engine.registry.hash != expected_registry:
        raise ValueError("physics registry identity differs from the dataset")

    component_ids = list(engine.compiled_target.component_order)
    shape = (len(evidence), len(component_ids))
    teacher_hit = np.zeros(shape, dtype=np.float64)
    teacher_energy = np.zeros(shape, dtype=np.float64)
    proxy_hit = np.zeros(shape, dtype=np.float64)
    proxy_energy = np.zeros(shape, dtype=np.float64)
    observed = np.zeros(shape, dtype=bool)
    weights = np.asarray([item.design_weight for item in evidence], dtype=np.float64)
    proxy = FastSamplingProxy(engine)
    for row, item in enumerate(evidence):
        targets = extract_component_hurdle_targets(item.result)
        proxy_result = proxy.evaluate(deepcopy(item.case))
        for column, component_id in enumerate(component_ids):
            fragment = item.result["components"][component_id]["fragment_metrics"]
            teacher_hit[row, column] = float(fragment["hit_indicator"]["mean"])
            teacher_energy[row, column] = float(fragment["deposited_energy_j"]["mean"])
            expected_hits = float(
                proxy_result["component_metrics"][component_id]["expected_hits"]
            )
            proxy_hit[row, column] = -math.expm1(-max(0.0, expected_hits))
            proxy_energy[row, column] = float(
                proxy_result["component_metrics"][component_id][
                    "expected_damage_energy_j"
                ]
            )
            observed[row, column] = bool(
                targets["components"][component_id]["p_hit"]["mask"]
            )
    return (
        proxy_hit,
        teacher_hit,
        proxy_energy,
        teacher_energy,
        observed,
        weights,
        source_audit,
        engine,
    )


def _metrics(proxy_hit, teacher_hit, proxy_energy, teacher_energy, observed, weights):
    pair_weights = np.repeat(weights[:, None], proxy_hit.shape[1], axis=1)
    calibration = weighted_calibration(
        proxy_hit[observed], teacher_hit[observed], weight=pair_weights[observed]
    )
    positive = observed & (proxy_energy > 0.0) & (teacher_energy > 0.0)
    complete = np.all(observed, axis=1)
    proxy_unreached = observed & (proxy_hit == 0.0)
    false_unreached = proxy_unreached & (teacher_hit > 0.0)
    unreached_count = int(np.sum(proxy_unreached))
    false_unreached_count = int(np.sum(false_unreached))
    false_unreached_rate = (
        None if unreached_count == 0 else false_unreached_count / unreached_count
    )
    if unreached_count:
        z = 1.959963984540054
        p = false_unreached_rate
        denominator = 1.0 + z * z / unreached_count
        center = (p + z * z / (2.0 * unreached_count)) / denominator
        half = (
            z
            * math.sqrt(
                p * (1.0 - p) / unreached_count
                + z * z / (4.0 * unreached_count * unreached_count)
            )
            / denominator
        )
        wilson_upper = min(1.0, center + half)
    else:
        wilson_upper = None
    return {
        "brier": weighted_brier(
            proxy_hit[observed], teacher_hit[observed], pair_weights[observed]
        ),
        "ece": calibration["ece"],
        "jensen_shannon_fixed_bins": jensen_shannon_binned(
            proxy_hit[observed], teacher_hit[observed], bin_edges=_P_HIT_JS_EDGES
        ),
        "log1p_wasserstein1": (
            wasserstein1_log_positive(proxy_energy[positive], teacher_energy[positive])
            if np.any(positive)
            else None
        ),
        "mmd2": (
            rbf_mmd2(proxy_hit[complete], teacher_hit[complete])["mmd2"]
            if np.any(complete)
            else None
        ),
        "proxy_unreached_pair_count": unreached_count,
        "teacher_positive_among_proxy_unreached": false_unreached_count,
        "false_unreached_rate": false_unreached_rate,
        "false_unreached_wilson_upper": wilson_upper,
        "identified_case_component_count": int(np.sum(observed)),
        "paired_positive_energy_count": int(np.sum(positive)),
        "domain_complete_case_count": int(np.sum(complete)),
    }


def _interval(values):
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    return {
        "valid_resamples": int(array.size),
        "percentile_95_interval": np.quantile(array, [0.025, 0.975]).tolist(),
        "bootstrap_median": float(np.median(array)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--physics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026112201)
    args = parser.parse_args()
    if args.resamples < 100:
        raise ValueError("resamples must be at least 100")

    arrays = _arrays(args.pilot, args.target, args.physics)
    proxy_hit, teacher_hit, proxy_energy, teacher_energy, observed, weights = arrays[:6]
    source_audit, engine = arrays[6:]
    point = _metrics(
        proxy_hit, teacher_hit, proxy_energy, teacher_energy, observed, weights
    )
    rng = np.random.default_rng(args.seed)
    draws = {
        key: []
        for key in (
            "brier",
            "ece",
            "jensen_shannon_fixed_bins",
            "log1p_wasserstein1",
            "mmd2",
            "false_unreached_rate",
        )
    }
    for _ in range(args.resamples):
        index = rng.integers(0, proxy_hit.shape[0], size=proxy_hit.shape[0])
        result = _metrics(
            proxy_hit[index],
            teacher_hit[index],
            proxy_energy[index],
            teacher_energy[index],
            observed[index],
            weights[index],
        )
        for key in draws:
            draws[key].append(result[key])

    report = {
        "schema_version": "proxy-teacher-case-cluster-bootstrap@1.0.0",
        "analysis_role": "development_descriptive_uncertainty_not_qmc_design_variance",
        "source": {
            **source_audit,
            "test_result_json_lines_decoded": 0,
            "test_metrics_computed": False,
            "test_open_count_increment": 0,
            "target_path": str(args.target),
            "target_file_sha256": file_hash(args.target),
            "physics_path": str(args.physics),
            "physics_file_sha256": file_hash(args.physics),
            "engine_target_hash": engine.target_hash,
            "engine_registry_hash": engine.registry.hash,
        },
        "bootstrap": {
            "unit": "case_with_all_67_components",
            "method": "paired_nonparametric_case_cluster_percentile_bootstrap",
            "resamples": args.resamples,
            "seed": args.seed,
            "scope_note": (
                "Quantifies finite development-case uncertainty conditional on this "
                "single QMC design; it is not between-scramble design variance."
            ),
        },
        "weights": {
            "unique_values": np.unique(weights).tolist(),
            "all_equal": bool(np.all(weights == weights[0])),
        },
        "point_estimates": point,
        "intervals": {key: _interval(values) for key, values in draws.items()},
    }
    report["report_hash"] = content_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report_hash": report["report_hash"],
                "test_result_json_lines_decoded": 0,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
