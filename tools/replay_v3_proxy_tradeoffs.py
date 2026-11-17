#!/usr/bin/env python3
"""Replay V3 proxy trade-offs on frozen development candidate pools."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.normalized_sampling import (
    ENVELOPE_BANDS,
    design_weight_quality,
    global_marginal_coverage_probabilities,
    pivotal_sample,
)
from tools.evaluate_normalized_sampling_strategy import _allocate_counts, _gini


WEIGHT_GRID = (
    (0.00, 0.00),
    (0.10, 0.00),
    (0.20, 0.00),
    (0.30, 0.00),
    (0.40, 0.00),
    (0.60, 0.00),
    (0.20, 0.10),
    (0.30, 0.10),
    (0.40, 0.10),
    (0.60, 0.10),
    (0.20, 0.20),
    (0.30, 0.20),
    (0.40, 0.20),
    (0.60, 0.20),
    (0.30, 0.30),
    (0.40, 0.30),
    (0.60, 0.30),
    (0.40, 0.40),
    (0.60, 0.40),
)


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _prepare_strata(
    eligible: list[dict[str, Any]],
    *,
    sample_size: int,
    calibration: dict[str, Any],
) -> tuple[list[np.ndarray], list[int]]:
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_cell[str(row["approach_profile_cell"])].append(row)
    cell_allocations = _allocate_counts(
        {cell: len(values) for cell, values in by_cell.items()},
        total=sample_size,
        weights={cell: 1.0 for cell in by_cell},
        minimum_one=True,
    )
    row_index = {
        str(row["candidate_id"]): index for index, row in enumerate(eligible)
    }
    strata: list[np.ndarray] = []
    quotas: list[int] = []
    for cell in sorted(by_cell):
        by_band: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in by_cell[cell]:
            by_band[str(row["idea1_envelope_band"])].append(row)
        allocations = _allocate_counts(
            {band: len(by_band.get(band, [])) for band in ENVELOPE_BANDS},
            total=cell_allocations[cell],
            weights={
                band: float(calibration["target_band_fractions"][band])
                for band in ENVELOPE_BANDS
            },
            minimum_one=True,
        )
        for band in ENVELOPE_BANDS:
            band_rows = sorted(
                by_band.get(band, []), key=lambda row: str(row["candidate_id"])
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
    return strata, quotas


def _evaluate(
    eligible: list[dict[str, Any]],
    *,
    strata: list[np.ndarray],
    quotas: list[int],
    tree_penalty: float,
    u_penalty: float,
    seed: int,
) -> dict[str, Any]:
    component_ids = eligible[0]["idea1_envelope"]["component_ids"]
    support = np.asarray(
        [
            [
                row["idea1_envelope"]["component_support_indicator"][identifier]
                for identifier in component_ids
            ]
            for row in eligible
        ],
        dtype=np.float64,
    )
    auxiliary = np.asarray(
        [
            [
                max(map(float, row["tree_outputs"].values()), default=0.0),
                float(row["idea1_envelope"]["u_penetration"]),
            ]
            for row in eligible
        ],
        dtype=np.float64,
    )
    probabilities = global_marginal_coverage_probabilities(
        support,
        strata=strata,
        quotas=quotas,
        auxiliary_features=auxiliary,
        auxiliary_weights=[-tree_penalty, -u_penalty],
        smoothing=5.0,
        score_exponent=1.0,
        iterations=80,
        damping=0.2,
        rho_min=0.08,
    )
    selected = np.zeros(len(eligible), dtype=bool)
    for offset, indices in enumerate(strata):
        selected[indices] = pivotal_sample(
            probabilities[indices], seed=seed + 1009 * offset
        )
    selected_rows = [row for row, keep in zip(eligible, selected, strict=True) if keep]
    support_counts = support[selected].sum(axis=0)
    tree_max = np.asarray(
        [max(map(float, row["tree_outputs"].values()), default=0.0) for row in selected_rows]
    )
    utilization = np.asarray(
        [float(row["idea1_envelope"]["normalized_utilization"]) for row in selected_rows]
    )
    u_penetration = np.asarray(
        [float(row["idea1_envelope"]["u_penetration"]) for row in selected_rows]
    )
    return {
        "tree_penalty": tree_penalty,
        "u_penalty": u_penalty,
        "selected_count": len(selected_rows),
        "component_at_least_20": int(np.count_nonzero(support_counts >= 20)),
        "component_at_least_50": int(np.count_nonzero(support_counts >= 50)),
        "component_count_gini": _gini(support_counts),
        "proxy_tree_saturation_fraction": float(np.mean(tree_max >= 0.95)),
        "proxy_utilization_saturation_fraction": float(
            np.mean(utilization >= 0.95)
        ),
        "mean_u_penetration": float(np.mean(u_penetration)),
        "design_weight_quality": design_weight_quality(
            probabilities, selected
        ),
        "minimum_inclusion_probability": float(np.min(probabilities)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--trial-dir", type=Path, action="append", required=True)
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026111681)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    trials: list[dict[str, Any]] = []
    for trial_index, trial_dir in enumerate(args.trial_dir):
        rows = _read_jsonl_gz(trial_dir / "proxy_rows.jsonl.gz")
        eligible = [row for row in rows if row["eligible_model_domain"]]
        strata, quotas = _prepare_strata(
            eligible, sample_size=args.sample_size, calibration=calibration
        )
        evaluations = [
            _evaluate(
                eligible,
                strata=strata,
                quotas=quotas,
                tree_penalty=tree_penalty,
                u_penalty=u_penalty,
                seed=args.seed + 100_003 * trial_index,
            )
            for tree_penalty, u_penalty in WEIGHT_GRID
        ]
        trials.append(
            {
                "trial": trial_dir.name,
                "eligible_count": len(eligible),
                "stratum_count": len(strata),
                "evaluations": evaluations,
            }
        )

    aggregate: list[dict[str, Any]] = []
    for tree_penalty, u_penalty in WEIGHT_GRID:
        values = [
            next(
                item
                for item in trial["evaluations"]
                if item["tree_penalty"] == tree_penalty
                and item["u_penalty"] == u_penalty
            )
            for trial in trials
        ]
        aggregate.append(
            {
                "tree_penalty": tree_penalty,
                "u_penalty": u_penalty,
                "component_at_least_20_mean": float(
                    np.mean([item["component_at_least_20"] for item in values])
                ),
                "component_at_least_50_mean": float(
                    np.mean([item["component_at_least_50"] for item in values])
                ),
                "component_count_gini_mean": float(
                    np.mean([item["component_count_gini"] for item in values])
                ),
                "proxy_tree_saturation_fraction_mean": float(
                    np.mean(
                        [item["proxy_tree_saturation_fraction"] for item in values]
                    )
                ),
                "proxy_utilization_saturation_fraction_mean": float(
                    np.mean(
                        [
                            item["proxy_utilization_saturation_fraction"]
                            for item in values
                        ]
                    )
                ),
                "mean_u_penetration": float(
                    np.mean([item["mean_u_penetration"] for item in values])
                ),
                "ess_fraction_min": float(
                    np.min(
                        [
                            item["design_weight_quality"]["ess_fraction"]
                            for item in values
                        ]
                    )
                ),
                "weight_cv_max": float(
                    np.max(
                        [
                            item["design_weight_quality"]["weight_cv"]
                            for item in values
                        ]
                    )
                ),
            }
        )
    baseline = next(
        row
        for row in aggregate
        if row["tree_penalty"] == 0.0 and row["u_penalty"] == 0.0
    )
    for row in aggregate:
        row["delta_vs_v2"] = {
            key: float(row[key] - baseline[key])
            for key in (
                "component_at_least_20_mean",
                "component_at_least_50_mean",
                "component_count_gini_mean",
                "proxy_tree_saturation_fraction_mean",
                "proxy_utilization_saturation_fraction_mean",
                "mean_u_penetration",
            )
        }

    result = {
        "schema_version": "idea1-v3-proxy-tradeoff-replay@1.0.0",
        "evidence_role": "development_replay_only",
        "sample_size_per_evaluation": args.sample_size,
        "selection_seed": args.seed,
        "auxiliary_features": ["proxy_tree_max", "u_penetration"],
        "trials": trials,
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"aggregate": aggregate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
