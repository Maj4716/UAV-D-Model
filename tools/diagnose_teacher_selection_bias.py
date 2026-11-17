#!/usr/bin/env python3
"""Diagnose proxy features associated with teacher censoring and saturation."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _auc(y: np.ndarray, score: np.ndarray) -> float | None:
    positive = y == 1
    n_positive = int(positive.sum())
    n_negative = int((~positive).sum())
    if n_positive == 0 or n_negative == 0:
        return None
    ranks = rankdata(score, method="average")
    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def _rate_table(
    rows: list[dict[str, Any]], key: str, target: str
) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(int(row[target]))
    return [
        {
            key: name,
            "n": len(values),
            "rate": float(np.mean(values)),
        }
        for name, values in sorted(groups.items())
    ]


def diagnose(
    frame_dir: Path,
    teacher_dirs: list[Path],
    strategies: list[str],
) -> dict[str, Any]:
    proxy_by_id: dict[str, dict[str, Any]] = {}
    teacher_by_id: dict[str, dict[str, Any]] = {}
    arm_by_id: dict[str, set[str]] = defaultdict(set)

    if len(strategies) != len(teacher_dirs):
        raise ValueError("strategy and teacher-dir counts must match")
    for strategy, teacher_dir in zip(strategies, teacher_dirs, strict=True):
        path = frame_dir / strategy / "selected_proxy_rows.jsonl.gz"
        for row in _read_jsonl_gz(path):
            proxy_by_id[row["candidate_id"]] = row
        for row in _read_jsonl_gz(teacher_dir / "teacher_metrics.jsonl.gz"):
            teacher_by_id[row["candidate_id"]] = row
            arm_by_id[row["candidate_id"]].add(strategy)

    rows: list[dict[str, Any]] = []
    for candidate_id in sorted(proxy_by_id.keys() & teacher_by_id.keys()):
        proxy = proxy_by_id[candidate_id]
        teacher = teacher_by_id[candidate_id]
        proxy_tree_max = max(proxy["tree_outputs"].values(), default=0.0)
        teacher_tree_max = max(teacher["tree_outputs"].values(), default=0.0)
        envelope = proxy["idea1_envelope"]
        rows.append(
            {
                "candidate_id": candidate_id,
                "strategies": sorted(arm_by_id[candidate_id]),
                "profile_id": proxy["profile_id"],
                "approach_profile_cell": proxy["approach_profile_cell"],
                "envelope_band": proxy["idea1_envelope_band"],
                "proxy_u_penetration": float(envelope["u_penetration"]),
                "proxy_normalized_utilization": float(
                    envelope["normalized_utilization"]
                ),
                "proxy_tree_max": float(proxy_tree_max),
                "proxy_boundary_information": float(
                    0.5 * envelope["tree_information"]
                    + 0.5 * envelope["utilization_information"]
                ),
                "proxy_supported_component_count": int(
                    proxy["supported_component_count"]
                ),
                "teacher_domain_censored": int(not teacher["domain_complete"]),
                "teacher_tree_saturated": int(teacher_tree_max >= 0.95),
            }
        )

    arm_summaries: dict[str, dict[str, Any]] = {}
    for strategy in strategies:
        arm_rows = [row for row in rows if strategy in row["strategies"]]
        complete = [row for row in arm_rows if not row["teacher_domain_censored"]]
        censored = [row for row in arm_rows if row["teacher_domain_censored"]]
        arm_summaries[strategy] = {
            "row_count": len(arm_rows),
            "teacher_domain_censored_fraction": float(
                np.mean([row["teacher_domain_censored"] for row in arm_rows])
            ),
            "proxy_tree_saturated_fraction_all": float(
                np.mean([row["proxy_tree_max"] >= 0.95 for row in arm_rows])
            ),
            "proxy_tree_saturated_fraction_complete_domain": float(
                np.mean([row["proxy_tree_max"] >= 0.95 for row in complete])
            ),
            "teacher_tree_saturated_fraction_all": float(
                np.mean([row["teacher_tree_saturated"] for row in arm_rows])
            ),
            "teacher_tree_saturated_fraction_complete_domain": float(
                np.mean([row["teacher_tree_saturated"] for row in complete])
            ),
            "teacher_tree_saturated_fraction_censored": float(
                np.mean([row["teacher_tree_saturated"] for row in censored])
            ),
            "mean_proxy_tree_max_all": float(
                np.mean([row["proxy_tree_max"] for row in arm_rows])
            ),
            "mean_proxy_tree_max_complete_domain": float(
                np.mean([row["proxy_tree_max"] for row in complete])
            ),
        }

    targets = ("teacher_domain_censored", "teacher_tree_saturated")
    features = (
        "proxy_u_penetration",
        "proxy_normalized_utilization",
        "proxy_tree_max",
        "proxy_boundary_information",
        "proxy_supported_component_count",
    )
    aucs: dict[str, dict[str, float | None]] = {}
    for target in targets:
        y = np.asarray([row[target] for row in rows], dtype=int)
        aucs[target] = {}
        for feature in features:
            score = np.asarray([row[feature] for row in rows], dtype=float)
            aucs[target][feature] = _auc(y, score)

    threshold_stats: list[dict[str, Any]] = []
    for feature in ("proxy_tree_max", "proxy_normalized_utilization"):
        for threshold in (0.80, 0.90, 0.95):
            high = [row for row in rows if row[feature] >= threshold]
            threshold_stats.append(
                {
                    "feature": feature,
                    "threshold": threshold,
                    "n": len(high),
                    "fraction": len(high) / len(rows),
                    "domain_censored_rate": (
                        float(
                            np.mean(
                                [row["teacher_domain_censored"] for row in high]
                            )
                        )
                        if high
                        else None
                    ),
                    "tree_saturated_rate": (
                        float(
                            np.mean(
                                [row["teacher_tree_saturated"] for row in high]
                            )
                        )
                        if high
                        else None
                    ),
                }
            )

    by_cell = _rate_table(rows, "approach_profile_cell", "teacher_domain_censored")
    by_cell.sort(key=lambda item: (-item["rate"], -item["n"], item["approach_profile_cell"]))
    return {
        "schema_version": "teacher-selection-diagnostic@1.0.0",
        "unique_candidate_count": len(rows),
        "teacher_domain_censored_fraction": float(
            np.mean([row["teacher_domain_censored"] for row in rows])
        ),
        "teacher_tree_saturated_fraction": float(
            np.mean([row["teacher_tree_saturated"] for row in rows])
        ),
        "arm_summaries": arm_summaries,
        "univariate_auc": aucs,
        "threshold_stats": threshold_stats,
        "domain_censoring_by_profile": _rate_table(
            rows, "profile_id", "teacher_domain_censored"
        ),
        "domain_censoring_by_envelope_band": _rate_table(
            rows, "envelope_band", "teacher_domain_censored"
        ),
        "highest_domain_censoring_cells": by_cell[:8],
        "lowest_domain_censoring_cells": list(reversed(by_cell[-8:])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-dir", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, action="append", required=True)
    parser.add_argument("--strategy", action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = diagnose(args.frame_dir, args.teacher_dir, args.strategy)
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
