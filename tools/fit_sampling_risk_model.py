#!/usr/bin/env python3
"""Fit and cross-seed validate the frozen sampling-risk logistic models."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.sampling_risk_model import (
    SAMPLING_RISK_MODEL_VERSION,
    sampling_risk_feature_names,
    sampling_risk_features,
)


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _fit_model(features: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0, ddof=0)
    scale[scale <= 1.0e-12] = 1.0
    standardized = (features - mean) / scale
    estimator = LogisticRegression(
        C=0.3,
        l1_ratio=0.0,
        solver="lbfgs",
        max_iter=2000,
        random_state=0,
    )
    estimator.fit(standardized, labels)
    return {
        "model_family": "standardized_l2_logistic_regression",
        "regularization_c": 0.3,
        "feature_names": list(sampling_risk_feature_names()),
        "feature_mean": [float(value) for value in mean],
        "feature_scale": [float(value) for value in scale],
        "coefficients": [float(value) for value in estimator.coef_[0]],
        "intercept": float(estimator.intercept_[0]),
        "training_count": int(features.shape[0]),
        "positive_count": int(np.count_nonzero(labels)),
    }


def _predict(features: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    logits = (features - mean) / scale @ coefficients + float(model["intercept"])
    logits = np.clip(logits, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "count": int(labels.size),
        "positive_fraction": float(np.mean(labels)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "mean_predicted_probability": float(np.mean(probabilities)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        nargs=4,
        action="append",
        metavar=("DESIGN_SEED", "FRAME_DIR", "STRATEGY", "TEACHER_DIR"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records: dict[tuple[int, str], dict[str, Any]] = {}
    source_records: list[dict[str, Any]] = []
    for raw_seed, raw_frame, strategy, raw_teacher in args.source:
        seed = int(raw_seed)
        frame = Path(raw_frame).resolve()
        teacher = Path(raw_teacher).resolve()
        proxy_path = frame / strategy / "selected_proxy_rows.jsonl.gz"
        teacher_path = teacher / "teacher_metrics.jsonl.gz"
        proxy_rows = {
            str(row["candidate_id"]): row for row in _read_jsonl_gz(proxy_path)
        }
        teacher_rows = {
            str(row["candidate_id"]): row for row in _read_jsonl_gz(teacher_path)
        }
        if proxy_rows.keys() != teacher_rows.keys():
            raise RuntimeError(f"proxy/teacher identities differ for {strategy}")
        for candidate_id in sorted(proxy_rows):
            teacher_row = teacher_rows[candidate_id]
            tree_max = max(
                map(float, teacher_row["tree_outputs"].values()), default=0.0
            )
            record = {
                "design_seed": seed,
                "candidate_id": candidate_id,
                "features": sampling_risk_features(proxy_rows[candidate_id]),
                "domain_complete": int(teacher_row["domain_complete"]),
                "tree_saturated": int(tree_max >= 0.95),
            }
            key = (seed, candidate_id)
            if key in records:
                previous = records[key]
                if (
                    previous["domain_complete"] != record["domain_complete"]
                    or previous["tree_saturated"] != record["tree_saturated"]
                    or not np.array_equal(previous["features"], record["features"])
                ):
                    raise RuntimeError(f"duplicate candidate disagrees: {key}")
            records[key] = record
        source_records.append(
            {
                "design_seed": seed,
                "frame_directory": str(frame),
                "strategy": strategy,
                "teacher_directory": str(teacher),
                "proxy_rows_hash": _file_hash(proxy_path),
                "teacher_metrics_hash": _file_hash(teacher_path),
                "arm_count": len(proxy_rows),
            }
        )

    ordered = [records[key] for key in sorted(records)]
    features = np.vstack([record["features"] for record in ordered])
    groups = np.asarray([record["design_seed"] for record in ordered], dtype=int)
    domain_labels = np.asarray(
        [record["domain_complete"] for record in ordered], dtype=int
    )
    tree_labels = np.asarray(
        [record["tree_saturated"] for record in ordered], dtype=int
    )
    seeds = sorted(set(map(int, groups)))
    if len(seeds) < 2:
        raise ValueError("cross-seed validation requires at least two design seeds")

    domain_oof = np.full(len(ordered), np.nan, dtype=np.float64)
    tree_oof = np.full(len(ordered), np.nan, dtype=np.float64)
    folds: list[dict[str, Any]] = []
    for held_out in seeds:
        train = groups != held_out
        test = groups == held_out
        domain_model = _fit_model(features[train], domain_labels[train])
        domain_oof[test] = _predict(features[test], domain_model)

        tree_train = train & (domain_labels == 1)
        tree_test = test & (domain_labels == 1)
        tree_model = _fit_model(features[tree_train], tree_labels[tree_train])
        tree_oof[tree_test] = _predict(features[tree_test], tree_model)
        folds.append(
            {
                "held_out_design_seed": held_out,
                "domain_complete": _metrics(
                    domain_labels[test], domain_oof[test]
                ),
                "tree_saturation_given_complete": _metrics(
                    tree_labels[tree_test], tree_oof[tree_test]
                ),
            }
        )

    complete = domain_labels == 1
    domain_model = _fit_model(features, domain_labels)
    tree_model = _fit_model(features[complete], tree_labels[complete])
    artifact = {
        "schema_version": SAMPLING_RISK_MODEL_VERSION,
        "fit_tool_version": "fit-sampling-risk-model@1.0.0",
        "intended_use": (
            "pre-outcome sampling utility only; predicts teacher-domain "
            "completeness and conditional teacher-tree saturation"
        ),
        "unique_development_candidate_count": len(ordered),
        "development_design_seeds": seeds,
        "source_arms": source_records,
        "cross_seed_validation": {
            "folds": folds,
            "pooled_domain_complete": _metrics(domain_labels, domain_oof),
            "pooled_tree_saturation_given_complete": _metrics(
                tree_labels[complete], tree_oof[complete]
            ),
        },
        "domain_complete_model": domain_model,
        "tree_saturation_given_complete_model": tree_model,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "unique_development_candidate_count": len(ordered),
                "development_design_seeds": seeds,
                "cross_seed_validation": artifact["cross_seed_validation"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
