from __future__ import annotations

import gzip
import json
from pathlib import Path
from copy import deepcopy

import numpy as np
import pytest

from tools.run_phase0_learning_diagnostics import (
    DEFAULT_CONFIG,
    _probability_report,
    aggregate_runs,
    feature_views,
    leakage_checks,
    load_phase0_dataset,
    load_reference_memberships,
    main,
    nested_train_indices,
    train_model,
    weighted_regression_report,
)


def _micro_dataset(root: Path) -> Path:
    root.mkdir()
    split = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.uint8)
    x = np.asarray(
        [[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [10.0, 20.0], [11.0, 22.0], [12.0, 24.0]],
        dtype=np.float32,
    )
    train = split == 0
    mean = np.mean(x[train], axis=0, dtype=np.float64).astype(np.float32)
    scale = np.std(x[train], axis=0, dtype=np.float64).astype(np.float32)
    normalized = ((x - mean) / scale).astype(np.float32)
    design = np.asarray(
        [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [100.0, 200.0], [101.0, 201.0], [102.0, 202.0]],
        dtype=np.float32,
    )
    complete = np.asarray([1, 0, 1, 1, 0, 1], dtype=np.uint8)
    response = complete.astype(bool)
    support = np.asarray(
        [[0.0, 1.0], [np.nan, np.nan], [1.0, 0.0], [1.0, 0.0], [np.nan, np.nan], [0.0, 1.0]],
        dtype=np.float32,
    )
    severity = support.copy()
    tree = np.asarray(
        [[0.2], [np.nan], [0.8], [0.7], [np.nan], [0.3]], dtype=np.float32
    )
    groups = np.asarray([f"g{index}" for index in range(6)])
    np.savez(
        root / "tensors.npz",
        x=x,
        x_normalized=normalized,
        x_design=design,
        x_train_mean=mean,
        x_train_scale=scale,
        domain_complete=complete,
        response_mask=response,
        split_code=split,
        component_support=support,
        component_severity=severity,
        tree_outputs=tree,
        candidate_group_id=groups,
    )
    (root / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "idea1-hybrid-neural-dataset@1.1.0",
                "sealed_test_split": None,
            }
        ),
        encoding="utf-8",
    )
    (root / "feature_schema.json").write_text(
        json.dumps(
            {
                "component_ids": ["c0", "c1"],
                "tree_output_ids": ["t0"],
            }
        ),
        encoding="utf-8",
    )
    rows = []
    for index, group in enumerate(groups):
        probability = 0.5 if index in {1, 4} else 1.0
        rows.append(
            {
                "observation_id": f"reference:{group}",
                "candidate_group_id": group,
                "strategy": "reference_uniform",
                "split": "train" if split[index] == 0 else "validation",
                "inclusion_probability": probability,
                "design_weight_raw": 1.0 / probability,
            }
        )
    rows.append(
        {
            "observation_id": "enriched:g3",
            "candidate_group_id": "g3",
            "strategy": "coverage",
            "split": "validation",
            "inclusion_probability": 1.0,
            "design_weight_raw": 1.0,
        }
    )
    with gzip.open(root / "selection_observations.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return root


def test_split_normalization_reference_weights_and_leakage_checks(tmp_path: Path) -> None:
    root = _micro_dataset(tmp_path / "dataset")
    dataset = load_phase0_dataset(root)
    views, normalization = feature_views(dataset)
    arrays = dataset["arrays"]

    assert np.mean(views["x_design_train_standardized"][:3], axis=0) == pytest.approx(
        [0.0, 0.0], abs=1.0e-7
    )
    assert normalization["x_design_train_standardized"]["fit_split"] == "train"
    assert normalization["x_design_train_standardized"]["mean"] == pytest.approx(
        [1.0, 2.0]
    )

    reference = load_reference_memberships(
        root / "selection_observations.jsonl.gz",
        arrays["candidate_group_id"],
        arrays["split_code"],
    )
    validation = reference["split_names"] == "validation"
    assert reference["indices"][validation].tolist() == [3, 4, 5]
    assert reference["weights"][validation].tolist() == [1.0, 2.0, 1.0]
    checks = leakage_checks(arrays, reference)
    assert checks["passed"] is True
    assert checks["train_validation_candidate_overlap_count"] == 0
    assert checks["test_split_accessed"] is False


def test_learning_curve_subsets_are_nested_and_outcome_blind() -> None:
    split = np.asarray([0] * 12 + [1] * 4)
    quarter = nested_train_indices(split, fraction=0.25, seed=17)
    half = nested_train_indices(split, fraction=0.5, seed=17)
    full = nested_train_indices(split, fraction=1.0, seed=17)
    assert set(quarter).issubset(set(half))
    assert set(half).issubset(set(full))
    assert len(quarter) == 3
    assert len(half) == 6
    assert len(full) == 12
    assert max(full) < 12


def test_weighted_probability_and_regression_metrics_preserve_design_weights() -> None:
    targets = np.asarray([[0.0], [1.0], [1.0]])
    predictions = np.asarray([[0.2], [0.8], [0.6]])
    mask = np.ones_like(targets, dtype=bool)
    weights = np.asarray([1.0, 2.0, 1.0])
    probability = _probability_report(
        targets,
        predictions,
        mask,
        ("task",),
        weights,
        (0.5,),
        calibration_bins=2,
    )
    task = probability["tasks"][0]
    assert task["weighted_prevalence"] == pytest.approx(0.75)
    assert task["brier"] == pytest.approx((0.04 + 2 * 0.04 + 0.16) / 4)
    assert task["average_precision"] is not None

    regression = weighted_regression_report(
        targets,
        predictions,
        mask,
        ("task",),
        weights,
        (0.5,),
    )
    assert regression["tasks"][0]["rmse"] == pytest.approx(
        np.sqrt((0.04 + 2 * 0.04 + 0.16) / 4)
    )


def _fake_run(seed: int, average_precision: float, brier: float) -> dict:
    summary = {
        "domain_average_precision": average_precision,
        "domain_brier": brier,
        "domain_ece": 0.1,
        "support_macro_average_precision": average_precision,
        "support_macro_brier": brier,
        "support_macro_ece": 0.1,
        "tree_macro_soft_average_precision": average_precision,
        "tree_macro_brier": brier,
        "tree_macro_ece": 0.1,
        "severity_macro_rmse": 0.2,
        "severity_macro_mae": 0.1,
        "severity_positive_macro_rmse": 0.3,
        "severity_positive_macro_mae": 0.2,
        "component_bottleneck_count": 1,
    }
    return {
        "feature_view": "x_normalized",
        "train_fraction": 0.25,
        "seed": seed,
        "evaluation": {
            "summary": summary,
            "component_support": {
                "tasks": [
                    {
                        "task_name": "c0",
                        "average_precision": average_precision,
                        "brier": brier,
                        "weighted_prevalence": 0.25,
                        "soft_event_mass": 2.0,
                    }
                ]
            },
            "tree_outputs": {
                "tasks": [
                    {
                        "task_name": "t0",
                        "average_precision": average_precision,
                        "brier": brier,
                        "ece": 0.1,
                        "calibration_intercept": 0.0,
                        "calibration_slope": 1.0,
                    }
                ]
            },
        },
    }


def test_metric_aggregation_reports_mean_and_sample_standard_deviation() -> None:
    aggregate = aggregate_runs(
        [_fake_run(1, 0.2, 0.4), _fake_run(2, 0.4, 0.2), _fake_run(3, 0.6, 0.3)]
    )["x_normalized"]["0.25"]
    metric = aggregate["summary"]["support_macro_average_precision"]
    assert metric == pytest.approx({"mean": 0.4, "std": 0.2, "n": 3})
    assert aggregate["components"][0]["average_precision"] == pytest.approx(
        {"mean": 0.4, "std": 0.2, "n": 3}
    )


def test_tiny_multitask_training_uses_censored_rows_only_for_domain(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    dataset = load_phase0_dataset(_micro_dataset(tmp_path / "dataset"))
    arrays = dataset["arrays"]
    views, _ = feature_views(dataset)
    config = deepcopy(DEFAULT_CONFIG)
    config["model"]["hidden_widths"] = [4]
    config["training"]["epochs"] = 1
    config["training"]["batch_size"] = 3
    model, losses = train_model(
        views["x_normalized"],
        arrays,
        np.flatnonzero(arrays["split_code"] == 0),
        config,
        seed=7,
    )
    assert model is not None
    assert len(losses) == 1
    assert set(losses[0]) == {
        "component_severity",
        "component_support",
        "domain_complete",
        "total",
        "tree_outputs",
    }


def test_config_only_freezes_preregistered_settings_before_dataset_access(tmp_path: Path) -> None:
    output = tmp_path / "experiment"
    result = main(
        [
            str(tmp_path / "dataset-does-not-exist"),
            "--output-directory",
            str(output),
            "--config-only",
        ]
    )
    assert result == 0
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert len(config["training"]["seeds"]) == 3
    assert config["training"]["train_fractions"] == [0.25, 0.5, 1.0]
    assert not (output / "results.json").exists()
