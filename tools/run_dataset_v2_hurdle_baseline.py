#!/usr/bin/env python
"""Run train-only hurdle learning curves and freeze a validation-selected model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.hurdle_baseline import (  # noqa: E402
    DEFAULT_ENCODER_ABLATION_BUNDLES,
    TARGET_ENCODER_METHODS,
    run_hurdle_learning_curves,
    run_target_encoder_ablation,
)
from damage_engine.dataset_v2 import (  # noqa: E402
    load_dataset_v2_core,
    validate_dataset_v2,
)
from damage_engine.errors import InputValidationError  # noqa: E402
from damage_engine.utils import content_hash, file_hash  # noqa: E402


def _write_inadmissible_encoder_ablation_report(
    dataset: Path,
    output: Path,
    *,
    fractions: tuple[float, ...] | list[float],
    seeds: tuple[int, ...] | list[int],
    epochs: int,
) -> dict | None:
    """Close the validation gate with evidence when p_hit is unobservable.

    Conditional hurdle heads can remain observable even when the primary p_hit
    validation mask is empty.  They cannot substitute for the frozen encoder
    selection metric, so the CLI emits a machine-readable inadmissibility
    result instead of losing the per-bundle decision in an exception.
    """

    validation = validate_dataset_v2(dataset, recompute_views=False)
    seal_path = dataset / "sealed_test.json"
    if not seal_path.is_file():
        raise InputValidationError("dataset-v2 lacks sealed_test.json")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("test_open_count") != 0:
        raise InputValidationError("sealed test has already been opened")
    core = load_dataset_v2_core(dataset)
    labels = core["component_labels"]
    splits = core["sampling_weights"]["split"].astype(str)

    def split_counts(split: str) -> dict[str, int]:
        selected = splits == split
        return {
            "case_count": int(np.sum(selected)),
            "p_hit_observation_count": int(np.sum(labels["label_mask"][selected])),
            "conditional_energy_observation_count": int(
                np.sum(labels["conditional_log1p_energy_mask"][selected])
            ),
            "conditional_hit_count_observation_count": int(
                np.sum(labels["conditional_log1p_hit_count_mask"][selected])
            ),
        }

    observations = {
        split: split_counts(split) for split in ("train", "validation", "test")
    }
    if observations["validation"]["p_hit_observation_count"] > 0:
        return None
    if output.exists():
        raise InputValidationError(
            "target encoder ablation output directory already exists",
            context={"path": str(output)},
        )

    failure = {
        "code": "no_observable_validation_p_hit",
        "message": "validation contains no observable p_hit labels",
        "selection_metric": "mean_full_fraction_seed_p_hit_weighted_brier",
    }
    report = {
        "schema_version": "hurdle-target-encoder-ablation@1.0.0",
        "status": "all_bundles_inadmissible",
        "reason": failure,
        "selection_scope": "reference_validation_only",
        "selection_metric": failure["selection_metric"],
        "selected_bundle_id": None,
        "dataset_id": validation["dataset_id"],
        "dataset_manifest_hash": validation["manifest_hash"],
        "dataset_manifest_sha256": file_hash(dataset / "manifest.json"),
        "requested_fractions": [float(value) for value in fractions],
        "requested_seeds": [int(value) for value in seeds],
        "requested_epochs": int(epochs),
        "observation_counts": observations,
        "candidate_bundles": [
            {
                "bundle_id": name,
                "methods": dict(methods),
                "status": "inadmissible",
                "error": failure,
                "validation_p_hit_brier": None,
                "validation_p_hit_ece": None,
            }
            for name, methods in DEFAULT_ENCODER_ABLATION_BUNDLES.items()
        ],
        "conditional_head_assessment": {
            "validation_energy_observation_count": observations["validation"][
                "conditional_energy_observation_count"
            ],
            "validation_hit_count_observation_count": observations["validation"][
                "conditional_hit_count_observation_count"
            ],
            "eligible_as_p_hit_selection_substitute": False,
        },
        "test_access_policy": {
            "labels_used_in_loss": False,
            "labels_used_in_metrics": False,
            "labels_used_in_selection": False,
            "test_result_json_lines_decoded": int(
                seal.get("test_result_json_lines_decoded", 0)
            ),
            "open_count": int(seal["test_open_count"]),
        },
    }
    report["report_hash"] = content_hash(report)
    output.mkdir(parents=True)
    (output / "ablation_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fraction", action="append", type=float)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--task-admission",
        type=Path,
        help="frozen dataset-v2 component-task admission JSON",
    )
    parser.add_argument(
        "--energy-encoder",
        choices=sorted(TARGET_ENCODER_METHODS),
        default="empirical_quantile_normal",
    )
    parser.add_argument(
        "--hit-count-encoder",
        choices=sorted(TARGET_ENCODER_METHODS),
        default="box_cox",
    )
    parser.add_argument(
        "--encoder-ablation",
        action="store_true",
        help=(
            "compare identity, Box-Cox, quantile-normal and the Section-9 "
            "hybrid with identical fractions and seeds"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = {
        "fractions": args.fraction or (0.25, 0.50, 1.0),
        "seeds": args.seed or (17, 29, 43),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "hidden_size": args.hidden_size,
        "device": args.device,
    }
    if args.encoder_ablation:
        result = _write_inadmissible_encoder_ablation_report(
            args.dataset,
            args.output,
            fractions=common["fractions"],
            seeds=common["seeds"],
            epochs=common["epochs"],
        )
        if result is None:
            result = run_target_encoder_ablation(
                args.dataset,
                args.output,
                task_admission_path=args.task_admission,
                **common,
            )
        summary = {
            "output": str(args.output),
            "candidate_count": len(result["candidate_bundles"]),
            "selected_bundle_id": result["selected_bundle_id"],
            "selection_scope": result["selection_scope"],
            "status": result.get("status", "completed"),
            "test_access_policy": result["test_access_policy"],
        }
    else:
        result = run_hurdle_learning_curves(
            args.dataset,
            args.output,
            energy_encoder_method=args.energy_encoder,
            hit_count_encoder_method=args.hit_count_encoder,
            task_admission_path=args.task_admission,
            **common,
        )
        summary = {
            "output": str(args.output),
            "dataset_id": result["dataset_id"],
            "run_count": len(result["runs"]),
            "target_encoder_methods": result["target_encoder_methods"],
            "selected": result["selected_frozen_model"],
            "test_access_policy": result["test_access_policy"],
        }
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
