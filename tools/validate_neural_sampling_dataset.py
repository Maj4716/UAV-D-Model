#!/usr/bin/env python3
"""Independently validate an Idea-1 hybrid neural dataset package."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.learning_input_identity import (
    loitering_impact_contact_pose_training_vector,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _close(actual: np.ndarray, expected: np.ndarray, *, name: str) -> None:
    if not np.allclose(actual, expected, rtol=2.0e-6, atol=2.0e-6):
        raise RuntimeError(f"tensor disagrees with packaged teacher rows: {name}")


def _validate_cases(
    path: Path,
    arrays: dict[str, np.ndarray],
    component_ids: list[str],
    tree_ids: list[str],
) -> int:
    count = 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if not line.strip():
                continue
            row = json.loads(line)
            _require(
                row["candidate_group_id"] == arrays["candidate_group_id"][index],
                "case order disagrees with candidate_group_id tensor",
            )
            _require(
                row["candidate_id"] == arrays["candidate_id"][index],
                "case order disagrees with candidate_id tensor",
            )
            _require(
                int(row["design_seed"]) == int(arrays["design_seed"][index]),
                "case order disagrees with design_seed tensor",
            )
            expected_x = np.asarray(
                loitering_impact_contact_pose_training_vector(row["case"]),
                dtype=np.float32,
            )
            _close(arrays["x"][index], expected_x, name="x")
            teacher = row["teacher"]
            complete = bool(teacher["domain_complete"])
            _require(
                complete == bool(arrays["domain_complete"][index]),
                "domain_complete tensor disagrees with teacher row",
            )
            if complete:
                components = teacher["components"]
                expected_components = {
                    "component_support": np.asarray(
                        [components[key]["positive_support"] for key in component_ids],
                        dtype=np.float32,
                    ),
                    "component_hits": np.asarray(
                        [components[key]["hits"] for key in component_ids],
                        dtype=np.float32,
                    ),
                    "component_penetrations": np.asarray(
                        [components[key]["penetrations"] for key in component_ids],
                        dtype=np.float32,
                    ),
                    "component_energy_j": np.asarray(
                        [components[key]["deposited_energy_j"] for key in component_ids],
                        dtype=np.float32,
                    ),
                    "component_severity": np.asarray(
                        [
                            components[key]["fragment_damage_severity"]
                            for key in component_ids
                        ],
                        dtype=np.float32,
                    ),
                }
                for name, expected in expected_components.items():
                    _close(arrays[name][index], expected, name=name)
                _close(
                    arrays["tree_outputs"][index],
                    np.asarray(
                        [teacher["tree_outputs"][key] for key in tree_ids],
                        dtype=np.float32,
                    ),
                    name="tree_outputs",
                )
                _close(
                    arrays["summary_targets"][index],
                    np.asarray(
                        [
                            teacher["positive_component_count"],
                            teacher["total_component_hits"],
                            teacher["total_component_penetrations"],
                            teacher["total_component_deposited_energy_j"],
                            teacher["teacher_damage_severity"],
                        ],
                        dtype=np.float32,
                    ),
                    name="summary_targets",
                )
            count += 1
    return count


def _validate_memberships(
    path: Path, *, valid_groups: set[str]
) -> tuple[int, Counter[str], Counter[tuple[int, str]]]:
    count = 0
    identifiers: set[str] = set()
    strategy_counts: Counter[str] = Counter()
    arm_counts: Counter[tuple[int, str]] = Counter()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row: dict[str, Any] = json.loads(line)
            identifier = str(row["observation_id"])
            _require(identifier not in identifiers, "duplicate selection observation")
            identifiers.add(identifier)
            group = str(row["candidate_group_id"])
            _require(group in valid_groups, "selection references an unknown candidate")
            probability = float(row["inclusion_probability"])
            weight = float(row["design_weight_raw"])
            _require(
                0.0 < probability <= 1.0 and math.isfinite(probability),
                "invalid inclusion probability",
            )
            _require(
                math.isclose(weight, 1.0 / probability, rel_tol=2.0e-10),
                "raw design weight is not inverse inclusion probability",
            )
            for field in (
                "predicted_domain_complete_probability",
                "predicted_tree_saturation_given_complete_probability",
            ):
                value = row.get(field)
                _require(
                    value is None
                    or (math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0),
                    f"invalid optional risk prediction: {field}",
                )
            strategy = str(row["strategy"])
            seed = int(row["design_seed"])
            strategy_counts[strategy] += 1
            arm_counts[(seed, strategy)] += 1
            count += 1
    return count, strategy_counts, arm_counts


def validate(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    manifest = json.loads(
        (directory / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (directory / "feature_schema.json").read_text(encoding="utf-8")
    )
    _require(
        manifest["schema_version"] == "idea1-hybrid-neural-dataset@1.1.0",
        "unexpected dataset schema version",
    )
    _require(manifest["status"] == "development_ready", "dataset is not ready")
    for filename, record in manifest["files"].items():
        _require(
            _sha256(directory / filename) == record["sha256"],
            f"file hash mismatch: {filename}",
        )

    archive = np.load(directory / "tensors.npz", allow_pickle=False)
    arrays = {name: archive[name] for name in archive.files}
    count = int(manifest["unique_candidate_count"])
    component_ids = list(schema["component_ids"])
    tree_ids = list(schema["tree_output_ids"])
    required_shapes = {
        "x": (count, 15),
        "x_normalized": (count, 15),
        "x_design": (count, 15),
        "x_train_mean": (15,),
        "x_train_scale": (15,),
        "domain_complete": (count,),
        "response_mask": (count,),
        "split_code": (count,),
        "component_support": (count, len(component_ids)),
        "component_hits": (count, len(component_ids)),
        "component_penetrations": (count, len(component_ids)),
        "component_energy_j": (count, len(component_ids)),
        "component_severity": (count, len(component_ids)),
        "tree_outputs": (count, len(tree_ids)),
        "summary_targets": (count, len(schema["summary_target_names"])),
        "candidate_group_id": (count,),
        "candidate_id": (count,),
        "design_seed": (count,),
    }
    for name, shape in required_shapes.items():
        _require(name in arrays and arrays[name].shape == shape, f"bad shape: {name}")

    for name in ("x", "x_normalized", "x_design", "x_train_mean", "x_train_scale"):
        _require(np.isfinite(arrays[name]).all(), f"non-finite learner tensor: {name}")
    _require(np.all(arrays["x_train_scale"] > 0.0), "normalization scale is not positive")
    split = arrays["split_code"]
    _require(set(np.unique(split)).issubset({0, 1}), "invalid split code")
    train = split == 0
    expected_mean = np.mean(arrays["x"][train], axis=0, dtype=np.float64).astype(
        np.float32
    )
    expected_scale = np.std(arrays["x"][train], axis=0, dtype=np.float64).astype(
        np.float32
    )
    expected_scale[expected_scale < 1.0e-12] = 1.0
    _close(arrays["x_train_mean"], expected_mean, name="x_train_mean")
    _close(arrays["x_train_scale"], expected_scale, name="x_train_scale")
    _close(
        arrays["x_normalized"],
        (arrays["x"] - expected_mean) / expected_scale,
        name="x_normalized",
    )

    complete = arrays["domain_complete"].astype(bool)
    _require(
        np.array_equal(complete, arrays["response_mask"]),
        "response mask disagrees with domain completeness",
    )
    response_names = (
        "component_support",
        "component_hits",
        "component_penetrations",
        "component_energy_j",
        "component_severity",
        "tree_outputs",
        "summary_targets",
    )
    for name in response_names:
        _require(np.isfinite(arrays[name][complete]).all(), f"bad complete labels: {name}")
        _require(np.isnan(arrays[name][~complete]).all(), f"fake censored labels: {name}")

    groups = [str(value) for value in arrays["candidate_group_id"]]
    _require(len(groups) == len(set(groups)), "candidate groups are not unique")
    train_groups = {groups[index] for index in np.flatnonzero(split == 0)}
    validation_groups = {groups[index] for index in np.flatnonzero(split == 1)}
    _require(not train_groups & validation_groups, "train/validation leakage")

    case_count = _validate_cases(
        directory / "cases.jsonl.gz", arrays, component_ids, tree_ids
    )
    observation_count, strategy_counts, arm_counts = _validate_memberships(
        directory / "selection_observations.jsonl.gz", valid_groups=set(groups)
    )
    _require(case_count == count, "case row count mismatch")
    _require(
        observation_count == int(manifest["selection_observation_count"]),
        "selection observation count mismatch",
    )
    _require(all(value == 2000 for value in arm_counts.values()), "arm is not 2000 rows")
    _require(
        dict(sorted(strategy_counts.items()))
        == manifest["strategy_observation_counts"],
        "strategy counts disagree with manifest",
    )
    _require(
        int(np.count_nonzero(complete)) == int(manifest["complete_domain_count"]),
        "complete-domain count disagrees with manifest",
    )
    return {
        "schema_version": "idea1-neural-dataset-validation@1.0.0",
        "status": "passed",
        "dataset_schema_version": manifest["schema_version"],
        "unique_candidate_count": count,
        "selection_observation_count": observation_count,
        "source_arm_count": len(arm_counts),
        "all_source_arms_exactly_2000": True,
        "complete_domain_count": int(np.count_nonzero(complete)),
        "domain_censored_count": int(np.count_nonzero(~complete)),
        "train_count": int(np.count_nonzero(split == 0)),
        "validation_count": int(np.count_nonzero(split == 1)),
        "canonical_input_recomputed_from_every_case": True,
        "all_teacher_tensors_recomputed_from_every_complete_case": True,
        "censored_targets_are_nan": True,
        "normalization_recomputed_from_train_only": True,
        "file_hashes_verified": True,
        "selection_probabilities_and_weights_verified": True,
        "no_train_validation_candidate_overlap": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.dataset_directory)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
