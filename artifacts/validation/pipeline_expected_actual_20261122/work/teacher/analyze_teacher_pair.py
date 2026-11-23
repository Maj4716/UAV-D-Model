"""Sealed-test-safe paired audit of the v2/v3 recovery pilots.

The script reads only dataset-v2 development cores and their already-generated
audits.  Test rows in those cores are opaque placeholders with all label masks
false; the script asserts that contract and excludes them before using labels
or component-state trace records.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[5]
CONTROL = ROOT / "artifacts/datasets/recovery_paired_control_v2_r8_20261122_retry1"
BOUND = ROOT / "artifacts/datasets/recovery_paired_bound_v3_r8_20261122"
OUTPUT = Path(__file__).with_name("teacher_pair_metrics.json")
BOOTSTRAP_SEED = 2026112201
BOOTSTRAP_DRAWS = 10_000
EPSILON = 0.01
TARGETS = ("p_hit", "p_degraded", "p_failed")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "maximum": float(np.max(values)),
    }


def _bootstrap_mean_difference(
    control: np.ndarray, bound: np.ndarray, rng: np.random.Generator
) -> dict[str, Any]:
    control = np.asarray(control, dtype=np.float64)
    bound = np.asarray(bound, dtype=np.float64)
    if control.shape != bound.shape or control.ndim != 1:
        raise ValueError("paired bootstrap inputs must be aligned vectors")
    difference = bound - control
    indices = rng.integers(0, difference.size, size=(BOOTSTRAP_DRAWS, difference.size))
    draws = np.mean(difference[indices], axis=1)
    return {
        "estimand": "v3_minus_v2_paired_case_mean_difference",
        "paired_case_count": int(difference.size),
        "estimate": float(np.mean(difference)),
        "percentile_95_interval": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def _bootstrap_flat_median_difference(
    control: np.ndarray, bound: np.ndarray, rng: np.random.Generator
) -> dict[str, Any]:
    control = np.asarray(control, dtype=np.float64)
    bound = np.asarray(bound, dtype=np.float64)
    if control.shape != bound.shape or control.ndim != 2:
        raise ValueError("paired median bootstrap inputs must be aligned case matrices")
    case_count = control.shape[0]
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for draw in range(BOOTSTRAP_DRAWS):
        indices = rng.integers(0, case_count, size=case_count)
        draws[draw] = np.median(bound[indices]) - np.median(control[indices])
    return {
        "estimand": "v3_minus_v2_flattened_component_median_difference_case_cluster_bootstrap",
        "paired_case_count": case_count,
        "estimate": float(np.median(bound) - np.median(control)),
        "percentile_95_interval": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def _affected_fraction_by_case(root: Path, split_by_id: dict[str, str]) -> dict[str, float]:
    path = root / "dataset_v2_development/core/component_state_trace.jsonl.gz"
    result: dict[str, float] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            case_id = str(row["case_id"])
            split = split_by_id[case_id]
            if split == "test":
                continue
            records = row["component_state_trace"]["records"]
            values = [
                float(record["domain_censoring"]["penetration_affected_replicate"])
                for record in records
            ]
            if len(values) != 8:
                raise ValueError(f"development case {case_id} does not have R=8")
            result[case_id] = float(np.mean(values))
    return result


def _load(root: Path) -> dict[str, Any]:
    core = root / "dataset_v2_development/core"
    inputs = np.load(core / "inputs.npz", allow_pickle=False)
    labels = np.load(core / "component_labels.npz", allow_pickle=False)
    weights = np.load(core / "sampling_weights.npz", allow_pickle=False)
    ids = inputs["case_ids"].astype(str)
    if not np.array_equal(ids, labels["case_ids"].astype(str)):
        raise ValueError("input and label case order differs")
    if not np.array_equal(ids, weights["case_ids"].astype(str)):
        raise ValueError("input and weight case order differs")
    split = weights["split"].astype(str)
    test = split == "test"
    development = ~test
    for name in TARGETS:
        if np.any(labels[f"{name}_mask"][test]):
            raise ValueError(f"sealed test {name} mask is not all false")
    if np.any(labels["conditional_log1p_energy_mask"][test]):
        raise ValueError("sealed test conditional energy mask is not all false")
    if np.any(labels["conditional_log1p_hit_count_mask"][test]):
        raise ValueError("sealed test conditional hit-count mask is not all false")
    sealed = _json(root / "dataset_v2_development/sealed_test.json")
    if not (
        sealed["test_result_json_lines_decoded"] == 0
        and sealed["test_open_count"] == 0
        and sealed["test_labels_present_in_core"] is False
        and sealed["test_label_masks_all_false"] is True
    ):
        raise ValueError("sealed-test contract is not closed")
    split_by_id = dict(zip(ids.tolist(), split.tolist(), strict=True))
    affected = _affected_fraction_by_case(root, split_by_id)
    dev_ids = ids[development]
    affected_vector = np.asarray([affected[value] for value in dev_ids], dtype=np.float64)
    return {
        "root": root,
        "inputs": inputs,
        "labels": labels,
        "weights": weights,
        "case_ids": ids,
        "split": split,
        "development": development,
        "development_case_ids": dev_ids,
        "affected_fraction": affected_vector,
        "sealed": sealed,
        "audit": _json(root / "development_audit.json"),
        "manifest": _json(root / "dataset_v2_development/manifest.json"),
        "sampling_quality": _json(root / "audits/sampling_quality.json"),
        "formal_checkpoint": _json(root / "formal_dataset/checkpoint.json"),
        "formal_manifest": _json(root / "formal_dataset/manifest.json"),
    }


def _target_case_observation_rate(data: dict[str, Any], name: str) -> np.ndarray:
    mask = data["labels"][f"{name}_mask"][data["development"]]
    return np.mean(mask, axis=1)


def _interval_width(data: dict[str, Any], name: str) -> np.ndarray:
    interval = data["labels"][f"{name}_identification_interval"][data["development"]]
    return interval[..., 1] - interval[..., 0]


def _epsilon_complete_vector(data: dict[str, Any]) -> np.ndarray:
    """Match the dataset-v2 audit definition case by case.

    A case is epsilon-complete when every component interval is no wider than
    epsilon for all three probability targets.
    """
    complete = np.ones(int(np.sum(data["development"])), dtype=bool)
    for name in TARGETS:
        width = _interval_width(data, name)
        target_complete = np.all(width <= EPSILON + 1.0e-12, axis=1)
        expected_mask = width <= EPSILON + 1.0e-12
        observed_mask = data["labels"][f"{name}_mask"][data["development"]].astype(bool)
        if not np.array_equal(expected_mask, observed_mask):
            raise ValueError(f"{name} mask disagrees with epsilon interval rule")
        complete &= target_complete
    return complete


def _support(data: dict[str, Any], name: str, split_name: str) -> dict[str, Any]:
    selected = data["split"] == split_name
    values = data["labels"][name][selected]
    mask = data["labels"][f"{name}_mask"][selected]
    positive = np.sum(mask & (values > 0.0), axis=0)
    negative = np.sum(mask & (values <= 0.0), axis=0)
    observed = np.sum(mask, axis=0)
    return {
        "case_count": int(np.sum(selected)),
        "positive_count": {
            "minimum": int(np.min(positive)),
            "median": float(np.median(positive)),
            "maximum": int(np.max(positive)),
            "zero_component_count": int(np.sum(positive == 0)),
        },
        "negative_count": {
            "minimum": int(np.min(negative)),
            "median": float(np.median(negative)),
            "maximum": int(np.max(negative)),
            "zero_component_count": int(np.sum(negative == 0)),
        },
        "observed_count": {
            "minimum": int(np.min(observed)),
            "median": float(np.median(observed)),
            "maximum": int(np.max(observed)),
        },
    }


def _dataset_report(data: dict[str, Any]) -> dict[str, Any]:
    dev = data["development"]
    labels = data["labels"]
    affected = data["affected_fraction"]
    epsilon_complete = _epsilon_complete_vector(data)
    affected_replicate_count_float = float(np.sum(affected) * 8.0)
    affected_replicate_count = int(round(affected_replicate_count_float))
    if not np.isclose(affected_replicate_count_float, affected_replicate_count):
        raise ValueError("affected fractions do not correspond to R=8 replicate counts")
    audit_development = data["audit"]["development"]
    if not np.isclose(
        float(np.mean(affected)),
        float(audit_development["affected_replicate_fraction"]["mean"]),
    ):
        raise ValueError("derived affected fraction disagrees with development audit")
    if int(np.sum(epsilon_complete)) != int(audit_development["epsilon_complete_case_count"]):
        raise ValueError("derived epsilon-complete cases disagree with development audit")
    sampling = data["sampling_quality"]
    manifest = data["manifest"]
    checkpoint = data["formal_checkpoint"]
    completed_case_count = sum(
        len(shard["case_ids"]) for shard in checkpoint["completed_shards"]
    )
    if not (
        manifest["case_count"] == 60
        and sampling["selected_count"] == 60
        and checkpoint["status"] == "complete"
        and completed_case_count == 60
    ):
        raise ValueError("archive cardinality or completion status is inconsistent")
    report: dict[str, Any] = {
        "root": str(data["root"].relative_to(ROOT)).replace("\\", "/"),
        "development_case_count": int(np.sum(dev)),
        "split_counts": {
            name: int(np.sum(data["split"] == name))
            for name in ("train", "validation", "test")
        },
        "affected_replicate_fraction": {
            **_summary(affected),
            "affected_case_count": int(np.sum(affected > 0.0)),
            "affected_case_rate": float(np.mean(affected > 0.0)),
            "affected_replicate_count": affected_replicate_count,
            "total_development_replicate_count": int(affected.size * 8),
        },
        "fully_observed_case": {
            "count": int(np.sum(np.all(labels["label_mask"][dev].astype(bool), axis=1))),
            "rate": float(np.mean(np.all(labels["label_mask"][dev].astype(bool), axis=1))),
        },
        "epsilon_complete": {
            "epsilon": EPSILON,
            "definition": "all component identification intervals at most epsilon for all three probability targets",
            "count": int(np.sum(epsilon_complete)),
            "rate": float(np.mean(epsilon_complete)),
        },
        "target_observation_rate": {},
        "identification_interval_width": {},
        "conditional_observation_rate": {
            "energy": float(np.mean(labels["conditional_log1p_energy_mask"][dev])),
            "hit_count": float(np.mean(labels["conditional_log1p_hit_count_mask"][dev])),
        },
        "support": {},
        "development_support": {
            name: {
                "positive": audit_development["component_positive_support"][name],
                "negative": audit_development["component_negative_support"][name],
                "observed": audit_development["component_observed_support"][name],
            }
            for name in TARGETS
        },
        "missingness_ks": {
            key: data["audit"]["missingness_ks"][key]
            for key in (
                "alpha",
                "tested_record_count",
                "evaluated_record_count",
                "significant_record_count",
                "maximum_ks_statistic",
            )
        },
        "sampling_quality": {
            "candidate_count": sampling["candidate_count"],
            "selected_count": sampling["selected_count"],
            "selected_split_counts": sampling["sets"]["selected"]["split_counts"],
            "positivity_min": sampling["positivity_min"],
            "positivity_max": sampling["positivity_max"],
            "design_weight": sampling["design_weight"],
            "effective_sample_size": sampling["effective_sample_size"],
            "ess_fraction": sampling["ess_fraction"],
            "registry_assignment_status": sampling["registry_assignment_status"],
            "coverage_status": sampling["coverage_status"],
            "coverage_guaranteed": sampling["coverage_guaranteed"],
            "quality_gates": sampling["quality_gates"],
            "quality_gate_passed": sampling["quality_gate_passed"],
            "quality_gate_scope": sampling["quality_gate_scope"],
        },
        "archive_integrity": {
            "dataset_v2": {
                key: manifest[key]
                for key in (
                    "schema_version",
                    "dataset_id",
                    "status",
                    "provisional_reason",
                    "core_tree_independent",
                    "case_count",
                    "component_count",
                    "label_spec_id",
                    "source_code_hash",
                    "manifest_hash",
                )
            },
            "dataset_validation_identity": data["audit"]["validation_identity"],
            "formal_checkpoint": {
                "schema_version": checkpoint["schema_version"],
                "status": checkpoint["status"],
                "completed_shard_count": len(checkpoint["completed_shards"]),
                "completed_case_count": completed_case_count,
                "dataset_id": checkpoint["dataset_id"],
                "root_manifest_hash": checkpoint["root_manifest_hash"],
            },
            "result_access": data["audit"]["result_access"],
            "development_audit_report_hash": data["audit"]["report_hash"],
        },
        "training_readiness": data["audit"]["training_readiness"],
        "component_task_admission_summary": {
            name: {
                "train_eligible_component_count": int(
                    data["audit"]["component_task_admission"][name]["train_eligible_component_count"]
                ),
                "validation_evaluable_component_count": int(
                    data["audit"]["component_task_admission"][name]["validation_evaluable_component_count"]
                ),
                "admitted_component_count": int(
                    data["audit"]["component_task_admission"][name]["admitted_component_count"]
                ),
            }
            for name in TARGETS
        },
        "sealed_test": data["sealed"],
        "source_hashes": {
            "development_audit_sha256": _sha256(data["root"] / "development_audit.json"),
            "dataset_v2_manifest_sha256": _sha256(
                data["root"] / "dataset_v2_development/manifest.json"
            ),
            "sealed_test_sha256": _sha256(
                data["root"] / "dataset_v2_development/sealed_test.json"
            ),
            "sampling_quality_sha256": _sha256(
                data["root"] / "audits/sampling_quality.json"
            ),
            "formal_checkpoint_sha256": _sha256(
                data["root"] / "formal_dataset/checkpoint.json"
            ),
            "formal_manifest_sha256": _sha256(
                data["root"] / "formal_dataset/manifest.json"
            ),
        },
    }
    for name in TARGETS:
        observation = _target_case_observation_rate(data, name)
        widths = _interval_width(data, name)
        report["target_observation_rate"][name] = float(np.mean(observation))
        if not np.isclose(
            report["target_observation_rate"][name],
            float(audit_development["target_observation_rate"][name]),
        ):
            raise ValueError(f"derived {name} observation rate disagrees with audit")
        report["identification_interval_width"][name] = _summary(widths.reshape(-1))
        report["support"][name] = {
            split_name: _support(data, name, split_name)
            for split_name in ("train", "validation")
        }
    return report


def main() -> None:
    control = _load(CONTROL)
    bound = _load(BOUND)
    if not np.array_equal(control["case_ids"], bound["case_ids"]):
        raise ValueError("v2/v3 case identities differ")
    if not np.array_equal(control["split"], bound["split"]):
        raise ValueError("v2/v3 split identities differ")
    if not np.array_equal(control["inputs"]["x"], bound["inputs"]["x"]):
        raise ValueError("v2/v3 learner inputs differ")
    if not np.array_equal(control["development_case_ids"], bound["development_case_ids"]):
        raise ValueError("v2/v3 development case order differs")

    control_epsilon_complete = _epsilon_complete_vector(control)
    bound_epsilon_complete = _epsilon_complete_vector(bound)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparisons: dict[str, Any] = {
        "affected_replicate_fraction_mean": _bootstrap_mean_difference(
            control["affected_fraction"], bound["affected_fraction"], rng
        ),
        "epsilon_complete_rate": _bootstrap_mean_difference(
            control_epsilon_complete.astype(float),
            bound_epsilon_complete.astype(float),
            rng,
        ),
        "target_observation_rate": {},
        "identification_interval_width_mean": {},
        "identification_interval_width_median": {},
        "conditional_observation_rate": {},
    }
    for name in TARGETS:
        comparisons["target_observation_rate"][name] = _bootstrap_mean_difference(
            _target_case_observation_rate(control, name),
            _target_case_observation_rate(bound, name),
            rng,
        )
        control_width = _interval_width(control, name)
        bound_width = _interval_width(bound, name)
        comparisons["identification_interval_width_mean"][name] = (
            _bootstrap_mean_difference(
                np.mean(control_width, axis=1), np.mean(bound_width, axis=1), rng
            )
        )
        comparisons["identification_interval_width_median"][name] = (
            _bootstrap_flat_median_difference(control_width, bound_width, rng)
        )
    for metric in ("energy", "hit_count"):
        key = (
            "conditional_log1p_energy_mask"
            if metric == "energy"
            else "conditional_log1p_hit_count_mask"
        )
        comparisons["conditional_observation_rate"][metric] = _bootstrap_mean_difference(
            np.mean(control["labels"][key][control["development"]], axis=1),
            np.mean(bound["labels"][key][bound["development"]], axis=1),
            rng,
        )

    report = {
        "schema_version": "pipeline-teacher-paired-audit@1.0.0",
        "analysis_id": "pipeline-expected-actual-20261122-v1",
        "analysis_scope": "paired_train_validation_only_with_opaque_test_placeholders",
        "analysis_status": "development_descriptive_exploratory",
        "bootstrap": {
            "unit": "paired_case",
            "seed": BOOTSTRAP_SEED,
            "draws": BOOTSTRAP_DRAWS,
            "interval": "percentile_95",
        },
        "identity_checks": {
            "case_ids_exact_match": True,
            "split_assignments_exact_match": True,
            "learner_inputs_exact_match": True,
            "test_case_id_order_hash_exact_match": (
                control["sealed"]["test_case_id_order_hash"]
                == bound["sealed"]["test_case_id_order_hash"]
            ),
        },
        "v2_control": _dataset_report(control),
        "v3_bound": _dataset_report(bound),
        "paired_comparisons": comparisons,
        "interpretation_guards": {
            "sealed_test_results_decoded": 0,
            "test_metrics_computed": False,
            "physical_validation_claim": False,
            "bootstrap_scope": (
                "quantifies uncertainty across the 45 fixed development cases; "
                "it does not quantify independent scrambled-design or real-world uncertainty"
            ),
        },
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "output_sha256": _sha256(OUTPUT),
        "development_case_count": 45,
        "test_result_json_lines_decoded": 0,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
