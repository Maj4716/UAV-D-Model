"""Generic evidence-record, array-mask and metadata-binding review.

No domain engine is constructed, no domain labels are generated, and no model
is fitted. The adapter receives vocabulary strings and abstract task metadata
only; no physical or response arrays are supplied to it.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from damage_engine.scientific_validation import validate_scientific_validation_record
from damage_engine.system_learning import build_system_learning_contract
from damage_engine.multitask_targets import MultiTaskTargetAdapter


def general_record():
    return {
        "schema_version": "1.0.0",
        "independent_data_sources": [{"source_id": "abstract-heldout-source", "role": "validation",
            "content_hash": "sha256:" + "1" * 64, "used_for_calibration": False}],
        "dimensional_error_metrics": [{"metric_id": "abstract-error", "mechanism": "generic-system", "unit": "1",
            "observed_error": 0.1, "acceptance_limit": 0.2, "passed": True}],
        "interval_coverage": [{"metric_id": "abstract-coverage", "mechanism": "generic-system",
            "nominal": 0.95, "observed": 0.95, "sample_count": 20, "passed": True}],
        "mechanism_stratified_errors": [{"stratum_id": "abstract-group", "metric_id": "abstract-error",
            "value": 0.1, "acceptance_limit": 0.2, "passed": True}],
        "extrapolation_failure_rules": [{"rule_id": "abstract-boundary", "condition": "outside declared scope", "action": "refuse"}],
        "uncertainty_ledger": {"parameter_epistemic": "assessed", "model_form": "assessed", "numerical_error": "assessed"},
        "approval": {"approver_id": "artificial-reviewer", "approved_at": "2026-10-13T00:00:00Z", "protocol_version": "abstract-protocol-1"},
    }


def accepted(record):
    try:
        validate_scientific_validation_record(record)
        return {"accepted_by_structure_gate": True}
    except Exception as error:
        return {"accepted_by_structure_gate": False, "error_type": type(error).__name__, "error": str(error)}


def toy_contract(names=("abstract-task-A",), weights_mask=None):
    values = np.asarray([[0.2, 0.4], [0.8, 0.9]])[:, :len(names)]
    return build_system_learning_contract(
        values, np.ones_like(values, dtype=bool), names,
        split_names=["train", "train"], sampling_roles=["reference_train", "reference_train"],
        dataset_id="abstract-dataset-A", dataset_manifest_hash="sha256:" + "a" * 64,
        test_status="observed_diagnostic_only", design_weight_mask=weights_mask,
        minimum_soft_event_mass={"train": 0.1, "validation": 0.1, "test": 0.1},
    )


def vocabulary_metadata(system_names):
    # Existing API vocabulary is required, but no response data is supplied.
    return {
        "dataset_id": "abstract-dataset-B",
        "dataset_manifest_hash": "sha256:" + "b" * 64,
        "system_output_names": list(system_names),
        "component_aux_label_names": [
            "fragment_hits", "fragment_penetrations", "fragment_mean_arrival_speed_mps",
            "fragment_mean_penetration_margin", "fragment_deposited_energy_j",
            "blast_effective_peak_pressure_pa", "blast_effective_impulse_pa_s",
            "blast_positive_duration_s", "blast_arrival_time_s",
        ],
    }


def main():
    evidence = {
        "scope": "abstract records, artificial scalar arrays and adapter metadata only; no engine, domain labels, model fitting or pilot generation",
        "python": platform.python_version(), "numpy": np.__version__,
        "scientific_structure_gate": {},
    }
    coverage = general_record()
    coverage["interval_coverage"][0].update(observed=0.0)
    evidence["scientific_structure_gate"]["zero_observed_coverage_with_passed_true"] = {
        **accepted(coverage), "coverage_record": coverage["interval_coverage"][0],
        "interpretation": "The gate does not bind or recompute a coverage acceptance criterion. Acceptance is a shape/declaration result, not demonstrated statistical adequacy.",
    }
    duplicate = general_record()
    duplicate["independent_data_sources"].append({"source_id": "abstract-calibration-source", "role": "calibration",
        "content_hash": duplicate["independent_data_sources"][0]["content_hash"], "used_for_calibration": True})
    evidence["scientific_structure_gate"]["same_content_claimed_calibration_and_holdout"] = {
        **accepted(duplicate), "source_records": duplicate["independent_data_sources"],
        "interpretation": "Conflicting uses of the same content identity are not cross-checked; no distinct subset-membership identity is recorded.",
    }
    boolean_count = general_record()
    boolean_count["interval_coverage"][0]["sample_count"] = True
    evidence["scientific_structure_gate"]["boolean_sample_count"] = {
        **accepted(boolean_count), "sample_count": True,
    }
    contract = toy_contract()
    metadata = vocabulary_metadata(("abstract-task-A",))
    adapter = MultiTaskTargetAdapter(metadata, system_learning_contract=contract)
    evidence["adapter_cross_archive_identity"] = {
        "contract_dataset_id": contract["dataset_binding"]["dataset_id"],
        "contract_manifest_hash": contract["dataset_binding"]["manifest_hash"],
        "metadata_dataset_id": metadata["dataset_id"], "metadata_manifest_hash": metadata["dataset_manifest_hash"],
        "construction_accepted": True, "eligibility_enforced": adapter._eligibility_enforced,
        "selected_task_names": list(adapter._system_task_names),
        "interpretation": "Vocabulary equality is enforced, but actual archive identity is not verified by this adapter.",
    }
    reordered = toy_contract(("abstract-task-A", "abstract-task-B"))
    reordered["tasks"].reverse()
    adapter_reordered = MultiTaskTargetAdapter(vocabulary_metadata(("abstract-task-A", "abstract-task-B")),
        system_learning_contract=reordered, requested_system_tasks=("abstract-task-A",))
    evidence["adapter_task_order_mismatch"] = {
        "bound_vocabulary": reordered["dataset_binding"]["system_output_names"],
        "task_record_order": [item["task_name"] for item in reordered["tasks"]],
        "requested_task_names": ["abstract-task-A"], "selected_task_names": list(adapter_reordered._system_task_names),
        "selected_indices": adapter_reordered._system_indices, "eligibility_enforced": adapter_reordered._eligibility_enforced,
        "interpretation": "Task-record position is used as the tensor-axis index even though the declared vocabulary provides a separate name order.",
    }
    no_weights = toy_contract(weights_mask=np.asarray([False, False]))
    task = no_weights["tasks"][0]
    evidence["label_availability_vs_weight_availability"] = {
        "input_label_mask": [[True], [True]], "input_design_weight_mask": [False, False],
        "reported_available_count": task["support"]["all_train"]["available_count"],
        "reported_label_available": task["label_available"], "status": task["status"], "reason": task["reason"],
        "interpretation": "Label availability and applicability of design weights share one filtered denominator and one no-label reason.",
    }
    evidence["boundary"] = "These reproduce API consistency gaps only; outer archive admission and actual research use were not exercised. No acceptance thresholds or domain calibration choices are proposed."
    destination = Path(__file__).with_suffix(".json")
    if destination.exists():
        raise RuntimeError("Preserve existing evidence and choose a fresh output filename.")
    destination.write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"evidence": str(destination), "coverage_record_accepted": evidence["scientific_structure_gate"]["zero_observed_coverage_with_passed_true"]["accepted_by_structure_gate"],
        "cross_archive_adapter_accepted": True, "requested_task": "abstract-task-A", "selected_task": list(adapter_reordered._system_task_names)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
