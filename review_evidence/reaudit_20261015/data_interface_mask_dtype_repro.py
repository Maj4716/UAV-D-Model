"""Read-only interface audit: synthetic masks and numeric dtype boundaries.

No simulator, training, domain-parameter choice, sampling, or existing dataset
mutation occurs. Existing artificial test fixtures supply only toy tensors.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from damage_engine.multitask_targets import MultiTaskTargetAdapter
from damage_engine.training_view import TrainingViewDataset, fit_training_view_artifact
from damage_engine.training_view import _source_contracts
from damage_engine.normalization import make_training_normalization_artifact, apply_training_normalization


def load_fixture(filename: str, name: str):
    specification = importlib.util.spec_from_file_location(name, ROOT / "tests" / filename)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main() -> None:
    report = {"scope": "synthetic mask and dtype contract tests; no simulator or training",
              "executable": sys.executable, "torch": torch.__version__}
    fixtures = load_fixture("test_multitask_targets.py", "artificial_multitask_fixture")
    adapter = MultiTaskTargetAdapter(fixtures.metadata(), allow_unfiltered_system_tasks_for_software_audit=True)
    first = fixtures.sample(0.25)
    first["component_aux_label_mask"][0, 0] = False
    second = fixtures.sample(0.0)
    second["component_aux_label_mask"][0, 0] = False
    first_output = adapter(first)
    second_output = adapter(second)
    report["unavailable_gate_placeholder_changes_dependent_mask"] = {
        "scope": "only adapter array-selection behavior; explicit software-audit override",
        "gate_available_first": first_output["fragment_nonzero_gate"]["mask"].tolist(),
        "gate_available_second": second_output["fragment_nonzero_gate"]["mask"].tolist(),
        "dependent_mask_with_positive_masked_placeholder": first_output["fragment_conditional_response"]["mask"].tolist(),
        "dependent_mask_with_zero_masked_placeholder": second_output["fragment_conditional_response"]["mask"].tolist(),
    }
    assert not first_output["fragment_nonzero_gate"]["mask"].any()
    assert first_output["fragment_conditional_response"]["mask"].any()
    assert not second_output["fragment_conditional_response"]["mask"].any()

    view_fixtures = load_fixture("test_training_view_contract.py", "artificial_training_fixture")
    dataset = view_fixtures.ArtificialTrainingDataset()
    artifact = fit_training_view_artifact(dataset, proxy_boundary="end_to_end")
    numeric_boundaries = []
    for value in (1.0e300, 1.0e-300):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            view = TrainingViewDataset(dataset, artifact, proxy_boundary="end_to_end",
                                       task_loss_weights={"system-events": value})
            observed = view[0]["task_loss_weights"]
        numeric_boundaries.append({"input_float64": value,
                                   "accepted": True,
                                   "output_dtype": str(observed.dtype),
                                   "output_value_text": str(observed.item()),
                                   "output_is_finite": bool(torch.isfinite(observed).all()),
                                   "output_is_positive": bool((observed > 0).all()),
                                   "warnings": [str(item.message) for item in captured]})
    report["task_weight_cast_invalidates_validated_contract"] = numeric_boundaries
    assert numeric_boundaries[0]["output_is_finite"] is False
    assert numeric_boundaries[1]["output_is_positive"] is False

    # A standalone generic numeric representation, without domain input names.
    tiny_scale_artifact = make_training_normalization_artifact(
        fit_case_ids=["toy-train-a", "toy-train-b"],
        source_contracts=_source_contracts(dataset, ["toy-feature"], ["1"]),
        feature_names=["toy-feature"], feature_units=["1"],
        feature_values=np.asarray([[0.0], [1.0e-40]], dtype=np.float32),
        feature_mask=np.ones((2, 1), dtype=bool),
        proxy_boundary={"mode": "end_to_end", "deployment_available": True,
                        "derived_input_contract_hash": None},
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        transformed = apply_training_normalization(
            np.asarray([1.0], dtype=np.float32), np.asarray([True]), tiny_scale_artifact)
    report["finite_generic_input_and_valid_artifact_produce_nonfinite_normalization"] = {
        "fitted_scale": tiny_scale_artifact["features"][0]["scale"],
        "input": [1.0], "input_available": [True],
        "output_value_text": str(transformed[0]),
        "output_is_finite": bool(np.isfinite(transformed).all()),
        "warnings": [str(item.message) for item in captured],
    }
    assert not np.isfinite(transformed).all()
    destination = Path(__file__).with_suffix(".json")
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
