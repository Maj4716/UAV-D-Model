from copy import deepcopy

import numpy as np
import pytest

from damage_engine.errors import InputValidationError
from damage_engine.multitask_targets import MultiTaskTargetAdapter
from damage_engine.system_learning import build_system_learning_contract


def _source_hashes():
    return {
        "target_hash": "sha256:" + "1" * 64,
        "physics_registry_hash": "sha256:" + "2" * 64,
        "engine_code_hash": "sha256:" + "3" * 64,
        "damage_tree_definition_hash": "sha256:" + "4" * 64,
        "label_spec_hash": "sha256:" + "b" * 64,
        "sampling_design_hash": "sha256:" + "5" * 64,
        "split_registry_hash": "sha256:" + "6" * 64,
    }


def metadata():
    return {
        "component_aux_label_names": (
            "fragment_hits", "fragment_penetrations",
            "fragment_mean_arrival_speed_mps", "fragment_mean_penetration_margin",
            "fragment_deposited_energy_j", "blast_effective_peak_pressure_pa",
            "blast_effective_impulse_pa_s", "blast_positive_duration_s",
            "blast_arrival_time_s",
        ),
        "system_output_names": ("mission_degraded", "mission_loss"),
        "system_learning_binding": {
            "dataset_id": "synthetic",
            "manifest_hash": "sha256:" + "a" * 64,
            "source_contract_hashes": _source_hashes(),
        },
    }


def sample(hit_mean=0.0):
    aux = np.zeros((1, 9), dtype=np.float32)
    aux[0, 0] = hit_mean
    return {
        "component_aux_y": aux,
        "component_aux_label_mask": np.ones_like(aux, dtype=bool),
        "component_y": np.zeros((1, 3, 2), dtype=np.float32),
        "component_label_mask": np.ones((1, 3, 2), dtype=bool),
        "armor_y": np.zeros((1, 2), dtype=np.float32),
        "armor_label_mask": np.ones((1, 2), dtype=bool),
        "system_y": np.asarray([0.3, 0.1], dtype=np.float32),
        "system_label_mask": np.ones(2, dtype=bool),
        "sampling_design_weight_raw": np.asarray(1.0, dtype=np.float64),
        "sampling_weight_mask": np.asarray(True),
    }


def test_multitask_adapter_preserves_conditioning_and_direct_system_output():
    adapter = MultiTaskTargetAdapter(
        metadata(), allow_unfiltered_system_tasks_for_software_audit=True
    )
    zero = adapter(sample(0.0))
    hit = adapter(sample(0.25))
    assert not zero["fragment_nonzero_gate"]["target"].item()
    assert not zero["fragment_conditional_response"]["mask"].any()
    assert hit["fragment_conditional_response"]["mask"].all()
    assert hit["system_tree"]["source"].startswith("direct_archived")
    assert hit["system_tree"]["activation"] == "independent_sigmoid"
    assert hit["sampling_risk_weight"]["must_not_be_reused_as"] == (
        "training_class_weight", "minibatch_sampling_probability"
    )


def test_multitask_adapter_requires_system_outputs_and_masks():
    value = sample()
    value.pop("system_y")
    with pytest.raises(InputValidationError, match="missing staged"):
        MultiTaskTargetAdapter(
            metadata(), allow_unfiltered_system_tasks_for_software_audit=True
        )(value)


def _learning_contract():
    return build_system_learning_contract(
        np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        np.ones((3, 2), dtype=bool),
        ("mission_degraded", "mission_loss"),
        split_names=("train", "train", "validation"),
        sampling_roles=(
            "reference_train",
            "reference_train",
            "reference_validation",
        ),
        dataset_id="synthetic",
        dataset_manifest_hash="sha256:" + "a" * 64,
        test_status="sealed_unobserved",
        source_contract_hashes=_source_hashes(),
        minimum_soft_event_mass={"train": 1.0, "validation": 1.0, "test": 1.0},
    )


def test_adapter_applies_explicit_training_eligibility_without_rewriting_masks():
    value = sample()
    original_mask = value["system_label_mask"].copy()
    adapter = MultiTaskTargetAdapter(
        metadata(), system_learning_contract=_learning_contract()
    )
    result = adapter(value)["system_tree"]
    assert result["task_names"] == ("mission_degraded",)
    assert result["target"].shape == (1,)
    assert result["mask"].tolist() == [True]
    assert result["eligibility_enforced"] is True
    assert np.array_equal(value["system_label_mask"], original_mask)


def test_adapter_refuses_constant_requested_task_and_unbound_selection():
    with pytest.raises(InputValidationError, match="not eligible"):
        MultiTaskTargetAdapter(
            metadata(),
            system_learning_contract=_learning_contract(),
            requested_system_tasks=("mission_loss",),
        )
    with pytest.raises(InputValidationError, match="require a bound"):
        MultiTaskTargetAdapter(
            metadata(), requested_system_tasks=("mission_degraded",)
        )


def test_fragment_conditioning_cannot_be_opened_by_an_unavailable_placeholder():
    value = sample(3.0)
    value["component_aux_label_mask"][0, 0] = False
    result = MultiTaskTargetAdapter(
        metadata(), allow_unfiltered_system_tasks_for_software_audit=True
    )(value)
    assert result["fragment_nonzero_gate"]["target"].tolist() == [False]
    assert not result["fragment_conditional_response"]["mask"].any()


def test_adapter_uses_dataset_axis_when_contract_task_records_are_reordered():
    contract = _learning_contract()
    contract["tasks"] = list(reversed(contract["tasks"]))
    result = MultiTaskTargetAdapter(
        metadata(),
        system_learning_contract=contract,
        requested_system_tasks=("mission_degraded",),
    )(sample())["system_tree"]
    assert result["task_names"] == ("mission_degraded",)
    assert result["target"].tolist() == pytest.approx([0.3])


@pytest.mark.parametrize("field", ["dataset_id", "manifest_hash", "source_contract_hashes"])
def test_adapter_refuses_cross_archive_contract_bindings(field):
    contract = deepcopy(_learning_contract())
    if field == "source_contract_hashes":
        contract["dataset_binding"][field] = {
            **_source_hashes(),
            "label_spec_hash": "sha256:" + "c" * 64,
        }
    elif field == "manifest_hash":
        contract["dataset_binding"][field] = "sha256:" + "d" * 64
    else:
        contract["dataset_binding"][field] = "different"
    with pytest.raises(InputValidationError, match="different dataset archive"):
        MultiTaskTargetAdapter(metadata(), system_learning_contract=contract)


def test_adapter_refuses_missing_or_malformed_required_source_hashes():
    missing = deepcopy(_learning_contract())
    missing["dataset_binding"]["source_contract_hashes"].pop("target_hash")
    with pytest.raises(InputValidationError, match="binding is incomplete"):
        MultiTaskTargetAdapter(metadata(), system_learning_contract=missing)

    malformed_contract = deepcopy(_learning_contract())
    malformed_metadata = metadata()
    malformed_contract["dataset_binding"]["source_contract_hashes"][
        "target_hash"
    ] = "sha256:" + "z" * 64
    malformed_metadata["system_learning_binding"]["source_contract_hashes"][
        "target_hash"
    ] = "sha256:" + "z" * 64
    with pytest.raises(InputValidationError, match="complete sha256"):
        MultiTaskTargetAdapter(
            malformed_metadata, system_learning_contract=malformed_contract
        )

    malformed_manifest_contract = deepcopy(_learning_contract())
    malformed_manifest_metadata = metadata()
    malformed_manifest = "sha256:" + "z" * 64
    malformed_manifest_contract["dataset_binding"]["manifest_hash"] = (
        malformed_manifest
    )
    malformed_manifest_metadata["system_learning_binding"]["manifest_hash"] = (
        malformed_manifest
    )
    with pytest.raises(InputValidationError, match="manifest hashes"):
        MultiTaskTargetAdapter(
            malformed_manifest_metadata,
            system_learning_contract=malformed_manifest_contract,
        )


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown"])
def test_adapter_refuses_incomplete_or_ambiguous_contract_task_records(mutation):
    contract = deepcopy(_learning_contract())
    if mutation == "missing":
        contract["tasks"].pop()
    elif mutation == "duplicate":
        contract["tasks"][1]["task_name"] = contract["tasks"][0]["task_name"]
    else:
        contract["tasks"][1]["task_name"] = "unknown"
    with pytest.raises(InputValidationError, match="cover the dataset vocabulary"):
        MultiTaskTargetAdapter(metadata(), system_learning_contract=contract)


def test_adapter_requires_contract_unless_software_audit_override_is_explicit():
    with pytest.raises(InputValidationError, match="contract is required"):
        MultiTaskTargetAdapter(metadata())

    adapter = MultiTaskTargetAdapter(
        metadata(), allow_unfiltered_system_tasks_for_software_audit=True
    )
    assert adapter.objective_contract["system_task_selection"] == {
        "task_names": ["mission_degraded", "mission_loss"],
        "eligibility_enforced": False,
        "software_audit_override": True,
        "constant_tasks_are_not_relabelled_as_missing": True,
    }

    with pytest.raises(InputValidationError, match="explicit boolean"):
        MultiTaskTargetAdapter(
            metadata(), allow_unfiltered_system_tasks_for_software_audit="true"
        )
