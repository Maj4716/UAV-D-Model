"""Mask-preserving staged supervision for control-surrogate training.

The adapter does not recompute damage-tree outputs.  It only exposes tensors
already archived by the engine and keeps design weights separate from task or
minibatch weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .content_contract import is_sha256_identity
from .errors import InputValidationError
from .system_learning import (
    SYSTEM_LEARNING_REQUIRED_SOURCE_HASH_FIELDS,
    assert_trainable_tasks,
)


def _positions(names: Sequence[str], requested: Sequence[str], group: str) -> list[int]:
    if not names or len(set(names)) != len(names) or any(not name for name in names):
        raise InputValidationError(f"{group} vocabulary must be unique and nonempty")
    if len(set(requested)) != len(requested) or any(not name for name in requested):
        raise InputValidationError(f"requested {group} names must be unique and nonempty")
    lookup = {name: index for index, name in enumerate(names)}
    missing = [name for name in requested if name not in lookup]
    if missing:
        raise InputValidationError(
            f"{group} vocabulary is missing required staged targets",
            context={"missing": missing},
        )
    return [lookup[name] for name in requested]


def _selected(value, indices: list[int]):
    return value[..., indices]


@dataclass(frozen=True)
class MultiTaskTargetAdapter:
    """Build staged targets from one Torch/NumPy dataset sample.

    ``metadata`` is the dataset loader's metadata mapping.  The returned masks
    remain authoritative; unavailable quantities are never filled and trained
    as physical zeroes.
    """

    metadata: Mapping[str, object]
    system_learning_contract: Mapping | None = None
    requested_system_tasks: Sequence[str] | None = None
    allow_exploratory_low_mass: bool = False
    allow_unfiltered_system_tasks_for_software_audit: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.allow_unfiltered_system_tasks_for_software_audit, bool):
            raise InputValidationError(
                "software-audit override must be an explicit boolean"
            )
        aux = tuple(self.metadata.get("component_aux_label_names", ()))
        if not aux:
            raise InputValidationError("component auxiliary vocabulary is required")
        object.__setattr__(
            self,
            "_fragment_indices",
            _positions(
                aux,
                (
                    "fragment_hits",
                    "fragment_penetrations",
                    "fragment_mean_arrival_speed_mps",
                    "fragment_mean_penetration_margin",
                    "fragment_deposited_energy_j",
                ),
                "component auxiliary",
            ),
        )
        object.__setattr__(
            self,
            "_blast_indices",
            _positions(
                aux,
                (
                    "blast_effective_peak_pressure_pa",
                    "blast_effective_impulse_pa_s",
                    "blast_positive_duration_s",
                    "blast_arrival_time_s",
                ),
                "component auxiliary",
            ),
        )
        system_names = tuple(self.metadata.get("system_output_names", ()))
        if not system_names:
            raise InputValidationError("direct system output vocabulary is required")
        if len(set(system_names)) != len(system_names) or any(
            not isinstance(name, str) or not name for name in system_names
        ):
            raise InputValidationError(
                "direct system output vocabulary must contain unique nonempty strings"
            )
        if self.system_learning_contract is None:
            if self.requested_system_tasks is not None:
                raise InputValidationError(
                    "requested system tasks require a bound system-learning contract"
                )
            if not self.allow_unfiltered_system_tasks_for_software_audit:
                raise InputValidationError(
                    "system-learning contract is required for system-task exposure; "
                    "unfiltered outputs require an explicit software-audit override"
                )
            system_indices = tuple(range(len(system_names)))
            selected_names = system_names
            eligibility_enforced = False
        else:
            dataset_binding = self.metadata.get("system_learning_binding")
            contract_binding = self.system_learning_contract.get("dataset_binding")
            if not isinstance(dataset_binding, Mapping) or not isinstance(
                contract_binding, Mapping
            ):
                raise InputValidationError(
                    "system-learning contract requires complete dataset binding metadata"
                )
            for field in ("dataset_id", "manifest_hash", "source_contract_hashes"):
                if field not in dataset_binding or field not in contract_binding:
                    raise InputValidationError(
                        "system-learning dataset binding is incomplete",
                        context={"field": field},
                    )
            if any(
                not isinstance(binding["dataset_id"], str)
                or not binding["dataset_id"]
                for binding in (dataset_binding, contract_binding)
            ):
                raise InputValidationError(
                    "system-learning dataset IDs must be nonempty strings"
                )
            if any(
                not is_sha256_identity(
                    binding["manifest_hash"],
                    acceptance="legacy_lowercase_python_hex",
                )
                for binding in (dataset_binding, contract_binding)
            ):
                raise InputValidationError(
                    "system-learning manifest hashes must be complete lowercase sha256 identities"
                )
            dataset_hashes = dataset_binding["source_contract_hashes"]
            contract_hashes = contract_binding["source_contract_hashes"]
            if not isinstance(dataset_hashes, Mapping) or not isinstance(
                contract_hashes, Mapping
            ):
                raise InputValidationError(
                    "system-learning source-contract binding must be an object"
                )
            missing_hashes = sorted(
                (SYSTEM_LEARNING_REQUIRED_SOURCE_HASH_FIELDS - set(dataset_hashes))
                | (SYSTEM_LEARNING_REQUIRED_SOURCE_HASH_FIELDS - set(contract_hashes))
            )
            if missing_hashes:
                raise InputValidationError(
                    "system-learning dataset binding is incomplete",
                    context={"source_contract_hashes": missing_hashes},
                )
            invalid_hashes = sorted(
                name
                for name in set(dataset_hashes) | set(contract_hashes)
                if any(
                    not is_sha256_identity(
                        mapping.get(name),
                        acceptance="legacy_lowercase_python_hex",
                    )
                    for mapping in (dataset_hashes, contract_hashes)
                )
            )
            if invalid_hashes:
                raise InputValidationError(
                    "system-learning source-contract hashes must be complete sha256 identities",
                    context={"source_contract_hashes": invalid_hashes},
                )
            if any(
                dataset_binding[field] != contract_binding[field]
                for field in ("dataset_id", "manifest_hash", "source_contract_hashes")
            ):
                raise InputValidationError(
                    "system-learning contract is bound to a different dataset archive"
                )
            bound_names = tuple(
                map(
                    str,
                    self.system_learning_contract.get("dataset_binding", {}).get(
                        "system_output_names", ()
                    ),
                )
            )
            if bound_names != system_names:
                raise InputValidationError(
                    "system-learning contract output vocabulary does not match the dataset"
                )
            task_records = self.system_learning_contract.get("tasks")
            if not isinstance(task_records, Sequence) or isinstance(
                task_records, (str, bytes)
            ):
                raise InputValidationError(
                    "system-learning contract tasks must be a complete sequence"
                )
            contract_task_names = [
                item.get("task_name") if isinstance(item, Mapping) else None
                for item in task_records
            ]
            if (
                any(not isinstance(name, str) or not name for name in contract_task_names)
                or len(set(contract_task_names)) != len(contract_task_names)
                or set(contract_task_names) != set(system_names)
            ):
                raise InputValidationError(
                    "system-learning contract task records must cover the dataset vocabulary exactly once"
                )
            requested = (
                tuple(self.requested_system_tasks)
                if self.requested_system_tasks is not None
                else tuple(
                    self.system_learning_contract.get("selection_policy", {}).get(
                        "default_task_names", ()
                    )
                )
            )
            # Eligibility lives in the contract records, but tensor positions
            # live only in the dataset vocabulary.  Contract record order is
            # intentionally irrelevant.
            assert_trainable_tasks(
                self.system_learning_contract,
                requested,
                allow_exploratory_low_mass=self.allow_exploratory_low_mass,
            )
            system_indices = tuple(
                _positions(system_names, requested, "system output")
            )
            selected_names = tuple(system_names[index] for index in system_indices)
            eligibility_enforced = True
        object.__setattr__(self, "_system_indices", list(system_indices))
        object.__setattr__(self, "_system_task_names", selected_names)
        object.__setattr__(self, "_eligibility_enforced", eligibility_enforced)

    @property
    def objective_contract(self) -> dict:
        return {
            "schema_version": "1.0.0",
            "stages": [
                {
                    "stage": "fragment_nonzero_gate",
                    "target": "mean_fragment_hits_gt_zero",
                    "recommended_loss": "sigmoid_binary_cross_entropy_or_brier",
                },
                {
                    "stage": "fragment_conditional_response",
                    "target": "penetration_speed_margin_energy_given_nonzero_mean_hits",
                    "recommended_loss": "masked_huber_or_count_likelihood",
                },
                {
                    "stage": "blast_local_response",
                    "target": "effective_pressure_impulse_duration_arrival",
                    "recommended_loss": "masked_scaled_huber_or_gaussian_nll",
                },
                {
                    "stage": "component_mechanism_state",
                    "target": "archived_component_y",
                    "recommended_loss": "masked_probability_scoring_rule",
                },
                {
                    "stage": "armor_response",
                    "target": "archived_armor_y",
                    "recommended_loss": "masked_response_specific_objectives",
                },
                {
                    "stage": "system_tree",
                    "target": "direct_archived_system_y",
                    "recommended_loss": "independent_sigmoid_soft_bce_or_brier_not_softmax",
                },
            ],
            "weight_semantics": {
                "sampling_design_weight_raw": "reference-risk correction only",
                "training_class_weight": "separate model configuration",
                "minibatch_sampling_probability": "separate sampler configuration",
            },
            "system_output_policy": "never_reconstruct_from_aggregated_leaf_means",
            "system_task_selection": {
                "task_names": list(self._system_task_names),
                "eligibility_enforced": self._eligibility_enforced,
                "software_audit_override": bool(
                    self.allow_unfiltered_system_tasks_for_software_audit
                ),
                "constant_tasks_are_not_relabelled_as_missing": True,
            },
        }
    def __call__(self, sample: Mapping[str, object]) -> dict:
        required = {
            "component_aux_y",
            "component_aux_label_mask",
            "component_y",
            "component_label_mask",
            "armor_y",
            "armor_label_mask",
            "system_y",
            "system_label_mask",
        }
        missing = sorted(required - set(sample))
        if missing:
            raise InputValidationError(
                "sample is missing staged supervision tensors", context={"missing": missing}
            )

        aux = sample["component_aux_y"]
        aux_mask = sample["component_aux_label_mask"]
        fragment = _selected(aux, self._fragment_indices)
        fragment_mask = _selected(aux_mask, self._fragment_indices)
        hit_mean = fragment[..., 0]
        hit_available = fragment_mask[..., 0]
        nonzero_gate = hit_available & (hit_mean > 0)
        conditional_mask = fragment_mask[..., 1:] & nonzero_gate[..., None]

        design_weight = sample.get("sampling_design_weight_raw")
        design_mask = sample.get("sampling_weight_mask")
        if (design_weight is None) != (design_mask is None):
            raise InputValidationError(
                "sampling design weight and mask must be supplied together"
            )

        return {
            "fragment_nonzero_gate": {
                "target": nonzero_gate,
                "mask": hit_available,
                "source": "component_aux_y.fragment_hits",
            },
            "fragment_conditional_response": {
                "target": fragment[..., 1:],
                "mask": conditional_mask,
                "condition": "mean_fragment_hits_gt_zero",
            },
            "blast_local_response": {
                "target": _selected(aux, self._blast_indices),
                "mask": _selected(aux_mask, self._blast_indices),
            },
            "component_mechanism_state": {
                "target": sample["component_y"],
                "mask": sample["component_label_mask"],
            },
            "armor_response": {
                "target": sample["armor_y"],
                "mask": sample["armor_label_mask"],
            },
            "system_tree": {
                "target": _selected(sample["system_y"], self._system_indices),
                "mask": _selected(sample["system_label_mask"], self._system_indices),
                "task_names": self._system_task_names,
                "eligibility_enforced": self._eligibility_enforced,
                "source": "direct_archived_per_replicate_tree_then_aggregate",
                "activation": "independent_sigmoid",
            },
            "sampling_risk_weight": {
                "value": design_weight,
                "mask": design_mask,
                "must_not_be_reused_as": (
                    "training_class_weight",
                    "minibatch_sampling_probability",
                ),
            },
        }
