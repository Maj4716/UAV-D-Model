"""Fail-closed contracts for learning overlapping soft system events.

The helpers in this module operate on archived labels; they never change a
label, a mask, or a dataset split.  Task eligibility is decided from Train
only.  Validation, Reference Test, and Challenge outcomes may be summarized
for an audit, but cannot make a task trainable.
"""

from __future__ import annotations

from math import isfinite
from typing import Iterable, Mapping, Sequence

import numpy as np

from .content_contract import is_sha256_identity as _is_sha256_identity
from .errors import InputValidationError
from .estimands import (
    nonnegative_effective_sample_size,
    restore_scaled_total,
    scale_positive_weights,
)


SYSTEM_LEARNING_REQUIRED_SOURCE_HASH_FIELDS = frozenset(
    {
        "target_hash",
        "physics_registry_hash",
        "engine_code_hash",
        "damage_tree_definition_hash",
        "label_spec_hash",
        "sampling_design_hash",
        "split_registry_hash",
    }
)


def is_sha256_identity(value: object) -> bool:
    """Compatibility shim for the historical lowercase-hex acceptance set."""

    return _is_sha256_identity(
        value, acceptance="legacy_lowercase_python_hex"
    )


def system_learning_archive_binding(
    manifest: Mapping | None, manifest_hash: str | None
) -> dict:
    """Extract the immutable archive identity used by system-task adapters."""

    if not isinstance(manifest, Mapping):
        return {}
    source = manifest.get("source_contracts")
    engine = manifest.get("engine")
    source = source if isinstance(source, Mapping) else {}
    engine = engine if isinstance(engine, Mapping) else {}
    hashes = {
        name: source[name]
        for name in (
            "damage_tree_definition_hash",
            "label_spec_hash",
            "sampling_design_hash",
            "split_registry_hash",
            "training_design_hash",
            "stratification_spec_hash",
        )
        if isinstance(source.get(name), str)
    }
    target_hashes = source.get("target_hashes")
    if (
        isinstance(target_hashes, Sequence)
        and not isinstance(target_hashes, (str, bytes))
        and len(target_hashes) == 1
        and isinstance(target_hashes[0], str)
    ):
        hashes["target_hash"] = target_hashes[0]
    if isinstance(engine.get("configuration_hash"), str):
        hashes["physics_registry_hash"] = engine["configuration_hash"]
    if isinstance(engine.get("code_hash"), str):
        hashes["engine_code_hash"] = engine["code_hash"]
    split_policy = manifest.get("split_policy")
    if (
        "split_registry_hash" not in hashes
        and isinstance(split_policy, Mapping)
        and isinstance(split_policy.get("split_registry_hash"), str)
    ):
        hashes["split_registry_hash"] = split_policy["split_registry_hash"]
    return {
        "dataset_id": manifest.get("dataset_id"),
        "manifest_hash": manifest_hash,
        "source_contract_hashes": hashes,
    }


SYSTEM_LEARNING_CONTRACT_VERSION = "1.0.0"
REFERENCE_TRAIN_ROLES = frozenset({"reference_train", "train_pilot"})
ENRICHED_TRAIN_ROLES = frozenset(
    {"geometry_stratified_train", "adaptive_train", "adaptive_train_round"}
)
REFERENCE_EVALUATION_ROLES = frozenset(
    {"reference_validation", "reference_test"}
)
CHALLENGE_ROLES = frozenset({"rare_stress_test", "challenge"})
_KNOWN_ROLES = (
    REFERENCE_TRAIN_ROLES
    | ENRICHED_TRAIN_ROLES
    | REFERENCE_EVALUATION_ROLES
    | CHALLENGE_ROLES
)


def _string_vector(values: Sequence[str], count: int, name: str) -> np.ndarray:
    array = np.asarray(values).astype(str)
    if array.shape != (count,) or any(not value for value in array.tolist()):
        raise InputValidationError(f"{name} must contain one nonempty value per case")
    return array


def _validated_system_arrays(
    targets,
    masks,
    task_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    values = np.asarray(targets, dtype=np.float64)
    available = np.asarray(masks, dtype=bool)
    names = tuple(map(str, task_names))
    if values.ndim != 2 or available.shape != values.shape:
        raise InputValidationError("system targets and masks must share shape [case, task]")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise InputValidationError("system targets must be nonempty")
    if len(names) != values.shape[1] or len(set(names)) != len(names) or any(
        not name for name in names
    ):
        raise InputValidationError("system task names must be unique and match the task axis")
    observed = values[available]
    if np.any(~np.isfinite(observed)):
        raise InputValidationError("available system targets must be finite")
    if np.any((observed < 0.0) | (observed > 1.0)):
        raise InputValidationError("available system soft targets must lie in [0, 1]")
    return values, available, names


def _role_split_guard(splits: np.ndarray, roles: np.ndarray) -> None:
    unknown_splits = sorted(set(splits.tolist()) - {"train", "validation", "test"})
    if unknown_splits:
        raise InputValidationError(
            "unknown dataset split in system-learning audit",
            context={"splits": unknown_splits},
        )
    unknown_roles = sorted(set(roles.tolist()) - _KNOWN_ROLES)
    if unknown_roles:
        raise InputValidationError(
            "unknown sampling role in system-learning audit",
            context={"roles": unknown_roles},
        )
    for split, role in zip(splits.tolist(), roles.tolist()):
        if split == "train" and role not in REFERENCE_TRAIN_ROLES | ENRICHED_TRAIN_ROLES:
            raise InputValidationError(
                "non-Train sampling role is present in the Train split",
                context={"role": role},
            )
        if split == "validation" and role != "reference_validation":
            raise InputValidationError(
                "Validation must contain only the frozen Reference role",
                context={"role": role},
            )
        if split == "test" and role not in {"reference_test"} | CHALLENGE_ROLES:
            raise InputValidationError(
                "Test contains a role that is neither Reference Test nor Challenge",
                context={"role": role},
            )


def _scaled_positive_weights(
    weights: np.ndarray, *, field: str
) -> tuple[np.ndarray, float, float]:
    """Compatibility delegate to the shared estimand implementation."""

    return scale_positive_weights(weights, field=field)


def _restore_weight_total(scaled_total: float, scale: float) -> tuple[float | None, str]:
    """Compatibility delegate to the shared estimand implementation."""

    return restore_scaled_total(scaled_total, scale)


def _stable_effective_sample_size(contributions: np.ndarray) -> float:
    """Compatibility delegate for a nonnegative soft-mass estimand."""

    return nonnegative_effective_sample_size(contributions)


def _support_summary(
    values: np.ndarray,
    masks: np.ndarray,
    selected: np.ndarray,
    task_index: int,
    *,
    zero_tolerance: float,
    design_weights: np.ndarray,
    design_weight_mask: np.ndarray,
) -> dict:
    label_use = selected & masks[:, task_index]
    weight_use = selected & design_weight_mask
    joint_use = label_use & design_weight_mask
    observed = values[label_use, task_index]
    joint_observed = values[joint_use, task_index]
    weights = design_weights[joint_use]
    counts = {
        "label_available_count": int(np.count_nonzero(label_use)),
        "weight_eligible_count": int(np.count_nonzero(weight_use)),
        "joint_analysis_count": int(np.count_nonzero(joint_use)),
    }
    if observed.size == 0:
        return {
            "available_count": 0,
            **counts,
            "nonzero_count": 0,
            "zero_count": 0,
            "soft_event_mass": 0.0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "is_constant": None,
            "design_weight_sum": 0.0,
            "design_weight_sum_status": "no_joint_analysis_rows",
            "design_weight_scale": None,
            "design_weight_sum_scaled": 0.0,
            "effective_sample_size": 0.0,
            "soft_event_effective_sample_size": 0.0,
        }
    minimum = float(np.min(observed))
    maximum = float(np.max(observed))
    nonzero = int(np.count_nonzero(observed > zero_tolerance))
    if weights.size:
        _, weight_scale, scaled_weight_sum = _scaled_positive_weights(
            weights, field="joint analysis design weights"
        )
        weight_sum, weight_sum_status = _restore_weight_total(
            scaled_weight_sum, weight_scale
        )
        effective_sample_size = _stable_effective_sample_size(weights)
        soft_effective_sample_size = _stable_effective_sample_size(
            weights * joint_observed
        )
    else:
        weight_scale = None
        scaled_weight_sum = 0.0
        weight_sum = 0.0
        weight_sum_status = "no_joint_analysis_rows"
        effective_sample_size = 0.0
        soft_effective_sample_size = 0.0
    return {
        "available_count": int(observed.size),
        **counts,
        "nonzero_count": nonzero,
        "zero_count": int(observed.size - nonzero),
        "soft_event_mass": float(np.sum(observed, dtype=np.float64)),
        "minimum": minimum,
        "maximum": maximum,
        "mean": float(np.mean(observed, dtype=np.float64)),
        "is_constant": bool(maximum - minimum <= zero_tolerance),
        "design_weight_sum": weight_sum,
        "design_weight_sum_status": weight_sum_status,
        "design_weight_scale": weight_scale,
        "design_weight_sum_scaled": scaled_weight_sum,
        "effective_sample_size": effective_sample_size,
        "soft_event_effective_sample_size": soft_effective_sample_size,
    }


def build_system_learning_contract(
    targets,
    masks,
    task_names: Sequence[str],
    *,
    split_names: Sequence[str],
    sampling_roles: Sequence[str],
    dataset_id: str,
    dataset_manifest_hash: str,
    test_status: str,
    design_weights=None,
    design_weight_mask=None,
    source_contract_hashes: Mapping[str, str] | None = None,
    task_definition_blockers: Mapping[str, Sequence[Mapping]] | None = None,
    minimum_soft_event_mass: Mapping[str, float] | None = None,
    zero_tolerance: float = 1.0e-12,
    training_seeds: Iterable[int] = (1729, 3253, 4933, 6521, 8161),
) -> dict:
    """Build a JSON-ready, Train-selected system-event learning contract.

    ``test_status`` must explicitly say whether Reference Test labels remain
    sealed.  This prevents a previously inspected Test split from silently
    being presented as confirmatory evidence.
    """

    values, available, names = _validated_system_arrays(targets, masks, task_names)
    count = values.shape[0]
    splits = _string_vector(split_names, count, "split_names")
    roles = _string_vector(sampling_roles, count, "sampling_roles")
    _role_split_guard(splits, roles)
    if design_weights is None:
        weights = np.ones(count, dtype=np.float64)
    else:
        weights = np.asarray(design_weights, dtype=np.float64)
        if weights.shape != (count,):
            raise InputValidationError("design_weights must contain one value per case")
    if design_weight_mask is None:
        weight_mask = np.ones(count, dtype=bool)
    else:
        weight_mask = np.asarray(design_weight_mask, dtype=bool)
        if weight_mask.shape != (count,):
            raise InputValidationError("design_weight_mask must contain one value per case")
    if np.any(weight_mask & (~np.isfinite(weights) | (weights <= 0.0))):
        raise InputValidationError("available design weights must be finite and positive")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise InputValidationError("dataset_id must be nonempty")
    if not is_sha256_identity(dataset_manifest_hash):
        raise InputValidationError("dataset_manifest_hash must be a complete sha256 identity")
    if test_status not in {"sealed_unobserved", "observed_diagnostic_only"}:
        raise InputValidationError(
            "test_status must be sealed_unobserved or observed_diagnostic_only"
        )
    tolerance = float(zero_tolerance)
    if not isfinite(tolerance) or tolerance < 0.0:
        raise InputValidationError("zero_tolerance must be finite and nonnegative")
    thresholds = {
        "train": 200.0,
        "validation": 50.0,
        "test": 100.0,
        **dict(minimum_soft_event_mass or {}),
    }
    if set(thresholds) != {"train", "validation", "test"} or any(
        not isfinite(float(value)) or float(value) <= 0.0
        for value in thresholds.values()
    ):
        raise InputValidationError(
            "minimum_soft_event_mass requires finite positive train/validation/test values"
        )
    thresholds = {key: float(value) for key, value in thresholds.items()}
    seeds = tuple(int(value) for value in training_seeds)
    if len(seeds) < 5 or len(set(seeds)) != len(seeds) or any(value < 0 for value in seeds):
        raise InputValidationError("at least five unique nonnegative training seeds are required")
    bindings = dict(source_contract_hashes or {})
    if any(
        not isinstance(key, str)
        or not key
        or not is_sha256_identity(value)
        for key, value in bindings.items()
    ):
        raise InputValidationError("source contract bindings must be complete sha256 identities")
    blocker_map = {
        str(task_name): [dict(item) for item in items]
        for task_name, items in dict(task_definition_blockers or {}).items()
    }
    unknown_blocker_tasks = sorted(set(blocker_map) - set(names))
    if unknown_blocker_tasks:
        raise InputValidationError(
            "task-definition blocker references an unknown system task",
            context={"tasks": unknown_blocker_tasks},
        )
    if any(not items for items in blocker_map.values()):
        raise InputValidationError("task-definition blocker lists must be nonempty")

    selectors = {
        "reference_train": (splits == "train") & np.isin(roles, tuple(REFERENCE_TRAIN_ROLES)),
        "enriched_train": (splits == "train") & np.isin(roles, tuple(ENRICHED_TRAIN_ROLES)),
        "all_train": splits == "train",
        "reference_validation": (splits == "validation") & (roles == "reference_validation"),
        "reference_test": (splits == "test") & (roles == "reference_test"),
        "challenge": (splits == "test") & np.isin(roles, tuple(CHALLENGE_ROLES)),
    }
    tasks: list[dict] = []
    default_tasks: list[str] = []
    exploratory_tasks: list[str] = []
    excluded_tasks: list[str] = []
    for task_index, name in enumerate(names):
        support = {
            group: _support_summary(
                values,
                available,
                selected,
                task_index,
                zero_tolerance=tolerance,
                design_weights=weights,
                design_weight_mask=weight_mask,
            )
            for group, selected in selectors.items()
        }
        train = support["all_train"]
        definition_blockers = blocker_map.get(name, [])
        secondary_issues = []
        if train["available_count"] and train["is_constant"]:
            secondary_issues.append("constant_train")
        if train["available_count"] and train["soft_event_mass"] < thresholds["train"]:
            secondary_issues.append("below_train_soft_event_mass_gate")
        if definition_blockers:
            status = "blocked_task_definition_incomplete"
            reason = "The provisional damage-tree mapping omits an unresolved source component."
        elif train["available_count"] == 0:
            status = "excluded_no_train_labels"
            reason = "Train contains no available labels for this task."
        elif train["is_constant"]:
            if train["maximum"] <= tolerance:
                status = "excluded_constant_zero_train"
            else:
                status = "excluded_constant_nonzero_train"
            reason = "Train labels are constant; the input-to-output relation is not identifiable."
        elif train["soft_event_mass"] < thresholds["train"]:
            status = "exploratory_low_event_mass"
            reason = "Train is nonconstant but below the preregistered engineering event-mass gate."
        else:
            status = "default_train_eligible"
            reason = "Train is nonconstant and reaches the declared soft-event-mass gate."

        evaluation_gates = {
            split: (
                support["reference_validation" if split == "validation" else "reference_test"][
                    "soft_event_mass"
                ]
                >= thresholds[split]
            )
            for split in ("validation", "test")
        }
        two_part_gate_eligible = bool(
            train["available_count"]
            and train["zero_count"] > 0
            and train["nonzero_count"] > 0
        )
        two_part_positive_magnitude_eligible = bool(
            two_part_gate_eligible
            and train["soft_event_mass"] >= thresholds["train"]
        )
        if status == "default_train_eligible":
            default_tasks.append(name)
        elif status == "exploratory_low_event_mass":
            exploratory_tasks.append(name)
        else:
            excluded_tasks.append(name)
        tasks.append(
            {
                "task_name": name,
                "status": status,
                "reason": reason,
                "secondary_issues": secondary_issues,
                "support": support,
                "label_available": train["available_count"] > 0,
                "train_eligible": status == "default_train_eligible",
                "two_part_gate_eligible": two_part_gate_eligible,
                "two_part_positive_magnitude_eligible": two_part_positive_magnitude_eligible,
                "two_part_model_eligible": (
                    not definition_blockers
                    and two_part_gate_eligible
                    and two_part_positive_magnitude_eligible
                ),
                "reference_evaluation_mass_gate": evaluation_gates,
                "eligibility_selected_from": "all_train_only",
                "task_definition_complete": not definition_blockers,
                "task_definition_blockers": definition_blockers,
            }
        )

    return {
        "schema_version": SYSTEM_LEARNING_CONTRACT_VERSION,
        "dataset_binding": {
            "dataset_id": dataset_id,
            "manifest_hash": dataset_manifest_hash,
            "source_contract_hashes": bindings,
            "system_output_names": list(names),
        },
        "task_semantics": {
            "type": "overlapping_soft_event_probabilities",
            "activation": "independent_sigmoid",
            "prohibited": ["softmax", "argmax", "row_normalization", "hard_label_rewrite"],
        },
        "selection_policy": {
            "eligibility_basis": "Train outcomes only",
            "validation_and_test_cannot_enable_a_task": True,
            "minimum_soft_event_mass": thresholds,
            "threshold_status": "engineering_starting_point_to_preregister_not_a_scientific_law",
            "zero_tolerance": tolerance,
            "default_task_names": default_tasks,
            "default_task_indices": [names.index(name) for name in default_tasks],
            "system_training_eligibility_mask": [name in default_tasks for name in names],
            "exploratory_task_names": exploratory_tasks,
            "excluded_task_names": excluded_tasks,
            "blocked_task_names": [
                item["task_name"]
                for item in tasks
                if item["status"] == "blocked_task_definition_incomplete"
            ],
        },
        "test_policy": {
            "status": test_status,
            "confirmatory_use_allowed": test_status == "sealed_unobserved",
            "requires_new_sealed_reference_test": test_status == "observed_diagnostic_only",
            "challenge_reported_separately": True,
        },
        "loss_policy": {
            "primary": ["masked_soft_bce_with_logits", "masked_brier_from_logits"],
            "constant_tasks_in_aggregate_loss": "forbidden",
            "design_weight": "reference-risk correction only",
            "task_weight": "separate model configuration",
            "minibatch_probability": "separate sampler configuration",
            "two_part_model": "allowed only when Train has both exact zero and positive support",
        },
        "evaluation_protocol": {
            "baselines": ["train_prevalence", "linear_or_gam", "tree"],
            "primary_metrics": ["brier", "log_loss", "calibration_intercept", "calibration_slope", "ece"],
            "rare_event_metrics": [
                "soft_average_precision",
                "binary_pr_auc_requires_preregistered_event_definition",
                "case_level_bootstrap_requires_preregistered_resampling_contract",
            ],
            "supplementary_only": ["auroc"],
            "report_per_task": True,
            "reference_and_challenge_metrics_must_not_be_pooled": True,
            "training_seeds": list(seeds),
        },
        "tasks": tasks,
        "research_boundary": (
            "This contract checks label information and experimental separation. "
            "It does not validate physical parameters, real-world prevalence, or external validity."
        ),
    }


def assert_trainable_tasks(
    contract: Mapping,
    requested_tasks: Sequence[str],
    *,
    allow_exploratory_low_mass: bool = False,
) -> tuple[int, ...]:
    """Return bound output-axis indices, rejecting ineligible tasks."""

    tasks = list(contract.get("tasks", ()))
    if not tasks or any(not isinstance(item, Mapping) for item in tasks):
        raise InputValidationError("system-learning contract tasks must be nonempty records")
    task_names = [item.get("task_name") for item in tasks]
    if (
        any(not isinstance(name, str) or not name for name in task_names)
        or len(set(task_names)) != len(task_names)
    ):
        raise InputValidationError(
            "system-learning contract task names must be unique and nonempty"
        )
    binding = contract.get("dataset_binding")
    bound_names = binding.get("system_output_names") if isinstance(binding, Mapping) else None
    if (
        not isinstance(bound_names, list)
        or len(bound_names) != len(set(bound_names))
        or set(bound_names) != set(task_names)
    ):
        raise InputValidationError(
            "system-learning contract tasks must cover its bound output vocabulary exactly once"
        )
    lookup = {name: (index, item) for index, (name, item) in enumerate(zip(task_names, tasks))}
    requested = tuple(map(str, requested_tasks))
    if not requested or len(set(requested)) != len(requested):
        raise InputValidationError("requested system tasks must be unique and nonempty")
    missing = sorted(set(requested) - set(lookup))
    if missing:
        raise InputValidationError("requested system task is absent from the contract", context={"tasks": missing})
    allowed = {"default_train_eligible"}
    if allow_exploratory_low_mass:
        allowed.add("exploratory_low_event_mass")
    refused = {
        name: lookup[name][1].get("status")
        for name in requested
        if lookup[name][1].get("status") not in allowed
    }
    if refused:
        raise InputValidationError(
            "requested system tasks are not eligible for the selected training mode",
            context={"tasks": refused},
        )
    bound_lookup = {name: index for index, name in enumerate(bound_names)}
    return tuple(bound_lookup[name] for name in requested)


def fit_reference_prevalence_baseline(
    targets,
    masks,
    task_names: Sequence[str],
    *,
    split_names: Sequence[str],
    sampling_roles: Sequence[str],
    design_weights=None,
) -> dict:
    """Fit a constant baseline using Reference Train only.

    Enriched Train is intentionally excluded.  Its event rate reflects its
    proposal and must not silently replace the declared Reference estimand.
    """

    values, available, names = _validated_system_arrays(targets, masks, task_names)
    count = values.shape[0]
    splits = _string_vector(split_names, count, "split_names")
    roles = _string_vector(sampling_roles, count, "sampling_roles")
    _role_split_guard(splits, roles)
    selected = (splits == "train") & np.isin(roles, tuple(REFERENCE_TRAIN_ROLES))
    if not np.any(selected):
        raise InputValidationError("Reference Train is empty; prevalence baseline cannot be fit")
    weights = (
        np.ones(count, dtype=np.float64)
        if design_weights is None
        else np.asarray(design_weights, dtype=np.float64)
    )
    if weights.shape != (count,) or np.any(~np.isfinite(weights) | (weights <= 0.0)):
        raise InputValidationError("design_weights must be finite, positive and aligned to cases")
    predictions = []
    support = []
    for index, name in enumerate(names):
        use = selected & available[:, index]
        if not np.any(use):
            raise InputValidationError(
                "Reference Train has no available target for a baseline task",
                context={"task_name": name},
            )
        scaled_weights, weight_scale, denominator = _scaled_positive_weights(
            weights[use], field="Reference Train design weights"
        )
        prediction = float(
            np.sum(scaled_weights * values[use, index], dtype=np.float64)
            / denominator
        )
        design_weight_sum, design_weight_sum_status = _restore_weight_total(
            denominator, weight_scale
        )
        if not isfinite(prediction):
            raise InputValidationError(
                "Reference Train prevalence baseline is non-finite",
                context={"task_name": name},
            )
        predictions.append(prediction)
        support.append(
            {
                "task_name": name,
                "available_count": int(np.count_nonzero(use)),
                "design_weight_sum": design_weight_sum,
                "design_weight_sum_status": design_weight_sum_status,
                "design_weight_scale": weight_scale,
                "design_weight_sum_scaled": denominator,
                "prediction": prediction,
            }
        )
    return {
        "schema_version": "1.0.0",
        "baseline": "reference_train_weighted_prevalence",
        "task_names": list(names),
        "predictions": predictions,
        "support": support,
        "fit_roles": sorted(REFERENCE_TRAIN_ROLES),
        "excluded_roles": sorted(ENRICHED_TRAIN_ROLES),
    }


def _weighted_soft_calibration(
    targets: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    *,
    epsilon: float,
) -> dict:
    """Fit the standard logistic recalibration model to fractional outcomes.

    The fitted model is ``sigmoid(intercept + slope * logit(probability))``.
    Fractional targets enter the Bernoulli cross-entropy directly, which is
    the same proper-score objective used by soft BCE.  Boundary-only targets,
    constant predictions, singular information, or a non-finite MLE are
    reported as undefined instead of emitting a misleading finite estimate.
    """

    positive_mass = float(np.sum(weights * targets, dtype=np.float64))
    negative_mass = float(np.sum(weights * (1.0 - targets), dtype=np.float64))
    if positive_mass <= 0.0:
        return {
            "calibration_intercept": None,
            "calibration_slope": None,
            "calibration_status": "undefined_no_positive_soft_mass",
            "calibration_iterations": 0,
        }
    if negative_mass <= 0.0:
        return {
            "calibration_intercept": None,
            "calibration_slope": None,
            "calibration_status": "undefined_no_negative_soft_mass",
            "calibration_iterations": 0,
        }

    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    logit = np.log(clipped) - np.log1p(-clipped)
    design = np.column_stack((np.ones_like(logit), logit))
    weighted_design = design * np.sqrt(weights)[:, None]
    if np.linalg.matrix_rank(weighted_design) < 2:
        return {
            "calibration_intercept": None,
            "calibration_slope": None,
            "calibration_status": "undefined_constant_prediction",
            "calibration_iterations": 0,
        }

    def objective(coefficients: np.ndarray) -> float:
        linear = design @ coefficients
        return float(
            np.sum(
                weights * (np.logaddexp(0.0, linear) - targets * linear),
                dtype=np.float64,
            )
        )

    coefficients = np.asarray([0.0, 1.0], dtype=np.float64)
    current = objective(coefficients)
    scale = max(1.0, float(np.sum(weights, dtype=np.float64)))
    for iteration in range(1, 101):
        linear = design @ coefficients
        fitted = np.empty_like(linear)
        nonnegative = linear >= 0.0
        fitted[nonnegative] = 1.0 / (1.0 + np.exp(-linear[nonnegative]))
        exponential = np.exp(linear[~nonnegative])
        fitted[~nonnegative] = exponential / (1.0 + exponential)
        gradient = design.T @ (weights * (fitted - targets))
        if float(np.max(np.abs(gradient))) <= 1.0e-10 * scale:
            if np.all(np.isfinite(coefficients)) and np.max(np.abs(coefficients)) < 1.0e8:
                return {
                    "calibration_intercept": float(coefficients[0]),
                    "calibration_slope": float(coefficients[1]),
                    "calibration_status": "defined_soft_binomial_mle",
                    "calibration_iterations": iteration - 1,
                }
            break
        curvature = weights * fitted * (1.0 - fitted)
        information = design.T @ (curvature[:, None] * design)
        if (
            np.linalg.matrix_rank(information) < 2
            or not np.all(np.isfinite(information))
        ):
            return {
                "calibration_intercept": None,
                "calibration_slope": None,
                "calibration_status": "undefined_singular_information",
                "calibration_iterations": iteration - 1,
            }
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            return {
                "calibration_intercept": None,
                "calibration_slope": None,
                "calibration_status": "undefined_singular_information",
                "calibration_iterations": iteration - 1,
            }
        accepted = False
        step_scale = 1.0
        for _ in range(40):
            candidate = coefficients - step_scale * step
            candidate_objective = objective(candidate)
            if np.isfinite(candidate_objective) and candidate_objective <= current:
                coefficients = candidate
                current = candidate_objective
                accepted = True
                break
            step_scale *= 0.5
        if not accepted:
            return {
                "calibration_intercept": None,
                "calibration_slope": None,
                "calibration_status": "undefined_optimizer_stalled",
                "calibration_iterations": iteration - 1,
            }
    return {
        "calibration_intercept": None,
        "calibration_slope": None,
        "calibration_status": "undefined_nonfinite_or_nonconverged_mle",
        "calibration_iterations": 100,
    }


def _weighted_soft_average_precision(
    targets: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> dict:
    """Integrate weighted soft precision over weighted soft recall.

    Equal prediction scores are processed as one threshold group, so the
    result cannot depend on arbitrary row order within ties.  This fractional
    event-mass score reduces to ordinary non-interpolated average precision
    for binary targets.  It is deliberately not labelled binary PR-AUC.
    """

    positive = weights * targets
    negative = weights * (1.0 - targets)
    total_positive = float(np.sum(positive, dtype=np.float64))
    total_negative = float(np.sum(negative, dtype=np.float64))
    if total_positive <= 0.0:
        return {
            "soft_average_precision": None,
            "soft_average_precision_status": "undefined_no_positive_soft_mass",
        }
    if total_negative <= 0.0:
        return {
            "soft_average_precision": None,
            "soft_average_precision_status": "undefined_no_negative_soft_mass",
        }

    order = np.argsort(-probabilities, kind="stable")
    ordered_probability = probabilities[order]
    ordered_positive = positive[order]
    ordered_weight = weights[order]
    cumulative_positive = 0.0
    cumulative_weight = 0.0
    average_precision = 0.0
    start = 0
    while start < len(order):
        stop = start + 1
        while (
            stop < len(order)
            and ordered_probability[stop] == ordered_probability[start]
        ):
            stop += 1
        group_positive = float(
            np.sum(ordered_positive[start:stop], dtype=np.float64)
        )
        cumulative_positive += group_positive
        cumulative_weight += float(
            np.sum(ordered_weight[start:stop], dtype=np.float64)
        )
        average_precision += (
            group_positive / total_positive
        ) * (cumulative_positive / cumulative_weight)
        start = stop
    return {
        "soft_average_precision": float(average_precision),
        "soft_average_precision_status": "defined_fractional_event_mass",
    }


def evaluate_system_probabilities(
    targets,
    predictions,
    masks,
    task_names: Sequence[str],
    *,
    evaluated_tasks: Sequence[str],
    design_weights=None,
    baseline_predictions: Sequence[float] | None = None,
    calibration_bins: int = 10,
    epsilon: float = 1.0e-12,
) -> dict:
    """Return per-task proper scores without silently averaging all outputs."""

    values, available, names = _validated_system_arrays(targets, masks, task_names)
    probability = np.asarray(predictions, dtype=np.float64)
    if probability.shape != values.shape or np.any(~np.isfinite(probability)):
        raise InputValidationError("predictions must be finite and aligned with system targets")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise InputValidationError("probabilities must lie in [0, 1]")
    requested = tuple(map(str, evaluated_tasks))
    if not requested or len(requested) != len(set(requested)):
        raise InputValidationError("evaluated_tasks must be explicit, unique and nonempty")
    missing = sorted(set(requested) - set(names))
    if missing:
        raise InputValidationError("evaluated task is absent", context={"tasks": missing})
    weights = (
        np.ones(values.shape[0], dtype=np.float64)
        if design_weights is None
        else np.asarray(design_weights, dtype=np.float64)
    )
    if weights.shape != (values.shape[0],) or np.any(
        ~np.isfinite(weights) | (weights <= 0.0)
    ):
        raise InputValidationError("design_weights must be finite, positive and aligned to cases")
    bins = int(calibration_bins)
    if bins < 2:
        raise InputValidationError("calibration_bins must be at least two")
    eps = float(epsilon)
    if not isfinite(eps) or eps <= 0.0 or eps >= 0.5:
        raise InputValidationError("epsilon must lie in (0, 0.5)")
    baseline = None
    if baseline_predictions is not None:
        baseline = np.asarray(baseline_predictions, dtype=np.float64)
        if baseline.shape != (len(names),) or np.any(
            ~np.isfinite(baseline) | (baseline < 0.0) | (baseline > 1.0)
        ):
            raise InputValidationError("baseline_predictions must align with the task axis")

    rows = []
    for name in requested:
        index = names.index(name)
        use = available[:, index]
        if not np.any(use):
            raise InputValidationError(
                "evaluated task has no available targets", context={"task_name": name}
            )
        y = values[use, index]
        p = probability[use, index]
        w, weight_scale, denominator = _scaled_positive_weights(
            weights[use], field="evaluation design weights"
        )
        brier = float(np.sum(w * (p - y) ** 2, dtype=np.float64) / denominator)
        clipped = np.clip(p, eps, 1.0 - eps)
        log_loss = float(
            -np.sum(
                w * (y * np.log(clipped) + (1.0 - y) * np.log1p(-clipped)),
                dtype=np.float64,
            )
            / denominator
        )
        bin_index = np.minimum((p * bins).astype(np.int64), bins - 1)
        calibration = []
        ece = 0.0
        for bin_number in range(bins):
            member = bin_index == bin_number
            if not np.any(member):
                continue
            bin_weight = float(np.sum(w[member], dtype=np.float64))
            raw_bin_weight, raw_bin_weight_status = _restore_weight_total(
                bin_weight, weight_scale
            )
            mean_prediction = float(
                np.sum(w[member] * p[member], dtype=np.float64) / bin_weight
            )
            mean_target = float(
                np.sum(w[member] * y[member], dtype=np.float64) / bin_weight
            )
            ece += bin_weight / denominator * abs(mean_prediction - mean_target)
            calibration.append(
                {
                    "bin": bin_number,
                    "count": int(np.count_nonzero(member)),
                    "weight": raw_bin_weight,
                    "weight_status": raw_bin_weight_status,
                    "weight_scaled": bin_weight,
                    "mean_prediction": mean_prediction,
                    "mean_soft_target": mean_target,
                }
            )
        weighted_soft_event_mass_scaled = float(
            np.sum(w * y, dtype=np.float64)
        )
        weighted_soft_event_mass, weighted_soft_event_mass_status = (
            _restore_weight_total(weighted_soft_event_mass_scaled, weight_scale)
        )
        design_weight_sum, design_weight_sum_status = _restore_weight_total(
            denominator, weight_scale
        )
        if not all(isfinite(value) for value in (brier, log_loss, ece)):
            raise InputValidationError(
                "weighted system evaluation produced a non-finite metric",
                context={"task_name": name},
            )
        row = {
            "task_name": name,
            "available_count": int(y.size),
            "soft_event_mass": float(np.sum(y, dtype=np.float64)),
            "design_weight_sum": design_weight_sum,
            "design_weight_sum_status": design_weight_sum_status,
            "design_weight_scale": weight_scale,
            "design_weight_sum_scaled": denominator,
            "weighted_soft_event_mass": weighted_soft_event_mass,
            "weighted_soft_event_mass_status": weighted_soft_event_mass_status,
            "weighted_soft_event_mass_scaled": weighted_soft_event_mass_scaled,
            "brier": brier,
            "log_loss": log_loss,
            "ece": float(ece),
            "calibration_bins": calibration,
            **_weighted_soft_calibration(y, p, w, epsilon=eps),
            **_weighted_soft_average_precision(y, p, w),
            "binary_pr_auc": None,
            "binary_pr_auc_status": (
                "not_computed_requires_preregistered_binary_event_definition"
            ),
        }
        if baseline is not None:
            baseline_brier = float(
                np.sum(w * (baseline[index] - y) ** 2, dtype=np.float64)
                / denominator
            )
            row["baseline_brier"] = baseline_brier
            if baseline_brier <= 0.0:
                row["brier_skill_score"] = None
                row["brier_skill_status"] = "undefined_zero_baseline_error"
            else:
                row["brier_skill_score"] = 1.0 - brier / baseline_brier
                row["brier_skill_status"] = "defined"
        rows.append(row)
    return {
        "schema_version": "1.0.0",
        "evaluated_tasks": list(requested),
        "macro_policy": "only_explicit_evaluated_tasks",
        "macro_task_names": list(requested),
        "macro_task_count": len(requested),
        "macro_weighting": "equal_weight_per_explicit_task",
        "macro_evidence_boundary": (
            "The caller must supply only tasks admitted by a Train-fitted "
            "eligibility contract; evaluation outcomes cannot establish eligibility."
        ),
        "macro_brier": float(np.mean([row["brier"] for row in rows])),
        "macro_log_loss": float(np.mean([row["log_loss"] for row in rows])),
        "metric_semantics": {
            "calibration": (
                "Weighted soft-binomial maximum likelihood for "
                "sigmoid(intercept + slope * logit(prediction))."
            ),
            "soft_average_precision": (
                "Non-interpolated weighted fractional-event-mass average "
                "precision with equal scores grouped at one threshold."
            ),
            "binary_pr_auc": (
                "Not computed without a preregistered binary event definition."
            ),
        },
        "case_level_bootstrap": {
            "status": "not_computed_requires_preregistered_resampling_contract",
            "required_fields": [
                "resampling_unit",
                "replicate_count",
                "confidence_level",
                "random_seed",
            ],
        },
        "tasks": rows,
    }


def _torch_inputs(logits, targets, mask):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs
        raise InputValidationError("PyTorch is required for neural loss evaluation") from exc
    if not (torch.is_tensor(logits) and torch.is_tensor(targets) and torch.is_tensor(mask)):
        raise InputValidationError("logits, targets and mask must be PyTorch tensors")
    if logits.shape != targets.shape or logits.shape != mask.shape or logits.ndim < 1:
        raise InputValidationError("logits, targets and mask must share a nonempty shape")
    observed = targets[mask.to(dtype=torch.bool)]
    if observed.numel() == 0:
        raise InputValidationError("masked loss has no available targets")
    if not torch.isfinite(logits).all() or not torch.isfinite(observed).all():
        raise InputValidationError("available neural loss inputs must be finite")
    if torch.any((observed < 0.0) | (observed > 1.0)):
        raise InputValidationError("available soft targets must lie in [0, 1]")
    return torch


def _loss_weight(torch, raw, mask, design_weight, task_weight):
    weight = mask.to(dtype=raw.dtype)
    if design_weight is not None:
        if not torch.is_tensor(design_weight) or design_weight.shape != raw.shape[:-1]:
            raise InputValidationError(
                "design_weight must be a tensor over case axes, separate from the task axis"
            )
        if not torch.isfinite(design_weight).all() or torch.any(design_weight <= 0.0):
            raise InputValidationError("design_weight must be finite and positive")
        weight = weight * design_weight.to(dtype=raw.dtype).unsqueeze(-1)
    if task_weight is not None:
        if not torch.is_tensor(task_weight) or task_weight.shape != (raw.shape[-1],):
            raise InputValidationError("task_weight must be a tensor over the task axis")
        if not torch.isfinite(task_weight).all() or torch.any(task_weight <= 0.0):
            raise InputValidationError("task_weight must be finite and positive")
        weight = weight * task_weight.to(dtype=raw.dtype)
    denominator = weight.sum()
    if not bool(torch.isfinite(denominator)) or float(denominator.detach().cpu()) <= 0.0:
        raise InputValidationError("masked loss has zero effective weight")
    return weight, denominator


def masked_soft_bce_with_logits(
    logits,
    targets,
    mask,
    *,
    design_weight=None,
    task_weight=None,
):
    """Mean independent-sigmoid soft BCE over available, weighted cells."""

    torch = _torch_inputs(logits, targets, mask)
    mask_bool = mask.to(dtype=torch.bool)
    safe_targets = torch.where(
        mask_bool,
        targets.to(dtype=logits.dtype),
        torch.zeros((), dtype=logits.dtype, device=logits.device),
    )
    raw = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, safe_targets, reduction="none"
    )
    weight, denominator = _loss_weight(
        torch, raw, mask, design_weight, task_weight
    )
    return (raw * weight).sum() / denominator


def masked_brier_from_logits(
    logits,
    targets,
    mask,
    *,
    design_weight=None,
    task_weight=None,
):
    """Mean Brier score after independent sigmoid activation."""

    torch = _torch_inputs(logits, targets, mask)
    mask_bool = mask.to(dtype=torch.bool)
    safe_targets = torch.where(
        mask_bool,
        targets.to(dtype=logits.dtype),
        torch.zeros((), dtype=logits.dtype, device=logits.device),
    )
    raw = (torch.sigmoid(logits) - safe_targets) ** 2
    weight, denominator = _loss_weight(
        torch, raw, mask, design_weight, task_weight
    )
    return (raw * weight).sum() / denominator
