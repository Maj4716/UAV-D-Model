"""Deterministic p0/LHS case construction with geometry-correct transforms."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import groupby
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .errors import InputValidationError
from .label_spec import LabelSpec
from .input_identity import (
    FrozenCase,
    _canonical_subtree_cache,
    case_identity_hash,
)
from .learning_input_identity import (
    LOITERING_IMPACT_CONTACT_POSE_TRAINING_FEATURE_NAMES,
    LOITERING_IMPACT_CONTACT_POSE_TRAINING_FEATURE_UNITS,
    LOITERING_IMPACT_STATE_FEATURE_NAMES,
    LOITERING_IMPACT_STATE_FEATURE_UNITS,
    LOITERING_IMPACT_TRAINING_FEATURE_NAMES,
    UAV_STATE_FEATURE_NAMES,
    UAV_STATE_FEATURE_UNITS,
    UAV_TRAINING_FEATURE_NAMES,
    fixed_scenario_fingerprint,
    learning_input_fingerprint,
    legacy_learning_input_fingerprint,
    loitering_impact_contact_pose_fixed_scenario_fingerprint,
    loitering_impact_contact_pose_learning_input_fingerprint,
    loitering_impact_contact_pose_training_vector,
    loitering_impact_fixed_scenario_fingerprint,
    loitering_impact_learning_input_fingerprint,
    loitering_impact_state_vector,
    uav_state_vector,
)
from .sampling_spec import SamplingSpec, latent_dimensions
from .split_registry import SplitRegistry
from .utils import content_hash
from .contact_contract import audit_s_contact
from .contact_manifold_sampling import apply_contact_manifold
from .impact import is_impact_case, resolve_impact_detonation
from .munition_profiles import apply_munition_profile


_CONTACT_DATA_ROLE_BY_SAMPLING_ROLE = {
    "reference_train": "model_train_reference",
    "reference_validation": "model_validation",
    "reference_test": "sealed_reference_test",
    "geometry_stratified_train": "contact_enriched_train",
    "rare_stress_test": "contact_challenge",
}
_CONTACT_INPUT_REFUSAL_CODES = frozenset(
    {
        "case_not_mapping",
        "encounter_missing",
        "contact_trigger_invalid",
        "trajectory_origin_invalid",
        "velocity_invalid",
        "velocity_zero",
        "trajectory_origin_inside_armor",
        "orientation_invalid",
        "orientation_velocity_misaligned",
        "armor_intersection_only_behind_origin",
        "no_forward_armor_intersection",
        "forward_contact_distance_not_positive",
    }
)


def randomized_lhs(sample_count: int, dimension: int, *, seed: int) -> np.ndarray:
    """Generate a randomized Latin hypercube in the open unit cube."""

    if sample_count < 1 or dimension < 1 or int(seed) < 0:
        raise InputValidationError("LHS count/dimension must be positive and seed nonnegative")
    rng = np.random.default_rng(int(seed))
    design = np.empty((sample_count, dimension), dtype=np.float64)
    for axis in range(dimension):
        design[:, axis] = (rng.permutation(sample_count) + rng.random(sample_count)) / sample_count
    # Generator.random() is specified on [0, 1), so an exact zero is
    # theoretically possible.  Keep the public open-cube contract exact: the
    # contact-manifold transform intentionally rejects boundary coordinates.
    np.clip(
        design,
        np.nextafter(0.0, 1.0),
        np.nextafter(1.0, 0.0),
        out=design,
    )
    return design


def _pointer_tokens(pointer: str) -> list[str]:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise InputValidationError("sampling paths must be RFC 6901 JSON pointers")
    return [token.replace("~1", "/").replace("~0", "~") for token in pointer[1:].split("/")]


def json_pointer_get(document: Any, pointer: str) -> Any:
    value = document
    for token in _pointer_tokens(pointer):
        if isinstance(value, list):
            try:
                value = value[int(token)]
            except (ValueError, IndexError) as exc:
                raise InputValidationError("sampling pointer does not identify a list item", context={"path": pointer}) from exc
        elif isinstance(value, dict) and token in value:
            value = value[token]
        else:
            raise InputValidationError("sampling pointer does not exist in the base case", context={"path": pointer})
    return value


def json_pointer_set(document: Any, pointer: str, replacement: Any) -> None:
    tokens = _pointer_tokens(pointer)
    if not tokens:
        raise InputValidationError("sampling cannot replace the entire case document")
    parent = document
    for token in tokens[:-1]:
        if isinstance(parent, list):
            try:
                parent = parent[int(token)]
            except (ValueError, IndexError) as exc:
                raise InputValidationError("sampling pointer does not identify a list item", context={"path": pointer}) from exc
        elif isinstance(parent, dict) and token in parent:
            parent = parent[token]
        else:
            raise InputValidationError("sampling pointer does not exist in the base case", context={"path": pointer})
    leaf = tokens[-1]
    if isinstance(parent, list):
        try:
            parent[int(leaf)] = replacement
        except (ValueError, IndexError) as exc:
            raise InputValidationError("sampling pointer does not identify a list item", context={"path": pointer}) from exc
    elif isinstance(parent, dict) and leaf in parent:
        parent[leaf] = replacement
    else:
        raise InputValidationError("sampling pointer does not exist in the base case", context={"path": pointer})


def _unit_direction(u: float, v: float) -> list[float]:
    z = 1.0 - 2.0 * u
    radius = math.sqrt(max(0.0, 1.0 - z * z))
    azimuth = 2.0 * math.pi * v
    return [radius * math.cos(azimuth), radius * math.sin(azimuth), z]


def _haar_quaternion(u: float, v: float, w: float) -> list[float]:
    # Shoemake's transform produces Haar-uniform unit quaternions in xyzw.
    quaternion = np.asarray(
        [
            math.sqrt(1.0 - u) * math.sin(2.0 * math.pi * v),
            math.sqrt(1.0 - u) * math.cos(2.0 * math.pi * v),
            math.sqrt(u) * math.sin(2.0 * math.pi * w),
            math.sqrt(u) * math.cos(2.0 * math.pi * w),
        ],
        dtype=np.float64,
    )
    # q and -q are the same rotation.  A fixed sign prevents duplicate
    # representations and discontinuous identities.
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return quaternion.tolist()


def _transform(variable: dict[str, Any], latent: np.ndarray) -> Any:
    distribution = variable["distribution"]
    if distribution == "uniform":
        return float(variable["low"] + latent[0] * (variable["high"] - variable["low"]))
    if distribution == "log_uniform":
        low, high = math.log(variable["low"]), math.log(variable["high"])
        return float(math.exp(low + latent[0] * (high - low)))
    if distribution == "integer_uniform":
        low, high = int(variable["low"]), int(variable["high"])
        return min(high, low + int(math.floor(latent[0] * (high - low + 1))))
    if distribution == "choice":
        values = variable["values"]
        return deepcopy(values[min(len(values) - 1, int(math.floor(latent[0] * len(values))))])
    if distribution == "unit_sphere_direction":
        return _unit_direction(float(latent[0]), float(latent[1]))
    if distribution == "sphere_shell":
        direction = np.asarray(_unit_direction(float(latent[0]), float(latent[1])))
        return (np.asarray(variable["center"], dtype=np.float64) + float(variable["radius"]) * direction).tolist()
    if distribution == "sphere_volume":
        minimum = float(variable["radius_min"])
        maximum = float(variable["radius_max"])
        radius = (minimum ** 3 + float(latent[0]) * (maximum ** 3 - minimum ** 3)) ** (1.0 / 3.0)
        direction = np.asarray(_unit_direction(float(latent[1]), float(latent[2])))
        return (np.asarray(variable["center"], dtype=np.float64) + radius * direction).tolist()
    if distribution == "haar_quaternion":
        return _haar_quaternion(*map(float, latent))
    raise InputValidationError("unsupported sampling distribution", context={"distribution": distribution})


def _derived_seed(namespace_seed: int, candidate_id: str) -> int:
    digest = content_hash({"namespace_seed": int(namespace_seed), "candidate_id": candidate_id})
    return int(digest.split(":", 1)[1][:16], 16) & ((1 << 63) - 1)


def _role_aware_candidate_identity(
    spec: SamplingSpec,
    allocation: dict[str, Any],
    latent_row: np.ndarray,
    *,
    global_index: int,
) -> tuple[str, str, int, int]:
    """Derive an identity without deciding when an attempt consumes its index."""

    digest = content_hash(
        {
            "sampling_spec_hash": spec.hash,
            "global_index": global_index,
            "round": allocation["round"],
            "stream": allocation["stream"],
            "role": allocation["role"],
            "stratum_id": allocation["stratum_id"],
            "proposal_id": allocation["proposal_id"],
            "latent": latent_row.tolist(),
        }
    ).split(":", 1)[1]
    candidate_id = "candidate-" + digest[:24]
    return (
        candidate_id,
        "lineage-" + digest[:24],
        _derived_seed(spec.seeds["screening"], candidate_id),
        _derived_seed(spec.seeds["formal"], candidate_id),
    )


def _configure_formal_case(
    case: dict[str, Any],
    *,
    candidate_id: str,
    lineage_id: str,
    formal_seed: int,
    replicate_count: int,
    adaptive_ci_half_width: Any,
) -> None:
    """Apply the shared formal-run identity and analysis configuration."""

    case["case_id"] = "formal-" + candidate_id.split("-", 1)[1]
    analysis = case.setdefault("analysis", {})
    analysis["aleatory_seed"] = formal_seed
    analysis["epistemic_sample_id"] = lineage_id
    analysis["replicates"] = replicate_count
    analysis["adaptive_ci_half_width"] = adaptive_ci_half_width
    analysis.pop("adaptive_physical_ci", None)


def _check_constraints(case: dict, constraints: list[dict]) -> None:
    for constraint in constraints:
        values = [float(json_pointer_get(case, path)) for path in constraint["paths"]]
        observed = float(sum(values))
        limit = float(constraint["limit"])
        satisfied = observed <= limit if constraint["kind"] == "sum_leq" else observed >= limit
        if not satisfied:
            raise InputValidationError(
                "sample violates a declared joint constraint",
                context={"kind": constraint["kind"], "observed": observed, "limit": limit},
            )


def _constraints_satisfied(case: dict, constraints: list[dict]) -> bool:
    """Evaluate declared legal-domain constraints without hiding bad pointers."""

    for constraint in constraints:
        values = [float(json_pointer_get(case, path)) for path in constraint["paths"]]
        observed = float(sum(values))
        limit = float(constraint["limit"])
        if constraint["kind"] == "sum_leq" and observed > limit:
            return False
        if constraint["kind"] == "sum_geq" and observed < limit:
            return False
    return True


def _condition_matches(case: dict, condition: dict[str, Any]) -> bool:
    observed = json_pointer_get(case, condition["path"])
    operator = condition["operator"]
    if operator == "equals":
        return observed == condition["value"]
    if operator == "not_equals":
        return observed != condition["value"]
    if operator == "in":
        return observed in condition["values"]
    if operator == "not_in":
        return observed not in condition["values"]
    try:
        left = float(observed)
        right = float(condition["value"])
    except (TypeError, ValueError) as exc:
        raise InputValidationError(
            "ordered sampling predicates require numeric values",
            context={"path": condition["path"], "operator": operator},
        ) from exc
    if not math.isfinite(left) or not math.isfinite(right):
        raise InputValidationError(
            "ordered sampling predicates require finite numeric values",
            context={"path": condition["path"], "operator": operator},
        )
    return {
        "lt": left < right,
        "le": left <= right,
        "gt": left > right,
        "ge": left >= right,
    }[operator]


def _apply_sampling_variables(
    case: dict, base_case: dict, sampling_spec: SamplingSpec, latent: np.ndarray
) -> dict[str, bool]:
    """Apply independent/shared-quantile variables then conditional activation."""

    variables = {item["name"]: item for item in sampling_spec.variables}
    joint_by_member: dict[str, dict[str, Any]] = {}
    for block in sampling_spec.variable_blocks:
        if block["kind"] == "joint":
            for name in block["variable_names"]:
                joint_by_member[name] = block
    consumed_blocks: set[str] = set()
    offset = 0
    for variable in sampling_spec.variables:
        block = joint_by_member.get(variable["name"])
        if block is not None:
            if block["block_id"] in consumed_blocks:
                continue
            quantile = latent[offset : offset + 1]
            offset += 1
            for name in block["variable_names"]:
                member = variables[name]
                json_pointer_set(case, member["path"], _transform(member, quantile))
            consumed_blocks.add(block["block_id"])
            continue
        width = latent_dimensions(variable["distribution"])
        json_pointer_set(case, variable["path"], _transform(variable, latent[offset : offset + width]))
        offset += width
    if offset != len(latent):
        raise InputValidationError(
            "sampling latent layout does not match the declared variable blocks",
            context={"consumed": offset, "available": len(latent)},
        )
    block_states: dict[str, bool] = {}
    for block in sampling_spec.variable_blocks:
        if block["kind"] != "conditional":
            continue
        active = _condition_matches(case, block["condition"])
        block_states[block["block_id"]] = active
        if not active:
            for name in block["variable_names"]:
                path = variables[name]["path"]
                json_pointer_set(case, path, deepcopy(json_pointer_get(base_case, path)))
    return block_states


def _matches_geometry_stratum(
    case: dict,
    stratum_id: str,
    sampling_spec: SamplingSpec,
    *,
    physics_registry=None,
) -> bool:
    if stratum_id == "p0":
        return True
    definitions = {item["stratum_id"]: item for item in sampling_spec.geometry_strata}
    try:
        definition = definitions[stratum_id]
    except KeyError as exc:
        raise InputValidationError(
            "sampling allocation references an unknown geometry stratum",
            context={"stratum_id": stratum_id},
        ) from exc
    screening = None
    for clause in definition["predicate"]["all"]:
        source = clause["source"]
        if source == "case_input":
            document = case
        elif source == "prelabel_screening":
            if physics_registry is None:
                raise InputValidationError(
                    "prelabel_screening geometry strata require a read-only physics registry",
                    context={"stratum_id": stratum_id},
                )
            if screening is None:
                from .prelabel_geometry import prelabel_geometry_features

                screening = prelabel_geometry_features(case, physics_registry)
            document = screening
        else:  # SamplingSpec validates this; keep execution fail-closed.
            raise InputValidationError(
                "unsupported pre-label stratum predicate source",
                context={"stratum_id": stratum_id, "source": source},
            )
        if not _condition_matches(document, clause):
            return False
    return True


@dataclass
class ReferencePlan:
    cases: list[dict]
    candidate_rows: list[dict]
    latent_design: np.ndarray
    split_registry: SplitRegistry


def _validate_fixed_scenario_plan(cases: list[dict], sampling_spec: SamplingSpec) -> str | None:
    """Fail before labels if a model-input plan varies hidden scenario state."""

    if not sampling_spec.has_versioned_model_input:
        return None
    if sampling_spec.is_contact_pose_model_input:
        fingerprint = loitering_impact_contact_pose_fixed_scenario_fingerprint
    elif sampling_spec.is_uav_model_input:
        fingerprint = fixed_scenario_fingerprint
    else:
        fingerprint = loitering_impact_fixed_scenario_fingerprint
    identities = {fingerprint(case) for case in cases}
    if len(identities) != 1:
        subject = "UAV-only" if sampling_spec.is_uav_model_input else "loitering-impact"
        raise InputValidationError(
            f"{subject} sampling plans require exactly one frozen threat/scenario identity",
            context={
                "fixed_scenario_hashes": sorted(identities),
                "case_count": len(cases),
            },
        )
    return next(iter(identities))


def _validate_uav_training_plan(cases: list[dict], sampling_spec: SamplingSpec) -> None:
    """Exercise the final versioned model transform before labels."""

    if not sampling_spec.has_versioned_model_input:
        return
    train_cases = [
        case
        for case in cases
        if (case.get("metadata") or {}).get("frozen_split") == "train"
    ]
    if not train_cases:
        raise InputValidationError("versioned model-input plan has no frozen Train cases")

    if sampling_spec.is_contact_pose_model_input:
        from .training_view import (
            transform_loitering_impact_contact_pose as input_transform,
        )

        vector = loitering_impact_contact_pose_training_vector
        raw_names = LOITERING_IMPACT_CONTACT_POSE_TRAINING_FEATURE_NAMES
        raw_units = LOITERING_IMPACT_CONTACT_POSE_TRAINING_FEATURE_UNITS
        training_names = LOITERING_IMPACT_CONTACT_POSE_TRAINING_FEATURE_NAMES
    elif sampling_spec.is_uav_model_input:
        from .training_view import transform_uav_state as input_transform

        vector = uav_state_vector
        raw_names = UAV_STATE_FEATURE_NAMES
        raw_units = UAV_STATE_FEATURE_UNITS
        training_names = UAV_TRAINING_FEATURE_NAMES
    else:
        from .training_view import (
            transform_loitering_impact_state as input_transform,
        )

        vector = loitering_impact_state_vector
        raw_names = LOITERING_IMPACT_STATE_FEATURE_NAMES
        raw_units = LOITERING_IMPACT_STATE_FEATURE_UNITS
        training_names = LOITERING_IMPACT_TRAINING_FEATURE_NAMES
    raw = np.asarray([vector(case) for case in train_cases], dtype=np.float32)
    raw_mask = np.ones_like(raw, dtype=bool)
    values, mask, names, _units, _transforms = input_transform(
        raw,
        raw_mask,
        raw_names,
        raw_units,
    )
    if (
        values.shape != (len(train_cases), len(training_names))
        or mask.shape != values.shape
        or not mask.all()
        or not np.isfinite(values).all()
        or tuple(names) != tuple(training_names)
    ):
        dimension = len(training_names)
        raise InputValidationError(
            f"versioned model-input plan failed its final {dimension}-dimensional tensor transform"
        )

    declared = {
        name: feature["role"]
        for feature in sampling_spec.feature_roles
        for name in feature["tensor_feature_names"]
    }
    if tuple(declared) != tuple(training_names):
        raise InputValidationError(
            "versioned feature roles do not exactly cover the final training tensor"
        )
    for index, name in enumerate(names):
        observed = values[:, index]
        observed_range = float(np.ptp(observed))
        roundoff_bound = float(
            8.0 * abs(np.spacing(np.float32(np.max(np.abs(observed)))))
        )
        constant = observed_range <= roundoff_bound
        role = declared[name]
        if role == "learnable_variable" and constant:
            raise InputValidationError(
                f"declared learnable_variable feature {name!r} is constant on Train"
            )
        if role == "fixed_context" and not constant:
            raise InputValidationError(
                f"declared fixed_context feature {name!r} varies on Train"
            )


class CaseFactory:
    """Map an LHS latent design into schema-compatible formal cases."""

    def __init__(
        self,
        base_case: dict,
        sampling_spec: SamplingSpec,
        label_spec: LabelSpec,
        *,
        target_hash: str,
        damage_tree_definition_hash: str,
        physics_registry_hash: str,
        engine_code_hash: str,
        physics_registry=None,
        compiled_target=None,
        stratification_spec_id: str = "none",
        stratification_spec_hash: str | None = None,
    ):
        # Preserve FrozenCase input metadata as well as schema-compatible keys.
        self.base_case = deepcopy(base_case)
        self.sampling_spec = sampling_spec
        self.label_spec = label_spec
        self.target_hash = target_hash
        self.damage_tree_definition_hash = damage_tree_definition_hash
        self.physics_registry_hash = physics_registry_hash
        self.physics_registry = physics_registry
        self.compiled_target = compiled_target
        self.engine_code_hash = engine_code_hash
        self.stratification_spec_id = stratification_spec_id
        self.stratification_spec_hash = stratification_spec_hash or content_hash(
            {"stratification_spec_id": stratification_spec_id, "policy": "none"}
        )
        for variable in sampling_spec.variables:
            json_pointer_get(self.base_case, variable["path"])
        for block in sampling_spec.variable_blocks:
            if block["kind"] == "conditional":
                json_pointer_get(self.base_case, block["condition"]["path"])
        for stratum in sampling_spec.geometry_strata:
            for clause in stratum["predicate"]["all"]:
                if clause["source"] == "case_input":
                    json_pointer_get(self.base_case, clause["path"])
                elif self.physics_registry is None:
                    raise InputValidationError(
                        "prelabel_screening geometry strata require a read-only physics registry",
                        context={"stratum_id": stratum["stratum_id"]},
                    )

    def _finalize_case_physics(self, case: dict) -> dict:
        """Synchronize sampled profiles and derive deterministic impact fields."""

        if case.get("munition") is not None:
            apply_munition_profile(case)
        if is_impact_case(case):
            if self.compiled_target is None:
                raise InputValidationError(
                    "impact-aware sampling requires a compiled target"
                )
            case = resolve_impact_detonation(case, self.compiled_target)
        return case

    def build_reference_plan(self) -> ReferencePlan:
        if self.sampling_spec.has_contact_conditioning:
            return self._build_contact_conditioned_plan()
        if self.sampling_spec.is_extended:
            return self._build_extended_plan()
        budgets = self.sampling_spec.budgets
        p0_hash = content_hash(self.sampling_spec.data["reference_distribution"])
        if budgets.get("rare_stress_test", 0):
            raise InputValidationError(
                "rare_stress_test requires its own explicit proposal specification; p0 pilot init only accepts zero"
            )
        roles: list[tuple[str, str, str]] = []
        # Validation and test are materialized first and the complete registry
        # is frozen before any engine label is evaluated.
        roles.extend(("reference_validation", "reference", "validation") for _ in range(budgets["validation"]))
        roles.extend(("reference_test", "reference", "test") for _ in range(budgets["test"]))
        roles.extend(("train_pilot", "reference", "train") for _ in range(budgets["train_pilot"]))
        design = randomized_lhs(
            len(roles), self.sampling_spec.latent_dimension,
            seed=self.sampling_spec.seeds["design"],
        )
        registry = SplitRegistry(self.sampling_spec.data["split_policy"]["registry_id"])
        pending: list[tuple[dict, dict, str, str]] = []
        observed_learning_inputs: dict[str, tuple[str, str]] = {}
        for global_index, ((role, stream, split), latent_row) in enumerate(zip(roles, design)):
            case = deepcopy(self.base_case)
            offset = 0
            for variable in self.sampling_spec.variables:
                width = latent_dimensions(variable["distribution"])
                json_pointer_set(case, variable["path"], _transform(variable, latent_row[offset:offset + width]))
                offset += width
            case = self._finalize_case_physics(case)
            _check_constraints(case, self.sampling_spec.data["reference_distribution"]["constraints"])
            candidate_digest = content_hash(
                {
                    "sampling_spec_hash": self.sampling_spec.hash,
                    "global_index": global_index,
                    "role": role,
                    "latent": latent_row.tolist(),
                }
            ).split(":", 1)[1]
            candidate_id = "candidate-" + candidate_digest[:24]
            lineage_id = "lineage-" + candidate_digest[:24]
            screen_seed = _derived_seed(self.sampling_spec.seeds["screening"], candidate_id)
            formal_seed = _derived_seed(self.sampling_spec.seeds["formal"], candidate_id)
            _configure_formal_case(
                case,
                candidate_id=candidate_id,
                lineage_id=lineage_id,
                formal_seed=formal_seed,
                replicate_count=self.sampling_spec.formal_replicates,
                adaptive_ci_half_width=self.sampling_spec.data["formal_policy"][
                    "adaptive_ci_half_width"
                ],
            )
            learning_hash = legacy_learning_input_fingerprint(case)
            previous = observed_learning_inputs.get(learning_hash)
            if previous is not None and previous[0] != split:
                raise InputValidationError(
                    "duplicate learning input crosses frozen split boundaries",
                    context={"learning_input_hash": learning_hash, "previous_split": previous[0], "split": split, "previous_candidate_id": previous[1], "candidate_id": candidate_id},
                )
            observed_learning_inputs.setdefault(learning_hash, (split, candidate_id))
            case.setdefault("metadata", {})["split_group"] = lineage_id
            epistemic_group_id = lineage_id
            registry.register(
                lineage_id=lineage_id,
                split=split,
                role=role,
                epistemic_group_id=epistemic_group_id,
            )
            row = {
                "candidate_id": candidate_id,
                "lineage_id": lineage_id,
                "epistemic_group_id": epistemic_group_id,
                "learning_input_hash": learning_hash,
                "duplicate_of_candidate_id": previous[1] if previous is not None else None,
                "round": 0,
                "stream": stream,
                "role": role,
                "stratum": "p0",
                "design_group_id": lineage_id,
                "split_group": lineage_id,
                "split": split,
                "p0_id": self.sampling_spec.distribution_id,
                "p0_hash": p0_hash,
                "acquisition_model_hash": None,
                "score_rarity": 0.0,
                "score_uncertainty": 0.0,
                "score_coverage": 0.0,
                "score_total": 0.0,
                "inclusion_probability": 1.0,
                "selected": True,
                "weight_mask": True,
                "random_draw": None,
                "selection_seed": self.sampling_spec.seeds["selection"],
                "design_weight_raw": 1.0,
                "screening_case_hash": None,
                "formal_case_hash": None,
                "target_hash": self.target_hash,
                "tree_hash": self.damage_tree_definition_hash,
                "label_spec_hash": self.label_spec.hash,
                "registry_hash": self.physics_registry_hash,
                "engine_code_hash": self.engine_code_hash,
                "screen_seed": screen_seed,
                "screen_count": 0,
                "formal_seed": formal_seed,
                "formal_count": self.sampling_spec.formal_replicates,
                "replicate_count": None,
                "uncertainty": None,
                "refusal": None,
                "latent": latent_row.tolist(),
            }
            pending.append((case, row, split, epistemic_group_id))
        registry_hash = registry.freeze()
        # The complete registry is identical for every frozen case.  Building a
        # fresh N-assignment document inside the N-case loop makes plan
        # construction O(N^2) in transient memory and CPU.  Reuse one snapshot
        # object while preserving the existing per-case serialized contract.
        registry_document = registry.definition_document()
        canonical_subtrees = _canonical_subtree_cache(
            (registry_document, self.label_spec.data)
        )
        cases: list[dict] = []
        rows: list[dict] = []
        for case, row, split, epistemic_group_id in pending:
            sampling = {
                "schema_version": "1.0.0",
                "candidate_id": row["candidate_id"],
                "lineage_id": row["lineage_id"],
                "design_group_id": row["design_group_id"],
                "round": 0,
                "stream": row["stream"],
                "role": row["role"],
                "stratum": row["stratum"],
                "role_names": ["train_pilot", "reference_validation", "reference_test", "adaptive_train", "rare_stress_test"],
                "stratum_names": ["p0"],
                "selection_probability": 1.0,
                "design_weight_raw": 1.0,
                "weight_mask": True,
                "sampling_spec_id": self.sampling_spec.spec_id,
                "sampling_spec_hash": self.sampling_spec.hash,
                "p0_id": self.sampling_spec.distribution_id,
                "p0_hash": p0_hash,
                "label_spec_id": self.label_spec.spec_id,
                "label_spec_hash": self.label_spec.hash,
                "label_spec_document": self.label_spec.data,
                "stratification_spec_id": self.stratification_spec_id,
                "stratification_spec_hash": self.stratification_spec_hash,
                "split_registry_id": registry.registry_id,
                "split_registry_hash": registry_hash,
                "split_registry_document": registry_document,
                "frozen_split": split,
                "epistemic_group_id": epistemic_group_id,
                "screen_seed": row["screen_seed"],
                "formal_seed": row["formal_seed"],
                "damage_tree_definition_hash": self.damage_tree_definition_hash,
                "target_hash": self.target_hash,
                "physics_registry_hash": self.physics_registry_hash,
                "engine_code_hash": self.engine_code_hash,
            }
            case["metadata"]["sampling"] = sampling
            case["metadata"]["frozen_split"] = split
            case["metadata"]["split_registry_hash"] = registry_hash
            row["split_registry_hash"] = registry_hash
            loaded_dependency = (
                case.input_identity["dependencies"][0]
                if isinstance(case, FrozenCase)
                and case.input_identity.get("dependencies")
                else None
            )
            row["formal_case_hash"] = case_identity_hash(
                case,
                loaded_dependency=loaded_dependency,
                _canonical_subtrees=canonical_subtrees,
            )
            cases.append(case)
            rows.append(row)
        return ReferencePlan(cases=cases, candidate_rows=rows, latent_design=design, split_registry=registry)

    def _build_contact_conditioned_plan(self) -> ReferencePlan:
        """Build the opt-in 1.3 S-CONTACT plan with a complete attempt ledger."""

        spec = self.sampling_spec
        conditioning = spec.contact_conditioning
        if conditioning is None or not spec.is_loitering_impact_model_input:
            raise InputValidationError(
                "contact-conditioned plans require the loitering-impact model-input contract"
            )
        if self.compiled_target is None:
            raise InputValidationError(
                "S-CONTACT sampling requires a compiled target"
            )
        p0_hash = content_hash(spec.data["reference_distribution"])
        allocations = spec.role_allocations()
        registry = SplitRegistry(spec.data["split_policy"]["registry_id"])
        pending: list[tuple[dict, dict, str, str, dict]] = []
        rows: list[dict] = []
        latent_rows: list[np.ndarray] = []
        observed_learning_inputs: dict[str, tuple[str, str, str, str]] = {}
        role_names = [item["role"] for item in spec.stream_roles]
        stratum_names = [
            "p0",
            *[item["stratum_id"] for item in spec.geometry_strata],
        ]
        policy = spec.legal_sampling["invalid_candidate_policy"]
        maximum_attempts = int(
            spec.legal_sampling["max_attempts_per_allocation"]
        )
        constraints = spec.data["reference_distribution"]["constraints"]
        global_index = 0
        refusal_counts: dict[str, int] = {}

        def contact_data_role(allocation: dict[str, Any]) -> str:
            try:
                return _CONTACT_DATA_ROLE_BY_SAMPLING_ROLE[allocation["role"]]
            except KeyError as exc:  # SamplingSpec should have rejected this.
                raise InputValidationError(
                    "sampling role has no S-CONTACT information-role mapping",
                    context={"role": allocation.get("role")},
                ) from exc

        def contact_audit(case: dict, allocation: dict[str, Any]) -> dict:
            return audit_s_contact(
                case,
                self.compiled_target,
                data_role=contact_data_role(allocation),
                estimand=conditioning["estimand"],
                selection_declaration=conditioning["selection_declaration"],
            )

        def candidate_learning_hash(case: dict) -> str:
            try:
                if spec.is_contact_pose_model_input:
                    return loitering_impact_contact_pose_learning_input_fingerprint(
                        case
                    )
                return loitering_impact_learning_input_fingerprint(case)
            except (KeyError, TypeError, ValueError):
                # Invalid contact candidates still need a deterministic ledger
                # identity, even when they cannot satisfy the learner contract.
                return legacy_learning_input_fingerprint(case)

        def candidate_row(
            case: dict,
            allocation: dict[str, Any],
            latent_row: np.ndarray,
            identity: tuple[str, str, int, int],
            *,
            selected: bool,
            learning_hash: str,
            duplicate_of: str | None = None,
            refusal: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            candidate_id, lineage_id, screen_seed, formal_seed = identity
            replicate_count = spec.replicates_for(
                allocation["role"], allocation["stratum_id"]
            )
            row = {
                "candidate_id": candidate_id,
                "lineage_id": lineage_id,
                "epistemic_group_id": lineage_id,
                "learning_input_hash": learning_hash,
                "duplicate_of_candidate_id": duplicate_of,
                "round": int(allocation["round"]),
                "stream": allocation["stream"],
                "role": allocation["role"],
                "stratum": allocation["stratum_id"],
                "design_group_id": lineage_id,
                "split_group": lineage_id,
                "split": allocation["split"],
                "p0_id": spec.distribution_id,
                "p0_hash": p0_hash,
                "acquisition_model_hash": None,
                "score_rarity": 0.0,
                "score_uncertainty": 0.0,
                "score_coverage": 0.0,
                "score_total": 0.0,
                "inclusion_probability": 1.0,
                "selected": selected,
                "weight_mask": selected
                and allocation["stream"] == "reference",
                "random_draw": None,
                "selection_seed": spec.seeds["selection"],
                "design_weight_raw": 1.0,
                "screening_case_hash": None,
                "formal_case_hash": None,
                "target_hash": self.target_hash,
                "tree_hash": self.damage_tree_definition_hash,
                "label_spec_hash": self.label_spec.hash,
                "registry_hash": self.physics_registry_hash,
                "engine_code_hash": self.engine_code_hash,
                "screen_seed": screen_seed,
                "screen_count": 0,
                "formal_seed": formal_seed,
                "formal_count": replicate_count if selected else 0,
                "replicate_count": None,
                "uncertainty": None,
                "admission_purpose": "s_contact_input_admission",
                "refusal": refusal,
                "latent": latent_row.tolist(),
            }
            if not selected:
                row.update(
                    {
                        "executed": False,
                        "successful": False,
                        "admitted": False,
                    }
                )
                row["formal_case_hash"] = content_hash(
                    {
                        "candidate_id": candidate_id,
                        "formal_case_status": "not_created",
                        "refusal": refusal,
                    }
                )
            return row

        def register_row(row: dict[str, Any], latent_row: np.ndarray) -> None:
            registry.register(
                lineage_id=row["lineage_id"],
                split=row["split"],
                role=row["role"],
                epistemic_group_id=row["epistemic_group_id"],
            )
            rows.append(row)
            latent_rows.append(latent_row.copy())

        def record_refusal(
            case: dict,
            allocation: dict[str, Any],
            latent_row: np.ndarray,
            identity: tuple[str, str, int, int],
            *,
            code: str,
            message: str,
            context: dict[str, Any],
            duplicate_of: str | None = None,
        ) -> bool:
            refusal_context = {
                "sampling_allocation": {
                    "round": int(allocation["round"]),
                    "role": allocation["role"],
                    "stream": allocation["stream"],
                    "split": allocation["split"],
                    "stratum_id": allocation["stratum_id"],
                    "proposal_id": allocation["proposal_id"],
                },
                "contact_contract_id": conditioning["contract_id"],
                "contact_contract_hash": conditioning["contract_hash"],
                "selection_declaration_hash": conditioning[
                    "selection_declaration_hash"
                ],
                "formal_teacher_executed": False,
                "refusal_context": context,
            }
            refusal = {
                "code": code,
                "message": message,
                "context_json": json.dumps(
                    refusal_context,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                    separators=(",", ":"),
                ),
            }
            row = candidate_row(
                case,
                allocation,
                latent_row,
                identity,
                selected=False,
                learning_hash=candidate_learning_hash(case),
                duplicate_of=duplicate_of,
                refusal=refusal,
            )
            register_row(row, latent_row)
            refusal_counts[code] = refusal_counts.get(code, 0) + 1
            return False

        def accept_candidate(
            allocation: dict[str, Any], latent_row: np.ndarray
        ) -> bool:
            nonlocal global_index
            attempt_index = global_index
            global_index += 1
            identity = _role_aware_candidate_identity(
                spec,
                allocation,
                latent_row,
                global_index=attempt_index,
            )
            case = deepcopy(self.base_case)
            independent_width = spec.independent_latent_dimension
            _apply_sampling_variables(
                case,
                self.base_case,
                spec,
                latent_row[:independent_width],
            )
            if spec.has_contact_manifold:
                manifold = spec.contact_manifold
                if manifold is None:  # SamplingSpec validation is fail-closed.
                    raise InputValidationError(
                        "contact-manifold SamplingSpec lacks its declaration"
                    )
                apply_contact_manifold(
                    case,
                    manifold,
                    latent_row[independent_width:],
                )

            # Run the input/geometry portion before impact resolution.  A valid
            # unresolved candidate normally stops at persisted_first_contact_mismatch;
            # the stable geometry refusals occur earlier and are safe to ledger.
            try:
                contact_audit(case, allocation)
            except InputValidationError as error:
                reason_code = error.context.get("reason_code")
                if reason_code != "persisted_first_contact_mismatch":
                    if reason_code not in _CONTACT_INPUT_REFUSAL_CODES:
                        raise
                    if policy == "fail_closed":
                        raise
                    return record_refusal(
                        case,
                        allocation,
                        latent_row,
                        identity,
                        code=str(reason_code),
                        message=error.message,
                        context=dict(error.context),
                    )

            case = self._finalize_case_physics(case)
            eligibility = contact_audit(case, allocation)
            if spec.has_contact_manifold:
                manifold_record = case.get("metadata", {}).get(
                    "contact_manifold_sampling"
                )
                if isinstance(manifold_record, dict):
                    manifold_record["status"] = "admitted_s_contact"
                    manifold_record["s_contact_contract_hash"] = eligibility[
                        "contract_hash"
                    ]
            legal = _constraints_satisfied(case, constraints)
            in_stratum = _matches_geometry_stratum(
                case,
                allocation["stratum_id"],
                spec,
                physics_registry=self.physics_registry,
            )
            if not legal or not in_stratum:
                reason = (
                    "joint constraints" if not legal else "pre-label geometry stratum"
                )
                if policy == "fail_closed":
                    raise InputValidationError(
                        f"sample does not satisfy its declared {reason}",
                        context={
                            "role": allocation["role"],
                            "stratum_id": allocation["stratum_id"],
                        },
                    )
                code = (
                    "declared_joint_constraint_unsatisfied"
                    if not legal
                    else "declared_geometry_stratum_unsatisfied"
                )
                return record_refusal(
                    case,
                    allocation,
                    latent_row,
                    identity,
                    code=code,
                    message=f"sample does not satisfy its declared {reason}",
                    context={
                        "role": allocation["role"],
                        "stratum_id": allocation["stratum_id"],
                    },
                )

            learning_hash = (
                loitering_impact_contact_pose_learning_input_fingerprint(case)
                if spec.is_contact_pose_model_input
                else loitering_impact_learning_input_fingerprint(case)
            )
            previous = observed_learning_inputs.get(learning_hash)
            if previous is not None:
                previous_split, previous_id, previous_role, previous_stream = previous
                boundary_crossed = previous_split != allocation["split"] or (
                    allocation["split"] != "train"
                    and (previous_role, previous_stream)
                    != (allocation["role"], allocation["stream"])
                )
                if boundary_crossed:
                    if policy == "fail_closed":
                        raise InputValidationError(
                            "duplicate learning input crosses frozen split or "
                            "evaluation-role boundaries",
                            context={
                                "learning_input_hash": learning_hash,
                                "previous_split": previous_split,
                                "split": allocation["split"],
                                "previous_role": previous_role,
                                "role": allocation["role"],
                                "previous_candidate_id": previous_id,
                                "candidate_id": identity[0],
                            },
                        )
                    return record_refusal(
                        case,
                        allocation,
                        latent_row,
                        identity,
                        code="duplicate_learning_input_boundary",
                        message=(
                            "duplicate learning input crosses frozen split or "
                            "evaluation-role boundaries"
                        ),
                        context={
                            "learning_input_hash": learning_hash,
                            "previous_split": previous_split,
                            "split": allocation["split"],
                            "previous_role": previous_role,
                            "role": allocation["role"],
                            "previous_candidate_id": previous_id,
                        },
                        duplicate_of=previous_id,
                    )

            candidate_id, lineage_id, _screen_seed, formal_seed = identity
            replicate_count = spec.replicates_for(
                allocation["role"], allocation["stratum_id"]
            )
            _configure_formal_case(
                case,
                candidate_id=candidate_id,
                lineage_id=lineage_id,
                formal_seed=formal_seed,
                replicate_count=replicate_count,
                adaptive_ci_half_width=spec.data["formal_policy"][
                    "adaptive_ci_half_width"
                ],
            )
            case.setdefault("metadata", {})["split_group"] = lineage_id
            row = candidate_row(
                case,
                allocation,
                latent_row,
                identity,
                selected=True,
                learning_hash=learning_hash,
                duplicate_of=previous[1] if previous is not None else None,
            )
            register_row(row, latent_row)
            pending.append(
                (
                    case,
                    row,
                    allocation["split"],
                    lineage_id,
                    eligibility,
                )
            )
            observed_learning_inputs.setdefault(
                learning_hash,
                (
                    allocation["split"],
                    candidate_id,
                    allocation["role"],
                    allocation["stream"],
                ),
            )
            return True

        allocation_key = lambda item: (
            item["round"],
            item["role"],
            item["stream"],
            item["split"],
            item["stratum_id"],
            item["proposal_id"],
        )
        for key, members_iter in groupby(allocations, key=allocation_key):
            members = list(members_iter)
            required_count = len(members)
            template = members[0]
            accepted_count = 0
            attempts = 0
            batch_index = 0
            while accepted_count < required_count and attempts < maximum_attempts:
                remaining = required_count - accepted_count
                # A no-refusal 1.3 plan retains the same allocation-sized LHS
                # geometry as 1.1/1.2.  Rejections trigger another deterministic
                # LHS for only the still-unfilled budget.
                batch_size = min(maximum_attempts - attempts, remaining)
                design_seed = _derived_seed(
                    spec.seeds["design"],
                    "design:" + ":".join(map(str, (*key, batch_index))),
                )
                batch = randomized_lhs(
                    batch_size, spec.latent_dimension, seed=design_seed
                )
                for latent_row in batch:
                    attempts += 1
                    if accept_candidate(template, latent_row):
                        accepted_count += 1
                        if accepted_count == required_count:
                            break
                batch_index += 1
                if policy == "fail_closed":
                    break
            if accepted_count != required_count:
                raise InputValidationError(
                    "unable to fill a declared role/stratum budget within the "
                    "frozen attempt limit",
                    context={
                        "round": template["round"],
                        "role": template["role"],
                        "stratum_id": template["stratum_id"],
                        "accepted": accepted_count,
                        "required": required_count,
                        "attempts": attempts,
                        "candidate_rows_recorded": len(rows),
                        "refusal_code_counts": dict(sorted(refusal_counts.items())),
                    },
                )

        registry_hash = registry.freeze()
        registry_document = registry.definition_document()
        training_design = spec.training_design_document
        canonical_subtrees = _canonical_subtree_cache(
            (
                registry_document,
                spec.data,
                training_design,
                self.label_spec.data,
                role_names,
                stratum_names,
            )
        )
        for row in rows:
            row["split_registry_hash"] = registry_hash

        cases: list[dict] = []
        for case, row, split, epistemic_group_id, eligibility in pending:
            contact_binding = {
                "contract_id": eligibility["contract_id"],
                "contract_hash": eligibility["contract_hash"],
                "mode": eligibility["contact_mode"],
                "estimand": eligibility["estimand"]["id"],
                "selection_declaration_hash": eligibility["selection"][
                    "content_hash"
                ],
                "eligibility_record": eligibility,
            }
            sampling = {
                "schema_version": spec.schema_version,
                "candidate_id": row["candidate_id"],
                "lineage_id": row["lineage_id"],
                "design_group_id": row["design_group_id"],
                "round": row["round"],
                "stream": row["stream"],
                "role": row["role"],
                "stratum": row["stratum"],
                "role_names": role_names,
                "stratum_names": stratum_names,
                "selection_probability": 1.0,
                "design_weight_raw": 1.0,
                "weight_mask": row["weight_mask"],
                "sampling_spec_id": spec.spec_id,
                "sampling_spec_hash": spec.hash,
                "sampling_spec_document": spec.data,
                "training_design_document": training_design,
                "training_design_hash": spec.training_design_hash,
                "contact_conditioning": contact_binding,
                "p0_id": spec.distribution_id,
                "p0_hash": p0_hash,
                "label_spec_id": self.label_spec.spec_id,
                "label_spec_hash": self.label_spec.hash,
                "label_spec_document": self.label_spec.data,
                "stratification_spec_id": self.stratification_spec_id,
                "stratification_spec_hash": self.stratification_spec_hash,
                "split_registry_id": registry.registry_id,
                "split_registry_hash": registry_hash,
                "split_registry_document": registry_document,
                "frozen_split": split,
                "epistemic_group_id": epistemic_group_id,
                "screen_seed": row["screen_seed"],
                "formal_seed": row["formal_seed"],
                "damage_tree_definition_hash": self.damage_tree_definition_hash,
                "target_hash": self.target_hash,
                "physics_registry_hash": self.physics_registry_hash,
                "engine_code_hash": self.engine_code_hash,
            }
            case["metadata"]["sampling"] = sampling
            case["metadata"]["frozen_split"] = split
            case["metadata"]["split_registry_hash"] = registry_hash
            loaded_dependency = (
                case.input_identity["dependencies"][0]
                if isinstance(case, FrozenCase)
                and case.input_identity.get("dependencies")
                else None
            )
            row["formal_case_hash"] = case_identity_hash(
                case,
                loaded_dependency=loaded_dependency,
                _canonical_subtrees=canonical_subtrees,
            )
            cases.append(case)

        design = np.asarray(latent_rows, dtype=np.float64).reshape(
            len(latent_rows), spec.latent_dimension
        )
        if len(rows) != design.shape[0]:
            raise InputValidationError(
                "S-CONTACT candidate ledger and latent design are not aligned"
            )
        _validate_fixed_scenario_plan(cases, spec)
        _validate_uav_training_plan(cases, spec)
        return ReferencePlan(
            cases=cases,
            candidate_rows=rows,
            latent_design=design,
            split_registry=registry,
        )

    def _build_extended_plan(self) -> ReferencePlan:
        """Build a role-aware plan from the outcome-blind 1.1 design contract."""

        spec = self.sampling_spec
        p0_hash = content_hash(spec.data["reference_distribution"])
        allocations = spec.role_allocations()
        registry = SplitRegistry(spec.data["split_policy"]["registry_id"])
        pending: list[tuple[dict, dict, str, str]] = []
        accepted_latents: list[np.ndarray] = []
        observed_learning_inputs: dict[str, tuple[str, str, str, str]] = {}
        role_names = [item["role"] for item in spec.stream_roles]
        stratum_names = ["p0", *[item["stratum_id"] for item in spec.geometry_strata]]
        policy = spec.legal_sampling["invalid_candidate_policy"]
        maximum_attempts = int(spec.legal_sampling["max_attempts_per_allocation"])
        constraints = spec.data["reference_distribution"]["constraints"]
        global_index = 0

        def accept_candidate(allocation: dict[str, Any], latent_row: np.ndarray) -> bool:
            nonlocal global_index
            case = deepcopy(self.base_case)
            _apply_sampling_variables(case, self.base_case, spec, latent_row)
            case = self._finalize_case_physics(case)
            legal = _constraints_satisfied(case, constraints)
            in_stratum = _matches_geometry_stratum(
                case,
                allocation["stratum_id"],
                spec,
                physics_registry=self.physics_registry,
            )
            if not legal or not in_stratum:
                if policy == "fail_closed":
                    reason = "joint constraints" if not legal else "pre-label geometry stratum"
                    raise InputValidationError(
                        f"sample does not satisfy its declared {reason}",
                        context={
                            "role": allocation["role"],
                            "stratum_id": allocation["stratum_id"],
                        },
                    )
                return False
            candidate_id, lineage_id, screen_seed, formal_seed = (
                _role_aware_candidate_identity(
                    spec,
                    allocation,
                    latent_row,
                    global_index=global_index,
                )
            )
            replicate_count = spec.replicates_for(
                allocation["role"], allocation["stratum_id"]
            )
            _configure_formal_case(
                case,
                candidate_id=candidate_id,
                lineage_id=lineage_id,
                formal_seed=formal_seed,
                replicate_count=replicate_count,
                adaptive_ci_half_width=spec.data["formal_policy"][
                    "adaptive_ci_half_width"
                ],
            )
            if spec.is_uav_model_input:
                learning_hash = learning_input_fingerprint(case)
            elif spec.is_loitering_impact_model_input:
                learning_hash = loitering_impact_learning_input_fingerprint(case)
            else:
                learning_hash = legacy_learning_input_fingerprint(case)
            previous = observed_learning_inputs.get(learning_hash)
            if previous is not None:
                previous_split, previous_id, previous_role, previous_stream = previous
                boundary_crossed = previous_split != allocation["split"] or (
                    allocation["split"] != "train"
                    and (previous_role, previous_stream)
                    != (allocation["role"], allocation["stream"])
                )
                if boundary_crossed:
                    if policy == "reject_and_resample":
                        return False
                    raise InputValidationError(
                        "duplicate learning input crosses frozen split or evaluation-role boundaries",
                        context={
                            "learning_input_hash": learning_hash,
                            "previous_split": previous_split,
                            "split": allocation["split"],
                            "previous_role": previous_role,
                            "role": allocation["role"],
                            "previous_candidate_id": previous_id,
                            "candidate_id": candidate_id,
                        },
                    )
            case.setdefault("metadata", {})["split_group"] = lineage_id
            epistemic_group_id = lineage_id
            registry.register(
                lineage_id=lineage_id,
                split=allocation["split"],
                role=allocation["role"],
                epistemic_group_id=epistemic_group_id,
            )
            weight_mask = allocation["stream"] == "reference"
            row = {
                "candidate_id": candidate_id,
                "lineage_id": lineage_id,
                "epistemic_group_id": epistemic_group_id,
                "learning_input_hash": learning_hash,
                "duplicate_of_candidate_id": previous[1] if previous is not None else None,
                "round": int(allocation["round"]),
                "stream": allocation["stream"],
                "role": allocation["role"],
                "stratum": allocation["stratum_id"],
                "design_group_id": lineage_id,
                "split_group": lineage_id,
                "split": allocation["split"],
                "p0_id": spec.distribution_id,
                "p0_hash": p0_hash,
                "acquisition_model_hash": None,
                "score_rarity": 0.0,
                "score_uncertainty": 0.0,
                "score_coverage": 0.0,
                "score_total": 0.0,
                "inclusion_probability": 1.0,
                "selected": True,
                "weight_mask": weight_mask,
                "random_draw": None,
                "selection_seed": spec.seeds["selection"],
                "design_weight_raw": 1.0,
                "screening_case_hash": None,
                "formal_case_hash": None,
                "target_hash": self.target_hash,
                "tree_hash": self.damage_tree_definition_hash,
                "label_spec_hash": self.label_spec.hash,
                "registry_hash": self.physics_registry_hash,
                "engine_code_hash": self.engine_code_hash,
                "screen_seed": screen_seed,
                "screen_count": 0,
                "formal_seed": formal_seed,
                "formal_count": replicate_count,
                "replicate_count": None,
                "uncertainty": None,
                "refusal": None,
                "latent": latent_row.tolist(),
            }
            pending.append((case, row, allocation["split"], epistemic_group_id))
            accepted_latents.append(latent_row.copy())
            observed_learning_inputs.setdefault(
                learning_hash,
                (
                    allocation["split"],
                    candidate_id,
                    allocation["role"],
                    allocation["stream"],
                ),
            )
            global_index += 1
            return True

        allocation_key = lambda item: (
            item["round"], item["role"], item["stream"], item["split"],
            item["stratum_id"], item["proposal_id"],
        )
        for key, members_iter in groupby(allocations, key=allocation_key):
            members = list(members_iter)
            required_count = len(members)
            template = members[0]
            accepted_count = 0
            attempts = 0
            batch_index = 0
            while accepted_count < required_count and attempts < maximum_attempts:
                remaining = required_count - accepted_count
                if policy == "fail_closed":
                    batch_size = remaining
                else:
                    batch_size = min(maximum_attempts - attempts, max(8, 2 * remaining))
                design_seed = _derived_seed(
                    spec.seeds["design"],
                    "design:" + ":".join(map(str, (*key, batch_index))),
                )
                batch = randomized_lhs(
                    batch_size, spec.latent_dimension, seed=design_seed
                )
                for latent_row in batch:
                    attempts += 1
                    if accept_candidate(template, latent_row):
                        accepted_count += 1
                        if accepted_count == required_count:
                            break
                batch_index += 1
                if policy == "fail_closed":
                    break
            if accepted_count != required_count:
                raise InputValidationError(
                    "unable to fill a declared role/stratum budget within the frozen attempt limit",
                    context={
                        "round": template["round"],
                        "role": template["role"],
                        "stratum_id": template["stratum_id"],
                        "accepted": accepted_count,
                        "required": required_count,
                        "attempts": attempts,
                    },
                )

        registry_hash = registry.freeze()
        # See the legacy path above: one frozen definition is sufficient for
        # all cases and avoids repeatedly materializing the full assignment
        # ledger during large plan construction.
        registry_document = registry.definition_document()
        cases: list[dict] = []
        rows: list[dict] = []
        training_design = spec.training_design_document
        canonical_subtrees = _canonical_subtree_cache(
            (
                registry_document,
                spec.data,
                training_design,
                self.label_spec.data,
                role_names,
                stratum_names,
            )
        )
        for case, row, split, epistemic_group_id in pending:
            sampling = {
                "schema_version": "1.1.0",
                "candidate_id": row["candidate_id"],
                "lineage_id": row["lineage_id"],
                "design_group_id": row["design_group_id"],
                "round": row["round"],
                "stream": row["stream"],
                "role": row["role"],
                "stratum": row["stratum"],
                "role_names": role_names,
                "stratum_names": stratum_names,
                "selection_probability": 1.0,
                "design_weight_raw": 1.0,
                "weight_mask": row["weight_mask"],
                "sampling_spec_id": spec.spec_id,
                "sampling_spec_hash": spec.hash,
                "sampling_spec_document": spec.data,
                "training_design_document": training_design,
                "training_design_hash": spec.training_design_hash,
                "p0_id": spec.distribution_id,
                "p0_hash": p0_hash,
                "label_spec_id": self.label_spec.spec_id,
                "label_spec_hash": self.label_spec.hash,
                "label_spec_document": self.label_spec.data,
                "stratification_spec_id": self.stratification_spec_id,
                "stratification_spec_hash": self.stratification_spec_hash,
                "split_registry_id": registry.registry_id,
                "split_registry_hash": registry_hash,
                "split_registry_document": registry_document,
                "frozen_split": split,
                "epistemic_group_id": epistemic_group_id,
                "screen_seed": row["screen_seed"],
                "formal_seed": row["formal_seed"],
                "damage_tree_definition_hash": self.damage_tree_definition_hash,
                "target_hash": self.target_hash,
                "physics_registry_hash": self.physics_registry_hash,
                "engine_code_hash": self.engine_code_hash,
            }
            case["metadata"]["sampling"] = sampling
            case["metadata"]["frozen_split"] = split
            case["metadata"]["split_registry_hash"] = registry_hash
            row["split_registry_hash"] = registry_hash
            loaded_dependency = (
                case.input_identity["dependencies"][0]
                if isinstance(case, FrozenCase)
                and case.input_identity.get("dependencies")
                else None
            )
            row["formal_case_hash"] = case_identity_hash(
                case,
                loaded_dependency=loaded_dependency,
                _canonical_subtrees=canonical_subtrees,
            )
            cases.append(case)
            rows.append(row)
        design = np.asarray(accepted_latents, dtype=np.float64).reshape(
            len(accepted_latents), spec.latent_dimension
        )
        _validate_fixed_scenario_plan(cases, spec)
        _validate_uav_training_plan(cases, spec)
        return ReferencePlan(
            cases=cases,
            candidate_rows=rows,
            latent_design=design,
            split_registry=registry,
        )


def load_case_template(path: str | Path) -> dict:
    value = Path(path).resolve()
    try:
        document = json.loads(value.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputValidationError("unable to load base case template", context={"path": str(value)}) from exc
    if not isinstance(document, dict):
        raise InputValidationError("base case template must contain one JSON object")
    warhead = document.get("warhead", {})
    if not isinstance(warhead, dict):
        raise InputValidationError("base case warhead must be a JSON object")
    source = warhead.get("fragment_source", {})
    if not isinstance(source, dict):
        raise InputValidationError("base case fragment_source must be a JSON object")
    if "fragment_list_ref" in source:
        reference = source["fragment_list_ref"]
        if not isinstance(reference, str) or not reference.strip():
            raise InputValidationError("base case fragment_list_ref must be a nonempty path string")
        if not Path(reference).is_absolute():
            source["fragment_list_ref"] = str((value.parent / reference).resolve())
    return document
