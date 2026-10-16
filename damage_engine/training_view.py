"""Leak-resistant training views over immutable damage dataset archives.

This module never rewrites ``tensors.npz``.  It derives model-facing
representations from a validated loader, fits statistics on an explicitly
Train-filtered view, and binds the resulting artifact to archive/source
identities.  Software integrity here does not establish physical validity or
research-training admission.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .errors import InputValidationError
from .learning_input_identity import (
    LOITERING_IMPACT_SAMPLING_MODEL_INPUT_CONTRACT,
    LOITERING_IMPACT_STATE_FEATURE_NAMES,
    LOITERING_IMPACT_STATE_FEATURE_UNITS,
    LOITERING_IMPACT_MODEL_INPUT_DEFINITION,
    UAV_ROTATION_6D_FEATURE_NAMES,
    UAV_SAMPLING_MODEL_INPUT_CONTRACT,
    UAV_STATE_FEATURE_NAMES,
    UAV_STATE_FEATURE_UNITS,
    UAV_MODEL_INPUT_DEFINITION,
    model_input_definition_from_metadata,
)
from .normalization import (
    TRAINING_FEATURE_ROLES,
    apply_training_normalization,
    make_training_normalization_artifact,
    validate_training_normalization_artifact,
)
from .utils import content_hash, file_hash


POSITION_FEATURES = (
    "detonation_x_m",
    "detonation_y_m",
    "detonation_z_m",
)
QUATERNION_FEATURES = (
    "orientation_qx",
    "orientation_qy",
    "orientation_qz",
    "orientation_qw",
)
POSITION_VIEW_FEATURES = (
    "detonation_direction_x",
    "detonation_direction_y",
    "detonation_direction_z",
    "detonation_distance_m",
)
ROTATION_6D_FEATURES = (
    "orientation_rotation_6d_col0_x",
    "orientation_rotation_6d_col0_y",
    "orientation_rotation_6d_col0_z",
    "orientation_rotation_6d_col1_x",
    "orientation_rotation_6d_col1_y",
    "orientation_rotation_6d_col1_z",
)
UAV_QUATERNION_FEATURES = UAV_STATE_FEATURE_NAMES[6:]
UAV_ROTATION_6D_FEATURES = UAV_ROTATION_6D_FEATURE_NAMES


def _as_numpy(value: Any, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _training_schema(
    raw_names: Sequence[str], raw_units: Sequence[str]
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    names = list(map(str, raw_names))
    units = list(map(str, raw_units))
    if len(names) != len(units) or len(set(names)) != len(names):
        raise InputValidationError("model-input feature names/units are invalid")
    has_position = all(name in names for name in POSITION_FEATURES)
    if all(name in names for name in UAV_QUATERNION_FEATURES):
        quaternion_features = UAV_QUATERNION_FEATURES
        rotation_features = UAV_ROTATION_6D_FEATURES
    else:
        quaternion_features = QUATERNION_FEATURES
        rotation_features = ROTATION_6D_FEATURES
    has_quaternion = all(name in names for name in quaternion_features)
    position_indices = [names.index(name) for name in POSITION_FEATURES] if has_position else []
    quaternion_indices = [names.index(name) for name in quaternion_features] if has_quaternion else []
    position_first = min(position_indices) if position_indices else -1
    quaternion_first = min(quaternion_indices) if quaternion_indices else -1

    output_names: list[str] = []
    output_units: list[str] = []
    transforms: list[dict[str, Any]] = []
    for index, (name, unit) in enumerate(zip(names, units)):
        if index in position_indices:
            if index == position_first:
                output_names.extend(POSITION_VIEW_FEATURES)
                output_units.extend(("1", "1", "1", "m"))
                transforms.append(
                    {
                        "kind": "position_direction_and_distance",
                        "inputs": list(POSITION_FEATURES),
                        "outputs": list(POSITION_VIEW_FEATURES),
                        "zero_distance_policy": "distance_available_direction_masked",
                    }
                )
            continue
        if index in quaternion_indices:
            if index == quaternion_first:
                output_names.extend(rotation_features)
                output_units.extend(("1",) * 6)
                transforms.append(
                    {
                        "kind": "quaternion_xyzw_to_rotation_matrix_first_two_columns",
                        "inputs": list(quaternion_features),
                        "outputs": list(rotation_features),
                        "invariance": "q_and_minus_q_identical",
                    }
                )
            continue
        output_names.append(name)
        output_units.append(unit)
        transforms.append({"kind": "identity", "inputs": [name], "outputs": [name]})
    return output_names, output_units, transforms


def transform_case_physics(
    values: np.ndarray,
    mask: np.ndarray,
    feature_names: Sequence[str],
    feature_units: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[dict[str, Any]]]:
    """Create stable model coordinates while preserving missingness masks."""

    array = np.asarray(values, dtype=np.float64)
    available = np.asarray(mask, dtype=bool)
    squeeze = array.ndim == 1
    if squeeze:
        array = array[None, :]
        available = available[None, :]
    if array.ndim != 2 or available.shape != array.shape:
        raise InputValidationError("model-input values/mask must share shape [B,F]")
    names = list(map(str, feature_names))
    units = list(map(str, feature_units))
    if array.shape[1] != len(names):
        raise InputValidationError("model-input values do not match their feature names")
    output_names, output_units, transforms = _training_schema(names, units)
    columns: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    has_position = all(name in names for name in POSITION_FEATURES)
    quaternion_features = (
        UAV_QUATERNION_FEATURES
        if all(name in names for name in UAV_QUATERNION_FEATURES)
        else QUATERNION_FEATURES
    )
    has_quaternion = all(name in names for name in quaternion_features)
    position_indices = [names.index(name) for name in POSITION_FEATURES] if has_position else []
    quaternion_indices = [names.index(name) for name in quaternion_features] if has_quaternion else []
    position_first = min(position_indices) if position_indices else -1
    quaternion_first = min(quaternion_indices) if quaternion_indices else -1

    for index in range(len(names)):
        if index in position_indices:
            if index != position_first:
                continue
            selected = np.all(available[:, position_indices], axis=1)
            position = array[:, position_indices]
            distance = np.linalg.norm(position, axis=1)
            finite = np.isfinite(position).all(axis=1) & np.isfinite(distance)
            distance_mask = selected & finite
            direction_mask = distance_mask & (distance > 0.0)
            direction = np.zeros_like(position)
            direction[direction_mask] = position[direction_mask] / distance[direction_mask, None]
            for axis in range(3):
                columns.append(direction[:, axis])
                masks.append(direction_mask)
            columns.append(np.where(distance_mask, distance, 0.0))
            masks.append(distance_mask)
            continue
        if index in quaternion_indices:
            if index != quaternion_first:
                continue
            selected = np.all(available[:, quaternion_indices], axis=1)
            quaternion = array[:, quaternion_indices]
            norms = np.linalg.norm(quaternion, axis=1)
            if np.any(selected & (~np.isfinite(quaternion).all(axis=1) | (norms <= 0.0))):
                raise InputValidationError("available orientation quaternion must be finite and nonzero")
            q = np.zeros_like(quaternion)
            q[selected] = quaternion[selected] / norms[selected, None]
            x, y, z, w = (q[:, axis] for axis in range(4))
            # First two rotation-matrix columns, flattened column-major.  Every
            # term is quadratic in q, hence q and -q are exactly equivalent.
            rotation_6d = np.column_stack(
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y + z * w),
                    2.0 * (x * z - y * w),
                    2.0 * (x * y - z * w),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z + x * w),
                ]
            )
            rotation_6d[~selected] = 0.0
            for axis in range(6):
                columns.append(rotation_6d[:, axis])
                masks.append(selected.copy())
            continue
        column_mask = available[:, index]
        if np.any(column_mask & ~np.isfinite(array[:, index])):
            raise InputValidationError(f"available feature {names[index]!r} must be finite")
        columns.append(np.where(column_mask, array[:, index], 0.0))
        masks.append(column_mask.copy())

    output = np.column_stack(columns).astype(np.float32, copy=False)
    output_mask = np.column_stack(masks).astype(bool, copy=False)
    if output.shape[1] != len(output_names):
        raise AssertionError("training feature transformation schema mismatch")
    if squeeze:
        output = output[0]
        output_mask = output_mask[0]
    return output, output_mask, output_names, output_units, transforms


def transform_uav_state(
    values: np.ndarray,
    mask: np.ndarray,
    feature_names: Sequence[str],
    feature_units: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[dict[str, Any]]]:
    """Transform the versioned UAV-only input to Cartesian+rotation-6D.

    Position and velocity remain three Cartesian coordinates each. Only the
    quaternion is replaced by the continuous six-dimensional representation,
    yielding a 12-value training tensor.
    """

    if tuple(map(str, feature_names)) != UAV_MODEL_INPUT_DEFINITION.feature_names:
        raise InputValidationError("uav_state feature names do not match the versioned contract")
    if tuple(map(str, feature_units)) != UAV_MODEL_INPUT_DEFINITION.feature_units:
        raise InputValidationError("uav_state feature units do not match the versioned contract")
    return transform_case_physics(values, mask, feature_names, feature_units)


def transform_loitering_impact_state(
    values: np.ndarray,
    mask: np.ndarray,
    feature_names: Sequence[str],
    feature_units: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[dict[str, Any]]]:
    """Validate and retain the six-value loitering-impact state.

    The first three columns are already one-hot encoded and the remaining
    coordinates are scalar deployment inputs, so no representation expansion
    is needed before Train-only normalization.
    """

    if tuple(map(str, feature_names)) != (
        LOITERING_IMPACT_MODEL_INPUT_DEFINITION.feature_names
    ):
        raise InputValidationError(
            "loitering-impact feature names do not match the versioned contract"
        )
    if tuple(map(str, feature_units)) != (
        LOITERING_IMPACT_MODEL_INPUT_DEFINITION.feature_units
    ):
        raise InputValidationError(
            "loitering-impact feature units do not match the versioned contract"
        )
    return transform_case_physics(values, mask, feature_names, feature_units)


_MODEL_INPUT_TRAINING_TRANSFORMS = {
    UAV_MODEL_INPUT_DEFINITION.version: transform_uav_state,
    LOITERING_IMPACT_MODEL_INPUT_DEFINITION.version: (
        transform_loitering_impact_state
    ),
}


def _model_input_binding(
    metadata: Mapping[str, Sequence[str]],
) -> tuple[list[str], list[str], str, str, Any]:
    definition = model_input_definition_from_metadata(metadata)
    if definition is not None:
        names = list(
            metadata.get(f"{definition.metadata_prefix}_feature_names", ())
        )
        units = list(
            metadata.get(f"{definition.metadata_prefix}_feature_units", ())
        )
        if (
            names != list(definition.feature_names)
            or units != list(definition.feature_units)
        ):
            label = (
                "loitering-impact"
                if definition is LOITERING_IMPACT_MODEL_INPUT_DEFINITION
                else "UAV"
            )
            raise InputValidationError(
                f"dataset {label} model-input metadata is inconsistent"
            )
        transform = _MODEL_INPUT_TRAINING_TRANSFORMS[definition.version]
        return (
            names,
            units,
            definition.tensor_field,
            definition.mask_field,
            transform,
        )
    # Explicit backward-compatible read path for tensor contracts <=1.7.
    names = list(metadata.get("case_physics_feature_names", ()))
    units = list(metadata.get("case_physics_feature_units", ()))
    if not names:
        raise InputValidationError("dataset lacks a supported model-input tensor")
    return (
        names,
        units,
        "case_physics_x",
        "case_physics_feature_mask",
        transform_case_physics,
    )


def _source_contracts(dataset: Any, names: Sequence[str], units: Sequence[str]) -> dict[str, Any]:
    manifest = getattr(dataset, "manifest", None)
    if not isinstance(manifest, dict):
        raise InputValidationError("training view requires a validated dataset manifest")
    source = manifest.get("source_contracts")
    if not isinstance(source, dict):
        raise InputValidationError("dataset manifest is missing source_contracts")
    directory = Path(getattr(dataset, "directory", ""))
    manifest_path = directory / "manifest.json"
    manifest_hash = getattr(dataset, "_manifest_hash", None)
    if not isinstance(manifest_hash, str) and manifest_path.is_file():
        manifest_hash = file_hash(manifest_path)
    required_scalar = (
        "split_registry_hash",
        "damage_tree_definition_hash",
        "sampling_design_hash",
    )
    required_list = ("target_hashes", "parameter_set_hashes")
    result = {
        "dataset_id": manifest.get("dataset_id"),
        "manifest_hash": manifest_hash,
        "tensor_contract_version": manifest.get("tensor_contract", {}).get("version"),
    }
    model_input_contract = manifest.get("model_input_contract")
    if model_input_contract is not None:
        if not isinstance(model_input_contract, dict):
            raise InputValidationError("dataset model_input_contract must be an object")
        result["model_input_contract_hash"] = content_hash(model_input_contract)
        result["model_input_tensor_field"] = model_input_contract.get("tensor_field")
    for name in required_scalar:
        result[name] = source.get(name)
    for name in required_list:
        value = source.get(name)
        result[name] = list(value) if isinstance(value, list) else value
    for name in ("label_spec_hash", "stratification_spec_hash"):
        if name in source:
            result[name] = source[name]
    for name in (
        "fixed_scenario_hash",
        "fixed_scenario_identity_version",
        "learning_input_identity_version",
    ):
        if name in source:
            result[name] = source[name]
    training_design_document = source.get("training_design_document")
    training_design_hash = source.get("training_design_hash")
    if training_design_document is not None or training_design_hash is not None:
        if not isinstance(training_design_document, dict):
            raise InputValidationError(
                "dataset manifest training_design_document must be an object"
            )
        if content_hash(training_design_document) != training_design_hash:
            raise InputValidationError(
                "dataset manifest training_design_document does not match training_design_hash"
            )
        result["training_design_hash"] = training_design_hash
    # This hash is replaced by make_training_normalization_artifact after the
    # transformed and optional derived schema is known.
    result["feature_schema_hash"] = content_hash(
        {"feature_names": list(names), "feature_units": list(units)}
    )
    return result


def _declared_feature_roles(
    dataset: Any,
    feature_names: Sequence[str],
    *,
    proxy_boundary: str,
) -> tuple[dict[str, str] | None, str | None]:
    """Resolve the SamplingSpec 1.1 tensor-role contract, if one is present.

    Legacy manifests have no embedded training-design document and retain the
    explicitly marked Train-statistics compatibility path.  Once either design
    field is present, the binding fails closed: the document hash, boundary and
    final model-facing feature coverage must all agree.
    """

    manifest = getattr(dataset, "manifest", None)
    source = manifest.get("source_contracts") if isinstance(manifest, dict) else None
    if not isinstance(source, dict):
        raise InputValidationError("dataset manifest is missing source_contracts")
    document = source.get("training_design_document")
    design_hash = source.get("training_design_hash")
    if document is None and design_hash is None:
        return None, None
    if not isinstance(document, dict) or content_hash(document) != design_hash:
        raise InputValidationError(
            "training design document does not match its declared identity"
        )
    contract_mode = document.get("contract_mode")
    if contract_mode not in {
        "distribution_design_1.1",
        "uav_control_state_only_1.2",
        "loitering_impact_state_only_1.2",
    }:
        raise InputValidationError(
            "embedded training design is not a supported SamplingSpec distribution design"
        )
    if contract_mode == "uav_control_state_only_1.2" and (
        document.get("model_input_contract") != UAV_SAMPLING_MODEL_INPUT_CONTRACT
    ):
        raise InputValidationError(
            "embedded UAV training design has an invalid model-input contract"
        )
    if contract_mode == "loitering_impact_state_only_1.2" and (
        document.get("model_input_contract")
        != LOITERING_IMPACT_SAMPLING_MODEL_INPUT_CONTRACT
    ):
        raise InputValidationError(
            "embedded loitering-impact training design has an invalid model-input contract"
        )
    feature_roles = document.get("feature_roles")
    if not isinstance(feature_roles, list) or not feature_roles:
        raise InputValidationError("training design feature_roles must be nonempty")

    declared: dict[str, str] = {}
    allowed = {
        "learnable_variable",
        "fixed_context",
        "structural_missing",
        "derived_input",
    }
    for item in feature_roles:
        if not isinstance(item, dict):
            raise InputValidationError("training design feature role must be an object")
        role = item.get("role")
        tensor_names = item.get("tensor_feature_names")
        if role not in allowed or not isinstance(tensor_names, list) or not tensor_names:
            raise InputValidationError(
                "training design feature role lacks a valid tensor_feature_names mapping"
            )
        for name in tensor_names:
            if not isinstance(name, str) or not name or name in declared:
                raise InputValidationError(
                    "training design tensor feature names must be globally unique"
                )
            declared[name] = role

    expected_names = set(map(str, feature_names))
    if set(declared) != expected_names:
        raise InputValidationError(
            "training design feature roles must exactly cover the final training tensor schema",
            context={
                "missing": sorted(expected_names - set(declared)),
                "unexpected": sorted(set(declared) - expected_names),
            },
        )
    proxy_scope = document.get("proxy_scope")
    expected_geometry_boundary = (
        "end_to_end" if proxy_boundary == "end_to_end" else "solver_proxy"
    )
    if (
        not isinstance(proxy_scope, dict)
        or proxy_scope.get("geometry_boundary") != expected_geometry_boundary
    ):
        raise InputValidationError(
            "training design proxy boundary does not match the requested training view"
        )
    return declared, str(design_hash)


def _path_schema(metadata: Mapping[str, Sequence[str]]) -> tuple[list[str], list[str]]:
    components = list(metadata.get("component_ids", ()))
    armor = list(metadata.get("armor_ids", ()))
    features = list(metadata.get("armor_path_feature_names", ()))
    units = list(metadata.get("armor_path_feature_units", ()))
    if len(features) != len(units):
        raise InputValidationError("armor path names/units are inconsistent")
    names: list[str] = []
    output_units: list[str] = []
    for component in components:
        for panel in armor:
            for feature, unit in zip(features, units):
                names.append(f"component_armor_path::{component}::{panel}::{feature}")
                output_units.append(str(unit))
    return names, output_units


def _flatten_path_sample(sample: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if "component_armor_path_x" not in sample or "component_armor_path_mask" not in sample:
        raise InputValidationError("solver_surrogate training view requires component armor path fields")
    values = _as_numpy(sample["component_armor_path_x"], dtype=np.float32).reshape(-1)
    mask = _as_numpy(sample["component_armor_path_mask"], dtype=bool).reshape(-1)
    if values.shape != mask.shape:
        raise InputValidationError("component armor path values/mask disagree")
    return values, mask


def fit_training_view_artifact(
    dataset: Any,
    *,
    proxy_boundary: str,
    derived_input_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit a normalization artifact from an explicitly Train-only loader."""

    if getattr(dataset, "split", None) != "train":
        raise InputValidationError(
            "normalization fitting requires a dataset constructed with split='train'"
        )
    if proxy_boundary not in {"end_to_end", "solver_surrogate"}:
        raise InputValidationError("proxy_boundary must be end_to_end or solver_surrogate")
    metadata = getattr(dataset, "metadata", {})
    (
        raw_names,
        raw_units,
        input_tensor_field,
        input_mask_field,
        input_transform,
    ) = _model_input_binding(metadata)
    if input_tensor_field in {
        "uav_state_x",
        "loitering_impact_x",
    } and proxy_boundary != "end_to_end":
        raise InputValidationError(
            "the versioned state-only model-input contract supports the end_to_end proxy boundary only"
        )
    case_ids: list[str] = []
    rows: list[np.ndarray] = []
    row_masks: list[np.ndarray] = []
    path_rows: list[np.ndarray] = []
    path_masks: list[np.ndarray] = []
    transformed_names: list[str] | None = None
    transformed_units: list[str] | None = None
    transforms: list[dict[str, Any]] | None = None
    for index in range(len(dataset)):
        sample = dataset[index]
        if sample.get("split") != "train":
            raise InputValidationError("non-Train case encountered while fitting normalization")
        case_id = sample.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise InputValidationError("normalization fit case is missing case_id")
        transformed = input_transform(
            _as_numpy(sample[input_tensor_field]),
            _as_numpy(sample[input_mask_field]),
            raw_names,
            raw_units,
        )
        values, mask, names, units, current_transforms = transformed
        if transformed_names is None:
            transformed_names, transformed_units, transforms = names, units, current_transforms
        elif names != transformed_names or units != transformed_units:
            raise InputValidationError("training feature schema changed within the Train split")
        case_ids.append(case_id)
        rows.append(values)
        row_masks.append(mask)
        if proxy_boundary == "solver_surrogate":
            path_values, path_mask = _flatten_path_sample(sample)
            path_rows.append(path_values)
            path_masks.append(path_mask)
    if not rows:
        raise InputValidationError("cannot fit normalization on an empty Train split")
    if len(set(case_ids)) != len(case_ids):
        raise InputValidationError("normalization fit case identities are not unique")
    assert transformed_names is not None and transformed_units is not None and transforms is not None

    derived_names: list[str] = []
    if proxy_boundary == "solver_surrogate":
        if not isinstance(derived_input_contract, Mapping):
            raise InputValidationError(
                "solver_surrogate fitting requires a derived_input_contract"
            )
        if derived_input_contract.get("deployment_available") is not True:
            raise InputValidationError(
                "derived_input_contract must declare deployment_available=true"
            )
        path_names, path_units = _path_schema(metadata)
        if path_rows and len(path_names) != len(path_rows[0]):
            raise InputValidationError("armor path topology does not match path metadata")
        transformed_names.extend(path_names)
        transformed_units.extend(path_units)
        rows = [np.concatenate([row, path]) for row, path in zip(rows, path_rows)]
        row_masks = [np.concatenate([row, path]) for row, path in zip(row_masks, path_masks)]
        derived_names = path_names
        proxy = {
            "mode": "solver_surrogate",
            "deployment_available": True,
            "derived_input_contract_hash": content_hash(dict(derived_input_contract)),
            "derived_input_contract": deepcopy(dict(derived_input_contract)),
        }
        transforms.append(
            {
                "kind": "externally_computed_geometry_path",
                "inputs": ["component_armor_path_x", "component_armor_path_mask"],
                "outputs": path_names,
                "deployment_contract_hash": proxy["derived_input_contract_hash"],
            }
        )
    else:
        if derived_input_contract is not None:
            raise InputValidationError(
                "end_to_end boundary may not attach a derived_input_contract"
            )
        proxy = {
            "mode": "end_to_end",
            "deployment_available": True,
            "derived_input_contract_hash": None,
        }
    declared_roles, training_design_hash = _declared_feature_roles(
        dataset,
        transformed_names,
        proxy_boundary=proxy_boundary,
    )
    source = _source_contracts(dataset, transformed_names, transformed_units)
    return make_training_normalization_artifact(
        fit_case_ids=case_ids,
        source_contracts=source,
        feature_names=transformed_names,
        feature_units=transformed_units,
        feature_values=np.stack(rows),
        feature_mask=np.stack(row_masks),
        derived_feature_names=derived_names,
        declared_feature_roles=declared_roles,
        training_design_hash=training_design_hash,
        proxy_boundary=proxy,
        transforms=transforms,
    )


def _validate_artifact_source(dataset: Any, artifact: Mapping[str, Any]) -> None:
    document = validate_training_normalization_artifact(artifact)
    feature_names = [item["name"] for item in document["features"]]
    feature_units = [item["unit"] for item in document["features"]]
    current = _source_contracts(dataset, feature_names, feature_units)
    expected = document["source_contracts"]
    for field in (
        "dataset_id",
        "manifest_hash",
        "tensor_contract_version",
        "split_registry_hash",
        "damage_tree_definition_hash",
        "sampling_design_hash",
        "target_hashes",
        "parameter_set_hashes",
        "label_spec_hash",
        "stratification_spec_hash",
        "training_design_hash",
        "model_input_contract_hash",
        "model_input_tensor_field",
        "fixed_scenario_hash",
        "fixed_scenario_identity_version",
        "learning_input_identity_version",
    ):
        if field in expected and current.get(field) != expected.get(field):
            raise InputValidationError(
                f"normalization artifact source mismatch: {field}"
            )
    if current["feature_schema_hash"] != expected["feature_schema_hash"]:
        raise InputValidationError("normalization artifact feature schema mismatch")


class TrainingViewDataset:
    """Model-facing wrapper with explicit representation and weight channels.

    ``sampling_design_weight_raw`` remains the archive's reference-risk weight.
    Optional task-loss weights and minibatch probabilities are caller-supplied
    independent configurations and are never inferred from that design weight.
    """

    def __init__(
        self,
        dataset: Any,
        artifact: Mapping[str, Any],
        *,
        proxy_boundary: str,
        task_loss_weights: Mapping[str, float] | None = None,
        minibatch_probabilities: Sequence[float] | Mapping[str, float] | None = None,
    ):
        self.dataset = dataset
        self.artifact = validate_training_normalization_artifact(artifact)
        _validate_artifact_source(dataset, self.artifact)
        expected_boundary = self.artifact["proxy_boundary"]["mode"]
        if proxy_boundary != expected_boundary:
            raise InputValidationError(
                "loader proxy boundary does not match the normalization artifact"
            )
        self.proxy_boundary = proxy_boundary
        metadata = getattr(dataset, "metadata", {})
        (
            raw_names,
            raw_units,
            self.input_tensor_field,
            self.input_mask_field,
            self.input_transform,
        ) = _model_input_binding(metadata)
        if self.input_tensor_field in {
            "uav_state_x",
            "loitering_impact_x",
        } and proxy_boundary != "end_to_end":
            raise InputValidationError(
                "the versioned state-only model-input contract supports the end_to_end proxy boundary only"
            )
        self.raw_feature_names = tuple(raw_names)
        self.raw_feature_units = tuple(raw_units)
        transformed_names, transformed_units, _ = _training_schema(
            self.raw_feature_names, self.raw_feature_units
        )
        if proxy_boundary == "solver_surrogate":
            path_names, path_units = _path_schema(metadata)
            transformed_names.extend(path_names)
            transformed_units.extend(path_units)
        artifact_names = [item["name"] for item in self.artifact["features"]]
        artifact_units = [item["unit"] for item in self.artifact["features"]]
        if transformed_names != artifact_names or transformed_units != artifact_units:
            raise InputValidationError("loader training schema does not match normalization artifact")
        self.feature_names = tuple(artifact_names)
        self.feature_units = tuple(artifact_units)
        self.role_indices = {
            role: np.asarray(
                [self.feature_names.index(name) for name in self.artifact["feature_roles"][role]],
                dtype=np.int64,
            )
            for role in TRAINING_FEATURE_ROLES
        }
        task_names = tuple(metadata.get("label_task_names", ()))
        self.task_names = task_names
        if task_loss_weights is None:
            self.task_loss_weights = None
            self.task_weight_config_id = None
        else:
            if set(task_loss_weights) != set(task_names):
                raise InputValidationError("task_loss_weights must match label_task_names exactly")
            weights = np.asarray([task_loss_weights[name] for name in task_names], dtype=np.float64)
            if not np.isfinite(weights).all() or np.any(weights <= 0.0):
                raise InputValidationError("task_loss_weights must be finite and positive")
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                stored_weights = weights.astype(np.float32)
            invalid = ~np.isfinite(stored_weights) | (stored_weights <= 0.0)
            if np.any(invalid):
                raise InputValidationError(
                    "task_loss_weights are not finite and positive after float32 conversion",
                    context={
                        "training_dtype": "float32",
                        "task_names": [
                            task_names[index] for index in np.flatnonzero(invalid)
                        ],
                    },
                )
            self.task_loss_weights = stored_weights
            self.task_weight_config_id = content_hash(
                {
                    "kind": "task_loss_weights",
                    "dtype": "float32",
                    "names": list(task_names),
                    "values": stored_weights.tolist(),
                }
            )

        self.minibatch_probabilities: np.ndarray | None
        if minibatch_probabilities is None:
            self.minibatch_probabilities = None
            self.minibatch_config_id = None
        else:
            if isinstance(minibatch_probabilities, Mapping):
                ids = [dataset[index]["case_id"] for index in range(len(dataset))]
                if set(minibatch_probabilities) != set(ids):
                    raise InputValidationError("minibatch probability mapping must match dataset case IDs")
                probabilities = np.asarray(
                    [minibatch_probabilities[case_id] for case_id in ids], dtype=np.float64
                )
            else:
                probabilities = np.asarray(minibatch_probabilities, dtype=np.float64)
            if probabilities.shape != (len(dataset),) or not np.isfinite(probabilities).all():
                raise InputValidationError("minibatch probabilities must be finite on the dataset axis")
            if np.any(probabilities <= 0.0) or not np.isclose(np.sum(probabilities), 1.0):
                raise InputValidationError("minibatch probabilities must be positive and sum to one")
            self.minibatch_probabilities = probabilities
            self.minibatch_config_id = content_hash(
                {"kind": "minibatch_sampling_probability", "values": probabilities.tolist()}
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        source = self.dataset[index]
        values, mask, _, _, _ = self.input_transform(
            _as_numpy(source[self.input_tensor_field]),
            _as_numpy(source[self.input_mask_field]),
            self.raw_feature_names,
            self.raw_feature_units,
        )
        if self.proxy_boundary == "solver_surrogate":
            path_values, path_mask = _flatten_path_sample(source)
            values = np.concatenate([values, path_values])
            mask = np.concatenate([mask, path_mask])
        normalized = apply_training_normalization(values, mask, self.artifact)
        result = dict(source)
        result.update(
            {
                "training_x": torch.from_numpy(np.asarray(normalized, dtype=np.float32)),
                "training_feature_mask": torch.from_numpy(np.asarray(mask, dtype=bool)),
                "training_normalization_id": self.artifact["normalization_id"],
                "training_proxy_boundary": self.proxy_boundary,
            }
        )
        for role in TRAINING_FEATURE_ROLES:
            selected = self.role_indices[role]
            result[f"training_{role}_x"] = torch.from_numpy(
                np.asarray(normalized[selected], dtype=np.float32)
            )
            result[f"training_{role}_mask"] = torch.from_numpy(
                np.asarray(mask[selected], dtype=bool)
            )

        if "sampling_design_weight_raw" in source:
            result["design_weight_raw"] = source["sampling_design_weight_raw"]
            result["design_weight_mask"] = source.get("sampling_weight_mask")
        if self.task_loss_weights is not None:
            result["task_loss_weights"] = torch.from_numpy(self.task_loss_weights.copy())
            result["task_weight_config_id"] = self.task_weight_config_id
        if self.minibatch_probabilities is not None:
            result["minibatch_sampling_probability"] = torch.tensor(
                self.minibatch_probabilities[index], dtype=torch.float64
            )
            result["minibatch_config_id"] = self.minibatch_config_id
        return result

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            **getattr(self.dataset, "metadata", {}),
            "training_feature_names": self.feature_names,
            "training_feature_units": self.feature_units,
            "training_feature_roles": {
                role: tuple(self.artifact["feature_roles"][role])
                for role in TRAINING_FEATURE_ROLES
            },
            "training_normalization_id": self.artifact["normalization_id"],
            "training_proxy_boundary": self.proxy_boundary,
            "model_input_tensor_field": self.input_tensor_field,
            "design_weight_source_field": "sampling_design_weight_raw",
            "task_weight_config_id": self.task_weight_config_id,
            "minibatch_config_id": self.minibatch_config_id,
        }
