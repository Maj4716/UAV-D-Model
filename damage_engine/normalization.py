from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import warnings
from typing import Any, Mapping, Sequence

import numpy as np

from .content_contract import require_sha256_identity
from .errors import InputValidationError
from .utils import content_hash


_CM_TO_M = 0.01


def _scale_vector(
    value: Any, *, field: str, scale: float = _CM_TO_M
) -> list[float]:
    """Validate a JSON three-vector and apply a deterministic unit scale."""

    if not isinstance(value, list) or len(value) != 3:
        raise InputValidationError(
            f"{field} must contain exactly three numeric values"
        )
    try:
        return [float(item) * scale for item in value]
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"{field} must contain numeric values") from exc


def _coerce_quaternion(value: Any, *, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise InputValidationError(
            f"{field} must contain exactly four numeric values"
        )
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"{field} must contain numeric values") from exc


def _normalize_dimensionless_geometry(geometry: dict[str, Any]) -> None:
    if "orientation_xyzw" in geometry:
        geometry["orientation_xyzw"] = _coerce_quaternion(
            geometry["orientation_xyzw"], field="orientation_xyzw"
        )
    if "extrusion_direction" in geometry:
        geometry["extrusion_direction"] = _scale_vector(
            geometry["extrusion_direction"],
            field="extrusion_direction",
            scale=1.0,
        )


def _normalize_geometry(geometry: dict[str, Any], *, frame_unit: str) -> None:
    shape = geometry.get("shape")
    if frame_unit == "m":
        expected = {
            "box": ("center_m", "size_lwh_m"),
            "cylinder": ("center_m", "radius_m", "height_m"),
            "extruded_polygon": ("vertices_m", "extrusion_distance_m"),
        }.get(shape)
        forbidden = {
            "center_cm",
            "size_lwh_cm",
            "radius_cm",
            "height_cm",
            "vertices_cm",
            "extrusion_distance_cm",
        }
        if expected is None:
            return
        if any(name in geometry for name in forbidden):
            raise InputValidationError(
                "centimetre geometry fields require coordinate_frame.length_unit='cm'",
                context={"shape": shape},
            )
        missing = [name for name in expected if name not in geometry]
        if missing:
            raise InputValidationError(
                "geometry is missing SI fields",
                context={"shape": shape, "missing": missing},
            )
        if shape == "box":
            geometry["center_m"] = _scale_vector(
                geometry["center_m"], field="center_m", scale=1.0
            )
            geometry["size_lwh_m"] = _scale_vector(
                geometry["size_lwh_m"], field="size_lwh_m", scale=1.0
            )
        elif shape == "cylinder":
            geometry["center_m"] = _scale_vector(
                geometry["center_m"], field="center_m", scale=1.0
            )
            try:
                geometry["radius_m"] = float(geometry["radius_m"])
                geometry["height_m"] = float(geometry["height_m"])
            except (TypeError, ValueError) as exc:
                raise InputValidationError(
                    "radius_m and height_m must be numeric"
                ) from exc
        elif shape == "extruded_polygon":
            vertices = geometry["vertices_m"]
            if not isinstance(vertices, list):
                raise InputValidationError("vertices_m must be an array")
            geometry["vertices_m"] = [
                _scale_vector(vertex, field="vertices_m", scale=1.0)
                for vertex in vertices
            ]
            try:
                geometry["extrusion_distance_m"] = float(
                    geometry["extrusion_distance_m"]
                )
            except (TypeError, ValueError) as exc:
                raise InputValidationError(
                    "extrusion_distance_m must be numeric"
                ) from exc
        _normalize_dimensionless_geometry(geometry)
        return

    if frame_unit != "cm":
        raise InputValidationError(
            "coordinate_frame.length_unit must be 'm' or 'cm'",
            context={"length_unit": frame_unit},
        )

    metre_fields = {
        "center_m",
        "size_lwh_m",
        "radius_m",
        "height_m",
        "vertices_m",
        "extrusion_distance_m",
    }
    if any(name in geometry for name in metre_fields):
        raise InputValidationError(
            "metre and centimetre geometry fields cannot be mixed",
            context={"shape": shape},
        )

    if shape == "box":
        if "center_cm" not in geometry or "size_lwh_cm" not in geometry:
            raise InputValidationError(
                "box geometry in centimetres requires center_cm and size_lwh_cm"
            )
        geometry["center_m"] = _scale_vector(
            geometry.pop("center_cm"), field="center_cm"
        )
        geometry["size_lwh_m"] = _scale_vector(
            geometry.pop("size_lwh_cm"), field="size_lwh_cm"
        )
    elif shape == "cylinder":
        required = ("center_cm", "radius_cm", "height_cm")
        missing = [name for name in required if name not in geometry]
        if missing:
            raise InputValidationError(
                "cylinder geometry in centimetres is incomplete",
                context={"missing": missing},
            )
        geometry["center_m"] = _scale_vector(
            geometry.pop("center_cm"), field="center_cm"
        )
        try:
            geometry["radius_m"] = float(geometry.pop("radius_cm")) * _CM_TO_M
            geometry["height_m"] = float(geometry.pop("height_cm")) * _CM_TO_M
        except (TypeError, ValueError) as exc:
            raise InputValidationError(
                "radius_cm and height_cm must be numeric"
            ) from exc
    elif shape == "extruded_polygon":
        if "vertices_cm" not in geometry or "extrusion_distance_cm" not in geometry:
            raise InputValidationError(
                "extruded polygon in centimetres requires vertices_cm and extrusion_distance_cm"
            )
        vertices = geometry.pop("vertices_cm")
        if not isinstance(vertices, list):
            raise InputValidationError("vertices_cm must be an array")
        geometry["vertices_m"] = [
            _scale_vector(vertex, field="vertices_cm") for vertex in vertices
        ]
        try:
            geometry["extrusion_distance_m"] = (
                float(geometry.pop("extrusion_distance_cm")) * _CM_TO_M
            )
        except (TypeError, ValueError) as exc:
            raise InputValidationError(
                "extrusion_distance_cm must be numeric"
            ) from exc
    _normalize_dimensionless_geometry(geometry)


def normalize_target(target: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical SI representation of a target JSON object.

    Target files may use explicit ``*_m`` geometry fields or explicit ``*_cm``
    fields.  The two forms may not be mixed.  Response parameters retain their
    own unit metadata and are converted later by the parameter layer.
    """

    if not isinstance(target, dict):
        raise InputValidationError("target must be a JSON object")
    normalized = deepcopy(target)
    frame = normalized.get("coordinate_frame")
    if not isinstance(frame, dict):
        return normalized
    frame_unit = frame.get("length_unit")
    if frame_unit not in {"m", "cm"}:
        raise InputValidationError(
            "coordinate_frame.length_unit must be 'm' or 'cm'",
            context={"length_unit": frame_unit},
        )
    for collection_name in ("armor", "components"):
        collection = normalized.get(collection_name)
        if not isinstance(collection, list):
            continue
        for item in collection:
            if not isinstance(item, dict) or not isinstance(item.get("geometry"), dict):
                continue
            _normalize_geometry(item["geometry"], frame_unit=frame_unit)
    frame["length_unit"] = "m"
    return normalized


TRAINING_NORMALIZATION_SCHEMA_VERSION = "1.1.0"
LEGACY_TRAINING_NORMALIZATION_SCHEMA_VERSION = "1.0.0"
TRAINING_FEATURE_ROLES = (
    "varying_learnable",
    "context_fixed",
    "optional_masked",
    "derived",
)
SAMPLING_TO_TRAINING_FEATURE_ROLE = {
    "learnable_variable": "varying_learnable",
    "fixed_context": "context_fixed",
    "structural_missing": "optional_masked",
    "derived_input": "derived",
}


def _normalization_payload(document: Mapping[str, Any]) -> dict[str, Any]:
    payload = deepcopy(dict(document))
    payload.pop("normalization_id", None)
    return payload


def validate_training_normalization_artifact(
    document: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a defensive copy of a canonical training-normalization artifact.

    The artifact is deliberately separate from the immutable raw archive.  Its
    identity covers the exact Train case identities, the archive/source
    contracts, representation transforms, feature roles and fitted statistics.
    Constant features use an identity transform with a null scale; accepting an
    epsilon denominator here would manufacture variation that the Train split
    did not contain.
    """

    if not isinstance(document, Mapping):
        raise InputValidationError("normalization artifact must be a JSON object")
    result = deepcopy(dict(document))
    schema_version = result.get("schema_version")
    if schema_version not in {
        LEGACY_TRAINING_NORMALIZATION_SCHEMA_VERSION,
        TRAINING_NORMALIZATION_SCHEMA_VERSION,
    }:
        raise InputValidationError("unsupported training normalization schema_version")
    if result.get("artifact_type") != "training_view_normalization":
        raise InputValidationError("artifact_type must be training_view_normalization")
    fit = result.get("fit")
    if not isinstance(fit, dict) or fit.get("split") != "train":
        raise InputValidationError("normalization artifact must be fitted only on split='train'")
    case_ids = fit.get("case_ids")
    if (
        not isinstance(case_ids, list)
        or not case_ids
        or any(not isinstance(item, str) or not item for item in case_ids)
        or len(set(case_ids)) != len(case_ids)
    ):
        raise InputValidationError("fit.case_ids must contain unique nonempty Train identities")
    if fit.get("case_count") != len(case_ids):
        raise InputValidationError("fit.case_count does not match fit.case_ids")
    if fit.get("case_ids_hash") != content_hash(case_ids):
        raise InputValidationError("fit.case_ids_hash does not match fit.case_ids")

    source = result.get("source_contracts")
    if not isinstance(source, dict):
        raise InputValidationError("normalization artifact is missing source_contracts")
    for name in (
        "manifest_hash",
        "split_registry_hash",
        "damage_tree_definition_hash",
        "sampling_design_hash",
        "feature_schema_hash",
    ):
        field = f"source_contracts.{name}"
        require_sha256_identity(
            source.get(name),
            field=field,
            acceptance="legacy_prefix_length",
            failure_message=f"{field} must be a sha256 content hash",
        )
    for name in ("target_hashes", "parameter_set_hashes"):
        values = source.get(name)
        if not isinstance(values, list) or not values:
            raise InputValidationError(f"source_contracts.{name} must be a nonempty list")
        for index, value in enumerate(values):
            field = f"source_contracts.{name}[{index}]"
            require_sha256_identity(
                value,
                field=field,
                acceptance="legacy_prefix_length",
                failure_message=f"{field} must be a sha256 content hash",
            )

    roles = result.get("feature_roles")
    if not isinstance(roles, dict) or set(roles) != set(TRAINING_FEATURE_ROLES):
        raise InputValidationError("feature_roles must declare all four training-view roles")
    role_names: list[str] = []
    for role in TRAINING_FEATURE_ROLES:
        names = roles[role]
        if (
            not isinstance(names, list)
            or any(not isinstance(name, str) or not name for name in names)
            or len(names) != len(set(names))
        ):
            raise InputValidationError(f"feature_roles.{role} must contain unique names")
        role_names.extend(names)
    if len(role_names) != len(set(role_names)):
        raise InputValidationError("a training feature may belong to only one role")

    role_assignment = result.get("feature_role_assignment")
    declared_roles: dict[str, str] | None = None
    if schema_version == TRAINING_NORMALIZATION_SCHEMA_VERSION:
        if not isinstance(role_assignment, dict):
            raise InputValidationError(
                "training normalization 1.1 requires feature_role_assignment"
            )
        mode = role_assignment.get("mode")
        if mode == "legacy_observed_train_statistics":
            if set(role_assignment) != {"mode", "training_design_hash"}:
                raise InputValidationError(
                    "legacy feature_role_assignment contains unsupported fields"
                )
            if role_assignment.get("training_design_hash") is not None:
                raise InputValidationError(
                    "legacy feature-role inference may not claim a training design hash"
                )
            if source.get("training_design_hash") is not None:
                raise InputValidationError(
                    "a source training design requires declared feature-role assignment"
                )
        elif mode == "declared_sampling_spec_1.1":
            if set(role_assignment) != {
                "mode",
                "training_design_hash",
                "declared_feature_roles",
            }:
                raise InputValidationError(
                    "declared feature_role_assignment contains unsupported fields"
                )
            design_hash = require_sha256_identity(
                role_assignment.get("training_design_hash"),
                field="feature_role_assignment.training_design_hash",
                acceptance="legacy_prefix_length",
                failure_message=(
                    "feature_role_assignment.training_design_hash must be a "
                    "sha256 content hash"
                ),
            )
            if source.get("training_design_hash") != design_hash:
                raise InputValidationError(
                    "feature-role assignment disagrees with source training_design_hash"
                )
            raw_declared = role_assignment.get("declared_feature_roles")
            if (
                not isinstance(raw_declared, dict)
                or set(raw_declared) != set(role_names)
                or any(
                    not isinstance(name, str)
                    or role not in SAMPLING_TO_TRAINING_FEATURE_ROLE
                    for name, role in raw_declared.items()
                )
            ):
                raise InputValidationError(
                    "declared feature roles must cover every training tensor feature exactly once"
                )
            declared_roles = dict(raw_declared)
        else:
            raise InputValidationError("unsupported feature_role_assignment.mode")
    elif role_assignment is not None:
        raise InputValidationError(
            "legacy normalization 1.0 may not contain feature_role_assignment"
        )

    features = result.get("features")
    if not isinstance(features, list) or len(features) != len(role_names):
        raise InputValidationError("features must describe every role-assigned feature exactly once")
    feature_by_name: dict[str, dict[str, Any]] = {}
    for feature in features:
        if not isinstance(feature, dict):
            raise InputValidationError("each normalization feature must be an object")
        name = feature.get("name")
        role = feature.get("role")
        if name in feature_by_name or name not in role_names or role not in TRAINING_FEATURE_ROLES:
            raise InputValidationError("normalization feature name/role is duplicate or invalid")
        if name not in roles[role]:
            raise InputValidationError(f"normalization feature {name!r} disagrees with feature_roles")
        method = feature.get("normalization")
        available = feature.get("available_count")
        if type(available) is not int or not 0 <= available <= len(case_ids):
            raise InputValidationError(f"normalization feature {name!r} has invalid available_count")
        if method == "standardize":
            mean, scale = feature.get("mean"), feature.get("scale")
            if (
                not isinstance(mean, (int, float))
                or not isinstance(scale, (int, float))
                or not np.isfinite([mean, scale]).all()
                or float(scale) <= 0.0
                or feature.get("constant_among_available") is not False
            ):
                raise InputValidationError(f"normalization feature {name!r} has invalid fitted scale")
        elif method == "identity":
            if feature.get("scale") is not None:
                raise InputValidationError(
                    f"identity-normalized feature {name!r} must use scale=null, never epsilon"
                )
        else:
            raise InputValidationError(f"normalization feature {name!r} has unsupported method")
        if role == "varying_learnable" and (
            available != len(case_ids) or feature.get("constant_among_available") is not False
        ):
            raise InputValidationError(
                f"varying_learnable feature {name!r} is constant or missing on Train"
            )
        if declared_roles is not None:
            expected_role = SAMPLING_TO_TRAINING_FEATURE_ROLE[declared_roles[name]]
            if role != expected_role:
                raise InputValidationError(
                    f"normalization feature {name!r} disagrees with its declared SamplingSpec role"
                )
            if role == "context_fixed" and (
                available != len(case_ids)
                or feature.get("constant_among_available") is not True
            ):
                raise InputValidationError(
                    f"declared fixed_context feature {name!r} is missing or varies on Train"
                )
            if role == "optional_masked" and (
                available != 0 or feature.get("constant_among_available") is not None
            ):
                raise InputValidationError(
                    f"declared structural_missing feature {name!r} is present on Train"
                )
        feature_by_name[name] = feature

    proxy = result.get("proxy_boundary")
    if not isinstance(proxy, dict) or proxy.get("mode") not in {
        "end_to_end",
        "solver_surrogate",
    }:
        raise InputValidationError("proxy_boundary.mode must be end_to_end or solver_surrogate")
    derived = roles["derived"]
    if proxy["mode"] == "end_to_end" and derived:
        raise InputValidationError("end_to_end artifacts may not include derived solver inputs")
    if proxy["mode"] == "end_to_end" and (
        proxy.get("derived_input_contract_hash") is not None
        or "derived_input_contract" in proxy
    ):
        raise InputValidationError("end_to_end artifacts may not declare derived solver inputs")
    if proxy["mode"] == "solver_surrogate":
        contract_hash = proxy.get("derived_input_contract_hash")
        require_sha256_identity(
            contract_hash,
            field="proxy_boundary.derived_input_contract_hash",
            acceptance="legacy_prefix_length",
            failure_message=(
                "proxy_boundary.derived_input_contract_hash must be a sha256 "
                "content hash"
            ),
        )
        contract = proxy.get("derived_input_contract")
        if not isinstance(contract, dict) or content_hash(contract) != contract_hash:
            raise InputValidationError(
                "proxy_boundary derived_input_contract hash does not match its document"
            )
        if proxy.get("deployment_available") is not True:
            raise InputValidationError(
                "solver_surrogate derived inputs require an explicitly deployable preprocessing contract"
            )

    expected = content_hash(_normalization_payload(result))
    if result.get("normalization_id") != expected:
        raise InputValidationError("normalization_id does not match the artifact content")
    return result


def make_training_normalization_artifact(
    *,
    fit_case_ids: Sequence[str],
    source_contracts: Mapping[str, Any],
    feature_names: Sequence[str],
    feature_units: Sequence[str],
    feature_values: np.ndarray,
    feature_mask: np.ndarray,
    derived_feature_names: Sequence[str] = (),
    declared_feature_roles: Mapping[str, str] | None = None,
    training_design_hash: str | None = None,
    proxy_boundary: Mapping[str, Any],
    transforms: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Fit a deterministic, Train-only normalization artifact.

    The caller supplies already transformed feature coordinates.  This keeps
    representation decisions (for example quaternion-to-6D) explicit and makes
    the fitted statistics independently testable.  No target or validation
    labels are accepted by this API.
    """

    case_ids = [str(item) for item in fit_case_ids]
    values = np.asarray(feature_values, dtype=np.float64)
    mask = np.asarray(feature_mask, dtype=bool)
    names = [str(item) for item in feature_names]
    units = [str(item) for item in feature_units]
    derived = set(map(str, derived_feature_names))
    if values.ndim != 2 or mask.shape != values.shape:
        raise InputValidationError("training features and masks must share shape [B,F]")
    if values.shape != (len(case_ids), len(names)) or len(units) != len(names):
        raise InputValidationError("training feature axes do not match case/name/unit identities")
    if len(set(names)) != len(names) or any(not name for name in names):
        raise InputValidationError("training feature names must be unique and nonempty")
    if not derived.issubset(names):
        raise InputValidationError("derived_feature_names must be present in feature_names")
    if not np.isfinite(values[mask]).all():
        raise InputValidationError("available training features must be finite")

    declared: dict[str, str] | None = None
    if declared_feature_roles is not None:
        declared = {str(name): str(role) for name, role in declared_feature_roles.items()}
        if set(declared) != set(names):
            raise InputValidationError(
                "declared feature roles must cover every training tensor feature exactly once"
            )
        unknown_roles = sorted(
            set(declared.values()) - set(SAMPLING_TO_TRAINING_FEATURE_ROLE)
        )
        if unknown_roles:
            raise InputValidationError(
                "declared feature roles contain unsupported SamplingSpec roles",
                context={"roles": unknown_roles},
            )
        require_sha256_identity(
            training_design_hash,
            field="feature_role_assignment.training_design_hash",
            acceptance="legacy_prefix_length",
            failure_message=(
                "feature_role_assignment.training_design_hash must be a sha256 "
                "content hash"
            ),
        )
        if source_contracts.get("training_design_hash") != training_design_hash:
            raise InputValidationError(
                "declared feature roles disagree with source training_design_hash"
            )
        declared_derived = {
            name for name, role in declared.items() if role == "derived_input"
        }
        if declared_derived != derived:
            raise InputValidationError(
                "declared derived_input features must exactly match solver-derived tensor features"
            )
    elif training_design_hash is not None:
        raise InputValidationError(
            "training_design_hash requires declared feature roles"
        )
    elif source_contracts.get("training_design_hash") is not None:
        raise InputValidationError(
            "a source training design requires declared feature roles"
        )

    roles = {role: [] for role in TRAINING_FEATURE_ROLES}
    features: list[dict[str, Any]] = []
    for index, (name, unit) in enumerate(zip(names, units)):
        available = mask[:, index]
        count = int(np.sum(available))
        observed = values[available, index]
        observed_range = float(np.ptp(observed)) if count else None
        # Training tensors are float32.  Representation transforms can turn a
        # mathematically constant quantity (for example a fixed sphere radius)
        # into adjacent float32 values.  Treat at most eight float32 ULP-scale
        # roundoff steps as constant; this is a storage-resolution bound, not a
        # fitted epsilon denominator and is recorded in the artifact.
        roundoff_bound = (
            float(8.0 * abs(np.spacing(np.float32(np.max(np.abs(observed))))))
            if count
            else None
        )
        constant = count > 0 and bool(observed_range <= roundoff_bound)
        if declared is not None:
            role = SAMPLING_TO_TRAINING_FEATURE_ROLE[declared[name]]
            if role == "varying_learnable" and (
                count != len(case_ids) or constant
            ):
                raise InputValidationError(
                    f"declared learnable_variable feature {name!r} is missing or constant on Train"
                )
            if role == "context_fixed" and (
                count != len(case_ids) or not constant
            ):
                raise InputValidationError(
                    f"declared fixed_context feature {name!r} is missing or varies on Train"
                )
            if role == "optional_masked" and count != 0:
                raise InputValidationError(
                    f"declared structural_missing feature {name!r} is present on Train"
                )
        elif name in derived:
            role = "derived"
        elif count < len(case_ids):
            role = "optional_masked"
        elif constant:
            role = "context_fixed"
        else:
            role = "varying_learnable"
        roles[role].append(name)
        if count > 0 and not constant:
            mean = float(np.mean(observed))
            scale = float(np.std(observed, ddof=0))
            # Exact constant detection above prevents epsilon pseudo-variation.
            # A positive but subnormal scale is still the actual Train statistic.
            if not math.isfinite(scale) or scale <= 0.0:
                raise InputValidationError(f"feature {name!r} has an invalid Train standard deviation")
            method = "standardize"
        else:
            mean = float(observed[0]) if count else None
            scale = None
            method = "identity"
        features.append(
            {
                "name": name,
                "unit": unit,
                "role": role,
                "available_count": count,
                "missing_count": len(case_ids) - count,
                "constant_among_available": constant if count else None,
                "observed_range": observed_range,
                "constant_detection_roundoff_bound": roundoff_bound,
                "constant_detection": "float32_storage_resolution_8ulp",
                "normalization": method,
                "mean": mean,
                "scale": scale,
            }
        )

    source = deepcopy(dict(source_contracts))
    source["feature_schema_hash"] = content_hash(
        {"feature_names": names, "feature_units": units}
    )
    document: dict[str, Any] = {
        "schema_version": TRAINING_NORMALIZATION_SCHEMA_VERSION,
        "artifact_type": "training_view_normalization",
        "fit": {
            "split": "train",
            "case_count": len(case_ids),
            "case_ids": case_ids,
            "case_ids_hash": content_hash(case_ids),
        },
        "source_contracts": source,
        "proxy_boundary": deepcopy(dict(proxy_boundary)),
        "transforms": [deepcopy(dict(item)) for item in transforms],
        "feature_role_assignment": (
            {
                "mode": "declared_sampling_spec_1.1",
                "training_design_hash": training_design_hash,
                "declared_feature_roles": declared,
            }
            if declared is not None
            else {
                "mode": "legacy_observed_train_statistics",
                "training_design_hash": None,
            }
        ),
        "feature_roles": roles,
        "features": features,
    }
    document["normalization_id"] = content_hash(document)
    return validate_training_normalization_artifact(document)


def apply_training_normalization(
    values: np.ndarray,
    mask: np.ndarray,
    artifact: Mapping[str, Any],
) -> np.ndarray:
    """Apply fitted statistics while leaving unavailable coordinates at zero."""

    document = validate_training_normalization_artifact(artifact)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        array = np.asarray(values, dtype=np.float32)
    available = np.asarray(mask, dtype=bool)
    if array.shape != available.shape or array.shape[-1] != len(document["features"]):
        raise InputValidationError("training values/mask do not match normalization features")
    output = np.zeros_like(array, dtype=np.float32)
    for index, specification in enumerate(document["features"]):
        selected = available[..., index]
        feature_name = specification["name"]
        context = {
            "feature_name": feature_name,
            "normalization_id": document["normalization_id"],
            "training_dtype": "float32",
        }
        if np.any(~np.isfinite(array[..., index][selected])):
            raise InputValidationError(
                "available training feature is non-finite after float32 conversion",
                context=context,
            )
        if specification["normalization"] == "standardize":
            with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
                output[..., index][selected] = (
                    array[..., index][selected] - float(specification["mean"])
                ) / float(specification["scale"])
        else:
            output[..., index][selected] = array[..., index][selected]
        if np.any(~np.isfinite(output[..., index][selected])):
            raise InputValidationError(
                "normalized available training feature is non-finite",
                context=context,
            )
    return output


def save_training_normalization_artifact(
    document: Mapping[str, Any], path: str | Path
) -> Path:
    validated = validate_training_normalization_artifact(document)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(validated, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def load_training_normalization_artifact(path: str | Path) -> dict[str, Any]:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputValidationError("failed to read training normalization artifact") from exc
    return validate_training_normalization_artifact(document)
