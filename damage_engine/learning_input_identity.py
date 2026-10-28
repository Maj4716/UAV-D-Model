"""Versioned model-input and fixed-scenario identities.

The simulation case intentionally remains a complete teacher-engine input. A
learner must not inherit that complete input implicitly: projectile, blast,
environment, solver and Monte-Carlo controls are scenario/teacher state, not
learner variables. This module is the single executable definition of the
model-facing states and of their independently frozen scenario identities.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from .input_identity import semantic_case_value
from .utils import content_hash


UAV_STATE_CONTRACT_VERSION = "uav-state-input@1.0.0"
UAV_SAMPLING_MODEL_INPUT_CONTRACT = "uav_state_only@1.0.0"
FIXED_SCENARIO_IDENTITY_VERSION = "fixed-scenario-json@1.0.0"
LEARNING_INPUT_IDENTITY_VERSION = "uav-state-learning-input@2.0.0"

LOITERING_IMPACT_STATE_CONTRACT_VERSION = "loitering-impact-state-input@1.0.0"
LOITERING_IMPACT_SAMPLING_MODEL_INPUT_CONTRACT = (
    "loitering_impact_state_only@1.0.0"
)
LOITERING_IMPACT_FIXED_SCENARIO_IDENTITY_VERSION = (
    "loitering-impact-fixed-scenario-json@1.0.0"
)
LOITERING_IMPACT_LEARNING_INPUT_IDENTITY_VERSION = (
    "loitering-impact-learning-input@1.0.0"
)

LOITERING_IMPACT_CONTACT_POSE_STATE_CONTRACT_VERSION = (
    "loitering-impact-contact-pose-input@1.0.0"
)
LOITERING_IMPACT_CONTACT_POSE_SAMPLING_MODEL_INPUT_CONTRACT = (
    "loitering_impact_contact_pose@1.0.0"
)
LOITERING_IMPACT_CONTACT_POSE_FIXED_SCENARIO_IDENTITY_VERSION = (
    "loitering-impact-contact-pose-fixed-scenario-json@1.0.0"
)
LOITERING_IMPACT_CONTACT_POSE_LEARNING_INPUT_IDENTITY_VERSION = (
    "loitering-impact-contact-pose-learning-input@1.0.0"
)

UAV_STATE_FEATURE_NAMES = (
    "uav_position_relative_to_fixed_threat_x_m",
    "uav_position_relative_to_fixed_threat_y_m",
    "uav_position_relative_to_fixed_threat_z_m",
    "uav_velocity_relative_to_fixed_threat_x_mps",
    "uav_velocity_relative_to_fixed_threat_y_mps",
    "uav_velocity_relative_to_fixed_threat_z_mps",
    "uav_attitude_relative_to_fixed_threat_qx",
    "uav_attitude_relative_to_fixed_threat_qy",
    "uav_attitude_relative_to_fixed_threat_qz",
    "uav_attitude_relative_to_fixed_threat_qw",
)
UAV_STATE_FEATURE_UNITS = (
    "m", "m", "m", "m/s", "m/s", "m/s", "1", "1", "1", "1",
)
UAV_STATE_FEATURE_SOURCE_PATHS = (
    *("/encounter/detonation_position_m",) * 3,
    *("/encounter/velocity_target_mps",) * 3,
    *("/encounter/body_to_target_orientation_xyzw",) * 4,
)
UAV_STATE_FEATURE_TRANSFORMS = (
    *("negate_fixed_threat_position_in_target_frame",) * 3,
    *("negate_fixed_threat_velocity_in_target_frame",) * 3,
    *("normalize_then_invert_body_to_target_quaternion_and_canonicalize_sign",) * 4,
)
UAV_STATE_FEATURE_SEMANTICS = (
    "UAV x-position relative to the fixed threat/detonation origin, expressed in the frozen scenario frame.",
    "UAV y-position relative to the fixed threat/detonation origin, expressed in the frozen scenario frame.",
    "UAV z-position relative to the fixed threat/detonation origin, expressed in the frozen scenario frame.",
    "UAV x-velocity relative to the fixed threat, under the frozen-threat translational-frame convention.",
    "UAV y-velocity relative to the fixed threat, under the frozen-threat translational-frame convention.",
    "UAV z-velocity relative to the fixed threat, under the frozen-threat translational-frame convention.",
    "x component of the canonical UAV/target attitude relative to the fixed threat body frame.",
    "y component of the canonical UAV/target attitude relative to the fixed threat body frame.",
    "z component of the canonical UAV/target attitude relative to the fixed threat body frame.",
    "w component of the canonical UAV/target attitude relative to the fixed threat body frame.",
)
UAV_ROTATION_6D_FEATURE_NAMES = (
    "uav_attitude_relative_to_fixed_threat_rotation_6d_col0_x",
    "uav_attitude_relative_to_fixed_threat_rotation_6d_col0_y",
    "uav_attitude_relative_to_fixed_threat_rotation_6d_col0_z",
    "uav_attitude_relative_to_fixed_threat_rotation_6d_col1_x",
    "uav_attitude_relative_to_fixed_threat_rotation_6d_col1_y",
    "uav_attitude_relative_to_fixed_threat_rotation_6d_col1_z",
)
UAV_TRAINING_FEATURE_NAMES = (
    *UAV_STATE_FEATURE_NAMES[:6],
    *UAV_ROTATION_6D_FEATURE_NAMES,
)

LOITERING_IMPACT_PROFILE_IDS = (
    "small_front",
    "medium_front",
    "large_front",
)
LOITERING_IMPACT_STATE_FEATURE_NAMES = (
    "munition_profile_small_front",
    "munition_profile_medium_front",
    "munition_profile_large_front",
    "trajectory_origin_y_m",
    "trajectory_origin_z_m",
    "impact_speed_mps",
)
LOITERING_IMPACT_STATE_FEATURE_UNITS = ("1", "1", "1", "m", "m", "m/s")
LOITERING_IMPACT_STATE_FEATURE_SOURCE_PATHS = (
    *("/munition/profile_id",) * 3,
    "/encounter/trajectory_origin_m/1",
    "/encounter/trajectory_origin_m/2",
    "/encounter/velocity_target_mps/0",
)
LOITERING_IMPACT_STATE_FEATURE_TRANSFORMS = (
    "one_hot_equals_small_front",
    "one_hot_equals_medium_front",
    "one_hot_equals_large_front",
    "identity",
    "identity",
    "identity_positive_x_component",
)
LOITERING_IMPACT_STATE_FEATURE_SEMANTICS = (
    "One-hot indicator for the versioned small front-detonating munition profile.",
    "One-hot indicator for the versioned medium front-detonating munition profile.",
    "One-hot indicator for the versioned large munition profile.",
    "Nose-fuze trajectory-origin y coordinate in the target frame before impact resolution.",
    "Nose-fuze trajectory-origin z coordinate in the target frame before impact resolution.",
    "Positive x-directed impact speed in the target frame before impact resolution.",
)
LOITERING_IMPACT_TRAINING_FEATURE_NAMES = LOITERING_IMPACT_STATE_FEATURE_NAMES

LOITERING_IMPACT_CONTACT_POSE_QUATERNION_FEATURE_NAMES = (
    "body_to_target_orientation_qx",
    "body_to_target_orientation_qy",
    "body_to_target_orientation_qz",
    "body_to_target_orientation_qw",
)
LOITERING_IMPACT_CONTACT_POSE_ROTATION_6D_FEATURE_NAMES = (
    "body_to_target_rotation_6d_col0_x",
    "body_to_target_rotation_6d_col0_y",
    "body_to_target_rotation_6d_col0_z",
    "body_to_target_rotation_6d_col1_x",
    "body_to_target_rotation_6d_col1_y",
    "body_to_target_rotation_6d_col1_z",
)
LOITERING_IMPACT_CONTACT_POSE_STATE_FEATURE_NAMES = (
    *LOITERING_IMPACT_STATE_FEATURE_NAMES[:3],
    "trajectory_origin_x_m",
    "trajectory_origin_y_m",
    "trajectory_origin_z_m",
    "approach_velocity_target_x_mps",
    "approach_velocity_target_y_mps",
    "approach_velocity_target_z_mps",
    *LOITERING_IMPACT_CONTACT_POSE_QUATERNION_FEATURE_NAMES,
)
LOITERING_IMPACT_CONTACT_POSE_STATE_FEATURE_UNITS = (
    *("1",) * 3,
    *("m",) * 3,
    *("m/s",) * 3,
    *("1",) * 4,
)
LOITERING_IMPACT_CONTACT_POSE_STATE_FEATURE_SOURCE_PATHS = (
    *("/munition/profile_id",) * 3,
    *("/encounter/trajectory_origin_m",) * 3,
    *("/encounter/velocity_target_mps",) * 3,
    *("/encounter/body_to_target_orientation_xyzw",) * 4,
)
LOITERING_IMPACT_CONTACT_POSE_STATE_FEATURE_TRANSFORMS = (
    "one_hot_equals_small_front",
    "one_hot_equals_medium_front",
    "one_hot_equals_large_front",
    *("identity_target_frame_component",) * 6,
    *("normalize_and_canonicalize_sign",) * 4,
)
LOITERING_IMPACT_CONTACT_POSE_STATE_FEATURE_SEMANTICS = (
    *LOITERING_IMPACT_STATE_FEATURE_SEMANTICS[:3],
    "Nose-fuze trajectory-origin x coordinate in the frozen target frame before impact resolution.",
    "Nose-fuze trajectory-origin y coordinate in the frozen target frame before impact resolution.",
    "Nose-fuze trajectory-origin z coordinate in the frozen target frame before impact resolution.",
    "Target-frame x component of the pre-impact velocity jointly constructed on the contact manifold.",
    "Target-frame y component of the pre-impact velocity jointly constructed on the contact manifold.",
    "Target-frame z component of the pre-impact velocity jointly constructed on the contact manifold.",
    "x component of the canonical body-to-target attitude aligned with the approach velocity.",
    "y component of the canonical body-to-target attitude aligned with the approach velocity.",
    "z component of the canonical body-to-target attitude aligned with the approach velocity.",
    "w component of the canonical body-to-target attitude aligned with the approach velocity.",
)
LOITERING_IMPACT_CONTACT_POSE_TRAINING_FEATURE_NAMES = (
    *LOITERING_IMPACT_CONTACT_POSE_STATE_FEATURE_NAMES[:9],
    *LOITERING_IMPACT_CONTACT_POSE_ROTATION_6D_FEATURE_NAMES,
)
LOITERING_IMPACT_CONTACT_POSE_TRAINING_FEATURE_UNITS = (
    *("1",) * 3,
    *("m",) * 3,
    *("m/s",) * 3,
    *("1",) * 6,
)
LOITERING_IMPACT_CONTACT_POSE_TRAINING_FEATURE_SOURCE_PATHS = (
    *("/munition/profile_id",) * 3,
    *("/encounter/trajectory_origin_m",) * 3,
    *("/encounter/velocity_target_mps",) * 3,
    *("/encounter/body_to_target_orientation_xyzw",) * 6,
)
LOITERING_IMPACT_CONTACT_POSE_TRAINING_FEATURE_TRANSFORMS = (
    "one_hot_equals_small_front",
    "one_hot_equals_medium_front",
    "one_hot_equals_large_front",
    *("identity_target_frame_component",) * 6,
    *("normalized_quaternion_to_rotation_matrix_first_two_columns",) * 6,
)
LOITERING_IMPACT_CONTACT_POSE_TRAINING_FEATURE_SEMANTICS = (
    *LOITERING_IMPACT_CONTACT_POSE_STATE_FEATURE_SEMANTICS[:9],
    "Target-frame x component of the body x axis; invariant to quaternion sign.",
    "Target-frame y component of the body x axis; invariant to quaternion sign.",
    "Target-frame z component of the body x axis; invariant to quaternion sign.",
    "Target-frame x component of the body y axis; invariant to quaternion sign.",
    "Target-frame y component of the body y axis; invariant to quaternion sign.",
    "Target-frame z component of the body y axis; invariant to quaternion sign.",
)


MODEL_INPUT_EXCLUDED_TENSOR_FIELDS = (
    "case_x",
    "case_feature_mask",
    "case_physics_x",
    "case_physics_feature_mask",
    "solver_context_x",
    "solver_context_feature_mask",
    "component_x",
    "component_feature_mask",
    "armor_x",
    "armor_feature_mask",
    "component_armor_path_x",
    "component_armor_path_mask",
    "armor_vertices_m",
    "armor_vertex_offsets",
    "component_vertices_m",
    "component_vertex_offsets",
    "tree_edge_index",
    "tree_edge_type",
)


@dataclass(frozen=True)
class ModelInputDefinition:
    """Complete versioned definition of one learner-visible input tensor."""

    sampling_contract: str
    version: str
    tensor_field: str
    mask_field: str
    metadata_prefix: str
    feature_axis: str
    feature_names: tuple[str, ...]
    feature_units: tuple[str, ...]
    feature_source_paths: tuple[str, ...]
    feature_transforms: tuple[str, ...]
    feature_semantics: tuple[str, ...]
    fixed_scenario_identity_version: str
    learning_input_identity_version: str
    learning_input_payload_key: str
    evidence_boundary: str
    known_limitation: str

    def __post_init__(self) -> None:
        lengths = {
            len(self.feature_names),
            len(self.feature_units),
            len(self.feature_source_paths),
            len(self.feature_transforms),
            len(self.feature_semantics),
        }
        if len(lengths) != 1 or not self.feature_names:
            raise ValueError("model-input feature metadata must be nonempty and aligned")

    @property
    def shape(self) -> str:
        return f"[B,{self.feature_axis}]"

    @property
    def metadata_fields(self) -> dict[str, tuple[str, ...]]:
        prefix = self.metadata_prefix
        return {
            f"{prefix}_feature_names": self.feature_names,
            f"{prefix}_feature_units": self.feature_units,
            f"{prefix}_feature_source_paths": self.feature_source_paths,
            f"{prefix}_feature_transforms": self.feature_transforms,
            f"{prefix}_feature_semantics": self.feature_semantics,
        }

    @property
    def tensor_shapes(self) -> dict[str, str]:
        return {
            self.tensor_field: self.shape,
            self.mask_field: self.shape,
            **{
                field: f"[{self.feature_axis}]"
                for field in self.metadata_fields
            },
        }

    @property
    def axis_document(self) -> dict[str, object]:
        return {
            "axis_id": self.feature_axis,
            "size": len(self.feature_names),
            "names_field": f"{self.metadata_prefix}_feature_names",
            "units_field": f"{self.metadata_prefix}_feature_units",
        }

    def manifest_document(self, *, fixed_scenario_hash: str) -> dict:
        return {
            "version": self.version,
            "tensor_field": self.tensor_field,
            "mask_field": self.mask_field,
            "dtype": "float32",
            "shape": self.shape,
            "feature_axis": self.feature_axis,
            "features": [
                {
                    "name": name,
                    "unit": unit,
                    "semantic": semantic,
                    "source_path": source_path,
                    "transformation": transform,
                }
                for name, unit, semantic, source_path, transform in zip(
                    self.feature_names,
                    self.feature_units,
                    self.feature_semantics,
                    self.feature_source_paths,
                    self.feature_transforms,
                    strict=True,
                )
            ],
            "excluded_tensor_fields": list(MODEL_INPUT_EXCLUDED_TENSOR_FIELDS),
            "fixed_scenario_policy": "single_frozen_scenario_per_archive",
            "fixed_scenario_hash": fixed_scenario_hash,
            "evidence_boundary": self.evidence_boundary,
        }

    def metadata_matches(self, metadata: Mapping[str, object]) -> bool:
        return all(
            tuple(map(str, metadata.get(field, ()))) == expected
            for field, expected in self.metadata_fields.items()
        )

    def vector(self, case: dict) -> list[float]:
        """Extract the sole learner-visible vector for this definition."""

        return _MODEL_INPUT_VECTOR_ADAPTERS[self.version](case)

    def fixed_scenario_fingerprint(self, case: dict) -> str:
        """Bind teacher/scenario state excluded from the learner vector."""

        return _MODEL_INPUT_SCENARIO_ADAPTERS[self.version](case)

    def learning_input_value(self, case: dict) -> dict[str, float]:
        """Extract the named state that is bound into the learning identity."""

        return _MODEL_INPUT_VALUE_ADAPTERS[self.version](case)

    def learning_input_fingerprint(self, case: dict) -> str:
        """Hash this definition's canonical learner-visible state.

        The payload key and identity version live beside the tensor contract so
        adding a learner contract cannot silently reuse another contract's hash
        namespace or payload shape.
        """

        return content_hash(
            {
                "version": self.learning_input_identity_version,
                self.learning_input_payload_key: self.learning_input_value(case),
            }
        )


UAV_MODEL_INPUT_DEFINITION = ModelInputDefinition(
    sampling_contract=UAV_SAMPLING_MODEL_INPUT_CONTRACT,
    version=UAV_STATE_CONTRACT_VERSION,
    tensor_field="uav_state_x",
    mask_field="uav_state_feature_mask",
    metadata_prefix="uav_state",
    feature_axis="F_uav_state",
    feature_names=UAV_STATE_FEATURE_NAMES,
    feature_units=UAV_STATE_FEATURE_UNITS,
    feature_source_paths=UAV_STATE_FEATURE_SOURCE_PATHS,
    feature_transforms=UAV_STATE_FEATURE_TRANSFORMS,
    feature_semantics=UAV_STATE_FEATURE_SEMANTICS,
    fixed_scenario_identity_version=FIXED_SCENARIO_IDENTITY_VERSION,
    learning_input_identity_version=LEARNING_INPUT_IDENTITY_VERSION,
    learning_input_payload_key="uav_state",
    evidence_boundary=(
        "relative UAV state under the current fixed-threat frame convention; "
        "not an absolute world pose and not physical-validation evidence"
    ),
    known_limitation=(
        "The UAV state is reconstructed from the current encounter convention: "
        "position/velocity are the negatives of fixed-threat vectors and attitude "
        "is the inverse body-to-target rotation; the archive does not contain an "
        "independent absolute world pose."
    ),
)

LOITERING_IMPACT_MODEL_INPUT_DEFINITION = ModelInputDefinition(
    sampling_contract=LOITERING_IMPACT_SAMPLING_MODEL_INPUT_CONTRACT,
    version=LOITERING_IMPACT_STATE_CONTRACT_VERSION,
    tensor_field="loitering_impact_x",
    mask_field="loitering_impact_feature_mask",
    metadata_prefix="loitering_impact",
    feature_axis="F_loitering_impact",
    feature_names=LOITERING_IMPACT_STATE_FEATURE_NAMES,
    feature_units=LOITERING_IMPACT_STATE_FEATURE_UNITS,
    feature_source_paths=LOITERING_IMPACT_STATE_FEATURE_SOURCE_PATHS,
    feature_transforms=LOITERING_IMPACT_STATE_FEATURE_TRANSFORMS,
    feature_semantics=LOITERING_IMPACT_STATE_FEATURE_SEMANTICS,
    fixed_scenario_identity_version=(
        LOITERING_IMPACT_FIXED_SCENARIO_IDENTITY_VERSION
    ),
    learning_input_identity_version=(
        LOITERING_IMPACT_LEARNING_INPUT_IDENTITY_VERSION
    ),
    learning_input_payload_key="loitering_impact_state",
    evidence_boundary=(
        "three versioned munition-profile categories plus sampled pre-impact "
        "origin y/z and positive x speed; collision solution, target geometry "
        "and teacher physics are excluded, and the contract is not "
        "physical-validation evidence"
    ),
    known_limitation=(
        "The six-value loitering-impact state treats each munition profile as a "
        "categorical, code-versioned configuration; it does not expose or "
        "independently calibrate the profile's physical parameters."
    ),
)

LOITERING_IMPACT_CONTACT_POSE_MODEL_INPUT_DEFINITION = ModelInputDefinition(
    sampling_contract=(
        LOITERING_IMPACT_CONTACT_POSE_SAMPLING_MODEL_INPUT_CONTRACT
    ),
    version=LOITERING_IMPACT_CONTACT_POSE_STATE_CONTRACT_VERSION,
    tensor_field="loitering_impact_contact_pose_x",
    mask_field="loitering_impact_contact_pose_feature_mask",
    metadata_prefix="loitering_impact_contact_pose",
    feature_axis="F_loitering_impact_contact_pose",
    feature_names=LOITERING_IMPACT_CONTACT_POSE_TRAINING_FEATURE_NAMES,
    feature_units=LOITERING_IMPACT_CONTACT_POSE_TRAINING_FEATURE_UNITS,
    feature_source_paths=(
        LOITERING_IMPACT_CONTACT_POSE_TRAINING_FEATURE_SOURCE_PATHS
    ),
    feature_transforms=LOITERING_IMPACT_CONTACT_POSE_TRAINING_FEATURE_TRANSFORMS,
    feature_semantics=LOITERING_IMPACT_CONTACT_POSE_TRAINING_FEATURE_SEMANTICS,
    fixed_scenario_identity_version=(
        LOITERING_IMPACT_CONTACT_POSE_FIXED_SCENARIO_IDENTITY_VERSION
    ),
    learning_input_identity_version=(
        LOITERING_IMPACT_CONTACT_POSE_LEARNING_INPUT_IDENTITY_VERSION
    ),
    learning_input_payload_key="loitering_impact_contact_pose_state",
    evidence_boundary=(
        "three versioned munition-profile categories plus pre-impact target-frame "
        "origin, velocity and body attitude; collision solution and teacher "
        "physics are excluded, and approach weights describe a contact-conditioned "
        "design distribution rather than operational prevalence"
    ),
    known_limitation=(
        "The target-facing ray manifold and its approach weights are project-specific "
        "design choices. S-CONTACT proves first-armour-contact consistency but does "
        "not validate operational attack frequencies or physical response fidelity."
    ),
)

MODEL_INPUT_DEFINITIONS = (
    LOITERING_IMPACT_CONTACT_POSE_MODEL_INPUT_DEFINITION,
    LOITERING_IMPACT_MODEL_INPUT_DEFINITION,
    UAV_MODEL_INPUT_DEFINITION,
)
MODEL_INPUT_METADATA_FIELDS = tuple(
    field
    for definition in MODEL_INPUT_DEFINITIONS
    for field in definition.metadata_fields
)
MODEL_INPUT_AXIS_NAME_FIELDS = {
    definition.feature_axis: f"{definition.metadata_prefix}_feature_names"
    for definition in MODEL_INPUT_DEFINITIONS
}
MODEL_INPUT_MASK_PAIRS = {
    definition.tensor_field: definition.mask_field
    for definition in MODEL_INPUT_DEFINITIONS
}
MODEL_INPUT_SEMANTIC_SHAPES = {
    field: ("B", definition.feature_axis)
    for definition in MODEL_INPUT_DEFINITIONS
    for field in (definition.tensor_field, definition.mask_field)
}
_MODEL_INPUT_BY_VERSION = {
    definition.version: definition for definition in MODEL_INPUT_DEFINITIONS
}
_MODEL_INPUT_BY_SAMPLING_CONTRACT = {
    definition.sampling_contract: definition
    for definition in MODEL_INPUT_DEFINITIONS
}
_MODEL_INPUT_BY_TENSOR_FIELD = {
    definition.tensor_field: definition for definition in MODEL_INPUT_DEFINITIONS
}


def model_input_definition_for_version(version: object) -> ModelInputDefinition | None:
    return _MODEL_INPUT_BY_VERSION.get(version)


def model_input_definition_for_sampling_contract(
    contract: object,
) -> ModelInputDefinition | None:
    return _MODEL_INPUT_BY_SAMPLING_CONTRACT.get(contract)


def model_input_definition_for_tensor_field(
    tensor_field: object,
) -> ModelInputDefinition | None:
    return _MODEL_INPUT_BY_TENSOR_FIELD.get(tensor_field)


def model_input_definition_from_metadata(
    metadata: Mapping[str, object],
) -> ModelInputDefinition | None:
    """Resolve current model input while preserving historical read precedence."""

    for definition in MODEL_INPUT_DEFINITIONS:
        names = metadata.get(f"{definition.metadata_prefix}_feature_names", ())
        units = metadata.get(f"{definition.metadata_prefix}_feature_units", ())
        try:
            declared = len(names) > 0 or len(units) > 0  # type: ignore[arg-type]
        except TypeError:
            declared = bool(names) or bool(units)
        if declared:
            return definition
    return None


def _finite_vector(value: Any, *, length: int, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _canonical_inverse_quaternion_xyzw(value: Any) -> list[float]:
    x, y, z, w = _finite_vector(
        value, length=4, name="encounter.body_to_target_orientation_xyzw"
    )
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        raise ValueError("encounter.body_to_target_orientation_xyzw must be nonzero")
    # Inverse of a unit body->target rotation: target/UAV attitude relative to
    # the fixed threat body frame. Canonical sign removes the q/-q duplicate.
    quaternion = [-x / norm, -y / norm, -z / norm, w / norm]
    for component in reversed(quaternion):  # prefer w, then z/y/x at the tie
        if abs(component) > 1.0e-15:
            if component < 0.0:
                quaternion = [-item for item in quaternion]
            break
    return quaternion


def _canonical_quaternion_xyzw(value: Any) -> list[float]:
    """Normalize a body-to-target quaternion and remove the q/-q duplicate."""

    quaternion = _finite_vector(
        value, length=4, name="encounter.body_to_target_orientation_xyzw"
    )
    norm = math.sqrt(sum(item * item for item in quaternion))
    if norm <= 0.0:
        raise ValueError("encounter.body_to_target_orientation_xyzw must be nonzero")
    quaternion = [item / norm for item in quaternion]
    for component in reversed(quaternion):
        if abs(component) > 1.0e-15:
            if component < 0.0:
                quaternion = [-item for item in quaternion]
            break
    return quaternion


def uav_state_value(case: dict) -> dict[str, float]:
    """Return the canonical ten-value UAV-only learner state.

    The present engine stores the fixed threat/detonation origin in target
    coordinates. Therefore UAV-relative position and velocity are the
    negatives of those encounter vectors. The returned names deliberately do
    not claim a global/world position that the current case schema does not
    contain.
    """

    encounter = case.get("encounter")
    if not isinstance(encounter, dict):
        raise ValueError("case is missing encounter")
    threat_position = _finite_vector(
        encounter.get("detonation_position_m"),
        length=3,
        name="encounter.detonation_position_m",
    )
    threat_velocity = _finite_vector(
        encounter.get("velocity_target_mps"),
        length=3,
        name="encounter.velocity_target_mps",
    )
    attitude = _canonical_inverse_quaternion_xyzw(
        encounter.get("body_to_target_orientation_xyzw")
    )
    values = [
        *(-item for item in threat_position),
        *(-item for item in threat_velocity),
        *attitude,
    ]
    return dict(zip(UAV_STATE_FEATURE_NAMES, values, strict=True))


def uav_state_vector(case: dict) -> list[float]:
    state = uav_state_value(case)
    return [state[name] for name in UAV_STATE_FEATURE_NAMES]


def loitering_impact_state_value(case: dict) -> dict[str, float]:
    """Return the canonical six-value loitering-impact learner state.

    The contract intentionally exposes only the selected versioned munition
    profile and the three sampled approach variables. Collision distance,
    contact point/time, target topology and teacher physics remain audit-only
    data, rather than post-geometry leakage into the model input.
    """

    munition = case.get("munition")
    encounter = case.get("encounter")
    if not isinstance(munition, dict):
        raise ValueError("case is missing munition")
    if not isinstance(encounter, dict):
        raise ValueError("case is missing encounter")
    profile_id = munition.get("profile_id")
    if profile_id not in LOITERING_IMPACT_PROFILE_IDS:
        raise ValueError(
            "munition.profile_id must be one of the three versioned loitering profiles"
        )
    origin = _finite_vector(
        encounter.get("trajectory_origin_m"),
        length=3,
        name="encounter.trajectory_origin_m",
    )
    velocity = _finite_vector(
        encounter.get("velocity_target_mps"),
        length=3,
        name="encounter.velocity_target_mps",
    )
    if velocity[0] <= 0.0:
        raise ValueError("encounter.velocity_target_mps[0] must be positive")
    values = [
        *(1.0 if profile_id == name else 0.0 for name in LOITERING_IMPACT_PROFILE_IDS),
        origin[1],
        origin[2],
        velocity[0],
    ]
    return dict(
        zip(LOITERING_IMPACT_STATE_FEATURE_NAMES, values, strict=True)
    )


def loitering_impact_state_vector(case: dict) -> list[float]:
    state = loitering_impact_state_value(case)
    return [state[name] for name in LOITERING_IMPACT_STATE_FEATURE_NAMES]


def loitering_impact_contact_pose_state_value(case: dict) -> dict[str, float]:
    """Extract the pre-impact pose through a canonical quaternion intermediate.

    Direction and speed are represented by the three target-frame velocity
    components. The versioned archive and identity immediately convert attitude
    to rotation-6D, so this quaternion intermediate is never learner-visible.
    """

    munition = case.get("munition")
    encounter = case.get("encounter")
    if not isinstance(munition, dict):
        raise ValueError("case is missing munition")
    if not isinstance(encounter, dict):
        raise ValueError("case is missing encounter")
    profile_id = munition.get("profile_id")
    if profile_id not in LOITERING_IMPACT_PROFILE_IDS:
        raise ValueError(
            "munition.profile_id must be one of the three versioned loitering profiles"
        )
    origin = _finite_vector(
        encounter.get("trajectory_origin_m"),
        length=3,
        name="encounter.trajectory_origin_m",
    )
    velocity = _finite_vector(
        encounter.get("velocity_target_mps"),
        length=3,
        name="encounter.velocity_target_mps",
    )
    speed = math.sqrt(sum(item * item for item in velocity))
    if speed <= 0.0:
        raise ValueError("encounter.velocity_target_mps must be nonzero")
    orientation = _canonical_quaternion_xyzw(
        encounter.get("body_to_target_orientation_xyzw")
    )
    values = [
        *(1.0 if profile_id == name else 0.0 for name in LOITERING_IMPACT_PROFILE_IDS),
        *origin,
        *velocity,
        *orientation,
    ]
    return dict(
        zip(
            LOITERING_IMPACT_CONTACT_POSE_STATE_FEATURE_NAMES,
            values,
            strict=True,
        )
    )


def loitering_impact_contact_pose_state_vector(case: dict) -> list[float]:
    state = loitering_impact_contact_pose_state_value(case)
    return [
        state[name] for name in LOITERING_IMPACT_CONTACT_POSE_STATE_FEATURE_NAMES
    ]


def loitering_impact_contact_pose_learning_value(case: dict) -> dict[str, float]:
    """Return profile3 + origin3 + velocity3 + rotation-6D6 identity."""

    raw = loitering_impact_contact_pose_state_value(case)
    prefix = [
        raw[name]
        for name in LOITERING_IMPACT_CONTACT_POSE_STATE_FEATURE_NAMES[:9]
    ]
    x, y, z, w = (
        raw[name]
        for name in LOITERING_IMPACT_CONTACT_POSE_QUATERNION_FEATURE_NAMES
    )
    rotation_6d = [
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + z * w),
        2.0 * (x * z - y * w),
        2.0 * (x * y - z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z + x * w),
    ]
    return dict(
        zip(
            LOITERING_IMPACT_CONTACT_POSE_TRAINING_FEATURE_NAMES,
            [*prefix, *rotation_6d],
            strict=True,
        )
    )


def loitering_impact_contact_pose_training_vector(case: dict) -> list[float]:
    state = loitering_impact_contact_pose_learning_value(case)
    return [
        state[name]
        for name in LOITERING_IMPACT_CONTACT_POSE_TRAINING_FEATURE_NAMES
    ]


def fixed_scenario_value(case: dict) -> dict:
    """Return case content that must be frozen outside the learner tensor."""

    # Sampling metadata embeds the complete split registry in every case. Drop
    # it before the semantic deep copy so a B-case plan does not copy an O(B)
    # registry B times merely to prove that the teacher scenario is fixed.
    identity_input = {
        key: child
        for key, child in case.items()
        if key not in {"case_id", "metadata"}
    }
    value = semantic_case_value(identity_input)
    encounter = value.get("encounter")
    if isinstance(encounter, dict):
        for name in (
            "detonation_position_m",
            "velocity_target_mps",
            "body_to_target_orientation_xyzw",
        ):
            encounter.pop(name, None)
    analysis = value.get("analysis")
    if isinstance(analysis, dict):
        for name in (
            "aleatory_seed",
            "epistemic_sample_id",
            "replicates",
            "adaptive_ci_half_width",
            "adaptive_physical_ci",
            "capture_fragment_audit",
            "capture_replicate_trace",
        ):
            analysis.pop(name, None)
    return value


def fixed_scenario_fingerprint(case: dict) -> str:
    return content_hash(
        {
            "version": FIXED_SCENARIO_IDENTITY_VERSION,
            "scenario": fixed_scenario_value(case),
        }
    )


def loitering_impact_fixed_scenario_value(case: dict) -> dict:
    """Return the invariant scenario outside the six-value impact state.

    Impact resolution derives detonation/contact fields from the sampled
    trajectory, and applying a versioned profile derives the complete munition
    and warhead documents from ``profile_id``. Those deterministic descendants
    are therefore excluded together with the learner coordinates. Unsampled
    trajectory x and velocity y/z components remain bound into the identity.
    """

    identity_input = {
        key: child
        for key, child in case.items()
        if key not in {"case_id", "metadata", "munition", "warhead"}
    }
    value = semantic_case_value(identity_input)
    encounter = value.get("encounter")
    if isinstance(encounter, dict):
        origin = encounter.get("trajectory_origin_m")
        if isinstance(origin, list) and len(origin) == 3:
            encounter["trajectory_origin_m"] = [
                origin[0],
                "<model-input-y>",
                "<model-input-z>",
            ]
        velocity = encounter.get("velocity_target_mps")
        if isinstance(velocity, list) and len(velocity) == 3:
            encounter["velocity_target_mps"] = [
                "<model-input-speed>",
                velocity[1],
                velocity[2],
            ]
        for name in (
            "detonation_position_m",
            "detonation_time_s",
            "collision_distance_m",
            "collision_armor_id",
            "collision_normal_target_xyz",
        ):
            encounter.pop(name, None)
    analysis = value.get("analysis")
    if isinstance(analysis, dict):
        for name in (
            "aleatory_seed",
            "epistemic_sample_id",
            "replicates",
            "adaptive_ci_half_width",
            "adaptive_physical_ci",
            "capture_fragment_audit",
            "capture_replicate_trace",
        ):
            analysis.pop(name, None)
    return value


def loitering_impact_fixed_scenario_fingerprint(case: dict) -> str:
    return content_hash(
        {
            "version": LOITERING_IMPACT_FIXED_SCENARIO_IDENTITY_VERSION,
            "scenario": loitering_impact_fixed_scenario_value(case),
        }
    )


def loitering_impact_contact_pose_fixed_scenario_value(case: dict) -> dict:
    """Return scenario state invariant to the full sampled contact pose."""

    identity_input = {
        key: child
        for key, child in case.items()
        if key not in {"case_id", "metadata", "munition", "warhead"}
    }
    value = semantic_case_value(identity_input)
    encounter = value.get("encounter")
    if isinstance(encounter, dict):
        for name in (
            "trajectory_origin_m",
            "velocity_target_mps",
            "body_to_target_orientation_xyzw",
            "detonation_position_m",
            "detonation_time_s",
            "collision_distance_m",
            "collision_armor_id",
            "collision_normal_target_xyz",
        ):
            encounter.pop(name, None)
    analysis = value.get("analysis")
    if isinstance(analysis, dict):
        for name in (
            "aleatory_seed",
            "epistemic_sample_id",
            "replicates",
            "adaptive_ci_half_width",
            "adaptive_physical_ci",
            "capture_fragment_audit",
            "capture_replicate_trace",
        ):
            analysis.pop(name, None)
    return value


def loitering_impact_contact_pose_fixed_scenario_fingerprint(case: dict) -> str:
    return content_hash(
        {
            "version": (
                LOITERING_IMPACT_CONTACT_POSE_FIXED_SCENARIO_IDENTITY_VERSION
            ),
            "scenario": loitering_impact_contact_pose_fixed_scenario_value(case),
        }
    )


def loitering_impact_learning_input_fingerprint(case: dict) -> str:
    """Compatibility entry point for the loitering-impact input identity."""

    return LOITERING_IMPACT_MODEL_INPUT_DEFINITION.learning_input_fingerprint(
        case
    )


def loitering_impact_contact_pose_learning_input_fingerprint(case: dict) -> str:
    """Compatibility entry point for the contact-pose input identity."""

    return (
        LOITERING_IMPACT_CONTACT_POSE_MODEL_INPUT_DEFINITION
        .learning_input_fingerprint(case)
    )


def legacy_learning_input_fingerprint(case: dict) -> str:
    """Historical full-declared-input identity for pre-UAV contracts."""

    identity_input = {
        key: child
        for key, child in case.items()
        if key not in {"case_id", "metadata"}
    }
    value = semantic_case_value(identity_input)
    analysis = value.get("analysis")
    if isinstance(analysis, dict):
        for name in (
            "aleatory_seed",
            "epistemic_sample_id",
            "replicates",
            "adaptive_ci_half_width",
            "adaptive_physical_ci",
        ):
            analysis.pop(name, None)
    return content_hash({"version": "learning-input-json@1.0.0", "input": value})


def _model_input_definition_for_case(
    case: dict,
) -> ModelInputDefinition | None:
    """Resolve the current contract without changing historical precedence."""

    sampling = case.get("metadata", {}).get("sampling", {})
    if isinstance(sampling, dict) and sampling:
        sampling_spec = sampling.get("sampling_spec_document")
        if not isinstance(sampling_spec, dict) or sampling_spec.get(
            "schema_version"
        ) not in {"1.2.0", "1.3.0", "1.4.0"}:
            return None
        model_input_contract = sampling_spec.get("model_input_contract")
        try:
            return model_input_definition_for_sampling_contract(
                model_input_contract
            )
        except TypeError:
            # JSON-like historical cases could carry a malformed, unhashable
            # contract value. They previously fell through to legacy identity.
            return None
    elif isinstance(case.get("munition"), dict) and (
        case.get("encounter") or {}
    ).get("detonation_trigger") == "first_armor_contact":
        return LOITERING_IMPACT_MODEL_INPUT_DEFINITION
    return UAV_MODEL_INPUT_DEFINITION


def learning_input_fingerprint(case: dict) -> str:
    """Hash only the declared canonical model state, not teacher inputs.

    This remains a conservative duplicate/leakage guard rather than evidence of
    scientific independence. The fixed scenario is bound separately through
    :func:`fixed_scenario_fingerprint`.
    """

    definition = _model_input_definition_for_case(case)
    if definition is None:
        return legacy_learning_input_fingerprint(case)
    try:
        return definition.learning_input_fingerprint(case)
    except (KeyError, TypeError, ValueError):
        # Historical writer/unit-test cases can intentionally use a reduced
        # synthetic schema with no encounter object. They are not eligible for
        # tensor contract 1.8, but retain their original conservative identity
        # instead of being made unreadable by the UAV contract migration.
        return legacy_learning_input_fingerprint(case)


# Explicit adapters keep extraction/scenario behavior local while allowing the
# declarative definitions above to exist before their implementation functions
# are defined.  Adding a third learner contract changes this module, not every
# writer/loader caller.
_MODEL_INPUT_VECTOR_ADAPTERS = {
    UAV_STATE_CONTRACT_VERSION: uav_state_vector,
    LOITERING_IMPACT_STATE_CONTRACT_VERSION: loitering_impact_state_vector,
    LOITERING_IMPACT_CONTACT_POSE_STATE_CONTRACT_VERSION: (
        loitering_impact_contact_pose_training_vector
    ),
}
_MODEL_INPUT_SCENARIO_ADAPTERS = {
    UAV_STATE_CONTRACT_VERSION: fixed_scenario_fingerprint,
    LOITERING_IMPACT_STATE_CONTRACT_VERSION: (
        loitering_impact_fixed_scenario_fingerprint
    ),
    LOITERING_IMPACT_CONTACT_POSE_STATE_CONTRACT_VERSION: (
        loitering_impact_contact_pose_fixed_scenario_fingerprint
    ),
}
_MODEL_INPUT_VALUE_ADAPTERS = {
    UAV_STATE_CONTRACT_VERSION: uav_state_value,
    LOITERING_IMPACT_STATE_CONTRACT_VERSION: loitering_impact_state_value,
    LOITERING_IMPACT_CONTACT_POSE_STATE_CONTRACT_VERSION: (
        loitering_impact_contact_pose_learning_value
    ),
}
