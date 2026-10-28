"""Reference-design contracts for outcome-blind dataset case generation.

Version 1.0 remains the compact, independent-variable pilot contract.  Version
1.1 adds explicit design roles without treating an engineering design as an
operational distribution or as evidence of physical validity. Version 1.3
binds S-CONTACT admission. Version 1.4 adds a coupled target-facing contact
manifold and a learner-visible direction/attitude contract.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from .content_contract import validate_json_schema
from .errors import InputValidationError
from .utils import content_hash


SAMPLING_SPEC_SCHEMA_VERSION = "1.1.0"
UAV_SAMPLING_SPEC_SCHEMA_VERSION = "1.2.0"
CONTACT_CONDITIONED_SAMPLING_SPEC_SCHEMA_VERSION = "1.3.0"
CONTACT_MANIFOLD_SAMPLING_SPEC_SCHEMA_VERSION = "1.4.0"
LEGACY_SAMPLING_SPEC_SCHEMA_VERSION = "1.0.0"
_EXTENDED_FIELDS = frozenset(
    {
        "feature_roles",
        "variable_blocks",
        "geometry_strata",
        "stream_roles",
        "round_budgets",
        "replicate_policy",
        "key_projections",
        "proxy_scope",
        "legal_sampling",
    }
)
_ROLE_CONTRACTS = {
    "reference_train": ("reference", "train", True),
    "reference_validation": ("reference", "validation", False),
    "reference_test": ("reference", "test", False),
    "geometry_stratified_train": ("geometry_stratified", "train", True),
    "rare_stress_test": ("stress", "test", False),
}
_LATENT_DIMENSIONS = {
    "uniform": 1,
    "log_uniform": 1,
    "integer_uniform": 1,
    "choice": 1,
    "unit_sphere_direction": 2,
    "sphere_shell": 2,
    "sphere_volume": 3,
    "haar_quaternion": 3,
}


class SamplingSpec:
    """Validated design-distribution, budget, seed and quality contract."""

    def __init__(self, data: Mapping[str, Any]):
        document = json.loads(json.dumps(data, ensure_ascii=False, allow_nan=False))
        validate_json_schema(
            document,
            "sampling_spec.schema.json",
            failure_message="sampling specification failed schema validation",
            max_errors=30,
            compatibility="legacy_messages",
        )
        version = document["schema_version"]
        extended = _EXTENDED_FIELDS.intersection(document)
        model_input_contract = document.get("model_input_contract")
        if version == LEGACY_SAMPLING_SPEC_SCHEMA_VERSION and (
            extended or model_input_contract is not None
        ):
            raise InputValidationError(
                "sampling contract 1.1 fields require schema_version 1.1.0",
                context={"fields": sorted(extended)},
            )
        if version in {
            SAMPLING_SPEC_SCHEMA_VERSION,
            UAV_SAMPLING_SPEC_SCHEMA_VERSION,
            CONTACT_CONDITIONED_SAMPLING_SPEC_SCHEMA_VERSION,
            CONTACT_MANIFOLD_SAMPLING_SPEC_SCHEMA_VERSION,
        }:
            missing = sorted(_EXTENDED_FIELDS - set(document))
            if missing:
                raise InputValidationError(
                    "sampling contract 1.1 is missing required design fields",
                    context={"missing": missing},
                )
        names = [item["name"] for item in document["reference_distribution"]["variables"]]
        paths = [item["path"] for item in document["reference_distribution"]["variables"]]
        if len(names) != len(set(names)):
            raise InputValidationError("sampling variable names must be unique")
        if len(paths) != len(set(paths)):
            raise InputValidationError("sampling variable paths must be unique")
        for variable in document["reference_distribution"]["variables"]:
            distribution = variable["distribution"]
            if distribution not in _LATENT_DIMENSIONS:
                raise InputValidationError(
                    "unsupported sampling distribution",
                    context={"variable": variable["name"], "distribution": distribution},
                )
            if distribution == "log_uniform" and variable["low"] <= 0.0:
                raise InputValidationError(
                    "log_uniform lower bound must be positive",
                    context={"variable": variable["name"]},
                )
            if distribution in {"uniform", "log_uniform", "integer_uniform"}:
                if not variable["low"] < variable["high"]:
                    raise InputValidationError(
                        "sampling lower bound must be smaller than upper bound",
                        context={"variable": variable["name"]},
                    )
            if distribution == "integer_uniform" and (
                not float(variable["low"]).is_integer()
                or not float(variable["high"]).is_integer()
            ):
                raise InputValidationError(
                    "integer_uniform bounds must be integers",
                    context={"variable": variable["name"]},
                )
            if distribution == "sphere_volume":
                if variable.get("radius_min", 0.0) < 0.0 or not (
                    variable.get("radius_min", 0.0) < variable["radius_max"]
                ):
                    raise InputValidationError(
                        "sphere_volume requires 0 <= radius_min < radius_max",
                        context={"variable": variable["name"]},
                    )
        if version == SAMPLING_SPEC_SCHEMA_VERSION and model_input_contract is not None:
            raise InputValidationError(
                "model_input_contract requires SamplingSpec schema_version 1.2.0"
            )
        if version in {
            UAV_SAMPLING_SPEC_SCHEMA_VERSION,
            CONTACT_CONDITIONED_SAMPLING_SPEC_SCHEMA_VERSION,
            CONTACT_MANIFOLD_SAMPLING_SPEC_SCHEMA_VERSION,
        }:
            from .learning_input_identity import (
                LOITERING_IMPACT_CONTACT_POSE_SAMPLING_MODEL_INPUT_CONTRACT,
                LOITERING_IMPACT_SAMPLING_MODEL_INPUT_CONTRACT,
                UAV_SAMPLING_MODEL_INPUT_CONTRACT,
            )

            if model_input_contract not in {
                UAV_SAMPLING_MODEL_INPUT_CONTRACT,
                LOITERING_IMPACT_SAMPLING_MODEL_INPUT_CONTRACT,
                LOITERING_IMPACT_CONTACT_POSE_SAMPLING_MODEL_INPUT_CONTRACT,
            }:
                raise InputValidationError(
                    f"SamplingSpec {version} requires a supported versioned model-input contract"
                )
            if (
                model_input_contract
                == LOITERING_IMPACT_CONTACT_POSE_SAMPLING_MODEL_INPUT_CONTRACT
                and version != CONTACT_MANIFOLD_SAMPLING_SPEC_SCHEMA_VERSION
            ):
                raise InputValidationError(
                    "the loitering-impact contact-pose model-input contract "
                    "requires SamplingSpec schema_version 1.4.0"
                )
            if (
                version == CONTACT_CONDITIONED_SAMPLING_SPEC_SCHEMA_VERSION
                and model_input_contract
                != LOITERING_IMPACT_SAMPLING_MODEL_INPUT_CONTRACT
            ):
                raise InputValidationError(
                    "SamplingSpec 1.3.0 contact conditioning is limited to the "
                    "loitering-impact model-input contract"
                )
            if (
                version == CONTACT_MANIFOLD_SAMPLING_SPEC_SCHEMA_VERSION
                and model_input_contract
                != LOITERING_IMPACT_CONTACT_POSE_SAMPLING_MODEL_INPUT_CONTRACT
            ):
                raise InputValidationError(
                    "SamplingSpec 1.4.0 contact-manifold sampling requires the "
                    "loitering-impact contact-pose model-input contract"
                )
        conditioning = document.get("contact_conditioning")
        if (
            version
            not in {
                CONTACT_CONDITIONED_SAMPLING_SPEC_SCHEMA_VERSION,
                CONTACT_MANIFOLD_SAMPLING_SPEC_SCHEMA_VERSION,
            }
            and conditioning is not None
        ):
            raise InputValidationError(
                "contact_conditioning requires SamplingSpec schema_version 1.3.0 or 1.4.0"
            )
        if version in {
            CONTACT_CONDITIONED_SAMPLING_SPEC_SCHEMA_VERSION,
            CONTACT_MANIFOLD_SAMPLING_SPEC_SCHEMA_VERSION,
        }:
            from .contact_contract import (
                S_CONTACT_CONTRACT_HASH,
                S_CONTACT_CONTRACT_ID,
                S_CONTACT_ESTIMAND,
                S_CONTACT_MODE,
                validate_outcome_blind_selection,
            )

            if not isinstance(conditioning, dict):
                raise InputValidationError(
                    f"SamplingSpec {version} requires an S-CONTACT conditioning object"
                )
            expected_identity = {
                "contract_id": S_CONTACT_CONTRACT_ID,
                "mode": S_CONTACT_MODE,
                "estimand": S_CONTACT_ESTIMAND,
            }
            observed_identity = {
                key: conditioning.get(key) for key in expected_identity
            }
            if observed_identity != expected_identity:
                raise InputValidationError(
                    f"SamplingSpec {version} contact conditioning identity is unsupported",
                    context={
                        "expected": expected_identity,
                        "observed": observed_identity,
                    },
                )
            declared_contract_hash = conditioning.get("contract_hash")
            if declared_contract_hash not in {None, S_CONTACT_CONTRACT_HASH}:
                raise InputValidationError(
                    f"SamplingSpec {version} S-CONTACT contract hash is inconsistent",
                    context={
                        "expected": S_CONTACT_CONTRACT_HASH,
                        "observed": declared_contract_hash,
                    },
                )
            selection = validate_outcome_blind_selection(
                conditioning["selection_declaration"]
            )
            declared_selection_hash = conditioning.get(
                "selection_declaration_hash"
            )
            if declared_selection_hash not in {None, selection["content_hash"]}:
                raise InputValidationError(
                    f"SamplingSpec {version} selection declaration hash is inconsistent",
                    context={
                        "expected": selection["content_hash"],
                        "observed": declared_selection_hash,
                    },
                )
            # Computed hashes become part of SamplingSpec.hash even when callers
            # supplied only the complete source declarations.
            conditioning["contract_hash"] = S_CONTACT_CONTRACT_HASH
            conditioning["selection_declaration_hash"] = selection["content_hash"]
        manifold = document.get("contact_manifold")
        if (
            version != CONTACT_MANIFOLD_SAMPLING_SPEC_SCHEMA_VERSION
            and manifold is not None
        ):
            raise InputValidationError(
                "contact_manifold requires SamplingSpec schema_version 1.4.0"
            )
        if version == CONTACT_MANIFOLD_SAMPLING_SPEC_SCHEMA_VERSION:
            from .contact_manifold_sampling import validate_contact_manifold

            if document["reference_distribution"]["interpretation"] != "design_distribution":
                raise InputValidationError(
                    "contact-manifold approach weights are a design distribution, not operational prevalence"
                )
            document["contact_manifold"] = validate_contact_manifold(manifold)
        if version in {
            SAMPLING_SPEC_SCHEMA_VERSION,
            UAV_SAMPLING_SPEC_SCHEMA_VERSION,
            CONTACT_CONDITIONED_SAMPLING_SPEC_SCHEMA_VERSION,
            CONTACT_MANIFOLD_SAMPLING_SPEC_SCHEMA_VERSION,
        }:
            self._validate_extended_contract(document, names, paths)
        seeds = document["seeds"]
        if len(set(seeds.values())) != len(seeds):
            raise InputValidationError(
                "design, selection, split, screening and formal seeds must be distinct"
            )
        positivity = float(document["quality_gates"]["positivity_min"])
        if not math.isfinite(positivity) or not 0.0 < positivity <= 1.0:
            raise InputValidationError("quality_gates.positivity_min must lie in (0, 1]")
        # Store one canonical document privately; public access never exposes
        # the nested object that determines candidate and resume identity.
        self._canonical_json = json.dumps(document, ensure_ascii=False, sort_keys=True, allow_nan=False)

    @staticmethod
    def _validate_extended_contract(
        document: dict[str, Any], variable_names: list[str], variable_paths: list[str]
    ) -> None:
        feature_roles = document["feature_roles"]
        feature_ids = [item["feature_id"] for item in feature_roles]
        feature_paths = [item["path"] for item in feature_roles]
        if len(feature_ids) != len(set(feature_ids)) or len(feature_paths) != len(set(feature_paths)):
            raise InputValidationError("feature role IDs and paths must be unique")
        tensor_feature_names = [
            name
            for feature in feature_roles
            for name in feature["tensor_feature_names"]
        ]
        if len(tensor_feature_names) != len(set(tensor_feature_names)):
            raise InputValidationError(
                "tensor feature names must be globally unique across feature roles"
            )
        if document["schema_version"] in {
            UAV_SAMPLING_SPEC_SCHEMA_VERSION,
            CONTACT_CONDITIONED_SAMPLING_SPEC_SCHEMA_VERSION,
            CONTACT_MANIFOLD_SAMPLING_SPEC_SCHEMA_VERSION,
        }:
            from .learning_input_identity import (
                LOITERING_IMPACT_CONTACT_POSE_SAMPLING_MODEL_INPUT_CONTRACT,
                LOITERING_IMPACT_CONTACT_POSE_TRAINING_FEATURE_NAMES,
                LOITERING_IMPACT_SAMPLING_MODEL_INPUT_CONTRACT,
                LOITERING_IMPACT_TRAINING_FEATURE_NAMES,
                UAV_TRAINING_FEATURE_NAMES,
            )

            loitering_contract = (
                document["model_input_contract"]
                == LOITERING_IMPACT_SAMPLING_MODEL_INPUT_CONTRACT
            )
            contact_pose_contract = (
                document["model_input_contract"]
                == LOITERING_IMPACT_CONTACT_POSE_SAMPLING_MODEL_INPUT_CONTRACT
            )
            if loitering_contract:
                expected_names = LOITERING_IMPACT_TRAINING_FEATURE_NAMES
                description = "profile one-hot3, trajectory origin y/z and impact speed"
            elif contact_pose_contract:
                expected_names = LOITERING_IMPACT_CONTACT_POSE_TRAINING_FEATURE_NAMES
                description = (
                    "profile one-hot3, trajectory origin3, velocity3 and "
                    "body-to-target attitude rotation6d6"
                )
            else:
                expected_names = UAV_TRAINING_FEATURE_NAMES
                description = "position3, velocity3 and attitude rotation6d6"
            if tuple(tensor_feature_names) != tuple(expected_names):
                raise InputValidationError(
                    "SamplingSpec versioned feature roles must exactly declare "
                    f"{description}",
                    context={
                        "expected": list(expected_names),
                        "observed": tensor_feature_names,
                    },
                )
            if contact_pose_contract:
                expected_paths = {
                    **{
                        name: {"/munition/profile_id"}
                        for name in expected_names[:3]
                    },
                    **{
                        name: {
                            "/encounter/trajectory_origin_m",
                            f"/encounter/trajectory_origin_m/{axis}",
                        }
                        for axis, name in enumerate(expected_names[3:6])
                    },
                    **{
                        name: {
                            "/encounter/velocity_target_mps",
                            f"/encounter/velocity_target_mps/{axis}",
                        }
                        for axis, name in enumerate(expected_names[6:9])
                    },
                    **{
                        name: {"/encounter/body_to_target_orientation_xyzw"}
                        for name in expected_names[9:]
                    },
                }
            elif loitering_contract:
                expected_paths = {
                    **{
                        name: {"/munition/profile_id"}
                        for name in expected_names[:3]
                    },
                    expected_names[3]: {"/encounter/trajectory_origin_m/1"},
                    expected_names[4]: {"/encounter/trajectory_origin_m/2"},
                    expected_names[5]: {"/encounter/velocity_target_mps/0"},
                }
            else:
                expected_paths = {
                    **{
                        name: {"/encounter/detonation_position_m"}
                        for name in expected_names[:3]
                    },
                    **{
                        name: {
                            "/encounter/velocity_target_mps",
                            f"/encounter/velocity_target_mps/{axis}",
                        }
                        for axis, name in enumerate(expected_names[3:6])
                    },
                    **{
                        name: {"/encounter/body_to_target_orientation_xyzw"}
                        for name in expected_names[6:]
                    },
                }
            invalid_features = [
                item["feature_id"]
                for item in feature_roles
                if any(
                    item["path"] not in expected_paths[name]
                    for name in item["tensor_feature_names"]
                )
                or item["availability_stage"] != "raw_input"
                or item["deployment_available"] is not True
                or item["role"] not in {"learnable_variable", "fixed_context"}
            ]
            if invalid_features:
                state_name = (
                    "raw loitering-impact contact-pose state"
                    if contact_pose_contract
                    else (
                        "raw loitering-impact state"
                        if loitering_contract
                        else "raw UAV state"
                    )
                )
                raise InputValidationError(
                    "SamplingSpec versioned model inputs must come only from "
                    f"deployment-available {state_name}",
                    context={"feature_ids": invalid_features},
                )
        roles_by_path = {item["path"]: item["role"] for item in feature_roles}
        missing_learnable = [
            path for path in variable_paths if roles_by_path.get(path) != "learnable_variable"
        ]
        if missing_learnable:
            raise InputValidationError(
                "every sampled variable path must be declared learnable_variable",
                context={"paths": missing_learnable},
            )
        if (
            document["schema_version"]
            == CONTACT_MANIFOLD_SAMPLING_SPEC_SCHEMA_VERSION
        ):
            manifold_paths = {
                "/encounter/trajectory_origin_m",
                "/encounter/velocity_target_mps",
                "/encounter/body_to_target_orientation_xyzw",
            }
            missing_manifold_roles = sorted(
                path
                for path in manifold_paths
                if roles_by_path.get(path) != "learnable_variable"
            )
            if missing_manifold_roles:
                raise InputValidationError(
                    "every contact-manifold output must be declared learnable_variable",
                    context={"paths": missing_manifold_roles},
                )
            conflicting = sorted(
                path
                for path in variable_paths
                if any(
                    path == manifold_path or path.startswith(manifold_path + "/")
                    for manifold_path in manifold_paths
                )
            )
            if conflicting:
                raise InputValidationError(
                    "independent variables cannot overwrite contact-manifold outputs",
                    context={"paths": conflicting},
                )
        for feature in feature_roles:
            if (
                feature["role"] == "derived_input"
                and feature["availability_stage"] != "prelabel_geometry"
            ):
                raise InputValidationError(
                    "derived_input features must be available before labels",
                    context={"feature_id": feature["feature_id"]},
                )

        variables = {
            item["name"]: item for item in document["reference_distribution"]["variables"]
        }
        block_ids: set[str] = set()
        joint_members: set[str] = set()
        conditional_members: set[str] = set()
        conditional_conditions: list[tuple[str, str]] = []
        for block in document["variable_blocks"]:
            block_id = block["block_id"]
            if block_id in block_ids:
                raise InputValidationError("variable block IDs must be unique")
            block_ids.add(block_id)
            unknown = sorted(set(block["variable_names"]) - set(variable_names))
            if unknown:
                raise InputValidationError(
                    "variable block references unknown sampling variables",
                    context={"block_id": block_id, "variables": unknown},
                )
            members = set(block["variable_names"])
            if block["kind"] == "joint":
                overlap = sorted(joint_members & members)
                if overlap:
                    raise InputValidationError(
                        "sampling variables cannot belong to multiple joint blocks",
                        context={"variables": overlap},
                    )
                joint_members.update(members)
                unsupported = sorted(
                    name
                    for name in members
                    if _LATENT_DIMENSIONS[variables[name]["distribution"]] != 1
                )
                if unsupported:
                    raise InputValidationError(
                        "shared_quantile joint blocks only support scalar distributions",
                        context={"block_id": block_id, "variables": unsupported},
                    )
            else:
                overlap = sorted(conditional_members & members)
                if overlap:
                    raise InputValidationError(
                        "sampling variables cannot belong to multiple conditional blocks",
                        context={"variables": overlap},
                    )
                conditional_members.update(members)
                member_paths = {variables[name]["path"] for name in members}
                condition_path = block["condition"]["path"]
                conditional_conditions.append((block_id, condition_path))
                if condition_path in member_paths:
                    raise InputValidationError(
                        "a conditional block cannot condition on a value it disables",
                        context={"block_id": block_id, "path": condition_path},
                    )
        conditional_paths = {variables[name]["path"] for name in conditional_members}
        for block_id, condition_path in conditional_conditions:
            if condition_path in conditional_paths:
                raise InputValidationError(
                    "conditional blocks cannot depend on conditionally disabled variables",
                    context={"block_id": block_id, "path": condition_path},
                )

        strata = document["geometry_strata"]
        stratum_ids = [item["stratum_id"] for item in strata]
        if len(stratum_ids) != len(set(stratum_ids)) or "p0" in stratum_ids:
            raise InputValidationError("geometry stratum IDs must be unique and cannot be p0")
        for stratum in strata:
            for clause in stratum["predicate"]["all"]:
                if clause["source"] not in {"case_input", "prelabel_screening"}:
                    raise InputValidationError(
                        "unsupported pre-label stratum predicate source",
                        context={"stratum_id": stratum["stratum_id"], "source": clause["source"]},
                    )
                if clause["source"] == "prelabel_screening":
                    from .prelabel_geometry import PRELABEL_GEOMETRY_FIELDS

                    path = clause["path"]
                    if path.count("/") != 1 or path[1:] not in PRELABEL_GEOMETRY_FIELDS:
                        raise InputValidationError(
                            "prelabel_screening predicate references an unknown geometry feature",
                            context={"stratum_id": stratum["stratum_id"], "path": path},
                        )

        declared_roles: dict[str, dict[str, Any]] = {}
        for item in document["stream_roles"]:
            role = item["role"]
            if role in declared_roles:
                raise InputValidationError("stream role declarations must be unique")
            expected = _ROLE_CONTRACTS[role]
            observed = (item["stream"], item["split"], item["train_eligible"])
            if observed != expected:
                raise InputValidationError(
                    "stream role violates its fixed split and training eligibility",
                    context={"role": role, "expected": expected, "observed": observed},
                )
            declared_roles[role] = item

        rounds: set[int] = set()
        totals = {key: 0 for key in ("train_pilot", "validation", "test", "rare_stress_test")}
        allocation_keys: set[tuple[int, str, str]] = set()
        largest_allocation = 0
        for round_item in document["round_budgets"]:
            round_index = round_item["round"]
            if round_index in rounds:
                raise InputValidationError("round budget indices must be unique")
            rounds.add(round_index)
            for allocation in round_item["allocations"]:
                role = allocation["role"]
                if role not in declared_roles:
                    raise InputValidationError(
                        "round budget references an undeclared stream role",
                        context={"role": role},
                    )
                stratum_id = allocation["stratum_id"]
                key = (round_index, role, stratum_id)
                if key in allocation_keys:
                    raise InputValidationError("round role/stratum allocations must be unique")
                allocation_keys.add(key)
                if role.startswith("reference_"):
                    if stratum_id != "p0" or allocation["proposal_id"] != document["reference_distribution"]["distribution_id"]:
                        raise InputValidationError(
                            "reference roles must use the p0 stratum and reference distribution proposal"
                        )
                else:
                    if stratum_id not in stratum_ids:
                        raise InputValidationError(
                            "geometry/stress allocations require a declared pre-label stratum",
                            context={"role": role, "stratum_id": stratum_id},
                        )
                    if allocation["proposal_id"] == document["reference_distribution"]["distribution_id"]:
                        raise InputValidationError(
                            "geometry/stress allocations require an explicit non-p0 proposal identity"
                        )
                count = int(allocation["count"])
                largest_allocation = max(largest_allocation, count)
                if role in {"reference_train", "geometry_stratified_train"}:
                    totals["train_pilot"] += count
                elif role == "reference_validation":
                    totals["validation"] += count
                elif role == "reference_test":
                    totals["test"] += count
                else:
                    totals["rare_stress_test"] += count
        if 0 not in rounds:
            raise InputValidationError("round budgets must include round zero")
        if rounds != {0}:
            raise InputValidationError(
                "reference-plan round budgets must contain round zero only; "
                "adaptive rounds are frozen by the multiround checkpoint workflow"
            )
        if totals != {key: int(value) for key, value in document["budgets"].items()}:
            raise InputValidationError(
                "round budgets do not reconcile with aggregate budgets",
                context={"round_totals": totals, "budgets": document["budgets"]},
            )
        if document["legal_sampling"]["max_attempts_per_allocation"] < largest_allocation:
            raise InputValidationError(
                "legal sampling attempt limit cannot be smaller than an allocation",
                context={
                    "max_attempts_per_allocation": document["legal_sampling"]["max_attempts_per_allocation"],
                    "largest_allocation": largest_allocation,
                },
            )

        replicate_policy = document["replicate_policy"]
        if replicate_policy["default_replicates"] != document["formal_policy"]["replicates"]:
            raise InputValidationError(
                "replicate policy default must equal formal_policy.replicates"
            )
        override_keys: set[tuple[str, str | None]] = set()
        for override in replicate_policy["overrides"]:
            role = override["role"]
            stratum_id = override.get("stratum_id")
            if role not in declared_roles:
                raise InputValidationError("replicate override references an undeclared role")
            if stratum_id is not None and stratum_id not in set(stratum_ids) | {"p0"}:
                raise InputValidationError("replicate override references an undeclared stratum")
            key = (role, stratum_id)
            if key in override_keys:
                raise InputValidationError("replicate overrides must have unique role/stratum keys")
            override_keys.add(key)

        projection_ids: set[str] = set()
        known_feature_paths = set(feature_paths)
        for projection in document["key_projections"]:
            if projection["projection_id"] in projection_ids:
                raise InputValidationError("key projection IDs must be unique")
            projection_ids.add(projection["projection_id"])
            unknown_paths = sorted(set(projection["paths"]) - known_feature_paths)
            if unknown_paths:
                raise InputValidationError(
                    "key projections must reference declared feature-role paths",
                    context={"projection_id": projection["projection_id"], "paths": unknown_paths},
                )

        scope = document["proxy_scope"]
        expected_identity = (
            "fixed_target_hash" if scope["scope"] == "fixed_target" else "target_family_registry"
        )
        if scope["target_identity_policy"] != expected_identity:
            raise InputValidationError(
                "proxy scope and target identity policy disagree",
                context={"expected": expected_identity},
            )
        if document["schema_version"] in {
            UAV_SAMPLING_SPEC_SCHEMA_VERSION,
            CONTACT_CONDITIONED_SAMPLING_SPEC_SCHEMA_VERSION,
            CONTACT_MANIFOLD_SAMPLING_SPEC_SCHEMA_VERSION,
        } and (
            scope["scope"] != "fixed_target"
            or scope["geometry_boundary"] != "end_to_end"
        ):
            raise InputValidationError(
                "versioned model inputs require fixed_target/end_to_end proxy scope"
            )

    @property
    def data(self) -> dict[str, Any]:
        return json.loads(self._canonical_json)

    @property
    def hash(self) -> str:
        return content_hash(self.data)

    @classmethod
    def from_file(cls, path: str | Path) -> "SamplingSpec":
        value = Path(path)
        try:
            document = json.loads(value.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise InputValidationError(
                "unable to load sampling specification as UTF-8 JSON",
                context={"path": str(value)},
            ) from exc
        if not isinstance(document, dict):
            raise InputValidationError("sampling specification must be a JSON object")
        return cls(document)

    @property
    def spec_id(self) -> str:
        return str(self.data["sampling_spec_id"])

    @property
    def schema_version(self) -> str:
        return str(self.data["schema_version"])

    @property
    def is_extended(self) -> bool:
        return self.schema_version in {
            SAMPLING_SPEC_SCHEMA_VERSION,
            UAV_SAMPLING_SPEC_SCHEMA_VERSION,
            CONTACT_CONDITIONED_SAMPLING_SPEC_SCHEMA_VERSION,
            CONTACT_MANIFOLD_SAMPLING_SPEC_SCHEMA_VERSION,
        }

    @property
    def is_uav_model_input(self) -> bool:
        if self.schema_version != UAV_SAMPLING_SPEC_SCHEMA_VERSION:
            return False
        from .learning_input_identity import UAV_SAMPLING_MODEL_INPUT_CONTRACT

        return self.data.get("model_input_contract") == UAV_SAMPLING_MODEL_INPUT_CONTRACT

    @property
    def is_loitering_impact_model_input(self) -> bool:
        if self.schema_version not in {
            UAV_SAMPLING_SPEC_SCHEMA_VERSION,
            CONTACT_CONDITIONED_SAMPLING_SPEC_SCHEMA_VERSION,
            CONTACT_MANIFOLD_SAMPLING_SPEC_SCHEMA_VERSION,
        }:
            return False
        from .learning_input_identity import (
            LOITERING_IMPACT_CONTACT_POSE_SAMPLING_MODEL_INPUT_CONTRACT,
            LOITERING_IMPACT_SAMPLING_MODEL_INPUT_CONTRACT,
        )

        return self.data.get("model_input_contract") in {
            LOITERING_IMPACT_SAMPLING_MODEL_INPUT_CONTRACT,
            LOITERING_IMPACT_CONTACT_POSE_SAMPLING_MODEL_INPUT_CONTRACT,
        }

    @property
    def is_contact_pose_model_input(self) -> bool:
        if self.schema_version != CONTACT_MANIFOLD_SAMPLING_SPEC_SCHEMA_VERSION:
            return False
        from .learning_input_identity import (
            LOITERING_IMPACT_CONTACT_POSE_SAMPLING_MODEL_INPUT_CONTRACT,
        )

        return (
            self.data.get("model_input_contract")
            == LOITERING_IMPACT_CONTACT_POSE_SAMPLING_MODEL_INPUT_CONTRACT
        )

    @property
    def has_versioned_model_input(self) -> bool:
        return self.is_uav_model_input or self.is_loitering_impact_model_input

    @property
    def has_contact_conditioning(self) -> bool:
        return self.schema_version in {
            CONTACT_CONDITIONED_SAMPLING_SPEC_SCHEMA_VERSION,
            CONTACT_MANIFOLD_SAMPLING_SPEC_SCHEMA_VERSION,
        }

    @property
    def contact_conditioning(self) -> dict[str, Any] | None:
        value = self.data.get("contact_conditioning")
        return dict(value) if value is not None else None

    @property
    def contact_selection_hash(self) -> str | None:
        value = self.contact_conditioning
        return (
            None
            if value is None
            else str(value["selection_declaration_hash"])
        )

    @property
    def has_contact_manifold(self) -> bool:
        return (
            self.schema_version == CONTACT_MANIFOLD_SAMPLING_SPEC_SCHEMA_VERSION
        )

    @property
    def contact_manifold(self) -> dict[str, Any] | None:
        value = self.data.get("contact_manifold")
        return dict(value) if value is not None else None

    @property
    def distribution_id(self) -> str:
        return str(self.data["reference_distribution"]["distribution_id"])

    @property
    def variables(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.data["reference_distribution"]["variables"])

    @property
    def independent_latent_dimension(self) -> int:
        if not self.is_extended:
            return sum(_LATENT_DIMENSIONS[item["distribution"]] for item in self.variables)
        joint_members: set[str] = set()
        joint_dimensions = 0
        for block in self.variable_blocks:
            if block["kind"] == "joint":
                joint_members.update(block["variable_names"])
                joint_dimensions += 1
        return joint_dimensions + sum(
            _LATENT_DIMENSIONS[item["distribution"]]
            for item in self.variables
            if item["name"] not in joint_members
        )

    @property
    def contact_manifold_latent_dimension(self) -> int:
        if not self.has_contact_manifold:
            return 0
        from .contact_manifold_sampling import CONTACT_MANIFOLD_LATENT_DIMENSION

        return CONTACT_MANIFOLD_LATENT_DIMENSION

    @property
    def latent_dimension(self) -> int:
        return (
            self.independent_latent_dimension
            + self.contact_manifold_latent_dimension
        )

    @property
    def budgets(self) -> dict[str, int]:
        return {key: int(value) for key, value in self.data["budgets"].items()}

    @property
    def seeds(self) -> dict[str, int]:
        return {key: int(value) for key, value in self.data["seeds"].items()}

    @property
    def formal_replicates(self) -> int:
        return int(self.data["formal_policy"]["replicates"])

    @property
    def feature_roles(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.data.get("feature_roles", ()))

    @property
    def variable_blocks(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.data.get("variable_blocks", ()))

    @property
    def geometry_strata(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.data.get("geometry_strata", ()))

    @property
    def stream_roles(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.data.get("stream_roles", ()))

    @property
    def key_projections(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.data.get("key_projections", ()))

    @property
    def proxy_scope(self) -> dict[str, Any] | None:
        value = self.data.get("proxy_scope")
        return dict(value) if value is not None else None

    @property
    def legal_sampling(self) -> dict[str, Any]:
        return self.data.get(
            "legal_sampling",
            {"invalid_candidate_policy": "fail_closed", "max_attempts_per_allocation": 1},
        )

    @property
    def training_design_document(self) -> dict[str, Any]:
        """Return the parseable training-design boundary bound into each case."""
        if not self.is_extended:
            return {
                "schema_version": "1.0.0",
                "contract_mode": "legacy_independent",
                "feature_roles": [],
                "variable_blocks": [],
                "geometry_strata": [],
                "stream_roles": [],
                "round_budgets": [],
                "replicate_policy": {
                    "default_replicates": self.formal_replicates,
                    "outcome_adaptive": False,
                    "overrides": [],
                },
                "key_projections": [],
                "proxy_scope": None,
                "legal_sampling": self.legal_sampling,
            }
        data = self.data
        return {
            "schema_version": "1.0.0",
            "contract_mode": (
                "loitering_impact_contact_pose_1.4"
                if self.is_contact_pose_model_input
                else (
                    "uav_control_state_only_1.2"
                    if self.is_uav_model_input
                    else (
                        (
                            "loitering_impact_contact_conditioned_1.3"
                            if self.has_contact_conditioning
                            else "loitering_impact_state_only_1.2"
                        )
                        if self.is_loitering_impact_model_input
                        else "distribution_design_1.1"
                    )
                )
            ),
            **(
                {"model_input_contract": data["model_input_contract"]}
                if self.has_versioned_model_input
                else {}
            ),
            **(
                {"contact_conditioning": data["contact_conditioning"]}
                if self.has_contact_conditioning
                else {}
            ),
            **(
                {"contact_manifold": data["contact_manifold"]}
                if self.has_contact_manifold
                else {}
            ),
            **{key: data[key] for key in (
                "feature_roles",
                "variable_blocks",
                "geometry_strata",
                "stream_roles",
                "round_budgets",
                "replicate_policy",
                "key_projections",
                "proxy_scope",
                "legal_sampling",
            )},
        }

    @property
    def training_design_hash(self) -> str:
        return content_hash(self.training_design_document)

    def role_allocations(self) -> tuple[dict[str, Any], ...]:
        """Return normalized role allocations; legacy ordering stays byte-stable."""
        if not self.is_extended:
            budgets = self.budgets
            rows: list[dict[str, Any]] = []
            rows.extend(
                {"round": 0, "role": "reference_validation", "stream": "reference", "split": "validation", "stratum_id": "p0", "proposal_id": self.distribution_id}
                for _ in range(budgets["validation"])
            )
            rows.extend(
                {"round": 0, "role": "reference_test", "stream": "reference", "split": "test", "stratum_id": "p0", "proposal_id": self.distribution_id}
                for _ in range(budgets["test"])
            )
            rows.extend(
                {"round": 0, "role": "train_pilot", "stream": "reference", "split": "train", "stratum_id": "p0", "proposal_id": self.distribution_id}
                for _ in range(budgets["train_pilot"])
            )
            return tuple(rows)
        role_map = {item["role"]: item for item in self.stream_roles}
        priority = {
            "reference_validation": 0,
            "reference_test": 1,
            "rare_stress_test": 2,
            "reference_train": 3,
            "geometry_stratified_train": 4,
        }
        allocations: list[dict[str, Any]] = []
        for round_item in self.data["round_budgets"]:
            for item in round_item["allocations"]:
                contract = role_map[item["role"]]
                allocations.extend(
                    {
                        "round": int(round_item["round"]),
                        "role": item["role"],
                        "stream": contract["stream"],
                        "split": contract["split"],
                        "stratum_id": item["stratum_id"],
                        "proposal_id": item["proposal_id"],
                    }
                    for _ in range(int(item["count"]))
                )
        allocations.sort(key=lambda item: (item["round"], priority[item["role"]], item["stratum_id"]))
        return tuple(allocations)

    def replicates_for(self, role: str, stratum_id: str) -> int:
        if not self.is_extended:
            return self.formal_replicates
        policy = self.data["replicate_policy"]
        exact = None
        role_only = None
        for item in policy["overrides"]:
            if item["role"] != role:
                continue
            if item.get("stratum_id") == stratum_id:
                exact = int(item["replicates"])
            elif "stratum_id" not in item:
                role_only = int(item["replicates"])
        return exact or role_only or int(policy["default_replicates"])


def latent_dimensions(distribution: str) -> int:
    try:
        return _LATENT_DIMENSIONS[distribution]
    except KeyError as exc:
        raise InputValidationError(
            "unsupported sampling distribution", context={"distribution": distribution}
        ) from exc
