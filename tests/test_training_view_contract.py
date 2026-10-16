"""Training-view contract tests use artificial inputs only; no engine execution."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from damage_engine.errors import InputValidationError
from damage_engine.normalization import (
    load_training_normalization_artifact,
    save_training_normalization_artifact,
    validate_training_normalization_artifact,
)
from damage_engine.training_view import (
    POSITION_VIEW_FEATURES,
    ROTATION_6D_FEATURES,
    TrainingViewDataset,
    fit_training_view_artifact,
    transform_case_physics,
)
from damage_engine.utils import content_hash


torch = pytest.importorskip("torch")


RAW_NAMES = (
    "detonation_x_m",
    "detonation_y_m",
    "detonation_z_m",
    "orientation_qx",
    "orientation_qy",
    "orientation_qz",
    "orientation_qw",
    "varying_scalar",
    "fixed_context",
    "optional_value",
)
RAW_UNITS = ("m", "m", "m", "1", "1", "1", "1", "Pa", "1", "m/s")


def _declared_tensor_roles() -> dict[str, str]:
    return {
        POSITION_VIEW_FEATURES[0]: "learnable_variable",
        POSITION_VIEW_FEATURES[1]: "learnable_variable",
        POSITION_VIEW_FEATURES[2]: "fixed_context",
        POSITION_VIEW_FEATURES[3]: "learnable_variable",
        ROTATION_6D_FEATURES[0]: "learnable_variable",
        ROTATION_6D_FEATURES[1]: "fixed_context",
        ROTATION_6D_FEATURES[2]: "fixed_context",
        ROTATION_6D_FEATURES[3]: "fixed_context",
        ROTATION_6D_FEATURES[4]: "learnable_variable",
        ROTATION_6D_FEATURES[5]: "fixed_context",
        "varying_scalar": "learnable_variable",
        "fixed_context": "fixed_context",
        "optional_value": "structural_missing",
    }


def _training_design_document(
    roles: dict[str, str], *, geometry_boundary: str = "end_to_end"
) -> dict:
    return {
        "schema_version": "1.0.0",
        "contract_mode": "distribution_design_1.1",
        "feature_roles": [
            {
                "feature_id": f"tensor-{index}",
                "path": f"/artificial/tensor/{index}",
                "tensor_feature_names": [name],
                "role": role,
                "availability_stage": (
                    "not_applicable"
                    if role == "structural_missing"
                    else "prelabel_geometry"
                    if role == "derived_input"
                    else "raw_input"
                ),
                "deployment_available": role != "structural_missing",
            }
            for index, (name, role) in enumerate(roles.items())
        ],
        "variable_blocks": [],
        "geometry_strata": [],
        "stream_roles": [],
        "round_budgets": [],
        "replicate_policy": {
            "default_replicates": 2,
            "outcome_adaptive": False,
            "overrides": [],
        },
        "key_projections": [],
        "proxy_scope": {
            "scope": "fixed_target",
            "geometry_boundary": geometry_boundary,
            "target_identity_policy": "fixed_target_hash",
            "deployment_features_match_training": True,
            "evidence_status": "unverified",
        },
        "legal_sampling": {
            "invalid_candidate_policy": "reject_and_resample",
            "max_attempts_per_allocation": 100,
        },
    }


class ArtificialTrainingDataset:
    def __init__(
        self,
        *,
        split="train",
        changed_source=False,
        declared_roles: dict[str, str] | None = None,
        declared_geometry_boundary: str = "end_to_end",
    ):
        self.split = split
        self.directory = "."
        self._manifest_hash = content_hash({"manifest": "changed" if changed_source else "one"})
        hashes = {
            name: content_hash({"contract": name, "changed": changed_source})
            for name in (
                "split_registry_hash",
                "damage_tree_definition_hash",
                "sampling_design_hash",
                "label_spec_hash",
                "stratification_spec_hash",
            )
        }
        hashes["target_hashes"] = [content_hash({"target": "one"})]
        hashes["parameter_set_hashes"] = [content_hash({"physics": "one"})]
        if declared_roles is not None:
            training_design = _training_design_document(
                declared_roles,
                geometry_boundary=declared_geometry_boundary,
            )
            hashes["training_design_document"] = training_design
            hashes["training_design_hash"] = content_hash(training_design)
        self.manifest = {
            "dataset_id": "artificial-training-view",
            "tensor_contract": {"version": "1.6.0"},
            "source_contracts": hashes,
        }
        self._metadata = {
            "case_physics_feature_names": RAW_NAMES,
            "case_physics_feature_units": RAW_UNITS,
            "component_ids": ("component-a",),
            "armor_ids": ("panel-a",),
            "armor_path_feature_names": (
                "intersects_path",
                "entry_distance_m",
                "path_length_m",
                "incidence_cosine",
            ),
            "armor_path_feature_units": ("1", "m", "m", "1"),
            "label_task_names": ("system-events",),
        }
        values = np.asarray(
            [
                [1, 0, 0, 0, 0, 0, 1, 10, 7, 0],
                [0, 2, 0, 0, 0, 1, 0, 14, 7, 0],
            ],
            dtype=np.float32,
        )
        masks = np.ones_like(values, dtype=bool)
        masks[:, -1] = False
        self.samples = []
        for index in range(2):
            self.samples.append(
                {
                    "case_id": f"case-{index}",
                    "split": split,
                    "case_physics_x": torch.from_numpy(values[index].copy()),
                    "case_physics_feature_mask": torch.from_numpy(masks[index].copy()),
                    "component_armor_path_x": torch.tensor(
                        [[[
                            float(index),
                            0.25 + index,
                            0.1 * index,
                            0.8 + 0.1 * index,
                        ]]],
                        dtype=torch.float32,
                    ),
                    "component_armor_path_mask": torch.ones((1, 1, 4), dtype=torch.bool),
                    "sampling_design_weight_raw": torch.tensor(2.0 + index, dtype=torch.float64),
                    "sampling_weight_mask": torch.tensor(index == 0),
                }
            )

    @property
    def metadata(self):
        return self._metadata

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def test_quaternion_6d_is_sign_invariant_and_position_is_direction_plus_distance():
    values = np.asarray(
        [
            [3, 4, 0, 0.1, -0.2, 0.3, 0.9, 1, 2, 0],
            [3, 4, 0, -0.1, 0.2, -0.3, -0.9, 1, 2, 0],
        ],
        dtype=np.float64,
    )
    mask = np.ones_like(values, dtype=bool)
    transformed, transformed_mask, names, _, _ = transform_case_physics(
        values, mask, RAW_NAMES, RAW_UNITS
    )
    np.testing.assert_allclose(
        transformed[:, :4],
        [[0.6, 0.8, 0.0, 5.0], [0.6, 0.8, 0.0, 5.0]],
    )
    rotation_indices = [index for index, name in enumerate(names) if "rotation_6d" in name]
    assert transformed[0, rotation_indices] == pytest.approx(transformed[1, rotation_indices])
    assert transformed_mask[:, :4].all()


def test_train_only_artifact_is_hashed_and_constants_never_receive_epsilon(tmp_path):
    dataset = ArtificialTrainingDataset()
    artifact = fit_training_view_artifact(dataset, proxy_boundary="end_to_end")
    assert artifact["fit"]["split"] == "train"
    assert artifact["fit"]["case_ids"] == ["case-0", "case-1"]
    assert artifact["source_contracts"]["manifest_hash"] == dataset._manifest_hash
    assert artifact["schema_version"] == "1.1.0"
    assert artifact["feature_role_assignment"] == {
        "mode": "legacy_observed_train_statistics",
        "training_design_hash": None,
    }
    fixed = next(item for item in artifact["features"] if item["name"] == "fixed_context")
    missing = next(item for item in artifact["features"] if item["name"] == "optional_value")
    assert fixed["role"] == "context_fixed"
    assert fixed["normalization"] == "identity" and fixed["scale"] is None
    assert missing["role"] == "optional_masked"
    assert missing["normalization"] == "identity" and missing["scale"] is None
    destination = save_training_normalization_artifact(artifact, tmp_path / "normalizer.json")
    assert load_training_normalization_artifact(destination) == artifact

    tampered = deepcopy(artifact)
    tampered["features"][0]["mean"] = 123.0
    with pytest.raises(InputValidationError, match="normalization_id"):
        validate_training_normalization_artifact(tampered)

    historical = deepcopy(artifact)
    historical["schema_version"] = "1.0.0"
    historical.pop("feature_role_assignment")
    historical.pop("normalization_id")
    historical["normalization_id"] = content_hash(historical)
    assert validate_training_normalization_artifact(historical) == historical


def test_sampling_spec_11_roles_bind_exact_final_tensor_schema():
    roles = _declared_tensor_roles()
    dataset = ArtificialTrainingDataset(declared_roles=roles)
    artifact = fit_training_view_artifact(dataset, proxy_boundary="end_to_end")

    assignment = artifact["feature_role_assignment"]
    assert assignment["mode"] == "declared_sampling_spec_1.1"
    assert assignment["declared_feature_roles"] == roles
    assert assignment["training_design_hash"] == dataset.manifest["source_contracts"][
        "training_design_hash"
    ]
    assert artifact["feature_roles"]["varying_learnable"] == [
        name for name, role in roles.items() if role == "learnable_variable"
    ]
    assert artifact["feature_roles"]["context_fixed"] == [
        name for name, role in roles.items() if role == "fixed_context"
    ]
    assert artifact["feature_roles"]["optional_masked"] == ["optional_value"]


def test_sampling_spec_11_roles_fail_closed_on_coverage_and_observed_semantics():
    incomplete = _declared_tensor_roles()
    incomplete.pop("optional_value")
    with pytest.raises(InputValidationError, match="exactly cover"):
        fit_training_view_artifact(
            ArtificialTrainingDataset(declared_roles=incomplete),
            proxy_boundary="end_to_end",
        )

    wrong_fixed = _declared_tensor_roles()
    wrong_fixed["varying_scalar"] = "fixed_context"
    with pytest.raises(InputValidationError, match="fixed_context.*varies"):
        fit_training_view_artifact(
            ArtificialTrainingDataset(declared_roles=wrong_fixed),
            proxy_boundary="end_to_end",
        )

    present_structural = ArtificialTrainingDataset(
        declared_roles=_declared_tensor_roles()
    )
    present_structural.samples[0]["case_physics_feature_mask"][-1] = True
    with pytest.raises(InputValidationError, match="structural_missing.*present"):
        fit_training_view_artifact(
            present_structural,
            proxy_boundary="end_to_end",
        )


def test_sampling_spec_11_roles_bind_proxy_boundary_and_design_hash():
    dataset = ArtificialTrainingDataset(declared_roles=_declared_tensor_roles())
    dataset.manifest["source_contracts"]["training_design_hash"] = content_hash(
        {"tampered": True}
    )
    with pytest.raises(InputValidationError, match="declared identity"):
        fit_training_view_artifact(dataset, proxy_boundary="end_to_end")

    with pytest.raises(InputValidationError, match="proxy boundary"):
        fit_training_view_artifact(
            ArtificialTrainingDataset(
                declared_roles=_declared_tensor_roles(),
                declared_geometry_boundary="solver_proxy",
            ),
            proxy_boundary="end_to_end",
        )


def test_fit_rejects_non_train_or_mixed_split():
    with pytest.raises(InputValidationError, match="split='train'"):
        fit_training_view_artifact(
            ArtificialTrainingDataset(split="validation"),
            proxy_boundary="end_to_end",
        )
    dataset = ArtificialTrainingDataset()
    dataset.samples[1]["split"] = "test"
    with pytest.raises(InputValidationError, match="non-Train"):
        fit_training_view_artifact(dataset, proxy_boundary="end_to_end")


def test_proxy_boundary_gate_forbids_silent_derived_input_use():
    dataset = ArtificialTrainingDataset()
    with pytest.raises(InputValidationError, match="derived_input_contract"):
        fit_training_view_artifact(dataset, proxy_boundary="solver_surrogate")
    contract = {
        "provider_id": "artificial-geometry-preprocessor",
        "revision": "one",
        "deployment_available": True,
        "outputs": list(dataset.metadata["armor_path_feature_names"]),
    }
    artifact = fit_training_view_artifact(
        dataset,
        proxy_boundary="solver_surrogate",
        derived_input_contract=contract,
    )
    assert artifact["feature_roles"]["derived"]
    tampered = deepcopy(artifact)
    tampered["proxy_boundary"]["derived_input_contract"]["revision"] = "tampered"
    with pytest.raises(InputValidationError, match="contract hash"):
        validate_training_normalization_artifact(tampered)
    with pytest.raises(InputValidationError, match="proxy boundary"):
        TrainingViewDataset(dataset, artifact, proxy_boundary="end_to_end")
    view = TrainingViewDataset(dataset, artifact, proxy_boundary="solver_surrogate")
    assert view[0]["training_derived_x"].numel() == 4


def test_sampling_spec_11_declares_each_solver_derived_tensor_feature():
    path_names = [
        f"component_armor_path::component-a::panel-a::{name}"
        for name in (
            "intersects_path",
            "entry_distance_m",
            "path_length_m",
            "incidence_cosine",
        )
    ]
    roles = _declared_tensor_roles()
    roles.update({name: "derived_input" for name in path_names})
    dataset = ArtificialTrainingDataset(
        declared_roles=roles,
        declared_geometry_boundary="solver_proxy",
    )
    contract = {
        "provider_id": "artificial-geometry-preprocessor",
        "revision": "one",
        "deployment_available": True,
        "outputs": list(dataset.metadata["armor_path_feature_names"]),
    }
    artifact = fit_training_view_artifact(
        dataset,
        proxy_boundary="solver_surrogate",
        derived_input_contract=contract,
    )
    assert artifact["feature_roles"]["derived"] == path_names


def test_loader_binds_source_and_keeps_weight_channels_distinct():
    dataset = ArtificialTrainingDataset()
    artifact = fit_training_view_artifact(dataset, proxy_boundary="end_to_end")
    view = TrainingViewDataset(
        dataset,
        artifact,
        proxy_boundary="end_to_end",
        task_loss_weights={"system-events": 3.0},
        minibatch_probabilities=[0.25, 0.75],
    )
    sample = view[1]
    assert sample["design_weight_raw"].item() == 3.0
    assert sample["design_weight_mask"].item() is False
    assert sample["task_loss_weights"].tolist() == [3.0]
    assert sample["minibatch_sampling_probability"].item() == pytest.approx(0.75)
    assert view.metadata["task_weight_config_id"] != view.metadata["minibatch_config_id"]
    assert sample["training_feature_mask"].shape == sample["training_x"].shape
    assert not sample["training_optional_masked_mask"].any()

    with pytest.raises(InputValidationError, match="source mismatch"):
        TrainingViewDataset(
            ArtificialTrainingDataset(changed_source=True),
            artifact,
            proxy_boundary="end_to_end",
        )


def test_minibatch_probability_is_never_derived_from_design_weight():
    dataset = ArtificialTrainingDataset()
    artifact = fit_training_view_artifact(dataset, proxy_boundary="end_to_end")
    view = TrainingViewDataset(dataset, artifact, proxy_boundary="end_to_end")
    sample = view[0]
    assert "design_weight_raw" in sample
    assert "task_loss_weights" not in sample
    assert "minibatch_sampling_probability" not in sample
    with pytest.raises(InputValidationError, match="sum to one"):
        TrainingViewDataset(
            dataset,
            artifact,
            proxy_boundary="end_to_end",
            minibatch_probabilities=[2.0, 3.0],
        )


def test_mechanism_counts_keep_event_counts_separate_from_design_weight_ess():
    pytest.importorskip("scipy")
    from tools.audit_dataset_distribution import _mechanism_effective_counts

    report = _mechanism_effective_counts(
        splits=np.asarray(["train", "train", "test", "test"]),
        fragment_hit=np.asarray([True, False, True, False]),
        fragment_transition=np.asarray([False, False, True, False]),
        shielding=np.asarray([True, True, False, False]),
        weights=np.asarray([1.0, 2.0, 8.0, 4.0]),
        weight_mask=np.asarray([True, True, False, True]),
    )
    assert report["fragment_hit"]["all"]["case_count"] == 2
    assert report["fragment_hit"]["all"]["design_weight_eligible_count"] == 1
    assert report["fragment_hit"]["all"]["design_weight_ess"] == 1.0
    assert report["fragment_hit_without_equal_penetration"]["test"]["case_count"] == 1
