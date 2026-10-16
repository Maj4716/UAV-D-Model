from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .learning_input_identity import (
    MODEL_INPUT_METADATA_FIELDS,
    UAV_MODEL_INPUT_DEFINITION,
    model_input_definition_for_tensor_field,
)

from .utils import content_hash, file_hash
from .archive_validation import (
    ArchiveIntegrityError, SEMANTIC_NAME_FIELDS, semantic_vocabulary,
    validate_vocabulary,
    verify_archive_receipt, verify_monolithic_archive, verify_sharded_archive,
)
from .system_learning import system_learning_archive_binding


_DEFAULT_SAMPLE_FIELDS = (
    "case_x",
    "case_feature_mask",
    "component_x",
    "component_feature_mask",
    "armor_x",
    "armor_feature_mask",
    "armor_y",
    "armor_y_stddev",
    "armor_y_ci95",
    "armor_label_mask",
    "component_armor_path_x",
    "component_armor_path_mask",
    "component_y",
    "component_y_stddev",
    "component_y_ci95",
    "component_label_mask",
    "component_frechet_y",
    "component_frechet_y_stddev",
    "component_frechet_y_ci95",
    "component_frechet_label_mask",
    "component_aux_y",
    "component_aux_y_stddev",
    "component_aux_y_ci95",
    "component_aux_label_mask",
    "system_y",
    "system_y_stddev",
    "system_y_ci95",
    "system_label_mask",
    "requested_standard_mask",
    "aleatory_seeds",
    "replicate_counts",
    "component_uncertainty_mask", "component_sample_count",
    "component_frechet_uncertainty_mask", "component_frechet_sample_count",
    "armor_uncertainty_mask", "armor_sample_count",
    "component_aux_uncertainty_mask", "component_aux_sample_count",
    "system_uncertainty_mask", "system_sample_count",
)

_DEFAULT_SAMPLE_FIELDS_V15 = (
    "case_physics_x",
    "case_physics_feature_mask",
    *tuple(name for name in _DEFAULT_SAMPLE_FIELDS if name not in {"case_x", "case_feature_mask"}),
    "sampling_selection_probability",
    "sampling_design_weight_raw",
    "sampling_weight_mask",
    "sampling_round",
    "sampling_stratum_index",
    "sampling_role_index",
)
_DEFAULT_SAMPLE_FIELDS_V16 = (
    *_DEFAULT_SAMPLE_FIELDS_V15, "system_source_available_mask", "label_task_available_mask",
)
_DEFAULT_SAMPLE_FIELDS_V17 = (
    *_DEFAULT_SAMPLE_FIELDS_V16,
    "sampling_p0_density", "sampling_adaptive_density",
    "sampling_proposal_density", "sampling_p0_hierarchy_probability",
    "sampling_proposal_hierarchy_probability",
)
_DEFAULT_LEARNER_AUDIT_FIELDS_V18 = (
    *tuple(
        name
        for name in _DEFAULT_SAMPLE_FIELDS_V16
        if name
        not in {
            "case_physics_x",
            "case_physics_feature_mask",
            "component_x",
            "component_feature_mask",
            "armor_x",
            "armor_feature_mask",
            "component_armor_path_x",
            "component_armor_path_mask",
        }
    ),
)
_ADAPTIVE_SAMPLE_FIELDS = (
    "sampling_p0_density", "sampling_adaptive_density",
    "sampling_proposal_density", "sampling_p0_hierarchy_probability",
    "sampling_proposal_hierarchy_probability",
)


def _default_sample_fields(
    contract_version: str | None,
    sampling_envelope_version: str | None = None,
    model_input_tensor_field: str | None = None,
) -> tuple[str, ...]:
    if contract_version == "1.8.0":
        definition = (
            model_input_definition_for_tensor_field(model_input_tensor_field)
            or UAV_MODEL_INPUT_DEFINITION
        )
        fields = (
            definition.tensor_field,
            definition.mask_field,
            *_DEFAULT_LEARNER_AUDIT_FIELDS_V18,
        )
        return (
            *fields,
            *(_ADAPTIVE_SAMPLE_FIELDS if sampling_envelope_version == "1.2.0" else ()),
        )
    if contract_version == "1.7.0":
        return _DEFAULT_SAMPLE_FIELDS_V17
    if contract_version == "1.6.0":
        return _DEFAULT_SAMPLE_FIELDS_V16
    return _DEFAULT_SAMPLE_FIELDS_V15 if contract_version == "1.5.0" else _DEFAULT_SAMPLE_FIELDS

_STATIC_GRAPH_FIELDS = (
    "armor_vertices_m",
    "armor_vertex_offsets",
    "component_vertices_m",
    "component_vertex_offsets",
    "tree_edge_index",
    "tree_edge_type",
)

_STATIC_STANDARD_FIELDS = (
    "standard_output_mask",
)

_STRING_SAMPLE_FIELDS = (
    "case_ids",
    "case_hashes",
    "split_names",
    "split_group_ids",
    "split_group_sources",
    "leakage_keys",
    "epistemic_sample_ids",
    "sampling_candidate_ids",
    "sampling_lineage_ids",
    "sampling_epistemic_group_ids",
    "sampling_acquisition_model_hashes",
    "sampling_proposal_hashes",
    "sampling_round_lineage_ids",
    "sampling_selected_case_set_hashes",
)


_DEFAULT_STREAM_BY_SAMPLING_ROLE = {
    # SamplingSpec 1.1 roles.
    "reference_train": "reference",
    "reference_validation": "reference",
    "reference_test": "reference",
    "geometry_stratified_train": "geometry_stratified",
    "rare_stress_test": "stress",
    # SamplingSpec 1.0 and adaptive-round compatibility roles.
    "train_pilot": "reference",
    "reference_pilot": "reference",
    "adaptive_train": "adaptive",
}

_POLICY_SAMPLING_ROLES = {
    "reference_train": {
        "reference_train", "train_pilot", "reference_pilot",
        "geometry_stratified_train",
    },
    "adaptive_train_round": {"adaptive_train"},
    "reference_validation": {"reference_validation"},
    "sealed_reference_test": {"reference_test"},
    "rare_stress_test": {"rare_stress_test"},
}

def _normalize_role_filter(value, *, name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    values = (value,) if isinstance(value, str) else tuple(value)
    if not values or any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"{name} must contain one or more nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def _stream_by_sampling_role(manifest: dict | None) -> dict[str, str]:
    mapping = dict(_DEFAULT_STREAM_BY_SAMPLING_ROLE)
    source = manifest.get("source_contracts", {}) if isinstance(manifest, dict) else {}
    specification = source.get("sampling_spec_document")
    if isinstance(specification, dict):
        for record in specification.get("stream_roles", ()):
            if not isinstance(record, dict):
                continue
            role, stream = record.get("role"), record.get("stream")
            if isinstance(role, str) and role and isinstance(stream, str) and stream:
                if role in mapping and mapping[role] != stream:
                    raise ValueError(
                        f"sampling specification maps role {role!r} to conflicting stream {stream!r}"
                    )
                mapping[role] = stream
    return mapping


def _sampling_role_values(
    arrays: dict[str, np.ndarray], *, expected_count: int,
) -> np.ndarray | None:
    index = arrays.get("sampling_role_index")
    names = arrays.get("sampling_role_names")
    if index is None and names is None:
        return None
    if index is None or names is None:
        raise ValueError(
            "sampling_role_index and sampling_role_names must appear together"
        )
    index = np.asarray(index)
    names = np.asarray(names)
    if index.shape != (expected_count,) or index.dtype.kind not in "iu":
        raise ValueError("sampling_role_index must be an integer valid-case axis")
    if names.ndim != 1 or names.dtype.kind not in "US" or not len(names):
        raise ValueError("sampling_role_names must be a nonempty string vocabulary")
    if np.any(index < 0) or np.any(index >= len(names)):
        raise ValueError("sampling_role_index exceeds its vocabulary")
    return names[index].astype(str)


def _filter_case_indices(
    arrays: dict[str, np.ndarray],
    manifest: dict | None,
    *,
    split: str | None,
    sampling_roles: tuple[str, ...] | None,
    stream_roles: tuple[str, ...] | None,
    strict_filter_match: bool,
) -> tuple[np.ndarray, set[str], set[str]]:
    """Return selected indices plus pre/post-filter role sets.

    ``candidate_roles`` is evaluated after split selection and before role
    selection. It is used by sharded roots to detect a mixed test stream even
    when the two roles reside in different shards.
    """

    count = len(arrays["case_x"])
    selected = np.ones(count, dtype=bool)
    if split is not None:
        if "split_names" not in arrays:
            raise ValueError("role-aware filtering requires split_names")
        selected &= np.asarray(arrays["split_names"]).astype(str) == split

    role_values = _sampling_role_values(arrays, expected_count=count)
    if role_values is None:
        if sampling_roles is not None or stream_roles is not None:
            raise ValueError(
                "sampling_role/stream_role filtering requires sampling-role metadata"
            )
        return np.flatnonzero(selected).astype(np.int64, copy=False), set(), set()

    candidate_roles = set(map(str, role_values[selected]))
    mapping = _stream_by_sampling_role(manifest)
    unknown_roles = sorted(candidate_roles - set(mapping))
    if unknown_roles:
        raise ValueError(
            "stream_role cannot be derived for sampling roles: "
            + ", ".join(unknown_roles)
        )
    if split == "test" and sampling_roles is None and stream_roles is None:
        if "reference_test" in candidate_roles and "rare_stress_test" in candidate_roles:
            raise ValueError(
                "test split mixes Reference Test and Rare/Stress Test; select "
                "sampling_role or stream_role explicitly"
            )

    if sampling_roles is not None:
        selected &= np.isin(role_values, np.asarray(sampling_roles))
    if stream_roles is not None:
        relevant_roles = set(map(str, role_values[selected]))
        unknown = sorted(role for role in relevant_roles if role not in mapping)
        if unknown:
            raise ValueError(
                "stream_role cannot be derived for sampling roles: " + ", ".join(unknown)
            )
        stream_values = np.asarray([mapping.get(str(role), "") for role in role_values])
        selected &= np.isin(stream_values, np.asarray(stream_roles))

    indices = np.flatnonzero(selected).astype(np.int64, copy=False)
    selected_roles = set(map(str, role_values[indices]))
    if split == "test" and {
        "reference_test", "rare_stress_test"
    }.issubset(selected_roles):
        raise ValueError(
            "test split mixes Reference Test and Rare/Stress Test; select "
            "exactly one sampling_role or stream_role"
        )
    if strict_filter_match and (sampling_roles is not None or stream_roles is not None) and not len(indices):
        raise ValueError(
            "sampling_role/stream_role selection matched no cases after split filtering"
        )
    return indices, candidate_roles, selected_roles


def _coerce_dataset_policy(policy):
    if policy is None:
        return None
    from .dataset_policy import DatasetUsePolicy

    if isinstance(policy, DatasetUsePolicy):
        return policy
    if isinstance(policy, (str, Path)):
        return DatasetUsePolicy.from_file(policy)
    raise TypeError("policy must be DatasetUsePolicy, a policy path, or None")


def _validate_policy_selection(policy, selected_roles: set[str]) -> None:
    if policy is None or policy.classification == "software_interpolation_pilot":
        return
    allowed = _POLICY_SAMPLING_ROLES.get(policy.classification, set())
    if not selected_roles:
        raise ValueError(
            "dataset policy cannot be bound to a role-specific view without "
            "sampling-role metadata"
        )
    incompatible = sorted(selected_roles - allowed)
    if incompatible:
        raise ValueError(
            f"dataset policy classification {policy.classification!r} is incompatible "
            f"with selected sampling roles: {incompatible}"
        )


class TorchDamageDataset:
    """Efficient, spawn-safe map-style view of ``tensors.npz``.

    Arrays are decompressed once per process instead of once per sample. The
    object drops its in-memory arrays when pickled, so Windows ``DataLoader``
    workers reopen the archive independently rather than inheriting an
    unpickleable ``NpzFile`` handle.

    Returned tensors own copies by default and may safely be transformed in
    place. ``copy_on_read=False`` is an explicit performance option: returned
    sample and static tensors then alias the per-process cache and must not be
    changed in place. Admission overrides are retained in ``audit_overrides``.
    """

    def __init__(
        self,
        dataset_directory: str | Path,
        *,
        split: str | None = None,
        sampling_role: str | Iterable[str] | None = None,
        stream_role: str | Iterable[str] | None = None,
        fields: Iterable[str] | None = None,
        include_static_graph: bool | None = None,
        copy_on_read: bool = True,
        verify_integrity: bool = True,
        allow_split_leakage: bool = False,
        allow_legacy_manifest: bool = False,
        purpose: str = "software_development",
        evidence_root: str | Path | None = None,
        policy=None,
        operation: str | None = None,
        model_hash: str | None = None,
        require_research_admission: bool = False,
    ):
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "TorchDamageDataset requires PyTorch; install the project with the 'torch' extra"
            ) from exc
        self.directory = Path(dataset_directory)
        if split is not None and split not in {"train", "validation", "test"}:
            raise ValueError("split must be one of: train, validation, test")
        self.archive_path = self.directory / "tensors.npz"
        self.copy_on_read = bool(copy_on_read)
        self.verify_integrity = bool(verify_integrity)
        self.allow_split_leakage = bool(allow_split_leakage)
        self.allow_legacy_manifest = bool(allow_legacy_manifest)
        self.purpose = str(purpose)
        self.evidence_root = Path(evidence_root) if evidence_root is not None else None
        self.sampling_roles = _normalize_role_filter(
            sampling_role, name="sampling_role"
        )
        self.stream_roles = _normalize_role_filter(stream_role, name="stream_role")
        self.policy = _coerce_dataset_policy(policy)
        self.operation = operation
        self.model_hash = model_hash
        self.require_research_admission = bool(require_research_admission)
        if self.policy is None and any(
            value is not None for value in (operation, model_hash)
        ):
            raise ValueError("operation/model_hash require an explicit dataset policy")
        if self.policy is None and self.require_research_admission:
            raise ValueError("require_research_admission requires an explicit dataset policy")
        if self.policy is not None and not isinstance(operation, str):
            raise ValueError("policy loading requires an explicit operation")
        self._archive_receipt = verify_monolithic_archive(
            self.directory, verify_integrity=self.verify_integrity,
            allow_split_leakage=self.allow_split_leakage,
            allow_legacy_manifest=self.allow_legacy_manifest,
            purpose=self.purpose, evidence_root=self.evidence_root,
            validate_tensors=True,
        )
        self.manifest = self._archive_receipt.manifest
        self.audit_overrides = self._archive_receipt.audit_overrides
        self._manifest_hash = self._archive_receipt.manifest_hash
        if self.policy is not None:
            self.policy.assert_verified_archive_binding(self._archive_receipt)
            self.policy.assert_operation(
                operation,
                model_hash=self.model_hash,
                require_research_admission=self.require_research_admission,
            )
        contract_version = self.manifest.get("tensor_contract", {}).get("version") if self.manifest else None
        # Tensor 1.8 carries an explicit model-facing state contract. Its
        # default view must not silently carry target/armor/tree teacher
        # geometry into a learner. Legacy contracts keep their historical
        # graph default; callers can still request archived graph data.
        self.include_static_graph = (
            contract_version != "1.8.0"
            if include_static_graph is None
            else bool(include_static_graph)
        )
        sampling_envelope_version = (
            self.manifest.get("generation", {}).get("sampling", {}).get(
                "sampling_envelope_version"
            )
            if self.manifest
            else None
        )
        self.sample_fields = tuple(fields) if fields is not None else _default_sample_fields(
            contract_version,
            sampling_envelope_version,
            (self.manifest or {}).get("model_input_contract", {}).get(
                "tensor_field"
            ),
        )
        self._arrays: dict[str, np.ndarray] | None = None
        self._load_arrays()
        arrays = self._require_arrays()
        if fields is None and "system_source_available_mask" in arrays and "system_source_available_mask" not in self.sample_fields:
            self.sample_fields += ("system_source_available_mask",)
        if fields is None and self.allow_legacy_manifest and (
            not self.manifest
            or self.manifest.get("tensor_contract", {}).get("version") not in {"1.4.0", "1.5.0", "1.6.0", "1.7.0", "1.8.0"}
        ):
            self.sample_fields = tuple(
                name for name in self.sample_fields
                if name in arrays or not (name.endswith("_uncertainty_mask") or name.endswith("_sample_count"))
            )
        if "case_x" not in arrays:
            raise ValueError("tensors.npz is missing required field 'case_x'")
        batch_size = len(arrays["case_x"])
        for name in self.sample_fields:
            if name not in arrays:
                raise ValueError(f"tensors.npz is missing requested field {name!r}")
            if len(arrays[name]) != batch_size:
                raise ValueError(f"tensor field {name!r} does not share the case axis")
        if split is not None and "split_names" not in arrays:
            arrays = dict(arrays)
            arrays["split_names"] = self._split_names(arrays)
        self.indices, available_roles, selected_roles = _filter_case_indices(
            arrays,
            self.manifest,
            split=split,
            sampling_roles=self.sampling_roles,
            stream_roles=self.stream_roles,
            strict_filter_match=True,
        )
        self.available_sampling_roles = frozenset(available_roles)
        self.selected_sampling_roles = tuple(sorted(selected_roles))
        _validate_policy_selection(self.policy, selected_roles)
        self._stream_by_role = _stream_by_sampling_role(self.manifest)
        self.split = split

    def _load_arrays(self) -> None:
        names = set(self.sample_fields) | set(_STRING_SAMPLE_FIELDS) | {
            "case_x",
            "case_feature_names",
            "case_feature_units",
            "case_physics_feature_names",
            "case_physics_feature_units",
            *MODEL_INPUT_METADATA_FIELDS,
            "solver_context_feature_names",
            "solver_context_feature_units",
            "sampling_stratum_names",
            "sampling_role_names",
            "sampling_role_index",
            "label_task_names",
            "system_source_available_mask",
            "component_feature_names",
            "component_feature_units",
            "armor_feature_names",
            "armor_feature_units",
            "armor_label_names",
            "armor_label_units",
            "armor_path_feature_names",
            "armor_path_feature_units",
            "component_aux_label_names",
            "component_aux_label_units",
            "mechanism_names",
            "state_names",
            "frechet_bound_names",
            "frechet_bound_units",
            "system_output_names",
            "evaluation_standard_ids",
            "component_ids",
            "armor_ids",
            "graph_node_ids",
            "graph_node_types",
        }
        names.update(_STATIC_STANDARD_FIELDS)
        if self.include_static_graph:
            names.update(_STATIC_GRAPH_FIELDS)
        with np.load(self.archive_path, allow_pickle=False) as archive:
            available = set(archive.files)
            # Validate the complete stored contract even if the caller requests
            # only a subset of fields. The shared archive verifier has already
            # checked the complete stored contract; retain requested arrays here.
            all_arrays = {name: np.asarray(archive[name]) for name in available}
            self._arrays = {
                name: all_arrays[name] for name in names if name in available
            }

    def _require_arrays(self) -> dict[str, np.ndarray]:
        if self._arrays is None:
            self._load_arrays()
        assert self._arrays is not None
        return self._arrays

    def _split_names(self, arrays: dict[str, np.ndarray]) -> np.ndarray:
        if "split_names" in arrays:
            return arrays["split_names"].astype(str)
        # Compatibility with 1.0 datasets. New datasets carry split names in
        # NPZ and therefore do not require pyarrow merely to train.
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "legacy split filtering requires pyarrow; regenerate the dataset to remove this dependency"
            ) from exc
        table = pq.read_table(self.directory / "splits.parquet").to_pydict()
        split_by_case = dict(zip(table["case_id"], table["split"]))
        return np.asarray(
            [split_by_case[str(case_id)] for case_id in arrays["case_ids"].tolist()]
        )

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_arrays"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        receipt = verify_monolithic_archive(
            self.directory, verify_integrity=self.verify_integrity,
            allow_split_leakage=self.allow_split_leakage,
            allow_legacy_manifest=self.allow_legacy_manifest,
            purpose=self.purpose, evidence_root=self.evidence_root,
            validate_tensors=True,
        )
        if receipt.manifest_hash != self._manifest_hash:
            raise ArchiveIntegrityError(
                "dataset manifest changed while reopening a worker archive",
                integrity_kind="manifest_hash_mismatch",
                archive_scope="root",
                path=self.directory / "manifest.json",
            )
        self._archive_receipt = receipt
        self.manifest = receipt.manifest
        self.audit_overrides = receipt.audit_overrides
        self._load_arrays()

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> dict:
        import torch

        arrays = self._require_arrays()
        index = int(self.indices[item])

        def tensor(value: np.ndarray):
            array = np.asarray(value)
            if self.copy_on_read:
                array = array.copy()
            return torch.from_numpy(array)

        sample = {name: tensor(arrays[name][index]) for name in self.sample_fields}
        sample.update(
            {
                "case_id": str(arrays["case_ids"][index]),
                "case_hash": str(arrays["case_hashes"][index]) if "case_hashes" in arrays else "",
                "split": str(arrays["split_names"][index]) if "split_names" in arrays else "",
                "split_group_id": str(arrays["split_group_ids"][index]) if "split_group_ids" in arrays else "",
                "epistemic_sample_id": str(arrays["epistemic_sample_ids"][index]) if "epistemic_sample_ids" in arrays else "",
                "sampling_candidate_id": str(arrays["sampling_candidate_ids"][index]) if "sampling_candidate_ids" in arrays else "",
                "sampling_lineage_id": str(arrays["sampling_lineage_ids"][index]) if "sampling_lineage_ids" in arrays else "",
                "sampling_epistemic_group_id": str(arrays["sampling_epistemic_group_ids"][index]) if "sampling_epistemic_group_ids" in arrays else "",
                "sampling_acquisition_model_hash": str(arrays["sampling_acquisition_model_hashes"][index]) if "sampling_acquisition_model_hashes" in arrays else "",
                "sampling_proposal_hash": str(arrays["sampling_proposal_hashes"][index]) if "sampling_proposal_hashes" in arrays else "",
                "sampling_round_lineage_id": str(arrays["sampling_round_lineage_ids"][index]) if "sampling_round_lineage_ids" in arrays else "",
                "sampling_selected_case_set_hash": str(arrays["sampling_selected_case_set_hashes"][index]) if "sampling_selected_case_set_hashes" in arrays else "",
            }
        )
        if "sampling_role_index" in arrays and "sampling_role_names" in arrays:
            role = str(arrays["sampling_role_names"][int(arrays["sampling_role_index"][index])])
            sample["sampling_role"] = role
            sample["stream_role"] = self._stream_by_role.get(role, "")
        if "sampling_stratum_index" in arrays and "sampling_stratum_names" in arrays:
            sample["sampling_stratum"] = str(arrays["sampling_stratum_names"][int(arrays["sampling_stratum_index"][index])])
        if self.include_static_graph:
            for name in _STATIC_GRAPH_FIELDS:
                if name in arrays:
                    sample[name] = tensor(arrays[name])
        for name in _STATIC_STANDARD_FIELDS:
            if name in arrays:
                sample[name] = tensor(arrays[name])
        return sample

    @property
    def metadata(self) -> dict[str, object]:
        arrays = self._require_arrays()
        names = (
            "case_feature_names", "case_feature_units",
            "case_physics_feature_names", "case_physics_feature_units",
            *MODEL_INPUT_METADATA_FIELDS,
            "solver_context_feature_names", "solver_context_feature_units",
            "sampling_stratum_names", "sampling_role_names", "label_task_names",
            "component_feature_names", "component_feature_units",
            "armor_feature_names", "armor_feature_units",
            "armor_label_names", "armor_label_units",
            "armor_path_feature_names", "armor_path_feature_units",
            "component_aux_label_names", "component_aux_label_units",
            "mechanism_names", "state_names", "system_output_names",
            "frechet_bound_names", "frechet_bound_units",
            "evaluation_standard_ids",
            "component_ids", "armor_ids",
            "graph_node_ids", "graph_node_types",
        )
        result: dict[str, object] = {
            name: tuple(map(str, arrays[name].tolist()))
            for name in names
            if name in arrays
        }
        result["system_learning_binding"] = system_learning_archive_binding(
            self.manifest, self._manifest_hash
        )
        return result

    @property
    def target_hash(self) -> str:
        if not self.manifest:
            return ""
        hashes = self.manifest.get("source_contracts", {}).get("target_hashes", [])
        return str(hashes[0]) if len(hashes) == 1 else ""

    @property
    def topology_metadata(self) -> dict[str, object]:
        return {**self.metadata, "target_hash": self.target_hash}

    def training_view(self, artifact: dict, **kwargs):
        """Return an immutable-archive training view with a bound normalizer."""

        from .training_view import TrainingViewDataset

        return TrainingViewDataset(self, artifact, **kwargs)


class TorchShardedDamageDataset:
    """Lazy, spawn-safe map-style view of a completed sharded dataset.

    Initialization validates one complete shard at a time, retaining only
    topology and split identity metadata. Requested shard arrays are opened
    through :class:`TorchDamageDataset` on first access and retained in a
    bounded least-recently-used cache. Pickling drops that cache so Windows
    ``DataLoader`` workers reopen and validate archives in their own processes.
    """

    def __init__(
        self,
        dataset_directory: str | Path,
        *,
        split: str | None = None,
        sampling_role: str | Iterable[str] | None = None,
        stream_role: str | Iterable[str] | None = None,
        fields: Iterable[str] | None = None,
        include_static_graph: bool | None = None,
        copy_on_read: bool = True,
        max_cached_shards: int = 2,
        verify_integrity: bool = True,
        allow_split_leakage: bool = False,
        allow_legacy_manifest: bool = False,
        purpose: str = "software_development",
        evidence_root: str | Path | None = None,
        policy=None,
        operation: str | None = None,
        model_hash: str | None = None,
        require_research_admission: bool = False,
    ):
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "TorchShardedDamageDataset requires PyTorch; install the project with the 'torch' extra"
            ) from exc
        if split is not None and split not in {"train", "validation", "test"}:
            raise ValueError("split must be one of: train, validation, test")
        if int(max_cached_shards) < 1:
            raise ValueError("max_cached_shards must be at least one")

        self.directory = Path(dataset_directory)
        self.split = split
        self.sampling_roles = _normalize_role_filter(
            sampling_role, name="sampling_role"
        )
        self.stream_roles = _normalize_role_filter(stream_role, name="stream_role")
        self.sample_fields = tuple(fields) if fields is not None else None
        self.copy_on_read = bool(copy_on_read)
        self.max_cached_shards = int(max_cached_shards)
        self.verify_integrity = bool(verify_integrity)
        self.allow_split_leakage = bool(allow_split_leakage)
        self.allow_legacy_manifest = bool(allow_legacy_manifest)
        self.purpose = str(purpose)
        self.evidence_root = Path(evidence_root) if evidence_root is not None else None
        self.policy = _coerce_dataset_policy(policy)
        self.operation = operation
        self.model_hash = model_hash
        self.require_research_admission = bool(require_research_admission)
        if self.policy is None and any(
            value is not None for value in (operation, model_hash)
        ):
            raise ValueError("operation/model_hash require an explicit dataset policy")
        if self.policy is None and self.require_research_admission:
            raise ValueError("require_research_admission requires an explicit dataset policy")
        if self.policy is not None and not isinstance(operation, str):
            raise ValueError("policy loading requires an explicit operation")
        self._cache: OrderedDict[int, TorchDamageDataset] = OrderedDict()
        self._archive_receipt = verify_sharded_archive(
            self.directory,
            verify_integrity=self.verify_integrity,
            allow_split_leakage=self.allow_split_leakage,
            allow_legacy_manifest=self.allow_legacy_manifest,
            purpose=self.purpose,
            evidence_root=self.evidence_root,
            validate_tensors=True,
        )
        checkpoint = self._archive_receipt.checkpoint
        manifest = self._archive_receipt.manifest
        assert checkpoint is not None and manifest is not None
        self.checkpoint = checkpoint
        self.manifest = manifest
        self.audit_overrides = self._archive_receipt.audit_overrides
        self._manifest_hash = self._archive_receipt.manifest_hash
        contract_version = self.manifest.get("tensor_contract", {}).get("version")
        self.include_static_graph = (
            contract_version != "1.8.0"
            if include_static_graph is None
            else bool(include_static_graph)
        )
        if self.policy is not None:
            self.policy.assert_verified_archive_binding(self._archive_receipt)
            self.policy.assert_operation(
                operation,
                model_hash=self.model_hash,
                require_research_admission=self.require_research_admission,
            )
        if self.sample_fields is None:
            self.sample_fields = _default_sample_fields(
                self.manifest.get("tensor_contract", {}).get("version"),
                self.manifest.get("generation", {}).get("sampling", {}).get(
                    "sampling_envelope_version"
                ),
                self.manifest.get("model_input_contract", {}).get(
                    "tensor_field"
                ),
            )
        records = list(self.checkpoint["completed_shards"])
        verified_shard_manifests = self._archive_receipt.shard_manifests

        root = self.directory.resolve()
        shard_paths: list[Path] = []
        counts: list[int] = []
        self._case_registry: set[str] = set()
        self._candidate_registry: set[str] = set()
        self._group_registry: dict[tuple[str, str], str] = {}
        self._inspected_topology: dict | None = None
        self._candidate_sampling_roles: set[str] = set()
        self._selected_sampling_roles: set[str] = set()
        for record, shard_manifest in zip(records, verified_shard_manifests):
            raw_path = record.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError("completed shard record is missing a path")
            relative_path = Path(raw_path)
            if relative_path.is_absolute():
                raise ValueError("completed shard paths must be relative to the dataset root")
            shard_path = (self.directory / relative_path).resolve()
            try:
                shard_path.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    "completed shard path escapes the dataset root"
                ) from exc
            count = self._inspect_shard(shard_path, record, shard_manifest)
            shard_paths.append(shard_path)
            counts.append(count)

        if self.split == "test" and {"reference_test", "rare_stress_test"}.issubset(
            self._selected_sampling_roles
        ):
            raise ValueError(
                "test split mixes Reference Test and Rare/Stress Test; select "
                "exactly one sampling_role or stream_role"
            )
        if (
            (self.sampling_roles is not None or self.stream_roles is not None)
            and not self._selected_sampling_roles
        ):
            raise ValueError(
                "sampling_role/stream_role selection matched no cases after split filtering"
            )
        self.available_sampling_roles = frozenset(self._candidate_sampling_roles)
        self.selected_sampling_roles = tuple(sorted(self._selected_sampling_roles))
        _validate_policy_selection(self.policy, self._selected_sampling_roles)
        self._stream_by_role = _stream_by_sampling_role(self.manifest)

        if self.manifest.get("quality_summary", {}).get("n_valid") != len(self._case_registry):
            raise ValueError("root quality_summary.n_valid does not match committed shard valid axes")

        self.shard_records = tuple(records)
        self.shard_paths = tuple(shard_paths)
        self.shard_lengths = tuple(counts)
        self._active_shard_indices = tuple(
            index for index, count in enumerate(counts) if count > 0
        )
        self.offsets = np.cumsum(
            [0, *(counts[index] for index in self._active_shard_indices)],
            dtype=np.int64,
        )

    def _inspect_shard(
        self, shard_path: Path, record: dict, shard_manifest: dict,
    ) -> int:
        archive_path = shard_path / "tensors.npz"
        manifest_path = shard_path / "manifest.json"
        with np.load(archive_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
            for name in ("damage_tree_definition_hash", "label_spec_hash", "label_spec_document", "split_registry_hash"):
                if shard_manifest.get("source_contracts", {}).get(name) != self.manifest.get("source_contracts", {}).get(name):
                    raise ValueError(f"root and shard source contracts disagree on {name}")
            topology = {
                name: tuple(map(str, value)) for name, value in arrays.items()
                if value.ndim == 1 and ((name.endswith("_names") and name != "split_names") or name.endswith("_units") or name in {"component_ids", "armor_ids", "graph_node_ids", "graph_node_types", "evaluation_standard_ids"})
            }
            if self._inspected_topology is not None and topology != self._inspected_topology:
                raise ValueError("shards do not share the same topology and semantic names")
            self._inspected_topology = topology
            case_ids = set(map(str, arrays["case_ids"]))
            if self._case_registry & case_ids:
                raise ValueError("case_ids are duplicated across committed shards")
            self._case_registry.update(case_ids)
            candidate_field = "input_sampling_candidate_ids" if "input_sampling_candidate_ids" in arrays else "sampling_candidate_ids"
            if candidate_field in arrays:
                identifiers = list(map(str, arrays[candidate_field]))
                if len(set(identifiers)) != len(identifiers) or self._candidate_registry & set(identifiers):
                    raise ValueError("sampling candidate IDs are duplicated across committed input shards")
                self._candidate_registry.update(identifiers)
            for key in ("split_group_ids", "leakage_keys"):
                if key in arrays:
                    for group, split_name in zip(map(str, arrays[key]), map(str, arrays["split_names"])):
                        registry_key = (key, group)
                        if registry_key in self._group_registry and self._group_registry[registry_key] != split_name and not self.allow_split_leakage:
                            raise ValueError(f"{key} crosses split boundaries across shards")
                        self._group_registry[registry_key] = split_name
            if "case_x" not in archive.files:
                raise ValueError(f"{archive_path} is missing required field 'case_x'")
            if self.split is not None and "split_names" not in archive.files:
                raise ValueError(
                    f"{archive_path} is missing 'split_names'; regenerate the sharded dataset"
                )
            valid_count = shard_manifest.get("quality_summary", {}).get("n_valid")
            if not isinstance(valid_count, int) or valid_count < 0:
                raise ValueError(
                    f"{manifest_path} is missing a valid quality_summary.n_valid"
                )
            if "split_names" in arrays and len(arrays["split_names"]) != valid_count:
                raise ValueError(
                    f"{archive_path} split_names does not match the valid case axis"
                )
            selected, candidate_roles, selected_roles = _filter_case_indices(
                arrays,
                shard_manifest,
                split=self.split,
                sampling_roles=self.sampling_roles,
                stream_roles=self.stream_roles,
                strict_filter_match=False,
            )
            self._candidate_sampling_roles.update(candidate_roles)
            self._selected_sampling_roles.update(selected_roles)
            return int(len(selected))

    def _open_shard(self, shard_index: int) -> TorchDamageDataset:
        if shard_index in self._cache:
            dataset = self._cache.pop(shard_index)
            self._cache[shard_index] = dataset
            return dataset
        # Initialization establishes a root snapshot, but shards are opened
        # lazily.  Revalidate that receipt immediately before consuming a new
        # child so a post-construction archive drift cannot mix snapshots.
        verify_archive_receipt(self._archive_receipt)
        dataset = TorchDamageDataset(
            self.shard_paths[shard_index],
            split=self.split,
            sampling_role=self.sampling_roles,
            stream_role=self.stream_roles,
            fields=self.sample_fields,
            include_static_graph=self.include_static_graph,
            copy_on_read=self.copy_on_read,
            verify_integrity=self.verify_integrity,
            allow_split_leakage=self.allow_split_leakage,
            allow_legacy_manifest=self.allow_legacy_manifest,
            purpose=self.purpose, evidence_root=self.evidence_root,
        )
        expected_manifest_hash = self.shard_records[shard_index].get(
            "manifest_hash"
        )
        if dataset._manifest_hash != expected_manifest_hash:
            raise ArchiveIntegrityError(
                "shard manifest differs from the verified root snapshot",
                integrity_kind="manifest_hash_mismatch",
                archive_scope="shard",
                path=self.shard_paths[shard_index] / "manifest.json",
                context={"shard_index": shard_index},
            )
        self._cache[shard_index] = dataset
        while len(self._cache) > self.max_cached_shards:
            self._cache.popitem(last=False)
        return dataset

    def clear_cache(self) -> None:
        """Release all currently retained shard arrays."""

        self._cache.clear()

    @property
    def cached_shard_indices(self) -> tuple[int, ...]:
        """Shard indices in least- to most-recently-used order."""

        return tuple(self._cache)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state

    def __setstate__(self, state: dict) -> None:
        expected_receipt = state["_archive_receipt"]
        verify_archive_receipt(expected_receipt)
        self.__init__(
            state["directory"], split=state["split"],
            sampling_role=state["sampling_roles"], stream_role=state["stream_roles"],
            fields=state["sample_fields"],
            include_static_graph=state["include_static_graph"], copy_on_read=state["copy_on_read"],
            max_cached_shards=state["max_cached_shards"], verify_integrity=state["verify_integrity"],
            allow_split_leakage=state["allow_split_leakage"], allow_legacy_manifest=state["allow_legacy_manifest"],
            purpose=state["purpose"], evidence_root=state["evidence_root"],
            policy=state["policy"], operation=state["operation"],
            model_hash=state["model_hash"],
            require_research_admission=state["require_research_admission"],
        )
        snapshot_fields = (
            "dataset_id",
            "manifest_hash",
            "checkpoint_hash",
            "committed_files",
        )
        changed_fields = [
            name
            for name in snapshot_fields
            if getattr(self._archive_receipt, name) != getattr(expected_receipt, name)
        ]
        if changed_fields:
            raise ArchiveIntegrityError(
                "dataset snapshot changed while reopening a worker archive",
                integrity_kind="worker_snapshot_mismatch",
                archive_scope="root",
                path=self.directory / "manifest.json",
                context={
                    "changed_fields": changed_fields,
                    "expected_dataset_id": expected_receipt.dataset_id,
                    "observed_dataset_id": self._archive_receipt.dataset_id,
                },
            )

    def __len__(self) -> int:
        return int(self.offsets[-1]) if len(self.offsets) else 0

    def __getitem__(self, item: int) -> dict:
        length = len(self)
        index = int(item)
        if index < 0:
            index += length
        if not 0 <= index < length:
            raise IndexError(item)
        active_index = bisect_right(self.offsets, index) - 1
        shard_index = self._active_shard_indices[active_index]
        local_index = index - int(self.offsets[active_index])
        return self._open_shard(shard_index)[local_index]

    @property
    def metadata(self) -> dict[str, object]:
        if not self._active_shard_indices:
            raise RuntimeError("filtered sharded dataset contains no active shards")
        result = dict(self._open_shard(self._active_shard_indices[0]).metadata)
        result["system_learning_binding"] = system_learning_archive_binding(
            self.manifest, self._manifest_hash
        )
        return result

    @property
    def target_hash(self) -> str:
        hashes = self.manifest.get("source_contracts", {}).get("target_hashes", [])
        return str(hashes[0]) if len(hashes) == 1 else ""

    @property
    def topology_metadata(self) -> dict[str, object]:
        return {**self.metadata, "target_hash": self.target_hash}

    def training_view(self, artifact: dict, **kwargs):
        """Return an immutable-archive training view with a bound normalizer."""

        from .training_view import TrainingViewDataset

        return TrainingViewDataset(self, artifact, **kwargs)


ShardedDamageDataset = TorchShardedDamageDataset


def validated_committed_shard_tensor_paths(
    dataset_directory: str | Path,
) -> tuple[dict, tuple[Path, ...]]:
    """Validate a completed sharded archive and return its committed tensors.

    This read-only entry point deliberately reuses the production loader's
    checkpoint, manifest, exact-inventory, partition-hash, tensor-axis, and
    cross-shard checks.  Audit tools must not rediscover shards with a glob.
    """

    archive = TorchShardedDamageDataset(
        dataset_directory,
        fields=("case_x",),
        include_static_graph=False,
        verify_integrity=True,
    )
    return dict(archive.manifest), tuple(path / "tensors.npz" for path in archive.shard_paths)


def _ordered_semantic_union(
    datasets: Sequence[TorchDamageDataset | TorchShardedDamageDataset],
    field: str,
) -> tuple[str, ...]:
    """Merge local axis orders without alphabetizing meaningful positions.

    Shared names retain every ordering constraint declared by their source
    archives.  Lexical order is used only to resolve names that never coexist,
    which keeps the frozen union independent of source-list order.
    """
    sequences = [tuple(dataset.metadata.get(field, ())) for dataset in datasets]
    successors = {value: set() for sequence in sequences for value in sequence}
    indegree = {value: 0 for value in successors}
    for sequence in sequences:
        for left, right in zip(sequence, sequence[1:]):
            if right not in successors[left]:
                successors[left].add(right)
                indegree[right] += 1

    ready = sorted(value for value, degree in indegree.items() if degree == 0)
    ordered: list[str] = []
    while ready:
        value = ready.pop(0)
        ordered.append(value)
        for successor in sorted(successors[value]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
        ready.sort()
    if len(ordered) != len(indegree):
        raise ValueError(f"datasets declare incompatible ordering for semantic axis {field}")
    return tuple(ordered)


class TorchMultiTargetDataset:
    """Map-style concatenation of target-specific tensor datasets.

    The wrapper keeps each source archive and its topology separate.  Samples
    carry private topology metadata consumed by :func:`collate_damage_batch`;
    this avoids pretending that local component, armor, output, or graph axes
    have the same meaning across targets.
    """

    def __init__(
        self,
        datasets: Sequence[
            TorchDamageDataset | TorchShardedDamageDataset | str | Path
        ],
        *,
        split: str | None = None,
        fields: Iterable[str] | None = None,
        include_static_graph: bool | None = None,
        copy_on_read: bool | None = None,
        verify_integrity: bool | None = None,
        allow_split_leakage: bool | None = None,
        allow_legacy_manifest: bool | None = None,
        purpose: str | None = None,
        evidence_root: str | Path | None = None,
        vocabulary: dict | None = None,
        allow_legacy_semantic_combination: bool = False,
    ):
        if not datasets:
            raise ValueError("datasets must contain at least one dataset")
        if split is not None and split not in {"train", "validation", "test"}:
            raise ValueError("split must be one of: train, validation, test")
        requested_fields = tuple(fields) if fields is not None else None
        sources = []
        for item in datasets:
            if isinstance(item, (TorchDamageDataset, TorchShardedDamageDataset)):
                if split is not None and item.split != split:
                    raise ValueError("existing Dataset split differs from the requested split; construct a matching filtered Dataset")
                effective_fields = item.sample_fields if item.sample_fields is not None else _default_sample_fields(
                    (item.manifest or {}).get("tensor_contract", {}).get("version"),
                    (item.manifest or {}).get("generation", {}).get("sampling", {}).get(
                        "sampling_envelope_version"
                    ),
                    (item.manifest or {}).get("model_input_contract", {}).get(
                        "tensor_field"
                    ),
                )
                if requested_fields is not None and requested_fields != effective_fields:
                    raise ValueError("existing Dataset fields differ from the requested fields")
                for name, requested in (
                    ("include_static_graph", include_static_graph), ("copy_on_read", copy_on_read),
                    ("verify_integrity", verify_integrity), ("allow_split_leakage", allow_split_leakage),
                    ("allow_legacy_manifest", allow_legacy_manifest), ("purpose", purpose),
                ):
                    if requested is not None and getattr(item, name) != requested:
                        raise ValueError(f"existing Dataset {name} differs from the requested setting")
                if evidence_root is not None and item.evidence_root != Path(evidence_root):
                    raise ValueError("existing Dataset evidence_root differs from the requested setting")
                sources.append(item)
            else:
                directory = Path(item)
                loader = TorchShardedDamageDataset if (directory / "checkpoint.json").exists() else TorchDamageDataset
                sources.append(loader(
                    directory, split=split, fields=requested_fields,
                    include_static_graph=include_static_graph,
                    copy_on_read=True if copy_on_read is None else copy_on_read,
                    verify_integrity=True if verify_integrity is None else verify_integrity,
                    allow_split_leakage=False if allow_split_leakage is None else allow_split_leakage,
                    allow_legacy_manifest=False if allow_legacy_manifest is None else allow_legacy_manifest,
                    purpose="software_development" if purpose is None else purpose,
                    evidence_root=evidence_root,
                ))
        self.datasets = tuple(sources)
        self.audit_overrides = ()
        semantic_sources = {}
        task_sources = {}
        for dataset in self.datasets:
            source = (dataset.manifest or {}).get("source_contracts", {})
            identity = (source.get("damage_tree_definition_hash"), source.get("label_spec_hash"),
                        content_hash(source["label_spec_document"]) if isinstance(source.get("label_spec_document"), dict) else None)
            version = (dataset.manifest or {}).get("tensor_contract", {}).get("version")
            for output in dataset.metadata.get("system_output_names", ()):
                if output in semantic_sources:
                    other_identity, other_version = semantic_sources[output]
                    unknown = any(value is None for value in identity + other_identity)
                    if unknown or identity != other_identity:
                        legacy_only = version in {None, "1.3.0", "1.4.0", "1.5.0"} and other_version in {None, "1.3.0", "1.4.0", "1.5.0"}
                        forensic = allow_legacy_semantic_combination and legacy_only and all(item.purpose == "software_development" for item in self.datasets)
                        if not forensic:
                            raise ValueError(f"same-name output {output!r} has incompatible or unknown task/event definitions")
                        self.audit_overrides = ("allow_legacy_semantic_combination=True",)
                semantic_sources[output] = (identity, version)
            document = source.get("label_spec_document")
            if isinstance(document, dict):
                for task_name, task_definition in document.get("tasks", {}).items():
                    task_identity = (identity[0], content_hash(task_definition))
                    if task_name in task_sources and task_sources[task_name] != task_identity:
                        raise ValueError(f"same-name task {task_name!r} has incompatible event/output definitions")
                    task_sources[task_name] = task_identity
        names = {
            name: _ordered_semantic_union(self.datasets, name)
            for name in SEMANTIC_NAME_FIELDS
        }
        self.semantic_vocabulary = validate_vocabulary(vocabulary) if vocabulary is not None else semantic_vocabulary(names)
        for name, local_values in names.items():
            if not set(local_values).issubset(self.semantic_vocabulary["names"][name]):
                raise ValueError(f"dataset contains names absent from the frozen {name} vocabulary")
        self.semantic_vocabulary_hash = self.semantic_vocabulary["content_hash"]
        self.label_task_names = _ordered_semantic_union(self.datasets, "label_task_names")
        self.collate_fn = DamageBatchCollator(self.semantic_vocabulary, task_names=self.label_task_names)
        self.offsets = np.cumsum(
            [0, *(len(dataset) for dataset in self.datasets)], dtype=np.int64
        )

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, item: int) -> dict:
        length = len(self)
        index = int(item)
        if index < 0:
            index += length
        if not 0 <= index < length:
            raise IndexError(item)
        dataset_index = bisect_right(self.offsets, index) - 1
        local_index = index - int(self.offsets[dataset_index])
        dataset = self.datasets[dataset_index]
        sample = dict(dataset[local_index])
        sample["_topology"] = {
            **dataset.topology_metadata,
            "dataset_index": dataset_index,
        }
        sample["target_hash"] = dataset.target_hash
        sample["_semantic_vocabulary"] = self.semantic_vocabulary
        sample["_label_task_names"] = self.label_task_names
        return sample

    def save_semantic_vocabulary(self, path: str | Path) -> Path:
        """Save the vocabulary digest to bind a consumer artifact to these axes."""
        destination = Path(path)
        destination.write_text(json.dumps(self.semantic_vocabulary, ensure_ascii=False, indent=2), encoding="utf-8")
        return destination


MultiTargetDamageDataset = TorchMultiTargetDataset


class DamageBatchCollator:
    """Pickle-safe collator bound to a saved, versioned semantic vocabulary."""

    def __init__(self, vocabulary: dict, *, task_names: Sequence[str] | None = None):
        self.vocabulary = validate_vocabulary(vocabulary)
        self.task_names = tuple(task_names) if task_names is not None else None

    def __call__(self, samples: Sequence[dict]) -> dict:
        return collate_damage_batch(samples, vocabulary=self.vocabulary, task_names=self.task_names)


def make_damage_collator(vocabulary: dict, *, task_names: Sequence[str] | None = None) -> DamageBatchCollator:
    return DamageBatchCollator(vocabulary, task_names=task_names)


def _as_numpy(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _to_tensor(value: np.ndarray):
    import torch

    return torch.from_numpy(np.ascontiguousarray(value))


def _pad_axis0(arrays: list[np.ndarray], maximum: int, *, fill_last: bool = False) -> np.ndarray:
    if not arrays:
        raise ValueError("cannot pad an empty array collection")
    tail = arrays[0].shape[1:]
    if any(array.shape[1:] != tail for array in arrays):
        raise ValueError("tensor feature axes differ across target datasets")
    output = np.zeros((len(arrays), maximum, *tail), dtype=arrays[0].dtype)
    for index, array in enumerate(arrays):
        output[index, : len(array)] = array
        if fill_last and len(array) and len(array) < maximum:
            output[index, len(array) :] = array[-1]
    return output


def _pad_component_state_field(
    samples: list[dict],
    topologies: list[dict],
    field: str,
    component_count: int,
    mechanisms: tuple[str, ...],
    states: tuple[str, ...],
) -> np.ndarray:
    first = _as_numpy(samples[0][field])
    tail = first.shape[3:]
    output = np.zeros(
        (len(samples), component_count, len(mechanisms), len(states), *tail),
        dtype=first.dtype,
    )
    mechanism_index = {name: index for index, name in enumerate(mechanisms)}
    state_index = {name: index for index, name in enumerate(states)}
    for batch_index, (sample, topology) in enumerate(zip(samples, topologies)):
        value = _as_numpy(sample[field])
        local_mechanisms = topology.get("mechanism_names", ())
        local_states = topology.get("state_names", ())
        for local_mechanism, mechanism in enumerate(local_mechanisms):
            for local_state, state in enumerate(local_states):
                output[
                    batch_index,
                    : value.shape[0],
                    mechanism_index[str(mechanism)],
                    state_index[str(state)],
                ] = value[:, local_mechanism, local_state]
    return output


def _pad_component_frechet_field(
    samples: list[dict],
    topologies: list[dict],
    field: str,
    component_count: int,
    states: tuple[str, ...],
) -> np.ndarray:
    first = _as_numpy(samples[0][field])
    tail = first.shape[2:]
    output = np.zeros(
        (len(samples), component_count, len(states), *tail),
        dtype=first.dtype,
    )
    state_index = {name: index for index, name in enumerate(states)}
    for batch_index, (sample, topology) in enumerate(zip(samples, topologies)):
        value = _as_numpy(sample[field])
        local_states = topology.get("state_names", ())
        for local_state, state in enumerate(local_states):
            output[
                batch_index,
                : value.shape[0],
                state_index[str(state)],
            ] = value[:, local_state]
    return output


def _align_named_axis(
    samples: list[dict],
    topologies: list[dict],
    field: str,
    local_names_key: str,
    global_names: tuple[str, ...],
) -> np.ndarray:
    first = _as_numpy(samples[0][field])
    tail = first.shape[1:]
    output = np.zeros((len(samples), len(global_names), *tail), dtype=first.dtype)
    global_index = {name: index for index, name in enumerate(global_names)}
    for batch_index, (sample, topology) in enumerate(zip(samples, topologies)):
        value = _as_numpy(sample[field])
        local_names = topology.get(local_names_key, ())
        if len(local_names) != len(value):
            raise ValueError(
                f"{field!r} length does not match topology field {local_names_key!r}"
            )
        for local_index, name in enumerate(local_names):
            output[batch_index, global_index[str(name)]] = value[local_index]
    return output


def collate_damage_batch(samples: Sequence[dict], *, vocabulary: dict | None = None, task_names: Sequence[str] | None = None) -> dict:
    """Pad and align samples from one or more target topologies.

    Component and armor axes are padded with explicit presence masks. System
    outputs and evaluation standards are aligned by ID rather than local
    position. Damage-tree graphs are combined as a disjoint graph by applying
    cumulative node offsets to every edge.
    """

    if not samples:
        raise ValueError("samples must contain at least one item")
    items = [dict(sample) for sample in samples]
    if any("_topology" not in sample for sample in items):
        raise ValueError(
            "cross-target collation requires samples from TorchMultiTargetDataset"
        )
    topologies = [dict(sample.pop("_topology")) for sample in items]
    attached = [sample.pop("_semantic_vocabulary", None) for sample in items]
    attached_tasks = [sample.pop("_label_task_names", None) for sample in items]
    if task_names is None and any("label_task_available_mask" in sample for sample in items):
        if any(value is None for value in attached_tasks) or any(value != attached_tasks[0] for value in attached_tasks):
            raise ValueError("task masks require a shared frozen task vocabulary")
        task_names = attached_tasks[0]
    tasks = tuple(task_names or ())
    if len(tasks) != len(set(tasks)) or any(not isinstance(name, str) or not name for name in tasks):
        raise ValueError("task vocabulary requires unique nonempty names")
    if vocabulary is None:
        if any(value is None for value in attached):
            raise ValueError("collation requires a frozen dataset vocabulary; use make_damage_collator")
        vocabulary = attached[0]
    frozen = validate_vocabulary(vocabulary)
    if any(value is not None and validate_vocabulary(value) != frozen for value in attached):
        raise ValueError("samples belong to different frozen semantic vocabularies")
    for topology in topologies:
        for name in SEMANTIC_NAME_FIELDS:
            local_names = tuple(map(str, topology.get(name, ())))
            if len(set(local_names)) != len(local_names) or not set(local_names).issubset(frozen["names"][name]):
                raise ValueError(f"sample contains duplicate or unknown {name}")
    component_counts = [len(topology.get("component_ids", ())) for topology in topologies]
    armor_counts = [len(topology.get("armor_ids", ())) for topology in topologies]
    maximum_components = max(component_counts, default=0)
    maximum_armor = max(armor_counts, default=0)
    component_presence = np.zeros((len(items), maximum_components), dtype=bool)
    armor_presence = np.zeros((len(items), maximum_armor), dtype=bool)
    for index, count in enumerate(component_counts):
        component_presence[index, :count] = True
    for index, count in enumerate(armor_counts):
        armor_presence[index, :count] = True

    output: dict[str, object] = {
        "semantic_vocabulary_hash": frozen["content_hash"],
        "component_presence_mask": _to_tensor(component_presence),
        "armor_presence_mask": _to_tensor(armor_presence),
        "case_id": [str(sample.get("case_id", "")) for sample in items],
        "case_hash": [str(sample.get("case_hash", "")) for sample in items],
        "split": [str(sample.get("split", "")) for sample in items],
        "split_group_id": [str(sample.get("split_group_id", "")) for sample in items],
        "epistemic_sample_id": [
            str(sample.get("epistemic_sample_id", "")) for sample in items
        ],
        "sampling_candidate_id": [
            str(sample.get("sampling_candidate_id", "")) for sample in items
        ],
        "sampling_lineage_id": [
            str(sample.get("sampling_lineage_id", "")) for sample in items
        ],
        "sampling_epistemic_group_id": [str(sample.get("sampling_epistemic_group_id", "")) for sample in items],
        "sampling_acquisition_model_hash": [str(sample.get("sampling_acquisition_model_hash", "")) for sample in items],
        "sampling_proposal_hash": [str(sample.get("sampling_proposal_hash", "")) for sample in items],
        "sampling_round_lineage_id": [str(sample.get("sampling_round_lineage_id", "")) for sample in items],
        "sampling_selected_case_set_hash": [str(sample.get("sampling_selected_case_set_hash", "")) for sample in items],
        "sampling_role": [str(sample.get("sampling_role", "")) for sample in items],
        "stream_role": [str(sample.get("stream_role", "")) for sample in items],
        "sampling_stratum": [str(sample.get("sampling_stratum", "")) for sample in items],
        "target_hash": [str(sample.get("target_hash", "")) for sample in items],
        "target_dataset_index": _to_tensor(
            np.asarray(
                [int(topology.get("dataset_index", -1)) for topology in topologies],
                dtype=np.int64,
            )
        ),
        "component_ids": [tuple(topology.get("component_ids", ())) for topology in topologies],
        "armor_ids": [tuple(topology.get("armor_ids", ())) for topology in topologies],
    }

    shared_axis_metadata = (
        "case_feature_names", "case_feature_units",
        "case_physics_feature_names", "case_physics_feature_units",
        *MODEL_INPUT_METADATA_FIELDS,
        "solver_context_feature_names", "solver_context_feature_units",
        "sampling_stratum_names", "sampling_role_names",
        "component_feature_names", "component_feature_units",
        "armor_feature_names", "armor_feature_units",
        "armor_label_names", "armor_label_units",
        "armor_path_feature_names", "armor_path_feature_units",
        "component_aux_label_names", "component_aux_label_units",
        "frechet_bound_names", "frechet_bound_units",
    )
    for name in shared_axis_metadata:
        values = [tuple(topology.get(name, ())) for topology in topologies]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(
                f"cross-target collation requires a shared {name!r} contract"
            )
        output[name] = values[0]

    handled = {
        "case_id", "case_hash", "split", "split_group_id",
        "epistemic_sample_id", "target_hash", "sampling_candidate_id",
        "sampling_lineage_id", "sampling_role", "stream_role", "sampling_stratum",
        "sampling_epistemic_group_id", "sampling_acquisition_model_hash",
        "sampling_proposal_hash", "sampling_round_lineage_id",
        "sampling_selected_case_set_hash",
    }
    component_fields = (
        "component_x", "component_feature_mask", "component_aux_y",
        "component_aux_y_stddev", "component_aux_y_ci95",
        "component_aux_label_mask",
        "component_aux_uncertainty_mask", "component_aux_sample_count",
    )
    for field in component_fields:
        if all(field in sample for sample in items):
            arrays = [_as_numpy(sample[field]) for sample in items]
            output[field] = _to_tensor(_pad_axis0(arrays, maximum_components))
            handled.add(field)

    armor_fields = (
        "armor_x", "armor_feature_mask", "armor_y", "armor_y_stddev",
        "armor_y_ci95", "armor_label_mask",
        "armor_uncertainty_mask", "armor_sample_count",
    )
    for field in armor_fields:
        if all(field in sample for sample in items):
            arrays = [_as_numpy(sample[field]) for sample in items]
            output[field] = _to_tensor(_pad_axis0(arrays, maximum_armor))
            handled.add(field)

    path_fields = ("component_armor_path_x", "component_armor_path_mask")
    for field in path_fields:
        if all(field in sample for sample in items):
            first = _as_numpy(items[0][field])
            tail = first.shape[2:]
            padded = np.zeros(
                (len(items), maximum_components, maximum_armor, *tail),
                dtype=first.dtype,
            )
            for index, sample in enumerate(items):
                value = _as_numpy(sample[field])
                if value.shape[2:] != tail:
                    raise ValueError(f"{field!r} feature axes differ across targets")
                padded[index, : value.shape[0], : value.shape[1]] = value
            output[field] = _to_tensor(padded)
            handled.add(field)

    mechanisms = tuple(frozen["names"]["mechanism_names"])
    states = tuple(frozen["names"]["state_names"])
    for field in (
        "component_y", "component_y_stddev", "component_y_ci95",
        "component_label_mask",
        "component_uncertainty_mask", "component_sample_count",
    ):
        if all(field in sample for sample in items):
            output[field] = _to_tensor(
                _pad_component_state_field(
                    items,
                    topologies,
                    field,
                    maximum_components,
                    mechanisms,
                    states,
                )
            )
            handled.add(field)
    for field in (
        "component_frechet_y", "component_frechet_y_stddev",
        "component_frechet_y_ci95", "component_frechet_label_mask",
        "component_frechet_uncertainty_mask", "component_frechet_sample_count",
    ):
        if all(field in sample for sample in items):
            output[field] = _to_tensor(
                _pad_component_frechet_field(
                    items,
                    topologies,
                    field,
                    maximum_components,
                    states,
                )
            )
            handled.add(field)
    output["mechanism_names"] = mechanisms
    output["state_names"] = states

    system_outputs = tuple(frozen["names"]["system_output_names"])
    for field in ("system_y", "system_y_stddev", "system_y_ci95", "system_label_mask", "system_uncertainty_mask", "system_sample_count", "system_source_available_mask"):
        if all(field in sample for sample in items):
            output[field] = _to_tensor(
                _align_named_axis(
                    items,
                    topologies,
                    field,
                    "system_output_names",
                    system_outputs,
                )
            )
            handled.add(field)
    output["system_output_names"] = system_outputs

    standards = tuple(frozen["names"]["evaluation_standard_ids"])
    if all("requested_standard_mask" in sample for sample in items):
        output["requested_standard_mask"] = _to_tensor(
            _align_named_axis(
                items,
                topologies,
                "requested_standard_mask",
                "evaluation_standard_ids",
                standards,
            )
        )
        handled.add("requested_standard_mask")
    if all("standard_output_mask" in sample for sample in items):
        standard_mask = np.zeros(
            (len(items), len(standards), len(system_outputs)), dtype=bool
        )
        standard_index = {name: index for index, name in enumerate(standards)}
        output_index = {name: index for index, name in enumerate(system_outputs)}
        for batch_index, (sample, topology) in enumerate(zip(items, topologies)):
            local = _as_numpy(sample["standard_output_mask"])
            local_standards = topology.get("evaluation_standard_ids", ())
            local_outputs = topology.get("system_output_names", ())
            for local_standard, standard in enumerate(local_standards):
                for local_output, output_name in enumerate(local_outputs):
                    standard_mask[
                        batch_index,
                        standard_index[str(standard)],
                        output_index[str(output_name)],
                    ] = bool(local[local_standard, local_output])
        output["standard_output_mask"] = _to_tensor(standard_mask)
        handled.add("standard_output_mask")
    output["evaluation_standard_ids"] = standards
    if all("label_task_available_mask" in sample for sample in items):
        for topology in topologies:
            if not set(topology.get("label_task_names", ())).issubset(tasks):
                raise ValueError("sample contains tasks absent from the frozen task vocabulary")
        output["label_task_available_mask"] = _to_tensor(_align_named_axis(
            items, topologies, "label_task_available_mask", "label_task_names", tasks,
        ))
        output["label_task_names"] = tasks
        handled.add("label_task_available_mask")

    if all("tree_edge_index" in sample for sample in items):
        edge_parts: list[np.ndarray] = []
        edge_types: list[np.ndarray] = []
        graph_batch: list[int] = []
        node_offsets = [0]
        graph_node_ids: list[str] = []
        graph_node_types: list[str] = []
        for batch_index, (sample, topology) in enumerate(zip(items, topologies)):
            node_ids = tuple(map(str, topology.get("graph_node_ids", ())))
            node_types = tuple(map(str, topology.get("graph_node_types", ())))
            if len(node_ids) != len(node_types):
                raise ValueError("graph node IDs and node types have different lengths")
            offset = node_offsets[-1]
            edge = _as_numpy(sample["tree_edge_index"]).astype(np.int64, copy=False)
            edge_parts.append(edge + offset)
            edge_types.append(_as_numpy(sample["tree_edge_type"]))
            graph_batch.extend([batch_index] * len(node_ids))
            graph_node_ids.extend(
                f"{topology.get('target_hash', '')}:{node_id}" for node_id in node_ids
            )
            graph_node_types.extend(node_types)
            node_offsets.append(offset + len(node_ids))
        output["tree_edge_index"] = _to_tensor(
            np.concatenate(edge_parts, axis=1)
            if edge_parts
            else np.zeros((2, 0), dtype=np.int64)
        )
        output["tree_edge_type"] = _to_tensor(
            np.concatenate(edge_types)
            if edge_types
            else np.zeros((0,), dtype=np.int16)
        )
        output["graph_batch"] = _to_tensor(np.asarray(graph_batch, dtype=np.int64))
        output["graph_node_offsets"] = _to_tensor(
            np.asarray(node_offsets, dtype=np.int64)
        )
        output["graph_node_ids"] = tuple(graph_node_ids)
        output["graph_node_types"] = tuple(graph_node_types)
        handled.update(("tree_edge_index", "tree_edge_type"))

    for vertices_name, offsets_name, mask_name in (
        ("armor_vertices_m", "armor_vertex_offsets", "armor_vertex_mask"),
        ("component_vertices_m", "component_vertex_offsets", "component_vertex_mask"),
    ):
        if all(vertices_name in sample for sample in items):
            arrays = [_as_numpy(sample[vertices_name]) for sample in items]
            maximum = max(map(len, arrays), default=0)
            output[vertices_name] = _to_tensor(_pad_axis0(arrays, maximum))
            mask = np.zeros((len(items), maximum), dtype=bool)
            for index, array in enumerate(arrays):
                mask[index, : len(array)] = True
            output[mask_name] = _to_tensor(mask)
            handled.add(vertices_name)
        if all(offsets_name in sample for sample in items):
            arrays = [_as_numpy(sample[offsets_name]) for sample in items]
            maximum = max(map(len, arrays), default=0)
            output[offsets_name] = _to_tensor(
                _pad_axis0(arrays, maximum, fill_last=True)
            )
            handled.add(offsets_name)

    common_numeric = set.intersection(
        *(set(sample) for sample in items)
    ) - handled - {"_topology"}
    for field in sorted(common_numeric):
        values = [_as_numpy(sample[field]) for sample in items]
        if all(value.dtype.kind not in {"U", "S", "O"} for value in values):
            if all(value.shape == values[0].shape for value in values):
                output[field] = _to_tensor(np.stack(values))

    return output
