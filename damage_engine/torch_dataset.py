from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .utils import file_hash


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
)

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
)


class TorchDamageDataset:
    """Efficient, spawn-safe map-style view of ``tensors.npz``.

    Arrays are decompressed once per process instead of once per sample. The
    object drops its in-memory arrays when pickled, so Windows ``DataLoader``
    workers reopen the archive independently rather than inheriting an
    unpickleable ``NpzFile`` handle.
    """

    def __init__(
        self,
        dataset_directory: str | Path,
        *,
        split: str | None = None,
        fields: Iterable[str] | None = None,
        include_static_graph: bool = True,
        copy_on_read: bool = False,
    ):
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "TorchDamageDataset requires PyTorch; install the project with the 'torch' extra"
            ) from exc
        self.directory = Path(dataset_directory)
        self.archive_path = self.directory / "tensors.npz"
        if (self.directory / ".dataset.incomplete").exists():
            raise RuntimeError(
                "dataset generation is incomplete; rerun generation before training"
            )
        if not self.archive_path.exists():
            raise FileNotFoundError(self.archive_path)
        self.copy_on_read = bool(copy_on_read)
        self.include_static_graph = bool(include_static_graph)
        self.sample_fields = tuple(fields) if fields is not None else _DEFAULT_SAMPLE_FIELDS
        self._arrays: dict[str, np.ndarray] | None = None
        self._load_arrays()
        arrays = self._require_arrays()
        if "case_x" not in arrays:
            raise ValueError("tensors.npz is missing required field 'case_x'")
        batch_size = len(arrays["case_x"])
        for name in self.sample_fields:
            if name not in arrays:
                raise ValueError(f"tensors.npz is missing requested field {name!r}")
            if len(arrays[name]) != batch_size:
                raise ValueError(f"tensor field {name!r} does not share the case axis")
        self.indices = np.arange(batch_size, dtype=np.int64)
        if split is not None:
            if split not in {"train", "validation", "test"}:
                raise ValueError("split must be one of: train, validation, test")
            split_names = self._split_names(arrays)
            self.indices = np.flatnonzero(split_names == split).astype(np.int64, copy=False)
        self.split = split
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict | None:
        path = self.directory / "manifest.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_arrays(self) -> None:
        names = set(self.sample_fields) | set(_STRING_SAMPLE_FIELDS) | {
            "case_x",
            "case_feature_names",
            "case_feature_units",
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
            self._arrays = {
                name: np.asarray(archive[name]) for name in names if name in available
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
            }
        )
        if self.include_static_graph:
            for name in _STATIC_GRAPH_FIELDS:
                if name in arrays:
                    sample[name] = tensor(arrays[name])
        for name in _STATIC_STANDARD_FIELDS:
            if name in arrays:
                sample[name] = tensor(arrays[name])
        return sample

    @property
    def metadata(self) -> dict[str, tuple[str, ...]]:
        arrays = self._require_arrays()
        names = (
            "case_feature_names", "case_feature_units",
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
        return {
            name: tuple(map(str, arrays[name].tolist()))
            for name in names
            if name in arrays
        }

    @property
    def target_hash(self) -> str:
        if not self.manifest:
            return ""
        hashes = self.manifest.get("source_contracts", {}).get("target_hashes", [])
        return str(hashes[0]) if len(hashes) == 1 else ""

    @property
    def topology_metadata(self) -> dict[str, object]:
        return {**self.metadata, "target_hash": self.target_hash}


class TorchShardedDamageDataset:
    """Lazy, spawn-safe map-style view of a completed sharded dataset.

    Initialization reads only the split vector from each shard. Full shard
    arrays are opened through :class:`TorchDamageDataset` on first access and
    retained in a bounded least-recently-used cache. Pickling intentionally
    drops that cache so Windows ``DataLoader`` workers reopen shards in their
    own processes.
    """

    def __init__(
        self,
        dataset_directory: str | Path,
        *,
        split: str | None = None,
        fields: Iterable[str] | None = None,
        include_static_graph: bool = True,
        copy_on_read: bool = False,
        max_cached_shards: int = 2,
        verify_integrity: bool = True,
        allow_split_leakage: bool = False,
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
        self.sample_fields = tuple(fields) if fields is not None else None
        self.include_static_graph = bool(include_static_graph)
        self.copy_on_read = bool(copy_on_read)
        self.max_cached_shards = int(max_cached_shards)
        self.verify_integrity = bool(verify_integrity)
        self.allow_split_leakage = bool(allow_split_leakage)
        self._cache: OrderedDict[int, TorchDamageDataset] = OrderedDict()

        if (self.directory / ".dataset.incomplete").exists():
            raise RuntimeError(
                "dataset generation is incomplete; rerun or resume generation before training"
            )
        checkpoint_path = self.directory / "checkpoint.json"
        manifest_path = self.directory / "manifest.json"
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        self.checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if self.checkpoint.get("status") != "complete":
            raise RuntimeError(
                "sharded dataset checkpoint is not complete; resume generation before training"
            )
        if self.verify_integrity:
            expected_root_hash = self.checkpoint.get("root_manifest_hash")
            if (
                not isinstance(expected_root_hash, str)
                or file_hash(manifest_path) != expected_root_hash
            ):
                raise RuntimeError(
                    "root manifest is missing its checkpointed hash or has changed"
                )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint_dataset_id = self.checkpoint.get("dataset_id")
        manifest_dataset_id = self.manifest.get("dataset_id")
        if not checkpoint_dataset_id or checkpoint_dataset_id != manifest_dataset_id:
            raise RuntimeError(
                "checkpoint and root manifest dataset identities do not match"
            )
        leakage_passed = self.manifest.get("split_policy", {}).get(
            "leakage_audit_passed"
        )
        if leakage_passed is not True and not self.allow_split_leakage:
            raise RuntimeError(
                "dataset failed the global split-leakage audit; pass "
                "allow_split_leakage=True only for forensic inspection"
            )

        records = sorted(
            self.checkpoint.get("completed_shards", []),
            key=lambda record: int(record.get("index", -1)),
        )
        if not records:
            raise ValueError("checkpoint contains no completed shards")
        indices = [int(record.get("index", -1)) for record in records]
        if indices != list(range(len(records))):
            raise ValueError("completed shard indices must be contiguous and zero-based")

        root = self.directory.resolve()
        shard_paths: list[Path] = []
        counts: list[int] = []
        for record in records:
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
            count = self._inspect_shard(shard_path, record)
            shard_paths.append(shard_path)
            counts.append(count)

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

    def _inspect_shard(self, shard_path: Path, record: dict) -> int:
        if (shard_path / ".dataset.incomplete").exists():
            raise RuntimeError(f"shard generation is incomplete: {shard_path}")
        archive_path = shard_path / "tensors.npz"
        manifest_path = shard_path / "manifest.json"
        if not archive_path.exists():
            raise FileNotFoundError(archive_path)
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        if self.verify_integrity:
            expected_manifest_hash = record.get("manifest_hash")
            if (
                not isinstance(expected_manifest_hash, str)
                or file_hash(manifest_path) != expected_manifest_hash
            ):
                raise RuntimeError(
                    f"shard manifest is missing its checkpointed hash or has changed: {shard_path}"
                )
        shard_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if shard_manifest.get("dataset_id") != record.get("dataset_id"):
            raise RuntimeError(
                f"checkpoint and shard manifest identities do not match: {shard_path}"
            )
        if self.verify_integrity:
            for partition in shard_manifest.get("partitions", []):
                path_glob = partition.get("path_glob")
                expected_hash = partition.get("content_hash")
                if not isinstance(path_glob, str) or not isinstance(expected_hash, str):
                    raise RuntimeError(
                        f"shard partition is missing integrity metadata: {shard_path}"
                    )
                matches = sorted(shard_path.glob(path_glob))
                if len(matches) != 1 or file_hash(matches[0]) != expected_hash:
                    raise RuntimeError(
                        f"shard partition is missing or has changed: {shard_path / path_glob}"
                    )
        with np.load(archive_path, allow_pickle=False) as archive:
            if "case_x" not in archive.files:
                raise ValueError(f"{archive_path} is missing required field 'case_x'")
            if self.split is None:
                valid_count = shard_manifest.get("quality_summary", {}).get("n_valid")
                if not isinstance(valid_count, int) or valid_count < 0:
                    raise ValueError(
                        f"{manifest_path} is missing a valid quality_summary.n_valid"
                    )
                return valid_count
            if "split_names" not in archive.files:
                raise ValueError(
                    f"{archive_path} is missing 'split_names'; regenerate the sharded dataset"
                )
            split_names = np.asarray(archive["split_names"]).astype(str)
            valid_count = shard_manifest.get("quality_summary", {}).get("n_valid")
            if not isinstance(valid_count, int) or len(split_names) != valid_count:
                raise ValueError(
                    f"{archive_path} split_names does not match the valid case axis"
                )
            return int(np.count_nonzero(split_names == self.split))

    def _open_shard(self, shard_index: int) -> TorchDamageDataset:
        if shard_index in self._cache:
            dataset = self._cache.pop(shard_index)
            self._cache[shard_index] = dataset
            return dataset
        dataset = TorchDamageDataset(
            self.shard_paths[shard_index],
            split=self.split,
            fields=self.sample_fields,
            include_static_graph=self.include_static_graph,
            copy_on_read=self.copy_on_read,
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
        self.__dict__.update(state)
        self._cache = OrderedDict()

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
    def metadata(self) -> dict[str, tuple[str, ...]]:
        return self._open_shard(0).metadata

    @property
    def target_hash(self) -> str:
        hashes = self.manifest.get("source_contracts", {}).get("target_hashes", [])
        return str(hashes[0]) if len(hashes) == 1 else ""

    @property
    def topology_metadata(self) -> dict[str, object]:
        return {**self.metadata, "target_hash": self.target_hash}


ShardedDamageDataset = TorchShardedDamageDataset


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
        include_static_graph: bool = True,
        copy_on_read: bool = False,
    ):
        if not datasets:
            raise ValueError("datasets must contain at least one dataset")
        self.datasets: tuple[TorchDamageDataset, ...] = tuple(
            item
            if isinstance(item, (TorchDamageDataset, TorchShardedDamageDataset))
            else TorchDamageDataset(
                item,
                split=split,
                fields=fields,
                include_static_graph=include_static_graph,
                copy_on_read=copy_on_read,
            )
            for item in datasets
        )
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
        return sample


MultiTargetDamageDataset = TorchMultiTargetDataset


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


def _ordered_union(values: Iterable[Iterable[str]]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for sequence in values:
        for item in sequence:
            text = str(item)
            if text not in seen:
                seen.add(text)
                result.append(text)
    return tuple(result)


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


def collate_damage_batch(samples: Sequence[dict]) -> dict:
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
        "component_presence_mask": _to_tensor(component_presence),
        "armor_presence_mask": _to_tensor(armor_presence),
        "case_id": [str(sample.get("case_id", "")) for sample in items],
        "case_hash": [str(sample.get("case_hash", "")) for sample in items],
        "split": [str(sample.get("split", "")) for sample in items],
        "split_group_id": [str(sample.get("split_group_id", "")) for sample in items],
        "epistemic_sample_id": [
            str(sample.get("epistemic_sample_id", "")) for sample in items
        ],
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
        "epistemic_sample_id", "target_hash",
    }
    component_fields = (
        "component_x", "component_feature_mask", "component_aux_y",
        "component_aux_y_stddev", "component_aux_y_ci95",
        "component_aux_label_mask",
    )
    for field in component_fields:
        if all(field in sample for sample in items):
            arrays = [_as_numpy(sample[field]) for sample in items]
            output[field] = _to_tensor(_pad_axis0(arrays, maximum_components))
            handled.add(field)

    armor_fields = (
        "armor_x", "armor_feature_mask", "armor_y", "armor_y_stddev",
        "armor_y_ci95", "armor_label_mask",
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

    mechanisms = _ordered_union(
        topology.get("mechanism_names", ()) for topology in topologies
    )
    states = _ordered_union(topology.get("state_names", ()) for topology in topologies)
    for field in (
        "component_y", "component_y_stddev", "component_y_ci95",
        "component_label_mask",
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

    system_outputs = _ordered_union(
        topology.get("system_output_names", ()) for topology in topologies
    )
    for field in ("system_y", "system_y_stddev", "system_y_ci95", "system_label_mask"):
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

    standards = _ordered_union(
        topology.get("evaluation_standard_ids", ()) for topology in topologies
    )
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
