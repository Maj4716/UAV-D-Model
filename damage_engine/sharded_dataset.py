from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from itertools import islice
import json
from pathlib import Path
import shutil
from typing import Iterable, Iterator

import numpy as np

from .dataset import DatasetWriter, dataset_identity_hash
from .engine import ENGINE_CODE_HASH, DamageEngine
from .errors import InputValidationError
from .utils import content_hash, file_hash


CHECKPOINT_SCHEMA_VERSION = "1.0.0"


def _chunks(values: Iterable[dict], size: int) -> Iterator[list[dict]]:
    iterator = iter(values)
    while True:
        chunk = list(islice(iterator, size))
        if not chunk:
            return
        yield chunk


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class ShardedDatasetWriter:
    """Bounded-memory dataset generation with deterministic resume.

    Each shard is a complete DatasetWriter product, so existing Parquet/NPZ
    consumers remain usable.  The root manifest aggregates their partition
    globs and hashes; ``checkpoint.json`` records the exact ordered case hashes
    used for resume validation.
    """

    def __init__(
        self,
        output_directory: str | Path,
        *,
        shard_size: int,
        split_seed: int = 0,
        split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        tensor_compression: str = "compressed",
    ):
        if int(shard_size) < 1:
            raise ValueError("shard_size must be at least one")
        self.output_directory = Path(output_directory)
        self.shard_size = int(shard_size)
        self.split_seed = int(split_seed)
        self.split_ratios = tuple(float(value) for value in split_ratios)
        self.tensor_compression = tensor_compression
        # Reuse the canonical validation performed by DatasetWriter.
        DatasetWriter(
            self.output_directory / ".configuration-check",
            split_seed=self.split_seed,
            split_ratios=self.split_ratios,
            tensor_compression=self.tensor_compression,
        )
        configuration_check = self.output_directory / ".configuration-check"
        if configuration_check.exists():
            configuration_check.rmdir()

    def _identity(
        self,
        engine: DamageEngine,
        *,
        fail_closed: bool,
        workers: int,
        chunksize: int,
    ) -> dict:
        """Checkpoint identity, including execution settings used by every shard.

        This identity is deliberately stricter than the logical dataset id: it
        prevents a resumed run from publishing misleading mixed execution
        provenance, while the final dataset id remains independent of storage
        layout and worker count.
        """
        value = {
            "target_hash": engine.target_hash,
            "physics_registry_hash": engine.registry.hash,
            "engine_code_hash": ENGINE_CODE_HASH,
            "split_seed": self.split_seed,
            "split_ratios": list(self.split_ratios),
            "tensor_compression": self.tensor_compression,
            "shard_size": self.shard_size,
            "failure_policy": "fail_fast" if fail_closed else "isolate_and_record",
            "workers": int(workers),
            "chunksize": int(chunksize),
        }
        value["identity_hash"] = content_hash(value)
        return value

    def _new_checkpoint(self, identity: dict) -> dict:
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "status": "in_progress",
            "identity": identity,
            "completed_shards": [],
        }

    def _validate_checkpoint_structure(self, checkpoint: object) -> dict:
        if not isinstance(checkpoint, dict):
            raise InputValidationError("checkpoint root must be a JSON object")
        if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise InputValidationError(
                "checkpoint schema version is not supported",
                context={"schema_version": checkpoint.get("schema_version")},
            )
        if checkpoint.get("status") not in {"in_progress", "complete"}:
            raise InputValidationError(
                "checkpoint status is invalid",
                context={"status": checkpoint.get("status")},
            )
        if not isinstance(checkpoint.get("identity"), dict):
            raise InputValidationError("checkpoint identity must be an object")
        records = checkpoint.get("completed_shards")
        if not isinstance(records, list):
            raise InputValidationError("checkpoint completed_shards must be an array")
        indices: list[int] = []
        required = {
            "index",
            "path",
            "case_ids",
            "case_hashes",
            "manifest_hash",
            "dataset_id",
        }
        for position, record in enumerate(records):
            if not isinstance(record, dict) or not required.issubset(record):
                raise InputValidationError(
                    "checkpoint shard record is incomplete",
                    context={"record_position": position},
                )
            index = record.get("index")
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise InputValidationError(
                    "checkpoint shard index must be a non-negative integer",
                    context={"record_position": position},
                )
            expected_path = f"shards/part-{index:06d}"
            if record.get("path") != expected_path:
                raise InputValidationError(
                    "checkpoint shard path is not canonical",
                    context={"index": index, "expected_path": expected_path},
                )
            case_ids = record.get("case_ids")
            case_hashes = record.get("case_hashes")
            if (
                not isinstance(case_ids, list)
                or not isinstance(case_hashes, list)
                or not case_ids
                or len(case_ids) != len(case_hashes)
                or len(case_ids) > self.shard_size
            ):
                raise InputValidationError(
                    "checkpoint shard case lists are invalid",
                    context={"index": index},
                )
            indices.append(index)
        if indices != list(range(len(records))):
            raise InputValidationError(
                "checkpoint shard indices must be unique, ordered and contiguous",
                context={"indices": indices},
            )
        if checkpoint["status"] == "complete" and (
            not checkpoint.get("dataset_id")
            or not checkpoint.get("root_manifest_hash")
        ):
            raise InputValidationError(
                "completed checkpoint lacks root dataset identity or manifest hash"
            )
        return checkpoint

    def _load_checkpoint(self, identity: dict, *, resume: bool) -> dict:
        checkpoint_path = self.output_directory / "checkpoint.json"
        if resume:
            if not checkpoint_path.exists():
                raise InputValidationError(
                    "resume requested but checkpoint.json does not exist",
                    context={"output_directory": str(self.output_directory)},
                )
            checkpoint = self._validate_checkpoint_structure(
                json.loads(checkpoint_path.read_text(encoding="utf-8"))
            )
            if checkpoint.get("identity") != identity:
                raise InputValidationError(
                    "resume configuration does not match the checkpoint identity",
                    context={
                        "expected_identity_hash": checkpoint.get("identity", {}).get("identity_hash"),
                        "observed_identity_hash": identity["identity_hash"],
                    },
                )
            return checkpoint
        if self.output_directory.exists() and any(self.output_directory.iterdir()):
            raise InputValidationError(
                "sharded output directory must be empty unless --resume is used",
                context={"output_directory": str(self.output_directory)},
            )
        self.output_directory.mkdir(parents=True, exist_ok=True)
        checkpoint = self._new_checkpoint(identity)
        _write_json_atomic(checkpoint_path, checkpoint)
        return checkpoint

    @staticmethod
    def _verify_completed_shard(record: dict, final_directory: Path) -> dict:
        manifest_path = final_directory / "manifest.json"
        if not manifest_path.exists() or file_hash(manifest_path) != record.get("manifest_hash"):
            raise InputValidationError(
                "completed shard manifest is missing or has changed",
                context={"shard_index": record.get("index"), "path": str(manifest_path)},
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset_id") != record.get("dataset_id"):
            raise InputValidationError(
                "completed shard dataset identity has changed",
                context={"shard_index": record.get("index")},
            )
        for partition in manifest.get("partitions", []):
            paths = sorted(final_directory.glob(str(partition["path_glob"])))
            if len(paths) != 1 or file_hash(paths[0]) != partition.get("content_hash"):
                raise InputValidationError(
                    "completed shard partition is missing or has changed",
                    context={
                        "shard_index": record.get("index"),
                        "partition": partition.get("name"),
                        "path_glob": partition.get("path_glob"),
                    },
                )
        return manifest

    def _validate_completed_resume(
        self,
        cases: Iterable[dict],
        completed: dict[int, dict],
        shards_directory: Path,
        checkpoint: dict,
    ) -> dict:
        seen_case_ids: set[str] = set()
        observed_shards = 0
        for shard_index, case_chunk in enumerate(_chunks(cases, self.shard_size)):
            observed_shards += 1
            if shard_index not in completed:
                raise InputValidationError(
                    "a completed sharded dataset is immutable and cannot be extended by resume",
                    context={"unexpected_shard_index": shard_index},
                )
            case_ids = [str(case.get("case_id") or "") for case in case_chunk]
            if any(not value for value in case_ids) or any(
                value in seen_case_ids for value in case_ids
            ) or len(case_ids) != len(set(case_ids)):
                raise InputValidationError(
                    "dataset case_id values must be globally unique across shards"
                )
            seen_case_ids.update(case_ids)
            case_hashes = [content_hash(case) for case in case_chunk]
            record = completed[shard_index]
            if record.get("case_hashes") != case_hashes:
                raise InputValidationError(
                    "resume input cases do not match the completed shard",
                    context={"shard_index": shard_index},
                )
            self._verify_completed_shard(
                record, shards_directory / f"part-{shard_index:06d}"
            )
        if observed_shards != len(completed):
            raise InputValidationError(
                "resume input ended before all checkpointed shards were matched",
                context={"expected_shards": len(completed), "observed_shards": observed_shards},
            )
        manifest_path = self.output_directory / "manifest.json"
        if not manifest_path.exists():
            raise InputValidationError("completed dataset is missing its root manifest")
        expected_manifest_hash = checkpoint.get("root_manifest_hash")
        if not expected_manifest_hash or file_hash(manifest_path) != expected_manifest_hash:
            raise InputValidationError(
                "completed dataset root manifest is missing its checkpointed hash or has changed",
                context={"path": str(manifest_path)},
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint_dataset_id = checkpoint.get("dataset_id")
        if manifest.get("dataset_id") != checkpoint_dataset_id:
            raise InputValidationError(
                "completed dataset root identity does not match its checkpoint",
                context={
                    "manifest_dataset_id": manifest.get("dataset_id"),
                    "checkpoint_dataset_id": checkpoint_dataset_id,
                },
            )
        return manifest

    def run_cases(
        self,
        engine: DamageEngine,
        cases: Iterable[dict],
        *,
        fail_closed: bool = False,
        workers: int = 1,
        chunksize: int | None = None,
        resume: bool = False,
    ) -> dict:
        if workers < 1:
            raise ValueError("workers must be at least one")
        if chunksize is not None and chunksize < 1:
            raise ValueError("chunksize must be at least one")
        effective_chunksize = int(
            chunksize or max(1, self.shard_size // max(1, workers * 4))
        )
        identity = self._identity(
            engine,
            fail_closed=fail_closed,
            workers=workers,
            chunksize=effective_chunksize,
        )
        checkpoint = self._load_checkpoint(identity, resume=resume)
        checkpoint_path = self.output_directory / "checkpoint.json"
        shards_directory = self.output_directory / "shards"
        shards_directory.mkdir(parents=True, exist_ok=True)
        completed = {
            int(item["index"]): item for item in checkpoint.get("completed_shards", [])
        }
        if checkpoint.get("status") == "complete":
            manifest = self._validate_completed_resume(
                cases, completed, shards_directory, checkpoint
            )
            # A crash can occur after publishing the complete checkpoint but
            # before removing the marker.  Only clear it after the complete
            # dataset and every committed partition have been revalidated.
            (self.output_directory / ".dataset.incomplete").unlink(missing_ok=True)
            return manifest

        incomplete_path = self.output_directory / ".dataset.incomplete"
        incomplete_path.write_text(
            json.dumps({"identity_hash": identity["identity_hash"]}, ensure_ascii=False),
            encoding="utf-8",
        )
        seen_case_ids: set[str] = set()
        observed_case_hashes: list[str] = []
        observed_shard_count = 0

        for shard_index, case_chunk in enumerate(_chunks(cases, self.shard_size)):
            observed_shard_count += 1
            case_ids = [str(case.get("case_id") or "") for case in case_chunk]
            if any(not value for value in case_ids):
                raise InputValidationError(
                    "sharded dataset cases require non-empty case_id values",
                    context={"shard_index": shard_index},
                )
            duplicate_ids = sorted(
                value for value in case_ids if value in seen_case_ids
            )
            if len(case_ids) != len(set(case_ids)):
                duplicate_ids.extend(
                    value for value in case_ids if case_ids.count(value) > 1
                )
            if duplicate_ids:
                raise InputValidationError(
                    "dataset case_id values must be globally unique across shards",
                    context={"duplicates": sorted(set(duplicate_ids))},
                )
            seen_case_ids.update(case_ids)
            case_hashes = [content_hash(case) for case in case_chunk]
            observed_case_hashes.extend(case_hashes)
            shard_name = f"part-{shard_index:06d}"
            final_directory = shards_directory / shard_name

            if shard_index in completed:
                record = completed[shard_index]
                if record.get("case_hashes") != case_hashes:
                    raise InputValidationError(
                        "resume input cases do not match the completed shard",
                        context={"shard_index": shard_index},
                    )
                self._verify_completed_shard(record, final_directory)
                continue

            temporary_directory = shards_directory / f".{shard_name}.tmp"
            if temporary_directory.exists():
                shutil.rmtree(temporary_directory)
            if final_directory.exists():
                # A crash may occur after shard rename but before checkpoint
                # publication. Such an uncommitted shard is safe to regenerate.
                shutil.rmtree(final_directory)
            writer = DatasetWriter(
                temporary_directory,
                split_seed=self.split_seed,
                split_ratios=self.split_ratios,
                tensor_compression=self.tensor_compression,
            )
            shard_manifest = writer.run_cases(
                engine,
                case_chunk,
                fail_closed=fail_closed,
                workers=workers,
                chunksize=effective_chunksize,
            )
            temporary_directory.replace(final_directory)
            manifest_path = final_directory / "manifest.json"
            record = {
                "index": shard_index,
                "path": f"shards/{shard_name}",
                "case_ids": case_ids,
                "case_hashes": case_hashes,
                "manifest_hash": file_hash(manifest_path),
                "dataset_id": shard_manifest["dataset_id"],
            }
            checkpoint["completed_shards"].append(record)
            completed[shard_index] = record
            _write_json_atomic(checkpoint_path, checkpoint)

        if observed_shard_count == 0:
            raise ValueError("cases must contain at least one case")
        unexpected = sorted(index for index in completed if index >= observed_shard_count)
        if unexpected:
            raise InputValidationError(
                "resume input ended before all checkpointed shards were matched",
                context={"unexpected_shards": unexpected},
            )

        manifests = [
            json.loads(
                (shards_directory / f"part-{index:06d}" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            for index in range(observed_shard_count)
        ]
        root_manifest = self._aggregate_manifest(
            manifests,
            identity=identity,
            ordered_case_hashes=observed_case_hashes,
            fail_closed=fail_closed,
            workers=workers,
            chunksize=effective_chunksize,
        )
        root_manifest_path = self.output_directory / "manifest.json"
        _write_json_atomic(root_manifest_path, root_manifest)
        checkpoint["status"] = "complete"
        checkpoint["dataset_id"] = root_manifest["dataset_id"]
        checkpoint["root_manifest_hash"] = file_hash(root_manifest_path)
        _write_json_atomic(checkpoint_path, checkpoint)
        incomplete_path.unlink(missing_ok=True)
        return root_manifest

    def _aggregate_manifest(
        self,
        manifests: list[dict],
        *,
        identity: dict,
        ordered_case_hashes: list[str],
        fail_closed: bool,
        workers: int,
        chunksize: int | None,
    ) -> dict:
        first = manifests[0]
        contract = first["tensor_contract"]
        for index, manifest in enumerate(manifests[1:], start=1):
            if manifest["tensor_contract"] != contract:
                raise InputValidationError(
                    "tensor contracts differ between generated shards",
                    context={"shard_index": index},
                )

        target_schemas = {
            manifest["source_contracts"]["target_schema"] for manifest in manifests
        }
        case_schemas = {
            manifest["source_contracts"]["case_schema"] for manifest in manifests
        }
        if len(target_schemas) != 1 or len(case_schemas) != 1:
            raise InputValidationError("source schema versions differ between shards")
        evaluation_standards: dict[str, set[str]] = defaultdict(set)
        for manifest in manifests:
            for standard_id, output_ids in manifest["source_contracts"].get(
                "evaluation_standards", {}
            ).items():
                evaluation_standards[str(standard_id)].update(map(str, output_ids))
        source_contracts = {
            "target_schema": next(iter(target_schemas)),
            "case_schema": next(iter(case_schemas)),
            "damage_tree_versions": sorted(
                {
                    str(version)
                    for manifest in manifests
                    for version in manifest["source_contracts"].get(
                        "damage_tree_versions", []
                    )
                }
            ),
            "evaluation_standards": {
                standard_id: sorted(output_ids)
                for standard_id, output_ids in sorted(evaluation_standards.items())
            },
            "parameter_set_hashes": sorted(
                {
                    str(value)
                    for manifest in manifests
                    for value in manifest["source_contracts"].get(
                        "parameter_set_hashes", []
                    )
                }
            ),
            "parameter_set_ids": sorted(
                {
                    str(value)
                    for manifest in manifests
                    for value in manifest["source_contracts"].get(
                        "parameter_set_ids", []
                    )
                }
            ),
            "target_hashes": sorted(
                {
                    str(value)
                    for manifest in manifests
                    for value in manifest["source_contracts"].get(
                        "target_hashes", []
                    )
                }
            ),
        }

        partition_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for shard_index, manifest in enumerate(manifests):
            for partition in manifest["partitions"]:
                partition_groups[(partition["name"], partition["format"])].append(
                    {"shard_index": shard_index, **partition}
                )
        partitions = []
        for (name, format_name), items in partition_groups.items():
            partitions.append(
                {
                    "name": name,
                    "format": format_name,
                    "path_glob": f"shards/part-*/{items[0]['path_glob']}",
                    "row_count": sum(int(item["row_count"]) for item in items),
                    "content_hash": content_hash(
                        [item["content_hash"] for item in items]
                    ),
                }
            )

        refusal_counts: dict[str, int] = defaultdict(int)
        # Audit globally rather than trusting each shard's local result.  Read
        # the compact split arrays from NPZ so importing the core CLI does not
        # make the optional PyArrow dependency mandatory.
        split_rows: list[dict] = []
        for shard_index in range(len(manifests)):
            tensors_path = (
                self.output_directory
                / "shards"
                / f"part-{shard_index:06d}"
                / "tensors.npz"
            )
            with np.load(tensors_path, allow_pickle=False) as arrays:
                split_rows.extend(
                    {
                        "group_id": str(group_id),
                        "leakage_key": str(leakage_key),
                        "split": str(split),
                    }
                    for group_id, leakage_key, split in zip(
                        arrays["split_group_ids"],
                        arrays["leakage_keys"],
                        arrays["split_names"],
                    )
                )
        split_counts = {
            name: sum(str(row.get("split")) == name for row in split_rows)
            for name in ("train", "validation", "test")
        }
        for manifest in manifests:
            for code, count in manifest["quality_summary"]["refusal_counts"].items():
                refusal_counts[code] += int(count)
        total = sum(int(item["quality_summary"]["n_total"]) for item in manifests)
        valid = sum(int(item["quality_summary"]["n_valid"]) for item in manifests)
        refused = sum(int(item["quality_summary"]["n_refused"]) for item in manifests)
        unverified_fraction = (
            sum(
                float(item["quality_summary"]["unverified_parameter_fraction"])
                * int(item["quality_summary"]["n_total"])
                for item in manifests
            )
            / max(1, total)
        )
        fidelity_levels = sorted(
            {
                level
                for manifest in manifests
                for level in manifest["engine"].get("fidelity_levels", [])
            }
        )
        engine = dict(first["engine"])
        engine["fidelity_levels"] = fidelity_levels
        engine["fidelity_level"] = (
            fidelity_levels[0]
            if len(fidelity_levels) == 1
            else ("mixed" if fidelity_levels else "unknown")
        )
        known_limitations = list(
            dict.fromkeys(
                limitation
                for manifest in manifests
                for limitation in manifest.get("known_limitations", [])
                if "materializes cases, results" not in limitation
            )
        )
        known_limitations.append(
            "Static topology arrays are repeated in each shard; TorchShardedDamageDataset opens shards lazily but does not deduplicate those files on disk."
        )
        # Preserve the monolithic quality statistic without retaining every
        # interval in RAM.  The temporary memmap is bounded by the number of
        # scalar damage-tree outputs and is removed before publication.
        first_tensors = self.output_directory / "shards" / "part-000000" / "tensors.npz"
        with np.load(first_tensors, allow_pickle=False) as arrays:
            output_count = int(len(arrays["system_output_names"]))
        width_count = valid * output_count
        if width_count:
            width_path = self.output_directory / ".ci-half-widths.tmp"
            widths = np.memmap(width_path, dtype=np.float64, mode="w+", shape=(width_count,))
            position = 0
            try:
                for shard_index in range(len(manifests)):
                    results_path = (
                        self.output_directory
                        / "shards"
                        / f"part-{shard_index:06d}"
                        / "results.jsonl"
                    )
                    with results_path.open("r", encoding="utf-8") as stream:
                        for line in stream:
                            payload = json.loads(line)
                            if payload.get("status") != "ok":
                                continue
                            for stats in payload.get("damage_tree", {}).get(
                                "outputs", {}
                            ).values():
                                widths[position] = 0.5 * (
                                    float(stats["ci95_upper"])
                                    - float(stats["ci95_lower"])
                                )
                                position += 1
                if position != width_count:
                    raise InputValidationError(
                        "damage-tree interval count differs between shard results and tensor contract",
                        context={"expected": width_count, "observed": position},
                    )
                widths.flush()
                ci_quantiles = [
                    float(value)
                    for value in np.quantile(
                        widths, [0.5, 0.9, 0.99], overwrite_input=True
                    )
                ]
            finally:
                del widths
                width_path.unlink(missing_ok=True)
        else:
            ci_quantiles = []
        return {
            "schema_version": first["schema_version"],
            "dataset_id": "damage-dataset-"
            + dataset_identity_hash(
                target_hash=identity["target_hash"],
                physics_registry_hash=identity["physics_registry_hash"],
                engine_code_hash=identity["engine_code_hash"],
                ordered_case_hashes=ordered_case_hashes,
                split_seed=identity["split_seed"],
                split_ratios=identity["split_ratios"],
                tensor_contract_version=contract["version"],
            ).split(":", 1)[1][:24],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "engine": engine,
            "source_contracts": source_contracts,
            "generation": {
                "workers": int(workers),
                "chunksize": int(chunksize),
                "failure_policy": "fail_fast" if fail_closed else "isolate_and_record",
                "input_order_preserved": True,
                "tensor_compression": self.tensor_compression,
            },
            "randomness": first["randomness"],
            "partitions": partitions,
            "tensor_contract": contract,
            "split_policy": {
                **first["split_policy"],
                "leakage_audit_passed": DatasetWriter._check_split_leakage(split_rows),
                "split_counts": split_counts,
            },
            "quality_summary": {
                "n_total": total,
                "n_valid": valid,
                "n_refused": refused,
                "refusal_counts": dict(refusal_counts),
                "unverified_parameter_fraction": unverified_fraction,
                "mc_ci_half_width_quantiles": ci_quantiles,
                "coverage_report_id": None,
            },
            "known_limitations": known_limitations,
            "prohibited_uses": list(first["prohibited_uses"]),
        }

