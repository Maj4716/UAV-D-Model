from __future__ import annotations

from damage_engine.dataset import DatasetWriter
from damage_engine.engine import DamageEngine
from damage_engine.result_ledger import load_result_ledger, resolve_result_ledger
from damage_engine.sharded_dataset import ShardedDatasetWriter
from test_sampling_contract import _sampling_plan


def test_sampling_archive_defaults_to_gzip_and_plain_override_remains_readable(
    tmp_path, target_data, case_data, registry
):
    engine = DamageEngine(target_data, registry)
    _, _, plan = _sampling_plan(engine, case_data)

    compressed = tmp_path / "compressed"
    compressed_manifest = DatasetWriter(compressed).run_cases(engine, plan.cases)
    compressed_partition = next(
        item for item in compressed_manifest["partitions"] if item["name"] == "audits"
    )
    assert compressed_manifest["generation"]["result_compression"] == "gzip"
    assert compressed_partition["format"] == "jsonl_gzip"
    assert compressed_partition["path_glob"] == "results.jsonl.gz"
    assert not (compressed / "results.jsonl").exists()
    assert len(load_result_ledger(compressed, compressed_manifest)) == len(plan.cases)

    plain = tmp_path / "plain"
    plain_manifest = DatasetWriter(plain, result_compression="none").run_cases(
        engine, plan.cases
    )
    plain_partition = next(
        item for item in plain_manifest["partitions"] if item["name"] == "audits"
    )
    assert plain_manifest["generation"]["result_compression"] == "none"
    assert plain_partition["format"] == "jsonl"
    assert resolve_result_ledger(plain, plain_manifest).name == "results.jsonl"
    assert len(load_result_ledger(plain, plain_manifest)) == len(plan.cases)

    sharded = tmp_path / "sharded"
    root_manifest = ShardedDatasetWriter(sharded, shard_size=2).run_cases(
        engine, plan.cases
    )
    assert root_manifest["generation"]["result_compression"] == "gzip"
    audit_partitions = [
        item for item in root_manifest["partitions"] if item["name"] == "audits"
    ]
    assert len(audit_partitions) == 2
    assert {item["format"] for item in audit_partitions} == {"jsonl_gzip"}
    assert all(item["path_glob"].endswith("results.jsonl.gz") for item in audit_partitions)
