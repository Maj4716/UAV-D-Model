from __future__ import annotations

import json

import numpy as np
import pytest

from damage_engine.result_ledger import write_result_ledger
from tools.audit_dataset_training_readiness import _trace_audit


def _write_declared_ledger(root, records):
    shard = root / "shards" / "part-000000"
    shard.mkdir(parents=True)
    path, format_name = write_result_ledger(
        shard, records, compression="none"
    )
    child_partition = {
        "name": "audits",
        "format": format_name,
        "path_glob": path.name,
        "row_count": len(records),
        "content_hash": "sha256:" + "2" * 64,
    }
    (shard / "manifest.json").write_text(
        json.dumps({"partitions": [child_partition]}), encoding="utf-8"
    )
    root_partition = {
        **child_partition,
        "path_glob": f"shards/part-000000/{path.name}",
    }
    (root / "manifest.json").write_text(
        json.dumps({"partitions": [root_partition]}), encoding="utf-8"
    )


@pytest.mark.parametrize("compression", ["none", "gzip"])
def test_training_readiness_trace_audit_reads_manifest_declared_ledgers(
    tmp_path, compression,
):
    root = tmp_path / "dataset"
    shard = root / "shards" / "part-000000"
    shard.mkdir(parents=True)
    record = {
        "case_id": "case-0000",
        "status": "ok",
        "replicate_trace": {
            "records": [{
                "parameter_realization_hash": "sha256:" + "1" * 64,
                "system_outputs": {"mission_loss": 0.25},
                "fragment_metrics": {},
            }]
        },
    }
    path, format_name = write_result_ledger(
        shard, [record], compression=compression
    )
    content_hash = "sha256:" + "2" * 64
    child_partition = {
        "name": "audits",
        "format": format_name,
        "path_glob": path.name,
        "row_count": 1,
        "content_hash": content_hash,
    }
    (shard / "manifest.json").write_text(
        json.dumps({"partitions": [child_partition]}), encoding="utf-8"
    )
    root_partition = {
        **child_partition,
        "path_glob": f"shards/part-000000/{path.name}",
    }
    (root / "manifest.json").write_text(
        json.dumps({"partitions": [root_partition]}), encoding="utf-8"
    )

    report = _trace_audit(
        root,
        ["mission_loss"],
        np.asarray([[0.25]], dtype=np.float64),
    )

    assert report["case_count"] == 1
    assert report["total_replicate_records"] == 1
    assert report["trace_mean_vs_aggregate_absolute_error"]["mission_loss"][
        "max"
    ] == 0.0


def test_training_readiness_accepts_declared_online_moments_without_raw_trace(
    tmp_path,
):
    root = tmp_path / "dataset"
    _write_declared_ledger(
        root,
        [{
            "case_id": "case-0000",
            "status": "ok",
            "replicate_count": 4,
            "replicate_statistics_storage": {
                "mode": "online_sufficient_moments",
                "raw_trace_retained": False,
            },
        }],
    )

    report = _trace_audit(
        root,
        ["mission_loss"],
        np.asarray([[0.25]], dtype=np.float64),
    )

    assert report["status"] == "unavailable_raw_trace_not_retained"
    assert report["case_count"] == 1
    assert report["trace_available_case_count"] == 0
    assert report["trace_unavailable_case_count"] == 1
    assert report["storage_mode_counts"] == {
        "online_sufficient_moments": 1
    }
    assert report["declared_replicate_count_distribution"] == {"4": 1}
    assert report["fraction_with_any_system_output_variation"] is None


def test_training_readiness_rejects_missing_trace_claimed_as_retained(tmp_path):
    root = tmp_path / "dataset"
    _write_declared_ledger(
        root,
        [{
            "case_id": "case-0000",
            "status": "ok",
            "replicate_statistics_storage": {
                "mode": "online_sufficient_moments_plus_raw_trace",
                "raw_trace_retained": True,
            },
        }],
    )

    with pytest.raises(
        RuntimeError,
        match="trace presence disagrees with storage declaration",
    ):
        _trace_audit(
            root,
            ["mission_loss"],
            np.asarray([[0.25]], dtype=np.float64),
        )


def test_training_readiness_reports_partial_trace_availability(tmp_path):
    root = tmp_path / "dataset"
    _write_declared_ledger(
        root,
        [
            {
                "case_id": "case-0000",
                "status": "ok",
                "replicate_count": 1,
                "replicate_statistics_storage": {
                    "mode": "online_sufficient_moments_plus_raw_trace",
                    "raw_trace_retained": True,
                },
                "replicate_trace": {
                    "records": [{
                        "parameter_realization_hash": "sha256:" + "1" * 64,
                        "system_outputs": {"mission_loss": 0.25},
                        "fragment_metrics": {},
                    }]
                },
            },
            {
                "case_id": "case-0001",
                "status": "ok",
                "replicate_count": 4,
                "replicate_statistics_storage": {
                    "mode": "online_sufficient_moments",
                    "raw_trace_retained": False,
                },
            },
        ],
    )

    report = _trace_audit(
        root,
        ["mission_loss"],
        np.asarray([[0.25], [0.5]], dtype=np.float64),
    )

    assert report["status"] == "partial_trace_availability"
    assert report["case_count"] == 2
    assert report["trace_available_case_count"] == 1
    assert report["trace_unavailable_case_count"] == 1
    assert report["total_replicate_records"] == 1
    assert report["fraction_with_any_system_output_variation"] == 0.0
