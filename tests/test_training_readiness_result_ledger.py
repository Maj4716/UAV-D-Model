from __future__ import annotations

import json

import numpy as np
import pytest

from damage_engine.result_ledger import write_result_ledger
from tools.audit_dataset_training_readiness import _trace_audit


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
