from __future__ import annotations

from pathlib import Path

import pytest

from damage_engine.errors import InputValidationError
from damage_engine.result_ledger import (
    iter_result_ledger,
    resolve_result_ledger,
    write_result_ledger,
)


def _manifest(path: Path, format_name: str, count: int) -> dict:
    return {
        "partitions": [{
            "name": "audits",
            "format": format_name,
            "path_glob": path.name,
            "row_count": count,
            "content_hash": "sha256:" + "0" * 64,
        }]
    }


def test_gzip_result_ledger_is_deterministic_compact_and_readable(tmp_path):
    rows = [
        {"case_id": f"case-{index:04d}", "status": "ok", "payload": "repeat" * 200}
        for index in range(200)
    ]
    first = tmp_path / "first"
    second = tmp_path / "second"
    plain = tmp_path / "plain"
    first.mkdir()
    second.mkdir()
    plain.mkdir()
    gzip_path, gzip_format = write_result_ledger(first, rows, compression="gzip")
    repeated_path, _ = write_result_ledger(second, rows, compression="gzip")
    plain_path, plain_format = write_result_ledger(plain, rows, compression="none")

    assert gzip_format == "jsonl_gzip"
    assert plain_format == "jsonl"
    assert gzip_path.read_bytes() == repeated_path.read_bytes()
    assert gzip_path.stat().st_size < plain_path.stat().st_size * 0.1
    assert list(iter_result_ledger(gzip_path)) == rows
    assert list(iter_result_ledger(plain_path)) == rows
    assert resolve_result_ledger(first, _manifest(gzip_path, gzip_format, len(rows))) == gzip_path.resolve()
    assert resolve_result_ledger(plain) == plain_path.resolve()


def test_result_ledger_manifest_format_and_corruption_fail_closed(tmp_path):
    rows = [{"case_id": "case-0", "status": "ok"}]
    path, _ = write_result_ledger(tmp_path, rows, compression="gzip")
    wrong = _manifest(path, "jsonl", 1)
    with pytest.raises(InputValidationError, match="format and filename disagree"):
        resolve_result_ledger(tmp_path, wrong)

    payload = path.read_bytes()
    path.write_bytes(payload[:-4])
    with pytest.raises(InputValidationError, match="unreadable or corrupt"):
        list(iter_result_ledger(path))
