"""Selected generic archive/interface tests; no simulator or model training."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET

ROOT = Path(__file__).absolute().parents[2]
OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from damage_engine.utils import source_tree_hash

TESTS = [
    "tests/test_utils.py", "tests/test_normalization.py", "tests/test_schema_resources.py",
    "tests/test_statistics_cli_review.py", "tests/test_loader_contract_review.py",
    "tests/test_shard_integrity_review.py", "tests/test_writer_integrity_review.py",
    "tests/test_reaudit_contract_fix.py", "tests/test_reaudit_serialization_fix.py",
    "tests/test_reaudit_sampling_fix.py", "tests/test_reaudit_audit_fix.py",
    "tests/test_result_ledger_compression.py", "tests/test_training_readiness_result_ledger.py",
    "tests/test_loader_stream_separation.py", "tests/test_dataset_policy.py", "tests/test_dataset_policy_cli.py",
]
TEMP_ROOT = Path(r"C:\Users\Administrator\.codex\visualizations\2026\10\09\01a11ff4-76f9-7262-9bf8-10f9bed66e97")


def main():
    temp = TEMP_ROOT / ("reaudit_20261015_" + uuid.uuid4().hex[:10])
    temp.mkdir()
    env = {**os.environ, "TEMP": str(temp), "TMP": str(temp), "PYTHONUTF8": "1"}
    before = source_tree_hash(ROOT / "damage_engine")
    report = {"date": "2026-10-15", "timezone": "Asia/Shanghai", "python": sys.executable,
              "scope": "selected generic software, archive integrity and abstract data-interface tests only",
              "source_before": before, "test_files": TESTS, "process_temp": str(temp), "steps": []}
    def run(name, command, timeout):
        start = time.perf_counter()
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                encoding="utf-8", errors="replace", timeout=timeout)
        (OUT / (name + ".log")).write_text(result.stdout + "\n[stderr]\n" + result.stderr, encoding="utf-8")
        report["steps"].append({"name": name, "command": command, "exit_code": result.returncode,
                                "seconds": time.perf_counter() - start})
        print(json.dumps({"step": name, "exit_code": result.returncode}), flush=True)
        if result.returncode:
            print((result.stdout + result.stderr)[-6500:], flush=True)
        return result.returncode
    junit = OUT / "generic_tests.xml"
    test_code = run("generic_pytest", [sys.executable, "-m", "pytest", *TESTS, "-q", "-p", "no:cacheprovider",
                    "--basetemp", str(temp / "pytest"), "--junitxml", str(junit)], 360)
    if junit.exists():
        suites = list(ET.parse(junit).getroot().iter("testsuite"))
        report["pytest_summary"] = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
                                    for key in ("tests", "failures", "errors", "skipped")}
    snapshot = temp / "static_source"
    snapshot.mkdir()
    for directory in ("damage_engine", "tests"):
        shutil.copytree(ROOT / directory, snapshot / directory,
                        ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("README.md", "pyproject.toml", "requirements.txt", "requirements-lock.txt"):
        shutil.copy2(ROOT / name, snapshot / name)
    static_code = run("static_repro", [sys.executable,
        r"C:\Users\Administrator\.codex\skills\test-research-code\scripts\repro_check.py", str(snapshot), "--json"], 90)
    report["source_after"] = source_tree_hash(ROOT / "damage_engine")
    report["source_unchanged"] = before == report["source_after"]
    (OUT / "generic_checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": report.get("pytest_summary"), "source_unchanged": report["source_unchanged"]}), flush=True)
    raise SystemExit(0 if not test_code and not static_code and report["source_unchanged"] else 1)


if __name__ == "__main__":
    main()
