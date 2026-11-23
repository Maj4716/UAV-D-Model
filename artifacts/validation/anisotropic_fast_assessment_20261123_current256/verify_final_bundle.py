"""Verify the current anisotropic fast-assessment evidence bundle."""

from __future__ import annotations

from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.engine import ENGINE_CODE_HASH, DamageEngine
from damage_engine.registry import PhysicsRegistry
from damage_engine.utils import content_hash, file_hash, source_tree_hash


HISTORICAL = ROOT / "artifacts/validation/anisotropic_fast_assessment_20261123_v3"
TARGET = ROOT / "configs/targets/target.fragment_only.v1.json"
PHYSICS = ROOT / "configs/physics/loitering_impact.taylor_shapiro.obliquity_bound.v3.json"


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _load(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
    )


def _embedded_hash(value: dict[str, Any], field: str) -> bool:
    copy = dict(value)
    recorded = copy.pop(field, None)
    return isinstance(recorded, str) and recorded == content_hash(copy)


def _verify_hash_ledger(
    directory: Path,
    *,
    inputs: dict[str, Path],
) -> None:
    ledger = _load(directory / "hashes.json")
    assert set(ledger["inputs"]) == set(inputs)
    for name, path in inputs.items():
        assert file_hash(path) == ledger["inputs"][name]
    for name, digest in ledger["outputs"].items():
        assert file_hash(directory / name) == digest
    command = _load(directory / "manifest.json")
    assert isinstance(command, dict)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> None:
    checks: list[str] = []

    def check(name: str, condition: bool) -> None:
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    assessment = _load(HERE / "assessment.json")
    check("assessment embedded hash", _embedded_hash(assessment, "assessment_hash"))
    check("quick candidate count", assessment["candidate_count"] == 256)
    check(
        "quick candidate IDs unique",
        len(assessment["candidate_ids"]) == len(set(assessment["candidate_ids"])),
    )
    provenance = assessment["provenance"]
    check("quick excludes test", provenance["split_audit"]["test_case_count_included"] == 0)
    check("quick does not run teacher", provenance["teacher_executed"] is False)
    check("quick does not build dataset", provenance["formal_dataset_built"] is False)
    check("quick sealed test decode zero", provenance["sealed_test_result_rows_decoded"] == 0)

    live = provenance["live_proxy_identity"]
    current_tree = source_tree_hash(ROOT / "damage_engine")
    check("current source tree identity", live["damage_engine_source_tree_hash"] == current_tree)
    check("current engine identity", live["engine_code_hash"] == ENGINE_CODE_HASH)
    engine = DamageEngine(_load(TARGET), PhysicsRegistry.from_file(PHYSICS))
    check("current target identity", live["target_hash"] == engine.target_hash)
    check("current physics identity", live["physics_registry_hash"] == engine.registry.hash)

    tier_1 = assessment["tier_1_generator"]
    diagnosis = assessment["diagnosis"]
    check(
        "L1 declared mass gap",
        tier_1["declared_cell_comparison"]["total_variation"] > 0.1
        and diagnosis["generator_checks"]["declared_cell_mass_gate_pass"] is False,
    )
    check(
        "L1 marginal KS gap",
        diagnosis["generator_checks"]["maximum_uniform_marginal_ks"] > 0.1
        and diagnosis["generator_checks"]["uniform_marginal_ks_gate_pass"] is False,
    )
    check(
        "L1 cell support gate",
        diagnosis["generator_checks"]["cell_count_gate_pass"] is False
        and diagnosis["generator_checks"]["cell_rank_gate_pass"] is False,
    )

    tier_2 = assessment["tier_2_geometry_proxy"]
    rows = tier_2["exposure_field"]["candidate_rows"]
    check("L2 row alignment", [row["candidate_id"] for row in rows] == assessment["candidate_ids"])
    component_ids = list(tier_2["exposure_field"]["component_ids"])
    check("L2 component count", len(component_ids) == 67)
    for row in rows:
        check_ids = sorted(row["component_lambda"])
        if check_ids != component_ids:
            raise AssertionError("L2 component identity alignment")
        for field in ("component_lambda", "component_damage_energy_j"):
            values = list(row[field].values())
            if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in values):
                raise AssertionError(f"L2 finite nonnegative {field}")
    checks.extend(["L2 component identity alignment", "L2 finite nonnegative fields"])
    check(
        "L2 incomplete domain detected",
        tier_2["proxy_domain_coverage"]["complete_case_fraction"] == 146 / 256
        and diagnosis["proxy_checks"]["gate_pass"] is False,
    )
    check("L2 non-structural zero", tier_2["structural_zero_claim"] is False)

    role_expectations = {
        "development": {"train": 229, "validation": 27},
        "train": {"train": 229},
        "validation": {"validation": 27},
    }
    scope_expectations = {
        "development": "development_train_validation",
        "train": "train",
        "validation": "validation",
    }
    evidence_names = {
        "development": "existing_teacher_rows.json",
        "train": "train_teacher_rows.json",
        "validation": "validation_teacher_rows.json",
    }
    for name in ("development", "train", "validation"):
        evidence = _load(HERE / evidence_names[name])
        check(f"{name} teacher embedded hash", _embedded_hash(evidence, "report_hash"))
        observed_roles = Counter(row["split_role"] for row in evidence["teacher_rows"])
        check(f"{name} teacher roles", dict(observed_roles) == role_expectations[name])
        source = evidence["source_audit"]
        check(f"{name} test decode zero", source["test_result_json_lines_decoded"] == 0)
        check(f"{name} test metrics false", source["test_metrics_computed"] is False)
        check(f"{name} test open zero", source["test_open_count_increment"] == 0)

        directory = HERE / f"engine_calibration_{name}"
        calibration = _load(directory / "engine_calibration.json")
        check(f"{name} calibration embedded hash", _embedded_hash(calibration, "comparison_hash"))
        check(f"{name} aggregate role", calibration["split_role"] == scope_expectations[name])
        check(f"{name} calibration roles", calibration["teacher_role_counts"] == role_expectations[name])
        overall = calibration["overall"]
        check(
            f"{name} pair arithmetic",
            overall["pair_count"] == overall["case_count"] * overall["component_count"],
        )
        check(
            f"{name} probability MSE alias",
            overall["weighted_probability_mse"] == overall["weighted_brier"],
        )
        check(f"{name} engine gap", calibration["diagnosis"]["engine_status"] == "proxy_teacher_gap_detected")
        check(f"{name} false-unreached gate", calibration["diagnosis"]["false_unreached_gate_pass"] is False)
        check(f"{name} rank gate", calibration["diagnosis"]["rank_gate_pass"] is False)
        check(f"{name} non-structural zero", calibration["structural_zero_claim"] is False)
        cal_provenance = calibration["provenance"]
        check(f"{name} no teacher execution", cal_provenance["teacher_executed"] is False)
        check(f"{name} sealed decode zero", cal_provenance["sealed_test_result_rows_decoded"] == 0)
        identity = cal_provenance["teacher_source_audit"]["identity_comparison"]
        check(f"{name} target identity match", identity["target_hash_match"] is True)
        check(f"{name} physics identity match", identity["physics_registry_hash_match"] is True)
        check(f"{name} engine identity mismatch retained", identity["engine_code_hash_match"] is False)
        _verify_hash_ledger(
            directory,
            inputs={
                "assessment": HERE / "assessment.json",
                "config": ROOT / "configs/sampling/anisotropic_fast_assessment.v1.json",
                "teacher_evidence": HERE / evidence_names[name],
            },
        )
        checks.append(f"{name} hash ledger")

    _verify_hash_ledger(
        HERE,
        inputs={
            "cases": ROOT
            / "artifacts/datasets/recovery_role_r8_r32_train2000_eval300_20261122"
            / "rounds/round-000/formal_cases.jsonl.gz",
            "config": ROOT / "configs/sampling/anisotropic_fast_assessment.v1.json",
            "physics": PHYSICS,
            "target": TARGET,
        },
    )
    checks.append("quick hash ledger")

    historical_assessment = _load(HISTORICAL / "assessment.json")
    cache_identity = historical_assessment["provenance"]["source_audit"]["cache_identity"]
    check("historical cache bytes match", cache_identity["npz_hash_match"] is True)
    check(
        "historical cache source mismatch explicit",
        cache_identity["status"] == "historical_cache_current_source_differs"
        and cache_identity["damage_engine_source_tree_match"] is False
        and cache_identity["engine_code_hash_match"] is False,
    )
    approximation = _load(HISTORICAL / "quick_vs_full_x1.json")
    check(
        "historical L1 approximation",
        approximation["tier_1_generator_approximation"]["quick_vs_full_cell_mass_tv"] < 0.001,
    )
    check(
        "historical L2 approximation",
        approximation["tier_2_field_approximation"]["component_lambda_pearson"] > 0.99,
    )

    benchmark = _load(HERE / "performance_benchmark_v2.json")
    check("benchmark exposure equality", benchmark["shared_exposure_fields_exactly_equal"] is True)
    check("benchmark test input zero", benchmark["test_case_inputs_seen"] == 0)
    check("benchmark teacher read zero", benchmark["teacher_result_rows_read"] == 0)
    check("benchmark sealed decode zero", benchmark["sealed_test_result_rows_decoded"] == 0)
    check(
        "benchmark current source identity",
        benchmark["live_identity"]["damage_engine_source_tree_hash"] == current_tree,
    )

    python_files = [
        ROOT / "damage_engine/fast_assessment.py",
        ROOT / "damage_engine/fast_sampling_proxy.py",
        ROOT / "tools/run_anisotropic_fast_assessment.py",
        ROOT / "tests/test_fast_assessment.py",
        ROOT / "tests/test_anisotropic_fast_assessment_cli.py",
        HERE / "extract_existing_teacher_evidence.py",
        HERE / "verify_final_bundle.py",
    ]
    for path in python_files:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    checks.append("in-memory Python compilation")

    pytest_command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_anisotropic_fast_assessment_cli.py",
        "tests/test_fast_assessment.py",
        "tests/test_fast_sampling_proxy.py",
        "tests/test_fast_proxy_domain_and_layers.py",
        "tests/test_exposure_field.py",
        "tests/test_multifidelity.py",
        "-p",
        "no:cacheprovider",
    ]
    pytest_result = _run(pytest_command)
    check("targeted pytest", pytest_result.returncode == 0)

    help_result = _run(
        [sys.executable, "tools/run_anisotropic_fast_assessment.py", "calibrate", "--help"]
    )
    check("calibrate help", help_result.returncode == 0)
    check("NPZ calibration disabled", "--precomputed-npz" not in help_result.stdout)

    diff_result = _run(
        [
            "git",
            "diff",
            "--check",
            "--",
            "damage_engine/fast_sampling_proxy.py",
            "README.md",
            "docs/INDEX.md",
            "tools/README.md",
            "tests/test_fast_sampling_proxy.py",
        ]
    )
    check("scoped git diff check", diff_result.returncode == 0)

    qa_log = "\n".join(
        [
            "# Targeted pytest",
            subprocess.list2cmdline(pytest_command),
            pytest_result.stdout.rstrip(),
            pytest_result.stderr.rstrip(),
            "",
            "# Calibrate help",
            help_result.stdout.rstrip(),
            help_result.stderr.rstrip(),
            "",
            "# Scoped git diff check",
            diff_result.stdout.rstrip(),
            diff_result.stderr.rstrip(),
            "",
        ]
    )
    (HERE / "QA_LOG.txt").write_text(qa_log, encoding="utf-8")

    verification = {
        "schema_version": "anisotropic-fast-assessment-verification@1.0.0",
        "status": "pass",
        "check_count": len(checks),
        "checks": checks,
        "current_identity": {
            "engine_code_hash": ENGINE_CODE_HASH,
            "damage_engine_source_tree_hash": current_tree,
            "target_hash": engine.target_hash,
            "physics_registry_hash": engine.registry.hash,
        },
        "safety": {
            "test_case_count_included": 0,
            "sealed_test_result_rows_decoded": 0,
            "teacher_executed": False,
            "formal_dataset_built": False,
        },
        "pytest": {
            "returncode": pytest_result.returncode,
            "stdout": pytest_result.stdout.strip(),
        },
    }
    verification["verification_hash"] = content_hash(verification)
    verification_path = HERE / "verification.json"
    verification_path.write_text(
        json.dumps(verification, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    evidence_paths = [
        ROOT / "damage_engine/fast_assessment.py",
        ROOT / "damage_engine/fast_sampling_proxy.py",
        ROOT / "tools/run_anisotropic_fast_assessment.py",
        ROOT / "configs/sampling/anisotropic_fast_assessment.v1.json",
        ROOT / "docs/ANISOTROPIC_FAST_ASSESSMENT.md",
        ROOT / "tests/test_fast_assessment.py",
        ROOT / "tests/test_anisotropic_fast_assessment_cli.py",
        ROOT / "tests/test_fast_sampling_proxy.py",
        HERE / "assessment.json",
        HERE / "FINAL_REPORT.md",
        HERE / "IMPLEMENTATION_LOG.md",
        HERE / "existing_teacher_rows.json",
        HERE / "train_teacher_rows.json",
        HERE / "validation_teacher_rows.json",
        HERE / "performance_benchmark_v2.json",
        HERE / "engine_calibration_development/engine_calibration.json",
        HERE / "engine_calibration_train/engine_calibration.json",
        HERE / "engine_calibration_validation/engine_calibration.json",
        HISTORICAL / "quick_vs_full_x1.json",
        verification_path,
        HERE / "QA_LOG.txt",
    ]
    manifest = {
        "schema_version": "anisotropic-fast-assessment-evidence-manifest@1.0.0",
        "verification_status": "pass",
        "files": {
            path.relative_to(ROOT).as_posix(): file_hash(path)
            for path in evidence_paths
        },
    }
    manifest["manifest_hash"] = content_hash(manifest)
    (HERE / "evidence_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "check_count": len(checks),
                "verification": str(verification_path),
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
