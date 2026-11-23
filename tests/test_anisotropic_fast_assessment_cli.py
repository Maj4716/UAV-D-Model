from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from damage_engine.utils import content_hash
from damage_engine.errors import InputValidationError
from tools.run_anisotropic_fast_assessment import _teacher_rows, main


def _config(path: Path) -> Path:
    value = {
        "schema_version": "anisotropic-fast-assessment-config@1.0.0",
        "method_id": "test-fast-assessment",
        "default_max_cases": 16,
        "precomputed_selection": {"seed": 17},
        "generator": {
            "maximum_uniform_marginal_ks": 1.0,
            "maximum_expected_cell_total_variation": 1.0,
            "maximum_cell_total_variation": 1.0,
            "minimum_measure_ess_fraction": 0.1,
            "minimum_cell_count": 1,
            "require_full_cell_rank": False,
        },
        "proxy": {
            "reach_tolerance": 0.0,
            "minimum_complete_domain_fraction": 0.5,
        },
        "engine_calibration": {
            "teacher_positive_threshold": 0.0,
            "confidence": 0.95,
            "maximum_false_unreached_wilson_upper": 1.0,
            "minimum_proxy_teacher_spearman": -1.0,
        },
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_cli_reuses_npz_for_quick_and_requires_role_labelled_calibration_evidence(
    tmp_path: Path,
):
    source = tmp_path / "cached.npz"
    candidate_ids = np.asarray([f"c{index}" for index in range(6)])
    cell_ids = np.asarray(["a", "a", "a", "b", "b", "b"])
    latent = np.asarray(
        [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.7, 0.6], [0.8, 0.7], [0.9, 0.8]]
    )
    component_ids = np.asarray(["engine", "sensor"])
    lambdas = np.asarray(
        [[0.0, 0.2], [0.3, 0.0], [0.5, 0.1], [0.2, 0.7], [0.8, 0.4], [1.0, 0.6]]
    )
    np.savez_compressed(
        source,
        candidate_ids=candidate_ids,
        cell_ids=cell_ids,
        latent=latent,
        component_ids=component_ids,
        component_lambda=lambdas,
        component_damage_energy_j=10.0 * lambdas,
        proxy_domain_status=np.asarray(["complete_proxy_domain"] * 6),
        teacher_indices=np.asarray([0, 2, 4]),
        teacher_component_hit_probability=np.asarray(
            [[0.1, 0.2], [0.4, 0.1], [0.7, 0.3]]
        ),
    )
    config = _config(tmp_path / "config.json")
    quick = tmp_path / "quick"

    assert (
        main(
            [
                "quick",
                "--precomputed-npz",
                str(source),
                "--max-cases",
                "6",
                "--config",
                str(config),
                "--output-dir",
                str(quick),
            ]
        )
        == 0
    )
    assessment = json.loads((quick / "assessment.json").read_text(encoding="utf-8"))
    assert assessment["candidate_count"] == 6
    assert assessment["tier_3_engine"]["status"] == "not_evaluated"
    assert assessment["provenance"]["teacher_executed"] is False
    assert assessment["provenance"]["sealed_test_result_rows_decoded"] == 0

    teacher = {
        "schema_version": "test-teacher-evidence@1.0.0",
        "teacher_rows": [
            {
                "candidate_id": "c0",
                "split_role": "train",
                "component_hit_probability": {"engine": 0.1, "sensor": 0.2},
            },
            {
                "candidate_id": "c2",
                "split_role": "validation",
                "component_hit_probability": {"engine": 0.4, "sensor": 0.1},
            },
            {
                "candidate_id": "c4",
                "split_role": "train",
                "component_hit_probability": {"engine": 0.7, "sensor": 0.3},
            },
        ],
        "source_audit": {
            "test_result_json_lines_decoded": 0,
            "test_metrics_computed": False,
            "test_open_count_increment": 0,
        },
    }
    teacher["report_hash"] = content_hash(teacher)
    teacher_path = tmp_path / "teacher.json"
    teacher_path.write_text(json.dumps(teacher), encoding="utf-8")
    calibration = tmp_path / "calibration"
    assert (
        main(
            [
                "calibrate",
                "--assessment",
                str(quick / "assessment.json"),
                "--teacher-evidence",
                str(teacher_path),
                "--config",
                str(config),
                "--split-role",
                "development_train_validation",
                "--output-dir",
                str(calibration),
            ]
        )
        == 0
    )
    result = json.loads(
        (calibration / "engine_calibration.json").read_text(encoding="utf-8")
    )
    assert len(result["paired_candidate_ids"]) == 3
    assert result["teacher_role_counts"] == {"train": 2, "validation": 1}
    assert result["provenance"]["teacher_executed"] is False
    assert result["provenance"]["sealed_test_result_rows_decoded"] == 0

    with pytest.raises(SystemExit):
        main(
            [
                "calibrate",
                "--assessment",
                str(quick / "assessment.json"),
                "--precomputed-npz",
                str(source),
                "--output-dir",
                str(tmp_path / "rejected"),
            ]
        )


def test_quick_cli_excludes_test_cases_before_proxy_statistics(tmp_path: Path):
    cases = []
    exposure = []
    for index, split in enumerate(("train", "validation", "test")):
        identifier = f"case-{index}"
        cases.append(
            {
                "candidate_id": identifier,
                "cell_id": "a",
                "latent": [0.2 + 0.2 * index, 0.3 + 0.1 * index],
                "case": {
                    "case_id": identifier,
                    "metadata": {"frozen_split": split},
                },
            }
        )
        exposure.append(
            {
                "candidate_id": identifier,
                "cell_id": "a",
                "measure_weight": 1.0,
                "proxy_domain_status": "complete_proxy_domain",
                "component_lambda": {"component": float(index)},
                "component_damage_energy_j": {"component": float(index)},
            }
        )
    cases_path = tmp_path / "cases.json"
    exposure_path = tmp_path / "exposure.json"
    cases_path.write_text(json.dumps(cases), encoding="utf-8")
    exposure_path.write_text(json.dumps(exposure), encoding="utf-8")
    output = tmp_path / "filtered"

    assert (
        main(
            [
                "quick",
                "--cases",
                str(cases_path),
                "--exposure-rows",
                str(exposure_path),
                "--config",
                str(_config(tmp_path / "config.json")),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assessment = json.loads((output / "assessment.json").read_text(encoding="utf-8"))
    assert assessment["candidate_count"] == 2
    assert manifest["split_audit"]["test_case_count_included"] == 0
    assert manifest["split_audit"]["source_split_counts"]["test"] == 1


def test_quick_cli_reports_sparse_cell_gate(tmp_path: Path):
    cases = [
        {
            "candidate_id": "a0",
            "cell_id": "a",
            "latent": [0.2, 0.3],
            "case": {"case_id": "a0", "metadata": {"frozen_split": "train"}},
        },
        {
            "candidate_id": "b0",
            "cell_id": "b",
            "latent": [0.7, 0.8],
            "case": {"case_id": "b0", "metadata": {"frozen_split": "train"}},
        },
    ]
    exposure = [
        {
            "candidate_id": row["candidate_id"],
            "cell_id": row["cell_id"],
            "measure_weight": 1.0,
            "proxy_domain_status": "complete_proxy_domain",
            "component_lambda": {"component": 0.1},
            "component_damage_energy_j": {"component": 1.0},
        }
        for row in cases
    ]
    config = json.loads(_config(tmp_path / "config.json").read_text())
    config["generator"]["minimum_cell_count"] = 2
    config_path = tmp_path / "sparse-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    cases_path = tmp_path / "sparse-cases.json"
    exposure_path = tmp_path / "sparse-exposure.json"
    cases_path.write_text(json.dumps(cases), encoding="utf-8")
    exposure_path.write_text(json.dumps(exposure), encoding="utf-8")
    output = tmp_path / "sparse-output"

    assert main([
        "quick",
        "--cases", str(cases_path),
        "--exposure-rows", str(exposure_path),
        "--config", str(config_path),
        "--output-dir", str(output),
    ]) == 0
    assessment = json.loads((output / "assessment.json").read_text(encoding="utf-8"))
    checks = assessment["diagnosis"]["generator_checks"]
    assert checks["cell_count_gate_pass"] is False
    assert checks["undersized_cells"] == {"a": 1, "b": 1}
    assert assessment["diagnosis"]["generator_status"] == "insufficient_cell_support"


def test_teacher_evidence_rejects_hash_tampering_and_test_use(tmp_path: Path):
    evidence = {
        "teacher_rows": [
            {
                "candidate_id": "c0",
                "split_role": "validation",
                "component_hit_probability": {"component": 0.1},
            }
        ],
        "source_audit": {
            "test_result_json_lines_decoded": 0,
            "test_metrics_computed": False,
        },
    }
    evidence["report_hash"] = content_hash(evidence)
    path = tmp_path / "teacher.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    rows, metadata = _teacher_rows(path)
    assert len(rows) == 1
    assert metadata["evidence_integrity"]["status"] == "verified"

    tampered = dict(evidence)
    tampered["source_audit"] = {"test_result_json_lines_decoded": 1}
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(InputValidationError, match="report_hash"):
        _teacher_rows(path)

    tampered["report_hash"] = content_hash(
        {key: value for key, value in tampered.items() if key != "report_hash"}
    )
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(InputValidationError, match="test data"):
        _teacher_rows(path)

