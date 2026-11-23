"""Compare the 4,096-case quick rebuild with the frozen 100,000-case X1 field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.utils import content_hash, file_hash


NPZ = ROOT / "artifacts/experiments/X1_X2_20261120_frozen_v2/x1_candidate_exposure.npz"
FULL = ROOT / "artifacts/experiments/X1_X2_20261120_frozen_v2/x1_exposure_field.json"
CONFIG = ROOT / "configs/sampling/anisotropic_fast_assessment.v1.json"


def uniform_ks(values: np.ndarray) -> float:
    ordered = np.sort(values, kind="stable")
    count = ordered.size
    upper = np.arange(1, count + 1, dtype=np.float64) / count
    lower = np.arange(count, dtype=np.float64) / count
    return float(max(np.max(np.abs(upper - ordered)), np.max(np.abs(lower - ordered))))


def normalized_l1(left: np.ndarray, right: np.ndarray) -> float | None:
    scale = float(np.sum(np.abs(right)))
    return None if scale <= 0.0 else float(np.sum(np.abs(left - right)) / scale)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment", type=Path, default=HERE / "assessment.json")
    parser.add_argument("--output", type=Path, default=HERE / "quick_vs_full_x1.json")
    args = parser.parse_args()
    full = json.loads(FULL.read_text(encoding="utf-8"))
    quick = json.loads(args.assessment.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    expected = config["generator"]["expected_cell_mass"]
    with np.load(NPZ, allow_pickle=False) as archive:
        cell_ids = archive["cell_ids"].astype(str)
        latent = np.asarray(archive["latent"], dtype=np.float64)
    unique, counts = np.unique(cell_ids, return_counts=True)
    full_cell_mass = {
        str(cell): float(count / cell_ids.size)
        for cell, count in zip(unique, counts, strict=True)
    }
    full_declared_tv = 0.5 * sum(
        abs(full_cell_mass[cell] - float(expected[cell])) for cell in full_cell_mass
    )
    quick_cell_mass = quick["tier_1_generator"]["reference_cell_mass"]
    quick_full_cell_tv = 0.5 * sum(
        abs(quick_cell_mass[cell] - full_cell_mass[cell]) for cell in full_cell_mass
    )
    full_mean = np.mean(latent, axis=0)
    full_covariance = np.cov(latent, rowvar=False, ddof=0)
    full_eigenvalues = np.linalg.eigvalsh(full_covariance)[::-1]
    quick_ball = quick["tier_1_generator"]["reference"]["global"]
    quick_mean = np.asarray(quick_ball["mean"], dtype=np.float64)
    quick_covariance = np.asarray(quick_ball["covariance"], dtype=np.float64)
    full_ks = [uniform_ks(latent[:, axis]) for axis in range(latent.shape[1])]
    quick_ks = [
        float(item["uniform_ks"]) for item in quick_ball["uniform_marginals"]
    ]

    component_ids = list(full["component_ids"])
    quick_components = quick["tier_2_geometry_proxy"]["exposure_field"]["components"]
    full_lambda = np.asarray(
        [full["components"][item]["lambda_hajek"] for item in component_ids]
    )
    quick_lambda = np.asarray(
        [quick_components[item]["lambda_hajek"] for item in component_ids]
    )
    full_energy = np.asarray(
        [full["components"][item]["damage_energy_hajek_j"] for item in component_ids]
    )
    quick_energy = np.asarray(
        [quick_components[item]["damage_energy_hajek_j"] for item in component_ids]
    )
    full_reached = {item for item, value in zip(component_ids, full_lambda) if value > 0.0}
    quick_reached = {item for item, value in zip(component_ids, quick_lambda) if value > 0.0}
    full_complete = float(full["structural_audit"]["complete_proxy_domain_fraction"])
    quick_complete = float(
        quick["tier_2_geometry_proxy"]["proxy_domain_coverage"][
            "complete_case_fraction"
        ]
    )
    report = {
        "schema_version": "anisotropic-quick-vs-full-x1@1.0.0",
        "quick_candidate_count": int(quick["candidate_count"]),
        "full_candidate_count": int(full["candidate_count"]),
        "tier_1_generator_approximation": {
            "declared_cell_tv_quick": float(
                quick["tier_1_generator"]["declared_cell_comparison"][
                    "total_variation"
                ]
            ),
            "declared_cell_tv_full": full_declared_tv,
            "quick_vs_full_cell_mass_tv": quick_full_cell_tv,
            "latent_mean_l2_quick_vs_full": float(np.linalg.norm(quick_mean - full_mean)),
            "latent_covariance_relative_frobenius_gap": float(
                np.linalg.norm(quick_covariance - full_covariance, ord="fro")
                / np.linalg.norm(full_covariance, ord="fro")
            ),
            "full_eigenvalues_descending": full_eigenvalues.tolist(),
            "quick_eigenvalues_descending": quick_ball["eigenvalues_descending"],
            "maximum_uniform_ks_quick": max(quick_ks),
            "maximum_uniform_ks_full": max(full_ks),
            "maximum_uniform_ks_absolute_error": abs(max(quick_ks) - max(full_ks)),
        },
        "tier_2_field_approximation": {
            "complete_domain_fraction_quick": quick_complete,
            "complete_domain_fraction_full": full_complete,
            "complete_domain_fraction_error": quick_complete - full_complete,
            "component_lambda_pearson": float(np.corrcoef(quick_lambda, full_lambda)[0, 1]),
            "component_lambda_spearman": float(spearmanr(quick_lambda, full_lambda).statistic),
            "component_lambda_normalized_l1_error": normalized_l1(quick_lambda, full_lambda),
            "component_energy_pearson": float(np.corrcoef(quick_energy, full_energy)[0, 1]),
            "component_energy_spearman": float(spearmanr(quick_energy, full_energy).statistic),
            "component_energy_normalized_l1_error": normalized_l1(quick_energy, full_energy),
            "full_reached_component_count": len(full_reached),
            "quick_reached_component_count": len(quick_reached),
            "quick_reached_precision": len(full_reached & quick_reached) / len(quick_reached),
            "quick_reached_recall": len(full_reached & quick_reached) / len(full_reached),
            "full_reached_missing_in_quick": sorted(full_reached - quick_reached),
            "quick_only_reached": sorted(quick_reached - full_reached),
        },
        "runtime": quick["runtime"],
        "sealed_test_result_rows_decoded": 0,
        "inputs": {
            "npz_sha256": file_hash(NPZ),
            "full_x1_sha256": file_hash(FULL),
            "quick_assessment_sha256": file_hash(args.assessment),
            "config_sha256": file_hash(CONFIG),
        },
    }
    report["report_hash"] = content_hash(report)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
