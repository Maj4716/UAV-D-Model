"""Build the machine-readable bundle for the stage-wise expected/actual report."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
DATASET = ROOT / "artifacts/datasets/recovery_role_r8_r32_train2000_eval300_20261122"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def digest(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()


def stage_row(mechanism: dict[str, Any], name: str, scope: str) -> dict[str, Any]:
    row = mechanism["stage_comparisons"][name][scope]["all_mixed_R8_R32"]
    return {
        "paired_case_count": row["paired_case_count"],
        "expected_mean": row["expected"]["mean"],
        "actual_mean": row["actual"]["mean"],
        "actual_minus_expected_mean": row["actual_minus_expected_mean"],
        "actual_minus_expected_case_resampling_95_interval": row[
            "actual_minus_expected_mean_case_bootstrap_95_interval"
        ],
        "actual_to_expected_mean_ratio": row["actual_to_expected_mean_ratio"],
        "unit": mechanism["stage_comparisons"][name]["unit"],
    }


def main() -> int:
    work = OUT / "work"
    sampling_path = work / "sampling/sampling_metrics.json"
    teacher_path = work / "teacher/teacher_pair_metrics.json"
    development_path = work / "full_development/full_development_metrics.json"
    proxy_path = work / "proxy_teacher/full_proxy_teacher_distribution.json"
    uncertainty_path = work / "proxy_teacher/full_proxy_uncertainty.json"
    r8_pilot_path = work / "proxy_teacher/v3_proxy_teacher_bootstrap.json"
    stage_map_path = work / "stage_contracts/stage_map.json"
    contact_path = work / "contact_inputs/contact_input_metrics.json"
    mechanism_path = work / "mechanism_funnel/mechanism_pipeline_metrics.json"
    micro_path = work / "mechanism_funnel/micro_audit_metrics.json"
    mechanism_verification_path = work / "mechanism_funnel/verification.json"
    output_gap_path = work / "output_gap/output_gap_metrics.json"

    sampling = load(sampling_path)
    teacher = load(teacher_path)
    development = load(development_path)
    proxy = load(proxy_path)
    uncertainty = load(uncertainty_path)
    r8_pilot = load(r8_pilot_path)
    stage_map = load(stage_map_path)
    contact = load(contact_path)
    mechanism = load(mechanism_path)
    micro = load(micro_path)
    mechanism_verification = load(mechanism_verification_path)
    output_gap = load(output_gap_path)

    config = {
        "schema_version": "pipeline-expected-actual-analysis-config@2.0.0",
        "analysis_id": "pipeline-expected-actual-20261122-v2-stagewise",
        "analysis_status": "development_stagewise_quantitative_assessment",
        "analysis_date": "2026-11-22",
        "expectation_taxonomy": stage_map["expectation_taxonomy"],
        "populations": {
            "candidate_design": 4301,
            "formal_preoutcome_inputs": 2600,
            "formal_split_counts": {"train": 2000, "validation": 300, "test": 300},
            "development_outputs": {
                "case_count": 2300,
                "train_R8": 2000,
                "validation_R32": 300,
                "teacher_replicate_count": 25600,
                "component_count": 67,
            },
            "stratified_micro_audit": {
                "train_case_count": 30,
                "profile_region_cells": 15,
                "cases_per_cell": 2,
                "replicates_per_case": 8,
                "fragment_trace_count": 40800,
            },
        },
        "uncertainty": {
            "unit": "case",
            "draws": 5000,
            "scope": "case-resampling sensitivity conditional on fixed QMC designs",
            "not_covered": ["between_scramble_uncertainty", "physical_population_uncertainty"],
        },
        "preexisting_gates": {
            "sampling": {"positivity_min": 0.1, "ess_fraction_min": 0.5, "weight_cv_max": 2.0},
            "proxy_false_unreached_R8_wilson_upper_max": 0.03,
            "train_epsilon_complete_rate_min": 0.90,
        },
        "sealing": {
            "test_case_count": 300,
            "test_result_json_lines_decoded": 0,
            "test_open_count": 0,
            "test_metrics_computed": False,
            "merged_trace_placeholder_rows_skipped_before_json_loads": True,
        },
        "interpretation_boundaries": {
            "software_implementation": "assessed",
            "generated_data_distribution": "assessed",
            "proxy_teacher_gap": "assessed",
            "external_physical_distribution_gap": "not_estimable_without_external_reference",
        },
    }
    dump(OUT / "analysis_config.json", config)

    candidate = sampling["candidate_generation_and_selection"]
    contact_actual = contact["actual_distributions"]
    full_teacher = development["development_teacher_observability"]
    domain = mechanism["actual_mechanism_distribution"]["domain_censoring"]
    joint_scope = "teacher_and_proxy_domain_complete_cases_by_protocol"
    all_scope = "all_cases_by_protocol"
    probability_joint = mechanism[
        "component_probability_heads_teacher_and_proxy_complete_by_protocol"
    ]["all_mixed_R8_R32"]
    output_heads = output_gap["actual_label_distributions"]
    axis_diagnoses = Counter(
        row["axis_diagnostic"]["diagnosis"]
        for row in micro["fragment_source_contract"]["per_case_contracts"].values()
    )
    tree_joint = mechanism["damage_tree_outputs_domain_complete_cases"][
        "teacher_and_proxy_domain_complete_cases"
    ]

    metrics = {
        "schema_version": "pipeline-expected-actual-metrics@2.0.0",
        "analysis_id": config["analysis_id"],
        "scope": config["analysis_status"],
        "stage_00_03_design_contact_archive": {
            "candidate_count": candidate["candidate_count"],
            "selected_count": candidate["selected_count"],
            "selection_rate": candidate["overall_selection"]["rate"],
            "candidate_profile_tv": candidate[
                "profile_pre_contact_candidate_vs_declared"
            ]["total_variation_distance"],
            "candidate_region_tv": candidate[
                "region_pre_contact_candidate_vs_declared"
            ]["total_variation_distance"],
            "candidate_profile_region_tv": candidate[
                "profile_region_pre_contact_candidate_vs_declared"
            ]["total_variation_distance"],
            "selected_region_tv": candidate[
                "region_formal_selected_vs_pre_contact_declared"
            ]["total_variation_distance"],
            "selection_rate_by_region": candidate["selection_rate_by_region"],
            "contact_input": {
                "speed_mps": contact_actual["speed_mps"],
                "collision_distance_m": contact_actual["collision_distance_m"],
                "collision_time_s": contact_actual["collision_time_s"],
                "incidence_angle_from_normal_deg": contact_actual[
                    "incidence_angle_from_normal_deg"
                ],
                "deterministic_contract_checks": contact[
                    "deterministic_contract_checks"
                ],
            },
            "weight_quality": sampling["weight_quality"],
            "archive_assertions": sampling["assertion_summary"],
        },
        "stage_04_07_aggregate_mechanism": {
            "domain_censoring": domain,
            "joint_complete_stages": {
                name: stage_row(mechanism, name, joint_scope)
                for name in (
                    "source_fragments",
                    "armor_hits",
                    "armor_penetrations",
                    "component_hits",
                    "component_penetrations",
                    "component_deposited_energy_j",
                    "component_damage_energy_j",
                )
            },
            "all_case_operational_stages": {
                name: stage_row(mechanism, name, all_scope)
                for name in (
                    "source_fragments",
                    "armor_hits",
                    "armor_penetrations",
                    "component_hits",
                    "component_penetrations",
                    "component_deposited_energy_j",
                    "component_damage_energy_j",
                )
            },
            "joint_complete_conversion_rates": mechanism["conversion_rates"][
                "teacher_and_proxy_domain_complete_cases"
            ]["armor_hit_per_source_fragment"],
            "all_conversion_rates": mechanism["conversion_rates"],
        },
        "stage_04_09_micro_audit": {
            "selection": micro["selection"],
            "archive_equivalence": micro["archive_equivalence"],
            "fragment_source_contract": {
                key: value
                for key, value in micro["fragment_source_contract"].items()
                if key != "per_case_contracts"
            },
            "axis_diagnosis_counts": dict(axis_diagnoses),
            "fragment_path_funnel": micro["fragment_path_funnel"],
            "component_reachability": {
                key: value
                for key, value in micro["component_reachability"].items()
                if key != "by_component"
            },
            "penetration_probability_calibration": micro[
                "penetration_probability_calibration"
            ],
            "matching_margins": micro["matching_margins"],
            "post_armor_component_reach": micro["post_armor_component_reach"],
            "evidence_boundaries": micro["evidence_boundaries"],
        },
        "stage_08_11_teacher_labels_outputs": {
            "teacher_observability": full_teacher,
            "trace_contracts": output_gap["trace_contracts"],
            "label_contracts": output_gap["label_contracts"],
            "actual_label_distributions": {
                name: {
                    "development_observed_pairs": row["development_observed_pairs"],
                    "fixed_67_component_identified_case_mean": row[
                        "fixed_67_component_identified_case_mean"
                    ],
                    "complete_67_component_case_analysis": row[
                        "complete_67_component_case_analysis"
                    ],
                    "missingness_sensitive_case_mean_analysis": row[
                        "missingness_sensitive_case_mean_analysis"
                    ],
                }
                for name, row in output_heads.items()
            },
            "conditional_positive_distributions": output_gap[
                "conditional_positive_distributions"
            ],
            "profile_region_strata": output_gap["profile_region_strata"],
        },
        "stage_09_10_proxy_states_and_tree": {
            "joint_complete_probability_heads": probability_joint,
            "trace_aggregation_consistency": mechanism[
                "trace_aggregation_consistency"
            ],
            "joint_complete_tree_outputs": tree_joint,
        },
        "stage_12_13_proxy_admission_and_sealing": {
            "full_mixed_protocol_proxy_teacher": {
                "brier": proxy["metrics"]["hit_probability"]["brier"],
                "ece": proxy["metrics"]["hit_probability"]["ece"],
                "jensen_shannon_fixed_bins": proxy["metrics"][
                    "hit_probability"
                ]["jensen_shannon_fixed_bins"],
                "positive_energy_log1p_wasserstein1": proxy["metrics"][
                    "positive_energy"
                ]["log1p_wasserstein1"],
                "multivariate_hit_probability_mmd2": proxy["metrics"][
                    "multivariate_hit_probability"
                ]["mmd2"],
                "uncertainty": uncertainty["intervals"],
            },
            "r8_pilot_promotion_gate": {
                "rate": r8_pilot["point_estimates"]["false_unreached_rate"],
                "wilson_upper": r8_pilot["point_estimates"][
                    "false_unreached_wilson_upper"
                ],
                "gate_upper_max": 0.03,
                "gate_passed": False,
            },
            "task_admission": development["task_admission"],
            "global_training_gate": development["global_training_gate"],
            "sealed_test": config["sealing"],
        },
        "anisotropic_sphere_assessment": {
            "implemented_as_formal_estimator": False,
            "recommended_role": "diagnostic_visualization_or_proposal_only",
            "formal_replacement_status": "not_promoted",
            "required_validation": "train-fit validation-evaluate paired comparison under the same R protocol",
        },
        "external_physical_distribution_gap": {
            "status": "not_estimable",
            "reason": "no independent physical or operational reference distribution",
        },
        "mechanism_verification": mechanism_verification,
    }
    dump(OUT / "metrics.json", metrics)

    evidence_paths = [
        OUT / "REPORT.md",
        OUT / "ANALYSIS_PLAN.md",
        OUT / "build_report_bundle.py",
        OUT / "verify_report_bundle.py",
        OUT / "analysis_config.json",
        OUT / "metrics.json",
        sampling_path,
        work / "sampling/analyze_sampling_design.py",
        teacher_path,
        work / "teacher/TEACHER_PAIR_AUDIT.md",
        work / "teacher/analyze_teacher_pair.py",
        development_path,
        work / "full_development/SUMMARY.md",
        proxy_path,
        uncertainty_path,
        r8_pilot_path,
        work / "proxy_teacher/full_proxy_uncertainty.py",
        stage_map_path,
        work / "stage_contracts/STAGE_MAP.md",
        contact_path,
        work / "contact_inputs/SUMMARY.md",
        work / "contact_inputs/analyze_contact_inputs.py",
        work / "contact_inputs/verify_contact_inputs.py",
        mechanism_path,
        micro_path,
        mechanism_verification_path,
        work / "mechanism_funnel/MECHANISM_ANALYSIS_SUMMARY.md",
        work / "mechanism_funnel/analyze_mechanism_pipeline.py",
        work / "mechanism_funnel/run_stratified_micro_audit.py",
        output_gap_path,
        work / "output_gap/SUMMARY.md",
        work / "output_gap/analyze_output_distribution_gap.py",
        work / "output_gap/verify_output_gap.py",
        DATASET / "sampling_checkpoint.json",
        DATASET / "audits/sampling_quality.json",
        DATASET / "formal_dataset/manifest.json",
        DATASET / "development_audit.json",
        DATASET / "dataset_v2_development/manifest.json",
        DATASET / "dataset_v2_development/task_admission.json",
        DATASET / "dataset_v2_development/model_readiness.json",
        ROOT / "damage_engine/distribution_metrics.py",
        ROOT / "damage_engine/fast_sampling_proxy.py",
        ROOT / "damage_engine/fragment_audit.py",
        ROOT / "tools/evaluate_proxy_teacher_distribution.py",
    ]
    external_idea = Path(r"D:\Desktop\idea\idea2.md")
    if external_idea.exists():
        evidence_paths.append(external_idea)

    rows = []
    for path in evidence_paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        rows.append(
            {
                "path": str(resolved),
                "workspace_relative_path": (
                    str(resolved.relative_to(ROOT)) if ROOT in resolved.parents else None
                ),
                "size_bytes": resolved.stat().st_size,
                "sha256": digest(resolved),
            }
        )
    manifest = {
        "schema_version": "pipeline-assessment-evidence-manifest@2.0.0",
        "analysis_id": config["analysis_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_count": len(rows),
        "sealed_test_result_json_lines_decoded": 0,
        "files": rows,
    }
    dump(OUT / "evidence_manifest.json", manifest)
    print(
        json.dumps(
            {
                "analysis_id": config["analysis_id"],
                "metrics": str(OUT / "metrics.json"),
                "evidence_manifest": str(OUT / "evidence_manifest.json"),
                "evidence_count": len(rows),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
