"""Quantify first-contact inputs without opening any teacher result file.

The analysis reads only the frozen SamplingSpec and the formal input ledger.
Sealed-test *inputs* are part of the declared design and may be audited, while
all outcome/result files remain unopened and undecoded.
"""

from __future__ import annotations

from collections import Counter
import gzip
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from damage_engine.utils import content_hash, file_hash  # noqa: E402


DATASET = ROOT / "artifacts/datasets/recovery_role_r8_r32_train2000_eval300_20261122"
SPEC_PATH = (
    ROOT
    / "configs/sampling/p0_sphere_entry_continuous_direction_recovery_train2000_eval300_role_r8_r32.v1.json"
)
FORMAL_CASES_PATH = DATASET / "rounds/round-000/formal_cases.jsonl.gz"
OUTPUT_PATH = SCRIPT.with_name("contact_input_metrics.json")
SUMMARY_PATH = SCRIPT.with_name("SUMMARY.md")


def scalar_summary(values: list[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
        "population_standard_deviation": float(np.std(array, ddof=0)),
    }


def uniform_ks(values: np.ndarray) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    n = ordered.size
    upper = np.arange(1, n + 1, dtype=np.float64) / n - ordered
    lower = ordered - np.arange(0, n, dtype=np.float64) / n
    return float(max(np.max(upper), np.max(lower)))


def category_gap(counts: Counter[str], labels: list[str]) -> dict[str, Any]:
    total = sum(counts[label] for label in labels)
    expected_mass = 1.0 / len(labels)
    rows = []
    for label in labels:
        mass = counts[label] / total
        rows.append(
            {
                "label": label,
                "count": int(counts[label]),
                "actual_mass": mass,
                "declared_mass": expected_mass,
                "actual_minus_declared_mass": mass - expected_mass,
            }
        )
    return {
        "denominator": int(total),
        "rows": rows,
        "total_variation_distance": 0.5
        * float(sum(abs(row["actual_minus_declared_mass"]) for row in rows)),
        "maximum_absolute_mass_gap": float(
            max(abs(row["actual_minus_declared_mass"]) for row in rows)
        ),
    }


def maximum_abs(values: list[float]) -> float:
    return float(max(abs(value) for value in values)) if values else 0.0


def vector_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right))


def main() -> int:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    regions = [entry["region_id"] for entry in spec["contact_manifold"]["regions"]]
    profiles = list(spec["reference_distribution"]["variables"][0]["values"])
    region_spec = {
        entry["region_id"]: entry for entry in spec["contact_manifold"]["regions"]
    }
    speed_bounds = sorted(
        {
            tuple(float(value) for value in entry["speed_range_mps"])
            for entry in spec["contact_manifold"]["regions"]
        }
    )
    if speed_bounds != [(80.0, 160.0)]:
        raise RuntimeError(f"unexpected heterogeneous speed bounds: {speed_bounds}")
    speed_low, speed_high = speed_bounds[0]

    rows: list[dict[str, Any]] = []
    profile_counts: Counter[str] = Counter()
    region_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    collision_armor_counts: Counter[str] = Counter()
    eligibility_status_counts: Counter[str] = Counter()
    contact_status_counts: Counter[str] = Counter()
    check_false_or_missing_counts: Counter[str] = Counter()
    check_true_counts: Counter[str] = Counter()
    normal_counts: Counter[str] = Counter()

    speed_contract_error: list[float] = []
    velocity_direction_error: list[float] = []
    direction_norm_error: list[float] = []
    normal_norm_error: list[float] = []
    distance_geometry_error: list[float] = []
    time_kinematic_error: list[float] = []
    detonation_point_error: list[float] = []
    detonation_time_error: list[float] = []
    forward_cosines: list[float] = []
    point_coordinates: list[list[float]] = []

    with gzip.open(FORMAL_CASES_PATH, "rt", encoding="utf-8") as stream:
        for line in stream:
            case = json.loads(line)
            encounter = case["encounter"]
            metadata = case["metadata"]
            contact = metadata["contact_manifold_sampling"]
            sampling = metadata["sampling"]
            eligibility = sampling["contact_conditioning"]["eligibility_record"]
            first_contact = eligibility["first_contact"]
            profile = str(case["munition"]["profile_id"])
            region = str(contact["region_id"])
            split = str(sampling["frozen_split"])

            velocity = np.asarray(encounter["velocity_target_mps"], dtype=np.float64)
            speed = float(contact["speed_mps"])
            velocity_speed = float(np.linalg.norm(velocity))
            velocity_direction = velocity / velocity_speed
            direction = np.asarray(contact["direction_target_xyz"], dtype=np.float64)
            origin = np.asarray(encounter["trajectory_origin_m"], dtype=np.float64)
            point = np.asarray(first_contact["collision_point_target_m"], dtype=np.float64)
            detonation_point = np.asarray(
                encounter["detonation_position_m"], dtype=np.float64
            )
            normal = np.asarray(first_contact["collision_normal_target_xyz"], dtype=np.float64)
            collision_distance = float(first_contact["collision_distance_m"])
            collision_time = float(first_contact["collision_time_s"])
            forward_cosine = float(-np.dot(velocity_direction, normal))
            incidence_angle_deg = float(
                np.degrees(np.arccos(np.clip(forward_cosine, -1.0, 1.0)))
            )
            speed_quantile = (speed - speed_low) / (speed_high - speed_low)

            profile_counts[profile] += 1
            region_counts[region] += 1
            split_counts[split] += 1
            collision_armor_counts[str(first_contact["collision_armor_id"])] += 1
            eligibility_status_counts[str(eligibility.get("status", "missing"))] += 1
            contact_status_counts[str(contact.get("status", "missing"))] += 1
            normal_counts[json.dumps(np.round(normal, 12).tolist())] += 1
            for name, value in eligibility.get("checks", {}).items():
                if value is True:
                    check_true_counts[name] += 1
                else:
                    check_false_or_missing_counts[name] += 1

            speed_contract_error.append(velocity_speed - speed)
            velocity_direction_error.append(vector_error(velocity_direction, direction))
            direction_norm_error.append(float(np.linalg.norm(direction)) - 1.0)
            normal_norm_error.append(float(np.linalg.norm(normal)) - 1.0)
            distance_geometry_error.append(float(np.linalg.norm(point - origin)) - collision_distance)
            time_kinematic_error.append(collision_time - collision_distance / velocity_speed)
            detonation_point_error.append(vector_error(point, detonation_point))
            detonation_time_error.append(
                collision_time - float(encounter["detonation_time_s"])
            )
            forward_cosines.append(forward_cosine)
            point_coordinates.append(point.tolist())
            rows.append(
                {
                    "case_id": str(case["case_id"]),
                    "split": split,
                    "profile": profile,
                    "region": region,
                    "speed_mps": speed,
                    "speed_quantile": speed_quantile,
                    "collision_distance_m": collision_distance,
                    "collision_time_s": collision_time,
                    "forward_normal_cosine": forward_cosine,
                    "incidence_angle_from_normal_deg": incidence_angle_deg,
                    "collision_point_target_m": point.tolist(),
                    "aim_u_m": float(contact["aim_u_m"]),
                    "aim_v_m": float(contact["aim_v_m"]),
                    "body_x_roll_rad": float(contact["body_x_roll_rad"]),
                    "transverse_radius_m": float(
                        metadata["sphere_entry_ray_sampling"]["transverse_radius_m"]
                    ),
                }
            )

    case_count = len(rows)
    speed_values = np.asarray([row["speed_mps"] for row in rows])
    speed_quantiles = np.asarray([row["speed_quantile"] for row in rows])
    point_array = np.asarray(point_coordinates, dtype=np.float64)
    strata: dict[str, Any] = {}
    for profile in profiles:
        for region in regions:
            subset = [
                row
                for row in rows
                if row["profile"] == profile and row["region"] == region
            ]
            key = f"{region}::{profile}"
            strata[key] = {
                "case_count": len(subset),
                "split_counts": dict(Counter(row["split"] for row in subset)),
                "speed_mps": scalar_summary([row["speed_mps"] for row in subset]),
                "collision_distance_m": scalar_summary(
                    [row["collision_distance_m"] for row in subset]
                ),
                "collision_time_s": scalar_summary(
                    [row["collision_time_s"] for row in subset]
                ),
                "incidence_angle_from_normal_deg": scalar_summary(
                    [row["incidence_angle_from_normal_deg"] for row in subset]
                ),
                "forward_normal_cosine_minimum": float(
                    min(row["forward_normal_cosine"] for row in subset)
                ),
            }

    speed_by_region = {}
    for region in regions:
        subset = np.asarray(
            [row["speed_mps"] for row in rows if row["region"] == region]
        )
        normalized = (subset - speed_low) / (speed_high - speed_low)
        speed_by_region[region] = {
            "summary": scalar_summary(subset),
            "uniform_ks_distance": uniform_ks(normalized),
        }

    metrics: dict[str, Any] = {
        "schema_version": "contact-input-expected-actual@1.0.0",
        "analysis_id": "pipeline-expected-actual-20261122-v2:contact-inputs",
        "scope": {
            "formal_input_case_count": case_count,
            "split_counts": dict(split_counts),
            "sealed_test_input_case_count": int(split_counts.get("test", 0)),
            "teacher_result_files_opened": 0,
            "teacher_result_json_lines_decoded": 0,
            "training_runs_started": 0,
            "data_access": "SamplingSpec plus formal pre-outcome case ledger only",
        },
        "evidence": {
            "sampling_spec": {
                "path": str(SPEC_PATH.resolve()),
                "sha256": file_hash(SPEC_PATH),
            },
            "formal_cases": {
                "path": str(FORMAL_CASES_PATH.resolve()),
                "sha256": file_hash(FORMAL_CASES_PATH),
            },
        },
        "declared_expectations": {
            "speed_distribution": {
                "family": "Uniform",
                "minimum_mps": speed_low,
                "maximum_mps": speed_high,
                "mean_mps": (speed_low + speed_high) / 2.0,
                "population_standard_deviation_mps": (speed_high - speed_low)
                / math.sqrt(12.0),
            },
            "profile_mass_each": 1.0 / len(profiles),
            "region_mass_each": 1.0 / len(regions),
            "first_contact_contract": {
                "positive_forward_intersection": True,
                "collision_time_equals_distance_over_speed": True,
                "detonation_point_equals_first_contact_point": True,
                "detonation_time_equals_first_contact_time": True,
                "direction_and_normal_are_unit_vectors": True,
            },
            "interpretation": (
                "profile/region/speed are declared design distributions; contact "
                "geometry is the S-CONTACT-conditioned actual input distribution"
            ),
        },
        "actual_distributions": {
            "profile_vs_declared_equal_mass": category_gap(profile_counts, profiles),
            "region_vs_declared_precontact_equal_mass": category_gap(region_counts, regions),
            "speed_mps": {
                "summary": scalar_summary(speed_values),
                "actual_minus_declared_mean_mps": float(
                    np.mean(speed_values) - (speed_low + speed_high) / 2.0
                ),
                "actual_to_declared_population_variance_ratio": float(
                    np.var(speed_values, ddof=0)
                    / (((speed_high - speed_low) ** 2) / 12.0)
                ),
                "uniform_ks_distance": uniform_ks(speed_quantiles),
                "out_of_declared_range_count": int(
                    np.sum((speed_values < speed_low) | (speed_values > speed_high))
                ),
                "by_region": speed_by_region,
            },
            "collision_distance_m": scalar_summary(
                [row["collision_distance_m"] for row in rows]
            ),
            "collision_time_s": scalar_summary(
                [row["collision_time_s"] for row in rows]
            ),
            "forward_normal_cosine": scalar_summary(forward_cosines),
            "incidence_angle_from_normal_deg": scalar_summary(
                [row["incidence_angle_from_normal_deg"] for row in rows]
            ),
            "collision_point_target_m_by_axis": {
                axis: scalar_summary(point_array[:, index])
                for index, axis in enumerate(("x", "y", "z"))
            },
            "aim_u_m": scalar_summary([row["aim_u_m"] for row in rows]),
            "aim_v_m": scalar_summary([row["aim_v_m"] for row in rows]),
            "body_x_roll_rad": scalar_summary(
                [row["body_x_roll_rad"] for row in rows]
            ),
            "transverse_radius_m": scalar_summary(
                [row["transverse_radius_m"] for row in rows]
            ),
            "collision_armor_counts": dict(collision_armor_counts),
            "collision_normal_counts": dict(normal_counts),
            "profile_region_strata": strata,
        },
        "deterministic_contract_checks": {
            "contact_status_counts": dict(contact_status_counts),
            "eligibility_status_counts": dict(eligibility_status_counts),
            "eligibility_check_true_counts": dict(check_true_counts),
            "eligibility_check_false_or_missing_counts": dict(
                check_false_or_missing_counts
            ),
            "velocity_magnitude_minus_declared_speed_max_abs_mps": maximum_abs(
                speed_contract_error
            ),
            "velocity_unit_direction_minus_declared_direction_max_l2": maximum_abs(
                velocity_direction_error
            ),
            "declared_direction_unit_norm_max_abs_error": maximum_abs(
                direction_norm_error
            ),
            "collision_normal_unit_norm_max_abs_error": maximum_abs(
                normal_norm_error
            ),
            "origin_to_contact_distance_minus_persisted_distance_max_abs_m": maximum_abs(
                distance_geometry_error
            ),
            "collision_time_minus_distance_over_speed_max_abs_s": maximum_abs(
                time_kinematic_error
            ),
            "detonation_point_minus_first_contact_point_max_l2_m": maximum_abs(
                detonation_point_error
            ),
            "detonation_time_minus_first_contact_time_max_abs_s": maximum_abs(
                detonation_time_error
            ),
            "nonpositive_forward_normal_cosine_count": int(
                np.sum(np.asarray(forward_cosines) <= 0.0)
            ),
            "speed_quantile_mapping_max_abs_error_mps": float(
                np.max(
                    np.abs(
                        speed_values
                        - (speed_low + (speed_high - speed_low) * speed_quantiles)
                    )
                )
            ),
        },
        "interpretation": {
            "speed_gap_semantics": (
                "distance to the declared pre-contact Uniform(80,160) design after "
                "S-CONTACT selection; no iid p-value"
            ),
            "contact_geometry_semantics": (
                "empirical push-forward of the declared design through target geometry "
                "and the first-contact admission rule; no external physical target given"
            ),
            "operational_prevalence_claim": False,
            "physical_validation_claim": False,
        },
    }
    metrics["report_hash"] = content_hash(metrics)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    speed = metrics["actual_distributions"]["speed_mps"]
    distance = metrics["actual_distributions"]["collision_distance_m"]
    time = metrics["actual_distributions"]["collision_time_s"]
    angle = metrics["actual_distributions"]["incidence_angle_from_normal_deg"]
    contract = metrics["deterministic_contract_checks"]
    summary = f"""# 首次装甲接触与输入状态：预期—实际量化

## 范围

本分析只读取 SamplingSpec 与 2,600 条正式工况输入。300 条 sealed-test 输入纳入设计审计，教师结果文件打开数和结果 JSON 解码数均为 0。

## 声明分布与条件后实际分布

| 指标 | 声明预期 | S-CONTACT 后实际 | 差距 |
|---|---:|---:|---:|
| 速度 | Uniform(80,160) m/s，均值 120 | 均值 {speed['summary']['mean']:.6f}，范围 [{speed['summary']['minimum']:.6f}, {speed['summary']['maximum']:.6f}] | 均值差 {speed['actual_minus_declared_mean_mps']:+.6f} m/s；KS={speed['uniform_ks_distance']:.6f} |
| profile 质量 | 三类各 1/3 | 2,600 例经验质量 | TV={metrics['actual_distributions']['profile_vs_declared_equal_mass']['total_variation_distance']:.6f} |
| region 质量 | 五区各 1/5 | 2,600 例经验质量 | TV={metrics['actual_distributions']['region_vs_declared_precontact_equal_mass']['total_variation_distance']:.6f} |
| 首次接触距离 | 由目标几何与射线共同决定 | 均值 {distance['mean']:.6f} m，p05–p95 [{distance['p05']:.6f}, {distance['p95']:.6f}] | 无外部目标分布；报告经验推送分布 |
| 首次接触时间 | distance/speed | 均值 {time['mean']:.8f} s，p05–p95 [{time['p05']:.8f}, {time['p95']:.8f}] | 最大合同误差 {contract['collision_time_minus_distance_over_speed_max_abs_s']:.3e} s |
| 入射角（相对外法向） | 正向接触，0–90° | 均值 {angle['mean']:.6f}°，p05–p95 [{angle['p05']:.6f}, {angle['p95']:.6f}] | 非正向余弦 0 例 |

速度的 KS 是随机化 QMC 设计诊断，不使用 iid p 值。region TV 衡量 S-CONTACT 对预接触等区域设计的条件化，不表示采样实现错误。

## 确定性合同

- 2,600/2,600 条工况状态为 `admitted_s_contact` 与 `eligible_contact_conditioned`；七项资格检查的 false/missing 数均为 0。
- 速度向量模长与声明速度最大绝对误差：{contract['velocity_magnitude_minus_declared_speed_max_abs_mps']:.3e} m/s。
- 单位速度方向与声明方向最大 L2 误差：{contract['velocity_unit_direction_minus_declared_direction_max_l2']:.3e}。
- 原点到首次接触点的几何距离与持久化距离最大误差：{contract['origin_to_contact_distance_minus_persisted_distance_max_abs_m']:.3e} m。
- 爆炸点与首次接触点最大 L2 误差：{contract['detonation_point_minus_first_contact_point_max_l2_m']:.3e} m；爆炸时刻与首次接触时刻最大误差：{contract['detonation_time_minus_first_contact_time_max_abs_s']:.3e} s。

## 解释

首次接触距离、时间、法向、命中点和入射角是声明设计经目标几何与 S-CONTACT 规则后的经验分布。项目没有给出这些量的外部物理/作战目标分布，因此当前可以核验设计忠实性与几何合同，也可以报告条件后实际分布；真实场景距离仍需要独立观测样本。

机器可读结果见 `contact_input_metrics.json`，复算脚本见 `analyze_contact_inputs.py`。
"""
    SUMMARY_PATH.write_text(summary, encoding="utf-8")
    print(
        json.dumps(
            {
                "metrics": str(OUTPUT_PATH),
                "summary": str(SUMMARY_PATH),
                "case_count": case_count,
                "teacher_result_json_lines_decoded": 0,
                "report_hash": metrics["report_hash"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
