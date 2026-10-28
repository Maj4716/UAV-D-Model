from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency, ks_2samp


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "raw_data" / "dataset_5000_loitering_impact_fragment_only_v1"
ARCHIVE = DATASET / "formal_dataset"
OUT_DIR = Path(__file__).resolve().parent

KMFC_COMPONENTS = {
    "K": ["component-003", "component-046", "component-053", "component-054"],
    "M": [
        *(f"component-{i:03d}" for i in range(1, 40)),
        "component-058",
    ],
    "F": [
        *(f"component-{i:03d}" for i in range(40, 55)),
    ],
    "C": [
        *(f"component-{i:03d}" for i in range(58, 68)),
    ],
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float | None]:
    if total <= 0:
        return [None, None]
    p = successes / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [max(0.0, center - half), min(1.0, center + half)]


def _entropy(counts: list[int]) -> float:
    n = sum(counts)
    if n == 0:
        return 0.0
    p = np.asarray([v / n for v in counts if v], dtype=float)
    return float(-(p * np.log2(p)).sum())


def _distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    rounded = np.round(values, 12)
    unique, counts = np.unique(rounded, return_counts=True)
    discrete = {format(float(k), ".12g"): int(v) for k, v in zip(unique, counts)}
    n = int(values.size)
    pos = int(np.count_nonzero(values > 0.0))
    zero = n - pos
    p = pos / n if n else 0.0
    h = _entropy(counts.tolist())
    hhi = float(np.sum((counts / n) ** 2)) if n else None
    r2_levels = np.asarray([0.0, 0.5, 1.0])
    nearest_index = np.argmin(np.abs(values[:, None] - r2_levels[None, :]), axis=1)
    nearest = r2_levels[nearest_index]
    r2_counts = Counter(format(float(v), ".1f") for v in nearest)
    r2_count_vector = [r2_counts.get(k, 0) for k in ("0.0", "0.5", "1.0")]
    r2_entropy = _entropy(r2_count_vector)
    return {
        "n": n,
        "zero_count": zero,
        "zero_fraction": zero / n if n else None,
        "positive_count": pos,
        "positive_fraction": p if n else None,
        "positive_rate_wilson95": _wilson(pos, n),
        "exact_one_count": int(np.count_nonzero(values == 1.0)),
        "unique_value_count": int(unique.size),
        "value_counts": discrete,
        "discrete_entropy_bits": h,
        "r2_nearest_level_counts": dict(r2_counts),
        "r2_nearest_level_max_absolute_deviation": float(np.max(np.abs(values - nearest))) if n else None,
        "r2_quantized_entropy_bits": r2_entropy,
        "r2_quantized_normalized_entropy": r2_entropy / math.log2(3.0),
        "hhi": hhi,
        "effective_discrete_values": float(2**h),
        "binary_balance_effective_n": float(4 * n * p * (1 - p)),
        "mean": float(values.mean()) if n else None,
        "std": float(values.std(ddof=0)) if n else None,
    }


def _load() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[Path]]:
    shards = sorted((ARCHIVE / "shards").glob("part-*"))
    if not shards:
        raise RuntimeError(f"no shards under {ARCHIVE}")
    per_case = [
        "case_x",
        "case_feature_mask",
        "loitering_impact_x",
        "loitering_impact_feature_mask",
        "system_y",
        "system_label_mask",
        "system_sample_count",
        "component_aux_y",
        "component_aux_label_mask",
        "split_names",
        "case_ids",
        "replicate_counts",
    ]
    chunks: dict[str, list[np.ndarray]] = {k: [] for k in per_case}
    static: dict[str, np.ndarray] = {}
    for shard in shards:
        with np.load(shard / "tensors.npz", allow_pickle=False) as z:
            for key in per_case:
                chunks[key].append(z[key])
            if not static:
                for key in (
                    "case_feature_names",
                    "loitering_impact_feature_names",
                    "system_output_names",
                    "component_ids",
                    "component_aux_label_names",
                ):
                    static[key] = z[key].copy()
    arrays = {k: np.concatenate(v, axis=0) for k, v in chunks.items()}
    return arrays, static, shards


def _index(names: np.ndarray, name: str) -> int:
    matches = np.where(names.astype(str) == name)[0]
    if matches.size != 1:
        raise KeyError(name)
    return int(matches[0])


def _split_profile_tests(x: np.ndarray, splits: np.ndarray, profiles: np.ndarray) -> dict:
    out: dict[str, dict] = {"continuous_ks": {}, "profile_chi_square": {}}
    for j, name in enumerate(("trajectory_origin_y_m", "trajectory_origin_z_m", "impact_speed_mps"), start=3):
        pairs = {}
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            result = ks_2samp(x[splits == a, j], x[splits == b, j], method="auto")
            pairs[f"{a}_vs_{b}"] = {"D": float(result.statistic), "p_value": float(result.pvalue)}
        out["continuous_ks"][name] = pairs
    split_levels = ["train", "validation", "test"]
    profile_levels = ["small_front", "medium_front", "large_front"]
    table = np.asarray(
        [[np.count_nonzero((splits == s) & (profiles == p)) for p in profile_levels] for s in split_levels],
        dtype=int,
    )
    stat, p, dof, expected = chi2_contingency(table)
    out["profile_chi_square"] = {
        "table_rows": split_levels,
        "table_columns": profile_levels,
        "observed": table.tolist(),
        "chi2": float(stat),
        "dof": int(dof),
        "p_value": float(p),
        "minimum_expected_count": float(expected.min()),
    }
    return out


def _binary_heterogeneity(positive: np.ndarray, groups: np.ndarray, levels: tuple[str, ...]) -> dict:
    table = np.asarray(
        [[np.count_nonzero((groups == level) & positive), np.count_nonzero((groups == level) & ~positive)] for level in levels],
        dtype=int,
    )
    if table[:, 0].sum() == 0 or table[:, 1].sum() == 0:
        return {
            "estimable": False,
            "levels": list(levels),
            "observed_positive_negative": table.tolist(),
            "reason": "one outcome column is identically zero",
        }
    stat, p, dof, expected = chi2_contingency(table)
    rates = table[:, 0] / table.sum(axis=1)
    group_sizes = table.sum(axis=1)
    total_positive = int(table[:, 0].sum())
    draws = 100_000
    rng = np.random.default_rng(20261021 + total_positive + int(group_sizes[0]))
    simulated_positive = rng.multivariate_hypergeometric(group_sizes, total_positive, size=draws)
    expected_positive = group_sizes * total_positive / int(group_sizes.sum())
    expected_negative = group_sizes - expected_positive
    simulated_negative = group_sizes[None, :] - simulated_positive
    simulated_chi2 = (
        ((simulated_positive - expected_positive[None, :]) ** 2 / expected_positive[None, :]).sum(axis=1)
        + ((simulated_negative - expected_negative[None, :]) ** 2 / expected_negative[None, :]).sum(axis=1)
    )
    permutation_p = (int(np.count_nonzero(simulated_chi2 >= stat - 1e-15)) + 1) / (draws + 1)
    return {
        "estimable": True,
        "levels": list(levels),
        "observed_positive_negative": table.tolist(),
        "positive_rates": rates.tolist(),
        "maximum_to_minimum_rate_ratio": float(rates.max() / rates.min()) if rates.min() > 0 else None,
        "chi2": float(stat),
        "dof": int(dof),
        "p_value": float(p),
        "conditional_monte_carlo_p_value": float(permutation_p),
        "conditional_monte_carlo_draws": draws,
        "conditional_monte_carlo_standard_error_upper_bound": float(math.sqrt(permutation_p * (1 - permutation_p) / draws)),
        "minimum_expected_count": float(expected.min()),
        "asymptotic_warning": bool(expected.min() < 5),
    }


def build() -> dict:
    a, s, shards = _load()
    case_names = s["case_feature_names"].astype(str)
    li_names = s["loitering_impact_feature_names"].astype(str)
    output_names = s["system_output_names"].astype(str)
    component_ids = s["component_ids"].astype(str)
    aux_names = s["component_aux_label_names"].astype(str)
    splits = a["split_names"].astype(str)
    li = a["loitering_impact_x"].astype(float)
    profiles = np.asarray([("small_front", "medium_front", "large_front")[i] for i in np.argmax(li[:, :3], axis=1)])
    case_x = a["case_x"].astype(float)

    orient_idx = [_index(case_names, n) for n in ("orientation_qx", "orientation_qy", "orientation_qz", "orientation_qw")]
    vel_idx = [_index(case_names, n) for n in ("velocity_x_mps", "velocity_y_mps", "velocity_z_mps")]
    det_idx = [_index(case_names, n) for n in ("detonation_x_m", "detonation_y_m", "detonation_z_m")]
    velocities = case_x[:, vel_idx]
    directions = velocities / np.linalg.norm(velocities, axis=1, keepdims=True)
    orientations = case_x[:, orient_idx]
    detonation = case_x[:, det_idx]

    continuous = {}
    for j, name in enumerate(li_names[3:], start=3):
        v = li[:, j]
        hist, _ = np.histogram(v, bins=10, range=(((-3, 2) if j == 3 else (0.65, 1.8)) if j < 5 else (80, 160)))
        continuous[name] = {
            "minimum": float(v.min()),
            "maximum": float(v.max()),
            "mean": float(v.mean()),
            "std": float(v.std(ddof=0)),
            "unique_count": int(np.unique(v).size),
            "ten_bin_counts": hist.tolist(),
            "ten_bin_minimum": int(hist.min()),
            "ten_bin_maximum": int(hist.max()),
        }
    yz_hist, _, _ = np.histogram2d(li[:, 3], li[:, 4], bins=(10, 10), range=((-3, 2), (0.65, 1.8)))
    corr = np.corrcoef(li[:, 3:6], rowvar=False)

    labels: dict[str, dict] = {}
    for j, name in enumerate(output_names):
        valid = a["system_label_mask"][:, j]
        values = a["system_y"][valid, j]
        positive = valid & (a["system_y"][:, j] > 0.0)
        task = {
            "all": _distribution(values),
            "by_split": {},
            "by_profile": {},
            "positive_rate_heterogeneity": {
                "by_split": _binary_heterogeneity(positive, splits, ("train", "validation", "test")),
                "by_profile": _binary_heterogeneity(positive, profiles, ("small_front", "medium_front", "large_front")),
            },
        }
        for split in ("train", "validation", "test"):
            mask = valid & (splits == split)
            task["by_split"][split] = _distribution(a["system_y"][mask, j])
        for profile in ("small_front", "medium_front", "large_front"):
            mask = valid & (profiles == profile)
            task["by_profile"][profile] = _distribution(a["system_y"][mask, j])
        if np.any(positive):
            py = li[positive, 3]
            pz = li[positive, 4]
            ps = li[positive, 5]
            pyz_hist, _, _ = np.histogram2d(py, pz, bins=(10, 10), range=((-3, 2), (0.65, 1.8)))
            task["positive_input_support"] = {
                "trajectory_origin_y_min_max": [float(py.min()), float(py.max())],
                "trajectory_origin_z_min_max": [float(pz.min()), float(pz.max())],
                "impact_speed_min_max": [float(ps.min()), float(ps.max())],
                "yz_10x10_occupied_cells": int(np.count_nonzero(pyz_hist)),
                "detonation_x_counts": {
                    str(k): int(v)
                    for k, v in zip(*np.unique(np.round(detonation[positive, 0], 12), return_counts=True))
                },
            }
        else:
            task["positive_input_support"] = None
        labels[name] = task

    aux = a["component_aux_y"].astype(float)
    aux_mask = a["component_aux_label_mask"]
    metric_indices = {
        "hit": _index(aux_names, "fragment_hits"),
        "penetration": _index(aux_names, "fragment_penetrations"),
        "energy": _index(aux_names, "fragment_deposited_energy_j"),
    }
    upstream: dict[str, dict] = {}
    for category, wanted in KMFC_COMPONENTS.items():
        ids = [i for i, c in enumerate(component_ids) if c in wanted]
        block = {"component_count": len(ids), "stages": {}, "by_profile": {}}
        stage_masks = {}
        for stage, metric_idx in metric_indices.items():
            available = aux_mask[:, ids, metric_idx]
            positive = available & (aux[:, ids, metric_idx] > 0)
            case_positive = np.any(positive, axis=1)
            stage_masks[stage] = case_positive
            count = int(case_positive.sum())
            block["stages"][stage] = {
                "case_count": count,
                "case_fraction": count / len(splits),
                "wilson95": _wilson(count, len(splits)),
            }
        block["stage_monotonicity_violations"] = {
            "penetration_without_hit": int(np.count_nonzero(stage_masks["penetration"] & ~stage_masks["hit"])),
            "energy_without_penetration": int(np.count_nonzero(stage_masks["energy"] & ~stage_masks["penetration"])),
        }
        hit_n = int(stage_masks["hit"].sum())
        pen_n = int(stage_masks["penetration"].sum())
        energy_n = int(stage_masks["energy"].sum())
        block["conditional_survival"] = {
            "penetration_given_hit": pen_n / hit_n if hit_n else None,
            "energy_given_penetration": energy_n / pen_n if pen_n else None,
        }
        for profile in ("small_front", "medium_front", "large_front"):
            pm = profiles == profile
            block["by_profile"][profile] = {
                stage: int(np.count_nonzero(mask & pm)) for stage, mask in stage_masks.items()
            }
        upstream[category] = block

    m1 = a["system_y"][:, _index(output_names, "effect-m1")]
    m2 = a["system_y"][:, _index(output_names, "effect-m2")]
    k1 = a["system_y"][:, _index(output_names, "effect-k1")]
    k2 = a["system_y"][:, _index(output_names, "effect-k2")]
    c1 = a["system_y"][:, _index(output_names, "effect-c1")]
    c2 = a["system_y"][:, _index(output_names, "effect-c2")]

    manifest = json.loads((ARCHIVE / "manifest.json").read_text(encoding="utf-8"))
    report = {
        "schema_version": "1.0.0",
        "scope": "existing_full_archive_diagnostic_only",
        "dataset_id": manifest.get("dataset_id"),
        "engine_version": manifest.get("engine", {}).get("version"),
        "case_count": int(len(splits)),
        "split_counts": dict(Counter(splits.tolist())),
        "replicate_count_distribution": dict(Counter(map(int, a["replicate_counts"].tolist()))),
        "input_support": {
            "learnable_feature_names": li_names.tolist(),
            "profiles": dict(Counter(profiles.tolist())),
            "continuous": continuous,
            "continuous_pearson_correlation": corr.tolist(),
            "yz_10x10_occupied_cells": int(np.count_nonzero(yz_hist)),
            "yz_10x10_minimum_cell_count": int(yz_hist.min()),
            "yz_10x10_maximum_cell_count": int(yz_hist.max()),
            "orientation_unique_rows": int(np.unique(np.round(orientations, 12), axis=0).shape[0]),
            "orientation_values": np.unique(np.round(orientations, 12), axis=0).tolist(),
            "velocity_direction_unique_rows": int(np.unique(np.round(directions, 12), axis=0).shape[0]),
            "velocity_direction_values": np.unique(np.round(directions, 12), axis=0).tolist(),
            "detonation_x_unique_count": int(np.unique(np.round(detonation[:, 0], 12)).size),
            "detonation_x_value_counts": {str(k): int(v) for k, v in zip(*np.unique(np.round(detonation[:, 0], 12), return_counts=True))},
            "first_armor_contact_true_count": int(np.count_nonzero(case_x[:, _index(case_names, "impact_first_armor_contact")] == 1.0)),
            "split_distribution_tests": _split_profile_tests(li, splits, profiles),
        },
        "labels": labels,
        "ordinal_diagnostics": {
            "m2_greater_than_m1_count_if_m2_is_level2": int(np.count_nonzero(m2 > m1 + 1e-12)),
            "m1_greater_than_m2_count": int(np.count_nonzero(m1 > m2 + 1e-12)),
            "m1_equals_m2_count": int(np.count_nonzero(np.isclose(m1, m2, atol=1e-12, rtol=0))),
            "k2_le_k1_all": bool(np.all(k2 <= k1 + 1e-12)),
            "c2_le_c1_all": bool(np.all(c2 <= c1 + 1e-12)),
            "warning": "Constant-zero pairs satisfy ordering vacuously and provide no ordinal learnability evidence.",
        },
        "kmfc_candidate_upstream_support": upstream,
        "interpretation": {
            "structural_zero_operational_definition": "No positive case at the earliest archived physical support stages (fragment hit, penetration, deposited energy) for the components used by a task/category under this fixed archive and design measure.",
            "rare_positive_operational_definition": "Positive upstream support exists but occupies a small fraction of the current design measure, so prevalence and metric estimates have substantial finite-sample uncertainty.",
            "threshold_induced_zero_operational_definition": "Positive continuous upstream dose exists but a downstream state threshold or damage-tree gate maps it to zero; this requires paired dose/gate evidence and must not be inferred from a zero label alone.",
            "findings": [
                "K and C old system outputs are constant zero and unlearnable; C also has zero candidate-category hit/penetration/energy support in all 5000 archived cases.",
                "F has only 28 candidate-category upstream-positive cases, all through the narrow observed component support; this is rare positive support rather than a universal impossibility claim.",
                "M has broad upstream support, while current M1/M2 outputs are not an admissible nested ordinal pair if M2 is intended as level 2.",
                "Every case uses one orientation and one normalized velocity direction. The archive cannot identify directional/attitude effects or justify generalization beyond this fixed approach geometry.",
                "R=2 restricts direct Bernoulli probability resolution to 0, 0.5 and 1 and makes nondetection indistinguishable from a truly zero conditional probability without a convergence experiment.",
            ],
        },
        "deep_learning_implications": {
            "constant_zero_tasks": ["effect-k1", "effect-k2", "effect-c1", "effect-c2"],
            "constant_zero_task_label_entropy_bits": 0.0,
            "constant_zero_task_input_label_mutual_information_bits": 0.0,
            "f1_train_positive_count": labels["effect-f1"]["by_split"]["train"]["positive_count"],
            "f1_train_positive_fraction": labels["effect-f1"]["by_split"]["train"]["positive_fraction"],
            "f1_probability_of_zero_positive_cases_per_iid_batch": {
                str(batch): float((1 - labels["effect-f1"]["by_split"]["train"]["positive_fraction"]) ** batch)
                for batch in (32, 64, 128, 256)
            },
            "f1_validation_recall_step_per_positive_case": 1.0 / labels["effect-f1"]["by_split"]["validation"]["positive_count"],
            "f1_test_recall_step_per_positive_case": 1.0 / labels["effect-f1"]["by_split"]["test"]["positive_count"],
            "accuracy_of_constant_zero_classifier": {
                split: 1.0 - labels["effect-f1"]["by_split"][split]["positive_fraction"]
                for split in ("train", "validation", "test")
            },
            "recommendation": "Do not train or report K/C heads from this archive. Treat F as a separate rare-event/hurdle task only after acquiring substantially more independent positive cases in Train. Preserve a reference stream and keep outcome-enriched sampling weights separate from minibatch/task weights.",
        },
        "scientific_boundary": "All Train, Validation, and previously observed Test rows are included. Results are descriptive diagnostics of one unvalidated synthetic archive and fixed design distribution. They do not establish real-world prevalence, physical validity, or causal inevitability of sparsity.",
        "sources": {
            "archive_manifest": str((ARCHIVE / "manifest.json").relative_to(ROOT)).replace("\\", "/"),
            "archive_manifest_sha256": _sha256(ARCHIVE / "manifest.json"),
            "shard_tensor_paths": [str((p / "tensors.npz").relative_to(ROOT)).replace("\\", "/") for p in shards],
        },
    }
    return report


def markdown(report: dict) -> str:
    inp = report["input_support"]
    f1_support = report["labels"]["effect-f1"]["positive_input_support"]
    m1_support = report["labels"]["effect-m1"]["positive_input_support"]
    lines = [
        "# 标签稀疏与输入偏置独立统计审计",
        "",
        f"- 数据集：`{report['dataset_id']}`",
        f"- 样本数：{report['case_count']}；划分：`{report['split_counts']}`",
        f"- 每工况重复数：`{report['replicate_count_distribution']}`",
        "- 使用范围：诊断性全归档复审；Test 已经被观察，不构成确认性评估。",
        "",
        "## 输入支持",
        "",
        f"- 姿态唯一值：**{inp['orientation_unique_rows']}**；归一化速度方向唯一值：**{inp['velocity_direction_unique_rows']}**。",
        f"- 10×10 的 y-z 网格占用：{inp['yz_10x10_occupied_cells']}/100，单格 {inp['yz_10x10_minimum_cell_count']}–{inp['yz_10x10_maximum_cell_count']} 条。",
        f"- 首次装甲接触：{inp['first_armor_contact_true_count']}/{report['case_count']}。",
        "- 这说明 y、z、速度和三种弹型获得覆盖，但方向与姿态没有实验自由度；方向效应在该数据中不可辨识。",
        "",
        "## 七个系统标签",
        "",
        "| 标签 | 正值数 | 零值率 | 原始唯一值 | R=2量化熵(bit) | HHI | Train 正值 | Val 正值 | Test 正值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, block in report["labels"].items():
        a = block["all"]
        lines.append(
            f"| `{name}` | {a['positive_count']} | {a['zero_fraction']:.4f} | {a['unique_value_count']} | "
            f"{a['r2_quantized_entropy_bits']:.4f} | {a['hhi']:.4f} | {block['by_split']['train']['positive_count']} | "
            f"{block['by_split']['validation']['positive_count']} | {block['by_split']['test']['positive_count']} |"
        )
    lines.extend([
        "",
        "## KMFC 候选部件的上游支持",
        "",
        "| 类别 | 部件数 | 命中 case | 侵彻 case | 沉积能>0 case | P(侵彻|命中) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for cat, block in report["kmfc_candidate_upstream_support"].items():
        stages = block["stages"]
        cond = block["conditional_survival"]["penetration_given_hit"]
        cond_s = "—" if cond is None else f"{cond:.4f}"
        lines.append(
            f"| {cat} | {block['component_count']} | {stages['hit']['case_count']} | {stages['penetration']['case_count']} | "
            f"{stages['energy']['case_count']} | {cond_s} |"
        )
    od = report["ordinal_diagnostics"]
    lines.extend([
        "",
        "## 关键判定",
        "",
        f"- 若将 `M2` 解释为二级毁伤、`M1` 解释为一级毁伤，则 `M2 > M1` 有 **{od['m2_greater_than_m1_count_if_m2_is_level2']}** 条；当前二者不能直接作为嵌套有序标签。",
        "- K/C 的常量零对虽形式上满足 `level2 <= level1`，但属于真空满足，不提供任何可学习性证据。",
        "- C 在命中阶段即为 0/5000，调整下游阈值、损失权重或平滑门无法从现有归档中创造 C 信号。",
        "- F 已有非零上游支持但仅 28/5000，应按稀有事件处理；其来源是否是项目内生稀有，仍需改变方向、接触面与部件几何的反事实实验。",
        f"- 28 个 F1 正值全部落在同一碰撞 x 平面，z 仅覆盖 {f1_support['trajectory_origin_z_min_max'][0]:.4f}–{f1_support['trajectory_origin_z_min_max'][1]:.4f} m，且只占 10×10 y-z 网格的 {f1_support['yz_10x10_occupied_cells']} 格；这是强烈的局部几何支持证据。",
        f"- M1 正值同样全部落在该碰撞 x 平面，z 上限为 {m1_support['trajectory_origin_z_min_max'][1]:.4f} m；当前标签分布受接触面/内部几何强约束。",
        "- `R=2` 无法把 Monte Carlo 未观测到事件与真实条件概率为零可靠地区分；当前零质量是下界诊断，不是物理必然性证明。",
        f"- F1 的 Train 仅 {report['deep_learning_implications']['f1_train_positive_count']} 个正值；batch=128 的普通 iid 抽样中无正值概率为 {report['deep_learning_implications']['f1_probability_of_zero_positive_cases_per_iid_batch']['128']:.3f}。",
        f"- F1 的 Validation 只有 2 个正值，单个样本会使召回率跳变 {report['deep_learning_implications']['f1_validation_recall_step_per_positive_case']:.0%}；Test 也只有 15 个，单个样本对应 {report['deep_learning_implications']['f1_test_recall_step_per_positive_case']:.2%}。",
        "- F1 正值率的划分与弹型异质性在探索性检验中出现小 p 值，但共进行了多项标签/分组比较，且 Validation 期望格数偏小；不得把它当作确认性弹型效应。",
        "",
        "## 证据边界",
        "",
        report["scientific_boundary"],
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = build()
    out_json = OUT_DIR / "ml_stats_audit.json"
    out_md = OUT_DIR / "ml_stats_audit.md"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    out_md.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"status": "ok", "json": str(out_json), "markdown": str(out_md)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
