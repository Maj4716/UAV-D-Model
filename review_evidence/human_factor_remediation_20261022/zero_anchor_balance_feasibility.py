"""Bound how balanced a zero-anchored 0/1/2 teacher can become.

This audit never fits a threshold.  It uses only upstream positive-support
counts.  For any positive-threshold teacher that maps zero evidence to class
zero, the class-zero probability is at least the no-support fraction.  Hence a
three-way uniform class distribution is impossible whenever that fraction is
larger than one third.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PILOT_PATH = HERE / "controlled_factorial_diagnostic.json"
TREE_AUDIT_PATH = HERE / "damage_tree_validity_and_software_prototype_audit.json"
OUTPUT_PATH = HERE / "zero_anchor_balance_feasibility.json"
REPORT_PATH = HERE / "zero_anchor_balance_feasibility.md"
CATEGORIES = ("K", "M", "F", "C")


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def bound(*, positive: int, total: int) -> dict[str, Any]:
    if total <= 0 or positive < 0 or positive > total:
        raise ValueError("support counts must satisfy 0 <= positive <= total")
    support = positive / total
    zero = 1.0 - support
    excess = max(0.0, zero - 1.0 / 3.0)
    return {
        "case_count": total,
        "positive_support_count": positive,
        "positive_support_fraction": support,
        "class0_fraction_lower_bound": zero,
        "uniform_three_class_support_necessary_condition": zero <= 1.0 / 3.0,
        "minimum_total_variation_distance_from_uniform_lower_bound": excess,
        "interpretation": (
            "necessary condition only; passing does not prove that two stable, "
            "scientifically acceptable thresholds exist"
        ),
    }


def historical(tree_audit: dict[str, Any]) -> dict[str, Any]:
    inventory = tree_audit["candidate_kmfc_branch_inventory"]
    result: dict[str, Any] = {}
    for category in CATEGORIES:
        positive = int(
            inventory[category]["diagnostic_positive_support"][
                "any_positive_case_count"
            ]
        )
        result[category] = bound(positive=positive, total=5000)
    return result


def pilot_groups(pilot: dict[str, Any]) -> dict[str, Any]:
    by_level = pilot["aggregates"]["by_axis_and_r"]
    result: dict[str, Any] = {}
    for level, by_axis in sorted(by_level.items(), key=lambda item: int(item[0])):
        result[level] = {}
        for axis, summary in sorted(by_axis.items()):
            total = int(summary["accepted_case_count"])
            support = summary["category_ever_positive_count"]
            result[level][axis] = {
                category: bound(positive=int(support[category]), total=total)
                for category in CATEGORIES
            }
    return result


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# 零锚两级毁伤教师的类别均衡可行性边界",
        "",
        "本审计不拟合阈值。若平滑证据满足 `evidence=0 -> severity=0`，且一级阈值严格为正，则所有上游无正支撑工况都必须属于等级 0。因此，若无支撑比例为 `z`，必有 `P(level=0)>=z`。三类精确均匀的必要条件是 `z<=1/3`；当 `z>1/3` 时，到均匀三分类分布的总变差距离至少为 `z-1/3`。",
        "",
        "该界只是一项必要条件。通过该界不证明证据分布存在稳定阈值，也不证明阈值具有物理含义。",
        "",
        "## 旧 5000 例诊断归档",
        "",
        "| 类别 | 正支撑 | 等级0占比下界 | 能否满足三等分必要条件 | 距均匀分布的最小 TV 下界 |",
        "|---|---:|---:|---|---:|",
    ]
    for category in CATEGORIES:
        item = payload["historical_5000_diagnostic"][category]
        lines.append(
            f"| {category} | {item['positive_support_count']}/{item['case_count']} "
            f"({item['positive_support_fraction']:.2%}) | "
            f"{item['class0_fraction_lower_bound']:.2%} | "
            f"{'是' if item['uniform_three_class_support_necessary_condition'] else '否'} | "
            f"{item['minimum_total_variation_distance_from_uniform_lower_bound']:.3f} |"
        )

    levels = payload["controlled_pilot_by_axis_and_r"]
    for level in sorted(levels, key=int):
        lines.extend(
            [
                "",
                f"## 受控试验 R={level}",
                "",
                "| 源轴条件 | 类别 | 正支撑率 | 等级0占比下界 | 能否满足三等分必要条件 | 最小 TV 下界 |",
                "|---|---|---:|---:|---|---:|",
            ]
        )
        for axis, categories in levels[level].items():
            for category in CATEGORIES:
                item = categories[category]
                lines.append(
                    f"| `{axis}` | {category} | "
                    f"{item['positive_support_fraction']:.2%} | "
                    f"{item['class0_fraction_lower_bound']:.2%} | "
                    f"{'是' if item['uniform_three_class_support_necessary_condition'] else '否'} | "
                    f"{item['minimum_total_variation_distance_from_uniform_lower_bound']:.3f} |"
                )

    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "1. 对未通过必要条件的类别，改变两个硬阈值无法获得接近均匀的 0/1/2 分布；若强行做到，只能依赖非零截距、把缺失当零后再重标、或使用与上游证据不一致的规则，都会制造伪标签。",
            "2. 源轴反事实若通过该界，只能说明软件模型空间存在可分支撑，不能证明该源轴物理正确。",
            "3. 真正的修复顺序仍是先恢复并批准方向、接触面和源项的上游支撑，再在独立 `tree_design_fit` 上选择阈值，并在 `tree_design_holdout` 上验证。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    pilot = load(PILOT_PATH)
    tree_audit = load(TREE_AUDIT_PATH)
    payload = {
        "schema_version": "1.0.0",
        "scope": "zero_anchored_three_class_balance_necessary_condition",
        "historical_5000_diagnostic": historical(tree_audit),
        "controlled_pilot_by_axis_and_r": pilot_groups(pilot),
        "claim_boundary": (
            "Support-count lower bounds only; no threshold fitting, operational "
            "prevalence estimation, or physical validation."
        ),
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    REPORT_PATH.write_text(markdown(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
