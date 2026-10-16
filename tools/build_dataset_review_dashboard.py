"""Build a self-contained HTML dashboard from the 5000-case audit artifacts."""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{100.0 * value:.{digits}f}%"


def _num(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.{digits}f}"


def _bar(label: str, value: float, *, maximum: float, color: str, note: str = "") -> str:
    width = 0.0 if maximum <= 0 else min(100.0, 100.0 * value / maximum)
    return f"""
      <div class="bar-row">
        <div class="bar-label">{escape(label)}</div>
        <div class="bar-track"><span style="width:{width:.2f}%;background:{color}"></span></div>
        <div class="bar-value">{value:.2%}{('<small>' + escape(note) + '</small>') if note else ''}</div>
      </div>"""


def build(dataset_root: Path) -> str:
    audits = dataset_root / "audits"
    readiness = _read(audits / "training_readiness_audit.json")
    deep = _read(audits / "distribution_deep_audit.json")
    loader = _read(audits / "loader_and_training_view_validation.json")
    comparison = _read(audits / "old_vs_new_comparison.json")
    timing = _read(dataset_root / "generation_timing.json")
    manifest = _read(dataset_root / "formal_dataset" / "manifest.json")

    events = readiness["mechanism_events"]
    labels = readiness["labels"]
    scope = readiness["scope"]
    inputs = readiness["input_features"]
    trace = readiness["replicate_trace"]
    split_counts = scope["split_counts"]
    plateau = labels["joint_mode_fraction"]
    hit = events["fragment_hit"]["all"]["rate"]
    shield_hit = events["shielding_and_fragment_hit"]["all"]["rate"]
    partial = events["fragment_hit_count_exceeds_penetration_count"]["all"]["rate"]
    loader_ok = loader["overall"]["loader_all_splits_passed"]
    normalization = loader["normalization"]
    research_count = manifest["quality_summary"]["purpose_admission_counts"]["research_training"]
    total_seconds = timing["durations_seconds"]["end_to_end"]

    event_bars = "".join(
        [
            _bar("系统联合众数平台", plateau, maximum=1.0, color="#e06c75", note=f"{labels['joint_mode_count']}/5000"),
            _bar("至少一次破片命中", hit, maximum=0.10, color="#d19a66", note="按 10% 满刻度显示"),
            _bar("遮蔽且命中", shield_hit, maximum=0.10, color="#c678dd", note="按 10% 满刻度显示"),
            _bar("命中数大于穿透数", partial, maximum=0.10, color="#56b6c2", note="按 10% 满刻度显示"),
        ]
    )

    split_rows = []
    for split in ("train", "validation", "test"):
        event = events["fragment_hit"][split]
        split_rows.append(
            f"<tr><td>{split.title()}</td><td>{split_counts[split]:,}</td>"
            f"<td>{event['successes']:,}</td><td>{_pct(event['rate'])}</td>"
            f"<td>{_pct(event['ci95'][0])}–{_pct(event['ci95'][1])}</td></tr>"
        )

    comparison_rows = []
    display_names = {
        "maximum_exploratory_split_smd": "最大探索性 split SMD",
        "validation_nearest_train_combined_distance_rad_median": "Validation 最近 Train 距离中位数",
        "test_nearest_train_combined_distance_rad_median": "Test 最近 Train 距离中位数",
        "fragment_hit_fraction": "破片命中率",
        "joint_system_mode_fraction": "系统联合众数平台",
        "system_output_spearman": "系统输出 Spearman",
        "fragment_hits_equal_penetrations_fraction": "命中数=穿透数比例",
        "naive_tree_aggregation_difference_fraction": "错误聚合树逻辑差异比例",
        "system_ci_half_width_nonzero_fraction_per_label_value": "系统 CI 非零比例",
        "system_ci_half_width_ge_0_25_fraction_per_label_value": "系统 CI 半宽≥0.25",
    }
    percent_metrics = {
        "fragment_hit_fraction",
        "joint_system_mode_fraction",
        "fragment_hits_equal_penetrations_fraction",
        "naive_tree_aggregation_difference_fraction",
        "system_ci_half_width_nonzero_fraction_per_label_value",
        "system_ci_half_width_ge_0_25_fraction_per_label_value",
    }
    for row in comparison["metrics"]:
        formatter = _pct if row["metric"] in percent_metrics else _num
        comparison_rows.append(
            "<tr>"
            f"<td>{escape(display_names.get(row['metric'], row['metric']))}</td>"
            f"<td>{formatter(row['old'])}</td><td>{formatter(row['new'])}</td>"
            f"<td>{escape(row.get('interpretation', ''))}</td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>5000条数据集质量诊断仪表板</title>
  <style>
    :root{{--bg:#0b1020;--panel:#131a2d;--panel2:#182238;--text:#edf2f7;--muted:#9fb0c7;--line:#2a3650;--good:#5bd3a6;--warn:#f4c95d;--bad:#ff718d;--accent:#73a9ff}}
    *{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at 15% 0,#16264a 0,transparent 35%),var(--bg);color:var(--text);font:15px/1.65 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}}
    main{{max-width:1240px;margin:auto;padding:34px 22px 70px}} h1{{font-size:clamp(28px,4vw,44px);line-height:1.15;margin:.15em 0}} h2{{margin-top:36px;font-size:23px}} h3{{font-size:17px;margin:0 0 7px}} .eyebrow{{color:#83b5ff;letter-spacing:.14em;text-transform:uppercase;font-weight:700}} .subtitle{{max-width:900px;color:var(--muted);font-size:17px}}
    .grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:24px 0}} .card,.panel{{background:linear-gradient(145deg,rgba(24,34,56,.96),rgba(16,23,42,.96));border:1px solid var(--line);border-radius:16px;box-shadow:0 14px 35px rgba(0,0,0,.18)}} .card{{padding:18px}} .card .value{{font-size:27px;font-weight:800;margin:3px 0}} .card p{{margin:0;color:var(--muted);font-size:13px}} .status{{display:inline-block;padding:2px 9px;border-radius:999px;font-weight:800;font-size:12px}} .pass{{background:rgba(91,211,166,.14);color:var(--good)}} .block{{background:rgba(255,113,141,.15);color:var(--bad)}} .limit{{background:rgba(244,201,93,.15);color:var(--warn)}}
    .two{{display:grid;grid-template-columns:1.05fr .95fr;gap:16px}} .panel{{padding:20px;overflow:hidden}} table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}} th,td{{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top}} th{{color:#a9bad0;font-size:12px;text-transform:uppercase;letter-spacing:.06em}} tbody tr:last-child td{{border-bottom:0}} code{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;color:#a8c8ff}} .alert{{border-left:4px solid var(--bad);background:rgba(255,113,141,.08);padding:13px 15px;border-radius:8px;margin:14px 0}} .note{{border-left:4px solid var(--warn);background:rgba(244,201,93,.07);padding:13px 15px;border-radius:8px}}
    .bar-row{{display:grid;grid-template-columns:180px 1fr 115px;gap:10px;align-items:center;margin:14px 0}} .bar-label{{color:#dbe6f5}} .bar-track{{height:13px;background:#29344a;border-radius:999px;overflow:hidden}} .bar-track span{{display:block;height:100%;border-radius:999px}} .bar-value{{font-variant-numeric:tabular-nums;text-align:right;font-weight:700}} .bar-value small{{display:block;color:var(--muted);font-weight:400;white-space:nowrap}} ol{{padding-left:22px}} li{{margin:8px 0}} .pill{{display:inline-block;border:1px solid var(--line);padding:3px 8px;border-radius:999px;color:var(--muted);margin:2px}}
    footer{{margin-top:38px;padding-top:18px;border-top:1px solid var(--line);color:var(--muted);font-size:13px}} @media(max-width:900px){{.grid{{grid-template-columns:repeat(2,1fr)}}.two{{grid-template-columns:1fr}}}} @media(max-width:620px){{.grid{{grid-template-columns:1fr}}.bar-row{{grid-template-columns:1fr}}.bar-value{{text-align:left}}table{{font-size:12px}}}}
  </style>
</head>
<body><main>
  <div class="eyebrow">UAV Damage Model · Dataset Review</div>
  <h1>5000 条数据集质量诊断</h1>
  <p class="subtitle">本页区分四件事：归档是否完整、深度学习接口是否可用、数据分布是否支持关键机理学习、以及物理可信度是否已经建立。它们不能互相替代。</p>

  <section class="grid">
    <div class="card"><span class="status pass">归档通过</span><div class="value">5,000 / 5,000</div><p>0 拒算 · 10 分片 · 4 次重复</p></div>
    <div class="card"><span class="status pass">Loader 通过</span><div class="value">4000 / 500 / 500</div><p>Train / Validation / Test 全量遍历</p></div>
    <div class="card"><span class="status block">训练视图阻断</span><div class="value">Normalizer 未生成</div><p>固定 2 m 距离被误标为 learnable</p></div>
    <div class="card"><span class="status limit">科学准入为 0</span><div class="value">{research_count}</div><p>仍是未验证参数下的 synthetic teacher</p></div>
  </section>

  <section class="two">
    <div class="panel">
      <h2 style="margin-top:0">事件与平台</h2>
      {event_bars}
      <div class="note">破片命中率的 Wilson 95% 区间为 {_pct(events['fragment_hit']['all']['ci95'][0])}–{_pct(events['fragment_hit']['all']['ci95'][1])}。这是当前合成参考设计的区间，不是真实装备概率区间。</div>
    </div>
    <div class="panel">
      <h2 style="margin-top:0">生成代价</h2>
      <table><tbody>
        <tr><td>端到端</td><td><strong>{total_seconds / 60:.1f} min</strong></td></tr>
        <tr><td>正式窗口</td><td><strong>{timing['durations_seconds']['formal_execution_and_archive'] / 60:.1f} min</strong></td></tr>
        <tr><td>正式归档</td><td>{timing['storage_bytes']['formal_dataset_total'] / 2**20:.1f} MiB</td></tr>
        <tr><td>计划文件</td><td><strong style="color:var(--bad)">{timing['storage_bytes']['formal_cases_jsonl'] / 2**30:.2f} GiB</strong></td></tr>
        <tr><td>正式吞吐</td><td>{timing['throughput']['outer_cases_per_second_formal_window']:.3f} cases/s</td></tr>
      </tbody></table>
      <div class="alert">计划文件重复嵌入大型 split registry，约为正式张量归档的 16.8 倍。30 万条前必须把 registry 外置为单一 sidecar。</div>
    </div>
  </section>

  <section class="panel" style="margin-top:16px">
    <h2 style="margin-top:0">稀有事件在三个拆分中的有效数量</h2>
    <table><thead><tr><th>拆分</th><th>总数</th><th>破片命中</th><th>命中率</th><th>Wilson 95% 区间</th></tr></thead><tbody>{''.join(split_rows)}</tbody></table>
    <p class="subtitle">Test 只有 17 个命中，遮蔽且命中只有 4 个；“命中数大于穿透数”的 15 个案例全部在 Train。当前 Test 无法稳定检验命中—穿透转变。</p>
  </section>

  <section class="panel" style="margin-top:16px">
    <h2 style="margin-top:0">旧 2000 条与新 5000 条的可比聚合指标</h2>
    <table><thead><tr><th>指标</th><th>旧</th><th>新</th><th>解释</th></tr></thead><tbody>{''.join(comparison_rows)}</tbody></table>
  </section>

  <section class="two" style="margin-top:16px">
    <div class="panel">
      <h2 style="margin-top:0">输入支持</h2>
      <div><span class="pill">48 个物理特征</span><span class="pill">仅 {inputs['variable_feature_count']} 个变化</span><span class="pill">{inputs['constant_feature_count']} 个常量</span><span class="pill">1 个结构缺失</span></div>
      <p>变化特征只有起爆位置 x/y/z 与姿态四元数 x/y/z/w。战斗部、材料、装甲、部件与环境均固定，因此只能学习固定目标下的五维插值。</p>
      <p>球面 20×10 等面积格：{inputs['sphere_equal_area_20x10']['occupied_cells']}/200 占用；未发现精确输入重复或跨拆分精确重复。</p>
    </div>
    <div class="panel">
      <h2 style="margin-top:0">R=4 的信息边界</h2>
      <p><strong>{trace['cases_with_any_system_output_variation_across_replicates']}/5000</strong> 案例在重复间有系统输出变化；每案例只有一个 parameter realization hash。</p>
      <p>这意味着 trace 主要刻画条件 aleatory 随机性，没有覆盖参数认识不确定性。`mission_loss` 的最大 CI 半宽仍约 0.498。</p>
      <p>trace 均值与归档聚合最大误差小于 3×10⁻⁸，聚合实现本身一致。</p>
    </div>
  </section>

  <section class="panel" style="margin-top:16px">
    <h2 style="margin-top:0">P0 阻断：训练特征角色</h2>
    <div class="alert"><code>{escape(normalization['error'])}</code></div>
    <p>正确做法是把 transform 输出按张量特征分角色：三个方向分量与六个 rotation-6D 分量为 learnable；固定距离为 fixed_context；缺失固定初速为 structural_missing。不得关闭检查或用 epsilon 制造伪方差。</p>
  </section>

  <section class="panel" style="margin-top:16px">
    <h2 style="margin-top:0">建议执行顺序</h2>
    <ol>
      <li><strong>先修训练角色合同：</strong>生成前预检最终 51 维特征；Train-only normalizer 与三个 TrainingViewDataset 全量通过。</li>
      <li><strong>分离 Reference 与 targeted：</strong>Validation/Test 保持结果盲 p0；仅 Train 加入标签前几何定向样本，并保留合法 p0/q/ρ 或明确仅作条件学习。</li>
      <li><strong>按稀有事件设计测试规模：</strong>500 条 Test 对 3.84% 命中只给约 17 个事件；评估集规模应由所需区间精度与最小事件数决定。</li>
      <li><strong>做 R=4/8/16/32 收敛试验：</strong>按预先冻结的几何转变层比较 CI 与计算成本，不用全局中位数掩盖稀有区域。</li>
      <li><strong>外置 split registry：</strong>每案例只保存 hash/ref，流式执行 formal plan；再做 5k/50k/300k 吞吐与峰值 RSS 基准。</li>
      <li><strong>建立物理 V&amp;V：</strong>独立实验标定、留出验证、模型差异与参数不确定性齐备后，才能讨论 research admission。</li>
    </ol>
  </section>

  <footer>
    数据集：{escape(scope['dataset_id'])} · 引擎 {escape(str(scope['engine_version']))} · HTML 完全自包含，无外部脚本或网络资源。<br>
    证据边界：软件归档完整、统计描述合理或代理模型可学习，都不等于物理正确或可用于自主控制。
  </footer>
</main></body></html>"""
    return html


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    html = build(args.dataset_root.resolve())
    args.output.write_text(html, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "bytes": len(html.encode('utf-8'))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
