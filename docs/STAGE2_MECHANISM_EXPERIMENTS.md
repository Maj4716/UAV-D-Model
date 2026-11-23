# Stage 2 机理实验工具

状态：`implemented_not_large_scale_executed`  
合同：`stage2-mechanism-experiments@1.0.0`

入口：`tools/run_stage2_mechanism_experiments.py`

该工具实现 `idea2.md` 的 E3、E5 和 C2。输出绑定目标文件、物理注册表、引擎
代码、基础工况和随机种子哈希。结果用于软件教师内部的机理归因，不构成外部
物理确认。

## E3：2^4 物理因子 + 离线树因子

默认规模为 16 个物理角点 × 300 个 Reference 工况 × R=4：

| 因子 | 低水平 | 高水平 |
|---|---|---|
| A 装甲 | 10 mm | 1 mm |
| T 部件阈值 | E50 ×1 | E50 ×0.1 |
| F 方向 | legacy 空心 Taylor 环 | 同名义外角实心锥 |
| W 战斗部 | Gurney ×1 | Gurney ×1.3 |
| Tr 树（离线） | 当前树 | 部件状态恒等输出 |

实心锥的外角按每个弹型和对应 W 水平重新计算，避免把 Gurney 变化混入“外角
不匹配”。所有物理角点使用同一组有序基础工况和相同 `aleatory_seed`。Tr 从
`component_state_trace` 离线重算，不增加教师调用。报告包含部件零率、工况任一
支持率、正能量 Hill 尾指数、当前/恒等树的零率、饱和率、中间区、熵、域完整
率，以及五因子的主效应和两两交互；置信区间以基础工况为成对 bootstrap 单元。

```powershell
python tools/run_stage2_mechanism_experiments.py e3 `
  --output artifacts/experiments/E3_YYYYMMDD
```

只检查设计绑定，不运行教师：

```powershell
python tools/run_stage2_mechanism_experiments.py e3 `
  --dry-run --output artifacts/experiments/E3_plan
```

## E5：σ × τ 与方向噪声形状

文档默认网格为 σ={0.5°,2°,5°} × τ={0.04,0.08,0.16}，共 9 个角点 ×
300 工况 × R=4。τ 同时写入装甲和部件破片响应的对数-logistic 弹道极限过渡
宽度。

方向噪声支持三种同方差分布：

- `normal`：标准差 σ；
- `uniform`：区间半宽 √3σ；
- `laplace`：尺度 σ/√2。

默认只运行文档的 9 个正态角点。显式加入三种分布时得到 27 个角点：

```powershell
python tools/run_stage2_mechanism_experiments.py e5 `
  --distributions normal uniform laplace `
  --output artifacts/experiments/E5_YYYYMMDD
```

## C2：解析金标与半独立对照

C2 对三个弹型分别报告：

- 按 `casing_mass_basis=excludes_fragments` 计算的独立 Gurney 金标；
- 引擎实际生成速度及最大绝对误差；
- legacy 环角度的精确 clipped-normal 混合均值/标准差；
- 固定种子经验角度分布及金标误差；
- 每个破片响应模型在 V50 处的解析概率 0.5、经验穿透率、τ 和 10%–90%
  速度比。

```powershell
python tools/run_stage2_mechanism_experiments.py c2 `
  --output artifacts/experiments/C2_YYYYMMDD
```

方向重复数和 V50 每速度试验数可分别用 `--direction-repeats`、
`--v50-trials-per-speed` 调整。

## 输出

E3/E5：

```text
<output>/
├── config.json
├── run.log
├── report.json
└── variants/<variant_id>/
    ├── summary.json
    └── engine_results.minimal.jsonl.gz
```

压缩结果保留逐重复 `component_state_trace`，可继续用
`tools/recompute_tree_outputs.py` 对候选树离线重算。

## Taylor–Shapiro 边界

E3 的 F 因子严格是 legacy `taylor_ring` 空心环与同外角实心锥的比较。
`taylor_shapiro_ring` 使用不同的局部壳体几何和角度定义，工具会拒绝把它静默
作为 E3/E5 水平。Taylor–Shapiro 应使用独立注册表和独立实验臂评估。
