# 数据生成链：预期合同、实际字段与可量化差距

## 总体判断

当前归档可以量化完整的数据治理链和大部分聚合物理漏斗。可直接分析的环节包括候选设计、接触准入、首次装甲接触、破片总数、装甲/部件命中与侵彻总数、开孔、适用域删失、部件毁伤状态、毁伤树、标签、切分和权重。

正式运行设置为 `capture_fragment_audit=false`、`capture_component_state_trace=true`。因此，2,600 例归档保存了每工况的聚合破片漏斗和逐重复部件状态，但没有保存逐破片的方向、候选几何路径、阻断原因、入射角、事件级侵彻概率和能量残差。逐环节报告应将“聚合漏斗可量化”和“微观事件不可恢复”分开陈述。

## 四类“预期”的判定口径

本映射把每个环节的预期拆成四类：

1. **声明设计分布**：配置明确规定的 profile、区域、速度、Sobol 潜变量、配额或机制参数分布。它回答“样本是否按设计产生”。
2. **确定性合同/守恒关系**：给定输入后必须精确成立的身份、几何、公式、排序、计数、能量关系或聚合顺序。它回答“代码是否忠实执行合同”。
3. **模型内概率期望**：当前侵彻、毁伤与树模型定义的条件概率或 Monte Carlo 期望。它回答“随机实现和代理是否匹配当前软件教师”。
4. **外部物理期望**：独立试验、实测或经外部校准模型给出的真实物理/作战分布。关键机制层目前缺失这一类基准。

前三类可以建立设计忠实性、软件一致性和模型内校准，不能单独证明物理有效性。机器可读映射在每个 stage 的 `expected_by_class` 中固定保存这四类口径。

## 分层映射

| ID | 数据生成环节 | 预期对象 | 实际证据 | 可比较的主要指标 | 可观测性 |
|---|---|---|---|---|---|
| S00 | 冻结合同与身份 | 文件/配置/代码/输入哈希精确一致 | checkpoint、manifest、result provenance | 哈希错配、案例数错配 | 完整 |
| S01 | 候选参考设计 | scrambled Sobol；profile 1/3；region 1/5；8 维潜变量近似均匀 | candidate ledger | TV、KS、discrepancy、秩相关 | 完整 |
| S02 | S-CONTACT 准入 | `P0` 经首次正向装甲接触条件化得到 `P_contact` | refusal、selection、formal eligibility | 准入率、拒绝原因、选择诱导偏移 | 完整 |
| S03 | 首次装甲接触 | 正距离/时间、单位法向、命中点与重算一致、速度 80–160 m/s | formal cases、tensors、inputs | 约束违例、命中点/法向/速度/入射角分布 | 完整 |
| S04 | 破片源 | 固定数量/质量、Taylor-Shapiro 环、2° 高斯散布、Gurney 初速 | `summary.total_fragments` 与工况配置 | 破片数量精确一致性 | 仅聚合 |
| S05 | 射线路径与遮挡 | 距离有序路径，阻断后不得产生下游命中 | armor/component hit totals | 每破片装甲/部件命中率、分层漏斗 | 仅聚合 |
| S06 | 装甲侵彻与开孔 | 注册 ballistic-limit/obliquity 模型 | armor hit/penetration/OOD/hole/opening | 侵彻转化、OOD、开孔、复用、完整度 | 仅聚合 |
| S07 | 部件接触/侵彻/能量 | 部件响应与空间累计损伤能量模型 | component metrics、summary | 命中/侵彻、条件能量、速度、裕度、聚类 | 部件聚合 |
| S08 | 适用域删失 | OOD 不记为零；传播识别区间；ε=0.01 | coverage、trace、interval/mask | 受影响重复、区间宽度、ε-complete | 完整 |
| S09 | 部件毁伤状态 | `0≤failed≤degraded≤1`，逐重复后求均值 | state trace、component probabilities | 单调性、零质量、状态转化 | 完整 |
| S10 | 毁伤树 | 每重复求树，再对输出均值；树版本冻结 | state trace、tree outputs | 离线重算误差、依赖场景跨度 | 完整 |
| S11 | 标签提取 | hurdle target v2、条件标签、识别区间和 mask 合同 | component_labels、development audit | 观察率、区间、支持、准入、mask 一致性 | train/validation 完整 |
| S12 | 归档/切分/权重 | 2,600 例；2,000/300/300；无泄漏；封存 test | manifests、registry、weights | 数量、哈希、ESS、CV、泄漏、test 解码 | 完整 |
| S13 | 快速代理—教师 | 快速期望场与软件教师分布的同工况差距 | proxy reports | Brier、ECE、JS、W1、MMD²、false-unreached | 聚合输出完整 |

## 机制层的预期—实际定义

### 破片源

配置给出可精确核验的离散预期：

- `small_front`：60 枚、每枚 0.005 kg、6 环；
- `medium_front`：150 枚、每枚 0.008 kg、12 环；
- `large_front`：300 枚、每枚 0.010 kg、18 环。

每工况的 `summary.total_fragments.mean` 可以与上述值逐例比较。Taylor-Shapiro 方向、2° 散布和 Gurney 初速的实现合同存在于配置与代码中，当前完整归档没有逐破片实现值，不能从归档直接计算方向 KS、环级角度偏差或初速偏差。

Gurney 圆筒合同为

```text
v_G = G / sqrt(M_casing / M_charge + 0.5)
```

由冻结 profile 计算的速度分别为：`small_front=2028.153764 m/s`、`medium_front=1974.946523 m/s`、`large_front=1875.608356 m/s`。公式和生成器可以做代码级精确核验；正式归档没有逐破片初速，无法对 2,600 例做“期望速度—实现速度”逐事件比较。以上速度仍是未外部标定的模型输入。

### 几何、遮挡与侵彻

`simulate_fragments` 为每个破片构造 `target.ray_hits` 候选路径，然后按事件时间和路径顺序处理。未侵彻、残速耗尽和 OOD 会终止该破片的后续路径。归档保存了 `total_armor_hits`、`total_armor_penetrations`、`total_component_hits`、`total_component_penetrations`、开孔、复用和 OOD，因此可以形成聚合漏斗：

```text
fragments
  → processed armor hits
  → armor penetrations
  → processed component hits
  → component penetrations
  → deposited/damage energy
  → degraded/failed states
  → damage-tree outputs
```

候选几何交点、被遮挡的下游对象、阻断原因、逐事件入射角和逐事件预测侵彻概率只写入 fragment audit。当前归档不能把“未命中部件”进一步唯一分解为几何不可见、装甲未侵彻、前序部件阻断或残速耗尽。

适用域内的单层侵彻模型给出

```text
p_pen = logistic(log(v / v50) / transition_width_ratio)
```

事件级正确比较应把保存的 `penetration_probability` 与同一事件的 `perforated` 分支做 Brier、ECE/分箱校准或“实际频率减平均概率”。当前归档只保存命中和侵彻总量的重复统计，未保存事件级概率与分支对，因此只能报告装甲/部件侵彻转化率，不能验证这条概率响应曲线。`demo_armor_response` 和 `demo_component_response` 均为单工况软件调参结果，不是外部 V50 校准。

### 部件状态、毁伤树与标签

逐重复 `component_state_trace` 已保留，可精确重算当前毁伤树。该层的预期不是经验频率，而是冻结树的确定性映射；正确的比较指标是离线重算的最大绝对误差和身份哈希，而不是分布距离。

部件孤立空间簇的毁伤响应为：

```text
p_state = vulnerable_ratio × logistic(log_slope × log(E_cluster / E50))
```

其中 degraded 使用 `E50=350 J, log_slope=3.0`，failed 使用 `E50=1900 J, log_slope=3.5`。因此 `E_cluster=E50` 时实际概率是 `0.5 × vulnerable_ratio`，不是无条件的 0.5。正式 `component_state_trace` 保留逐重复状态概率，但不保留逐重复簇能量和位置；不能用“平均能量代入非线性响应”冒充逐重复重算。

毁伤树的可核验聚合顺序是：先对每个 replicate 的联合部件状态求树，再对系统输出做算术平均。逐重复状态已保留，因此该项可以精确离线重算。它证明当前树实现一致；树本身仍为 provisional，且缺少系统级实测失效分布。

标签层将 `p_hit`、条件 `log1p` 能量、条件 `log1p` 命中数、`p_degraded` 和 `p_failed` 与域识别区间共同保存。可比较观察率、区间宽度、正负支持和任务准入。sealed test 只保留 masked placeholders，结果保持零解码。

## 报告中应明确的四个边界

1. **接触条件分布没有解析密度。** `P_contact` 是 `P0` 经目标几何和拒绝规则得到的经验推送分布。可以比较候选与选中样本，若要计算绝对密度比误差，需要高分辨率几何基准或解析 Jacobian。
2. **微观破片漏斗未随 2,600 例归档。** 聚合命中/侵彻/能量漏斗可计算，逐破片归因需要另建启用 `capture_fragment_audit=true` 的冻结诊断样本。
3. **软件期望不等于外部物理期望。** 破片方向、弹药参数、响应参数和毁伤树仍标记为 unverified/provisional。当前可以计算“声明模型—软件实现”的差距，真实物理分布差距需要独立实测基准。
4. **部件状态可见不等于响应曲线可重算。** 逐重复状态概率已经保存，但逐重复部件能量、空间簇能量和位置没有进入 `component_state_trace`。E50 响应的代码合同可审计，正式归档不能完成逐重复能量—状态残差比较。

完整机器可读映射见 [`stage_map.json`](stage_map.json)。
