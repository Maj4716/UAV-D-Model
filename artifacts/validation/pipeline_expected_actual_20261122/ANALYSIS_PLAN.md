# 数据生成链逐环节预期—实际量化计划

- 分析 ID：`pipeline-expected-actual-20261122-v2-stagewise`
- 冻结日期：2026-11-22（Asia/Shanghai）
- 目标：对当前数据生成链逐环节建立分母、预期对象、实际统计量、差距指标和可解释边界。
- 数据边界：允许读取全部 pre-outcome 输入和 train/validation 结果；sealed-test 教师结果在 `json.loads` 前跳过，解码数、打开数与指标数均为 0。

## 1. 预期对象分类

每个环节必须标明预期属于哪一类：

1. **声明设计分布**：SamplingSpec 中的 profile、region、速度、潜变量和角色配额。评价设计忠实性。
2. **确定性合同或守恒关系**：哈希、S-CONTACT 几何、破片数、速度公式、状态单调性、标签聚合与毁伤树重算。评价违例数和数值误差。
3. **软件模型内部期望**：FastSamplingProxy、侵彻概率、E50 响应和软件教师的 replicate 均值。评价两种软件实现之间的差距。
4. **外部物理/作战期望**：独立试验、实测或外部标定模型给出的真实参照。当前项目未提供这一分母，因此真实分布距离不作数值伪估。

## 2. 分析总体

### 2.1 完整输入与归档

- 数据集：`recovery_role_r8_r32_train2000_eval300_20261122`。
- 候选 4,301，正式输入 2,600；train/validation/sealed test = 2,000/300/300。
- 可读取 SamplingSpec、候选账本、正式工况、split registry、checkpoint、manifest 和输入 tensor。

### 2.2 完整开发输出

- 2,000 个 train R8 + 300 个 validation R32，共 25,600 个教师重复。
- 67 个部件、7 个毁伤树输出。
- 读取 train/validation 教师结果、部件状态 trace 和 development labels。

### 2.3 当前版本微观机制审计

- 仅从 train 中抽取 15 个 region×profile 层，每层固定 2 例，共 30 例。
- 保持原 R8、aleatory seed、物理配置与输入不变，仅将 `capture_fragment_audit` 从 false 切换为 true。
- 先证明 summary、部件指标和树输出与归档结果数值等价，再分析逐破片事件。
- 该样本是等层诊断样本，聚合率不解释为 2,300 例开发总体发生率。

## 3. 逐环节指标

| 环节 | 主要指标 |
|---|---|
| 候选设计 | TV、边际 KS、centered discrepancy、Spearman |
| S-CONTACT | 准入率、拒绝原因、按 region/profile 选择率、条件后 TV |
| 首次接触 | 速度分布、距离、时间、入射角、资格合同违例 |
| 破片源 | 数量/质量合同、Gurney 速度、Taylor–Shapiro 方向残差 |
| 装甲和部件漏斗 | 每工况每重复期望/实际总量、actual−expected、case-resampling 区间、ratio-of-sums |
| 适用域 | proxy partial 和 teacher affected 工况率；双方 domain-complete 子集 |
| 事件侵彻 | 侵彻概率 vs Bernoulli 结果的 Brier/ECE，工况聚类重采样 |
| 能量/毁伤 | 吸收能量、damage energy、V50/E50 margin，状态单调性 |
| 毁伤树 | 逐重复离线重算误差，proxy–teacher 绝对差 |
| 标签和 split | 观测率、识别区间、零质量、固定 67 部件分母的 train–validation 差距 |
| 归档与封存 | 配额、哈希、泄漏、ESS/CV、test 零解码/零指标 |

## 4. 统计规则

- 候选设计是随机化 QMC；KS、discrepancy 和秩相关是设计诊断，不报 iid p 值。
- 主机制表使用 teacher 与 proxy 双方 domain-complete 工况。all-case 表表示当前部署输出；proxy partial 工况的期望总量是右删失下界。
- 工况是 bootstrap 单位；部件、破片和交点随工况整体重采样。区间表示固定 QMC 设计下的 case-resampling 敏感性，不表示重复 scramble 或现实物理总体不确定性。
- train/validation 输出使用固定 67 部件分母的识别区间；complete-67 作敏感性。按已观测部件求均值的可变分母结果仅用于删失诊断。
- Brier/ECE 比较 proxy 概率与教师 replicate 均值 soft label，包含 R8/R32 Monte Carlo 噪声；正能量 W1 仅在两者同时为正的部件对上定义。
- 拒绝、真实零和域删失分开统计；mask=false 不转换为零。
- 真实物理/作战分布未提供时，该距离记为“不可估计：缺少外部参照”，不记为 0。

## 5. 预先存在的门槛

- 采样质量：positivity ≥ 0.1、ESS fraction ≥ 0.5、weight CV ≤ 2。
- 代理结构零反证：同一 R 协议下 false-unreached Wilson 95% 上界 ≤ 3%。
- 概率任务支持：train 正/负各 ≥20，validation 正/负各 ≥5。
- train epsilon-complete 率 ≥ 90%。
- 本分析不为不利结果新增事后门槛，不读取 test 选择指标或结论。

## 6. 输出

- `REPORT.md`：逐环节定量评估与可行性结论。
- `metrics.json`：主报告的机器可读摘要。
- `work/stage_contracts/`：14 层合同和可观测性映射。
- `work/contact_inputs/`：首次接触与输入分布。
- `work/mechanism_funnel/`：完整聚合漏斗和 30 例微观诊断。
- `work/output_gap/`：标签、split 和毁伤树分布。
- `evidence_manifest.json`：文件大小和 SHA-256 证据清单。
