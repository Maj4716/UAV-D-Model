# idea2 全量实施与优化评估报告

日期：2026-11-20  
证据状态：主链实现与定向验证已形成；冻结 E0、E1 多目标、E2/E7、X1–X3、E4/E6 增强诊断和 §6.8 机理因子设计已完成；最终 R=32 v3 writer、严格开发包、开发审计、validation-only 学习评估与全仓软件质量检查均已完成。  
冻结引擎哈希：`sha256:33e6a78619b5e9028e7b3d4a7217ebb7e230f325f11da365e479665f519b59ad`

## 1. 结论

项目已经把 `idea2.md` 的核心方法落实为可执行代码、版本化配置、测试和证据链，覆盖引擎语义、Taylor–Shapiro 方向模型、删失区间、R 无关跨栏标签、逐重复部件状态、离线树重算、连续来向、暴露场、多保真、单调树族、目标编码器、dataset-v2 与封存测试纪律。E1 多目标复跑、最终 v3 writer、严格 dataset-v2、train/validation-only 开发审计、编码器消融、代理—教师分布和四臂代理学习评估均已完成。

Taylor–Shapiro 方法适合以独立 v2 物理候选进入当前试生成。它用有符号 Shapiro 偏转描述局部方向，并用 Taylor 角限制偏转幅值；实现绑定圆柱侧壁、前端轴上点起爆和明确的速度比定义域。该方法比旧 `impact_inward` 空心 Taylor 环具有更清楚的几何与公式来源。旧模型保持原身份，新模型使用独立 physics/case ID，因此历史数据仍可复现。当前 UAV 战斗部几何、2° 色散、装甲和部件阈值缺少外部标定，Taylor–Shapiro v2 的证据等级为“有文献依据的软件教师候选”。

最终试生成采用 Reference 主臂、SamplingSpec 1.7 连续来向、16 维首次装甲状态输入、Taylor–Shapiro v2、hurdle v2 标签和 R=32。暴露均衡臂已实现并完成工程运行，但 X1 教师反证率和 X2 预注册门未通过，因此没有并入最终试生成。该决策保持采样分布可解释，并避免把尚未通过校核的代理偏差写入正式训练数据。

严格开发包已实现 test 真标签零解码：test 结果行在 `json.loads` 前跳过，开发包只保留 test 输入、身份与全掩码占位；最终 `sealed_test.json` 记录 `test_result_json_lines_decoded=0`、`test_open_count=0`。开发审计显示 300 个 train+validation 样本中仅 2 个达到 ε=0.01 完整，52/67 部件没有任何概率目标正支持，validation 没有 ε-完整样本。因此当前数据适合软件开发、学习流程调试和受限探索，不适合确认性模型比较或研究级神经网络训练。

## 2. 方法可行性与科学性评估

### 2.1 Taylor–Shapiro 更新

v2 采用以下定义：

\[
q=\frac{V_f}{2V_d},\qquad \gamma_t=\arcsin q,
\]

\[
\gamma_s=\arctan\!\left[q\cos\left(\frac{\pi}{2}+\phi_2-\phi_1\right)\right],
\qquad |\gamma_s|\le\gamma_t.
\]

其中，`V_f` 为 Gurney 破片速度，`V_d` 为爆速，`φ1` 为局部壳体法线相对弹轴的极角，`φ2` 为爆轰波方向相对弹轴的极角。实现使用同一圆柱内壁环布局生成破片源点、局部法线和爆轰波方向，避免源点与方向几何不一致。

可行性判定：

- 公式、坐标约定、速度比和适用几何均已显式编码；
- `0<q≤1`，越界时直接拒绝，不进行静默截断；
- `σ=0` 时方向与随机种子无关，旧 Taylor 路径保持逐位不变；
- v2 使用独立注册表、案例 ID、特征 one-hot 和模型卡；
- 定向测试覆盖参考角、Taylor 界、几何布局、定义域拒绝、输入特征身份和旧路径回归。

科学适用范围为轴对称直圆柱侧壁破片战斗部、前端轴上点起爆和已知壳体内壁几何。端板破片、非轴对称预制破片层、复杂波形整形和有限加速历程需要单独模型。外部标定应提供真实几何、方向统计、V50、部件 E50 和色散参数。

### 2.2 第 9 部分纠正后的方法

| 原思路 | 最终方法 | 科学理由 |
|---|---|---|
| 用终止/透明双分支构造全局上下界 | 用受影响重复比例构造严格最坏识别区间；仅在逐标签证明单调性后收紧 | 装甲开孔、时序和后续可达性不满足全局单调性 |
| 用解析分布和 KL 修正期望 | 零概率用 Brier/ECE，正值用 `log1p` Wasserstein-1 和分位数，多变量用 RBF-MMD，分箱描述用 JS | 点质量、支撑不一致和高维密度使 KL 不稳定 |
| 用毁伤树改变标签分布以便学习 | 在下游模型内使用 train-only 可逆目标编码器，原尺度评价 | 学习变换与标签语义解耦，保持物理层和功能层稳定 |
| 在原始 15 维空间追求均匀 | 在连续来向、瞄点和速度等有效坐标上记录设计测度与权重 | 输入覆盖与输出分布分开治理 |

## 3. 七项决策的执行结果

1. **方向模型：** 最终试生成选用 Taylor–Shapiro v2；legacy Taylor 环保留为可复现对照。
2. **来向范围：** Reference 移除 `bottom_up`，使用连续方位/俯冲；`bottom_up` 只保留为压力测试语义。
3. **删失处理：** 采用 ε-完整与最坏识别区间；跳弹物理留给获得材料和试验依据后的新模型。
4. **树输出目标：** 保持逐重复评树后取平均；作用于部件期望概率的结果只作为派生视图。
5. **树语义：** 23 个未引用部件使用 provisional identity fallback；功能分组、F1/M1 和 K 级定义等待领域权威审定。
6. **输入合同：** 采用 16 维首次装甲状态作为冻结物理 core；代理量按需派生，只进入显式四臂消融，不改变物理输入合同。
7. **计算预算：** E1 的 32 个非 test 工况×R32 pilot 给出最坏 `p_hit` 均值估计 RMSE 0.0450，因此本轮 360 例试生成采用 R32。

## 4. 实施范围

| 模块 | 已实施 | 当前证据等级 |
|---|---|---|
| E-1/E-3/E-4 | 删失识别区间、R 无关标签、充分统计、逐重复 trace、离线树重算 | 单测与缩放链路通过；正式规模待补 |
| E-2 | legacy 模型卡、Taylor–Shapiro v2、独立注册表和回归测试 | 软件实现通过；物理标定待补 |
| E-5/E-6/E-7 | MC 噪声、参数区间、Morris/Sobol、§6.8 机理因子映射、装甲占位显式登记 | 32 点因子设计与缩放执行完成；正式效应预算、跨种子稳定性和权威参数待补 |
| E-8 | weighted noisy-OR、精确 k-of-n、2-additive Choquet、identity fallback | 数学与工程测试通过；树语义门未通过 |
| E-9/E-10 | 连续来向、球面测度、SamplingSpec 1.7、16 维输入、Schema 和依赖同步 | 最终 v3 与学习报告完成；全量 1,280 tests、compileall、pip check、5 组 Schema 镜像和差异检查通过 |
| G-1/G-2/G-6 | 跨栏目标、软标签、方差/CI、区间与异方差损失 | 实现与测试通过；最终统计审计完成并判定正式训练门失败 |
| G-4/G-5 | Reference、暴露场、固定样本暴露均衡臂、连续方向 | 工程运行完成；暴露臂未晋升 |
| G-7/G-8/G-9 | dataset-v2、可追加树视图、validation-only 选模、严格零解码开发包与 sealed test | 最终打包/验证/开发审计和学习报告完成，test 零解码/零打开；validation 选择不可定义，一次性 test 未执行 |

完整逐项状态见 [IMPLEMENTATION_MATRIX.md](IMPLEMENTATION_MATRIX.md)，执行过程见 [WORKLOG.md](WORKLOG.md)。

## 5. 已完成实验与关键发现

### 5.1 Phase 0、E1、E4、E6

- Phase 0 完成 2 种输入×3 种训练比例×3 个种子的 18 次训练；全量训练时 `x_design` 部件宏 PR-AUC 为 0.740，`x_normalized` 为 0.427。当前首要学习瓶颈是物理输入表征，随后是 validation 仅有 17–22 个正例的稀有部件。
- E1 修正版多目标复跑完成 32 个非 test 工况×R32。`p_hit/P(degraded)/P(failed)` 最大 RMSE 分别为 0.04503/0.04292/0.01911，建议 R 为 32/32/8；命中后命中数的 48 个可估部件在 R32 全部通过 0.10 容差。命中后能量最大 RMSE 为 1.011，19 个部件信息不足，3 个部件在候选 R1024 内仍不达 0.10。32/32 工况受域删失影响，平均受影响重复比例为 0.921875，因此 MC 方差不能代替删失识别区间。结果对象哈希为 `sha256:494d011a0fc86870af1d302492375e69e34b947ef92e4d3e20301335cba97ebe`；完整分析见 [E1_MULTITARGET_ANALYSIS.md](E1_MULTITARGET_ANALYSIS.md)。
- E4 增强诊断使用 5,479 个 complete case 和 41,262 个正观测；加权全模型 R²=0.76814。drop-one ΔR² 中 `ln(hits)=0.477556`、来向固定效应 `0.004877`、`ln(1/r²)=3.42e-6`、弹型固定效应 `2.71e-7`、`ln(mV0²)=0`。设计矩阵秩为 10/11、条件数为 `3.16e14`，`mV0²` 与弹型固定效应混叠。三弹型 Hill 范围为 1.192–53.903、1.093–29.692、0.897–24.739，当前数据不支持单一稳定尾指数。
- E6 覆盖 15 个实际来向×弹型单元；支持向量代理叶联合熵为 10.8394 bit，树输出联合熵为 6.0560 bit，描述性压缩比为 0.5587。该比率不是精确树叶信息保留率；精确叶向量和候选 view 对照待最终逐重复 trace。详见 [E4/E6 增强报告](../../experiments/E4_E6_enhanced_20261120/README.md)。

### 5.2 E0 与 E2/E7 冻结诊断

E0 使用 8 个基准工况×R8 完成冻结缩放复跑。距离 22/22 项等效且差异为 0；滚转仅 9/33 项等效。因此距离移出默认输入，滚转保留。结果哈希为 `sha256:82a5df98ff7f90e43397eb960994523c854b7a5c58d3c619d6a114983dfdedd6`，详见 [E0_ANALYSIS.md](E0_ANALYSIS.md)。

冻结版 E2/E7 使用 16 个工况、R=2、67 个部件，共记录 381,900 条碎片—部件轨迹：

- 98.383% 无几何相交，是当前漏斗的首要收缩环节；
- 6,175 条几何相交轨迹中，38.154% 在到达部件前被阻挡；
- 3,819 条实际部件命中中，93.401% 进入 `response_possible`；
- `M_armor` 有 73.069% 低于 V50；
- `M_comp` 有 6.441% 低于 degraded 阈值、88.164% 低于 failed 阈值；
- 733 个穿甲碎片中 425 个到达部件，`M_reach=57.981%`。

这些结果把稀疏主因定位为几何覆盖和路径可达性。详细分层见 [E2_E7_ANALYSIS.md](E2_E7_ANALYSIS.md)。该运行用于冻结软件教师的机制诊断；正式匹配结论需要扩大独立工况并标定装甲、阈值和方向参数。

### 5.3 E3/E5/C2/C3 与 §6.8 因子设计

- E3 完成 16 个因子配置×3 个工况、R=1，验证公共随机数、因子编码、逐重复 trace、离线树和 bootstrap 链；离线树误差为 0。
- E5 完成 normal 噪声下 9 个 `σ×τ` 组合×3 个工况、R=1，并完成 uniform/laplace 同方差路径测试。
- C2 三弹型 Gurney 最大误差为 `4.547e-13`；方向角偏差分别为 -0.1390°、-0.0758°、+0.0051°；V50 中点结果接近 0.5。
- C3 完成 12 次代理 Morris 和 28 次单工况 R1 教师 Sobol 评估。该结果验证敏感性执行链，正式排序需要扩大预算并检查跨种子稳定性。
- §6.8 机理因子设计完成 32 点 scrambled Sobol：最大绝对相关系数 0.190716、最大 VIF 1.048105，31/32 点可行，1 点因爆速越界拒绝；31 个可行点均以同一非 test 工况、R1 执行。结果哈希为 `sha256:4b08c678ed80e497e34208e74798ddb6636885ad49c630dfa6e161d773f56c24`。该结果验证因子可独立设计、映射和拒绝，尚不构成真实主效应结论。详见 [§6.8 因子设计报告](../../experiments/warhead_factor_design_20261120/README.md)。

### 5.4 X1–X3 与树候选

冻结 X1–X3 在 source/schema/engine 哈希前后一致的条件下完成 100,000 个代理候选、150×R8 教师审计、2,000 对 2,000 采样臂对照和配对多保真评估：

- X1 教师 false-unreached 为 12.845%，Wilson 上界为 13.53%，高于 3% 门；
- X2 共同工程门通过；最小纳入概率为 0.01133，未达到原 0.08 门，暴露臂 ESS 比例为 0.753；至少 20 个支持样本的部件数由 39 增至 58；
- X3 holdout 方差缩减为 5.09%，教师样本等效倍率为 1.054，收益较小。

冻结证据的晋升结论为失败，Reference 保持主臂。最终四臂代理学习报告覆盖 teacher-only、67 维派生代理特征、教师−代理残差和代理预训练后教师微调；四臂均因 validation `p_hit` 可观测数为 0 而 `inadmissible`，`selected_arm=null`，512 例代理预训练池在主验证门失败后未参与选择。代理量按需派生，不写入 16 维物理输入 core；X1 failed 与最终不可选择结论共同保持 `diagnostic_only_not_promoted`。冻结证据见 [状态文件](../../experiments/X1_X2_20261120_frozen_v2/FROZEN_EVIDENCE_STATUS.json)。

树层已实现 weighted noisy-OR、精确 Poisson-binomial k-of-n、受约束 2-additive Choquet、预注册选择器和独立验证 seed。23 个未引用部件缺少权威语义，当前 dataset-v2 使用 identity fallback，G4 保持未通过。

## 6. 可信度与阶段门

| 门 | 当前判定 | 主要原因 |
|---|---|---|
| G0 基线与诊断 | 未通过 | B0/E1 多目标/E4/E6 和缩放 E0/E2/E7 已覆盖主要问题；正式 E0/E1/E2/E7 规模未完成 |
| G1 引擎语义 | 部分完成 | R 无关标签、trace、方向身份、离线树、ε/MNAR 正式报告和全量 1,280 项回归已完成；C1 正式规模与数据完整性门仍失败 |
| G2 机理实验 | 未通过 | E3/E5/C2/C3 已缩放运行；正式预算、跨种子稳定性和 C4 专家断言缺失 |
| G3 暴露场/采样 | 未通过 | 冻结复跑已完成；X1 教师反证门和 X2 原预注册门失败，X3 收益小，Reference 保持主臂 |
| G4 树冻结 | 未通过 | 单调候选与选择流程完成；功能语义权威缺失 |
| G5 正式数据/训练 | 未通过 | v3 writer、严格 dataset-v2、开发审计和 validation-only 学习报告已完成；validation `p_hit` 可观测数为 0，四 bundle 与四臂均不可选择，`formal_neural_training_ready=false` |

当前可信度档案保持 `research_admission=false`、`physical_validation_claim_allowed=false`。该结论来自明确的外部证据缺口：C4 专家断言、23 个未引用部件语义、真实战斗部几何、装甲材料与厚度、V50/E50 和方向统计。软件测试通过后，软件一致性与物理可信度仍分别记录。

## 7. 最终采样策略

本次最终试生成固定为：

| 项目 | 冻结值 |
|---|---|
| 主采样臂 | Reference |
| SamplingSpec | 1.7 连续方向 |
| 来向 | 连续方位/俯冲，Reference 排除 bottom-up |
| 物理注册表 | Taylor–Shapiro v2 |
| 输入 | 16 维首次装甲状态 |
| 标签 | hurdle v2，主目标不依赖 R |
| 重复数 | R=32 |
| formal case | 360 |
| split | train/validation/sealed test，标签观察前冻结 |
| 暴露均衡臂 | 未晋升，不进入本次最终试生成 |

该策略优先满足身份稳定、设计权重可解释和封存测试可执行。暴露均衡臂保留为独立研究分支，待 X1 反证率和 X2 positivity/ESS 门同时通过后再评估晋升。

## 8. 最终运行结果

### 8.1 v3 数据集与 dataset-v2

运行目录：`artifacts/datasets/idea2_taylor_shapiro_continuous_r32_20261120_v3/`

| 指标 | 最终值 |
|---|---|
| writer 状态、shard 数、manifest 哈希 | `complete；23 shards；sha256:2ceef7e95643f91343b97ff191a326d7f30d7e63b0edc39bb8e14b9c1b1de522` |
| train / validation / sealed test 数量 | `240 / 60 / 60；观测标签前冻结` |
| 域完整率、拒绝率、识别区间宽度 | `开发集 ε-完整 2/300=0.006667；train 2/240，validation 0/60；writer 0 refused；三个概率目标区间宽度中位数均为 1.0` |
| 各部件正例数、条件量可用率、稀有部件数 | `三个概率目标正支持 min/median/max=0/0/2；52/67 部件零支持；条件能量/命中数观察率均为 0.134478` |
| dataset-v2 结构与哈希验证 | `通过；ID damage-dataset-v2-aa0fc301038df1c79ac21095；内部 manifest sha256:e0fa6c68…1178a；23 个源分片及源 manifest 全匹配` |
| sealed test `open_count` | `0；test_result_json_lines_decoded=0；test 指标未计算` |
| 严格开发包零解码回归 | `通过；最终 v3 开发包与故意损坏 test JSON 回归均证明 test 行在 JSON 解码前跳过` |

### 8.2 目标编码器与学习曲线

| 项目 | 最终值 |
|---|---|
| identity / Box–Cox / empirical quantile-normal / hybrid 同预算消融 | `四 bundle 全部 inadmissible：validation p_hit observation_count=0` |
| validation 选择的 bundle | `null；未用条件目标替代预注册 p_hit 选择指标` |
| 25% / 50% / 100% × 3 seed 原尺度指标 | `请求比例 0.25/0.5/1.0，seed 17/29/43，120 epochs；主验证门先失败，未产生可比较指标` |
| `p_hit` Brier/ECE、正值 MAE、学习曲线方向一致性 | `均为 null；train/validation/test 的 p_hit 观测数为 134/0/0；条件能量/命中数为 2,089/614/0` |
| teacher-only / 代理特征 / 残差 / 代理预训练四臂 | `四臂全部 inadmissible，selected_arm=null；代理预训练池 512 例未越过主验证门；X1 failed，diagnostic-only` |
| test 访问状态 | `test_result_json_lines_decoded=0；open_count=0；未用于 loss、metric 或 selection` |
| 报告身份 | `hurdle report sha256:620d289c…9c1a，file SHA-256 bdca7f79…a035；四臂 report sha256:48046246…ce88，file SHA-256 8ea914e2…173` |

### 8.3 代理—教师分布

| 指标 | 最终值 |
|---|---|
| design-weighted Brier / ECE | `0.0219319 / 0.0357256；仅 134 个可识别工况—部件对` |
| 正值 `log1p` Wasserstein-1 与分位数 | `W1=1.104171；仅 6 个正能量配对；Pearson/Spearman=0.709857/0.771429` |
| RBF-MMD / JS | `MMD²=0.797647（bandwidth=0.887653，仅 2 个完整工况）；JS=0.0312804` |
| region×profile 最差分层 | `不可定义：15 个分层中只有 pos_x::small_front 可估（134 对，Brier 0.0219319、ECE 0.0357256、MAE 0.0359897），其余 14 个为 0 对` |
| test JSON 解码数 | `0；报告只含 train/validation 300 例` |
| 报告身份 | `report sha256:6d4b1ccd…54dc7；file SHA-256 711363ff…b9c` |

### 8.4 冻结 X1–X3 复跑

| 门 | 冻结结果 |
|---|---|
| 引擎/source/schema 哈希复核 | `通过；执行前后一致，引擎 sha256:33e6…b59ad` |
| X1 false-unreached ≤3% | `失败：12.845%，Wilson 上界 13.53%` |
| X2 positivity / ESS / 覆盖门 | `共同门通过；原 min π 门失败（0.01133<0.08）；ESS=0.753；支持≥20 的部件 39→58` |
| X3 方差缩减与教师节省 | `5.09%；等效倍率 1.054，收益小` |
| 暴露均衡臂最终晋升 | `不晋升；Reference 保持主臂` |

### 8.5 最终训练适用性

最终判定：

- **软件/流程适用性：** 适合。writer、恢复、严格打包、哈希验证、零解码封存和开发审计链路均已执行。
- **探索性神经网络训练：** 仅适合受限的流程验证与诊断。训练须使用 mask、识别区间和精度权重，并明确 240 个 train 与极低正支持的范围。
- **确认性模型比较：** 不适合。validation 没有 ε-完整样本或可观测 `p_hit` 标签，四种编码器和四个代理学习臂均被正确拒绝选择。
- **研究级训练准入：** 不通过。240<2,000、train ε-完整率 0.008333、52/67 部件零正支持，可信度档案也保持 `research_admission=false`。
- **物理外推：** 不适合。方向统计、真实几何、材料、装甲厚度、V50/E50、响应阈值和树语义尚未完成外部标定。

判定依据包括完整率、有效训练样本、各部件正例支持、条件目标覆盖、设计权重 ESS、validation 选择门、代理—教师偏差、sealed test 访问记录和可信度档案。

### 8.6 最终软件质量检查

| 检查 | 结果 |
|---|---|
| 全量 pytest | `1,280 passed`；独立 collect 为 1,280 tests / 129 files |
| 编译与依赖 | `compileall` exit 0；`pip check` 无破损依赖 |
| Schema 镜像 | 5/5 同名镜像，0 mismatch |
| 差异与残留 | `git diff --check` exit 0；lock=false；incomplete marker=false；临时 shard=0 |

完整记录见 [FINAL_QA_20261120.md](FINAL_QA_20261120.md)。

## 9. 后续工作

1. 优先修复穿透适用域与输入覆盖，使 validation 获得可观测概率标签；达到分层支持门后重新生成独立数据，保持当前 sealed test 未开。
2. 由领域权威审定 23 个未引用部件、F1/M1、K 级和至少 10 条 C4 模式断言；随后执行 T1 正式选择与独立验证。
3. 用真实弹体几何、方向统计、材料、装甲厚度、V50 和 E50 标定 Taylor–Shapiro 与响应模型。
4. 扩大 E0/E1/E2/E3/E5/C3 至预注册规模；将缩放链路证据升级为稳健性结论。

## 10. 证据索引

- [全量实施矩阵](IMPLEMENTATION_MATRIX.md)
- [实施工作日志](WORKLOG.md)
- [E2/E7 冻结诊断](E2_E7_ANALYSIS.md)
- [E0 冻结诊断](E0_ANALYSIS.md)
- [E1 多目标噪声与 R 选择](E1_MULTITARGET_ANALYSIS.md)
- [最终数据集 manifest](../../datasets/idea2_taylor_shapiro_continuous_r32_20261120_v3/formal_dataset/manifest.json)
- [严格 dataset-v2 manifest](../../datasets/idea2_taylor_shapiro_continuous_r32_20261120_v3/dataset_v2_development/manifest.json)
- [开发审计](../../datasets/idea2_taylor_shapiro_continuous_r32_20261120_v3/development_audit.json)
- [编码器消融](../../datasets/idea2_taylor_shapiro_continuous_r32_20261120_v3/hurdle_learning/ablation_results.json)
- [代理—教师分布](../../datasets/idea2_taylor_shapiro_continuous_r32_20261120_v3/proxy_teacher_distribution.json)
- [四臂代理学习](../../datasets/idea2_taylor_shapiro_continuous_r32_20261120_v3/proxy_learning_ablation.json)
- [最终软件 QA](FINAL_QA_20261120.md)
- [E4/E6 增强报告](../../experiments/E4_E6_enhanced_20261120/README.md)
- [§6.8 因子设计报告](../../experiments/warhead_factor_design_20261120/README.md)
- [X1–X3 冻结证据](../../experiments/X1_X2_20261120_frozen_v2/FROZEN_EVIDENCE_STATUS.json)
- [Taylor–Shapiro 模型卡](../../../docs/FRAGMENT_DIRECTION_MODEL.md)
- [Stage 2 缩放报告](../idea2_optimization_20261120/stage2_representative_scaled/README.md)
- [Phase 0 摘要](../phase0/README.md)
- [可信度档案](credibility_dossier.json)

