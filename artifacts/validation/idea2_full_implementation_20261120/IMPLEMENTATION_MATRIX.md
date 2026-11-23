# idea2 全量实施矩阵

日期：2026-11-20  
证据截止点：冻结 E0、E1 多目标、E2/E7、X1–X3、E4/E6 增强诊断和 §6.8 机理因子设计均已完成；最终 R=32 v3 writer、严格开发包、开发审计、validation-only 学习评估和全仓软件质量检查均已完成。

本矩阵把“实现存在”“缩放运行完成”“文档规定规模完成”“需要外部权威”和“阶段门通过”分开记录。代码或单元测试存在，不等于研究实验已经按规定规模完成；软件教师一致性通过，也不等于外部物理有效性成立。

## 状态图例

| 标记 | 含义 |
|---|---|
| 代码已实现 | 接口、校验、CLI 或测试已进入工作区 |
| 缩放运行 | 端到端链路以低于文档规定的预算运行 |
| 全规模未跑 | `idea2.md` 规定的样本量、重复数或独立种子尚未完成 |
| 外部权威所需 | 需要领域专家、权威几何、材料/阈值来源或实测数据 |
| 子门通过 | 一个明确的软件、几何或统计子门有机器证据通过 |
| 门未通过 | 该阶段的全部准入条件没有同时满足 |

## 第 9 部分纠正后的执行原则

| 纠正项 | 最终决策 | 实现与证据 | 当前边界 |
|---|---|---|---|
| 全局 L/U | 不采用“终止传播/透明传播”为全部输出构造全局上下界。装甲开孔、传播顺序和后续可达性没有全局单调性保证。保留受影响重复比例给出的严格 `[max(0, point-q), min(1, point+q)]` 识别区间；双分支只能作为情景分析。 | `damage_engine/engine.py`、`damage_engine/hurdle_targets.py`、`configs/labels/loitering_impact.fragment_only.hurdle.v2.json`、D011 | 标签专用收紧界只有在逐标签证明单调性后才能加入；ε-完整率和非随机缺失全规模报告未完成。 |
| KL 替换 | 跨栏分解后比较：零/概率部分用 Brier 与校准；正值部分用 `log1p` 尺度 Wasserstein-1 和分位数；多变量用 RBF-MMD；需要单一分箱描述量时用 Jensen–Shannon。 | `damage_engine/distribution_metrics.py`、`tools/evaluate_proxy_teacher_distribution.py`、最终 `proxy_teacher_distribution.json`、对应测试 | 最终 train/validation-only 报告完成：134 个可识别工况—部件对、仅 2 个完整工况；Brier 0.02193、ECE 0.03573、JS 0.03128；正能量仅 6 对，log1p W1=1.10417；MMD²=0.79765。15 个分层中仅 `pos_x::small_front` 可估，不能据此推广到完整设计域。 |
| 目标编码器位置 | 编码器只属于下游训练流程：仅在 train 拟合、可逆、与模型共同保存、原尺度评价；不修改 label spec 的物理估计目标。 | `damage_engine/target_encoding.py`、`damage_engine/hurdle_baseline.py`、最终 `hurdle_learning/ablation_results.json`、对应测试、hurdle v2 中 `label_mutation=false` | identity/Box–Cox/经验分位数正态/section9 hybrid 四 bundle 均因 validation `p_hit` 可观测数为 0 而 `inadmissible`；`selected_bundle_id=null`，未用条件目标替代预注册选择指标。 |
| 执行顺序 | 基线先于标签、树和采样修改；物理层、功能树和学习变换分层管理。 | [Phase 0 摘要](../phase0/README.md)、D012–D015 | G0 只完成 B0 主体，未完成该门要求的全部机理诊断。 |

## E-1 至 E-10：引擎问题与修正

| ID | 已实施内容 | 证据 | 运行状态 | 判定 |
|---|---|---|---|---|
| E-1 域删失 | 采用受影响重复的严格最坏识别区间，区分 Monte Carlo 区间、删失识别区间和认知情景区间；宽区间只用区间监督。 | `engine.py`、`hurdle_targets.py`、`hurdle_learning.py`、hurdle v2、最终 [development_audit.json](../../datasets/idea2_taylor_shapiro_continuous_r32_20261120_v3/development_audit.json)、D011 | ε=0.01 时开发集仅 2/300 完整（train 2/240，validation 0/60）；受影响重复比例均值 0.9104，中位数 1.0。三个概率目标识别区间宽度中位数均为 1.0。Holm-KS 仅 18/276 项可评估，0 项校正后显著，样本过少不能据此认定随机缺失。 | 软件链完成；数据完整性门失败，正式神经网络训练不准入。 |
| E-2 方向模型 | 模型卡记录 legacy 空心环、实心锥与 Taylor–Shapiro 路线；Taylor–Shapiro 使用独立物理注册表和案例 ID，避免静默改变旧身份。 | [FRAGMENT_DIRECTION_MODEL.md](../../../docs/FRAGMENT_DIRECTION_MODEL.md)、`configs/physics/loitering_impact.taylor_shapiro.v2.json`、`tests/test_taylor_shapiro_direction.py` | legacy 的 C2 缩放对照已完成；Taylor–Shapiro 有代码与金标测试；外部物理来源、适用域和实测确认仍缺。 | 软件实现可用，物理模型保持 `unverified`。 |
| E-3 R 依赖标签 | 主目标改为 `p_hit`、`P(degraded)`、`P(failed)` 和显式命中后条件量；保存样本数、方差、SE、CI；二值 support 仅为固定阈值派生视图。 | `damage_engine/hurdle_targets.py`、`damage_engine/hurdle_learning.py`、hurdle v2、`tests/test_hurdle_targets.py` | R=4/R=16 语义不变性单测通过；文档要求的 200 工况 R4/R16 等效性全规模未跑。 | 代码已实现，label spec 仍为 `provisional`。 |
| E-4 物理轨迹/树解耦 | 新增轻量 `component_state_trace`、离线树重算、树定义哈希、dataset-v2 core/view 分层。权威树目标保持逐重复评树后平均。 | `damage_engine/engine.py`、`tools/recompute_tree_outputs.py`、`damage_engine/dataset_v2.py`、最终严格 dataset-v2、对应测试 | 单测在线/离线最大误差 `1.11e-16`；Stage 2 缩放运行误差为 0；最终严格开发包结构验证通过，dataset-v2 ID 为 `damage-dataset-v2-aa0fc301038df1c79ac21095`。 | 核心和正式数据视图工程子门通过；树语义门仍由 G4 管理。 |
| E-5 Monte Carlo 噪声 | 在线输出重复方差、命中后条件充分统计；提供二项损失、异方差高斯损失、精度权重和 E1 多目标方差分解入口。 | `online_statistics.py`、`hurdle_learning.py`、`tools/run_idea2_engine_experiments.py`、`artifacts/experiments/E1_multitarget_taylor_shapiro_continuous_20261120/results.json`、[E1 多目标分析](E1_MULTITARGET_ANALYSIS.md) | 32 个非 test 工况×R32 已完成：三个概率目标最大 RMSE 为 0.04503/0.04292/0.01911；命中后命中数 48 个可估部件均在 R32 通过 0.10 容差；命中后能量最大 RMSE 1.011，3 个部件在候选 R1024 内仍不达 0.10，19 个部件信息不足。 | R32 支持本次试生成的概率目标和命中后命中数；条件能量必须保留 mask/方差/精度权重。500 工况正式 E1 待补。 |
| E-6 参数不确定性 | 建立五组探索性区间与方向模型分类因子；实现 Morris/Sobol；另实现 §6.8 的 `V0/m/N/θ/σ` 机理因子映射、质量/几何检查、拒绝台账与 VIF 门。 | uncertainty v1、`warhead_mechanism_factor_design.v1.json`、`tools/run_c3_parameter_sensitivity.py`、`tools/run_warhead_mechanism_factor_design.py` | C3 完成 12 次 Morris 与 28 次教师 Sobol；机理因子完成 32 点 scrambled Sobol，max |corr|=0.190716、max VIF=1.048105，31/32 点以同一非 test 工况 R1 执行，1 点因爆速越界拒绝。 | 两条缩放链路完成；跨种子稳定性、正式预算和外部参数范围仍待补。 |
| E-7 装甲占位 | 10 mm 占位值被显式登记；装甲缩放 `[0.1,1.0]` 进入 E3 A 因子和 C3 参数合同。 | uncertainty v1、Stage 2 E3 配置与缩放报告 | E3 16×3、R1 缩放运行完成；权威厚度、材料来源和正式稳健性实验缺失。 | 外部权威所需。 |
| E-8 原始树直导 | 实现 weighted noisy-OR、精确 Poisson-binomial k-of-n、受约束 2-additive Choquet、identity fallback、独立开发/验证 seed 选择和 dataset-v2 树视图。 | `damage_engine/offline_tree_candidates.py`、`tree_candidate_selection.py`、[树实施记录](../tree/IMPLEMENTATION_LOG_20261120.md) | 37 项定向测试通过；23 个未引用部件仅生成 provisional identity fallback；未冻结获批候选。 | 数学/工程子门通过，T1 语义门未通过。 |
| E-9 离散来向 | 实现 SamplingSpec 1.7 连续方向、动态切向基、球面测度记录；主 Reference 排除 bottom-up，另保留压力测试语义。 | `damage_engine/continuous_direction_sampling.py`、`configs/sampling/p0_sphere_entry_continuous_direction_reference.v1.json`、最终 v3 manifest 与 SamplingSpec 1.7 writer/loader/manifest 定向测试 | 最终 R32 v3 writer 完成 360/360、23 shards、0 refused，split 为 240/60/60；dataset ID 为 `damage-dataset-99e74282d7d911ef4287476b`，根 manifest 哈希为 `sha256:2ceef7e95643f91343b97ff191a326d7f30d7e63b0edc39bb8e14b9c1b1de522`。 | Reference 连续来向正式试生成归档完成；统计适训门已审计并失败。 |
| E-10 一致性/维护 | README 采样依赖和入口已更新；requirements 与 Python 3.12/NumPy 2.4.6 环境对齐；`component-057` 缺失几何有显式导入例外；pytest cache 使用禁用策略。 | [environment.json](environment.json)、[component-057 例外](source_import_exception_component_057.json)、[最终 QA](FINAL_QA_20261120.md)、README | 当前工作树全量 pytest 为 1,280 passed；compileall 和 pip check 通过；5 组 Schema 镜像 0 mismatch；`git diff --check` exit 0；无 lock、incomplete marker 或临时 shard。 | 软件一致性与维护门通过。 |

## C1 至 C6：可信度档案

当前机器档案为 [credibility_dossier.json](credibility_dossier.json)。它明确给出 `research_admission=false` 和 `physical_validation_claim_allowed=false`。

| ID | 当前证据 | 缺口 | 状态 |
|---|---|---|---|
| C1 性质与极限 | 已有顺序、闭合、部分单调/零极限、门函数随机单调、重复收敛工具；端到端单调、镜像几何、Taylor–Shapiro `σ→0` 与冻结 E0 缩放检验完成。 | E0 200×R32 TOST、发布级收敛绑定和更广条件的性质覆盖。 | `incomplete`；缩放 E0 判定为“移除距离、保留滚转”。 |
| C2 解析/半独立对照 | 缩放运行覆盖三弹型 Gurney、legacy 角度金标、V50 中点；Taylor–Shapiro 独立测试和最终 v3 代理—教师报告已完成。 | 最终报告仅 2 个完整工况且 15 个分层中只有 1 个可估；仍缺正式方向重复、V50 试验规模和跨分层完整支持。 | 缩放与最终诊断报告完成，正式门未通过。 |
| C3 全局敏感性 | 参数合同、Morris/Sobol 设计/分析/执行代码和公共随机数绑定已实现；12 次代理 Morris 与 28 次单工况 R1 教师 Sobol 已运行。 | 正式预算、跨种子稳定性和稳健角点数据集。 | 缩放运行完成，正式门未通过。 |
| C4 专家模式断言 | 已生成 10 项空白权威模板，代码禁止空白断言通过。 | 具名领域权威提供断言、来源、适用条件和置信度，并执行检查。 | `external_required`。 |
| C5 不确定性报告 | MC CI、删失识别区间、参数情景区间已分离；多目标 E1 已报告概率与条件目标噪声；最终 v3 已报告 ε-完整率、识别区间、正支持、条件量覆盖、ESS 与 Holm-KS 缺失诊断。 | 在 500 个分层工况复核 E1；提高 ε-完整与正支持后重跑稳定的 region×profile 缺失诊断。 | `incomplete`；最终开发审计已绑定，统计准入失败。 |
| C6 版本化档案 | 计划、正式数据/学习证据路径、最终 QA 和内容哈希机制已绑定；dossier 在最终 QA 后重新生成。 | C1–C5 仍有正式规模、统计支持和外部权威缺口；新增证据后继续版本化重建。 | `incomplete`；研究准入未通过。 |

## G-1 至 G-9：数据集与学习设计

| ID | 已实施内容 | 运行状态 | 判定 |
|---|---|---|---|
| G-1 跨栏目标 | 域/到达/命中后量级/有序状态/树派生视图已进入 hurdle v2、dataset-v2 和损失库；`P(failed)≤P(degraded)` 由参数化保证。 | 组件与单测完成；最终多任务网络未训练；固定可微树层未作为权威标签。 | 代码已实现，模型验证未完成。 |
| G-2 软标签与方差 | 保存 success/trial、方差、SE、CI；MC 精度权重与设计权重、任务权重形状分离。 | 单测完成；E1 支持本次试生成使用 R32，方差地板和生产级自适应预算仍待预注册。 | 部分完成。 |
| G-3 输入合同 | 旧 15 维保持身份；16 维首次装甲状态已贯通 writer/loader；连续来向进入同一物理表征；代理量保持为派生学习信息。 | 16 维端到端测试与最终四臂报告完成；teacher-only、67 维代理特征、教师−代理残差和代理预训练四臂均因 validation `p_hit` 可观测数为 0 而不可选择。 | 工程子门通过；代理不写入冻结物理输入 core，也未因缺失验证标签而被晋升。 |
| G-4 采样臂 | Reference、暴露场、固定样本暴露均衡选择器、positivity/ESS/足迹审计已实现；选择只使用代理并限制在 train。 | 冻结复跑完成 10^5 候选、150×R8 教师与 2,000 对 2,000；执行依赖哈希前后一致。false-unreached=12.845%，原 positivity 门失败。 | 暴露臂不晋升；Reference 保持最终策略。 |
| G-5 连续来向 | SamplingSpec 1.7、两个方向潜变量、动态切向基、球面测度和 no-bottom Reference 已实现。 | 最终 R32 writer 完成 360/360，23 shards、0 refused；240/60/60 split 在观测标签前冻结；训练适用性审计为 false。 | Reference 主臂归档完成；统计训练门失败。 |
| G-6 删失数据 | ε-完整点+区间监督、宽区间仅区间损失、拒绝仅台账均有实现。 | 最终开发审计完成：ε-完整率 0.006667，三个概率目标识别区间宽度中位数均为 1.0。 | 代码实现通过，最终数据统计门失败。 |
| G-7 dataset-v2 | core inputs/trace/labels/weights、哈希验证、可追加树 view 和严格开发包转换器已实现。开发包保留 test 输入/身份，以全掩码占位替代真实 test 标签。 | 最终包 360 例、67 部件，结构/哈希验证通过；内部 manifest hash 为 `sha256:e0fa6c68af2b96ac7f74086139d30292274240a296bf9aa11ba9ee0c3231178a`。23 个源分片和源 manifest 全部匹配。 | 严格封存和归档工程子门通过。 |
| G-8 可学习性 | Phase 0 完成 2 输入×3 比例×3 seed；最终 hurdle 预注册 25%/50%/100%×3 seed×4 bundle 和四臂报告器保持 Reference validation 选择、test 未读。 | 最终 validation `p_hit` 可观测数为 0，四个 bundle 与四个代理臂全部 `inadmissible`，选择指标为 null；训练/验证/test 的 `p_hit` 观测数为 134/0/0。 | 流程门通过，最终可学习性门失败；不允许选模。 |
| G-9 封存测试 | 严格开发包在 `json.loads` 前跳过 test 结果行；test 真标签与 trace 不进入开发包，baseline 只见全掩码占位。先前已读 pilot 保持 diagnostic-only。 | 最终 `sealed_test.json` 记录 test 60 例、`test_result_json_lines_decoded=0`、`test_open_count=0`、所有 test label mask 为 false；test 指标未计算。 | 封存写入/训练边界通过；一次性确认性读取保持未执行。 |

## E0 至 E7：实验登记

这里的 E0–E7 是实验编号，与上面的 E-1–E-10 问题编号分开。

| ID | 实现 | 已执行证据 | 尚缺 | 状态 |
|---|---|---|---|---|
| E0 干扰维 | roll×distance 变体、成对 TOST 和完整指标入口已实现。 | 8 个基准工况×R8 冻结复跑完成：距离 22/22 项等效且差异为 0；滚转仅 9/33 项等效，结果哈希 `sha256:82a5df98ff7f90e43397eb960994523c854b7a5c58d3c619d6a114983dfdedd6`。 | 200 工况×R32 正式 TOST。 | 缩放结论：距离移出默认输入，滚转保留。 |
| E1 方差分解 | 逐工况均值/方差、随机与输入方差占比、R 选择字段已实现；多目标入口覆盖 `p_hit/P(degraded)/P(failed)` 与两项命中后条件量。 | 修正版多目标复跑完成 32 个非 test 工况×R32。概率目标最大 RMSE 为 0.04503/0.04292/0.01911；命中后命中数 48 个可估部件在 R32 全部通过；条件能量无统一可接受 R，结果对象哈希 `sha256:494d011a0fc86870af1d302492375e69e34b947ef92e4d3e20301335cba97ebe`。 | 500×R32 分层正式运行、19 个条件目标信息不足部件的覆盖改善和生产级自适应预算。 | 多目标缩放复跑完成；R32 支持概率目标和命中后命中数，不支持均匀精度的条件能量普通回归。 |
| E2 逐部件漏斗 | 破片审计按无几何交、提前阻挡、未穿透、穿透未损伤、响应可能分层。 | 16 工况×R2 冻结复跑完成：381,900 条轨迹中 98.383% 无几何相交；相交后 38.154% 被提前阻挡。 | 500 工况教师审计与正式漏斗报告。 | 冻结缩放完成，几何覆盖为首要收缩环节。 |
| E3 因子敲除 | 2^4 A/T/F/W + 离线 Tr、公共随机数、成对 bootstrap、主效应/交互已实现。 | 16×3、R1，48 case 缩放运行；离线树误差 0。 | 16×300、R4、预注册 bootstrap 与独立结论门。 | 缩放运行完成，不能解释效应。 |
| E4 尾部机理 | log 回归、来向/弹型固定效应、drop-one 增量 R²、分弹型 Hill 与截断诊断已实现。 | 5,479 complete case、41,262 正观测；增强模型加权 R²=0.76814，`ln(hits)` ΔR²=0.477556。矩阵秩 10/11、条件数 `3.16e14`，`mV0²` 与弹型固定效应混叠；三弹型 Hill 均不稳定。 | 在独立数据上复核；§6.8 弹型内因子设计用于解决混叠。 | 增强数值报告完成，稳定尾指数假设不成立。 |
| E5 噪声形状 | σ×τ 网格、normal/uniform/laplace 同方差生成和报告已实现。 | normal 9×3、R1，27 case 缩放运行。 | 默认 9×300、R4；若研究形状则 27×300、R4。 | 缩放运行完成，不能下机理结论。 |
| E6 树作用 | 离线树指标、region×profile 分层、零/中间/饱和区、二元熵和信息保留率入口已实现。 | 15 个实际单元已报告；支持向量代理叶联合熵 10.8394 bit、树输出联合熵 6.0560 bit，描述性比率 0.5587。工具接受最终逐重复 leaf state 与候选 view。 | 领域审定、精确叶向量候选对照和正式冻结。 | 分层数值报告完成，语义结论未完成。 |
| E7 匹配度 | `M_armor=log(v/V50)`、`M_comp=log(E/E50)` 与穿甲后精确可达性审计、总体和 region×profile 汇总已实现。 | 16 工况×R2 冻结复跑完成：73.069% 低于 V50，6.441% 低于 degraded、88.164% 低于 failed；`M_reach=57.981%`。 | 500 工况正式匹配审计。 | 冻结缩放完成，当前参数下几何/可达与 failed 裕量不足最突出。 |

Stage 2 缩放证据见 [stage2_representative_scaled/README.md](../idea2_optimization_20261120/stage2_representative_scaled/README.md)。该运行的 E3/E5 case 数为文档规模的 1%，replicate 计算量为 0.25%。

## X1–X3、T1、F1

| ID | 已实施内容 | 运行/权威状态 | 判定 |
|---|---|---|---|
| X1 暴露场 | QMC/代理暴露、可达状态、Hájek/ESS、足迹和教师反证审计实现。代理 non-reach 不被误称为结构零。 | 冻结复跑完成 100,000 候选与 150×R8；source/schema/engine 哈希前后一致。teacher false-unreached=12.845%（Wilson 上界 13.53%），超过 3%。 | 软件结构审计通过，科学晋升失败。 |
| X2 暴露均衡臂 | 固定样本量、正纳入概率、train-only、ESS/positivity 和 Reference 对照实现。 | 冻结 2,000 对 2,000：共同门通过，min π=0.01133 未达原 0.08 门；ESS 比例 0.753，≥20 支持部件由 39 增至 58。 | 覆盖改善明确，原预注册门未通过，Reference 保持主臂。 |
| X3 多保真 | 控制变量与四臂学习对照均已实现：teacher-only、代理特征、教师−代理残差、代理预训练后微调。 | 冻结控制变量留出方差缩减 5.09%、等效倍率 1.054；最终四臂均因 validation `p_hit` 可观测数为 0 而 `inadmissible`，代理预训练池 512 例因主验证门失败未进入选择。 | X1 失败且最终四臂不可选择；代理学习保持 diagnostic-only，不晋升。 |
| T1 树重设计 | 三类单调候选、预注册模板、独立 seed 选择、哈希与 dataset-v2 view 已实现。 | 数值定向测试通过；23 个未引用部件只有 provisional identity fallback。 | 外部语义权威缺失，G4 不可通过。 |
| F1 最终模型 | hurdle baseline、区间/二项/有序损失、target encoder、显式编码器 bundle 和 validation-only 消融入口已实现；test 强制保持 `open_count=0`。 | 最终报告为 `all_bundles_inadmissible`、`selected_bundle_id=null`；validation 无可观测 `p_hit`，四 bundle 指标均为 null；test decode/open 均为 0。 | 正确拒绝选模；确认性 test 不打开。 |

## 阶段门 G0 至 G5

| 门 | 已满足 | 未满足 | 结论 |
|---|---|---|---|
| G0 基线与诊断 | B0 18 次训练、E1 多目标缩放复跑、冻结 E0 与 E2/E7、E4/E6 增强报告完成；Reference validation；test 未读；输入表征、几何覆盖、稀有部件、条件能量方差、混叠和尾部不稳定问题已识别。 | E0/E1/E2/E7 正式规模与 E4/E6 独立数据确认未完成。 | 门未通过。 |
| G1 引擎语义 | E-3、E-4、模型卡、参数区间、D011–D015 已实现；离线树误差门、最终 ε/MNAR 审计和当前全仓 1,280 项回归完成。 | C1 仍缺 200×R32 E0 正式 TOST、形式参数角点与发布级收敛证据；最终数据 ε-完整率仅 0.006667。 | 门未通过；软件回归与 E-4 工程子门通过，性质规模和数据完整性子门失败。 |
| G2 机理实验 | E3/E5/C2 端到端缩放和 C3 的 12 次 Morris/28 次教师 Sobol 缩放运行完成；可信度基础设施存在。 | 正式规模、C3 跨种子稳定性、C4 专家断言、稳健角点与完整 C1–C4。 | 门未通过。 |
| G3 暴露场/采样器 | 球面几何 84/84、5D 降维 54/54、设计序列子门通过；冻结 X1 10^5、X2 2,000 对 2,000、X3 配对运行和哈希复核完成。 | X1 教师反证门与 X2 原门均失败；X3 收益小。 | 门未通过；Reference 保持主采样臂。 |
| G4 树冻结 | 候选族、哈希、预注册模板、独立验证执行器和 identity fallback 已实现。 | 领域语义审定、填写并冻结的预注册、独立 seed 正式选择/验证和 label spec 树哈希晋升。 | 门未通过，外部权威所需。 |
| G5 正式数据/训练 | 连续方向、Taylor–Shapiro、R32、hurdle v2、16 维合同、最终 writer、严格 dataset-v2、零解码封存、开发审计和 validation-only 学习报告均已完成。 | 240 个 train 小于 2,000 门，ε-完整率 0.008333，52/67 部件零正支持，validation `p_hit` 可观测数为 0；四 bundle 与四臂均不可选择，确认性 test 保持未开。 | 正式训练门失败；软件流程与受限探索可继续。 |

## 最终数据集结果

当前运行目录：`artifacts/datasets/idea2_taylor_shapiro_continuous_r32_20261120_v3/`

- 预标签冻结：360 个 formal case，`labels_observed_at_freeze=false`。
- writer 已按 SamplingSpec 1.7 / tensor contract 1.8 完成 360/360；23 shards、0 refused，split 为 240/60/60。
- 严格 dataset-v2、train/validation-only 开发审计、sealed-test 状态、validation-only 学习与分布指标均已形成。

| 项目 | 最终值 |
|---|---|
| 完整归档/哈希验证 | `complete；360/360；23 shards；0 refused；root manifest sha256:2ceef7e9…1de522` |
| 域完整率与识别区间宽度 | `开发集 ε-完整 2/300=0.006667；三个概率目标区间宽度中位数均为 1.0` |
| hurdle 目标可用率与正例支持 | `概率目标每部件正支持 min/median/max=0/0/2；52/67 零支持；条件目标观察率 0.134478` |
| dataset-v2 验证 | `通过；ID damage-dataset-v2-aa0fc301038df1c79ac21095；内部 manifest sha256:e0fa6c68…1178a` |
| G-8 学习曲线与三 seed 指标 | `预注册 25%/50%/100%×seed 17/29/43；validation p_hit 观测数=0，四 bundle 与四臂全部 inadmissible，未产生可比较指标或选择结果` |
| sealed Reference test 状态 | `60 例；test_result_json_lines_decoded=0；test_open_count=0；指标未计算` |
| 最终训练适用性结论 | `formal_neural_training_ready=false；可用于软件流程和受限探索，不适合确认性或研究级训练` |

当前数据集不得标为 `research_training`，G5 正式训练门不通过。
