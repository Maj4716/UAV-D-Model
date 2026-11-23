# idea2 实施工作日志

日期：2026-11-20  
范围：从前半部分基线到第 9 部分纠正后的继续实现。本文只记录已完成动作和已观察证据；最终 R=32 writer、严格开发包、开发审计、学习评估与全仓软件质量检查均已完成。

## 1. 基线冻结与可行性判断

1. 复核 `idea2.md` 的问题、优先级、实验登记和 G0–G5 门。
2. 决定先做开发数据基线，再改标签、树和采样，避免在未知瓶颈上调整物理参数。
3. 冻结 Phase 0 配置：同一两层共享 MLP、域/部件 support/严重度/树四头、2 个输入视图、25%/50%/100% 三种训练比例、3 个随机种子、Reference validation 加权评价、禁止访问 test。
4. 完成 18 次 Phase 0 训练。完整结果在 `artifacts/experiments/phase0_20261120/results.json`，紧凑摘要在 [phase0/README.md](../phase0/README.md)。

观察结论：`x_design` 全量训练的部件宏 PR-AUC 为 0.740，`x_normalized` 为 0.427；主要瓶颈先是物理输入表征，再是 17–22 个 validation 正例的稀有部件。`x_design` 是诊断上界，不作为生产合同。

## 2. 前半部分：P0 语义与身份修复

### 2.1 E-4 物理状态与树解耦

- 在引擎中加入轻量 `capture_component_state_trace`；轨迹包含逐重复部件机制状态、在线树输出、域状态、参数哈希和来源身份。
- 新增 `tools/recompute_tree_outputs.py`，按树定义哈希离线生成派生视图。
- 当前树在线逐重复评估后取均值，与离线重算的最大误差为 `1.11e-16`；Stage 2 缩放运行的误差为 0。
- 实现 dataset-v2 core/view 分层、core 哈希验证和树视图重算验证。

### 2.2 E-3 R 无关标签

- 新增 `damage_engine/hurdle_targets.py`，以 `p_hit`、有序状态概率和显式命中后条件量作为估计目标。
- 保存试验数、成功数、样本方差、标准误、95% CI 和域监督状态。
- 二值 support 固定为派生视图；CI 跨阈值时标记 ambiguous，不把 Monte Carlo 分辨率写进主标签。
- 新增 hurdle v2 label spec、dataset-v2 字段、二项/区间/异方差损失和有序状态参数化。
- R=4/R=16 单元级语义一致性验证通过；200 工况正式等效性实验仍未运行。

### 2.3 E-2 方向模型身份

- 写入 [破片方向模型卡](../../../docs/FRAGMENT_DIRECTION_MODEL.md)，把 legacy 空心环、实心锥和 Taylor–Shapiro 分开管理。
- Taylor–Shapiro 使用独立 physics/case ID 和测试，不覆盖旧冻结注册表。
- Stage 2 C2 缩放对照确认三弹型 Gurney 金标为浮点级一致，并检查 legacy 角度与 V50 中点；外部物理确认仍缺。

## 3. 第 9 部分纠正

### 3.1 撤回全局 L/U 假设

复查后确认，“OOD 破片停止”和“无损透明传播”没有形成全部标签全局上下界所需的单调性。装甲开孔、后续传播和时序会改变可达集合。

执行结果：

- 未实现伪全局 L/U；
- 保留受影响重复比例构造的严格最坏识别区间；
- 区分 Monte Carlo CI、删失识别区间和参数/模型情景区间；
- 增加区间损失与 ε-完整监督策略；
- D011 明确要求标签级单调性证明后才能收紧区间。

### 3.2 替换 KL

新增 `damage_engine/distribution_metrics.py`：

- 概率部分：Brier 与校准；
- 正值部分：`log1p` Wasserstein-1 与分位数；
- 多变量：RBF-MMD；
- 分箱单值描述：Jensen–Shannon。

这些指标已有性质测试；最终 train/validation-only 报告覆盖 300 例，但只有 134 个可识别工况—部件对和 2 个完整工况。命中概率 Brier/ECE/JS 为 0.021932/0.035726/0.031280，Pearson/Spearman 为 0.614724/0.486303；正能量仅 6 对，log1p W1=1.104171；MMD²=0.797647。15 个 region×profile 分层中仅 `pos_x::small_front` 可估，不能外推到其余 14 个分层。test 行解码数为 0。

### 3.3 把“适应分布”放入目标编码器

新增 `damage_engine/target_encoding.py`：identity、log1p、Box-Cox 和训练集分位数编码均可逆；只允许在 train 拟合，合同哈希漂移会拒绝，评价回到原始标签尺度。hurdle v2 明确 `label_mutation=false`。

目标编码器已接入 hurdle baseline，并提供 identity、Box–Cox、经验分位数正态及默认混合 bundle 的同预算 validation-only 消融。最终四 bundle 均因 validation `p_hit` 可观测数为 0 而 `inadmissible`，`selected_bundle_id=null`；train/validation/test 的 `p_hit` 观测数为 134/0/0，条件能量和命中数为 2,089/614/0。报告没有用条件目标替代预注册选择指标，test decode/open 均为 0。

## 4. 采样、输入与数据格式

### 4.1 球面入口与设计序列

- 球面重锚 G1：30 个几何对与 54 个教师对，共 84/84 通过。
- 6D→5D 降维：54/54 对教师结果一致。
- 同预算设计序列比较推荐 scrambled Sobol；旧配置和哈希保持不变。
- 新 16 维首次装甲状态合同由 profile one-hot、命中点、表面法向、单位撞击方向、速度和 body-y 组成；全部字段在标签计算前确定。

证据见 [sampling_sphere_v1/IMPLEMENTATION_LOG.md](../sampling_sphere_v1/IMPLEMENTATION_LOG.md)。

### 4.2 连续方向

- 实现 SamplingSpec 1.7 的连续方向潜变量、动态切向基和球面测度。
- 主 Reference 排除 bottom-up；压力测试语义保留。
- 新 Taylor–Shapiro 连续方向 R32 计划已在标签观察前冻结，360 个 formal case 来自 630 个候选。
- SamplingSpec 1.7 已贯通 required/optional/fixed 字段、验证、model input、writer 与 manifest；18 个案例×R2 的端到端小样确认 manifest 1.7、tensor contract 1.8 和 16 维输入。

### 4.3 dataset-v2 与 sealed-test 纪律

- 实现 core inputs/trace/labels/weights 与可追加 `tree_<hash>` view。
- sampler 转换器在缺少 component trace 时 fail closed。
- 新增严格开发包转换器 `tools/package_sampler_dataset_v2_development.py`：test 分区只保留输入和身份字段，真实标签与 trace 在 `json.loads` 前跳过，并以全掩码占位满足训练接口。故意写入非法 test JSON 的回归仍能完成打包、验证和 baseline，审计字段为 `test_result_json_lines_decoded=0`；9 项定向测试通过。
- 新增 `tools/audit_dataset_v2_development.py`：只审计 train/validation，输出 ε=0.01 完整率、识别区间宽度、region×profile 分层和 Holm 校正 KS 非随机缺失检验；非法 test JSON 的定向回归证明审计过程同样零解码，2 项定向测试通过。
- baseline 强制 validation-only；旧 360 pilot 的 test 已读取，因此固定为 diagnostic-only，不能作确认性 test。
- 最终 R32 计划使用独立 test 身份；在模型、阈值和训练规则冻结前不得读取其标签指标。

## 5. 可信度、机理和树基础设施

### 5.1 C1–C6

- 建立参数情景区间、可信度计划、内容哈希档案和 10 项领域断言空白模板。
- 新增厚度/速度/质量解析单调、部件阈值、零易损比和单调门性质测试。
- 当前 [credibility_dossier.json](credibility_dossier.json) 的研究准入为 false；C4 等待具名领域权威，C1/C2/C3/C5/C6 仍有公开缺口。
- `component-057` 因缺少坐标和姿态保持在目标之外，不补造几何；例外记录见 [source_import_exception_component_057.json](source_import_exception_component_057.json)。

### 5.2 Stage 2 缩放运行

完整日志见 [stage2_representative_scaled/README.md](../idea2_optimization_20261120/stage2_representative_scaled/README.md)。

| 工作项 | 文档规模 | 本轮规模 | 结果用途 |
|---|---:|---:|---|
| E3 | 16×300、R4 | 16×3、R1 | 验证公共随机数、因子编码、trace、离线树和 bootstrap 链；不解释效应 |
| E5 | 9×300、R4 | 9×3、R1 | 验证 σ×τ normal 网格；不判断长尾来源 |
| C2 方向 | 256 repeats | 16 repeats | 验证解析金标连接 |
| C2 V50 | 4096/速度 | 512/速度 | 验证随机实现靠近解析 0.5 |

E3/E5 case 数为正式设计的 1%，replicate 计算量为 0.25%。正式机理门保持未通过。

### 5.3 C3、X1–X3 与 T1

- C3：Morris 代理筛选、教师 Sobol、公共随机数和参数绑定代码完成；已执行 12 次 Morris 代理评估和 28 次单工况 R1 教师 Sobol。结果哈希为 `sha256:ff893547f77d6fa41fba2f86c4f4ecff6dc3c59d52137c05fc86db6cb6f4caee`；当前排序用于探索，正式预算、跨种子稳定性与角点冻结仍待完成。
- X1–X3 的首轮漂移结果只保留为工程诊断；冻结复跑已在 source/schema/engine 哈希前后一致的条件下完成，证据见 [冻结运行状态](../../experiments/X1_X2_20261120_frozen_v2/FROZEN_EVIDENCE_STATUS.json) 和 [运行日志](../../experiments/X1_X2_20261120_frozen_v2/RUN_LOG.md)。
- X1：冻结运行完成 100,000 个候选和 150×R8 教师审计。教师 false-unreached 为 12.845%，Wilson 上界为 13.53%，高于 3% 预注册门，暴露场未晋升。
- X2：冻结 2,000 对 2,000 完成；共同工程门通过，最小纳入概率 0.01133 未达到原 0.08 门，ESS 比例为 0.753；至少 20 个支持样本的部件数由 39 增至 58。覆盖改善没有改变晋升结论。
- X3：冻结配对控制变量的独立 holdout 方差缩减为 5.09%，教师样本等效倍率为 1.054，收益不足以支持晋升。
- `tools/run_proxy_learning_ablation.py` 已实现 teacher-only、67 维派生代理特征、教师−代理残差、代理预训练后教师微调四臂同预算对照。最终四臂均因 validation `p_hit` 可观测数为 0 而 `inadmissible`，`selected_arm=null`；512 例代理预训练池在主验证门失败后未进入选择。X1 仍为 failed，四臂保持 diagnostic-only；代理量不写入冻结 16 维物理输入 core。
- 最终试生成继续使用 Reference；冻结 X1–X3 已结束，不再列为待收口项。
- T1：weighted noisy-OR、k-of-n、2-additive Choquet、identity fallback、独立 seed 选择和预注册模板完成；37 项定向测试通过。23 个未引用部件缺少领域语义，任何候选都未获推广资格。

### 5.4 E0、E1、E2/E7 与 E4/E6 增强诊断

- E0 已完成 8 个基准工况×R8 冻结缩放复跑。距离 22/22 项等效且差异为 0；滚转仅 9/33 项等效。因此距离移出默认输入，滚转保留。结果哈希为 `sha256:82a5df98ff7f90e43397eb960994523c854b7a5c58d3c619d6a114983dfdedd6`，详见 [E0_ANALYSIS.md](E0_ANALYSIS.md)。
- E1 修正版多目标复跑使用 32 个非 test Reference 工况、每工况 R32。`p_hit/P(degraded)/P(failed)` 最大 RMSE 分别为 0.04503/0.04292/0.01911，对应建议 R32/R32/R8；命中后命中数的 48 个可估部件在 R32 全部通过 0.10 容差。命中后能量最大 RMSE 为 1.011，19 个部件信息不足，3 个部件在候选 R1024 内仍不达 0.10。32/32 工况受穿透适用域影响，受影响重复比例均值为 0.921875；MC 噪声与删失识别区间分别报告。结果对象哈希为 `sha256:494d011a0fc86870af1d302492375e69e34b947ef92e4d3e20301335cba97ebe`，详见 [E1_MULTITARGET_ANALYSIS.md](E1_MULTITARGET_ANALYSIS.md)。
- E2/E7 已完成 16 工况×R2 冻结审计，共 381,900 条碎片—部件轨迹：98.383% 无几何相交；相交轨迹中 38.154% 被提前阻挡；`M_armor` 有 73.069% 低于 V50；`M_comp` 有 6.441% 低于 degraded、88.164% 低于 failed；`M_reach=57.981%`。详见 [E2_E7_ANALYSIS.md](E2_E7_ANALYSIS.md)。
- E4 增强诊断使用 5,479 个 complete case 和 41,262 个正观测，加权全模型 R²=0.76814。drop-one ΔR² 中 `ln(hits)=0.477556`、来向固定效应 `0.004877`、`ln(1/r²)=3.42e-6`、弹型固定效应 `2.71e-7`、`ln(mV0²)=0`。设计矩阵秩为 10/11、条件数为 `3.16e14`，`mV0²` 与弹型固定效应混叠；三弹型 Hill 范围分别为 1.192–53.903、1.093–29.692、0.897–24.739，稳定尾指数假设不成立。
- E6 增强诊断覆盖 15 个实际单元；支持向量代理叶联合熵为 10.8394 bit，树输出联合熵为 6.0560 bit，描述性保留/压缩比为 0.5587。精确叶向量与候选 view 的比较仍需最终逐重复 trace。E4/E6 证据与 5 项定向测试见 [增强报告](../../experiments/E4_E6_enhanced_20261120/README.md)。

### 5.5 §6.8 战斗部机理因子设计

- 新增 `V0/m/N/θ/σ` 因子到物理配置的显式映射、质量/几何可行性检查、拒绝台账和 VIF 门。
- 32 点 scrambled Sobol 设计的最大绝对相关系数为 0.190716，最大 VIF 为 1.048105；31/32 点可行，1 点因爆速越界拒绝。
- 31 个可行点均以同一非 test 工况、R1 执行，结果哈希为 `sha256:4b08c678ed80e497e34208e74798ddb6636885ad49c630dfa6e161d773f56c24`；5 项定向测试通过。该运行解决设计可辨识性和执行链问题，正式效应结论仍需扩大重复与独立工况。详见 [因子设计报告](../../experiments/warhead_factor_design_20261120/README.md)。

## 6. 验证记录与证据边界

- Phase 0 泄漏检查通过，test 未访问。
- 球面几何、5D 降维和设计序列子门通过。
- 树候选/dataset-v2 定向测试记录为 37 passed。
- Taylor–Shapiro/C1 定向性质测试记录为 14 passed；SamplingSpec 1.7 相关定向测试约 50 passed；编码器消融定向测试记录为 10 passed；代理—教师报告器测试记录为 4 passed。
- 严格开发包、开发审计、四臂代理学习、§6.8 因子设计和 E4/E6 增强诊断的定向测试分别记录为 9、2、5、5、5 passed。
- 最终学习评估收口后，hurdle/分布/四臂相关 focused suite 为 15 passed；三项最终报告均保持 test decode/open=0。
- 最终工作树全量 pytest exit 0：独立 collect 为 1,280 tests / 129 files，执行结果为 1,280 passed。`compileall`、`pip check` 和 `git diff --check` 均 exit 0；5 组根目录/包内 Schema 镜像 0 mismatch；无 writer lock、incomplete marker 或临时 shard。详见 [FINAL_QA_20261120.md](FINAL_QA_20261120.md)。
- 软件测试证明实现符合声明的合同；方向模型、材料参数、装甲厚度、响应阈值和功能树语义仍需要外部证据。

## 7. 最终数据集运行与结果

目录：`artifacts/datasets/idea2_taylor_shapiro_continuous_r32_20261120_v3/`

writer 完成证据：

- round manifest：`labels_observed_at_freeze=false`；
- formal case：360；
- writer 在冻结引擎哈希 `sha256:33e6a78619b5e9028e7b3d4a7217ebb7e230f325f11da365e479665f519b59ad` 下完成 360/360；
- checkpoint 状态为 `complete`，共 23 shards、0 refused；split 为 train 240、validation 60、sealed test 60；
- dataset ID 为 `damage-dataset-99e74282d7d911ef4287476b`；根 manifest 哈希为 `sha256:2ceef7e95643f91343b97ff191a326d7f30d7e63b0edc39bb8e14b9c1b1de522`；
- writer 在 352/360 时因宿主进程结束留下陈旧锁。恢复前确认锁中 PID 已不存在，随后使用原冻结参数和 checkpoint resume 完成最后 8 例；既有 22 个已提交 shards 保持不变。
- 严格开发包首次转换在合法的条件目标 `sample_count=0` 上 fail closed。修正按 `mask=false` 表示条件量不可用，不补造标签；定向回归与重新打包通过，test 行仍在 JSON 解码前跳过。

后处理结果：

- 归档完整性：`complete`；360/360、23 shards、0 refused；manifest 哈希见上；
- 域完整性：ε=0.01 时开发集 2/300（0.006667），train 2/240（0.008333），validation 0/60；受影响重复比例均值 0.910417、中位数 1.0；三个概率目标识别区间宽度中位数均为 1.0；writer 0 refused；
- hurdle 支持：`p_hit/p_degraded/p_failed` 的每部件正支持最小/中位/最大均为 0/0/2，52/67 部件零正支持，67/67 均低于 20；条件能量与条件命中数观察率均为 0.134478；
- dataset-v2：ID `damage-dataset-v2-aa0fc301038df1c79ac21095`，内部 manifest hash `sha256:e0fa6c68af2b96ac7f74086139d30292274240a296bf9aa11ba9ee0c3231178a`；结构和哈希验证通过，23 个源分片与源 manifest 全部匹配；开发审计 hash `sha256:7352d000a92fba1a29c43f8f233dc971e1823f619f1af042ea4d95011a6b19ba`；
- validation-only 评估：预注册 25%/50%/100%×seed 17/29/43；四个编码器 bundle 与四个代理学习臂均因 validation `p_hit` 可观测数为 0 而不可选择，所有选择指标为 null；hurdle 报告 hash `sha256:620d289c0dfa76907d2b38cdbb29fabdfa57fdad615a81210d83986fa8759c1a`，四臂报告 hash `sha256:48046246a20d1d50fc043623953ff369fa36f55af32e840535891ee3c3d9ce88`；
- sealed test 保持 `test_result_json_lines_decoded=0`、`test_open_count=0`，60 例真标签与 trace 不进入开发包；当前 `formal_neural_training_ready=false`。

G5 的正式训练门已判定失败；学习消融只用于验证流程和描述在当前稀疏/删失数据上的行为。
