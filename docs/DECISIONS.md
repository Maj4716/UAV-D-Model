# 架构与研究决策

本页是当前有效决策的单一入口。接口、数据角色、统计对象或研究范围发生变化时，在本页新增一条有日期和状态的记录；已被替代的决策保留并标记 `Superseded`。

## 决策总览

| ID | 决策 | 状态 | 日期 |
|---|---|---|---|
| D001 | 项目定位为数据方法合成生成器 | Accepted | 2026-11-17 |
| D002 | 采样状态机、归档写入与归档验证分层拥有 | Accepted | 2026-10-16 |
| D003 | 15 维 learner input 与 estimand 使用单一合同 | Accepted | 2026-10-16 |
| D004 | 旧格式只读兼容，当前 writer 只写当前格式 | Accepted | 2026-10-16 |
| D005 | 正式样本必须满足首次正向装甲接触 | Accepted | 2026-11-13 |
| D006 | Reference + Coverage 双臂采样 | Accepted | 2026-11-17 |
| D007 | 真实零、域删失与拒绝分离 | Accepted | 2026-11-17 |
| D008 | 毁伤等级采用平滑中间门与最终离散判定 | Accepted for development | 2026-11-17 |
| D009 | 机器证据按稳定用途目录保留 | Accepted | 2026-11-17 |
| D010 | 参数写回必须通过预注册晋升门 | Accepted | 2026-11-16 |
| D011 | OOD 标签采用已证明有效的严格最坏界 | Accepted | 2026-11-20 |
| D012 | 部件命中概率是跨重复数的主学习目标 | Accepted for development | 2026-11-20 |
| D013 | 物理状态轨迹与毁伤树派生视图解耦 | Accepted for development | 2026-11-20 |
| D014 | 球面入射 Sobol 设计进入 Reference 评价路线 | Accepted for evaluation | 2026-11-20 |
| D015 | 首层装甲状态进入版本化学习输入候选 | Accepted for evaluation | 2026-11-20 |
| D016 | Taylor–Shapiro 作为独立 v2 方向候选，不覆盖 legacy | Accepted for evaluation | 2026-11-20 |
| D017 | 连续来向采用球面测度，bottom-up 仅作压力测试 | Accepted for evaluation | 2026-11-20 |
| D018 | 最终试生成采用 R=32，test 在选模期间保持封存 | Accepted for pilot | 2026-11-20 |
| D019 | 分布适配只进入 train-only 可逆目标编码器 | Accepted for development | 2026-11-20 |
| D020 | 缺少树语义权威时使用 identity-component fallback | Accepted for development | 2026-11-20 |
| D021 | Taylor–Shapiro 偏转改为朝爆轰传播方向，v2/v3 符号仅供复现 | Accepted for evaluation | 2026-11-23 |
| D022 | 瞄准窗口采用固定参考轴切向基，动态基仅供复现 | Accepted for evaluation | 2026-11-23 |

## D001：项目定位为数据方法合成生成器

**背景**

当前工作的核心产物是可追溯的合成数据、采样方法、训练接口和验证证据。

**决策**

项目范围固定为：

- 软件教师及其确定性/随机计算链；
- 面向目标的接触条件采样；
- 数据归档、掩码、权重、切分和加载；
- 覆盖增强、稀疏性分析和神经网络衔接；
- 软件一致性、统计有效性和数据层验证。

发布产物使用 `development_ready` 等数据角色标识。目标、模型和参数的验证状态随 manifest 保留。

**后果**

README、`docs/`、`artifacts/` 和测试围绕这条主线组织。项目状态不再通过按日期累积的平行报告表达。

## D002：分层拥有采样与归档完整性

**背景**

采样 checkpoint、分片 writer、CLI 和归档验证曾交叉调用私有实现，扩大了循环依赖和篡改处理差异。

**决策**

- `sampling_checkpoint` 拥有 Reference/Adaptive 状态机、轮次 lineage 和归档绑定事务；
- `sharded_dataset` 拥有分片执行、恢复和发布；
- `archive_validation` 拥有单体/分片重开、manifest Schema、目录 inventory、分区哈希、张量合同和稳定错误分类；
- `input_identity` 定义冻结案例源和输入捕获接口；
- CLI 只调用公开采样接口；
- 验证成功返回不可变 `VerifiedArchive`，policy 绑定前检查 receipt freshness；
- writer 在首次提交与完成态恢复时均用公共 verifier 重开归档，再清除 incomplete marker。

**后果**

包内依赖图不允许 `sampling_checkpoint ↔ sharded_dataset` 强连通分量。writer、loader、policy 和采样入口对相同篡改给出一致的 `ArchiveIntegrityError` 类别与上下文。

## D003：learner input 与 estimand 使用单一合同

**背景**

模型输入定义和 ESS 实现曾分散在 writer、loader、training view 与统计工具中。

**决策**

- `learning_input_identity.ModelInputDefinition` 是 learner-visible 状态的版本化单一来源；
- 当前接触姿态合同固定为 3 维弹型 one-hot、3 维起点、3 维速度和 6 维旋转，共 15 维；
- 教师中间量、求解器状态、目标拓扑与装甲几何留在审计归档，不进入默认 learner view；
- `estimands` 集中实现有限正设计权重 ESS、允许零质量的 soft-event ESS、稳定 Hájek/向量加权均值和 streaming moments；
- task weight 与 minibatch probability 独立于设计 estimand；
- 权重计算先按有限正最大值缩放，避免直接累加溢出。

**后果**

新增 learner input 必须同时定义名称、单位、来源、变换、张量、mask 与 manifest 语义。默认字段变化属于版本化合同变更。

## D004：兼容读取与生命周期

**背景**

仓库需要读取已有归档，同时保持当前写入合同清晰。

**决策**

- 旧 manifest、归一化、身份和 ledger reader 仅作只读兼容；
- 当前 writer 只生成声明版本的当前格式；
- 兼容符号按适配器或符号退役，提供迁移路径、测试和至少一个观察周期；
- `generate --record-refusal` 保留为带运行期警告的 deprecated no-op，下一次 breaking release 删除；
- `ordinal_projection` 保持显式 opt-in；
- `multitask_targets` 保持活动外部训练接口；
- 模板使用文件化单一来源；
- `schemas/` 与包内 Schema 镜像在外部消费者完成清点前继续保留。

**后果**

兼容代码不通过整模块注释处理。所有退役都有公开使用量或仓库调用证据，并在 breaking release 窗口执行。

## D005：正式样本满足首次正向装甲接触

**背景**

位置、速度和姿态独立采样会产生未命中、反向飞行和持久化接触字段不一致。

**决策**

正式候选由目标朝向射线流形联合生成轨迹起点、速度、瞄点、standoff 和 roll。教师执行前必须复算首次正向装甲交点。未命中、反向、零速度、目标内部起点和接触字段漂移进入拒绝账本。

**后果**

所有训练样本都满足面向装甲车的运动与姿态合同。拒绝记录不进入响应张量，也不编码成无毁伤。

## D006：Reference + Coverage 双臂采样

**背景**

均匀参考臂提供可解释的目标设计分布，覆盖增强臂提高有限预算下的部件和过渡区支持。V5 教师比较降低了条件树饱和，但预注册总门为 `overall_pass=false`。

**决策**

- 每轮保留等预算 Reference 和 V5 Coverage；
- 两臂在“来向 × 弹型 × U 带”内分层；
- V5 使用全局边际覆盖与 `+0.8 q_domain - 0.8 q_sat` 软排序；
- 所有成员关系、一阶纳入概率和原始设计权重保留；
- Reference 用于参考分布估计，Coverage 用于训练增强和分层诊断。

**后果**

采样器评价同时报告总体、域完整率和完整域内条件指标。发生率报告不得把 Coverage 的原始频率当作 Reference 频率。

## D007：真实零、域删失与拒绝分离

**背景**

把域删失或输入拒绝填为零会制造假负例，并改变部件覆盖、树饱和、采样排序和模型校准。

**决策**

- 完整域内数值 0 表示真实零响应；
- 域删失保存 `response_mask=false`，响应张量为 `NaN`；
- 输入或接触拒绝只进入拒绝/选择账本；
- 任务级部分可用性需要显式 task mask；
- 域头对全部候选训练，响应头只在可用 mask 内训练。

**后果**

writer、validator、loader 和 loss 共同检查 mask—有限性一致性。`nan_to_num` 不得用于构造响应负例。

## D008：平滑中间门与最终离散判定

**背景**

全硬逻辑树丢失阈值附近的信息；全平滑输出又不能直接承担稳定的等级判定。项目需要针对当前任务优化可学习性，同时保留清楚的毁伤等级语义。

**决策**

- 叶响应先转换为有界连续激活；
- 中间门可使用 sigmoid、softplus/softmin、平滑 k-of-n 或可解释的加权组合；
- 最终一级使用冻结阈值输出等级 0/1/2；
- 等级事件满足 `level-2 ⊆ level-1`；
- 神经网络输出累计概率时满足 `q2 <= q1`；
- 每个类别分别报告零质量、过渡占比、等级分布、条件熵和端点饱和；
- 阈值与温度通过预注册开发目标选择，并在独立设计种子上验证。

**后果**

树可针对本项目的可学习标签分布进行特化，但版本、部件语义、门函数、阈值和选择准则必须随数据包冻结。树修改触发标签版本和数据集身份变化。

## D009：稳定机器证据目录

**背景**

临时候选池、运行日志和按日期报告数量过多，增加了引用漂移与状态误判。

**决策**

- 最终开发数据位于 `artifacts/development_dataset/`；
- 采样稳定证据位于 `artifacts/validation/sampling/`；
- 引擎稳定证据位于 `artifacts/validation/engine/`；
- 穿透诊断位于 `artifacts/validation/penetration/`；
- `artifacts/refactor_20261117/` 保存本次重构前基线、清理清单与重构后验收结果；
- 临时池、缓存、图片副本和可再生中间结果不作为文档引用目标；
- 当前结论写入 [研究发现](RESEARCH_FINDINGS.md)，运行方法写入 [验证](VALIDATION.md)。

**后果**

文档只引用稳定用途路径，不引用编号报告。删除中间池前，先独立验证自包含数据包并保存内容哈希与 lineage 状态。

## D010：参数晋升与写回

**背景**

诊断排名可以找到候选领导者，但排名不等于满足预注册科学门。

**决策**

参数候选只有在以下条件同时满足时才能写回注册表：

- 冻结的候选空间、指标、门槛和选择时间点；
- 单点与多工况完成状态有效；
- 侵彻影响率、条件正响应、饱和、能量闭合和稳健性门通过；
- 来源在运行期间稳定；
- 选择结果包含非空 `selected_candidate_id`；
- 独立验证记录绑定候选、代码、输入和注册表哈希。

**后果**

当前单点与多工况探索的 `selected_candidate_id=null`，注册表保持原参数。诊断领导者只用于下一轮假设和实验设计。

## D011：OOD 标签使用严格最坏界

**背景**

超出当前部件或材料适用域的重复需要给出可审计的不确定性范围。把 OOD 部件分别解释为“终止传播”和“透明且不沉积能量”，无法形成全部系统输出的全局上下界；装甲开孔、后续传播和时序耦合会破坏所需的单调性。

**决策**

- 保留当前对受影响重复使用的严格最坏情况界；
- 只有在某个标签的传播链和门函数具备可证明单调性时，才增加该标签专用的收紧界；
- 每个界记录受影响重复、适用标签、假设和证明范围；
- 缺少单调性依据的双分支结果只作情景分析，不标记为上下界。

**后果**

OOD 不确定性仍可进入训练 mask、稳健损失和评价报告。新增收紧方法需要标签级性质测试，且不得改变旧数据集身份。

## D012：跨重复数的概率主目标

**背景**

固定重复数下的二值 support 会把 Monte Carlo 分辨率写进标签语义。同一底层命中概率在不同重复数下可能产生不同二值结果，使模型学习样本预算而非物理教师的概率对象。

**决策**

- 主目标定义为部件命中概率 `p_hit`，由重复级 `hit_indicator` 的样本均值估计；
- 同时保存重复数、样本标准差、标准误、95% 区间和不确定性 mask；
- 二值 support 是版本化派生视图，使用固定概率阈值和区间判定；区间跨阈值时标记 ambiguous 并关闭训练 mask；
- 域删失保留数值证据，响应训练 mask 为 false；
- 命中后能量和命中数只在提供显式条件充分统计时生成，不由无条件总量代替。

**后果**

R=4、R=16 等预算共享同一估计对象。实现入口为 `damage_engine/hurdle_targets.py`，批量适配器为 `tools/extract_hurdle_targets.py`；阈值、区间规则和目标版本随数据包冻结。

## D013：状态轨迹与树派生解耦

**背景**

物理教师状态、部件机制判定和系统毁伤树具有不同的演化频率。把树输出固化为物理核的唯一标签会放大语义修改成本，也会混淆逐重复非线性评估与在期望概率上评树这两个不同统计对象。

**决策**

- 软件教师可选保存重复级部件状态轨迹、域状态、参数身份和来源身份；
- 毁伤树作为可离线重算的版本化派生视图；
- 权威树目标对每个重复评估冻结树，再对重复结果取均值；
- 在期望部件概率上评树只作辅助诊断，并使用独立字段和 estimand 名称；
- 缺少领域授权的功能树语义时，训练和评价优先使用部件身份任务。

**后果**

树规则迭代无需重跑物理教师，且两类树统计量不会混用。离线入口为 `tools/recompute_tree_outputs.py`，重算结果必须与在线逐重复输出满足数值一致性门。

## D014：球面入射 Sobol Reference 评价路线

**背景**

现有 V5 Coverage 通过代理门，但独立软件教师总门为 `overall_pass=false`。球面入射参数化已通过几何与降维检查，需要用独立配置评价空间填充序列，并保持旧冻结配置和哈希不变。

**决策**

- 新球面入射 Reference 配置显式使用 scrambled Sobol；
- 非 2 的幂样本量由下一幂次 Sobol 设计取确定性前缀，并完整记录填充量、前缀和序列身份；
- Reference 继续承担参考分布基线，V5 不晋升为新基线；
- `bottom_up` 保留为压力测试方向，不进入当前主 Reference 设计；
- 连续方向采样在来向分布、测度和权重合同冻结后进入后续版本。

**后果**

旧 randomized-LHS 配置及六来向 Sobol 比较配置仍按原哈希复现。新路线使用独立配置、设计种子和数据身份；当前主评价入口为 `configs/sampling/p0_sphere_entry_sobol_no_bottom_impact_state.v1.json`，状态为 `provisional`，通过同预算评价门后才能晋升。

## D015：首层装甲状态学习输入候选

**背景**

仅使用来向代理量会遗漏实际瞄点、姿态和目标几何共同决定的首次装甲接触状态。首层装甲交点、法向和路径状态在教师执行前可确定，能够形成不读取响应的几何输入。

**决策**

- 新球面数据路线优先评价版本化的首层装甲状态输入；
- 输入只包含标签计算前可确定的几何与运动状态，并通过来源和泄漏检查；
- 来向代理量保留为可选消融，不与首层装甲状态混成未声明合同；
- 原 15 维 learner input 保持旧身份；新增字段使用新的模型输入定义、维度和 manifest 身份；
- 树语义缺少领域授权时，首层装甲状态先服务于部件概率和严重度任务。

**后果**

该候选使用 16 维合同 `loitering_impact_first_armor_state@1.0.0`，需要与旧 15 维输入在同一 Reference 验证集、相同训练预算和多随机种子下比较。通过泄漏、可学习性和稳定性门后，才能成为新数据包默认输入。

## D016：Taylor–Shapiro 使用独立 v2 身份

**决策**

- legacy `taylor_ring` 的配置、固定种子输出和数据身份保持不变；
- 新路径使用 `taylor_shapiro_ring`、独立 physics registry、case revision 和模型卡；
- v2 将 Shapiro 有符号偏转限制在 Taylor 幅值界内，仅支持已声明的圆柱内壁、前端轴上点起爆几何；
- 新模型的文献可溯源性不等于已完成弹药几何、色散、材料和目标响应标定。

**后果**

v2 可用于敏感性研究和新数据 pilot，不得用它重解释旧数据。模型说明和适用界见 `docs/FRAGMENT_DIRECTION_MODEL.md`。

## D017：连续来向与 bottom-up 分流

**决策**

- 水平来向在方位±30°、俯冲0–45°内按声明的球面测度采样；顶攻在偏离竖直0–45°内按均匀立体角采样；
- 切向基随实际方向重算，记录量度和采样身份；
- `bottom_up` 不进入主 Reference，仅进入独立 stress 语义。

**后果**

SamplingSpec 1.7 是新评价路线；旧离散来向仍用于复现。Reference 的设计权重不解释为作战方位的真实流行率。

## D018：R=32 的 pilot 预算与封存 test

**决策**

- 用 32 个非 test 工况、每工况 R=32 的 E1 方差分解选择本次试生成预算；
- 预注册 pilot 门为“67 部件中最坏 `p_hit` 均值估计器 RMSE≤0.05”；实测最坏值约 0.045，R=16 外推约 0.064；
- train/validation/test 在标签观察前以独立设计种子冻结；模型、阈值和训练规则冻结前不读 test 指标。

**后果**

R=32 是本次 360 例 pilot 的临时决策，不是对全域或所有条件正值目标的充分性证明。正式数据需扩大 E1 并单独预注册自适应重复分配。

## D019：分布适配不改写标签

**决策**

- identity、log1p、Box–Cox 和经验分位数编码器只允许在 train 拟合；
- 变换参数与模型绑定并可逆，所有发布评价回到原始标签尺度；
- 编码器不进入物理标签身份，不按数据分布改写毁伤语义。

## D020：未审定树语义时使用 identity fallback

**决策**

- 23 个现有树未引用部件仅生成部件—机理—状态 identity 输出，不补造功能分组或系统效应；
- noisy-OR、k-of-n 和 Choquet 候选可作数值诊断，只有独立 seed 门通过且具名语义权威批准后才能冻结；
- 缺少语义权威时，阶段 5 可先训练部件层，不将候选树写成权威标签。

## D021：Taylor–Shapiro 偏转朝爆轰传播方向

**背景**

审查（`artifacts/validation/pipeline_expected_actual_20261122/REVIEW_20261123.md` §4.1）发现，v2/v3 把 Shapiro 角施加到极角增大方向后，前端起爆的破片全部偏向起爆端（轴向 +0.5° 至 +7.6°）。Taylor（1941）的稳态流构造要求破片偏向爆轰波传播方向，前端起爆应向后偏转。原测试把错误方向写成了断言，微观审计又复用了同一函数，所以一直没有被发现。

**决策**

- 新增 `angle_convention: body_plus_x_polar_deflection_toward_detonation_propagation`，偏转角取 `atan(q·sin(φ₂−φ₁))`，幅值和 Taylor 上界不变，符号与 v2/v3 相反；
- 新物理注册表 `loitering_impact.taylor_shapiro.wave_directed.obliquity_bound.v4.json` 与案例 v4 只改这一处，其余与 v3 相同；
- v2/v3 的 `body_plus_x_polar_from_nose_local_normal_signed` 原样保留，仅用于复现已归档数据；
- 方向测试必须包含与引擎代码无关的独立判据（Taylor 实验室系速度合成、“偏离法线方向与爆轰切向分量同号”）。

**后果**

基于 v3 的数据集（含 `recovery_role_r8_r32_train2000_eval300_20261122`）的部件级命中格局需要用 v4 重建后才能作为正式结论。v3 归档仍可逐位复现：3 个 train 工况的教师结果重跑后完全相等。模型卡升为 `fragment-direction-model@3.0.0`。

## D022：瞄准窗口固定参考轴切向基

**背景**

SamplingSpec 1.7 的动态切向基取“与来向最不对齐的坐标轴”投影为 u，v=direction×u。v 的朝向随来向翻转，同一个 `aim_v_range_m=[0.75,1.7]` 对一半扇区落在目标中心上方，对另一半落在下方：瞄准带在上方时 S-CONTACT 准入 94.8%，下方时 26.1%；x 向近似镜像对称的目标，pos_x 与 neg_x 准入率为 77.4% 与 17.7%，正式集 region TV 达 0.2015。

**决策**

- 新增 `contact_manifold.aim_tangent_basis: target_up_horizontal_target_forward_top@1.0.0`：水平扇区以目标 +z、顶攻以目标 +y 在垂直来向平面上的投影为 v，u=v×direction；
- 四个正方向与正下方顶攻时，该基精确还原 `contact_pose.smoke.v1` 中显式声明的坐标框架；
- 新基使用独立的 measure contract（`continuous-direction-measure@1.1.0`）和实现哈希；未声明该字段的 SamplingSpec 维持原动态基，归档哈希不变；
- 新 SamplingSpec `p0_sphere_entry_continuous_direction_recovery_train2000_eval300_role_r8_r32.v2.json` 使用新基和新设计种子。

**后果**

同预算下 S-CONTACT 总准入率从 60.5% 升到 98.4%，四个水平扇区均为 100%，正式集 region TV 从 0.2015 降到 0.0127。旧计划重建后 2,600 个工况与归档逐位一致（存储字段除外）。D017“切向基随实际方向重算”的表述由本决策细化：方向本身仍逐例重算，切向基改为固定参考轴投影。

## 修改决策的流程

1. 在本页新增候选决策并标记 `Proposed`；
2. 写明受影响的 Schema、模块、数据身份、统计对象和迁移路径；
3. 增加或修改测试与验证计划；
4. 用新设计种子或独立数据验证；
5. 接受后更新状态、日期和相关稳定文档；
6. 被替代的决策标记 `Superseded by Dxxx`，不删除历史语义。
