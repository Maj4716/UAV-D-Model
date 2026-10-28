# 毁伤树有效性与 KMFC 专用软件原型审计

**审计日期：** 2026-10-22  
**审计范围：** 旧七输出映射、KMFC 两级有序合同、零锚平滑层、逐重复硬门、availability/mask、能量语义、深度学习接口  
**结论等级：** `project_specific / software_only / prototype_not_admitted`  
**物理有效性：** `not_established`  
**科研训练准入：** `false`

## 1. 核心结论

1. **旧七输出不能原地改名为新的 KMFC 八个累计输出。** 旧定义缺少 `effect-f2`；K1/K2 和 M1/M2 不共享同一组件事件集合；M 在 5000 个归档案例中有 1346 个严格 `M2 > M1` 违例。C1/C2 虽然使用相同人员组件的 degraded/failed 状态，只有结构兼容性，尚没有经批准的角色、人数、冗余和时间窗语义。
2. **旧树在当前软件合同下可重复，但不能据此证明毁伤定义有效。** 旧树输出与旧叶输入一致，只说明求值器执行了既有定义；它不能证明七个任务构成合理的 KMFC 分级，也不能证明现实装甲毁伤效能。
3. **“零锚连续证据 → 逐重复最终硬门 → 外层计数”可作为项目专用合成教师的软件骨架。** 新增原型强制四类 K/M/F/C、每类共同严重度轴、`level_ge_2 <= level_ge_1`、全零证据不产生正例、缺失证据输出 unavailable，并拒绝把穿透能量损失当作剩余动能。
4. **当前给出的尺度和阈值只是可执行哨兵值。** 它们没有使用现有已观察 Validation/Test 拟合，也没有物理标定含义。配置明确冻结为 `software_fixture_not_calibrated`，未来只能在独立 `tree_design_fit` 上选择，在 `tree_design_holdout` 上验证。
5. **当前 5000 例只能用于根因诊断，不能用于拟合新树。** 该归档中 K/F 各只有 28 个正支撑且来自同一案例集合、C 为零；M 有较宽支撑，可进入未来独立数据上的原型研究，但不能把当前已观察归档转用为最终阈值校准集。

机器可读结果见 `damage_tree_validity_and_software_prototype_audit.json`；可执行未拟合配置见 `kmfc_project_specific_software_prototype_v0_unfitted.json`。

## 2. 证据边界与方法

本审计复用了两个已经绑定冻结归档、目标和标签规范哈希的只读证据：

- `review_evidence/kmfc_ordinal_feasibility_audit_20261015.json`：旧七输出的支持度、叶映射和直接有序化可行性；
- `review_evidence/kmfc_v0_component_support_audit_20261015.json`：候选 KMFC 分支的组件域和归档正支撑。

本轮没有根据上述归档选择任何尺度、权重或阈值。软件性质使用人工构造的零、高值、缺失值和门控哨兵进行验证。这样可以证明实现是否满足数学合同，同时避免把当前已经观察过的 Validation/Test 固化进新教师。

本审计没有使用高保真外部仿真或实测数据，因此不评价真实毁伤概率、真实等级阈值、跨目标泛化和战斗部效能。

## 3. 旧七输出的有效性审计

### 3.1 支撑与输出库存

| 旧输出 | 可用案例 | 非零案例 | 归档 soft mass | 新 KMFC 直接复用结论 |
|---|---:|---:|---:|---|
| `effect-k1` | 5000 | 0 | 0 | 拒绝；与 K2 为不同组件事件 |
| `effect-k2` | 5000 | 0 | 0 | 拒绝；与 K1 为不同组件事件 |
| `effect-m1` | 5000 | 996 | 957.500 | 拒绝；与 M2 叶集合不同 |
| `effect-m2` | 5000 | 1770 | 1737.500 | 拒绝；有 1346 个严格顺序违例 |
| `effect-f1` | 5000 | 28 | 19.9995 | 拒绝；F2 不存在，F1 也不是完整新功能图 |
| `effect-c1` | 5000 | 0 | 0 | 仅结构兼容；语义未批准 |
| `effect-c2` | 5000 | 0 | 0 | 仅结构兼容；语义未批准 |

新合同需要八个累计输出：`K>=1`、`K>=2`、`M>=1`、`M>=2`、`F>=1`、`F>=2`、`C>=1`、`C>=2`。旧库存只有七项，缺少 F2。

### 3.2 旧输出到候选功能分支的 lineage

| 旧输出 | 旧叶事件 | 候选分支重叠 | 为什么不能直接作为新等级 |
|---|---|---|---|
| K1 | `component-003 failed` | `fuel_store` | K2 使用 `component-046`；两者不是共同严重度轴上的父子事件 |
| K2 | `component-046 failed` | `ammunition_store` | 不能由“弹药组件失效”推出“油箱失效” |
| M1 | `006–017 degraded`、`030–031 failed` | 左/右承载、主动/诱导轮 | M2 使用动力、履带、驾驶员等另一组叶；不是低高状态关系 |
| M2 | `001–005 failed`、`032–039 degraded`、`058 failed` | 动力、轮组、左右履带、驾驶员 | 5000 例中严格 `M2>M1` 为 1346 例，直接反证累计有序解释 |
| F1 | `047–048,051–054 failed` | 武器、供弹/弹药、对抗分支 | F2 缺失；感知、火控和其他武器部件没有形成完整共同严重度 |
| C1 | `058–067 degraded` | 基本乘员、载员 | 与 C2 的状态顺序形式兼容，但旧 `max(any person)` 不能表达角色和人数效应 |
| C2 | `058–067 failed` | 基本乘员、载员 | 仍需批准角色、冗余、人数和任务时间窗语义 |

这些重叠只用于追踪旧定义来源。新等级必须由同一类别严重度和父子硬门重新生成，不能将旧七列复制、重命名或强行排序。

## 4. 候选 KMFC 专用结构

原型配置使用经版本化的**功能分支严重度**作为类别层输入，避免组件数量多的分支在扁平平均中自动获得更大权重。

| 类别 | 候选分支 | 当前归档任一正支撑 | 至少两分支同时正 | 当前结论 |
|---|---|---:|---:|---|
| K | `fuel_store`、`ammunition_store` | 28 | 0 | 当前归档不能拟合阈值；仅弹药分支有支撑 |
| M | `powertrain`、`drive_and_idler`、左右承载、`suspension`、左右履带、`mobility_operator` | 2205 | 1331 | 可在未来独立校准数据上做受约束原型 |
| F | `sensing`、`fire_control`、`weapon`、`feed_and_ammunition`、`countermeasure` | 28 | 0 | 当前归档不能拟合阈值；只有供弹/弹药分支有支撑 |
| C | `essential_crew`、`embarked_personnel` | 0 | 0 | 当前归档为结构零，必须先修复可达性和采样覆盖 |

组件域沿用 KMFC-v0 候选 codebook：

- K：`003`；`046,053,054`；
- M：`001–039` 中批准的动力、轮组、承载、悬挂和履带组件，以及 `058`；
- F：`040–054` 中批准的感知、火控、武器、供弹/弹药和对抗组件；
- C：`058–067`。

K 与 F 共享弹药组件，M 与 C 共享驾驶员。这是显式跨类别依赖，因此 K/M/F/C 可同时为真，不能在四类之间使用 softmax。

## 5. 平滑层与最终硬门

对类别 `g`、重复 `r`，先把非负证据 `x` 变换为零锚分数：

\[
u_{g,j,r}=\frac{x_{g,j,r}}{x_{g,j,r}+s_{g,j}},\qquad u(0)=0,
\]

当前原型冻结指数为 1，以保证零点有限斜率。类别严重度采用正权幂平均：

\[
S_{g,r}=\left(\frac{\sum_j w_{g,j}u_{g,j,r}^{p_g}}
{\sum_jw_{g,j}}\right)^{1/p_g},\qquad w_{g,j}>0.
\]

最终离散等级在**每个重复内部**产生：

\[
B_{g,1,r}=A_{g,1,r}\mathbf1\{S_{g,r}\ge\tau_{g,1}\},
\]

\[
B_{g,2,r}=B_{g,1,r}A_{g,2,r}\mathbf1\{S_{g,r}\ge\tau_{g,2}\},
\quad 0<\tau_{g,1}<\tau_{g,2}\le1.
\]

独占等级为 `D=B1+B2`，因此只能取 0、1、2，且结构上恒有 `B2<=B1`。随后才计算：

\[
q_{g,1}=\frac{\sum_r B_{g,1,r}}{R_{g,\mathrm{available}}},\qquad
q_{g,2}=\frac{\sum_r B_{g,2,r}}{R_{g,\mathrm{available}}}.
\]

不能先平均 `S` 再过门。合成哨兵已经构造出 `mean(S)` 门控结果与逐重复事件频率不同的案例，测试强制保留正确顺序。

## 6. availability、等级 0 与未命中语义

原型采用 `all_required_evidence_and_gate_conditions`：

- 必需证据或必要门状态不可用：该重复对应类别 `availability=false`，严重度和等级序列化为 `null`；
- 必要门**已知为假**：这是有效的等级 0；
- 所有证据可用且为零：零锚平滑层严格输出 0，正一级阈值保证等级 0；
- 不允许把 NaN、缺失组件、未批准规则或不可解析叶填成标签 0。

这一区分对深度学习至关重要。训练损失、指标和有效样本量都必须使用类别级 availability mask；把 unavailable 当 0 会人为增加负例并产生与几何缺失、机制缺失高度相关的捷径。

## 7. 能量损失与剩余动能必须分离

当前破片实现包含两个不同物理量：

1. 归档 `deposited_energy_j` 是穿透前后动能差：

   \[
   E_{\mathrm{loss}}=\tfrac12m(v_{\mathrm{pre}}^2-v_{\mathrm{res}}^2).
   \]

2. 当前组件 degraded/failed 响应律使用部件处剩余动能：

   \[
   E_{\mathrm{res}}=\tfrac12m v_{\mathrm{res}}^2.
   \]

二者不等价，也不是同一变量的别名。原型只允许以下三个显式证据模型：

- `state_expected_severity`：由批准的累计组件状态导出的 `[0,1]` 证据；
- `approved_branch_severity`：由另一个已版本化下层聚合器产生的 `[0,1]` 功能分支证据；
- `residual_kinetic_energy_hill`：单位为 J 的 `component_residual_kinetic_energy_j`。

`deposited_energy_j`、`fragment_deposited_energy_j` 和 `penetration_energy_loss_j` 被显式拒绝为该教师的剩余能量输入。当前 5000 例支撑审计保存的是**案例级、重复均值的能量损失**，没有逐重复剩余动能，因此不能据此拟合 residual-energy 教师。

## 8. 可执行原型与参数边界

`kmfc_project_specific_software_prototype_v0_unfitted.json` 提供四类完整、可序列化和可哈希的配置。其当前数值为：

- 分支证据尺度：0.25；
- Hill 指数：1；
- 分支权重：等权；
- 幂平均阶数：2；
- 一级/二级哨兵阈值：0.30/0.65；
- 最小阈值间隔：0.20。

这些数值用于让零锚、单调、嵌套、缺失传播和重复内门控测试可执行，**不是优化所得阈值，也不是物理阈值**。配置同时写入：

```text
parameter_selection_policy = tree_design_fit_only_no_validation_or_test
parameter_status = software_fixture_not_calibrated
threshold_approval_status = not_approved_for_real_world_kmfc
calibration_artifact_hash = null
```

未来拟合后必须更新版本和内容哈希，并绑定 calibration artifact；不能原地覆盖当前文件。

## 9. 已验证的软件性质

本轮合成哨兵和单元测试验证了：

- 类别库存严格等于 K、M、F、C；
- 每类二级始终蕴含一级；
- 全零证据的严重度、一级和二级均严格为零；
- 硬门在每个重复内部执行，再聚合 successes/trials；
- 缺失证据和不可用必要条件传播为 unavailable/`null`；
- 已知为假的必要条件产生有效等级 0；
- 状态/分支证据强制位于 `[0,1]`；
- 剩余动能证据强制使用 `J` 和独立的 canonical quantity；
- 能量损失冒充剩余动能时编译失败；
- 结果保存配置哈希、证据模型、通道语义、逐重复累计等级和独占等级；
- 聚合结果分别保存 `successes_ge1`、`successes_ge2`、共同 trials、`n0/n1/n2` 和 Wilson 区间。

在用户指定的 `pytorch312-lmdt` 环境（Python 3.12.13、torch 2.8.0+cu128）中，`tests/test_kmfc_ordinal_teacher.py` 与 `tests/test_ordinal_projection.py` 合计 **51 passed**；原始记录为 `kmfc_targeted_tests.log`。

这些证据只支持“代码实现符合声明的数学合同”。它们不支持“阈值正确”“类别含义真实”或“现实毁伤概率有效”。

## 10. 与深度学习衔接的合同

新标签应建成四个 `ordinal_cumulative` 任务，每类两个累计目标，不应建成八个彼此独立的二元头。推荐网络输出：

\[
\hat q_{g,1}=\sigma(a_g),\qquad
\hat q_{g,2}=\hat q_{g,1}\sigma(b_g),
\]

从网络结构保证 `q2<=q1`。三个独占概率为：

\[
p_{g,0}=1-q_{g,1},\quad p_{g,1}=q_{g,1}-q_{g,2},\quad p_{g,2}=q_{g,2}.
\]

训练与评估应满足：

1. K/M/F/C 是可并发多任务，类别之间不使用 softmax；
2. 每类保存独立 availability mask，不可用行不进入损失或指标；
3. 保存 `successes/trials`，优先用二项或 beta-binomial 似然；如用累计 soft BCE，两个等级必须共享同一有效分母；
4. 报告 RPS、Brier/log loss、可靠性、每级 soft mass、ESS、阈值邻域误差和多 seed 方差；
5. design weight、任务权重和 minibatch 抽样概率必须分开；
6. 树参数只从 `tree_design_fit` 选择，网络超参数只从 `model_validation` 选择，密封 Test 不参与两者；
7. 旧七标签只能用于基线和迁移差异审计，不能作为新八输出的主监督。

## 11. 进入正式数据生成前的必要步骤

1. 审批并哈希冻结项目专用 codebook、各分支组件成员、下层分支聚合器和必要门语义。
2. 在结果盲的接触表面、方向和姿态设计下生成独立 `tree_design_fit` 与 `tree_design_holdout`，保存逐重复证据、availability、几何相交和穿透链。
3. 若采用能量证据，新增明确的逐重复 `component_residual_kinetic_energy_j`；能量损失继续使用独立字段，不得改名复用。
4. 只在 `tree_design_fit` 上拟合尺度、幂平均候选和有序阈值；阈值目标采用最低支持、不过度集中和稳定性约束，不能强迫精确三等分。
5. 在 `tree_design_holdout` 上验证零锚、单调、嵌套、子域稳定、阈值扰动敏感性和覆盖；K/F/C 若仍不可达，应返回 `coverage_unreachable`。
6. 冻结新树、LabelSpec、目标、registry 和 calibration artifact 的版本与内容哈希，再生成新的 Train/Validation/密封 Test。
7. 正式切换前运行旧树与新树的并行差异审计；旧归档保持只读，禁止原地覆盖。

## 12. 本轮交付边界

本轮没有修改旧 `damage_tree`、旧目标文件或旧七标签归档，也没有把原型接入正式引擎输出。这样可以先审查接口、数学性质和数据语义，再在新采样覆盖和独立校准数据具备后决定是否晋升。

当前可接受的表述是：

> 已实现一个项目专用、软件范围、未标定的 KMFC 有序教师骨架，并用合成哨兵证明零锚、逐重复硬门、等级嵌套、缺失传播和能量语义隔离。旧七输出不能直接迁移；物理有效性和科研训练准入仍未建立。

