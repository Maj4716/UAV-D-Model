# 巡飞弹撞击装甲车数据集科学性与神经网络训练审计

- 数据集：`damage-dataset-fb1af45b41880da2e8ef368a`
- 审计版本：`loitering-impact-dataset-audit@1.0.0`
- 正式样本 / 成功张量 / 拒绝：5000 / 5000 / 0
- 默认系统任务训练就绪：**否**
- 科研准入：**否**。本报告是描述性软件与数据审计，不是实物毁伤或物理模型验证。

## 1. 样本组成

| 维度 | 计数 |
|---|---:|
| 划分 `train` | 3000 |
| 划分 `validation` | 500 |
| 划分 `test` | 1500 |
| 弹型 `small_front` | 1662 |
| 弹型 `medium_front` | 1670 |
| 弹型 `large_front` | 1668 |

## 2. 设计输入分布

| 输入 | 最小 | P05 | 中位数 | P95 | 最大 | 均值±标准差 | 唯一值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `origin_y_m` | -3 | -2.772 | -0.4852 | 1.747 | 2 | -0.4959 ± 1.445 | 5000 |
| `origin_z_m` | 0.6504 | 0.7055 | 1.217 | 1.741 | 1.8 | 1.22 ± 0.3324 | 5000 |
| `impact_speed_mps` | 80.01 | 84.06 | 119.1 | 155.9 | 160 | 119.5 ± 22.95 | 5000 |
| `collision_distance_m` | 8.37 | 8.37 | 8.37 | 8.813 | 8.813 | 8.566 ± 0.2199 | 3 |
| `detonation_time_s` | 0.05233 | 0.05468 | 0.07174 | 0.1018 | 0.1101 | 0.07448 ± 0.01508 | 5000 |

三个连续设计变量的最大绝对 Pearson 相关系数为 0.009744。这是采样设计诊断，不是统计独立性的证明。

| 声明均匀变量 | 区间 | 占用箱/总箱 | 越界 | 最小/中位/最大箱计数 |
|---|---|---:|---:|---:|
| `trajectory_origin_y_m` | [-3.0, 2.0] | 20/20 | 0 | 236/248.5/272 |
| `trajectory_origin_z_m` | [0.65, 1.8] | 20/20 | 0 | 236/249/269 |
| `impact_speed_mps` | [80.0, 160.0] | 20/20 | 0 | 218/250/273 |

### 显式学习器输入契约

- 张量形状：`[5000, 6]`；特征：`['munition_profile_small_front', 'munition_profile_medium_front', 'munition_profile_large_front', 'trajectory_origin_y_m', 'trajectory_origin_z_m', 'impact_speed_mps']`。
- mask 全真 / 数值全有限 / one-hot 合法：True / True / True。
- 与冻结 formal case 重建的最大绝对误差：7.626e-06。
- Train 常量特征：`[]`；跨划分重复输入组：0。
- 连续/one-hot 非零 Train 标准差尺度比：69.43。
- 归一化要求：Fit continuous-feature normalization on Train only and apply the frozen artifact to Validation/Test. Keep the three one-hot profile indicators unscaled or explicitly document their transform. Do not normalize using all splits.

## 3. 重复与划分泄漏

- 设计输入额外重复行：0。
- case hash 额外重复行：0。
- 跨划分违规证据计数：0。

## 4. 标签可用性、常量与稀有性

系统标签 mask 总体可用率：**1**。

| 系统输出 | 可用数 | 均值 | 标准差 | 唯一值 | 零值率 | 一值率 | Train 非零/可用 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `effect-m2` | 5000 | 0.3475 | 0.4728 | 57 | 0.646 | 0.332 | 1064/3000 |
| `effect-k1` | 5000 | 0 | 0 | 1 | 1 | 0 | 0/3000 |
| `effect-m1` | 5000 | 0.1915 | 0.3886 | 51 | 0.8008 | 0.1668 | 590/3000 |
| `effect-k2` | 5000 | 0 | 0 | 1 | 1 | 0 | 0/3000 |
| `effect-f1` | 5000 | 0.004 | 0.05643 | 26 | 0.9944 | 0.0008 | 11/3000 |
| `effect-c1` | 5000 | 0 | 0 | 1 | 1 | 0 | 0/3000 |
| `effect-c2` | 5000 | 0 | 0 | 1 | 1 | 0 | 0/3000 |

组件级碎片事件（每 case 至少一个组件发生事件）：

- `fragment_hits`：2205/5000，率 0.441，Wilson 95% CI [0.4272882901123114, 0.45480229871720546]。
- `fragment_penetrations`：2205/5000，率 0.441，Wilson 95% CI [0.4272882901123114, 0.45480229871720546]。

## 5. Monte Carlo 与设计权重

- replicate 数：min=2，median=2，max=2。
- 系统标签 CI95 全宽：median=0，P95=0，max=1。
- 有效设计权重：n=5000，ESS=5000，ESS 比例=1。
- 注意：CI and standard-deviation arrays quantify simulator Monte Carlo uncertainty only; they do not cover target-model, munition-parameter, structural-model or real-world discrepancy。

## 6. 碰撞即起爆一致性

- 检查 5000 条，信息不完整 0 条。
- 时间误差最大值：0 s。
- 位置误差最大值：1.776e-15 m。
- 法向单位长度误差最大值：0。
- 违规计数：`{"trigger_not_first_armor_contact": 0, "nonzero_or_missing_delay": 0, "time_consistency": 0, "position_consistency": 0, "normal_consistency": 0, "missing_collision_armor": 0}`。

## 7. 问题与专业修改建议

### 1. [CRITICAL] constant-system-labels

- 发现：训练划分中的常量系统标签：effect-k1, effect-k2, effect-c1, effect-c2
- 后果：对应输出不可辨识，神经网络无法学习输入到标签的关系。
- 修改建议：扩大能触发状态变化的物理设计域，或从当前任务移除这些输出；不得用关闭 mask 或改标签数值绕过检查。

### 2. [CRITICAL] research-admission-not-established

- 发现：弹体/目标/物理响应仍含未验证来源，且归档未建立独立科研验证或 research_* 准入。
- 后果：当前数据最多证明软件链路可执行，不能支持真实毁伤精度、外推能力或论文中的物理有效性结论。
- 修改建议：使用可追溯实测/权威数据校准弹体与目标参数；预注册外部验证方案，保留冻结 holdout，并报告模型差异、不确定度和适用域。

### 3. [HIGH] low-monte-carlo-replicates

- 发现：每工况最少仅 2.0 个 MC 重复，概率分辨率和方差估计不足。
- 后果：软标签呈粗量化，CI/标准差在 R=2 时尤其不稳，稀有事件可能被误判为零。
- 修改建议：先做独立收敛研究；按最难收敛输出确定 R，建议将 R=2 仅作为流水线烟雾配置，并保留预先声明且不窥视验证/测试结果的停止规则。

### 4. [HIGH] rare-system-labels

- 发现：训练划分中至少一侧少于 100 条的系统标签：effect-k1, effect-k2, effect-f1, effect-c1, effect-c2
- 后果：分类损失和评估指标高度不稳定，模型可能学成恒定预测器。
- 修改建议：先做分层稀有事件扩充；设计权重与 minibatch/task 权重分离，并在冻结验证/测试集报告 PR-AUC、Brier、校准曲线及置信区间。

## 8. 科研证据边界

本审计可证明归档可读取、输入分布/划分/标签 mask/MC 数量及碰撞几何的一致性。它不能证明真实装甲毁伤精度、弹体参数真实性、响应模型校准、跨目标泛化或论文层面的科研有效性。

- 弹体验证状态：`{"unverified": 5000}`
- research 准入计数：`{"software_development": 5000, "research_training": 0, "research_validation": 0, "research_test": 0}`
- 独立科学验证：`False`

因此即使所有软件检查通过，`research_admission` 仍保持 `false`，直到完成独立校准与冻结外部验证。
