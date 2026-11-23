# V6 球面入口射线优化预注册

状态：`frozen_before_v6_execution`

## 1. 基线身份

- 基线提交：`ff2ef1f21c7d6bf73630bdb11331289278fb69dd`
- 基线 SamplingSpec：`configs/sampling/p0_contact_reference.json`
- 基线 SamplingSpec 哈希：`sha256:4589489789fbb0389f7da8ce1c5db5bd0162b70ba1933838e8edfe76b9c1062b`
- 归一化目标哈希：`sha256:8f03659312d50ec0ae497b77eb01dd4f8612aa4c61aa9e177aa90fde090492a2`
- 物理注册表哈希：`sha256:809b15a7010ed540403b299946caec96dc0c29aa03aba5e36aed61169af2ce45`
- estimand：`criterion_exceedance_probability_given_first_armor_contact`
- 模型输入：`loitering_impact_contact_pose@1.0.0`

现有 1.4 配置、冻结数据和历史验证产物只读保留。V6 使用 1.5 合同和独立产物目录。

## 2. V6-A 唯一主张

在常速直线飞行、相同有向物理射线、相同弹型、命中速度、滚转姿态和随机实现下，把轨迹起点从旧 standoff 平面平移到保守包络球的上游交点，不改变首次装甲接触后的软件教师科学响应。

该主张不包含作战命中概率、目标表面面积均匀、有限尺寸弹体扫掠接触、飞行段阻力或姿态演化，也不构成物理标定。

## 3. 冻结球体合同

- 算法：`target-aabb-circumsphere@1.0.0`
- 对象范围：`all`
- 生产 clearance：`0.01 m`
- 中心：全部编译对象 AABB 并集的中点
- 半径：AABB 半对角线长度加 clearance
- 越界射线：拒绝，不截断横向偏移

无 clearance 的当前目标参考值：

- center = `[0.0, 0.3230171203613281, 1.35686] m`
- radius = `4.359241643112272 m`

## 4. 身份与随机数

物理射线身份由归一化有向方向和 Pluecker moment `p x d` 构成，并绑定目标哈希。身份排除 standoff、球半径、clearance、球面起点、碰撞距离和飞行时间。

教师公共随机数分组绑定 physical ray ID、munition profile、impact speed、body-X roll 和 target hash。设计 case ID 继续绑定 SamplingSpec、设计序列和球体政策；设计身份与物理随机分组身份分离。

## 5. Phase 1 配对等价门 G1

### 5.1 几何层

覆盖六个 approach 的中心、四角和轴向边界。旧起点与球面起点必须满足：

- 命中/未命中状态完全一致；
- 首个 armor ID 完全一致；
- 命中点和法向量绝对误差不超过 `1e-10`；
- 命中对象顺序完全一致；
- 每个对象的物内路径长度绝对误差不超过 `1e-10 m`。

`entry_t`、`collision_distance_m`、`detonation_time_s` 随纵向起点平移，不属于等价字段。

### 5.2 教师层

覆盖 `6 approaches x 3 profiles`，使用至少三个固定教师 seed。双方使用相同 aleatory seed 和 epistemic group。必须满足：

- S-CONTACT 双方均通过；
- `scientific_completion_status` 与 domain censoring 状态完全一致；
- 部件统计、summary 和 damage-tree 字段集合完全一致；
- 所有有限科学数值满足 `atol=1e-12, rtol=1e-10`。

允许变化的字段仅限 case/provenance 身份、trajectory origin、碰撞距离、碰撞时间和由这些量直接派生的追溯信息。

### 5.3 判定

任何科学响应不一致、接触对象改变、球外射线被静默裁剪或旧 1.4 回归失败，G1 均失败。只有 G1 全部通过，才允许启用 5D reduced physical-ray 配置。

## 6. 后续阶段

### Phase 2：降维与序列

- B1：6D paired dummy-standoff；
- B2：5D reduced physical-ray；
- B3：在 B2 上单独比较 LHS 与低差异或最大间距设计；
- 每次只改变一个因素。

### Phase 3：V6-B 横向测度

横向 proposal 与球面重锚分离。均匀截面只解释为平行通量测度；若目标是表面面积测度，必须显式处理 `d^2b = |n dot d| dA` 和首次可见面约束。

Reference 必须为 proxy-partial 候选保留正纳入概率，并单列 `reference_audit`；否则不能回权到全部接触候选。

### Phase 4：固定教师重复数

使用独立 pilot 和嵌套公共随机数比较 `R in {4, 8, 16, 32, 64}`。确认前按 role/stratum 冻结 R；`outcome_adaptive=false` 保持不变。

### Phase 5-8：开发、确认与学习

- 至少三个独立 design frame，确认期推荐 5-10 个；
- 外层 frame、中层 cell x candidate-group、内层 replicate 三层推断；
- Coverage admission 与 Reference replacement 分开判定；
- 学习输入消融使用相同数据、模型、训练预算和至少三个训练 seed；
- 阈值、哈希与模型选择冻结后才打开 sealed test。

## 7. 主指标与双门

Coverage 主指标为：

`complete_non_saturated_yield = E_Q[domain_complete * (1 - tree_saturated)]`

Coverage admission 还必须检查完整域非劣、S-CONTACT、ESS/权重尾部、关键部件覆盖和计算成本。

Reference replacement 额外要求同一目标测度、完整支持、正纳入概率、可验证回权、proxy-partial audit 和封存确认通过。Coverage 通过不推出 Reference replacement。

## 8. 停止规则

- G1 失败：停止降维，定位几何、身份或时间耦合来源；
- 查看结果后改变确认阈值：该轮降级为开发证据；
- 任一方案出现零支持区域：不得宣称 Reference 替代；
- 单 seed 或单 frame 改善：不得升级为研究结论；
- 软件一致性通过：不得表述为物理标定通过。
