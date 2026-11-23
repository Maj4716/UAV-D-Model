# idea2 项目排查、优化与试生成评估

日期：2026-11-20  
结论：工程链路通过，最新 360 例数据包适合管线与探索性模型调试，尚不适合神经网络正式训练或确认性评价。

## 1. 可行性与科学性判断

`idea2.md` 的主线可行：先建立可学习性基线，再把标签改为跨重复数的概率对象，把物理状态与毁伤树解耦，最后改进采样和输入表征。这一顺序可以区分“采样覆盖不足、输入表征不足、标签统计不足、物理教师域不足”四类问题。

本轮采用了优先级最高且证据充分的路线：

1. 冻结 Phase 0 基线，禁止读取 test；
2. 增加重复级部件状态轨迹和离线树重算接口；
3. 以 `p_hit` 作为 R 无关主目标，二值 support 仅作带置信区间的派生视图；
4. 采用球面入射五维参数化和 scrambled Sobol；
5. 新建 16 维首层装甲状态输入合同；
6. 用新合同试生成数据并按采样、泄漏、域完整、标签质量和任务质量审计。

两项方法未直接执行：

- “终止传播/透明传播”双分支不能作为全部输出的全局上下界。装甲开孔、传播时序和后续部件可达性破坏所需单调性。本轮保留已证明有效的受影响重复最坏界，双分支只允许作情景分析。
- 连续来向和暴露场属于下一阶段。当前主要瓶颈是域删失与 R=4 标签分辨率；先扩大输入模型会增加算量，却不会解除这两个训练阻点。

对应决策已写入 [DECISIONS.md](../../../docs/DECISIONS.md)。

## 2. 前半部分基线结果

Phase 0 使用当前 10,589 例开发数据、同一两层共享 MLP、三种训练比例和三个随机种子。全部评价来自 2,000 例 Reference validation，test 未访问。完整结果见 [Phase 0 摘要](../phase0/README.md) 与 `artifacts/experiments/phase0_20261120/results.json`。

全量训练时：

| 输入 | 部件 support 宏平均 PR-AUC | 树输出软 PR-AUC | 正例严重度 MAE | PR-AUC<0.2 的部件数 |
| --- | ---: | ---: | ---: | ---: |
| 15 维采样坐标 `x_design` | 0.740 | 0.812 | 0.218 | 3/67 |
| 原物理姿态 `x_normalized` | 0.427 | 0.661 | 0.311 | 11/67 |

`x_design` 包含方向类别和采样分位数，属于采样器坐标，不是生产输入。该结果证明任务对紧凑目标编码敏感，也表明原物理姿态存在表征瓶颈。剩余低分部件的 validation 正例仅 17–22 个，第二瓶颈是稀有部件支持量。

## 3. 第九部分重新审视后的实现

### 3.1 状态轨迹与树解耦

- `DamageEngine` 可选输出重复级部件机制状态、在线系统输出、域标志、参数哈希和来源身份；
- `tools/recompute_tree_outputs.py` 可从状态轨迹离线重算树；
- 在线逐重复评树后取均值与离线重算的最大误差为 `1.11e-16`；
- 权威树目标保持“逐重复评估后平均”，期望概率上的树只作辅助视图。

### 3.2 R 无关的 hurdle 目标

- `damage_engine/hurdle_targets.py` 从 `fragment_metrics.hit_indicator.mean` 提取 `p_hit`；
- 同时保存样本数、样本标准差、标准误、95% 区间和域 mask；
- 二值 support 使用固定阈值 `0.05` 和 `ci95_excludes_fixed_probability_threshold`；
- 命中后能量与命中数缺少显式条件充分统计时标记 unavailable，不用无条件总量替代。

### 3.3 采样与学习输入

- 84 组球面入口配对验证和 54 组五维降维验证全部通过，教师数值最大差为 0；
- 预标签序列比较中，scrambled Sobol 排名第一：接触失败数为 0、最小单元接触率为 1.0、最坏二维 discrepancy 为 0.00111；
- 新配置移除主 Reference 中的 `bottom_up_pos_z`，保留其压力测试代码；
- 新 16 维输入为三弹型 one-hot、首层装甲命中点、表面法向、单位入射方向、速度和 body-y 轴；全部字段在教师标签执行前确定。

配置：[p0_sphere_entry_sobol_no_bottom_impact_state.v1.json](../../../configs/sampling/p0_sphere_entry_sobol_no_bottom_impact_state.v1.json)  
SamplingSpec hash：`sha256:1c9aa0a04832000311e2c8934ddd406224b7cd66ea7a4fc0664c9bc87d29eaf6`

## 4. 最新策略试生成

数据包：[no_bottom_impact_state_pilot](../sampling_sphere_v1/no_bottom_impact_state_pilot/)  
dataset id：`damage-dataset-0d329baeb6e488cd40309c56`  
tensor contract：`1.8.0`

| 项目 | 结果 |
| --- | ---: |
| 正式样本 | 360 |
| 有效 / 拒绝 | 360 / 0 |
| train / validation / test | 240 / 60 / 60 |
| 重复数 | R=4 |
| Reference ESS / ESS fraction | 360 / 1.0 |
| 权重 CV | 0 |
| 采样 positivity | 1.0 |
| 精确重复输入 | 0 |
| 跨切分重复输入 | 0 |

采样质量门全部通过。六个设计轴的十箱占用率均为 1.0，最大 KS 偏差为 0.0134。16 维模型输入 360 行全部唯一、mask 全真、无常量列；validation 和 test 相对 train 的最大标准化均值差分别为 0.125 和 0.103。

试生成中的 test 已被审计读取，状态固定为 `observed_diagnostic_only`，不能用于后续确认性结论。下一版需要重新生成密封 Reference test。

## 5. 标签与训练就绪度

### 5.1 事件与域完整性

| 指标 | 全部 | train | validation | test |
| --- | ---: | ---: | ---: | ---: |
| 至少一次破片命中 | 29.17% | 29.58% | 30.00% | 26.67% |
| 至少一次破片侵彻 | 25.00% | 25.42% | 26.67% | 21.67% |
| 域完整案例 | 137/360 | 92/240 | 25/60 | 20/60 |
| 域完整率 | 38.06% | 38.33% | 41.67% | 33.33% |

参数集仍为 `loitering-impact-unverified-1`。数据 manifest 将 360 例全部归为 `software_development`，`research_training=0`。

现有 10,589 例开发包的域完整率为 69.29%（7,337/10,589），新 pilot 为 38.06%。两者的采样总体不同，不能把差异直接归因于 Sobol；该结果足以否决当前新配置的基线晋升，状态继续为 `provisional`。

### 5.2 p_hit 与派生 support

360×67 共 24,120 个部件槽位：

- 14,941 个因域删失关闭训练 mask；
- 9,179 个槽位可用于 `p_hit`，其中 8,084 个为 0；
- train 中只有 92 个域完整案例；每部件非零 `p_hit` 案例数的最小值/中位数/最大值为 0/6/46，1 个部件完全没有非零案例；
- 阈值 0.05 的派生二值 support 得到 769 个正例、8,410 个区间跨阈值样本、0 个负例。

零命中且 R=4 时，Hoeffding 95% 上界为 0.679，无法判定 `p_hit<0.05`。同一规则要让零命中的上界低于 0.05，至少需要 R=738。因此，R=4 下应直接训练软 `p_hit` 或区间目标，不能训练当前派生二值 support。

### 5.3 系统任务

系统学习合同没有可进入默认训练的任务：6 个输出因 train 软事件质量不足而保持 exploratory，`effect-f1` 还因 `component-050` 的树阈值未解析而被阻断。所有系统输出的 R=4 区间半宽均大于 0.10。

审计入口：

- [sampling_quality.json](../sampling_sphere_v1/no_bottom_impact_state_pilot/audits/sampling_quality.json)
- [distribution_audit.json](../sampling_sphere_v1/no_bottom_impact_state_pilot/audits/distribution_audit.json)
- [training_readiness.json](../sampling_sphere_v1/no_bottom_impact_state_pilot/audits/training_readiness.json)
- [system_learning_contract.json](../sampling_sphere_v1/no_bottom_impact_state_pilot/audits/system_learning_contract.json)
- [hurdle_targets_summary.json](../sampling_sphere_v1/no_bottom_impact_state_pilot/audits/hurdle_targets_summary.json)（完整逐案例文件为同目录 `hurdle_targets.json`）

## 6. 训练适用性结论

当前 pilot 适合：

- 验证球面 Sobol、分片归档、恢复、哈希、16 维输入、loader 和审计链；
- 调试损失、mask、batch 和小规模网络；
- 估计域删失、稀疏程度和下一轮预算。

当前 pilot 不适合：

- 训练可交付的 67 部件神经网络；
- 训练派生二值 support；
- 选择或确认系统树模型；
- 声称现实物理预测能力。

阻点按优先级排序为：域完整率 38.06% → R=4 标签分辨率 → 稀有部件样本量 → `effect-f1` 树语义 → 破片方向和材料参数外部有效性。采样完整性和 16 维输入合同本身已经通过。

因此，Reference 稳定基线暂不切换到本次五来向 Sobol 配置。新配置保留为下一轮域覆盖修复和同预算 A/B 的候选臂。

## 7. 后续路线

1. **先修域覆盖。** 扩展侵彻模型适用域或缩小并重新声明 Reference 支持；以至少 90% 域完整率作为下一阶段工程准入门，同时单独报告被排除的目标区域。
2. **主目标保持软 `p_hit`。** R=4 只用于广覆盖；对靠近决策阈值、稀有部件和校准集采用预注册的自适应 R。若保留 Hoeffding+0.05 的二值规则，负例预算需要 R≥738。
3. **回收区间删失信息。** 在在线充分统计中增加部件级 `p_hit` 严格界，训练区间损失；在小型审计子集开启重复级状态轨迹，验证在线界与离线重算一致。
4. **分阶段扩样。** 域问题修复后先生成约 2,000 个 train 案例重跑学习曲线；达到部件支持和系统事件质量门后，再扩到 8,000–12,000。validation/test 只从 Reference 生成，Coverage 仅增强 train。
5. **做同预算输入消融。** 比较 16 维首撞状态、原 15 维接触姿态和采样坐标代理；使用同一网络、三种训练比例和至少三个种子。采样坐标只作诊断上界。
6. **完成系统语义与物理验证。** 解析 `component-050` 阈值；按 [FRAGMENT_DIRECTION_MODEL.md](../../../docs/FRAGMENT_DIRECTION_MODEL.md) 对 F0/F1/F2 做公共随机数敏感性试验，并为方向模型、材料参数和阈值补充外部依据。

## 8. 验证状态

相关模块定向测试、16 维 Writer→NPZ→Torch 端到端测试和完整仓库回归均通过：

- `pytest`：1,170 passed，240.42 s；
- `compileall`：通过；
- `pip check`：无损坏依赖；
- `git diff --check`：通过，仅有 Windows LF→CRLF 提示；
- 两套 schema 目录共有的 5 份 JSON 逐字节一致。
