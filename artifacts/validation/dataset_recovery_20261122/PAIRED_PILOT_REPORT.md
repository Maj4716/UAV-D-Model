# v2/v3 配对恢复 Pilot 对比报告

日期：2026-11-22  
范围：相同连续来向 Reference 设计、相同种子、相同标签合同；每个工况 `R=8`。总计 60 例，train / validation / sealed test = 30 / 15 / 15。

## 结论

v3 已经解决了阻止数据扩展的主要工程问题：开发集平均受影响重复比例从 82.78% 降到 3.61%，三个概率头的点识别率均提高到 99.7% 以上，识别区间中位宽度从 1 降到 0。这些结果支持将 v3 进入大样本恢复阶段。

60 例 pilot 不是正式训练集。train 只有 30 例，而部件准入门同时要求正例不少于 20、负例不少于 20；该门在 30 例上数学上不可能通过。因此 `formal_neural_training_ready=false` 由样本规模和稀有支持门决定，不再由大面积标签删失决定。

## 配对设计与封存约束

- v2 对照：`artifacts/datasets/recovery_paired_control_v2_r8_20261122_retry1`
- v3 高斜角单侧界：`artifacts/datasets/recovery_paired_bound_v3_r8_20261122`
- 开发审计只解码 train + validation 的 45 条结果。
- 两个数据集的 sealed-test 工况顺序哈希相同：`sha256:639d40ff4794fc55a4b76fb98412e2d19077e0c8842acff765b1b6dcff32bed0`。
- v2 与 v3 的 `test_result_json_lines_decoded=0`、`test_open_count=0`、`test_metrics_computed=false`。

## 主要结果

| 开发集指标（45 例） | v2 对照 | v3 单侧界 | 变化 |
|---|---:|---:|---:|
| 平均 affected replicate fraction | 0.8278 | 0.0361 | -0.7917 |
| affected replicate fraction 中位数 | 1.0000 | 0.0000 | -1.0000 |
| epsilon-complete 工况 | 4 / 45（8.89%） | 40 / 45（88.89%） | +80.00 pp |
| `p_hit` 点观测率 | 16.95% | 99.93% | +82.98 pp |
| `p_degraded` 点观测率 | 15.39% | 99.90% | +84.51 pp |
| `p_failed` 点观测率 | 12.94% | 99.73% | +86.80 pp |
| 条件能量 / 命中数观测率 | 1.96% | 10.98% | +9.02 pp |
| 设计权重 ESS fraction | 1.0000 | 1.0000 | 不变 |

### 识别区间宽度

| 概率头 | v2 均值 / 中位数 / P90 | v3 均值 / 中位数 / P90 | v3 最大值 |
|---|---:|---:|---:|
| `p_hit` | 0.713806 / 1 / 1 | 0.000083 / 0 / 0 | 0.125 |
| `p_degraded` | 0.725742 / 1 / 1 | 0.000115 / 0 / 0 | 0.125 |
| `p_failed` | 0.750374 / 1 / 1 | 0.000198 / 0 / 0 | 0.125 |

v3 仍保留了少量宽区间，说明无法证明非贯穿的事件没有被强行转换为点标签。`R=8` 时单个受影响重复对宽度的最小非零步长是 0.125，与观测到的最大宽度一致。

### train / validation 稳定性

| 版本与分割 | 平均 affected fraction | epsilon-complete | `p_hit` / `p_degraded` / `p_failed` 观测率 |
|---|---:|---:|---:|
| v2 train | 0.7875 | 4 / 30（13.33%） | 21.99% / 20.55% / 18.21% |
| v2 validation | 0.9083 | 0 / 15（0%） | 6.87% / 5.07% / 2.39% |
| v3 train | 0.0125 | 27 / 30（90.00%） | 99.90% / 99.85% / 99.70% |
| v3 validation | 0.0833 | 13 / 15（86.67%） | 100.00% / 100.00% / 99.80% |

v3 的改善同时出现在 train 和 validation，不是单一分割的偶然现象。validation 的 15 例过少，86.67% 的工况级 epsilon-complete 率需要在大样本上复核。

## 部件支持与准入

- v2 开发集中，`p_hit / p_degraded / p_failed` 分别有 19 / 22 / 28 个部件没有可观测正例。
- v3 开发集中，三个概率头都为 9 个部件零正例；逐部件可观测数的最小值为 44 / 45，中位数为 45 / 45。
- v3 validation 中有 9 个部件达到正负支持门；v2 为 0。
- v2 和 v3 的正式准入部件数都为 0。v3 的直接原因是 30 例 train 无法同时满足“正例 >= 20 且负例 >= 20”，不应据此将零正例部件判为结构不可达。

## 科学性边界

v3 是未经外部标定的软件教师候选。它只在以下显式形状约束下给出单侧结论：对同一事件潜变量分位 `Q`，弹道极限 `V_limit(theta, Q)` 随斜角逐路径单调不减。当到达速度不高于登记边界处的同分位弹道极限时，v3 证明非贯穿；不能证明的高斜角事件仍作为 OOD。

该 pilot 证明了软件一致性和标签可用性的显著改善，不证明高斜角真实响应、跳弹方向、破片或材料失效机理的物理正确性。物理验证和研究准入继续作为独立门。

## 下一步验收

1. 按新的、与结果无关的种子扩大 Reference train / validation / sealed test，不根据 pilot 结果重分割。
2. 优先复核每部件 train 正/负支持 >= 20、validation 正/负支持 >= 5，以及 train epsilon-complete >= 90%。
3. 用 25% / 50% / 100% 学习曲线和至少 3 个种子检查 Brier、ECE、种子方差与曲线斜率。
4. 在模型、阈值和准入任务冻结前，sealed test 保持零解码。

## 证据文件

- `artifacts/datasets/recovery_paired_control_v2_r8_20261122_retry1/development_audit.json`
- `artifacts/datasets/recovery_paired_bound_v3_r8_20261122/development_audit.json`
- `artifacts/datasets/recovery_paired_control_v2_r8_20261122_retry1/dataset_v2_development/sealed_test.json`
- `artifacts/datasets/recovery_paired_bound_v3_r8_20261122/dataset_v2_development/sealed_test.json`
