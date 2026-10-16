# 工程术语表

本表约束代码、归档和研究报告中的同名概念。术语一致不代表物理模型已经校准，也不代表数据获得研究准入。

| 术语 | 工程定义 | 不应混同 |
|---|---|---|
| case | 一次教师引擎求值所需的完整输入，包含场景、战斗部、求解与审计参数 | learner input |
| candidate | 采样设计产生的一次候选尝试；在接触条件模式中可以被拒绝且没有 formal case | successful case |
| lineage | 候选所属参考/自适应轮次及父轮次的内容绑定身份 | 运行时间或目录名 |
| formal case | 已通过标签无关筛选、冻结 split 并可交给教师引擎执行的 case | candidate attempt |
| outcome | formal case 的 `ok` 或 `refused` 执行结果及其内容身份 | 研究结论 |
| archive | manifest、checkpoint、张量、表格和结果账本组成的不可变数据产品 | 单个 `tensors.npz` |
| archive receipt | 经共同完整性实现验证后得到的 manifest、inventory 与身份结论 | 物理有效性证明 |
| learner input | 版本化、默认暴露给神经网络的最小状态；当前为 UAV state 或 loitering-impact state | teacher、solver、装甲/目标静态字段 |
| raw design weight | 对声明参考总体 estimand 的抽样设计修正量 | task weight / minibatch probability |
| Reference | outcome-blind 冻结的参考采样流，可用于预先声明的 Train/Validation/Test 角色 | targeted/enriched stream |
| Enriched Train | 只进入 Train 的稀有事件或自适应采样流，必须保存 proposal 与 raw design weight | Reference Test |
| Challenge | 单独报告的压力/稀有条件集合，不得并入 Reference 总体指标 | confirmatory Test |

## 证据层级

1. **软件完整性**：格式、哈希、状态机、重放和测试通过。
2. **数据集完整性**：输入、mask、权重、split 与归档 inventory 自洽。
3. **统计可靠性**：在明确 estimand 下，样本量、ESS、不确定度与泄漏控制达到预设判据。
4. **物理有效性**：通过外部试验、标定、灵敏度和适用域证据支持。
5. **研究准入**：满足冻结方案中全部独立验收门槛。

前一层不能自动推出后一层。
