# 工程术语表

本表统一代码、配置、归档和研究文档中的核心概念。当前项目只声明软件与数据层证据。

| 术语 | 工程定义 | 不应混同 |
|---|---|---|
| case | 一次软件教师求值所需的完整输入，包含场景、战斗部、求解与审计参数 | learner input |
| candidate | 采样设计产生的一次候选尝试；可以在正式执行前被拒绝 | successful case |
| lineage | 候选所属采样臂、设计种子、轮次及父轮次的内容绑定身份 | 运行时间或目录名 |
| formal case | 通过标签无关筛选、冻结 split 并可交给教师执行的 case | candidate attempt |
| outcome | formal case 的 `ok`、`refused` 或域状态及其内容身份 | 研究结论 |
| archive | manifest、checkpoint、张量、表格与结果账本组成的可核验数据产品 | 单个 `tensors.npz` |
| archive receipt | 由统一完整性实现生成的 manifest、inventory 与身份验证结论 | 模型有效性证明 |
| learner input | 版本化、默认暴露给神经网络的最小输入状态；当前接触姿态合同为 15 维 | 教师中间量、目标拓扑或求解器状态 |
| software teacher | 根据冻结输入、注册模型和随机键产生合成标签的软件实现 | 外部观测真值 |
| response mask | 指示某行响应标签是否可用于损失计算的布尔量 | 零毁伤标签 |
| domain censoring | 事件超出已注册软件模型适用域，响应保持未知并以 mask/`NaN` 传播 | 已知无响应 |
| raw design weight | 面向声明参考总体 estimand 的抽样设计修正量 | task weight 或 minibatch probability |
| Reference | outcome-blind 的分层参考采样臂，用于估计声明的设计分布 | Coverage 增强臂 |
| Coverage | 面向完整域、过渡区或部件支持的训练增强采样臂 | Reference 发生率 |
| selection observation | 某候选在某个采样臂中的一次入选记录，含一阶纳入概率 | 唯一候选样本 |
| design seed | 冻结一轮候选与选择随机性的种子，用于组间隔离和复现 | 教师 replicate seed |
| perforation event | 带形成时间、位置、法向和半径的一次穿孔历史记录 | 当前时刻的唯一开口数 |
| connected opening | 当前时刻重叠开口原语并集的连通分量 | 穿孔事件数 |
| unresolved penetration event | 几何接触已发生但倾角超出注册侵彻模型适用域的事件 | 已知未侵彻或零毁伤 |
| penetration-domain censoring bounds | 对含未解析侵彻事件的重复输出最坏情形上下界 | 普通采样置信区间 |
| gate-dependency sensitivity | 在声明、独立和完全正相关等依赖情景下传播同一叶边际量 | 由数据识别的相关系数 |
| development-ready | 可用于管线、模型、损失和采样方法开发的数据角色 | 封存确认集或最终应用数据 |
| research admission | 满足冻结研究方案中全部独立验收门的状态 | 测试通过或归档完整 |

## 证据层级

1. **输入与软件完整性**：Schema、哈希、状态机、重放和测试通过。
2. **数据集完整性**：输入、mask、权重、split 与归档 inventory 自洽。
3. **统计可靠性**：在明确 estimand 下报告样本量、ESS、不确定度与泄漏控制。
4. **研究准入**：冻结方案中全部独立比较门通过，且用途与数据角色一致。

较低层级通过不自动推出较高层级。本项目当前产物停留在软件、数据和开发期统计验证范围。
