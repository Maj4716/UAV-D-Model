# 通用 masked 数值接口复审（2026-10-15）

本补充仅检查两个匿名任务、两行人工数组的接口完整性，不读取正式领域标签的逐项分布，不执行仿真、训练、反向传播或采样。它不能证明现有正式数据已经触发下列问题，也不提供领域效能、标签分布或模型效果优化意见。

解释器：`D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe`；PyTorch `2.8.0+cu128`；NumPy `2.4.6`。复现退出码 0。

- [可重放脚本](D:/Desktop/UAV新/review_evidence/reaudit_20261015/generic_masked_loss_contract_repro.py)
- [完整 JSON](D:/Desktop/UAV新/review_evidence/reaudit_20261015/generic_masked_loss_contract_repro.json)

运行前后 `damage_engine` 源码树摘要完全相同：`sha256:4e767ff2c20a310d08b9cf8dad6de09988e75a8956f6eb18aa4af6d7aef298b3`。JSON 同时记录四个相关源码文件的前后摘要。

## 1. 确认：masked NaN 仍污染两种公开损失

位置：[system_learning.py:867](D:/Desktop/UAV新/damage_engine/system_learning.py:867)、[system_learning.py:910](D:/Desktop/UAV新/damage_engine/system_learning.py:910)、[system_learning.py:930](D:/Desktop/UAV新/damage_engine/system_learning.py:930)。

入口只检查 `targets[mask]` 有限，却对完整 targets 计算元素损失，再乘 mask。人工有效目标为 0.25、0.75；其余两个位置 mask=false。无效位置使用有限零占位时，两种损失均有限；仅把同一无效位置换成 NaN，两种损失都成功返回 NaN，没有异常。原因是计算后再乘零无法消除 NaN。

影响：不可用位置的表示改变了有效项的聚合结果，破坏 mask 契约。此为公共函数人工反例，不是当前正式标签含 NaN 的证明。

修订原则：无效项应在可能传播非有限值的数值运算之前被排除或隔离；返回值必须符合函数承诺的有限性。不能把真实零目标排除，也不能把未知目标改写成有效零监督。

验收：mask 外 NaN/有限占位替换不改变有效结果；mask 内 NaN 明确失败；真实零目标且 mask=true 仍参与；全 mask=false 明确失败。若未来增加梯度检查，应另行记录，本轮未执行反向传播。

## 2. 确认：有限正权重的聚合可以溢出并产生看似更好的零分数

位置：[system_learning.py:478](D:/Desktop/UAV新/damage_engine/system_learning.py:478)、[system_learning.py:484](D:/Desktop/UAV新/damage_engine/system_learning.py:484)、[system_learning.py:725](D:/Desktop/UAV新/damage_engine/system_learning.py:725)、[system_learning.py:754](D:/Desktop/UAV新/damage_engine/system_learning.py:754)。

两个权重均为有限正 float64 `1e308`，输入检查通过，权重和成为 infinity。人工常数基线返回两个 0，其 support 中的 weight sum 为 infinity；指标接口成功返回 Brier=0、log loss=0，而 ECE 为 NaN。所有预测固定为 0.5，并非人工目标本身，因此零分数不是实际吻合。

影响：单值有限性不保证聚合有限，非有限分母可能产生误导性有限结果。本例是 dtype 极端边界，不代表现有权重范围已触发。

修订原则：明确权重的可表示范围及聚合精度；检查累加量、分母和最终结果，而非仅检查原始单值。不可表示时应明确失败，不应把溢出分数作为有效评价。

验收：一般正权重结果不变；零权重按现有严格正值契约失败；有限输入引发聚合溢出时不返回有效分数；记录缩放不变性等通用数值不变量的验证边界。

## 3. 本轮通过的行为及证明边界

| 人工情况 | 两种 torch 损失 | 常数基线与公开指标 |
|---|---|---|
| mask=true 的真实零目标 | 有限结果 | 接受，零仍为真实目标 |
| mixed mask，有限占位 | 有限结果 | 接受 |
| mixed mask，仅无效位置 NaN | 返回 NaN，确认缺陷 | 正确筛选有效目标，返回有限主要分数 |
| 全 mask=false，有限或 NaN 占位 | 明确拒绝 | 明确拒绝 |
| 有效位置 NaN | 明确拒绝 | 明确拒绝 |
| case weight=0 | 明确拒绝 | 明确拒绝 |
| task weight=0 | 明确拒绝 | 不提供该参数，不适用 |

覆盖范围是 `system_learning.py` 的四个公开数值入口：两种 masked 损失、`fit_reference_prevalence_baseline` 和 `evaluate_system_probabilities`；指标内部的校准、soft average precision 随公共调用执行，本轮没有独立穷举这些内部算法。未重跑完整仓库测试。

## 4. outer case 与 replicate 口径

`system_learning.py:108-139` 对传入 `[case,task]` 轴上的可用外层行计数并累加 fractional targets；`effective_sample_size` 是权重诊断公式，不能解释为已证明的独立实验数量。`dataset.py:1954-1965` 的 `*_sample_count` 来自该外层案例内部的 `replicate_counts`，用于标记均值不确定性的可估计状态。两种数量不得相加、混称正例次数或据此声称独立性。

以上是源码口径核对，没有计算正式领域标签的非零、熵、稀缺性或按 split 的支持分布。soft target 不应仅因数值大于零便被无条件重新定义为 Bernoulli 正例；这要求独立且明确的数学对象定义，而非由存储 shape 推断。
