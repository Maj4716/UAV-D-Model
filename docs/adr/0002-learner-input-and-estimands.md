# ADR-0002：learner-input 与加权 estimand 的单一合同

- 状态：Accepted
- 日期：2026-10-16

## 背景

模型输入的名称、单位、来源、变换、张量字段和 manifest 描述曾散落在 writer、归档验证、Torch loader 与 training view。ESS 同时存在直接平方、缩放和允许零贡献的不同实现，容易混淆统计含义。

## 决策

1. `learning_input_identity` 的 `ModelInputDefinition` 是当前 learner-visible 状态的单一版本化定义，拥有状态提取、固定场景身份与学习输入身份。
2. writer、归档验证、Torch 默认字段、reader axis/mask/metadata 映射和 training view 从该定义派生；legacy `case_physics` 仅作为显式只读兼容路径。
3. teacher、solver、目标拓扑与装甲几何保留在审计归档，但不得进入默认 learner view。
4. `estimands` 集中数值稳定原语，并显式区分：
   - finite-positive raw design-weight ESS；
   - 允许零质量的 soft-event contribution ESS；
   - task weight 与 minibatch probability 不属于这两个 estimand。
5. 归一化权重、Hájek/向量加权均值和分片 streaming moments 必须先按有限正权重最大值缩放；不得直接累加可能溢出的 raw weight。

## 后果

- 新增 learner input 必须提供完整的名称、单位、来源、变换、语义、张量与 manifest 定义。
- 更改默认 learner 字段属于版本化合同变更，不能作为普通重构静默发生。
- 零权重资格由 mask 表达；结构性零事件不能通过重采样或 loss 技巧伪装为覆盖。

## 验收

- writer 生成的 feature metadata、manifest 和 tensor axis 来自同一 definition。
- loader 默认字段只含该 definition 的 tensor/mask 与允许的标签/审计身份。
- 极端共同缩放下 ESS、normalized weight、weighted mean 和 streaming moments 不溢出且不变。
- 全零 soft-event contribution 返回 0；非法负贡献 fail closed。
