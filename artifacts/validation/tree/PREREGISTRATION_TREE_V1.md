# 离线损伤树候选预注册模板 V1

状态：`TEMPLATE — 在查看开发集候选输出之前复制、填写并冻结`

本模板用于分离候选定义、开发集选择、独立验证和领域语义审定。填写后的记录应保存为只读研究记录，并计算内容哈希；不得用验证集或测试集结果反向修改候选、阈值或排序规则。

## 1. 研究身份

- 预注册 ID：`<required>`
- dataset-v2 `manifest_hash`：`<required>`
- 输入契约哈希：`<required>`
- 标签规范 ID：`<required>`
- 代码哈希：`<required>`
- 填写人、时间与版本：`<required>`

## 2. 语义审定边界

- 领域审定人/机构：`<required for promotion; otherwise not_provided>`
- 审定材料与版本：`<required for promotion>`
- 审定范围：`<components, mechanisms, states, functional groupings and outputs>`
- `semantic_authority.status`：`approved | unreviewed | not_provided`

没有领域审定时，只允许使用 `identity-component fallback`：它逐项透传部件—机理—状态概率，状态必须为 `provisional_no_domain_semantics`。它不声明功能分组、关键度、失效传播或系统效应。当前树未引用的 23 个部件不得由代码或研究人员自行赋予权威功能语义。

## 3. 冻结候选集合

逐项填写候选文件、`candidate_id` 和 `definition_hash`。候选仅允许使用已实现且可验证单调性的门：

- `weighted_noisy_or`：非负权重；
- `k_of_n`：整数 `1 <= k <= n`；
- `two_additive_choquet`：归一化、单调的 2-additive capacity；
- `identity`：仅用于无语义的部件状态透传。

| candidate_id | definition_hash | gate family | semantic authority | 文件 |
|---|---|---|---|---|
| `<required>` | `<required>` | `<required>` | `<required>` | `<required>` |

冻结后是否允许增删候选：`否`。若候选集合发生变化，必须建立新的预注册 ID 并重新开始选择。

## 4. 独立设计种子

- 开发/选择 seed：`<required>`
- 独立验证 seed：`<required and different>`
- 两个 seed 对应的样本集合是否互斥：`是`
- 测试 seed：`<optional, never used for selection or threshold tuning>`
- 权重字段：`design_weight | target_measure_weight`，冻结值：`<required>`

## 5. 数值验收门

默认门在查看候选输出前冻结；如需变更，必须在本节说明先验依据。

| 指标 | 冻结规则 |
|---|---:|
| 中间区间 | `0.01 <= p <= 0.99` |
| 最低加权中间占比 | `>= 0.30` |
| 饱和定义 | `p >= 0.99` |
| 最高加权饱和占比 | `<= 0.25` |
| 零值定义 | `p <= 1e-6` |
| 最高加权零值占比 | `<= 0.70` |
| 最低加权二元熵（nats） | `<required; software smoke-test default is 0>` |

每个输出都必须逐项通过门；禁止用多个输出的平均值掩盖失败输出。

## 6. 冻结选择规则

只在开发 seed 上执行：

1. 淘汰任一输出未通过数值门的候选；
2. 优先选择已经领域审定的候选；未审定候选只能形成 provisional 数值诊断；
3. 在同一审定等级内，最大化“所有输出中的最低中间占比”；
4. 若并列，最大化各输出的平均加权二元熵；
5. 若仍并列，选择字典序最小的 `definition_hash`。

独立验证 seed 仅评估一次。验证结果不得用于改写候选或门限。

## 7. 推广判据

只有同时满足以下条件才可标记 `promotion_eligible=true`：

1. 候选按冻结规则在开发 seed 上被选中；
2. 该候选在独立验证 seed 上的每个输出通过全部数值门；
3. `semantic_authority.status=approved`，且审定范围覆盖该候选的全部输入、门和输出。

数值通过只证明该采样分布下的覆盖与非退化，不证明物理正确性或领域语义有效性。identity fallback 无论数值结果如何都保持 provisional。

## 8. 偏离记录

| 时间 | 偏离内容 | 原因 | 是否建立新预注册 |
|---|---|---|---|
| `<none at freeze>` |  |  |  |

## 9. 冻结签名

- 冻结文件 SHA-256：`<required>`
- 冻结人：`<required>`
- 冻结时间：`<required>`
- 领域审定签名/引用：`<required for promotion>`
