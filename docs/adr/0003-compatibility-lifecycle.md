# ADR-0003：兼容读取与旧符号生命周期

- 状态：Accepted
- 日期：2026-10-16

## 分类

| 对象 | 状态 | 当前价值 | 退役条件 |
|---|---|---|---|
| legacy manifest/normalization/identity/ledger readers | compatibility-read-only | 读取已有外部归档 | 完成归档 inventory、提供迁移工具、一个发布周期零使用；新 writer 禁止生成旧格式 |
| `generate --record-refusal` | deprecated no-op | 保持旧脚本可运行 | 已发运行期警告；下一个 breaking release 删除 |
| `ordinal_projection` | opt-in research | 等待 KMFC codebook/阈值与物理标定 | 获批后版本化晋升，或一个明确周期无用户后单独退役 |
| `multitask_targets` | active opt-in | 外部训练 consumer interface | 不按 package 内 fan-in 判死代码 |
| evaluation/convergence/fragment/statistical audit modules | research diagnostic | 规划与只读诊断 | 不进入物理准入；按公开工具和报告使用量评估 |
| `tempered_prevalence_target` | experimental-dormant | 未来预注册稀有事件 policy 候选 | 一个发布周期无内部/文档外部调用后删除 |
| `NumericalError` | compatibility-reserved | 可能存在外部 refusal-code consumer | public-import/ledger inventory + breaking-release 窗口后删除 |
| `scientific_validation.example_scientific_validation_record` | retired duplicate | 无导出、无仓库调用，与已签入 JSON 模板重复 | 已按符号删除；`examples/scientific_validation_record.template.json` 是唯一模板来源，并由测试确认其 fail closed |
| 顶层 `schemas/*.json` 镜像 | external-copy candidate | 运行时不读取；可能被外部脚本引用 | 确认外部消费者并提供生成/parity 流程后再删除；不手工双写 |

## 规则

- 不注释整个 Python module；这会破坏导入和旧归档读取。
- 兼容代码必须是显式只读适配器，新归档只写当前格式。
- 退役按符号或适配器进行，并有测试、迁移路径和观察期证据。
- 文件化模板保持单一来源；不再为同一非准入占位结构维护无调用的 Python 复制。
- lifecycle 测试锁定 experimental-dormant 符号不在活动 package 调用路径上，并锁定 compatibility-reserved 异常的序列化 code/message/context。
- 任何研究/诊断模块的保留都不构成物理标定或研究准入。
