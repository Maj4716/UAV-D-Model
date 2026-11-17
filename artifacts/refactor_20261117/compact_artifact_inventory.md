# 紧凑证据库存清单

日期：2026-11-17

## 收敛结果

`artifacts/validation/` 现仅保留 8 个当前软件级验证工件和 1 个总清单。8 个工件合计 105,701 bytes；路径、SHA-256、字节数、范围和状态见 `artifacts/validation/manifest.json`。

| 相对于 `artifacts/validation/` 的路径 | bytes | 状态 |
|---|---:|---|
| `engine/acceptance_summary.json` | 2,700 | 引擎 0.11.0.dev0 软件验收已完成，未获得科研准入 |
| `penetration/p012_cumulative_finalists_r128.json` | 20,829 | 仅用于软件调参，不构成物理标定 |
| `penetration/p012_selected_verification_r512.json` | 3,835 | 仅用于软件调参，不构成物理标定 |
| `penetration/p012_threshold_selection_r256.json` | 10,030 | 仅用于软件调参，不构成物理标定 |
| `sampling/PREREGISTRATION_V5.md` | 4,014 | V5 独立确认的冻结协议 |
| `sampling/risk_model.json` | 15,434 | V5 冻结使用的 V4 风险模型，字节一致性已核对 |
| `sampling/v5_proxy_aggregate.json` | 38,420 | 代理门通过 |
| `sampling/v5_teacher_aggregate.json` | 10,439 | 教师门失败 |

## V5 结论边界

- `v5_proxy_aggregate.json` 的 `overall_proxy_gate_pass=true`。
- `v5_teacher_aggregate.json` 的 `preregistered_teacher_gate.overall_pass=false`。
- 教师门未通过的三项事前准则为 `complete_domain_ci_noninferior_3pp`、`positive_component_count_ci_improved` 和 `component_support_gini_lower`。
- 因此 V5 只能表述为已完成独立确认但未通过教师准入，不得表述为已正式接纳的采样策略。

`sampling/risk_model.json` 从 `review_evidence/idea1_sampling_20261116/v4_sampling_risk_model.json` 按原字节保留；两者 SHA-256 均为 `a4c5022ef185cf48684e969262bd23c674afea966f80ac4a5aa64976699e21e0`，压缩时比对结果为一致。

## 已移除的冗余工件

| 范围 | 文件数 | bytes | 处置 |
|---|---:|---:|---|
| `validation/engine/` 旧诊断、性能和 0.10 历史验证 | 6 | 410,251 | 仅保留当前 0.11 `acceptance_summary.json` |
| `validation/proxy/` | 5 | 45,361 | 整目录移除 |
| `validation/sampling/` 单轮代理试验和开发重放 | 2 | 157,417 | 保留事前协议、冻结模型和两份最终聚合 |
| **合计** | **13** | **613,029** | 中间态已移除 |

## 版本库边界

- `artifacts/development_dataset/` 保留为本地开发数据包，已通过 `.gitignore` 排除，避免将大型可再生数据直接写入 Git 历史。
- `data/generated/` 和 `artifacts/experiments/` 用于未来可再生输出，同样不进入 Git。
- `artifacts/validation/` 只保留小型、可审查的当前证据；`artifacts/refactor_20261117/` 保留本次重构收据。
