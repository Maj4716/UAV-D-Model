# 证据索引与版本关系

## 当前主要引用入口

| 主题 | 主证据 | 角色 |
|---|---|---|
| 11-17 仓库重构 | [`PROJECT_REFACTOR_REPORT.md`](../../refactor_20261117/PROJECT_REFACTOR_REPORT.md) | 当前 HEAD 的结构基线 |
| 球面入口与连续方向采样 | [`IMPLEMENTATION_LOG.md`](../sampling_sphere_v1/IMPLEMENTATION_LOG.md) | 采样与测度实现证据 |
| Idea2 全链实现 | [`FINAL_REPORT.md`](../idea2_full_implementation_20261120/FINAL_REPORT.md) | 11-20 实现与失败 pilot 的主报告 |
| dataset-v2 恢复 | [`FINAL_DATASET_READINESS_REPORT.md`](../dataset_recovery_20261122/FINAL_DATASET_READINESS_REPORT.md) | 旧 v3/v1 正式包的内部数据与学习证据 |
| 逐环节预期—实际分析 | [`REPORT.md`](../pipeline_expected_actual_20261122/REPORT.md) | 生成链量化主报告 |
| 独立复核 | [`REVIEW_20261123.md`](../pipeline_expected_actual_20261122/REVIEW_20261123.md) | 两项上游缺陷的发现证据 |
| 数据生成修复 | [`README.md`](../data_generation_fixes_20261123/README.md) | v4 方向与 SamplingSpec v2 切向基修复 |
| 快速代理收敛 | [`REPORT.md`](../proxy_convergence_20261123/REPORT.md) | 代理数值收敛、预声明门与推荐配置 |
| 各向异性三级快评 | [`FINAL_REPORT.md`](../anisotropic_fast_assessment_20261123_current256/FINAL_REPORT.md) | 当前快速评估主入口 |

`data_generation_fixes_20261123/README.md` §4 已取代 `proxy_convergence_20261123/REPORT.md` §9.4 对教师复现失败原因的判断：归档 Python 3.12.13/NumPy 2.4.6 环境为 12/12 逐重复一致，默认 Python 3.13.5/NumPy 2.1.3 环境为 2/12；差异来自运行环境浮点路径。该结论解决教师归档复现问题，`current256` L3 的代理/教师代码哈希仍未闭合。

## 历史与中间证据

| 路径 | 状态 |
|---|---|
| `artifacts/validation/penetration/` | 11-14 软件调参；不承担物理标定 |
| `artifacts/validation/engine/acceptance_summary.json` | 11-16 软件验收快照 |
| `artifacts/validation/phase0/` | 11-20 学习可分性诊断 |
| `artifacts/validation/anisotropic_fast_assessment_20261123*` 中非 `current256` 目录 | 中间迭代或历史缓存 |
| `artifacts/datasets/idea2_taylor_shapiro_continuous_r32_20261120*` | 早期 writer 和严格 v3 pilot |
| `artifacts/datasets/recovery_paired_*` | 11-22 配对诊断与重试 |
| `artifacts/experiments/` | 探索运行与机理诊断 |

## 需要更新的发现性入口

`artifacts/validation/manifest.json` 生成于 11 月 17 日，只索引 8 个紧凑工件、105,701 bytes。本次自清单排除报告目录后，`artifacts/validation/` 已有 432 个保留变化文件、226,915,863 bytes（216.40 MiB）；该 manifest 未覆盖 11 月 20—23 日新增证据。发布前应由本索引与 `current_change_inventory.csv` 生成新的根 manifest，并同步更新 `artifacts/README.md` 与 `docs/INDEX.md`。

