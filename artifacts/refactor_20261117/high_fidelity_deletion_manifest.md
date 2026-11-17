# 高保真仿真内容删除清单

## 范围

本清单覆盖已退出当前项目路线的 SCNet/LS-DYNA 高保真仿真、数值资格审查、标定设计、求解回传、提交包、运行缓存和过程文档。当前项目保留软件级数据生成、采样、标签合同、学习接口和可复现验证。

删除前盘点共计 **2058 个现存文件、8,438,541,335 bytes（约 7.858 GiB）**，其中 **988 个 Git 跟踪路径**。根 `damage_engine/` 与根 `tests/` 没有导入高保真子项目；`pyproject.toml` 的 pytest 收集范围为 `tests/`，删除不改变数据层运行依赖。

## 删除对象

| 对象 | 文件数 | 删除前字节数 | Git 跟踪数 | 处理理由 |
|---|---:|---:|---:|---|
| `scnet_lsdyna_plate_calibration_20261017/` | 1857 | 6,307,686,175 | 949 | 专用 deck、提交、回传解析、数值资格、标定设计、测试、工件、第三方副本和原始结果均属于已停止路线 |
| `LS_DYNA_1030_233727/` | 112 | 2,130,196,985 | 0 | 本地求解器原始回传，含 `d3plot*`、`d3part*`、`binout*` 和 dump |
| `review_evidence/scnet_lsdyna_20261017/` | 4 | 208,176 | 4 | 仅服务于高保真候选证据与本地训练准入审计 |
| `paper-workspace/` | 2 | 12,485 | 2 | 两份文件均为 R04/SCNet 工件复现记录 |
| 根目录高保真报告与 handoff，共 33 个文件 | 33 | 241,294 | 33 | R01–R06 运行记录、失败诊断、提交与下载清单已经失去维护用途 |
| `.tmp/pytest-cache-r06*` 与 `.tmp/pytest-github-scnet` | 46 | 168,407 | 0 | 可再生测试缓存 |
| `.archify/dataflow-uav-simulation-dataset-20261117-090249/` | 4 | 27,813 | 0 | 未交付的旧路线可视化候选，需删除或按新结构重生成 |

根目录 33 个文件的完整路径保存在同目录的 `high_fidelity_deletion_manifest.json` 中。

## 引用修复

1. `README.md` 删除 SCNet/LS-DYNA 状态链接，并将项目定义为软件级数据生成与验证工具。
2. `PROJECT_STATUS_2026-11-11.md` 退役，改用 `docs/` 下的当前数据层状态文档。
3. `.gitignore` 删除 `/uav_hf_*/`、`/LS_DYNA_*/` 和全部高保真子目录专用规则；保留通用缓存和生成数据规则。
4. `GLOSSARY.md` 与 `docs/adr/0003-compatibility-lifecycle.md` 移除“等待高保真标定”的当前路线描述。
5. `examples/physics.loitering_impact.json` 将 LS-DYNA 专名改为来源中性的 software-only 作用域说明。
6. 归并 73–81 号数据层报告时删除 R06/E7 后续动作，仅保留采样、教师、标签、数据合同和学习接口结论。
7. `review_evidence/project_refactor_20261117/release_audit_before.json` 是清理前临时快照；完成后用清理后审计替换。

## 删除后验收

- 所有清单路径均不存在；
- 保留文档没有指向已删除路径；
- 仓库中不再存在 SCNet、LS-DYNA、`d3plot*`、`d3part*`、`binout*`、`glstat` 和 R04–R06 工作流的活动引用；
- 根 pytest 收集与全量测试通过；
- README 与当前状态文档统一描述数据层软件范围；
- 清理后 release audit 不再扫描已删除子树与可再生缓存。
