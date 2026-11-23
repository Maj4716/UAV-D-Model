# 文档索引

本项目是面向数据方法研究的合成毁伤数据生成器。它提供可追溯的随机仿真、条件接触采样、数据归档、训练视图、采样优化和软件验证能力。当前产物用于数据管线开发、采样方法比较和神经网络接口验证。

## 阅读顺序

1. [架构](ARCHITECTURE.md)：理解计算链、模块边界和身份体系。
2. [数据管线](DATA_PIPELINE.md)：理解样本怎样生成、归档、切分和加载。
3. [采样与神经网络](SAMPLING_AND_ML.md)：理解双臂采样、训练输入、标签与评价。
4. [各向异性快速评估](ANISOTROPIC_FAST_ASSESSMENT.md)：独立运行生成器、暴露代理和既有教师残差诊断。
5. [破片方向模型卡](FRAGMENT_DIRECTION_MODEL.md)：查看当前方向假设、候选模型和晋升门。
6. [高斜角单侧非贯穿界](HIGH_OBLIQUITY_NONPERFORATION_BOUND.md)：查看 v3 形状约束、证明范围和物理边界。
7. [研究发现](RESEARCH_FINDINGS.md)：查看当前数据层实验支持的结论。
8. [验证](VALIDATION.md)：运行测试并核对数据包和机器证据。
9. [决策记录](DECISIONS.md)：查看长期有效的架构与研究决策。

工程术语见 [GLOSSARY.md](GLOSSARY.md)。安装和最短运行路径见根目录 [README.md](../README.md)。

## 当前范围

当前版本为 `0.11.0.dev0`。项目维护以下能力：

- 破片、侵彻、冲击波、部件响应和毁伤树的软件教师计算；
- 首次装甲接触条件下的来向、瞄点、速度、距离和滚转联合采样；
- 球面入射 scrambled Sobol Reference 设计及完整序列来源记录；
- Reference 与覆盖增强数据流的分账、选择概率、设计权重和切分治理；
- Parquet/NPZ/JSONL 归档、严格加载、训练视图和 PyTorch 数据接口；
- 可选重复级部件状态轨迹、离线树重算和跨重复数的命中概率目标；
- 零质量、条件正值、域删失、饱和、部件覆盖、采样效率和 Phase 0 多任务可学习性分析。

目标、材料、响应阈值和毁伤语义保留其验证状态。当前神经网络数据包的角色是 `development_ready`，不包含密封测试集。

## 稳定入口

| 对象 | 路径 |
|---|---|
| Python 包 | `damage_engine/` |
| JSON Schema | `schemas/` 与 `damage_engine/schemas/` |
| 示例输入 | `examples/` |
| 当前案例、目标、标签、采样与物理配置 | `configs/` |
| 原始源数据 | `data/source/` |
| 数据与审计工具 | `tools/` |
| 各向异性三级快评 | `tools/run_anisotropic_fast_assessment.py` 与 `docs/ANISOTROPIC_FAST_ASSESSMENT.md` |
| Phase 0 学习诊断 | `tools/run_phase0_learning_diagnostics.py` |
| 概率目标提取 | `tools/extract_hurdle_targets.py` |
| 离线树重算 | `tools/recompute_tree_outputs.py` |
| 最新球面 Sobol Reference 配置 | `configs/sampling/p0_sphere_entry_sobol_no_bottom_impact_state.v1.json` |
| idea2 优化与试生成报告 | `artifacts/validation/idea2_optimization_20261120/FINAL_REPORT.md` |
| 破片方向模型卡 | `docs/FRAGMENT_DIRECTION_MODEL.md` |
| 高斜角单侧非贯穿界 | `docs/HIGH_OBLIQUITY_NONPERFORATION_BOUND.md` |
| Stage 2 机理实验工具 | `docs/STAGE2_MECHANISM_EXPERIMENTS.md` |
| 测试 | `tests/` |
| 稳定验证证据 | `artifacts/validation/` |
| 当前神经网络开发数据包 | `artifacts/development_dataset/` |

## 文档维护规则

- README 只保存项目入口、安装、快速开始和导航。
- `docs/` 保存当前设计，不按日期追加平行报告。
- 可复核的最终结果写入 `artifacts/validation/<topic>/`，以 manifest 或验证报告作为入口。
- 运行日志、临时表、缓存和可再生中间池不作为长期文档。
- 影响接口或研究对象的决策写入 [DECISIONS.md](DECISIONS.md)。
