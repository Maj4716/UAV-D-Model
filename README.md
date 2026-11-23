# Damage Dataset Engine

本项目是面向深度学习数据方法研究的合成毁伤数据生成与验证工具。当前主线只覆盖软件教师、条件接触采样、数据归档、训练接口和统计验证；不包含高保真标定流程，也不作为最终应用系统。

当前版本：`0.11.0.dev0`。

## 核心能力

- 破片、分层侵彻、冲击波、部件响应与毁伤树的软件计算；
- 面向装甲目标的首次正向接触采样，拒绝未命中、反向飞行、零速度与接触身份漂移；
- Reference 与 Coverage 双臂采样、scrambled Sobol Reference 设计、选择概率、设计权重、域删失和拒绝分账；
- JSONL/Parquet/NPZ 数据归档、内容哈希、恢复、严格读取与 PyTorch 接口；
- 重复级部件状态轨迹、离线树重算和跨重复数的部件命中概率目标；
- 零质量、条件正响应、部件覆盖、树饱和、ESS、数据泄漏与多任务可学习性审计。

软件测试通过代表实现与数据合同一致，不代表目标参数、毁伤阈值或合成标签已获得外部真实性验证。当前稳定数据包的角色是 `development_ready`。

## 环境与安装

项目要求 Python 3.11 及以上。当前验证环境为 `pytorch312-lmdt`：

```powershell
conda activate pytorch312-lmdt
python -m pip install -e ".[dataset,sampling,test,torch]"
```

核心依赖为 NumPy 与 jsonschema；数据归档、采样设计、测试和 PyTorch 支持分别由 `dataset`、`sampling`、`test`、`torch` 可选依赖组提供。显式使用 scrambled Sobol 或 maximin LHS 的 SamplingSpec 需要安装 `.[sampling]`；未声明 `design_sequence` 的旧 randomized-LHS 路径不加载 SciPy。

## 快速验证

验证一组冻结输入：

```powershell
python -m damage_engine validate `
  --target configs/targets/target.fragment_only.v1.json `
  --case configs/cases/loitering_impact.fragment_only.v1.json `
  --physics configs/physics/loitering_impact.json
```

运行单案例并把结果写入可再生数据目录：

```powershell
New-Item -ItemType Directory -Force data/generated | Out-Null
python -m damage_engine simulate `
  --target configs/targets/target.fragment_only.v1.json `
  --case configs/cases/loitering_impact.fragment_only.v1.json `
  --physics configs/physics/loitering_impact.json `
  --output data/generated/single_case.json
```

只校验正式采样计划，不执行标签计算：

```powershell
python -m damage_engine sampler validate-plan `
  --target configs/targets/target.fragment_only.v1.json `
  --physics configs/physics/loitering_impact.json `
  --base-case configs/cases/loitering_impact.fragment_only.v1.json `
  --sampling-spec configs/sampling/p0_contact_reference.json `
  --label-spec configs/labels/loitering_impact.fragment_only.v1.json
```

该示例使用 1.4 接触位姿规范；`configs/sampling/loitering_impact.fragment_only.v1.json`
保留为 1.2 旧数据复现入口，不作为新计划模板。

## 当前开发数据包

本地开发数据位于 `artifacts/development_dataset/`：

- 16,000 条选择观测；
- 10,589 个去重候选；
- 7,337 条完整响应标签；
- 3,252 条域删失样本；
- train/validation 按独立设计种子隔离；
- 当前没有封存测试集。

独立重算式验证：

```powershell
python tools/validate_neural_sampling_dataset.py `
  artifacts/development_dataset `
  --output artifacts/development_dataset/validation_report.json
```

压缩案例与张量文件默认不进入普通 Git 历史；README、manifest、特征契约、验证报告和读取器可随代码审查。正式发布大数据时应使用专门的数据版本机制。

## 当前研究结论边界

- V5 Coverage 的代理门通过；独立软件教师总门 `overall_pass=false`，失败项见 `artifacts/validation/sampling/v5_teacher_aggregate.json`。
- Coverage 适合作为训练增强臂，不能替代 Reference，也不能用其原始频率代表参考总体发生率。
- 最新球面入射 Reference 配置使用 scrambled Sobol、五个主来向和首层装甲状态输入，并保留完整序列来源；它是状态为 `provisional` 的独立采样修订，未覆盖旧冻结配置。
- 该配置的 360 例试生成已完成，采样质量门通过；域完整率为 38.06%，R=4 派生 support 没有可判定负例，因此未晋升为稳定基线，数据包只用于管线与探索性调试。完整结论见 `artifacts/validation/idea2_optimization_20261120/FINAL_REPORT.md`。
- `p_hit` 是跨重复数的部件主目标；二值 support 由固定阈值和置信区间派生，区间跨阈值的样本不进入该二值任务。
- 权威树目标由逐重复树结果取均值得到；期望部件概率上的树结果只作辅助诊断。
- Phase 0 多任务诊断固定使用 validation-only 评价、三种训练比例、三个随机种子和两个输入视图，禁止访问 test split。
- 引擎参数搜索没有产生可晋升候选，`selected_candidate_id=null`。
- 目标参数状态仍为 `unverified`；数据适合管线、模型、损失和采样方法开发。

稳定证据入口为 `artifacts/validation/manifest.json`。

相关实现入口：

- `tools/run_anisotropic_fast_assessment.py` 与 `docs/ANISOTROPIC_FAST_ASSESSMENT.md`：不构建正式数据集的 L1 生成器、L2 方向暴露、L3 既有教师残差快评；
- `configs/sampling/p0_sphere_entry_sobol_no_bottom_impact_state.v1.json`：最新球面入射 Sobol Reference 配置；
- `tools/run_phase0_learning_diagnostics.py`：Phase 0 学习曲线与瓶颈诊断；
- `damage_engine/hurdle_targets.py` 与 `tools/extract_hurdle_targets.py`：概率主目标与批量提取；
- `tools/recompute_tree_outputs.py`：从重复级状态轨迹离线重算树输出；
- `docs/FRAGMENT_DIRECTION_MODEL.md`：破片方向假设、敏感性候选和晋升门。
- `artifacts/validation/idea2_optimization_20261120/FINAL_REPORT.md`：idea2 优化、试生成指标和训练适用性结论。

## 项目结构

| 路径 | 内容 |
|---|---|
| `damage_engine/` | 软件教师、采样、归档、训练视图与验证实现 |
| `configs/` | 当前案例、目标、物理、标签、采样和导入配置 |
| `data/source/` | 目标重建所需的原始来源表 |
| `data/generated/` | 本地可再生输出，默认忽略 |
| `artifacts/development_dataset/` | 当前神经网络开发数据包 |
| `artifacts/validation/` | 精简且带哈希的稳定机器证据 |
| `docs/` | 按主题归并的当前文档与决策 |
| `examples/` | 小型示例与探索性参数计划 |
| `schemas/` | 仓库级 JSON Schema 镜像 |
| `tests/` | 单元、集成、性质和回归测试 |
| `tools/` | 数据验证、诊断、基准和研究工具 |

## 文档

- [文档索引](docs/INDEX.md)
- [系统架构](docs/ARCHITECTURE.md)
- [数据管线](docs/DATA_PIPELINE.md)
- [采样与神经网络](docs/SAMPLING_AND_ML.md)
- [各向异性快速评估](docs/ANISOTROPIC_FAST_ASSESSMENT.md)
- [破片方向模型卡](docs/FRAGMENT_DIRECTION_MODEL.md)
- [研究发现](docs/RESEARCH_FINDINGS.md)
- [验证与验收](docs/VALIDATION.md)
- [架构与研究决策](docs/DECISIONS.md)
- [工程术语表](docs/GLOSSARY.md)

## 提交前检查

```powershell
python tools/sync_schema_mirrors.py
python -m compileall -q damage_engine tools tests
python -m pip check
python -m pytest -p no:cacheprovider
git diff --check
```

完整验收结果见 [项目清理与重构验收记录](artifacts/refactor_20261117/PROJECT_REFACTOR_REPORT.md)，并在 [验证与验收](docs/VALIDATION.md) 中维护单一状态记录。
