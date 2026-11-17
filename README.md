# Damage Dataset Engine

本项目是面向深度学习数据方法研究的合成毁伤数据生成与验证工具。当前主线只覆盖软件教师、条件接触采样、数据归档、训练接口和统计验证；不包含高保真标定流程，也不作为最终应用系统。

当前版本：`0.11.0.dev0`。

## 核心能力

- 破片、分层侵彻、冲击波、部件响应与毁伤树的软件计算；
- 面向装甲目标的首次正向接触采样，拒绝未命中、反向飞行、零速度与接触身份漂移；
- Reference 与 Coverage 双臂采样、选择概率、设计权重、域删失和拒绝分账；
- JSONL/Parquet/NPZ 数据归档、内容哈希、恢复、严格读取与 PyTorch 接口；
- 零质量、条件正响应、部件覆盖、树饱和、ESS 与数据泄漏审计。

软件测试通过代表实现与数据合同一致，不代表目标参数、毁伤阈值或合成标签已获得外部真实性验证。当前稳定数据包的角色是 `development_ready`。

## 环境与安装

项目要求 Python 3.11 及以上。当前验证环境为 `pytorch312-lmdt`：

```powershell
conda activate pytorch312-lmdt
python -m pip install -e ".[dataset,test,torch]"
```

核心依赖为 NumPy 与 jsonschema；数据归档、测试和 PyTorch 支持分别由可选依赖组提供。

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
  --sampling-spec configs/sampling/loitering_impact.fragment_only.v1.json `
  --label-spec configs/labels/loitering_impact.fragment_only.v1.json
```

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
- 引擎参数搜索没有产生可晋升候选，`selected_candidate_id=null`。
- 目标参数状态仍为 `unverified`；数据适合管线、模型、损失和采样方法开发。

稳定证据入口为 `artifacts/validation/manifest.json`。

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
