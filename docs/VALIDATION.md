# 验证与验收

## 1. 验证层次

项目把验证分成五层。每层回答不同问题，上一层通过不自动替代下一层。

| 层 | 核心问题 | 主要证据 |
|---|---|---|
| V1 输入与 Schema | 数据结构、单位、枚举和跨字段语义是否有效 | JSON Schema、语义校验、镜像一致性 |
| V2 软件实现 | 事件顺序、随机流、归档、恢复、掩码和统计量是否按合同工作 | 单元/集成/性质测试、编译检查 |
| V3 数据包 | 文件是否完整、可重算、无泄漏、无假零 | manifest、内容哈希、独立验证报告 |
| V4 采样研究 | 采样策略是否在预注册指标上改善 | Reference 对照、设计种子、bootstrap、ESS |
| V5 研究用途 | 参数、模型与标签语义的证据状态支持何种使用 | 参数注册表、模型覆盖、dataset policy |

当前数据包通过 V1–V3 的相应软件与数据检查，研究用途登记为 `development_ready`。V5 采样研究中代理门通过，但独立教师总门未通过。

## 2. 当前基线

重构前基线位于 `artifacts/refactor_20261117/pytest_before.xml`：

- 1,044 个测试；
- 0 failures；
- 0 errors；
- 0 skipped；
- 总测试时间 186.100 s。

该基线用于证明清理与结构重构前的代码状态。重构完成后应在同一环境重跑全套测试，并保存新的 JUnit 结果作为对照。

引擎研究摘要位于 `artifacts/validation/engine/acceptance_summary.json`。其中较早的 1,031 测试计数对应当时的实现快照；当前软件基线以 1,044 测试的 JUnit 为准。

## 3. 推荐验证环境

项目使用 Python 3.12 虚拟环境 `pytorch312-lmdt`。以下 PowerShell 命令把测试临时文件限制在仓库内：

```powershell
New-Item -ItemType Directory -Force -Path .tmp | Out-Null
$env:TEMP = (Resolve-Path .tmp).Path
$env:TMP = $env:TEMP
conda run -n pytorch312-lmdt python -m pip check
conda run -n pytorch312-lmdt python -m compileall -q damage_engine tools tests
conda run -n pytorch312-lmdt python -m pytest -p no:cacheprovider --basetemp .tmp/pytest-current --junitxml .tmp/pytest-current.xml
git diff --check
```

测试运行后核对 JUnit 的 `tests`、`failures`、`errors` 和 `skipped`，同时保存 Python、NumPy、PyArrow、PyTorch 与操作系统版本。

## 4. Schema 与合同验证

### 4.1 双镜像

`schemas/` 是仓库级合同入口，`damage_engine/schemas/` 是包内运行镜像。新增或修改 Schema 时：

1. 同步修改两处；
2. 比较文件集合和 SHA-256；
3. 运行 Schema 正反例测试；
4. 更新 schema/version 兼容矩阵；
5. 对旧数据执行显式迁移或拒绝加载。

### 4.2 失败关闭

以下情况应在教师执行或数据加载前拒绝：

- 未知字段、未知枚举或缺失必需字段；
- 非有限数值、单位混用或范围颠倒；
- 内容哈希、定义哈希或依赖快照不一致；
- SamplingSpec 与模型输入合同不兼容；
- 标签输出词汇与目标树不一致；
- 已提交分片被修改；
- 相同数据集 ID 指向不同内容。

## 5. 软件测试范围

当前测试套件覆盖：

- 事件时间线、破片顺序不变性和延迟提交；
- 分层侵彻、倾角适用域和能量闭合；
- 孔洞并集、连通开口和时变查询；
- 冲击波波前与正相位分段积分；
- 空间累计部件毁伤与完整链聚类；
- 毁伤树、依赖边界和有序投影；
- 接触流形、未命中/反向/零速拒绝；
- 稳定随机键、重复身份和输入内容绑定；
- 单体/分片归档、原子提交、恢复和篡改拒绝；
- 自适应纳入概率、Hájek 权重和 ESS；
- 训练视图、train-only 归一化和 PyTorch 加载；
- 掩码损失、权重分轴、校准与软标签指标；
- CLI 解析、空输入、无效 UTF-8 和错误定位。

新增功能至少提供一个正常路径、一个边界路径和一个拒绝路径。涉及随机算法时增加固定种子重放与顺序扰动测试。

## 6. 开发数据包验证

稳定数据包位于 `artifacts/development_dataset/`。独立验证命令为：

```powershell
conda run -n pytorch312-lmdt python tools/validate_neural_sampling_dataset.py `
  artifacts/development_dataset `
  --output .tmp/development_dataset_validation.json
```

当前 `validation_report.json` 状态为 `passed`，确认：

- 8 个源臂均严格包含 2,000 条选择观测；
- 16,000 条选择观测映射到 10,589 个唯一候选；
- 7,337 条完整域标签与 3,252 条域删失状态一致；
- 每个案例重新计算的 15 维输入与保存张量一致；
- 每个完整案例重新计算的教师张量与保存张量一致；
- 域删失响应均为 `NaN`；
- 纳入概率和设计权重通过检查；
- train/validation 候选组零交叉；
- 归一化统计只由 train 重算；
- manifest 声明的文件哈希与实际字节一致。

数据包内容哈希、来源身份与数据角色以 `artifacts/development_dataset/dataset_manifest.json` 为准。

## 7. 采样策略验证

稳定证据集合为：

| 文件 | 作用 |
|---|---|
| `artifacts/validation/sampling/PREREGISTRATION_V5.md` | 冻结假设、比较对象、指标和门槛 |
| `artifacts/validation/sampling/risk_model.json` | 冻结域完整与条件饱和风险模型 |
| `artifacts/validation/sampling/v5_proxy_aggregate.json` | 三个新设计种子的抽样前代理比较 |
| `artifacts/validation/sampling/v5_teacher_aggregate.json` | 2,000 对 2,000 教师比较与 bootstrap |

验收必须同时报告：

1. 总体响应或代理指标；
2. 域完整率；
3. 条件于完整域的响应指标；
4. 部件覆盖和支持不均衡；
5. 设计权重 ESS、最小纳入概率和权重尾部；
6. 预注册门的逐项结果。

V5 教师比较的 `overall_pass=false`。可进入生产流程的结论是“作为 Coverage 臂降低条件树饱和”，不是“替换 Reference”。

## 8. 引擎研究验证

`artifacts/validation/engine/acceptance_summary.json` 是当前软件教师的紧凑验收入口，汇总状态、性能、模型覆盖和研究用途。对应的中间搜索轨迹未作为长期工件保留。

当前候选选择结果为 `selected_candidate_id=null`，参数注册表保持原状态。模型覆盖检查显示树引用 44/67 个部件，目标 191 个参数的验证状态为 `unverified`，未解析压力—冲量面为 `RAW-PI-UNVERIFIED`。

## 9. 数据层发布门

一次开发数据发布需同时满足：

- 版本化输入、Schema、标签、树和代码身份已冻结；
- 接触合同对所有正式样本成立；
- 真实零、域删失和拒绝分离；
- 选择观测与唯一候选关系可追溯；
- 权重三轴分离并通过有限性与范围检查；
- 无候选组、重复组或规范化统计泄漏；
- 响应 mask 与 `NaN` 有严格一致性；
- Reference 与 Coverage 结果分账；
- 数据验证器从保存的案例重新计算输入与标签；
- 全套测试、编译检查、依赖检查和 `git diff --check` 通过；
- manifest、验证报告和 JUnit 结果随发布保存。

密封 test 的发布门在模型、训练规则和阈值冻结后执行。test 设计种子、案例与标签在最终评价前保持未使用状态。

## 10. 重构后复验记录

目录清理和引用修复后的最终复验记录如下：

| 项目 | 结果 |
|---|---|
| 环境 | `pytorch312-lmdt`；Python 3.12.13、NumPy 2.4.6、PyArrow 22.0.0、PyTorch 2.8.0+cu128 |
| 测试总数 | 1,044；用时 256.05 s |
| failures/errors/skipped | 0 / 0 / 0 |
| 数据包独立验证 | `passed`；10,589 个候选、7,337 个完整域标签、16,000 条选择观测 |
| Schema 镜像 | `tools/sync_schema_mirrors.py` 通过 |
| 依赖与编译 | `pip check`、`compileall` 通过 |
| CLI 合同 | 单案例校验、单案例执行、5,000 候选计划校验通过 |
| `git diff --check` | 通过 |
| 证据路径 | `artifacts/refactor_20261117/pytest_after.xml` 与 `artifacts/development_dataset/validation_report.json` |

重构前后测试数均为 1,044，且两次均为零失败、零错误、零跳过。当前 JUnit 是下一次提交前的软件回归基线。
