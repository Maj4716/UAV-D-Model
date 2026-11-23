# 数据生成链两处缺陷的修复与验证（2026-11-23）

来源：[`REVIEW_20261123.md`](../pipeline_expected_actual_20261122/REVIEW_20261123.md) §4.1、§4.2。决策记录：[`docs/DECISIONS.md`](../../../docs/DECISIONS.md) D021、D022。

两处修复都按版本开关实现：已有配置保持逐位不变，用来复现已归档的数据集；修正后的行为只通过新配置启用。

## 1. Taylor–Shapiro 偏转符号（D021）

| 项 | 内容 |
|---|---|
| 代码 | `damage_engine/fragments.py`：新增 `taylor_shapiro_deflection`（`atan(q·sin(φ₂−φ₁))`），`FragmentRingLayout.angle_convention` 按注册表分派；`taylor_shapiro_angles` 原样保留为 legacy |
| Schema | `damage_engine/schemas/physics_registry.schema.json`：`angle_convention` 增加 `body_plus_x_polar_deflection_toward_detonation_propagation` |
| 新配置 | `configs/physics/loitering_impact.taylor_shapiro.wave_directed.obliquity_bound.v4.json`、`configs/cases/loitering_impact.fragment_only.taylor_shapiro.wave_directed.obliquity_bound.v4.json` |
| 测试 | `tests/test_taylor_shapiro_direction.py`：Taylor 实验室系独立判据、与 legacy 精确镜像、v4 全部破片后倾且满足切向判据、v4 可执行、未知约定拒绝 |

验证（[`taylor_fix_verification.json`](taylor_fix_verification.json)，Python 3.12.13，即生成归档的环境）：

| 配置 | 弹型 | 符合 Taylor 方向的破片比例 | 名义轴向倾角均值 |
|---|---|---:|---:|
| v3（legacy） | small / medium / large | 0% / 0% / 0% | +5.53° / +5.87° / +5.14° |
| v4（修正） | small / medium / large | 100% / 100% / 100% | −5.53° / −5.87° / −5.14° |

legacy 端到端复现：用 v3 重跑 3 个归档 train 工况（R8 完整教师），summary、67 部件指标、毁伤树输出与归档结果完全相等。

## 2. 瞄准窗口切向基（D022）

| 项 | 内容 |
|---|---|
| 代码 | `damage_engine/continuous_direction_sampling.py`：新增 `reference_axis_tangent_basis`、`aim_tangent_basis` 分派，以及独立的 measure contract / 实现哈希；未声明 `aim_tangent_basis` 时行为与哈希不变 |
| Schema | `damage_engine/schemas/sampling_spec.schema.json`、`schemas/sampling_spec.schema.json`：增加 `contact_manifold.aim_tangent_basis` |
| 新配置 | `configs/sampling/p0_sphere_entry_continuous_direction_recovery_train2000_eval300_role_r8_r32.v2.json`（新基、新设计种子、新 split registry ID） |
| 测试 | `tests/test_continuous_direction_sampling.py`：归档哈希不变、与 `contact_pose.smoke.v1` 显式框架一致、v 恒朝上/朝前、镜像对称、合同身份、哈希不匹配与未知基拒绝、v2 规格加载 |

验证（[`aim_basis_fix_verification.json`](aim_basis_fix_verification.json)，同预算 2,600 个正式工况，只做几何，不运行教师）：

| 区域 | 旧基准入率 | 新基准入率 |
|---|---:|---:|
| pos_x | 77.4% | 100% |
| neg_x | 17.7% | 100% |
| pos_y | 42.2% | 100% |
| neg_y | 81.9% | 100% |
| top_cap | 83.0% | 92.2% |
| **合计** | **60.5%**（4,301 个候选） | **98.4%**（2,641 个候选） |

正式集 region TV（对等质量设计）从 0.2015 降到 0.0127；pos_x 与 neg_x 的准入率差从 59.7 个百分点（约 31 倍标准误）降到 0。

legacy 复现（[`legacy_reproduction_py312.json`](legacy_reproduction_py312.json)）：用归档的 SamplingSpec v1、base case 和 v3 物理重建计划，2,600 个正式工况与 `rounds/round-000/formal_cases.jsonl.gz` 逐位一致，数值最大差为 0。不比较的只有存储字段：`engine_code_hash`（任何代码改动都会变）和契约文档的内嵌/引用形式。用 Python 3.13 + numpy 2.1.3 重建时会出现 ≤8.9×10⁻¹⁵ 的末位差异，来自 numpy 版本，不来自本次修改。

## 3. 回归测试

在 `pytorch312-lmdt`（Python 3.12.13）环境运行完整 `tests/`：1,338 项全部通过，0 失败（[`full_test_suite.log`](full_test_suite.log)）。其中包括 legacy `taylor_ring` 固定种子输出的 SHA-256 逐位回归测试。

## 4. 与代理收敛研究的交叉核对

代理收敛研究（[`proxy_convergence_20261123`](../proxy_convergence_20261123/REPORT.md)）在本次修复之后，用它自己的两个 oracle 复核了 v3 是否保持不变。我又在 [`peer_checks/`](peer_checks/run_peer_checks.sh) 里重跑了这两个检查，并重跑了它的教师可复现性抽查。脚本会先备份该研究 `results/` 下原有的输出，跑完后恢复，所以那里的证据文件保持原样。

| 检查 | 环境 | 结果 |
|---|---|---|
| `check_reproduction.py`：az4/az4_s7777777/azall 对审查 v0/v1/v2，全部 2,300 工况 | Python 3.13.5 / numpy 2.1.3 | 全部逐位相同，最大差 0 |
| `check_final_module_consistency.py 60`：当前代理模块对已存运行，60 工况 | Python 3.13.5 / numpy 2.1.3 | 5 个配置全部逐位相同 |
| `check_teacher_equivalence.py 4`：完整教师重跑 12 个 train 工况，对比归档行 | **Python 3.12.13 / numpy 2.4.6（归档环境）** | **12/12 相同** |
| 同上 | Python 3.13.5 / numpy 2.1.3（默认 `python`） | 2/12 相同，10/12 不同；逐例结果与代理研究的 `teacher_equivalence_check.json` 完全一致 |

**结论：代理研究 §9.4 报告的“12 个工况中 10 个复现不了”，不是代码改动造成的，而是运行环境不同。** 归档的 `plan/frozen_hashes.json` 记录的生成环境是 Python 3.12.13、numpy 2.4.6（OpenBLAS）。默认 `python` 是 3.13.5、numpy 2.1.3（MKL），浮点结果有 10⁻¹⁵ 量级的末位差异。教师每个工况要做数千次 `rng.random() < p` 的分支判定，这点差异足以偶尔翻转一次临界分支，从而改变重复层面的命中与状态；聚合量仍在 MC 噪声内，这与该研究观察到的现象一致。在归档环境里，当前工作树（包含本次修复）逐位复现了全部 12 个归档教师行。

由此得到一条操作规则：凡是“重新生成再与归档逐位比较”的流程，都必须使用归档 `frozen_hashes.json` 中记录的解释器和 numpy 版本。代理本身是确定性计算，跨环境只有 10⁻¹⁵ 量级的差异，对代理研究的结论没有影响。

## 5. 尚未完成

- 用 v4 物理 + v2 SamplingSpec 重新生成正式数据集（需要运行 R8/R32 教师，计算量与原数据集相当）。
- 重新生成后，重跑 `pipeline_expected_actual` 分析，检查分层异质性与部件级标签。
- `fast_sampling_proxy.py` 自动继承新的方向约定（通过 `generate_fragments`），代理与新教师的对比应在新数据集上进行。
- `pipeline_expected_actual_20261122/work/mechanism_funnel/run_stratified_micro_audit.py` 是那份报告的冻结证据，它的 `_taylor_shapiro_nominal_directions` 写死了 legacy 的 `taylor_shapiro_angles`。审计 v4 数据前，要改为按 `layout.angle_convention` 分派，否则会把正确方向判成合同违例。

## 复现命令

在项目根目录、使用生成归档时的环境运行：

```bash
/d/ProgramData/Anaconda/envs/pytorch312-lmdt/python.exe artifacts/validation/data_generation_fixes_20261123/verify_taylor_fix.py
```

```bash
/d/ProgramData/Anaconda/envs/pytorch312-lmdt/python.exe artifacts/validation/data_generation_fixes_20261123/verify_aim_basis_fix.py
```

```bash
/d/ProgramData/Anaconda/envs/pytorch312-lmdt/python.exe artifacts/validation/data_generation_fixes_20261123/diff_legacy_case.py
```
