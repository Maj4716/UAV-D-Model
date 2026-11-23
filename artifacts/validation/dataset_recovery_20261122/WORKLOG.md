# 数据集训练适用性恢复工作日志

日期：2026-11-22  
范围：仅使用 train / validation 追溯与选模；既有和新建 sealed test 在模型冻结前保持零结果解码、零指标计算。

## 1. 失效链复核

1. v3 开发集 300 例中，`affected_replicate_fraction` 均值为 0.9104，中位数为 1；ε=0.01 完整样本为 2/300。
2. R=32 时非零 affected fraction 的最小步长为 1/32=0.03125，因此旧 ε 门实际等同于“没有任何 OOD 重复”。
3. 代表性分层审计的 OOD 事件中，装甲占 96.6%，部件占 3.4%；主要触发点是装甲 80° 高斜角适用域边界。
4. 旧引擎把任一 OOD 扩散到该重复的全部 67 个部件；旧标签提取器又要求全局 affected fraction 为零，并把 `p_hit/p_degraded/p_failed` 三个 mask 做 AND。
5. validation 无 `p_hit` 监督是 mask 坍塌，不是真实无命中。忽略旧 mask 后，train/validation 的 67 个部件均出现过 `p_hit>0`。

## 2. 科学决策

- 保留装甲 80°、部件 85°的已登记适用域；不把上限直接改成 89°，不做角度 clamp。
- OOD 路径终止继续作为 fail-closed 下端情景，不当作物理真值。
- 先修复目标特异删失与监督 mask；装甲 OOD 缺少反跳方向外包时继续对全部件使用严格 fallback。
- 新增高斜角机制只能作为独立版本。优先尝试“边界弹道极限的单调非贯穿证明”：只解析能由边界下界严格证明不贯穿的事件；无法证明的事件仍为 OOD。
- conditional energy / hit-count 在其分子或条件集合可能受 OOD 改变时关闭点监督。

## 3. 已完成修改

- 引擎记录 OOD 装甲/部件计数，并构造逐部件 `p_hit` 与状态概率识别界。
- 部件 OOD 的当前几何接触计入 `p_hit`；其状态和同射线下游仍保持删失。
- 标签合同升级为 `component-fragment-hurdle-targets-v2`，按每个目标自身区间宽度生成 mask。
- dataset-v2 新增 `p_hit_mask`、`p_degraded_mask`、`p_failed_mask`；旧 `label_mask` 仅保留为三者交集兼容字段。
- 条件正值目标在逐部件域影响不为零时 fail closed。
- baseline 的点损失、区间损失和验证指标分别使用目标特异 mask。

## 4. 高斜角 v3 与配对 pilot：已完成

- 新增版本化 Taylor–Shapiro 高斜角单调非贯穿边界：`configs/physics/loitering_impact.taylor_shapiro.obliquity_bound.v3.json`。
- 新增对应 case：`configs/cases/loitering_impact.fragment_only.taylor_shapiro.obliquity_bound.v3.json`。
- 该机制只在相同潜变量分位下能证明边界速度以内不贯穿时解除 OOD；无法证明的路径继续 fail closed。它不生成反跳方向，也不替代外部物理验证。
- 完成相同 60 个工况、R8 的 v2/v3 配对 pilot。affected fraction 均值由 0.8278 降至 0.0361，epsilon-complete 由 8.89% 升至 88.89%，三个概率头观测率均超过 99.7%。
- 配对证据记录于 `PAIRED_PILOT_REPORT.md`；规模与稀有支持分析记录于 `PILOT_SUPPORT_SCALING_AUDIT.md`。

## 5. 正式 2600 工况数据集：已完成

- 冻结采样配置：`configs/sampling/p0_sphere_entry_continuous_direction_recovery_train2000_eval300_role_r8_r32.v1.json`。
- 角色预算：train 2000×R8，validation 300×R32，sealed test 300×R32；总计 2600 个有效工况，拒绝 0。
- 源数据集 ID：`damage-dataset-074d300c1382eb8eee4cd87a`；dataset-v2 ID：`damage-dataset-v2-16969411c388e086d2757f88`。
- 冻结引擎源哈希：`sha256:b041499ab0e5bfd1b80813779ccd8e6867afaf9d9cc6cba0a7ff29a622a8ad0d`。
- 严格开发包：`artifacts/datasets/recovery_role_r8_r32_train2000_eval300_20261122/dataset_v2_development`。
- sealed test 的 300 条结果保持零解码、零打开、零指标计算。

## 6. 正式审计与逐任务准入：已完成

- 最新审计：`dataset_v2_development/development_audit.json`，hash `sha256:b11af730b39b65807dd723e426680f270b505465faa4cabc8e81f495ca551696`。
- 开发集三个概率头观测率为 99.6853%、99.6204%、99.5204%；epsilon-complete 为 92.6522%；affected fraction 均值为 2.4918%；ESS 比例为 1。
- 冻结 `dataset-v2-task-admission@1.1.0`，hash `sha256:c5b4ff603ec29eeffae51ad0dcd7fd2a34f1c9f90917f79d33234d4c3f6810c3`。
- 概率任务准入 64/67，拒绝 `component-011/019/020`。
- 条件能量与条件命中数任务准入 62/67，拒绝 `component-002/010/011/019/020`。

## 7. 两套独立基线与模型可学性：已完成

- PyTorch 五任务基线使用 3 个种子和 25%/50%/100% 学习曲线，最终产物位于 `dataset_v2_development/torch_hurdle_baseline_admission_v1_1`。
- 训练量从 25% 增至 100% 时，三概率头验证 Brier 均值分别由 0.044673/0.034970/0.016716 降至 0.041156/0.032082/0.014961；条件能量/命中数 MAE 由 0.887597/0.529635 降至 0.745157/0.438510。
- NumPy p_hit 基线位于 `dataset_v2_development/numpy_p_hit_baseline_admission_v1_1`；25%/50%/100% 验证 Brier 均值为 0.040556/0.037914/0.037287，选中模型为 0.037205。
- 独立 `model_readiness.json` 重建选中模型验证预测；相对 train-only climatology 的 micro/macro Brier skill 为 0.419645/0.419461，62/64 个准入部件 skill>0，全部可学性门通过。
- 两套基线均未使用 test 标签进行损失、指标或选择；独立可学性工具也未计算 test 预测。

## 8. 最终状态

- 软件一致性：通过当前测试与身份绑定检查。
- 统计训练适用性：概率任务限 64 部件、条件任务限 62 部件时通过；全 67 部件不通过。
- 模型可学性：validation 上有稳定学习曲线，并显著优于 train-only climatology。
- 物理可信度：仍为 `provisional`；Taylor–Shapiro 高斜角约束、目标响应参数与真实数据一致性尚未外部校准。
- 下一版全覆盖规模配置已冻结为 `configs/sampling/p0_sphere_entry_continuous_direction_recovery_train5000_eval2000_role_r8_r32.v1.json`，尚未执行。

