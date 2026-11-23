# 数据集恢复与神经网络训练适用性最终报告

日期：2026-11-22  
结论对象：`damage-dataset-v2-16969411c388e086d2757f88`  
正式开发包：`artifacts/datasets/recovery_role_r8_r32_train2000_eval300_20261122/dataset_v2_development`

## 1. 最终判决

当前数据集已经适合训练和验证合成教师代理网络，但适用范围必须由冻结的逐任务准入合同约束：

- `p_hit`、`p_degraded`、`p_failed`：准入 64/67 个部件。
- `conditional_log1p_energy`、`conditional_log1p_hit_count`：准入 62/67 个部件。
- 全 67 部件联合任务：不适合，正例或条件观测支持仍不足。
- 软件与统计层：在上述准入范围内通过。
- 模型可学性层：通过；多种子学习曲线随训练量增加而改善，独立 NumPy 模型明显优于 train-only climatology。
- 物理层：仍为 `provisional`。Taylor–Shapiro 高斜角形状约束、装甲/部件响应参数和真实分布尚未完成外部校准。

sealed test 的 300 个工况保持零结果解码、零打开、零预测、零指标；本报告不包含测试集泛化性能。

## 2. 证据身份

| 对象 | 身份或哈希 |
|---|---|
| 源数据集 | `damage-dataset-074d300c1382eb8eee4cd87a` |
| dataset-v2 | `damage-dataset-v2-16969411c388e086d2757f88` |
| dataset-v2 manifest | `sha256:72fb6a13d747600a2ef78898b62264a57cb65f0104e587d6f1af6ea643d8cff6` |
| 冻结引擎代码 | `sha256:b041499ab0e5bfd1b80813779ccd8e6867afaf9d9cc6cba0a7ff29a622a8ad0d` |
| 最新开发审计 | `sha256:b11af730b39b65807dd723e426680f270b505465faa4cabc8e81f495ca551696` |
| task-admission@1.1 | `sha256:c5b4ff603ec29eeffae51ad0dcd7fd2a34f1c9f90917f79d33234d4c3f6810c3` |
| 独立 model-readiness | `sha256:ab0c37f0d69f8e2f65de66b3e4ff3726896fffcc4eb8b1d9cd5436e00a317622` |

## 3. 原失效链与原因追溯

旧数据集的问题不是网络结构或数据拆分先导致的，而是监督标签在仿真与提取链中坍塌：

1. 高斜角装甲事件大量越过旧模型登记的 80° 适用域；旧开发集 affected replicate fraction 均值为 0.9104，300 个工况中只有 2 个达到 ε=0.01 完整门。
2. R32 下 affected fraction 的最小非零步长为 1/32=0.03125，因此 ε=0.01 实际要求工况内零 OOD。
3. 代表性 OOD 中约 96.6% 来自装甲；旧引擎将任一装甲 OOD 扩散为该重复全部 67 个部件未知。
4. 标签提取器再把三个概率 mask 做全局 AND，导致 validation 的 `p_hit` 监督消失。
5. 忽略旧 mask 后，全部 67 个部件都曾出现非零支持，说明根因是删失传播与 mask 语义，而非真实“完全无命中”。

修复大面积删失后，剩余问题转为参考分布下的稀有部件支持：采样为 outcome-blind scrambled Sobol，五个方向区内采用均匀立体角，未设置部件定向 geometry strata。内部、小截面或边缘位置部件不会自动获得足量正例；train 使用 R8 又使极低概率事件更容易得到零成功计数。

## 4. 已完成的机制与数据合同修改

### 4.1 Taylor–Shapiro 与高斜角 v3

- 仿真角度路径已迁移到版本化 Taylor–Shapiro 角度/弹道极限方法。
- 高斜角 v3 使用相同潜变量分位下的单调非贯穿边界：只有当边界条件足以证明不贯穿时才解除 OOD；无法证明的事件继续 fail closed。
- 该机制不推断反跳方向，也不把边界外插当作经过实测验证的物理真值。
- 配置：`configs/physics/loitering_impact.taylor_shapiro.obliquity_bound.v3.json`。
- case：`configs/cases/loitering_impact.fragment_only.taylor_shapiro.obliquity_bound.v3.json`。
- 证据边界：`docs/HIGH_OBLIQUITY_NONPERFORATION_BOUND.md`。

相同 60 工况、R8 的 v2/v3 配对 pilot 显示：affected fraction 均值由 0.8278 降至 0.0361，epsilon-complete 由 8.89% 升至 88.89%，三个概率头观测率均升至 99.7% 以上。这证明软件监督可用性恢复；它不证明外部物理精度。

### 4.2 组件局部删失与目标特异 mask

- OOD 影响改为逐装甲、逐部件记录，并生成逐部件概率识别区间。
- 部件自身 OOD 只影响该部件状态及相应下游路径；装甲 OOD 在缺少反跳方向模型时保留严格全局 fallback。
- `p_hit`、`p_degraded`、`p_failed` 使用各自识别区间与 mask；兼容字段 `label_mask` 仅为三者交集。
- 条件能量与条件命中数在条件集合可能受 OOD 改变时关闭点监督。
- 基线损失、验证指标和模型选择均使用目标特异 mask 与冻结 task-admission。

### 4.3 重复预算与数值稳健性

- train 使用 R8 以增加唯一输入覆盖；validation/test 使用 R32 以降低评估标签的 Monte Carlo 噪声。
- Box–Cox 逆变换前按 train-only 编码范围投影，避免网络预测越过可逆数值域；投影次数和比例进入验证报告。
- sealed test 标签不进入核心开发数组，所有工具在 split 判定后跳过 test 结果解码。

## 5. 正式数据集统计

采样配置：`configs/sampling/p0_sphere_entry_continuous_direction_recovery_train2000_eval300_role_r8_r32.v1.json`。

| 指标 | 结果 |
|---|---:|
| train / validation / sealed test | 2000 / 300 / 300 |
| 有效 / 拒绝工况 | 2600 / 0 |
| 输入完整率 | 100% |
| `p_hit` 开发观测率 | 99.6853% |
| `p_degraded` 开发观测率 | 99.6204% |
| `p_failed` 开发观测率 | 99.5204% |
| epsilon-complete 开发工况率 | 92.6522% |
| affected replicate fraction 均值 | 2.4918% |
| 设计权重范围 | 1.0–1.0 |
| ESS 比例 | 1.0 |
| Holm 缺失性诊断显著项 | 0 / 222 |
| sealed-test 结果解码 / 打开 | 0 / 0 |

三个概率识别区间宽度的中位数、P90、P95、P99 均为 0；均值依次为 `3.2994e-4`、`4.0045e-4`、`5.6019e-4`。剩余删失已经从主导性问题降为局部监督缺口。

## 6. 逐任务准入

准入仅使用 train/validation 标签支持，不使用验证性能或 test 结果。

| 任务 | 准入部件 | 拒绝部件 |
|---|---:|---|
| `p_hit` | 64 | `011, 019, 020` |
| `p_degraded` | 64 | `011, 019, 020` |
| `p_failed` | 64 | `011, 019, 020` |
| `conditional_log1p_energy` | 62 | `002, 010, 011, 019, 020` |
| `conditional_log1p_hit_count` | 62 | `002, 010, 011, 019, 020` |

概率门为 train 正/负例各不少于 20、validation 正/负例各不少于 5；条件门为 train 观测不少于 20、validation 观测不少于 5。所有被拒任务仍保留原始标签，不能参与当前损失、验证汇总或能力声明。

## 7. PyTorch 五任务基线

产物：`dataset_v2_development/torch_hurdle_baseline_admission_v1_1/results.json`。每个训练比例使用 3 个固定种子；表中为 validation 均值 ± 总体标准差。

### 7.1 三概率头学习曲线

| train fraction | `p_hit` Brier | `p_degraded` Brier | `p_failed` Brier |
|---:|---:|---:|---:|
| 0.25 | 0.044673 ± 0.000834 | 0.034970 ± 0.000684 | 0.016716 ± 0.000566 |
| 0.50 | 0.041834 ± 0.000584 | 0.032785 ± 0.000538 | 0.015313 ± 0.000339 |
| 1.00 | 0.041156 ± 0.000429 | 0.032082 ± 0.000420 | 0.014961 ± 0.000178 |

三个概率头均随训练量增加单调改善，种子间离散度同时下降。冻结模型选中 `fraction-1.00-seed-43`，三个 Brier 为 0.040822、0.031717、0.014811；有序状态最大违例为 0。

### 7.2 条件任务学习曲线

MAE 在原始条件标签的 `log1p` 尺度上计算。

| train fraction | 条件能量 MAE | 条件命中数 MAE |
|---:|---:|---:|
| 0.25 | 0.887597 ± 0.015752 | 0.529635 ± 0.017926 |
| 0.50 | 0.797517 ± 0.011655 | 0.471443 ± 0.010893 |
| 1.00 | 0.745157 ± 0.004279 | 0.438510 ± 0.005896 |

冻结模型的条件能量/命中数 MAE 为 0.739834/0.431390。条件命中数 Box–Cox 逆变换投影率为 7.58%，已被显式记录；这属于数值域控制证据，不代表该尾部误差已经物理校准。

## 8. NumPy 独立 p_hit 基线与 climatology 比较

NumPy 产物：`dataset_v2_development/numpy_p_hit_baseline_admission_v1_1/results.json`。其 25%/50%/100% validation Brier 均值为 0.040556/0.037914/0.037287，随训练量单调下降；选中模型 Brier 为 0.037205。

独立工具从 `selected_model.npz` 重建 validation 预测，并用每个部件的 train-only 均值建立 climatology：

| 指标 | 模型 | train-only climatology | Brier skill |
|---|---:|---:|---:|
| micro Brier | 0.037205 | 0.064107 | 0.419645 |
| macro Brier | 0.037155 | 0.064000 | 0.419461 |

- 62/64 个概率准入部件的 skill>0。
- `component-022` 与 `component-021` 的 skill 分别为 -0.0252、-0.00313，是下一轮模型与采样诊断重点。
- 25%→100% 的 Brier 相对改善为 8.06%。
- 重建 Brier 与训练报告的绝对误差为 0；全部可学性门通过。

这组结果表明数据中存在可重复学习、并超越常数基线的输入—响应信号。它不提供 test 泛化误差，也不把 validation 结果转换成真实物理准确率。

## 9. 分层适用性结论

| 层级 | 判决 | 依据 |
|---|---|---|
| 数据与软件一致性 | 通过 | 2600/2600 有效；身份哈希闭合；目标 mask、数值逆变换和 sealed-test 合同已验证 |
| 统计训练适用性 | 有条件通过 | 概率 64、条件 62；ESS=1，观测率高；全 67 支持门未过 |
| 神经网络可学性 | 通过 | PyTorch 五任务学习曲线改善；NumPy p_hit 对 climatology 的 Brier skill≈0.42 |
| sealed-test 泛化 | 未评价 | test 零解码、零打开、零预测、零指标 |
| 物理可信度 | `provisional` | 高斜角形状约束和响应参数缺少外部实验/高保真校准 |
| 真实任务/安全使用 | 未准入 | 当前标签是未完成物理校准的合成教师输出 |

因此，当前数据集可用于固定目标、固定输入合同下的代理网络开发、消融、校准方法研究和训练管线验证；必须加载 task-admission@1.1，并将模型声明限制到对应部件任务。它不适合宣称全 67 部件能力，也不适合作为真实毁伤效能或自主控制决策的认证数据。

## 10. 下一步

下一版全覆盖规模配置已经冻结：

`configs/sampling/p0_sphere_entry_continuous_direction_recovery_train5000_eval2000_role_r8_r32.v1.json`

其预算为 5000 train×R8、2000 validation×R32、300 sealed test×R32，状态为 `provisional`，尚未执行。建议执行顺序：

1. 保留当前参考流，并新增 outcome-blind、pre-label 的稀有部件几何增强流，优先覆盖 `002/010/011/019/020`；通过重要性权重回到声明参考分布。
2. 在 5000/2000 数据上重新冻结逐任务准入，不沿用当前 64/62 名单。
3. 对 `component-021/022` 增加分层误差诊断，判断负 skill 来自采样稀疏、模型容量还是标签 MC 噪声。
4. 完成 Taylor–Shapiro 高斜角边界、装甲响应和部件阈值的外部实验或高保真校准后，再评估物理研究准入。
5. 只有在模型、超参数、准入任务和验收门全部冻结后，才按单次协议开启 sealed test。

