# FastSamplingProxy 数值收敛研究与晋升门复评

- **日期：** 2026-11-23
- **对象：** [`damage_engine/fast_sampling_proxy.py`](../../../damage_engine/fast_sampling_proxy.py)（本次改为 `conical-multigroup-expected-transport@0.4.0`）
- **数据：** [`recovery_role_r8_r32_train2000_eval300_20261122`](../../datasets/recovery_role_r8_r32_train2000_eval300_20261122) 的 2,000 train（R=8）+ 300 validation（R=32）开发工况。教师引擎 build `sha256:b041499a…`。sealed test 结果解码数为 0（经 `tools.evaluate_proxy_teacher_distribution.load_development_evidence`）。
- **上游：** [`REVIEW_20261123.md`](../pipeline_expected_actual_20261122/REVIEW_20261123.md) §3.2–§3.5
- **复算：** 全部脚本在 [`scripts/`](scripts/README.md)，结果在 `results/`，生成的表格见 [`results/report_tables.md`](results/report_tables.md)
- **证据边界：** 这里衡量的是代理与当前软件教师的一致程度，不代表物理真实。所有数字都建立在当前数据生成链之上；审查指出的两项数据生成缺陷（Taylor–Shapiro 偏转符号、瞄准窗口切向基）不在本任务范围内，修复后需要重跑。

---

## 0. 结论摘要

| # | 问题 | 结论 | 关键数字（全部 2,300 个开发工况） |
|---|---|---|---|
| 1 | 方位射线数 | 4 条/环远未收敛。每枚破片一条射线后，`p_hit` 均值偏差基本消失 | `p_hit` 均值比 0.504→0.978；Brier 0.0426→0.0075 |
| 2 | 收敛后剩余误差的来源 | 主要是**单次散布实现的采样噪声**，不是结构差距。对 K 次源散布实现取平均即可消除 | Brier(K)=a+b/K：a=0.0015，b=0.0059。教师 MC 噪声底 0.0013，系统性残差约 0.0002 |
| 3 | `layer_quadrature_samples`、`reach_probability_cutoff` | 对本目标无影响或影响很小（目标没有声明层叠结构，层叠交互数为 0） | q=4/16/64 结果逐位一致；cutoff=0 时 FU 1.65%→1.57% |
| 4 | `p_failed` 差距 | 原因确认是累积模型。改用教师的空间簇能量累积后基本闭合 | 全集 0.717→0.952；审查用的 173 例子集 0.615→0.935；failed/hit 0.170→0.257（教师 0.274） |
| 5 | 代理 OOD 质量约 7% | **新发现：** 这 7% 本身也是代理的产物。只要任一代表分支越界，旧规则就删失整个 packet；按教师的分位抽样语义只删失越界的概率质量后，降到 0.02% | 期望被拒接触数 10.8→0.030/工况（教师未解析撞击 0.031/重复）；装甲命中数代理/教师 0.898→1.004 |
| 6 | 晋升门（train R8 2,000 例，预声明配置 = 全破片射线） | **通过** | false-unreached 1.649%，Wilson 95% 上界 1.724% ≤ 3% |
| 7 | 推荐配置 | 代理用作教师期望时，使用新常量 `TEACHER_PREDICTOR_OPTIONS`（全破片射线、K=8、registry 累积、branch_probability 删失）。构造函数默认值保持不变，理由见 §8 | Brier `p_hit` 0.0023、`p_failed` 0.0005；FU 0.40%；0.79 s/工况（约为 az4 的 15 倍） |
| 8 | 运行环境 | 教师 Monte Carlo 对 Python/numpy 版本敏感，代理不敏感。在生成归档时的环境（Python 3.12.13 / numpy 2.4.6）下，当前代码逐位复现归档教师行；在本研究的环境（3.13.5 / numpy 2.1.3）下，抽查的 12 例中有 10 例在重复层面不同 | 代理在两个环境间最大差 2.6×10⁻¹²，未触及状态 0 次翻转，门结果完全相同（§9.4） |

---

## 1. 流程与可信度保障

### 1.1 并发修改

- 开工时 `fast_sampling_proxy.py` 的 sha256 为 `3b2bafbb…`（10:52 他人新增 `output_scope` 的版本），我的改动全部叠加在这个版本之上。改动前先用 [`snapshot_baseline.py`](scripts/snapshot_baseline.py) 存了未修改模块的完整输出（[`baseline/pre_edit_outputs.json`](baseline/pre_edit_outputs.json)）。
- 之后每次写入前都检查了 mtime，没有发现别人的并发修改。每个 sweep run 都记录了运行前后的模块哈希（`results/runs/*.json`），运行期间源文件若有变化，脚本会直接报错退出。
- `fragments.py` 在 11:15 被其他会话改过（新增 Taylor 符号约定）。由于 v3 registry 声明的是 legacy 约定，代理输出仍与审查 10:54 的 `proxy_variants.npz` 逐位一致（见 §1.3）。

### 1.2 晋升门预声明

[`GATE_DECLARATION.json`](GATE_DECLARATION.json) 写于 11:21:29，11:22:40 补了一次修订。两次都在本研究任何代理运行和教师侧指标计算之前；第一次 run 在 11:29:47 完成。修订内容是把门的实现对齐到原报告产生 7.56%/7.60% 时使用的 identified-pair 口径（`bootstrap_proxy_teacher.py::_metrics`），原 `exposure_audit` 的无掩码口径改作次要结果。
声明中如实写明：审查已经报告过全破片配置的 1.65%，所以这次声明并不对该数字盲。它的作用是防止用本次 sweep 的结果反过来挑选门配置。门配置固定为全破片射线、默认种子、单次实现；其余配置一律算作事后（post-hoc）结果。

### 1.3 数值 oracle

| 检查 | 结果 | 文件 |
|---|---|---|
| 改动后的模块（默认选项）在 40 个工况 × 4 种配置上复现改动前**全部**输出键 | 0 处不一致（只有 `schema_version` 变化，并新增 6 个键） | `results/backward_compat_check.json` |
| `az4`、`az4_s7777777`、`azall` 在 2,300 例上复现审查的 v0/v1/v2 | `lambda`、`p_deg`、`p_fail`、OOD 分数、域标志全部逐位一致 | `results/reproduction_check.json` |
| 在 domain-censoring 补丁之前跑出的 5 个 run，在最终模块下复算 60 例 | 逐位一致 | `results/final_module_consistency.json` |
| 审查 §3.4 的 173 例子集 | 0.533/0.887（`p_hit`）、0.583/0.962（`p_deg`）、0.477/0.615（`p_fail`）、0.170 vs 0.274（failed/hit）全部复现 | T6 |

---

## 2. 模块改动（全部为 opt-in，默认行为逐位不变）

| 选项 / 输出 | 说明 |
|---|---|
| `source_realizations=K`（默认 1） | 对 K 个独立源散布实现取平均。第 0 个实现使用历史种子，所以 K=1 时结果与历史逐位相同 |
| `damage_accumulation="registry"`（默认 `"independent_hits"`） | 按每个部件响应模型声明的 `damage_accumulation_model` 累积损伤。对 `spatial_cumulative_damage_energy`：在代表射线入射点上做 complete-link 聚类（直径 ≤ `damage_correlation_length_m`，与教师 `ComponentDamageAccumulator._spatial_clusters` 的支撑规则相同）；簇内成员各自独立地取“未到达 / 到达并穿透 / 到达并停止 / 越界”四个分支，先对到达成员的损伤能量**求和**，再代入 logistic；簇与簇之间独立。能量分布精确枚举，原子数超过 4,096 时在 0.1% 的 log1p 能量网格上合并。也支持 `independent_hits` 和 `cumulative_damage_energy` |
| `domain_censoring="branch_probability"`（默认 `"any_branch"`） | 教师每次接触只抽一个弹道极限分位数，越界时只拒绝这一个分位数。新选项只删失被拒绝的那部分分位质量，已解析的分支照常保留。被拒绝的部件接触仍计为命中，但不沉积能量，与教师一致 |
| `out_of_domain_mass_fraction`（新输出，同时写入 `proxy_domain_coverage`） | OOD 破片质量分数。`proxy_domain_coverage.comparison_quantity` 标明比较时用它；`proxy_domain_status` 为兼容保留，并附语义说明：它会随射线数和种子增长，不能用于比较 |
| `out_of_domain_packet_weight_by_role` | 按装甲 / 部件拆分的 OOD 权重 |
| `component_hit_probabilities`（新输出） | 每个实现的 `1−Π(1−reach)^weight` 再对实现取平均，是独立破片到达的并事件概率。Poisson 近似 `1−exp(−λ)` 系统性偏低（T1、T4） |
| `component_damage_cluster_counts` | 仅在 registry 模式下输出 |
| `TEACHER_PREDICTOR_OPTIONS`、`ONE_RAY_PER_FRAGMENT` | 推荐配置常量（只读 mapping） |

测试：[`tests/test_fast_sampling_proxy.py`](../../../tests/test_fast_sampling_proxy.py) 新增 22 个测试，覆盖以下内容：默认值不变；参数校验；K 次实现的平均恒等式；并事件概率的上下界；OOD 质量分数；complete-link 直径约束；簇能量分布与暴力枚举一致（rel 1e-12）；确定性情形下与教师 `ComponentDamageAccumulator` 一致（abs 1e-12）；registry 模式只改变状态输出；`independent_hits` 模型在 registry 模式下等于历史结果；branch_probability 的拒绝概率与教师逐次抽样一致（4,000 次抽样，3 个速度）；域内与历史结果逐位一致；部件接触被拒时计入命中但不计能量；无拒绝时与历史一致；预设常量。
`tests/test_fast_sampling_proxy.py` 和 `tests/test_fast_proxy_domain_and_layers.py` 共 36 个测试通过。代理的下游测试（exposure_field、fast_assessment、p0_statistics_admission、sensitivity_execution、taylor_shapiro_direction、anisotropic CLI）共 59 个也全部通过。CLI 测试需要 `--basetemp` 指向一个可写目录，这是本机 Windows 临时目录的权限问题，与本次改动无关。

---

## 3. 收敛研究

除特别说明外，指标都在全部 2,300 个开发工况上计算，只使用教师 hurdle 掩码为 identified 的部件对，**不按代理域标志筛选子集**。“比值”指代理均值 / 教师均值（pooled）。FU 为 false-unreached（代理 λ=0 而教师 `hit_indicator` 均值 > 0 的部件对比例）。“union”指 `component_hit_probabilities`，“Poisson”指 `1−exp(−λ)`。“vs ref”的参照是同一选项组合下 K=16 的 run，也就是代理自身的收敛参照，不使用教师信息。s/case 取自 [`benchmark_runtime.py`](scripts/benchmark_runtime.py)：每个配置在单线程进程里跑 60 例（每个弹型 20 例），10 个进程并发，宿主机 16 核。带 * 的数字取自 sweep worker 的计时（14 个进程并发），偏高。

### T1. 方位射线数（默认种子，K=1，历史累积）

| config | rays/case | s/case | p_hit 比 Poisson | p_hit 比 union | Brier Poisson | Brier union | train FU（Wilson 上界） | val FU | RMS Δp_hit vs ref | 未触及状态翻转 vs ref | OOD 质量均值 | any-OOD 工况 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `az2` | 24 | 0.059* | 0.307 | 0.307 | 0.0580 | 0.0580 | 9.69% (9.86%) | 11.55% | 0.2370 | 15.17% | 0.0667 | 629 |
| `az4`（历史默认） | 48 | 0.052 | 0.497 | 0.504 | 0.0423 | 0.0426 | 7.60% (7.75%) | 9.23% | 0.2022 | 12.38% | 0.0701 | 1094 |
| `az8` | 96 | 0.077 | 0.748 | 0.785 | 0.0205 | 0.0215 | 4.18% (4.29%) | 5.64% | 0.1404 | 8.01% | 0.0697 | 1752 |
| `az16` | 166 | 0.116 | 0.890 | 0.968 | 0.0063 | 0.0077 | 1.79% (1.87%) | 2.91% | 0.0771 | 4.86% | 0.0691 | 2114 |
| `azall`（每片一条） | 170 | 0.122 | 0.896 | 0.978 | 0.0060 | 0.0075 | 1.65% (1.72%) | 2.73% | 0.0752 | 4.61% | 0.0691 | 2124 |

- 收敛是单调的。az16 已经接近 azall：small 每环 10 枚、medium 每环 12 枚、large 每环 16 枚，在 16 条射线下只有 medium 和 large 的最后一环（18、28 枚）还被压缩。
- 总期望命中数几乎不随射线数变化，但 `p_hit` 均值随射线数单调上升。这与审查 §3.2 的凹函数机理一致：λ 集中在少数部件上时，Σ p_hit 会变小。
- **OOD 质量分数在各射线数下几乎不变（0.067–0.070），any-OOD 工况数却从 629 涨到 2,124。** 后者是离散化产物，前者才是可比的量。

### T2. 层积分样本数（全破片射线）

| config | layer_quadrature_samples | 层叠交互次数（全部工况） | Brier union | train FU | p_fail 比 |
|---|---:|---:|---:|---:|---:|
| `azall_q4` | 4 | 0 | 0.0075 | 1.65% | 0.717 |
| `azall` | 16 | 0 | 0.0075 | 1.65% | 0.717 |
| `azall_q64` | 64 | 0 | 0.0075 | 1.65% | 0.717 |

`target.fragment_only.v1.json` 的 69 个对象都是 `layer_stack_id: null`，所以这个参数对本数据集不起作用，三种设置的结果逐位一致。它的正确性另由 `test_fast_proxy_honors_declared_layer_stack_with_shared_penetration_kernel` 覆盖。`reach_probability_cutoff` 从 1e-5 改为 0（`azall_cut0`）时，train FU 从 1.65% 降到 1.57%，validation FU 从 2.73% 降到 2.45%，`p_hit` 比值、Brier 和 `p_fail` 不变（`results/convergence_tables.md`）；耗时没有单独做基准。默认值 1e-5 可以保留。

### T3. 源散布种子（K=1）

| 射线 | 种子数 | p_hit 比 union（最小–最大） | Brier union | train FU | any-OOD 工况 | OOD 质量均值 | 种子间 RMS Δp_hit（两两平均） | 种子间未触及状态翻转 |
|---|---:|---|---|---|---|---|---:|---:|
| az4 | 4 | 0.493–0.504 | 0.0425–0.0427 | 7.60%–7.74% | 1042–1094 | 0.0691–0.0714 | 0.0762 | 1.07% |
| azall | 4 | 0.974–0.978 | 0.0072–0.0077 | 1.65%–1.75% | 2099–2124 | 0.0685–0.0695 | 0.1094 | 2.63% |

聚合指标几乎不受种子影响，**单个部件对的预测却变动很大**（RMS 0.11）。原因在于单次散布实现只是一个确定性点预测：窄部件要么被某枚破片击中，要么完全不被击中。种子间方差 0.109²/2 = 0.0060，与 T4 拟合出的 b = 0.0059 吻合。审查 v1 观察到的“271 例域标志翻转”就是这一项方差的表现。

### T4. 源散布实现数 K（全破片射线）

| config | K | s/case | p_hit 比 union | Brier union [95% CI] | Brier Poisson | train FU（Wilson 上界） | val FU | RMS Δp_hit vs K16 | RMS Δp_fail vs K16（同选项） | any-OOD 工况 | OOD 质量均值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `azall` | 1 | 0.122 | 0.978 | 0.0075 [0.0072, 0.0078] | 0.0060 | 1.65% (1.72%) | 2.73% | 0.0752 | 0.0266 | 2124 | 0.0691 |
| `azall_k2` | 2 | 0.470* | 0.978 | 0.0046 [0.0044, 0.0047] | 0.0041 | 1.10% (1.17%) | 1.89% | 0.0519 | 0.0185 | 2184 | 0.0689 |
| `azall_k4` | 4 | 0.401 | 0.978 | 0.0030 [0.0029, 0.0032] | 0.0032 | 0.70% (0.76%) | 1.28% | 0.0338 | 0.0117 | 2225 | 0.0691 |
| `azall_k8` | 8 | 1.856* | 0.978 | 0.0023 [0.0022, 0.0024] | 0.0027 | 0.43% (0.47%) | 0.81% | 0.0197 | 0.0069 | 2255 | 0.0689 |
| `azall_k16` | 16 | 3.834* | 0.978 | 0.0020 [0.0019, 0.0021] | 0.0024 | 0.24% (0.27%) | 0.53% | — | — | 2270 | 0.0688 |
| `az4_k8` | 8 | 0.823* | 0.502 | 0.0399 [0.0385, 0.0412] | 0.0397 | 7.00% (7.14%) | 8.49% | 0.1953 | 0.0903 | 1443 | 0.0701 |
| `azall_reg_dom` | 1 | 0.131 | 0.983 | 0.0075 [0.0072, 0.0078] | 0.0059 | 1.61% (1.68%) | 2.67% | 0.0755 | 0.0338 | 2117 | 0.0002 |
| `azall_k4_reg_dom` | 4 | 0.426 | 0.984 | 0.0030 [0.0029, 0.0031] | 0.0031 | 0.67% (0.72%) | 1.27% | 0.0339 | 0.0149 | 2219 | 0.0002 |
| **`azall_k8_reg_dom`（推荐）** | 8 | 0.792 | 0.983 | 0.0023 [0.0022, 0.0024] | 0.0026 | 0.40% (0.44%) | 0.78% | 0.0198 | 0.0086 | 2251 | 0.0002 |
| `azall_k16_reg_dom` | 16 | 1.539 | 0.984 | 0.0019 [0.0018, 0.0020] | 0.0023 | 0.22% (0.25%) | 0.51% | — | — | 2263 | 0.0002 |

区间为工况聚类 bootstrap 95% 区间（2,000 次）。教师 Monte Carlo 噪声底 E[q̂(1−q̂)/(R−1)]：全集 0.0013，train R8 0.0014，validation R32 0.0004。

- **拟合 Brier(K) = a + b/K**（[`analyze_residuals.py`](scripts/analyze_residuals.py)）：推荐选项下 a = 0.00152，b = 0.0060。a 比教师噪声底只高 0.0002，也就是说，在每片一条射线、K→∞ 时，`p_hit` 的系统偏差平方只剩约 0.0002。K=8 时 Brier 0.0023 ≈ 教师噪声 0.0013 + 代理自身 MC 方差 0.0007 + 偏差 0.0002。
- 在 validation（R32）上，推荐配置的 Brier 为 0.0015，噪声底 0.0004；K=16 时为 0.0011。教师 `p_hit` 质量落在代理 < 0.01 处的比例：az4 51.8%，azall 6.4%，推荐配置 1.1%。
- K 只能消除散布噪声，不能修正方位压缩：`az4_k8` 的比值仍是 0.50。
- K≥4 时 union 优于 Poisson，而且 union 的比值（0.98）不随 K 变化，Poisson 的比值则从 0.90 慢慢升到 0.93。K=1 时 Poisson 的 Brier 反而更低，因为它把概率往下压，相当于对单次实现的噪声做了收缩。
- K 增大时，FU 下降的同时 unreached 对数也在减少（114,806→109,972）：在任一散布实现下可能被击中的部件，越来越多地被判为“可达”。

---

## 4. 损伤累积：空间簇能量求和

### T5. 历史逐撞击乘积 vs registry 空间簇

| config | p_deg 比 | p_fail 比 [95% CI] | Brier p_deg | Brier p_fail [95% CI] | failed/hit 代理 | failed/hit 教师 | degraded/hit 代理 | degraded/hit 教师 | RMS Δp_fail vs K16（同累积、同删失） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `az4` | 0.528 | 0.499 [0.488, 0.510] | 0.0333 | 0.0159 [0.0152, 0.0166] | 0.405 | 0.495 | 0.763 | 0.812 | 0.1234 |
| `az4_reg` | 0.532 | 0.606 [0.594, 0.618] | 0.0333 | 0.0147 [0.0140, 0.0154] | 0.491 | 0.495 | 0.768 | 0.812 | 0.1190 |
| `azall` | 0.972 | 0.717 [0.710, 0.724] | 0.0061 | 0.0056 [0.0054, 0.0059] | 0.281 | 0.398 | 0.689 | 0.764 | 0.0702 |
| `azall_reg` | 0.977 | **0.952** [0.947, 0.956] | 0.0059 | 0.0016 [0.0015, 0.0016] | 0.373 | 0.398 | 0.694 | 0.764 | 0.0333 |
| `azall_k8` | 0.971 | 0.717 [0.710, 0.724] | 0.0022 | 0.0050 [0.0048, 0.0052] | 0.269 | 0.375 | 0.712 | 0.757 | 0.0651 |
| `azall_k8_reg` | 0.977 | 0.955 [0.952, 0.957] | 0.0021 | 0.0005 [0.0005, 0.0005] | 0.354 | 0.375 | 0.716 | 0.757 | 0.0086 |
| `azall_k16` | 0.971 | 0.717 [0.710, 0.724] | 0.0020 | 0.0049 [0.0047, 0.0051] | 0.268 | 0.372 | 0.717 | 0.757 | 0.0648 |
| `azall_k16_reg` | 0.977 | 0.956 [0.954, 0.959] | 0.0018 | 0.0004 [0.0004, 0.0005] | 0.353 | 0.372 | 0.721 | 0.757 | — |
| **`azall_k8_reg_dom`** | 0.983 | **0.962** [0.959, 0.964] | 0.0020 | 0.0005 [0.0005, 0.0005] | 0.356 | 0.375 | 0.718 | 0.757 | 0.0086 |
| `azall_k16_reg_dom` | 0.983 | 0.963 [0.961, 0.966] | 0.0017 | 0.0004 [0.0004, 0.0004] | 0.355 | 0.372 | 0.723 | 0.757 | — |

条件比值只在“教师命中 > 0 且代理命中 > 0”的部件对上计算。代理命中的集合随配置变化，所以各行的教师列不完全相同。

### T6. 与审查 §3.4 的连续性（173 例三方完整子集）

| config | p_hit 比（Poisson） | p_hit 比（union） | p_deg 比 | p_fail 比 | failed/hit 代理（Poisson 分母，与审查同口径） | failed/hit 教师 |
|---|---:|---:|---:|---:|---:|---:|
| `az4` | 0.533 | 0.547 | 0.583 | 0.477 | 0.246 | 0.329 |
| `azall` | 0.887 | 0.979 | 0.962 | **0.615** | **0.170** | **0.274** |
| `azall_reg` | 0.887 | 0.979 | 0.981 | **0.935** | **0.257** | 0.274 |
| `azall_k16_reg` | 0.928 | 0.989 | 0.990 | 0.952 | 0.238 | 0.251 |
| `azall_k8_reg_dom` | 0.930 | 0.996 | 0.998 | 0.948 | 0.241 | 0.254 |

**解读：**

- 审查 §3.4 的机理预测得到验证。只把累积模型换成 registry，而且只改变状态输出（测试保证 λ、能量等暴露量逐位不变），`p_failed` 比值就从 0.615 升到 0.935（子集），全集从 0.717 升到 0.952。failed/hit 从 0.170 升到 0.257，教师为 0.274。`p_degraded` 只从 0.962 变到 0.981：degraded 的阈值低，簇求和对它影响小，与 Jensen 方向的分析一致。
- 按暴露水平分段（全集，`azall` vs `azall_reg`，`results/residual_metrics.json` 的 `p_failed_by_lambda_band`）：λ∈(2,5] 时，教师 `p_fail` 0.290，历史累积 0.192，registry 0.280；λ>5 时分别为 0.811、0.598、0.797。差距集中在多次命中的部件上，符合簇求和机理。
- 4 射线配置下 registry 也有改进（0.499→0.606），但方位压缩仍是主要误差。压缩 packet 的簇按平均权重计重数，只是近似。
- registry 累积的额外耗时约 7%（0.122→0.131 s/工况）。

---

## 5. 适用域：用 OOD 质量分数替代 any-OOD 标志

### T5b. 删失口径

| config | OOD 质量均值 | any-OOD 标志工况 | 代理期望被拒接触 / 工况 | 教师未解析撞击 / 重复 | 装甲命中 代理/教师 | 部件命中 代理/教师 | 吸收能量 代理/教师 | p_hit 比 union | Brier union | train FU | p_fail 比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `azall` | 0.0691 | 2124 | 10.810 | 0.031 | 0.898 | 0.996 | 0.918 | 0.978 | 0.0075 | 1.65% | 0.717 |
| `azall_dom` | 0.0002 | 2117 | 0.029 | 0.031 | 1.004 | 1.000 | 0.929 | 0.983 | 0.0075 | 1.61% | 0.726 |
| `azall_k8_reg` | 0.0689 | 2255 | 10.792 | 0.031 | 0.899 | 0.997 | 0.921 | 0.978 | 0.0023 | 0.43% | 0.955 |
| `azall_k8_reg_dom` | 0.0002 | 2251 | 0.030 | 0.031 | 1.006 | 1.001 | 0.931 | 0.983 | 0.0023 | 0.40% | 0.962 |

**机理：** v3 registry 给装甲（验证域上限 80°）和部件（上限 85°）都声明了 `pathwise_monotone_ballistic_limit_lower_bound` 高倾角界（到 89.9°）。在这个区间内，教师每次接触只抽一个弹道极限分位数：若抽到的极限低于到达速度，则拒绝（OOD，概率 p）；否则判为认证的不穿透。代理的旧做法是在中位数分位和 0.5·p 分位上各求一次“代表分支”，只要其中一个越界就抛出异常，**整个 packet 的剩余到达概率都被删失**，后续接触也随之被截断。结果是代理平均每工况删失约 10.8 次接触（OOD 质量约 7%），而教师平均每个重复只有 0.031 次未解析撞击。审查 §3.3 已经指出 any-OOD 标志依赖射线数；这里进一步表明，7% 的 OOD 质量本身也主要来自删失规则，并不是工况本身的属性。

改用 `domain_censoring="branch_probability"` 后：

- 代理期望被拒接触数为 0.030/工况，教师为 0.031/重复，量级一致。代理被拒质量 ≥ 0.01 枚破片的工况有 307 例（K=8），教师有任一未解析撞击的工况为 213 例。
- 代理被拒量与教师未解析撞击数的 Spearman 相关从 0.05–0.09 升到 0.44（K=8）。代理 OOD 质量与教师受影响重复比例的相关从 0.01 升到 0.44。
- 装甲命中数从 0.898 恢复到 1.004（相对教师）。被错误删失的 packet 原本会继续撞击后面的装甲。
- `p_hit`、FU 和 `p_fail` 小幅改善（0.978→0.983，1.65%→1.61%，0.955→0.962）。部件命中原本只受轻微影响：被删的 packet 多半停在装甲上。

**比较时的处理：**

- 代理输出新增 `out_of_domain_mass_fraction`，并在 `proxy_domain_coverage.comparison_quantity` 中标明比较量。`proxy_domain_status` 为兼容保留，附语义警告。
- 本研究的所有比较都在全部工况上计算，没有按标志筛选。在 `azall` 下按 OOD 质量分段（=0、≤0.05、≤0.1、≤0.2、>0.2），各段 `p_hit` 比值为 0.97–0.99、Brier 为 0.005–0.009，差别不大。在推荐配置下，2,255 个工况的 OOD 质量 ≤ 0.05，其余 45 例为 0，已经不需要按 OOD 质量分层（`results/domain_metrics.json`）。
- 即使在 branch_probability 下，any-OOD 标志仍有 2,251 例为真：哪怕只有 1e-10 的被拒质量也会触发它。这个标志已无信息量。

---

## 6. 毁伤树输出

### T7. 代理/教师均值比（教师侧无受影响重复的 2,087 例；筛选只用教师信息）

| output | 教师均值 | `az4` | `azall` | `azall_reg` | `azall_reg_dom` | `azall_k8_reg_dom` | `azall_k16_reg_dom` |
|---|---:|---:|---:|---:|---:|---:|---:|
| effect-c1 | 0.4301 | 0.757 | 0.971 | 0.989 | 0.989 | 0.980 | 0.981 |
| effect-c2 | 0.2060 | 0.373 | 0.394 | 0.951 | 0.951 | 0.950 | 0.954 |
| effect-f1 | 0.1773 | 0.454 | 0.672 | 0.949 | 0.960 | 0.949 | 0.948 |
| effect-k1 | 0.0321 | 0.277 | 0.308 | 0.807 | 0.814 | 0.848 | 0.849 |
| effect-k2 | 0.0118 | 0.247 | 0.426 | 0.788 | 0.802 | 0.810 | 0.811 |
| effect-m1 | 0.2121 | 0.543 | 0.832 | 0.854 | 0.861 | 0.865 | 0.860 |
| effect-m2 | 0.5703 | 0.564 | 0.981 | 0.993 | 0.994 | 0.983 | 0.982 |

审查 §3.5 的两个根因在这里都得到独立验证。方位离散化主要影响由 degraded 叶子构成的输出（c1、m2：从 0.76/0.56 升到 0.97/0.98）。累积模型主要影响由 failed 叶子构成的输出（c2：0.394→0.951，f1：0.672→0.949）。k1、k2（各含 1 个 failed 叶子）和 m1 仍有 13–19% 的低估，见 §9.1。

---

## 7. 晋升门复评（train R8 2,000 例，同一引擎 build 的正式数据集）

### T8

| 状态 | config | identified 对 | 代理 unreached 对 | 其中教师为正 | FU | Wilson 95% 上界 | 工况聚类 bootstrap 95% | 门 ≤ 3% |
|---|---|---:|---:|---:|---:|---:|---|---|
| **预声明** | `azall` | 133,708 | 114,806 | 1,893 | **1.649%** | **1.724%** | [1.56, 1.74] | **PASS** |
| 事后 | `azall_reg_dom` | 133,708 | 114,731 | 1,842 | 1.605% | 1.680% | [1.52, 1.69] | PASS |
| 事后 | `azall_k8_reg_dom`（推荐） | 133,708 | 109,972 | 438 | 0.398% | 0.437% | [0.36, 0.44] | PASS |

- 门评估的前提条件全部满足：教师 code hash 与声明一致（`b041499a…`）；声明配置的 λ 与审查 v2 在 2,300 例上逐位一致；声明配置的 run（11:29:47）晚于声明时间（11:21:29）。
- 次要口径（`damage_engine.exposure_audit`，不加掩码）：1,899/115,027 = 1.651%，Wilson 上界 1.726%。
- 补充（不参与判定）：validation R32 为 2.728%，Wilson 上界 2.983%，离 3% 很近。原因是 R32 下教师能检出更低的命中概率，同一个代理零点更容易被证伪。推荐配置在 validation 上为 0.78%。
- 预声明配置分层（train）：large 1.81%，medium 1.88%，small 1.28%；按区域，pos_x 最高，为 2.22%（Wilson 上界 2.40%），pos_y 最低，为 0.57%。所有层的 Wilson 上界都低于 3%。
- 与原报告用 45 例 pilot（不同 build `987c0d4d…`）得出的 7.56% 相比：默认 4 射线配置在本数据集上为 7.60%（T1），两者一致，说明原结论“不晋升”只对 4 射线成立。
- 这个门检验的是“代理零点被教师证伪的比例”。通过它并不意味着代理零点是结构零（`structural_zero_claim: False` 维持不变）。

**判定：** 按预声明的配置和口径，代理在 train R8 上通过 false-unreached 晋升门。

---

## 8. 推荐默认配置

| 用途 | 推荐配置 | 代价（s/工况） | 主要指标（全集） |
|---|---|---:|---|
| 代理作为教师的“期望”（对比报告、门、分布预测） | `FastSamplingProxy(engine, **TEACHER_PREDICTOR_OPTIONS)`，即全破片射线、K=8、`damage_accumulation="registry"`、`domain_censoring="branch_probability"`；命中概率读 `component_hit_probabilities` | 0.79（约为 az4 的 15 倍，也只是一次 R8 教师运行的一小部分：本机抽查教师约 13 s/工况，未做严格基准） | `p_hit` 比 0.983、Brier 0.0023；`p_deg` 0.983；`p_fail` 0.962、Brier 0.0005；FU 0.40%；OOD 质量 0.0002 |
| 大规模候选筛选（暴露场等，成本敏感） | 至少使用 `azimuths_per_ring=ONE_RAY_PER_FRAGMENT` 和 `domain_censoring="branch_probability"`（K=1） | 0.13（az4 的 2.5 倍） | `p_hit` 比 0.983、FU 1.61%；去掉了 2 倍的 `p_hit` 偏差和 7% 的虚假 OOD 质量 |
| 需要更低噪声的参照 | 同第一行，K=16 | 1.54 | Brier 0.0019；FU 0.22% |

选 K=8 的理由：它位于 Brier(K) 曲线的拐点附近。K 从 8 加到 16，Brier 只降 0.0004（`p_hit`）和 0.0001（`p_fail`），耗时却翻倍。

**构造函数默认值本次没有修改。** 研究结果足以支持把 `azimuths_per_ring` 和 `domain_censoring` 的默认值改成上表第二行，但没有改，原因有三：

1. 仓库里有 14 处构造代理的调用（约 11 个模块和工具，如 `damage_engine/sensitivity_execution.py:255`、`tools/build_exposure_field.py:111`、`tools/run_x1_x2_exposure_experiment.py:87`），都没有传 `damage_accumulation` 或 `domain_censoring`。翻转这两个默认值会改变它们的全部输出，也会改变今天 11:27 被其他会话改过的 `damage_engine/fast_assessment.py` 收到的结果。另一方面，这些调用大多显式传入 `azimuths_per_ring`，命令行默认值多为 4（如 `tools/run_anisotropic_fast_assessment.py:685`），所以只改构造函数的射线数默认值并不能让这些工具收敛，还得同时改各工具自己的参数默认值。
2. 已归档的证据（如 `tools/refine_fast_sampling_proxy_resolution.py` 依赖的记录）将不再能复现。
3. 任务要求只修改代理模块、测试和本文件夹。

建议作为一项单独、协调后的变更来翻转默认值，并同时修改依赖 `proxy_domain_status` 做筛选的调用点（§10）。

---

## 9. 剩余差距与局限

### 9.1 推荐配置下仍存在的差距

- **`p_failed` −3.8%。** 在 60 个 failed 教师质量 ≥ 10 的部件上，比值分位数 p10/p50/p90 为 0.81/0.93/0.99。最差的是 component-061、063、064、066：它们的 `p_hit` 比值约 1.0，`p_fail` 比值却只有 0.76–0.79，都是被多枚破片命中的大部件。另外，吸收能量也低 7%（0.931）。**假设（未检验）：** 单层侵彻只用两个代表分位点（0.5p 和 p+0.5(1−p)）近似穿透后残速和沉积能量的条件分布；同一 logistic 在阈值下方是凸的，把能量分布收成两点会低估 E[σ]。这与审查 §3.4 末段对 +8.42% 沉积能量差距的猜测是同一个机理。可以用每分支多个分位点（类似 `layer_quadrature_samples`）的消融来检验。
- **`p_hit` −1.7%。** 在 64 个命中教师质量 ≥ 20 的部件上，比值分位数为 0.88/0.99/1.01。component-026 和 component-046 为 0.77–0.78。
- **毁伤树：** k1、k2、m1 为 0.81–0.87，其余输出 ≥ 0.95。k1、k2 各只含一个 failed 叶子，所以直接继承了对应部件的 `p_fail` 差距。
- 残差的分解（K=8）：`p_hit` Brier 0.0023 ≈ 教师噪声 0.0013 + 代理 MC 方差 0.0007 + 偏差约 0.0002。偏差部分已经比教师噪声小一个数量级。

### 9.2 近似和假设

- registry 累积使用代理的确定性射线位置（每个实现一组），并假设簇内成员的到达相互独立，与教师中不同破片相互独立一致。每个成员的分支能量只用两点代表。对压缩 packet（射线数少于全部破片），簇的重数取成员的平均权重，只是近似。
- branch_probability 删失沿用同一套代表分位点：被拒绝的分支把它的整段概率质量计为 OOD。单层情形下，被拒的总是低分位一侧（高倾角界的结构保证了这一点），所以拒绝概率与教师的抽样一致（测试 `test_branch_probability_matches_teacher_sampling_of_high_obliquity_bound`）。层叠结构下拒绝概率按格点计数，本目标没有层叠结构，这部分未经数据检验。
- 门判定与 FU 只依赖 λ 是否为 0，与累积和删失选项无关；K 会改变 λ 为 0 的集合。

### 9.3 统计

- Bootstrap 以工况为单位，条件于单一的 QMC 设计，没有覆盖设计间方差。
- 本研究在同一批开发工况上比较了 29 种配置。推荐配置是事后选出的，它的门结果只是次要证据；预声明的结果是 1.649%。
- 运行时间是本机 16 核、10 个进程并发时测得的。绝对值随机器变化，相对倍数更可靠。

### 9.4 运行环境与教师可复现性

[`check_teacher_equivalence.py`](scripts/check_teacher_equivalence.py) 用当前代码对 12 个 train 工况（每个弹型 4 例）重跑完整的 `DamageEngine`，再与归档行比较：

| 环境 | 与归档逐位一致 | 文件 |
|---|---:|---|
| Python 3.12.13 / numpy 2.4.6 + OpenBLAS（生成归档时的环境 `pytorch312-lmdt`，记录在 `plan/frozen_hashes.json`） | **12/12** | `results/teacher_equivalence_check.py312.json` |
| Python 3.13.5 / numpy 2.1.3 + MKL（本研究的 sweep 环境，默认 `python`） | 2/12 | `results/teacher_equivalence_check.py313.json`（`teacher_equivalence_check.json` 为同一环境的首次运行） |

本报告的初稿把 3.13 下的不一致写成了“代码漂移”，这个判断不对。代码并没有改变教师行为：numpy 版本带来的末位差异（审查会话测得 ≤ 8.9×10⁻¹⁵）会在逐事件的 Bernoulli 判定 `rng.random() < p` 上被放大，导致单个重复的分支翻转；聚合计数仍在 MC 噪声范围内。py312 的重跑是审查会话在 `artifacts/validation/data_generation_fixes_20261123/peer_checks/` 中用本研究的脚本完成的，结果原样复制到这里。

代理是确定性的期望传输，不含分支抽样，对环境不敏感。[`check_environment_sensitivity.py`](scripts/check_environment_sensitivity.py) 在 Python 3.12.13 下对 2,300 例重算了预声明配置 `azall`，与存储的 3.13 run 相比：λ 最大差 2.6×10⁻¹²，`p_fail` 最大差 2.3×10⁻¹²，λ=0 状态翻转 0 次，train R8 门结果完全相同（1.649%，Wilson 上界 1.724%）（`results/environment_sensitivity_azall_py312.json`）。

结论：本研究的门评估和全部代理指标都不受运行环境影响。今后凡是要把重新生成的教师结果与归档结果逐位比较，都必须使用生成归档时的环境。

---

## 10. 后续建议

1. **调用点改用 OOD 质量分数。** 以下位置仍按 `proxy_domain_status` 计数或筛选：`tools/run_proxy_learning_ablation.py:721`、`tools/evaluate_normalized_sampling_strategy.py:290`、`damage_engine/fast_assessment.py:625`。`tools/evaluate_proxy_teacher_distribution.py` 固定使用 `1−exp(−λ)`，也没有 K、累积和删失选项；`damage_engine/exposure_field.py` 只读取 λ 与能量。本任务按约束没有修改这些文件。
2. **协调翻转构造函数默认值**（§8 第二行），与第 1 项一起做。
3. **固定教师的运行环境**：凡是要与归档逐位比较的教师重跑，都使用 `pytorch312-lmdt`（Python 3.12.13 / numpy 2.4.6 + OpenBLAS）。数据集的 `plan/frozen_hashes.json` 已经记录了解释器、numpy 版本、numpy build hash 和 RNG 身份；建议在重跑工具里把它作为前置检查，环境不符时拒绝做逐位比较（§9.4）。
4. **检验 §9.1 的两点分支假设**：给单层侵彻加多分位积分作为 opt-in 选项，看 `p_fail` 的剩余 3.8% 和吸收能量的 7% 能否随之收敛。
5. Taylor 符号和瞄准基修复之后，用 `scripts/` 原样重跑整个研究（只需替换数据集路径）。

---

## 11. 文件清单

| 路径 | 内容 |
|---|---|
| `GATE_DECLARATION.json` | 晋升门预声明（含修订记录） |
| `baseline/pre_edit_outputs.json` | 未修改模块的输出快照（兼容性 oracle） |
| `baseline/development_case_order.json` | 开发工况顺序 |
| `baseline/fast_sampling_proxy.sweep_version.py.txt` | 第一轮 sweep 时的模块副本（补丁前） |
| `results/runs/*.npz/.json` | 29 个配置 × 2,300 工况的代理输出与来源信息（模块哈希、参数、耗时） |
| `results/convergence_metrics.json` | 全部指标 |
| `results/report_tables.md` | 本报告各表的原始生成结果 |
| `results/gate_evaluation.json` | 门评估（预声明 + 事后） |
| `results/domain_metrics.json`、`residual_metrics.json`、`runtime_benchmark.json` | 域、残差、耗时 |
| `results/backward_compat_check.json`、`reproduction_check.json`、`final_module_consistency.json` | 数值 oracle |
| `results/teacher_equivalence_check*.json`、`environment_sensitivity_azall_py312.json` | 教师可复现性抽查（py312 / py313）、代理与门结果的环境敏感性 |
| `scripts/` | 复算脚本，顺序见 [`scripts/README.md`](scripts/README.md) |
| `work/` | 可再生缓存（解码后的输入与教师数组），可删除 |
