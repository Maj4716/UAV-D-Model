# v2/v3 配对 pilot 接触域—教师域—标签—归档量化审计

## 结论

v3 的高斜率约束在同一组 45 个 development case 上大幅缩小了教师域截尾：受影响重复比例均值从 82.778% 降至 3.611%，配对差为 -79.167 个百分点（95% bootstrap CI -88.611 至 -68.611）。三个概率头的可观察率均提高到 99.7% 以上，识别区间均宽缩小 99.97% 以上。这证明 v3 修复对该配对 pilot 的“教师可识别性”有实质效果。

当前瓶颈已转为样本量和类别支持。v3 中每个概率头的 train eligible component 仍为 0/67，最终 admitted component 为 0/67；数据集也仍以 `provisional` 归档，v3 的明示原因是高斜率形状约束尚未物理验证。因此，v3 已达到软件与标签层面的开发分析可用性，未达到正式神经网络训练或物理研究准入条件。

## 配对设计

- control：`recovery_paired_control_v2_r8_20261122_retry1`
- bound：`recovery_paired_bound_v3_r8_20261122`
- 配对单元：45 个 train/validation case（30/15），每个 case 有 8 个重复。
- case ID、split assignment、learner input 张量逐项完全相同。
- 置信区间：固定 seed `2026112201`，10,000 次 case-level paired percentile bootstrap。
- test：15 个案例只核对密封元数据和全 false 占位 mask，结果行解码数为 0，未计算 test 指标。

## 逐环节实际比较

| 环节/指标 | v2 control | v3 bound | v3-v2 配对差 [95% CI] |
|---|---:|---:|---:|
| 受教师域截尾影响的重复 | 298/360 | 13/360 | 比例 -0.79167 [-0.88611, -0.68611] |
| 受影响 case | 41/45 | 6/45 | 描述统计 |
| epsilon-complete case，epsilon=0.01 | 4/45 (8.889%) | 40/45 (88.889%) | +0.80000 [0.68889, 0.91111] |
| `p_hit` 可观察率 | 16.949% | 99.934% | +0.82985 [0.73665, 0.90913] |
| `p_degraded` 可观察率 | 15.390% | 99.900% | +0.84511 [0.75357, 0.92537] |
| `p_failed` 可观察率 | 12.935% | 99.735% | +0.86799 [0.77844, 0.95257] |
| `p_hit` 识别区间均宽 | 0.713806 | 0.0000829 | -0.71372 [-0.80858, -0.61028] |
| `p_degraded` 识别区间均宽 | 0.725742 | 0.0001149 | -0.72563 [-0.81915, -0.62296] |
| `p_failed` 识别区间均宽 | 0.750374 | 0.0001979 | -0.75018 [-0.84798, -0.64565] |
| 条件 energy mask 观察率 | 1.957% | 10.978% | +0.09022 [0.07363, 0.10680] |
| 条件 hit-count mask 观察率 | 1.957% | 10.978% | +0.09022 [0.07363, 0.10746] |

epsilon-complete 按现有 audit 定义计算：三个概率头的全部 67 个 component 识别区间宽度均不超过 0.01。v2 的三个区间宽中位数均为 1，v3 均为 0；case-cluster bootstrap 的中位数差为 -1，各头 95% CI 上界分别为 -0.875、-0.875 和 -0.99297。

## 标签支持与训练准入

v3 的 development observed support 已接近完整：三个概率头均为每个 component 至少 44、中位 45、最多 45 个可观察 case。但是阳性支持仍稀疏：每个概率头的阳性支持为最小 0、中位 4、最大 32，9/67 component 完全没有阳性，64/67 低于 20。负性支持为最小 13、中位 41、最大 45，2/67 低于 20。

按冻结准入规则，v3 的三个概率头均为：

- train eligible component：0/67；
- validation evaluable component：9/67；
- admitted component：0/67。

v3 train epsilon-complete 门通过（27/30=90%），ESS 门通过；样本数与正/负支持门未通过，`formal_neural_training_ready=false`。

## 抽样、泄漏与归档

v2/v3 的抽样质量记录相同：98 个 candidate 中选 60 个，selection probability 为 1，design weight 均为 1，CV=0，ESS=60，ESS fraction=1。`candidate_set_integrity`、`declared_group_isolation`、`stream_role_split_isolation`、positivity、ESS 和 weight-CV 门全部通过，registry assignment 为 `assignments_verified`。该报告的 coverage 状态是 `descriptive_only_no_predeclared_coverage_protocol`，`coverage_guaranteed=false`。

两个 formal checkpoint 均为 `complete`，各含 4 个 shard、60 个 case；dataset-v2 validator 均为 `dataset_valid=true`。两个 dataset-v2 manifest 均为 `provisional`。test 记录访问计数均为 seen=15、decoded=0，test mask 全 false，test open count=0。

missingness KS 审计中，v2/v3 均有 234 个候选比较、30 个可评估比较，Holm 0.05 下显著数都为 0；最大 KS 从 0.60 降至 0.25。这项结果支持“当前未检出轴向性 missingness 偏移”，定性结论仍由只有 30 项可评估比较的功效边界限定。

## 证据与复现

- 可复现脚本：`work/teacher/analyze_teacher_pair.py`
- 完整机器可读结果：`work/teacher/teacher_pair_metrics.json`
- 输出 SHA-256：`77c1b251640e07fba5865ca44463ed66b4e668c21cf6906719af87ffb5072504`
- 关键输入：两个 pilot 目录中的 `development_audit.json`、`audits/sampling_quality.json`、`dataset_v2_development/manifest.json`、`dataset_v2_development/sealed_test.json`、`dataset_v2_development/core/*.npz`、`formal_dataset/checkpoint.json`。
- 编译检查：`python -m py_compile artifacts\validation\pipeline_expected_actual_20261122\work\teacher\analyze_teacher_pair.py`，exit 0。
- 执行：`python artifacts\validation\pipeline_expected_actual_20261122\work\teacher\analyze_teacher_pair.py`，exit 0。

bootstrap 区间表达这 45 个固定 development case 上的配对 case 差异。对独立 scrambled design 和现实物理误差的不确定性，需要额外实验层级才能估计。
