# V6 球面入口射线实施日志

## 2026-11-20 基线冻结

- Git HEAD：`ff2ef1f21c7d6bf73630bdb11331289278fb69dd`
- 工作区开始时为 clean。
- 在 `D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe` 中运行几何、SamplingSpec、CaseFactory、S-CONTACT 六组目标测试，全部通过。
- 系统 Python `D:\ProgramData\Anaconda\python.exe` 缺少可选 PyTorch，导致 `TorchDamageDataset` 单测出现环境失败；切换到项目的 PyTorch 环境后复核通过。
- 旧 1.4 SamplingSpec 哈希：`sha256:4589489789fbb0389f7da8ce1c5db5bd0162b70ba1933838e8edfe76b9c1062b`。
- 归一化目标哈希：`sha256:8f03659312d50ec0ae497b77eb01dd4f8612aa4c61aa9e177aa90fde090492a2`。

## 实施顺序

1. 公共包络球与物理射线重锚 API；
2. SamplingSpec 1.5 与 CaseFactory 分发；
3. 配对几何和教师等价验证；
4. 设计权重、耦合 bootstrap 与双门准入；
5. G1 通过后再启用 5D 降维和后续设计比较。

每个阶段均以失败测试开始，并保持旧 1.4 配置、哈希和冻结产物不变。

## 2026-11-20 Phase 1：球面重锚与 G1

- 新增公共保守包络球、球定义哈希、物理有向射线身份和球面上游交点重锚。
- 新增 SamplingSpec 1.5：保留旧 6D 布局用于配对控制，`standoff_quantile` 明确为 no-op；旧 1.4 哈希保持不变。
- 首次 G1 运行中，30 个几何对通过，但 54 个教师对有 36 个失败。定位结果表明：同一物理直线仅因纵向起点不同，首次接触坐标出现约 `8.88e-16 m` 的重构差；碎片射线从装甲边界出发时放大了该差异。
- `CompiledTarget.ray_hits` 现将世界坐标命中点规范到 `1e-12 m` 网格；`entry_t`、`exit_t`、物内路径长、BVH 候选和碰撞时间计算保持原定义。该改动小于 `1e-10 m` 几何门和 `1e-8 m` 冲击位置合同容差。
- 对照运行关闭坐标规范化后复现旧冻结计划哈希 `sha256:319165...7ec5`；最终坐标规范化计划哈希为 `sha256:8c684efb32dd7773150cd4525fffd39eb98206643dfa30d8a7a5b0fb2c20c31d`。逐字段差异仅为命中/起爆坐标的亚皮米级舍入及其派生 case hash。
- 最终源码版 G1：30 个几何对和 54 个教师对，共 84/84 通过。报告：`paired_equivalence.json`。

## 2026-11-20 Phase 2：删除采样 standoff

- 新增 SamplingSpec 1.6 和 `reduced_physical_ray_5d`：物理潜变量为 `approach_selector + aim_u + aim_v + speed + roll`；standoff 不再是设计潜变量。
- 1.5 继续冻结为 6D 配对控制；1.6 使用独立配置、分布身份和 split 身份。
- SamplingSpec 哈希：
  - 1.4：`sha256:4589489789fbb0389f7da8ce1c5db5bd0162b70ba1933838e8edfe76b9c1062b`
  - 1.5：`sha256:ca7bebc8e5af1b96f756f0b2fdbdfe05d118d87c01e5c04861d149eb52590024`
  - 1.6：`sha256:dcb28b9a46d9802653150800fe8232f24aff411e5d1439c23684a5baf1dca056`
- B1/B2 使用六来向、三弹型、三个教师 seed、每例 R=4；54/54 对的物理射线 ID、S-CONTACT、首次装甲对象、几何与教师科学输出全部一致。报告：`layout_reduction_equivalence.json`。

## 2026-11-20 Phase 3：联合设计与支持

- 新增 `randomized_lhs`、`scrambled_sobol`、`maximin_lhs` 三种严格开单位立方设计及一维 KS、二维投影 centered discrepancy、最近邻距离诊断。
- B3 固定 6 来向 × 3 弹型，每格 32 个候选，只改变 `aim_u/aim_v/speed/roll`，共比较 1728 个候选；不执行教师。
- 结果：
  - scrambled Sobol：576/576 S-CONTACT；
  - maximin LHS：575/576；
  - randomized LHS：570/576；
  - 三者均覆盖两个首次装甲对象和四个入射角区间。
- 冻结的 outcome-blind 字典序规则推荐 `scrambled_sobol`；报告：`design_sequence_comparison.json`。
- 新增支持审计与扩池计划：分别报告候选数、接触率、完整代理域率、Wilson 区间、最小纳入概率和零支持；只有几何可达与 proposal-overlap 证据均成立时才归类为预算不足。
- 新增 Reference-audit 固定样本量选择器：按 stratum × proxy-domain 分配整数配额，所有候选保持正纳入概率，proxy-partial 不再固定为零支持。

## 2026-11-20 统计合同与学习视图

- 新增 Operational / Reference / Coverage 三类版本化测度合同；Operational 未提供时显式记录 unavailable，不把 Reference 解释为作战频率。
- 设计权重、目标测度重要性权重和训练目标权重使用三个独立字段。去重混合纳入概率使用 `1 - product(1-rho_k)`；相关选择没有联合纳入概率时 fail closed。
- 新增候选级耦合、分层 exponential multiplier bootstrap。cluster 使用 `design_seed + candidate_group_id`；裸 physical-ray ID 仅作为几何身份，避免同一射线的不同弹型互相别名。
- Coverage admission 与 Reference replacement 为两个独立门；proxy-partial 零支持时 Reference replacement 固定失败。
- 新增 16 维 `loitering-impact-first-armor-state-input@1.0.0`：profile one-hot3、首次装甲命中点3、法向3、单位撞击方向3、命中速度1、body-y3。该视图要求部署侧先执行确定性射线求交，不读取教师毁伤标签。

## 最终验收

- 最终 `ENGINE_CODE_HASH`：`sha256:ddc794d0fccf4dd59d765f0a14c9ac5a3899c9d6cffd1deec7f75b9526cf8c52`。
- 最终 G1、B1/B2、B3 报告均记录上述引擎哈希且 `overall_pass=true`；三个报告的运行开始/结束源码哈希相同，`source_stable_during_run=true`。
- G1：`artifacts/validation/sampling_sphere_v1/paired_equivalence.json`，SHA-256 `d8694ef062345c0397f539d0d3bb22283417c44e33976535071b079ce1d47d42`。
- B1/B2：`artifacts/validation/sampling_sphere_v1/layout_reduction_equivalence.json`，SHA-256 `b4180447717a97301805faaf8a2aa1c3e384adc1a059ca5225fc481a0d1461c7`。
- B3：`artifacts/validation/sampling_sphere_v1/design_sequence_comparison.json`，SHA-256 `dada86169dd2df2bece200cdcab44cfcb117759a1c8c72a35f761216727400fa`。
- schema 镜像一致；SamplingSpec schema SHA-256：`6328ee21a5c819fc5a073580df067fb2b6657733502f1419e462f18020858ea5`。
- Python 语法编译、`git diff --check` 通过。
- 项目环境 `D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe` 下完整收集 1146 项测试，最终全量运行退出码为 0。

## 证据边界与下一门

- 当前通过的是软件一致性、几何/设计支持与统计合同门，不是破片、侵彻、部件脆弱性或冲击波的物理标定。
- 16 维 impact-state 视图已实现并通过单测，但尚未接入 `DatasetWriter` 或正式生产数据包。
- 尚未生成正式多独立 frame 的 V6 数据包，也未打开 sealed Reference test。
- 下一步应按至少三个独立 design frame 运行 scrambled Sobol 候选池、快速代理、Reference/Reference-audit/Coverage 选择和固定 R pilot；达到 Coverage admission 后仍需单独判定 Reference replacement。
