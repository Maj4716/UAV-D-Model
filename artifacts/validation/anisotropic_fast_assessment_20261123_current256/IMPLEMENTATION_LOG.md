# 实现日志

## 目标

在已有 `FastSamplingProxy`、暴露场、教师审计和多保真代码上建立三级各向异性快速评估；默认避开正式数据集构建、随机教师执行和封存测试结果，使生成器问题与代理/仿真链路问题可分别诊断。

## 实施记录

1. 检索并确认已有基础模块：`fast_sampling_proxy.py`、`exposure_field.py`、`exposure_audit.py`、`multifidelity.py` 和 X1/X2 历史实验。
2. 对最小深接口、可扩展适配器和默认工作流三套设计进行并行评审，采用“两个公共 API + quick/calibrate 双入口 + exposure-only 计算范围”的组合。
3. 新增 `damage_engine.fast_assessment`：L1 经验各向异性椭球、声明/实现质量、选择与 IPW；L2 暴露场和适用域；L3 固定工况概率误差、秩相关、false-unreached 与控制变量效率。
4. 优化 `FastSamplingProxy.evaluate`：保留默认 full 合同，新增 exposure 范围并跳过部件状态与毁伤树计算；回归测试确认共享暴露字段完全一致。
5. 新增 CLI、配置、方法文档和测试；输出 assessment/report/manifest/hashes/command 五件套并拒绝覆盖。
6. 加固证据边界：quick 默认排除 test；calibrate 拒绝 test/sealed 行和已打开/解码 test 的元数据；核验 assessment/report 内部哈希；无角色证明的 NPZ 不再进入 L3。
7. 修正统计口径：混合 229 train + 27 validation 标记为 `development_train_validation`，并分别生成 train、validation 子集结果；软概率差的 `weighted_brier` 增加明确别名 `weighted_probability_mse`。
8. 完善单元质量门：生产配置要求每单元至少 20 例并检查 7 维满秩；稀疏单元只作告警。
9. 在当前源码身份下重跑 256 例 L1+L2；从既有开发归档提取角色化教师证据，保持 test 结果零解码；完成 development/train/validation 三套 L3。
10. 执行当前源码性能复测和历史 4,096/100,000 近似性复核；将缓存统计时间与实时射线计算时间分开报告。

## 文件变更

- 新增：`damage_engine/fast_assessment.py`
- 修改：`damage_engine/fast_sampling_proxy.py`
- 新增：`tools/run_anisotropic_fast_assessment.py`
- 新增：`configs/sampling/anisotropic_fast_assessment.v1.json`
- 新增：`docs/ANISOTROPIC_FAST_ASSESSMENT.md`
- 新增/修改测试：`tests/test_fast_assessment.py`、`tests/test_anisotropic_fast_assessment_cli.py`、`tests/test_fast_sampling_proxy.py`
- 更新入口：`README.md`、`docs/INDEX.md`、`tools/README.md`

## 已执行验收

- 定向回归：45 tests passed。
- 当前 quick：256 工况，test 输入纳入 0，教师执行 false，正式数据集构建 false，封存 test 结果解码 0。
- 当前 L3：256/229/27 三套角色口径均完成，教师执行 false，封存 test 结果解码 0。
- 字段一致性：exposure-only 与 full 的共享暴露字段完全相同。
- 最终自动验收结果记录在 `verification.json` 与 `QA_LOG.txt`。

