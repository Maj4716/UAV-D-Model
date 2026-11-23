# 恢复数据集角色重复预注册

冻结日期：2026-11-22（生产运行前）  
适用配置：`p0-sphere-entry-continuous-direction-recovery-train2000-eval300-role-r8-r32-v1`

## 目的与预算

- Reference train：2,000 个唯一工况，R=8。
- Reference validation：300 个唯一工况，R=32。
- Sealed Reference test：300 个唯一工况，R=32。
- 分割在结果生成前冻结；模型选择期间 test 结果 JSON 解码数必须为 0。

## 重复数决策

既有 E1 与新 v3 R8 pilot 都表明，p_hit / p_degraded 在 R8 的最坏 RMS 标准误差约为 0.09 / 0.08，只有 R32 同时通过 0.05 容差。因此 validation/test 使用 R32。Train 使用 R8，以增加固定预算下的唯一几何输入；训练保留 success/trial、Monte Carlo 标准误和设计权重，不将 R8 点估计宣称为单工况高精度真值。

## 准入规则

1. 数据身份、Schema、完整性校验通过；train/validation/test 无输入重复。
2. train 的 p_hit 目标观测率不低于 0.90；ESS fraction 不低于 0.50。
3. 每个任务头单独准入：train 正/负支持均不少于 20，validation 正/负支持均不少于 5。
4. 未过支持门的部件不进入该版本模型选择或性能声称；原始标签保留，供后续富集。
5. 学习曲线使用 25% / 50% / 100% train 和三个种子；只用 validation 选模。

## 预先声明的局限

pilot 尺度外推表明 2,000/300 不足以保证 67 个部件全部通过支持门。本运行的完成条件是产生一个可审计的、任务特异准入的神经网络训练候选，不会将未过门部件写成“已适合训练”。如果要全部件一次入场，当前 Reference 分布的点估计规模约为 train 5,000 / validation 2,000，或先验证稀有部件富集臂。

该准入只表示合成软件教师的统计/软件可学性。Taylor–Shapiro 方向候选、高斜角形状约束及响应参数仍未获得外部实验标定。
