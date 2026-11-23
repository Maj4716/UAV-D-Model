# 各向异性快速评估：L3 引擎校准

- 配对工况：229
- 工况—部件对：15343
- 运行时间：0.169 s
- 引擎差距状态：`proxy_teacher_gap_detected`

## 配对差距

- 加权概率 MSE（兼容字段 weighted_brier）：0.042075
- weighted ECE：0.051398
- weighted MAE：0.052771
- weighted RMSE：0.205123
- Spearman：0.5873167469824749
- false-unreached：0.07654200390734021
- false-unreached Wilson 上界：0.08100901459570002
- 最差单元：neg_x::large_front（RMSE=0.307541）

该层只量化快速代理与当前软件教师的差距，不是外部物理校准。
