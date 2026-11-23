# 各向异性快速评估：L3 引擎校准

- 配对工况：27
- 工况—部件对：1809
- 运行时间：0.065 s
- 引擎差距状态：`proxy_teacher_gap_detected`

## 配对差距

- 加权概率 MSE（兼容字段 weighted_brier）：0.047052
- weighted ECE：0.058845
- weighted MAE：0.059479
- weighted RMSE：0.216916
- Spearman：0.6194231463955296
- false-unreached：0.09994015559545183
- false-unreached Wilson 上界：0.11525074237124155
- 最差单元：neg_x::large_front（RMSE=0.294795）

该层只量化快速代理与当前软件教师的差距，不是外部物理校准。
