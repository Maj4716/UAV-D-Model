# 各向异性快速评估：L3 引擎校准

- 配对工况：150
- 工况—部件对：10050
- 运行时间：0.482 s
- 引擎差距状态：`proxy_teacher_gap_detected`

## 配对差距

- weighted Brier：0.071344
- weighted ECE：0.085240
- weighted MAE：0.086209
- weighted RMSE：0.267103
- Spearman：0.40666126364553956
- false-unreached：0.12845076225793162
- false-unreached Wilson 上界：0.1352537903295978
- 最差单元：neg_x::large_front（RMSE=0.345795）

该层只量化快速代理与当前软件教师的差距，不是外部物理校准。
