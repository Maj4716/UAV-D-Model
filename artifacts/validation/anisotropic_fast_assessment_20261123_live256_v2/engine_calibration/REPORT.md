# 各向异性快速评估：L3 引擎校准

- 配对工况：256
- 工况—部件对：17152
- 运行时间：0.068 s
- 引擎差距状态：`proxy_teacher_gap_detected`

## 配对差距

- weighted Brier：0.042600
- weighted ECE：0.052178
- weighted MAE：0.053478
- weighted RMSE：0.206398
- Spearman：0.5911664134834762
- false-unreached：0.07898519027682309
- false-unreached Wilson 上界：0.08326576771555434
- 最差单元：neg_x::large_front（RMSE=0.303352）

该层只量化快速代理与当前软件教师的差距，不是外部物理校准。
