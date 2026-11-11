# R06p 0.50 ms 终止时长扩展提交说明

不要压缩。只上传整个 `uav_hf_r06p_0800_terminal_horizon_0p50` 文件夹，选择 `r06p_0800_terminal_horizon_0p50.k` 作为 head/input deck。

硬性资源参数：1 个节点、恰好 32 个 MPP ranks、`ls-dyna_R15_mmp_d`、head memory 512 MB、
child memory 256 MB、命令行参数留空、walltime 12:00:00。提交预览必须包含 `--ntasks-per-node=32`。

使用全新工作目录 `/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06p_0800_terminal_horizon_0p50`。
本轮从 R06o 精确回传 deck 分支；模型、材料、接触、自适应 SPH、TSSFAC=0.05、DT2MS=0 均冻结。
唯一模型因素是终止时刻 0.04→0.50 ms；同时按预注册计划放宽输出频率以控制数据量。

循环保护仍为 829,930。若未到 0.50 ms 即达到循环上限或失败，不要修改参数重启，直接下载完整目录。
无论完成或失败，保留 `.k`、`mes*`、`glstat`、`matsum`、`ncforc`、`binout*`、所有 d3plot/d3part、
stdout/stderr、Slurm 脚本和资源记录。之后只需发送回传路径。

本轮是响应获取，不是物理验证，也不是仿真引擎或毁伤树标定。
