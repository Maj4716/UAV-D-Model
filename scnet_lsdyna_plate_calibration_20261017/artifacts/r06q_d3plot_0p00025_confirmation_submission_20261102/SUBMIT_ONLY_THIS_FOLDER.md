# R06q D3PLOT 0.00025 ms 测量确认提交说明

不要压缩。只上传整个 `uav_hf_r06q_0800_d3plot_0p00025_confirmation` 文件夹，选择 `r06q_0800_d3plot_0p00025_confirmation.k` 作为 head/input deck。

硬性资源参数：1 个节点、恰好 32 个 MPP ranks、`ls-dyna_R15_mmp_d`、head memory 512 MB、
child memory 256 MB、命令行参数留空、walltime 12:00:00。提交预览必须包含 `--ntasks-per-node=32`。

使用全新工作目录 `/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06q_0800_d3plot_0p00025_confirmation`。
本轮物理模型与 R06p 完全相同，只把 D3PLOT 输出间隔从 0.0005 改为 0.00025 ms。
终止时刻 0.50 ms、D3PART/ASCII 0.00025 ms、TSSFAC=0.05、DT2MS=0 和 829,930 循环保护均不变。

若提前达到循环上限或失败，不要修改参数重启，直接下载完整目录。保留 `.k`、`mes*`、`glstat`、
`matsum`、`ncforc`、`binout*`、所有 d3plot/d3part、stdout/stderr、Slurm 脚本和资源记录。
之后只需发送回传路径。

本轮是测量分辨率确认，不是物理验证，也不是仿真引擎或毁伤树标定。
