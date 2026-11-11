# R06h 直接文件夹提交说明

不要压缩。只上传整个 `uav_hf_r06h_0800_tssfac01` 文件夹，并在平台选择 `r06h_0800_tssfac01.k` 作为 head/input deck。

资源参数：1 个节点、32 MPP ranks、`ls-dyna_R15_mmp_d`、命令行参数留空。使用全新工作目录
`/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06h_0800_ts01`。

本轮唯一变量是 `CONTROL_TIMESTEP.TSSFAC: 0.2 -> 0.1`，`DT2MS=0`。当前模型已经从爆炸后的破片初始状态开始，
不再添加爆炸前飞行或起爆阶段，也不允许在这次时间步对照中改变初始状态。

完成后下载完整回传：`.k`、`d3hsp/o=`、`mes*`、`glstat`、`binout*`、完整 `d3plot*`、完整 `d3part*`、
stdout/stderr、Slurm 脚本和资源记录。如果平台返回一个路径，之后只需把那个路径发给我。
