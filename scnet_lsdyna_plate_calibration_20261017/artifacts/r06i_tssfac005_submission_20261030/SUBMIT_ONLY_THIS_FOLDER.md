# R06i 终止式时间步细化提交说明

不要压缩。只上传整个 `uav_hf_r06i_0800_tssfac005` 文件夹，并在平台选择 `r06i_0800_tssfac005.k` 作为 head/input deck。

资源参数：1 个节点、32 MPP ranks、`ls-dyna_R15_mmp_d`、命令行参数留空。使用全新工作目录
`/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06i_0800_ts005`。

本轮唯一变量是 `CONTROL_TIMESTEP.TSSFAC: 0.1 -> 0.05`，`DT2MS=0`。当前模型继续从爆炸后破片初始状态开始，
不改变初始几何、速度、材料、接触、失效或输出设置。

这是自动减半的终止算例。回传后不会自动生成 TSSFAC=0.025；将根据结果选定时间步，或转入破片删除/材料失效/接触诊断。

完成后下载完整回传：`.k`、`d3hsp/o=`、`mes*`、`glstat`、`binout*`、完整 `d3plot*`、完整 `d3part*`、
stdout/stderr、Slurm 脚本和资源记录。之后只需把回传路径发给我。
