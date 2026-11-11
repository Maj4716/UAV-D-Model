# R06k 碎片负体积删除机制隔离提交说明

不要压缩。只上传整个 `uav_hf_r06k_0800_fragment_psfail_plate` 文件夹，并在平台选择 `r06k_0800_fragment_psfail_plate.k` 作为 head/input deck。

资源参数：1 个节点、32 MPP ranks、`ls-dyna_R15_mmp_d`、命令行参数留空。使用全新工作目录
`/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06k_0800_fpsfail`。

本轮保持 `TSSFAC=0.05`、`DT2MS=0`、碎片 MAT_015 `EROD=1`、板材 `EROD=0` 和 ERODING 接触不变。
唯一变量是 `CONTROL_SOLID.PSFAIL: 0 -> 910001`。集合 910001 只包含板材 part 1002，因此碎片 part 2002
被排除在 ERODING 接触自动启用的负体积删除准则之外。

这是机制隔离，不保证正常结束。时间步坍缩、严重畸变或求解器终止同样是需要完整保留的诊断证据，
不得修改模型以追求完成，也不得把所得响应作为物理标签、仿真引擎标定目标或毁伤树标定目标。

完成或中止后都下载完整回传：`.k`、`d3hsp/o=`、`mes*`、`glstat`、`binout*`、所有 `d3plot*`、所有
`d3part*`、stdout/stderr、Slurm 脚本和资源记录。之后只需把回传路径发给我。
