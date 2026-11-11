# R06j 碎片侵蚀机制隔离提交说明

不要压缩。只上传整个 `uav_hf_r06j_0800_fragment_erod1` 文件夹，并在平台选择 `r06j_0800_fragment_erod1.k` 作为 head/input deck。

资源参数：1 个节点、32 MPP ranks、`ls-dyna_R15_mmp_d`、命令行参数留空。使用全新工作目录
`/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06j_0800_ferod1`。

本轮固定 `TSSFAC=0.05`、`DT2MS=0`，唯一变量是碎片材料 MAT_015 Card 4 的 `EROD: 0 -> 1`；
板材 EROD 仍为 0。初始几何、爆炸后破片速度、材料强度/失效参数、接触、网格、终止时间和输出设置不变。

这只是失效后碎片删除机制诊断。EROD=1 会保留失效单元并清零其偏应力，因此所得响应不得作为物理标签、
仿真引擎标定目标或毁伤树标定目标。即使求解正常结束，也必须先审计 HVAR6、删除质量、负体积、能量和接触证据。

完成后下载完整回传：`.k`、`d3hsp/o=`、`mes*`、`glstat`、`binout*`、完整 `d3plot*`、完整 `d3part*`、
stdout/stderr、Slurm 脚本和资源记录。之后只需把回传路径发给我。
