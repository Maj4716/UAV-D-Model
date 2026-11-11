# R06o 板件 MAT EROD=1 隔离提交说明

不要压缩。只上传整个 `uav_hf_r06o_0800_plate_erod1_isolation` 文件夹，选择 `r06o_0800_plate_erod1_isolation.k` 作为 head/input deck。

硬性资源参数：1 个节点、恰好 32 个 MPP ranks、`ls-dyna_R15_mmp_d`、head memory 512 MB、
child memory 256 MB、命令行参数留空。提交预览必须包含 `--ntasks-per-node=32`。

使用全新工作目录 `/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06o_0800_plate_erod1`。
相对回传 R06n，本轮仅改变 TITLE 和板件 MAT_JOHNSON_COOK 的 EROD 字段 0→1；碎片自适应 SPH、
接触、TSSFAC=0.05、DT2MS=0、0.04 ms 时长及 829,930 循环保护不变。

本轮可能因保留失效板件单元而发生时间步塌缩。若提前停止或达到循环上限，不要改参数重启；直接下载完整目录。
无论完成或失败，保留 `.k`、`mes*`、`glstat`、`matsum`、`ncforc`、`binout*`、所有 d3plot/d3part、
stdout/stderr、Slurm 脚本和资源记录。之后只需发送回传路径。

这是数值失效路径隔离，不是物理、仿真引擎或毁伤树标定。
