# R06m 自适应实体转 SPH 能力验证提交说明

不要压缩。只上传整个 `uav_hf_r06m_0800_adaptive_sph_canary` 文件夹，在平台选择 `r06m_0800_adaptive_sph_canary.k` 作为 head/input deck。

资源参数：1 个节点、32 MPP ranks、`ls-dyna_R15_mmp_d`、head memory 512 MB、child memory 256 MB，
命令行参数留空。使用全新工作目录 `/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06m_0800_adaptive_sph`。

本轮从与 R06j 相同的爆炸瞬间后状态开始，800 m/s、TSSFAC=0.05、DT2MS=0。碎片实体 2002
失效后转换为 SPH part 2003，每个六面体生成 8 个粒子。实体接触仍为 SOFT=2；SOFT=1 只用于
SPH 对实体的专用非对称节点-面接触，因此不是重复已经失败的 R06l 全局 SOFT=1 路线。

防失控边界：必须在 829,930 个循环以内到达问题时间 0.04 ms。如果先达到循环上限，
该次只能记为能力验证失败/部分结果，不算正常完成，不要重启或擅自修改参数，直接下载完整回传。

无论完成、提前停机还是解析失败，都下载完整目录，尤其保留 `.k`、`d3hsp/o=`、`mes*`、`glstat`、
`matsum`、`ncforc`、`binout*`、所有 `d3plot*`、所有 `d3part*`、stdout/stderr、Slurm 脚本和资源记录。
之后只需把回传路径发给我。

这是解析/运行/质量-能量/输出能力验证，不是物理标定；不得把响应直接用于仿真引擎或毁伤树标定。
