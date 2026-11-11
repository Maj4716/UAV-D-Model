# R06n 精确 32-rank 自适应 SPH 确认提交说明

不要压缩。只上传整个 `uav_hf_r06n_0800_adaptive_sph_32rank_confirmation` 文件夹，选择 `r06n_0800_adaptive_sph_32rank_confirmation.k` 作为 head/input deck。

硬性资源参数：1 个节点、**恰好 32 个 MPP ranks**、`ls-dyna_R15_mmp_d`、head memory 512 MB、
child memory 256 MB、命令行参数留空。请在提交预览/生成的 Slurm 脚本中确认
`--ntasks-per-node=32`；若平台自动改为 64，不要把该作业视为本轮有效确认。

使用全新工作目录 `/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06n_0800_adaptive_sph_32rank`。
本轮仍从爆炸瞬间后的状态开始，模型、接触、输出、0.04 ms 时长和 829,930 循环保护均不变；
与回传 R06m 的输入 deck 相比只有 TITLE 标识行改变。本轮不要顺便延长时长，否则会把资源和时长两个因素混在一起。

无论完成、提前停机或平台资源不符，都下载完整作业目录。之后只需把回传路径发给我。
这是资源一致性与工程复现确认，不是物理标定；不得用于仿真引擎或毁伤树拟合。
