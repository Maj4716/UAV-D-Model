# R06r 2000 m/s 数值上锚点提交说明

不要压缩。只上传整个 `uav_hf_r06r_2000_speed_upper_anchor` 文件夹，选择 `r06r_2000_speed_upper_anchor.k` 作为 head/input deck。

硬性资源参数：1 个节点、恰好 32 个 MPP ranks、`ls-dyna_R15_mmp_d`、head memory 512 MB、
child memory 256 MB、命令行参数留空、walltime 12:00:00。提交预览必须包含 `--ntasks-per-node=32`。

使用全新工作目录 `/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06r_2000_speed_upper_anchor`。
本轮唯一物理因子是 part 2002 初速度从 800 改为 2000 m/s；D3PLOT/D3PART 0.0001 ms 是测量控制，
ENDCYC=2,500,000 是未触发时不影响解的执行保护。终止时刻 0.50 ms、ASCII 0.00025 ms、
TSSFAC=0.05、DT2MS=0，以及材料、接触、自适应 SPH、网格和 32 ranks 均冻结不变。

预计完整输出约 32.8 GB（30.55 GiB），请预留足够空间。若提前达到循环上限或失败，不要修改参数重启，
直接下载完整目录。保留 `.k`、`mes*`、`glstat`、`matsum`、`ncforc`、`binout*`、所有 d3plot/d3part、
stdout/stderr、Slurm 脚本和资源记录。之后只需发送回传路径。

本轮是数值速度上锚点，不是物理验证，也不是仿真引擎或毁伤树标定。
