# R06u 板件自适应 SPH 能力验证提交说明

不要压缩。只上传整个 `uav_hf_r06u_1200_plate_adaptive_sph_canary` 文件夹，选择 `r06u_1200_plate_adaptive_sph_canary.k` 作为 head/input deck。

硬性资源参数：1 个节点、恰好 32 个 MPP ranks、`ls-dyna_R15_mmp_d`、head memory 512 MB、
child memory 256 MB、命令行参数留空、walltime 12:00:00。提交预览必须包含 `--ntasks-per-node=32`。

使用全新工作目录 `/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06u_1200_plate_adaptive_sph_canary`。
本算例从已完成的 R06s 分支，PSFAIL 恢复为 0，并增加板件失败单元 part 1002 → 同材料 SPH part 1003 的局部转换。
速度 1200 m/s；短时终点 0.15 ms，超过 R06t 的 0.10872 ms 崩溃点；ENDCYC=2,500,000。
D3PLOT/D3PART 0.00015 ms、ASCII 0.00025 ms、TSSFAC=0.05、DT2MS=0，其他材料、网格和资源均冻结。

计算完成后不要立即下载大文件。第一阶段只下载：提交 deck、manifest.json、job_contract.json、Slurm 脚本、
平台 log、std.out、std.err、glstat、matsum、sleout、nodout、ncforc、若存在则 sldsph_frag，以及 mes0000–mes0031。
先发送该小文件目录路径；不要下载 d3plot、d3part 或 binout。

本轮只验证板件局部 SPH 替代是否能越过 R06t 崩溃点，不是物理验证，也不是仿真引擎或毁伤树标定。
