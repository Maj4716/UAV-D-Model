# R06s 1200 m/s 数值稳定性诊断提交说明

不要压缩。只上传整个 `uav_hf_r06s_1200_speed_stability_diagnostic` 文件夹，选择 `r06s_1200_speed_stability_diagnostic.k` 作为 head/input deck。

硬性资源参数：1 个节点、恰好 32 个 MPP ranks、`ls-dyna_R15_mmp_d`、head memory 512 MB、
child memory 256 MB、命令行参数留空、walltime 12:00:00。提交预览必须包含 `--ntasks-per-node=32`。

使用全新工作目录 `/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06s_1200_speed_stability_diagnostic`。
唯一物理因子是 part 2002 初速度从 2000 改为 1200 m/s；D3PLOT/D3PART 0.00015 ms 是测量控制。
终止时刻 0.50 ms、ENDCYC=2,500,000、ASCII 0.00025 ms、TSSFAC=0.05、DT2MS=0，
以及材料、接触、自适应 SPH、网格和 32 ranks 均冻结不变。

计算完成后不要立即下载大文件。第一阶段只下载：提交 deck、manifest.json、job_contract.json、Slurm 脚本、
平台 log、std.out、std.err、glstat、mes0000 至 mes0031。先发送该小文件目录路径；不要下载 d3plot、d3part 或 binout。

本轮用于判断2000 m/s的板件负体积和滑移能失败是否局限于极端速度，不是物理验证，也不是标定。
