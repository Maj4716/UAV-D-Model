# R06u 原样重提交：提交前身份核对

本文件夹是对正确 R06u 的逐字节原样重交付；没有修改任何模型参数。

1. 不要压缩，只上传整个 `uav_hf_r06u_1200_plate_adaptive_sph_canary_retry` 文件夹。
2. 必须选择 `r06u_1200_plate_adaptive_sph_canary.k` 作为 head/input deck。
3. 该 deck 的 SHA-256 必须是 `sha256:7bfb99a49bb757632237b2e2c625500fef27e811988eac3149df7a3e056647d4`。
4. 若界面出现 `r06t` 或 `psfail` 文件名，立即取消提交；那是旧输入卡。
5. 使用全新工作目录 `/work/home/acd7dt5yv4/LS_DYNA/uav_hf_r06u_1200_plate_adaptive_sph_canary_retry`，不要复用上次目录。
6. 提交预览中的 `i=` 必须以 `/r06u_1200_plate_adaptive_sph_canary.k` 结尾。

资源仍为：1 节点、恰好 32 ranks、`ls-dyna_R15_mmp_d`、head memory 512 MB、
child memory 256 MB、命令行参数留空、walltime 12:00:00。

R06u 终止时刻应为 0.15 ms，`CONTROL_SOLID.PSFAIL=0`，板件 part 1002 失败后转入同材料 SPH part 1003。
本轮只验证能否越过 R06t 的 0.10872 ms 崩溃点，不用于物理有效性、仿真引擎标定或毁伤树标定。

回传仍按小文件优先：deck、manifest、job_contract、Slurm、平台 log、stdout/stderr、
glstat/matsum/sleout/nodout/ncforc、若有 sldsph_frag、mes0000--mes0031；先不要下载 d3plot/d3part/binout。
