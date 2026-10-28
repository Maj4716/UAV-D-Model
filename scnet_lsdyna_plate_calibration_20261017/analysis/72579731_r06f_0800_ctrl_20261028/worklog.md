# Job 72579731 / R06f 800 m/s control 回传审查工作日志

日期：2026-10-28

1. 将 `D:\Desktop\UAV新\uav_hf_r06f_0800_ctrl` 作为只读原始证据目录；未在该目录内改名、移动、删除或生成分析文件。
2. 盘点 1,093 个文件、2,824,683,203 bytes；确认核心族为 14 个 `d3plot*`、3 个 `d3part*`、2 个 `binout*`、512 个 `mes*`、512 个并行重启分片与完整平台日志。
3. 核对 candidate manifest 的 9 个白名单输入；文件大小与 SHA-256 全部匹配。回传 deck SHA-256 为 `111911562b703b8af1e96810b08edd8c31fbeea739ddddd2921b8bcd07c517af`。
4. 扫描 Slurm、portal 变量、host file、作业日志与 stdout，确认实际资源为 16 nodes / 512 MPP ranks，而冻结合同要求 1 node / 32 ranks。
5. 扫描全部 512 个 `mes*`、solver `o=`、stdout/stderr；确认 512/512 ranks 正常结束、无 numbered error、stderr 为 0 bytes。
6. 用 `parse_lsdyna_ascii.py` 解析 GLSTAT 与 rank 日志，生成 `ascii_metrics.json`；能量比为 `0.955775–1.00268`，hourglass energy 为 0，无质量缩放，存在 66 个唯一负体积侵蚀事件。
7. 将 66 个负体积事件绑定到 deck：plate PID 1002 为 26 个，fragment PID 2002 为 40 个；首次 `0.0032354 ms`，末次 `0.012784 ms`。
8. 用项目固定的 vendored `lasso-python 2.0.4` 打开完整 `d3plot*`、`d3part*` 与 `binout*`；核心二进制族均可读且无断号。
9. 运行 R06 单作业适配器，生成 `r06f_single_job_analysis.json`。其 capture 与 full-D3PLOT HVAR6 结果有效；其 direct-contact 段因空 BINOUT inventory 而误报，已在紧凑审计中明确作废。
10. 验证 D3PART 修复：802 states、686 fragment solids、history shape `(802,686,1,7)`；HVAR6 在 alive 元素上全有限。
11. 在 402 个共享时刻将 D3PART 与 full D3PLOT 逐元素交叉核对：686 IDs、251,657 个 alive 值、7 个历史通道最大绝对差均为 0，alive mask 与时刻完全一致。
12. 枚举 BINOUT 逻辑库，确认 `JOBINFO/glstat/matsum/ncforc/sleout/nodout` 均存在；NCFORC 为时变 ragged 节点集合，必须保留每态 ID，不能补零伪装为固定矩阵。
13. 按冻结门前指标与 Job 72397333 探索性比较；qM/qP/qE 三项超出容差，qV 相对差通过。但因资源从 32 ranks 变为 512 ranks，该比较不具备正式因果解释资格。
14. 判定为 `successful_solver_run_but_resource_confounded_invalid_recanary`；未放行 TSSFAC=0.4、2000 m/s 或 treatment，也未给予 P0/P1、引擎拟合或数据集准入。

边界：本轮只审查回传证据并制定后续策略，没有修改原始回传、输入物理参数、材料参数、接触模型、仿真引擎或数据集生成器。
