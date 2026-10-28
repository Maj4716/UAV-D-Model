# Job 72630855 TSSFAC=0.4 回传审查与 R06g 细化交付日志

日期：2026-10-28  
原始回传：`D:\Desktop\UAV新\uav_hf_r06f_0800_tssfac04`（只读分析）

## 回传审查

1. 文件身份：113 个文件、2,129,839,228 bytes；submitted deck SHA-256 为 `afb0f34112ccfdfcfd95bf442c9b45e2cb1590fdad8d76e98fbbc3fb5d6baee7`，与冻结提交包一致。
2. 资源身份：Slurm `-N 1`、每节点 32 task；作业日志 32 total cores、`mpirun -np 32`；求解器 32 MPP procs，全部一致。
3. 求解状态：Job 72630855 正常结束，problem time 0.040005 ms，7066 cycles，stderr 为空，32/32 rank 正常结束，无 error/fatal。
4. 输出完整性：完整 `d3plot`–`d3plot13`、`d3part`–`d3part02`、2 个 binout、32 个 mes、restart/full 文件和平台记录均存在。
5. 独立读取：D3PART 802 states，full D3PLOT 402 states；402 个共享状态七通道完全一致，alive mask mismatch=0，HVAR6 全部有限。
6. NCFORC：801 个变长状态，3365–3398 nodes/state，逐状态 ID 唯一，坐标、力和压力有限；物理单位、符号和能量闭合仍未验证。
7. 全局数值：能量比范围 0.987558–1.00263，hourglass energy=0；10 个负体积单元，其中 plate 2、fragment 8；无质量缩放。

## 冻结门禁比较

相对 Job 72617857（TSSFAC=0.8）：

| 指标 | TSSFAC=0.8 | TSSFAC=0.4 | 差值/结果 |
|---|---:|---:|---|
| qM | 0.873177835 | 0 | 0.873177835，FAIL |
| qP | 0.561307455 | 0 | 0.561307455，FAIL |
| qE | 0.360875291 | 0 | 0.360875291，FAIL |
| qV/v0 | 0.642832917 | null | 不可比，fail-closed |
| cycles | 2863 | 7066 | 明显改变 |
| 负体积单元 | 66 | 10 | 明显减少 |
| 最终存活弹体单元 | 612 | 656 | 明显增加 |

TSSFAC=0.4 没有确认的 capture-plane crossing，但 deleted-upstream mass fraction 仍为 0.0437318，超过“确认未穿透”所需的更严格 0.005 门槛，因此分类仍是 `transition_or_indeterminate`，不能把 q=0 解释成已证明未穿透。

结论：求解和测量渠道通过，但 0.8→0.4 时间步敏感性门禁失败；TSSFAC=0.8 响应不能视为时间步稳健结果。2000 m/s 和全部 treatment 继续冻结。

## 下一作业实现

- 新建 revision：`R06_timestep_refinement_20261028`。
- 新作业：`r06g_0800_tssfac02`。
- 唯一数值因子：TSSFAC `0.4 -> 0.2`；`DT2MS=0`。
- 外部控制：Job 72630855。
- 保持 800 m/s、0.75 mm、几何、网格、材料、接触、终止时间、输出合同和 1 node/32 ranks 不变。
- 新门禁加入 no-crossing-aware qV 规则和 deleted-upstream mass fraction 绝对差 0.01。

## 实施与验证

- 新增可复现回传审计器、R06g 设计/合同生成器、R06g 暂存器和 5 个专项测试。
- 扩展生成器/验证器，使其支持 R06g 生命周期和 0.4→0.2 的字段级隔离。
- 为多个 Windows 原子目录发布入口加入只针对 WinError 5/32 的有限短暂重试，消除安全扫描造成的偶发回归失败。
- 最终全套可执行回归：129 passed、0 failed；另有 1 个历史外部下载夹具测试因本机缺失 `D:\Downloads\uav_hf_r06d_0800_ctrl.zip` 而明确排除。
- R06g 专项：5 passed；相关脚本 `py_compile`：PASS。

## 最终提交入口

规范目录必须带 `_v2`：

`D:\Desktop\UAV新\scnet_lsdyna_plate_calibration_20261017\artifacts\r06g_tssfac02_submission_20261028_v2`

只上传其中的 `uav_hf_r06g_0800_tssfac02.zip`。不带 `_v2` 的同名目录是封装过程旧版，已写入 superseded 标记，不得提交。
