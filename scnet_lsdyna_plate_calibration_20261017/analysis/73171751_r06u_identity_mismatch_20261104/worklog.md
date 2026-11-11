# R06u 标签作业 73171751 输入身份错误审查与原样重提交工作日志

日期：2026-11-04

## 只读输入审查

- 回传目录：`D:\Desktop\UAV新\uav_hf_r06u_1200_plate_adaptive_sph_canary`。
- 原始回传保持只读，审查输出写入独立 analysis 目录。
- 平台作业名和 Slurm `-J` 标记为 `uav_hf_r06u_1200_plate_adaptive_sph_canary`，作业号 73171751。
- 实际命令行 `i=` 指向 `r06t_1200_fragment_only_psfail_diagnostic.k`，而不是 R06u deck。
- 回传 `manifest.json`、`job_contract.json` 和 deck 均属于冻结 R06t；没有发现 R06u deck。
- 实际 deck SHA-256 为 `11f62db3f8b27d1087988d5ad2abc572bddbd89570f6065500ddfe390a6459ae`，与冻结 R06t 完全一致。
- 正确 R06u deck SHA-256 为 `7bfb99a49bb757632237b2e2c625500fef27e811988eac3149df7a3e056647d4`，两者不相同。

## 重复失败签名

- 实际运行再次在 0.10872 ms、cycle 603398 终止。
- 终止错误仍为板件 part 1002 的单元 23830 `complex sound speed`。
- 终止时间步仍为 `1.99e-27 ms`。
- 该时间、循环、单元和时间步与 R06t Job 73087868 完全一致。
- 因此本次是旧 R06t 输入卡的确定性复跑，不是 R06u 数值结果，不能据此修改 R06u。

## 裁决

- 输入身份门禁失败，R06u 尚未执行。
- 不下载本次作业的 `d3plot*`、`d3part*` 或 `binout*`。
- 不修改材料、接触、侵蚀、自适应 SPH、时间步或输出控制。
- 不进行仿真引擎拟合、毁伤树拟合或参考标签准入。
- 唯一下一步是把此前正确 R06u 逐字节复制到新文件夹，并在全新平台工作目录中重提交。

## 防错重提交交付

- 上传目录：`D:\Desktop\UAV新\scnet_lsdyna_plate_calibration_20261017\artifacts\r06u_plate_adaptive_sph_canary_resubmission_20261104\uav_hf_r06u_1200_plate_adaptive_sph_canary_retry`。
- Head deck：`r06u_1200_plate_adaptive_sph_canary.k`。
- Head deck SHA-256：`7bfb99a49bb757632237b2e2c625500fef27e811988eac3149df7a3e056647d4`。
- 上传目录内 9 个文件均与冻结的正确 R06u 逐字节一致；没有 ZIP。
- 全新平台工作目录：`/work/home/acd7dt5yv4/LS_DYNA/uav_hf_r06u_1200_plate_adaptive_sph_canary_retry`。
- 提交预览中 `i=` 必须以 `/r06u_1200_plate_adaptive_sph_canary.k` 结尾；出现 `r06t` 或 `psfail` 必须取消提交。

## 验证

- 新增输入身份审计可重复生成，并明确 `r06u_was_executed=false`。
- Python 编译检查通过。
- 身份防错专项测试 3/3 通过。
- 受控全项目回归 199/199 通过；18 项依赖当前不在工作区的历史外部回传、下载 ZIP 或大型二进制，按精确测试名排除。
- 临时测试目录固定到工作区，避免 Windows 沙箱默认 TEMP ACL 干扰业务回归判定。

