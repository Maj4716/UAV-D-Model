# Job 72197064 原样重提清单

## 决策

本次失败发生在 ANSYS 许可证连接阶段，LS-DYNA 尚未读取 deck。保持 `R05_capture_ledger_preregistered` 不变，不创建 R06，不改材料、接触、时间步、D3PART、网格、ranks 或内存。

## 重提前

1. 优先确认 SCNet 的 `ls-dyna_R15_mmp_d` 许可证服务已经恢复；若无法确认，只进行一次隔离重试。
2. 使用新的空工作目录：`/work/home/acd7dt5yv4/LS_DYNA/uav_hf_p0_plate_r05_retry1`。
3. 作业名建议为 `uav_hf_p0_plate_r05_retry1`，仅用于区分运行身份，不代表新模型修订。
4. 从权威候选目录上传，不使用下载回传目录中的副本：
   `D:\Desktop\UAV新\scnet_lsdyna_plate_calibration_20261017\manual_upload_20261021_r05_candidate`
5. head/input 只选择 `p0_velocity_mesh_calibration.k`。
6. 上传前核对 deck：20,605,206 bytes；SHA-256 `6fbd753a113d0214e7ba4ff5136dc526d151e6f2c52c94e70399b29c4b9d3d0c`。

## 保持不变的设置

- 软件：`ls-dyna_R15_mmp_d`；
- 节点：1；
- ranks：32；
- walltime：04:00:00；
- head memory：512 MB；
- child memory：256 MB；
- command-line parameters：空；
- auto-fill previous parameters：off。

## 运行后第一道停止门

先检查 stdout 是否出现：

```text
[license/info] Successfully checked out 1 of "dyna".
[license/info] Successfully checked out 31 of "FEAT_DYSMP".
[license/info] Successfully started "ANSYS LSDYNA".
```

若再次出现 `FlexNet Licensing error:-15,570`，停止重复提交并向 SCNet 报障；不要修改 deck 规避许可证问题。

若许可证通过，才继续检查 `d3hsp` 和最早的 `mes0000` 关键词解析结果。只有解析、正常终止、真实 D3PART、捕获账本、速度括区及三网格门全部通过，P0 才能进入响应标定。

## 可直接发送给 SCNet 的报障文本

> LS-DYNA Job 72197064 于 2026-10-22 11:21:08 在计算节点 b12r4n09 使用 `ls-dyna_mpp_d_R15_0_2` 启动时，连接 ANSYS license server `27890@login01` 失败，stdout 报 `FlexNet Licensing error:-15,570`、System Error 115、LS-DYNA Error 70022，应用退出码 12。相同软件版本、32 ranks 和许可模式曾在 Job 72095164（节点 i15r4n06，2026-10-21）成功签出 `dyna` 和 31 个 `FEAT_DYSMP`。请检查 `lmgrd`/vendor daemon 状态、license host/port 配置，以及计算节点到 `login01:27890` 的 DNS、路由和防火墙连通性。
