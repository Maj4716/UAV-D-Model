# Job 73179287 最小补充下载清单

请从平台作业 73179287 只补下载以下 5 个小型 ASCII 数据库，并放入原目录：

`D:\Desktop\UAV新\uav_hf_r06u_1200_plate_adaptive_sph_canary_retry`

必需文件：

1. `glstat`（平台下载后名为 `glstat.glstat` 也可）
2. `matsum`（`matsum.matsum` 也可）
3. `sleout`（`sleout.sleout` 也可）
4. `nodout`（`nodout.nodout` 也可）
5. `ncforc`（`ncforc.ncforc` 也可）

如果平台目录存在 `sldsph_frag`，一并下载；它是可选的辅助证据。

暂时不要下载：`d3plot*`、`d3part*`、`binout*`。

补齐后再次发送同一个本地目录路径。下一轮将核验最小时间步、总能量、滑移能、沙漏能、板件 SPH part 1003 正质量，以及接触/节点账本；只有这些门禁通过后才决定是否下载二进制结果。

