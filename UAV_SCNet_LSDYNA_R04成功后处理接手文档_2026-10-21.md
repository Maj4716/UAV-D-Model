# UAV 项目接手文档：SCNet LS-DYNA R04 成功计算后的结果审查

**生成日期：** 2026-10-21  
**项目根目录：** `D:\Desktop\UAV新`  
**下一轮重点：** 接收并审查 SCNet R04 成功作业的完整回传结果，完成数值资格判定，并据此提出仿真引擎、数据集生成器和 KMFC 毁伤评估的校准建议。

## 1. 当前状态

- SCNet P0 作业 `uav_hf_p0_plate_r04` 已计算成功，Job ID 为 `72095164`。
- 使用的软件条目为 `ls-dyna_R15_mmp_d`，本轮为单节点、32 个 MPP ranks 的双精度计算。
- 平台工作目录截图显示已经生成：`binout0000`、`glstat`、完整的 `d3plot`～`d3plot10`、32 个 `mes0000`～`mes0031`、实际输入 deck、作业日志以及重启动文件。
- 用户尚未把成功作业的结果 ZIP 回传。本轮最后一步是告知用户应下载哪些文件，并估算总大小。
- 截图中共有 101 个文件；全部文件原始大小估计约 `1.68–1.70 GB`。若排除 `d3dump0.*` 和 `d3full01`，分析所需文件约 `1.06 GB`。

## 2. 下一轮首先要做的事

1. 接收用户上传的成功作业 ZIP，保持原文件名和目录结构，不要先修改或重命名。
2. 计算 ZIP 的 SHA-256，执行 CRC、路径穿越和条目清单检查，再安全展开到项目 `results/` 下以 Job ID 命名的新目录。
3. 对远端实际 `p0_velocity_mesh_calibration.k` 做 SHA-256 身份绑定。R04 预期 deck：
   - 大小：`20,604,806` 字节；
   - SHA-256：`312F2CE1C42DD070097DAD1036D1D670790704C55C5B534A3B845A0709709E08`；
   - 首行：`*KEYWORD LONG=Y`；
   - 第 14 行末尾应为 `...,0,0,0`。
4. 先读 `mes0000`、全部 `mes*`、作业 `.out/.log`、`std.out.*`、`std.err.*` 和 `status.out`，确认正常终止，并汇总所有 warning/error。不要只凭平台状态“成功”认定数值有效。
5. 使用 `binout0000`、`glstat` 和完整 `d3plot*` 文件族完成数值资格检查：
   - 实际终止时间、时间步及 CPU/MPI 信息；
   - 总能量、内能、动能、小时玻璃能和质量变化；
   - 初始穿透、接触警告、负体积、单元删除及异常能量跳变；
   - 七个子模型是否互不干扰；
   - 五个速度点的穿孔状态和剩余速度；
   - 2000 m/s 三个网格层级的响应一致性与离散误差。
6. 用 `p0_case_part_map.csv` 绑定七个工况与 part ID，严禁凭 d3plot 视觉位置猜测工况。
7. 只有 P0 通过解析、稳定性、能量与网格资格门后，才可以提出 P1 计算、仿真引擎参数校准以及 KMFC 毁伤树阈值调整建议。

## 3. 用户应回传的文件

最佳方案是下载并回传整个工作目录。若需要减小体积，至少包含：

```text
d3plot, d3plot01 ... d3plot10
binout0000
glstat
mes0000 ... mes0031
job_uav_hf_p0_plate_r04_72095164.log
job_uav_hf_p0_plate_r04_72095164.out
job_LS_DYNA_*.slurm
std.out.72095164
std.err.72095164
status.out
p0_velocity_mesh_calibration.k
p0_case_part_map.csv
manifest.json
jc4340_reference.json
static_validation.json
```

若存在，也要下载 `d3hsp`、`matsum`、`sleout`、`nodout`、`rcforc`、`elout`。截图中未看到这些独立 ASCII 文件，相关数据库可能写入了 `binout0000`。

`d3dump0.0000`～`d3dump0.0031` 和 `d3full01` 主要用于重启动，结果分析可以不下载；若用户已下载整个目录，则保留原样。

## 4. 现有权威文档与产物

不要在接手文档中重新推导此前三次失败，直接读取以下文件：

- R03 根因、R04 修复和静态审查：`D:\Desktop\UAV新\43_SCNet_LSDYNA_P0_R03失败根因与R04全量校验修复_2026-10-21.md`
- R04 SCNet 提交参数与回传清单：`D:\Desktop\UAV新\40_SCNet_LSDYNA_P0手工提交与回传操作清单_2026-10-19.md`
- P0 科学设计及接受边界：`D:\Desktop\UAV新\39_SCNet_LSDYNA破片装甲P0设计_提交与回传校准记录_2026-10-17.md`
- SCNet 方案总记录：`D:\Desktop\UAV新\38_SCNet_LSDYNA标定执行记录与仿真引擎数据集修改建议_2026-10-17.md`
- R04 项目目录：`D:\Desktop\UAV新\scnet_lsdyna_plate_calibration_20261017`
- R04 上传副本：`D:\Desktop\UAV新\scnet_lsdyna_plate_calibration_20261017\manual_upload_20261021_r04`
- 静态校验报告：`D:\Desktop\UAV新\scnet_lsdyna_plate_calibration_20261017\validation\static_validation.json`
- 结果解析脚本：`D:\Desktop\UAV新\scnet_lsdyna_plate_calibration_20261017\scripts\parse_lsdyna_ascii.py`
- 后处理自测：`D:\Desktop\UAV新\scnet_lsdyna_plate_calibration_20261017\scripts\selftest_postprocessor.py`
- 完整本地审计包：`D:\Desktop\UAV新\scnet_lsdyna_plate_calibration_20261017\artifacts\scnet_lsdyna_plate_calibration_p0_20261017.zip`

官方字段依据：

- LS-DYNA R15 Keyword Manual Vol. I：`https://ftp.lstc.com/anonymous/outgoing/web/ls-dyna_manuals/R15/LS-DYNA_Manual_Volume_I_R15.pdf`
- SCNet LS-DYNA 模板说明：`https://www.scnet.cn/help/docs/mainsite/hpc/software-examples/use/CAE/LSDYNA/index.html`

## 5. 已完成的工程修复

只保留结论，详细 diff 见第 4 节文档：

- R01：普通字段宽度与长数值冲突；
- R02：必需卡片/字段不完整及整数类型问题；
- R03：`*CONTROL_CONTACT` Card 2 的整数型 `ECDT` 被写成浮点科学计数；
- R04：`ECDT` 改为严格整数，并建立 23 种关键字、110 个关键字块的逐卡字段类型合同检查；
- R04 在本地通过静态验证、Python 编译、关键字负向回归、后处理自测、引用闭合、网格体积、质量、空间隔离和哈希绑定检查；
- R03 与 R04 deck 只有第 14 行的一个 token 不同，未改动物理模型。

## 6. 科学结论边界

- “作业成功”只证明求解器完成运行，不等于模型具有物理真实性。
- P0 是 8 g 立方破片、10 mm 参考钢板、四分之一对称、中心法向正撞的数值资格模型。
- 当前 4340 Johnson–Cook/EOS 参数是参考材料假设，并非用户真实装甲或战斗部的试验标定。
- P0 仅用于速度响应、穿孔过渡和网格行为检查，不能直接证明完整车辆毁伤、爆炸破片场或 KMFC 标签的真实性。
- 高保真结果的当前证据级别仍是 `high_fidelity_synthetic_reference_candidate`；只有通过数值资格检查后才能升格为本项目的外部数值校准证据。
- 不得为了让 KMFC 标签均匀而反向调整 LS-DYNA 材料参数；标签均衡只能在通过物理资格的响应域内通过采样和有序判定设计实现。
- 在 P0 接受前不要批量提交 24 个 P1 工况。

## 7. 建议技能

下一位代理应根据工作阶段调用以下技能：

- `$ai-research`：组织 P0 资格判定、校准假设和证据链；
- `$test-research-code`：检查后处理入口、可重复性、哈希和回传包完整性；
- `$statistical-analysis`：分析速度响应、网格收敛、误差和不确定性；
- `$scientific-writing`：把求解结果、限制和修改建议写成科研级 Markdown 报告；
- `$scientific-visualization`：生成剩余速度曲线、能量历史和网格敏感性图时使用。

## 8. 工作约定

- 使用中文输出，结论必须绑定文件、日志行、哈希或计算结果。
- Python 环境优先使用 `D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe`。
- Windows 临时测试应把 `TEMP/TMP` 指向仓库内可写临时目录，避免沙箱系统临时目录的权限错误。
- 记录分析过程和接受/拒绝理由；区分静态检查、求解器正常终止、数值稳定、物理有效和项目校准五个层级。
- 不覆盖原始 SCNet 回传文件；所有派生 CSV、JSON、图表和报告写入独立分析目录。
- SCNet 账号标识和远端个人目录已从本接手文档中省略；如确有必要，读取现有提交清单，不要在公开报告中传播账户信息。
