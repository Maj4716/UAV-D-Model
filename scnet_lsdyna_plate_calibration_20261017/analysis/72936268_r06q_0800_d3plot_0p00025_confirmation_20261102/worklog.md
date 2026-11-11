# R06q Job 72936268 回传处理与 R06r 交付工作日志

日期：2026-11-02

## 输入与不可变性

- 原始回传：`D:\Desktop\UAV新\uav_hf_r06q_0800_d3plot_0p00025_confirmation`
- 原始回传按只读证据处理，未在其中写入或修改文件。
- 共 188 个文件，14,277,708,013 bytes。
- 回传 deck SHA-256：`a6229cda9778adb13b4c174346a4356deea3e62b175fda9cf8c7dbf12be60002`，与提交清单一致。

## 回传处理

1. 复核平台 Job ID、deck、软件、节点数、MPP ranks、Slurm 与 stdout/stderr 身份。
2. 复核 0.50 ms 正常终止、循环保护、能量、质量、自适应 SPH、板件保留、负体积归属和跨格式一致性。
3. 重建全时程 solid/SPH 载体账本、捕获平面事件、尾段分离状态与主 QoI。
4. 将 R06q 与 R06p 比较，确认只提高 D3PLOT 测量分辨率，没有改变数值解。
5. 生成正式审查 `r06q_d3plot_confirmation_return_audit.json`。

## R06q 结论

- 1 节点 / 32 ranks，R15.0.2 MPP double，正常到达 0.50 ms。
- 平台耗时 2,344 s；solver elapsed 2,325 s；775,413 cycles；stderr 0 bytes；无 mass scaling。
- 能量比范围 1.0–1.0；终态滑移能占初始能 0.022668；小时玻璃比 0。
- 45 个负体积单元全部属于破片，且与 45 个自适应 SPH 转换母单元集合完全相同；板件删除为 0。
- 45 个转换实体对应 360 个活动 SPH，严格 8:1；质量闭合最大相对误差 `3.90e-15`；SPH 删除为 0。
- D3PLOT/D3PART 均为 2,002 states，7 通道最大差为 0，alive mask mismatch 为 0。
- D3PLOT 状态数从 R06p 的 1,002 增至 2,002；最大连续向下步长从 0.446989 mm 降至 0.228597 mm，满足冻结的 `<=0.25 mm` 门槛。
- 无 solid/SPH 捕获平面穿越、无平面下激活、无删除区间歧义，最小捕获平面余量 6.803782 mm，尾段稳定分离。
- 主响应可用：`confirmed_nonperforation`，`qM=qP=qE=0`，`qV/v0=null`。
- R06p/R06q 循环数、ASCII 关键指标、转换母单元集合和目标板件 HVAR6 一致；确认输出频率只修复测量分辨率。
- 准入边界：仅准入 800 m/s、0.75 mm 网格的一个确定性数值未贯穿点；物理有效性、V50、材料标定、引擎拟合、毁伤树拟合和数据集晋升仍不成立。

## 下一实验的选择

- 选择 2000 m/s 作为一个数值上锚点，以单个算例尽快尝试建立 `[800, 2000] m/s` 的未贯穿/贯穿括区。
- 唯一物理因子：part 2002 初速度 800 -> 2000 m/s。
- 测量控制：D3PLOT 和 D3PART 0.00025 -> 0.0001 ms；按 R06q 实测值线性规划，预计最大向下步长 0.228597 mm，仍低于 0.25 mm。
- 执行保护：ENDCYC 829,930 -> 2,500,000；在未触发时不改变解。
- 冻结：0.50 ms、ASCII 0.00025 ms、TSSFAC=0.05、DT2MS=0、材料、接触、自适应 SPH、网格和 1 节点/32 ranks。
- 若 R06r 确认贯穿，下一轮释放 1200 m/s 中点；若仍未贯穿，不自动继续升速，先审查速度范围与模型形式。

## 构建与验证

- R06r deck 仅第 3、8、19、24、45102 行与 R06q 父 deck 不同；其余 45,115 行一致。
- 专项测试：6/6 通过。
- 受控全项目回归：184/184 通过；另有 12 项依赖当前工作区已不在场的 R06d–R06n 原始回传/下载 ZIP 证据，作为不可执行外部证据测试排除。
- 交付根目录白名单通过；上传目录含 9 个普通文件；ZIP 数为 0；复制后逐文件大小与 SHA-256 复核通过。
- R06r head deck SHA-256：`511cbde9d33b1422981bc5716b0c9ae16a3fcd14f091e1dec003d053f3808e48`。

## 交付

- 直接上传目录：`D:\Desktop\UAV新\scnet_lsdyna_plate_calibration_20261017\artifacts\r06r_2000_speed_upper_anchor_submission_20261102\uav_hf_r06r_2000_speed_upper_anchor`
- Head deck：`r06r_2000_speed_upper_anchor.k`
- 资源：`ls-dyna_R15_mmp_d`，1 节点，恰好 32 MPP ranks，head memory 512 MB，child memory 256 MB，命令行留空，walltime 12:00:00。
- 规划输出：约 32.805 GB（30.55 GiB），不是实测值。

