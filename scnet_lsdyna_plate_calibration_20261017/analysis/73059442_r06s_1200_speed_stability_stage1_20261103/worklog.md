# R06s Job 73059442 第一阶段审查与 R06t 交付工作日志

日期：2026-11-03

## 输入与身份

- 只读审查目录：`D:\Desktop\UAV新\uav_hf_r06s_1200_speed_stability_diagnostic`。
- 40 个第一阶段小文件，共 9,624,756 bytes；未修改原始回传。
- Job ID `73059442`；deck SHA-256 `eac511d8579d9cad92f08c41b5241129801df46b297140a3f601acacd5b0dd0e`，与冻结提交一致。
- LS-DYNA R15.0.2 MPP double；1 节点、32/32 ranks；stderr 为 0；无实际 mass scaling。
- Slurm walltime 为 72:00:00，冻结合同为 12:00:00。作业在 2 h 24 min 内完成，因此该偏差只属于分配元数据，不改变本次数值解。

## 第一阶段结果

- 正常到达 0.50 ms；1,177,110 cycles，低于 2,500,000 循环上限。
- Solver elapsed 8,614 s；平台时间 8,631 s。
- 总能量比范围 1.0–1.0；小时玻璃/内能最大绝对比 0。
- 终态滑移能绝对占比 1.212979%，通过 5% 门槛。
- 最小时间步 `3.29177e-7 ms`。
- 负体积单元 346 个：板件 70、破片 276；板件事件从 0.0092731 ms 持续至 0.46428 ms。
- 目标板件单元 39948 未报告负体积，但冻结门槛要求板件负体积总数为 0，因此第一阶段仍失败。

## 下载和科学裁决

- 停止下载 R06s 的 `d3plot*`、`d3part*` 与 `binout*`，避免约 22.51 GB 的无效重型下载。
- R06s 证明 1200 m/s 下全局能量与滑移能问题已消失，但板件负体积模型形式问题仍在。
- 没有准入终态贯穿响应、速度括区间、物理有效性、仿真引擎标定、毁伤树标定或参考数据标签。
- 停止继续速度扫描，释放一个板件负体积机制隔离算例 R06t。

## R06t 设计与构建

- 父 deck：R06s 原样回传 deck。
- 唯一模型形式因子：`CONTROL_SOLID.PSFAIL` 从 `0` 改为现有破片集合 `900001`；该集合只含 part 2002，排除板件 part 1002。
- 仅两行不同：TITLE 标识行和 CONTROL_SOLID 行；其余 deck 行逐行一致。
- 1200 m/s、0.50 ms、ENDCYC 2,500,000、D3PLOT/D3PART 0.00015 ms、ASCII 0.00025 ms、TSSFAC 0.05、DT2MS 0、材料、接触、自适应 SPH、网格及资源全部冻结。
- 预注册最小时间步底线 `3.29177e-8 ms`，即 R06s 观测最小值的十分之一；板件负体积仍要求为 0。
- 若板件消息消失但时间步或能量门槛崩溃，则拒绝该抑制方案，下一步转向局部板件失效表示，不继续调接触参数。

## 验证

- R06t 专项测试 6/6 通过。
- 受控全项目回归 190/190 通过；18 项依赖当前不在工作区的历史回传、已移除大型二进制或下载 ZIP，按精确测试名排除。
- Python 编译检查通过；JSON 严格生成；`git diff --check` 无空白错误。
- 上传目录含 9 个普通文件，共 2,575,941 bytes；无 ZIP。
- Head deck SHA-256：`11f62db3f8b27d1087988d5ad2abc572bddbd89570f6065500ddfe390a6459ae`。

## 交付

直接提交目录：

`D:\Desktop\UAV新\scnet_lsdyna_plate_calibration_20261017\artifacts\r06t_fragment_only_psfail_submission_20261103\uav_hf_r06t_1200_fragment_only_psfail_diagnostic`

计算后仍先只下载第一阶段小文件，不下载 `d3plot*`、`d3part*` 或 `binout*`。
