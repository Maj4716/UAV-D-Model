# R06t Job 73087868 失败审查与 R06u 交付工作日志

日期：2026-11-04

## 输入与执行身份

- 只读审查目录：`D:\Desktop\UAV新\uav_hf_r06t_1200_fragment_only_psfail_diagnostic`。
- 39 个回传文件，共 7,642,872 bytes；未修改原始回传。
- Job ID `73087868`；deck SHA-256 `11f62db3f8b27d1087988d5ad2abc572bddbd89570f6065500ddfe390a6459ae`，与冻结提交一致。
- LS-DYNA R15.0.2 MPP double；1 节点、32 ranks；许可证正常；stderr 为 0；无实际 mass scaling。
- Slurm walltime 72:00:00，与冻结合同 12:00:00 不同，但本轮在 1 h 24 min 内错误终止，分配上限不是失败原因。

## 失败证据

- 求解进入动态阶段，不是提交、许可证或关键词读取失败。
- 错误终止于 0.10872 ms、603,398 cycles；solver elapsed 5,080 s。
- 终止错误：`Error 40510 (SOL+510) complex sound speed in solid element #23830`。
- 元素 23830 属于板件 part 1002；终止时间步骤降为 `1.99e-27 ms`。
- 终止前负体积单元 415 个，全部属于破片；板件负体积消息为 0。
- R06t 因而证明 PSFAIL=900001 能抑制板件自动负体积删除，但不能给板件失效区提供可继续演化的材料/单元表示。
- R06s 中板件单元 23830 已属于负体积集合；R06t 只是将同一局部问题从删除转化为复杂声速和时间步崩溃。

## 裁决

- 否决 fragment-only PSFAIL 抑制方案，不重提 R06t。
- 不下载 R06t 的 `d3plot*`、`d3part*` 或 `binout*`。
- 不进行更多接触参数试错。
- 当前没有终态响应、物理有效性、引擎标定、毁伤树标定或参考标签准入。

## R06u 设计

- 从正常完成的 R06s deck 分支，而不是从失败的 R06t 分支。
- `CONTROL_SOLID.PSFAIL` 恢复为 0。
- 保留破片 part 2002 → SPH part 2003 的既有转换。
- 新增板件 part 1002 → 同材料、同 EOS 的 SPH part 1003；NQ=2，即每个失效六面体生成 8 个 SPH 粒子。
- 新增 part 1003 对 solid set 920001 的专用 SOFT=1 节点-面接触；固体侵蚀接触仍保持 SOFT=2。
- 本轮不增加 SPH-SPH 显式接触，终态物理响应继续禁止；本轮只验证板件转换解析、运行、质量和能量能力。
- 终止时刻缩短为 0.15 ms，覆盖 R06t 的 0.10872 ms 崩溃点，避免在模型形式尚未通过前直接运行完整 0.50 ms。
- 速度 1200 m/s、网格 0.75 mm、TSSFAC 0.05、DT2MS 0、材料常数、输出步长及 1 节点/32 ranks 冻结。

## 方法依据

- ANSYS LS-DYNA 指南说明，Adaptive Solid to SPH 可在 Lagrangian solid element 失效时以 SPH 粒子替代，并继承节点与积分点状态。
- NQ=2 对六面体对应 2×2×2=8 个粒子。
- 依据：`https://ansyshelp.ansys.com/public/Views/Secured/corp/v252/en/lsdyna_ug/lsdyna_ug_adaptive_solid_sph.html`。
- 关键词定义依据：`https://ansyshelp.ansys.com/public/Views/Secured/Doc_Assets/Release/v242/LS-DYNA_Manual_Volume_I_R14.pdf`。

## 验证与交付

- R06u 专项测试 6/6 通过。
- 受控全项目回归 196/196 通过；18 项依赖当前不在场历史外部回传/大型二进制/ZIP 的测试按精确测试名排除。
- Python 编译检查通过；`git diff --check` 无空白错误。
- 上传目录 9 个普通文件，共 2,579,673 bytes；无 ZIP。
- Head deck SHA-256：`7bfb99a49bb757632237b2e2c625500fef27e811988eac3149df7a3e056647d4`。

直接提交目录：

`D:\Desktop\UAV新\scnet_lsdyna_plate_calibration_20261017\artifacts\r06u_plate_adaptive_sph_canary_submission_20261104\uav_hf_r06u_1200_plate_adaptive_sph_canary`
