# R06f 800 m/s、TSSFAC=0.4 提交准备工作日志

日期：2026-10-28  
状态：`ready_for_user_submission_not_submitted_not_solver_validated`

## 1. 决策输入

- 采用平台 Job 72617857 作为外部控制基线；该作业已经通过正确资源复算审查。
- 只改变 `*CONTROL_TIMESTEP` 的 `TSSFAC`：`0.8 -> 0.4`。
- 保持 `DT2MS=0`，不启用质量缩放；保持 800 m/s、0.75 mm 以及其余模型、网格、材料、接触和输出设置不变。
- 本轮只释放一个诊断作业；2000 m/s 控制和所有 treatment 继续冻结。

## 2. 实施改动

1. 修复 `scripts/r06_single_job_adapter.py`：
   - 从真实 BINOUT 建立库存；
   - 直接读取 NCFORC 的变长状态并保留逐状态节点 ID；
   - 将未请求的 INTFOR 记录为 `not_requested_under_contract`；
   - 将 SLEOUT 明确标为 energy-only，不再把不可用证据当作零。
2. 新增冻结设计生成器 `scripts/create_p0_r06f_timestep_sensitivity_design.py`，生成单因素设计和输出合同。
3. 扩展 `scripts/generate_p0_r06_diagnostics.py`，支持按作业写入 TSSFAC。
4. 扩展 `scripts/validate_p0_r06_diagnostics.py`，与不可变 R06f 控制 deck 做字段级差异检查。
5. 新增 `scripts/stage_r06f_timestep_diagnostic.py`，只暂存一个 ZIP、外部清单、提交说明和阶段清单。
6. 为 Windows 临时目录发布加入短暂的有限重试，消除安全扫描导致的瞬时 `WinError 5/32`，不改变 ZIP 内容。
7. 新增/扩展测试，覆盖 NCFORC 变长状态、合同语义、单因素隔离、两次独立构建字节一致性和四文件暂存白名单。

## 3. 验证结果

- 批次静态就绪：PASS；1 个作业、1 个单因素实验。
- deck 差异：只允许 TITLE 与 `CONTROL_TIMESTEP` 第 2 字段；实际 TSSFAC `0.8 -> 0.4`，`DT2MS=0`。
- 资源合同：1 node / 32 MPP ranks。
- ZIP 独立复核：PASS；10 个显式白名单成员；head deck 为 `r06f_0800_tssfac04.k`。
- ZIP SHA-256：`54bb13c10c93da23925c75009c8a8ef9552fb86bd0052c1f2f5ad4718269e3cc`。
- ZIP 大小：413,598 bytes。
- 可执行全套回归：124 passed、0 failed；另有 1 个历史测试因本机缺失 `D:\Downloads\uav_hf_r06d_0800_ctrl.zip` 外部夹具而明确排除。
- 新增 TSSFAC 专项测试：4 passed；测量适配器专项测试：12 passed；相关脚本 `py_compile`：PASS。

## 4. 交付边界

静态、测试和封装通过只说明算例内部一致且适合用户提交。尚未在目标平台求解，因此不能据此声称：

- LS-DYNA R15.0.2 目标解析/运行已通过；
- 数值响应或失败机制已稳定；
- P0/P1、仿真引擎拟合或参考数据集标签已准入；
- 物理有效性已经建立。

## 5. 本轮提交动作

只上传 `uav_hf_r06f_0800_tssfac04.zip`，选择 `r06f_0800_tssfac04.k` 为 head deck。提交后先确认 `-N 1`、`--ntasks-per-node=32`、`mpirun -np 32` 和求解器 `MPP execution with 32 procs`；不满足则停止作业。

结果回传后，先审查正常终止、运行完整性、完整 D3PLOT/D3PART、HVAR6、NCFORC、能量和相对 Job 72617857 的 q 指标，再决定是否释放 2000 m/s 控制。
