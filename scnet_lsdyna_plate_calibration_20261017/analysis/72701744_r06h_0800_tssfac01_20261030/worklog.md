# Job 72701744 / R06h 回传处理与 R06i 交付日志

日期：2026-10-30

## 输入与身份

- 回传目录：`D:\Desktop\UAV新\uav_hf_r06h_0800_tssfac01`
- 平台 Job ID：`72701744`
- 冻结作业：`r06h_0800_tssfac01`
- 包修订：`R06_timestep_confirmation_20261029`
- 求解器：LS-DYNA R15.0.2 MPP double precision
- 输入 deck SHA-256：`6e4fb3806722df91df8500ea3e2a6758ef6a07c36e913f4e8eccc7a9083374c3`
- 原始回传：112 个普通文件，2,130,999,055 bytes。

## 处理记录

1. 保持原始回传只读，核对 `manifest.json`、deck 大小和 SHA-256。
2. 对 Slurm、平台 log 和 `std.out` 交叉验证 1 节点、32 MPP ranks 和 R15.0.2 运行身份。
3. 运行单作业适配器，生成 `r06h_single_job_analysis.json`。
4. 独立重读 D3PART 和 full D3PLOT，验证 7 个通道、alive mask、HVAR6 和 NCFORC。
5. 解析 32 个 `mes*` 与 `glstat`，将负体积元素回查到 deck part。
6. 按预注册 0.2→0.1 门禁与 Job 72639935 比较，生成 `r06h_tssfac01_return_audit.json`。
7. 冻结 R06i TSSFAC=0.05 终止式细化设计、输出契约和单作业 batch。
8. 生成无 ZIP 的直接提交目录，并按文件白名单、大小和 SHA-256 复核。

## R06h 执行和测量结果

- 求解：正常终止；38,768 cycles；终止问题时间约 0.040000 ms。
- 资源：1 节点、32 ranks，`mpirun -np 32`，身份通过。
- D3PART/full-D3PLOT：802/402 帧；共享 402 帧的 7 通道最大绝对差 0；alive mask 差异 0。
- HVAR6：alive 无效值数 0；可读。
- NCFORC：801 帧可读且数值有限；物理语义、符号和单位仍未标定。
- 能量比范围：0.992358–1.00011；小时玻璃能为 0。
- 最终存活破片单元：635（Job 72639935 为 655）。
- 负体积元素：41158、41180、41187、41208、41257，均属于 fragment part 2002。
- 侵蚀能占初始能量：0.058380859375（基线为 0.04333559375）。

## 0.2→0.1 预注册门禁

- qM/qP/qE 诊断值：均为 0，与基线绝对差 0，通过。
- qV：两边均为 `null`，且均无确认穿越，按冻结规则记为不适用但通过。
- deleted-upstream mass fraction：0.07434402197408253。
- Job 72639935 基线：0.045189503469741425。
- 绝对差：0.029154518504341108，超过容差 0.01，失败。
- 主 QoI 可用性：失败；deleted-upstream 超过 0.05 上限，所以 qM/qP/qE/qV 正式输出必须保持 unavailable/null，不得把诊断值 0 当成可用非穿透标签。
- 总结：R06h 求解成功，但时间步确认失败。

## 下一步决策

授权一个 R06i 终止式时间步细化：800 m/s、0.75 mm、TSSFAC=0.05、DT2MS=0。它与 Job 72701744 的 deck
只允许 TITLE 和 `CONTROL_TIMESTEP.TSSFAC: 0.1 -> 0.05` 不同。

终止规则：

- R06i 后不自动生成 TSSFAC=0.025。
- 若比较门禁和响应可用性全部通过，才可将 0.05 作为局部生产候选并评估是否开放 2000 m/s。
- 若诊断值一致但响应仍不可用，停止时间步减半，转入破片删除/材料失效/接触诊断。
- 若任一比较门禁失败，停止时间步减半，将当前控制算例记为数值/模型形式不稳定。

本轮不授权 2000 m/s、treatment、P0/P1 准入、仿真引擎拟合、毁伤树拟合或参考数据集提升。

## 验证记录

- R06i 定向回归：5/5 通过。
- 项目受控测试：137 项选中测试 exit code 0。
- 排除 3 项历史外部夹具测试：1 项依赖已不存在的 `D:\Downloads` R06d ZIP；2 项依赖用户已移动的 R06f/R06g `candidate_bundle_manifest.json`。
- 最终交付：9 个普通文件，ZIP 数 0，文件白名单、大小和哈希全部通过。
- head deck SHA-256：`e2e8da8e55304b7fe2deb2c1790e00bf7fea9c64145c6c09a954c50492f3dd88`。

## 交付位置

- 交付根目录：`artifacts/r06i_tssfac005_submission_20261030`
- 直接上传目录：`artifacts/r06i_tssfac005_submission_20261030/uav_hf_r06i_0800_tssfac005`
- head/input deck：`r06i_0800_tssfac005.k`

计算完成后，用户只需发送完整回传路径；继续按同一流程自动审计并决定选定时间步或转入删除/材料/接触机制诊断。
