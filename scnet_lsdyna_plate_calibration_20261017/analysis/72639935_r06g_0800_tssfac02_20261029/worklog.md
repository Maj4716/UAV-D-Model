# Job 72639935 / R06g 回传处理与 R06h 交付日志

日期：2026-10-29

## 输入与身份

- 回传目录：`D:\Desktop\UAV新\uav_hf_r06g_0800_tssfac02`
- 平台 Job ID：`72639935`
- 冻结作业：`r06g_0800_tssfac02`
- 求解器：LS-DYNA R15.0.2 MPP double precision
- 输入 deck SHA-256：`b5f4da367e9ef6eaae7098e47a84610eb55c51760cc9bf8dbb4721693b7a732a`
- 资源身份：1 节点、32 MPP ranks，通过 Slurm、平台日志与 `mes0000` 三重核对。

## 执行记录

1. 对原始回传运行单作业适配器，生成 `r06g_single_job_analysis.json`。
2. 独立重读 D3PART 和 full D3PLOT，校验 7 个通道、alive mask、HVAR6 和 NCFORC，生成 `independent_measurement_validation.json`。
3. 解析 32 个 rank 日志和 `glstat`，生成 `ascii_metrics.json`。
4. 生成可复现审计 `r06g_tssfac02_return_audit.json`，并冻结返回 deck 快照。
5. 逐个解析 `mes*` 中的负体积事件，再回查 deck 元素与 part 归属。
6. 冻结 R06h TSSFAC=0.1 单因子设计和输出契约，生成、静态校验单作业 batch。
7. 按用户要求生成直接文件夹交付，未创建 ZIP。

## R06g 结果与门禁

- 正常终止：通过；17,931 cycles；终止问题时间 0.040001 ms。
- D3PART/full-D3PLOT 交叉：通过；共享 402 帧；7 通道最大绝对差 0；alive mask 差异 0。
- HVAR6：可读，alive 非有限/哨兵值数 0。
- NCFORC：801 帧可读且数值有限；物理语义和单位仍待独立验证。
- 能量比范围：0.995864–1.00044；小时玻璃能为 0。
- 负体积唯一元素：41159、41201、41208；均属于 fragment part 2002，作为次级数值告警保留。
- 诊断响应：`qM=0`、`qP=0`、`qE=0`、`qV=null`，确认穿越数为 0。
- deleted-upstream mass fraction：0.045189503469741425；相对 Job 72630855 的绝对差约 0.00145773，通过 0.01 门禁。
- 0.4→0.2 时间步门禁：通过。
- 回应分类仍是 `transition_or_indeterminate`；4.519% 的 deleted-upstream 未解析质量不能被当成确认非穿透。

## 下一步决策

只授权一个 R06h：800 m/s、0.75 mm、TSSFAC=0.1、DT2MS=0。理由是 0.8→0.4 曾发生强不连续，
而 0.4→0.2 刚进入局部一致区；用一个成本较低的 0.1 点确认细时间步平台后，才选定生产时间步。

本轮不授权 2000 m/s、treatment、P0/P1 准入、仿真引擎拟合、毁伤树拟合或参考数据集提升。

## 爆炸瞬间起算约定

当前破片-板算例已经直接从爆炸后破片初始状态开始，没有仿真爆炸前飞行或起爆阶段。R06h 中初始几何、速度、
质量和其他状态完全冻结，以保证唯一变量是 TSSFAC。未来若换成整弹/全战斗部模型，可从爆炸时刻起算，但必须先冻结并审计
质量、动量、能量、破片状态和边界状态的映射契约。

## 验证记录

- R06h 定向回归：5/5 通过（审计复现、设计/契约复现、单因子 deck、batch 复现、无 ZIP 直接文件夹）。
- 项目受控测试：134 项选中测试 exit code 0。
- 排除 1 项历史测试：它依赖已不存在的 `D:\Downloads\uav_hf_r06d_0800_ctrl.zip`，与 R06h 不同源。
- 最终交付：9 个普通文件，2,591,201 bytes，ZIP 数 0，哈希清单通过。
- head deck SHA-256：`6e4fb3806722df91df8500ea3e2a6758ef6a07c36e913f4e8eccc7a9083374c3`。
- 与 Job 72639935 冻结 deck 的允许差异：TITLE 和 `CONTROL_TIMESTEP.TSSFAC` 0.2→0.1；`DT2MS=0`；其余块完全一致。

## 交付位置

- 交付根目录：`artifacts/r06h_tssfac01_submission_20261029`
- 直接上传目录：`artifacts/r06h_tssfac01_submission_20261029/uav_hf_r06h_0800_tssfac01`
- head/input deck：`r06h_0800_tssfac01.k`

用户回传该作业后，之后只需发送回传路径；按同一流程自动完成身份核对、数值审计、测量复核、门禁判定和下一个无压缩提交目录生成。
