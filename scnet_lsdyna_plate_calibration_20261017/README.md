# SCNet / LS-DYNA 破片—装甲 P0 标定包

**状态：** R04 已完成回传审查但响应资格不通过；历史 R05 候选随后以 Job `72252362` 成功运行。该作业的哈希绑定 run-integrity 为 `PASS`，修复后的适配器可读取真实 D3PART，但独立 integration acceptance 尚未签发，且 7 个工况只有 600 m/s 的诊断账本可用，其余 6 个均因捕获面前删除超限而 `unavailable`。R06d 两个 control（Jobs `72269229`、`72269336`）均在 cycle 0 的 R15.0.2 关键词读取阶段失败，历史 R06d 包已撤回。R06e 已修复数据库卡片合同、重新生成四个作业并通过本地静态与发布校验；当前只允许提交 `r06e_0800_ctrl` 作为 Wave 0，状态仍是 `ready_for_user_submission_not_submitted_not_solver_validated`。P0 响应拟合、P1、引擎拟合和数据集晋级仍为 `DENIED`。  
**证据等级上限：** `high_fidelity_synthetic_reference_candidate`。  
**单位制：** kg–mm–ms–kN–GPa–K，因此速度数值 `mm/ms` 与 `m/s` 相同。

本包用于资格检查当前项目的低层装甲响应模型。R04 的五个 0.75 mm 速度点全部出现穿孔代理，且 2000 m/s 三网格响应不一致，因此只保留为数值诊断证据。R05 在一份 deck 中并列布置七个互不接触的四分之一对称子模型：0.75 mm 速度扫描为 600/800/1000/1200/2000 m/s，2000 m/s 的 1.00/0.75/0.50 mm 三网格压力测试保持不变。Job `72252362` 的返回目录 `05/` 是只读历史证据；所有阶段 A/B 派生物写入 `analysis/72252362_r05_stage_ab_20261023/`。

P0 固定为 8 g、10 mm、法向、立方体面迎击和 AISI 4340 Johnson–Cook 中心材料卡。四分之一对称只适用于这一中心切片。质量、厚度、斜入射、姿态和材料不确定性的 24 个 P1 三维工况已经在 `design/p1_preregistered_cases.csv` 中冻结，但只有 P0 通过求解器与数值 QA 后才允许提交。

## R06d 失败与 R06e 解析修复候选

R06d 不再把历史七工况大 deck 当作新单工况变体的直接因果基线，而是配置两个配对实验、共四个独立作业：

1. `r06d_0800_ctrl` 与 `r06d_0800_tend008`：唯一因素为终止时刻 `0.04 → 0.08 ms`，用于检验 800 m/s 的观察窗与尾窗稳定性；
2. `r06d_2000_ctrl` 与 `r06d_2000_psfail_plate`：`*CONTROL_SOLID` Card 1 仅有 `PSFAIL: 0 → 910001`，其中 910001 只含 plate PID 1005，用于归因 fragment 负体积删除的贡献。

R06d 两个 control 的回传 ZIP、manifest 和本地发布 deck 哈希完全一致，许可证和 32-rank 启动正常；但 `*DATABASE_EXTENT_BINARY` 只写了 Card 1，R15.0.2 将后续 `*DATABASE_BINARY_D3PART` 吞作缺失的 Card 2，继而产生 `KEY+246`、级联 `KEY+450` 和最终 `KEY+133`。同时，`*DATABASE_BINARY_INTFOR_FILE` timing card 写了五字段，而该 parser 只接受四字段，产生 `KEY+457`。两作业均未进入时间积分，因此没有数值或物理结论。

R06e 保持网格、坐标、连接、材料、EOS、接触、时间步、速度、边界条件、两个预注册因素和五字段 D3PART 不变，只修复共享测量序列化：`*DATABASE_EXTENT_BINARY` 显式写完整两卡，Card 1 为 `6,0,3,0,1,1,1,1`（仅覆盖 `NEIPH=6`，其余为 R15 默认），Card 2 为 `0,0,0,1,1,1,2,1`；INTFOR timing card 改为四字段 `5e-5,0,0,0` 并保留下一卡 `IOOPT=0`。生成器与验证器均加入这一精确合同和反回归测试。

历史失败包 `manual_upload_20261023_r06_diagnostics/` 与 `artifacts/r06_diagnostics_20261023/` 只作证据，不得再次提交。新候选目录为 `manual_upload_20261025_r06e_parser_repair/`，完整四包发布目录为 `artifacts/r06e_parser_repair_20261025/`。为避免越过闸门，下一次只使用 `artifacts/r06e_wave0_canary_20261025/uav_hf_r06e_0800_ctrl.zip`；只有该作业无相关 KEY 诊断、越过 cycle 0、正常终止且真实 D3PART/INTFOR 可读后，才提交 `r06e_2000_ctrl`。两个 control 均通过后才允许提交两个 treatment。

```powershell
& 'D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe' scripts\analyze_r06_wave1_keyword_failure.py
& 'D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe' scripts\create_p0_r06_parser_repair_design.py
& 'D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe' scripts\generate_p0_r06_diagnostics.py `
  --design design\p0_r06_keyword_parser_repair_design.json `
  --output-dir manual_upload_20261025_r06e_parser_repair
& 'D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe' scripts\validate_p0_r06_diagnostics.py `
  --batch-dir manual_upload_20261025_r06e_parser_repair --write-reports
& 'D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe' scripts\build_p0_r06_diagnostic_bundles.py `
  --batch-dir manual_upload_20261025_r06e_parser_repair `
  --output-dir artifacts\r06e_parser_repair_20261025
& 'D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe' scripts\stage_r06e_wave0_canary.py
```

上述 `PASS` 只证明 R06e 候选的版本、哈希、关键字字段、单工况身份、配对因素隔离和打包一致性；本机没有 LS-DYNA 可执行文件，目标求解器解析/运行仍须由 SCNet 作业确认。R06e 诊断结果也不会自动成为 P0 响应或训练标签。

## R05 Job 72252362 与阶段 A/B 结论

Job `72252362` 使用 LS-DYNA MPP double R15.0.2、1 节点/32 ranks 正常终止。阶段 A 已生成覆盖 75 个关键文件、总计 2,944,160,835 bytes 的哈希清单，并确认 32/32 rank 正常终止、fatal/error 为 0、总能量比 `0.968466–1.00216`、沙漏能/峰值内能为 0、added-mass parts 为 0、`dt2ms=0`。`o=` 求解器文本输出按冻结等价规则满足 d3hsp，BINOUT 逻辑数据库满足 matsum/nodout/sleout。

真实 D3PART 回归包含 802 states、7,737 nodes、5,680 fragment solids；实际被 solid 引用的 PID 精确为 2001–2007，plate PID 1001–1007 只是零实体 catalog 元数据。时序门改为“名义输出间隔 + 区间内正的 GLSTAT 显式步长 + 4 ULP 时间字容差”，最大实际间隔 `6.462719e-5 ms`，最大实际/可实现上界比 `0.995374`；7 个工况均无 alive-to-alive 的 `0.5×mesh` 跨步位移超限。该结果状态仍为 `real_d3part_exercised_pending_independent_integration_acceptance`，不会追溯提升历史 R05 的准入等级。

阶段 B 对 32 个 `mes*` 去重得到 2,248 个唯一失效实体：负体积 628 个、其他实体失效 1,620 个，无跨类型重复。fragment 的“其他实体失效”显著多于负体积失效，因此单独限制负体积作用域不能解决主要删除。2000 m/s 三网格的捕获面前删除质量比例为 75.600%（1.00 mm）、65.306%（0.75 mm）、12.200%（0.50 mm），表现出严重网格依赖。只有 600 m/s 可诊断为 confirmed nonperforation；其余 6 个主 QoI 均不可用。

完整 D3PLOT 的 82 个保存状态还用于提取 800 m/s 与 2000 m/s、0.75 mm 两个重点工况的 alive-solid 有效塑性应变和实体静水压力历史；后者严格由 `p=-(sxx+syy+szz)/3` 派生、单位为 GPa，并非接触压力。直接接触穿透与接触压力因 D3PLOT flags 和 BINOUT 变量均未写出而保持 `unavailable_not_written`。三组图表保留结构零、原始事件点和冻结 5% 门，不进行平滑或因果归因。

先前 Job `72197064` 的 FlexNet `-15,570` 许可失败仍保留为历史提交诊断，但已被成功运行的 `72252362` 取代为当前结果来源。

## 阶段 A/B 入口与证据

主 QoI 是仅含 fragment 零件的高频 D3PART 在 `z=-12 mm` 捕获面上的初始参考质量账本：`qM/qP/qE/qV_over_v0`。中心历史节点和末态存活 part 均值只作诊断。`qP` 固定为负 z 出射方向为正的标量动量，`qE` 固定为八节点算术质心的平动动能；不可用输入不会被当作零响应。

```powershell
$env:TEMP = 'D:\Desktop\UAV新\scnet_lsdyna_plate_calibration_20261017\.tmp'
$env:TMP = $env:TEMP
& 'D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe' scripts\build_r05_run_integrity.py `
  --result-dir 05 `
  --output-dir analysis\72252362_r05_stage_ab_20261023 `
  --run-id 72252362 `
  --solver-build 'LS-DYNA MPP double R15.0.2-1-g1e7cfa42b1 I8R8 Intel-MPI 2018'
& 'D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe' scripts\extract_r05_d3part.py `
  --package-dir manual_upload_20261021_r05_candidate `
  --d3part 05\d3part `
  --run-id 72252362 `
  --solver-build 'LS-DYNA MPP double R15.0.2-1-g1e7cfa42b1 I8R8 Intel-MPI 2018' `
  --output analysis\72252362_r05_stage_ab_20261023\r05_d3part_extraction_integration_pending.json
& 'D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe' scripts\analyze_r05_failure_modes.py `
  --return-dir 05 `
  --output-dir analysis\72252362_r05_stage_ab_20261023 `
  --replace-owned
& 'D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe' scripts\plot_r05_failure_modes.py `
  --events-csv analysis\72252362_r05_stage_ab_20261023\failure_events_unique.csv `
  --summary-csv analysis\72252362_r05_stage_ab_20261023\failure_summary_by_case.csv `
  --design-json 05\p0_cases_r05.json `
  --output-dir analysis\72252362_r05_stage_ab_20261023 `
  --replace
& 'D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe' -m unittest discover -s tests -p 'test_r05*.py' -v
& 'D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe' -m unittest discover -s tests -p 'test_r06*.py' -v
```

阶段 A 的 run-integrity 审计、真实 D3PART extraction 与阶段 B 的失效诊断相互独立；只有独立的 integration acceptance 与 run-integrity 都明确 PASS，并桥接到同一 run/deck/design/D3PART family 身份时，才允许产生 assessor 准入载荷。当前未生成该桥接载荷。

R06 的真实输出合同冻结在 `design/p0_r06_real_output_contract.json`：D3PART 严格使用五字段 `DT,CYCL,IOOPT,LCDT,PSETID`，PID 门采用 used-solid 语义，并固定上述 solver-realizable 时序门及回传等价规则。该文件的 `r06_deck_status=not_generated`、`r06_submission_status=not_submitted` 是生成前冻结的历史快照，刻意不追改；当前生命周期由 R06 design、batch/job manifest、静态 readiness 与 release verification 独立记录，也不得改写 `manual_upload_20261021_r05_candidate/` 或 Job `72252362` 的历史证据。

历史 R05 候选位于 `manual_upload_20261021_r05_candidate/`，白名单审计包位于 `artifacts/uav_hf_p0_plate_r05_candidate.zip`；其 manifest 字面状态仍是提交前状态，不能用来覆盖实际 Job 的独立 run identity。完整阶段 A/B 结论见项目上级目录 `52_SCNet_P0_R05_阶段A与B执行报告_2026-10-23.md`。

## 历史 R04 本地生成与验证

```powershell
& 'D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe' scripts\generate_p0.py
& 'D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe' scripts\generate_p1_design.py
& 'D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe' scripts\validate_package.py
& 'D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe' scripts\selftest_keyword_format.py
& 'D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe' scripts\selftest_postprocessor.py
& 'D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe' -m unittest discover -s tests -v
```

本子包的阶段 0–2 ASCII 后处理依赖记录在 `requirements.txt`。解析与测试使用 Python 3.12；`numpy` 和 `matplotlib` 仅用于派生数值与静态图表，原始 SCNet 文件保持只读。

## R04 回传数据与阶段 0–2 复核入口

输入数据来自 SCNet Job `72095164` 的原始下载包：

```text
D:\Desktop\UAV新\uav_hf_p0_plate_r04.zip
```

该 ZIP 的已审计 SHA-256 为 `2d1339f14b742eecdd121d57d400bc6c279c6dedfdbb08add7d4c634abe2f8c5`。安全展开副本位于 `results/72095164_R15_mpp_d/`；用户下载后的展开目录位于项目根目录 `uav_hf_p0_plate_r04/`。两者都作为只读来源，所有派生产物写入独立 `analysis/` 目录。

从项目根目录运行：

```powershell
$env:TEMP = 'D:\Desktop\UAV新\.tmp\scnet_r04_stage02'
$env:TMP = $env:TEMP
& 'D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe' `
  scnet_lsdyna_plate_calibration_20261017\scripts\analyze_r04_numerical_qa.py `
  --result-dir scnet_lsdyna_plate_calibration_20261017\results\72095164_R15_mpp_d `
  --download-dir uav_hf_p0_plate_r04 `
  --archive uav_hf_p0_plate_r04.zip `
  --intake-analysis scnet_lsdyna_plate_calibration_20261017\analysis\72095164_return_audit_20261021 `
  --case-map scnet_lsdyna_plate_calibration_20261017\decks\p0_case_part_map.csv `
  --deck scnet_lsdyna_plate_calibration_20261017\decks\p0_velocity_mesh_calibration.k `
  --output-dir scnet_lsdyna_plate_calibration_20261017\analysis\72095164_r04_stage0_2
```

该入口重新核对 ZIP 与两份展开目录的 100 个文件哈希，解析真实 `glstat` 点号格式，扫描全部 32 个 `mes*`，把负体积事件绑定到冻结的 case/part map，并生成全局数值 QA。它不执行回传脚本，不修改 deck，不读取封存测试集，也不把全局 QA 当作逐工况穿孔、物理有效或研究准入。

## R04 阶段 3–5 逐工况响应与准入入口

二进制读取器局部安装在 `.vendor/lasso-python-2.0.4`，固定
`lasso-python==2.0.4` 与 `h5py==3.16.0`。原始回传继续保持只读。从项目根目录运行：

```powershell
$env:TEMP = 'D:\Desktop\UAV新\.tmp\scnet_r04_stage35'
$env:TMP = $env:TEMP
& 'D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe' `
  scnet_lsdyna_plate_calibration_20261017\scripts\analyze_r04_stage3_5.py `
  --result-dir scnet_lsdyna_plate_calibration_20261017\results\72095164_R15_mpp_d `
  --case-map scnet_lsdyna_plate_calibration_20261017\decks\p0_case_part_map.csv `
  --stage0-2-qa scnet_lsdyna_plate_calibration_20261017\analysis\72095164_r04_stage0_2\numerical_qa.json `
  --vendor-dir scnet_lsdyna_plate_calibration_20261017\.vendor\lasso-python-2.0.4 `
  --output-dir scnet_lsdyna_plate_calibration_20261017\analysis\72095164_r04_stage3_5
```

该入口核验 BINOUT/d3plot 可读性，按冻结 case map 提取存活 part 的质量、速度、能量和单元删除；
中心历史节点若末态已无存活实体支撑则显式标为 unavailable。随后对冻结速度扫描、三层网格和 P0/P1
准入门做 fail-closed 判定。它不修改 deck，不提交 P1，不拟合本地引擎，不生成数据集，不训练模型，也不读取封存验证/测试。

静态 `PASS` 只证明案例身份、哈希、长字段声明、数值字段宽度、全量关键字块基数、卡片数、字段数与 `I/F` 类型合同、网格引用、正的结构化六面体方向、初始间隙、离散体积/质量积分、P1 分层和必需关键词在本包内自洽。R01 因普通字段模式与 15–16 字符数值项冲突失败；R02 因必需卡片/字段不完整且 `PHASE` 类型错误失败；R03 虽补全卡片，却把 `*CONTROL_CONTACT` Card 2 的整数型 `ECDT` 写成浮点科学计数，且旧校验器漏检了该字段。R04 将 `ECDT` 显式序列化为整数，并对生成器输出的全部 23 种关键字、110 个关键字块实施逐卡逐字段合同检查。静态检查仍不能替代 LS-DYNA 实际解析，也不证明能量、接触、侵蚀或穿透结果合理。

## SCNet 提交与回传合同

1. 历史 R05 已冻结，不再原样重提；不允许提交新的七工况 R06。当前只允许按 Wave 1/Wave 2 顺序提交 `manual_upload_20261023_r06_diagnostics/` 中独立版本化、每包单工况的 successor deck。
2. 后续候选使用平台的 LS-DYNA MPP 双精度模板；记录求解器完整 build、可执行文件身份、MPI 实现、队列、节点/ranks、申请与实耗资源。
3. 提交输入、设计、case map、材料与回传文件都必须进入独立 `run_identity` 和哈希清单，不改写历史候选的 manifest 或 `05/`。
4. 回传必须包含 `mes*`、`glstat`、`binout*`、`d3plot*`、`d3part*`、solver stdout/stderr、Slurm 脚本与资源记录。命令绑定且纳入哈希清单的唯一 solver `o=` 输出可作为 d3hsp 等价文本；BINOUT 内的 matsum/nodout/sleout 逻辑数据库可替代独立 ASCII 文件。
5. 上述等价规则只证明回传完整性，不证明响应、数值收敛或物理有效性。

`artifacts/uav_hf_p0_plate_r05_candidate.zip` 只作为历史 R05 审计与传递包，不再作为当前提交入口。

## 结果准入

P0 至少满足以下条件后，才能进入响应拟合：

- 正常终止且无未解释的 fatal/error termination；
- 总能量比候选范围 0.95–1.05；
- 小时玻璃能相对峰值内能的绝对比候选上限 5%；本 deck 使用全积分 S/R 实体，显著非零小时玻璃能需解释；
- 不使用质量缩放；若求解器报告附加质量，则 P0 拒绝；
- 负体积、删除单元、接触警告必须按 part 和工况解释；
- D3PART 相邻状态间隔必须不大于“名义间隔 + 同区间最大正 GLSTAT 显式时间步 + 4 ULP 时间字容差”；缺失、非有限、不单调或覆盖不足的 solver-step 证据一律 fail closed；
- alive-to-alive 跨捕获面的定向质心位移不得超过 `0.5×mesh`；超限以及 alive-to-deleted 区间内可能跨面均计入 temporal ambiguity，而不是补成确定穿孔；
- D3PART 初始离散质量相对 case map 预期质量误差不超过 `0.5%`；账本内部质量闭合误差不超过 `1e-6`，跨步歧义质量不超过 1%，捕获前删除质量不超过 5%，且末 10% 时间窗稳定；缺失或非有限输入一律 fail closed；
- 数值穿孔定义为 `qM>=2%` 且 `qP>=1%`；数值未穿孔除 `qM<=0.5%` 且 `qP<=0.5%` 外，还要求 `qM+跨步歧义率+捕获前删除率+正向缺失参考质量率<=0.5%`；中间区间保持不确定；
- 速度扫描至少各含一个确定未穿孔点和确定穿孔点，最近括区宽度不超过 200 m/s，且不得出现穿孔到未穿孔反转；`qV_mps=qV_over_v0*impact_speed_mps`，只比较完整速度序列中相邻且都确定穿孔的工况，向下跳变不得超过 100 m/s；
- 2000 m/s 三网格分类一致且主 QoI 全部可用；粗到中网格各无量纲主量绝对差不超过 0.10，中到细网格不超过 0.05。

这些阈值是本项目预注册的候选工程门，不是普遍适用的 LS-DYNA 标准；新增的 successor 时序/PID规则不能追溯把 R05 Job `72252362` 变成合格响应。

## 与本地引擎的关系

P0 不直接输出 KMFC。通过的连续响应只用于拟合新的、版本化的装甲响应模型，包括确定性数值转变速度/区间（不是实验 V50）、残余速度和能量传递。现有 `demo_armor_response` 不覆盖；现有已观察 Test 不参与选参；P1 holdout 不参与材料卡或公式选择。

## 主要证据来源

- [SCNet LS-DYNA 使用说明](https://www.scnet.cn/help/docs/mainsite/hpc/software-examples/use/CAE/LSDYNA/index.html)
- [Ansys LS-DYNA Projectile 官方示例](https://lsdyna.ansys.com/projectile/)
- [Ansys LS-DYNA 结构冲击建议](https://lsdyna.ansys.com/recommendations-for-structural-impact-2/)
- [Ansys LS-DYNA Hourglass 说明](https://lsdyna.ansys.com/hourglass/)
- [Ansys LS-DYNA Total Energy 说明](https://lsdyna.ansys.com/total-energy-2/)
- [Johnson–Cook 1983 原始论文扫描件](https://ia600406.us.archive.org/4/items/AConstitutiveModelAndDataForMetals/A%20constitutive%20model%20and%20data%20for%20metals_text.pdf)

材料参数的逐字段来源与限制记录在 `materials/jc4340_reference.json`。
