# UAV Damage Model 项目状态

更新日期：2026-11-11（Asia/Shanghai）

GitHub：`https://github.com/Maj4716/UAV-D-Model`

主分支：`main`

## 1. 项目目标

本项目由两条相互约束的工作线组成：

1. `damage_engine`：破片、冲击波、装甲/部件响应、毁伤树、数据集生成、张量归档和训练接口；
2. `scnet_lsdyna_plate_calibration_20261017`：使用 LS-DYNA 高保真算例逐步验证时间步、接触、侵蚀、自适应 SPH、质量/能量账本和响应测量，为后续仿真引擎与毁伤树标定提供候选证据。

最终目标是用经过数值资格、物理解释和不确定性审查的高保真结果约束仿真引擎及毁伤树。当前尚未达到该目标。

## 2. 当前分层状态

| 层级 | 当前状态 | 说明 |
|---|---|---|
| 软件实现与合同一致性 | 已建立 | 引擎、数据集合同、审计脚本、LS-DYNA 单因素生成/回传审查和回归测试均已实现 |
| LS-DYNA 输入与资源身份 | R06u 通过 | Job 73179287 使用正确 deck、1 节点、32 ranks |
| 短时解析/运行能力 | R06u 通过 | 正常到达 0.15 ms，越过 R06t 的 0.10872 ms 崩溃点 |
| 数值工程准入 | 等待 | R06u 回传缺少 GLSTAT/MATSUM/SLEOUT/NODOUT/NCFORC，能量、正式最小时间步、质量和接触门尚未完成 |
| 物理有效性 | 未建立 | 自适应 SPH 能运行不等于板件断裂/破片化模型已被实验确认 |
| 仿真引擎标定 | 未准入 | 尚无满足全部数值和物理门的标定响应 |
| 毁伤树标定 | 未准入 | 不允许根据当前合成/诊断结果拟合阈值或条件关系 |
| 正式训练数据准入 | 未准入 | 高保真候选不得直接提升为参考标签或 `research_admission=true` |

测试通过或求解器正常终止只证明对应的软件/运行门，不自动证明数值稳健性、物理有效性或研究准入。

## 3. 高保真仿真进展

### 3.1 已完成的主要诊断链

- R06m：自适应 solid-to-SPH 能力成立，但平台实际使用 64 ranks，与冻结的 32-rank 合同不一致。
- R06n：32-rank 精确确认得到有效负结果；板件材料失效删除和资源分解敏感性阻止时长扩展。
- R06o：板件 `EROD=1` 单因素隔离通过，自适应 SPH、质量和能量工程门通过。
- R06p：扩展到 0.50 ms 并正常完成，但事前冻结的 D3PLOT 位移分辨率门失败，主要 QoI 继续不可用。
- R06q：通过更密 D3PLOT 对测量分辨率进行确认，随后才放行单一 2000 m/s 上锚点。
- R06r：2000 m/s 正常完成，但终态滑移能占初始能量绝对值 6.89495%，超过预注册 5% 上限；同时出现大量板件负体积事件，第一阶段工程门失败。
- R06s：1200 m/s 正常完成且总能量门通过，但仍有 70 个板件负体积单元，不能成为标定点。
- R06t：fragment-only PSFAIL 抑制板件删除消息，却在 0.10872 ms 触发板件单元 23830 `complex sound speed`，方案被否决。
- R06u 首次标签作业：作业名为 R06u，实际误运行冻结 R06t deck；输入身份门失败，不作为 R06u 证据。
- R06u Job 73179287：正确 R06u 正常完成 0.15 ms、244,053 cycles、32/32 ranks 正常终止；单元 23830 在 0.042562 ms 触发失效后，求解继续到终点，未再出现复杂声速错误。

### 3.2 R06u 当前门禁

R06u 已证明板件 part 1002 → 同材料 SPH part 1003 方案具备解析和短时运行能力，但当前本地回传缺少：

- `glstat`：正式最小时间步、总能量、滑移能和沙漏能；
- `matsum`：part 1003 正质量及分部质量/能量辅助证据；
- `sleout`：接触能量账本；
- `nodout`：历史节点运动；
- `ncforc`：节点接触力。

因此不得生成或提交 R06v，也不得直接运行 0.50 ms R06u 扩展。下一步唯一允许动作是从 Job 73179287 补下载上述五类小型 ASCII 文件；若存在则附带 `sldsph_frag`。在 Stage-1 通过前，不下载 `d3plot*`、`d3part*` 或 `binout*`。

最新详细报告：

- [72_SCNet_P0_R06u_Job73179287_Stage1部分通过与最小补充下载清单_2026-11-04.md](72_SCNet_P0_R06u_Job73179287_Stage1部分通过与最小补充下载清单_2026-11-04.md)
- [R06u 机器可读部分审计](scnet_lsdyna_plate_calibration_20261017/analysis/73179287_r06u_plate_adaptive_sph_stage1_20261104/r06u_stage1_partial_return_audit.json)
- [R06u 最小补充下载清单](scnet_lsdyna_plate_calibration_20261017/analysis/73179287_r06u_plate_adaptive_sph_stage1_20261104/DOWNLOAD_NEXT_MINIMAL.md)

## 4. 软件与数据引擎状态

当前软件版本为 `0.7.0.dev3`。主要能力包括：

- 破片空气阻力、BVH 候选筛选、逐层装甲侵彻、内部部件命中与状态概率；
- 冲击波比例距离、入射/反射载荷、装甲 P-I 响应和开孔混合透射；
- 破片—冲击波顺序耦合、Fréchet 并集上下界和版本化毁伤树；
- SamplingSpec 1.0–1.3、UAV-only/巡飞弹输入合同、候选拒绝账本和恢复身份校验；
- 分片数据集写入、checkpoint/resume、严格归档验证、PyTorch 适配器及 mask-safe 训练/评价；
- S-CONTACT 条件接触审计与独立的 software-only KMFC 有序投影原型。

仓库内演示物理参数及教师响应仍为 `unverified`。本地 5000 条软件归档证明生成/归档链可运行，但不构成现实装备效能或正式科研标签。

## 5. 仓库结构

- `src/damage_engine/`：核心仿真与数据合同实现；
- `schemas/`：版本化输入、标签、采样和归档 Schema；
- `tests/`：引擎与数据管线回归测试；
- `tools/`：数据审计、训练合同、可达性和可视化工具；
- `scnet_lsdyna_plate_calibration_20261017/design/`：R06 系列预注册设计与输出合同；
- `scnet_lsdyna_plate_calibration_20261017/scripts/`：deck 生成、提交暂存、ASCII/二进制解析和回传审计；
- `scnet_lsdyna_plate_calibration_20261017/tests/`：高保真交付与审计回归；
- `scnet_lsdyna_plate_calibration_20261017/analysis/`：轻量机器审计、工作日志与冻结父 deck；
- `scnet_lsdyna_plate_calibration_20261017/manual_upload_*` 与 `artifacts/`：可复核的直接提交目录。

## 6. 验证与复现

基础安装与完整软件测试：

```powershell
python -m pip install -r requirements.txt
python -m pip install -e ".[torch]"
python -m pytest -q
```

2026-11-11 使用本机 `pytorch312-lmdt` 环境执行根项目测试，全部通过。最近一次高保真工作区受控回归为 202/202 通过；另有 18 项依赖未随 Git 保存的历史外部回传、下载 ZIP 或大型二进制，按精确测试名排除。R06u Stage-1 部分回传专项测试为 3/3 通过。这些结果不替代目标平台复算或物理验证。

## 7. GitHub 同步边界

GitHub 保存：

- 源代码、Schema、设计合同、生成与审计脚本、测试；
- 轻量 JSON/Markdown 审计证据、冻结父 deck；
- 历史直接提交目录与交付说明；
- 项目状态和科研边界文档。

GitHub 不保存：

- `uav_hf_*`、`LS_DYNA_*` 原始求解回传；
- `d3plot*`、`d3part*`、`binout*` 等大型结果族；
- 可从原始结果重复生成的约 61 MiB 逐状态 `r06*_single_job_analysis.json`；
- 本地 5000 条数据集、临时目录、缓存和第三方 vendored 运行时。

上述排除是存储与证据管理边界，不表示原始证据可以删除。原始回传继续在本地保持只读，并通过 SHA-256、manifest、job contract 与外部审计 JSON 绑定。

## 8. 下一里程碑

1. 补齐 R06u Job 73179287 的五类小型 ASCII 数据库；
2. 通过正式最小时间步、能量、part 1003 正质量和接触/节点账本门；
3. 若 Stage-1 通过，再下载二进制结果验证 8:1 转换、solid+SPH 质量闭合和接触响应；
4. 只有二进制门也通过，才设计单一 0.50 ms 确认算例；
5. 获得多个通过数值与物理解释门的速度/工况点后，才开始仿真引擎与毁伤树标定。

当前最重要的原则是保持单因素、预注册、输入身份可追溯，并把 `unavailable` 保留为 `null + reason`，不得为了获得期望标签而改写毁伤树或物理语义。
