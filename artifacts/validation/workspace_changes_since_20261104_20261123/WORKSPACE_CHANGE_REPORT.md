# 2026-11-04 以来工作区保留改动审计与同行汇报报告

## 1. 核心结论

自 2026-11-04 以来，项目完成了从“高保真求解标定工作区”向“软件教师、数据生成、采样设计、学习接口与统计验证仓库”的主线切换，并在 11 月 20—23 日集中形成了连续方向采样、dataset-v2、逐任务准入、快速代理收敛和各向异性三级快评体系。

当前应向同行传达四点：

1. **仓库方向已经改变。** 11 月 11 日同步的 R06u 高保真工作流于 11 月 17 日退出活动树；当前主线是软件层数据生成与验证。
2. **当前工作树显著领先于 Git HEAD。** 相对 `ff2ef1f`，已有 43 个 tracked 文件发生未提交修改（+5,495/-396 行）；另有 135 个研究功能文件尚未跟踪，覆盖 28 个核心模块、25 个配置、30 个工具、45 个测试和 7 个文档/示例。本次审计另新增 1 个清单生成工具。
3. **数据链已经完成逐环节量化，并发现两个会改变正式分布的上游缺陷。** Taylor–Shapiro 方向符号已由 v3 的 0% 合同符合率修复为 v4 的 100%；瞄准切向基修复后 S-CONTACT 接受率由 60.5% 提升至 98.4%，region TV 由 0.2015 降至 0.0127。
4. **当前正式数据需要按新链路重建。** 11 月 22 日的 2,600 工况数据包证明了旧链路下的内部一致性、监督可用性和验证集可学习性；它使用 v3 物理与 SamplingSpec v1。当前 v4 物理与 SamplingSpec v2 修复了分布上游，因此下一份正式数据应使用新身份重新生成，并重跑代理与教师比较。

报告将软件一致性、统计证据、物理有效性和科研准入分层呈现。当前工程诊断体系已建立；外部物理标定仍需独立实验或现场参照。

## 2. 审计范围与快照

- 截止口径：2026-11-04 00:00（Asia/Shanghai，含当日）至 2026-11-23。
- 分支与 HEAD：`main`，`ff2ef1f21c7d6bf73630bdb11331289278fb69dd`。
- 快照时间、精确文件数和字节数以 [`inventory_summary.json`](inventory_summary.json) 为准。
- 扫描排除：`.git/`、`.pytest_cache/`、`__pycache__/`、`.pyc/.pyo` 以及本报告输出目录自身。
- Git 历史用于重建已提交路径；当前未跟踪和忽略文件以文件系统 mtime 识别。已在当前快照前删除、且从未进入 Git 的临时文件无法由现有工作区重建。

### 2.1 当前规模

| 层级 | 数量 | 规模/状态 | 解释 |
|---|---:|---:|---|
| 截止日后 Git 提交 | 3 | 1,863 条历史路径变化；其中 1,181 条删除路径记录 | 包含已退出活动树的高保真工作线 |
| 当前 tracked 未提交差异 | 43 文件 | +5,495/-396 行 | 尚未形成正式 Git 基线 |
| 当前未跟踪研究功能文件 | 135 文件 | 约 1.51 MB | 28 模块、25 配置、30 工具、45 测试、7 文档/示例 |
| 本次审计工具 | 1 文件 | `tools/build_workspace_change_inventory.py` | 生成本报告的机器清单 |
| 当前保留变化文件 | 1,885 文件 | 520,258,306 bytes（496.16 MiB） | 含代码、数据、验证和探索工件；不含本报告目录 |
| 其中 `artifacts/` | 1,602 文件 | 515,108,070 bytes（491.25 MiB） | 数据集、验证证据和探索运行占主体 |
| 扫描错误 | 0 | — | 当前纳入路径均可读取元数据 |

完整明细：

- [`commit_summary.csv`](commit_summary.csv)：提交级统计；
- [`historical_commit_changes.csv`](historical_commit_changes.csv)：每个提交的路径级状态；
- [`tracked_worktree_diff.csv`](tracked_worktree_diff.csv)：43 个 tracked 未提交文件及行数；
- [`current_change_inventory.csv`](current_change_inventory.csv)：当前保留文件的状态、主题、大小与 mtime；
- [`directory_summary.csv`](directory_summary.csv)：顶层目录汇总；
- [`git_status_snapshot.txt`](git_status_snapshot.txt)：审计时 Git 状态快照。

## 3. 已提交历史

| 日期 | 提交 | 标准 Git 统计 | 作用 | 当前状态 |
|---|---|---:|---|---|
| 11-11 10:08 | `56ba16e` `checkpoint: sync R06u calibration workflow and project status` | 535 文件，+1,869,720/-11 | 同步 R06m–R06u 高保真 LS-DYNA 工作流、R06u Stage-1 证据和项目状态 | 主体已由 11-17 重构移出活动树，保留在 Git 历史 |
| 11-11 10:09 | `77d779f` `docs: update relocated GitHub URL` | 1 文件，+1/-1 | 更新项目状态文档中的仓库地址 | 文档随后随重构退出活动树 |
| 11-17 09:58 | `ff2ef1f` `refactor: focus project on data generation and validation` | 1,327 文件，+37,943/-4,959,272 | 退役高保真工作区与旧研究池，建立当前代码、配置、数据、验证和文档结构 | 当前 HEAD |

11 月 17 日重构删除 9,617,546,016 bytes（8.96 GiB），将活动树收敛到 276 个文件、73,362,650 bytes；当时 1,044 项测试通过。重构时开发包包含 10,589 个唯一候选、7,337 个完整响应、3,252 个域删失样本和 16,000 条选择观测。详见 [`PROJECT_REFACTOR_REPORT.md`](../../refactor_20261117/PROJECT_REFACTOR_REPORT.md)。

## 4. 现存工作时间线

现存 `artifacts/` 的时间戳从 11 月 14 日开始；11 月 4—13 日的工作主要通过 Git 历史和后续重构结果追踪。以下时间线只报告当前仍可读取的证据或可复核的提交。

### 4.1 11 月 14—17 日：软件教师调参与仓库转向

- **11 月 14 日，穿透/毁伤阈值软件调参。** R512 最终复核给出装甲穿透率 0.5251、部件穿透率 0.4295、命中后正毁伤率 0.600、降级率 0.3275、失效率 0.1353，能量闭合最大误差为 0。该工件标记为 `software_tuning_only_not_physical_calibration`。证据：[`p012_selected_verification_r512.json`](../penetration/p012_selected_verification_r512.json)。
- **11 月 16 日，引擎 0.11 软件验收。** 1,031 项测试通过；67 个部件中 44 个被毁伤树引用，191 个目标参数保持 `unverified`，`research_admission_ready=false`。证据：[`acceptance_summary.json`](../engine/acceptance_summary.json)。
- **11 月 17 日，V5/开发数据与仓库重构。** V5 代理门通过，独立教师总门未通过；开发集可用于软件开发，未建立封存测试结论。随后仓库将高保真求解和旧研究池移出活动树，当前结构正式形成。

### 4.2 11 月 20 日：球面采样、连续方向与 Idea2 全链实施

- **球面入口与连续方向采样。** SamplingSpec 扩展到 1.5/1.6，完成 30 个几何对和 54 个教师对的 6D→5D 等价检查，共 84/84 通过；scrambled Sobol 设计 576/576 工况满足 S-CONTACT。增加 Operational/Reference/Coverage 测度、权重、聚类 bootstrap 和 16 维首层装甲状态输入。证据：[`sampling_sphere_v1/IMPLEMENTATION_LOG.md`](../sampling_sphere_v1/IMPLEMENTATION_LOG.md)。
- **Idea2 机制与数据链。** 建立 Taylor–Shapiro v2、局部删失、dataset-v2、hurdle 目标、暴露场、敏感性和离线树候选。Phase 0 中 `x_design` 宏 PR-AUC 为 0.740，高于原始姿态的 0.427；最终 v3 360 工况包仅 2/300 个开发工况 ε-complete，52/67 部件为零正支持，因此定位为流程与诊断数据。封存 test 结果零解码；1,280 项测试通过。证据：[`idea2_full_implementation_20261120/FINAL_REPORT.md`](../idea2_full_implementation_20261120/FINAL_REPORT.md)。

### 4.3 11 月 22 日：数据恢复与逐环节预期—实际比较

- **监督可用性恢复。** 新数据包包含 2,000 train、300 validation、300 sealed test；train 使用 R8，validation/test 使用 R32。三概率头观察率为 99.52%–99.69%，ε-complete 率为 92.6522%；概率头准入 64/67，条件头准入 62/67。NumPy `p_hit` 在 validation 上的 Brier 为 0.037205，train-only 气候基线为 0.064107，Brier skill 为 0.419645。封存 test 结果零解码。证据：[`FINAL_DATASET_READINESS_REPORT.md`](../dataset_recovery_20261122/FINAL_DATASET_READINESS_REPORT.md)。
- **逐环节量化。** 分析覆盖 4,301 个候选、2,600 个正式工况和 25,600 个教师重复。候选在 profile、region 及交叉层面的 TV 分别为 0.001240、0.001302、0.003984；S-CONTACT 条件化后 region TV 增至 0.201538，第一处主要分布变化定位在接触条件化。全部 2,600 个首次接触通过 7 项几何检查；10,291,200 个状态边界检查和 179,200 个离线树比对均为零违例。外部真实分布没有可用参照，因此外部差距未估计。证据：[`pipeline_expected_actual_20261122/REPORT.md`](../pipeline_expected_actual_20261122/REPORT.md)。

### 4.4 11 月 23 日：独立审查、上游修复、代理收敛和各向异性快评

- **Taylor–Shapiro 方向符号修复。** v3 的 small/medium/large 方向合同符合率均为 0%，v4 均为 100%；legacy v3 的 3 个归档工况可逐位复现。
- **瞄准切向基修复。** 同预算 2,600 个正式工况的 S-CONTACT 总接受率由 60.5% 提升到 98.4%，region TV 由 0.2015 降到 0.0127；四个水平区域接受率均达到 100%。修复使用新配置和新 split 身份，旧配置行为保持可复现。完整回归记录为 1,338 项通过。证据：[`data_generation_fixes_20261123/README.md`](../data_generation_fixes_20261123/README.md)。
- **快速代理收敛。** 4 条方位射线/环扩展为每枚破片一条射线后，`p_hit` 均值比由 0.504 提升到 0.978，Brier 由 0.0426 降到 0.0075。预声明的 train R8 门得到 false-unreached 1.649%，Wilson 95% 上界 1.724%≤3%。事后推荐的 K=8、registry 累积和 branch-probability 删失配置得到 `p_hit` Brier 0.0023、`p_failed` Brier 0.0005、false-unreached 0.40%，约 0.79 s/工况；构造函数默认值尚未翻转，三个下游调用点仍使用旧 `proxy_domain_status` 口径。初始抽查在默认 Python 3.13.5/NumPy 2.1.3 环境下仅 2/12 逐重复复现；随后在归档 Python 3.12.13/NumPy 2.4.6 环境下达到 12/12，根因定位为环境浮点差异触发临界随机分支，而非当前代码改变。证据：[`proxy_convergence_20261123/REPORT.md`](../proxy_convergence_20261123/REPORT.md) 与 [`data_generation_fixes_20261123/README.md`](../data_generation_fixes_20261123/README.md)。
- **各向异性三级快评。** L1 测得声明→S-CONTACT 单元质量 TV 0.200781、最大 KS 0.146919；L2 完整域 146/256=0.570313，66/67 部件可达；L3 对既有教师的 RMSE 0.206398、Spearman 0.591166、false-unreached 0.078985。历史 4,096/100,000 对照的单元质量 TV 误差为 0.000598，暴露 Pearson/Spearman 为 0.999492/0.989776，支持其作为全局快速诊断。当前 256 例 L1+L2 用时 38.104 s。归档环境的 12/12 教师复现解决了归档可复现性问题；`current256` 包内代理与教师代码哈希仍不相同，因此 L3 仍定位为跨版本残差诊断。证据：[`anisotropic_fast_assessment_20261123_current256/FINAL_REPORT.md`](../anisotropic_fast_assessment_20261123_current256/FINAL_REPORT.md)。

## 5. 当前未提交代码改动

### 5.1 tracked 差异

| 类别 | 文件数 | 新增行 | 删除行 | 主要内容 |
|---|---:|---:|---:|---|
| 核心代码 | 17 | 3,656 | 334 | 采样、工况生成、破片方向、侵彻、教师轨迹、概率目标和快速代理 |
| Schema 镜像 | 9 | 280 | 34 | 采样版本、设计序列、重复轨迹、Taylor–Shapiro 和学习输入合同 |
| 测试 | 9 | 1,198 | 1 | 连续接触、教师轨迹、代理、删失、统计和训练就绪审计 |
| 文档 | 3 | 232 | 9 | 决策记录、索引和项目说明 |
| 工具 | 2 | 114 | 7 | 数据集训练就绪审计与工具说明 |
| 依赖 | 3 | 15 | 11 | SciPy 采样可选依赖及锁定环境 |
| **合计** | **43** | **5,495** | **396** | 当前尚未提交 |

主要实现变化：

1. **采样与工况生成**：SamplingSpec 1.5/1.6/1.7、球面入口、5D/7D 连续方向、outcome-blind 设计序列、固定参考轴切向基、region 权重和实现哈希。
2. **破片与侵彻**：独立 `taylor_shapiro_ring`、版本化弹药几何、legacy/wave-directed 两套方向约定、高斜角单侧非贯穿界和 V50 分支概率。
3. **教师输出与统计**：逐重复部件—机理—状态轨迹、逐重复毁伤树评估、`p_hit` 主目标、四种重复存储模式及逐部件删失最坏界。
4. **快速代理**：全破片射线、K 次散布实现、registry 状态累积、branch-probability 删失、OOD 质量和部件状态审计。

逐文件行数见 [`tracked_worktree_diff.csv`](tracked_worktree_diff.csv)。

### 5.2 未跟踪研究功能文件

| 主题 | 核心模块 | 配置 | 工具 | 测试 | 文档/示例 | 合计 | 静态测试函数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reference/球面入口/连续方向采样 | 8 | 10 | 3 | 13 | 0 | 34 | 104 |
| dataset-v2、hurdle 目标与学习准入 | 8 | 2 | 13 | 16 | 2 | 41 | 68 |
| 暴露场、多保真与各向异性快评 | 5 | 4 | 6 | 5 | 1 | 21 | 18 |
| Idea2 机理、敏感性与可信度 | 5 | 9 | 7 | 9 | 3 | 33 | 37 |
| 离线毁伤树候选 | 2 | 0 | 1 | 2 | 1 | 6 | 10 |
| **合计** | **28** | **25** | **30** | **45** | **7** | **135** | **237** |

这 135 个文件在审计开始时全部为 `??`。它们已经形成“配置—核心模块—CLI—测试—文档”链条，但尚未进入正式版本历史。本次整理新增的 `tools/build_workspace_change_inventory.py` 单独计为审计工具。

## 6. 数据集、验证与探索工件

当前变化的主体是本地数据和验证证据。按路径规则汇总，主要占用如下：

| 主题 | 文件数 | 约 MiB | 说明 |
|---|---:|---:|---|
| dataset recovery | 570 | 142.73 | 2,000/300/300 正式包、shards、学习基线与准入结果 |
| dataset-v2 与学习 | 74 | 87.67 | 早期 v1/v2/v3 数据与学习输出 |
| 各向异性快评 | 94 | 68.79 | 多次迭代、历史 4,096 缓存及 current256 最终包 |
| 开发数据及其他核心工件 | 52 | 67.24 | 11-17 重构保留的开发数据等 |
| 快速代理收敛 | 106 | 43.13 | 29 组配置、2,300 工况输出、门和复算脚本 |
| 探索实验 | 32 | 26.87 | C3、E1、E4/E6、F1、X1/X2 等探索运行 |
| Idea2 机理与可信度 | 418 | 23.95 | 机制变体、敏感性、诊断和报告 |

迭代关系：

- `anisotropic_fast_assessment_20261123_current256` 是当前汇报入口；`_v2/_v3/_live256` 等目录保留为中间迭代与历史近似性证据。
- `idea2_taylor_shapiro_continuous_r32_20261120_v3` 是 11 月 20 日严格版本；11 月 22 日 recovery 数据替代其训练用途。
- 11 月 22 日 recovery 正式包使用旧 v3 方向和 SamplingSpec v1；11 月 23 日修复后，它保留为历史链路证据，当前链路的正式数据需重建。
- `sampling_sphere_v1/no_bottom_impact_state_pilot` 是 360 工况工程 pilot；其身份与 11 月 22 日正式角色数据不同。
- `artifacts/experiments/` 是探索与机理诊断证据，不承担最终准入结论。

## 7. 证据层级与当前判决

| 层级 | 已有证据 | 当前判决 |
|---|---|---|
| 软件实现与合同 | Schema、哈希、legacy 复现、几何/教师配对、封存 test 零解码、1,338 项完整回归记录及代理/快评定向测试 | 工程链完整；发布前应在冻结当前组合树后再运行一次全量测试 |
| 统计与数据可用性 | 2,600 工况 recovery 包、逐任务准入、validation 学习基线、逐环节 TV/KS/Brier/FU 分解 | 旧 v3/v1 链路的内部证据充分；新 v4/v2 链路待重建并复算 |
| 快速筛选 | 代理收敛、4,096/100,000 各向异性近似对照、L1/L2/L3 分层接口 | 可用于方案筛选、分布诊断和固定工况残差定位 |
| 当前引擎归因 | L3 包内记录的代理/教师代码哈希不一致；归档环境重跑教师已达到 12/12 逐重复一致 | 正式归因仍应生成同源码树哈希、同环境身份的固定工况教师证据 |
| 外部物理有效性 | 目标参数仍含 `unverified`，没有实验/现场发生率参照 | 尚未完成外部物理标定 |
| 科研准入 | sealed test 未开启，当前正式数据需要按新生成链重建 | 当前状态适合开发与诊断，正式研究结论待新数据和外部参照 |

## 8. 当前最重要的技术判断

1. **主要生成器偏差已定位到接触条件化实现。** 候选设计本身的 TV 约为 0.001–0.004；旧切向基在 S-CONTACT 后把 region TV 放大到约 0.20。新固定参考轴基将其压到 0.0127。
2. **Taylor–Shapiro v3 的符号错误会改变部件命中格局。** v4 采用朝爆轰传播方向的偏转；基于 v3 的部件级分布应重新生成后再引用。
3. **旧快速代理的主要差距来自角向离散、单次散布噪声和累积/删失语义。** 全破片射线、K 次散布、registry 累积和 branch-probability 删失共同把代理与软件教师的 Brier 和 false-unreached 明显压低。
4. **各向异性三级快评已经实现生成器—代理—教师分层。** L1 完全不运行引擎；L2 用确定性暴露核；L3 只读取已有非测试教师证据。它能独立定位分布迁移和代理残差，外部物理参照仍是第四层证据。
5. **当前最大工程风险是身份未冻结。** 43 个 tracked 文件和 135 个研究功能文件尚未提交。归档环境下教师 12/12 逐重复复现，默认环境下仅 2/12，说明解释器、NumPy/BLAS 与源码、配置和 split 一样需要冻结。

## 9. 建议的下一轮工作

按以下顺序推进：

1. 冻结当前源码、配置、Schema、环境和 split registry；更新 `docs/INDEX.md` 与验证根 manifest。
2. 使用 v4 Taylor–Shapiro 物理和 SamplingSpec v2 重新生成 2,000/300/300 数据，维持 train R8、validation/test R32 和 sealed-test 零解码协议。
3. 在新数据上重跑 `pipeline_expected_actual`，复核 region/profile 分布、局部删失、部件支持和树输出。运行 v4 审计前，先修改冻结脚本 `run_stratified_micro_audit.py`，使 `_taylor_shapiro_nominal_directions` 按 `layout.angle_convention` 分派，避免把 v4 正确方向误报为合同违例。
4. 重跑代理收敛与晋升门；用完全匹配当前代码哈希的教师工况重跑各向异性 L3。
5. 固定模型与门后按单次协议开启 sealed test；随后引入外部实验或现场参照完成物理校准。

## 10. 同行汇报用一句话

项目已经从高保真工作区转成可审计的数据生成与验证平台，完成了采样—接触—破片—侵彻—教师—标签—学习的逐环节量化，并通过独立审查修复了两处会扭曲正式分布的上游缺陷；当前决定性工作是按 v4 物理与 SamplingSpec v2 重建正式数据，再用同一源码身份复核代理、教师和各向异性三级快评。

