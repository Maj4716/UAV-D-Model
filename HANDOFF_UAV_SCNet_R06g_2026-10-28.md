# UAV SCNet / LS-DYNA 高保真标定交接（R06g）

日期：2026-10-28（Asia/Shanghai）  
稳定位置：`D:\Desktop\UAV新\HANDOFF_UAV_SCNet_R06g_2026-10-28.md`  
Git：`main`，HEAD `7fde837`（本地领先 `origin/main` 2 个提交，未推送）

## 1. 项目最终目标与科研边界

最终目标是使用通过数值资格审查、物理解释审查并带有不确定性记录的高保真 LS-DYNA 结果，标定：

1. 仿真引擎的源项、接触/侵彻响应与部件毁伤映射；
2. 毁伤树的阈值、条件关系和有序标签；
3. 数据集生成器的条件分布、覆盖策略和可用性掩码。

必须持续区分四层结论：软件一致性、统计稳定性、物理有效性、研究准入。测试通过或 LS-DYNA 正常终止都不等于物理标定完成，也不自动允许结果进入正式训练集。`unavailable` 必须保存为 `null + reason`，不得改写为 0；不得为了标签均衡而修改毁伤树语义。参考分布样本与定向/加密诊断样本必须分账。

## 2. 当前科学状态

- 可靠基线：Job `72617857`，`800 m/s`、`0.75 mm`、`TSSFAC=0.8`。
- 最新回传：Job `72630855`，同工况 `TSSFAC=0.4`；求解器正常终止、资源为 1 节点/32 ranks、输出可读。
- 但冻结的时间步敏感性门失败：`qM/qP/qE` 从 `0.8731778/0.5613075/0.3608753` 变为 `0/0/0`，`qV` 为 `null`。循环数 `2863 -> 7066`，负体积单元 `66 -> 10`，末态存活破片单元 `612 -> 656`。
- `TSSFAC=0.4` 被归类为 `transition/indeterminate`：上游删除质量分数 `0.0437318`，高于严格非穿透门 `0.005`。
- 结论：P0 尚未满足时间步稳健性；`2000 m/s` 及所有 treatment 算例继续冻结，不能进入正式高保真标定或数据集准入。

完整证据：

- `D:\Desktop\UAV新\60_SCNet_P0_R06f_TSSFAC04_Job72630855回传审查与R06g_TSSFAC02提交交付_2026-10-28.md`
- `D:\Desktop\UAV新\scnet_lsdyna_plate_calibration_20261017\analysis\72630855_r06f_0800_tssfac04_20261028\r06f_tssfac04_return_audit.json`
- 基线审计：`D:\Desktop\UAV新\scnet_lsdyna_plate_calibration_20261017\analysis\72617857_r06f_0800_ctrl_20261028\r06f_return_audit.json`

## 3. 下一步唯一允许的动作

只提交 R06g 单一诊断作业：`800 m/s`、`0.75 mm`、`TSSFAC=0.2`、`DT2MS=0`。

唯一提交 ZIP：

`D:\Desktop\UAV新\scnet_lsdyna_plate_calibration_20261017\artifacts\r06g_tssfac02_submission_20261028_v2\uav_hf_r06g_0800_tssfac02.zip`

- ZIP SHA-256：`0d7af79e4e9399edecb8d87a9ebd7ef0cd0090db5a61a6eab1aba24cee1ef8f3`
- 头文件：`r06g_0800_tssfac02.k`
- deck SHA-256：`b5f4da367e9ef6eaae7098e47a84610eb55c51760cc9bf8dbb4721693b7a732a`
- 提交说明：同目录 `SUBMIT_ONLY_THIS_JOB.md`
- 冻结设计/合同：
  - `D:\Desktop\UAV新\scnet_lsdyna_plate_calibration_20261017\design\p0_r06g_timestep_refinement_design.json`
  - `D:\Desktop\UAV新\scnet_lsdyna_plate_calibration_20261017\design\p0_r06g_timestep_output_contract.json`

不要提交同名非 `_v2` 目录或其他历史包。不要同时提交 `2000 m/s`、侵彻/材料 treatment 或参数扫描。

## 4. R06g 回传后的验收顺序

1. 将原始回传目录视为只读证据；记录作业 ID、文件树、时间戳和哈希。
2. 核验 deck/合同身份、1 节点/32 ranks、正常终止、输出族完整性和二进制/ASCII 可读性。
3. 运行单作业适配器、独立测量与 ASCII 数值 QA；不得只依赖一个后处理路径。
4. 以 Job `72630855`（`TSSFAC=0.4`）为直接比较基线，执行预注册门：
   - `qM/qP/qE` 绝对差各 `<= 0.01`；
   - 上游删除质量分数绝对差 `<= 0.01`；
   - 若两次 `qV` 都可用，相对差 `<= 0.05`；只有两次都确认穿越质量为 0 时才允许两者均为 `null`，其余情况失败；
   - 同时检查能量平衡、HVAR6、NCFORC、负体积事件及删除/侵蚀账本。
5. 即使 `0.4 -> 0.2` 通过，也只形成“可决定是否追加 `TSSFAC=0.1`”的证据；不得自动解冻 `2000 m/s`。若不通过，继续冻结并调查接触、侵蚀、网格与时间积分稳健性。

## 5. 高保真结果如何用于引擎、毁伤树与数据集

只将通过上述门的工况写入高保真证据账本，并保存输入身份、QoI、availability/reason、数值质量、测量不确定性和适用域。随后：

1. **引擎标定**：用控制工况先约束源项/接触/响应参数及其不确定性；定向 treatment 只做单因素辨识，不与控制样本混算。
2. **毁伤树标定**：用高保真响应与部件状态估计阈值/条件关系，保留区间和冲突证据；不得根据期望标签比例倒推阈值。
3. **生成器标定**：参考分布用于总体训练/评估，定向加密用于覆盖稀疏边界；两者保留来源标志与独立权重。结构性零不重采样，测不到的量使用掩码。
4. **准入**：冻结标签语义、P0/P1 版本和数据划分；只有软件、统计、数值、物理解释四层证据齐全后，才提升 `research_admission`。

## 6. 验证、清理与 Git 状态

- 本轮完整可运行测试：`129 passed, 0 failed`。另有 1 个历史测试因外部文件 `D:\Downloads\uav_hf_r06d_0800_ctrl.zip` 不存在而排除，不应误记为业务回归。
- R06g 定向测试：`5/5 passed`；这只证明静态/软件交付一致性，不证明目标求解器、物理有效性或数据准入。
- 已安全移除缓存、`__pycache__`、临时目录及被 `_v2` 取代的 R06g 非最终目录，共 `14,611` 个文件、`155,970,181` bytes（约 `148.74 MiB`）。
- 大型原始证据 `uav_hf_r06f_0800_tssfac04`、项目 `05`、`results` 与分析证据被有意保留，并由 `.gitignore` 排除，不应在未归档前删除。
- `D:\Desktop\UAV新\.pytest_cache` 是 0 字节的 ACL 异常空目录，当前账户无法枚举/删除；它不影响 Git 或测试，可后续由管理员处理。
- Git 检查点：`7fde837 checkpoint: preserve high-fidelity calibration and R06g handoff`；未推送。
- `.git` 已从约 `7.59 GiB` 清理至 `37.44 MiB`；`git fsck --full` 无输出/无异常，检查点创建后的工作树曾验证为干净，`main...origin/main [ahead 2]`。

## 7. 建议新对话使用的技能

- `ai-research`：继续预注册诊断假设、单因素设计、泄漏隔离与证据分层。
- `statistical-analysis`：标定参数、重复/不确定性、敏感性和准入判据。
- `scientific-visualization`：绘制时间步收敛、QoI/能量/侵蚀账本与不确定性图，不隐藏 `null` 或失败工况。

交接时不重复平台账号、令牌或账号相关路径；需要提交界面信息时只参考最终包内的提交说明。下一位执行者应从“提交唯一 R06g ZIP，等待并审计其回传”开始，而不是重新生成包或并行推进其他工况。
