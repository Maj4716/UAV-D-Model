# SCNet R04 阶段 3–5 逐工况响应、网格判定与准入结论

- 作业：`72095164`
- 总结论：`completed_reject_for_response_calibration`
- P0 响应标定准入：`denied`
- P1 提交准入：`denied`

## 阶段 3：数据库与逐工况提取

`binout0000` 与完整 `d3plot*` 文件族均由项目局部 LASSO 2.0.4 成功打开。BINOUT 实际目录为 `JOBINFO/glstat/matsum/nodout/sleout`；d3plot 识别 184,949 节点、163,730 实体单元和 14 个 part。

| 工况 | 入速 m/s | 网格 mm | 存活 part 平均残速 m/s | 中心节点残速 | 破片存活质量 | 板删除参考体积 | 穿孔代理 |
|---|---:|---:|---:|---:|---:|---:|---|
| `p0_v1200_h0p75` | 1200 | 0.75 | 526.0 | 733.7 | 73.0% | 0.087% | True |
| `p0_v1600_h0p75` | 1600 | 0.75 | 855.2 | 不可用 | 45.0% | 0.180% | True |
| `p0_v2000_h0p75` | 2000 | 0.75 | 1300.2 | 不可用 | 38.0% | 0.271% | True |
| `p0_v2300_h0p75` | 2300 | 0.75 | 758.5 | 不可用 | 19.4% | 0.722% | True |
| `p0_v2600_h0p75` | 2600 | 0.75 | 1206.6 | 不可用 | 16.6% | 0.634% | True |
| `p0_v2000_h1p00` | 2000 | 1.00 | 322.6 | 不可用 | 46.4% | 0.686% | True |
| `p0_v2000_h0p50` | 2000 | 0.50 | 1854.4 | 1876.7 | 88.0% | 0.039% | True |

中心历史节点只有在末态仍有存活实体单元支撑时才作为可用 QoI。其余工况虽然 NODOUT 仍保留数值，但节点支撑已全部删除，记录可能是冻结值，已按 unavailable 处理。

穿孔代理定义为：仍存活的破片实体质心越过初始板背面且存活 part 的平均 z 速度继续向后。它是数值几何代理，不是试验穿孔判据。

## 阶段 4：速度与网格行为

- 速度扫描穿孔转变是否被括住：`False`；五个中网格点均为穿孔代理。
- 存活 part 平均残速是否随入速单调：`False`。
- 三网格存活 part 平均残速：`[322.608406164252, 1300.1533018170833, 1854.4214898552082]` m/s（1.00/0.75/0.50 mm）。
- 粗→中相对差：`75.19%`；中→细相对差：`29.89%`。
- 三网格破片存活质量率：`[46.4, 38.05, 87.95]`%；跨网格极差 `49.90` 个百分点。
- 中心节点三网格 QoI 可用性：`[False, False, True]`。
- 设计文档要求“进入预注册误差带”，但设计 JSON 与报告中没有发现数值误差带；不能事后选择宽松阈值宣告通过。

## 阶段 5：门控结论

| Gate | 状态 |
|---|---|
| stage0_2_archive_input_solver_global_energy | `pass` |
| binary_database_readability | `pass` |
| database_identity_and_dynamic_case_isolation | `pass` |
| case_level_qoi_completeness | `partial` |
| mesh_perforation_proxy_consistency | `pass` |
| mesh_residual_response_consistency | `fail` |
| speed_scan_transition_bracketing | `fail` |
| p0_response_calibration_admission | `denied` |
| p1_submission_admission | `denied` |

P0 的文件身份、求解完成和全局能量门继续成立；但响应层资格失败：速度扫描没有未穿孔点，三网格残速与存活质量高度敏感，中心历史节点大多因支撑删除而不可用。因此本次 R04 只能作为读取链、侵蚀和网格问题的数值诊断证据，不能进入装甲响应拟合、V50 标定、KMFC 阈值修改或科研数据生成。

## 下一轮最小修复设计

1. 在提交前明确残余速度主 QoI（推荐固定捕获面上的质量—速度联合分布及守恒闭合），不要依赖可能被删除的单个中心节点。
2. 扩展更低速度直到至少得到一个可靠未穿孔点；冻结后再运行，不依据结果临时挑点。
3. 对 2000 m/s 网格组统一输出存活/侵蚀质量、动量和捕获面数据，并预注册数值容差。
4. 先用最小三网格复算闭合响应；通过前继续禁止 P1、材料反调、训练数据再生成和 sealed Test 使用。

## 结论边界

Stages 3-5 establish binary readability and diagnose case-level numerical response. The rejection is a P0 numerical-response qualification decision, not proof that LS-DYNA, the platform, or the reference material is universally invalid. No experimental physical validation or real-world effectiveness claim is established.
