# SCNet / LS-DYNA 标定执行记录与仿真引擎、数据集修改建议

**日期：** 2026-10-17  
**项目：** `D:\Desktop\UAV新`  
**文档性质：** 平台实查、已有作业核验、本地数据复审与下一步标定方案  
**证据清单：** `review_evidence/scnet_lsdyna_20261017/evidence_manifest.json`  
**结论状态：** `calibration_not_completed`；`physical_validity=not_established`

---

## 1. 执行结论

本次已经在登录状态下核验 SCNet 的 LS-DYNA 能力、当前作业、已完成作业、求解脚本与输出目录，并在本地 `pytorch312-lmdt` 环境重新审计当前 5000 例正式归档。结论如下。

1. **SCNet 具备运行本项目所需 LS-DYNA MPP 作业的条件，而且账号中已经存在一条受哈希和授权门保护的高保真计算链。** 平台模板可接收 `.key/.k/.inc/.dyn`；现有作业实际使用 LS-DYNA R14.0.0 双精度 MPP、Intel MPI 2018 和 240 ranks。[证据：SCNET-LIVE-001、002、004]
2. **已有一项 240 核作业正常结束。** 日志记录 314,461 个循环、22,167 s 求解时间和正常终止，折算约 1,477.8 core-hours；目录中存在 `glstat` 与 `d3hsp`。[证据：SCNET-LIVE-003]
3. **另有一项 240 核、4 节点续算正在进行。** 观察时已推进到约 7.25 ms，时间步约 `4.78e-8 s`，日志出现节点删除。运行中状态不能作为数值质量通过的证据。[证据：SCNET-LIVE-002]
4. **上述 SCNet 作业还不能直接标定本地引擎。** 当前没有取得输入 deck 与本地目标的几何/材料/坐标映射，也没有取得按 67 个组件和 2 个装甲对象汇总的响应量。一次“正常终止”只说明求解器完成了输入问题，并不证明模型与本项目相符，更不证明物理正确。
5. **没有新提交收费作业。** 账号中已有多项长时间作业；重复提交一个未完成输入绑定和输出定义的作业会产生大量无效 core-hours。下载 `glstat` 超时后，SCNet 页面访问授权被终止，本次没有绕过权限继续访问。[证据：SCNET-LIVE-005]
6. **本地复审仍拒绝科研训练准入。** 当前归档为 5000/5000 成功、0 拒绝，碰撞合同一致；但 K1、K2、C1、C2 恒零，F1 仅 28/5000 非零，所有工况均为 `R=2`，源项、装甲和部件响应仍未验证。[证据：LOCAL-001—004]

因此，本轮可形成可靠的 **平台执行与证据缺口判定**，但不能声称已完成参数校准、KMFC 标定或物理验证。下一步应先把已有 LS-DYNA 结果转成可与本地引擎逐项对齐的连续响应数据，再决定是否新增计算。

---

## 2. 已核验的 SCNet 作业事实

### 2.1 平台与求解器

SCNet 的 ANSYS LS-DYNA 模板显示 `V231,user`，而账号中现有项目作业使用独立部署的 LS-DYNA `R14.0.0` 双精度 MPP 可执行文件。两者必须视为不同求解器 cohort；不得仅因平台界面都显示 ANSYS/LS-DYNA 就混合结果。

现有 `run.slurm` 在求解前校验：

- 提交授权变量；
- source manifest 绝对路径和 manifest SHA-256；
- `restart.k` SHA-256；
- source gate 脚本 SHA-256；
- LS-DYNA 可执行文件 SHA-256；
- 实际 MPI ranks 必须为 240。

这套门控适合沿用到本项目，尤其应把 solver build、输入 deck、几何、材料、映射与输出提取脚本都绑定到 manifest。

### 2.2 已完成作业

| 项目 | 已观察值 | 可支持的结论 |
|---|---:|---|
| CPU | 240 核 | 资源规模事实 |
| 求解时间 | 22,167 s | 约 6 h 9 min |
| 计算循环 | 314,461 | 日志事实 |
| 终止 | Normal termination | 求解器正常结束 |
| `glstat` | 31.88 MB | 存在全局统计文件 |
| `d3hsp` 分段 | 178.31 MB | 存在详细求解日志 |
| 估算消耗 | 1,477.8 core-hours | `240 × 22167 / 3600`，不等同人民币费用 |

上述信息不足以判断能量守恒、小时玻璃能、质量缩放、接触穿透、负体积、删除单元合理性或目标部件损伤。

### 2.3 运行中作业

运行中作业申请 4 节点、240 核、约 851.7 GB 内存、24 h 时限。观察时日志已经写出多个 `d3plot` 时刻，并在约 6.75 ms 后出现节点删除。节点删除可能来自预期失效/侵蚀，也可能掩盖网格畸变和材料失稳；必须结合 `d3hsp`、`messag`、能量曲线、删除单元所属 part 及网格敏感性判断，不能仅凭“继续运行”判为合理。

24 h 时限对应最多 5,760 core-hours。当前条件下，在没有输出合同与 deck 绑定前再提交同规模扫描不具备科研或成本合理性。

---

## 3. SCNet 结果与本地项目尚未闭合的关键差距

### 3.1 缺少输入身份绑定

必须证明 SCNet deck 与下列本地对象之间的对应关系：

| 本地对象 | 当前状态 | 必需的高保真绑定 |
|---|---|---|
| 2 个装甲对象 | `demo_steel`、厚度与边界条件未验证 | LS-DYNA part/section/material、厚度、坐标变换、边界条件 |
| 67 个组件 | 全部使用 `demo_electronics` 与同一响应模型 | part 集合到 `component-001…067` 的多对一映射及缺失说明 |
| 三个弹型 profile | 参数来自未验证代码移植 | deck 中源项、几何、质量与 profile revision 的一一对应 |
| 碎片源 | Taylor ring + Gurney + 固定 drag | LS-DYNA 破片云的质量、速度、方向和时间窗统计 |
| 当前 case | 仅 y/z、正 x 速度变化 | 同坐标系下的预碰撞输入、接触点、姿态、速度与时间零点 |

只要任一绑定缺失，SCNet 数据就只能称为 `unbound_high_fidelity_candidate`。

### 3.2 缺少可拟合的 QoI

`glstat` 和 `d3plot` 是原始求解输出，不是本地引擎可以直接拟合的监督量。每个高保真 case 至少应导出：

- **数值质量：** 正常终止、能量分量、小时玻璃能、质量缩放增量、最小时间步、负体积/接触告警、删除单元数及所属 part；
- **装甲响应：** 是否穿孔、残余速度、孔口/损伤面积、残余厚度、塞块/次生破片质量与速度；
- **破片云：** 在固定捕获面和时间窗上的质量、速度、方向、动能联合分布，而非仅报告均值；
- **组件响应：** 每个映射组件的峰值/积分响应、塑性功、失效单元比例和可用性 mask；
- **系统层监督：** 如有人工或权威 KMFC 评定，应作为单独来源保存；不得把 LS-DYNA 的单元删除直接等同 KMFC 等级。

### 3.3 当前 SCNet 作业的适用性未知

已有作业名称和目录表明它们是分轮次、分阶段的高保真计算，但尚未取得 source manifest 内容。当前不能确认其目标几何、材料、源项与本项目装甲车一致，也不能确认是破片、冲击波还是耦合模型。将其直接用于本项目参数拟合会引入不可追溯的 domain mismatch。

---

## 4. 本地数据复审结果

本轮在 `pytorch312-lmdt` 中重新运行 `tools/audit_loitering_impact_dataset.py`。生成：

- `review_evidence/scnet_lsdyna_20261017/local_dataset_audit.json`
- `review_evidence/scnet_lsdyna_20261017/local_dataset_audit.md`

主要结果：

| 项目 | 结果 |
|---|---:|
| 正式样本 | 5000 |
| Train / Validation / Test | 3000 / 500 / 1500 |
| 三 profile 数量 | 1662 / 1670 / 1668 |
| 接触合同违规 | 0 |
| 每工况 MC 重复 | 固定为 2 |
| `effect-m2` Train 非零 | 1064 / 3000 |
| `effect-m1` Train 非零 | 590 / 3000 |
| `effect-f1` Train 非零 | 11 / 3000 |
| `effect-k1/k2/c1/c2` Train 非零 | 全部 0 |
| research admission | false |

KMFC-v0 组件级支持为：

- K：28/5000；
- M：2205/5000；
- F：28/5000；
- C：0/5000；
- K 与 F 的 28 个正例集合完全相同。

训练就绪审计还因归档缺失逐重复 trace 而 fail closed。它说明当前均值、标准差和 CI 数组不能独立重算到逐重复联合事件，尤其不适合用来验证新有序硬门。

本轮运行 63 项接触、毁伤树、学习输入、系统学习和科学验证测试，全部通过；这证明当前软件合同未在本轮检查中出现回归，但不提高物理证据等级。

---

## 5. 高保真标定的正确分层

### 5.1 第一层：数值验证，不拟合任何物理参数

先对一个固定、非选择性的基准 case 做：

1. 网格加密；
2. 时间步/质量缩放敏感性；
3. 接触算法和侵蚀控制敏感性；
4. 8/16/32/64 ranks 或平台可行的 MPP 缩放；
5. 双精度结果重现性；
6. 关键 QoI 的收敛和数值误差带。

建议预注册数值筛查门。可把能量漂移、小时玻璃能比、附加质量比和关键 QoI 网格变化设为显式阈值；阈值应根据模型类型给出理由。`5%` 或 `5–10%` 可作为工程候选范围，不能写成所有问题通用的 LS-DYNA 标准。

### 5.2 第二层：低层物理代理标定

按机制分别标定，不从最终 KMFC 正例率反推物理参数。

#### A. 破片源

用 LS-DYNA 捕获面的联合分布替换或校准当前 Taylor ring/Gurney 假设：

- 破片质量分布；
- 方向分布和轴向覆盖；
- 初速分布及质量—速度相关；
- 总破片质量与能量闭合；
- 不同 profile 的独立版本。

输出应进入新的 `fragment_list_ref` 或参数化 source artifact，并记录 deck、提取脚本和结果哈希。

#### B. 装甲穿透

当前 `damage_engine/fragments.py::_ballistic_response()` 使用一条未验证的幂律 V50 曲线和 sigmoid 过渡。应使用板级高保真矩阵拟合：

- 系数与厚度、质量、斜入射指数；
- 过渡宽度；
- 残余速度关系；
- 穿孔/未穿孔和孔面积 QoI。

标定域使用无量纲变量更稳健，例如 `v/v50`、相对厚度和入射余弦。拟合结果必须生成新的 response-model ID；不得原地覆盖 `demo_armor_response`。

#### C. 组件响应

当前 67 个组件共用 `demo_component_response`，损伤概率只由残余动能、统一阈值和 vulnerable-area ratio 生成。这是当前结构零与失真的主要风险之一。

建议先按物理族分组，而不是为 67 个组件各拟一套自由参数：

- 动力/传动；
- 行走与履带；
- 光电/电子；
- 武器/供弹；
- 人员等效响应。

每族拟合零锚定、单调的连续响应曲线，并用层级收缩限制小样本过拟合。没有外部部件失效标准时，LS-DYNA 只能提供 `high_fidelity_synthetic_reference`，不能把某一应变或删除比例自动命名为真实失效概率。

#### D. 破片—冲击波耦合

当前正式数据只运行 fragment 机制。`damage_engine/engine.py::_combine_states()` 对 fragment/blast 使用条件独立并报告 Fréchet 界。只有在获得匹配的联合高保真 case 后，才能增加版本化的 `mechanism_coupling_model_id`；在此之前不能用现有爆炸作业替换 fragment-only 标签。

### 5.3 第三层：KMFC 专用树标定

LS-DYNA 部件响应不是 KMFC 真值。KMFC 树应继续采用 35 号总纲的双路线：

- 显式 Hill/power-mean 有序树作为必需基线；
- 固定 codebook、非负、零锚、单调的浅层神经化树作为候选；
- 每类共享一个连续严重度轴，再在逐重复层执行 `level 2 => level 1` 的硬门；
- `tree_design_fit` 拟合，`tree_design_holdout` 选择；
- 当前只允许 M-only 原型，K/F/C 继续标为 `unidentified/unavailable`，直到获得分支覆盖和监督锚点。

若目标是提高训练效率，可以把等级覆盖作为项目专用教师的次级选择目标；不能通过改变低层材料/源项参数去“制造均匀 KMFC 标签”。

---

## 6. 建议的高保真数据交换合同

### 6.1 每个 case 的 manifest

```json
{
  "hf_case_id": "...",
  "role": "hf_numerical_verification | hf_response_fit | hf_response_holdout | hf_tree_fit | hf_tree_holdout | hf_external_validation",
  "solver_build": "...",
  "solver_executable_sha256": "...",
  "precision": "double",
  "mpi_implementation": "...",
  "mpi_ranks": 0,
  "slurm_job_id": "...",
  "input_deck_sha256": "...",
  "geometry_sha256": "...",
  "material_cards_sha256": "...",
  "part_component_map_sha256": "...",
  "unit_system": "explicit",
  "local_case_hash": "...",
  "termination_status": "...",
  "numerical_qa_status": "pass | fail | unavailable",
  "output_artifact_hashes": {}
}
```

### 6.2 建议输出表

| 文件 | 粒度 | 主要字段 |
|---|---|---|
| `hf_case_manifest.json` | case | 身份、版本、哈希、资源、角色、QA |
| `part_component_map.csv` | part | part ID、component ID、映射状态、坐标变换 |
| `numerical_qa.json` | case | 能量、质量、时间步、接触/侵蚀/删除告警 |
| `armor_response.parquet` | case × panel | 穿孔、残余速度、孔面积、残厚、mask |
| `fragment_cloud.parquet` | case × fragment/bin | 质量、速度、方向、时间、捕获面、权重 |
| `component_response.parquet` | case × component | 连续响应、积分响应、失效比例、mask |
| `system_assessment.parquet` | case × assessor | KMFC 等级、依据、盲法状态、置信度 |

所有 unavailable 必须保留 mask 与原因，不能用 0 代替。

---

## 7. 仿真引擎修改建议

### P0：在接受高保真数据前完成

1. **新增高保真 artifact schema 与校验器。** 校验 solver/input/material/geometry/map/output hash，并拒绝跨 cohort 混合。
2. **保留 demo 模型。** 新增独立 ID，如 `lsdyna_plate_response@1`；旧归档仍绑定旧 ID。
3. **逐重复 trace 成为正式树数据的必需字段。** 缺失 trace 的归档只能做描述审计，不能用于新硬门或联合事件重算。
4. **新增 `source_fidelity` 与 `physical_validity`。** 至少区分 `demo_unvalidated`、`high_fidelity_synthetic_reference`、`empirically_calibrated`。
5. **把 SCNet 实际可执行文件 hash 作为主 solver 身份。** 平台 UI 的应用名不能替代求解器 build。

### P1：低层物理标定

1. 将碎片源参数从 registry 常量拆成可版本化 calibration artifact；支持联合 fragment list。
2. 将 V50、残速和孔面积拆为三个可独立校准的响应模块，避免一条曲线承担全部误差。
3. 将组件响应从单一 `demo_electronics` 改为有限的组件族模型；每族保留适用域、参数协方差和 holdout 误差。
4. 在 `engine.py` 中把 fragment/blast dependence 变为显式模型 ID；未标定时保持当前假设与 Fréchet 界，不隐藏不确定性。
5. 任何高保真拟合都必须输出参数 posterior/bootstrap 区间和 OOD 拒绝范围。

### P2：KMFC 树与学习接口

1. 实现 35 号总纲规定的零锚、单调、有序树节点和累计/独占双编码。
2. 高保真组件响应仅进入 teacher/audit schema；默认 learner 输入不得读取 d3plot、接触结果、组件损伤或求解器 QA。
3. 对 M 建立显式树、受约束序数逻辑和浅层单调网络三者的同协议比较；K/F/C 在支持不足时不训练。
4. 校准、树设计、网络训练和最终 test 使用不同 lineage，所有派生 case 随父组切分。

---

## 8. 数据集生成器修改建议

### 8.1 修复当前输入偏置，同时保持“必然接触”规则

当前正式样本虽然 100% 接触，但速度恒为正 x 方向、姿态为单一规范姿态，只有 y/z 起点和速度大小变化。这会把单一来向、有限外表面和固定姿态固化进学生模型。

下一版应直接在合法接触流形上采样：

- 计划接触面或外表面族；
- 面内归一化坐标；
- 朝向目标的入射方向/斜入射角；
- 绕速度轴的 roll；
- 正速度大小；
- profile。

由“接触点 + 入射方向 + 预碰撞距离”反构轨迹，使射线必定与目标相交。正式样本继续拒绝 miss、反向、零速、起点在装甲内和姿态—速度不一致，但不再固定为单一方向和姿态。

学习器只能读取部署时可得的事前量。若使用“计划接触点”，必须与求解后的实际接触点分开命名；后者默认属于 teacher/audit，避免结果泄漏。

### 8.2 新的数据角色

| 数据角色 | 用途 | 是否可调参 |
|---|---|---:|
| `hf_numerical_verification` | 网格/时间步/接触/并行验证 | 否 |
| `hf_response_fit` | 源项、装甲、组件低层拟合 | 是 |
| `hf_response_holdout` | 检查低层代理 | 否 |
| `tree_design_fit` | 拟合专用树 | 是 |
| `tree_design_holdout` | 选择/拒绝树候选 | 否 |
| `reference_train` | 学生模型训练基准 | 是 |
| `contact_enriched_train` | 阈值附近与稀有等级富集 | 是，需 design weight |
| `reference_validation` | 学生模型选择 | 否 |
| `sealed_reference_test` | 一次性确证评价 | 否 |
| `hf_external_validation` | 独立物理/高保真外部验证 | 否 |

### 8.3 高保真样本的选择

高保真计算昂贵，应采用多保真设计：

1. 用当前低保真引擎生成大候选池；
2. 仅在 Train/calibration 池按几何覆盖、低保真不确定性和阈值邻域选择高保真点；
3. 保留一部分随机参考点，避免只计算“最有希望”的工况；
4. 保存候选概率、选择概率、round、proposal 与 design weight；
5. 高保真 holdout 按几何/材料/profile lineage 分组冻结；
6. 不用已观察 Test 或学生网络准确率决定下一个 LS-DYNA case。

---

## 9. 下一轮 SCNet 执行顺序

### Gate 0：先回收已有结果

重新获得 SCNet 页面访问后，优先下载或在平台侧导出：

1. 已完成作业的 source manifest、输入 deck hash、`run.slurm`、`glstat`、`d3hsp`；
2. 运行中作业完成后的相同文件；
3. deck 的 part/material/section 清单；
4. 能够映射到本地对象的部件响应 CSV/Parquet；
5. 后处理脚本及 SHA-256。

若 source manifest 证明 deck 不属于本项目目标，则把它保留为平台/流程验证，不进入标定。

### Gate 1：数值质量

先对已有完成作业做能量、小时玻璃、质量缩放、接触、负体积、侵蚀和网格证据审查。任何 fatal/未解释异常都先修模型，不做参数拟合。

### Gate 2：最小追加计算

只有在已有结果不能覆盖数值敏感性时，才准备一个固定基准的短时缩放/网格作业。提交前应展示：

- deck 与本地 case/target/registry 的 hash 绑定；
- 预计节点、核数、时限和最大 core-hours；
- 预注册 QoI 与停止规则；
- 输出和后处理合同；
- SCNet 费用规则下的预算上限。

实际点击“提交作业”会产生费用，应在该步骤单独确认。

### Gate 3：标定矩阵

先做 M 类可达组件和低层装甲/破片响应；K/F/C 采用覆盖诊断矩阵，不直接拟合完整八输出。标定矩阵应使用无量纲水平和项目批准的适用域，避免从最终标签均衡目标倒推材料或源项参数。

---

## 10. 验收标准

| Gate | 通过条件 | 当前状态 |
|---|---|---|
| SC-0 平台可执行 | LS-DYNA 作业可正常结束，版本/资源可追溯 | **部分通过** |
| SC-1 输入绑定 | deck、几何、材料、part map 与本地 hash 闭合 | **未通过** |
| SC-2 数值质量 | 网格/时间步/接触/能量/侵蚀 QA 通过 | **未通过** |
| SC-3 QoI 回传 | 装甲、破片云、组件连续响应可读并有 mask | **未通过** |
| SC-4 低层 holdout | 源项/装甲/组件代理在独立 holdout 达标 | **未通过** |
| SC-5 KMFC 树 | 有序、单调、零锚，M 原型先通过 | **未通过** |
| SC-6 新数据合同 | S-CONTACT 多来向、逐重复 trace、多保真角色完整 | **未通过** |
| SC-7 学习准入 | 新 sealed test、无泄漏、概率与等级校准通过 | **未通过** |
| SC-8 物理有效性 | 独立实验或权威外部数据验证 | **未建立** |

---

## 11. 最终科学判断

SCNet 上已有的 LS-DYNA 计算链证明了平台执行能力和良好的输入身份控制，也提供了一项正常结束的高成本候选结果。它尚未完成与本地装甲车模型、67 个组件、三 profile 和 KMFC 语义的绑定，因此目前不能据此修改物理参数，更不能把现有标签稀疏归因于“战斗部威力过小”。

对项目最有价值的下一步不是立即增加样本数量，而是把已有高保真结果转换为可审核的连续响应与数值质量证据。完成这一层后，优先标定碎片源、装甲响应和组件族响应；只在独立监督存在时标定 KMFC 树。M 可以作为首个原型，K/F/C 必须先恢复独立分支支持。新生成器应保持全部样本朝向并接触目标，同时在接触流形上增加表面、面内位置、入射方向和 roll 的覆盖，修复当前固定来向偏置。

在上述 Gate 通过之前，本项目可以继续标记为 `software_development` 和 `high_fidelity_candidate_integration`；不得标记为 `research_training`、`physical_calibration_completed` 或 `real_world_effectiveness_validated`。

---

## 12. 复现与来源

### 本地证据

- `review_evidence/scnet_lsdyna_20261017/evidence_manifest.json`
- `review_evidence/scnet_lsdyna_20261017/local_dataset_audit.json`
- `review_evidence/scnet_lsdyna_20261017/local_dataset_audit.md`
- `review_evidence/kmfc_v0_component_support_audit_20261015.json`
- `review_evidence/kmfc_ordinal_feasibility_audit_20261015.json`
- `raw_data/target.rebuilt.fragment_only.v1.json`
- `examples/physics.loitering_impact.json`
- `35_下一轮科研整改总纲_KMFC专用有序毁伤树与条件接触采样合同_2026-10-15.md`

### 本轮命令与结果

- Python：`D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe`
- PyTorch：`2.8.0+cu128`
- 数据审计：成功；`research_admission=false`
- 训练就绪审计：因缺失 replicate trace 按设计 fail closed
- 相关 pytest：63 passed；另有 pytest cache 的 Windows 权限警告，不影响测试结果

### SCNet 官方入口

- LS-DYNA 使用说明：<https://www.scnet.cn/help/docs/mainsite/hpc/software-examples/use/CAE/Ansys-Lsdyna/index.html>
- 计费规则：<https://www.scnet.cn/help/docs/mainsite/hpc/advanced-guide/billing-rule/>

本报告中的 SCNet 数值来自 2026-10-17 登录页面的现场观察；下载未完成，尚未由本地文件 hash 复核。正式科研使用前，应由项目负责人打开原始 deck、manifest 与输出文件逐项确认。
