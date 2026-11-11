# Damage Dataset Engine

> 当前项目状态、SCNet/LS-DYNA 高保真标定进展、证据边界与下一步见
> [PROJECT_STATUS_2026-11-11.md](PROJECT_STATUS_2026-11-11.md)。

面向破片/冲击波毁伤评估与神经网络教师数据生产的概率仿真内核。当前版本为 `0.7.0.dev3`：SamplingSpec 1.3 增加显式 S-CONTACT 条件接触准入、输入阶段拒绝账本和恢复校验；训练接口补齐归档身份/任务轴绑定、mask-safe loss 与极端权重稳定性；独立的 software-only 有序投影提供逐重复硬门和精确计数，但不替换旧毁伤树。仓库内 `examples/physics.demo.json`、`examples/physics.loitering_impact.json` 及 `raw_data/import_config.json` 中的响应参数仍是 `unverified` 演示/代码参数，因此输出不能解释为真实装备毁伤效能。

## 已实现的计算链

- 破片：质量/初速/方向联合记录或参数源采样 → 空气阻力 → BVH 宽相位 → 外装甲逐层侵彻 → 串列内部部件命中与残速续追 → 状态概率。
- 冲击波：比例距离查表 → 入射/反射载荷 → 装甲 P-I 失效 → 开孔面积混合透射 → 部件 P-I 状态概率。
- 耦合：`sequential` 允许破片造成的装甲完整度影响后续冲击波透射；部件层保留条件独立点估计，同时输出 Fréchet 并集上下界，避免在相关证据不足时把独立假设当成事实。
- 系统评价：版本化毁伤树支持 `independent` 和 `fully_dependent` 门，并可把输出节点组织成多个命名评价标准视图。

## 安装与验证

建议先建立隔离环境，再使用已验证的软件基线：

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

`requirements.txt`（并保留同内容的 `requirements-lock.txt`）固化了本仓库软件验证所用版本；PyTorch 是可选依赖，应按实际 CPU/CUDA 环境单独安装。

六份运行时 JSON Schema（含 `label_spec` 与 `sampling_spec`）随 `damage_engine.schemas` 一并打包，并通过 `importlib.resources` 读取；校验不依赖源码树旁的顶层 `schemas` 目录。

## 快速开始

```powershell
python -m damage_engine hash-target --target examples/target.demo.json
python -m damage_engine validate `
  --target examples/target.demo.json `
  --case examples/case.demo.json `
  --physics examples/physics.demo.json
python -m damage_engine simulate `
  --target examples/target.demo.json `
  --case examples/case.demo.json `
  --physics examples/physics.demo.json `
  --output result.demo.json
python -m damage_engine generate `
  --target examples/target.demo.json `
  --cases examples/cases.demo.jsonl `
  --physics examples/physics.demo.json `
  --output-directory dataset_demo `
  --workers 4
```

目标几何可使用显式 `*_m` 或 `*_cm` 字段，但同一目标禁止混用。载入后统一规范化为 SI；`hash-target` 输出规范化后的内容哈希，案例中的 `target_ref.content_hash` 必须引用该哈希。

## 装甲车碰撞起爆与三型巡飞弹

`damage_engine.munition_profiles` 只允许 `small_front`、`medium_front`、`large_front` 三个稳定弹型 ID；不保留参考工程中的中型后端起爆型，也不会把旧数值 ID `2` 静默重解释为大型。弹体/战斗部尺寸、壁厚、装药、Gurney 常数、爆速、破片数量/质量、环数、轴向覆盖、扩散和阻力系数来自历史参考工程 `loitering_munition_damage_twin/simulation/engine.py`，并明确标记为 `unverified` 代码参数而非试验标定值。归档中保留的原始绝对路径仅是冻结 provenance 字符串，运行时不会访问该路径；本轮为保持 case/content identity 未改写它。

`detonation_trigger=first_armor_contact` 使用连续射线求交，沿 `trajectory_origin_m`（定义为 `t=0` 的弹鼻触发点）和 `velocity_target_mps` 取得第一个 `role=armor` 的交点，并强制写入：

- `detonation_position_m = first armor contact point`；
- `detonation_time_s = collision_distance_m / ||velocity_target_mps||`；
- `detonation_delay_s = 0`；
- 命中装甲 ID、表面法向和 `continuous-ray-first-armor-v1` 模型 ID。

零速度、未命中、起点位于装甲内部、弹轴与速度不一致或派生字段被篡改均 fail closed。当前实现是弹鼻点质点模型，弹体长/直径尚未用于扫掠体接触；碰撞面只包括两块导入装甲，轮、履带、天线等还不是外部碰撞面。参考 `FRONT` 的主破片轴为机体系 `-X_B`，代码按此保留，不把“前端起爆”误改成 `+X_B` 定向。

先校验并运行 12 条三弹型烟雾计划：

```powershell
python -m damage_engine sampler validate-plan `
  --target raw_data/target.rebuilt.json `
  --physics examples/physics.loitering_impact.json `
  --base-case raw_data/case.loitering_impact.v1.json `
  --sampling-spec raw_data/sampling_spec.loitering_impact.smoke.v1.json `
  --label-spec raw_data/label_spec.loitering_impact.v1.json

python -m damage_engine sampler pilot `
  --target raw_data/target.rebuilt.json `
  --physics examples/physics.loitering_impact.json `
  --base-case raw_data/case.loitering_impact.v1.json `
  --sampling-spec raw_data/sampling_spec.loitering_impact.smoke.v1.json `
  --label-spec raw_data/label_spec.loitering_impact.v1.json `
  --output-directory raw_data/dataset_loitering_impact_smoke_v1 `
  --workers 1 --shard-size 12 --fail-fast
```

该计划使用通用 SamplingSpec 1.1，并把三型 one-hot 与结构参数写入 `case_x`；它不属于 UAV-only 12 维控制输入合同。烟雾案例暂只启用破片机理，因为现有近场冲击波表是合成占位表且不覆盖贴面起爆的全部比例距离；不得通过无依据外推把它伪装成有效近场模型。参考 `FRONT -> -X_B` 与正向入射组合后，基准正面接触案例为零破片命中；12 条 smoke 中 5 条存在命中、7 条为零命中，说明结果强依赖接触装甲和几何。该归档只验收碰撞解算、弹型切换、张量与归档链路，明确禁止直接作为正式毁伤学习数据使用。

### 本地 5000 条巡飞弹撞击数据集（不随源码提交）

工作目录中可另行保存生成数据集 `raw_data/dataset_5000_loitering_impact_fragment_only_v1`；该约 69 MB 的归档不属于源码提交，clean checkout 不包含它。现有本地副本只含 `small_front`、`medium_front`、`large_front`，强制 `first_armor_contact` 零延迟起爆，并使用 `loitering_impact_state_only@1.0.0` 的 6 维模型输入（三弹型 one-hot + 轨迹 y/z + 撞击速度）。它完成了 5000/5000 案例的软件归档，但因弹体/目标/响应参数未完成独立标定，且四个系统输出在 Train 中恒为零，`research_admission=false`。不得把“归档完成”解释为“科研有效”。下列审计命令需要先恢复或重新生成该本地归档。

```powershell
python tools/audit_loitering_impact_dataset.py `
  --dataset-root raw_data/dataset_5000_loitering_impact_fragment_only_v1/formal_dataset
```

### S-CONTACT 与 KMFC 有序投影原型

SamplingSpec 1.3 只接受 `loitering_impact_state_only@1.0.0`，并把完整的 S-CONTACT 合同哈希、结果盲选择声明及其哈希绑定到采样身份。正式候选必须满足：轨迹起点位于装甲外、速度有限且非零、机体 `+X` 与速度对齐、速度正向射线存在首次装甲交点，且保存的接触点、时间、距离、法向和装甲 ID 能由冻结几何复算。`reject_and_resample` 会把每次输入/几何拒绝写入候选账本，拒绝行保持 `selected=false`、`weight_mask=false`、`formal_count=0`，并明确记录正式教师未执行；checkpoint 恢复时会重新核对完整尝试账本。旧 SamplingSpec 1.0—1.2 的生成路径保持原行为。

S-CONTACT 数据只估计 `criterion_exceedance_probability_given_first_armor_contact`。它不能估计命中率、制导成功率、任务成功率或无条件 KMFC/毁伤概率。结果盲声明只接受合同中逐项列出的前标签 case 输入路径；`/kmfc/...`、`/predictions/...`、result/outcome/label/teacher/内部响应、严重度或稀有度分数等未批准路径一律拒绝。合同哈希同时覆盖该白名单、禁止依赖标记和撞击复算容差。

`damage_engine.ordinal_projection` 是与旧 `damage_tree.py` 分离、需显式调用的 provisional software-only 原型。它固定 KMFC-v0 的 Hill 指数为 1，要求逐输入预声明最大斜率与数值容差，使用零锚定单调变换和正权幂平均，并在每个 Monte Carlo 重复内部执行两级嵌套硬门；聚合保存 `successes/trials/n0/n1/n2/q1/q2` 和 Wilson 边际区间。缺失证据传播为 unavailable，不会补零。当前 K/M/F/C codebook、F2、外部阈值和独立校准尚未批准，因此该模块不能把旧七输出改名为新八输出，也未用于重生成现有 5000 条归档。

### 5000 条数据的任务选择、破片追踪与收敛合同

`0.7.0.dev3` 将归档标签可用性与网络任务可训练性分开。`tools/audit_system_learning_contract.py` 只用 Train 选择任务，先调用生产 loader 的严格 committed-inventory 校验，再绑定 dataset manifest 以及 target、physics、engine、毁伤树、标签、采样和切分合同哈希。loader 向 adapter 提供同一组 dataset ID、manifest hash 和源合同哈希；任一字段缺失、非小写十六进制 SHA-256 或不一致都会拒绝。资格按任务名判定，张量列只按归档 `system_output_names` 定位。对当前冻结归档，默认可训练输出仅为 `effect-m2` 和 `effect-m1`；`effect-k1/k2/c1/c2` 因 Train 恒为零而 fail closed；`effect-f1` 虽有少量非零支持，但 `component-050` 阈值尚未解析，任务定义未完成，不允许训练。工程起点 `Train/Validation/Test=200/50/100` 是待预注册的软事件质量门槛，不是科学定理。

```powershell
python tools/audit_system_learning_contract.py `
  --dataset raw_data/dataset_5000_loitering_impact_fragment_only_v1/formal_dataset `
  --target raw_data/target.rebuilt.fragment_only.v1.json `
  --test-status observed_diagnostic_only `
  --output review_evidence/optimization_20261014/system_learning_contract.json
```

`MultiTaskTargetAdapter` 默认要求绑定该合同，并仅返回已放行的 system head；显式请求恒定、未解析或未注册任务会拒绝。只有不进入训练的软件审计可以显式设置 `allow_unfiltered_system_tasks_for_software_audit=true` 查看全部原始输出，返回合同会保留 `eligibility_enforced=false` 和 override 标记。`masked_soft_bce_with_logits` 和 `masked_brier_from_logits` 使用逐输出 sigmoid 语义，保留 label mask，并将设计权重与任务权重作为独立轴处理。逐任务评价器报告 Brier、log loss、ECE、软二项逻辑再校准的截距/斜率和加权 soft average precision；二值 PR-AUC 与 case-level bootstrap 只有在事件阈值及重采样单位、次数、置信度和随机种子预注册后才允许计算，缺少合同时返回机器可读的“未计算”状态。宏平均只包含调用方显式列出的、已由 Train 合同放行的任务。`training`、`normalizer_fit`、`acquisition_fit`、选模及评价操作现在会自动要求相应的 `research_*` 准入，不能通过关闭可选开关绕过。

破片调试在 `analysis.capture_fragment_audit=true` 时记录每条射线的完整候选路径、已处理路径、阻断原因、入/出交点、装甲等效厚度、V50、侵彻余量与部件响应概率。默认关闭，开启不改变物理抽样结果。以下命令只从 `reference_train` 按弹型、首接触装甲和预定 y 中心分层，将每个部件的原因互斥划分为不相交、上游阻断、命中未穿透、穿透无响应和可能响应：

```powershell
python tools/audit_fragment_reachability.py `
  --target raw_data/target.rebuilt.fragment_only.v1.json `
  --physics examples/physics.loitering_impact.json `
  --sampling-run raw_data/dataset_5000_loitering_impact_fragment_only_v1 `
  --system-output effect-k1 --system-output effect-k2 `
  --system-output effect-f1 --system-output effect-c1 --system-output effect-c2 `
  --trace-output review_evidence/optimization_20261014/fragment_reachability_train_trace.jsonl.gz `
  --output review_evidence/optimization_20261014/fragment_reachability_train_audit.json
```

收敛诊断工具已支持固定嵌套前缀 `R=4/8/16/32/64/128`，默认执行下限为 72 个工况。报告给出相对最大前缀的逐任务绝对误差 P95/P99、排序相关、标签翻转率和阈值邻域翻转率；最大前缀只是数值参考，不是物理真值。CLI 只证明本次调用没有按新仿真结果追加/停止，不能自行证明上游案例是 outcome-blind、已外部预注册或满足交叉分层配额；这些状态会在报告中显式标为 `not_established_by_this_report`。

本地历史归档的 Train-only 就绪审计曾尝试固定 `3 弹型 × 2 首接触装甲 × 低/过渡/高响应 × 每层 4` 的 72 工况设计，但多个过渡/高响应层为零，结果为 `quota_unreachable`。因此没有执行正式 R=128 仿真，也没有生成 `replicate_convergence_r128.json`。该次机器证据属于未随源码提交的本地审查产物；只有新 Train-only 可达性设计补齐全部冻结配额，并单独绑定经审查的选择计划、阈值、误差门槛和停止规则后，才运行该工具。直接对 sampling-run 取前 72 条只能作为数值探索，不能标记为正式分层收敛研究。

现有 Test 结果已被审查读取，只能作诊断和软件回归。最终模型比较仍需要全新、从未参与窗口发现、采样、早停或超参选择的密封 Reference Test。上述合同和审计工具不改写当前 5000 条冻结归档，也不代替弹体/装甲/响应参数标定或外部验证。

## UAV-only 参考分布采样与最小 p0 pilot

`examples/label_spec.demo.json` 把学习任务绑定到当前毁伤树定义哈希，并明确 `mission_degraded` 与 `mission_loss` 是可同时发生的多标签事件概率，不能使用 softmax 强制归一。新的 `examples/sampling_spec.uav_control.demo.json` 使用 SamplingSpec 1.2 和 `model_input_contract=uav_state_only@1.0.0`：只采样无人机相对位置、相对速度与相对姿态，弹体/威胁源、目标、环境、物理注册表和求解设置由配套 `examples/case.uav_control.demo.json` 冻结在场景外侧，不进入控制模型输入。其 2/1/1 条 train/validation/test 与每工况 2 次 replicate 只用于软件管线 smoke test，不具备统计放行能力。

`examples/sampling_spec.demo.json` 是 SamplingSpec 1.0 的 legacy 兼容样例，只用于复现旧合同；新建无人机控制数据集不要以它作为默认模板。

先做无标签计划校验：

```powershell
python -m damage_engine sampler validate-plan `
  --target examples/target.demo.json `
  --physics examples/physics.demo.json `
  --base-case examples/case.uav_control.demo.json `
  --sampling-spec examples/sampling_spec.uav_control.demo.json `
  --label-spec examples/label_spec.demo.json
```

如需先审阅计划而不运行教师标签，可独立冻结候选账本、formal cases 和 split registry：

```powershell
python -m damage_engine sampler init `
  --target examples/target.demo.json `
  --physics examples/physics.demo.json `
  --base-case examples/case.uav_control.demo.json `
  --sampling-spec examples/sampling_spec.uav_control.demo.json `
  --label-spec examples/label_spec.demo.json `
  --output-directory raw_data/dataset_pilot_demo
```

若已执行 `init`，运行 pilot 时必须对同一目录显式加 `--resume`；若输出目录不存在，可直接执行下面命令并去掉 `--resume`：

```powershell
python -m damage_engine sampler pilot `
  --target examples/target.demo.json `
  --physics examples/physics.demo.json `
  --base-case examples/case.uav_control.demo.json `
  --sampling-spec examples/sampling_spec.uav_control.demo.json `
  --label-spec examples/label_spec.demo.json `
  --output-directory raw_data/dataset_pilot_demo `
  --workers 1 `
  --shard-size 2 `
  --resume

python -m damage_engine sampler audit `
  --output-directory raw_data/dataset_pilot_demo
```

`init` 在任何教师输出被观测前冻结 validation/test 与切分注册表，formal seed 与 screening seed 分离。当前 p0 pilot 的入选概率为 1，原始设计权重为 1；后续自适应采样只允许进入 train。`--resume` 只接受目标、物理注册表、基准案例、标签合同、采样合同和种子均与 checkpoint 一致的已提交计划；文件被改动、不完整标记或身份不匹配会 fail closed。SamplingSpec 1.2 还会在教师求解前检查全部正式案例是否共享同一个 `fixed_scenario_hash`，避免弹体/目标/环境在模型看不见时暗中变化。

SamplingSpec 1.1 在旧 1.0 合同之外增加 `feature_roles`、联合/条件变量块、预标签 geometry strata、Reference/geometry-stratified/stress 流、round budget、分层重复策略、关键投影和代理边界。SamplingSpec 1.2 在此基础上要求模型输入严格匹配已声明的版本化合同：`uav_state_only@1.0.0` 覆盖 position3 + velocity3 + attitude rotation-6D6，`loitering_impact_state_only@1.0.0` 覆盖三弹型 one-hot + 轨迹 y/z + 撞击速度。目标、装甲、碰撞求解、物理注册表和教师输出仍只能进入审计字段，不得混入默认学习器。Validation/Test 只能来自 Reference 流；geometry-stratified 只进入 Train；Rare/Stress 独立进入 Test 且 `sampling_weight_mask=false`。无法满足冻结预算或使用标签后分层时直接拒绝。旧 1.0/1.1 文档仍按原路径读取，不会被静默升级为 1.2。

SamplingSpec 1.0 参考档案使用 manifest 1.4；通用 SamplingSpec 1.1 使用 manifest 1.5、tensor 1.6；adaptive 归档继续使用 manifest 1.6、tensor 1.7；带 UAV-only 模型输入合同的 SamplingSpec 1.2 使用 manifest 1.7、tensor 1.8。旧档案不会自动升级。tensor 1.8 明确区分模型输入与教师审计字段：

- `uav_state_x` / `uav_state_feature_mask`：UAV 合同的默认原始输入，10 维 position3 + velocity3 + quaternion4；训练视图把 quaternion4 转为 rotation-6D6，最终 `training_x` 为 12 维；
- `loitering_impact_x` / `loitering_impact_feature_mask`：撞击合同的默认输入，6 维三弹型 one-hot + 轨迹 y/z + 撞击速度；直接生成 6 维 `training_x`；
- `case_physics_x`、`solver_context_x` 及其 mask：仍可归档用于教师复核和历史兼容，但 UAV-only loader 默认不返回，禁止无意并入控制网络；
- `sampling_selection_probability`、`sampling_design_weight_raw`、`sampling_weight_mask` 及 round/role/stratum 索引、词表和 lineage/candidate ID；
- 数据身份中的毁伤树、`label_spec`、`sampling_spec`/p0、分层定义和 split registry 哈希。

正式归档完成后，可从 sampling-run 根目录或直接从 `formal_dataset/` 生成自包含张量语义页面；工具会按 Manifest 自动识别 tensor 1.8 的 `uav_state_x`，不会再假设固定案例数、分片数或字段数：

```powershell
python tools/build_tensor_semantics_dashboard.py `
  --dataset-root raw_data/your_sampling_run `
  --output tensor_semantics.uav_control.html
```

tensor 1.6 另外保存可解析且摘要匹配的 `source_contracts.label_spec_document`、`generation.sampling.split_registry_document`，以及包括拒算记录的完整输入候选身份轴。tensor 1.7 再保存每个 adaptive case 的 `p0/q/rho`、目标层级概率、proposal/model/round/set 身份和一般设计权重。`system_source_available_mask` 保守传播机制与图依赖可用性；`label_task_available_mask` 和 `label_task_names` 记录任务可用性。序数与互斥任务需要完整输出，互斥任务必须显式包含背景并满足概率和为1；独立输出任务仍使用逐输出 mask。输出含义由保存的定义约束。

PyTorch loader 对 tensor 1.8 只暴露当前 manifest 所声明的默认模型输入（`uav_state_x` 或 `loitering_impact_x`）；`case_physics_x`、`solver_context_x`、`component_x`、`armor_x`、`component_armor_path_x`、对应 mask 以及固定目标/装甲/毁伤树静态图继续保存在归档中，但只有显式指定 `fields` / `include_static_graph=True` 的审计兼容路径才能读取，不能静默进入网络。Reference 候选直接来自 `p0` 时，`sampling_design_weight_raw=1/ρ`；`sampling_weight_mask=false` 的 geometry/stress 数值不能用于总体加权诊断。一般 adaptive proposal 在独立的多轮候选账本中分别保存 `p0`、`q`、两种层级概率和 `ρ`，设计权重为 `p0·P0/(q·Q·ρ)`，只有 `q=p0` 且层级概率相同时才化简为 `1/ρ`。子集读取仍先校验整个档案。多目标合并遇到同名但不明或不同的输出/任务定义会拒绝；历史软件取证例外为 `allow_legacy_semantic_combination=True`，会记录覆盖项。

`TorchDamageDataset` 与 `TorchShardedDamageDataset` 可用 `sampling_role`/`stream_role` 形成角色视图，并可绑定 `DatasetUsePolicy`、operation、用途准入和冻结模型 hash。`test` 中同时存在 `reference_test` 与 `rare_stress_test` 时，未显式选择角色会 fail closed。该门禁是 loader 级逻辑隔离；同归档直接文件访问和最终模型冻结后的 Test 延迟执行仍需单独的进程/存储权限方案。

训练视图从 manifest 中的冻结设计文档读取四类特征角色：`learnable_variable`、`fixed_context`、`structural_missing`、`derived_input`。SamplingSpec 1.1 要求 `tensor_feature_names` 精确覆盖最终张量特征；Train-only normalization 将位置拆为方向和距离、把四元数转为对 `q/-q` 不变的 rotation-6D，并保留 mask。设计权重、任务权重和 minibatch 概率是三个独立配置，不能由 `1/rho` 相互推导。

采样 checkpoint 2.1 使用 `initialized` / `archive_complete`：完成要求真实正式档案、提交文件、完整有序输入、候选和实际 registry assignment 一致，并分别记录成功、拒算及用途准入数量。初始化到完成持有同一采样写入锁。2.1 参考候选只保留 Zstd Parquet，不再重复写 candidate JSONL；formal case 使用确定性 `jsonl.gz`，逐案结果使用确定性 `results.jsonl.gz`；`sampling_spec`、`label_spec`、split registry 等重复大对象外置为受哈希约束的 `plan/` sidecar，恢复时校验后再水合。旧 2.0 的未压缩 JSONL 仍可读，但不会写入新归档。`archive_complete` 仅表示归档检查完成；质量与科学准入另外记录，coverage/independence 不作无证据保证。外部输入内容快照及执行环境纳入恢复身份。旧 checkpoint 1.0 不自动迁移。

多轮 adaptive 流程为 `adaptive-init → adaptive-propose → adaptive-run-selected → adaptive-update → adaptive-audit → adaptive-finalize`。proposal input 1.1 必须包含与 candidate ID 一一对齐的完整 candidate cases；提交时生成只读候选、选中正式案例及索引账本。`run-selected` 只能执行内容绑定的 Train 案例，`update` 只能从通过 manifest、checkpoint、分片、张量和文件哈希复核的正式归档派生 outcome，外部 outcome 文件不能替代正式归档。

正式 Parquet 表使用 `table_schema_version=1.0.0` 显式 schema，Reference 候选表使用 `sampling-candidates@2.0.0`，adaptive 候选表使用 `adaptive-sampling-candidates@1.1.0`；JSONL/Parquet 字段和内容必须一致，未知列及未知嵌套字段拒绝。`analysis.capture_replicate_trace=true` 可选保存聚合前的逐重复联合输出及 `parameter_realization_hash`，供重复数收敛研究使用；它不自动建立 CI 覆盖。冻结 split 不证明物理独立或真实任务覆盖。输出仍为 `synthetic_teacher`；演示物理参数、LabelSpec 和 p0 为 provisional/unverified，软件验收不构成科研训练准入。

通用软件验收命令（测试临时目录必须是当前用户可写位置）：

```powershell
$env:TEMP='D:\tmp'; $env:TMP='D:\tmp'
& 'D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe' -m pytest -p no:cacheprovider
```

历史整改曾包含额外的本地大归档 loader smoke、保留哈希和隔离 wheel 检查；这些生成数据与机器摘要未纳入源码提交，不能由 clean checkout 直接复现。当前可复现的验收以完整 pytest、[本轮架构实施日志](37_架构深度优化实施日志_2026-10-16.md)及其中明确列出的证据缺口为准。保留的旧 pilot 不自动迁移；软件验收不证明物理有效性、采样代表性、教师不确定性、学习效益或科研准入。

### 从 raw_data 重建目标

`List.csv` 的重复毁伤列名按列位置解析，`armor.csv` 的两个截面按同名对象重建为拉伸多边形：

```powershell
python -m damage_engine import-raw `
  --components raw_data/List.csv `
  --armor raw_data/armor.csv `
  --config raw_data/import_config.json `
  --output raw_data/target.rebuilt.json `
  --diagnostics raw_data/import_report.json `
  --skip-incomplete-components
```

当前导入结果为 67/68 个部件、2 个装甲体；缺少 XYZ 的“无线电台”被隔离并记录，不会默认放在原点。原始文件没有提供装甲厚度、易损率和材料响应，因此 `import_config.json` 中的 10 mm/0.2 等值只是明示的 `unverified` 占位参数。原始毁伤码阈值的状态映射也是临时 codebook；获得正式准则后应更新配置并重新导入。`--strict-components` 可用于生产放行时对任何不完整几何 fail closed。

## 输入与评价标准

外装甲支持拉伸多边形的顶点、拉伸方向/距离、等效厚度、易损面积比例及冲击波响应；内部部件支持 box、cylinder、extruded polygon、位置、完整 `xyzw` 姿态、尺寸、厚度、易损率及 P-I 阈值。

如需同一毁伤树输出多个评价口径，可在 `damage_tree.standards` 中声明：

```json
{
  "standards": [
    {"id": "mission-effectiveness", "outputs": ["mission_loss", "mission_degraded"]},
    {"id": "mission-loss-only", "outputs": ["mission_loss"]}
  ]
}
```

案例的 `analysis.evaluation_standard_ids` 选择需要返回的视图。未声明 `standards` 时，为兼容旧输入，毁伤树 `version` 自动成为唯一标准 ID，并包含全部 `outputs`。

`warhead.fragment_source.fragment_list_ref` 可引用 UTF-8 CSV/JSON/JSONL 离散联合源项。每条记录需要 `mass_kg`、`direction_body_xyz`、`initial_velocity_mps`，可选 `id` 和 `multiplicity`；CSV 也可将方向写成独立 x/y/z 列。加载器会检查有限值、正质量/速度、单位方向、重复 ID、展开数量和金属质量守恒，并在 provenance 中记录文件 SHA-256。

除概率输出的 `adaptive_ci_half_width` 外，`analysis.adaptive_physical_ci` 可对连续物理量分别设置绝对/相对半宽，且必须与 `replicates >= 32` 同时使用：

```json
{
  "adaptive_physical_ci": {
    "component.blast.incident_peak_pressure_pa": {
      "absolute_half_width": 5000.0,
      "relative_half_width": 0.05,
      "relative_floor": 1000.0
    },
    "component.fragment.deposited_energy_j": {
      "absolute_half_width": 2.0
    }
  }
}
```

停止条件要求所有已申明的概率门和物理量门同时满足；不支持的 selector 或被禁用的机理会被拒绝。

## 主要输出

- 部件级破片、冲击波和综合状态概率；
- 命中数、穿透数、着速、侵彻裕度和能量沉积；
- 入射/有效峰值超压、冲量、正相持续时间、到达时间和遮蔽厚度；
- 装甲破片后/综合完整度、反射载荷及冲击波失效概率；
- 部件两机制条件独立并集点估计及 Fréchet 并集上下界；
- 毁伤树全部节点、输出节点及所请求评价标准视图；
- Monte Carlo 均值、标准差、Student-t/正态 95% 区间和收敛元数据；
- JSONL 审计结果、Parquet 长表、NPZ 张量和带哈希的 manifest。

## 张量接口与版本兼容

`tensors.npz` 的核心字段如下：

- `case_x [B,71]`：交汇/碰撞、三型 one-hot 与弹体结构、战斗部、源项统计、方向、阻力、环境、机理、耦合、保真度与采样控制；旧案例没有弹型元数据时对应结构特征由 mask 标为不可用；
- `component_x [B,C,38]`：形状、位置、完整旋转矩阵、尺寸、厚度、易损率、P-I 阈值和暴露类别；
- `armor_x [B,A,18]`：装甲几何、拉伸、面积/周长/顶点数与响应参数；
- `component_armor_path_x [B,C,A,4]`：遮挡、入射距离、路径长度和入射余弦；
- `component_y [B,C,3,S]`：破片/冲击波/综合 × 部件毁伤状态；
- `component_frechet_y [B,C,S,2]`：两机制并集的 lower/upper 界，仅两机制同时启用时 mask 有效；
- `armor_y [B,A,13]`：装甲完整度、入射/反射峰值与冲量、持续/到达时间、入射余弦及失效概率；
- `component_aux_y [B,C,14]`：可解释的破片与冲击波中间量；
- `system_y [B,O]`：毁伤树输出节点；
- 对应的标准差、95% 区间、feature/label mask、名称、单位、split/group/leakage key、随机种子和实际重复数；
- 拉伸多边形顶点 CSR 数组及数值化毁伤树边。

禁用机理对应标签使用 mask 标记为不可用，不把人工零值当训练标签。分组切分优先级为 `split_group → condition_cluster → physical_case_hash`，防止同一物理工况的 Monte Carlo 重复跨 train/validation/test 泄漏。

PyTorch 可选适配器：

```python
from torch.utils.data import DataLoader
from damage_engine.torch_dataset import (
    TorchDamageDataset,
    TorchMultiTargetDataset,
    TorchShardedDamageDataset,
    collate_damage_batch,
)

dataset = TorchDamageDataset("dataset_demo", split="train")
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

sharded = TorchShardedDamageDataset(
    "dataset_sharded", split="train", max_cached_shards=2
)
multi_target = TorchMultiTargetDataset([dataset, sharded])
multi_loader = DataLoader(
    multi_target,
    batch_size=16,
    collate_fn=multi_target.collate_fn,
    num_workers=4,
)
```

请使用 `multi_target.collate_fn`：它在数据集初始化时冻结全局状态、机制、标准和输出词表，batch 子集、shuffle 和尾 batch 不会改变语义位置。裸 `collate_damage_batch` 需要显式传入词表。对象轴动态 padding 并有 presence mask，图索引保持节点命名空间。两种 loader 默认经同一公共 verifier 验证版本化 manifest Schema、checkpoint/shard inventory、文件哈希、切分审计和张量语义，并以 `copy_on_read=True` 防止直接原地变换污染缓存。`VerifiedArchive` receipt 同时绑定 root manifest、checkpoint、shard manifest 与全部已提交分区；policy 消费 receipt 时会复核 freshness。无 `schema_version` 的历史 manifest 仅保留既有只读语义兼容，当前 writer 不生成该格式。已有 Dataset 实例若与外层 split/fields 等明确要求冲突会报错。兼容/取证绕过必须显式请求，且记录于 `audit_overrides`，不能据此宣布研究准入。

1.4 新增每种标签的独立 uncertainty mask 与 sample count。一次观测仍保留均值；JSON 方差/区间为 null，张量的相应项为不可估计占位并由 mask 排除。标准轴 `E_standard` 与图边轴 `E_edge` 分别声明。标签来源明确为 `synthetic_teacher`；`n_valid` 保留兼容含义“执行成功数”，另列研究用途准入数。`purpose="research_training"` 等用途需要独立、匹配当前对象/定义摘要且未过期的证据；当前演示资料没有此类证据。

### 分片生成与恢复

```powershell
python -m damage_engine generate `
  --target raw_data/target.rebuilt.json `
  --cases raw_data/cases.smoke.jsonl `
  --physics examples/physics.demo.json `
  --output-directory raw_data/dataset_smoke `
  --workers 4 `
  --shard-size 1024

# 中断后必须使用相同顺序的案例、workers/chunksize、失败策略和分片配置
python -m damage_engine generate `
  --target raw_data/target.rebuilt.json `
  --cases raw_data/cases.smoke.jsonl `
  --physics examples/physics.demo.json `
  --output-directory raw_data/dataset_smoke `
  --workers 4 `
  --shard-size 1024 `
  --resume
```

每个分片在临时目录中完成后才改名；checkpoint 2.0 绑定完整有限输入快照、预期数量、环境与执行配置。相对外部文件路径以案例文件所在目录为基准；Python API 需显式 `cases_base_directory`。运行期间使用已冻结内容，外部同路径内容改变、缩短输入或环境改变会拒绝恢复。逻辑输入身份按内容与格式计算，移动相同内容文件不改变该身份，路径另存执行 provenance。完成态不允许追加；1.x checkpoint 不静默迁移。

根 manifest 1.2 列出提交集合中的精确文件，不使用 `part-*` 通配符。孤立分片/临时目录会拒绝发布并保留证据。输出目录在写入前取得排他锁；硬中断可能留下 `.目录名.writer.lock`，必须核实所有者已停止并显式归档锁后才能恢复，程序不自行删除不明锁或目录。原子改名、文件 flush/fsync 只支持已测的进程中断防护，尚未验证存储故障或掉电事务。

Parquet 是按列读取接口，NPZ 是单拓扑张量档案。分片限制主要结果的单片规模，但全局 case ID/hash、split 审计元数据和 checkpoint 仍为 O(N)，累计 checkpoint 重写量还会增长。读取初始化的完整语义校验需要逐片解压，启动 I/O 尚未测量。历史本地人工元数据测量上限为 50,000 条，其机器产物未纳入源码提交，也不能作为完整管线或百万级档案的容量保证。

## 性能复测

```powershell
python -m benchmarks.benchmark_engine --cases 40 --replicates 4 --warmup 2
python -m benchmarks.benchmark_geometry
python tools/benchmark_plan_identity.py --cases 1000
python tools/benchmark_engine_hotpath.py --cases 200 --rounds 3
python -m benchmarks.benchmark_dataset --cases 48 --replicates 2 --parallel-workers 2
```

热点优化包括拉伸多边形预计算/标量 Möller–Trumbore 求交、静态目标预编译、BVH 宽相位、共享 canonical-subtree identity cache 和确定性多进程批处理。基准命令同时检查科学输出 digest、formal identity、serial/parallel tensor 与 dataset identity；它们仍只是当前机器的 software-performance snapshot。正式性能结论应绑定硬件、Python/依赖版本、目标复杂度和案例集，并保存重构前基线；缺少前测时不得宣称“无性能退化”。

## 可信度边界

- `L3` 会强制要求物理注册表为 `validated`、目标参数为 `validated`、来源为 `approved`；演示配置会被拒绝。
- 超出注册模型适用域时结构化拒答，不隐式外推。
- `fully_dependent` 是完全正相关的可审计包络，不是经试验拟合的相关系数。
- 目标级峰值、冲量和持续时间阈值会覆盖装甲 `failed` 状态或部件首个非完好状态的 P-I 尺度；更高毁伤状态仍使用物理注册表中的分级参数。
- 张量接口已提供 `armor_y` 和 Fréchet 界，但它们仍是同一套未标定教师模型的输出，不是实验真值。
- 破片偏转/跳弹/二次破片/孔洞重叠以及冲击波完整时程/多路径/泄压网络未经标定模型实现；不得将经验降阶结果宣称为 CFD/FEM 结果。
- 2026-10-09 在 `pytorch312-lmdt` 验收环境执行真实 PyTorch 人工档案回归；具体结果以本轮整改日志与验收 JSON 为准。小型 spawn/CUDA 张量检查不能代替完整训练或规模吞吐/RSS 验收。
- 软件测试通过只证明实现行为与当前规格一致，不证明演示物理参数或模型已被实物试验确认。

实现背景见 `01_仿真引擎工程实现.md`；基线评估见 `02_仿真引擎机理与性能评估整改报告.docx`；本轮 raw 数据重建、风险闭环和验证证据见 `03_原始装甲数据重建与风险整改报告.md`。

## 2026-10-09 审查整改与兼容边界

0.5.0 / 张量契约 1.4.0 修改依据为 `06_代码科研审查与深度学习数据衔接修改意见.md`，实际处理和未闭环项见 `07_审查整改工作日志_2026-10-09.md`。旧数据与原始反例保留原样；新代码不把旧字段声明自动升级为新科学证据。新旧档案要区分契约版本，缺失身份/质量信息的旧 NPZ 只能显式取证读取或重新审计。

`interval` 现仅表示上下界，带 RNG 的调用会拒绝；原先隐式均匀采样应由使用者在有依据时明确声明为 `uniform`，不能自动迁移。现有 `lognormal` 的 value 表示中位数，数值采样算法保留。重复输出标准差是经验离散度，近似 CI 是条件均值区间，不是认知不确定性或现实预测区间，也没有任意停止时间或联合 95% 覆盖保证。人工稀有 Bernoulli 诊断已观察到明显覆盖不足，该方法风险仍未闭环。

切分的独立单位和跨数据集 split registry 应由研究设计提供；哈希分组不证明独立性或覆盖平衡。许可、原始数据授权、正式评价定义和对象域适用性需要明确依据，目前不推定授权或物理有效性。
