# 人为因素修复与物理剩余因素辨识执行日志

- 开始时间：2026-10-22 23:53（Asia/Shanghai）
- 工作区：`D:\Desktop\UAV新`
- Git HEAD：`7c9e088955f1830a7590d016a28eecf0654c0f1d`（`main`）
- Python：`D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe`，3.12.13
- PyTorch：2.8.0+cu128；NumPy：2.4.6
- 开始时 D 盘可用空间：184,140,222,464 bytes（约 171.49 GiB）

## 已冻结的证据边界

1. 旧 5000 条正式归档只用于描述历史分布；其引擎身份为 dev1，不能与当前 dev3 工况逐条混合推断。
2. 当前 Test 已用于诊断，后续模型确认必须重新封存独立测试集。
3. 新方向/姿态设计是项目专用、接触条件下的设计分布，不解释为真实作战发生率。
4. 诊断源轴翻转只用于因果辨识，不直接替换生产物理参数。
5. 高保真标定尚未完成；本轮候选毁伤教师只允许标记为 `project_specific/software_only`。

## 执行计划

1. 新增多方向、目标朝向、首接触约束下的姿态/滚转采样合同；所有变化自由度进入 learner-facing 输入。
2. 保留拒绝候选账本，未命中、反向或不满足接触合同的候选不执行教师、不赋正式标签。
3. 运行方向、滚转、弹型、源轴与重复数的受控析因诊断，并量化各因素对 KMFC 上游支持的影响。
4. 新增不覆盖旧树的 KMFC 两级候选教师：零锚平滑证据、逐重复硬门、等级嵌套与 availability/mask。
5. 运行针对性与全量测试，检查输入身份、固定场景身份、接触合同、顺序性、数据加载和可复现性。
6. 形成科研报告，区分已证实人为近因、仍待标定的物理因素、当前可以执行的改进以及禁止外推的结论。

## 基线验证

- 命令范围：`test_loitering_model_input_contract.py`、`test_review35_contact_conditioning.py`、`test_contact_contract.py`、`test_case_factory_geometry.py`、`test_ordinal_projection.py`
- 结果：78 passed，0 failed，0 error，0 skipped。
- JUnit：`review_evidence/pytest_baseline_20261022.xml`

## 文件清理策略

- 保留旧正式归档、R04/R05 证据、历史报告和源码。
- 只删除本轮产生且已确认无复核价值的临时缓存/中间文件；删除前后记录空间。
- 当前 `.tmp` 仅约 0.0003 GiB，暂不为节省空间删除证据。

## 已执行整改与验收

### 接触流形与学习输入

- `SamplingSpec` 升级到 1.4，新增六来向、装甲面内瞄点、离面距离、速度和绕弹体前向轴滚转；弹型参数另作为一个潜变量维度。
- 正式样本要求速度和弹体 `+X` 轴均指向由 S-CONTACT 求得的首次装甲接触；不相交候选进入拒绝账本，`formal_teacher_executed=false`，不得伪装成零毁伤标签。
- 训练视图包含 profile 3、origin 3、velocity 3、连续 6D 旋转 6，共 15 维；四元数 `q/-q` 映射到相同 learner identity。
- smoke：18/18 正式样本通过接触准入，六来向各 3 条，body `+X`/velocity 最小余弦为 1.0，Torch 批张量形状为 `[18,15]`。
- 拒绝试验：22 个候选中 18 个正式样本、4 个 `no_forward_armor_intersection`，4 个拒绝样本均未执行教师。
- 定向/兼容测试：47 passed。

### 毁伤树原型

- 新增 opt-in 的 KMFC 0/1/2 有序教师软件原型；不覆盖旧七输出树。
- 连续证据采用零锚 Hill 激活和正权幂均值聚合；最终在逐重复层执行硬门，并由同一类别严重度及父级约束保证 level 2 ⇒ level 1。
- 缺失证据输出 unavailable/null；已知无证据输出 level 0，防止把未知误当作无毁伤。
- 配置参数仅为软件哨兵值，尚未标定；`research_training_admission=false`。
- 定向测试：51 passed。

### 受控因果诊断

- 固定平衡诊断网格共有 162 个配对候选；两个源轴条件共 324 个已接触工况，Rmax=8，共 2592 条重复轨迹。
- 162 个独立随机流按完整候选键生成，并仅在同一候选的两个源轴条件之间配对；R=1/2/4/8 使用嵌套前缀。
- 当前归档源轴 `-body X` 在 R=8 时的上游支持率：K 8.64%、M 11.11%、F 14.81%、C 7.41%。
- 诊断翻转 `+body X` 在 R=8 时的上游支持率：K 72.22%、M 100%、F 82.72%、C 99.38%。
- 配对风险差分别为 +0.636、+0.889、+0.679、+0.920；这证明源轴符号是当前软件中最大的已测敏感因素，也反证“固定弹型—目标无论轴语义如何都必然同等稀疏”。它尚未证明历史 `-body-X` 是人为错误；该翻转仍是诊断干预，未获准作为生产物理参数。
- `controlled_factorial_validation.json` 的 `overall_pass=true`；全部区间只解释为固定诊断网格配对稳定性区间。

### 最终软件回归

- 当前 `ENGINE_CODE_HASH`：`sha256:8866e874dd6bc306e6c087251312a53d6f6681dbb6c5e7da6fd2bed32412e00a`。
- 全量 pytest：892 passed，0 failed，0 error，0 skipped，77.454 s。
- JUnit：`review_evidence/human_factor_remediation_20261022/pytest_full_final_20261023.xml`。
- `python -m compileall -q damage_engine` 通过；`python -m pip check` 返回 `No broken requirements found.`。
- 最终结论边界：软件合同和因果近因辨识已通过；源轴物理语义、战斗部参数及两级阈值仍需独立高保真/实验标定，因此正式研究训练准入保持 false。

## 空间管理结果

- 最终析因持久证据约 3.70 MB；未生成新的大规模正式归档。
- 本轮 pytest 隔离产物约 83,689,036 bytes（79.81 MiB），均位于 `D:\Desktop\UAV新\.tmp`。
- 递归删除命令被运行环境自动审批拒绝；未尝试绕过，临时目录原样保留。
- 完成时 D 盘可用空间为 184,031,293,440 bytes（约 171.39 GiB），临时占用不足可用空间的 0.05%。
