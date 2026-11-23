# C 链实现与验收记录（2026-11-20）

## 已实现范围

1. `damage_engine/offline_tree_candidates.py`
   - weighted noisy-OR：严格采用 `1 - product(1 - w_j p_j)`，约束 `w_j in [0,1]`；
   - k-of-n：泊松二项分布精确尾概率；
   - 2-additive Choquet：Möbius 容量归一化与完整单调性检查，允许受约束的正/负交互；
   - 所有候选均计算稳定的 SHA-256 内容哈希；声明哈希不一致时拒绝；
   - 无领域审定时生成 identity-component fallback，状态固定为 `provisional_no_domain_semantics`，只透传部件—机理—状态，不建立功能分组或系统效应。
   - 已对当前 `target.fragment_only.v1.json` 中结构上未被树引用的 23 个部件生成可编译 fallback，并记录目标文件哈希；共 46 个 degraded/failed 透传输出。
2. `damage_engine/dataset_v2.py`
   - core 固定包含 inputs、逐重复 component-state trace、component labels、sampling weights；
   - core 文件逐项哈希，字段与形状严格验证，禁止树输出混入；
   - `views/tree_<definition_hash>.npz` 只保存从 core 轨迹派生的树输出；验证器逐重复重算后比较；
   - 添加 view 不改变任何 core 文件或哈希。
3. `damage_engine/tree_candidate_selection.py`
   - 开发 seed 与独立验证 seed 必须不同且都实际存在；
   - 门限、权重字段、排序和候选哈希写入报告；
   - 开发集选择不读取验证结果；只有独立验证过门且 `semantic_authority.status=approved` 才可推广；
   - 未审定候选可做数值诊断，但保持 provisional。
4. CLI 与研究记录
   - `tools/build_dataset_v2.py`：pack、add-tree-view、identity-fallback；
   - `tools/validate_dataset_v2.py`：哈希、契约、派生视图重算验证；
   - `tools/select_offline_tree_candidate.py`：预注册 ID 下的独立 seed 选择/验证；
   - `PREREGISTRATION_TREE_V1.md`、候选 JSON 示例和 CLI 最小流程。

## 科学边界

- weighted noisy-OR 的概率解释依赖声明的因果抑制/条件独立结构；权重必须由领域审定或校准支持。
- Poisson-binomial k-of-n 使用边缘状态概率时依赖条件独立假设；存在共同原因依赖时，应使用联合重复状态或另建依赖模型。
- Choquet 输出在数学上是 `[0,1]` 的单调聚合分数；没有外部校准时不能自动解释为频率概率。
- 数值门只检查 Reference 权重下的覆盖、零值、饱和与信息量，不证明物理有效性。23 个现有树未引用部件仍等待领域权威逐项审定。

## 验收证据

执行命令：

```powershell
D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe -m pytest -q -p no:cacheprovider `
  tests/test_offline_tree_candidates.py `
  tests/test_dataset_v2.py `
  tests/test_tree_candidate_selection.py `
  tests/test_dataset_v2_cli.py `
  tests/test_damage_tree.py `
  tests/test_recompute_tree_outputs.py
```

结果：`37 passed`。

覆盖的关键验收项：三类门的范围与坐标单调性质、Choquet 非单调容量拒绝、哈希稳定与篡改拒绝、identity fallback 的 provisional 边界、core/view 分离、view 可重算性、独立 seed、未审定禁止推广、CLI 端到端、现有默认树与原离线重算回归。

`damage_engine/damage_tree.py` 和现有 target 配置未修改；C 链没有切换仿真引擎默认树。
