# 2026-10-15 通用归档与审计算法审查

本审查只验证现有存储表示、身份、数量、状态集合，以及人工无害数据上的通用审计算法。未运行领域引擎、生成标签、训练模型，亦未计算或优化领域输入窗口/边界/几何覆盖。

## 既有档案：表示一致，证据具有明确版本边界

对象为 `raw_data/dataset_5000_loitering_impact_fragment_only_v1`，共 64 个文件、4 个分片、5,000 个 formal cases。解释器为 `D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe`；NumPy/PyArrow 版本及全部绑定源码 SHA-256 见证据 JSON。

`io_consistency.py` 完整执行 795,294 个检查组、106,705,000 个标量比较，未发现跨格式不匹配或未处理异常。检查组是标量/数组/集合/身份检查的混合，不能称为独立样本或独立统计实验。NPZ 根据语义 ID 和名称与已有 JSONL 统计对象对照，Parquet 按 case/component/armor/mechanism/state/metric/output 连接，不依靠行号相等来替代语义主键。每族 mean 的位置数如下；同时比较 stddev、两个 CI 端点、availability、uncertainty mask 和 count；Parquet 还比较统计对象的 `uncertainty_estimable`。

| 标签族 | mean 标量位置数 | JSONL→NPZ / JSONL→Parquet |
| --- | ---: | --- |
| component | 2,010,000 | 一致 |
| component_frechet | 1,340,000 | 一致，包含两个嵌套 bound 对象 |
| component_aux | 4,690,000 | 一致 |
| armor | 130,000 | 一致 |
| system | 35,000 | 一致 |

根及分片 partition hash、manifest/checkpoint 身份、Parquet 显式 schema 与声明行数、formal transport 的冻结 sidecar hydration 后 case hash、provenance、采样身份、冻结 registry split 均通过。`formal_cases.jsonl.gz` 是紧凑传输表示，需通过既有 `_iter_hydrated_formal_cases` 恢复共享注册表和合同文档再验逻辑身份；直接对传输行求 hash 会产生假的 5,000 项差异。源码位置为 `damage_engine/sampling_checkpoint.py:403`、`:479`、`:538`，canonical subtree 缓存在 `damage_engine/input_identity.py:76`。

candidate frame、selected、executed、successful 均为同一组 5,000 个 ID；refused 为 0。它们的显式 ID 集合与 completion summary 一致，逐输入 status/valid mask 亦一致。declared admitted 集合的数量与根 manifest 一致，且属于 successful 集合；这是对准入声明存储一致性的检查，不是独立科研准入验证。预结果 candidates ledger 中 `executed/successful` 各有 5,000 个 null，表示冻结时未知，不能按 false 解释。完成摘要在 `damage_engine/sampling_checkpoint.py:1395-1396` 区分各集合；当前 `sampling_audit.py:208-244` 也将未提供的 outcome 集合标为 `not_evaluated`。

档案声明 `engine.version=0.7.0.dev1`、`engine.code_hash=sha256:bf3b325e5504d180a22a14aab4948751f374f8a67eb77d6db956707d2770e787`；本次当前源码是 `0.7.0.dev2`、`ENGINE_CODE_HASH=sha256:55cf8377026d7fc3c8ef5a4828f3403991de61b825ecc7b23b4ba8905a70aa41`。两者不同。因此结论是“历史档案内已有表示一致”，不能写成“这些标签已与当前源码执行结果一致”。未执行新旧代码对照生成。

全部 5,000 个 result 均无 `replicate_trace`；未发现顶层 successes/trials/success_count/trial_count 合同。故无法核查 aggregate↔replicate 原始记录、逐重复联合对齐或从原始事件重算 success/trial 统计，也没有从均值反推成功次数。当前代码仅在输入显式启用时保存 trace：`damage_engine/engine.py:300`、`:518-519`；trace 构造位于 `:81-126`。这项缺口不等同于均值错误，也不由 hash 或跨格式复制一致弥补。

档案前后全部文件 SHA-256 相同，绑定源码前后 SHA-256 相同，清单保存为 `io_inventory_before.json` 和 `io_inventory_after.json`。首次检查器适配错误已保留为 `io_consistency_attempt1.json`：它错误地直接 hash 紧凑 transport，以及错误使用 Arrow nested-leaf 投影导致重名列；这些错误已修正，首轮错误计数不是档案缺陷。

## 人工无害数据确认的通用算法问题

证据为 `io_audit_algorithms_toy.py/.json`；该脚本不读取任何真实领域输入。以下问题未被外推为当前正式档案已经发生相应数值错误。

正向合同证据另存为 `io_sampling_contracts_toy.py/.json`：主 `sampling_audit` 对重复 candidate、跨 split 的 lineage/epistemic group、概率越界、负权重、非 bool selected、NaN/越界 latent 及 executed 非 selected 子集等 9 项人工异常均拒绝；正常人工例的加权 CDF/ESS 与手算相符，selected/executed/successful/admitted/refused 分层和严格 JSON 序列化通过。由此不能把下面描述工具 helper 的缺陷错误地归到已经存在这些防护的主采样合同。

### IO-A1（P1）：描述审计使用 glob 读入未承诺的分片

`tools/audit_loitering_impact_dataset.py:201-232` 与 `tools/audit_dataset_distribution.py:353` 直接发现 `part-*/tensors.npz`。前者检查已读数组的行数和轴名称，却不先验证 manifest/checkpoint 的 committed inventory、文件 hash 或 incomplete marker。人工目录中 manifest/checkpoint 只列一个分片，但 helper 会再读入未承诺的 `part-999999`，返回两条记录。最小 fixture 并未模拟完整合法档案；它证明 helper 没有咨询这些文件，而非证明已绕过严格 Torch archive loader。

修订原则：在审计入口复用只读归档验证器，再按唯一已承诺文件清单读取；额外分片应明确拒绝或独立记录，不能静默纳入统计。验收：对人工合法归档分别加入 orphan、incomplete marker、修改分片字节/轴名称；默认应停止，并留下具体完整性原因。

### IO-A2（P2）：两个组各自常数但均值不同，SMD 被记为 0

`tools/audit_dataset_distribution.py:110`、`tools/audit_loitering_impact_dataset.py:457` 在 pooled SD 为 0 时统一返回 0。人工三个 split 各自常数且彼此不同，所有比较的 SMD=0，而 KS=1。标准化效应在这个退化分母下不可按“无差异”解释；原始均值分离仍存在。

修订原则：区分“同一常数”“不同常数”“空组/缺失”，报告可序列化的 `not_estimable`/退化原因和未标准化差值，不用 0 代替未知。验收覆盖这三种情形及普通非零方差手算例；缺失比较不应使 maximum drift 默认为已证实的 0。

### IO-A3（P2）：ESS 对公共尺度不具数值稳定性

`damage_engine/adaptive_sampling.py:538-542`、`tools/audit_dataset_distribution.py:272-278`、`tools/audit_loitering_impact_dataset.py:896` 直接计算 `(sum w)^2/sum(w^2)`。三条相等正权重的 ESS 应始终为 3；人工公共尺度 `1e200` 导致非有限值，`1e-200` 导致非有限值或 0。均为有限、正的输入。部分上层会先缩放，但其他调用点直接传 raw weight，因此核心 API 仍不满足数值合同。

修订原则：先检查有限正向量和维度，再作公共尺度归一化或稳定求和；各调用者共用一个稳定实现。验收为正公共尺度不变性、结果有限且位于 `[1,n]`、单元素/空/负/非有限输入合同，以及最终 JSON 严格序列化。

### IO-A4（P1）：无效 eligible weight 被静默排除，probability>1 不计为违例

`tools/audit_loitering_impact_dataset.py:888-913` 将 eligible 但非正/非有限的权重从计算中排除，未单独报告 invalid weight count；positivity 检查只排除 `<=0` 和非有限值，遗漏 `>1`。人工 3 条 eligible 记录含一条负权重和一条 probability=1.1，helper 不报验证错误、只用两条权重得到 ESS fraction=1，概率违例数仍为 0。

修订原则：区分业务 mask=false 与数据本身无效；默认拒绝任何 eligible 无效权重/概率，并保存原分母、有效分母与明确违例数。验收覆盖负/零/NaN/Inf 权重、越界概率、mask 缺失和形状错位；不能仅通过过滤后 ESS 作为有效性判断。

### 其余两项为尚需明确的描述统计合同

`tools/audit_loitering_impact_dataset.py:408-425` 对声明 `low==high` 的人工常数轴调用 histogram，NumPy 自动扩展区间；输出仍显示原声明区间，缺少 `not_estimable` 原因。普通端点 toy 中左右闭合与 out-of-range 计数正常，但 NaN 被过滤后没有 missing count。应在执行前区分常数轴与合法区间，并保留全部分母。

`tools/audit_dataset_distribution.py:38-64`、`tools/audit_loitering_impact_dataset.py:60-90` 的 summary 会删除非有限值。人工 4 项输入仅报告 2 项 finite count，未提供原 count 或 invalid count。仅有限值描述本身可以成立，但不能据此推断全部 eligible 输入有效；应由调用方明确 fail-closed 或同时呈现 missing/invalid 分母。这是合同缺口，未据此宣称真实档案存在 NaN 污染。

重复审计目前使用 exact byte signature（例如 `tools/audit_loitering_impact_dataset.py:639-646`）。其结果不能等同于已完成“近重复”检验；后者需要预先声明、与实际领域数值无关的度量/尺度/容差合同。此次未计算领域近重复、覆盖、相关或 split shift。

## 通用解释与验收边界

空间填充、均匀性或权重 ESS 是相对于声明设计及特定 eligible 集合的描述。现有声明将 reference 标为 `design_distribution`；没有经过验证的 operational reference 及其连接证据时，不能把设计覆盖解释为真实使用频率或外部有效性。文件 hash、schema、跨格式一致和无害 toy 回归均不构成物理有效性、Monte Carlo CI 覆盖率或科研训练准入证据。
