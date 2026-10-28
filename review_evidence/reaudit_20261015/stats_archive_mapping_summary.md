# 正式归档的表示一致性复核

日期：2026-10-15。解释器：`pytorch312-lmdt` / Python 3.12.13。

结论：对 `raw_data/dataset_5000_loitering_impact_fragment_only_v1` 的四个正式分片、5000 条既有结果，在以下明确范围内完成全量表示核对，未发现差异。64 个归档文件的前后逐文件内容哈希完全相同。没有构造领域 engine、运行领域仿真、重新生成数据或训练模型。

## 已完成的范围

- 四类主表：component、component_aux、armor、system。按案例及对象/机制/状态/指标等语义键匹配，核对 JSON 与 Parquet 的 mean、stddev、CI 两端、sample_count、uncertainty_estimable；再核对 Parquet 与 NPZ 的相应数值、availability、uncertainty mask 和有效 count。
- 第五类 Fréchet：核对 JSON 与 NPZ 的均值、标准差、CI 两端，并按可用性处理不可用统计量；没有逐字段核对 Parquet 中的嵌套 Fréchet struct。
- cases/components/armor 的 Parquet 特征、feature mask 与 NPZ；case_physics 和 solver_context 对 case_x 的按名称切片；冻结输入到现存六维 training view 的声明字段映射。
- 案例顺序、case hash、seed、epistemic ID、实际 R、参数特征内容摘要及结果/分片 engine hash 的存储一致性。

共执行 195152 组比较、121925000 次标量比较；包含同一信息在多种表示间的重复对照，不能解释成独立样本量或统计有效性证据。JSON/Parquet 按原精度精确比较，NPZ 按明确 float32 转换后精确比较，mask/count 精确比较，NaN 按对应缺失状态比较。

## 当前源码的相关保护

- `damage_engine/dataset.py:1380–1396,1455–1466`：部件标签的映射与可用性。
- `damage_engine/dataset.py:1557–1571`、`damage_engine/label_spec.py:208–217`：系统输出继承来源 availability；禁用来源不自动成为已知零。
- `damage_engine/dataset.py:1938–1942`、`damage_engine/label_spec.py:163–205`：LabelSpec 已进入系统标签数值及任务 mask 验收；旧“只有独立 helper”的缺口已修复。
- `damage_engine/dataset.py:1943–1965`：区分均值可用性、样本计数和不确定性可估计性。
- `damage_engine/utils.py:310–317`、`damage_engine/engine.py:950–965`：条件统计、固定认知实现与逐重复随机参数的作用范围已明示。
- `damage_engine/dataset.py:1789–1794,1861–1888`：审计输入切片及六维视图的名称、来源和转换声明。

## 不能据此推导的结论

归档生成版本为 `0.7.0.dev1`，当前源码声明为 `0.7.0.dev2`，代码 hash 不同。存储表示吻合不证明两个版本执行等价，也不是当前引擎的领域回归结果。

归档没有逐重复 replicate_trace；不能从聚合统计量独立重算逐重复联合对应关系、系统聚合过程或 CI 经验覆盖率。没有重算全部几何特征，没有逐字段核对 Parquet 嵌套 Fréchet struct，也未验收后续训练制品的完整调用链。

本复核不证明物理有效性、真实总体代表性、科研用途准入或训练效果。数值表示一致与科学验证是不同结论。

## 证据

- `stats_archive_mapping_audit.py`：最终审计脚本。
- `stats_archive_mapping_audit.json`：最终结果，`status=completed`、`result_count=5000`、`mismatch_counts={}`、`archive_files_unchanged=true`。
- `stats_archive_mapping_audit_attempt1.json`：保留的脚本初次字段别名错误；它不是正式归档缺陷证据。

最终状态：**上述限定范围的归档表示一致性通过；科学有效性与当前版本执行等价性未验证。**
