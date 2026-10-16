# ADR-0001：采样运行与归档完整性的所有权

- 状态：Accepted
- 日期：2026-10-16

## 背景

采样 checkpoint 与分片 writer 曾互相导入，并由 CLI 调用采样模块的私有 locked 实现。归档 checkpoint 结构、分片目录 inventory 和已提交 shard 校验也被采样模块通过 writer 私有方法间接调用。

## 决策

1. `sampling_checkpoint` 拥有 reference/adaptive 状态机、候选/轮次 lineage 与 archive 绑定事务。
2. `sharded_dataset` 拥有分片执行、恢复和发布流程。
3. 中立的 `archive_validation` 深模块拥有 monolithic/sharded 归档重开、版本化 manifest Schema、checkpoint 结构、目录 inventory、已提交 shard、tensor 合同验证和稳定错误分类；采样、writer、loader 与 policy 均调用它。
4. `input_identity` 定义 captured case source/input interface；具体 reference 传输仍由采样模块实现。
5. CLI 只调用公开采样 interface。locked helper 留在其 module 内部。
6. 成功验证返回不可变 `VerifiedArchive` receipt，绑定 root manifest、checkpoint、shard manifest 与全部已提交分区；新 policy 绑定必须先执行 receipt freshness 校验。旧 manifest-only policy API 仅作为兼容边界保留，不代表整个分区已验证。
7. monolithic 与 sharded writer 在首次提交以及完成态 resume 时，都必须用公共 verifier 重开归档；只有重开成功后才能删除 incomplete marker 或报告提交完成。
8. 无 `schema_version` 的历史 manifest 继续走既有只读语义合同；当前 writer 不生成该格式。声明版本的 manifest 必须通过公共 Schema 校验。

## 后果

- package 导入图不得包含 `sampling_checkpoint ↔ sharded_dataset` 强连通分量。
- checkpoint/schema/ledger 版本不因本决策改变。
- 任意源码变化都会正确改变 `ENGINE_CODE_HASH`，因此 dataset/provenance 身份可以变化；case、candidate、lineage、split/spec/tree 身份公式不得变化。
- 软件完整性通过不升级为物理有效性或研究准入。

## 验收

- AST 依赖测试证明无循环、CLI 不导入下划线 checkpoint 符号。
- reference、adaptive、resume、tamper 与旧归档读取测试通过。
- writer compatibility delegate 与共同完整性实现统一给出 `ArchiveIntegrityError`、稳定 `integrity_kind` 和关键 context。
- monolithic/sharded 首次提交、完成态 resume、loader worker reopen 和 receipt-aware policy 均通过公共 verifier。
- tamper matrix 覆盖 writer 首次提交、monolithic/sharded loader、policy、真实 sampling-checkpoint、版本化 Schema、同步 hash 后的 tensor 语义损坏，以及 receipt 后的 checkpoint/shard manifest/partition 漂移；不同入口返回统一错误类别和关键 context。
