# 软件契约可复现检查清单

2026-10-09。本轮依据 06 审查意见完成通用数据可靠性整改；不执行领域仿真、训练或效应参数校准。

| 静态 essential | 结果 | 证据含义 |
|---|---|---|
| README | OK | 当前安装、数据契约及验收说明存在 |
| ENV | OK | 库依赖范围、历史顶层包锁和实际验收环境另列 |
| SEED | OK | 当前源码中发现 seed 声明；不认证完整训练确定性 |
| ENTRYPOINT | OK | CLI 与验收脚本入口存在 |
| DATA | OK | 数据说明与来源/用途边界存在 |
| RESULTS_CMD | OK | README 中有可运行命令 |

静态扫描对象是当前 package/tests/docs 的独立副本，未包含历史 wheel 安装副本；结果见 `review_evidence/remediation_static_repro.txt` 与 `remediation_acceptance.json`。

实际检查：128 个通用/人工回归，0 失败；真实 PyTorch Windows spawn、复制隔离、词表稳定、缺项及哈希/质量拒绝；离线 wheel 在独立目录安装后进行人工读取与 CUDA 张量往返。命令：

```powershell
& 'D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe' review_evidence/run_remediation_acceptance.py
```

环境信息保存在 `remediation_environment.json`、验收前后 environment ID 与 `requirements-validation-pytorch312-20261009.txt`；后者为 158 个已安装发行版版本快照，没有 wheel 摘要，不是跨平台完整锁。原始数据权利、独立科学证据和对象适用性未推定。完整训练、实际规模和掉电恢复未验证。

本清单证明本轮软件契约行为，不代替研究结果复现或现实验证。后续如需正式归档，可单独使用 prepare-artifacts；若有论文指标，另用 verify-results 对照真实结果，不由本清单宣称通过。

## 2026-10-10 采样协调器 / pilot 增量检查

- 静态 essential 复查：README / ENV / SEED / ENTRYPOINT / DATA / RESULTS_CMD 全部 `OK`。
- `pytorch312-lmdt` 完整套件：`244 passed in 20.17s`。默认沙箱临时路径会对文件 `resolve` 误报 `WinError 5`，改用可写纯 ASCII `D:\tmp` 后全部进入断言并通过。
- 真实执行的最小 pilot：`raw_data/dataset_pilot_v3`，4/4 有效、0 拒算、2/1/1 切分，manifest 1.3 / Tensor Contract 1.5，checkpoint=`pilot_complete`。
- 已用 `TorchShardedDamageDataset` 验证分片哈希、长度和 1.5 默认输入。
- 验收只证明软件契约与试点文件的完整性。物理注册表仍为 `unverified`，标签仍为 `synthetic_teacher`，不具备 research training/validation/test 准入。

详细实施、失败路径与剩余风险见工作区 `11_采样协调器与数据集生成器实现日志_2026-10-10.md`。

## 2026-10-11 第12号意见通用软件整改验收

以当前冻结源码的 `review_evidence/remediation_20261011/acceptance_final.json` 为准：指定 `pytorch312-lmdt` 的18个通用/人工数据测试文件 **372 passed，0 failures/errors/skips**，另明确排除一项领域仿真测试。启用运行 guard 拒绝领域引擎/效应执行；未训练模型，不把本次结果与前轮128/244或分项数量相加。

| 静态 essential | 当前结果 | 证据与处理 |
|---|---|---|
| README | OK | 当前安装及数据/用途边界存在 |
| ENV | WARN | PEP621依赖范围未锁定；实际环境版本已捕获，但没有可移植完整锁或wheel哈希 |
| SEED | OK | 静态发现seed声明；没有新作完整训练/GPU确定性验证 |
| ENTRYPOINT | OK | CLI及本轮验收入口存在 |
| DATA | OK | 有数据指令和来源/准入边界 |
| RESULTS_CMD | OK | README有现存运行命令；本轮没有报告训练指标 |

静态扫描由 `test-research-code/scripts/repro_check.py` 执行，扫描当前源码的独立副本，六项present、5 OK/1 WARN、0 missing、exit0。保留ENV WARN的原因是本轮环境捕获证明“在哪个现有环境验收”，尚未验证从空环境重建所有依赖；未为消除警告改变兼容依赖范围。证据为 `static_repro_final.json/txt`，不复用上轮六项OK结论。

入口：

```powershell
& 'D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe' review_evidence/remediation_20261011/run_acceptance.py
```

该脚本捕获 `environment_final.json` 与 `requirements-validation-pytorch312-20261011.txt`，然后执行有领域运行guard的人工回归并计算源码/原档案摘要。seed行为保留原实现，由人工规范与冻结身份回归检查，没有修改采样策略。版本清单没有wheel摘要，不是跨平台完整锁；回归时环境中旧发行版metadata0.5.0与源码0.6.0.dev2分开记录。

新0.6.0.dev2 wheel使用本机已有构建依赖在 `--no-index` 下构建，以 `--target` 安装于独立目录；空cwd、PYTHONPATH仅指向该目录的人工tensor1.6读取通过，source/distribution版本及实现/源码/schema摘要与372验收一致。真实PyTorch0/2spawn persistent workers各两epoch检查mixed weight mask、raw weight及task mask。`wheel_smoke.json`记录此结果；没有验证从零离线构建运行依赖或执行完整训练。

历史pilot只读5,045个检查仍发现16个既有Parquet嵌套字段缺失，`all_checks_passed=false`；未迁移。234个原审查/pilot/reaudit文件及集合保留，末端11项交付对账成立。完成状态只表示归档完整，scientific quality/admission单列未建立。完整逐项变更、失败历史与未决项见 `13_审查整改工作日志_2026-10-11.md`、`completion_summary.json`、`final_delivery_review.md`。

若后续做正式归档/论文指标对照，分别由prepare-artifacts和verify-results承担；本清单不宣称获得artifact badge或研究结果复现。

## 2026-10-11 第16号数据分布整改验收

最终软件验收以 `review_evidence/optimization_20261011/completion_summary.json` 的 run `20261011T121824_762841Z` 为准：指定 `pytorch312-lmdt` 下 28 个合同/人工数据测试文件 **472 passed，0 failures/errors/skips**。验收 guard 拦截引擎、fragment/blast、fragment generation、dataset worker 和进程池入口；没有生成新领域标签或训练模型。

| 检查 | 结果 | 证据边界 |
|---|---|---|
| README / ENV / SEED / ENTRYPOINT / DATA / RESULTS_CMD | 6 OK，0 WARN，0 MISSING | `repro_check.py` 扫描干净源码快照；只证明软件制品具备复现入口与说明。 |
| 源码稳定性 | 通过 | 测试前后 tree hash 均为 `sha256:318337601471a470748def25986be3cac5a65a3d774c6f3cf509df8051e45f27`。 |
| 原审查与 2000 条归档保留 | 115/115 文件，前后均无缺失/新增/变化 | 证明文件未被本轮覆写，不证明数据科学有效。 |
| 当前 2000 policy/loader | manifest 精确绑定；Train `train_pilot` 1600 条只读加载通过 | 数据仍为 `software_interpolation_pilot / software_development`，不是 research train/validation/test。 |
| schema 镜像 | 五组公开/包内 schema 字节一致 | package-only physics registry schema 为有意布局。 |
| adaptive 执行链 | 人工 reference→content-bound proposal→run-selected→manifest 1.6/tensor 1.7→archive-derived update 通过 | 不证明 acquisition/OOF、proposal、物理标签或估计量的科学正确性。 |
| wheel | 离线 `pip wheel --no-deps --no-build-isolation`、隔离 `--target` 安装与 smoke 通过 | wheel SHA-256 为 `sha256:86ba66576d2bc3000ef88c2e10590d20cccd7730270cdc2e8868269dd8393ccc`；未验证从空机器离线重建所有运行依赖。 |

复现入口：

```powershell
& 'D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe' review_evidence/optimization_20261011/run_acceptance.py
& 'D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe' review_evidence/optimization_20261011/run_wheel_acceptance.py
```

首次沙箱验收因 Windows `Path.resolve(strict=True)` 对已创建临时文件返回 WinError 5 而失败，失败 JUnit/log 保留；最终运行按权限规则在沙箱外使用相同命名环境和 no-domain guard。PyPA `build` 模块在当前环境不可执行，wheel 脚本保留该失败后使用本机已有 pip/setuptools/wheel 的无网络 fallback。

当前规模门仍为 blocked。尚缺进程/存储级 Test 密封、完整输出任务/损失/评价合同、参数有效域来源、独立物理标定、教师收敛与 CI 覆盖、学习曲线/预注册实际改进、外部验证、多目标 OOD 合同和 research label admission。本清单不能用来声称物理可信、统计代表性、模型有效或研究结果复现。

## 2026-10-21 SCNet R04 回传文件与运行证据审计

本次仅做归档完整性、输入身份和程序运行证据检查，不执行武器效能分析、响应校准或领域仿真。

静态扫描对象为独立子包 `scnet_lsdyna_plate_calibration_20261017/`：README、SEED、ENTRYPOINT、RESULTS_CMD 为 OK，ENV、DATA 为 MISSING，退出码 1。根项目存在环境文件，子包也存在 selftest 脚本，因此该启发式结果不代表整个项目无环境或无测试。完整原始结果保存在该子包 `analysis/72095164_return_audit_20261021/repro_check.json`。

| 已执行检查 | 结果 | 证据边界 |
|---|---|---|
| ZIP CRC、路径/类型检查 | 100 条目通过 | 不证明数据库格式或科学正确性 |
| 展开后独立 SHA-256 复核 | 100 文件一致，原 ZIP 未变 | `verification.json` |
| R04 输入及五个上传附件身份 | 与上传基线一致 | `input_identity.json` |
| 日志扫描 | 32/32 rank 有正常终止标记 | 编号警告保留，不作数值资格结论 |
| 本轮审计脚本 | 实际执行成功，编译检查通过 | 不执行回传脚本或物理后处理 |

日志索引可复核命令（仅重写派生审计文件）：

```powershell
& 'D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe' 'scnet_lsdyna_plate_calibration_20261017\analysis\72095164_return_audit_20261021\audit_runtime.py'
```

环境记录在同目录 `audit_environment.json`，未安装/升级依赖。当前剩余项为数据库格式与元数据可读性检查、输出缺项来源核对，以及独立子包运行说明。详见根目录 `44_SCNet_R04回传归档审计与下一步规划_2026-10-21.md`。本轮不进入正式投稿归档或论文指标复现流程，不声称通过 artifact 评定。
