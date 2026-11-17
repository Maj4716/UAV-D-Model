# 项目清理与重构验收记录

## 结果

项目已收敛为软件层合成数据生成与验证仓库。高保真工作区、原始求解回传、旧中间研究池、旧数据归档、49 份已归并的编号文档、一次性仪表板工具与可再生缓存均已移除；未创建 Git 提交。

精确计量的删除量为 9,617,546,016 bytes（8.96 GiB），另有未纳入字节台账的测试临时目录和字节码缓存。当前工作区除 `.git` 与异常 ACL 的忽略缓存入口外，共 276 个文件、73,362,650 bytes；其中主要体积来自本地保留且 Git 忽略的开发数据载荷。

## 新结构

- `damage_engine/`：当前软件教师与数据管线实现；
- `configs/`：按 cases、targets、physics、sampling、labels、import 分层；
- `data/source/`：来源表，`data/generated/` 用于可再生输出；
- `artifacts/development_dataset/`：本地神经网络开发数据包；
- `artifacts/validation/`：带 SHA-256 的紧凑稳定证据；
- `docs/`：8 份按主题维护的当前文档；
- `tools/`：保留可重复执行的验证、诊断、基准与研究工具。

根目录只保留项目入口、依赖文件和上述功能目录。现行代码、配置、测试、工具与文档中，旧 `raw_data`、`review_evidence`、本机绝对路径和已退役求解工作流引用均为 0。

## 软件与数据验收

| 检查 | 结果 |
|---|---|
| 全量 pytest | 1,044 passed；0 failed/errors/skipped；256.05 s |
| 重构前后测试数 | 1,044 → 1,044 |
| Schema 镜像 | 通过 |
| `compileall` | 通过 |
| `pip check` | 通过 |
| CLI 单案例合同 | 通过 |
| 5,000 候选采样计划 | 通过；未执行标签 |
| PyTorch 数据读取 | train 7,965；validation 2,624；输入 15 维；部件 67 维；树 7 维 |
| 开发数据独立重算 | `passed` |
| `git diff --check` | 通过 |
| 稳定证据哈希 | 8/8 通过 |

最终 JUnit 为 `pytest_after.xml`，开发数据验证报告位于 `../development_dataset/validation_report.json`。

## 数据研究边界

当前数据包包含 10,589 个唯一候选、7,337 个完整域响应、3,252 个域删失样本和 16,000 条选择观测。它没有封存测试集，角色为 `development_ready`。

V5 代理门通过，教师总门 `overall_pass=false`。Coverage 作为训练增强臂保留，Reference 继续承担参考分布估计。目标参数保持 `unverified`，引擎候选的 `selected_candidate_id=null`。

## 发布审计

保守发布审计的总提示由 1,610 条降至 374 条，身份泄漏提示为 0。剩余提示主要是测试常量、`set` 使用启发式、两处注释示例、冻结预注册文件名以及本地忽略的大数据载荷；它们不是本次目录清理的失败证据，也未在未授权情况下改动可能影响数值结果的实现。

## 已知残留

根目录仍有一个被 `.gitignore` 覆盖的 `.pytest_cache` 目录入口。Windows ACL 拒绝读取、接管和删除；最终测试使用 `-p no:cacheprovider`，该目录不参与运行、证据或提交。其余临时目录和字节码缓存已清除。
