# R06 配对诊断候选执行日志

日期：2026-10-23（Asia/Shanghai）  
范围：从 R05 阶段 A/B 证据出发，设计、生成、校验并封装“可由用户提交但尚未提交”的 R06 诊断候选。  
历史只读来源：`05/`，Job `72252362`。  
当前结论：`ready_for_user_submission_not_submitted_not_solver_validated`。

## 1. 设计决策

1. 没有直接制作两个 treatment 并拿 R05 七工况大 deck 当作基线。单工况化会同时改变模型规模、MPP 分区和 I/O；因此将两个诊断改成两个配对实验，共四个 solver job。
2. 800 m/s、0.75 mm 配对只改变 `termination_time_ms: 0.04 -> 0.08`，检验 R05 的晚期首次穿越与尾窗不稳定是否来自观察窗不足。
3. 2000 m/s、0.75 mm 配对只改变 `*CONTROL_SOLID` Card 1 的 `PSFAIL: 0 -> 910001`；set 910001 只包含 plate PID 1005，用来量化 eroding contact 自动负体积判据对 fragment 删除的贡献。
4. PSFAIL 只作归因，不预设为修复。即使历史 76 个 fragment 负体积事件全部消失，按事件数代理估算的捕获前删除仍至少为 `(448-76)/686 = 54.227%`，远高于冻结 5% 门。
5. 接触族切换与 fragment `MAT_015 EROD=1` 被记录为后续备选，但未混入本轮主对照：前者会削弱侵蚀后新暴露面的接触处理，后者会保留失去偏应力承载的失败质量，均不适合直接成为生产响应。

设计冻结文件：`design/p0_r06_diagnostic_design.json`，SHA-256 `9443fc7397c78db6d9fb7fd0b2b55ac185972f52fae5c90ea3641df8598da402`。

## 2. 共同观测升级

四个 job 使用完全相同的观测配置；这些配置不计为 pair 内的物理因素：

- D3PLOT：`0.0001 ms`；
- 五字段 D3PART：`5e-5,0,0,0,900001`，set 900001 只含当前 fragment PID；
- `*DATABASE_EXTENT_BINARY`：`NEIPH=6`，请求 MAT_015 的 Johnson–Cook damage `D` 历史变量；
- contact Card 1：`SAPR/SBPR=1`；
- `*DATABASE_BINARY_INTFOR_FILE`：文件名 `intfor`、间隔 `5e-5 ms`；
- `*DATABASE_EXTENT_INTFOR`：压力、剪切、力、gap，并请求 `NPEN=1`；
- NCFORC、GLSTAT、MATSUM、SLEOUT、NODOUT：`5e-5 ms`；
- `DT2MS=0`、timestep `ERODE=0`，没有质量缩放或时间步侵蚀。

INTFOR 的实际字段、符号、SOFT=2 下的 penetration 支持以及 NEIPH=6 的真实可读性仍须由目标 R15.0.2 回传确认；静态校验没有把它们写成已验证结果。

## 3. 生成与身份

生成器：`scripts/generate_p0_r06_diagnostics.py`。它执行非覆盖式 staging 发布，且强制 `package_revision` 以 `R06_` 开头，拒绝 R05 revision。

| Wave | Job | 唯一因素水平 | 节点 / 单元 | 原 R05 ID 区间 | Deck SHA-256 |
|---|---|---|---:|---|---|
| 1 | `r06d_0800_ctrl` | `T=0.04 ms` | 23,775 / 20,902 | node 23776–47550；element 20903–41804 | `37f5f2ef...f8725daa` |
| 1 | `r06d_2000_ctrl` | `PSFAIL=0` | 23,775 / 20,902 | node 95101–118875；element 83609–104510 | `3b618832...3d177ae2` |
| 2 | `r06d_0800_tend008` | `T=0.08 ms` | 23,775 / 20,902 | node 23776–47550；element 20903–41804 | `3eb30f6c...f24fc6d8` |
| 2 | `r06d_2000_psfail_plate` | `PSFAIL=910001` | 23,775 / 20,902 | node 95101–118875；element 83609–104510 | `ee31fbdc...691af2f5` |

每个 deck 的节点坐标、单元连接、节点/单元 ID 和 PID 均逐项与历史 R05 deck 核对一致。控制/变体的自动差分证明：800 pair 只允许 TITLE 与 CONTROL_TERMINATION 差异；2000 pair 只允许 TITLE 与 CONTROL_SOLID 差异，且后者仅 Card 1 field 5 从 0 变为 910001。

旧 `scripts/generate_p0_r05.py` 的命令行入口增加了 frozen-revision 防误用保护。默认不再允许用历史 `R05_capture_ledger_preregistered` 身份重建不同字节的五字段 deck；只有显式测试 fixture override 才能执行。

## 4. 静态校验与打包

静态校验器：`scripts/validate_p0_r06_diagnostics.py`。

每个 job 均通过：

- 长字段模式、全部 30 类关键字/40 个 block 的字段类型合同；
- 单工况 case map、manifest、材料、设计、successor contract 哈希；
- 五字段 D3PART、NEIPH=6、INTFOR_FILE、EXTENT_INTFOR、NCFORC；
- fragment-only D3PART set 与 plate-only PSFAIL set；
- 历史 R05 网格逐节点/逐单元一致；
- `DT2MS=0`、`ERODE=0`、1 node/32 ranks 资源身份；
- pair 内单因素隔离。

Readiness：`manual_upload_20261023_r06_diagnostics/R06_SUBMISSION_READINESS.json`，SHA-256 `57ea35602a874897b4b52cf8c1c3c462f4e82c278b25101d42ae81aa7152b754`。

打包器：`scripts/build_p0_r06_diagnostic_bundles.py`。每个 ZIP 采用显式白名单，只含一个 `.k` head deck；不包含 `05/`、`analysis/`、`.vendor/` 或旧候选。两次独立构建逐 ZIP 字节相同。

| Wave | ZIP | Bytes | SHA-256 |
|---|---|---:|---|
| 1 | `uav_hf_r06d_0800_ctrl.zip` | 412,925 | `5298582d8fe163f5141eb29738f4a2ab514635a1de41610317f8e4734099f7d7` |
| 1 | `uav_hf_r06d_2000_ctrl.zip` | 423,125 | `bd4e0a5ba8eeebee2d6e26d15a8791e57d6a86fe8f3f688cec2f96d7065abbc2` |
| 2 | `uav_hf_r06d_0800_tend008.zip` | 412,996 | `97743dbc58f4e5aa6fc07a3c4c3f2e89bb007209e901805fd307d4ce1c8e9859` |
| 2 | `uav_hf_r06d_2000_psfail_plate.zip` | 423,223 | `a71db73c63bd6430053e84ac890ee264ce52c95900283d3dac3217a564b84816` |

Release manifest：`artifacts/r06_diagnostics_20261023/R06_RELEASE_MANIFEST.json`，SHA-256 `93a473bc5dc6fe11905f9939acd677391c2ffc71468c1b43654f2370505e0820`。

## 5. 独立 release verification

`scripts/verify_p0_r06_release.py` 重新执行了以下检查：

- fresh static readiness 与磁盘存档逐字段一致；
- 4 个 ZIP 的大小/哈希、外部 manifest、内嵌 manifest 和成员哈希一致；
- ZIP 无重复成员、绝对路径或 `..` 路径，且每包恰好一个 head deck；
- 历史 R05 candidate 的 deck/design/case-map/manifest/static-validation 固定哈希不变；
- `05/` 的 75 个关键文件、2,944,160,835 bytes 逐文件 SHA-256 不变；
- `05/` 完整目录仍为 120 文件、3,606,867,357 bytes。

结果：`analysis/r06_submission_readiness_20261023/r06_release_verification.json` 为 `PASS`，SHA-256 `5c999d87e10cde52ac403f309d2e9a5fe905006548fdb7ab6d8e910870f0a42a`。

## 6. 软件验证

- 新增/修改脚本 `py_compile` 通过；
- 新生成 JSON 共 36 个，严格解析通过，无重复 key、NaN 或 Infinity；
- 最终全量回归：`103 tests in 126.432s, OK`；
- 另有包/合同重点回归：`27 tests in 108.311s, OK`；
- 本机未发现 LS-DYNA 可执行文件，未进行本地 solver parse 或短算。

## 7. 停止点与提交顺序

工程状态已经达到“可由用户提交 R06 诊断作业”，但没有代替用户访问 SCNet 或实际提交。

1. Wave 1：先提交 `r06d_0800_ctrl`、`r06d_2000_ctrl`。
2. 只有两个 control 都通过 R15.0.2 解析、32 ranks 正常终止、能量/沙漏/质量门和真实 D3PART/INTFOR 可读性后，才提交或解释 Wave 2。
3. Wave 2：提交 `r06d_0800_tend008`、`r06d_2000_psfail_plate`。
4. 四个 job 必须使用各自 `UPLOAD_README.txt` 中的唯一 job name/work directory；不得复用目录或开启平台“自动填充上次参数”。

当前仍未满足：目标 solver parser/runtime、真实新增输出可读性、配对响应结论、P0 准入、P1 准入和独立物理有效性。因此不得据此拟合正式仿真引擎或生成参考数据集标签。
