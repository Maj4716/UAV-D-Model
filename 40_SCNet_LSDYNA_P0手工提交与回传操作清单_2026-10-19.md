# SCNet / LS-DYNA P0 手工提交与回传操作清单

**日期：** 2026-10-21（R04 修订）  
**作业目的：** 对 8 g 立方破片冲击 10 mm 参考钢板的速度响应和网格敏感性进行 P0 数值资格检查。  
**证据上限：** `high_fidelity_synthetic_reference_candidate`；不是实弹试验或真实装甲材料验证。  
**提交方式：** SCNet LS-DYNA 模板提交。  

> R01（Job ID `71903517`）因字段宽度失败；R02（Job ID `72021951`）因必需卡片/字段不完整而失败；R03（Job ID `72092170`）因把整数型 `ECDT` 写成浮点科学计数而失败。以下内容已切换到完成全量关键字类型合同检查的 R04；不得复用 R01/R02/R03 deck、远程目录或作业名。

---

## 1. 本次只提交什么

### 1.1 求解器输入文件

必须作为 LS-DYNA 输入选择的文件只有：

```text
D:\Desktop\UAV新\scnet_lsdyna_plate_calibration_20261017\manual_upload_20261021_r04\p0_velocity_mesh_calibration.k
```

- 文件大小：`20,604,806` 字节；
- SHA-256：`312f2ce1c42dd070097dad1036d1d670790704c55c5b534a3b845a0709709e08`；
- 第一行必须为：`*KEYWORD LONG=Y`；
- 包含 7 个互相隔离的 P0 子模型；
- 不要把 ZIP 文件选作 LS-DYNA 输入；
- 不要把 `p0_case_part_map.csv` 选作求解输入。

### 1.2 建议一同上传到工作目录的审计文件

以下文件不参与求解，但建议与 `.k` 放在同一远程目录：

```text
p0_case_part_map.csv
manifest.json
jc4340_reference.json
static_validation.json
UPLOAD_README.txt
```

这些文件用于回传后核对工况、part ID、材料来源与上传前哈希。

---

## 2. 在文件管理中上传

1. 打开 **控制台 → 高性能计算 → 文件管理**。
2. 本次明确选择 **西北一区（西安）**，因为所需软件版本 `ls-dyna_R15_mmp_d` 只在该区域显示。SCNet 不同资源中心的文件和软件相互独立。
3. 在个人目录下建立纯英文目录：

   ```text
   LSDYNA/uav_hf_p0_plate_r04
   ```

4. 进入该目录，点击 **上传 → 上传文件**。
5. 从下面的本地目录选择全部 6 个文件：

   ```text
   D:\Desktop\UAV新\scnet_lsdyna_plate_calibration_20261017\manual_upload_20261021_r04
   ```

6. 上传完成后核对 `.k` 文件大小必须为 `20,604,806` 字节，并可在线预览确认首行为 `*KEYWORD LONG=Y`、第 14 行末尾为 `...,0,0,0`。若任一项不同，删除错误上传并重新上传，不要提交。

该 deck 大于 10 MB，不使用 `rz`；普通文件管理上传即可，无需为 20.6 MB 文件启用超过 2 GB 才建议使用的快传。

SCNet 官方建议工作目录和文件名避免中文及空格，并要求上传位置与计算资源中心一致：[上传输入文件说明](https://www1.scnet.cn/help/docs/mainsite/hpc/beginner-guide/upload-example/)。

---

## 3. 选择 LS-DYNA 模板

1. 打开 **控制台 → 高性能计算 → 作业提交**。
2. 搜索：

   ```text
   dyna
   ```

3. 在 ANSYS LS-DYNA 卡片中点击 **模板**，不要创建图形桌面。
4. 若搜索不到 LS-DYNA，进入 **模板订阅/模板中心** 搜索并订阅后返回。
5. 软件版本选择：

   ```text
   ls-dyna_R15_mmp_d
   ```

   该条目位于 **西北一区（西安）**。页面名称写作 `mmp`，结合 `d` 后缀，它是列表中唯一同时明确表达 MPP 和双精度的 R15 版本。不要选择 `s_R13.1.1`、`s_R14`、`ls-dyna_R15_mmp_s` 等单精度版本，也不要选择并行模式不明确的 `d_R15` 或 `d_R14`。

SCNet 官方说明 LS-DYNA 模板需要选择节点、核心、工作目录和 `.k` 输入，并提醒 SMP/MPP 不匹配可能报错：[LS-DYNA 模板说明](https://www.scnet.cn/help/docs/mainsite/hpc/software-examples/use/CAE/LSDYNA/index.html)。

---

## 4. 表单应填写的值

| 页面字段 | 本次填写值 | 说明 |
|---|---|---|
| 作业名称 | `uav_hf_p0_plate_r04` | 与 R01/R02/R03 失败作业分离 |
| 求解方式 | `MPP` | 不选 SMP |
| 精度 | `double` / 双精度 | 若有该字段 |
| MPI | `Intel MPI` | 若模板提供选项；同时记录具体版本 |
| 软件版本 | `ls-dyna_R15_mmp_d` | 西北一区（西安）；本次固定为 R15 MPP 双精度 cohort |
| 节点数 | `1` | P0 单节点资格运行；当前截图中的 `4` 必须改为 `1` |
| 每节点核数 | 优先 `32` | 选择西安资源后重新查看节点总核数；若节点至少 32 核，P0 先用 32 核 |
| 总 tasks/ranks | `32` | 单节点 32 个 MPP ranks |
| CPUs per task | `1` | 若有该字段 |
| 运行时限 | `04:00:00` | 当前截图中的 72 小时必须改为 4 小时；实际提前完成即停止计费 |
| GPU | `0` / 不申请 | 本作业为 CPU MPP |
| 独占节点 | 关闭/留空 | `--exclusive` 可能按整节点计费 |
| 工作目录 | `/work/home/acd7dt5yv4/LSDYNA/uav_hf_p0_plate_r04` | 新建 R04 目录；不要覆盖 R01/R02/R03 证据 |
| 输入文件 | `p0_velocity_mesh_calibration.k` | 只选 `.k` |
| 重启动/续算 | 关闭 | 这是全新 P0 作业 |
| 命令行参数 | **完全清空** | 必须删除截图中来自旧作业的 `r=/work/.../d3dump01`；`r=d3dump01` 是 LS-DYNA 小重启动参数，本作业没有对应的初始计算 |
| 头内存占用 | `512 MB` | 当前西安模板配置；用于 MPP 分解/0 号进程，本次 163,730 单元显式 P0 可接受，运行后以 `d3hsp` 实耗复核 |
| 子内存占用 | `256 MB` | 当前西安模板配置；32 ranks 时合计静态上限约 8.45 GB，低于页面分配的 111.5625 GB |
| 自动填充上次参数 | **关闭** | 防止旧作业的重启动路径、资源规模和内存再次写回表单 |

### 队列选择原则

1. 使用账号已购买且支持 LS-DYNA MPP 的 CPU 共享队列。
2. 点开队列“详情”，记录：资源中心、队列名、单节点总核数、内存、核存比和页面单价。
3. 选择 `ls-dyna_R15_mmp_d` 后使用西安 CPU 共享队列。先查看西安节点规格；单节点至少 32 核时填 32 核。若平台只允许整节点或强制独占，则先停止提交并记录页面说明。
4. 不选 GPU 队列，不为这个 16.4 万单元 P0 直接申请多节点。
5. 若页面显示预估费用，以页面显示为准。

共享队列公开计费公式为“单价 × 申请核心数 × 实际运行时间”；独占参数会覆盖申请核心数并按节点全部计费：[SCNet 计费规则](https://www1.scnet.cn/help/docs/mainsite/hpc/advanced-guide/billing-rule/)。

### 算力上限

| 配置 | 4 小时墙钟上限对应的最大核时 |
|---|---:|
| 1 节点 × 32 核 | 128 core-hours |
| 强制整节点 64 核 | 256 core-hours |

截图中的 `4 节点 × 60 核 × 72 小时` 对应最大 `17,280 core-hours`，是本次建议上限 `128 core-hours` 的 135 倍，不适合作为首次资格运行。

实际费用还取决于页面单价与实际运行时间。不要在没有错误证据的情况下扩大到多节点。

---

## 5. 点击提交前的最终核对

逐项确认：

- [ ] 模板明确为 LS-DYNA；
- [ ] 求解方式为 MPP，精度为 double；
- [ ] 1 节点，32 核，4 小时；
- [ ] 未启用 GPU；
- [ ] 未启用独占节点；
- [ ] 工作目录属于西北一区（西安），并包含同一批 6 个上传文件；
- [ ] 输入文件名是 `p0_velocity_mesh_calibration.k`；
- [ ] 命令行参数为空，不含 `r=`、`d3dump` 或任何旧作业路径；
- [ ] 头内存为 512 MB，子内存为 256 MB；若日志出现 `OTH+21` 或明确内存不足，再依据 `d3hsp` 提示提高，不盲目重提；
- [ ] “自动填充上次参数”已关闭；
- [ ] 页面输入文件大小与本地 20,604,806 字节一致，首行是 `*KEYWORD LONG=Y`，第 14 行末尾为 `...,0,0,0`；
- [ ] 页面显示的单价和预估费用可以接受；
- [ ] 保存或截图完整提交参数。

若任何一项不一致，先不要点击提交。求解器固定使用 `ls-dyna_R15_mmp_d`，不要仅凭 `d_R15` 中的 `d` 推定该版本同时属于 MPP。

---

## 6. 提交后立即记录

提交成功后，在 **高性能计算 → 作业管理** 中记录：

```text
job_id=
job_name=uav_hf_p0_plate_r04
resource_center=
queue=
solver_version=
solver_build=
precision=double
mpi_implementation=
nodes=1
ranks=
walltime_limit=04:00:00
unit_price=
submission_time=
work_directory=
```

同时保存两张截图：

1. 提交表单的完整参数；
2. 作业管理中的 Job ID、状态和资源配置。

作业状态可能依次出现排队、运行、完成或失败；SCNet 的作业管理页可查看日志、错误信息和工作目录：[作业管理说明](https://www.scnet.cn/help/docs/mainsite/hpc/beginner-guide/job-management/index.html)。

---

## 7. 运行后的判定与回传

### 7.1 若作业数分钟内失败

不要直接改核数或重复提交。先下载：

```text
Slurm 标准输出
Slurm 标准错误
d3hsp
messag 或 mes*
p0_velocity_mesh_calibration.k
manifest.json
```

把这些文件和 Job ID 回传，先用实际输入 deck 的 SHA-256 绑定作业身份，再判断关键词、许可证、MPI、内存、工作目录或版本问题。

### 7.2 若正常完成

从作业管理进入工作目录，下载整个目录或至少下载：

```text
p0_velocity_mesh_calibration.k
p0_case_part_map.csv
manifest.json
d3hsp
messag
mes*
glstat
matsum
sleout
nodout
d3plot*
slurm-*.out
slurm-*.err
```

不要重命名或覆盖原始文件。建议平台打包下载为：

```text
uav_hf_p0_plate_r04_<JOBID>.zip
```

SCNet 支持从作业管理直接进入工作目录下载，多个文件会打包：[结果下载说明](https://www.scnet.cn/help/docs/mainsite/hpc/beginner-guide/download-file/)。

### 7.3 回传给本项目

把结果 ZIP 放到：

```text
D:\Desktop\UAV新\scnet_lsdyna_plate_calibration_20261017\results\<JOBID>_<solver-cohort>\
```

然后提供 Job ID 和提交参数截图。后处理将依次执行：身份哈希核对、正常终止检查、能量与小时玻璃能检查、负体积/侵蚀审计、三个网格层级一致性、五个速度点穿孔与剩余速度提取。P0 未通过前不要提交 24 个 P1 工况。

---

## 8. 严禁修改的内容

- 不修改 `.k` 文件内容；
- 不增加质量缩放；
- 不把结束时间从 0.02 ms 改成其他值；
- 不把 MPP 改为 SMP；
- 不用 ZIP、CSV 或 JSON 代替 `.k` 作为输入；
- 不用当前 KMFC 标签分布选择材料参数；
- 不复用其他任务的输出、可执行文件路径或工作目录；
- 不在 P0 通过前批量提交 P1。

