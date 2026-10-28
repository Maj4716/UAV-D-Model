# 只提交这个 R06f Wave 0 作业

本目录是本轮唯一提交入口。只上传：

- `uav_hf_r06f_0800_ctrl.zip`

解压后在平台中选择 `r06f_0800_ctrl.k` 作为 head/input deck。请使用新的工作目录，
不要复用 R06e 工作目录，也不要同时提交 TSSFAC=0.4、2000 m/s 或 treatment。

## SCNet 参数

```text
job_name=uav_hf_r06f_0800_c
work_directory=/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06f_0800_c
software=ls-dyna_R15_mmp_d
nodes=1
ranks=32
walltime=04:00:00
head_memory=512MB
child_memory=256MB
command_line_parameters=<empty>
auto_fill_previous_parameters=off
head_deck=r06f_0800_ctrl.k
```

## 必须完整下载的回传

- submitted `.k`、solver `o=`/`d3hsp`、`mes*`；
- `glstat`、`matsum`、`sleout`、`nodout`、`binout*`；
- 完整 `d3plot*`、`d3part*`；
- stdout/stderr、Slurm 脚本和资源记录；
- 若意外产生 `intfor*` 或 `ncforc*`，也一并保留。

`intfor*` 不再是强制回传：R15.0.2 已用 Warning 20340 证明侵蚀接触不提供
binary INTFOR。`NCFORC` 仍被 best-effort 请求，但缺失不阻断本轮，且绝不能填成 0。

## 本轮通过条件

1. 无 KEY 错误，越过 cycle 0，32 ranks 正常结束；
2. run-integrity 通过，且无质量缩放；
3. D3PART 捕获账本可读；完整 D3PLOT HVAR6 可读并正确使用 alive mask；
4. 相对 Job 72397333，门前诊断 `qM/qP/qE` 绝对差均不超过 0.01，`qV` 相对差不超过 0.05；
5. missing、sentinel 和删除后归零均按 null/reason 处理，不伪装为有效零值。

即使本轮通过，下一步也只允许生成并提交 `800 m/s, TSSFAC=0.4` 单因素诊断，
不会直接释放 2000 m/s 或 treatment，也不会把本作业写入 P0/P1、引擎拟合或参考训练标签。

## ZIP 内原始说明

```text
SCNet LS-DYNA P0 R06 diagnostic job

STATUS
  Prepared and statically validated only after static_validation.json reports PASS.
  NOT submitted, NOT solver-validated, NOT admitted for P0/P1 or training labels.

PAIR / FACTOR
  experiment_id=r06f_recanary_800_measurement_contract
  factor=measurement_contract_revision_only
  role=measurement_repair_recanary_control
  termination_time_ms=0.04
  plate_only_psfail=false

SOLVER INPUT (select this file only as the LS-DYNA head/input file)
  r06f_0800_ctrl.k

EXPECTED SOLVER INPUT IDENTITY
  size_bytes=2557676
  sha256:111911562b703b8af1e96810b08edd8c31fbeea739ddddd2921b8bcd07c517af
  first_line=*KEYWORD LONG=Y
  package_revision=R06_measurement_contract_repair_20261027
  job_id=r06f_0800_ctrl

SCNET SETTINGS
  job_name=uav_hf_r06f_0800_c
  work_directory=/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06f_0800_c
  software=ls-dyna_R15_mmp_d
  nodes=1
  ranks=32
  walltime=04:00:00
  head_memory=512MB
  child_memory=256MB
  command_line_parameters=<empty; R06f does not request unsupported binary INTFOR>
  auto_fill_previous_parameters=off

UPLOAD SEQUENCE
  wave_0=r06f_0800_ctrl
  Follow the frozen gates in the packaged design. Do not advance to a later wave
  until the preceding parser/runtime/output-readability gate passes.

REQUIRED RETURN
  Required: submitted .k, solver o= output or d3hsp, mes*, glstat, matsum, sleout, nodout, binout*, d3plot*, d3part*, solver stdout/stderr, Slurm script and resource record.
  Optional but preserve if present: intfor*, ncforc*.
  Do not rename or merge job directories. A missing optional direct-contact file
  is unavailable evidence, never a zero response.

BOUNDARY
  This job follows the frozen design and submission wave. A static PASS does not
  prove target-solver parsing, output readability, numerical quality, response
  availability, physical validity, P0 admission, or dataset-label admission.
```
