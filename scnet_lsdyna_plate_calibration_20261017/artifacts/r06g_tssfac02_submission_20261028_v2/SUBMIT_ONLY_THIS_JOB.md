# 只提交这个 R06g TSSFAC=0.2 细化作业

本目录是本轮唯一提交入口。只上传：

- `uav_hf_r06g_0800_tssfac02.zip`

解压后在平台中选择 `r06g_0800_tssfac02.k` 作为 head/input deck。必须使用新的工作目录，
不要复用 Job 72630855、Job 72617857 或其他 R06 目录，也不要同时提交 2000 m/s
或 treatment。

## SCNet 参数

```text
job_name=uav_hf_r06g_0800_ts02
work_directory=/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06g_0800_ts02
software=ls-dyna_R15_mmp_d
nodes=1
ranks=32
walltime=04:00:00
head_memory=512MB
child_memory=256MB
command_line_parameters=<empty>
auto_fill_previous_parameters=off
head_deck=r06g_0800_tssfac02.k
```

## 开跑前资源握手

必须看到 `-N 1`、`--ntasks-per-node=32`、32 total cores、`mpirun -np 32`；
求解器启动后还应出现 `MPP execution with 32 procs`。任一不符就停止该作业。

## 必须完整下载的回传

- submitted `.k`、solver `o=`/`d3hsp`、`mes*`；
- `glstat`、`matsum`、`sleout`、`nodout`、`binout*`；
- 完整 `d3plot*`、`d3part*`；
- stdout/stderr、Slurm 脚本和资源记录；
- 若产生独立 `ncforc*` 或意外 `intfor*`，也一并保留。

## 本轮因子和门禁

- 唯一因子：`CONTROL_TIMESTEP.TSSFAC` 从 0.4 改为 0.2；
- `DT2MS=0`，不得启用质量缩放；
- 外部控制：Job 72630855；
- `qM/qP/qE` 绝对差分别不超过 0.01；
- deleted-upstream mass fraction 绝对差不超过 0.01；
- qV 仅在两边均可用时比较 5% 相对差；若两边均为零确认穿越质量，qV 才可记为不适用；
- 还必须通过正常终止、运行完整性、D3PART/HVAR6、NCFORC 和能量检查。

本包静态验证通过，但尚未经过平台 LS-DYNA 求解。它不能用于 P0/P1、引擎拟合、
物理有效性结论或参考数据集标签。

## ZIP 内原始说明

```text
SCNet LS-DYNA P0 R06 diagnostic job

STATUS
  Prepared and statically validated only after static_validation.json reports PASS.
  NOT submitted, NOT solver-validated, NOT admitted for P0/P1 or training labels.

PAIR / FACTOR
  experiment_id=r06g_timestep_refinement_800
  factor=control_timestep_tssfac
  role=no_mass_scaling_timestep_refinement
  termination_time_ms=0.04
  plate_only_psfail=false
  tssfac=0.2

SOLVER INPUT (select this file only as the LS-DYNA head/input file)
  r06g_0800_tssfac02.k

EXPECTED SOLVER INPUT IDENTITY
  size_bytes=2557651
  sha256:b5f4da367e9ef6eaae7098e47a84610eb55c51760cc9bf8dbb4721693b7a732a
  first_line=*KEYWORD LONG=Y
  package_revision=R06_timestep_refinement_20261028
  job_id=r06g_0800_tssfac02

SCNET SETTINGS
  job_name=uav_hf_r06g_0800_ts02
  work_directory=/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06g_0800_ts02
  software=ls-dyna_R15_mmp_d
  nodes=1
  ranks=32
  walltime=04:00:00
  head_memory=512MB
  child_memory=256MB
  command_line_parameters=<empty; R06g does not request unsupported binary INTFOR>
  auto_fill_previous_parameters=off

UPLOAD SEQUENCE
  wave_0=r06g_0800_tssfac02
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
