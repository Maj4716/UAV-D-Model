# 各向异性快速评估方法

## 1. 方法定位

本方法把数据集生成器、轻量几何代理和仿真引擎拆成三个可单独执行与判决的证据层。默认命令只执行前两层，不构建正式数据集、不运行随机教师、不读取封存测试结果。L1 完全不调用引擎；L2 不运行随机教师，但复用目标几何、空气阻力和侵彻响应，因此属于流程隔离与残差定位，不是外部独立物理模型。

项目已有的 `FastSamplingProxy` 是当前各向异性核：它把 Taylor/Taylor-Shapiro 环形方向分布压缩成带权确定性射线包，复用目标射线求交、空气阻力和侵彻响应。这里的“各向异性球”指围绕遭遇条件叠加的方向性能力场，不新增会忽略方向、遮挡和装甲的各向同性衰减球。

三级结构如下：

1. **L1 生成器层**：检查候选工况、单元质量、潜变量覆盖、采样权重和选择足迹。
2. **L2 各向异性场层**：计算部件期望到达量、损伤能量、有限帧可达状态、代理适用域和分单元包络。
3. **L3 引擎校准层**：只在显式提供已有教师证据时，计算代理—教师差距和 false-unreached 门。L3 缺省状态为 `not_evaluated`。

三级判决互不替代。生成器通过不表示代理完整，代理通过也不表示仿真引擎已获得物理验证。

当前默认配置还把来向五区域与三弹型的声明等质量设计（15 个单元各 `1/15`）同 S-CONTACT 后的实现帧比较。该 TV 距离量化的是“声明的预接触设计测度→接触条件化实现测度”的变化，用于定位生成器/条件化环节，不等同于作战发生率误差。配置中的数值门为诊断门，尚未作为研究结论门预注册。

## 2. 快速入口

### 2.1 从工况文件运行 L1+L2

```powershell
python tools/run_anisotropic_fast_assessment.py quick `
  --cases path/to/cases.jsonl.gz `
  --config configs/sampling/anisotropic_fast_assessment.v1.json `
  --target configs/targets/target.fragment_only.v1.json `
  --physics configs/physics/loitering_impact.taylor_shapiro.obliquity_bound.v3.json `
  --splits train,validation `
  --max-cases 4096 `
  --output-dir artifacts/validation/anisotropic_fast_run
```

`--cases` 支持 JSON、JSONL 和 JSONL.GZ。JSON 可以是单个工况、工况列表、含 `cases` 的对象，或含 `candidate_ids` 与 `candidate_cases` 的候选池。每行可声明 `candidate_id`、`cell_id` 和 `measure_weight`。

分割身份依次读取 `metadata.frozen_split` 与 `metadata.sampling.frozen_split`。默认保留 train、validation 和未声明分割的前验工况，始终排除 test；报告中的 `test_case_count_included` 固定为 0。

### 2.2 使用预计算暴露行

```powershell
python tools/run_anisotropic_fast_assessment.py quick `
  --cases path/to/cases.json `
  --exposure-rows path/to/exposure_rows.json `
  --output-dir artifacts/validation/anisotropic_fast_precomputed_rows
```

`--exposure-rows` 绕过 `FastSamplingProxy`，用于复算已冻结的 L1/L2 统计。文件可以是行列表，或包含 `candidate_rows`、`exposure_rows`、`rows` 之一的对象。

### 2.3 数秒复用现有 X1 NPZ

```powershell
python tools/run_anisotropic_fast_assessment.py quick `
  --precomputed-npz artifacts/experiments/X1_X2_20261120_frozen_v2/x1_candidate_exposure.npz `
  --max-cases 4096 `
  --output-dir artifacts/validation/anisotropic_fast_x1
```

NPZ 路径与 `--cases` 二选一。冻结 X1 缓存只用于方法近似性回放；其源码身份与当前版本不同，不能替代当前 live 验证。入口读取：

- `candidate_ids`、`cell_ids`、`latent`；
- `component_ids`、`component_lambda`、`component_damage_energy_j`；
- `proxy_domain_status`。

当 `--max-cases` 小于原始规模时，入口按 `cell_ids` 进行固定种子的比例分层抽样。`manifest.json` 记录原始规模、使用规模、抽样规则和索引哈希；不会简单截取候选前缀。配置也支持 `equal_cell_stratified`。

### 2.4 比较一个已冻结的采样选择

`--selection` 接收：

```json
{
  "selected_candidate_ids": ["case-001", "case-017"],
  "inclusion_probabilities": {
    "case-001": 0.2,
    "case-017": 0.1
  }
}
```

选择概率必须位于 `(0, 1]`，且所有选择工况都必须存在于本次参考帧。L2 使用逆纳入概率 Hájek 权重报告恢复误差和未加权采样足迹偏移。

## 3. 引擎校准入口

### 3.1 导入已有教师行

```powershell
python tools/run_anisotropic_fast_assessment.py calibrate `
  --assessment artifacts/validation/anisotropic_fast_run/assessment.json `
  --teacher-evidence path/to/teacher_rows.json `
  --output-dir artifacts/validation/anisotropic_fast_calibration
```

教师证据可以是行列表，或包含 `teacher_rows`、`rows` 之一的对象。每行至少包含 `candidate_id` 和 `component_hit_probability`。该命令没有现场教师执行选项。

### 3.2 角色、完整性与身份要求

`calibrate` 只接受 `--teacher-evidence`。教师行需要带有明确的非测试角色；混合 train/validation 时，应使用 `--split-role development_train_validation`，顶层角色必须覆盖所有行角色。没有可信角色清单的 NPZ 可以用于 L1/L2 历史回放，不能直接进入 L3。

若教师证据包含 `report_hash`，入口会先复算并核对；同时拒绝 test/sealed-test 行，以及元数据中任何非零 test 解码/打开计数或已计算 test 指标。结果衡量代理相对输入软件教师证据的差距。只有目标、物理注册表和引擎代码身份全部匹配时，才能称为当前同源比较。

## 4. 输出合同

`quick --output-dir DIR` 严格拒绝覆盖已有目录，并生成：

- `assessment.json`：三级机器可读结果；L3 默认 `not_evaluated`；
- `REPORT.md`：中文摘要和三个独立判决；
- `manifest.json`：输入、分割、抽样与输出清单；
- `hashes.json`：输入和输出 SHA-256；
- `COMMAND.txt`：可重放命令。

`calibrate` 生成同类证据文件，主结果名为 `engine_calibration.json`。quick 与 calibrate 都只接受新的 `--output-dir`，并严格拒绝覆盖。

## 5. Python 公共接口

```python
from damage_engine.fast_assessment import (
    compare_engine_evidence,
    evaluate_fast_assessment,
)

assessment = evaluate_fast_assessment(case_rows, proxy=fast_proxy, config=config)
comparison = compare_engine_evidence(
    assessment,
    teacher_rows,
    split_role="development_train_validation",
    config=config,
)
```

两个接口分别承担 outcome-blind L1+L2 与已有证据 L3，避免一个高参数入口同时触发生成、教师和校准。

## 6. 解释规则

- `proxy_unreached_in_declared_frame` 表示当前有限候选帧和代理分辨率没有到达；它不是结构零。
- Reference 与 Coverage 是两种测度。富集臂的未加权分布用于描述训练足迹，带设计权重的结果用于恢复声明的 Reference 量。
- L1 失败优先修复候选生成、条件化或选择概率。
- 单元样本数或秩门失败时，只使用全局椭球；稀疏单元椭球不作为稳定协方差结论。
- L2 失败优先修复方向核、几何遮挡、代理分辨率或适用域。
- L3 失败定位代理与软件教师之间的差距；应按 region×profile 分层优化代理或引擎接口。
- 真实作战频率和物理准确率需要独立实验或现场参照。本工具不从软件内部一致性推导该结论。

## 7. 复用关系

统一入口只组合现有模块：

- `damage_engine.fast_sampling_proxy.FastSamplingProxy`：各向异性 Taylor 环射线核；
- `damage_engine.exposure_field`：部件暴露场、Hájek 聚合和采样足迹；
- `damage_engine.exposure_audit`：教师对代理非到达的反证；
- `damage_engine.multifidelity`：配对控制变量与方差缩减；
- `damage_engine.fast_assessment`：三级组合合同。

正式标签、完整数据集构建、训练和封存测试保持在原有链路中，不由本入口触发。
