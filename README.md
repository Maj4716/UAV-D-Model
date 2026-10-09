# Damage Dataset Engine

面向破片/冲击波毁伤评估与神经网络教师数据生产的概率仿真内核。当前版本为 `0.4.0`。它已经形成可安装、可复现、可分片批处理的 **L1 工程内核**，并具备进入 L2 参数标定的接口；仓库内 `examples/physics.demo.json` 及 `raw_data/import_config.json` 中的响应参数仍是 `unverified` 演示/占位值，因此输出不能解释为真实装备毁伤效能。

## 已实现的计算链

- 破片：质量/初速/方向联合记录或参数源采样 → 空气阻力 → BVH 宽相位 → 外装甲逐层侵彻 → 串列内部部件命中与残速续追 → 状态概率。
- 冲击波：比例距离查表 → 入射/反射载荷 → 装甲 P-I 失效 → 开孔面积混合透射 → 部件 P-I 状态概率。
- 耦合：`sequential` 允许破片造成的装甲完整度影响后续冲击波透射；部件层保留条件独立点估计，同时输出 Fréchet 并集上下界，避免在相关证据不足时把独立假设当成事实。
- 系统评价：版本化毁伤树支持 `independent` 和 `fully_dependent` 门，并可把输出节点组织成多个命名评价标准视图。

## 安装与验证

建议先建立隔离环境，再使用已验证的软件基线：

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

`requirements.txt`（并保留同内容的 `requirements-lock.txt`）固化了本仓库软件验证所用版本；PyTorch 是可选依赖，应按实际 CPU/CUDA 环境单独安装。

四份运行时 JSON Schema 随 `damage_engine.schemas` 一并打包，并通过 `importlib.resources` 读取；校验不依赖源码树旁的顶层 `schemas` 目录。

## 快速开始

```powershell
python -m damage_engine hash-target --target examples/target.demo.json
python -m damage_engine validate `
  --target examples/target.demo.json `
  --case examples/case.demo.json `
  --physics examples/physics.demo.json
python -m damage_engine simulate `
  --target examples/target.demo.json `
  --case examples/case.demo.json `
  --physics examples/physics.demo.json `
  --output result.demo.json
python -m damage_engine generate `
  --target examples/target.demo.json `
  --cases examples/cases.demo.jsonl `
  --physics examples/physics.demo.json `
  --output-directory dataset_demo `
  --workers 4
```

目标几何可使用显式 `*_m` 或 `*_cm` 字段，但同一目标禁止混用。载入后统一规范化为 SI；`hash-target` 输出规范化后的内容哈希，案例中的 `target_ref.content_hash` 必须引用该哈希。

### 从 raw_data 重建目标

`List.csv` 的重复毁伤列名按列位置解析，`armor.csv` 的两个截面按同名对象重建为拉伸多边形：

```powershell
python -m damage_engine import-raw `
  --components raw_data/List.csv `
  --armor raw_data/armor.csv `
  --config raw_data/import_config.json `
  --output raw_data/target.rebuilt.json `
  --diagnostics raw_data/import_report.json `
  --skip-incomplete-components
```

当前导入结果为 67/68 个部件、2 个装甲体；缺少 XYZ 的“无线电台”被隔离并记录，不会默认放在原点。原始文件没有提供装甲厚度、易损率和材料响应，因此 `import_config.json` 中的 10 mm/0.2 等值只是明示的 `unverified` 占位参数。原始毁伤码阈值的状态映射也是临时 codebook；获得正式准则后应更新配置并重新导入。`--strict-components` 可用于生产放行时对任何不完整几何 fail closed。

## 输入与评价标准

外装甲支持拉伸多边形的顶点、拉伸方向/距离、等效厚度、易损面积比例及冲击波响应；内部部件支持 box、cylinder、extruded polygon、位置、完整 `xyzw` 姿态、尺寸、厚度、易损率及 P-I 阈值。

如需同一毁伤树输出多个评价口径，可在 `damage_tree.standards` 中声明：

```json
{
  "standards": [
    {"id": "mission-effectiveness", "outputs": ["mission_loss", "mission_degraded"]},
    {"id": "mission-loss-only", "outputs": ["mission_loss"]}
  ]
}
```

案例的 `analysis.evaluation_standard_ids` 选择需要返回的视图。未声明 `standards` 时，为兼容旧输入，毁伤树 `version` 自动成为唯一标准 ID，并包含全部 `outputs`。

`warhead.fragment_source.fragment_list_ref` 可引用 UTF-8 CSV/JSON/JSONL 离散联合源项。每条记录需要 `mass_kg`、`direction_body_xyz`、`initial_velocity_mps`，可选 `id` 和 `multiplicity`；CSV 也可将方向写成独立 x/y/z 列。加载器会检查有限值、正质量/速度、单位方向、重复 ID、展开数量和金属质量守恒，并在 provenance 中记录文件 SHA-256。

除概率输出的 `adaptive_ci_half_width` 外，`analysis.adaptive_physical_ci` 可对连续物理量分别设置绝对/相对半宽，且必须与 `replicates >= 32` 同时使用：

```json
{
  "adaptive_physical_ci": {
    "component.blast.incident_peak_pressure_pa": {
      "absolute_half_width": 5000.0,
      "relative_half_width": 0.05,
      "relative_floor": 1000.0
    },
    "component.fragment.deposited_energy_j": {
      "absolute_half_width": 2.0
    }
  }
}
```

停止条件要求所有已申明的概率门和物理量门同时满足；不支持的 selector 或被禁用的机理会被拒绝。

## 主要输出

- 部件级破片、冲击波和综合状态概率；
- 命中数、穿透数、着速、侵彻裕度和能量沉积；
- 入射/有效峰值超压、冲量、正相持续时间、到达时间和遮蔽厚度；
- 装甲破片后/综合完整度、反射载荷及冲击波失效概率；
- 部件两机制条件独立并集点估计及 Fréchet 并集上下界；
- 毁伤树全部节点、输出节点及所请求评价标准视图；
- Monte Carlo 均值、标准差、Student-t/正态 95% 区间和收敛元数据；
- JSONL 审计结果、Parquet 长表、NPZ 张量和带哈希的 manifest。

## 张量接口 1.3.0

`tensors.npz` 的核心字段如下：

- `case_x [B,59]`：交汇、战斗部、源项统计、方向、阻力、环境、机理、耦合、保真度与采样控制；
- `component_x [B,C,38]`：形状、位置、完整旋转矩阵、尺寸、厚度、易损率、P-I 阈值和暴露类别；
- `armor_x [B,A,18]`：装甲几何、拉伸、面积/周长/顶点数与响应参数；
- `component_armor_path_x [B,C,A,4]`：遮挡、入射距离、路径长度和入射余弦；
- `component_y [B,C,3,S]`：破片/冲击波/综合 × 部件毁伤状态；
- `component_frechet_y [B,C,S,2]`：两机制并集的 lower/upper 界，仅两机制同时启用时 mask 有效；
- `armor_y [B,A,13]`：装甲完整度、入射/反射峰值与冲量、持续/到达时间、入射余弦及失效概率；
- `component_aux_y [B,C,14]`：可解释的破片与冲击波中间量；
- `system_y [B,O]`：毁伤树输出节点；
- 对应的标准差、95% 区间、feature/label mask、名称、单位、split/group/leakage key、随机种子和实际重复数；
- 拉伸多边形顶点 CSR 数组及数值化毁伤树边。

禁用机理对应标签使用 mask 标记为不可用，不把人工零值当训练标签。分组切分优先级为 `split_group → condition_cluster → physical_case_hash`，防止同一物理工况的 Monte Carlo 重复跨 train/validation/test 泄漏。

PyTorch 可选适配器：

```python
from torch.utils.data import DataLoader
from damage_engine.torch_dataset import (
    TorchDamageDataset,
    TorchMultiTargetDataset,
    TorchShardedDamageDataset,
    collate_damage_batch,
)

dataset = TorchDamageDataset("dataset_demo", split="train")
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

sharded = TorchShardedDamageDataset(
    "dataset_sharded", split="train", max_cached_shards=2
)
multi_target = TorchMultiTargetDataset([dataset, sharded])
multi_loader = DataLoader(
    multi_target,
    batch_size=16,
    collate_fn=collate_damage_batch,
    num_workers=4,
)
```

`collate_damage_batch` 会按名称对齐状态、评价标准和系统输出，padding 部件/装甲/路径轴并生成 presence mask，同时将多目标图合成不相交批图。`TorchShardedDamageDataset` 会校验根 manifest、分片 manifest 及所有分区哈希，默认拒绝未通过全局 split-leakage 审计的数据集，惰性打开分片并用 LRU 限制同时常驻数量。缓存不进入 pickle，用于 Windows `spawn` 工作进程重新打开本地分片。

### 分片生成与恢复

```powershell
python -m damage_engine generate `
  --target raw_data/target.rebuilt.json `
  --cases raw_data/cases.smoke.jsonl `
  --physics examples/physics.demo.json `
  --output-directory raw_data/dataset_smoke `
  --workers 4 `
  --shard-size 1024

# 中断后必须使用相同顺序的案例、workers/chunksize、失败策略和分片配置
python -m damage_engine generate `
  --target raw_data/target.rebuilt.json `
  --cases raw_data/cases.smoke.jsonl `
  --physics examples/physics.demo.json `
  --output-directory raw_data/dataset_smoke `
  --workers 4 `
  --shard-size 1024 `
  --resume
```

每个分片在临时目录中完成后才原子改名；`checkpoint.json` 记录有序 case hash、分片 manifest hash 和执行配置，完成态数据集不允许用 `--resume` 追加。逻辑 `dataset_id` 包含引擎代码、物理注册表、目标、有序案例和张量契约，但不受 worker 数、压缩或 shard size 影响。

Parquet 是跨目标按列读取的主接口，NPZ 是目标内稠密张量快照。分片使毁伤结果、张量和 Parquet 行的峰值内存受 `shard_size` 限制，但全局 case ID/hash、split 审计元数据和 checkpoint 仍随案例数 O(N) 增长，静态拓扑文件也会在各分片重复。当前还没有多写者锁，同一输出目录必须只有一个生成进程。百万级案例需进一步把全局索引迁移到 SQLite/分片索引并减少 checkpoint 重写。

## 性能复测

```powershell
python -m benchmarks.benchmark_engine --cases 40 --replicates 4 --warmup 2
python -m benchmarks.benchmark_geometry
```

热点优化包括拉伸多边形预计算/标量 Möller–Trumbore 求交、静态目标预编译、BVH 宽相位和确定性多进程批处理。本次重建的 69 对象目标上，20,000 条随机射线的 BVH 候选集与暴力/向量参考完全一致，最终复测为 `0.1889 s` 对 `0.4514 s`，约 `2.389×`。该数字只是当前机器和该几何的微基准；正式性能结论应绑定硬件、Python/依赖版本、目标复杂度和案例集。

## 可信度边界

- `L3` 会强制要求物理注册表为 `validated`、目标参数为 `validated`、来源为 `approved`；演示配置会被拒绝。
- 超出注册模型适用域时结构化拒答，不隐式外推。
- `fully_dependent` 是完全正相关的可审计包络，不是经试验拟合的相关系数。
- 目标级峰值、冲量和持续时间阈值会覆盖装甲 `failed` 状态或部件首个非完好状态的 P-I 尺度；更高毁伤状态仍使用物理注册表中的分级参数。
- 张量接口已提供 `armor_y` 和 Fréchet 界，但它们仍是同一套未标定教师模型的输出，不是实验真值。
- 破片偏转/跳弹/二次破片/孔洞重叠以及冲击波完整时程/多路径/泄压网络未经标定模型实现；不得将经验降阶结果宣称为 CFD/FEM 结果。
- 当前环境未安装真实 PyTorch，只验证了结构、pickle 和轻量替身；目标 CPU/CUDA 环境的 Windows `spawn` 吞吐/内存仍是放行项。
- 软件测试通过只证明实现行为与当前规格一致，不证明演示物理参数或模型已被实物试验确认。

实现背景见 `01_仿真引擎工程实现.md`；基线评估见 `02_仿真引擎机理与性能评估整改报告.docx`；本轮 raw 数据重建、风险闭环和验证证据见 `03_原始装甲数据重建与风险整改报告.md`。
