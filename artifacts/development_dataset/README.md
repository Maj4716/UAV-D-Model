# 思路一混合采样神经网络开发数据集

这是本轮采样优化的最终开发数据包。它把分层均匀参考臂与思路一覆盖臂合并、按候选样本去重，并保留每次抽样成员关系和一阶纳入概率。

## 数据规模

- 8 个源采样臂，每臂严格 2,000 条，共 16,000 条抽样观测。
- 去重后 10,589 个候选样本。
- 训练集 7,965 条，其中 5,482 条具有完整响应标签。
- 验证集 2,624 条，其中 1,855 条具有完整响应标签。
- 完整响应标签共 7,337 条；域删失样本 3,252 条。
- 当前没有封存测试集。

## 文件

- `tensors.npz`：训练张量、掩码、分组 ID 和划分代码。
- `feature_schema.json`：输入、部件、毁伤树和汇总标签顺序。
- `cases.jsonl.gz`：完整输入、代理结果、教师结果及抽样成员关系。
- `selection_observations.jsonl.gz`：16,000 条抽样观测及纳入概率、原始设计权重。
- `dataset_manifest.json`：来源、计数、质量检查和核心文件 SHA-256。
- `validation_report.json`：独立重算式校验结果。
- `pytorch_dataset.py`：最小 PyTorch 读取器。

## 输入张量

- `x`：项目既有 `loitering_impact_contact_pose@1.0.0` 契约，15 维物理输入。
- `x_normalized`：仅用训练集均值和总体标准差标准化后的 `x`；推荐作为首轮模型输入。
- `x_design`：15 维采样设计坐标，含剖面/方向独热、4 个连续分位坐标和滚转角正余弦；适合采样诊断或辅助输入。
- `x_train_mean`、`x_train_scale`：训练集标准化统计量。

`feature_schema.json` 给出每一列的确切名称。不要把 `x_design` 当成物理输入契约的同义字段。

## 标签与掩码

- `domain_complete`：所有样本均可用的域完整性二分类标签。
- `response_mask`：是否允许使用响应标签；与 `domain_complete` 一致。
- `component_support`：67 个部件的正支持标签。
- `component_hits`、`component_penetrations`、`component_energy_j`、`component_severity`：67 个部件的连续响应。
- `tree_outputs`：7 个毁伤树输出。
- `summary_targets`：5 个样本级汇总响应。

当 `response_mask=False` 时，所有响应目标均为 `NaN`。训练时只对 `response_mask=True` 的行计算响应损失；域删失行只监督域完整性头。把这些 `NaN` 填成零会制造假负样本。

## PyTorch 读取

```python
from pathlib import Path
from pytorch_dataset import Idea1HybridDataset

root = Path("artifacts/development_dataset")
train = Idea1HybridDataset(root, split="train", feature_tensor="x_normalized")
validation = Idea1HybridDataset(root, split="validation", feature_tensor="x_normalized")

batch = train[0]
mask = batch["response_mask"]
```

推荐模型结构是一个域完整性头，加上在 `response_mask` 下训练的部件支持头、部件连续响应头、毁伤树头和汇总响应头。各头分别按有效元素数归一化损失，避免 67 维部件头压倒其他任务。

## 抽样权重

去重样本可能同时属于参考臂和思路一臂，因此不存在一个天然唯一的“合并后权重”。推荐先用去重后的混合集做无权重表征学习，再进行两类敏感性分析：

1. 仅用分层均匀参考臂及其权重评估设计分布指标。
2. 在 `selection_observations.jsonl.gz` 的臂内使用 `design_weight_raw=1/inclusion_probability` 复算加权指标。

不要把多个臂的原始权重直接相加后用作单一样本权重。

## 划分与用途

训练集按设计种子 `2026111641/61/81` 分组，验证集使用独立设计种子 `2026111721`，候选 ID 无交叉。验证集结果已经用于本轮最终比较，因此它适合开发期回归检查，不是未来论文结果的封存测试集。

这些标签来自 R=4 软件教师，用于网络结构、损失函数、数据管线和采样策略开发。该数据包没有封存测试集；确认性评价应生成独立且未参与开发的数据版本。

## 复核命令

```powershell
D:\ProgramData\Anaconda\envs\pytorch312-lmdt\python.exe tools\validate_neural_sampling_dataset.py `
  artifacts\development_dataset `
  --output artifacts\development_dataset\validation_report.json
```
