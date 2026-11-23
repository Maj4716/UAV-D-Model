# dataset-v2 CLI 最小流程

`dataset-v2` 的 core 不包含树输出；树候选只写入内容寻址的 `views/tree_<hash>.npz`。以下命令不改变仿真引擎的默认树。

模型冻结前，已完成的 sampler pilot 使用严格封存开发包。train/validation
结果必须含轻量 `component_state_trace`；test 结果保持在源 pilot 中，打包器
不解码其 JSON，并在 core 中写入全掩码占位：

```powershell
python tools/package_sampler_dataset_v2_development.py `
  --pilot artifacts/experiments/F1_taylor_shapiro_continuous_20261120/pilot `
  --output artifacts/experiments/F1_taylor_shapiro_continuous_20261120/dataset_v2_development

python tools/audit_dataset_v2_development.py `
  --dataset artifacts/experiments/F1_taylor_shapiro_continuous_20261120/dataset_v2_development `
  --pilot artifacts/experiments/F1_taylor_shapiro_continuous_20261120/pilot `
  --epsilon 0.01 `
  --output artifacts/experiments/F1_taylor_shapiro_continuous_20261120/development_audit.json

python tools/run_dataset_v2_hurdle_baseline.py `
  --dataset artifacts/experiments/F1_taylor_shapiro_continuous_20261120/dataset_v2_development `
  --output artifacts/experiments/F1_taylor_shapiro_continuous_20261120/hurdle_learning

python tools/run_proxy_learning_ablation.py `
  --dataset artifacts/experiments/F1_taylor_shapiro_continuous_20261120/dataset_v2_development `
  --pilot artifacts/experiments/F1_taylor_shapiro_continuous_20261120/pilot `
  --target configs/targets/target.fragment_only.v1.json `
  --physics configs/physics/loitering_impact.taylor_shapiro.v2.json `
  --output artifacts/experiments/F1_taylor_shapiro_continuous_20261120/proxy_learning_ablation.json `
  --proxy-pool-size 512 `
  --x1-gate-status failed
```

训练命令只在 Train 拟合归一化与可逆目标编码器，只用 Reference validation
早停和选模，并保持 `sealed_test.json.test_open_count=0` 与
`test_result_json_lines_decoded=0`。冻结模型、阈值和一次性 test 协议后，再用
`package_sampler_dataset_v2.py` 执行完整发布归档。

代理学习消融只使用 train 教师标签和 validation 指标。独立 Sobol 代理池不调用
教师；X1 教师反证门未通过时，报告固定为 `diagnostic_only_not_promoted`，不改变
正式标签或最终采样臂。

```powershell
python tools/build_dataset_v2.py pack `
  --inputs work/inputs.npz `
  --component-labels work/component_labels.npz `
  --sampling-weights work/sampling_weights.npz `
  --component-state-traces work/results.jsonl.gz `
  --input-contract configs/input_contract.json `
  --dataset-id uav-dataset-v2-demo `
  --label-spec-id component-labels-v2 `
  --source-code-hash sha256:<代码哈希> `
  --status provisional `
  --provisional-reason "awaiting domain semantic review" `
  --output work/dataset_v2

python tools/build_dataset_v2.py add-tree-view `
  --dataset work/dataset_v2 `
  --candidate examples/offline_tree_candidate.demo.json

python tools/validate_dataset_v2.py work/dataset_v2
```

没有领域审定时，从真实部件 ID 清单生成 identity fallback。该命令只生成逐状态透传候选，不建立 23 个未引用部件的功能语义：

```powershell
python tools/build_dataset_v2.py identity-fallback `
  --components work/component_ids.json `
  --output work/identity_fallback.json
```

开发 seed 与验证 seed 必须来自独立设计，并在查看输出前填完 `artifacts/validation/tree/PREREGISTRATION_TREE_V1.md`：

```powershell
python tools/select_offline_tree_candidate.py `
  --dataset work/dataset_v2 `
  --candidate work/candidate_a.json `
  --candidate work/candidate_b.json `
  --development-seed 41001 `
  --validation-seed 41002 `
  --preregistration-id TREE-2026-001 `
  --output work/tree_selection_report.json
```

输入 NPZ 的固定字段：

- `inputs.npz`：`case_ids`, `x`, `feature_names`，可选 `input_mask`；
- `component_labels.npz`：`case_ids`, `component_ids`, `p_hit`, `p_hit_std_error`, `p_hit_ci95`, `p_degraded`, `p_failed`, `label_mask`；
- `sampling_weights.npz`：`case_ids`, `selection_probability`, `design_weight`, `target_measure_weight`, `training_weight`, `split`, `design_seed`。
