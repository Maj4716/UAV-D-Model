# review_20261123：REPORT.md 独立审查的复核脚本与结果

审查意见正文见上级目录 [`REVIEW_20261123.md`](../REVIEW_20261123.md)。本目录只放复核脚本和它们的机器可读输出。所有脚本只读取 pre-outcome 输入和 train/validation 教师结果，不解码 sealed test 结果。

| 脚本 | 作用 | 输出 |
|---|---|---|
| `check_taylor_tilt.py` | 用 Taylor（1941）稳态流推导独立判断破片倾斜方向，并与实现对比 | `check_taylor_tilt.json` |
| `check_aim_window_basis.py` | 把 4,301 个候选的 latent 还原为瞄准点，按 region×瞄准带上下侧统计 S-CONTACT 准入率 | `aim_window_basis.json` |
| `run_proxy_variants.py` | 在 2,300 个开发工况上重跑 FastSamplingProxy：v0 默认、v1 换 seed、v2 不压缩方位包 | `proxy_variants.npz` |
| `analyze_review_checks.py` | 复现报告 §8.1，做根因归因、子集组成、噪声底、零质量、false-unreached、微观 SE、Lambert 基线、基线比较 | `review_checks.json` |
| `proxy_v2_full_scope.py` | 在全开发集、teacher-complete、validation 三个范围比较 v0/v2 的 Brier、域标志、部件覆盖数 | `proxy_v2_full_scope.json` |
| `decompose_tree_gap.py` | 把毁伤树差距分解为“输入状态偏差”和“聚合次序（Jensen）效应” | `tree_gap_decomposition.json` |
| `tree_with_v2_states.py` | 把 v2 代理状态代入毁伤树，看各输出差距收窄多少 | `tree_with_v2_states.json` |
| `label_heterogeneity_by_aim_side.py` | 检验瞄准带上下侧是否解释 region×profile 标签异质性 | `label_heterogeneity_by_aim_side.json` |
| `taylor_sign_sensitivity.py` | 用代理做反事实：翻转 Shapiro 角符号后部件命中格局变化多少 | `taylor_sign_sensitivity.json` |

复现顺序（项目根目录执行；读取 parquet 的脚本需使用生成数据时的 conda 环境）：

```bash
python artifacts/validation/pipeline_expected_actual_20261122/review_20261123/check_taylor_tilt.py
```

```bash
/d/ProgramData/Anaconda/envs/pytorch312-lmdt/python.exe artifacts/validation/pipeline_expected_actual_20261122/review_20261123/check_aim_window_basis.py
```

```bash
python artifacts/validation/pipeline_expected_actual_20261122/review_20261123/run_proxy_variants.py
```

```bash
python artifacts/validation/pipeline_expected_actual_20261122/review_20261123/analyze_review_checks.py
```

```bash
python artifacts/validation/pipeline_expected_actual_20261122/review_20261123/proxy_v2_full_scope.py
```

```bash
python artifacts/validation/pipeline_expected_actual_20261122/review_20261123/decompose_tree_gap.py
```

```bash
python artifacts/validation/pipeline_expected_actual_20261122/review_20261123/tree_with_v2_states.py
```

```bash
python artifacts/validation/pipeline_expected_actual_20261122/review_20261123/label_heterogeneity_by_aim_side.py
```

```bash
python artifacts/validation/pipeline_expected_actual_20261122/review_20261123/taylor_sign_sensitivity.py
```
