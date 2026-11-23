# Reproduction scripts

Run from this folder with the project's Python 3.13 environment. All scripts read
only train/validation rows; sealed-test result lines are skipped before JSON
decoding by `tools.evaluate_proxy_teacher_distribution.load_development_evidence`.
Set `PYTHONIOENCODING=utf-8` on Windows consoles (GBK cannot print some table glyphs).

| order | script | what it does | output |
|---:|---|---|---|
| 0 | `snapshot_baseline.py` | full proxy outputs of the **unmodified** module (40 cases × v0/v1/v2/exposure); run once, before editing the proxy | `../baseline/pre_edit_outputs.json` |
| 1 | `common.py` | decode development evidence once; cache case inputs and teacher arrays | `../work/development_cases.json.gz`, `../work/teacher_arrays.npz`, `../work/source_audit.json` |
| 1b | `extract_teacher_tree.py` | cache teacher damage-tree blocks | `../work/teacher_damage_tree_raw.json.gz` |
| 2 | `check_backward_compat.py` | every pre-edit output key reproduced bit-for-bit by the edited module (default options) | `../results/backward_compat_check.json` |
| 3 | `run_sweep.py NAME …` | proxy-only sweep over 2,300 cases (14 worker processes, `SWEEP_WORKERS`); `--list` prints configurations | `../results/runs/NAME.npz/.json` |
| 4 | `check_reproduction.py` | `az4`/`az4_s7777777`/`azall` equal review v0/v1/v2 arrays bit-for-bit (gate oracle) | `../results/reproduction_check.json` |
| 4b | `check_final_module_consistency.py [N]` | runs made before the domain-censoring patch are bit-identical under the final module | `../results/final_module_consistency.json` |
| 5 | `analyze_sweep.py` | convergence + teacher-agreement metrics for every run | `../results/convergence_metrics.json`, `../results/convergence_tables.md` |
| 6 | `analyze_domain.py` | OOD mass fraction as a continuous covariate; proxy refused contacts vs teacher unresolved impacts | `../results/domain_metrics.json` |
| 7 | `analyze_residuals.py` | Brier(K)=a+b/K extrapolation; worst components of the preset | `../results/residual_metrics.json` |
| 8 | `evaluate_gate.py [POSTHOC …]` | pre-declared train-R8 gate (`azall`) plus post-hoc runs | `../results/gate_evaluation.json` |
| 9 | `benchmark_runtime.py [N]` | clean single-thread runtime per configuration (N cases per profile) | `../results/runtime_benchmark.json` |
| 10 | `build_report_tables.py` | assembles the REPORT tables | `../results/report_tables.md` |
| — | `check_teacher_equivalence.py [N]` | re-run the full teacher on N train cases per profile against archived rows; bitwise equality needs the archive's environment (Python 3.12.13 / numpy 2.4.6, `pytorch312-lmdt`) | `../results/teacher_equivalence_check.json` (py312/py313 copies alongside) |
| — | `check_environment_sensitivity.py [RUN]` | re-evaluate a stored run under the current interpreter, compare and recompute the gate | `../results/environment_sensitivity_RUN_pyXY.json` |
| — | `_apply_branch_censoring_patch.py` | record of the anchored patch that added `domain_censoring` to the proxy (already applied; refuses to re-apply) | — |

Exact sequence used for the report:

```bash
python snapshot_baseline.py            # before any module edit
python common.py && python extract_teacher_tree.py
python check_backward_compat.py
python run_sweep.py az4 azall
python check_reproduction.py
python run_sweep.py az2 az8 az16 azall_q4 azall_q64 azall_cut0 az4_s7777777 az4_s2026112301 az4_s2026112302 azall_s7777777 azall_s2026112301 azall_s2026112302 az4_reg azall_reg azall_k2 azall_k4 az4_k8 azall_k4_reg azall_k8 azall_k8_reg azall_k16 azall_k16_reg
python run_sweep.py azall_dom azall_reg_dom azall_k4_reg_dom azall_k8_reg_dom azall_k16_reg_dom
python check_final_module_consistency.py 60
python analyze_sweep.py && python analyze_domain.py && python analyze_residuals.py
python evaluate_gate.py azall_reg_dom azall_k8_reg_dom
python benchmark_runtime.py 20
python build_report_tables.py
python check_teacher_equivalence.py 4
```

`work/` holds regenerable caches (decoded inputs and teacher arrays); delete and
rebuild with `common.py` and `extract_teacher_tree.py`.
