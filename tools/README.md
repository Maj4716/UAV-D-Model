# 工具入口

本目录只保留可重复执行、仍服务于当前软件数据链路的命令行工具。所有命令均从项目根目录运行；输入应显式指向 `configs/` 或待验证的数据包，输出写入用户指定的实验目录。

## 配置与基础验证

- `sync_schema_mirrors.py`：校验并同步包内、根目录两套 JSON Schema 镜像。
- `build_loitering_fragment_only_plan.py`：从基础装甲目标配置派生当前破片专用目标、标签、案例与采样合同。
- `benchmark_engine_hotpath.py`、`benchmark_online_statistics.py`、`benchmark_plan_identity.py`：分别检查引擎热路径、在线统计和采样计划身份稳定性。

## 采样设计与代理评估

- `run_anisotropic_fast_assessment.py`：三级各向异性快评入口；`quick` 独立执行生成器 L1 与暴露代理 L2，`calibrate` 只读取带角色的既有非测试教师证据执行 L3，不构建正式数据集或现场运行教师。
- 显式 scrambled Sobol 或 maximin LHS 设计需要先安装 `python -m pip install -e ".[sampling]"`；旧 SamplingSpec 的 randomized-LHS 默认路径不加载 SciPy。
- `evaluate_sphere_entry_sampling.py`：执行旧 6D 横截面方案、球面重锚 6D 配对控制与 5D 物理射线方案的几何/教师等价门，分别生成 G1 和 B1/B2 报告。
- `evaluate_sphere_entry_designs.py`：按冻结的 outcome-blind 规则比较 randomized LHS、scrambled Sobol 与 maximin LHS，生成 B3 联合设计诊断报告；该工具只运行快速几何代理，不调用教师。
- `evaluate_normalized_sampling_strategy.py`：生成并评估归一化接触采样方案。
- `package_sampler_dataset_v2.py`：把已完成的 sampler pilot 确定性打包为树无关 dataset-v2；该完整归档入口会机械解码全部 split，适用于已获准打开 test 的发布阶段。
- `package_sampler_dataset_v2_development.py`：生成严格封存的开发包；保留 test 输入、权重和 split 身份，核心标签与部件轨迹使用全掩码占位，test 结果行保持零 JSON 解码。模型冻结前的训练与验证使用此入口。
- `audit_dataset_v2_development.py`：只审计严格开发包的 train/validation，报告识别区间、ε-完整率、逐部件正例、条件目标覆盖、ESS、region×profile 分层和 Holm 校正 KS 缺失诊断；test 结果行在 JSON 解码前跳过。
- `freeze_dataset_v2_task_admission.py`：根据开发审计中的 train/validation 支持门冻结各任务可训练部件；概率头使用正负样本门，条件回归头使用有效观测数门；校验数据集与审计哈希，记录逐部件拒绝原因，并拒绝任何已开启或已解码 test 的输入。
- `run_dataset_v2_hurdle_baseline.py`：按 25%/50%/100% 与三随机种子运行跨栏多任务学习曲线；输入归一化和可逆目标编码只在 train 拟合，模型选择只读取 Reference validation。
- `run_proxy_learning_ablation.py`：在严格封存的 dataset-v2 开发包上，以相同教师子集、随机种子和微调预算比较 teacher-only、67 维代理特征、教师减代理残差、代理预训练后教师微调四臂；test 真标签保持零解码、零评价。
- `evaluate_fast_sampling_proxy.py`：对显式传入的候选池执行快速代理筛选；`--proposal-input` 为必填项。
- `refine_fast_sampling_proxy_resolution.py`、`evaluate_refined_sampling_selection.py`：细化代理分辨率并复核冻结选择；候选池同样必须显式提供。
- `fit_sampling_risk_model.py`、`replay_v3_proxy_tradeoffs.py`、`replay_v4_risk_tradeoffs.py`：拟合风险模型并重放冻结的代理/风险权衡。
- `diagnose_teacher_selection_bias.py`：检查教师选择造成的输入偏置与覆盖偏移。
- `estimate_sampling_plan_storage.py`、`plan_reference_evaluation.py`：估算存储量并规划参考评估。

## 物理与统计诊断

- `evaluate_multicondition_penetration.py`、`sweep_penetration_parameters.py`：执行多工况侵彻诊断与参数扫描。
- `run_replicate_convergence_study.py`：评估重复数与统计收敛性。
- `audit_fragment_reachability.py`、`audit_loitering_impact_dataset.py`：审计破片可达性与巡飞弹数据分布。
- `run_stage2_mechanism_experiments.py`：执行 idea2 E3 2^4 因子敲除、E5 σ×τ/同方差方向噪声形状和 C2 Gurney/角度/V50 金标检查；默认规模可用 `--dry-run` 先绑定。

## 数据包审计

- `validate_neural_sampling_dataset.py`：验证当前神经网络开发数据包的哈希、张量重算、分组切分与来源臂覆盖。
- `validate_uav_control_dataset.py`：验证旧格式 UAV 控制数据合同；仅用于读取需要迁移的外部数据包。
- `audit_dataset_distribution.py`、`audit_dataset_training_readiness.py`、`audit_system_learning_contract.py`、`build_dataset_health_report.py`：生成分布、训练就绪性、学习合同和健康状态报告。

当前已保留的开发数据包位于 `artifacts/development_dataset/`。该目录是本地生成产物，不作为配置源；重新生成实验时应使用新的输出目录，避免覆盖既有证据。
