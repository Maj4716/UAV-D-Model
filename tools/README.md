# 工具入口

本目录只保留可重复执行、仍服务于当前软件数据链路的命令行工具。所有命令均从项目根目录运行；输入应显式指向 `configs/` 或待验证的数据包，输出写入用户指定的实验目录。

## 配置与基础验证

- `sync_schema_mirrors.py`：校验并同步包内、根目录两套 JSON Schema 镜像。
- `build_loitering_fragment_only_plan.py`：从基础装甲目标配置派生当前破片专用目标、标签、案例与采样合同。
- `benchmark_engine_hotpath.py`、`benchmark_online_statistics.py`、`benchmark_plan_identity.py`：分别检查引擎热路径、在线统计和采样计划身份稳定性。

## 采样设计与代理评估

- `evaluate_normalized_sampling_strategy.py`：生成并评估归一化接触采样方案。
- `evaluate_fast_sampling_proxy.py`：对显式传入的候选池执行快速代理筛选；`--proposal-input` 为必填项。
- `refine_fast_sampling_proxy_resolution.py`、`evaluate_refined_sampling_selection.py`：细化代理分辨率并复核冻结选择；候选池同样必须显式提供。
- `fit_sampling_risk_model.py`、`replay_v3_proxy_tradeoffs.py`、`replay_v4_risk_tradeoffs.py`：拟合风险模型并重放冻结的代理/风险权衡。
- `diagnose_teacher_selection_bias.py`：检查教师选择造成的输入偏置与覆盖偏移。
- `estimate_sampling_plan_storage.py`、`plan_reference_evaluation.py`：估算存储量并规划参考评估。

## 物理与统计诊断

- `evaluate_multicondition_penetration.py`、`sweep_penetration_parameters.py`：执行多工况侵彻诊断与参数扫描。
- `run_replicate_convergence_study.py`：评估重复数与统计收敛性。
- `audit_fragment_reachability.py`、`audit_loitering_impact_dataset.py`：审计破片可达性与巡飞弹数据分布。

## 数据包审计

- `validate_neural_sampling_dataset.py`：验证当前神经网络开发数据包的哈希、张量重算、分组切分与来源臂覆盖。
- `validate_uav_control_dataset.py`：验证旧格式 UAV 控制数据合同；仅用于读取需要迁移的外部数据包。
- `audit_dataset_distribution.py`、`audit_dataset_training_readiness.py`、`audit_system_learning_contract.py`、`build_dataset_health_report.py`：生成分布、训练就绪性、学习合同和健康状态报告。

当前已保留的开发数据包位于 `artifacts/development_dataset/`。该目录是本地生成产物，不作为配置源；重新生成实验时应使用新的输出目录，避免覆盖既有证据。
