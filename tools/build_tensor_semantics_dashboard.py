from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "raw_data" / "dataset_5000_software_demo_v2"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "22_全张量语义与代码定位可视化.html"


ROLE_LABELS = {
    "feature": "输入特征",
    "label": "标签/响应",
    "uncertainty": "不确定性",
    "mask": "可用性掩码",
    "index": "索引",
    "metadata": "元数据",
}


BASE_SEMANTICS = {
    "case_x": "兼容性案例向量：物理输入与求解/采集控制的合并表示。新合同仍可为教师审计保存它，但不应绕过版本化 model_input_contract 进入网络。",
    "case_feature_mask": "case_x 每一维是否在本案例中有定义；false 表示结构缺失或不适用，不能把对应数值 0 当成物理零。",
    "component_x": "目标部件的几何、姿态、材料响应阈值和暴露类别，共 38 维；当前数据只有一个 flight-control 部件。",
    "component_feature_mask": "component_x 各维是否可用；与部件特征同时进入需要处理缺失值的模型。",
    "armor_x": "装甲板几何、挤出方向、尺寸和响应阈值，共 18 维；当前数据只有一个 armor-front。",
    "armor_feature_mask": "armor_x 各维是否可用。",
    "armor_y": "装甲完整度与冲击波载荷响应的条件蒙特卡洛均值，共 13 个量；当前 LabelSpec 未把它声明为正式任务。",
    "armor_label_mask": "armor_y 中每个指标是否由本案例启用的物理机理产生。",
    "component_armor_path_x": "起爆点到部件路径与装甲的几何关系：是否相交、入射距离、装甲内路径长度、入射角余弦。",
    "component_armor_path_mask": "路径特征是否可定义；未相交时部分几何量不应被当作有效零值。",
    "component_y": "部件在 fragment、blast、combined 三种机理下达到 degraded/failed 状态的软概率均值。",
    "component_label_mask": "component_y 对应机理和状态是否可用；combined 需要 fragment 与 blast 均启用。",
    "component_frechet_y": "破片与冲击波联合损伤概率在未知依赖结构下的 Fréchet 下界/上界。它是联合概率边界，不是第三种物理标签。",
    "component_frechet_label_mask": "Fréchet 联合界是否可用；需要 fragment 与 blast 同时启用。",
    "component_aux_y": "部件级机理诊断量：命中/穿透计数、到达速度、穿透裕度、沉积能量以及冲击波压力、冲量、持续时间和遮蔽量。",
    "component_aux_label_mask": "component_aux_y 每个机理诊断量是否可用。",
    "system_y": "损伤树输出的系统事件概率。当前两列依次为 mission_degraded 与 mission_loss，是非互斥软标签。",
    "system_label_mask": "系统标签是否既被当前评估标准请求、又能从已启用机理的叶节点计算。训练 loss 必须使用该掩码。",
    "system_source_available_mask": "不考虑评估标准请求时，损伤树输出的全部源叶节点是否可用。用于区分“源缺失”和“未请求”。",
    "requested_standard_mask": "案例请求了哪些评估标准；当前轴只有 demo-tree-1。",
    "case_ids": "成功案例的稳定标识，与 B 轴一一对应。",
    "case_hashes": "规范化案例内容哈希；用于追溯输入身份和发现完全相同案例。",
    "split_names": "案例所属 train、validation 或 test 拆分。",
    "split_group_ids": "分组拆分单元标识；同组不得跨拆分。",
    "split_group_sources": "split_group_ids 的来源类型，说明分组依据。",
    "leakage_keys": "防止相同物理输入跨拆分泄漏的精确键。",
    "aleatory_seeds": "每个案例的条件随机性种子；复现实例内蒙特卡洛轨迹。",
    "epistemic_sample_ids": "参数认识不确定性样本标识；当前每个案例只有一个参数实现，不能解释为完整认识不确定性覆盖。",
    "replicate_counts": "每个案例实际完成的条件蒙特卡洛重复数；具体 R 由冻结的 replicate_policy 与角色覆盖项决定。",
    "input_case_ids": "原始输入顺序中的案例标识，包含成功和拒绝案例；与 N 轴对应。",
    "input_case_status": "每条原始输入的执行状态。",
    "input_case_valid_mask": "原始输入是否成功进入 B 轴正式张量。",
    "input_to_valid_index": "N 轴输入映射到 B 轴成功案例的索引；拒绝案例使用 -1。",
    "component_ids": "C 轴部件名称表。",
    "armor_ids": "A 轴装甲名称表。",
    "mechanism_names": "M 轴机理名称表：fragment、blast、combined。",
    "state_names": "S 轴有损状态名称表：degraded、failed；intact 未作为独立监督列。",
    "frechet_bound_names": "D 轴 Fréchet 边界名称：lower、upper。",
    "frechet_bound_units": "Fréchet 边界单位表，均为无量纲概率。",
    "system_output_names": "O 轴系统事件名称表，决定 system_y 的列顺序。",
    "evaluation_standard_ids": "E_standard 轴评估标准名称表。",
    "standard_output_mask": "每个评估标准包含哪些系统输出。",
    "case_feature_names": "case_x 的 59 个列名。",
    "case_feature_units": "case_x 的逐列单位。",
    "component_feature_names": "component_x 的 38 个列名。",
    "component_feature_units": "component_x 的逐列单位。",
    "armor_feature_names": "armor_x 的 18 个列名。",
    "armor_feature_units": "armor_x 的逐列单位。",
    "armor_label_names": "armor_y 的 13 个响应名称。",
    "armor_label_units": "armor_y 的逐响应单位。",
    "armor_path_feature_names": "component_armor_path_x 的 4 个列名。",
    "armor_path_feature_units": "路径特征的逐列单位。",
    "component_aux_label_names": "component_aux_y 的 14 个诊断量名称。",
    "component_aux_label_units": "部件诊断量的逐列单位。",
    "component_material_ids": "每个部件引用的材料标识。",
    "component_type_ids": "每个部件的类型标识。",
    "fragment_response_model_ids": "每个部件使用的破片响应模型标识。",
    "blast_response_model_ids": "每个部件使用的冲击波响应模型标识。",
    "armor_material_ids": "每块装甲引用的材料标识。",
    "armor_fragment_response_model_ids": "每块装甲使用的破片响应模型标识。",
    "armor_blast_response_model_ids": "每块装甲使用的冲击波响应模型标识。",
    "armor_vertices_m": "所有装甲多边形顶点的扁平表，单位米；由 armor_vertex_offsets 分段。",
    "armor_vertex_offsets": "装甲顶点的前缀偏移，长度 A+1；第 i 块装甲顶点位于 offsets[i]:offsets[i+1]。",
    "component_vertices_m": "挤出多边形部件的顶点扁平表；当前部件是 box，因此 Vc=0。",
    "component_vertex_offsets": "部件顶点的前缀偏移，长度 C+1。",
    "graph_node_ids": "固定目标图的节点标识，包含部件状态叶节点和损伤树节点。",
    "graph_node_types": "每个图节点的类型。",
    "tree_edge_index": "COO 图边索引，形状 [2,E_edge]；第一行源节点、第二行目标节点。",
    "tree_edge_type": "每条边的类型整数编码。",
    "tree_edge_type_names": "边类型编码表：部件状态到叶节点、损伤树依赖。",
    "case_physics_x": "教师仿真的完整物理工况输入。它是旧合同的默认输入；在 tensor 1.8 UAV-only 归档中仅保留作教师审计/兼容，默认 loader 不暴露。",
    "case_physics_feature_mask": "case_physics_x 每一维是否存在；fixed_initial_velocity_mps 当前全部缺失。",
    "solver_context_x": "11 维求解器/采集控制：启用机理、耦合模式、保真度、重复数和自适应停止阈值。它不应伪装成部署侧物理输入。",
    "solver_context_feature_mask": "solver_context_x 各维是否可用。",
    "case_physics_feature_names": "case_physics_x 的 48 个列名。",
    "case_physics_feature_units": "case_physics_x 的逐列单位。",
    "solver_context_feature_names": "solver_context_x 的 11 个列名。",
    "solver_context_feature_units": "solver_context_x 的逐列单位。",
    "uav_state_x": "tensor 1.8 的 10 维 UAV-only 原始模型输入：相对位置 3、相对速度 3、规范化并取逆后的相对姿态四元数 4。弹体、目标、环境和求解器参数不在该张量中。",
    "uav_state_feature_mask": "uav_state_x 的逐维可用性掩码；当前合同要求 10 维全部存在且有限。",
    "uav_state_feature_names": "uav_state_x 的 10 个版本化列名。",
    "uav_state_feature_units": "uav_state_x 的逐列单位。",
    "uav_state_feature_source_paths": "每个 UAV 输入维度追溯到 simulation case 的原始 JSON 路径。",
    "uav_state_feature_transforms": "从教师案例坐标约定转换到 UAV 相对固定威胁坐标约定的逐维变换。",
    "uav_state_feature_semantics": "每个 UAV 输入维度的版本化自然语言语义。",
    "sampling_selection_probability": "正式案例在当前采样设计下的入选概率 rho。",
    "sampling_design_weight_raw": "原始设计权重 1/rho，用于设计分布估计；不是类别权重，也不是 minibatch 抽样概率。",
    "sampling_weight_mask": "当前案例的设计权重是否适用于参考风险估计。",
    "sampling_round": "案例被选入的采样轮次。",
    "sampling_stratum_index": "案例所属采样层的整数索引，由 sampling_stratum_names 解码。",
    "sampling_role_index": "案例采样角色的整数索引，由 sampling_role_names 解码。",
    "sampling_stratum_names": "Q 轴采样层名称表。",
    "sampling_role_names": "R 轴采样角色名称表。",
    "sampling_candidate_ids": "成功案例对应的候选标识。",
    "sampling_lineage_ids": "成功案例的采样谱系标识，用于阻止同谱系跨拆分。",
    "sampling_epistemic_group_ids": "成功案例的认识不确定性分组标识。",
    "input_sampling_candidate_ids": "N 轴全部输入的候选标识。",
    "input_sampling_lineage_ids": "N 轴全部输入的采样谱系标识。",
    "input_sampling_roles": "N 轴全部输入的文本采样角色。",
    "input_sampling_epistemic_group_ids": "N 轴全部输入的认识不确定性分组标识。",
    "input_frozen_splits": "标签执行前已冻结的 N 轴拆分分配。",
    "label_task_names": "T 轴正式学习任务名称。当前只有 system_events。",
    "label_task_available_mask": "每个案例的正式任务是否有完整可用标签。",
}


USE_OVERRIDES = {
    "case_x": "旧接口兼容；不建议作为当前默认模型输入",
    "case_physics_x": "旧合同模型输入；tensor 1.8 UAV-only 归档仅用于教师审计/兼容",
    "uav_state_x": "tensor 1.8 默认模型输入；训练视图把 quaternion4 转为 rotation-6D6 并只用 Train 拟合归一化",
    "solver_context_x": "求解/采集配置；用于审计或求解器代理边界，不属于端到端部署输入",
    "component_x": "固定目标条件或多目标模型的部件节点特征",
    "armor_x": "装甲节点/条件特征",
    "component_armor_path_x": "求解器代理可选派生输入；部署时必须能独立计算",
    "system_y": "当前 LabelSpec 唯一正式监督目标",
    "component_y": "部件级辅助监督/诊断；当前未列入正式 LabelSpec",
    "component_frechet_y": "依赖假设敏感性与稳健性诊断",
    "component_aux_y": "机理分阶段建模、事件条件评估和错误诊断",
    "armor_y": "装甲响应辅助监督/诊断；当前未列入正式 LabelSpec",
    "sampling_design_weight_raw": "参考分布风险估计；禁止直接当类别权重或抽样概率",
    "tree_edge_index": "图模型拓扑；当前所有案例共享同一目标图",
    "system_label_mask": "正式 loss 的标签有效性门控",
    "system_source_available_mask": "区分损伤树源不可用与评估标准未请求",
}


FEATURE_SEMANTICS = {
    "uav_position_relative_to_fixed_threat_x_m": "冻结场景坐标系中，无人机相对固定威胁/起爆原点的 X 位置",
    "uav_position_relative_to_fixed_threat_y_m": "冻结场景坐标系中，无人机相对固定威胁/起爆原点的 Y 位置",
    "uav_position_relative_to_fixed_threat_z_m": "冻结场景坐标系中，无人机相对固定威胁/起爆原点的 Z 位置",
    "uav_velocity_relative_to_fixed_threat_x_mps": "冻结威胁平移参考系中，无人机相对速度的 X 分量",
    "uav_velocity_relative_to_fixed_threat_y_mps": "冻结威胁平移参考系中，无人机相对速度的 Y 分量",
    "uav_velocity_relative_to_fixed_threat_z_mps": "冻结威胁平移参考系中，无人机相对速度的 Z 分量",
    "uav_attitude_relative_to_fixed_threat_qx": "规范化、取逆并固定符号后的无人机相对威胁姿态四元数 x 分量",
    "uav_attitude_relative_to_fixed_threat_qy": "规范化、取逆并固定符号后的无人机相对威胁姿态四元数 y 分量",
    "uav_attitude_relative_to_fixed_threat_qz": "规范化、取逆并固定符号后的无人机相对威胁姿态四元数 z 分量",
    "uav_attitude_relative_to_fixed_threat_qw": "规范化、取逆并固定符号后的无人机相对威胁姿态四元数 w 分量",
    "detonation_x_m": "目标坐标系下起爆点 X 坐标",
    "detonation_y_m": "目标坐标系下起爆点 Y 坐标",
    "detonation_z_m": "目标坐标系下起爆点 Z 坐标",
    "detonation_time_s": "起爆时刻",
    "orientation_qx": "弹体到目标坐标系姿态四元数 x 分量",
    "orientation_qy": "弹体到目标坐标系姿态四元数 y 分量",
    "orientation_qz": "弹体到目标坐标系姿态四元数 z 分量",
    "orientation_qw": "弹体到目标坐标系姿态四元数 w 分量",
    "velocity_x_mps": "目标坐标系相对速度 X 分量",
    "velocity_y_mps": "目标坐标系相对速度 Y 分量",
    "velocity_z_mps": "目标坐标系相对速度 Z 分量",
    "charge_mass_kg": "装药质量",
    "casing_mass_kg": "壳体质量",
    "blast_equivalent_mass_kg": "冲击波等效装药质量",
    "fragment_total_count": "生成的破片总数",
    "fragment_mass_mean_kg": "破片质量分布均值",
    "fragment_mass_stddev_kg": "破片质量分布标准差",
    "fragment_mass_q10_kg": "破片质量 10% 分位数",
    "fragment_mass_q50_kg": "破片质量中位数",
    "fragment_mass_q90_kg": "破片质量 90% 分位数",
    "fragment_density_kg_m3": "破片材料密度",
    "fixed_initial_velocity_mps": "固定初速度；当前 Gurney 模型下结构不适用",
    "gurney_constant_mps": "Gurney 速度常数",
    "detonation_velocity_mps": "爆速",
    "direction_axis_body_x": "破片锥轴在弹体坐标系的 X 分量",
    "direction_axis_body_y": "破片锥轴在弹体坐标系的 Y 分量",
    "direction_axis_body_z": "破片锥轴在弹体坐标系的 Z 分量",
    "direction_half_angle_deg": "破片方向锥半角",
    "direction_spread_sigma_deg": "方向附加散布标准差",
    "drag_coefficient": "破片常阻力系数 Cd",
    "area_shape_factor": "破片迎风面积形状因子",
    "air_density_kg_m3": "环境空气密度",
    "ambient_pressure_pa": "环境压力",
    "temperature_k": "环境温度",
    "requested_replicates": "请求的条件蒙特卡洛重复数",
    "adaptive_ci_half_width": "自适应停止的置信区间半宽阈值",
    "center_x_m": "几何中心 X 坐标",
    "center_y_m": "几何中心 Y 坐标",
    "center_z_m": "几何中心 Z 坐标",
    "extent_x_m": "轴对齐包围盒 X 尺寸",
    "extent_y_m": "轴对齐包围盒 Y 尺寸",
    "extent_z_m": "轴对齐包围盒 Z 尺寸",
    "principal_axis_x": "几何主轴 X 分量",
    "principal_axis_y": "几何主轴 Y 分量",
    "principal_axis_z": "几何主轴 Z 分量",
    "box_size_l_m": "盒体长度",
    "box_size_w_m": "盒体宽度",
    "box_size_h_m": "盒体高度",
    "cylinder_radius_m": "圆柱半径",
    "cylinder_height_m": "圆柱高度",
    "extrusion_direction_x": "挤出方向 X 分量",
    "extrusion_direction_y": "挤出方向 Y 分量",
    "extrusion_direction_z": "挤出方向 Z 分量",
    "extrusion_distance_m": "多边形挤出距离",
    "polygon_area_m2": "截面多边形面积",
    "polygon_perimeter_m": "截面多边形周长",
    "vertex_count": "截面顶点数量",
    "equivalent_thickness_m": "部件或装甲的等效厚度",
    "vulnerable_area_ratio": "有效易损面积比例",
    "blast_peak_threshold_pa": "冲击波峰值压力响应阈值",
    "blast_impulse_threshold_pa_s": "冲击波冲量响应阈值；未给出时掩码为 false",
    "blast_duration_threshold_s": "冲击波正相持续时间响应阈值；未给出时掩码为 false",
    "intersects_path": "起爆点到部件的射线路径是否穿过该装甲",
    "entry_distance_m": "从起爆点到装甲入射点的距离",
    "path_length_m": "射线在装甲内部的路径长度",
    "incidence_cosine": "射线与装甲表面法向夹角余弦",
    "post_fragment_integrity": "破片作用后的装甲完整度，1 为完整、0 为完全失效",
    "post_combined_integrity": "破片与冲击波顺序耦合后的装甲完整度",
    "blast_distance_m": "起爆点到响应对象的距离",
    "blast_incident_peak_pressure_pa": "自由场入射峰值超压",
    "blast_incident_impulse_pa_s": "自由场入射冲量",
    "blast_reflected_peak_pressure_pa": "装甲表面反射峰值超压",
    "blast_reflected_impulse_pa_s": "装甲表面反射冲量",
    "blast_effective_peak_pressure_pa": "考虑遮蔽/传递后的部件有效峰值压力",
    "blast_effective_impulse_pa_s": "考虑遮蔽/传递后的部件有效冲量",
    "blast_positive_duration_s": "冲击波正相持续时间",
    "blast_arrival_time_s": "冲击波到达时间",
    "blast_incidence_cosine": "冲击波方向与装甲法向夹角余弦",
    "blast_failure_probability": "装甲冲击波失效概率",
    "blast_pre_integrity": "冲击波作用前装甲完整度",
    "blast_post_integrity": "冲击波作用后装甲完整度",
    "fragment_hits": "命中部件的破片数",
    "fragment_penetrations": "穿透并作用于部件的破片数",
    "fragment_mean_arrival_speed_mps": "命中破片平均到达速度",
    "fragment_mean_penetration_margin": "平均穿透裕度；正值表示超过弹道极限",
    "fragment_deposited_energy_j": "破片向部件沉积的总能量",
    "blast_shielding_effective_thickness_m": "冲击波传播路径上的有效遮蔽厚度",
    "blast_shielding_panel_count": "冲击波传播路径上的遮蔽板数量",
    "mission_degraded": "综合失效或冲击波降级导致的任务降级事件概率",
    "mission_loss": "飞控部件综合 failed 的标准投影，即任务损失事件概率",
    "lower": "Fréchet 联合概率下界 max(p_fragment,p_blast)",
    "upper": "Fréchet 联合概率上界 min(1,p_fragment+p_blast)",
    "degraded": "部件达到或超过降级状态的累积概率",
    "failed": "部件达到失效状态的概率",
    "fragment": "仅破片机理",
    "blast": "仅冲击波机理",
    "combined": "按条件独立并集组合破片与冲击波：1-(1-pf)(1-pb)",
}


CODE_LOCATIONS = {
    "uav_state_x": [
        ("damage_engine/learning_input_identity.py", 24, "名称/单位/坐标变换合同"),
        ("damage_engine/dataset.py", 1, "归档编码与写入"),
        ("damage_engine/torch_dataset.py", 1, "默认模型输入过滤"),
        ("damage_engine/training_view.py", 1, "quaternion4 到 rotation-6D6 的训练变换"),
    ],
    "uav_state_feature_mask": [
        ("damage_engine/dataset.py", 1, "逐维可用性写入"),
        ("damage_engine/archive_validation.py", 1, "完整性与有限值校验"),
    ],
    "case_x": [("damage_engine/dataset.py", 20, "列名/单位"), ("damage_engine/dataset.py", 352, "编码"), ("damage_engine/dataset.py", 1529, "写入")],
    "component_x": [("damage_engine/dataset.py", 78, "列名/单位"), ("damage_engine/dataset.py", 439, "编码"), ("damage_engine/dataset.py", 1531, "写入")],
    "armor_x": [("damage_engine/dataset.py", 101, "列名/单位"), ("damage_engine/dataset.py", 524, "编码"), ("damage_engine/dataset.py", 1533, "写入")],
    "component_armor_path_x": [("damage_engine/dataset.py", 115, "列名/单位"), ("damage_engine/dataset.py", 565, "几何编码"), ("damage_engine/dataset.py", 1539, "写入")],
    "armor_y": [("damage_engine/dataset.py", 120, "标签名称"), ("damage_engine/dataset.py", 1311, "聚合"), ("damage_engine/dataset.py", 1535, "写入")],
    "component_y": [("damage_engine/fragments.py", 901, "破片响应"), ("damage_engine/blast.py", 237, "冲击波响应"), ("damage_engine/engine.py", 214, "机理组合"), ("damage_engine/dataset.py", 1202, "聚合")],
    "component_frechet_y": [("damage_engine/engine.py", 214, "上下界"), ("damage_engine/dataset.py", 1240, "聚合"), ("damage_engine/dataset.py", 1545, "写入")],
    "component_aux_y": [("damage_engine/dataset.py", 194, "名称/单位"), ("damage_engine/dataset.py", 1258, "聚合"), ("damage_engine/dataset.py", 1549, "写入")],
    "system_y": [("examples/target.demo.json", 124, "树定义"), ("damage_engine/damage_tree.py", 151, "树求值"), ("damage_engine/dataset.py", 1375, "聚合"), ("damage_engine/dataset.py", 1553, "写入")],
    "case_physics_x": [("damage_engine/dataset.py", 59, "物理/控制拆分"), ("damage_engine/dataset.py", 1632, "写入"), ("damage_engine/torch_dataset.py", 57, "默认加载"), ("damage_engine/training_view.py", 396, "训练变换")],
    "solver_context_x": [("damage_engine/dataset.py", 62, "列定义"), ("damage_engine/dataset.py", 1634, "写入")],
    "sampling_selection_probability": [("damage_engine/dataset.py", 1640, "写入"), ("damage_engine/sampling_spec.py", 1, "采样策略合同")],
    "sampling_design_weight_raw": [("damage_engine/dataset.py", 1643, "写入"), ("damage_engine/training_view.py", 511, "训练边界")],
    "label_task_names": [("damage_engine/dataset.py", 1716, "任务验证"), ("damage_engine/label_spec.py", 1, "任务声明合同")],
    "label_task_available_mask": [("damage_engine/dataset.py", 1716, "任务可用性"), ("damage_engine/label_spec.py", 42, "LabelSpec 校验")],
}


def html_escape(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


_SOURCE_LINE_CACHE: dict[str, list[str]] = {}


def _source_lines(relative_path: str) -> list[str]:
    if relative_path not in _SOURCE_LINE_CACHE:
        path = PROJECT_ROOT / relative_path
        _SOURCE_LINE_CACHE[relative_path] = (
            path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
        )
    return _SOURCE_LINE_CACHE[relative_path]


def _nearest_source_line(relative_path: str, symbol: str, hint: int) -> int:
    candidates = [
        line_number
        for line_number, line in enumerate(_source_lines(relative_path), start=1)
        if symbol in line
    ]
    return min(candidates, key=lambda value: abs(value - hint)) if candidates else hint


def semantic_for(name: str) -> str:
    if name in BASE_SEMANTICS:
        return BASE_SEMANTICS[name]
    for suffix, explanation in (
        ("_stddev", "在每案条件蒙特卡洛重复之间的样本标准差；只有对应 uncertainty_mask 为 true 时可解释。"),
        ("_ci95", "均值的近似 95% 置信区间，末轴顺序为 [lower, upper]；不可估计的位置写 NaN。"),
        ("_sample_count", "每个标签单元实际用于均值/不确定性统计的重复数；标签不可用时为 0。"),
        ("_uncertainty_mask", "该单元的不确定性统计是否可用：标签可用、样本数至少 2 且标准差/区间均有限。"),
    ):
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            base_name = base + "_y" if base in {"armor", "component", "component_frechet", "component_aux", "system"} else base
            return f"{BASE_SEMANTICS.get(base_name, base_name)} {explanation}"
    if name.endswith("_mask"):
        return "布尔可用性或选择掩码。false 表示对应数值不得按有效观测参与模型、损失或统计。"
    if name.endswith("_names"):
        return "语义轴名称表，用于把整数位置恢复为可读类别。"
    if name.endswith("_units"):
        return "与对应名称轴对齐的单位表。"
    if name.startswith("input_sampling_"):
        return "N 轴原始输入对应的采样身份字段；保留标签执行前的完整输入顺序和谱系。"
    if name.startswith("sampling_"):
        return "B 轴成功案例对应的采样设计字段；用于拆分、设计权重和采样谱系审计。"
    return "Manifest 中声明并存储于 tensors.npz 的合同字段。"


def use_for(name: str, role: str) -> str:
    if name in USE_OVERRIDES:
        return USE_OVERRIDES[name]
    if name.endswith("_stddev") or name.endswith("_ci95") or name.endswith("_uncertainty_mask") or name.endswith("_sample_count"):
        return "不确定性加权、质量门控与评估；不是额外独立标签"
    if role == "mask":
        return "控制输入/标签/统计是否有效，避免把缺失值误当物理零"
    if role == "index":
        return "建立轴映射、批处理和可重复追溯；通常不直接作为连续模型特征"
    if role == "metadata":
        return "解码语义轴、追溯来源、拆分和采样身份；通常不直接进入网络"
    if role == "feature":
        return "候选模型输入；是否进入最终网络取决于代理边界和特征角色合同"
    if role == "label":
        return "教师响应或辅助监督；只有 LabelSpec 声明的任务是正式训练目标"
    return "数据合同与审计"


def locations_for(name: str) -> list[dict[str, Any]]:
    base = name
    for suffix in ("_stddev", "_ci95", "_sample_count", "_uncertainty_mask"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            if base in {"armor", "component", "component_frechet", "component_aux", "system"}:
                base += "_y"
            break
    if name in CODE_LOCATIONS:
        entries = CODE_LOCATIONS[name]
    elif base in CODE_LOCATIONS:
        entries = list(CODE_LOCATIONS[base]) + [("damage_engine/dataset.py", 1721, "不确定性门控")]
    elif name.startswith("uav_state_"):
        entries = [
            ("damage_engine/learning_input_identity.py", 24, "UAV-only 名称/单位/变换合同"),
            ("damage_engine/dataset.py", 1, "tensor 1.8 归档写入"),
            ("damage_engine/torch_dataset.py", 1, "默认 Loader 模型输入合同"),
        ]
    elif name.startswith("sampling_") or name.startswith("input_sampling_") or name == "input_frozen_splits":
        entries = [("damage_engine/dataset.py", 1624, "采样张量写入"), ("damage_engine/torch_dataset.py", 57, "Loader 合同")]
    elif name.startswith("case_") or name in {"aleatory_seeds", "epistemic_sample_ids", "replicate_counts"}:
        entries = [("damage_engine/dataset.py", 1515, "案例元数据写入"), ("damage_engine/torch_dataset.py", 516, "逐样本读取")]
    elif name.startswith("input_"):
        entries = [("damage_engine/dataset.py", 1515, "输入到有效案例映射")]
    elif name.startswith("component_"):
        entries = [("damage_engine/dataset.py", 1531, "部件张量写入"), ("damage_engine/torch_dataset.py", 18, "Loader 字段")]
    elif name.startswith("armor_"):
        entries = [("damage_engine/dataset.py", 1533, "装甲张量写入"), ("damage_engine/torch_dataset.py", 18, "Loader 字段")]
    elif name.startswith("system_") or name.startswith("label_task_"):
        entries = [("damage_engine/dataset.py", 1375, "系统标签聚合"), ("damage_engine/dataset.py", 1553, "写入")]
    elif name.startswith("graph_") or name.startswith("tree_"):
        entries = [("damage_engine/dataset.py", 1618, "图结构写入"), ("damage_engine/torch_dataset.py", 86, "静态图加载")]
    else:
        entries = [("damage_engine/dataset.py", 1528, "张量合同写入"), ("damage_engine/dataset.py", 1966, "Manifest 字段声明")]
    source_lines = _source_lines("damage_engine/dataset.py")
    writer_line = next(
        (
            line_number
            for line_number, line in enumerate(source_lines, start=1)
            if f'"{name}"' in line
        ),
        None,
    )
    if writer_line is None and name.endswith("_sample_count"):
        writer_line = 1740
    elif writer_line is None and name.endswith("_uncertainty_mask"):
        writer_line = 1741
    elif writer_line is None and name.startswith("input_sampling_"):
        writer_line = 1671
    if writer_line is not None and not any(
        path == "damage_engine/dataset.py" and line == writer_line
        for path, line, _ in entries
    ):
        entries = list(entries) + [("damage_engine/dataset.py", writer_line, "NPZ 字段写入")]
    resolved_entries = []
    seen_locations: set[tuple[str, int]] = set()
    for path, line, why in entries:
        symbol = name if name in "\n".join(_source_lines(path)) else base
        resolved_line = _nearest_source_line(path, symbol, line)
        location_key = (path, resolved_line)
        if location_key not in seen_locations:
            resolved_entries.append(
                {"path": path, "line": resolved_line, "why": why}
            )
            seen_locations.add(location_key)
    return resolved_entries


def category_for(name: str, role: str) -> str:
    if name.startswith("sampling_") or name.startswith("input_sampling_") or name == "input_frozen_splits":
        return "sampling"
    if name.startswith("graph_") or name.startswith("tree_") or "vertices" in name or "vertex_offsets" in name:
        return "topology"
    if name.endswith("_stddev") or name.endswith("_ci95") or name.endswith("_sample_count") or name.endswith("_uncertainty_mask"):
        return "uncertainty"
    if role == "feature":
        return "feature"
    if role == "label":
        return "label"
    if role == "mask":
        return "mask"
    return "metadata"


def number_for_json(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 8)
    if isinstance(value, (int, bool, str)) or value is None:
        return value
    return str(value)


def preview(array: np.ndarray, limit: int = 24) -> dict[str, Any]:
    flat = array.reshape(-1)
    values = [number_for_json(item) for item in flat[:limit]]
    return {"values": values, "shown": len(values), "total": int(flat.size)}


def statistics(array: np.ndarray) -> dict[str, Any]:
    flat = array.reshape(-1)
    result: dict[str, Any] = {"elements": int(flat.size)}
    if array.dtype.kind == "b":
        true_count = int(np.count_nonzero(flat))
        result.update(
            {
                "kind": "bool",
                "true": true_count,
                "false": int(flat.size - true_count),
                "true_rate": round(true_count / flat.size, 8) if flat.size else None,
                "unique": 2 if true_count not in {0, flat.size} else (1 if flat.size else 0),
            }
        )
        return result
    if array.dtype.kind in {"i", "u", "f"}:
        numeric = flat.astype(np.float64, copy=False)
        finite = numeric[np.isfinite(numeric)]
        result.update({"kind": "numeric", "finite": int(finite.size), "nan_or_inf": int(flat.size - finite.size)})
        if finite.size:
            unique = np.unique(finite)
            result.update(
                {
                    "min": number_for_json(np.min(finite)),
                    "max": number_for_json(np.max(finite)),
                    "mean": number_for_json(np.mean(finite)),
                    "median": number_for_json(np.median(finite)),
                    "std": number_for_json(np.std(finite)),
                    "unique": int(unique.size),
                    "unique_preview": [number_for_json(v) for v in unique[:12]],
                }
            )
        return result
    strings = flat.astype(str)
    unique, counts = np.unique(strings, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    result.update(
        {
            "kind": "string",
            "unique": int(unique.size),
            "top": [
                {"value": str(unique[index]), "count": int(counts[index])}
                for index in order[:8]
            ],
        }
    )
    return result


def feature_semantic(name: str) -> str:
    if name in FEATURE_SEMANTICS:
        return FEATURE_SEMANTICS[name]
    if name.startswith("mass_family_"):
        return f"破片质量分布是否为 {name.removeprefix('mass_family_')} 的 one-hot 指示"
    if name.startswith("shape_"):
        return f"几何/破片形状是否为 {name.removeprefix('shape_')} 的 one-hot 指示"
    if name.startswith("initial_velocity_"):
        return f"初速度模型是否为 {name.removeprefix('initial_velocity_')} 的 one-hot 指示"
    if name.startswith("direction_") and name.removeprefix("direction_") in {"isotropic", "cone", "taylor_ring"}:
        return f"方向模型是否为 {name.removeprefix('direction_')} 的 one-hot 指示"
    if name.startswith("mechanism_"):
        return f"{name.removeprefix('mechanism_').removesuffix('_enabled')} 机理是否启用"
    if name.startswith("coupling_"):
        return f"耦合模式是否为 {name.removeprefix('coupling_')} 的 one-hot 指示"
    if name.startswith("fidelity_"):
        return f"保真度是否为 {name.removeprefix('fidelity_')} 的 one-hot 指示"
    if name.startswith("exposure_"):
        return f"暴露类别是否为 {name.removeprefix('exposure_')} 的 one-hot 指示"
    if name.startswith("rotation_r"):
        return f"部件姿态旋转矩阵元素 {name.removeprefix('rotation_')}"
    return "与名称和单位轴对齐的合同维度；详细来源见选中张量的代码定位。"


def resolve_dataset_layout(dataset_root: Path) -> tuple[Path, Path]:
    """Return ``(sampling_run_root, tensor_archive_root)`` for either accepted input form."""
    root = dataset_root.resolve()
    formal = root / "formal_dataset"
    if (formal / "manifest.json").is_file():
        return root, formal
    if (root / "manifest.json").is_file():
        run_root = root.parent if root.name == "formal_dataset" and (root.parent / "plan").is_dir() else root
        return run_root, root
    raise RuntimeError(
        f"no manifest.json found under dataset root or formal_dataset: {root}"
    )


def load_all_arrays(
    fields: list[dict[str, Any]], archive_root: Path
) -> tuple[dict[str, np.ndarray], list[Path]]:
    shard_paths = sorted(archive_root.glob("shards/part-*/tensors.npz"))
    if not shard_paths and (archive_root / "tensors.npz").is_file():
        shard_paths = [archive_root / "tensors.npz"]
    if not shard_paths:
        raise RuntimeError("no tensor shards found")
    expected = [field["name"] for field in fields]
    sample_axis = {
        field["name"]: str(field["shape"]).startswith("[B") or str(field["shape"]).startswith("[N")
        for field in fields
    }
    pieces: dict[str, list[np.ndarray]] = {name: [] for name in expected}
    static: dict[str, np.ndarray] = {}
    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as archive:
            if set(archive.files) != set(expected):
                missing = sorted(set(expected) - set(archive.files))
                extra = sorted(set(archive.files) - set(expected))
                raise RuntimeError(f"tensor contract mismatch in {shard_path}: missing={missing}, extra={extra}")
            for name in expected:
                value = np.asarray(archive[name])
                if sample_axis[name]:
                    pieces[name].append(value.copy())
                elif name not in static:
                    static[name] = value.copy()
                else:
                    if value.dtype.kind in {"f", "c"}:
                        same = np.array_equal(static[name], value, equal_nan=True)
                    else:
                        same = np.array_equal(static[name], value)
                    if not same:
                        raise RuntimeError(f"static tensor differs across shards: {name}")
    arrays = {
        name: np.concatenate(pieces[name], axis=0) if sample_axis[name] else static[name]
        for name in expected
    }
    return arrays, shard_paths


def role_by_raw_feature(sampling_spec: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in sampling_spec.get("feature_roles", []):
        feature_id = item["feature_id"]
        feature_path = item["path"]
        role = item["role"]
        raw_names: list[str] = []
        if feature_path == "/encounter/detonation_position_m":
            raw_names = [
                "detonation_x_m", "detonation_y_m", "detonation_z_m",
                "uav_position_relative_to_fixed_threat_x_m",
                "uav_position_relative_to_fixed_threat_y_m",
                "uav_position_relative_to_fixed_threat_z_m",
            ]
        elif feature_path == "/encounter/body_to_target_orientation_xyzw":
            raw_names = [
                "orientation_qx", "orientation_qy", "orientation_qz", "orientation_qw",
                "uav_attitude_relative_to_fixed_threat_qx",
                "uav_attitude_relative_to_fixed_threat_qy",
                "uav_attitude_relative_to_fixed_threat_qz",
                "uav_attitude_relative_to_fixed_threat_qw",
                *item["tensor_feature_names"],
            ]
        elif feature_path == "/encounter/velocity_target_mps":
            raw_names = [
                "velocity_x_mps", "velocity_y_mps", "velocity_z_mps",
                "uav_velocity_relative_to_fixed_threat_x_mps",
                "uav_velocity_relative_to_fixed_threat_y_mps",
                "uav_velocity_relative_to_fixed_threat_z_mps",
            ]
        elif feature_path.startswith("/encounter/velocity_target_mps/"):
            axis = int(feature_path.rsplit("/", 1)[-1])
            raw_names = [
                ("velocity_x_mps", "velocity_y_mps", "velocity_z_mps")[axis],
                (
                    "uav_velocity_relative_to_fixed_threat_x_mps",
                    "uav_velocity_relative_to_fixed_threat_y_mps",
                    "uav_velocity_relative_to_fixed_threat_z_mps",
                )[axis],
            ]
        elif feature_id == "detonation_position":
            raw_names = ["detonation_x_m", "detonation_y_m", "detonation_z_m"]
        elif feature_id == "body_orientation":
            raw_names = ["orientation_qx", "orientation_qy", "orientation_qz", "orientation_qw"]
        else:
            raw_names = list(item["tensor_feature_names"])
        for name in raw_names:
            result[name] = role
    return result


def channel_catalog(
    arrays: dict[str, np.ndarray],
    tensor_name: str,
    channel_names: list[str],
    channel_units: list[str],
    mask_name: str | None,
    raw_roles: dict[str, str],
    representative_indices: dict[str, int],
) -> list[dict[str, Any]]:
    values = arrays[tensor_name]
    matrix = values.reshape((values.shape[0], -1))
    if matrix.shape[1] != len(channel_names):
        raise RuntimeError(f"channel count mismatch for {tensor_name}: {matrix.shape[1]} != {len(channel_names)}")
    masks = arrays[mask_name].reshape((values.shape[0], -1)) if mask_name else np.ones_like(matrix, dtype=bool)
    rows = []
    for index, (name, unit) in enumerate(zip(channel_names, channel_units)):
        valid = masks[:, index].astype(bool)
        selected = matrix[valid, index]
        stat = statistics(selected) if selected.size else {"kind": "numeric", "elements": 0, "finite": 0, "nan_or_inf": 0}
        available = int(np.count_nonzero(valid))
        if not available:
            status = "结构缺失"
        elif stat.get("unique", 0) <= 1:
            status = "常量"
        else:
            status = "变化"
        rows.append(
            {
                "index": index,
                "name": name,
                "unit": unit,
                "semantic": feature_semantic(name.split(" / ")[-1]),
                "declared_role": raw_roles.get(name.split(" / ")[-1], "not_declared_here"),
                "status": status,
                "available": available,
                "available_rate": round(available / len(valid), 8),
                "stats": stat,
                "examples": {
                    split: number_for_json(matrix[row_index, index]) if bool(masks[row_index, index]) else None
                    for split, row_index in representative_indices.items()
                },
            }
        )
    return rows


def _load_contract_document(
    run_root: Path,
    manifest: dict[str, Any],
    filename: str,
    manifest_key: str,
) -> dict[str, Any]:
    path = run_root / "plan" / filename
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    embedded = manifest.get("source_contracts", {}).get(manifest_key)
    if isinstance(embedded, dict):
        return embedded
    raise RuntimeError(f"missing {filename} and embedded {manifest_key}")


def build_payload(dataset_root: Path = DEFAULT_DATASET_ROOT) -> dict[str, Any]:
    run_root, archive_root = resolve_dataset_layout(dataset_root)
    manifest_path = archive_root / "manifest.json"
    audit_path = run_root / "audits" / "training_readiness_audit.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = (
        json.loads(audit_path.read_text(encoding="utf-8"))
        if audit_path.is_file()
        else {}
    )
    sampling_spec = _load_contract_document(
        run_root, manifest, "sampling_spec.json", "sampling_spec_document"
    )
    label_spec = _load_contract_document(
        run_root, manifest, "label_spec.json", "label_spec_document"
    )
    fields = manifest["tensor_contract"]["fields"]
    arrays, shard_paths = load_all_arrays(fields, archive_root)
    splits = arrays["split_names"].astype(str)
    representative_indices = {
        split: int(np.flatnonzero(splits == split)[0])
        for split in ("train", "validation", "test")
    }
    representatives = {
        split: {
            "index": index,
            "case_id": str(arrays["case_ids"][index]),
            "split": split,
        }
        for split, index in representative_indices.items()
    }
    tensor_rows = []
    for field in fields:
        name = field["name"]
        array = arrays[name]
        sample_aligned = str(field["shape"]).startswith("[B") or str(field["shape"]).startswith("[N")
        examples = {}
        if sample_aligned:
            for split, index in representative_indices.items():
                examples[split] = preview(np.asarray(array[index]))
        else:
            examples["static"] = preview(array, limit=80)
        tensor_rows.append(
            {
                "name": name,
                "dtype": str(array.dtype),
                "contract_dtype": field["dtype"],
                "symbolic_shape": field["shape"],
                "actual_shape": list(array.shape),
                "role": field["semantic_role"],
                "role_label": ROLE_LABELS[field["semantic_role"]],
                "unit": field.get("unit"),
                "category": category_for(name, field["semantic_role"]),
                "semantic": semantic_for(name),
                "dataset_use": use_for(name, field["semantic_role"]),
                "sample_aligned": sample_aligned,
                "statistics": statistics(array),
                "examples": examples,
                "locations": locations_for(name),
            }
        )

    raw_roles = role_by_raw_feature(sampling_spec)
    feature_specs: dict[str, tuple[list[str], list[str], str | None]] = {
        "case_x": (arrays["case_feature_names"].astype(str).tolist(), arrays["case_feature_units"].astype(str).tolist(), "case_feature_mask"),
        "case_physics_x": (arrays["case_physics_feature_names"].astype(str).tolist(), arrays["case_physics_feature_units"].astype(str).tolist(), "case_physics_feature_mask"),
        "solver_context_x": (arrays["solver_context_feature_names"].astype(str).tolist(), arrays["solver_context_feature_units"].astype(str).tolist(), "solver_context_feature_mask"),
        "component_x": (
            [f"{component} / {name}" for component, name in product(arrays["component_ids"].astype(str), arrays["component_feature_names"].astype(str))],
            arrays["component_feature_units"].astype(str).tolist() * len(arrays["component_ids"]),
            "component_feature_mask",
        ),
        "armor_x": (
            [f"{armor} / {name}" for armor, name in product(arrays["armor_ids"].astype(str), arrays["armor_feature_names"].astype(str))],
            arrays["armor_feature_units"].astype(str).tolist() * len(arrays["armor_ids"]),
            "armor_feature_mask",
        ),
        "component_armor_path_x": (
            [f"{component} / {armor} / {name}" for component, armor, name in product(arrays["component_ids"].astype(str), arrays["armor_ids"].astype(str), arrays["armor_path_feature_names"].astype(str))],
            arrays["armor_path_feature_units"].astype(str).tolist() * len(arrays["component_ids"]) * len(arrays["armor_ids"]),
            "component_armor_path_mask",
        ),
        "armor_y": (
            [f"{armor} / {name}" for armor, name in product(arrays["armor_ids"].astype(str), arrays["armor_label_names"].astype(str))],
            arrays["armor_label_units"].astype(str).tolist() * len(arrays["armor_ids"]),
            "armor_label_mask",
        ),
        "component_y": (
            [f"{component} / {mechanism} / {state}" for component, mechanism, state in product(arrays["component_ids"].astype(str), arrays["mechanism_names"].astype(str), arrays["state_names"].astype(str))],
            ["1"] * int(np.prod(arrays["component_y"].shape[1:])),
            "component_label_mask",
        ),
        "component_frechet_y": (
            [f"{component} / {state} / {bound}" for component, state, bound in product(arrays["component_ids"].astype(str), arrays["state_names"].astype(str), arrays["frechet_bound_names"].astype(str))],
            ["1"] * int(np.prod(arrays["component_frechet_y"].shape[1:])),
            "component_frechet_label_mask",
        ),
        "component_aux_y": (
            [f"{component} / {name}" for component, name in product(arrays["component_ids"].astype(str), arrays["component_aux_label_names"].astype(str))],
            arrays["component_aux_label_units"].astype(str).tolist() * len(arrays["component_ids"]),
            "component_aux_label_mask",
        ),
        "system_y": (arrays["system_output_names"].astype(str).tolist(), ["1"] * len(arrays["system_output_names"]), "system_label_mask"),
    }
    if "uav_state_x" in arrays:
        feature_specs = {
            "uav_state_x": (
                arrays["uav_state_feature_names"].astype(str).tolist(),
                arrays["uav_state_feature_units"].astype(str).tolist(),
                "uav_state_feature_mask",
            ),
            **feature_specs,
        }
    feature_catalogs = {
        tensor_name: channel_catalog(
            arrays,
            tensor_name,
            names,
            units,
            mask_name,
            raw_roles,
            representative_indices,
        )
        for tensor_name, (names, units, mask_name) in feature_specs.items()
    }
    counts_by_role = {
        role: sum(row["role"] == role for row in tensor_rows)
        for role in ROLE_LABELS
    }
    try:
        archive_label = archive_root.relative_to(run_root).as_posix() or "."
    except ValueError:
        archive_label = archive_root.as_posix()
    sources = [
        f"{archive_label}/manifest.json",
        f"{len(shard_paths)} × {archive_label}/shards/part-*/tensors.npz"
        if len(shard_paths) > 1
        else f"{archive_label}/{shard_paths[0].relative_to(archive_root).as_posix()}",
    ]
    if audit_path.is_file():
        sources.append("audits/training_readiness_audit.json")
    sources.extend(
        [
            "plan/sampling_spec.json",
            "plan/label_spec.json",
            "damage_engine/dataset.py",
            "damage_engine/torch_dataset.py",
            "damage_engine/training_view.py",
            "damage_engine/learning_input_identity.py",
            "damage_engine/fragments.py",
            "damage_engine/blast.py",
            "damage_engine/engine.py",
            "damage_engine/damage_tree.py",
        ]
    )
    admission = manifest.get("label_admission", {})
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": {
            "dataset_id": manifest["dataset_id"],
            "manifest_version": manifest["schema_version"],
            "tensor_contract_version": manifest["tensor_contract"]["version"],
            "engine_version": manifest["engine"]["version"],
            "case_count": int(len(arrays["case_ids"])),
            "shard_count": len(shard_paths),
            "field_count": len(fields),
            "split_counts": {split: int(np.count_nonzero(splits == split)) for split in ("train", "validation", "test")},
            "scientific_evidence_status": admission.get("scientific_evidence_status", "unknown"),
            "approved_uses": admission.get("approved_uses", []),
            "model_input_contract": manifest.get("model_input_contract", {}),
            "model_input_tensor": manifest.get("model_input_contract", {}).get(
                "tensor_field", "case_physics_x"
            ),
        },
        "representatives": representatives,
        "counts_by_role": counts_by_role,
        "axes": manifest["tensor_contract"]["axes"],
        "tensors": tensor_rows,
        "feature_catalogs": feature_catalogs,
        "input_audit": audit.get("input_features", {}),
        "label_spec": label_spec,
        "sources": sources,
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>全张量语义与代码定位</title>
  <style>
    :root{--bg:#07111f;--panel:#0d1b2d;--panel2:#11243a;--line:#263c55;--text:#e8f0f8;--muted:#9fb0c3;--cyan:#47d7e8;--blue:#6aa8ff;--green:#57d39b;--orange:#ffb45b;--purple:#b38cff;--red:#ff7383;--shadow:0 18px 45px rgba(0,0,0,.25);color-scheme:dark}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:radial-gradient(circle at 15% 0%,rgba(71,215,232,.10),transparent 28%),var(--bg);color:var(--text);font:14px/1.65 Inter,"Segoe UI","Microsoft YaHei",sans-serif}a{color:var(--cyan);text-decoration:none}a:hover{text-decoration:underline}code{font-family:"Cascadia Code",Consolas,monospace;color:#bfe9ff}button,input,select{font:inherit}.wrap{width:min(1480px,calc(100% - 32px));margin:auto}.hero{padding:42px 0 24px}.kicker{letter-spacing:.16em;text-transform:uppercase;color:var(--cyan);font-size:12px}.hero h1{margin:6px 0 10px;font-size:clamp(28px,4vw,52px);line-height:1.12}.hero p{max-width:980px;color:var(--muted);margin:0}.metrics{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin:22px 0}.metric,.panel,.detail{background:linear-gradient(180deg,rgba(17,36,58,.96),rgba(13,27,45,.96));border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow)}.metric{padding:15px}.metric b{display:block;font-size:24px;color:var(--text)}.metric span{color:var(--muted)}.notice{padding:14px 16px;border-left:4px solid var(--orange);background:rgba(255,180,91,.09);border-radius:10px;color:#ffd8a8;margin:14px 0 0}.nav{position:sticky;top:0;z-index:5;background:rgba(7,17,31,.88);backdrop-filter:blur(12px);border-bottom:1px solid var(--line)}.nav .wrap{display:flex;gap:8px;overflow:auto;padding:10px 0}.nav a{white-space:nowrap;padding:7px 11px;border-radius:9px;color:var(--muted)}.nav a:hover{background:var(--panel2);color:var(--text);text-decoration:none}section{padding:34px 0}.section-head{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:16px}.section-head h2{font-size:25px;margin:0}.section-head p{color:var(--muted);margin:0;max-width:720px}.panel{padding:18px}.flow{display:grid;grid-template-columns:1fr auto 1fr auto 1fr auto 1fr;align-items:stretch;gap:10px}.node{padding:16px;background:rgba(255,255,255,.025);border-radius:12px}.node b{display:block;color:var(--cyan);margin-bottom:5px}.node small{color:var(--muted)}.arrow{align-self:center;color:var(--blue);font-size:24px}.axis-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:9px}.axis{padding:11px;background:rgba(255,255,255,.025);border-radius:10px}.axis b{display:block}.axis span{color:var(--muted)}.toolbar{display:grid;grid-template-columns:minmax(250px,1.2fr) minmax(180px,.5fr) minmax(210px,.6fr);gap:10px;margin-bottom:12px}.control{width:100%;background:#081625;border:1px solid var(--line);color:var(--text);border-radius:10px;padding:10px 12px}.filters{display:flex;flex-wrap:wrap;gap:7px;margin:10px 0 16px}.chip{border:1px solid var(--line);background:transparent;color:var(--muted);padding:7px 10px;border-radius:999px;cursor:pointer}.chip.active{background:var(--cyan);color:#031016;border-color:transparent}.layout{display:grid;grid-template-columns:minmax(0,1.65fr) minmax(330px,.85fr);gap:14px;align-items:start}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:14px;background:rgba(13,27,45,.75)}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:10px 11px;border-bottom:1px solid rgba(38,60,85,.72);vertical-align:top}th{position:sticky;top:0;background:#10233a;color:#bcd0e5;font-weight:600;z-index:1}tr[data-name]{cursor:pointer}tr[data-name]:hover,tr.selected{background:rgba(71,215,232,.08)}.name{font-family:"Cascadia Code",Consolas,monospace;color:#c7edff}.badge{display:inline-block;padding:2px 7px;border-radius:999px;background:rgba(106,168,255,.13);color:#b9d4ff;font-size:12px;white-space:nowrap}.badge.label{background:rgba(255,180,91,.13);color:#ffd39a}.badge.mask{background:rgba(179,140,255,.13);color:#d9c3ff}.badge.uncertainty{background:rgba(87,211,155,.13);color:#aef0ce}.detail{padding:18px;position:sticky;top:64px}.detail h3{margin:0 0 4px;font-size:20px;overflow-wrap:anywhere}.muted{color:var(--muted)}.kv{display:grid;grid-template-columns:110px 1fr;gap:8px 10px;margin:15px 0}.kv dt{color:var(--muted)}.kv dd{margin:0;overflow-wrap:anywhere}.value-box{background:#071523;border:1px solid var(--line);border-radius:11px;padding:11px;margin-top:10px;overflow:auto}.value-box pre{margin:0;white-space:pre-wrap;overflow-wrap:anywhere;color:#d9efff}.code-list{display:grid;gap:7px}.code-ref{display:block;padding:8px 10px;background:rgba(255,255,255,.025);border-radius:9px}.sample-switch{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}.sample-switch button{border:1px solid var(--line);background:transparent;color:var(--muted);border-radius:8px;padding:6px 9px;cursor:pointer}.sample-switch button.active{background:var(--blue);color:#07111f;border-color:transparent}.count{color:var(--muted);margin-left:auto}.empty{padding:26px;text-align:center;color:var(--muted)}.feature-tools{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px}.feature-tools select{min-width:260px}.status{font-weight:600}.status.variable,.status.变化{color:var(--green)}.status.constant,.status.常量{color:var(--orange)}.status.missing,.status.结构缺失{color:var(--red)}.role-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;background:var(--blue)}.wide-table td:nth-child(2){min-width:280px}.formula{font-family:"Cascadia Code",Consolas,monospace;color:#dff8ff;background:#071523;padding:12px;border-radius:10px;overflow:auto}.provenance{display:grid;grid-template-columns:1fr 1fr;gap:14px}.source-list{columns:2;column-gap:28px}.source-list li{break-inside:avoid;margin:4px 0;color:var(--muted)}footer{padding:28px 0 46px;color:var(--muted);border-top:1px solid var(--line)}@media(max-width:1050px){.metrics{grid-template-columns:repeat(3,1fr)}.layout{grid-template-columns:1fr}.detail{position:static}.flow{grid-template-columns:1fr}.arrow{transform:rotate(90deg);justify-self:center}.provenance{grid-template-columns:1fr}}@media(max-width:680px){.wrap{width:min(100% - 20px,1480px)}.metrics{grid-template-columns:repeat(2,1fr)}.toolbar{grid-template-columns:1fr}.section-head{display:block}.source-list{columns:1}.kv{grid-template-columns:1fr}.wide-hide{display:none}}
  </style>
</head>
<body>
  <header class="hero"><div class="wrap">
    <div class="kicker">Tensor Contract Explorer · Live archive</div>
    <h1>全张量语义与代码定位</h1>
    <p>逐项解释所选冻结归档中的全部张量字段。每项均来自实际 Manifest 与 NPZ 分片，并关联语义、用途、全局统计、代表样本值和代码位置。</p>
    <div class="metrics">
      <div class="metric"><b id="m-fields">—</b><span>张量字段</span></div>
      <div class="metric"><b id="m-cases">—</b><span>有效案例</span></div>
      <div class="metric"><b id="m-shards">—</b><span>NPZ 分片</span></div>
      <div class="metric"><b id="m-features">—</b><span>输入特征张量</span></div>
      <div class="metric"><b id="m-labels">—</b><span>标签张量</span></div>
    </div>
    <div class="notice"><b>证据边界：</b>这里的“当前值”是冻结归档的真实数值，但物理注册表和阈值仍标记为未验证；它们是 synthetic teacher 输出，不是已经过实物标定的毁伤真值。</div>
  </div></header>
  <nav class="nav"><div class="wrap"><a href="#flow">数据流</a><a href="#axes">轴词典</a><a href="#catalog">张量目录</a><a href="#channels">逐维展开</a><a href="#labels">监督语义</a><a href="#sources">来源与边界</a></div></nav>

  <main>
    <section id="flow"><div class="wrap">
      <div class="section-head"><div><div class="kicker">01 / Data flow</div><h2>从配置到模型样本</h2></div><p>张量语义不是单独存在于 dataset.py，而是沿定义、求解、归档、加载四层形成。</p></div>
      <div class="panel flow">
        <div class="node"><b>输入与语义源</b><small>case / target / physics / sampling / label spec</small></div><div class="arrow">→</div>
        <div class="node"><b>物理教师</b><small>fragments.py · blast.py · engine.py · damage_tree.py</small></div><div class="arrow">→</div>
        <div class="node"><b>张量归档</b><small>dataset.py 编码合同字段，Manifest 冻结轴与 dtype</small></div><div class="arrow">→</div>
        <div class="node"><b>模型视图</b><small>torch_dataset.py 按 model_input_contract 过滤；training_view.py 再变换/归一化</small></div>
      </div>
    </div></section>

    <section id="axes"><div class="wrap">
      <div class="section-head"><div><div class="kicker">02 / Axis dictionary</div><h2>先读懂 shape 中的字母</h2></div><p>符号 shape 来自 Tensor Contract；数字是这批冻结数据的实际轴长。</p></div>
      <div class="panel axis-grid" id="axis-grid"></div>
    </div></section>

    <section id="catalog"><div class="wrap">
      <div class="section-head"><div><div class="kicker">03 / Complete catalog</div><h2>全部张量逐项检索</h2></div><p>点击任意行查看完整语义、用途、全量统计、三拆分代表样本和源代码位置。</p></div>
      <div class="toolbar">
        <input id="search" class="control" type="search" placeholder="搜索名称、语义、用途或代码路径…" aria-label="搜索张量">
        <select id="role-filter" class="control" aria-label="按合同角色筛选"><option value="all">全部合同角色</option></select>
        <select id="shape-filter" class="control" aria-label="按轴对齐筛选"><option value="all">全部轴类型</option><option value="sample">B/N 样本对齐</option><option value="static">静态词表/拓扑</option></select>
      </div>
      <div class="filters" id="category-filters" aria-label="按类别筛选"></div>
      <div class="layout">
        <div class="table-wrap"><table><thead><tr><th>名称</th><th>角色</th><th>shape</th><th class="wide-hide">dtype</th><th>当前摘要</th></tr></thead><tbody id="tensor-body"></tbody></table><div class="empty" id="empty" hidden>没有匹配的张量。</div></div>
        <aside class="detail" id="detail" aria-live="polite"></aside>
      </div>
    </div></section>

    <section id="channels"><div class="wrap">
      <div class="section-head"><div><div class="kicker">04 / Channel explorer</div><h2>主要张量逐维展开</h2></div><p>这里将复合轴展平成可读维度，展示单位、具体语义、全归档统计和真实样本值。</p></div>
      <div class="panel">
        <div class="feature-tools">
          <select id="feature-select" class="control" aria-label="选择要展开的张量"></select>
          <select id="feature-status" class="control" aria-label="筛选变化状态"><option value="all">全部维度</option><option value="变化">仅变化</option><option value="常量">仅常量</option><option value="结构缺失">仅结构缺失</option></select>
          <input id="feature-search" class="control" type="search" placeholder="搜索维度名称或语义…" aria-label="搜索维度">
        </div>
        <div class="table-wrap"><table class="wide-table"><thead><tr><th>#</th><th>维度 / 语义</th><th>单位</th><th>角色/状态</th><th>有效率</th><th>全量统计</th><th>当前样本值</th></tr></thead><tbody id="feature-body"></tbody></table></div>
      </div>
    </div></section>

    <section id="labels"><div class="wrap">
      <div class="section-head"><div><div class="kicker">05 / Supervision</div><h2>当前正式监督目标</h2></div><p>LabelSpec 只正式声明 system_events；其他 y 张量可用于辅助监督或诊断，但需要另行版本化任务定义。</p></div>
      <div class="provenance">
        <div class="panel"><h3><code>system_y[:, 0]</code> · mission_degraded</h3><p>完全相关 OR：飞控综合失效，或飞控因冲击波降级。</p><div class="formula">max(P(combined.failed), P(blast.degraded))</div></div>
        <div class="panel"><h3><code>system_y[:, 1]</code> · mission_loss</h3><p>标准投影：直接等于飞控部件综合 failed 概率。</p><div class="formula">P(mission_loss) = P(combined.failed)</div></div>
      </div>
      <div class="notice"><b>训练含义：</b>两个输出是非互斥软概率，应使用独立 sigmoid 与 soft BCE/Brier；不能用 softmax、argmax 或行归一化。combined 部件概率使用 <code>1-(1-p_fragment)(1-p_blast)</code>。</div>
    </div></section>

    <section id="sources"><div class="wrap">
      <div class="section-head"><div><div class="kicker">06 / Provenance</div><h2>现场来源与验收边界</h2></div><p id="generated-at"></p></div>
      <div class="provenance">
        <div class="panel"><h3>读取来源</h3><ul class="source-list" id="source-list"></ul></div>
        <div class="panel"><h3>解释规则</h3><ul><li>全量统计跨所有已提交分片合并；静态词表/拓扑要求所有分片完全一致。</li><li>“当前样本值”提供每个拆分首个真实案例，不代表总体分布。</li><li>mask=false 时数值只作占位，页面显示为“缺失/不适用”。</li><li>Loader 可读和合同一致只证明软件完整性，不证明物理有效或研究准入。</li></ul></div>
      </div>
    </div></section>
  </main>
  <footer><div class="wrap">自包含离线页面 · 无网络请求 · 数据来自所选冻结归档。</div></footer>

  <script id="payload" type="application/json">__PAYLOAD__</script>
  <script>
    const DATA=JSON.parse(document.getElementById('payload').textContent);
    const categoryLabels={all:'全部',feature:'输入',label:'标签',uncertainty:'不确定性',mask:'掩码',sampling:'采样',topology:'拓扑',metadata:'元数据/索引'};
    const roleClass={label:'label',mask:'mask',uncertainty:'uncertainty'};
    let selectedName=DATA.tensors[0].name, selectedCategory='all', selectedSample='train';
    const fmt=n=>n===null||n===undefined?'—':(typeof n==='number'?(Math.abs(n)>=1e5||(Math.abs(n)>0&&Math.abs(n)<1e-4)?n.toExponential(4):Number(n.toPrecision(7)).toString()):String(n));
    function statText(s){if(s.kind==='bool')return `true ${s.true.toLocaleString()} (${fmt((s.true_rate||0)*100)}%)`;if(s.kind==='string')return `${s.unique.toLocaleString()} 个唯一值 · ${s.top.slice(0,2).map(x=>x.value+'×'+x.count).join('，')}`;if(!s.finite)return '无有限值';return `min ${fmt(s.min)} · median ${fmt(s.median)} · max ${fmt(s.max)}`}
    function previewText(p){if(!p)return '—';const suffix=p.total>p.shown?`\n… 仅显示 ${p.shown}/${p.total} 个元素`:'';return JSON.stringify(p.values,null,2)+suffix}
    document.getElementById('m-fields').textContent=DATA.dataset.field_count;
    document.getElementById('m-cases').textContent=DATA.dataset.case_count.toLocaleString();
    document.getElementById('m-shards').textContent=DATA.dataset.shard_count;
    document.getElementById('m-features').textContent=DATA.counts_by_role.feature;
    document.getElementById('m-labels').textContent=DATA.counts_by_role.label;
    document.getElementById('generated-at').textContent=`生成：${DATA.generated_at} · dataset ${DATA.dataset.dataset_id}`;
    const axis=document.getElementById('axis-grid');DATA.axes.forEach(a=>{const el=document.createElement('div');el.className='axis';el.innerHTML=`<b>${a.axis_id} = ${a.size}</b><span>${a.names_field||'固定/位置轴'}${a.units_field?' · '+a.units_field:''}</span>`;axis.appendChild(el)});
    const roleFilter=document.getElementById('role-filter');Object.entries(DATA.counts_by_role).forEach(([role,count])=>{const o=document.createElement('option');o.value=role;o.textContent=`${DATA.tensors.find(x=>x.role===role)?.role_label||role} (${count})`;roleFilter.appendChild(o)});
    const cats=['all','feature','label','uncertainty','mask','sampling','topology','metadata'];const catHost=document.getElementById('category-filters');cats.forEach(cat=>{const b=document.createElement('button');b.type='button';b.className='chip'+(cat==='all'?' active':'');b.dataset.cat=cat;b.textContent=categoryLabels[cat];b.addEventListener('click',()=>{selectedCategory=cat;catHost.querySelectorAll('button').forEach(x=>x.classList.toggle('active',x===b));renderCatalog()});catHost.appendChild(b)});
    function filteredTensors(){const q=document.getElementById('search').value.trim().toLowerCase(),role=roleFilter.value,shape=document.getElementById('shape-filter').value;return DATA.tensors.filter(t=>(selectedCategory==='all'||t.category===selectedCategory)&&(role==='all'||t.role===role)&&(shape==='all'||(shape==='sample'&&t.sample_aligned)||(shape==='static'&&!t.sample_aligned))&&(!q||[t.name,t.semantic,t.dataset_use,...t.locations.map(x=>x.path)].join(' ').toLowerCase().includes(q)))}
    function renderCatalog(){const rows=filteredTensors(),body=document.getElementById('tensor-body');body.innerHTML='';document.getElementById('empty').hidden=rows.length>0;rows.forEach(t=>{const tr=document.createElement('tr');tr.dataset.name=t.name;tr.classList.toggle('selected',t.name===selectedName);tr.innerHTML=`<td><span class="name">${t.name}</span></td><td><span class="badge ${roleClass[t.role]||''}">${t.role_label}</span></td><td><code>${t.symbolic_shape}</code><br><span class="muted">${t.actual_shape.join('×')}</span></td><td class="wide-hide"><code>${t.dtype}</code></td><td>${statText(t.statistics)}</td>`;tr.addEventListener('click',()=>{selectedName=t.name;renderCatalog();renderDetail()});body.appendChild(tr)});document.querySelector('.count')?.remove();const count=document.createElement('span');count.className='count';count.textContent=`显示 ${rows.length} / ${DATA.tensors.length}`;catHost.appendChild(count);if(!rows.some(x=>x.name===selectedName)&&rows.length){selectedName=rows[0].name;renderDetail()}}
    function codeLinks(locations){return locations.map(l=>`<a class="code-ref" href="${l.path}" target="_blank"><code>${l.path}:${l.line}</code><br><span class="muted">${l.why}</span></a>`).join('')}
    function renderDetail(){const t=DATA.tensors.find(x=>x.name===selectedName),host=document.getElementById('detail');const sampleKeys=t.sample_aligned?['train','validation','test']:['static'];if(!sampleKeys.includes(selectedSample))selectedSample=sampleKeys[0];const rep=t.sample_aligned?DATA.representatives[selectedSample]:null;host.innerHTML=`<div class="kicker">Selected tensor</div><h3>${t.name}</h3><span class="badge ${roleClass[t.role]||''}">${t.role_label}</span><dl class="kv"><dt>符号 shape</dt><dd><code>${t.symbolic_shape}</code></dd><dt>实际 shape</dt><dd><code>${t.actual_shape.join(' × ')}</code></dd><dt>dtype / 单位</dt><dd><code>${t.dtype}</code> · ${t.unit||'见轴单位/无单位'}</dd><dt>具体语义</dt><dd>${t.semantic}</dd><dt>数据集作用</dt><dd>${t.dataset_use}</dd><dt>全量统计</dt><dd>${statText(t.statistics)} · ${t.statistics.elements.toLocaleString()} 元素</dd></dl><hr><b>当前值</b><div class="sample-switch">${sampleKeys.map(k=>`<button type="button" data-sample="${k}" class="${k===selectedSample?'active':''}">${k==='static'?'静态值':k+' · '+DATA.representatives[k].case_id}</button>`).join('')}</div><div class="value-box"><pre>${previewText(t.examples[selectedSample])}</pre></div><hr><b>代码位置</b><div class="code-list">${codeLinks(t.locations)}</div>`;host.querySelectorAll('[data-sample]').forEach(b=>b.addEventListener('click',()=>{selectedSample=b.dataset.sample;renderDetail();renderFeatures()}))}
    ['search','role-filter','shape-filter'].forEach(id=>document.getElementById(id).addEventListener(id==='search'?'input':'change',renderCatalog));
    const featureSelect=document.getElementById('feature-select');Object.keys(DATA.feature_catalogs).forEach(name=>{const o=document.createElement('option');o.value=name;o.textContent=`${name} (${DATA.feature_catalogs[name].length} 维)`;featureSelect.appendChild(o)});
    function renderFeatures(){const name=featureSelect.value||featureSelect.options[0].value,status=document.getElementById('feature-status').value,q=document.getElementById('feature-search').value.trim().toLowerCase(),body=document.getElementById('feature-body');body.innerHTML='';DATA.feature_catalogs[name].filter(r=>(status==='all'||r.status===status)&&(!q||(`${r.name} ${r.semantic}`).toLowerCase().includes(q))).forEach(r=>{const s=r.stats;const role=r.declared_role==='not_declared_here'?'—':r.declared_role;const tr=document.createElement('tr');tr.innerHTML=`<td>${r.index}</td><td><span class="name">${r.name}</span><br><span class="muted">${r.semantic}</span></td><td><code>${r.unit||'—'}</code></td><td><span class="muted">${role}</span><br><span class="status ${r.status}">${r.status}</span></td><td>${fmt(r.available_rate*100)}%</td><td>${statText(s)}</td><td><code>${fmt(r.examples[selectedSample]??r.examples.train)}</code><br><span class="muted">${selectedSample}</span></td>`;body.appendChild(tr)})}
    featureSelect.addEventListener('change',renderFeatures);document.getElementById('feature-status').addEventListener('change',renderFeatures);document.getElementById('feature-search').addEventListener('input',renderFeatures);
    const sources=document.getElementById('source-list');DATA.sources.forEach(s=>{const li=document.createElement('li');li.innerHTML=`<code>${s}</code>`;sources.appendChild(li)});
    renderCatalog();renderDetail();renderFeatures();
  </script>
</body>
</html>
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a self-contained tensor semantics dashboard from a committed dataset archive."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Sampling-run root (containing formal_dataset/) or a direct tensor archive root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination HTML path.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    output_path = arguments.output.resolve()
    payload = build_payload(arguments.dataset_root)
    html = HTML_TEMPLATE.replace(
        "__PAYLOAD__",
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"),
    )
    output_path.write_text(html, encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "bytes": output_path.stat().st_size,
                "tensors": len(payload["tensors"]),
                "cases": payload["dataset"]["case_count"],
                "shards": payload["dataset"]["shard_count"],
                "feature_catalogs": {name: len(rows) for name, rows in payload["feature_catalogs"].items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
