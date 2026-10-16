from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "27_仿真引擎可变参数与代码位置可视化.html"

CASE_PATH = ROOT / "raw_data" / "case.5000.uav_control.v3.json"
SAMPLING_PATH = ROOT / "raw_data" / "sampling_spec.5000.uav_control.v3.json"
TARGET_PATH = ROOT / "examples" / "target.demo.json"
PHYSICS_PATH = ROOT / "examples" / "physics.demo.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


CASE = load_json(CASE_PATH)
SAMPLING = load_json(SAMPLING_PATH)
TARGET = load_json(TARGET_PATH)
PHYSICS = load_json(PHYSICS_PATH)


def compact(value: Any) -> str:
    if value is None:
        return "未配置"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def rec(
    ident: str,
    layer: str,
    subgroup: str,
    name: str,
    path: str,
    current: Any,
    status: str,
    affects: str,
    constraint: str,
    effect: str,
    sources: list[str],
    *,
    note: str = "",
) -> dict[str, Any]:
    return {
        "id": ident,
        "layer": layer,
        "subgroup": subgroup,
        "name": name,
        "path": path,
        "current": compact(current),
        "status": status,
        "affects": affects,
        "constraint": constraint,
        "effect": effect,
        "sources": sources,
        "note": note,
    }


encounter = CASE["encounter"]
warhead = CASE["warhead"]
fragment = warhead["fragment_source"]
blast_source = warhead["blast_source"]
environment = CASE["environment"]
analysis = CASE["analysis"]
armor = TARGET["armor"][0]
component = TARGET["components"][0]
armor_response = armor["response"]
component_response = component["response"]
fragment_source_model = PHYSICS["fragment_source_models"][fragment["model_id"]]
armor_fragment_model = PHYSICS["fragment_response_models"][armor_response["fragment_response_model_id"]]
component_fragment_model = PHYSICS["fragment_response_models"][component_response["fragment_response_model_id"]]
blast_field_model = PHYSICS["blast_field_models"][blast_source["model_id"]]
component_blast_model = PHYSICS["blast_response_models"][component_response["blast_response"]["model_id"]]
transmission_model = PHYSICS["transmission_models"][armor_response["material_id"]]


RECORDS: list[dict[str, Any]] = [
    rec(
        "uav-position",
        "UAV状态",
        "当前学习变量",
        "无人机相对威胁位置 / 起爆点目标系位置",
        "/encounter/detonation_position_m",
        {"template_m": encounter["detonation_position_m"], "sample": "sphere_volume, r=1.5..5.0 m"},
        "当前采样",
        "采样/训练 + 物理响应",
        "3 个有限数；V3 在以原点为中心的球体体积内采样，半径 1.5–5.0 m。模型侧取相反数作为 UAV 相对固定威胁的位置。",
        "同时控制爆源到装甲/部件的距离、射线方向、冲击波尺度距离；也是 UAV 学习输入的 3 个位置量。",
        [
            "raw_data/sampling_spec.5000.uav_control.v3.json:10-18",
            "damage_engine/schemas/simulation_case.schema.json:59-71",
            "damage_engine/learning_input_identity.py:101-132",
            "damage_engine/fragments.py:811-825",
            "damage_engine/blast.py:246-269",
        ],
        note="case 字段表示固定威胁/爆源在目标坐标系中的位置；uav_state_value 对其取负后才是 UAV 相对威胁位置。",
    ),
    rec(
        "uav-velocity",
        "UAV状态",
        "当前学习变量",
        "无人机相对威胁速度 / 目标系平移速度",
        "/encounter/velocity_target_mps/{0,1,2}",
        {"template_mps": encounter["velocity_target_mps"], "sample_each_axis": [-30.0, 30.0]},
        "当前采样",
        "采样/训练 + 物理响应",
        "3 个有限数；V3 对 x/y/z 分别均匀采样 [-30, 30] m/s。模型侧取相反数。",
        "叠加到每个破片的目标系初速度；同时成为 UAV 学习输入的 3 个速度量。",
        [
            "raw_data/sampling_spec.5000.uav_control.v3.json:19-39",
            "damage_engine/schemas/simulation_case.schema.json:59-71",
            "damage_engine/learning_input_identity.py:101-132",
            "damage_engine/fragments.py:770-803",
        ],
    ),
    rec(
        "uav-attitude",
        "UAV状态",
        "当前学习变量",
        "无人机姿态 / 弹体坐标到目标坐标旋转",
        "/encounter/body_to_target_orientation_xyzw",
        {"template_xyzw": encounter["body_to_target_orientation_xyzw"], "sample": "Haar SO(3)"},
        "当前采样",
        "采样/训练 + 物理响应",
        "4 元 xyzw 单位四元数；V3 使用 Haar quaternion。学习侧归一化、求逆并消除 q/-q 重复，再转 6D 旋转表示。",
        "旋转破片方向分布；学习侧形成 6 个连续姿态特征。",
        [
            "raw_data/sampling_spec.5000.uav_control.v3.json:40-44",
            "damage_engine/schemas/simulation_case.schema.json:48-71",
            "damage_engine/learning_input_identity.py:83-132",
            "damage_engine/fragments.py:719-769",
            "damage_engine/training_view.py:181-205",
        ],
    ),
    rec(
        "detonation-time",
        "UAV状态",
        "遭遇条件",
        "起爆时刻",
        "/encounter/detonation_time_s",
        encounter.get("detonation_time_s"),
        "V3固定",
        "身份/追溯",
        "可选，数值且 ≥ 0。",
        "当前核心片段/冲击波计算未读取该值；会进入 case 身份与归档。",
        [
            "raw_data/case.5000.uav_control.v3.json:9-14",
            "damage_engine/schemas/simulation_case.schema.json:59-72",
        ],
        note="这是被 schema 接受但当前求解器未消费的字段，不能把改变它等同于改变动力学。",
    ),
    rec(
        "coordinate-frame",
        "无人机结构",
        "坐标系",
        "目标坐标系定义",
        "/coordinate_frame/{handedness,x_axis,y_axis,z_axis,length_unit}",
        TARGET["coordinate_frame"],
        "V3固定",
        "几何解释",
        "右手系；长度单位只能 m 或 cm；几何字段后缀必须与 length_unit 一致。",
        "决定所有 UAV 装甲、部件几何和遭遇向量的解释方式。",
        [
            "examples/target.demo.json:5-12",
            "damage_engine/schemas/target.schema.json:63-74",
            "damage_engine/validation.py:529-540",
            "damage_engine/normalization.py:194-229",
        ],
    ),
    rec(
        "armor-geometry",
        "无人机结构",
        "装甲几何",
        "装甲多边形顶点",
        "/armor/*/geometry/vertices_{m|cm}",
        armor["geometry"]["vertices_m"],
        "V3固定",
        "物理响应",
        "至少 3 个三维点；必须共面且与 extrusion_direction 一致。",
        "定义破片/冲击波射线与装甲的首次交点和遮挡轮廓。",
        [
            "examples/target.demo.json:18-28",
            "damage_engine/schemas/target.schema.json:194-233",
            "damage_engine/geometry.py:660-711",
            "damage_engine/validation.py:350-389",
        ],
    ),
    rec(
        "armor-extrusion",
        "无人机结构",
        "装甲几何",
        "装甲挤出方向与几何厚度",
        "/armor/*/geometry/{extrusion_direction,extrusion_distance_{m|cm}}",
        {
            "direction": armor["geometry"]["extrusion_direction"],
            "distance_m": armor["geometry"]["extrusion_distance_m"],
        },
        "V3固定",
        "物理响应",
        "方向为有限单位向量；距离 > 0。",
        "形成装甲实体的体积；注意它不同于响应模型中的等效厚度。",
        [
            "examples/target.demo.json:18-28",
            "damage_engine/schemas/target.schema.json:194-233",
            "damage_engine/geometry.py:690-696",
        ],
    ),
    rec(
        "component-geometry-kind",
        "无人机结构",
        "部件几何",
        "部件几何类型",
        "/components/*/geometry/shape",
        component["geometry"]["shape"],
        "V3固定",
        "物理响应",
        "box、cylinder 或 extruded_polygon。",
        "选择不同的求交几何实现，改变投影面积、遮挡和命中位置。",
        [
            "examples/target.demo.json:71-76",
            "damage_engine/schemas/target.schema.json:116-240",
            "damage_engine/geometry.py:660-711",
        ],
    ),
    rec(
        "component-center",
        "无人机结构",
        "部件几何",
        "部件中心位置",
        "/components/*/geometry/center_{m|cm}",
        component["geometry"]["center_m"],
        "V3固定",
        "物理响应",
        "box/cylinder 为三维有限向量；单位与坐标系一致。",
        "决定爆距、传播方向、装甲遮挡路径和破片命中位置。",
        [
            "examples/target.demo.json:71-76",
            "damage_engine/schemas/target.schema.json:116-193",
            "damage_engine/geometry.py:679-688",
            "damage_engine/blast.py:358-401",
        ],
    ),
    rec(
        "component-dimensions",
        "无人机结构",
        "部件几何",
        "部件尺寸",
        "/components/*/geometry/{size_lwh,radius,height,vertices,extrusion_distance}",
        {"size_lwh_m": component["geometry"]["size_lwh_m"]},
        "V3固定",
        "物理响应",
        "所有长度必须 > 0；形状对应字段互斥，m/cm 不能混用。",
        "改变截获面积、射线进入/退出距离和部件暴露体积。",
        [
            "examples/target.demo.json:71-76",
            "damage_engine/schemas/target.schema.json:116-240",
            "damage_engine/geometry.py:660-711",
        ],
    ),
    rec(
        "component-orientation",
        "无人机结构",
        "部件几何",
        "部件局部姿态",
        "/components/*/geometry/orientation_xyzw",
        component["geometry"].get("orientation_xyzw"),
        "V3固定",
        "物理响应",
        "box/cylinder 需要单位四元数 xyzw。",
        "旋转几何体，改变受弹/受爆表面法向和斜侵彻角。",
        [
            "examples/target.demo.json:71-76",
            "damage_engine/schemas/target.schema.json:116-193",
            "damage_engine/geometry.py:679-688",
        ],
    ),
    rec(
        "material-selection",
        "无人机结构",
        "材料与层系",
        "材料与层系标识",
        "/{armor|components}/*/response/{material_id,layer_stack_id}",
        {
            "armor": [armor_response["material_id"], armor_response.get("layer_stack_id")],
            "component": [component_response["material_id"], component_response.get("layer_stack_id")],
        },
        "V3固定",
        "模型选择",
        "material_id 非空；若用于冲击波遮挡，必须在 transmission_models 中存在。",
        "选择材料传递模型；layer_stack_id 当前主要保留身份/追溯语义。",
        [
            "examples/target.demo.json:29-54",
            "examples/target.demo.json:77-111",
            "damage_engine/schemas/target.schema.json:242-292",
            "damage_engine/blast.py:134-168",
        ],
    ),
    rec(
        "equivalent-thickness",
        "无人机结构",
        "材料响应",
        "等效厚度",
        "/{armor|components}/*/response/equivalent_thickness",
        {
            "armor": armor_response["equivalent_thickness"],
            "component": component_response["equivalent_thickness"],
        },
        "V3固定",
        "物理响应 + 不确定性",
        "参数对象：value、unit(m/cm/mm)、valid_range、uncertainty、校准与来源；实际值必须 > 0。",
        "进入破片 V50 计算和冲击波穿透衰减；可按 uncertainty 设为案例级认知抽样或复制级随机抽样。",
        [
            "damage_engine/schemas/target.schema.json:89-114",
            "damage_engine/schemas/target.schema.json:242-263",
            "damage_engine/parameters.py:28-84",
            "damage_engine/parameters.py:87-152",
            "damage_engine/fragments.py:848-880",
            "damage_engine/blast.py:134-168",
        ],
    ),
    rec(
        "vulnerable-ratio",
        "无人机结构",
        "材料响应",
        "易损面积比例",
        "/{armor|components}/*/response/vulnerable_area_ratio",
        {
            "armor": armor_response["vulnerable_area_ratio"],
            "component": component_response["vulnerable_area_ratio"],
        },
        "V3固定",
        "物理响应 + 不确定性",
        "单位 1 时 [0,1]，单位 % 时 [0,100]；带 valid_range 与 uncertainty。",
        "乘入破片单次损伤概率和冲击波状态概率。",
        [
            "damage_engine/schemas/target.schema.json:255-263",
            "damage_engine/parameters.py:54-57",
            "damage_engine/parameters.py:110-114",
            "damage_engine/fragments.py:1017-1029",
            "damage_engine/blast.py:181-234",
        ],
    ),
    rec(
        "fragment-response-link",
        "无人机结构",
        "材料响应",
        "破片响应模型选择",
        "/{armor|components}/*/response/fragment_response_model_id",
        {
            "armor": armor_response["fragment_response_model_id"],
            "component": component_response["fragment_response_model_id"],
        },
        "V3固定",
        "模型选择",
        "非空且必须指向 registry.fragment_response_models。",
        "决定弹道极限、穿透转变、孔洞面积与残余能量毁伤曲线。",
        [
            "damage_engine/schemas/target.schema.json:242-291",
            "damage_engine/fragments.py:957-983",
            "damage_engine/fragments.py:1018-1029",
        ],
    ),
    rec(
        "blast-response-link",
        "无人机结构",
        "材料响应",
        "冲击波响应模型选择",
        "/{armor|components}/*/response/blast_response/model_id",
        {
            "armor": armor_response["blast_response"]["model_id"],
            "component": component_response["blast_response"]["model_id"],
        },
        "V3固定",
        "模型选择",
        "非空且必须指向 registry.blast_response_models。",
        "选择 P-I-时长损伤状态曲线。",
        [
            "damage_engine/schemas/target.schema.json:265-291",
            "damage_engine/blast.py:298-334",
            "damage_engine/blast.py:409-439",
        ],
    ),
    rec(
        "blast-thresholds",
        "无人机结构",
        "材料响应",
        "局部冲击波阈值覆盖",
        "/{armor|components}/*/response/blast_response/{peak_overpressure_threshold,impulse_threshold,duration_threshold}",
        {
            "component_peak": component_response["blast_response"].get("peak_overpressure_threshold"),
            "impulse": component_response["blast_response"].get("impulse_threshold"),
            "duration": component_response["blast_response"].get("duration_threshold"),
        },
        "可选能力",
        "物理响应 + 不确定性",
        "均为可选参数对象；压力 Pa/kPa/MPa，冲量 Pa*s/kPa*ms/kPa*s/MPa*ms，时长 s/ms；实现要求正值。",
        "覆盖首个损伤状态或装甲 failed 状态的 registry P50/I50/时长尺度。",
        [
            "examples/target.demo.json:99-110",
            "damage_engine/schemas/target.schema.json:265-290",
            "damage_engine/parameters.py:59-82",
            "damage_engine/parameters.py:115-150",
            "damage_engine/blast.py:298-334",
            "damage_engine/blast.py:419-438",
        ],
    ),
    rec(
        "parameter-uncertainty",
        "无人机结构",
        "不确定性",
        "目标参数不确定性模型",
        "/.../uncertainty/{kind,parameters,epistemic}",
        {"demo_kind": "none", "epistemic": True},
        "V3固定",
        "不确定性传播",
        "kind ∈ none/normal/lognormal/uniform/interval/empirical；interval 仅表示边界，不授权随机抽样；lognormal 的 value 是中位数。",
        "epistemic=true 时每案例只抽一次；否则每个 aleatory replicate 抽样。",
        [
            "damage_engine/schemas/target.schema.json:76-114",
            "damage_engine/parameters.py:17-84",
            "damage_engine/parameters.py:87-152",
        ],
    ),
    rec(
        "component-exposure",
        "无人机结构",
        "部件语义",
        "部件舱室与暴露等级",
        "/components/*/{compartment_id,exposure_class}",
        {"compartment_id": component.get("compartment_id"), "exposure_class": component["exposure_class"]},
        "V3固定",
        "物理响应",
        "exposure_class ∈ external/semi_exposed/internal。",
        "external 使用反射载荷；其余经过装甲传递并叠加 compartment_amplification。",
        [
            "examples/target.demo.json:113-115",
            "damage_engine/schemas/target.schema.json:308-343",
            "damage_engine/blast.py:358-408",
        ],
    ),
    rec(
        "component-states",
        "无人机结构",
        "部件语义",
        "部件类型、功能标签和有序损伤状态",
        "/components/*/{component_type,functional_tags,damage_states}",
        {
            "component_type": component["component_type"],
            "functional_tags": component["functional_tags"],
            "damage_states": component["damage_states"],
        },
        "V3固定",
        "标签语义",
        "functional_tags 至少 1 个；damage_states 至少 2 个且唯一，顺序用于强制概率单调。",
        "定义输出标签、响应模型状态匹配以及毁伤树叶节点的可引用状态。",
        [
            "examples/target.demo.json:65-122",
            "damage_engine/schemas/target.schema.json:308-343",
            "damage_engine/validation.py:680-724",
            "damage_engine/fragments.py:1056-1060",
            "damage_engine/blast.py:412-445",
        ],
    ),
    rec(
        "damage-tree",
        "无人机结构",
        "系统毁伤逻辑",
        "毁伤树节点与依赖假设",
        "/damage_tree/nodes/*/{node_type,inputs,component_id,mechanism,damage_state,k,dependency_model_id}",
        {"version": TARGET["damage_tree"]["version"], "node_count": len(TARGET["damage_tree"]["nodes"])},
        "V3固定",
        "系统标签",
        "node_type ∈ leaf/and/or/not/k_of_n/functional/standard_projection；门节点显式声明 independent 或 fully_dependent。",
        "把部件级片段/冲击波/combined 状态映射为系统级任务降级/任务丧失概率。",
        [
            "examples/target.demo.json:124-176",
            "damage_engine/schemas/target.schema.json:356-443",
            "damage_engine/damage_tree.py:207-305",
        ],
    ),
    rec(
        "damage-tree-outputs",
        "无人机结构",
        "系统毁伤逻辑",
        "毁伤树输出和评估标准",
        "/damage_tree/{outputs,standards}",
        {"outputs": TARGET["damage_tree"]["outputs"], "requested": analysis["evaluation_standard_ids"]},
        "V3固定",
        "系统标签",
        "outputs 至少 1 个且唯一；analysis.evaluation_standard_ids 必须引用已声明标准/树版本。",
        "控制最终 system_y/评估标准投影的集合。",
        [
            "damage_engine/schemas/target.schema.json:445-480",
            "damage_engine/schemas/simulation_case.schema.json:233-289",
            "damage_engine/engine.py:412-416",
            "damage_engine/engine.py:841-873",
        ],
    ),
    rec(
        "uav-provenance",
        "无人机结构",
        "证据与边界",
        "几何/材料来源、校准和验证状态",
        "/{armor|components}/*/{provenance,response/*/{source_id,calibration_id,validation_status,valid_range}}",
        {"armor_status": armor["provenance"]["status"], "parameter_status": "unverified"},
        "V3固定",
        "证据/追溯",
        "provenance.status ∈ unverified/reviewed/approved；parameter.validation_status ∈ unverified/calibrated/validated。",
        "不直接改变求解公式，但决定结果可否被当作校准/验证证据。",
        [
            "damage_engine/schemas/target.schema.json:89-114",
            "damage_engine/schemas/target.schema.json:345-354",
            "examples/target.demo.json:32-61",
        ],
    ),
    rec(
        "warhead-masses",
        "弹体",
        "弹体质量",
        "装药质量与壳体质量",
        "/warhead/{charge_mass_kg,casing_mass_kg}",
        {"charge_mass_kg": warhead["charge_mass_kg"], "casing_mass_kg": warhead["casing_mass_kg"]},
        "V3固定",
        "物理响应",
        "可选字段但 Gurney 圆柱模型要求二者 > 0；还参与破片金属质量预算。",
        "决定 Gurney 初速和允许的破片总质量比例。",
        [
            "raw_data/case.5000.uav_control.v3.json:15-20",
            "damage_engine/schemas/simulation_case.schema.json:209-221",
            "damage_engine/fragments.py:515-542",
            "damage_engine/fragments.py:737-751",
        ],
    ),
    rec(
        "fragment-count",
        "弹体",
        "破片源",
        "破片总数",
        "/warhead/fragment_source/total_count",
        fragment["total_count"],
        "V3固定",
        "物理响应 + 性能",
        "整数且 ≥ 1；外部 fragment_list 展开后的总 multiplicity 必须完全相等。",
        "线性改变射线追踪工作量和潜在命中次数。",
        [
            "raw_data/case.5000.uav_control.v3.json:20-35",
            "damage_engine/schemas/simulation_case.schema.json:164-189",
            "damage_engine/fragments.py:545-616",
            "damage_engine/fragments.py:477-488",
        ],
    ),
    rec(
        "fragment-mass-distribution",
        "弹体",
        "破片源",
        "破片质量分布族与参数",
        "/warhead/fragment_source/mass_distribution/{family,parameters}",
        fragment["mass_distribution"],
        "V3固定",
        "物理响应 + 随机性",
        "fixed:value_kg；lognormal:median_kg/log_sigma；weibull:scale_kg/shape；empirical:values_kg/probabilities，质量均 > 0。",
        "决定每枚破片质量、等效迎风面积、动能、V50 质量项与总金属预算。",
        [
            "raw_data/case.5000.uav_control.v3.json:20-28",
            "damage_engine/schemas/simulation_case.schema.json:74-162",
            "damage_engine/fragments.py:545-616",
        ],
    ),
    rec(
        "fragment-mass-range",
        "弹体",
        "破片源",
        "破片质量有效范围",
        "/warhead/fragment_source/mass_distribution/valid_range_kg",
        fragment["mass_distribution"].get("valid_range_kg"),
        "V3固定",
        "域门控",
        "两个正数且 lower ≤ upper；超出时 fail closed，不做截断。",
        "限制质量分布/外部清单的适用域。",
        [
            "damage_engine/schemas/simulation_case.schema.json:82-90",
            "damage_engine/fragments.py:320-334",
            "damage_engine/fragments.py:604-615",
        ],
    ),
    rec(
        "fragment-shape-density",
        "弹体",
        "破片源",
        "破片形状类别与材料密度",
        "/warhead/fragment_source/{shape_class,density_kg_m3}",
        {"shape_class": fragment["shape_class"], "density_kg_m3": fragment["density_kg_m3"]},
        "V3固定",
        "物理响应",
        "shape_class ∈ sphere/cube/cylinder/irregular/empirical；density > 0。",
        "密度与 area_shape_factor 共同决定等效迎风面积；shape_class 当前主要保留分类/追溯语义。",
        [
            "damage_engine/schemas/simulation_case.schema.json:164-189",
            "damage_engine/fragments.py:619-632",
            "damage_engine/fragments.py:803-825",
        ],
    ),
    rec(
        "fragment-model-links",
        "弹体",
        "破片源",
        "破片源/方向/阻力/初速模型标识",
        "/warhead/fragment_source/{model_id,drag_model_id,direction_model_id,initial_velocity_model_id}",
        {key: fragment[key] for key in ("model_id", "drag_model_id", "direction_model_id", "initial_velocity_model_id")},
        "V3固定",
        "模型选择",
        "均为非空字符串；model_id 必须指向 physics registry。",
        "model_id 选择实际嵌套源模型；其余三个 ID 由输入契约记录模型声明并参与一致性检查。",
        [
            "damage_engine/schemas/simulation_case.schema.json:164-189",
            "damage_engine/fragments.py:683-686",
            "damage_engine/validation.py:430-528",
        ],
    ),
    rec(
        "gurney-constant",
        "弹体",
        "破片初速",
        "Gurney 常数",
        "/warhead/fragment_source/gurney_constant_mps",
        fragment.get("gurney_constant_mps"),
        "V3固定",
        "物理响应",
        "Gurney 初速模型下必须为有限正数。",
        "与装药/壳体/破片质量一起决定破片初速。",
        [
            "raw_data/case.5000.uav_control.v3.json:20-35",
            "damage_engine/schemas/simulation_case.schema.json:164-189",
            "damage_engine/fragments.py:737-751",
        ],
    ),
    rec(
        "detonation-velocity",
        "弹体",
        "破片方向",
        "爆速",
        "/warhead/fragment_source/detonation_velocity_mps",
        fragment.get("detonation_velocity_mps"),
        "V3固定",
        "物理响应",
        "可选正数；taylor_ring 方向模型必须提供。",
        "与 Gurney 速度共同给出 Taylor 环的极角中心。cone/isotropic 下不使用。",
        [
            "damage_engine/schemas/simulation_case.schema.json:164-189",
            "damage_engine/fragments.py:635-673",
            "damage_engine/fragments.py:761-768",
        ],
    ),
    rec(
        "fragment-list-ref",
        "弹体",
        "确定性破片清单",
        "外部破片清单引用",
        "/warhead/fragment_source/fragment_list_ref",
        fragment.get("fragment_list_ref"),
        "可选能力",
        "物理响应 + 追溯",
        "null 或本地非空路径；支持 UTF-8 CSV/JSON/JSONL；内容哈希进入输入身份。",
        "启用后绕过随机质量/方向/初速生成，直接使用联合破片记录。",
        [
            "damage_engine/schemas/simulation_case.schema.json:164-189",
            "damage_engine/fragments.py:281-318",
            "damage_engine/fragments.py:683-716",
            "damage_engine/engine.py:273-280",
        ],
    ),
    rec(
        "fragment-list-fields",
        "弹体",
        "确定性破片清单",
        "单枚破片联合字段",
        "fragment_list/*/{id,mass_kg,direction_body_xyz,initial_velocity_mps,multiplicity}",
        "V3 未启用 fragment_list_ref",
        "可选能力",
        "物理响应",
        "mass/speed > 0；方向为三维单位向量；multiplicity 为正整数；ID 唯一；展开数必须等于 total_count。",
        "同时锁定单枚破片质量、弹体系方向、初速和重复数，保留它们的联合关系。",
        [
            "damage_engine/fragments.py:34-74",
            "damage_engine/fragments.py:95-172",
            "damage_engine/fragments.py:339-488",
        ],
    ),
    rec(
        "blast-equivalent-mass",
        "弹体",
        "冲击波源",
        "等效爆炸质量",
        "/warhead/blast_source/equivalent_mass_kg",
        blast_source["equivalent_mass_kg"],
        "V3固定",
        "物理响应",
        "有限正数。",
        "进入 Hopkinson 型尺度距离 r/W^(1/3)，并缩放冲量与正相持续时间。",
        [
            "raw_data/case.5000.uav_control.v3.json:37-43",
            "damage_engine/schemas/simulation_case.schema.json:191-207",
            "damage_engine/blast.py:30-89",
            "damage_engine/blast.py:246-269",
        ],
    ),
    rec(
        "blast-model-links",
        "弹体",
        "冲击波源",
        "冲击波场模型与等效基准",
        "/warhead/blast_source/{model_id,equivalency_basis,pressure_time_model_id,source_id}",
        blast_source,
        "V3固定",
        "模型选择 + 追溯",
        "model/source/basis 非空；pressure_time_model_id 当前只能 scaled-table。",
        "选择尺度距离表，并记录等效质量的依据。",
        [
            "damage_engine/schemas/simulation_case.schema.json:191-207",
            "damage_engine/blast.py:246-250",
        ],
    ),
    rec(
        "warhead-identity",
        "弹体",
        "身份与版本",
        "弹体标识、修订版与来源",
        "/warhead/{warhead_id,revision,source_id}",
        {key: warhead[key] for key in ("warhead_id", "revision", "source_id")},
        "V3固定",
        "身份/追溯",
        "非空字符串。",
        "不直接进入物理公式，但进入 case 哈希、归档和场景冻结。",
        [
            "raw_data/case.5000.uav_control.v3.json:15-45",
            "damage_engine/schemas/simulation_case.schema.json:209-221",
            "damage_engine/learning_input_identity.py:140-172",
        ],
    ),
    rec(
        "source-initial-velocity",
        "物理模型",
        "破片源模型",
        "初速模型类型与固定速度",
        "/fragment_source_models/*/initial_velocity/{kind,velocity_mps}",
        fragment_source_model["initial_velocity"],
        "标定模型",
        "物理响应",
        "kind ∈ fixed/gurney_cylinder；fixed 要求 velocity_mps > 0。",
        "选择固定初速或 Gurney 圆柱公式。",
        [
            "examples/physics.demo.json:5-23",
            "damage_engine/schemas/physics_registry.schema.json:49-76",
            "damage_engine/fragments.py:737-753",
        ],
    ),
    rec(
        "source-direction",
        "物理模型",
        "破片源模型",
        "方向分布类型、轴、锥角与环带散布",
        "/fragment_source_models/*/direction/{kind,axis_body,half_angle_deg,spread_sigma_deg}",
        fragment_source_model["direction"],
        "标定模型",
        "物理响应 + 随机性",
        "kind ∈ isotropic/cone/taylor_ring；axis 为三维向量；锥半角 (0,180]；sigma ≥ 0。",
        "决定破片在弹体系内的方向分布，再由 encounter 四元数旋入目标系。",
        [
            "examples/physics.demo.json:8-13",
            "damage_engine/schemas/physics_registry.schema.json:77-102",
            "damage_engine/fragments.py:635-673",
        ],
    ),
    rec(
        "source-drag",
        "物理模型",
        "破片源模型",
        "阻力系数与迎风面积形状因子",
        "/fragment_source_models/*/drag/{kind,coefficient,area_shape_factor}",
        fragment_source_model["drag"],
        "标定模型",
        "物理响应",
        "kind 当前只能 constant_cd；coefficient 和 area_shape_factor > 0。",
        "共同决定破片面积和指数速度衰减。",
        [
            "examples/physics.demo.json:14-18",
            "damage_engine/schemas/physics_registry.schema.json:103-112",
            "damage_engine/fragments.py:619-632",
            "damage_engine/fragments.py:803-845",
        ],
    ),
    rec(
        "source-mass-budget",
        "物理模型",
        "破片源模型",
        "壳体质量口径与最大破片金属比例",
        "/fragment_source_models/*/{casing_mass_basis,maximum_fragment_mass_fraction_of_total_metal}",
        {
            "casing_mass_basis": fragment_source_model["casing_mass_basis"],
            "max_fraction": fragment_source_model["maximum_fragment_mass_fraction_of_total_metal"],
        },
        "标定模型",
        "域门控",
        "basis ∈ includes_fragments/excludes_fragments；最大比例在 (0,1]。",
        "防止破片质量与壳体质量口径不一致；超预算时拒绝案例。",
        [
            "damage_engine/schemas/physics_registry.schema.json:113-125",
            "damage_engine/fragments.py:515-542",
        ],
    ),
    rec(
        "source-valid-speed",
        "物理模型",
        "破片源模型",
        "破片速度有效域",
        "/fragment_source_models/*/valid_speed_mps",
        fragment_source_model["valid_speed_mps"],
        "标定模型",
        "域门控",
        "两个数；运行时要求生成/清单速度落在范围内，否则 fail closed。",
        "限定源模型适用的速度范围。",
        [
            "damage_engine/schemas/physics_registry.schema.json:119-125",
            "damage_engine/fragments.py:718-733",
            "damage_engine/fragments.py:754-800",
        ],
    ),
    rec(
        "ballistic-limit",
        "物理模型",
        "破片响应模型",
        "弹道极限系数与指数",
        "/fragment_response_models/*/ballistic_limit/{coefficient,thickness_exponent,mass_exponent,obliquity_exponent}",
        {
            "armor": armor_fragment_model["ballistic_limit"],
            "component": component_fragment_model["ballistic_limit"],
        },
        "标定模型",
        "物理响应",
        "coefficient > 0；三个指数为数值。",
        "计算 V50 = C·t^a·m^b·sec(theta)^c，是破片穿透概率的核心模型。",
        [
            "examples/physics.demo.json:25-54",
            "damage_engine/schemas/physics_registry.schema.json:128-167",
            "damage_engine/fragments.py:848-880",
        ],
    ),
    rec(
        "ballistic-transition",
        "物理模型",
        "破片响应模型",
        "穿透转变宽度与最大斜侵角",
        "/fragment_response_models/*/ballistic_limit/{transition_width_ratio,maximum_obliquity_deg}",
        {
            "armor": {
                "transition": armor_fragment_model["ballistic_limit"]["transition_width_ratio"],
                "max_deg": armor_fragment_model["ballistic_limit"]["maximum_obliquity_deg"],
            },
            "component": {
                "transition": component_fragment_model["ballistic_limit"]["transition_width_ratio"],
                "max_deg": component_fragment_model["ballistic_limit"]["maximum_obliquity_deg"],
            },
        },
        "标定模型",
        "物理响应",
        "transition > 0；最大角 (0,90]°。",
        "控制 V/V50 附近 sigmoid 陡峭度；超过最大角直接判定不穿透。",
        [
            "damage_engine/schemas/physics_registry.schema.json:133-151",
            "damage_engine/fragments.py:860-880",
        ],
    ),
    rec(
        "fragment-state-response",
        "物理模型",
        "破片响应模型",
        "破片残余能量毁伤状态曲线",
        "/fragment_response_models/*/damage_states/*/{residual_energy_50_j,log_slope}",
        component_fragment_model["damage_states"],
        "标定模型",
        "物理响应",
        "每个状态 residual_energy_50_j、log_slope 均 > 0；状态必须与部件 damage_states 对齐。",
        "穿透后按残余动能给出部件降级/失效概率。",
        [
            "examples/physics.demo.json:48-53",
            "damage_engine/schemas/physics_registry.schema.json:153-167",
            "damage_engine/fragments.py:1014-1029",
        ],
    ),
    rec(
        "hole-multiplier",
        "物理模型",
        "破片响应模型",
        "装甲孔洞面积倍率",
        "/fragment_response_models/*/hole_area_multiplier",
        {"armor": armor_fragment_model.get("hole_area_multiplier"), "component": component_fragment_model.get("hole_area_multiplier")},
        "标定模型",
        "耦合响应",
        "可选且 > 0，运行默认 1。",
        "破片先穿甲时扩大孔洞面积，改变 sequential 模式下后续冲击波穿透。",
        [
            "damage_engine/schemas/physics_registry.schema.json:153-167",
            "damage_engine/fragments.py:976-983",
        ],
    ),
    rec(
        "blast-scaled-table",
        "物理模型",
        "冲击波场模型",
        "尺度距离—压力/冲量/时长表",
        "/blast_field_models/*/scaled_distance_table/*/{z_m_kg13,peak_pressure_pa,scaled_impulse_pa_s_kgm13,scaled_duration_s_kgm13}",
        blast_field_model["scaled_distance_table"],
        "标定模型",
        "物理响应",
        "至少 2 行，所有量 > 0；插值要求适用域内的正值。",
        "经 log-log 插值得到自由场峰值压力、冲量和正相持续时间。",
        [
            "examples/physics.demo.json:56-64",
            "damage_engine/schemas/physics_registry.schema.json:169-203",
            "damage_engine/blast.py:30-89",
        ],
    ),
    rec(
        "blast-reflection",
        "物理模型",
        "冲击波场模型",
        "反射系数模型",
        "/blast_field_models/*/reflection/{minimum_factor,maximum_factor,pressure_scale_pa,angular_exponent}",
        blast_field_model["reflection"],
        "标定模型",
        "物理响应",
        "min/max ≥ 0，pressure_scale > 0，angular_exponent ≥ 0。",
        "根据入射超压和夹角调节反射峰压与冲量。",
        [
            "examples/physics.demo.json:65-70",
            "damage_engine/schemas/physics_registry.schema.json:205-215",
            "damage_engine/blast.py:92-113",
        ],
    ),
    rec(
        "compartment-amplification",
        "物理模型",
        "冲击波场模型",
        "舱室放大系数",
        "/blast_field_models/*/compartment_amplification/{external,semi_exposed,internal}",
        blast_field_model["compartment_amplification"],
        "标定模型",
        "物理响应",
        "各值 ≥ 0；缺省运行值为 1。",
        "对非 external 部件的传递后压力和冲量进行乘法放大。",
        [
            "examples/physics.demo.json:71-74",
            "damage_engine/schemas/physics_registry.schema.json:216-224",
            "damage_engine/blast.py:392-408",
        ],
    ),
    rec(
        "gas-properties",
        "物理模型",
        "冲击波场模型",
        "气体比热比、气体常数与参考温度",
        "/blast_field_models/*/{heat_capacity_ratio,specific_gas_constant_j_kgk,reference_temperature_k}",
        {key: blast_field_model[key] for key in ("heat_capacity_ratio", "specific_gas_constant_j_kgk", "reference_temperature_k")},
        "标定模型",
        "物理响应",
        "gamma > 1，R > 0，Tref > 0。案例 temperature_k 存在时覆盖参考温度。",
        "决定音速，进而决定冲击波到达时间。",
        [
            "examples/physics.demo.json:75-78",
            "damage_engine/schemas/physics_registry.schema.json:169-203",
            "damage_engine/blast.py:70-89",
        ],
    ),
    rec(
        "blast-state-response",
        "物理模型",
        "冲击波响应模型",
        "P-I-时长毁伤曲线",
        "/blast_response_models/*/damage_states/*/{peak_50_pa,impulse_50_pa_s,peak_exponent,impulse_exponent,duration_50_s,duration_exponent,log_slope}",
        component_blast_model["damage_states"],
        "标定模型",
        "物理响应",
        "P50/I50/指数/log_slope > 0；时长尺度与时长指数必须成对出现。",
        "用压力、冲量、持续时间和易损比例计算部件/装甲状态概率。",
        [
            "examples/physics.demo.json:81-108",
            "damage_engine/schemas/physics_registry.schema.json:225-259",
            "damage_engine/blast.py:181-234",
            "damage_engine/blast.py:409-439",
        ],
    ),
    rec(
        "transmission",
        "物理模型",
        "遮挡传递模型",
        "峰压/冲量半值厚度与最小透射",
        "/transmission_models/*/{peak_half_thickness_m,impulse_half_thickness_m,minimum_transmission}",
        transmission_model,
        "标定模型",
        "物理响应",
        "两个半值厚度 > 0；minimum_transmission ∈ [0,1]。",
        "通过装甲路径厚度、入射角和完整度衰减峰压与冲量。",
        [
            "examples/physics.demo.json:110-116",
            "damage_engine/schemas/physics_registry.schema.json:260-274",
            "damage_engine/blast.py:116-178",
        ],
    ),
    rec(
        "registry-status",
        "物理模型",
        "身份与证据",
        "物理注册表标识与验证状态",
        "/{config_id,validation_status}",
        {"config_id": PHYSICS["config_id"], "validation_status": PHYSICS["validation_status"]},
        "V3固定",
        "证据/追溯",
        "validation_status ∈ unverified/calibrated/validated。",
        "不改变数值公式，但引擎会对非 validated 注册表发出“不可视为现实认证”的警告。",
        [
            "examples/physics.demo.json:1-5",
            "damage_engine/schemas/physics_registry.schema.json:1-46",
            "damage_engine/engine.py:417-419",
            "damage_engine/engine.py:915-932",
        ],
    ),
    rec(
        "air-density",
        "环境",
        "大气",
        "空气密度",
        "/environment/air_density_kg_m3",
        environment["air_density_kg_m3"],
        "V3固定",
        "物理响应",
        "有限正数。",
        "线性进入破片阻力衰减系数；当前不进入冲击波自由场标度。",
        [
            "raw_data/case.5000.uav_control.v3.json:46-50",
            "damage_engine/schemas/simulation_case.schema.json:223-231",
            "damage_engine/fragments.py:829-845",
            "damage_engine/fragments.py:926-928",
        ],
    ),
    rec(
        "ambient-pressure",
        "环境",
        "大气",
        "环境压力",
        "/environment/ambient_pressure_pa",
        environment["ambient_pressure_pa"],
        "V3固定",
        "物理响应",
        "有限正数。",
        "进入冲击波速度修正和反射强度尺度。",
        [
            "raw_data/case.5000.uav_control.v3.json:46-50",
            "damage_engine/schemas/simulation_case.schema.json:223-231",
            "damage_engine/blast.py:30-89",
            "damage_engine/blast.py:92-113",
        ],
    ),
    rec(
        "temperature",
        "环境",
        "大气",
        "环境温度",
        "/environment/temperature_k",
        environment["temperature_k"],
        "V3固定",
        "物理响应",
        "有限正数。",
        "覆盖场模型参考温度，用于音速与冲击波到达时间；不改变当前压力/冲量表值。",
        [
            "raw_data/case.5000.uav_control.v3.json:46-50",
            "damage_engine/schemas/simulation_case.schema.json:223-231",
            "damage_engine/blast.py:70-89",
            "damage_engine/blast.py:246-269",
        ],
    ),
    rec(
        "fidelity",
        "求解控制",
        "机制选择",
        "保真级别",
        "/analysis/fidelity_level",
        analysis["fidelity_level"],
        "V3固定",
        "身份/追溯",
        "L0/L1/L2/L3。",
        "当前运行核心未按该值切换算法；它作为契约/追溯字段存在。",
        [
            "raw_data/case.5000.uav_control.v3.json:51-61",
            "damage_engine/schemas/simulation_case.schema.json:233-290",
        ],
        note="修改 fidelity_level 本身不会自动启用更高阶物理模型。",
    ),
    rec(
        "mechanisms",
        "求解控制",
        "机制选择",
        "启用毁伤机制",
        "/analysis/mechanisms",
        analysis["mechanisms"],
        "V3固定",
        "求解路径",
        "至少 1 个，唯一；元素只能 fragment/blast。",
        "决定执行破片、冲击波或二者；未启用机制输出归零。",
        [
            "damage_engine/schemas/simulation_case.schema.json:233-290",
            "damage_engine/engine.py:273-350",
        ],
    ),
    rec(
        "randomness",
        "求解控制",
        "随机性",
        "偶然随机种子与认知样本标识",
        "/analysis/{aleatory_seed,epistemic_sample_id}",
        {"aleatory_seed": analysis["aleatory_seed"], "epistemic_sample_id": analysis["epistemic_sample_id"]},
        "V3固定",
        "随机性/复现",
        "seed 为非负整数；sample_id 非空。",
        "aleatory_seed 派生复制级参数/破片/冲击波 RNG；epistemic_sample_id 与 target_hash 派生案例级认知 RNG。",
        [
            "damage_engine/schemas/simulation_case.schema.json:233-290",
            "damage_engine/engine.py:281-321",
            "damage_engine/engine.py:937-953",
        ],
    ),
    rec(
        "replicates",
        "求解控制",
        "蒙特卡洛",
        "最大复制数与概率 CI 自适应阈值",
        "/analysis/{replicates,adaptive_ci_half_width}",
        {"template_replicates": analysis["replicates"], "adaptive_ci_half_width": analysis.get("adaptive_ci_half_width"), "V3_role_policy": "train=4, validation/test=8, geometry-stratified=32"},
        "V3固定",
        "数值收敛 + 性能",
        "replicates ≥ 1；启用 adaptive_ci_half_width 时 replicates ≥ 32，阈值在 (0,1)。",
        "控制最大蒙特卡洛次数；满足概率半宽与物理指标半宽后可提前停止。",
        [
            "raw_data/sampling_spec.5000.uav_control.v3.json:164-171",
            "damage_engine/schemas/simulation_case.schema.json:233-302",
            "damage_engine/engine.py:281-321",
            "damage_engine/engine.py:404-475",
        ],
    ),
    rec(
        "physical-ci",
        "求解控制",
        "蒙特卡洛",
        "物理量自适应 CI 容差",
        "/analysis/adaptive_physical_ci/<metric>/{absolute_half_width,relative_half_width,relative_floor}",
        analysis.get("adaptive_physical_ci"),
        "可选能力",
        "数值收敛",
        "至少绝对或相对半宽之一；正值；选择器必须属于引擎支持的片段/冲击波/装甲/摘要指标。",
        "按带单位物理量逐序列检查 95% CI，而非只看概率标签。",
        [
            "damage_engine/schemas/simulation_case.schema.json:261-280",
            "damage_engine/engine.py:44-77",
            "damage_engine/engine.py:404-475",
            "damage_engine/engine.py:525-718",
        ],
    ),
    rec(
        "coupling",
        "求解控制",
        "机制耦合",
        "破片—冲击波耦合方式",
        "/analysis/coupling_model",
        analysis["coupling_model"],
        "V3固定",
        "耦合响应",
        "none 或 sequential。",
        "sequential 把破片后的装甲完整度传给冲击波传递；部件状态概率仍在每复制内按条件独立合并，并输出 Fréchet 联合边界。",
        [
            "damage_engine/schemas/simulation_case.schema.json:281-283",
            "damage_engine/engine.py:335-370",
            "damage_engine/engine.py:417-426",
        ],
    ),
    rec(
        "capture-flags",
        "求解控制",
        "诊断输出",
        "破片审计与复制轨迹开关",
        "/analysis/{capture_fragment_audit,capture_replicate_trace}",
        {"capture_fragment_audit": analysis.get("capture_fragment_audit"), "capture_replicate_trace": analysis.get("capture_replicate_trace")},
        "V3固定",
        "输出体积 + 可审计性",
        "布尔值，默认 false。",
        "控制逐命中审计事件和逐复制联合输出；不改变物理计算本身，但显著改变归档体积。",
        [
            "damage_engine/schemas/simulation_case.schema.json:281-289",
            "damage_engine/engine.py:288-289",
            "damage_engine/engine.py:397-403",
            "damage_engine/engine.py:505-508",
        ],
    ),
    rec(
        "evaluation-standards",
        "求解控制",
        "输出选择",
        "评估标准 ID",
        "/analysis/evaluation_standard_ids",
        analysis["evaluation_standard_ids"],
        "V3固定",
        "输出选择",
        "至少 1 个、唯一、非空；必须在毁伤树标准集合中可解析。",
        "决定结果中 evaluation_standards 的投影，不改变底层命中/压力计算。",
        [
            "damage_engine/schemas/simulation_case.schema.json:284-289",
            "damage_engine/engine.py:412-416",
            "damage_engine/engine.py:841-873",
        ],
    ),
    rec(
        "parameter-set-id",
        "求解控制",
        "身份与证据",
        "参数集标识",
        "/parameter_set_id",
        CASE["parameter_set_id"],
        "V3固定",
        "身份/追溯",
        "非空字符串。",
        "进入案例语义身份和归档，用于区分不同参数快照；不直接进入公式。",
        [
            "raw_data/case.5000.uav_control.v3.json:63-69",
            "damage_engine/schemas/simulation_case.schema.json:7-35",
            "damage_engine/input_identity.py:283-365",
        ],
    ),
]


SOURCE_PATTERN = re.compile(r"^(.*?):(\d+)(?:-(\d+))?$")


def validate_source_locations(records: list[dict[str, Any]]) -> None:
    errors: list[str] = []
    cache: dict[Path, int] = {}
    for record in records:
        for source in record["sources"]:
            match = SOURCE_PATTERN.match(source)
            if not match:
                errors.append(f"invalid source syntax: {source}")
                continue
            relative, start_raw, end_raw = match.groups()
            path = ROOT / relative
            if not path.is_file():
                errors.append(f"missing source file: {relative}")
                continue
            count = cache.setdefault(path, len(path.read_text(encoding="utf-8").splitlines()))
            start = int(start_raw)
            end = int(end_raw or start_raw)
            if start < 1 or end < start or end > count:
                errors.append(f"out-of-range source: {source} (file has {count} lines)")
    if errors:
        raise RuntimeError("source-location validation failed:\n" + "\n".join(errors))


validate_source_locations(RECORDS)

sample_variables = SAMPLING["reference_distribution"]["variables"]
training_feature_names = [
    name
    for role in SAMPLING["feature_roles"]
    for name in role["tensor_feature_names"]
]
frozen_hashes_path = ROOT / "raw_data" / "dataset_5000_uav_control_v3" / "plan" / "frozen_hashes.json"
frozen_hashes = load_json(frozen_hashes_path) if frozen_hashes_path.exists() else {}

DATA = {
    "generatedOn": date.today().isoformat(),
    "title": "仿真引擎可变参数与代码位置",
    "engineVersion": "0.7.0.dev3",
    "currentScenario": "UAV-control 5000 v3",
    "records": RECORDS,
    "counts": {
        "records": len(RECORDS),
        "sampleDeclarations": len(sample_variables),
        "rawUavDimensions": 10,
        "trainingDimensions": len(training_feature_names),
        "layers": len({item["layer"] for item in RECORDS}),
    },
    "trainingFeatureNames": training_feature_names,
    "sampleVariables": sample_variables,
    "frozenHashes": {
        key: frozen_hashes.get(key)
        for key in (
            "target_hash",
            "physics_registry_hash",
            "engine_code_hash",
            "damage_tree_definition_hash",
            "base_case_hash",
            "sampling_spec_hash",
        )
    },
    "layerCounts": Counter(item["layer"] for item in RECORDS),
    "statusCounts": Counter(item["status"] for item in RECORDS),
    "sources": [
        {
            "role": "当前 V3 基础案例",
            "path": "raw_data/case.5000.uav_control.v3.json",
            "scope": "遭遇、弹体、环境、分析控制的当前固定值",
        },
        {
            "role": "当前 V3 采样设计",
            "path": "raw_data/sampling_spec.5000.uav_control.v3.json",
            "scope": "实际变化的 UAV 位置、速度和姿态范围",
        },
        {
            "role": "目标/UAV 基线",
            "path": "examples/target.demo.json",
            "scope": "当前冻结目标哈希对应的合成装甲、部件和毁伤树",
        },
        {
            "role": "物理模型基线",
            "path": "examples/physics.demo.json",
            "scope": "当前合成破片、冲击波、响应和传递系数",
        },
        {
            "role": "案例输入契约",
            "path": "damage_engine/schemas/simulation_case.schema.json",
            "scope": "遭遇/弹体/环境/分析字段与硬约束",
        },
        {
            "role": "目标输入契约",
            "path": "damage_engine/schemas/target.schema.json",
            "scope": "UAV 几何、材料、不确定性和毁伤树字段",
        },
        {
            "role": "物理注册表契约",
            "path": "damage_engine/schemas/physics_registry.schema.json",
            "scope": "标定模型系数与表格字段",
        },
        {
            "role": "运行时入口",
            "path": "damage_engine/engine.py",
            "scope": "字段如何驱动 Monte Carlo、机制耦合、收敛与输出",
        },
    ],
}


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>仿真引擎可变参数与代码位置</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #667085;
      --line: #d8dee9;
      --navy: #123b6d;
      --blue: #1f6feb;
      --cyan: #148aa3;
      --amber: #b86b00;
      --green: #18794e;
      --purple: #6941c6;
      --red: #b42318;
      --sample: #e8f1ff;
      --fixed: #eef8f2;
      --optional: #fff6df;
      --calibration: #f4edff;
      --shadow: 0 12px 34px rgba(16, 24, 40, .08);
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      background: radial-gradient(circle at 12% -10%, #e5efff 0, transparent 34rem), var(--bg);
      color: var(--ink);
      font-family: Inter, "Segoe UI", "Microsoft YaHei", system-ui, sans-serif;
      line-height: 1.55;
    }
    button, input, select { font: inherit; }
    button { cursor: pointer; }
    code { font-family: "Cascadia Code", Consolas, monospace; }
    .shell { max-width: 1480px; margin: 0 auto; padding: 28px 24px 56px; }
    .hero {
      position: relative;
      overflow: hidden;
      padding: 30px;
      border-radius: 24px;
      color: #fff;
      background: linear-gradient(125deg, #102f55, #145c83 58%, #1c7b8f);
      box-shadow: var(--shadow);
    }
    .hero::after {
      content: "";
      position: absolute;
      width: 460px;
      height: 460px;
      border-radius: 50%;
      right: -170px;
      top: -260px;
      border: 80px solid rgba(255,255,255,.08);
    }
    .eyebrow { letter-spacing: .08em; text-transform: uppercase; font-size: 12px; opacity: .8; }
    h1 { margin: 5px 0 8px; font-size: clamp(30px, 4vw, 50px); line-height: 1.08; }
    .hero p { max-width: 980px; margin: 0; color: rgba(255,255,255,.86); }
    .hero-meta { display: flex; flex-wrap: wrap; gap: 10px 18px; margin-top: 20px; font-size: 13px; }
    .hero-meta span { padding: 5px 9px; border: 1px solid rgba(255,255,255,.22); border-radius: 999px; background: rgba(255,255,255,.08); }
    .stats { display: grid; grid-template-columns: repeat(5, minmax(0,1fr)); gap: 14px; margin: 18px 0; }
    .stat, .panel { background: var(--panel); border: 1px solid var(--line); box-shadow: var(--shadow); }
    .stat { padding: 18px; border-radius: 16px; }
    .stat .value { font-size: 30px; font-weight: 760; color: var(--navy); line-height: 1; }
    .stat .label { margin-top: 8px; color: var(--muted); font-size: 13px; }
    .panel { margin-top: 18px; padding: 22px; border-radius: 18px; }
    .panel h2 { margin: 0 0 5px; font-size: 22px; }
    .panel-sub { margin: 0 0 17px; color: var(--muted); font-size: 14px; }
    .scope-grid { display: grid; grid-template-columns: 1.1fr 1fr; gap: 22px; align-items: start; }
    .flow { display: grid; grid-template-columns: 1fr 44px 1fr 44px 1fr; align-items: stretch; gap: 8px; }
    .flow-card { min-height: 150px; padding: 16px; border: 1px solid var(--line); border-radius: 14px; background: #fbfcff; }
    .flow-card strong { display: block; color: var(--navy); margin-bottom: 8px; }
    .flow-card ul { margin: 0; padding-left: 18px; color: #344054; font-size: 13px; }
    .arrow { display: grid; place-items: center; color: var(--blue); font-size: 26px; font-weight: 800; }
    .scope-note { border-left: 4px solid var(--amber); padding: 14px 16px; background: #fffaf0; border-radius: 0 12px 12px 0; }
    .scope-note strong { display: block; color: #8a4b00; }
    .scope-note p { margin: 6px 0 0; color: #604b2b; font-size: 13px; }
    .bars { display: grid; gap: 9px; }
    .bar-row { display: grid; grid-template-columns: 104px 1fr 34px; gap: 10px; align-items: center; font-size: 13px; }
    .bar-track { height: 10px; background: #edf0f5; border-radius: 99px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--blue), var(--cyan)); }
    .bar-count { text-align: right; font-variant-numeric: tabular-nums; color: var(--muted); }
    .pipeline { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 15px; }
    .step { position: relative; padding: 14px; border-top: 4px solid var(--blue); background: #f8faff; border-radius: 10px; }
    .step .n { color: var(--blue); font-weight: 800; font-size: 12px; }
    .step strong { display: block; margin: 3px 0; }
    .step code { display: block; color: var(--muted); font-size: 11px; word-break: break-all; }
    .toolbar { position: sticky; top: 0; z-index: 5; margin: -4px -4px 16px; padding: 12px 4px; background: rgba(255,255,255,.94); backdrop-filter: blur(10px); }
    .filters { display: grid; grid-template-columns: minmax(220px, 1.6fr) repeat(3, minmax(145px,.65fr)); gap: 10px; }
    input, select { width: 100%; padding: 10px 12px; border: 1px solid #c8d0dc; border-radius: 10px; background: #fff; color: var(--ink); }
    input:focus, select:focus, button:focus-visible { outline: 3px solid rgba(31,111,235,.24); outline-offset: 1px; }
    .filter-summary { display: flex; justify-content: space-between; gap: 12px; margin-top: 9px; color: var(--muted); font-size: 13px; }
    .table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 14px; }
    table { width: 100%; border-collapse: collapse; background: #fff; min-width: 1180px; }
    th { position: sticky; top: 81px; z-index: 2; padding: 11px 12px; text-align: left; color: #475467; background: #f1f4f8; border-bottom: 1px solid var(--line); font-size: 12px; }
    td { vertical-align: top; padding: 12px; border-bottom: 1px solid #e7ebf1; font-size: 13px; }
    tbody tr:hover { background: #f8fbff; }
    tbody tr:last-child td { border-bottom: 0; }
    .param-btn { border: 0; padding: 0; background: transparent; color: var(--navy); font-weight: 720; text-align: left; }
    .path { display: block; margin-top: 5px; color: var(--muted); font-size: 11px; word-break: break-all; }
    .current { max-width: 330px; color: #344054; word-break: break-word; }
    .tag { display: inline-flex; align-items: center; padding: 3px 8px; border-radius: 999px; font-size: 11px; white-space: nowrap; border: 1px solid rgba(16,24,40,.08); }
    .tag.sample { background: var(--sample); color: #174ea6; }
    .tag.fixed { background: var(--fixed); color: var(--green); }
    .tag.optional { background: var(--optional); color: #8a4b00; }
    .tag.calibration { background: var(--calibration); color: var(--purple); }
    .source-list { display: grid; gap: 4px; }
    .source-list code { color: #24517d; font-size: 11px; white-space: nowrap; }
    .drawer { display: none; margin-top: 14px; padding: 20px; border: 1px solid #a9c7ee; border-radius: 14px; background: #f7fbff; }
    .drawer.open { display: block; }
    .drawer-head { display: flex; justify-content: space-between; gap: 14px; align-items: flex-start; }
    .drawer h3 { margin: 0; font-size: 20px; }
    .drawer button { border: 1px solid #c6d3e2; background: #fff; border-radius: 8px; padding: 6px 10px; color: #475467; }
    .detail-grid { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 12px 22px; margin-top: 14px; }
    .detail-block { min-width: 0; }
    .detail-block .k { color: var(--muted); font-size: 12px; margin-bottom: 3px; }
    .detail-block .v { color: #27364a; word-break: break-word; }
    .detail-block.full { grid-column: 1 / -1; }
    .source-index { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 10px; }
    .source-card { padding: 13px; border-left: 3px solid var(--blue); background: #f8faff; border-radius: 0 10px 10px 0; }
    .source-card strong { display: block; }
    .source-card code { display: block; margin: 4px 0; color: #24517d; word-break: break-all; font-size: 12px; }
    .source-card span { color: var(--muted); font-size: 12px; }
    .hashes { display: grid; gap: 6px; }
    .hash-row { display: grid; grid-template-columns: 180px 1fr; gap: 10px; font-size: 12px; }
    .hash-row code { word-break: break-all; color: #344054; }
    .legend { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    .foot { margin-top: 18px; color: var(--muted); font-size: 12px; text-align: center; }
    .hidden { display: none !important; }
    @media (max-width: 980px) {
      .stats { grid-template-columns: repeat(3, 1fr); }
      .scope-grid { grid-template-columns: 1fr; }
      .filters { grid-template-columns: 1fr 1fr; }
      .pipeline { grid-template-columns: 1fr 1fr; }
      th { top: 131px; }
    }
    @media (max-width: 660px) {
      .shell { padding: 14px 12px 38px; }
      .hero, .panel { padding: 17px; border-radius: 15px; }
      .stats { grid-template-columns: 1fr 1fr; }
      .flow { grid-template-columns: 1fr; }
      .arrow { transform: rotate(90deg); min-height: 25px; }
      .filters { grid-template-columns: 1fr; }
      .pipeline, .source-index, .detail-grid { grid-template-columns: 1fr; }
      .detail-block.full { grid-column: auto; }
      .hash-row { grid-template-columns: 1fr; }
      th { position: static; }
      .filter-summary { flex-direction: column; }
    }
    @media print {
      body { background: #fff; }
      .shell { max-width: none; padding: 0; }
      .hero, .panel, .stat { box-shadow: none; }
      .toolbar { position: static; }
      .filters, .drawer, .param-btn { display: none; }
      table { min-width: 0; }
      th { position: static; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header class="hero">
      <div class="eyebrow">Damage Engine · Code-backed Parameter Atlas</div>
      <h1>仿真引擎可变参数与代码位置</h1>
      <p>从当前 UAV-control 5000 v3 场景、三个 JSON Schema 和实际运行时消费代码交叉提取。每一行均给出当前值/状态、约束、影响路径和精确源码行号。</p>
      <div class="hero-meta" id="hero-meta"></div>
    </header>

    <section class="stats" id="stats" aria-label="参数审计概览"></section>

    <section class="panel">
      <h2>先看结论：当前真正变化的只有 UAV 状态</h2>
      <p class="panel-sub">V3 数据生成器冻结目标、弹体、环境和教师求解器，只采样位置、速度与姿态；这和“引擎支持配置”是两个不同层次。</p>
      <div class="scope-grid">
        <div class="flow" aria-label="当前参数流">
          <div class="flow-card">
            <strong>当前采样：UAV 状态</strong>
            <ul>
              <li>位置 3 维：球体体积 r=1.5–5.0 m</li>
              <li>速度 3 维：每轴 −30–30 m/s</li>
              <li>姿态 4 维四元数：Haar SO(3)</li>
            </ul>
          </div>
          <div class="arrow" aria-hidden="true">→</div>
          <div class="flow-card">
            <strong>固定教师场景</strong>
            <ul>
              <li>合成 UAV 装甲/部件几何</li>
              <li>合成弹体、破片与冲击波模型</li>
              <li>固定环境与耦合/复制策略</li>
            </ul>
          </div>
          <div class="arrow" aria-hidden="true">→</div>
          <div class="flow-card">
            <strong>输出</strong>
            <ul>
              <li>12 维训练特征：3+3+6D 姿态</li>
              <li>部件与装甲毁伤概率/物理量</li>
              <li>系统毁伤树输出</li>
            </ul>
          </div>
        </div>
        <div>
          <div class="scope-note">
            <strong>模型边界</strong>
            <p>代码中没有 UAV 质量、惯量、气动导数、升阻力曲线、推进、电机/电池、控制面、角速度或六自由度飞行动力学参数。这里的“无人机参数”是相对位置/速度/姿态，以及作为受毁目标的几何、材料和部件易损性；不能据此声称在做完整飞行动力学仿真。</p>
          </div>
          <div class="legend" aria-label="状态图例">
            <span class="tag sample">当前采样</span>
            <span class="tag fixed">V3固定</span>
            <span class="tag optional">可选能力</span>
            <span class="tag calibration">标定模型</span>
          </div>
          <div class="bars" id="status-bars" style="margin-top:16px"></div>
        </div>
      </div>
    </section>

    <section class="panel">
      <h2>从配置到求解的四层定位</h2>
      <p class="panel-sub">找参数时要区分“当前值”“允许输入”“模型系数”和“运行时真正读取”。</p>
      <div class="pipeline">
        <div class="step"><div class="n">01 · 当前值</div><strong>V3 case / sampling spec</strong><code>raw_data/case.5000.uav_control.v3.json</code></div>
        <div class="step"><div class="n">02 · 输入契约</div><strong>JSON Schema</strong><code>damage_engine/schemas/*.schema.json</code></div>
        <div class="step"><div class="n">03 · 标定模型</div><strong>Physics registry</strong><code>examples/physics.demo.json</code></div>
        <div class="step"><div class="n">04 · 实际消费</div><strong>Runtime</strong><code>engine.py · fragments.py · blast.py</code></div>
      </div>
    </section>

    <section class="panel" id="inventory-panel">
      <h2>参数总表</h2>
      <p class="panel-sub">点击参数名可展开约束、影响说明和全部源码位置。搜索支持中文名、JSON 路径、当前值与文件路径。</p>
      <div class="toolbar">
        <div class="filters">
          <input id="search" type="search" placeholder="搜索参数、路径、值或源码…" aria-label="搜索参数">
          <select id="layer-filter" aria-label="按层级筛选"><option value="">全部层级</option></select>
          <select id="status-filter" aria-label="按状态筛选"><option value="">全部状态</option></select>
          <select id="affect-filter" aria-label="按影响筛选"><option value="">全部影响</option></select>
        </div>
        <div class="filter-summary"><span id="filter-count" aria-live="polite"></span><span>提示：<code>当前采样</code> ≠ 所有可配置参数</span></div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th style="width:14%">层级 / 子类</th>
              <th style="width:21%">参数与 JSON 路径</th>
              <th style="width:22%">当前值 / 配置</th>
              <th style="width:11%">当前状态</th>
              <th style="width:15%">主要影响</th>
              <th style="width:17%">关键代码位置</th>
            </tr>
          </thead>
          <tbody id="parameter-rows"></tbody>
        </table>
      </div>
      <div class="drawer" id="drawer" aria-live="polite"></div>
    </section>

    <section class="panel">
      <h2>源码与配置索引</h2>
      <p class="panel-sub">本报告的“当前值”与“支持范围”来自不同文件，以下是证据角色分工。</p>
      <div class="source-index" id="source-index"></div>
    </section>

    <section class="panel">
      <h2>当前冻结身份</h2>
      <p class="panel-sub">这些哈希把当前目标、物理模型、毁伤树、基础案例和采样计划绑定到 V3 归档；修改参数后应生成新的身份，而不是沿用旧证据。</p>
      <div class="hashes" id="hashes"></div>
    </section>

    <section class="panel">
      <h2>证据边界</h2>
      <div class="scope-grid">
        <div>
          <strong>本 HTML 已确认</strong>
          <ul>
            <li>字段存在于当前工作区的 schema、V3 配置或运行时读取路径。</li>
            <li>表中所有 <code>file:line-line</code> 在生成时都做了文件存在与行号范围校验。</li>
            <li>V3 当前采样域是位置、速度、姿态；其余教师场景参数被冻结。</li>
          </ul>
        </div>
        <div>
          <strong>本 HTML 不证明</strong>
          <ul>
            <li>合成参数具有现实物理有效性或科研准入资格。</li>
            <li>修改任意 schema 字段就一定会改变结果；已明确标出当前未被核心求解器消费的字段。</li>
            <li>当前固定场景能覆盖其他弹体、材料、无人机拓扑或飞行动力学域。</li>
          </ul>
        </div>
      </div>
    </section>
    <div class="foot" id="foot"></div>
  </main>

  <script id="inventory-data" type="application/json">__DATA__</script>
  <script>
    (() => {
      const data = JSON.parse(document.getElementById('inventory-data').textContent);
      const records = data.records;
      const $ = (id) => document.getElementById(id);
      const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
      const statusClass = (status) => status === '当前采样' ? 'sample' : status === 'V3固定' ? 'fixed' : status === '可选能力' ? 'optional' : 'calibration';
      const formatCurrent = (value) => escapeHtml(value).replace(/,/g, ',<wbr>');

      $('hero-meta').innerHTML = [
        `引擎 ${escapeHtml(data.engineVersion)}`,
        escapeHtml(data.currentScenario),
        `生成日期 ${escapeHtml(data.generatedOn)}`,
        `${data.counts.records} 个参数条目`
      ].map(text => `<span>${text}</span>`).join('');

      const stats = [
        [data.counts.sampleDeclarations, 'V3 采样声明（速度按 3 轴拆分）'],
        [data.counts.rawUavDimensions, 'UAV 原始状态维度（3+3+4）'],
        [data.counts.trainingDimensions, '训练特征维度（3+3+6）'],
        [data.counts.records, '已定位参数条目'],
        [data.counts.layers, '参数层级']
      ];
      $('stats').innerHTML = stats.map(([value,label]) => `<div class="stat"><div class="value">${value}</div><div class="label">${escapeHtml(label)}</div></div>`).join('');

      const maxStatus = Math.max(...Object.values(data.statusCounts));
      $('status-bars').innerHTML = Object.entries(data.statusCounts).map(([name,count]) => `
        <div class="bar-row"><span>${escapeHtml(name)}</span><div class="bar-track"><div class="bar-fill" style="width:${Math.max(5, count/maxStatus*100)}%"></div></div><span class="bar-count">${count}</span></div>
      `).join('');

      const layers = [...new Set(records.map(r => r.layer))];
      const statuses = [...new Set(records.map(r => r.status))];
      const affects = [...new Set(records.flatMap(r => r.affects.split(' + ')))].sort();
      $('layer-filter').insertAdjacentHTML('beforeend', layers.map(v => `<option>${escapeHtml(v)}</option>`).join(''));
      $('status-filter').insertAdjacentHTML('beforeend', statuses.map(v => `<option>${escapeHtml(v)}</option>`).join(''));
      $('affect-filter').insertAdjacentHTML('beforeend', affects.map(v => `<option>${escapeHtml(v)}</option>`).join(''));

      const rowHtml = (r) => {
        const sources = r.sources.slice(0,2).map(s => `<code>${escapeHtml(s)}</code>`).join('');
        return `<tr data-id="${escapeHtml(r.id)}">
          <td><strong>${escapeHtml(r.layer)}</strong><br><span class="path">${escapeHtml(r.subgroup)}</span></td>
          <td><button class="param-btn" type="button" data-open="${escapeHtml(r.id)}">${escapeHtml(r.name)}</button><code class="path">${escapeHtml(r.path)}</code></td>
          <td class="current"><code>${formatCurrent(r.current)}</code></td>
          <td><span class="tag ${statusClass(r.status)}">${escapeHtml(r.status)}</span></td>
          <td>${escapeHtml(r.affects)}</td>
          <td><div class="source-list">${sources}${r.sources.length>2 ? `<span class="path">另 ${r.sources.length-2} 处，点击展开</span>` : ''}</div></td>
        </tr>`;
      };

      function filteredRecords() {
        const query = $('search').value.trim().toLowerCase();
        const layer = $('layer-filter').value;
        const status = $('status-filter').value;
        const affect = $('affect-filter').value;
        return records.filter(r => {
          const haystack = [r.layer,r.subgroup,r.name,r.path,r.current,r.status,r.affects,r.constraint,r.effect,r.note,...r.sources].join(' ').toLowerCase();
          return (!query || haystack.includes(query)) && (!layer || r.layer===layer) && (!status || r.status===status) && (!affect || r.affects.includes(affect));
        });
      }

      function renderRows() {
        const shown = filteredRecords();
        $('parameter-rows').innerHTML = shown.map(rowHtml).join('');
        $('filter-count').textContent = `显示 ${shown.length} / ${records.length} 条`;
        $('drawer').classList.remove('open');
        $('drawer').innerHTML = '';
      }

      function openDetail(id) {
        const r = records.find(item => item.id === id);
        if (!r) return;
        $('drawer').innerHTML = `
          <div class="drawer-head"><div><div class="tag ${statusClass(r.status)}">${escapeHtml(r.status)}</div><h3>${escapeHtml(r.name)}</h3><code class="path">${escapeHtml(r.path)}</code></div><button type="button" id="close-drawer">关闭</button></div>
          <div class="detail-grid">
            <div class="detail-block"><div class="k">层级</div><div class="v">${escapeHtml(r.layer)} · ${escapeHtml(r.subgroup)}</div></div>
            <div class="detail-block"><div class="k">影响类型</div><div class="v">${escapeHtml(r.affects)}</div></div>
            <div class="detail-block full"><div class="k">当前值 / 配置</div><div class="v"><code>${formatCurrent(r.current)}</code></div></div>
            <div class="detail-block full"><div class="k">输入约束</div><div class="v">${escapeHtml(r.constraint)}</div></div>
            <div class="detail-block full"><div class="k">进入计算后的作用</div><div class="v">${escapeHtml(r.effect)}</div></div>
            ${r.note ? `<div class="detail-block full"><div class="k">重要说明</div><div class="v">${escapeHtml(r.note)}</div></div>` : ''}
            <div class="detail-block full"><div class="k">全部代码位置</div><div class="source-list">${r.sources.map(s => `<code>${escapeHtml(s)}</code>`).join('')}</div></div>
          </div>`;
        $('drawer').classList.add('open');
        $('close-drawer').addEventListener('click', () => $('drawer').classList.remove('open'));
        $('drawer').scrollIntoView({behavior:'smooth',block:'nearest'});
      }

      ['search','layer-filter','status-filter','affect-filter'].forEach(id => $(id).addEventListener(id==='search'?'input':'change', renderRows));
      $('parameter-rows').addEventListener('click', event => {
        const button = event.target.closest('[data-open]');
        if (button) openDetail(button.dataset.open);
      });
      renderRows();

      $('source-index').innerHTML = data.sources.map(item => `<div class="source-card"><strong>${escapeHtml(item.role)}</strong><code>${escapeHtml(item.path)}</code><span>${escapeHtml(item.scope)}</span></div>`).join('');
      $('hashes').innerHTML = Object.entries(data.frozenHashes).map(([key,value]) => `<div class="hash-row"><strong>${escapeHtml(key)}</strong><code>${escapeHtml(value || '当前文件中不可用')}</code></div>`).join('');
      $('foot').textContent = `基于工作区代码静态审计生成 · ${data.generatedOn} · 物理注册表当前为 unverified，页面仅证明代码参数面与位置，不证明现实有效性。`;
    })();
  </script>
</body>
</html>
'''


def main() -> None:
    data_json = json.dumps(DATA, ensure_ascii=False, separators=(",", ":"))
    html = TEMPLATE.replace("__DATA__", data_json)
    OUTPUT.write_text(html, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "bytes": OUTPUT.stat().st_size,
                "parameter_records": len(RECORDS),
                "source_locations": sum(len(item["sources"]) for item in RECORDS),
                "sample_declarations": len(sample_variables),
                "training_dimensions": len(training_feature_names),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
