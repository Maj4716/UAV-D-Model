const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const TARGET_PATH = path.join(ROOT, "raw_data", "target.rebuilt.json");
const REPORT_PATH = path.join(ROOT, "raw_data", "import_report.json");
const CONFIG_PATH = path.join(ROOT, "raw_data", "import_config.json");
const OUTPUT_PATH = path.join(ROOT, "28_装甲车毁伤目标导入诊断可视化.html");

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function add(a, b) {
  return a.map((value, index) => value + b[index]);
}

function quaternionRotate(point, quaternion) {
  const [x, y, z, w] = quaternion;
  const dot = x * point[0] + y * point[1] + z * point[2];
  const cross = [
    y * point[2] - z * point[1],
    z * point[0] - x * point[2],
    x * point[1] - y * point[0],
  ];
  const normSquared = x * x + y * y + z * z;
  return [
    2 * dot * x + (w * w - normSquared) * point[0] + 2 * w * cross[0],
    2 * dot * y + (w * w - normSquared) * point[1] + 2 * w * cross[1],
    2 * dot * z + (w * w - normSquared) * point[2] + 2 * w * cross[2],
  ];
}

function meshForGeometry(geometry) {
  if (geometry.shape === "box") {
    const [hx, hy, hz] = geometry.size_lwh_cm.map((value) => value / 2);
    const local = [
      [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
      [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ];
    const vertices = local.map((point) =>
      add(quaternionRotate(point, geometry.orientation_xyzw), geometry.center_cm)
    );
    return {
      vertices,
      edges: [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]],
      faces: [[0,1,2,3],[4,7,6,5],[0,4,5,1],[1,5,6,2],[2,6,7,3],[3,7,4,0]],
    };
  }

  if (geometry.shape === "cylinder") {
    const segments = 20;
    const vertices = [];
    for (const localZ of [-geometry.height_cm / 2, geometry.height_cm / 2]) {
      for (let index = 0; index < segments; index += 1) {
        const angle = (2 * Math.PI * index) / segments;
        const local = [
          geometry.radius_cm * Math.cos(angle),
          geometry.radius_cm * Math.sin(angle),
          localZ,
        ];
        vertices.push(add(quaternionRotate(local, geometry.orientation_xyzw), geometry.center_cm));
      }
    }
    const edges = [];
    const faces = [];
    for (let index = 0; index < segments; index += 1) {
      const next = (index + 1) % segments;
      edges.push([index, next], [segments + index, segments + next], [index, segments + index]);
      faces.push([index, next, segments + next, segments + index]);
    }
    faces.push([...Array(segments).keys()].reverse());
    faces.push([...Array(segments).keys()].map((index) => segments + index));
    return { vertices, edges, faces };
  }

  if (geometry.shape === "extruded_polygon") {
    const base = geometry.vertices_cm;
    const extrusion = geometry.extrusion_direction.map(
      (value) => value * geometry.extrusion_distance_cm
    );
    const top = base.map((point) => add(point, extrusion));
    const vertices = base.concat(top);
    const count = base.length;
    const edges = [];
    const faces = [];
    for (let index = 0; index < count; index += 1) {
      const next = (index + 1) % count;
      edges.push([index, next], [count + index, count + next], [index, count + index]);
      faces.push([index, next, count + next, count + index]);
    }
    faces.push([...Array(count).keys()].reverse());
    faces.push([...Array(count).keys()].map((index) => count + index));
    return { vertices, edges, faces };
  }

  throw new Error(`Unsupported geometry shape: ${geometry.shape}`);
}

function boundsForObjects(objects) {
  const minimum = [Infinity, Infinity, Infinity];
  const maximum = [-Infinity, -Infinity, -Infinity];
  for (const object of objects) {
    for (const point of object.mesh.vertices) {
      for (let axis = 0; axis < 3; axis += 1) {
        minimum[axis] = Math.min(minimum[axis], point[axis]);
        maximum[axis] = Math.max(maximum[axis], point[axis]);
      }
    }
  }
  return {
    min: minimum,
    max: maximum,
    center: minimum.map((value, index) => (value + maximum[index]) / 2),
    size: maximum.map((value, index) => value - minimum[index]),
  };
}

function pointInPolygon(point, polygon) {
  let inside = false;
  for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index++) {
    const [yi, zi] = polygon[index];
    const [yj, zj] = polygon[previous];
    const crosses = (zi > point[1]) !== (zj > point[1]);
    if (crosses && point[0] < ((yj - yi) * (point[1] - zi)) / (zj - zi) + yi) {
      inside = !inside;
    }
  }
  return inside;
}

function centerInsideArmor(center, armorObjects) {
  return armorObjects.some((armor) => {
    const geometry = armor.geometry;
    const x0 = geometry.vertices_cm[0][0];
    const x1 = x0 + geometry.extrusion_direction[0] * geometry.extrusion_distance_cm;
    const lower = Math.min(x0, x1);
    const upper = Math.max(x0, x1);
    const polygon = geometry.vertices_cm.map((point) => [point[1], point[2]]);
    return center[0] >= lower && center[0] <= upper && pointInPolygon([center[1], center[2]], polygon);
  });
}

function identityQuaternion(value) {
  return value.length === 4 && value.every((item, index) => item === [0, 0, 0, 1][index]);
}

function safeJson(value) {
  return JSON.stringify(value).replace(/</g, "\\u003c");
}

const target = readJson(TARGET_PATH);
const report = readJson(REPORT_PATH);
const config = readJson(CONFIG_PATH);
const appliedRecords = target.metadata.raw_damage_tree_records.filter(
  (record) => record.apply_to_provisional_tree
);
const treeComponents = new Set(appliedRecords.map((record) => record.component_id));

const armor = target.armor.map((item) => ({ ...item, mesh: meshForGeometry(item.geometry) }));
const components = target.components.map((item) => {
  const codes = item.functional_tags.map((tag) => tag.replace("damage-code:", ""));
  return {
    ...item,
    mesh: meshForGeometry(item.geometry),
    codes,
    treeCovered: treeComponents.has(item.id),
    centerOutsideArmor: !centerInsideArmor(item.geometry.center_cm, target.armor),
    identityRotation: identityQuaternion(item.geometry.orientation_xyzw),
    blastThresholdMissing: !item.response.blast_response.peak_overpressure_threshold,
  };
});

const armorBounds = boundsForObjects(armor);
const componentBounds = boundsForObjects(components);
const allBounds = boundsForObjects(armor.concat(components));
const outsideComponents = components.filter((item) => item.centerOutsideArmor);
const uncoveredComponents = components.filter((item) => !item.treeCovered);
const identityComponents = components.filter((item) => item.identityRotation);
const missingBlastThresholds = components.filter((item) => item.blastThresholdMissing);

const diagnostics = [
  {
    severity: "blocker",
    title: "部件导入不完整：无线电台被隔离",
    finding: "原始 68 条部件记录只导入 67 条；序号 57“无线电台”缺少 X/Y/Z 坐标，没有进入目标模型。",
    consequence: "与通信相关的命中、毁伤树贡献和系统效应会缺失。当前结果不能称为完整装甲车模型。",
    evidence: ["raw_data/import_report.json:3-23", "raw_data/List.csv:58", "damage_engine/raw_import.py:260-276"],
    action: "补齐无线电台三维坐标后，以 strict-components 模式重新导入并重新冻结目标哈希。",
  },
  {
    severity: "blocker",
    title: "材料与暴露类别被全局默认值覆盖",
    finding: `67/67 个部件均为 demo_electronics 且 exposure_class=internal；其中 ${outsideComponents.length} 个部件中心位于两块装甲实体之外。`,
    consequence: "车轮、履带、炮塔、火炮、天线、乘员和弹药的穿透/冲击波响应可能被错误地按同一电子部件与内部暴露规则处理。",
    evidence: ["raw_data/import_config.json:14-24", "damage_engine/raw_import.py:327-365", "raw_data/target.rebuilt.json:231-288"],
    action: "为每类部件建立材料、响应模型、暴露类别和舱室映射；外部件至少需要显式 external/partially_shielded 语义。",
  },
  {
    severity: "blocker",
    title: "装甲响应仍是未验证占位值",
    finding: "上/下装甲均使用 demo_steel、10 mm 等效厚度与 0.2 易损面积比，来源为 RAW-IMPORT-PROFILE-PLACEHOLDER。",
    consequence: "几何遮挡可以检查，但穿透概率、剩余速度、反射载荷和真实毁伤效能没有物理放行依据。",
    evidence: ["raw_data/import_config.json:26-52", "raw_data/target.rebuilt.json:60-101", "03_原始装甲数据重建与风险整改报告.md:138-151"],
    action: "按上/下/侧/炮塔等真实分区录入材料与厚度，并绑定批准来源、标定 ID、适用域和不确定度。",
  },
  {
    severity: "high",
    title: "坐标系和姿态解释是待确认假设",
    finding: `当前假设右手系 X右/Y前/Z上、外禀 XYZ 角度制；原始表有 11 条空 Rotation，跳过无线电台后模型中 ${identityComponents.length} 个部件采用单位姿态。`,
    consequence: "若数据导出工具的轴向、角度顺序或内/外禀约定不同，定向箱体和圆柱会发生系统性旋转错误。",
    evidence: ["raw_data/import_config.json:4-11", "damage_engine/raw_import.py:70-103", "raw_data/target.rebuilt.json:6445-6455"],
    action: "用至少 3 个方向不对称部件与源 CAD/三视图进行人工基准对齐，确认轴向和欧拉角语义。",
  },
  {
    severity: "high",
    title: "毁伤树仅覆盖部分组件，且标准仍是临时映射",
    finding: `${treeComponents.size}/67 个组件进入可执行毁伤树；${uncoveredComponents.length} 个不贡献当前 7 个系统输出。并列机关炮 F1 规则缺阈值，只保留元数据。`,
    consequence: "即使未覆盖组件被击毁，当前系统级 M1/M2/F1/K1/K2/C1/C2 输出也可能完全不变。",
    evidence: ["raw_data/import_report.json:45-58", "raw_data/target.rebuilt.json:4675-4754", "raw_data/import_config.json:53-68"],
    action: "确认正式代码表与门逻辑，补齐并列机关炮阈值，并决定 22 个 unassigned 组件是否应进入系统效应。",
  },
  {
    severity: "medium",
    title: "部分冲击波阈值缺失",
    finding: `${missingBlastThresholds.length}/67 个已导入组件没有 peak_overpressure_threshold；导入器没有擅自补 0。`,
    consequence: "这是诚实缺失而非解析失败，但相关部件的冲击波状态不能按统一阈值解释。",
    evidence: ["raw_data/List.csv", "damage_engine/raw_import.py:309-354"],
    action: "区分“不适用”和“缺测”，分别用显式机制掩码与经验证阈值表示。",
  },
  {
    severity: "pass",
    title: "装甲截面几何重建通过导入公差",
    finding: "两组装甲均由匹配的 6 顶点 YZ 截面沿 +X 拉伸；最大截面/平面残差低于 0.001 cm 配置阈值。",
    consequence: "能够确认 CSV → 拉伸多边形的数值重建一致，但不能据此确认车体外形或装甲厚度真实。",
    evidence: ["raw_data/import_report.json:24-44", "damage_engine/raw_import.py:372-501"],
    action: "保留当前几何导入；另行与 CAD/测绘基准核对轮廓、朝向和车辆原点。",
  },
];

const payload = {
  generatedOn: "2026-10-14",
  target: {
    schema_version: target.schema_version,
    target_id: target.target_id,
    target_revision: target.target_revision,
    coordinate_frame: target.coordinate_frame,
    raw_import: target.metadata.raw_import,
    damage_tree_outputs: target.damage_tree.outputs,
    damage_tree_standard: target.damage_tree.standards[0],
  },
  report,
  config,
  armor,
  components,
  bounds: { armor: armorBounds, components: componentBounds, all: allBounds },
  counts: {
    sourceRows: report.component_rows,
    imported: components.length,
    skipped: report.components_skipped.length,
    armor: armor.length,
    boxes: components.filter((item) => item.geometry.shape === "box").length,
    cylinders: components.filter((item) => item.geometry.shape === "cylinder").length,
    outsideCenters: outsideComponents.length,
    treeCovered: treeComponents.size,
    treeUncovered: uncoveredComponents.length,
    identityRotation: identityComponents.length,
    missingBlastThresholds: missingBlastThresholds.length,
  },
  diagnostics,
};

const htmlTemplate = String.raw`<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>装甲车毁伤目标导入诊断可视化</title>
  <style>
    :root{--bg:#0c1115;--panel:#131a20;--panel2:#182129;--line:#2b3741;--text:#edf2f4;--muted:#9dafba;--lime:#d8ff45;--cyan:#49d6c8;--amber:#ffbf47;--red:#ff625e;--blue:#68a9ff;--green:#65d68d;--armor:#86a6b7}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:radial-gradient(circle at 15% 0,#18232b 0,#0c1115 42%);color:var(--text);font-family:"Segoe UI","Microsoft YaHei",sans-serif;line-height:1.55}
    a{color:var(--cyan)}code,.mono{font-family:"Cascadia Code",Consolas,monospace}.shell{width:min(1500px,calc(100% - 34px));margin:auto;padding:28px 0 56px}
    header{border:1px solid var(--line);background:linear-gradient(125deg,rgba(216,255,69,.08),rgba(73,214,200,.04));padding:26px 28px;border-radius:18px;display:grid;grid-template-columns:1fr auto;gap:22px;align-items:start;box-shadow:0 18px 50px #0006}
    h1{font-size:clamp(28px,4vw,48px);line-height:1.08;margin:0 0 12px;letter-spacing:-1.5px}h2{font-size:24px;margin:0 0 16px}h3{font-size:17px;margin:0 0 8px}.eyebrow{color:var(--lime);font-weight:800;letter-spacing:.12em;text-transform:uppercase;font-size:12px}.lead{max-width:900px;color:#cbd5db;margin:0}.hash{font-size:11px;color:var(--muted);word-break:break-all;text-align:right;max-width:390px}
    .verdict{display:inline-flex;gap:9px;align-items:center;background:#ff625e18;color:#ffb6b3;border:1px solid #ff625e66;border-radius:999px;padding:7px 12px;font-weight:800;font-size:12px;margin-bottom:12px}.dot{width:9px;height:9px;border-radius:50%;background:var(--red);box-shadow:0 0 14px var(--red)}
    .metrics{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin:14px 0 20px}.metric,.panel{background:linear-gradient(180deg,#151d23,#11171c);border:1px solid var(--line);border-radius:14px}.metric{padding:14px}.metric strong{display:block;font-size:25px;color:var(--lime)}.metric span{font-size:12px;color:var(--muted)}
    .section{margin-top:26px}.grid-main{display:grid;grid-template-columns:minmax(0,1.65fr) minmax(310px,.75fr);gap:14px}.panel{padding:17px}.viewer-wrap{position:relative;border:1px solid #2b3841;border-radius:12px;background:linear-gradient(#10171d,#0a0f13);overflow:hidden}.viewer-wrap canvas{display:block;width:100%;height:610px;cursor:grab}.viewer-wrap canvas:active{cursor:grabbing}.viewer-note{position:absolute;left:12px;bottom:10px;font-size:11px;color:#bbcad2;background:#071015d9;border:1px solid #2c3d46;padding:6px 9px;border-radius:7px;pointer-events:none}.legend{position:absolute;right:10px;top:10px;background:#071015dd;border:1px solid #2c3d46;padding:8px 10px;border-radius:9px;font-size:11px;display:grid;gap:4px}.sw{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px}
    .toolbar{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:11px;align-items:center}.toolbar button,.toolbar select,.toolbar input[type="search"]{background:#0e151a;color:var(--text);border:1px solid #35434d;border-radius:8px;padding:8px 10px}.toolbar button{cursor:pointer}.toolbar button.active{background:var(--lime);color:#101410;border-color:var(--lime);font-weight:800}.toolbar label{font-size:12px;color:var(--muted);display:flex;gap:6px;align-items:center}.toolbar input[type="range"]{width:110px}
    .detail{display:grid;gap:12px;align-content:start}.detail-card{background:#0e151a;border:1px solid #2a3841;border-radius:11px;padding:13px}.detail-card .id{color:var(--cyan);font-size:11px}.kv{display:grid;grid-template-columns:110px 1fr;gap:6px 10px;font-size:12px}.kv dt{color:var(--muted)}.kv dd{margin:0;word-break:break-word}.pill{display:inline-block;border:1px solid #40505b;border-radius:999px;padding:2px 7px;font-size:10px;margin:2px 3px 2px 0}.pill.warn{border-color:#ffbf4770;color:#ffd991}.pill.bad{border-color:#ff625e70;color:#ffaaa7}.pill.good{border-color:#65d68d70;color:#92e5ab}
    .diag-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}.diag{border-left:4px solid var(--amber);padding:15px 16px;background:#11181d;border-radius:10px;border-top:1px solid var(--line);border-right:1px solid var(--line);border-bottom:1px solid var(--line)}.diag.blocker{border-left-color:var(--red)}.diag.high{border-left-color:var(--amber)}.diag.medium{border-left-color:var(--blue)}.diag.pass{border-left-color:var(--green)}.sev{font-size:10px;font-weight:900;letter-spacing:.1em}.blocker .sev{color:var(--red)}.high .sev{color:var(--amber)}.medium .sev{color:var(--blue)}.pass .sev{color:var(--green)}.diag p{font-size:13px;color:#c5d0d6;margin:7px 0}.evidence{color:var(--muted);font-size:11px}.action{padding-top:8px;border-top:1px dashed #33414a;margin-top:9px!important;color:#e8eef1!important}
    .flow{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;align-items:stretch}.flow .step{position:relative;padding:14px;background:#10171c;border:1px solid var(--line);border-radius:10px;min-height:112px}.flow .step:not(:last-child):after{content:"→";position:absolute;right:-13px;top:40%;z-index:2;color:var(--lime);font-weight:900;background:var(--bg);padding:0 3px}.step b{display:block;margin-bottom:6px}.step small{color:var(--muted)}
    .gate{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.gate-card{padding:14px;border:1px solid var(--line);border-radius:10px;background:#10171c}.gate-card strong{display:block;margin-bottom:4px}.gate-card.pass strong{color:var(--green)}.gate-card.partial strong{color:var(--amber)}.gate-card.block strong{color:var(--red)}
    .table-wrap{overflow:auto;max-height:620px;border:1px solid var(--line);border-radius:11px}table{border-collapse:collapse;width:100%;font-size:12px}th{position:sticky;top:0;background:#1b252d;color:#dfe9ed;text-align:left;padding:10px;border-bottom:1px solid #3a4852;z-index:1}td{padding:9px 10px;border-bottom:1px solid #26323a;color:#c6d1d7;white-space:nowrap}tbody tr{cursor:pointer}tbody tr:hover,tbody tr.selected{background:#d8ff4510}td.name{font-weight:700;color:#eff4f6}.status-out{color:#ffd27a}.status-in{color:#83df9f}.foot{margin-top:20px;padding:14px;color:var(--muted);font-size:12px;border-top:1px solid var(--line)}
    @media(max-width:1100px){.metrics{grid-template-columns:repeat(3,1fr)}.grid-main{grid-template-columns:1fr}.diag-grid{grid-template-columns:1fr}.flow{grid-template-columns:1fr}.flow .step:not(:last-child):after{content:"↓";right:auto;left:50%;top:auto;bottom:-17px}.gate{grid-template-columns:repeat(2,1fr)}}
    @media(max-width:650px){.shell{width:min(100% - 18px,1500px)}header{grid-template-columns:1fr;padding:20px}.hash{text-align:left}.metrics{grid-template-columns:repeat(2,1fr)}.viewer-wrap canvas{height:460px}.gate{grid-template-columns:1fr}}
  </style>
</head>
<body>
<main class="shell">
  <header>
    <div>
      <div class="eyebrow">Imported target · geometry & semantics audit</div>
      <h1>装甲车毁伤目标<br>导入诊断可视化</h1>
      <p class="lead">基于当前 <span class="mono">raw_data/target.rebuilt.json</span> 的真实几何和导入元数据。结论：模型可以被引擎读取并用于软件链路烟测，但导入不完整、材料/暴露类别默认化、装甲参数占位和临时毁伤准则使它尚不能用于可信的真实毁伤结论。</p>
    </div>
    <div>
      <div class="verdict"><span class="dot"></span>结构可加载 · 物理解释未放行</div>
      <div class="hash mono">target_id: <span id="target-id"></span><br>revision: <span id="target-rev"></span><br>content hash（既有冻结证据）:<br>sha256:c6745e13b03c2bcd00ae6db95841f937dd59bc337a33d1edf420a0afefc855e6</div>
    </div>
  </header>

  <div class="metrics" id="metrics"></div>

  <section class="section panel">
    <div class="eyebrow">Interactive geometry</div>
    <h2>真实导入几何与部件定位</h2>
    <div class="toolbar">
      <button data-view="iso" class="view-btn active">自由旋转</button>
      <button data-view="top" class="view-btn">俯视 X–Y</button>
      <button data-view="front" class="view-btn">前视 X–Z</button>
      <button data-view="side" class="view-btn">侧视 Y–Z</button>
      <select id="code-filter"><option value="all">全部毁伤代码</option></select>
      <label><input id="outside-only" type="checkbox">仅看中心在装甲外</label>
      <label><input id="show-armor" type="checkbox" checked>显示装甲</label>
      <label>装甲透明度 <input id="armor-opacity" type="range" min="0" max="0.7" value="0.2" step="0.02"></label>
      <button id="reset-view">复位</button>
    </div>
    <div class="grid-main">
      <div class="viewer-wrap">
        <canvas id="viewer"></canvas>
        <div class="legend"><span><i class="sw" style="background:#86a6b7"></i>装甲实体</span><span><i class="sw" style="background:#d8ff45"></i>已选部件</span><span><i class="sw" style="background:#ffbf47"></i>中心在装甲外</span><span><i class="sw" style="background:#49d6c8"></i>中心在装甲内</span></div>
        <div class="viewer-note">拖拽旋转 · 滚轮缩放 · 点击最近部件 · 坐标单位 cm</div>
      </div>
      <aside class="detail" id="detail"></aside>
    </div>
  </section>

  <section class="section">
    <div class="eyebrow">Findings</div>
    <h2>导入问题与影响边界</h2>
    <div class="diag-grid" id="diagnostics"></div>
  </section>

  <section class="section panel">
    <div class="eyebrow">Import lineage</div>
    <h2>原始数据到可执行目标的链路</h2>
    <div class="flow">
      <div class="step"><b>① List.csv</b><small>68 条部件、22 列；箱体/圆柱、坐标、旋转、等效参数、3 组毁伤槽位。</small></div>
      <div class="step"><b>② armor.csv</b><small>24 行点数据；上/下装甲各两个 X 截面，每截面 6 顶点。</small></div>
      <div class="step"><b>③ import_config.json</b><small>补入 CSV 未提供的坐标约定、默认材料、暴露类别、装甲占位响应和临时代码映射。</small></div>
      <div class="step"><b>④ raw_import.py</b><small>校验、单位保留、欧拉角转四元数、装甲拉伸重建、异常隔离与毁伤树生成。</small></div>
      <div class="step"><b>⑤ target.rebuilt.json</b><small>2 个装甲实体 + 67 个部件 + 7 个临时系统输出；1 个部件被隔离。</small></div>
    </div>
  </section>

  <section class="section panel">
    <div class="eyebrow">Release gate</div>
    <h2>现在可以相信到什么程度</h2>
    <div class="gate">
      <div class="gate-card pass"><strong>通过：文件与 Schema 链路</strong><span>JSON 能加载，几何对象数量、形状类型与导入诊断一致。</span></div>
      <div class="gate-card partial"><strong>部分通过：几何重建</strong><span>装甲截面误差在公差内；坐标轴、原点、旋转语义仍需与源 CAD 对齐。</span></div>
      <div class="gate-card block"><strong>阻断：模型完整性</strong><span>无线电台缺坐标被隔离；23 个组件未进入可执行毁伤树。</span></div>
      <div class="gate-card block"><strong>阻断：物理有效性</strong><span>材料、装甲厚度、暴露类别、响应模型和代码表均未获批准证据。</span></div>
    </div>
  </section>

  <section class="section panel">
    <div class="eyebrow">Component inventory</div>
    <h2>67 个已导入组件逐项检查</h2>
    <div class="toolbar">
      <input id="search" type="search" placeholder="搜索 ID / 名称 / 毁伤代码">
      <label><input id="tree-only" type="checkbox">仅看未进入毁伤树</label>
      <span id="table-count" style="color:var(--muted);font-size:12px"></span>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>ID</th><th>名称</th><th>形状</th><th>中心 [x,y,z] cm</th><th>毁伤代码</th><th>毁伤树</th><th>相对装甲</th><th>暴露</th><th>材料</th><th>来源</th></tr></thead>
        <tbody id="component-body"></tbody>
      </table>
    </div>
  </section>

  <div class="foot">生成日期：2026-10-14。该页只读取当前工作区数据并将其嵌入单文件；没有修改仿真引擎、目标 JSON 或原始 CSV。几何“中心在装甲外”是诊断提示，不自动等同于错误——车轮、履带、天线等本可在外部，真正需要核对的是它们仍被统一标记为 <span class="mono">internal</span>。</div>
</main>
<script>
const DATA = __PAYLOAD__;
const colorByCode = {M1:"#6ea8fe",M2:"#49d6c8",F1:"#ff8f70",K1:"#d8ff45",K2:"#ffbf47",C1:"#c58aff",C2:"#ff72b6",unassigned:"#81909a"};
const state = {view:"iso",yaw:-0.72,pitch:0.52,zoom:1,selected:"component-001",drag:false,lastX:0,lastY:0,projectedCenters:[]};
const canvas = document.getElementById("viewer");
const ctx = canvas.getContext("2d");
const codeFilter = document.getElementById("code-filter");
const outsideOnly = document.getElementById("outside-only");
const showArmor = document.getElementById("show-armor");
const armorOpacity = document.getElementById("armor-opacity");

document.getElementById("target-id").textContent = DATA.target.target_id;
document.getElementById("target-rev").textContent = DATA.target.target_revision;

const metrics = [
  [DATA.counts.imported + "/" + DATA.counts.sourceRows,"部件成功导入"],
  [DATA.counts.armor,"拉伸装甲实体"],
  [DATA.counts.boxes + " + " + DATA.counts.cylinders,"箱体 + 圆柱"],
  [DATA.counts.outsideCenters,"中心在装甲外"],
  [DATA.counts.treeCovered + "/" + DATA.counts.imported,"进入毁伤树"],
  [fmt(DATA.bounds.all.size[0]) + "×" + fmt(DATA.bounds.all.size[1]) + "×" + fmt(DATA.bounds.all.size[2]),"总包围尺寸 cm"],
];
document.getElementById("metrics").innerHTML = metrics.map(function(item){return '<div class="metric"><strong>'+item[0]+'</strong><span>'+item[1]+'</span></div>'}).join("");

const allCodes = Array.from(new Set(DATA.components.flatMap(function(item){return item.codes}))).sort();
allCodes.forEach(function(code){const option=document.createElement("option");option.value=code;option.textContent=code==="unassigned"?"未分配代码":code;codeFilter.appendChild(option)});

document.getElementById("diagnostics").innerHTML = DATA.diagnostics.map(function(item){
  const label={blocker:"阻断项",high:"高优先级",medium:"需补充",pass:"已通过的软件检查"}[item.severity];
  return '<article class="diag '+item.severity+'"><div class="sev">'+label+'</div><h3>'+item.title+'</h3><p>'+item.finding+'</p><p><b>影响：</b>'+item.consequence+'</p><div class="evidence mono">证据：'+item.evidence.join(" · ")+'</div><p class="action"><b>建议：</b>'+item.action+'</p></article>';
}).join("");

function fmt(value,digits){return Number(value).toFixed(digits===undefined?1:digits)}
function componentColor(item){
  if(item.id===state.selected)return "#d8ff45";
  if(item.centerOutsideArmor)return "#ffbf47";
  return colorByCode[item.codes[0]] || "#49d6c8";
}
function visibleComponents(){
  return DATA.components.filter(function(item){
    if(codeFilter.value!=="all" && !item.codes.includes(codeFilter.value))return false;
    if(outsideOnly.checked && !item.centerOutsideArmor)return false;
    return true;
  });
}
function rotateForView(point){
  const center=DATA.bounds.all.center;
  let x=point[0]-center[0],y=point[1]-center[1],z=point[2]-center[2];
  if(state.view==="top")return [x,-y,z];
  if(state.view==="front")return [x,-z,-y];
  if(state.view==="side")return [y,-z,x];
  const cy=Math.cos(state.yaw),sy=Math.sin(state.yaw),cp=Math.cos(state.pitch),sp=Math.sin(state.pitch);
  const x1=cy*x-sy*y,y1=sy*x+cy*y;
  return [x1,cp*z-sp*y1,sp*z+cp*y1];
}
function resizeCanvas(){
  const ratio=Math.min(window.devicePixelRatio||1,2);
  const rect=canvas.getBoundingClientRect();
  const width=Math.max(1,Math.round(rect.width*ratio)),height=Math.max(1,Math.round(rect.height*ratio));
  if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height}
  ctx.setTransform(ratio,0,0,ratio,0,0);
  return {width:rect.width,height:rect.height};
}
function draw(){
  const size=resizeCanvas();ctx.clearRect(0,0,size.width,size.height);
  const components=visibleComponents();
  const objects=(showArmor.checked?DATA.armor:[]).concat(components);
  const projectedRaw=[];
  objects.forEach(function(object){object.mesh.vertices.forEach(function(point){projectedRaw.push(rotateForView(point))})});
  if(!projectedRaw.length)return;
  const minX=Math.min.apply(null,projectedRaw.map(function(p){return p[0]})),maxX=Math.max.apply(null,projectedRaw.map(function(p){return p[0]}));
  const minY=Math.min.apply(null,projectedRaw.map(function(p){return p[1]})),maxY=Math.max.apply(null,projectedRaw.map(function(p){return p[1]}));
  const scale=Math.min((size.width-50)/Math.max(1,maxX-minX),(size.height-50)/Math.max(1,maxY-minY))*state.zoom;
  const midX=(minX+maxX)/2,midY=(minY+maxY)/2;
  function screen(point){const p=rotateForView(point);return [size.width/2+(p[0]-midX)*scale,size.height/2+(p[1]-midY)*scale,p[2]]}
  const faces=[];
  objects.forEach(function(object){
    const projected=object.mesh.vertices.map(screen);
    object.mesh.faces.forEach(function(face){faces.push({object:object,points:face.map(function(index){return projected[index]}),depth:face.reduce(function(sum,index){return sum+projected[index][2]},0)/face.length})});
  });
  faces.sort(function(a,b){return a.depth-b.depth});
  faces.forEach(function(face){
    const isArmor=face.object.role==="armor";
    const color=isArmor?"#86a6b7":componentColor(face.object);
    ctx.beginPath();face.points.forEach(function(p,index){if(index===0)ctx.moveTo(p[0],p[1]);else ctx.lineTo(p[0],p[1])});ctx.closePath();
    ctx.globalAlpha=isArmor?Number(armorOpacity.value):(face.object.id===state.selected?.62:.22);ctx.fillStyle=color;ctx.fill();
  });
  ctx.globalAlpha=1;
  objects.forEach(function(object){
    const projected=object.mesh.vertices.map(screen);const isArmor=object.role==="armor";ctx.strokeStyle=isArmor?"#a9c0cc":componentColor(object);ctx.lineWidth=object.id===state.selected?2.2:(isArmor?1.2:.75);ctx.globalAlpha=isArmor?.75:.86;
    object.mesh.edges.forEach(function(edge){ctx.beginPath();ctx.moveTo(projected[edge[0]][0],projected[edge[0]][1]);ctx.lineTo(projected[edge[1]][0],projected[edge[1]][1]);ctx.stroke()});
  });ctx.globalAlpha=1;
  const origin=DATA.bounds.all.center;const axisLength=90;const o=screen(origin);[[[axisLength,0,0],"X","#ff6b66"],[[0,axisLength,0],"Y","#65d68d"],[[0,0,axisLength],"Z","#68a9ff"]].forEach(function(axis){const endpoint=screen([origin[0]+axis[0][0],origin[1]+axis[0][1],origin[2]+axis[0][2]]);ctx.strokeStyle=axis[2];ctx.fillStyle=axis[2];ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(o[0],o[1]);ctx.lineTo(endpoint[0],endpoint[1]);ctx.stroke();ctx.fillText(axis[1],endpoint[0]+4,endpoint[1]-4)});
  state.projectedCenters=components.map(function(item){const p=screen(item.geometry.center_cm);return {id:item.id,x:p[0],y:p[1]}});
}

function selectedItem(){return DATA.components.find(function(item){return item.id===state.selected})||DATA.components[0]}
function geometryText(item){const g=item.geometry;if(g.shape==="box")return "box · "+g.size_lwh_cm.map(function(v){return fmt(v)}).join(" × ")+" cm";return "cylinder · R="+fmt(g.radius_cm)+" / H="+fmt(g.height_cm)+" cm"}
function updateDetail(){
  const item=selectedItem();state.selected=item.id;
  const threshold=item.response.blast_response.peak_overpressure_threshold;
  const tags=item.codes.map(function(code){return '<span class="pill">'+code+'</span>'}).join("");
  const flags=(item.centerOutsideArmor?'<span class="pill warn">中心在装甲外</span>':'<span class="pill good">中心在装甲内</span>')+(item.treeCovered?'<span class="pill good">已进毁伤树</span>':'<span class="pill bad">未进毁伤树</span>');
  document.getElementById("detail").innerHTML='<div class="detail-card"><div class="id mono">'+item.id+'</div><h3>'+item.name+'</h3>'+tags+'<div style="margin-top:6px">'+flags+'</div></div><div class="detail-card"><h3>几何</h3><dl class="kv"><dt>形状/尺寸</dt><dd>'+geometryText(item)+'</dd><dt>中心 cm</dt><dd class="mono">['+item.geometry.center_cm.map(function(v){return fmt(v,3)}).join(", ")+']</dd><dt>四元数 xyzw</dt><dd class="mono">['+item.geometry.orientation_xyzw.map(function(v){return fmt(v,4)}).join(", ")+']</dd><dt>单位姿态</dt><dd>'+(item.identityRotation?'是（原 Rotation 为空或恰为零旋转）':'否')+'</dd></dl></div><div class="detail-card"><h3>响应与溯源</h3><dl class="kv"><dt>材料</dt><dd>'+item.response.material_id+'</dd><dt>暴露类别</dt><dd>'+item.exposure_class+'</dd><dt>等效厚度</dt><dd>'+fmt(item.response.equivalent_thickness.value,2)+' '+item.response.equivalent_thickness.unit+' · '+item.response.equivalent_thickness.validation_status+'</dd><dt>易损面积比</dt><dd>'+fmt(item.response.vulnerable_area_ratio.value,3)+' · '+item.response.vulnerable_area_ratio.validation_status+'</dd><dt>超压阈值</dt><dd>'+(threshold?fmt(threshold.value,3)+' '+threshold.unit:'缺失')+'</dd><dt>来源</dt><dd class="mono">'+item.provenance.source_id+'</dd><dt>证据状态</dt><dd>'+item.provenance.status+'</dd></dl></div><div class="detail-card"><h3>坐标解释</h3><p style="font-size:12px;color:var(--muted);margin:0">右手系；X 向右，Y 向前，Z 向上。该约定来自导入配置，仍需源 CAD/数据提供方确认。</p></div>';
  renderTable();draw();
}

function selectComponent(id){state.selected=id;updateDetail();const row=document.querySelector('tr[data-id="'+id+'"]');if(row)row.scrollIntoView({block:"nearest"})}
canvas.addEventListener("mousedown",function(event){state.drag=true;state.lastX=event.clientX;state.lastY=event.clientY});
window.addEventListener("mouseup",function(){state.drag=false});
window.addEventListener("mousemove",function(event){if(!state.drag||state.view!=="iso")return;state.yaw+=(event.clientX-state.lastX)*.009;state.pitch=Math.max(-1.4,Math.min(1.4,state.pitch+(event.clientY-state.lastY)*.009));state.lastX=event.clientX;state.lastY=event.clientY;draw()});
canvas.addEventListener("wheel",function(event){event.preventDefault();state.zoom=Math.max(.45,Math.min(4,state.zoom*Math.exp(-event.deltaY*.001)));draw()},{passive:false});
canvas.addEventListener("click",function(event){if(!state.projectedCenters.length)return;const rect=canvas.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top;let best=null;state.projectedCenters.forEach(function(item){const distance=(item.x-x)*(item.x-x)+(item.y-y)*(item.y-y);if(!best||distance<best.distance)best={id:item.id,distance:distance}});if(best&&best.distance<1600)selectComponent(best.id)});
window.addEventListener("resize",draw);
document.querySelectorAll(".view-btn").forEach(function(button){button.addEventListener("click",function(){state.view=button.dataset.view;document.querySelectorAll(".view-btn").forEach(function(item){item.classList.toggle("active",item===button)});draw()})});
document.getElementById("reset-view").addEventListener("click",function(){state.yaw=-.72;state.pitch=.52;state.zoom=1;draw()});
[codeFilter,outsideOnly,showArmor,armorOpacity].forEach(function(control){control.addEventListener("input",function(){draw();renderTable()})});

const search=document.getElementById("search"),treeOnly=document.getElementById("tree-only"),tbody=document.getElementById("component-body");
function tableItems(){const query=search.value.trim().toLowerCase();return visibleComponents().filter(function(item){if(treeOnly.checked&&item.treeCovered)return false;const hay=(item.id+" "+item.name+" "+item.codes.join(" ")).toLowerCase();return !query||hay.includes(query)})}
function renderTable(){const items=tableItems();document.getElementById("table-count").textContent="显示 "+items.length+" / "+DATA.components.length;tbody.innerHTML=items.map(function(item){return '<tr data-id="'+item.id+'" class="'+(item.id===state.selected?'selected':'')+'"><td class="mono">'+item.id+'</td><td class="name">'+item.name+'</td><td>'+item.geometry.shape+'</td><td class="mono">['+item.geometry.center_cm.map(function(v){return fmt(v,1)}).join(", ")+']</td><td>'+item.codes.join(" / ")+'</td><td>'+(item.treeCovered?'已覆盖':'未覆盖')+'</td><td class="'+(item.centerOutsideArmor?'status-out':'status-in')+'">'+(item.centerOutsideArmor?'中心在外':'中心在内')+'</td><td>'+item.exposure_class+'</td><td>'+item.response.material_id+'</td><td class="mono">'+item.provenance.source_id+'</td></tr>'}).join("");tbody.querySelectorAll("tr").forEach(function(row){row.addEventListener("click",function(){selectComponent(row.dataset.id)})})}
[search,treeOnly].forEach(function(control){control.addEventListener("input",renderTable)});

updateDetail();
</script>
</body>
</html>`;

const html = htmlTemplate.replace("__PAYLOAD__", safeJson(payload));
fs.writeFileSync(OUTPUT_PATH, html, "utf8");
console.log(JSON.stringify({
  output: OUTPUT_PATH,
  bytes: Buffer.byteLength(html),
  components: components.length,
  armor: armor.length,
  outsideCenters: outsideComponents.length,
  treeCovered: treeComponents.size,
  diagnostics: diagnostics.length,
}, null, 2));
