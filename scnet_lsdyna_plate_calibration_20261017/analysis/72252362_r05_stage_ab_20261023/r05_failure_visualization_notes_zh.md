# R05 阶段 B 失效模式图表说明与替代文本

## 证据边界

这些图只展示 Job 72252362 返回文件中已经记录的数值事件和捕获账本诊断。它们不证明真实裂纹、穿孔或任何物理因果，也不构成 P0、材料模型、仿真引擎或数据集生成器准入。图中未绘制误差条：输入是一次确定性求解的事件计数与质量账本比例，没有重复样本或已定义的统计不确定性。

## 图 A：各 case 的失效单元计数

替代文本：零基线分组柱状图，按七个 case（或当前输入中的全部 case）分别展示 plate/fragment 与 negative-volume/other-solid-failure 四种组合；颜色区分部件、纹理区分失效文本类别，并直接标注计数。当前完整输入共包含 2248 个唯一 MES 失效事件，结构零保留为 0，不从图或源数据中删除。

说明：柱高是唯一单元计数，不是发生概率；`other_solid_failure` 仅表示求解器文本没有进一步给出机理。

## 图 B：重点工况的事件时间—初始 z

替代文本：两个并排散点面板，每个点对应一个唯一 MES 失效事件；横轴是事件时间，纵轴是该单元由输入 deck 参考几何计算的初始质心 z。800 m/s、0.75 mm 工况有 155 个点，2000 m/s、0.75 mm 工况有 692 个点。颜色区分 plate/fragment，圆点与 X 标记区分两类求解器失效文本。点之间不连接、不插值、不平滑，也不做随机抖动或分箱。

重要语义：纵轴是初始参考位置，不代表失效瞬时位置或变形后位置；散点的接近、先后或聚集不能单独解释为物理因果。

## 图 C：捕获前删除质量比例与冻结门

替代文本：从零开始的水平条形图，逐 case 显示 `deleted_upstream_mass_fraction`，并以黑色虚线标出冻结的 5% 可用性门。当前 6/7 个 case 超过该门；最高值是 `p0_v2000_h1p00` 的 75.6%。颜色、斜线纹理和直接百分比标签共同编码是否超过门限。

说明：门限来自冻结设计 `qualification_gates.capture_ledger.deleted_upstream_mass_fraction_max`。超过门限表示主 QoI 按合同不可用，不是对物理现象的概率判断。

## 可复核数据与导出

- 图 A 源数据：`r05_failure_modes_panel_a_source_data.csv`，显式包含所有 case×类别组合和结构零。
- 图 B 源数据：`r05_failure_modes_panel_b_source_data.csv`，保留每个点的 case、EID、原始时间和初始 deck z；没有聚合或平滑。
- 图 C 源数据：`r05_failure_modes_panel_c_source_data.csv`，保留每个 case 的原始比例、冻结门及判定。
- PNG 为 200 ppi、RGB、无 alpha 通道、白色不透明背景；SVG 保留矢量文字/图元并使用白色不透明背景。
- 尺寸、像素、哈希、调色板、透明度和全部变换记录见 `r05_failure_visualization_manifest.json`。
