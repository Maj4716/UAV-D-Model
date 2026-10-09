from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "02_仿真引擎机理与性能评估整改报告.docx"

NAVY = "1F4E78"
LIGHT_BLUE = "D9EAF7"
LIGHT_GRAY = "F2F4F6"
MID_GRAY = "D9DEE3"
DARK = RGBColor(0x1F, 0x1F, 0x1F)
RED = RGBColor(0xA6, 0x1B, 0x1B)


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_width(cell, width_inches: float) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(int(width_inches * 1440)))
    tc_w.set(qn("w:type"), "dxa")


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def prevent_row_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def set_cell_margins(cell, top=55, start=65, bottom=55, end=65) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_run_font(run, *, size: float | None = None, bold: bool | None = None,
                 color: RGBColor | None = None, east_asia: str = "等线",
                 latin: str = "Aptos") -> None:
    run.font.name = latin
    run._element.rPr.rFonts.set(qn("w:eastAsia"), east_asia)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = color


def set_paragraph_font(paragraph, *, size: float = 10.0, bold: bool = False,
                       color: RGBColor = DARK) -> None:
    for run in paragraph.runs:
        set_run_font(run, size=size, bold=bold, color=color)


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("第 ")
    set_run_font(run, size=8.5, color=RGBColor(0x66, 0x66, 0x66))
    field_begin = OxmlElement("w:fldChar")
    field_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    field_end = OxmlElement("w:fldChar")
    field_end.set(qn("w:fldCharType"), "end")
    run._r.append(field_begin)
    run._r.append(instr)
    run._r.append(field_end)
    tail = paragraph.add_run(" 页")
    set_run_font(tail, size=8.5, color=RGBColor(0x66, 0x66, 0x66))


def add_table(doc: Document, headers: list[str], rows: list[list[str]],
              widths: list[float] | None = None, font_size: float = 8.4) -> object:
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    table.autofit = False
    header = table.rows[0]
    set_repeat_table_header(header)
    prevent_row_split(header)
    for index, text in enumerate(headers):
        cell = header.cells[index]
        set_cell_shading(cell, NAVY)
        set_cell_margins(cell)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        if widths:
            set_cell_width(cell, widths[index])
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_after = Pt(0)
        run = paragraph.add_run(text)
        set_run_font(run, size=font_size, bold=True, color=RGBColor(255, 255, 255))
    for row_index, values in enumerate(rows):
        row = table.add_row()
        prevent_row_split(row)
        for index, value in enumerate(values):
            cell = row.cells[index]
            if row_index % 2:
                set_cell_shading(cell, LIGHT_GRAY)
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if widths:
                set_cell_width(cell, widths[index])
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.05
            run = paragraph.add_run(str(value))
            set_run_font(run, size=font_size, color=DARK)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def add_caption(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(4)
    paragraph.paragraph_format.space_after = Pt(4)
    run = paragraph.add_run(text)
    set_run_font(run, size=8.5, color=RGBColor(0x55, 0x55, 0x55))


def add_body(doc: Document, text: str, *, bold_prefix: str | None = None,
             color: RGBColor = DARK) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.first_line_indent = Cm(0.74)
    paragraph.paragraph_format.space_after = Pt(5)
    paragraph.paragraph_format.line_spacing = 1.25
    if bold_prefix and text.startswith(bold_prefix):
        first = paragraph.add_run(bold_prefix)
        set_run_font(first, size=10.2, bold=True, color=color)
        second = paragraph.add_run(text[len(bold_prefix):])
        set_run_font(second, size=10.2, color=color)
    else:
        run = paragraph.add_run(text)
        set_run_font(run, size=10.2, color=color)


def add_bullet(doc: Document, text: str, *, level: int = 0) -> None:
    paragraph = doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
    paragraph.paragraph_format.space_after = Pt(3)
    paragraph.paragraph_format.line_spacing = 1.15
    set_paragraph_font(paragraph, size=10.0)
    if not paragraph.runs:
        run = paragraph.add_run(text)
        set_run_font(run, size=10.0)
    else:
        paragraph.runs[0].text = text
        set_run_font(paragraph.runs[0], size=10.0)


def add_command(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.left_indent = Cm(0.55)
    paragraph.paragraph_format.right_indent = Cm(0.2)
    paragraph.paragraph_format.space_after = Pt(3)
    run = paragraph.add_run(text)
    set_run_font(run, size=8.4, east_asia="等线", latin="Consolas")


def configure_styles(doc: Document) -> None:
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Aptos"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "等线")
    normal.font.size = Pt(10.2)
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE

    title = styles["Title"]
    title.font.name = "Aptos Display"
    title._element.rPr.rFonts.set(qn("w:eastAsia"), "等线")
    title.font.size = Pt(27)
    title.font.bold = True
    title.font.color.rgb = DARK

    for name, size, before, after in (
        ("Heading 1", 16, 14, 7),
        ("Heading 2", 12.5, 10, 5),
        ("Heading 3", 10.8, 8, 4),
    ):
        style = styles[name]
        style.font.name = "Aptos Display"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "等线")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = DARK
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True


def configure_sections(doc: Document) -> None:
    for section in doc.sections:
        section.page_width = Inches(8.5)
        section.page_height = Inches(11)
        section.top_margin = Inches(0.72)
        section.bottom_margin = Inches(0.66)
        section.left_margin = Inches(0.72)
        section.right_margin = Inches(0.72)
        section.header_distance = Inches(0.25)
        section.footer_distance = Inches(0.28)


def add_header_footer(section) -> None:
    section.different_first_page_header_footer = True
    header = section.header
    paragraph = header.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("仿真引擎机理与性能评估整改报告｜v0.3.0")
    set_run_font(run, size=8.0, color=RGBColor(0x77, 0x77, 0x77))
    add_page_number(section.footer.paragraphs[0])


def build_report() -> None:
    result = json.loads((ROOT / "result.demo.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "dataset_demo" / "manifest.json").read_text(encoding="utf-8"))

    engine_hash = result["provenance"]["engine_code_hash"]
    schema_hash = result["provenance"]["schema_tree_hash"]
    dataset_id = manifest["dataset_id"]
    achieved_width = result["convergence"]["achieved_max_half_width"]

    doc = Document()
    configure_styles(doc)
    configure_sections(doc)

    # Cover
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(58)
    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("仿真引擎机理与性能评估整改报告")
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_before = Pt(8)
    subtitle.paragraph_format.space_after = Pt(28)
    run = subtitle.add_run("破片／冲击波毁伤计算、毁伤树评价与神经网络数据接口")
    set_run_font(run, size=13.0, color=RGBColor(0x3F, 0x3F, 0x3F))

    status = doc.add_paragraph()
    status.alignment = WD_ALIGN_PARAGRAPH.CENTER
    status.paragraph_format.space_after = Pt(18)
    r1 = status.add_run("软件验证：通过　")
    set_run_font(r1, size=12.0, bold=True, color=RGBColor(0x1F, 0x6D, 0x3B))
    r2 = status.add_run("｜　物理确认：待完成")
    set_run_font(r2, size=12.0, bold=True, color=RED)

    cover_rows = [
        ["项目目录", str(ROOT)],
        ["引擎版本", "damage-dataset-engine 0.3.0"],
        ["报告日期", "2026-09-26"],
        ["当前可声明等级", "L1 工程可算；具备 L2 参数标定与分层确认接口"],
        ["禁止性结论", "演示参数未经实物试验确认，不得作为真实装备效能或自主控制放行依据"],
    ]
    add_table(doc, ["项目", "内容"], cover_rows, widths=[1.45, 5.55], font_size=9.2)
    footer_note = doc.add_paragraph()
    footer_note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer_note.paragraph_format.space_before = Pt(18)
    rr = footer_note.add_run("本报告区分“软件实现验证”与“物理模型确认”。二者不可互相替代。")
    set_run_font(rr, size=9.0, color=RGBColor(0x66, 0x66, 0x66))
    doc.add_page_break()

    # 1
    doc.add_heading("1　执行摘要与总评", level=1)
    add_body(
        doc,
        "结论：本轮已把原有可运行原型整改为具备严格输入契约、可追溯概率输出、确定性批处理、分组防泄漏数据集和可安装发布包的工程内核。代码层面的高风险错误已经修复，73 项自动化测试全部通过；但仓库中的物理注册表仍明确标记为 unverified，尚无与真实战斗部、装甲和精密部件试验数据的标定/确认记录，因此不能宣称“高保真已得到验证”，也不能把演示数值解释为真实毁伤概率。",
        bold_prefix="结论：",
    )
    summary_rows = [
        ["软件回归", "73/73 通过", "几何、破片、冲击波、毁伤树、规范化、数据集、校验与统计工具"],
        ["单进程吞吐", "31.05 cases/s", "40 案例×4 重复，三次复测中位；整改前 10.73 cases/s，约 2.89×"],
        ["批数据吞吐", "43.35 cases/s", "48 案例×2 重复、2 workers、压缩 NPZ；串行中位 34.23 cases/s"],
        ["数据一致性", "通过", "串行/2-worker 的 67 个 NPZ 数组逐项完全一致，dataset_id 一致"],
        ["安装包", "通过", "wheel 0.3.0 含 4 个 JSON Schema；源码树外安装后可校验并完成仿真"],
        ["演示物理", "未确认", "参数未验证；演示案例自适应目标半宽 0.05 未达到，实际为 %.4f" % achieved_width],
    ]
    add_caption(doc, "表 1　整改后总体状态")
    add_table(doc, ["维度", "结论", "证据／解释"], summary_rows, widths=[1.25, 1.35, 4.4], font_size=8.8)

    add_body(
        doc,
        "评价边界：当前适合软件集成、算法开发、试验设计和参数标定准备；完成分层物理确认、误差量化、适用域冻结和独立复核后，才可进入指定范围的教师数据生产。代理网络即使能高精度拟合本引擎，也只说明复制了当前教师映射，不能自动消除教师模型偏差。",
        bold_prefix="评价边界：",
    )

    # 2
    doc.add_heading("2　评估范围、输入覆盖与判据", level=1)
    add_body(
        doc,
        "本次检查覆盖 damage_engine 全部运行路径、4 份 JSON Schema、示例目标/案例/物理注册表、批量生成器、Torch 适配器、发布配置和演示产物。方法包括源码审查、反例驱动测试、几何随机对偶、数值极值测试、确定性对照、cProfile 热点剖析、串并行数据逐数组对比、manifest Schema 校验以及 wheel 构建/源码树外安装烟测。",
    )
    mapping_rows = [
        ["外装甲", "顶点、拉伸方向/距离", "extruded_polygon；显式 *_m 或 *_cm，载入后统一 SI；方向须为单位向量"],
        ["外装甲", "等效厚度、易损面积比例", "m/cm/mm 与 1/% 均可；校验范围、有限性、不确定度和来源"],
        ["外装甲", "峰值/冲量/持续时间阈值", "覆盖 failed 状态 P-I 尺度；没有 failed 响应模型时拒答"],
        ["内部部件", "名称、形状、L/H/W 或 R/H", "box、cylinder、extruded_polygon；尺寸和中心支持 m/cm"],
        ["内部部件", "X/Y/Z 与姿态", "位置规范化为 m；姿态固定为 xyzw 单位四元数并转完整 3×3 旋转矩阵特征"],
        ["内部部件", "厚度、易损率、P-I 阈值", "进入物理响应和训练特征；阈值覆盖首个非完好状态，更高状态由注册表分级参数定义"],
        ["毁伤树", "部件事件、逻辑门、评价标准", "DAG 编译；支持 independent / fully_dependent；多个标准映射为输出视图与 mask"],
        ["战斗部", "源项参数预留", "质量分布、数量、形状、密度、初速、方向、阻力、爆炸等效量均由案例/注册表给出；未实现的 fragment_list_ref 明确拒答"],
    ]
    add_caption(doc, "表 2　用户需求字段到当前接口的映射")
    add_table(doc, ["对象", "用户字段", "当前实现与守卫"], mapping_rows, widths=[1.0, 1.75, 4.25], font_size=8.55)

    doc.add_heading("2.1　可信度判据", level=2)
    add_bullet(doc, "Verification（验证）：代码是否正确实现了声明的数学/接口规格——本轮重点完成。")
    add_bullet(doc, "Validation（确认）：模型是否在预期用途和适用域内代表现实——必须依赖试验/高保真基准，当前未完成。")
    add_bullet(doc, "Uncertainty Quantification（不确定度量化）：区分随机、认知和数值误差，并验证区间覆盖——当前已有传播接口，尚缺正式确认数据。")
    add_bullet(doc, "因此，本报告不会用自动化测试数量替代物理可信度，也不会把 L2 案例字段误写成已经达到 L2/L3 的证据。")

    # 3
    doc.add_heading("3　核心机理评估", level=1)
    doc.add_heading("3.1　破片通道", level=2)
    add_body(
        doc,
        "当前计算链为：源项质量/速度/方向采样 → 目标系速度叠加 → 空气阻力衰减 → 装甲与部件按射线距离排序 → 等效弹道厚度与经验侵彻响应 → 残速、沉积能量和部件状态概率。整改后，同一破片穿透前一内部部件后会从出射位置携残速继续追踪，不再在第一次部件命中后无条件停止；每次相遇后的目标系速度还会重新接受模型适用域检查。",
    )
    add_body(
        doc,
        "评价：该通道已经能够表达几何遮挡、串列侵彻、入射角、剩余速度和多状态概率，适合作为可校准降阶模型。它仍未表达偏转/跳弹、二次破片、孔洞重叠以及质量—初速—方向联合分布；离散实测破片清单接口尚未实现，因此当前对非空 fragment_list_ref 采取拒答而不是静默忽略。",
    )

    doc.add_heading("3.2　冲击波通道", level=2)
    add_body(
        doc,
        "当前计算链为：等效质量与比例距离查表 → 环境温度相关到达时间 → 入射角反射 → 装甲 failed 状态 P-I 响应 → 装甲完整度更新 → 完好面积/开孔面积混合透射 → 舱室放大 → 部件峰值—冲量—持续时间 P-I 响应。P-I 运算已改为对数域，避免极端载荷比值溢出；目标级峰值、冲量和持续时间阈值可覆盖相应 50% 响应尺度。",
    )
    add_body(
        doc,
        "评价：与仅用单一超压阈值相比，当前能区分峰值、冲量和持续时间，并显式输出反射载荷与装甲失效概率。仍属于中心点自由场和经验透射的降阶模型，不包含完整压力—时间波形、多路径反射、开口泄压网络、结构动力学或 CFD；装甲之间对其他装甲载荷的相互遮挡也未建模。",
    )

    doc.add_heading("3.3　耦合、毁伤树与评价标准", level=2)
    add_body(
        doc,
        "sequential 模式将破片后的装甲完整度传给冲击波透射；面积混合关系可写为 T = I·T_intact + (1−I)·1，其中 I 为完整投影面积比例。部件层综合状态仍使用 P_comb = 1−(1−P_frag)(1−P_blast)，结果和警告中均明确标注“给定重复样本的条件独立”假设。",
    )
    add_body(
        doc,
        "毁伤树不再默认所有输入独立。independent 门只有在输入事件不重叠时才允许；fully_dependent 采用完全正相关的可审计包络：OR=max、AND=min、k-of-n 为第 k 大边际概率。目标可声明多个标准 ID，每个标准选择稳定的输出节点集合；单案例 JSON、Parquet 与 NPZ 均保留标准—输出映射和逐案例请求 mask。",
    )

    mechanism_rows = [
        ["破片", "几何命中、串列侵彻、残速/能量、多状态", "降阶弹道可校准；适用域越界拒答", "无偏转、二次破片、孔洞重叠和实测清单加载"],
        ["冲击波", "比例距离、反射、面积混合透射、P-I", "峰值/冲量/持续时间均参与", "无完整时程、泄压网络、多路径或结构动力学"],
        ["耦合", "破片开孔影响冲击波遮蔽", "避免把破孔误当均匀减薄", "部件层两机制仍条件独立"],
        ["毁伤树", "显式依赖门、DAG、多评价标准", "共享事件不得误标独立", "尚无条件概率、Copula 或标定相关矩阵"],
    ]
    add_caption(doc, "表 3　机理能力与边界")
    add_table(doc, ["模块", "已实现", "主要改进", "仍有限制"], mechanism_rows,
              widths=[0.8, 2.1, 1.9, 2.2], font_size=8.3)

    # 4
    doc.add_heading("4　发现的问题、后果、原理与实际修复", level=1)
    add_body(doc, "以下均为本轮实际检查到的缺陷或高概率失效模式。表中“已修复”表示代码和测试已落地；需要真实试验数据才能解决的内容列入第 8 章，不以软件改动冒充物理确认。")

    physics_issues = [
        ["P0-01", "内部部件命中后无条件 break，且未正确延续残速。后果：串列部件不会被同一破片继续作用，系统共因毁伤被低估。", "穿透后应从出射位置继续传播；只有未穿透或残速耗尽才终止。", "已修复：更新 residual 与 previous_distance；双部件串列审计测试验证两者各命中一次。"],
        ["P0-02", "只检查初始 Gurney 速度，目标系速度叠加后可能越出标定域。后果：以越域数值产生正常标签。", "适用域必须施加在真正进入响应模型的状态量上。", "已修复：逐片检查最终速度有限、为正且位于 valid_speed_mps；越界结构化拒答。"],
        ["P0-03", "装甲冲击波响应和阈值原先未进入计算。后果：装甲只挡内部部件，自己永不因冲击波退化。", "装甲是受载对象；反射载荷应驱动 P-I 失效并反过来改变透射。", "已修复：计算装甲入射/反射载荷、failed 概率、前后完整度；缺 failed 模型但给阈值时拒答。"],
        ["P0-04", "破孔耦合曾把完整度解释为均匀减薄。后果：厚板少量孔洞被错误视为整板变薄，透射严重失真。", "孔洞是一部分开放面积，应做开放区与完好区的面积平均。", "已修复：T=I·T_intact+(1−I)，多板逐板相乘；0/0.5/1 完整度解析排序测试通过。"],
        ["P1-01", "P-I 幂运算在极端比值下会溢出。后果：NaN/Inf 污染概率与训练标签。", "在 log 域做幂与求和，最后再进入 sigmoid。", "已修复：logaddexp 稳定计算，零载荷精确为 0；1e±300 极值测试通过。"],
        ["P1-02", "冲击波到达时间忽略案例温度。后果：冷热环境的声速与时序相同。", "传播速度随温度变化，至少应采用声明气体参数。", "已修复：使用 environment.temperature_k；冷热到达时间顺序测试通过。"],
        ["P1-03", "峰值阈值已消费，但目标级冲量/持续时间阈值仅校验未进响应。后果：用户输入无效且不报警。", "进入 Schema 的物理字段要么被计算消费，要么拒答。", "已修复：不确定度采样和 SI 转换覆盖三类阈值，并覆盖首个非完好/failed 状态的 P-I 尺度；响应排序测试通过。"],
        ["P1-04", "fragment_list_ref 可填写但未加载。后果：离散破片清单被静默忽略。", "未实现能力必须 fail closed。", "已修复：非空引用返回明确 InputValidationError；保留字段用于后续实现。"],
        ["P1-05", "拉伸多边形每条射线对每个三角形调用大量 NumPy cross。后果：破片通道占绝大多数运行时间。", "静态三角几何应预计算，窄相位采用标量 Möller–Trumbore。", "已修复：缓存边/法向并使用标量求交；200 条随机射线与等价有向盒对照。"],
    ]
    add_caption(doc, "表 4　物理与几何问题整改")
    add_table(doc, ["编号", "问题与可能后果", "解决原理", "实际解决与证据"], physics_issues,
              widths=[0.62, 2.25, 1.8, 2.33], font_size=7.75)

    logic_issues = [
        ["P0-06", "相关毁伤事件被独立 OR/AND 组合。示例 0.28 与 0.40 会被算成 0.568。后果：系统毁伤概率虚高。", "概率门必须声明依赖模型；共享祖先/同部件分级事件不能伪装独立。", "已修复：支持 independent / fully_dependent；重叠事件独立门编译失败；示例改为完全相关，结果为 0.40。"],
        ["P0-07", "evaluation_standard_ids 原先只能等于 tree.version。后果：无法按多个评价口径输出。", "底层状态/节点保持稳定，标准仅定义输出投影视图。", "已修复：damage_tree.standards 声明多视图；结果、manifest、Parquet、NPZ、Torch 全链路保留标准映射与请求 mask。"],
        ["P1-06", "DAG 可出现重复节点/事件/输入、循环、孤儿或未知引用。后果：规则错误可能静默进入批量标签。", "毁伤树应在仿真前一次性强类型编译。", "已修复：唯一性、引用、循环、可达性、k 范围、概率有限性与叶状态覆盖全部校验。"],
        ["P1-07", "小样本 95% 区间固定使用 1.96。后果：少量重复时区间偏窄、过度自信。", "未知方差且样本少时使用 Student-t 临界值。", "已修复：df 1–30 使用 t 值，较大样本转 1.96；概率区间裁剪到 [0,1]，非有限数拒绝。"],
        ["P1-08", "自适应停止只看部分输出。后果：某些分机制状态尚未收敛时提前结束。", "停止规则应覆盖全部训练概率标签，并明确排除的指标。", "已修复：覆盖树输出和 fragment/blast/combined 全部状态；至少 32 次；输出实际最大半宽和 target_met。"],
        ["P1-09", "部件综合仍采用条件独立，却未在结果中声明。后果：用户把降阶近似当作真实相关性。", "模型形式假设必须进入结果与数据卡。", "已修复：combined_dependency_assumption、警告和 manifest 限制均显式记录；未伪称完全耦合。"],
    ]
    add_caption(doc, "表 5　统计、毁伤树与评价标准问题整改")
    add_table(doc, ["编号", "问题与可能后果", "解决原理", "实际解决与证据"], logic_issues,
              widths=[0.62, 2.25, 1.8, 2.33], font_size=7.75)

    data_issues = [
        ["P0-08", "厘米字段只在 Schema 出现，运行时仍按米字段读取。后果：用户给 cm 时失败或数量级错误。", "输入在几何编译和哈希前必须规范化为唯一 SI 表示。", "已修复：normalize_target 深拷贝转换；禁止 m/cm 混用；m 与等价 cm 得到同一 canonical hash，CLI/直调/仿真测试通过。"],
        ["P0-09", "训练特征缺姿态、装甲几何/路径、源项分布统计和中间标签。后果：网络无法区分关键工况，只能学习错误捷径。", "教师数据要覆盖决定性输入，并保留可解释中间量与有效掩码。", "已修复：case 59、component 38、armor 18、path 4、aux label 14 维；完整旋转、任意多边形顶点、图边与单位均导出。"],
        ["P0-10", "随机行级 split 可能把同一物理工况的不同种子拆开。后果：测试精度虚高。", "按物理工况组划分；标签视图不应改变物理分组。", "已修复：split_group→condition_cluster→physical_case_hash；物理哈希排除 seed/重复数/标准 ID；自动泄漏审计。"],
        ["P1-10", "禁用机理的人造 0 会被当有效标签。后果：代理网络学习到错误零概率。", "值和可用性必须分离。", "已修复：component/system/aux label mask；标准请求决定 system_label_mask。"],
        ["P1-11", "批处理中单案例异常可使整批丢失，重复 case_id 和集合迭代还会破坏顺序。", "批生成默认隔离失败，所有输出顺序必须由输入和稳定轴定义。", "已修复：默认 isolate-and-record，可选 --fail-fast；拒绝重复 ID；串并行保持输入顺序。"],
        ["P1-12", "Torch 适配器每取样本重复解压 NPZ，并持有不易 pickle 的状态。后果：Windows 多 worker 性能差或启动失败。", "每进程一次加载，数据对象保持可 pickle。", "已修复：懒加载缓存、spawn 兼容、split 直接写 NPZ；字段选择和 copy_on_read 可配。"],
        ["P1-13", "写数据时缺少完成标志，manifest 可能先于其余文件可见。后果：训练读到半成品。", "manifest 最后原子发布，未完成目录必须拒读。", "已修复：.dataset.incomplete 标志、manifest 原子替换、Torch 读取器拒绝未完成数据集。"],
        ["P1-14", "wheel 只发现 damage_engine，运行期 Schema 位于包外。后果：源码可运行，安装后校验失败。", "运行时资源必须进入发布物并做源码树外烟测。", "已修复：schemas 作为 package-data 打包；最终 wheel 含 4 个 Schema，安装后仿真成功。"],
        ["P1-15", "代码哈希原先不含 Schema。后果：修改输入契约却可能保留同一引擎指纹。", "决定运行语义的代码与 Schema 必须共同进入指纹。", "已修复：ENGINE_CODE_HASH 合并 Python 源码树和 schema_tree_hash；结果/manifest 均记录。"],
        ["P1-16", "依赖只有下界，环境无法精确重建。后果：不同 NumPy/jsonschema/pyarrow 版本产生漂移。", "验证基线需要精确锁定。", "已修复：requirements.txt/requirements-lock.txt 固定验证版本；repro_check 所有必备项通过。"],
    ]
    add_caption(doc, "表 6　数据集、发布与复现问题整改")
    add_table(doc, ["编号", "问题与可能后果", "解决原理", "实际解决与证据"], data_issues,
              widths=[0.62, 2.25, 1.8, 2.33], font_size=7.65)

    # 5
    doc.add_heading("5　面向神经网络的数据接口评估", level=1)
    add_body(
        doc,
        "整改后的数据产品同时保留 Parquet 长表、JSONL 审计结果和 NPZ 张量。Parquet 适合跨案例按列筛选与后续分片；NPZ 适合固定拓扑、小中规模训练。所有数组在 manifest 中声明 dtype、shape、语义角色、单位和内容哈希。输出不仅包含最终系统概率，还包含部件状态、破片/冲击波中间量及 Monte Carlo 不确定度，支持多任务代理模型。",
    )
    tensor_rows = [
        ["case_x", "[B,59]", "交汇、战斗部、源项统计、环境、机理、耦合、保真度、采样控制"],
        ["component_x", "[B,C,38]", "形状、位置、3×3 姿态、尺寸、厚度、易损率、P-I 阈值、暴露"],
        ["armor_x", "[B,A,18]", "装甲几何、拉伸、面积/周长/顶点数、响应参数"],
        ["component_armor_path_x", "[B,C,A,4]", "遮挡、入射距离、路径长度、入射余弦"],
        ["component_y", "[B,C,3,S]", "破片/冲击波/综合 × 部件状态概率"],
        ["component_aux_y", "[B,C,14]", "命中、穿透、着速、裕度、能量、峰值、冲量、时序、遮蔽"],
        ["system_y", "[B,O]", "稳定的毁伤树输出轴；由 system_label_mask 选择案例有效标签"],
        ["standard_output_mask", "[E,O]", "评价标准到输出节点的固定映射"],
        ["requested_standard_mask", "[B,E]", "逐案例请求了哪些评价标准"],
        ["不确定度与掩码", "同主标签轴", "stddev、ci95、feature/label mask、实际 replicate_count"],
        ["图与多边形", "CSR/edge arrays", "任意顶点、毁伤树边、节点/边类型，避免丢失拓扑"],
    ]
    add_caption(doc, "表 7　张量契约 1.1.0 的核心字段")
    add_table(doc, ["字段", "形状", "学习语义"], tensor_rows, widths=[1.85, 1.25, 3.9], font_size=8.35)

    doc.add_heading("5.1　数据集质量门", level=2)
    add_bullet(doc, "每个案例保留 case_hash、target_hash、registry_hash、engine_code_hash、随机种子、认知样本 ID 与实际重复数。")
    add_bullet(doc, "split 按物理组生成；同一物理输入只改变 MC seed、重复数或评价口径时仍进入同一组。")
    add_bullet(doc, "禁用机理、未请求标准和缺失参数均通过 mask 表达，不以数值 0 混淆“真实零”与“不可用”。")
    add_bullet(doc, "批生成默认记录拒答，不将拒答案例伪装成正常训练标签；输入到有效样本索引显式保存。")
    add_bullet(doc, "当前缺口：装甲完整度/失效尚无 armor_y；跨目标 C/A 轴仍需项目自定义 padding/collation；全量驻内存，暂无分片与断点续算。")

    # 6
    doc.add_heading("6　性能评估与优化效果", level=1)
    add_body(
        doc,
        "所有性能数字均来自同一台 Windows 11、Python 3.13.5 环境和 synthetic demo 目标，不能直接外推到复杂无人机模型。优化验收同时要求结果等价；串行与并行张量逐数组完全一致。",
    )
    perf_rows = [
        ["端到端引擎", "40 cases × 4 reps", "3.7265 s；10.734 cases/s", "1.2882 s；31.052 cases/s", "2.89×"],
        ["cProfile 总计", "1 case × 40 reps", "1.498 s", "0.475 s", "3.15×"],
        ["simulate_fragments", "同上", "1.425 s", "0.405 s", "3.52×"],
        ["三角窄相位", "同上", "1.166 s", "0.087 s", "13.40×"],
        ["数据集串行", "48 cases × 2 reps", "—", "1.4024 s；34.23 cases/s", "基线"],
        ["数据集 2 workers", "同上、压缩", "—", "1.1073 s；43.35 cases/s", "1.27× vs 串行"],
    ]
    add_caption(doc, "表 8　性能复测（3 次运行中位；cProfile 为单次剖析）")
    add_table(doc, ["指标", "负载", "整改前", "整改后", "变化"], perf_rows,
              widths=[1.35, 1.45, 1.55, 1.75, 0.9], font_size=8.2)
    add_body(
        doc,
        "解释：最大收益来自拉伸多边形窄相位的预计算与标量实现。当前最终热点转移到 ray_hits、宽相位 candidate 选择和弹道响应。简单 demo 上双进程仅获得约 1.27×，说明进程启动、序列化和文件写出占比已不可忽略；复杂目标/更多重复时并行收益可能更高，应在目标硬件和真实目标规模上重新冻结基准。",
    )
    add_body(
        doc,
        "存储权衡：48 案例基准中压缩 tensors.npz 为 26,122 B，不压缩为 132,708 B，约 5.08×；本次小样本压缩并未变慢。大规模生产仍应以分片 Parquet/Arrow 为主，而不是把单个 NPZ 无限放大。",
    )

    # 7
    doc.add_heading("7　验证证据与可复现结果", level=1)
    verification_rows = [
        ["自动化测试", "python -m pytest -q", "73 项全部通过"],
        ["语法/导入", "python -m compileall -q damage_engine benchmarks tests", "通过"],
        ["输入校验", "hash-target / validate / simulate", "规范化哈希一致；demo 成功"],
        ["数据生成", "workers=1 与 workers=2", "3/3 有效、0 拒答；67 个数组完全相同"],
        ["manifest", "Draft 2020-12 校验", "0 errors"],
        ["发布包", "pip wheel + --target 安装", "v0.3.0；4 Schema；源码树外 status=ok"],
        ["复现检查", "test-research-code repro_check", "README/ENV/SEED/ENTRYPOINT/DATA/RESULTS_CMD 全部 OK"],
        ["演示收敛", "40 reps, target half-width=0.05", "未达到；最大半宽 %.4f，正确报告 target_met=false" % achieved_width],
    ]
    add_caption(doc, "表 9　最终验证矩阵")
    add_table(doc, ["证据", "方法", "结果"], verification_rows, widths=[1.4, 2.7, 2.9], font_size=8.6)

    doc.add_heading("7.1　最终可追溯标识", level=2)
    trace_rows = [
        ["Engine code hash", engine_hash],
        ["Schema tree hash", schema_hash],
        ["Target canonical hash", result["provenance"]["target_hash"]],
        ["Physics registry hash", result["provenance"]["physics_registry_hash"]],
        ["Dataset ID", dataset_id],
        ["Wheel", "damage_dataset_engine-0.3.0-py3-none-any.whl；SHA-256 3b0bda12095558bde93c80207834e7baf2cdc9109bb775c4d87d33782c06a49b"],
    ]
    add_table(doc, ["标识", "值"], trace_rows, widths=[1.65, 5.35], font_size=8.2)

    # 8
    doc.add_heading("8　尚未解决的风险与后续优先级", level=1)
    add_body(
        doc,
        "以下不是可以靠补几行代码诚实消除的问题。它们决定本引擎能否从“工程可算”升级为“适用域内可信”。在完成前，应把所有输出标为模型预测而非真实效能事实。",
    )
    remaining_rows = [
        ["R0-01", "正式物理标定/确认缺失", "最高", "建立单板、双层、部件、舱室、系统分层试验；分离 calibration/validation；冻结误差与覆盖率判据。"],
        ["R0-02", "演示注册表全部 unverified", "最高", "用批准来源替换 demo 参数；每个模型记录适用域、证据、标定 ID 与独立复核。"],
        ["R1-01", "破片源项非联合分布，离散清单未实现", "高", "实现 fragment_list_ref 解析、质量守恒和联合采样；与实测破片谱/更高保真源项交叉确认。"],
        ["R1-02", "无偏转/跳弹/二次破片/孔洞重叠", "高", "按预期用途分级增加模型；先用单元试验确认，再接入系统。"],
        ["R1-03", "冲击波无完整时程、多路径、泄压网络", "高", "引入经标定的舱室降阶模型或高保真耦合；不要把经验指数透射宣称为 CFD。"],
        ["R1-04", "部件机制联合仍为条件独立", "高", "用共享随机变量、共同原因节点或经数据标定的条件模型；证据不足时输出上下界。"],
        ["R1-05", "装甲标签未形成 armor_y", "高", "下一张量版本加入装甲完整度、失效概率、反射峰值/冲量及不确定度/mask。"],
        ["R1-06", "自适应停止不覆盖连续物理量", "中高", "按 Pa、Pa·s、J、m/s 等分别定义绝对/相对误差门；稀有事件采用专门准则。"],
        ["R2-01", "批生成全量驻内存", "中", "实现分片、checkpoint/resume、失败分片重跑和最终事务式目录发布。"],
        ["R2-02", "复杂目标宽相位仍近似 O(F·N)", "中", "在真实部件数与破片规模上剖析后引入 BVH/空间索引；用金样验证事件等价。"],
        ["R2-03", "跨目标训练缺统一 collation", "中", "定义 ragged/graph batching、稳定类型词表、padding/mask 和目标拓扑留出方案。"],
        ["R2-04", "真实 PyTorch DataLoader 未在本机实测", "中", "安装与目标训练环境一致的 CPU/CUDA 版本，做 spawn、吞吐和内存基准。"],
        ["R3-01", "顶层包名 schemas 可能冲突", "低", "后续迁入 damage_engine/schemas 并用 importlib.resources；保持 wheel 烟测。"],
    ]
    add_caption(doc, "表 10　剩余风险与建议")
    add_table(doc, ["编号", "风险", "优先级", "建议闭环"], remaining_rows,
              widths=[0.65, 1.85, 0.65, 3.85], font_size=7.8)

    doc.add_heading("8.1　进入神经网络训练前的放行门", level=2)
    add_bullet(doc, "冻结预期用途和适用域；定义哪些目标、战斗部、角度、距离、材料和环境允许训练/推理。")
    add_bullet(doc, "至少完成关键物理子模型的独立验证，并给出误差/区间覆盖，而非只看系统最终等级。")
    add_bullet(doc, "建立按目标拓扑、战斗部族和工况簇分组的外部测试集；不得只做随机行切分。")
    add_bullet(doc, "代理模型同时预测中间量、部件状态和系统输出；检查单调性、守恒、反事实一致性和概率校准。")
    add_bullet(doc, "上线数字孪生前加入 OOD 检测、不确定度阈值、回退到仿真器/保守策略以及人工复核；控制器单独做安全论证。")

    # 9
    doc.add_heading("9　主要改动文件", level=1)
    file_rows = [
        ["物理/几何", "damage_engine/fragments.py, blast.py, geometry.py, parameters.py, registry.py", "串列追踪、P-I 稳定性、装甲失效、面积混合透射、温度、阈值、热点优化和域守卫"],
        ["编排/统计", "damage_engine/engine.py, utils.py, normalization.py", "规范化哈希、多标准结果、依赖/收敛元数据、Student-t 区间、代码+Schema 指纹"],
        ["毁伤树/校验", "damage_engine/damage_tree.py, validation.py", "显式依赖、DAG 完整性、多标准、单位/姿态/有限性/模型引用校验"],
        ["数据集", "damage_engine/dataset.py, torch_dataset.py", "张量 1.1、分组 split、标准 mask、失败隔离、原子发布、spawn 兼容"],
        ["契约/发布", "schemas/*.json, pyproject.toml, requirements*.txt", "Schema 收紧与打包、版本 0.3.0、依赖锁定"],
        ["验证/基准", "tests/*.py, benchmarks/*.py", "73 项测试、引擎与数据生成基准"],
        ["示例/说明", "README.md, result.demo.json, dataset_demo*", "最新命令、限制、演示结果与串并行数据产物"],
    ]
    add_table(doc, ["类别", "文件", "核心变化"], file_rows, widths=[1.0, 2.65, 3.35], font_size=8.15)

    # 10
    doc.add_heading("10　最终结论", level=1)
    add_body(
        doc,
        "本轮整改解决了会直接扭曲毁伤结果或训练数据的关键实现问题：串列部件漏算、装甲冲击波不受损、破孔耦合错误、相关事件误判独立、厘米输入未执行、阈值字段未消费、数据泄漏、标签语义缺失、串并行不稳定、安装包缺 Schema 以及主要几何热点。整改后的引擎已经具备作为“可校准教师模型工程底座”的结构与接口。",
    )
    add_body(
        doc,
        "同时，真实可信度的决定因素仍是物理参数与模型的验证/确认。建议下一阶段不再盲目增加公式，而是先确定预期用途和批准适用域，获取分层试验/高保真基准，按 R0/R1 项完成标定、独立确认与误差量化；随后再大规模生产教师数据并训练代理网络。只有这样，数字孪生和智能控制使用的才是带证据边界的预测，而不是更快的未知偏差。",
    )

    # Appendix
    doc.add_page_break()
    doc.add_heading("附录 A　复现命令", level=1)
    commands = [
        "python -m pip install -r requirements.txt",
        "python -m pytest -q",
        "python -m damage_engine hash-target --target examples\\target.demo.json",
        "python -m damage_engine validate --target examples\\target.demo.json --case examples\\case.demo.json --physics examples\\physics.demo.json",
        "python -m damage_engine simulate --target examples\\target.demo.json --case examples\\case.demo.json --physics examples\\physics.demo.json --output result.demo.json",
        "python -m damage_engine generate --target examples\\target.demo.json --cases examples\\cases.demo.jsonl --physics examples\\physics.demo.json --output-directory dataset_demo --workers 1",
        "python -m benchmarks.benchmark_engine --cases 40 --replicates 4 --warmup 2",
        "python -m benchmarks.benchmark_dataset --cases 48 --replicates 2 --parallel-workers 2",
        "python -m pip wheel . --no-deps --no-build-isolation -w .wheel_test",
    ]
    for command in commands:
        add_command(doc, command)

    doc.add_heading("附录 B　参考依据", level=1)
    references = [
        "[R1] NASA, NASA-STD-7009B: Standard for Models and Simulations, 2024. https://standards.nasa.gov/sites/default/files/standards/NASA/B/1/NASA-STD-7009B-Final-3-5-2024.pdf",
        "[R2] NISTIR 8298, A Summary of Industrial Verification, Validation, and Uncertainty Quantification Procedures in Computational Fluid Dynamics, 2020. https://doi.org/10.6028/NIST.IR.8298",
        "[R3] U.S. Department of Defense, UFC 3-340-02, Structures to Resist the Effects of Accidental Explosions. https://www.wbdg.org/dod/ufc/ufc-3-340-02",
        "[R4] Apache Arrow Documentation, Parquet and Python data interfaces. https://arrow.apache.org/docs/python/parquet.html",
        "[R5] PyTorch Documentation, torch.utils.data. https://docs.pytorch.org/docs/stable/data.html",
        "[R6] JSON Schema, Draft 2020-12. https://json-schema.org/draft/2020-12",
    ]
    for reference in references:
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.left_indent = Cm(0.35)
        paragraph.paragraph_format.first_line_indent = Cm(-0.35)
        paragraph.paragraph_format.space_after = Pt(5)
        run = paragraph.add_run(reference)
        set_run_font(run, size=8.8, color=DARK)

    doc.add_heading("附录 C　报告适用声明", level=1)
    add_body(
        doc,
        "本报告的结论绑定于 2026-09-26 工作区快照、上列代码/Schema 指纹、示例目标与演示物理注册表。后续任何代码、参数、Schema、毁伤树或适用域变化都应触发相应级别的回归、再标定与再确认。报告中的性能数字是本机工程基准，不是产品性能承诺；报告中的演示概率是软件路径检查结果，不是实装效能结论。",
    )

    configure_sections(doc)
    for section in doc.sections:
        add_header_footer(section)
    doc.core_properties.title = "仿真引擎机理与性能评估整改报告"
    doc.core_properties.subject = "破片/冲击波毁伤引擎、毁伤树和神经网络数据接口整改"
    doc.core_properties.author = "Codex"
    doc.core_properties.keywords = "毁伤仿真, 破片, 冲击波, 毁伤树, 数据集, V&V, UQ"
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build_report()
