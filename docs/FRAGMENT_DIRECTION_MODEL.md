# 破片方向模型卡

模型卡版本：`fragment-direction-model@3.0.0`

状态：`unverified_engineering_hypothesis`

适用配置：`configs/physics/loitering_impact.json`

## 当前定义

三种弹型均使用 `taylor_ring`，锥轴为首次装甲接触面的内法向
（`axis_mode: impact_inward`）。代码按

\[
\theta_0=\operatorname{atan2}(2V_0,D)
\]

计算基础锥角，并按破片环加入 0–10° 的轴向修正；每个环的极角与方位角
再叠加标准差 2° 的正态扰动。当前锥轴和破片起点不随弹体滚转改变，弹体
速度只作为破片初速度的平移分量叠加。

配置中的来源为 `legacy_code_port_reference_not_retained`。仓库内没有可复核的
原始推导、试验数据或权威模型来源。因此，当前公式是软件教师的一项工程
假设，不具备物理标定状态。

## 已确认的软件性质

- `impact_inward` 使破片场围绕接触面内法向形成空心环带；
- 基础锥角由 Gurney 初速和爆速共同决定；
- 弹体滚转不进入当前方向变换；
- 方向模型会直接改变装甲后部件的几何可达性，是部件零膨胀的高敏感候选；
- 软件等价、几何回归和随机数复现只能证明实现一致，不能证明该方向模型反映现实。

## 候选敏感性臂

| ID | 定义 | 用途 |
|---|---|---|
| F0 | 当前 `impact_inward` 空心 Taylor 环 | 冻结软件基线 |
| F1 | 以同一外角为边界的实心锥 | 检验空心轴区对可达性和稀疏的贡献 |
| F2 | 以弹轴为轴的侧向破片带 | 检验弹体姿态、俯冲角与侧向破片机理 |

F1、F2 是敏感性模型，不是自动晋升候选。比较结果可以说明输出对方向假设的
敏感程度，不能单独判定哪一个模型更接近真实破片场。

## 当前决策

1. 保留 F0 作为现有数据版本的软件教师，注册表状态继续为 `unverified`。
2. F0、F1、F2 使用相同工况和公共随机数进行因子对照；分别报告部件到达率、
   结构零、命中后能量、域完整率和树饱和率。
3. 任何方向模型变更使用新的物理注册表 ID、数据版本和模型卡版本。
4. 在方向模型获得外部依据前，姿态与滚转不作为已确认的物理解释变量；它们
   保留在对照输入中，并在不变性实验通过后从默认学习输入移出。

## 晋升门

方向模型进入正式数据版本需要同时满足：

- 公式、坐标轴和参数范围有可追溯来源；
- 固定工况的三维方向统计回归测试通过；
- 关键稀疏、尾部和完整率结论在预注册参数范围内保持方向一致；
- 新旧模型的差异报告绑定代码、配置、目标和随机种子哈希；
- 数据 manifest 明确记录模型版本与适用范围。

## v2 Taylor–Shapiro 候选

适用配置：`configs/physics/loitering_impact.taylor_shapiro.v2.json`  
适用案例：`configs/cases/loitering_impact.fragment_only.taylor_shapiro.v2.json`  
状态：`unverified_literature_grounded_candidate`

“Taylor–Shapiro”在本项目中不是一个新的固定锥角名称，而是“Shapiro 有符号
偏转 + Taylor 幅值边界”的版本化实现。令 (V_f) 为 Gurney 破片初速，
(V_d) 为爆速，φ₁ 为弹轴正向与局部壳体外法线的极角，φ₂ 为弹轴
正向与起爆点至该壳体点的爆轰传播方向的极角，则

\[
q=\frac{V_f}{2V_d},\qquad
\gamma_t=\arcsin q,
\]

\[
\gamma_s=\arctan\!\left[q\cos\left(\frac{\pi}{2}+\phi_2-\phi_1\right)\right],
\qquad |\gamma_s|\le\gamma_t.
\]

γₛ 是相对局部壳体法线、沿极角增大方向计量的有符号偏转；最终极角为
φₛ=φ₁+γₛ。v2 明确采用弹体 `+X` 为朝弹鼻的零极角方向，前端轴上点起爆，
圆柱内壁为破片源面。环中心沿 `-X` 排列，因此前端起爆时 γₛ 为负，名义方向
从侧向壳体法线向 `+X` 偏转。

> **2026-11-23 更正：上一段的偏转方向与 Taylor（1941）相反。** 按上式取 γₛ 并
> 施加到极角增大方向，破片会偏向起爆端；Taylor 的稳态流构造要求破片偏向
> 爆轰波传播方向，前端起爆时应向 `-X`（向后）偏转。v2/v3 的
> `angle_convention: body_plus_x_polar_from_nose_local_normal_signed` 保留原样，
> 只用于复现已归档数据；新数据使用下文的 v4。证据见
> `artifacts/validation/pipeline_expected_actual_20261122/REVIEW_20261123.md` §4.1
> 和 `artifacts/validation/data_generation_fixes_20261123/`。源点与方向使用同一个环布局对象；不再使用 F0 的
`impact_inward` 轴和 0–10° 经验轴向修正。

v2 的 2° 色散采用名义方向切平面上的二维独立高斯扰动，只作为未标定的对照
参数。它不属于 Shapiro 公式，也不代表已经获得靶场标定。

### 适用前提与拒绝条件

- 轴对称直圆柱壳体、前端轴上点起爆、已知内壁半径和环位置；
- (0<q\le1)，超过 Taylor 定义域直接拒绝，不静默截断；
- 注册表与弹型中的环数、轴向覆盖率、总破片数必须一致；
- 新模型只用于新物理注册表和新数据版本，F0 输出保持原路径。

端板、复杂波形整形、非轴对称预制破片层、有限加速历程和稀疏波端部效应均不
在 v2 模型内。若真实弹体不是普通圆柱侧壁破片战斗部，应建立新的几何模型，
不能把 v2 自动解释为真实弹型。

## v4 波向一致的 Taylor–Shapiro（偏转符号修正）

适用配置：`configs/physics/loitering_impact.taylor_shapiro.wave_directed.obliquity_bound.v4.json`
适用案例：`configs/cases/loitering_impact.fragment_only.taylor_shapiro.wave_directed.obliquity_bound.v4.json`
状态：`unverified_literature_grounded_candidate`

v4 与 v3 只差一处：`angle_convention` 改为
`body_plus_x_polar_deflection_toward_detonation_propagation`，偏转角取

\[
\delta=\arctan\!\left[q\sin(\phi_2-\phi_1)\right],\qquad |\delta|\le\gamma_t,
\]

仍沿极角增大方向施加。子午面内极角增大方向的单位切向量为
(−sin φ₁, cos φ₁)，爆轰传播方向为 (cos φ₂, sin φ₂)，二者点积
sin(φ₂−φ₁) 就是爆轰波沿壳体的切向分量，所以破片偏向爆轰传播方向。
φ₁=90° 时退化为经典 Taylor 情形：波向前传播时破片前倾，向后传播时后倾。
δ 恰好等于 v2/v3 中 γₛ 的相反数，幅值和 Taylor 上界不变。

前端起爆时三个弹型的名义轴向倾角为 −5.1° 至 −5.9°（均值，Gurney 初速
2,000 m/s 时），100% 的破片满足“偏离法线的方向与爆轰切向分量同号”这一
与角度约定无关的判据。实现见 `damage_engine/fragments.py` 的
`taylor_shapiro_deflection`，独立判据测试见
`tests/test_taylor_shapiro_direction.py`。

### 可追溯来源

- G. I. Taylor, *Analysis of the Explosion of a Long Cylindrical Bomb
  Detonated at One End*, R.C.193 (1941)：
  <https://archives.trin.cam.ac.uk/index.php/analysis-of-the-explosion-of-a-long-cylindrical-bomb-detonated-at-one-end>
- U.S. Army, *AMCP 706-290 Warheads—General* (1964)：
  <https://www.bulletpicker.com/pdf/AMCP-706-290.pdf>
- Felix, Colwill and Harris, *Journal of Defense Modeling and Simulation*
  19(4), 747–757：<https://doi.org/10.1177/1548512920982673>
- Banai, *CMES* 88(5), 351–366：<https://doi.org/10.3970/cmes.2012.088.351>
- U.S. Navy patent US8285528B1：
  <https://patents.google.com/patent/US8285528B1/en>
- Chen et al., *Mathematical Problems in Engineering* 2015:723463：
  <https://doi.org/10.1155/2015/723463>

