# SCNet / LS-DYNA 破片—装甲 P0 设计、提交与回传校准记录

**日期：** 2026-10-17  
**项目：** `D:\Desktop\UAV新`  
**本地包：** `scnet_lsdyna_plate_calibration_20261017`  
**当前阶段：** `r01_field_width_failed / r02_required_cards_failed / r03_repaired_locally / resubmission_pending`  
**证据上限：** `high_fidelity_synthetic_reference_candidate`；不等同于实弹试验、真实车辆验证或经验标定。

---

## 1. 本轮目标与决策边界

本轮不从 KMFC 标签频率反向调节材料或威力参数，而是先建立“破片质量—速度—装甲厚度—斜入射—姿态—材料不确定性”到板级连续响应的外部高保真参考。第一份作业只承担 P0 数值资格检查和中心切片速度响应检查；不直接生成 K/M/F/C 等级，不用于重写当前 Test，也不以提高标签均匀度作为物理参数选择标准。

经确认的后续设计域为：

| 因子 | 范围/水平 |
|---|---|
| 破片质量 | 5、8、10 g |
| 初速 | 1200–2600 m/s |
| 板厚 | 6–14 mm |
| 相对法线斜角 | 0–60° |
| 姿态 | face、edge、corner |
| 材料认识不确定性 | low、central、high |

算力不设固定硬上限。申请资源必须服从 SCNet 模板和队列限制，并在 manifest 中记录申请值、实际实耗和扩容依据。

## 2. P0 作业设计

P0 把 7 个相互隔离的四分之一对称子模型并列写入同一 LS-DYNA deck：

- 8 g、10 mm、法向、立方体面迎击、4340 中心材料；
- 中等网格 0.75 mm 下的 1200、1600、2000、2300、2600 m/s 速度扫描；
- 2000 m/s 下的 1.00、0.75、0.50 mm 三层网格检查；
- 中等网格 2000 m/s 工况同时充当独立响应留出点；
- 全积分 S/R 实体，未启用质量缩放；
- 输出 `d3plot`、`glstat`、`matsum`、`sleout`、`nodout` 及历史节点。

四分之一对称只对居中、法向、face-on 中心切片成立。P1 的斜入射和 edge/corner 姿态必须改用全模型或经过独立证明的相应对称模型，不能复制 P0 的四分之一边界。

## 3. 材料模型与认识限制

P0 使用 AISI 4340 Johnson–Cook 公开参考参数。强度和失效参数分别绑定 Johnson–Cook 1983/1985 来源；LS-DYNA `*MAT_JOHNSON_COOK` 固体单元配套 `*EOS_GRUNEISEN`。EOS 中心值来自公开的通用钢参考，并非 4340 特定热处理的独立标定，因此在包内被明确标记为可检验的认识假设。

该设计可以回答“在统一公开参考体系中，当前简化装甲响应曲线是否处于合理数量级”，不能回答“项目装甲车真实钢种的抗毁伤能力是多少”。后续 low/central/high 材料组用于传播认识不确定性，不能被解释成真实批次公差。

## 4. 本地验证结果

`pytorch312-lmdt` 环境中的静态校验已通过：

| 检查 | 结果 |
|---|---:|
| P0 工况 | 7 |
| 节点 | 184,949 |
| 实体单元 | 163,730 |
| part ID | 14 个且唯一 |
| 单元方向 | 结构化六面体角点有向体积全部为正 |
| 初始板—破片间隙 | 7/7 均为 1.0 mm |
| 每个四分之一破片网格质量 | 约 0.002 kg；乘 4 后为 8 g |
| 每个四分之一板体积 | 36,000 mm³ |
| P1 拟合/留出 | 18/6 |
| P1 各离散因子拟合计数 | 每水平 6 |
| P1 各离散因子留出计数 | 每水平 2 |
| 后处理器合成自测 | PASS |

静态 PASS 只支持包内一致性。LS-DYNA 关键词解析、接触稳定性、能量闭合、侵蚀、穿孔状态和网格收敛均仍待平台实算。

## 5. P0 准入门

P0 进入响应拟合前至少应满足：

1. LS-DYNA 正常终止，无未解释的 fatal/error termination；
2. 总能量比候选区间 0.95–1.05；
3. 小时玻璃能/峰值内能绝对比候选上限 5%；全积分模型出现显著小时玻璃能时必须解释；
4. 无质量缩放；附加质量非零时拒绝；
5. 负体积、删除单元和接触告警可定位到工况与 part；
6. 三层网格的穿孔状态一致，剩余速度进入预注册误差带；
7. 速度扫描至少包含一个未穿孔点和一个穿孔点，否则先扩展速度边界，不能直接拟合 V50。

这些是本项目候选工程门，不是 LS-DYNA 的普适标准。能量门通过也不能单独证明物理正确。

## 6. 回传数据合同

必须保留：输入 deck、case map、完整 solver build、MPI 实现、节点/ranks、队列、墙钟申请与实耗、作业 ID、`d3hsp`、`messag/mes*`、`glstat`、`matsum`、`sleout`、`nodout`、`d3plot*`、标准输出/错误以及各文件 SHA-256。

回传后按以下顺序处理：

1. 先判定求解器、输入和输出身份是否与 manifest 一致；
2. 再判定终止、能量、质量、时间步、负体积和侵蚀等数值 QA；
3. 逐工况提取穿孔状态、剩余速度、能量转移及可用性 mask；
4. 只用 P0 拟合候选板响应形式和网格误差，不触碰当前 Test；
5. P0 通过后才生成 P1 全模型 deck；18 个 `hf_response_fit` 用于拟合，6 个 `hf_response_holdout` 只用于外推检查；
6. 新模型使用独立版本 ID，保留现有 `demo_armor_response` 与旧归档身份。

## 7. 平台状态、R01 失败与 R02 修复

已登录的 SCNet 内置浏览器标签存在。首次尝试读取新作业模板时，Browser Use origin gate 拒绝了对 `https://www.scnet.cn` 的页面观察。用户随后通过 `@浏览器` 显式附加同一标签，第二次访问仍返回 `A saved user permission setting or managed policy denies access`。本机诊断确认设置界面与 `C:\Users\Administrator\.codex\config.toml` 都已把该精确 origin 的 `access`、`downloads` 设为 `allow`，且未发现用户级或常见机器级 `requirements.toml`。用户完整重启 Codex 后，新标签已位于 `#/job-submit?tab=template`，但第三次绑定仍被同一 origin gate 拒绝。因此普通进程缓存已被排除，当前剩余诊断为账户/工作区下发的不可见管理策略或 Browser Use 权限合并缺陷。安全策略禁止通过其他浏览器接口或底层命令规避。当前未填写表单、未上传、未产生费用、未提交作业。

2026-10-19，用户再次显式调用 Computer Use 并授权直接在平台提交。提交前本地静态校验再次 PASS，deck SHA-256 未变化；Computer Use 尝试创建 SCNet 提交标签时仍被 origin gate 拒绝，且错误明确禁止替换浏览器表面或间接绕过。该尝试没有建立网络连接、上传文件或创建作业。

由于自动控制无法通过站点安全门，已生成 `40_SCNet_LSDYNA_P0手工提交与回传操作清单_2026-10-19.md`，并把六个待上传文件集中到 `scnet_lsdyna_plate_calibration_20261017/manual_upload_20261019/`。手工方案固定为 MPP 双精度、单节点共享队列、优先 30–32 核、4 小时墙钟上限、关闭独占节点与 GPU；求解输入仅选择 `.k`，其余五个文件作为审计附件保存在同一工作目录。

2026-10-19，用户按手工清单提交 R01，SCNet Job ID 为 `71903517`。平台实际启动 `LS-DYNA mpp d R15.0.2`、Intel MPI 2018、32 ranks、512/256 MB，证明版本、队列和启动路径均已绑定成功；作业在 4 秒、0 个计算循环后于关键词读取阶段失败。`mes0000` 报告数值项过长、数据格式不正确，并以 `keyword input phase failed with: 227 errors` 终止。根因是生成器输出 15–16 字符的 `.9e` 数值，而 R01 使用未声明长字段的普通 `*KEYWORD`。

2026-10-20，生成器已改为 `*KEYWORD LONG=Y`，静态校验器新增标准 10 字符/长格式 20 字符边界检查，并用原 R01 deck 验证能够捕获该缺陷。R02 最大数值项宽度为 16，格式检查及原有网格、质量、哈希和 P1 分层检查均 PASS。R02 独立上传目录为 `scnet_lsdyna_plate_calibration_20261017/manual_upload_20261020_r02/`，应使用作业名 `uav_hf_p0_plate_r02` 和新远程目录提交。完整证据与排除过程见 `41_SCNet_LSDYNA_P0_R01失败根因与R02修复_2026-10-20.md`。

2026-10-20，R02 以 Job ID `72021951` 提交。求解器确认 `long option: yes`，证明 R01 修复已生效，但随后因 `*CONTROL_CONTACT` 缺少第 2 张必需卡，以及七组 `*INITIAL_VELOCITY_GENERATION` 的结构不完整并把整数型 `PHASE` 写成科学计数浮点数，而在关键词阶段汇总 17 个错误。2026-10-21，R03 已补齐控制卡和初速度卡、显式使用整数型 `PHASE/IRIGID`，明确设置四分之一模型的 `ISYM=1, EROSOP=1, IADJ=1`，并将接触控制改为有来源的 R15 默认值加 `ISLCHK=2` 完整初始穿透检查。校验器现在会拒绝 R02 的缺卡和字段类型。完整证据见 `42_SCNet_LSDYNA_P0_R02失败根因与R03修复_2026-10-21.md`。

## 8. 可复核提交物

| 对象 | 路径/摘要 |
|---|---|
| 求解输入 deck | `scnet_lsdyna_plate_calibration_20261017/decks/p0_velocity_mesh_calibration.k` |
| deck SHA-256 | `7063a40d26dc7e044674b446788af348b3ea47beb78368c6be48c7b278e0ef93`（R03） |
| 确定性复现包 | `scnet_lsdyna_plate_calibration_20261017/artifacts/scnet_lsdyna_plate_calibration_p0_20261017.zip` |
| ZIP SHA-256 | `2e2ae3d044588a1cc08f020043162768589f4c4b36195f5028251a4ee418a573`（R03） |
| ZIP 完整性 | 17 项（含内嵌 manifest）；CRC 自检无错误；不含 `results/` 或 `manual_upload_*` |
| 静态校验报告 | `scnet_lsdyna_plate_calibration_20261017/validation/static_validation.json` |
| P1 冻结设计 | `scnet_lsdyna_plate_calibration_20261017/design/p1_preregistered_cases.csv` |

平台直接求解时上传未压缩 `.k`；ZIP 用于审计、复现与回传身份核对。

## 9. 来源

- [SCNet LS-DYNA 使用说明](https://www.scnet.cn/help/docs/mainsite/hpc/software-examples/use/CAE/LSDYNA/index.html)
- [SCNet 高级作业提交说明](https://www2.scnet.cn/help/docs/mainsite/hpc/advanced-guide/advanced-submit-job/index.html)
- [Ansys LS-DYNA Projectile 官方示例](https://lsdyna.ansys.com/projectile/)
- [Ansys LS-DYNA 结构冲击建议](https://lsdyna.ansys.com/recommendations-for-structural-impact-2/)
- [Ansys LS-DYNA Hourglass 说明](https://lsdyna.ansys.com/hourglass/)
- [Ansys LS-DYNA Total Energy 说明](https://lsdyna.ansys.com/total-energy-2/)
- [LS-DYNA Keyword User's Manual, MAT015](https://lsdyna.ansys.com/wp-content/uploads/2025/02/LS-DYNA_manual_Vol_II_R6.0.0.pdf)
- [Johnson–Cook 1983 原始论文扫描件](https://ia600406.us.archive.org/4/items/AConstitutiveModelAndDataForMetals/A%20constitutive%20model%20and%20data%20for%20metals_text.pdf)
