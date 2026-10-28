# 基于 R04 Stage 0-2 的高保真仿真接入建议

## 当前可用证据

R04 已建立可信的归档身份、输入身份、求解器正常终止和全局能量历史。最终总能量比位于候选区间，小时玻璃能为零；但 `874` 个单元发生负体积侵蚀，且逐工况穿孔、残速和网格一致性尚未建立。因此当前结果只能用于完善高保真数据合同和质量门，不能直接替换本地装甲响应参数或 KMFC 阈值。

## 仿真引擎的优化顺序

1. 增加版本化 `high_fidelity_run` 与 `numerical_qa` artifact，绑定 solver build、deck/material/geometry/case-map hash、输出 hash 和 QA 状态；QA 未通过时 fail closed。
2. 保留 `demo_armor_response`，以后新增独立的高保真响应模型 ID。先把穿孔状态、残余速度和能量转移拆成三个可单独校准的模块，避免一条经验曲线承担全部误差。
3. 高保真结果只进入 teacher/audit 通道；求解器状态、接触结果、删除单元和 d3plot 派生量不得成为部署时 learner 输入。
4. 每个拟合 artifact 保存适用域、网格层级、材料认识变体、不确定度和 OOD 拒绝边界；不要在适用域外静默外推。
5. 将负体积、侵蚀能、时间步突降和网格不一致作为样本级 mask/拒绝原因，而不是把数值异常编码成真实毁伤标签。

## 数据集生成器的优化顺序

1. 继续冻结参考 `p0`、Validation/Test 和标签语义；高保真自适应选点只允许在 Train/calibration 池进行。
2. 建立 `hf_numerical_verification`、`hf_response_fit`、`hf_response_holdout` 三类明确角色，P1 的 18/6 fit/holdout 身份保持冻结。
3. 保存连续监督量及可用性 mask：穿孔、残速、吸收/转移能量、侵蚀比例、响应是否稳定；不要把 unavailable 写成 0，也不要直接把单元删除映射成 KMFC 等级。
4. 用低保真大候选池加高保真小样本做多保真残差校准或代理模型；选点依据应是输入空间覆盖、代理不确定度和预注册边界邻域，不是已观察 Test 或标签均衡目标。
5. 保留随机参考点和独立高保真 holdout；输出选择概率、design weight、父 case lineage 和求解器 cohort，禁止跨网格/材料/求解器 cohort 无标识混池。
6. 只有 Stage 3-4 证明 P0 的逐工况响应与网格行为可接受后，才允许用这些连续 QoI 校准本地引擎，再生成新版本数据；旧数据和旧 response-model ID 不覆盖。

## 当前停止线

在 BINOUT/d3plot 可读、侵蚀空间位置得到复核、残速稳定且三层网格结论一致之前，不提交 P1，不修改材料参数，不重写 KMFC 树，不生成新的科研训练数据。
