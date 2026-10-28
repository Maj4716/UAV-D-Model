# R06e 下一次提交计划

1. 只提交 `r06e_0800_ctrl`（Wave 0），验证 R15.0.2 关键词解析和完整运行。
2. Wave 0 必须同时满足：无 KEY 错误、修复卡片无 excess-field warning、cycle > 0、正常终止、D3PART 与 INTFOR 均可读。
3. Wave 0 通过后只提交 `r06e_2000_ctrl`；两个 control 都通过 run-integrity 与输出可读性后，才进入 treatment。
4. 最后提交 `r06e_0800_tend008` 与 `r06e_2000_psfail_plate`，分别只与自己的 control 配对解释。
5. 任一阶段失败即停止；不得把失败或 unavailable 响应写成 0，也不得把诊断作业加入参考训练分布。
