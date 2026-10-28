# Job 72397333 / R06e 800 m/s control 回传审查工作日志

日期：2026-10-27

1. 将 `D:\Desktop\UAV新\06` 作为只读原始证据；未改名、未移动、未写入该目录。
2. 核对 `manifest.json`、candidate manifest、job contract、deck 大小与 SHA-256；提交身份一致。
3. 扫描 solver `o=`、stdout、stderr 和全部 32 个 `mes*`；确认旧 KEY 解析错误全部消失，32/32 ranks 正常结束。
4. 核对实际资源为 1 node / 32 ranks / R15.0.2 MPP double / 512M+256M，许可证成功取得。
5. 解析 ASCII GLSTAT：能量比范围 `0.957156–1.00268`，hourglass/peak internal 为 `0`，无质量缩放。
6. 用项目固定的 `lasso-python 2.0.4` 只读打开 `binout*`、`d3part*`、`d3plot*`。
7. 确认 D3PART 802 states、fragment 686 solids，solver-realizable timeline 通过；完整 D3PLOT 402 states、20,902 solids。
8. 确认 `Warning 20340` 与 `*CONTACT_ERODING_SINGLE_SURFACE` 的组合导致 INTFOR 不可用；未把缺失值解释为 0。
9. 交叉比对 402 个共同状态：D3PART 唯一历史通道逐值等于完整 D3PLOT 第 7 通道，不是目标 JC damage；目标 damage 位于完整 D3PLOT 第 6 通道。
10. 按现有冻结捕获账本算法计算 800 m/s 响应，并与 R05 Job 72252362 同工况比较；两者都不具备主 QoI 可用性，且轨迹不复现。
11. 严格按冻结 R06e Wave 0 条文判定 `FAIL/HOLD`；将“成功执行”与“输出门通过”分开记录。
12. 提出 prospective R06f：先修测量合同和适配器，再做 800 m/s recanary 与时间步敏感性，之后才释放 2000 m/s control。

边界：本日志没有修改物理参数、材料参数、接触模型、仿真引擎或数据集生成器，也没有产生 P0/P1 或训练标签准入。
