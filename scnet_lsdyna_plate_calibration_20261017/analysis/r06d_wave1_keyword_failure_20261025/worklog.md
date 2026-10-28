# R06d Wave 1 关键词失败审计工作日志

- 对两份下载 ZIP 做只读清单、CRC、路径安全、哈希和 manifest 绑定检查；未修改原始 ZIP。
- 两份回传 deck 均与本地发布版逐字节一致，排除平台改写和误上传。
- 两份作业均成功启动 R15.0.2、取得许可证并进入 keyword reader。
- 首个致命错误均为 `*DATABASE_EXTENT_BINARY` 缺 Card 2；D3PART 报错是解析错位的级联结果。
- `*DATABASE_BINARY_INTFOR_FILE` 的五字段 timing card 另触发 `KEY+457`，目标 parser 只接受四字段。
- 两份回传均无 d3plot/d3part/intfor/glstat 等积分结果，因此不做数值或物理解释。

## 哈希绑定

- `r06d_0800_ctrl` / Job `72269229`：
  - ZIP `sha256:fe6718e5081580a71db144409f43f9d1fd866343cc244026655c0ec2922ff93f`；57 个成员；CRC PASS。
  - deck `sha256:37f5f2ef4e6d0276e28d05aaf15c4114a4ae26e7655c11ba3ff1b2a8f8725daa`；manifest 与本地发布版一致。
- `r06d_2000_ctrl` / Job `72269336`：
  - ZIP `sha256:cd382fb6e569293b8a7e05462185c3817d22dd604b49da884c5a1e72800be72f`；57 个成员；CRC PASS。
  - deck `sha256:3b61883283b0a68ede37ee40069af5e0ad85ce6fef57a43f572e56573d177ae2`；manifest 与本地发布版一致。

## 修复边界

修复只允许改共享数据库输出卡片的序列化；网格、材料、EOS、接触、时间步、终止时间、速度、边界条件和两个预注册单因素均保持不变。
