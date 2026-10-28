# 受控诊断试验的空间占用与清理记录

> 状态：`validated_current_identity`。

## 持久化产物

| 文件 | 字节 | SHA-256 |
|---|---:|---|
| `review_evidence/human_factor_remediation_20261022/controlled_factorial_diagnostic.py` | 66574 | `sha256:90f142cd1182ded9ff240dcf191cc502cd5267bb37cad13305ea12b4700d0ae6` |
| `review_evidence/human_factor_remediation_20261022/controlled_factorial_diagnostic.json` | 3318160 | `sha256:362fa6abe44c756130ecb0f3777d175a3021d5d6375d18c543f2bce2c2fbcaa4` |
| `review_evidence/human_factor_remediation_20261022/controlled_factorial_diagnostic.md` | 15233 | `sha256:c982fd97faa647bc84d7aa8bba8a1e7d76000a4dc5485c21c7826e166afc37f7` |
| `review_evidence/human_factor_remediation_20261022/candidate_ledger.jsonl` | 267670 | `sha256:f87b49500e642b71f71db22775190b8d4896b6af27933db75d3f870e78b51582` |
| `review_evidence/human_factor_remediation_20261022/validate_controlled_factorial_diagnostic.py` | 12124 | `sha256:b7a708c535ddf38270cfe50043b2e622b7febda54bb45cd4c643286ac81896f0` |
| `review_evidence/human_factor_remediation_20261022/controlled_factorial_validation.json` | 3009 | `sha256:243ea222d909b1bac9e90f14e045ed6214ad73a66218e13f8bb95821c7359cff` |
| `review_evidence/human_factor_remediation_20261022/write_space_and_cleanup_record.py` | 5597 | `sha256:0d4fd2d57cc9f5d9fa6227746fd632281ec1996e36a1742cdc0571aa3ebedd26` |
| `review_evidence/human_factor_remediation_20261022/run_controlled_factorial_after_freeze.ps1` | 3459 | `sha256:98e6273de0177741e0f35c656c6f1407a0461a5a9e21cc965c8620d4983759e4` |
| `review_evidence/human_factor_remediation_20261022/run.log` | 3869 | `sha256:59a3d45513673259ee423004923110e99e588e7443ec956a3a6f43a04160e8d5` |

上述诊断产物合计 **3695695 字节**（不含本空间记录自身）。

## 临时目录清理

- 临时目录：`.tmp/pilot_distribution_exp_20261023`。
- 清理前已解析到工作区内部；目录为 0 字节、0 个条目。
- 因目录为空，使用非递归 `Remove-Item` 删除；删除后 `Test-Path=False`。

## 磁盘边界

- 运行开始可用空间：184070041600 字节。
- 清理后可用空间：184068247552 字节。
- 同一磁盘上有并发代理和其他进程，因此不把两次系统级可用空间之差归因于本试验；可归因的持久占用以上表逐文件大小为准。
- 仿真逐重复轨迹只驻留内存；未复制正式数据集，也未创建新的正式归档。
