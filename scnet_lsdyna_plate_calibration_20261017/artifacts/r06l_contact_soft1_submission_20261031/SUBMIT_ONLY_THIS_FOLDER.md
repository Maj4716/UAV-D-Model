# R06l 接触 SOFT=1 单因素诊断提交说明

不要压缩。只上传整个 `uav_hf_r06l_0800_contact_soft1` 文件夹，并在平台选择 `r06l_0800_contact_soft1.k` 作为 head/input deck。

资源参数：1 个节点、32 MPP ranks、`ls-dyna_R15_mmp_d`、head memory 512 MB、child memory 256 MB、
命令行参数留空。使用全新工作目录 `/work/home/acd7dt5yv4/LSDYNA/uav_hf_r06l_0800_soft1`。

本轮继续从爆炸瞬间后的同一碎片状态开始。保持 `PSFAIL=910001`、`TSSFAC=0.05`、`DT2MS=0`、
碎片 MAT_015 `EROD=1`、板材 `EROD=0` 不变。唯一变量是 ERODING_SINGLE_SURFACE Optional Card A
的 `SOFT: 2 -> 1`。

这是数值机制诊断，不是物理标定。即使正常结束，也必须检查碎片删除/负体积、总能量比、滑移能量、
最小时间步和循环数；不得把本轮响应直接作为仿真引擎或毁伤树的标定目标。

完成或中止后都下载完整回传：`.k`、`d3hsp/o=`、`mes*`、`glstat`、`binout*`、所有 `d3plot*`、
所有 `d3part*`、stdout/stderr、Slurm 脚本和资源记录。之后只需把回传路径发给我。
