**第 10 课报告：ModelRunner：把执行计划变成张量与 Attention metadata**

状态：待学习 / 机制练习完成且硬件待实测 / 待验收 / 已通过。
日期：
实际投入：原理 __；源码 __；实验 __；复盘 __ 分钟；合计 __ 分钟。
后台机器时间（与人的投入分开）：

**主要产物与推导**

SchedulerOutput到模型输入的张量/状态表。
适用假设、变量和单位：
手算与边界检查：
教学模拟与真实运行的区别：

**环境与负载**

| 项目 | 实际值 | 证据/待查 |
| --- | --- | --- |
| 平台/SKU、卡数、拓扑/NUMA | | |
| 镜像tag/digest、驱动、CUDA/CANN | | |
| vLLM、插件、PyTorch/torch_npu | | |
| 模型/tokenizer revision、architectures | | |
| dtype、backend、runner、TP | | |
| 长度分布、seed、并发/到达率 | | |
| APC、chunked、async、token预算 | | |
| 预热、测量边界、重复、失败 | | |

**源码映射**

| 文件/符号 | 输入输出/shape | 状态拥有者、进程/设备边界 | 本课结论 |
| --- | --- | --- | --- |
| Worker._make_model_runner | | | |
| SchedulerOutput | | | |
| _update_states / _prepare_inputs / execute_model | | | |
| CommonAttentionMetadata | | | |

**实验与分析**

完整命令、有效配置、原始JSON与日志路径：
固定变量与唯一控制变量：

| 条件/重复 | 主要指标与单位 | 正确性/误差/失败 | 原始文件 |
| --- | --- | --- | --- |
| 对照 | | | |
| 干预 | | | |

观察（数据直接支持）：
假设（机制解释）：
待验证（能区分候选机制的下一项测量）：
噪声、负收益、异常和未运行项：

**自测**

1. 复算q=[1,3,2]的段边界与positions。

我的答案：

2. position与slot有什么区别？

我的答案：

3. slot_mapping与block_table分别做什么？

我的答案：

4. 新请求与已缓存请求为什么不同发送？

我的答案：

5. seq_lens上界能代替精确长度吗？

我的答案：

**验收与下一步**

已掌握：
未通过项和最小补课：
验收证据：
下一课调整：

未实测项必须保留为待完成；容量或CPU模拟不能替代硬件基线与阶段验收。
