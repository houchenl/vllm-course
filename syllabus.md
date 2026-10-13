**分阶段与逐周课表 v1.0**

规则确认与模型/硬件信息更新：2026-10-13。58 周 × 5 小时 = 290 小时；第 59–60 周为可选机动，不增加必修内容。每周默认原理 75 分钟、源码 75 分钟、实验 120 分钟、复盘 30 分钟。每周只有一个主要产物，检查点整理复用前几周的记录。

**课程如何对应你的工作**

基础实验用一个已支持的小型 Dense/GQA 模型建立可控基线，建议已有的 Qwen3-0.6B；其官方 config 给出 28 层、8 个 KV heads、head_dim 128，适合核对容量公式。模型来源、revision 与运行栈必须记录。[模型官方配置](https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json)

DeepSeek-V4.1-Flash 与 GLM-5.3-Flash 是贯穿全程的工作案例：前期识别模型入口和基础公式边界，中期深入模型、缓存、量化、通信和图执行，后期验证稀疏/混合 Attention、投机解码、PD 或卸载。DeepSeek 的具体路径以 checkpoint 的 `architectures` 为准；本地已有 `deepseek_v41` 及 `glm5next` 实现。GLM 路径名称与商品模型名不同，不能仅用目录名选择实现。

H20 与 910C 为主要实测平台，910B 用于补充迁移与拓扑对照。早期基线先在当前最容易使用的空闲单卡完成，另一平台使用同一负载补测；后期使用单机多卡、双机。双机设备数量、显存、互联与网卡以环境盘点为准，不从“8 卡/16 卡”推断。

用户报告 vLLM 0.31.0、vLLM-Ascend 0.30.0rc1。Ascend 的公开与本地兼容表将 0.30.0rc1 配套到 vLLM 0.30.0，因此第一课记录 NVIDIA 镜像和 Ascend 镜像各自安装的 vLLM，不自行替换现有部署。CUDA/CANN 版本读取镜像实际环境；宿主机驱动单独记录。[官方版本策略](https://docs.vllm.ai/projects/ascend/en/main/community/versioning_policy.html)

**阶段安排**

| 阶段 | 学习周 / 时数 | 主线 | 阶段验收 |
| --- | --- | --- | --- |
| A | 1–14 / 70h | 推理基础、V1 执行链、调度与 KV、基础 TP 和测量 | 请求执行图、容量估算、基线与一次瓶颈解释 |
| B | 15–20 / 30h | 模型适配、权重/存储、量化、编译与 Graph | 一个模型适配或修复案例及正确性证据 |
| C | 21–26 / 30h | CUDA、Triton、算子与性能工具 | 一个代表性算子的测量与修改报告 |
| D | 27–32 / 30h | Ascend 栈、Ascend C、NPU 执行与调优 | H20/910C 对照及一个 Ascend 算子或执行路径改进 |
| E | 33–38 / 30h | MoE、TP/PP/DP/EP、通信与双机 | MoE 通信/算子分解及拓扑选择报告 |
| F | 39–44 / 30h | 超长上下文、CP、混合状态、线性与稀疏 Attention | 两个工作模型的缓存/Attention 分析及代表性实测 |
| G | 45–50 / 30h | 投机解码、PD 分离、KV 卸载与多级存储 | 先进特性的收益边界与失败处理报告 |
| H | 51–54 / 20h | 多模态理解与视觉生成 | 分阶段性能拆解与质量检查 |
| I | 55–58 / 20h | 真实问题结课项目 | 正确性、瓶颈、修改、性能、复现与适用范围 |

每四周复盘一次，使用当周 30 分钟复盘时段。第 4 周校准学习速度；第 14、20、26、32、38、44、50、54、58 周进行阶段验收。工作案例可替换相同技能目标的实验。

**逐周任务**

表中代码组对应[源码地图](source-map.md)。目录级入口用于定位；每周实际必读由讲义限定为 3–5 个文件或符号，避免把整个代码组当作当周阅读量。

| 周 | 主要问题与知识 | 源码入口 | 当周主要产物 |
| --- | --- | --- | --- |
| 01 | 请求如何生成；Prefill/Decode、KV、TTFT/TPOT；诊断已有基础 | A、B | 第一课报告：执行链草图、容量计算、4 组离线基线 |
| 02 | Tensor 的 shape/stride/layout/dtype 与设备异步执行如何影响推理 | B、F | 一个布局/同步小实验，解释隐式复制或同步 |
| 03 | 算力、带宽、容量为何是不同约束；硬件层次与 Roofline 入门 | F、J | 真实设备资源表与一个算术强度估算 |
| 04 | 如何公平测量；冷启动/预热/稳态、流式指标、并发与到达率 | M | 小模型在线基线与测量口径，完成首次节奏校准 |
| 05 | 客户端、EngineCore 和 Worker 各自拥有什么状态；IPC 和进程边界 | A、B、E | 请求及输出跨进程路径图 |
| 06 | Scheduler 如何安排 token；请求状态、预算、抢占和恢复 | C | 三个请求的手工调度轨迹与源码变量对应 |
| 07 | KV 如何分页、分配和释放；块表、逻辑占用与预分配池 | D | 一个分页分配示例与显存容量核对 |
| 08 | 前缀命中怎么判断；哈希、引用计数、复用边界和缓存失效 | C、D | 冷/热前缀对照与缓存收益解释 |
| 09 | Chunked Prefill 怎样与 Decode 混排；异步调度减少哪些等待 | C、E | 长短请求混跑，比较一个 token 预算参数 |
| 10 | Runner 如何准备输入、维护 batch、生成 Attention metadata 并采样 | E、F | 从 SchedulerOutput 到模型输入的张量/状态表 |
| 11 | GQA/MLA、FlashAttention 与分页组织分别解决什么问题；backend 分派 | F、N | 实际 backend 选择证据与 Attention 成本图 |
| 12 | TP 如何切分线性层和 heads；通信原语如何进入计算图 | F、J | 单卡与 TP=2 的形状/消息量/正确性对照 |
| 13 | 单机多卡为什么不线性加速；拓扑、batch 和 TP 扩展 | J、M | 2–3 个 TP 配置的吞吐与延迟曲线 |
| 14 | 主干检查点：解释完整请求并用证据定位一个限制因素 | A–F、J、M | 主干报告，复用已有基线与执行图 |
| 15 | 模型注册、config 和实现如何对应；Qwen Dense 与两个工作模型入口 | G、N、O | 三个模型的 architectures → 实现 → 缓存类型映射 |
| 16 | checkpoint 如何加载；融合权重、参数切片、绑权重、mmap 和 I/O | G | 加载时间分解及一个权重映射核对 |
| 17 | 模型适配的最小闭环：接口、单层参照、权重映射与回归 | G、N、O | 一个可审查的适配或修复小改动 |
| 18 | FP8/INT8/INT4 与 KV 量化；scale、layout、kernel 支持和精度 | H、N、O | 一种支持的量化方案的质量/内存/性能对照 |
| 19 | torch.compile、Graph 捕获、形状、融合与 fallback | I、E | Eager/Graph 或编译策略的可控对照 |
| 20 | 模型适配检查点；把代表模型带到另一平台 | G、K、M | 适配测试、质量检查与跨平台初步报告 |
| 21 | CUDA 线程、grid/block/warp、边界和基本同步 | F、官方 CUDA 教程 | 一个简单 CUDA 内核与边界正确性检查 |
| 22 | GPU 存储层次、合并访存、Stream/Event 与 Nsight Systems | F、I | 一个访存/同步假设及测量证据 |
| 23 | Triton 的 program、pointer、mask 与 tiling | F、官方 Triton 教程 | 一个 Triton 基础内核及不同 shape 的结果 |
| 24 | 归约、RMSNorm/Softmax、融合、数值稳定性和误差阈值 | F | 一个融合/tiling 修改的正确性与性能对照 |
| 25 | GEMM、Tensor Core、分块、复用与实际 shape；Nsight Compute | F、L | 一个 GEMM shape/配置分析，解释利用率 |
| 26 | Attention 热点：布局、IO、occupancy、kernel dispatch 和融合 | F、N、O | 一个内核或 backend 配置修改报告，允许收益为负 |
| 27 | 910C 的软件栈：torch_npu、CANN、平台插件、运行时与版本边界 | K | NPU 调用栈与镜像/驱动/依赖记录 |
| 28 | NPU Runner、Attention metadata、Cache 与 Graph 路径 | K、N、O | GPU/NPU 同一执行步骤的接口与数据布局对照 |
| 29 | Ascend C 从零：Cube/Vector、片上存储、搬运、队列和同步 | K、厂商文档 | 简单 CopyIn/Compute/CopyOut 算子的执行图 |
| 30 | Ascend C 算子：Tiling、流水、尾块、dtype 与测试 | K | 一个简单算子的实现或优化与正确性结果 |
| 31 | NPU 性能分析：主机同步、搬运、算子融合、图执行与通信 | K、M | 一份对应 CANN 工具版本的 profiler 分析 |
| 32 | H20/910C 对照；910B、RTX PRO 5000 的迁移差异 | J、K、M | 同负载对照与硬件迁移决策表 |
| 33 | MoE 参数量、激活参数与内存；router、top-k、专家和 token 分布 | L、N、O | MoE 单层成本与路由分布分析 |
| 34 | FusedMoE 的 prepare/expert/finalize；Grouped GEMM 与 dispatch/combine | L、K | MoE 单层耗时拆解，区分路由/通信/计算 |
| 35 | TP/PP/DP/EP 的组合、进程组和负载均衡；EPLB 的代价 | J、L | 一个并行方案的设备/组/消息量图 |
| 36 | 双机 NCCL/HCCL、RDMA、NVLink/HCCS、拓扑与通信重叠 | J、K | 一个 collectives/传输基线及通信瓶颈判断 |
| 37 | DeepSeek/GLM 两个工作模型深入对照；Qwen/Kimi/MiniMax 的差异迁移 | L、N、O | 五系列适配矩阵，深入两个工作模型的关键差异 |
| 38 | MoE 与双机检查点：容量、通信、负载均衡、吞吐和尾延迟 | J–O | 一份代表性 MoE 实测与并行选择报告 |
| 39 | 超长上下文：有效上下文、RoPE、缓存容量、Prefill 与质量 | D、N、O | 上下文长度扩展曲线与质量检查 |
| 40 | Prefill CP 和 Decode CP；KV 分片、Ring/AllGather 与布局 | J、N、O、CP 文档 | 分片/通信成本计算与本版本支持矩阵 |
| 41 | CP 实测：单机/双机、通信范围、容量与 TTFT/Decode 收益 | J、K、M | 一种实际支持的 CP 配置对照，不推定任意组合可用 |
| 42 | 混合 Attention/SSM：固定状态、按 token KV、分组和恢复 | D、P、O | GLM-5.3-Flash 层类型与缓存/状态表 |
| 43 | 线性注意力：递推/分块、KDA/GDN、卷积状态、数值与融合 | O、P、K | GLM KDA 或可用小模型的分块/递推正确性验证 |
| 44 | 稀疏注意力：压缩、索引、top-k、缓存平面和质量；避免只算稀疏主算子 | N、O、Q | DeepSeek/GLM 索引与 Attention 全链路成本分析 |
| 45 | 投机原理：Draft/Verify、拒绝采样、接受长度和额外成本 | R、C | 手工接受/拒绝例子与速度收益估算 |
| 46 | MTP/EAGLE/实际 proposer、状态回退、负载与图执行 | R、N、O、K | 一种支持方案的接受率、延迟、吞吐与正确性对照 |
| 47 | PD 机制：调度/执行端 connector、KV 生命周期、路由与传输 | S、C | PD 时序图与 KV 传输量/时间估算 |
| 48 | PD 实测：同卡数比较、P/D 比例、尾延迟、故障恢复 | S、J、K、M | 合设/分离对照及一种传输失败情形分析 |
| 49 | KV 卸载：权重 vs KV、复用 vs 活动换出、CPU/pinned memory 与预取 | T、D、S | CPU KV 层次的命中/搬运/重算成本对照 |
| 50 | 多级缓存：NVMe/远端、chunk/IO 大小、一致性与恢复 | T、S、G | 一种支持层次的实验，或明确接口边界的可执行设计 |
| 51 | 图像/视频输入：Processor、编码器、token 预算与缓存 | U、N、O | 两个工作模型或小型 VLM 的输入到视觉 token 流程 |
| 52 | 多模态调优：媒体处理、编码、语言模型、并行与质量 | U、K、M | 多阶段耗时拆解与一个可控优化 |
| 53 | 视觉生成：Diffusion/DiT、去噪循环、VAE 与阶段调度 | vLLM-Omni 官方源码/文档 | 一个生成管线的阶段/内存/质量基线 |
| 54 | 视觉生成性能：Attention、批处理、序列并行与缓存策略 | vLLM-Omni、F、J | 一种支持优化的速度/质量对照，明确与自回归引擎的区别 |
| 55 | 结课问题定义：选择 DeepSeek 或 GLM 在 H20/910C 上的真实问题 | 相应代码组 | 问题、SLO、版本、负载、正确性门槛与实验预算 |
| 56 | 建立证据：定位热点、构造最小复现、筛选干预 | 相应代码组 | 假设与 profiler/通信/缓存证据 |
| 57 | 实施与验证：一个主要改动、回归、性能与稳定性 | 相应代码组 | 可审查修改与原始测量记录 |
| 58 | 独立答辩：解释机制、失败边界、复现及迁移条件 | 相应代码组 | 完整结课报告与待验证事项 |

**实验范围与前置条件**

每个硬件实验前确认“模型 → 架构 → backend → 精度 → 并行/高级特性”的支持链。所需前置检查通常在前一课形成，不把完整环境部署计为额外作业。CPU/单层替代实验能验证局部机制，不能证明全尺寸模型的容量、通信或稳定性。

对于 CP、投机、PD、外部缓存及视觉生成：具备配套软件、模型与硬件时完成代表性实测；当前版本不支持时记录具体边界，以实际支持的平台或小模型验证原理，并留下迁移方案。主干、模型适配、算子、MoE 和结课项目必须有实际测量。

第一阶段会提前介绍工作模型的高级结构，但使用标准小模型推导基本公式。DeepSeek-V4.1 的非对称结构、压缩/稀疏缓存与 GLM 的混合状态会在第 15、28、37、39–46、51–52 周逐步深入，不能把 Dense/GQA 的公式直接套到它们的总缓存或全模型执行次数。

**学习材料的使用方式**

每课首先定位对应版本源码，随后用文档和论文解释原理。以下资料是索引，不要求一次读完：

- [官方 V1 架构](https://docs.vllm.ai/en/stable/design/arch_overview/)与[调优说明](https://docs.vllm.ai/en/latest/configuration/optimization/)。
- [PagedAttention 原论文](https://arxiv.org/abs/2309.06180)、[FlashAttention 原论文](https://arxiv.org/abs/2205.14135)。
- [CUDA 编程模型](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)、[Triton 教程](https://triton-lang.org/main/getting-started/tutorials/)、[Ascend 文档](https://www.hiascend.com/document)。厂商 API 按实际工具链版本选择。
- [Context Parallel](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/)、[Hybrid KV cache](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)、[投机解码](https://docs.vllm.ai/en/latest/features/speculative_decoding/)、[PD 分离](https://docs.vllm.ai/en/latest/features/disagg_prefill/)、[KV 卸载](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/)。
- [vLLM-Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/)。

各阶段详细讲义以 HTML 教材格式逐课准备，结合最近的自测、报告和环境事实收缩必读范围。阶段 A 的第 1–14 课已分别准备，按课表入口阅读；当前建议从[第一课](lessons/01-inference-lifecycle.html)开始。后续阶段尚未编写。正式[网页课表](index.html)同时提供阶段导航与 58 周安排。
