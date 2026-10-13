**vLLM 与 AI Infra 系统学习课程规则 v1.0（已确认）**

制定及确认日期：2026-10-13。用户已确认规则。配套[HTML 课程索引与课表](index.html)与[第一课 HTML 教材](lessons/01-inference-lifecycle.html)已建立。课程规则以仓库内 course-rules.md 为编写源，本文状态以本段为准。

学员基础：学过约 12 小时 PyTorch 入门并做过简单项目；熟悉 Transformer 原理；了解通信原语和并行方式；CUDA、Triton、Ascend C 与硬件知识需要从基础补齐。每周投入 5 小时。工作覆盖 DeepSeek、Qwen、GLM、Kimi、MiniMax 系列，包含 Dense、MoE、多模态理解及视觉生成。2026-10-13 补充：常用模型为 DeepSeek-V4.1-Flash、GLM-5.3-Flash；Ascend 以 910C 为主、910B 为补充。主要硬件为 H20、RTX PRO 5000、910B、910C；已描述资源包括双机 H20 和两类双机 910B，910C 的具体资源、各环境每节点卡数、SKU 和网络拓扑将在实验准备时核实。用户报告 vLLM 0.31.0、vLLM-Ascend 0.30.0rc1；CUDA/CANN 及两个镜像内完整版本分别实测记录。

已检查的源码依据：`vllm-main-261013` 中的 V1 EngineCore、Scheduler、KVCacheManager、GPU Worker、模型实现、设计文档，以及当前工作区 `vllm-ascend-main-1013` 的平台插件和版本策略。尚未确认主线快照的精确 commit；目录名不作为版本证明。目录中同时存在传统 GPUModelRunner 与实验性的 Model Runner V2，必须按运行配置区分实际执行路径。Ascend 主线版本策略要求配套 `.github/vllm-main-verified.commit` 指定的 vLLM commit，目前文件内容为 `8cc9aa5ad350f6f80cf3bebb298d410803ff75a9`；尚未确认本地 vLLM 快照与该 commit 相同。

**1. 用岗位能力定义目标。**

课程结束的验收围绕四项能力：解释一次请求的完整执行过程；依据测量定位算力、访存、通信、CPU 调度或存储瓶颈；完成代表性模型的适配或修复，并验证正确性；针对目标硬件和业务负载给出有证据的性能优化方案。每项能力都要有源码定位和实验产物，不能只用术语记忆或启动成功作为证明。

**2. 主干优先，按依赖关系推进。**

| 层次 | 学习范围 | 达到的深度 |
| --- | --- | --- |
| 第一轮主干 | Prefill/Decode、容量与性能估算、请求生命周期、V1 调度、分页 KV cache、连续批处理、Chunked Prefill、前缀缓存、基础测量、单卡与单机 TP | 能解释流程、定位代码、建立基线并分析一次性能问题 |
| 核心深入 | 模型注册与权重加载、算子与 Attention backend、编译与 Graph、量化、TP/PP/DP/EP、MoE 通信与负载均衡、Ascend 平台适配、双机拓扑 | 能修改代表性路径，并通过正确性和性能验证 |
| 专题必修 | 投机解码、PD 分离、超长上下文与 CP、混合 Attention/SSM、稀疏与线性注意力、KV 卸载、多模态理解、视觉生成 | 能解释机制、定位实现边界、分析成本收益并完成代表性实验 |
| 按需扩展 | 完整高性能 Attention/GEMM 内核重写、其他引擎全面比较、训练系统、集群平台的深入运维 | 根据岗位需求增加，避免挤占 vLLM 主干 |

高级专题在主干阶段先给概览和依赖图，之后回到同一问题深入实现。工作中的紧急问题可以提前作为案例，但后续补齐其前置知识。

**3. 基础知识与当前源码问题绑定。**

先做一次短诊断，再决定补课量。Transformer 只补推理相关细节与架构变体；PyTorch 补张量 shape、stride、layout、dtype、数值误差、异步执行、同步和内存行为；硬件补计算单元、存储层次、带宽与 Roofline；通信补消息量、延迟、有效带宽、拓扑和计算通信重叠；存储补权重读取、mmap/page cache、主机内存、NUMA、pinned memory、NVMe 和远端缓存。每项基础知识必须说明它如何影响当前代码与性能。

**4. 同一知识点分三轮掌握。**

第一轮建立直观模型并走通最小例子；第二轮推导关键公式、状态变化和源码实现；第三轮面对真实模型与硬件验证边界、故障和优化收益。复杂内容拆分到多周，避免一周同时要求熟练掌握新架构、新内核语言和新分布式方案。

**5. 每课采用统一教学结构。**

2026-10-13 补充确认：每节课以 HTML 作为正式交付格式，使用类似优秀美式理工科教材的编排方法。正文采用原创中文的连贯推演，包含清晰定义、适用假设、变量与单位、编号公式和图表、完整解题过程、概念检查、源码导读、受控实验、分级习题与参考答案。支持本地离线阅读、移动端和打印；具体规范见 [HTML 教材编写规范](editorial-guide.html)。实验脚本和数据记录作为配套附件保留各自适合的格式。

同日补充：需要插图的教学内容调用画图工具绘制，并嵌入 HTML。每张插图明确教学目标、适用假设和图例，核对数量、箭头及状态关系，配编号、中文图注和替代文本；原图与提示词保留在课程目录。插图数量依据讲解需要确定。

工作问题与目标 → 直观图解 → 必要的公式和张量/状态示例 → 核心源码 → 最小实验 → 结果解释 → 自测。每课明确必读与选读、前置知识和预算，通常限定 3–5 个核心文件或符号。读源码关注输入输出、shape/layout、状态拥有者、进程/设备边界、分配释放与性能关键路径。涉及并发、内存生命周期、数值正确性时再逐行分析。

**6. 固定阅读基线，单独锁定运行环境。**

以用户提供的主线源码为阅读锚点，以 V1 引擎为主线；旧架构仅在解释演进或版本差异时对照。Model Runner V2 单列为演进内容，依据实际配置判断使用哪个 runner。GPU 与 Ascend 实验分别记录 vLLM、插件、PyTorch/torch_npu、CUDA/CANN、驱动、通信库、容器和模型 revision。Ascend 发布版按兼容表配对，Ascend 主线按 `.github/vllm-main-verified.commit` 配对，不默认两个 main 源码快照互相兼容；必要时对照阅读版本与运行版本。

源码讲解引用文件、类、方法和短片段；确认后为阅读快照补充 tag/commit 或归档指纹。资料优先使用对应版本源码、官方文档与设计文档、原论文、厂商文档；社区文章作为辅助线索。对关键结论标明源码事实、文档说明、理论估算或实验观察，并说明版本、硬件与负载前提。逐课核查功能组合的支持情况，区分可用、实验性、待验证与不支持。

**7. 每周 5 小时是全部学习任务的预算。**

默认分配：原理 1.25 小时，源码 1.25 小时，实验 2 小时，复盘与自测 0.5 小时。实验周与原理周可调整，总量保持 5 小时。环境配置与排障计入预算；遇到复杂环境问题，采用已验证容器或将问题拆为独立任务。机器后台运行时间与人的投入时间分别记录。每周一个主要目标、一个主要产物；长课题分多周。可替换部分实验为工作案例，不能要求同时额外完成两套任务。

**8. 所有指定专题都进入必修体系。**

| 专题 | 必须掌握的核心 | 所需前置基础与实验关注点 |
| --- | --- | --- |
| 投机解码 | Draft/Verify、采样与拒绝机制、接受率、有效生成长度、KV/状态回退、MTP/EAGLE 等代表性方案 | 调度、采样、KV；比较额外计算与延迟收益，覆盖不同并发和输入输出长度 |
| PD 分离 | Prefill/Decode 资源特性、KV connector 的调度端与执行端、传输、路由、缓存生命周期和失败恢复 | 调度、KV、通信；与同卡数同负载的合设方案比较 TTFT、ITL、尾延迟与有效吞吐 |
| 超长上下文与 CP | RoPE 与有效上下文、Prefill CP 和 Decode CP 的区别、KV 分片、分布式 Attention、通信与容量代价 | Attention、TP、显存估算、拓扑；分别验证长上下文质量、容量、TTFT 和 Decode 性能，区分论文方案与当前实现 |
| 混合 Attention/SSM | 不同层的 KV/状态需求、递推与分块执行、Hybrid cache 分组、前缀复用及状态恢复 | Cache、Attention；验证连续生成、前缀命中与抢占恢复，认识固定大小状态与按 token 增长 KV 的差异 |
| 稀疏与线性注意力 | 稀疏模式、索引/路由成本、线性/递推表达、数据布局、融合与质量约束 | Attention 与内核基础；同时评估选择/索引成本、真实加速和质量，逐模型区分架构 |
| KV cache 卸载 | KV 与权重卸载的区别、缓存复用与活动请求状态换出的区别、GPU/CPU/NVMe/远端层次、命中率、预取和一致性 | Cache、异步搬运、存储；先 CPU 再外部层次，比较回迁与重算成本，分别核实 CUDA 和 Ascend 路径 |
| 多模态理解 | Processor、视觉编码器、图像/视频 token 预算、编码缓存、语言模型执行与调度 | 模型适配、张量布局、调度；拆分媒体处理、编码与语言模型各阶段耗时 |
| 视觉生成 | Diffusion/DiT、多步去噪、VAE、阶段编排、批处理与序列并行、生成质量 | 算子、通信与多模态；以每图/视频延迟、吞吐和质量评估，借助 vLLM-Omni 等相关项目讲解实际执行边界 |

每个专题至少完成“原理 + 相关源码/接口 + 成本收益模型 + 一个可执行的代表性实验设计”。具备环境并且该版本支持时实测；否则明确待实测项，不能写成已验证。核心专题和结课项目需要实际测量。

**9. 算子与硬件从零开始，逐步达到工作可用深度。**

CUDA 学线程组织、存储层次、合并访存、Stream/Event、同步和测量，并完成基础内核练习；Triton 从向量操作、归约和融合入手，再读或修改 RMSNorm、Softmax、GEMM/Attention 的代表性代码；Ascend C 学 Cube/Vector、搬运、片上存储、Tiling、流水与同步，并完成一个简单算子的实现或优化。使用 NVIDIA Nsight Systems/Compute 和 Ascend 对应性能工具分析实际热点。Triton、CUDA、Ascend C 的概念逐项对照，但依据各自硬件和软件版本验证实现。

Attention 深入区分分页缓存组织与 FlashAttention 的 IO 优化，结合实际 backend 和分派路径阅读。必修验收要求能够定位热点、解释实现并修改代表性内核或配置；完整重写高性能 Attention/GEMM 为扩展项目。

**10. 模型按架构差异组织，硬件按真实资源安排。**

选择一个较小 Dense 模型建立全链路基线，再进入 MoE 和 MLA/混合状态架构，之后覆盖多模态。DeepSeek、Qwen、GLM、Kimi、MiniMax 五个系列均有代表性源码案例，比较 config、权重映射、Attention、MoE、量化、并行及缓存差异；具体型号依据版本支持、模型可得性和显存确定。同一系列的不同模型逐个确认架构。

H20 与 910C 是主要实测平台，910B 为补充。实验按单卡 → 单机多卡 → 双机升级。优先让小模型在两种平台复现，再将结论带入大模型。至少实测一个 Dense、一个 MoE、一个混合或多模态代表模型；缩小结构或单层实验只验证局部机制，全尺寸容量、通信负载和稳定性需要另行验证。RTX PRO 5000 与 910B 必须纳入硬件差异与迁移案例，具备相应机器时完成补充实测。

实验准备核实每节点设备数量、显存、互联、网卡、CPU/NUMA、内存、磁盘、软件栈和模型资源。不能从“8 卡/16 卡”或芯片名称推断具体拓扑与实际带宽。其他引擎可做小范围对照，阅读与项目仍以 vLLM 为中心。

**11. 性能实验遵循同一证据标准。**

先明确业务目标：交互延迟、吞吐、长上下文容量、资源成本或可用性。固定模型与 tokenizer revision、输入输出长度及分布、精度、并发或到达率、前缀命中条件、并行配置和设备数量。区分冷启动、预热/编译与稳态，先做单因素实验再考虑组合优化。保留命令、配置、数据、日志和 profiler 文件，进行足够的重复测量并解释噪声。

常用指标为 TTFT、ITL/TPOT、请求端到端延迟、P50/P95/P99、请求和 token 吞吐、满足延迟 SLO 的有效吞吐、内存占用、通信时间、失败率。KV 命中率与投机接受率用于解释机制。跨平台使用共同负载，并说明精度、软件版本及资源差异；同时评价各平台合适的优化配置，区分公平对照与平台最佳结果。支持度、容量、稳态性能、启动/加载耗时分别记录。

**12. 正确性、精度和稳定性与性能一起验收。**

对算子做边界 shape、dtype、layout 与数值误差检查；对模型验证权重映射、代表性层或 logits、生成任务与适当的质量指标；对系统验证长短请求混跑、抢占/恢复、异常退出和反复运行。跨硬件及不同 kernel 不要求默认逐位一致，也不能只用一个生成样例判断正确。量化与稀疏方案同时报告质量变化；投机解码验证其采样语义，避免只比较 greedy 文本。

**13. 用阶段产物检验掌握程度。**

| 检查点 | 必须提交的证据 |
| --- | --- |
| 主干理解 | 请求执行图、关键对象/状态表、KV/显存估算、一个可复现的性能基线 |
| 性能诊断 | 一份 profiler 分析：假设、热点证据、干预实验、收益或失败原因 |
| 模型与平台适配 | 代表性修改、相关测试、质量检查、H20/910C 的迁移或对比报告，910B 补充 |
| 专题能力 | 一个双机或先进特性案例，包含收益边界、版本支持与异常处理分析 |
| 结课项目 | 一个真实岗位问题的完整解决报告：问题、版本、正确性、瓶颈、修改、性能、适用范围与复现步骤 |

每课设置少量解释、估算、源码定位或设计实验题。未通过关键检查点时，缩小任务并补齐对应知识。优化未产生收益也可成为合格产物，前提是证据充分且能解释原因。无需承诺固定加速比例或必须向上游提交 PR。

**14. 周期、反馈和课程调整。**

建议先按 48–60 个学习周、240–300 小时预算，前 12–16 周完成主干第一轮，再深入适配、算子和高级专题；这是根据当前基础与覆盖范围提出的可调整预算，不是能力达成保证。经过最初 4 周再校准速度。大型模型与新硬件的深入专项可延长。

每 4 周根据自测、实验报告与工作变化调整后续深度和顺序。讲解使用中文，保留英文术语与代码符号；初次出现解释含义，公式说明变量、单位和适用条件。逐步减少实验脚手架，最终让学员独立提出假设、查源码和设计验证。

确认后维护课程索引、学习进度、源码地图、实验记录与问题清单，记录已经掌握和仍待验证的内容。课堂推进前先核对本地记录，保持跨会话连续性。

**资料依据（规则的设计仍属于课程建议）**

- V1 进程架构：[本地设计文档](../vllm-main-261013/docs/design/arch_overview.md)；[官方架构说明](https://docs.vllm.ai/en/stable/design/arch_overview/)。
- Model Runner V2 的实验状态：[本地 README](../vllm-main-261013/vllm/v1/worker/gpu/README.md)。
- Ascend 配套版本：[本地版本策略](../vllm-ascend-main-1013/docs/source/community/versioning_policy.md)、[本地主线验证 commit](../vllm-ascend-main-1013/.github/vllm-main-verified.commit)；[官方版本策略](https://docs.vllm.ai/projects/ascend/en/main/community/versioning_policy.html)。本地主线配套要求以本地指定 commit 和对应依赖为准。
- 分页缓存：[PagedAttention 原论文](https://arxiv.org/abs/2309.06180)。
- 专题机制：[投机解码](https://docs.vllm.ai/en/latest/features/speculative_decoding/)、[PD 分离](https://docs.vllm.ai/en/latest/features/disagg_prefill/)、[Context Parallel](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/)、[Hybrid KV cache](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)、[KV 卸载](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/)。运行能力以配套版本为准，不能从 latest 文档推定本地支持。
- 内核基础：[CUDA 编程模型](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)、[Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/)、[Ascend C 编程范式示例文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha001/opdevg/Ascendcopdevg/atlas_ascendc_10_0016.html)。Ascend C 文档示例说明编程概念，API 和工具使用须对照实验环境的 CANN 版本。
- 视觉生成扩展：[vLLM-Omni 官方文档](https://docs.vllm.ai/projects/vllm-omni/en/latest/)。

**本地源码入口**

- [架构概览](../vllm-main-261013/docs/design/arch_overview.md)
- [EngineCore](../vllm-main-261013/vllm/v1/engine/core.py)
- [Scheduler](../vllm-main-261013/vllm/v1/core/sched/scheduler.py)
- [KVCacheManager](../vllm-main-261013/vllm/v1/core/kv_cache_manager.py)
- [GPU Worker](../vllm-main-261013/vllm/v1/worker/gpu_worker.py)
- [传统 GPUModelRunner](../vllm-main-261013/vllm/v1/worker/gpu_model_runner.py)
- [Model Runner V2](../vllm-main-261013/vllm/v1/worker/gpu/model_runner.py)
- [模型注册](../vllm-main-261013/vllm/model_executor/models/registry.py)
- [默认权重加载](../vllm-main-261013/vllm/model_executor/model_loader/default_loader.py)
- [Ascend 平台插件](../vllm-ascend-main-1013/vllm_ascend/platform.py)
- [Ascend 主线版本策略](../vllm-ascend-main-1013/docs/source/community/versioning_policy.md)
- [Ascend 主线验证 commit](../vllm-ascend-main-1013/.github/vllm-main-verified.commit)

规则已于 2026-10-13 确认，并按同日补充的工作模型及 Ascend 硬件优先级更新。后续按已确认规则和每四周复盘调整课表。
