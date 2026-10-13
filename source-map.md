**教学源码地图**

基于 2026-10-13 工作区阅读快照。路径存在不表示运行版本已启用该功能；配置、设备分派和实际调用栈必须核对。每周使用下面的代码组定位，再由讲义选出 3–5 个必读文件/符号。当前提交和 73 个教学文件 SHA256 见[阶段 A 源码记录](records/stage-a-source-snapshot.json)。原始[教学指纹](records/source-snapshot.json)保留旧目录标识；原56锚点中54个与当前文件一致，Ascend platform.py 与 sparse_mla_attention.py 两项不同。提交标识不证明运行镜像支持或版本兼容。

| 代码组 | 主题 | 已核对的主要入口 |
| --- | --- | --- |
| A | 架构 | [arch_overview.md](../vllm/docs/design/arch_overview.md) |
| B | 离线客户端、请求和引擎 | [LLM.generate](../vllm/vllm/entrypoints/llm.py)、[OfflineInferenceMixin](../vllm/vllm/entrypoints/common/offline.py)、[LLMEngine](../vllm/vllm/v1/engine/llm_engine.py)、[EngineCore](../vllm/vllm/v1/engine/core.py)、[EngineCoreClient](../vllm/vllm/v1/engine/core_client.py) |
| C | 调度 | [Scheduler](../vllm/vllm/v1/core/sched/scheduler.py)；重点 schedule、update_from_output、请求状态与 token 预算 |
| D | 缓存 | [KVCacheManager](../vllm/vllm/v1/core/kv_cache_manager.py)、[BlockPool](../vllm/vllm/v1/core/block_pool.py)、[KVCacheSpec](../vllm/vllm/v1/kv_cache_interface.py)、[CacheCoordinator](../vllm/vllm/v1/core/kv_cache_coordinator.py) |
| E | 执行 | [MultiprocExecutor](../vllm/vllm/v1/executor/multiproc_executor.py)、[GPU Worker](../vllm/vllm/v1/worker/gpu_worker.py)、[传统 GPUModelRunner](../vllm/vllm/v1/worker/gpu_model_runner.py)、[Model Runner V2](../vllm/vllm/v1/worker/gpu/model_runner.py)；用 Worker._make_model_runner 判断分支 |
| F | 基础层与 Attention | [linear.py](../vllm/vllm/model_executor/layers/linear.py)、[layernorm.py](../vllm/vllm/model_executor/layers/layernorm.py)、[Attention selector](../vllm/vllm/v1/attention/selector.py)、[Attention backends](../vllm/vllm/v1/attention/backends)、[csrc](../vllm/csrc) |
| G | 模型与权重 | [Registry](../vllm/vllm/model_executor/models/registry.py)、[Qwen3](../vllm/vllm/model_executor/models/qwen3.py)、[DefaultModelLoader](../vllm/vllm/model_executor/model_loader/default_loader.py) |
| H | 量化 | [量化注册入口](../vllm/vllm/model_executor/layers/quantization/__init__.py)、[Ascend 量化](../vllm-ascend/vllm_ascend/quantization) |
| I | 编译与 Graph | [compilation](../vllm/vllm/compilation)、[torch_compile 设计](../vllm/docs/design/torch_compile.md)、[CUDA Graph 设计](../vllm/docs/design/cuda_graphs.md) |
| J | 分布式 | [parallel_state.py](../vllm/vllm/distributed/parallel_state.py)、[device_communicators](../vllm/vllm/distributed/device_communicators)、[并行部署说明](../vllm/docs/serving/parallelism_scaling.md) |
| K | Ascend 平台与算子 | [NPUPlatform](../vllm-ascend/vllm_ascend/platform.py)、[Worker](../vllm-ascend/vllm_ascend/worker/worker.py)、[Runner V1](../vllm-ascend/vllm_ascend/worker/model_runner_v1.py)、[Runner V2](../vllm-ascend/vllm_ascend/worker/v2/model_runner.py)、[LayerNorm](../vllm-ascend/vllm_ascend/ops/layernorm.py)、[Attention](../vllm-ascend/vllm_ascend/attention)、[distributed](../vllm-ascend/vllm_ascend/distributed)、[csrc](../vllm-ascend/csrc) |
| L | MoE 与负载均衡 | [FusedMoE](../vllm/vllm/model_executor/layers/fused_moe/fused_moe.py)、[EPLB](../vllm/vllm/distributed/eplb)、[Qwen MoE](../vllm/vllm/model_executor/models/qwen3_moe.py)、[Kimi](../vllm/vllm/model_executor/models/kimi_k25.py)、[MiniMax](../vllm/vllm/model_executor/models/minimax_m2.py) |
| M | 测量 | [serve benchmark](../vllm/vllm/benchmarks/serve.py)、[metrics/stats.py](../vllm/vllm/v1/metrics/stats.py)、[课程离线脚本](labs/lesson01_baseline.py) |
| N | DeepSeek-V4.1 工作案例 | [注册后入口](../vllm/vllm/models/deepseek_v41/__init__.py)、[NVIDIA model](../vllm/vllm/models/deepseek_v41/nvidia/model.py)、[NVIDIA vl_model](../vllm/vllm/models/deepseek_v41/nvidia/vl_model.py)、[Attention](../vllm/vllm/models/deepseek_v41/attention.py)、[sparse_mla](../vllm/vllm/models/deepseek_v41/sparse_mla.py)、[decoder replay](../vllm/vllm/models/deepseek_v41/decoder_replay_layers.py)、[Ascend model 目录](../vllm-ascend/vllm_ascend/models/deepseek_v41)、[Ascend DSA](../vllm-ascend/vllm_ascend/attention/dsa_v41.py) |
| O | GLM-5.3-Flash 工作案例 | [glm5next 入口](../vllm/vllm/models/glm5next/__init__.py)、[common/model](../vllm/vllm/models/glm5next/common/model.py)、[common/kda](../vllm/vllm/models/glm5next/common/kda.py)、[common/attention](../vllm/vllm/models/glm5next/common/attention.py)、[Ascend model](../vllm-ascend/vllm_ascend/models/glm5next/model.py)、[Ascend cache](../vllm-ascend/vllm_ascend/models/glm5next/cache_config.py)、[Ascend KDA](../vllm-ascend/vllm_ascend/models/glm5next/kda.py) |
| P | 混合与状态模型 | [Qwen3 Next](../vllm/vllm/model_executor/models/qwen3_next.py)、[Mamba/GDN 层](../vllm/vllm/model_executor/layers/mamba)、[Hybrid KV 设计](../vllm/docs/design/hybrid_kv_cache_manager.md) |
| Q | 稀疏索引与 Attention | [sparse_attn_indexer](../vllm/vllm/model_executor/layers/sparse_attn_indexer.py)、[sparse_mla_attention](../vllm/vllm/model_executor/layers/attention/sparse_mla_attention.py)，以及 N/O 的模型专用实现 |
| R | 投机解码 | [传统 Runner 的 spec_decode](../vllm/vllm/v1/spec_decode)、[Runner V2 的 spec_decode](../vllm/vllm/v1/worker/gpu/spec_decode)、[Ascend V2 spec_decode](../vllm-ascend/vllm_ascend/worker/v2/spec_decode)、[投机文档](../vllm/docs/features/speculative_decoding/README.md)；按照实际 runner 选择路径 |
| S | PD / KV 传输 | [kv_transfer](../vllm/vllm/distributed/kv_transfer)、[PD 设计/使用文档](../vllm/docs/features/disagg_prefill.md) |
| T | 卸载与层次缓存 | [kv_offload](../vllm/vllm/v1/kv_offload)、[使用文档](../vllm/docs/features/kv_offloading_usage.md)；Ascend 独立核查 connector/扩展支持 |
| U | 多模态 | [multimodal](../vllm/vllm/multimodal)、[Qwen VL](../vllm/vllm/model_executor/models/qwen3_vl.py)，以及 N/O 中的 processor、视觉和条件生成路径 |

两个工作模型的正式映射，以用户 checkpoint 的 `config.json`、Registry 和实际加载日志为证据。上述模型路径已在本地定位，但尚未运行用户镜像或 checkpoint；不能把源码存在写成该镜像已支持。

视觉生成使用 [vLLM-Omni 官方文档](https://docs.vllm.ai/projects/vllm-omni/en/latest/)定位对应版本源码；当前工作区已有同级 vllm-omni 仓库；阶段 A 未对其锁定教学与运行版本，进入第 53 周前另行核对。
