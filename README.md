**vLLM 与 AI Infra 学习课程**

正式阅读入口：[HTML 课程首页与课表](index.html)。每节课使用 HTML，采用理工科教材的定义、推导、例题、源码导读和实验结构。讲义支持离线阅读、手机布局和打印为 PDF。

需要插图的讲解使用画图工具生成教学图片，核对后嵌入 HTML。原图保存在 assets/figures，提示词与核对记录保存在 records；正式讲义内嵌图片，单个 HTML 离线打开仍可阅读。

第一课插图：[Prefill/Decode 与 KV 复用](assets/figures/lesson01-kv-reuse-v1.png)、[逻辑 KV / 分配块 / 预分配池](assets/figures/lesson01-kv-pool-v1.png)；[完整提示词与技术核对记录](records/lesson01-illustrations.json)。

课程规则已于 2026-10-13 确认。基线为 58 个学习周、290 小时；可再使用 2 个机动周，总预算不超过 300 小时。学习周按实际投入推进。

- [已确认规则](rules.html)
- [分阶段与逐周课表](index.html)
- [第一课：一次生成请求如何运行](lessons/01-inference-lifecycle.html)
- [HTML 教材讲义编写规范](editorial-guide.html)
- [源码地图](source-map.html)
- [学习进度与环境记录](records/progress.html)
- [第一课报告模板](records/lesson01-report-template.html)
- [阶段 A 教学源码指纹](records/stage-a-source-snapshot.json)
- [原始阅读快照记录](records/source-snapshot.json)
- [第一课容量估算脚本](labs/lesson01_estimate.py)
- [第一课离线基线脚本](labs/lesson01_baseline.py)

工作案例：DeepSeek-V4.1-Flash、GLM-5.3-Flash。实验平台以 H20 与 910C 为主，910B 为补充；RTX PRO 5000 纳入迁移案例。阅读基线为工作区的 vLLM 与 Ascend 源码快照；用户报告运行版本为 vLLM 0.31.0、vLLM-Ascend 0.30.0rc1，镜像内部的完整版本组合尚待记录。

阶段 A（第 1–14 课）现已逐课准备，含独立 HTML 讲义、报告模板与受控实验附件。入口见 index.html 的阶段 A；准备完成不会改变学员学习和验收状态。新增 7 张教学插图，原图保存在 assets/figures/stage-a-*-v1.png，完整提示词与核对记录见 records/stage-a-illustrations.json，使用内置 image_gen 生成。

推荐第一周顺序：读第一课原理 → 完成容量例题 → 按指定范围读源码 → 在已有验证环境运行实验 → 填写报告与自测。无需先安装本地 Windows 版 vLLM，也无需先编译完整源码。

本轮已准备课程和实验文件，尚未连接远端加速器环境。CPU 教学练习的验证与真实 GPU/NPU 性能实测分别记录。硬件实验结果由实际运行产生，学习进度不会因讲义已准备而自动标为完成。

HTML 为正式交付，原有 Markdown 保留为记录或早期稿件。课程规则编写源为 [course-rules.md](course-rules.md)，第 1–14 课正文编写源位于 materials/*.body.html，课程清单位于 records/stage-a-lessons.json，逐课报告编写源位于 records/lessonXX-report-template.md。后续课件按 [编写规范](editorial-guide.html) 与 templates/lesson-template.html 编写，保持同一套教学标准。

**本地阅读与构建**

当前课程仓库为 `ie/vllm-course`，远程为 [houchenl/vllm-course](https://github.com/houchenl/vllm-course)。打开 index.html 即可阅读；正式讲义包含内嵌样式、脚本和插图。

在仓库根目录使用 Python 3.10 或更新版本重新生成网页，只依赖标准库，不需要安装 vLLM：

```bash
python tools/build_textbook.py
```

源码链接面向本地阅读环境，要求 `vllm` 和 `vllm-ascend` 与课程仓库同级。源码仓库不随本课程仓库提交；单独克隆课程仍可阅读讲义，但需放置相应源码仓库才能打开本地源码链接。当前提交、阅读锚点与 SHA256 见 records/stage-a-source-snapshot.json；原始指纹保留在 records/source-snapshot.json，两个阅读版本的差异另有记录。

Git 保存讲义、课表、规则、编写源、插图、实验脚本和配套记录。`.venv/`、`.uv-cache/`、`__pycache__/`、`.browser-check/` 为本地运行或检查产物，已忽略。

**阶段 A 实验附件**

- labs/stage_a_mechanisms.py：标准库 Roofline、调度、分页、前缀引用与 Runner 输入的教学推演。
- labs/lesson02_tensor.py、lesson11_attention.py、lesson12_tp.py：使用镜像已有 PyTorch，分别验证布局/计时、GQA/在线 softmax、TP 切分等式；不是高性能实现。
- labs/stage_a_bench.py：第 4/13 课 CLI 计划与原始日志；默认只生成计划，--run 才执行已有独立学习服务。
- labs/lesson08_prefix.py、lesson09_mixed.py：使用本地 tokenizer 与流式 completion API，分别观察前缀首次/重复和长短固定轨迹；不报告逐 token ITL。
- labs/stage_a_plot.py：读取真实离线基线 JSON 生成独立 SVG；中位数与重复 min/max，不代表置信区间。

构建只用标准库。硬件实验在已验证镜像中执行，不在本地 macOS 安装完整 vLLM。PyTorch 数值练习可在 CPU 环境运行；远端实验分别记录 CUDA/CANN、设备、插件与模型 revision。静态检查入口为 tools/check_textbook.py。
