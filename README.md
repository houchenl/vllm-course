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
- [教学源码指纹](records/source-snapshot.json)
- [第一课容量估算脚本](labs/lesson01_estimate.py)
- [第一课离线基线脚本](labs/lesson01_baseline.py)

工作案例：DeepSeek-V4.1-Flash、GLM-5.3-Flash。实验平台以 H20 与 910C 为主，910B 为补充；RTX PRO 5000 纳入迁移案例。阅读基线为工作区的 vLLM 与 Ascend 源码快照；用户报告运行版本为 vLLM 0.31.0、vLLM-Ascend 0.30.0rc1，镜像内部的完整版本组合尚待记录。

推荐本周顺序：读第一课原理 → 完成容量例题 → 按指定范围读源码 → 在已有验证环境运行实验 → 填写报告与自测。无需先安装本地 Windows 版 vLLM，也无需先编译完整源码。

本轮已准备课程和实验文件，尚未连接远端加速器环境。硬件实验结果由实际运行产生，学习进度不会因讲义已准备而自动标为完成。

HTML 为正式交付，原有 Markdown 保留为记录或早期稿件。课程规则编写源为 [course-rules.md](course-rules.md)，第一课正文编写源为 materials/01-inference-lifecycle.body.html。后续课件按 [编写规范](editorial-guide.html) 与 templates/lesson-template.html 编写，保持同一套教学标准。

**本地阅读与构建**

当前课程仓库为 `ie/vllm-course`，远程为 [houchenl/vllm-course](https://github.com/houchenl/vllm-course)。打开 index.html 即可阅读；正式讲义包含内嵌样式、脚本和插图。

在仓库根目录使用 Python 3.10 或更新版本重新生成网页，只依赖标准库，不需要安装 vLLM：

```bash
python tools/build_textbook.py
```

源码链接面向本地阅读环境，要求 `vllm-main-261013` 和 `vllm-ascend-main-1013` 与课程仓库同级。源码快照不随本课程仓库提交；单独克隆课程仍可阅读讲义，但需放置相应源码快照才能打开本地源码链接。具体阅读锚点与 SHA256 见 records/source-snapshot.json。

Git 保存讲义、课表、规则、编写源、插图、实验脚本和配套记录。`.venv/`、`.uv-cache/`、`__pycache__/`、`.browser-check/` 为本地运行或检查产物，已忽略。
