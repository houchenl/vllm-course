"""Build offline HTML course documents with the Python standard library.

Run from the course directory. HTML lessons are authored as body fragments;
the stylesheet and scripts are embedded so the delivered files need no CDN.
"""
from __future__ import annotations

import html
import base64
import json
import mimetypes
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "assets/textbook.css").read_text(encoding="utf-8")
BOOK_JS = (ROOT / "assets/book.js").read_text(encoding="utf-8")
LESSONS = json.loads((ROOT / "records/stage-a-lessons.json").read_text(encoding="utf-8"))
ROUTES = {
    (ROOT / "syllabus.md").resolve(): ROOT / "index.html",
    (ROOT / "README.md").resolve(): ROOT / "index.html",
    (ROOT / "source-map.md").resolve(): ROOT / "source-map.html",
    (ROOT / "editorial-guide.md").resolve(): ROOT / "editorial-guide.html",
    (ROOT / "course-rules.md").resolve(): ROOT / "rules.html",
    (ROOT / "records/progress.md").resolve(): ROOT / "records/progress.html",
    (ROOT / "records/lesson01-report-template.md").resolve(): ROOT / "records/lesson01-report-template.html",
    (ROOT / "lessons/01-inference-lifecycle.md").resolve(): ROOT / "lessons/01-inference-lifecycle.html",
}
for lesson in LESSONS:
    slug = lesson["slug"]
    ROUTES[(ROOT / "lessons" / (slug + ".md")).resolve()] = ROOT / "lessons" / (slug + ".html")
    week = lesson["week"]
    ROUTES[(ROOT / "records" / f"lesson{week}-report-template.md").resolve()] = ROOT / "records" / f"lesson{week}-report-template.html"


def link_target(value: str, source: Path, output: Path) -> str:
    parts = urlsplit(value)
    if parts.scheme or value.startswith("#"):
        return value
    target = (source.parent / parts.path).resolve()
    target = ROUTES.get(target, target)
    relative = Path(os.path.relpath(target, output.parent)).as_posix()
    return relative + ("?" + parts.query if parts.query else "") + ("#" + parts.fragment if parts.fragment else "")


def inline(value: str, source: Path, output: Path) -> str:
    chunks = re.split(r"(`[^`]+`|\[[^\]]+\]\([^)]+\))", value)
    rendered = []
    for chunk in chunks:
        if chunk.startswith("`") and chunk.endswith("`"):
            rendered.append("<code>" + html.escape(chunk[1:-1]) + "</code>")
        elif match := re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", chunk):
            label, href = match.groups()
            href = link_target(href, source, output)
            rendered.append('<a href="' + html.escape(href, quote=True) + '">' + html.escape(label) + "</a>")
        else:
            escaped = html.escape(chunk)
            rendered.append(re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped))
    return "".join(rendered)


def markdown(text: str, source: Path, output: Path, prefix: str, skip_title: bool = True) -> str:
    """Render the small Markdown subset used by course records and tables."""
    lines = text.splitlines()
    result: list[str] = []
    paragraph: list[str] = []
    index = 0
    section = 0
    list_kind: str | None = None

    def flush() -> None:
        if paragraph:
            result.append("<p>" + inline(" ".join(paragraph), source, output) + "</p>")
            paragraph.clear()

    def close_list() -> None:
        nonlocal list_kind
        if list_kind:
            result.append("</" + list_kind + ">")
            list_kind = None

    while index < len(lines):
        line = lines[index].strip()
        if not line:
            flush()
            close_list()
            index += 1
            continue
        if line.startswith("```"):
            flush()
            close_list()
            code = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code.append(lines[index])
                index += 1
            result.append("<pre><code>" + html.escape("\n".join(code)) + "</code></pre>")
            index += 1
            continue
        if re.fullmatch(r"\*\*.+\*\*", line):
            flush()
            close_list()
            if skip_title and index == 0:
                index += 1
                continue
            section += 1
            result.append(f'<h2 id="{prefix}-{section}">{inline(line[2:-2], source, output)}</h2>')
            index += 1
            continue
        if line.startswith("|"):
            flush()
            close_list()
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                    rows.append(cells)
                index += 1
            result.append('<div class="table-scroll"><table><thead><tr>')
            for cell in rows[0]:
                result.append("<th scope=\"col\">" + inline(cell, source, output) + "</th>")
            result.append("</tr></thead><tbody>")
            for row in rows[1:]:
                result.append("<tr>")
                for column, cell in enumerate(row):
                    anchor = f' id="group-{cell}"' if prefix == "map" and column == 0 and re.fullmatch(r"[A-U]", cell) else ""
                    result.append("<td" + anchor + ">" + inline(cell, source, output) + "</td>")
                result.append("</tr>")
            result.append("</tbody></table></div>")
            continue
        match = re.match(r"(- |\d+\. )(.+)", line)
        if match:
            flush()
            kind = "ul" if match.group(1).startswith("-") else "ol"
            if list_kind != kind:
                close_list()
                result.append("<" + kind + ">")
                list_kind = kind
            result.append("<li>" + inline(match.group(2), source, output) + "</li>")
        else:
            close_list()
            paragraph.append(line)
        index += 1
    flush()
    close_list()
    return "\n".join(result)


def header(title: str, subtitle: str, label: str = "课程资料") -> str:
    return f'<header class="chapter-head" id="top"><p class="eyebrow">VLLM &amp; AI INFRA · {html.escape(label)}</p><h1>{html.escape(title)}</h1><p class="lead">{subtitle}</p><div class="meta">更新：2026-10-14 · HTML 教材版 · 离线阅读 / A4 打印</div></header>'


def embed_images(body: str, output: Path) -> str:
    """Keep image assets editable locally while making delivered HTML standalone."""
    def embed(match: re.Match[str]) -> str:
        prefix, value, suffix = match.groups()
        href = html.unescape(value)
        if href.startswith("data:image/"):
            return match.group(0)
        parts = urlsplit(href)
        if parts.scheme or parts.netloc or not parts.path:
            raise ValueError(f"Textbook illustrations must be local assets: {href}")
        image_path = (output.parent / parts.path).resolve()
        if not image_path.is_relative_to(ROOT):
            raise ValueError(f"Illustration must be inside the course directory: {href}")
        mime, _ = mimetypes.guess_type(image_path.name)
        if mime not in {"image/png", "image/jpeg", "image/webp"}:
            raise ValueError(f"Unsupported illustration format: {image_path.name}")
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return prefix + f"data:{mime};base64,{encoded}" + suffix
    return re.sub(r'(<img\b[^>]*\bsrc=")([^"<>]+)(")', embed, body)


def page(output: Path, title: str, body: str, scripts: str = "", toc: list[tuple[str, str]] | None = None) -> None:
    body = embed_images(body, output)
    if output.parent == ROOT / "lessons":
        position = next((i for i, lesson in enumerate(LESSONS) if lesson["slug"] == output.stem), None)
        if position is not None:
            adjacent = []
            if position > 0:
                lesson = LESSONS[position - 1]
                adjacent.append(f'<a href="{lesson["slug"]}.html">← 上一课：{html.escape(lesson["title"])}</a>')
            if position + 1 < len(LESSONS):
                lesson = LESSONS[position + 1]
                adjacent.append(f'<a href="{lesson["slug"]}.html">下一课：{html.escape(lesson["title"])} →</a>')
            body += '<nav class="lesson-navigation" aria-label="相邻课程">' + "<br>".join(adjacent) + "</nav>"
    home = Path(os.path.relpath(ROOT / "index.html", output.parent)).as_posix()
    if toc is None:
        toc = [(identifier, html.unescape(re.sub(r"<[^>]+>", "", label))) for identifier, label in re.findall(r'<h2 id="([^"]+)">(.*?)</h2>', body, re.S)]
        if not toc:
            toc = [(identifier, html.unescape(re.sub(r"<[^>]+>", "", label))) for identifier, label in re.findall(r'<section id="([^"]+)">\s*<h2>(.*?)</h2>', body, re.S)]
    items = "".join(f'<li><a href="#{html.escape(identifier)}">{html.escape(label)}</a></li>' for identifier, label in toc)
    document = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><meta name="description" content="面向模型适配与推理调优的 vLLM 源码课程：{html.escape(title)}"><title>{html.escape(title)} · vLLM 与 AI Infra</title><style>
{CSS}
</style></head><body>
<a class="skip" href="#main">跳到正文</a>
<div class="bookbar"><a href="{home}">vLLM 与 AI Infra / 工程师课程</a><span class="edition">TEXTBOOK EDITION · 2026.10</span><button id="print-chapter" type="button" class="bar-action no-print">打印 / 存为 PDF</button></div>
<div class="layout"><nav class="toc" aria-label="本页目录"><details open><summary>本页目录 / CONTENTS</summary><ol>{items}</ol></details><div class="navnote">读懂机制 · 定位源码<br>建立证据 · 验证边界<br><br>每周 5 小时，按实际学习进度推进。</div></nav><main id="main">
{body}
</main></div><footer>中文原创课程讲义 · 源码、理论估算与实测分开记录 · 本地 HTML 无外部字体或脚本依赖</footer>
<script>
{BOOK_JS}
{scripts}
</script></body></html>
'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    print(output.relative_to(ROOT).as_posix())


def build_index() -> None:
    source = ROOT / "syllabus.md"
    output = ROOT / "index.html"
    text = source.read_text(encoding="utf-8")
    phases = []
    weeks = []
    for line in text.splitlines():
        if line.startswith("|"):
            row = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if re.fullmatch(r"[A-I]", row[0]):
                phases.append(row)
            elif re.fullmatch(r"\d{2}", row[0]):
                weeks.append(row)
    if len(phases) != 9 or [row[0] for row in weeks] != [f"{week:02}" for week in range(1, 59)]:
        raise ValueError("Expected nine phases and exactly 58 ordered weeks")
    body = header("vLLM 与 AI Infra", "从请求执行主干，走向模型适配、算子分析和跨硬件推理调优。", "课程入口与分阶段课表")
    body += '<div class="box"><div class="label">阶段 A · 14 课已准备</div><p><a class="lesson-ready" href="lessons/01-inference-lifecycle.html">从第 1 课开始：一次生成请求如何运行 →</a></p><p>第 1–14 课均有独立讲义、例题、源码导读、实验与报告模板。<a href="#phase-A">查看阶段 A 全部课程</a>。讲义准备、硬件实测与学员验收分别记录；后续阶段按学习反馈推进。</p></div>'
    intro = text.split("**阶段安排**")[0]
    body += markdown(intro, source, output, "intro")
    body += '<section id="roadmap"><h2>分阶段路线 · 58 周 / 290 小时</h2><div class="phase-strip">'
    for letter, budget, topics, assessment in phases:
        body += f'<div class="phase-card"><span class="tag">阶段 {letter} · {html.escape(budget)}</span><h3><a href="#phase-{letter}">{html.escape(topics)}</a></h3><p>{html.escape(assessment)}</p></div>'
    body += '</div><p class="small">每 4 周复盘；第 14、20、26、32、38、44、50、54、58 周验收。第 59–60 周为可选机动。工作案例可替换相同技能目标的实验。</p></section>'
    body += '<section id="weekly"><h2>逐周课表</h2><p>源码代码组对应<a href="source-map.html">源码地图</a>。每课限定 3–5 个核心文件或符号；代码组不等于整组必读。阶段 A 第 1–14 课已编写，阶段 B–I 仍为后续学习安排。</p>'
    for phase in phases:
        letter, budget, topics, assessment = phase
        span = re.match(r"(\d+)–(\d+)", budget)
        if span is None:
            raise ValueError(budget)
        start, end = map(int, span.groups())
        body += f'<h3 id="phase-{letter}">阶段 {letter}　{html.escape(topics)}</h3><p class="small">第 {start}–{end} 周 · 验收：{html.escape(assessment)}</p>'
        body += '<div class="table-scroll"><table><thead><tr><th scope="col">周</th><th scope="col">主要问题与知识</th><th scope="col">源码组</th><th scope="col">主要产物</th></tr></thead><tbody>'
        for row in weeks[start - 1:end]:
            week, topics, groups, product = row
            ready = next((lesson for lesson in LESSONS if lesson["week"] == week), None)
            cell = f'<a href="lessons/{ready["slug"]}.html">{week}<br>已编写</a>' if ready else week
            if ready:
                topics = f'<a href="lessons/{ready["slug"]}.html">{html.escape(ready["title"])}</a><br>' + inline(topics, source, output)
            else:
                topics = inline(topics, source, output)
            body += f'<tr data-week="{week}"><td class="nowrap">{cell}</td><td>{topics}</td><td>{inline(groups, source, output)}</td><td>{inline(product, source, output)}</td></tr>'
        body += '</tbody></table></div>'
    body += '</section>'
    tail = "**实验范围与前置条件**" + text.split("**实验范围与前置条件**", 1)[1]
    body += markdown(tail, source, output, "reference", skip_title=False)
    body += '<section id="documents"><h2>课程资料与记录</h2><ul><li><a href="rules.html">已确认课程规则</a> · <a href="editorial-guide.html">HTML 教材编写规范</a></li><li><a href="source-map.html">源码地图</a> · <a href="records/stage-a-source-snapshot.json">阶段 A 源码记录</a> · <a href="records/source-snapshot.json">原始教学指纹</a></li><li><a href="records/progress.html">学习进度</a> · <a href="records/lesson14-report-template.html">阶段 A 验收模板</a></li><li><a href="records/stage-a-illustrations.json">阶段 A 插图提示词与核对记录</a></li></ul><p>阶段 A 的硬件实验和学员验收尚未完成，详见各课报告。课程准备检查只覆盖讲义、构建与本地可执行练习。</p></section>'
    toc = [("top", "课程与工作背景"), ("roadmap", "分阶段路线"), ("weekly", "逐周课表")]
    toc += [("phase-" + row[0], "阶段 " + row[0] + " · 第 " + row[1].split(" / ")[0] + " 周") for row in phases]
    toc += [("documents", "课程资料与记录")]
    page(output, "课程入口与课表", body, toc=toc)


def build_documents() -> None:
    descriptions = [
        (ROOT / "source-map.md", "教学源码地图", "由工程问题定位代码组，再由每课限定必读范围。", "map"),
        (ROOT / "editorial-guide.md", "HTML 教材讲义编写规范", "用定义、推导、例题、源码和实验，建立可迁移的工程能力。", "editorial"),
        (ROOT / "course-rules.md", "已确认课程规则", "任务目标、学习顺序、预算与证据标准。", "rules"),
        (ROOT / "records/progress.md", "学习进度与环境事实", "区分讲义准备、实际学习、设备实测和通过验收。", "progress"),
        (ROOT / "records/lesson01-report-template.md", "第一课报告模板", "记录执行轨迹、容量估算、源码职责和基线数据。", "report"),
    ]
    for source, title, description, prefix in descriptions:
        output = ROUTES[source.resolve()]
        body = header(title, description)
        if prefix in {"progress", "report"}:
            body += '<p class="small">填写或更新请使用配套 <a href="' + source.name + '">Markdown 记录文件</a>；此页用于阅读与打印。</p>'
        body += markdown(source.read_text(encoding="utf-8"), source, output, prefix)
        page(output, title, body)
    for lesson in LESSONS[1:]:
        source = ROOT / "records" / f'lesson{lesson["week"]}-report-template.md'
        output = ROUTES[source.resolve()]
        body = header(f'第 {lesson["week"]} 课报告模板', lesson["title"], "学习与实验记录")
        body += f'<p class="small">使用 <a href="{source.name}">Markdown 编写源</a>填写；<a href="../lessons/{lesson["slug"]}.html">返回本课讲义</a>。</p>'
        body += markdown(source.read_text(encoding="utf-8"), source, output, "report")
        page(output, f'第 {lesson["week"]} 课报告模板', body)


def build_template() -> None:
    body = header("第 XX 课　课程标题", "待填写：本课解决的一个具体工程问题。", "讲义编写模板")
    names = [
        ("orientation", "问题、目标与时间预算", "写明前置知识、能力目标、5 小时预算、必做产物与验收。"),
        ("concept", "定义与直观模型", "解释术语，给出状态/数据流图、输入输出与可推演的小例子。需要插图时调用画图工具生成，核对技术关系，配图注和替代文本，嵌入 HTML。"),
        ("derivation", "机制与关键推导", "先写适用假设与变量单位，再推导关键公式，说明近似与忽略项。"),
        ("example", "完整例题", "按已知、建模、计算、检查和工程解释展开，写出中间步骤。"),
        ("source", "源码导读", "限定 3–5 个核心文件/符号，注明快照、输入输出、状态和进程/设备边界。"),
        ("lab", "受控实验", "写明假设、固定变量、控制变量、命令、测量边界、原始数据与重复次数。"),
        ("problems", "习题与验收", "设置概念、估算、源码和实验设计题，区分必做与选做，并给出参考推理。"),
        ("references", "证据与进一步阅读", "在相关结论附近链接一手资料，分别记录源码、估算和实测。"),
    ]
    for identifier, title, prompt in names:
        body += f'<section id="{identifier}"><h2>{title}</h2><p class="small">编写提示：{prompt}</p></section>'
    body += '<details class="answer"><summary>参考解答（编写占位）</summary><div class="answer-body"><p>填写推理、单位检查与适用条件。</p></div></details>'
    body += '<p>此文件是编写模板，不是已完成课程。</p>'
    page(ROOT / "templates/lesson-template.html", "新课编写模板", body)


def main() -> None:
    for lesson in LESSONS:
        fragment = (ROOT / "materials" / (lesson["slug"] + ".body.html")).read_text(encoding="utf-8")
        script = (ROOT / "assets" / ("lesson01.js" if lesson["week"] == "01" else "stage-a.js")).read_text(encoding="utf-8")
        page(ROOT / "lessons" / (lesson["slug"] + ".html"), f'第 {lesson["week"]} 课 · {lesson["title"]}', fragment, scripts=script)
    build_index()
    build_documents()
    build_template()


if __name__ == "__main__":
    main()
