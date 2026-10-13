"""Check generated offline course pages, local links, anchors and lesson coverage."""

from collections import Counter
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote,urlsplit

ROOT=Path(__file__).resolve().parents[1]
VOID={"area","base","br","col","embed","hr","img","input","link","meta","param","source","track","wbr"}


class Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids=[]
        self.links=[]
        self.stack=[]
        self.errors=[]
        self.weeks=[]
        self.images=0
        self.math=0
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag not in VOID:
            self.stack.append(tag)
        if tag=="img":
            self.images+=1
            if not attrs.get("alt"):
                self.errors.append("image missing alternative text")
        if tag=="math":
            self.math+=1
        if tag=="script" and "src" in attrs:
            self.errors.append("script dependency is not embedded")
        if tag=="link" and attrs.get("rel")=="stylesheet":
            self.errors.append("stylesheet is not embedded")
        if tag=="img" and not attrs.get("src","").startswith("data:image/"):
            self.errors.append("illustration is not embedded")
        if "data-week" in attrs:
            self.weeks.append(attrs["data-week"])
        for key in ["href","src"]:
            if attrs.get(key):
                self.links.append(attrs[key])
    def handle_endtag(self,tag):
        if tag in VOID:
            return
        if not self.stack or self.stack[-1]!=tag:
            self.errors.append(f"unpaired closing {tag}; stack tail {self.stack[-3:]}")
        else:
            self.stack.pop()
    def handle_startendtag(self,tag,attrs):
        self.handle_starttag(tag,attrs)
        if tag not in VOID:
            self.handle_endtag(tag)


def main():
    pages={}
    errors=[]
    count=0
    for path in sorted(ROOT.rglob("*.html")):
        if any(part.startswith(".") for part in path.relative_to(ROOT).parts):
            continue
        if path.parent==ROOT/"materials":
            continue
        parser=Page()
        parser.feed(path.read_text(encoding="utf-8"))
        pages[path.resolve()]=parser
        parser.errors.extend(f"duplicate id {key}" for key,n in Counter(parser.ids).items() if n>1)
        if parser.stack:
            parser.errors.append(f"unclosed tags {parser.stack}")
        errors.extend(f"{path.relative_to(ROOT)}: {e}" for e in parser.errors)
    for path,parser in pages.items():
        for link in parser.links:
            parts=urlsplit(link)
            if parts.scheme or parts.netloc:
                continue
            count+=1
            target=(path.parent/unquote(parts.path)).resolve() if parts.path else path
            if not target.exists():
                errors.append(f"{path.relative_to(ROOT)}: missing {link}")
            elif parts.fragment and target in pages and unquote(parts.fragment) not in pages[target].ids:
                errors.append(f"{path.relative_to(ROOT)}: missing anchor {link}")
    lessons=json.loads((ROOT/"records/stage-a-lessons.json").read_text())
    if [row["week"] for row in lessons]!=[f"{i:02}" for i in range(1,15)]:
        errors.append("expected exactly fourteen ordered stage A lessons")
    for lesson in lessons:
        path=(ROOT/"lessons"/(lesson["slug"]+".html")).resolve()
        if path not in pages:
            errors.append(f"missing lesson {lesson['week']}")
            continue
        ids=pages[path].ids
        if any(key not in ids for key in ["orientation","source","lab","references"]):
            errors.append(f"lesson {lesson['week']} missing required sections")
        if not (ROOT/"records"/f"lesson{lesson['week']}-report-template.html").is_file():
            errors.append(f"lesson {lesson['week']} missing report")
        if lesson["week"]!="01" and not 3<=len(lesson["sources"])<=5:
            errors.append(f"lesson {lesson['week']} must limit source reading to 3–5 entries")
    if pages[(ROOT/"index.html").resolve()].weeks!=[f"{i:02}" for i in range(1,59)]:
        errors.append("syllabus must contain exactly 58 ordered weeks")
    report=dict(pages=len(pages),stage_a_lessons=len(lessons),local_links_and_anchors=count,
                embedded_images=sum(p.images for p in pages.values()),
                mathml_equations=sum(p.math for p in pages.values()),errors=errors)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if errors:
        raise SystemExit(1)


if __name__=="__main__":
    main()
