"""Record selected teaching file hashes and Git identities; do not infer runtime support."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def git(repo,*args):
    return subprocess.check_output(["git","-C",str(repo),*args],text=True).strip()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error("choose a new snapshot path; historical fingerprints are preserved")
    old=json.loads((ROOT/"records/source-snapshot.json").read_text())
    rename={"vllm-main-261013":"vllm","vllm-ascend-main-1013":"vllm-ascend"}
    previous={}
    for row in old["files"]:
        p=Path(row["path"])
        path=(Path(rename[p.parts[0]])/Path(*p.parts[1:])).as_posix()
        previous[path]=row["sha256"]
    paths=set(previous)
    for lesson in json.loads((ROOT/"records/stage-a-lessons.json").read_text()):
        paths.update(s["path"] for s in lesson["sources"])
    paths.update(["vllm/docs/design/prefix_caching.md","vllm/docs/configuration/optimization.md",
                  "vllm/docs/design/hybrid_kv_cache_manager.md","vllm/docs/serving/parallelism_scaling.md",
                  "vllm/vllm/v1/worker/gpu/README.md"])
    files=[]
    for path in sorted(paths):
        content=(ROOT.parent/path).read_bytes()
        sha=hashlib.sha256(content).hexdigest()
        files.append(dict(path=path,sha256=sha,bytes=len(content),
                          historical_sha256=previous.get(path),
                          historical_match=sha==previous[path] if path in previous else None))
    repos={}
    for name in ["vllm","vllm-ascend"]:
        repo=ROOT.parent/name
        repos[name]=dict(commit=git(repo,"rev-parse","HEAD"),
                         dirty_status=git(repo,"status","--porcelain"))
    result=dict(created_client_date="2026-10-13",repositories=repos,
                scope="selected teaching files plus Git commit identities; not runtime validation or upstream support claims",
                historical_snapshot="records/source-snapshot.json",runtime_verified=False,
                ascend_required_main_vllm_commit=(ROOT.parent/"vllm-ascend/.github/vllm-main-verified.commit").read_text().strip(),
                files=files,
                selected_file_set_sha256=hashlib.sha256("\n".join(f'{f["path"]}\t{f["sha256"]}' for f in files).encode()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as output:
        json.dump(result,output,ensure_ascii=False,indent=2)
    print(f"Recorded {len(files)} selected files: {args.output}")


if __name__=="__main__":
    main()
