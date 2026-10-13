"""Observe first/repeated prefix requests without target-prefix preflight traffic."""

import argparse
import json
from pathlib import Path

from streaming_client import observe,token_workload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url",default="http://127.0.0.1:8000")
    parser.add_argument("--model",default="course-dense")
    parser.add_argument("--tokenizer",required=True)
    parser.add_argument("--cache-condition",choices=["enabled-fresh","disabled"],required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    path=Path(args.output)
    if path.exists():
        parser.error("choose a new output filename")
    if not Path(args.tokenizer).is_dir():
        parser.error("tokenizer must be a local directory")
    common=token_workload(args.tokenizer,768,variant=17)
    unrelated=token_workload(args.tokenizer,896,variant=91)
    # Rotate the initial tokens to ensure the warmup starts with a different prefix.
    first=next((token for token in unrelated if token!=common[0]),None)
    if first is None:
        parser.error("cannot construct a distinct warmup prefix")
    unrelated[0]=first
    suffix=token_workload(args.tokenizer,128,variant=23)
    target=common+suffix
    shifted=list(target)
    shifted[0]=first
    workloads=[("unrelated-warmup",unrelated),("target-first",target),
               ("target-repeat-1",target),("target-repeat-2",target),
               ("target-repeat-3",target),("changed-prefix-control",shifted)]
    rows=[]
    for label,prompt in workloads:
        row=observe(args.base_url,args.model,prompt)
        row["condition"]=label
        rows.append(row)
        if not row["success"]:
            break
    report=dict(arguments=vars(args),rows=rows,
                cache_label_is_user_assertion=True,
                cold_proven_by_client=False,
                evidence_required="server APC flag, clean target prefix state, hit statistics and runtime versions",
                not_measured=["server cache hit count","per-token ITL","pure prefill time"])
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
    print(f"Saved {len(rows)} requests to {path}")
    if any(not row["success"] for row in rows):
        raise SystemExit("Request failed; preserved response/error in JSON")


if __name__=="__main__":
    main()
