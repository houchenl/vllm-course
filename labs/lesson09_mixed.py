"""Replay a fixed long/short token workload; record actual dispatch deviations."""

import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import time
from pathlib import Path

from streaming_client import observe,token_workload


def positive(value):
    number=int(value)
    if number<1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url",default="http://127.0.0.1:8000")
    parser.add_argument("--model",default="course-dense")
    parser.add_argument("--tokenizer",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--requests",type=positive,default=24)
    parser.add_argument("--interval-ms",type=float,default=40)
    parser.add_argument("--workers",type=positive,default=16)
    args=parser.parse_args()
    path=Path(args.output)
    if path.exists():
        parser.error("choose a new output filename")
    if not 1 <= args.requests <= 1000 or not 1 <= args.workers <= 64 or not 0 <= args.interval_ms <= 10000:
        parser.error("workload limits: requests≤1000, workers≤64, finite interval 0..10000ms")
    if not Path(args.tokenizer).is_dir():
        parser.error("tokenizer must be local")
    short=token_workload(args.tokenizer,128,variant=31)
    long=token_workload(args.tokenizer,1024,variant=47)
    # Common warmup for each new service configuration, kept out of measured rows.
    warmup=[observe(args.base_url,args.model,short),observe(args.base_url,args.model,long)]
    rows=[]
    if all(row["success"] for row in warmup):
        epoch=time.monotonic()
        def run(index,kind,planned):
            row=observe(args.base_url,args.model,long if kind=="long" else short)
            row.update(index=index,kind=kind,planned_seconds=planned-epoch,
                       actual_start_seconds=row["started_monotonic"]-epoch,
                       client_dispatch_lag_seconds=row["started_monotonic"]-planned)
            return row
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[]
            for i in range(args.requests):
                planned=epoch+i*args.interval_ms/1000
                delay=planned-time.monotonic()
                if delay>0:
                    time.sleep(delay)
                kind="long" if i%4==0 else "short"
                futures.append(pool.submit(run,i,kind,planned))
            for future in as_completed(futures):
                rows.append(future.result())
        rows.sort(key=lambda row:row["index"])
    report=dict(arguments=vars(args),warmup=warmup,rows=rows,
                workload="one long then three short; fixed scheduled interarrival times",
                caveat="thread pool and OS may delay dispatch; actual start and lag are recorded",
                not_measured=["per-token ITL","server queue time","CPU-device overlap"])
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
    print(f"Saved {len(rows)} measured requests to {path}")
    if any(not r["success"] for r in warmup+rows):
        raise SystemExit("Failed requests preserved; inspect JSON")


if __name__=="__main__":
    main()
