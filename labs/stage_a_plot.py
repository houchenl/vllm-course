"""Plot real offline baseline JSON as standalone SVG, with repeat min/max ranges."""

import argparse
import json
import math
import statistics
from pathlib import Path


def collect(paths):
    groups={}
    for path in paths:
        result=json.loads(path.read_text())
        if not result.get("measurement","").startswith("completed offline batch"):
            raise ValueError(f"{path}: expected offline batch measurements")
        config=result["engine_arguments"]
        tp=config["tensor_parallel_size"]
        if not isinstance(tp,int) or isinstance(tp,bool) or tp<1:
            raise ValueError("TP must be a positive integer")
        identity=json.dumps({
            "engine_arguments":{k:v for k,v in config.items() if k!="tensor_parallel_size"},
            "model_revision":result["model_revision"],
            "image_id":result.get("image_id"),
            "platform_device_type":result.get("platform_device_type"),
            "packages":result.get("packages"),
            "effective_settings":result.get("effective_settings"),
        },sort_keys=True)
        for row in result["rows"]:
            case=(row["input_tokens_per_request"],row["output_tokens_per_request"],row["batch_requests"])
            if not all(isinstance(x,int) and not isinstance(x,bool) and x>0 for x in case):
                raise ValueError("workload lengths and request counts must be positive integers")
            key=(case,tp)
            group=groups.setdefault(key,dict(identity=identity,times=[],rates=[],files=[]))
            if group["identity"]!=identity:
                raise ValueError("incompatible runs within a TP/case group")
            elapsed=row["batch_wall_seconds"]
            rate=row["aggregate_output_tokens_per_second"]
            if not all(isinstance(x,(int,float)) and math.isfinite(x) and x>0 for x in [elapsed,rate]):
                raise ValueError("non-finite or nonpositive measurements")
            if row["actual_output_counts"]!=[case[1]]*case[2]:
                raise ValueError("output counts differ from requested workload")
            if row["actual_input_counts"]!=[case[0]]*case[2]:
                raise ValueError("input counts differ from requested workload")
            if not math.isclose(rate,case[1]*case[2]/elapsed,rel_tol=1e-6):
                raise ValueError("throughput does not match completed output counts and wall time")
            group["times"].append(elapsed*1000)
            group["rates"].append(rate)
            group["files"].append(str(path))
    identities={g["identity"] for g in groups.values()}
    if len(identities)!=1:
        raise ValueError("models, revisions or fixed engine settings differ across plotted runs")
    cases=sorted({case for case,tp in groups})
    if not groups or any(len({tp for case,tp in groups if case==c})<2 for c in cases):
        raise ValueError("each workload requires at least two measured TP settings")
    return groups,cases


def plot(groups,cases):
    width=1000
    height=120+len(cases)*260
    lines=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
           '<title id="title">Real offline TP scaling measurements</title>',
           '<desc id="desc">Each workload has throughput and batch wall time panels. Points are medians; error bars are repeat minimum and maximum, not confidence intervals.</desc>',
           '<rect width="100%" height="100%" fill="#fbfaf6"/>',
           '<g font-family="sans-serif" fill="#243140">',
           '<text x="50" y="35" font-size="22">TP scaling · actual offline batches</text>',
           '<text x="50" y="60" font-size="13">Median + repeat min/max; these are not streaming latency metrics or confidence intervals.</text>']
    for case_index,case in enumerate(cases):
        points=sorted((tp,g) for (c,tp),g in groups.items() if c==case)
        top=100+case_index*260
        lines.append(f'<text x="50" y="{top}" font-size="15">P={case[0]}, N={case[1]}, B={case[2]}</text>')
        for panel,metric,label in [(0,"rates","Aggregate output tokens/s"),(1,"times","Batch wall time / ms")]:
            left=70+panel*490
            baseline=top+180
            maximum=max(max(g[metric]) for tp,g in points)*1.1
            lines.extend([f'<text x="{left}" y="{top+23}" font-size="13">{label}</text>',
                          f'<path d="M{left} {top+40}V{baseline}H{left+360}" fill="none" stroke="#9aa8a1"/>'])
            def yy(v):
                return baseline-v/maximum*130
            xy=[]
            for i,(tp,g) in enumerate(points):
                x=left+40+i*280/(len(points)-1)
                median=statistics.median(g[metric])
                y=yy(median)
                xy.append(f'{x},{y}')
                lo,hi=yy(min(g[metric])),yy(max(g[metric]))
                lines.append(f'<path d="M{x} {lo}V{hi}M{x-5} {lo}H{x+5}M{x-5} {hi}H{x+5}" stroke="#176759" fill="none"/>')
                lines.append(f'<circle cx="{x}" cy="{y}" r="4" fill="#176759"/>')
                lines.append(f'<text x="{x}" y="{y-10}" text-anchor="middle" font-size="12">{median:.3g} (n={len(g[metric])})</text>')
                lines.append(f'<text x="{x}" y="{baseline+20}" text-anchor="middle" font-size="12">TP={tp}</text>')
            lines.append(f'<polyline points="{" ".join(xy)}" fill="none" stroke="#176759"/>')
    lines.append('</g></svg>')
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs",nargs="+",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output filename")
    try:
        groups,cases=collect(args.inputs)
    except (ValueError,KeyError,OSError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as output:
        output.write(plot(groups,cases))
    print(args.output)


if __name__=="__main__":
    main()
