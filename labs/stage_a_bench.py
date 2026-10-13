"""Save controlled vLLM CLI plans; execute only with --run in a validated image."""

import argparse
import json
import shutil
import shlex
import subprocess
import time
from pathlib import Path


def positive(value):
    number=int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def make_command(args,case,name,num_prompts):
    return [args.vllm_bin,"bench","serve","--base-url",args.base_url,
            "--backend","vllm","--endpoint","/v1/completions",
            "--model",args.model,"--served-model-name",args.served_model,
            "--tokenizer",args.model,"--dataset-name","random",
            "--random-input-len",str(case["input"]),"--random-output-len","32",
            "--random-range-ratio","0","--seed","17","--temperature","0",
            "--num-prompts",str(num_prompts),"--ignore-eos",
            "--request-rate",str(case["rate"]),"--max-concurrency",str(case["concurrency"]),
            "--percentile-metrics","ttft,tpot,itl,e2el","--metric-percentiles","50,95,99",
            "--save-result","--save-detailed","--result-dir",str(args.output_dir),
            "--result-filename",name+".json"]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lesson",type=int,choices=[4,13],required=True)
    parser.add_argument("--model",required=True,help="local tokenizer/checkpoint path")
    parser.add_argument("--served-model",default="course-dense")
    parser.add_argument("--base-url",default="http://127.0.0.1:8000")
    parser.add_argument("--vllm-bin",default="vllm")
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--num-prompts",type=positive,default=100)
    parser.add_argument("--repeats",type=positive,default=3)
    parser.add_argument("--run",action="store_true")
    args=parser.parse_args()
    if args.output_dir.exists():
        parser.error("choose a new output directory; results are never overwritten")
    cases=[dict(name="serial",input=128,rate="inf",concurrency=1),
           dict(name="concurrent",input=128,rate="inf",concurrency=4),
           dict(name="open-low",input=128,rate=2,concurrency=16)]
    if args.lesson==13:
        cases=[dict(name=f"p{length}-c{c}",input=length,rate="inf",concurrency=c)
               for length,c in [(128,1),(128,4),(1024,4)]]
    jobs=[]
    for case in cases:
        name=case["name"]+"-warmup"
        jobs.append(dict(kind="warmup",command=make_command(args,case,name,8),name=name))
        for repeat in range(args.repeats):
            name=case["name"]+f"-r{repeat+1}"
            jobs.append(dict(kind="measurement",command=make_command(args,case,name,args.num_prompts),name=name))
    args.output_dir.mkdir(parents=True)
    plan=dict(lesson=args.lesson,executed=False,server_restart_performed=False,
              server_configuration="must be saved separately with startup log, image and model revisions",
              seed=17,output_length=32,cases=cases,jobs=jobs,
              caveat="benchmark can send a test request before timing; do not use for a true cold APC measurement")
    plan_path=args.output_dir/"plan.json"
    plan_path.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding="utf-8")
    for job in jobs:
        print(shlex.join(job["command"]))
    if not args.run:
        print("Plan saved. No vLLM process or server request was executed.")
        return
    if not shutil.which(args.vllm_bin):
        parser.error("vLLM CLI unavailable in this environment; saved plan remains unexecuted")
    help_result=subprocess.run([args.vllm_bin,"bench","serve","--help"],capture_output=True,text=True)
    (args.output_dir/"cli-help.txt").write_text(help_result.stdout+help_result.stderr,encoding="utf-8")
    required=["--random-input-len","--random-output-len","--random-range-ratio",
              "--max-concurrency","--served-model-name","--save-detailed","--temperature"]
    if help_result.returncode or any(flag not in help_result.stdout for flag in required):
        parser.error("installed CLI differs; inspect saved help and adapt the plan before executing")
    statuses=[]
    for job in jobs:
        started=time.monotonic()
        with (args.output_dir/(job["name"]+".log")).open("x",encoding="utf-8") as log:
            completed=subprocess.run(job["command"],stdout=log,stderr=subprocess.STDOUT)
        statuses.append(dict(name=job["name"],kind=job["kind"],exit_code=completed.returncode,
                             command_wall_seconds=time.monotonic()-started))
        (args.output_dir/"execution.json").write_text(json.dumps(statuses,indent=2),encoding="utf-8")
        if completed.returncode:
            raise SystemExit("Benchmark failed; inspect the saved log. Remaining jobs were not run.")
    plan["executed"]=True
    plan_path.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
