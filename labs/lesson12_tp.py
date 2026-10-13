"""Verify column/row partition algebra on CPU; no distributed performance claim."""

import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    path=Path(args.output)
    if path.exists():
        parser.error("choose a new output filename")
    import torch
    torch.manual_seed(17)
    rows=[]
    with torch.inference_mode():
        for tokens,d,f,p in [(1,8,12,2),(7,8,12,2),(3,12,24,4)]:
            x=torch.randn(tokens,d)
            w1=torch.randn(d,f)
            w2=torch.randn(f,d)
            bias=torch.randn(d)
            reference=torch.relu(x@w1)@w2+bias
            first=w1.chunk(p,dim=1)
            second=w2.chunk(p,dim=0)
            partial=[torch.relu(x@a)@b for a,b in zip(first,second)]
            partitioned=torch.stack(partial).sum(0)+bias
            torch.testing.assert_close(reference,partitioned,rtol=2e-5,atol=2e-5)
            wrong_bias=torch.stack([part+bias for part in partial]).sum(0)
            extra_bias=wrong_bias-partitioned
            torch.testing.assert_close(extra_bias,(p-1)*bias.expand_as(extra_bias),rtol=2e-5,atol=2e-5)
            rows.append(dict(tokens=tokens,hidden=d,intermediate=f,tp=p,
                             local_w1_shape=list(first[0].shape),local_w2_shape=list(second[0].shape),
                             partial_shape=list(partial[0].shape),
                             max_abs_error=(reference-partitioned).abs().max().item(),
                             payload_bytes=tokens*d*x.element_size()))
    report=dict(kind="CPU algebra verification",torch=torch.__version__,seed=17,dtype="float32",
                rtol=2e-5,atol=2e-5,rows=rows,
                not_measured=["actual collective traffic","GPU/NPU TP performance","full model quality"])
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
