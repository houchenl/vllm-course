"""Check a small causal GQA reference and tiled online-softmax recurrence.

This deliberately materializes small scores; it is not a fast attention kernel.
"""

import argparse
import json
from pathlib import Path


def tiled_attention(torch,q,k,v,tile):
    length,dim = q.shape[-2:]
    m = torch.full((*q.shape[:-1],1),-torch.inf,device=q.device,dtype=q.dtype)
    l = torch.zeros_like(m)
    a = torch.zeros_like(q)
    queries = torch.arange(length,device=q.device)[:,None]
    for start in range(0,length,tile):
        end = min(start+tile,length)
        score = q @ k[...,start:end,:].transpose(-1,-2) / dim**0.5
        keys = torch.arange(start,end,device=q.device)[None,:]
        score = score.masked_fill(keys > queries,-torch.inf)
        block_max = score.amax(-1,keepdim=True)
        next_m = torch.maximum(m,block_max)
        alpha = torch.exp(m-next_m)
        probability = torch.exp(score-next_m)
        l = alpha*l + probability.sum(-1,keepdim=True)
        a = alpha*a + probability @ v[...,start:end,:]
        m = next_m
    return a/l


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",choices=["cpu","cuda","npu"],default="cpu")
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        parser.error("choose a new output filename")
    import torch
    if args.device == "npu":
        import torch_npu
    torch.manual_seed(17)
    rows=[]
    with torch.inference_mode():
        for length,tile in [(1,1),(7,3),(16,5),(33,8)]:
            q=torch.randn(1,4,length,8,device=args.device,dtype=torch.float32)
            k=torch.randn(1,2,length,8,device=args.device,dtype=torch.float32)
            v=torch.randn_like(k)
            repeated_k=k.repeat_interleave(2,dim=1)
            repeated_v=v.repeat_interleave(2,dim=1)
            scores=q @ repeated_k.transpose(-1,-2) / 8**0.5
            mask=torch.ones(length,length,device=args.device,dtype=torch.bool).triu(1)
            ref=scores.masked_fill(mask,-torch.inf).softmax(-1) @ repeated_v
            tiled=tiled_attention(torch,q,repeated_k,repeated_v,tile)
            torch.testing.assert_close(ref,tiled,rtol=2e-5,atol=2e-5)
            grouped=[]
            for head in range(4):
                s=q[:,head] @ k[:,head//2].transpose(-1,-2) / 8**0.5
                grouped.append(s.masked_fill(mask,-torch.inf).softmax(-1) @ v[:,head//2])
            grouped=torch.stack(grouped,dim=1)
            torch.testing.assert_close(ref,grouped,rtol=2e-5,atol=2e-5)
            rows.append(dict(length=length,tile=tile,q_shape=list(q.shape),kv_shape=list(k.shape),
                             max_abs_tiled_error=(ref-tiled).abs().max().cpu().item(),
                             max_abs_grouped_error=(ref-grouped).abs().max().cpu().item()))
    report=dict(kind="teaching_numerical_reference",torch=torch.__version__,device=args.device,
                dtype="float32",seed=17,mask="full causal",rtol=2e-5,atol=2e-5,
                rows=rows,not_measured=["backend selection","FlashAttention hardware speed","model quality"])
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
