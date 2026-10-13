"""Inspect tensor storage and compare submission versus synchronized timing."""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path


def inspect(t):
    return dict(shape=list(t.shape),stride=list(t.stride()),
                storage_offset=t.storage_offset(),logical_bytes=t.numel()*t.element_size(),
                storage_bytes=t.untyped_storage().nbytes(),contiguous=t.is_contiguous())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",choices=["cpu","cuda","npu"],default="cpu")
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("choose a new output file")
    import torch
    if args.device == "npu":
        import torch_npu  # Registers the NPU device and torch.npu API.
    api = getattr(torch,args.device,None) if args.device != "cpu" else None
    if api and not api.is_available():
        parser.error("requested accelerator is unavailable")
    x = torch.arange(6,dtype=torch.float32).reshape(2,3)
    y = x.transpose(0,1)
    z = y.contiguous()
    torch.testing.assert_close(y,z,rtol=0,atol=0)
    shared = x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()
    copied = y.untyped_storage().data_ptr() != z.untyped_storage().data_ptr()
    a = torch.randn(512,512,device=args.device)
    b = torch.randn_like(a)
    def sync():
        if api:
            api.synchronize()
    with torch.inference_mode():
        for _ in range(5):
            _ = a @ b
        sync()
        rows = []
        for _ in range(5):
            sync()
            start_event = end_event = None
            if args.device == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            started = time.perf_counter()
            for _ in range(20):
                result = a @ b
            submitted = time.perf_counter()
            if end_event:
                end_event.record()
            sync()
            finished = time.perf_counter()
            rows.append(dict(submission_wall_seconds=submitted-started,
                             synchronized_wall_seconds=finished-started,
                             device_event_ms=start_event.elapsed_time(end_event) if start_event else None))
        finite = bool(torch.isfinite(result).all().cpu().item())
    report = dict(device=args.device,torch=importlib.metadata.version("torch"),
                  dtype=str(a.dtype),cuda_runtime=torch.version.cuda,
                  x=inspect(x),transpose=inspect(y),contiguous_copy=inspect(z),
                  transpose_shares_storage=shared,contiguous_allocates= copied,
                  matmul_shape=[512,512,512],operations_per_repeat=20,warmup_operations=5,
                  measurements=rows,finite_output=finite,
                  timing_scope="CPU perf_counter submission and synchronized completion; CUDA event on current stream only",
                  not_measured=["vLLM performance","production TTFT","NPU event interval"])
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
