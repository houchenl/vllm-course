"""Measure fixed-size offline batches in an existing validated vLLM environment.

The wall time includes prefill, decode and client-side output processing. It does
not provide streaming TTFT, per-request latency or pure decode throughput.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import time
from pathlib import Path


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Existing local checkpoint directory")
    parser.add_argument("--output", required=True, help="New result JSON file")
    parser.add_argument("--input-lengths", nargs="+", type=positive_int, default=[128, 1024])
    parser.add_argument("--batch-sizes", nargs="+", type=positive_int, default=[1, 4])
    parser.add_argument("--output-length", type=positive_int, default=32)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--max-model-len", type=positive_int, default=2048)
    parser.add_argument("--memory-utilization", type=float, default=0.5)
    parser.add_argument("--model-revision", default="unrecorded")
    parser.add_argument("--image-id", default="unrecorded")
    args = parser.parse_args()
    model_path = Path(args.model).resolve()
    result_path = Path(args.output)
    if not model_path.is_dir():
        parser.error("--model must be an existing local checkpoint directory")
    if result_path.exists():
        parser.error("--output already exists; choose a new filename")
    if max(args.input_lengths) + args.output_length > args.max_model_len:
        parser.error("input length plus output length exceeds --max-model-len")
    if not 0 < args.memory_utilization <= 1:
        parser.error("--memory-utilization must be in (0, 1]")

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.platforms import current_platform

    config = {
        "model": str(model_path),
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": args.max_model_len,
        "max_num_seqs": max(args.batch_sizes),
        "gpu_memory_utilization": args.memory_utilization,
        "enable_prefix_caching": False,
        "enforce_eager": True,
        "seed": 17,
    }
    init_start = time.perf_counter()
    llm = LLM(**config)
    model_init_seconds = time.perf_counter() - init_start
    effective_config = llm.llm_engine.vllm_config
    tokenizer = llm.get_tokenizer()
    seed_ids = tokenizer.encode(
        "This synthetic workload measures inference execution with a fixed length. ",
        add_special_tokens=False,
    )
    if not seed_ids:
        raise RuntimeError("Tokenizer produced no seed tokens")
    params = SamplingParams(
        temperature=0, max_tokens=args.output_length,
        min_tokens=args.output_length, ignore_eos=True,
    )
    rows = []
    summaries = []
    for input_length in args.input_lengths:
        prompt_ids = (seed_ids * (input_length // len(seed_ids) + 1))[:input_length]
        for batch_size in args.batch_sizes:
            prompts = [{"prompt_token_ids": list(prompt_ids)} for _ in range(batch_size)]
            llm.generate(prompts, params, use_tqdm=False)
            case_rows = []
            for repeat in range(args.repeats):
                started = time.perf_counter()
                outputs = llm.generate(prompts, params, use_tqdm=False)
                elapsed = time.perf_counter() - started
                input_counts = [len(item.prompt_token_ids or []) for item in outputs]
                output_counts = [len(item.outputs[0].token_ids) for item in outputs]
                if len(outputs) != batch_size or input_counts != [input_length] * batch_size:
                    raise RuntimeError("Measured input/request counts differ from the workload")
                if output_counts != [args.output_length] * batch_size:
                    raise RuntimeError("Measured output lengths differ from the workload")
                row = {
                    "input_tokens_per_request": input_length,
                    "output_tokens_per_request": args.output_length,
                    "batch_requests": batch_size,
                    "repeat": repeat + 1,
                    "batch_wall_seconds": elapsed,
                    "aggregate_output_tokens_per_second": sum(output_counts) / elapsed,
                    "batch_requests_per_second": batch_size / elapsed,
                    "actual_input_counts": input_counts,
                    "actual_output_counts": output_counts,
                }
                rows.append(row)
                case_rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
            summaries.append({
                "input_tokens_per_request": input_length,
                "output_tokens_per_request": args.output_length,
                "batch_requests": batch_size,
                "median_batch_wall_seconds": statistics.median(
                    row["batch_wall_seconds"] for row in case_rows
                ),
                "median_aggregate_output_tokens_per_second": statistics.median(
                    row["aggregate_output_tokens_per_second"] for row in case_rows
                ),
                "min_batch_wall_seconds": min(row["batch_wall_seconds"] for row in case_rows),
                "max_batch_wall_seconds": max(row["batch_wall_seconds"] for row in case_rows),
            })
    result = {
        "measurement": "completed offline batch wall time; not streaming TTFT/ITL",
        "python": platform.python_version(),
        "vllm_module_file": getattr(vllm, "__file__", None),
        "vllm_reported_version": getattr(vllm, "__version__", None),
        "packages": {name: package_version(name) for name in (
            "vllm", "vllm-ascend", "torch", "torch-npu", "transformers", "triton"
        )},
        "platform_device_type": getattr(current_platform, "device_type", "unknown"),
        "visible_devices": {name: os.environ.get(name) for name in (
            "CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES"
        )},
        "image_id": args.image_id,
        "model_revision": args.model_revision,
        "engine_arguments": config,
        "effective_settings": {
            "use_v2_model_runner": getattr(effective_config, "use_v2_model_runner", None),
            "max_num_batched_tokens": getattr(
                effective_config.scheduler_config, "max_num_batched_tokens", None
            ),
            "enable_chunked_prefill": getattr(
                effective_config.scheduler_config, "enable_chunked_prefill", None
            ),
            "enable_prefix_caching": getattr(
                effective_config.cache_config, "enable_prefix_caching", None
            ),
            "cache_dtype": getattr(effective_config.cache_config, "cache_dtype", None),
        },
        "workload_arguments": vars(args),
        "model_init_seconds": model_init_seconds,
        "warmup_runs_per_case": 1,
        "rows": rows,
        "summaries": summaries,
        "not_measured": ["TTFT", "ITL", "per-request latency", "quality", "pure decode time"],
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("x", encoding="utf-8") as result_file:
        json.dump(result, result_file, ensure_ascii=False, indent=2)
    print(f"Saved {len(rows)} measurements to {result_path}")


if __name__ == "__main__":
    main()
