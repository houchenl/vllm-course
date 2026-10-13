"""Estimate standard full-attention KV storage for a single unsharded device."""

import argparse
import json
import math


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=positive_int, default=32)
    parser.add_argument("--kv-heads", type=positive_int, default=8)
    parser.add_argument("--head-dim", type=positive_int, default=128)
    parser.add_argument("--bytes-per-element", type=positive_int, default=2)
    parser.add_argument("--tokens", type=positive_int, default=4096)
    parser.add_argument("--requests", type=positive_int, default=8)
    parser.add_argument("--block-size", type=positive_int, default=16)
    args = parser.parse_args()
    bytes_per_token = (
        2 * args.layers * args.kv_heads * args.head_dim * args.bytes_per_element
    )
    rounded_tokens = math.ceil(args.tokens / args.block_size) * args.block_size
    ideal_bytes = bytes_per_token * args.tokens * args.requests
    paged_bytes = bytes_per_token * rounded_tokens * args.requests
    print(json.dumps({
        "scope": "standard full attention, one unsharded device, no KV sharing",
        "assumptions": vars(args),
        "kv_bytes_per_token": bytes_per_token,
        "ideal_kv_bytes": ideal_bytes,
        "ideal_kv_gib": ideal_bytes / 2**30,
        "block_rounded_kv_gib": paged_bytes / 2**30,
        "excludes": [
            "weights", "activation/workspace", "graphs", "allocator/backend padding",
            "quantization metadata", "hybrid/recurrent/compressed cache state",
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
