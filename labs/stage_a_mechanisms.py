"""Deterministic teaching models, not vLLM execution or hardware measurements."""

import argparse
import json
import math


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def roofline(m, k, n, element_bytes, compute, bandwidth):
    flops = 2 * m * k * n
    traffic = element_bytes * (m * k + k * n + m * n)
    return {
        "flops": flops, "minimum_traffic_bytes": traffic,
        "arithmetic_intensity_flops_per_byte": flops / traffic,
        "compute_lower_bound_us": flops / (compute * 1e12) * 1e6,
        "memory_lower_bound_us": traffic / (bandwidth * 1e9) * 1e6,
        "assumptions": "one read of X/W and one write of Y; decimal TFLOP/s and GB/s",
    }


def schedule(prompts, outputs, budget):
    requests = [dict(id=chr(65+i), prompt=p, computed=0, produced=0)
                for i, p in enumerate(prompts)]
    trace = []
    while any(r["produced"] < outputs for r in requests):
        remaining = budget
        allocations = []
        for r in requests:
            if not remaining:
                break
            if r["produced"] == outputs:
                continue
            target = r["prompt"] + r["produced"]
            count = min(target - r["computed"], remaining)
            r["computed"] += count
            remaining -= count
            sampled = r["computed"] == target
            if sampled:
                r["produced"] += 1
            allocations.append(dict(request=r["id"], scheduled=count,
                                    computed=r["computed"], produced=r["produced"],
                                    sampled=sampled, finished=r["produced"] == outputs))
        trace.append(dict(step=len(trace)+1, allocations=allocations,
                          used=budget-remaining, remaining=remaining))
    work = sum(r["computed"] for r in requests)
    expected = sum(prompts) + len(prompts) * (outputs-1)
    if work != expected:
        raise RuntimeError("teaching trajectory violated work conservation")
    return dict(trace=trace, total_work_positions=work,
                expected_positions=expected,
                assumptions="A/B/C order; synchronous chunks; no KV limit, APC, async, spec, abort or preemption")


def pages(lengths, block, per_token, pool_blocks):
    counts = [(t + block - 1) // block for t in lengths]
    slots = sum(counts) * block
    return dict(lengths=lengths, blocks_per_request=counts,
                needed_blocks=sum(counts), slots=slots,
                logical_bytes=sum(lengths)*per_token,
                allocated_payload_bytes=slots*per_token,
                tail_bytes=(slots-sum(lengths))*per_token,
                pool_payload_bytes=pool_blocks*block*per_token,
                fits_pool=sum(counts) <= pool_blocks,
                mapping_example=dict(block_table=[7,2,9], position=9,
                                     block_size=4, physical_block=9, slot=37),
                assumptions="independent unshared requests; no alignment or lookahead; example mapping is separate")


def inputs(counts, computed):
    starts = [0]
    positions, req_indices = [], []
    for i, (q, c) in enumerate(zip(counts, computed)):
        starts.append(starts[-1]+q)
        positions.extend(range(c, c+q))
        req_indices.extend([i]*q)
    return dict(query_start_loc=starts, positions=positions,
                request_indices=req_indices, seq_lens=[c+q for c,q in zip(computed,counts)],
                last_indices=[end-1 for end in starts[1:]],
                assumptions="all requests reach sampling boundary; no spec, padding, CP or async")


def prefix():
    references = {"P1": 1, "P2": 1}
    history = [dict(event="A uses full prefix", refs=dict(references))]
    for event, delta in [("B touches prefix",1), ("A frees",-1), ("B frees",-1)]:
        references = {key:count+delta for key,count in references.items()}
        history.append(dict(event=event,refs=dict(references)))
    return dict(history=history, hashes_can_remain_at_zero=True,
                reallocating_requires_old_hash_invalidation=True,
                max_full_hit_examples=[dict(prompt=12,block=4,hit=8),dict(prompt=13,block=4,hit=12)],
                assumptions="full-prefix references only; not an implementation of vLLM hash/eviction policy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    r = commands.add_parser("roofline")
    for key, default in [("m",32),("k",4096),("n",4096),("element-bytes",2)]:
        r.add_argument("--"+key,type=positive,default=default)
    r.add_argument("--compute-tflops",type=positive_float,default=100)
    r.add_argument("--bandwidth-gbps",type=positive_float,default=1000)
    s = commands.add_parser("schedule")
    s.add_argument("--prompts",type=positive,nargs="+",default=[6,2,1])
    s.add_argument("--outputs",type=positive,default=3)
    s.add_argument("--budget",type=positive,default=4)
    p = commands.add_parser("pages")
    p.add_argument("--lengths",type=positive,nargs="+",default=[5,8,9])
    p.add_argument("--block-size",type=positive,default=4)
    p.add_argument("--bytes-per-token",type=positive,default=1024)
    p.add_argument("--pool-blocks",type=positive,default=10)
    q = commands.add_parser("inputs")
    q.add_argument("--counts",type=positive,nargs="+",default=[1,3,2])
    q.add_argument("--computed",type=int,nargs="+",default=[4,8,2])
    commands.add_parser("prefix")
    args = parser.parse_args()
    if args.command == "roofline":
        result = roofline(args.m,args.k,args.n,args.element_bytes,
                          args.compute_tflops,args.bandwidth_gbps)
    elif args.command == "schedule":
        if len(args.prompts) > 26 or sum(args.prompts)+len(args.prompts)*args.outputs > 100000:
            parser.error("teaching workload too large; use ≤26 requests and ≤100000 positions")
        result = schedule(args.prompts,args.outputs,args.budget)
    elif args.command == "pages":
        result = pages(args.lengths,args.block_size,args.bytes_per_token,args.pool_blocks)
    elif args.command == "inputs":
        if len(args.counts) != len(args.computed) or any(c < 0 for c in args.computed):
            parser.error("counts and nonnegative computed positions must have the same length")
        if sum(args.counts) > 100000:
            parser.error("teaching workload too large")
        result = inputs(args.counts,args.computed)
    else:
        result = prefix()
    print(json.dumps({"kind":"teaching_model", **result},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
