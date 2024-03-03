import argparse
from functools import partial
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from syncbn import SyncBatchNorm
from utils import add_spaces

torch.set_num_threads(1)


def init_process(local_rank, fn, args, batch_size, hid_dim):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(args.backend, rank=local_rank, world_size=args.size)
    # size = dist.get_world_size()
    return fn(local_rank, args, batch_size, hid_dim)


def one_worker_bench(rank, args, batch_size, hid_dim):
    device = torch.device(rank)
    if args.norm_type == "lib":
        bn = nn.SyncBatchNorm(num_features=hid_dim, momentum=0.1, eps=1e-5, affine=False)
    else:
        bn = SyncBatchNorm(num_features=hid_dim)
    bn.to(device)
    inputs = torch.randn([batch_size, hid_dim], device=device, requires_grad=True)

    outputs = bn(inputs)
    loss = outputs.sum()
    loss.backward()

    return torch.cuda.max_memory_allocated(device)


def run_benchmark(args, batch_size, hid_dim):
    ctx = torch.multiprocessing.get_context("spawn")
    fn = partial(
        init_process,
        fn=one_worker_bench,
        args=args,
        batch_size=batch_size,
        hid_dim=hid_dim
    )

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with ctx.Pool(processes=args.size) as pool:
        memory_result = pool.starmap(fn, zip(range(args.size)))
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end), sum(memory_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1, help="Number of workers")
    parser.add_argument("--norm_type", type=str, default="custom", help="Type of bnorm: `custom` or `lib`")
    parser.add_argument("--backend", type=str, default="nccl", help="Backend for distributed processes")
    args = parser.parse_args()

    print(f"Started measuring for BatchNorm type = {args.norm_type}")
    print("| Hidden size | Batch size  | Time (s)    | Memory (Mb)")
    for hid_dim in [128, 256, 512, 1024]:
        for batch_size in [32, 64]:
            time, memory = run_benchmark(args, batch_size, hid_dim)
            time_str, memory_str = f'{time / 1000:.6f}', f'{memory / 2**20:.6f}'
            print(f"| {add_spaces(str(hid_dim))}|"
                  f" {add_spaces(str(batch_size))}|"
                  f" {add_spaces(time_str)}|"
                  f" {add_spaces(memory_str)}")
