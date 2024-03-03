import argparse
from functools import partial
import numpy as np
import os

import torch
import torch.distributed as dist
import torch.nn as nn


from train import run_training
from utils import add_spaces


def init_process(local_rank, fn, args, backend="nccl"):
    """Initialize the distributed environment."""
    dist.init_process_group(backend, rank=local_rank)
    size = dist.get_world_size()
    return fn(local_rank, size, args)


def run_benchmark(args):
    ctx = torch.multiprocessing.get_context("spawn")
    fn = partial(
        init_process,
        fn=run_training,
        args=args
    )

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with ctx.Pool(processes=args.size) as pool:
        result = pool.starmap(fn, zip(range(args.size)))
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end), result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm_type", type=str, default="custom", help="Type of bnorm: `custom` or `lib`")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for each process")
    parser.add_argument("--grad_accum", type=int, default=1, help="Grad accumulation steps")
    parser.add_argument("--n_epoch", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--run_val", type=bool, default=False, help="Determines run validation epoch or no")
    parser.add_argument("--size", type=int, default=1, help="Number of workers")
    parser.add_argument("--backend", type=str, default="nccl", help="Backend for distributed processes")
    parser.add_argument("--device", type=str, default="cuda", help="Device `cuda` or `cpu`")
    args = parser.parse_args()

    time, result = run_benchmark(args)

    print(f"Training for BNorm type = {args.norm_type}; Grad Acc = {args.grad_accum}; Num epoch ")
    if args.run_val:
        train_acs, val_acs, mems = zip(*result)
        print(f"Train acc: "
              f"{np.mean(train_acs):.3f}.\n"
              f"Val acc: {np.mean(val_acs):.3f}.\n"
              f"Time: {time / 1000:.4f} s.\n"
              f"Memory: {np.sum(mems):.4f} Mb.\n")
    else:
        print("Skipping val metrics since validation epoch was disabled...\n")
        train_acs, mems = zip(*result)
        print(f"Train acc: "
              f"{np.mean(train_acs):.3f}.\n"
              f"Time: {time / 1000:.4f} s.\n"
              f"Memory: {np.sum(mems):.4f} Mb.\n")
