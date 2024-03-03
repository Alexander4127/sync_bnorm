import os

import torch
import torch.distributed as dist


def run_sequential(rank, size, num_iter=10):
    """
    Prints the process rank sequentially according to its number over `num_iter` iterations,
    separating the output for each iteration by `---`
    Example (3 processes, num_iter=2):
    ```
    Process 0
    Process 1
    Process 2
    ---
    Process 0
    Process 1
    Process 2
    ```
    """
    tns = torch.zeros(1)
    for idx in range(num_iter):
        if (rank == 0 and idx > 0 and size > 1) or rank > 0:
            dist.recv(tensor=tns, src=(size + rank - 1) % size)

        if rank == 0 and idx > 0:
            print("---", flush=True)
        print(f"Process {rank}", flush=True)

        if size > 1 and not (idx == num_iter - 1 and rank == size - 1):
            dist.send(tensor=tns, dst=(rank + 1) % size)


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(rank=local_rank, backend="gloo")

    run_sequential(local_rank, dist.get_world_size())
