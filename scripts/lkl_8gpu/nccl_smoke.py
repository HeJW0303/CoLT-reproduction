#!/usr/bin/env python3

import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    value = torch.tensor([float(rank)], device=f"cuda:{local_rank}")
    dist.all_reduce(value)
    expected = world_size * (world_size - 1) / 2
    if value.item() != expected:
        raise RuntimeError(f"rank {rank}: expected {expected}, got {value.item()}")
    dist.barrier()
    print(f"rank={rank} local_rank={local_rank} device={torch.cuda.get_device_name(local_rank)} NCCL_OK")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
