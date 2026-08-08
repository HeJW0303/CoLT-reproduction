#!/usr/bin/env python3
"""Exercise the CoLT FSDP-to-Transformers weight synchronizer on multiple GPUs."""

from __future__ import annotations

import os
import sys
import types

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def _provide_codetiming_stub_if_missing() -> None:
    try:
        import codetiming  # noqa: F401
    except ImportError:
        module = types.ModuleType("codetiming")

        class Timer:
            def __init__(self, *args, **kwargs):
                del args, kwargs

        module.Timer = Timer
        sys.modules["codetiming"] = module


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("FSDP sync check requires CUDA.")
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    _provide_codetiming_stub_if_missing()
    from verl.workers.sharding_manager.fsdp_transformers import FSDPTransformersShardingManager

    torch.manual_seed(1729)
    source = nn.Sequential(nn.Linear(7, 11), nn.GELU(), nn.Linear(11, 5)).cuda()
    expected = {name: value.detach().clone() for name, value in source.state_dict().items()}
    device_mesh = init_device_mesh("cuda", mesh_shape=(dist.get_world_size(),), mesh_dim_names=("fsdp",))
    actor = FSDP(
        source,
        device_id=torch.cuda.current_device(),
        use_orig_params=False,
        device_mesh=device_mesh,
    )
    rollout = nn.Sequential(nn.Linear(7, 11), nn.GELU(), nn.Linear(11, 5)).cuda()
    for parameter in rollout.parameters():
        parameter.data.zero_()

    manager = FSDPTransformersShardingManager(
        module=actor,
        inference_model=rollout,
        use_param_offload=False,
        seed=2026,
    )
    manager.prepare_rollout()
    for name, actual in rollout.state_dict().items():
        torch.testing.assert_close(actual, expected[name], rtol=0.0, atol=0.0)
    manager.release_rollout()
    if manager.loaded:
        raise RuntimeError("FSDP Transformers manager remained loaded after release.")

    dist.barrier()
    if dist.get_rank() == 0:
        print(f"FSDP Transformers sync passed on {dist.get_world_size()} ranks.")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
