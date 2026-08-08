# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2026 CoLT contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import torch
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import PreTrainedModel

from ...utils.fsdp_utils import load_fsdp_model, offload_fsdp_model
from .base import BaseShardingManager


class FSDPTransformersShardingManager(BaseShardingManager):
    """Synchronize an FSDP actor into a per-rank Transformers rollout copy."""

    def __init__(
        self,
        module: FSDP,
        inference_model: PreTrainedModel,
        use_param_offload: bool,
        seed: int,
    ) -> None:
        self.module = module
        self.inference_model = inference_model
        self.use_param_offload = use_param_offload
        self.loaded = False
        self.freed_bytes = 0
        self.training_random_state = torch.cuda.get_rng_state()
        torch.cuda.manual_seed(seed + torch.distributed.get_rank())
        self.rollout_random_state = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(self.training_random_state)

    @torch.no_grad()
    def _sync_weights(self) -> None:
        if self.use_param_offload:
            load_fsdp_model(self.module)

        actor_state = get_model_state_dict(self.module)
        rollout_state = self.inference_model.state_dict(keep_vars=True)
        actor_keys = set(actor_state)
        rollout_keys = set(rollout_state)
        if actor_keys != rollout_keys:
            missing = sorted(rollout_keys - actor_keys)
            unexpected = sorted(actor_keys - rollout_keys)
            raise RuntimeError(
                "Actor/rollout state dictionaries differ: "
                f"missing_from_actor={missing[:20]}, unexpected_from_actor={unexpected[:20]}"
            )

        for name, sharded_value in actor_state.items():
            full_value = sharded_value.full_tensor() if isinstance(sharded_value, DTensor) else sharded_value
            target = rollout_state[name]
            target.copy_(full_value.to(device=target.device, dtype=target.dtype))

        del actor_state, rollout_state
        if self.use_param_offload:
            offload_fsdp_model(self.module)
        self.inference_model.eval()

    def prepare_rollout(self) -> None:
        if self.loaded:
            raise RuntimeError("Transformers rollout is already prepared.")
        self.loaded = True
        self._sync_weights()
        self.training_random_state = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(self.rollout_random_state)

    def release_rollout(self) -> None:
        if not self.loaded:
            raise RuntimeError("Transformers rollout is not prepared.")
        self.loaded = False
        self.rollout_random_state = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(self.training_random_state)
        self.module.train()
