# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2026 CoLT contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Any, Optional, Union

import numpy as np
import torch
from tensordict import TensorDict
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    PreTrainedTokenizer,
    ProcessorMixin,
)

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image, process_video
from ...utils.torch_dtypes import PrecisionType
from .base import BaseRollout
from .config import RolloutConfig


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int):
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    return np.repeat(value, repeats, axis=0)


def _trim_left_prompt_padding(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    response_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep rollout teacher-forcing shape identical to the batch-1 actor path."""
    prompt_length = input_ids.shape[1] - response_length
    if prompt_length <= 0:
        raise ValueError(
            "CoLT response length must be smaller than the input sequence length: "
            f"input_length={input_ids.shape[1]}, response_length={response_length}."
        )

    valid_prompt_lengths = attention_mask[:, :prompt_length].sum(dim=-1)
    if not torch.all(valid_prompt_lengths == valid_prompt_lengths[0]):
        return input_ids, attention_mask, position_ids

    valid_prompt_length = int(valid_prompt_lengths[0].item())
    left_padding = prompt_length - valid_prompt_length
    if left_padding <= 0:
        return input_ids, attention_mask, position_ids

    return (
        input_ids[:, left_padding:],
        attention_mask[:, left_padding:],
        position_ids[..., left_padding:],
    )


def _repeat_cached_latent_outputs(latent_outputs, repeats: int):
    """Clone a batch-1 latent prefix for an optional response-only batch."""
    repeated_outputs = copy.deepcopy(latent_outputs)
    if repeated_outputs.hidden_states is not None:
        repeated_outputs.hidden_states = tuple(
            hidden_state.repeat_interleave(repeats, dim=0)
            if isinstance(hidden_state, torch.Tensor)
            else hidden_state
            for hidden_state in repeated_outputs.hidden_states
        )
    repeated_outputs.past_key_values.batch_repeat_interleave(repeats)
    return repeated_outputs


class CoLTTransformersRollout(BaseRollout):
    """Correctness-first CoLT rollout using the vendored Transformers model."""

    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
    ) -> None:
        super().__init__()
        if config.tensor_parallel_size != 1:
            raise ValueError("CoLTTransformersRollout requires tensor_parallel_size=1.")
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            raise ValueError("CoLT Transformers rollout requires tokenizer.pad_token_id.")

        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=config.trust_remote_code)
        if type(model_config) in AutoModelForImageTextToText._model_mapping.keys():
            auto_class = AutoModelForImageTextToText
        else:
            auto_class = AutoModelForCausalLM
        dtype = PrecisionType.to_dtype(config.dtype)
        self.inference_model = auto_class.from_pretrained(
            model_path,
            config=model_config,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
            trust_remote_code=config.trust_remote_code,
        ).to(torch.cuda.current_device(), dtype=dtype)
        rollout_parameter_dtype = next(self.inference_model.parameters()).dtype
        if rollout_parameter_dtype != dtype:
            raise RuntimeError(
                "CoLT rollout model dtype does not match rollout configuration: "
                f"expected={dtype}, actual={rollout_parameter_dtype}."
            )
        self.inference_model.requires_grad_(False)
        self.inference_model.eval()
        if not hasattr(self.inference_model, "latent_reasoning_generate"):
            raise TypeError(
                "Loaded rollout model does not expose latent_reasoning_generate; ensure the vendored CoLT Transformers "
                "source is first on PYTHONPATH."
            )
        if not getattr(self.inference_model, "latent_reasoning_mode", False):
            raise ValueError("Loaded rollout model has latent reasoning disabled.")

        self.sampling_params = {
            "n": config.n,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
        }

    @contextmanager
    def update_sampling_params(self, **kwargs):
        old_values = {}
        for key, value in kwargs.items():
            if key in self.sampling_params:
                old_values[key] = self.sampling_params[key]
                self.sampling_params[key] = value
        try:
            yield
        finally:
            self.sampling_params.update(old_values)

    def _prepare_multimodal_inputs(
        self,
        multi_modal_data: Optional[dict[str, Any]],
        meta_info: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        if multi_modal_data is None:
            return {}
        if self.processor is None:
            raise ValueError("Multimodal rollout input requires a processor.")

        images = [
            process_image(image, meta_info["min_pixels"], meta_info["max_pixels"])
            for image in multi_modal_data.get("images", [])
        ]
        videos = [process_video(video) for video in multi_modal_data.get("videos", [])]
        if images and videos:
            raise ValueError("A single rollout sample cannot contain both images and videos.")
        if images:
            values = dict(self.processor.image_processor(images=images, return_tensors="pt"))
        elif videos:
            values = dict(self.processor.image_processor(images=None, videos=videos, return_tensors="pt"))
        else:
            return {}
        device = torch.cuda.current_device()
        return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in values.items()}

    def _prepare_batched_multimodal_inputs(
        self,
        batch_multi_modal_data: list[Optional[dict[str, Any]]],
        meta_info: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        batched_values: dict[str, list[torch.Tensor]] = {}
        for multi_modal_data in batch_multi_modal_data:
            for key, value in self._prepare_multimodal_inputs(multi_modal_data, meta_info).items():
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"CoLT multimodal rollout input {key!r} must be a tensor.")
                batched_values.setdefault(key, []).append(value)
        return {key: torch.cat(values, dim=0) for key, values in batched_values.items()}

    @torch.no_grad()
    def _generate_model_batch(
        self,
        prompt_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        prompt_position_ids: torch.Tensor,
        batch_multi_modal_data: list[Optional[dict[str, Any]]],
        meta_info: dict[str, Any],
        eos_token_id: int,
        colt_cached_latent_outputs=None,
    ) -> torch.Tensor:
        if prompt_ids.ndim != 2 or prompt_attention_mask.ndim != 2:
            raise ValueError("CoLT batched generation expects rank-2 token and attention tensors.")
        if prompt_position_ids.ndim == 3:
            prompt_position_ids = prompt_position_ids.transpose(0, 1)
        if prompt_position_ids.ndim not in {2, 3}:
            raise ValueError("CoLT batched generation expects rank-2 or rank-3 position_ids.")
        if prompt_ids.shape[0] != len(batch_multi_modal_data):
            raise ValueError(
                "CoLT generation multimodal batch must align with prompt batch: "
                f"prompts={prompt_ids.shape[0]}, multimodal={len(batch_multi_modal_data)}."
            )

        device = torch.cuda.current_device()
        model_inputs = (
            {}
            if colt_cached_latent_outputs is not None
            else self._prepare_batched_multimodal_inputs(batch_multi_modal_data, meta_info)
        )
        temperature = float(self.sampling_params["temperature"])
        generated = self.inference_model.latent_reasoning_generate(
            input_ids=prompt_ids.to(device),
            attention_mask=prompt_attention_mask.to(device),
            position_ids=prompt_position_ids.to(device),
            num_hidden_generations=self.config.num_hidden_generations,
            max_new_tokens=self.config.response_length,
            do_sample=temperature > 0.0,
            temperature=max(temperature, 1e-5),
            top_p=float(self.sampling_params["top_p"]),
            top_k=int(self.sampling_params["top_k"]),
            eos_token_id=eos_token_id,
            colt_cached_latent_outputs=colt_cached_latent_outputs,
            **model_inputs,
        )
        return generated[:, prompt_ids.shape[1] :].detach().cpu()

    @torch.no_grad()
    def _generate_batch(
        self,
        prompt_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        prompt_position_ids: torch.Tensor,
        multi_modal_data: Optional[dict[str, Any]],
        meta_info: dict[str, Any],
        eos_token_id: int,
        batch_size: int,
    ) -> list[list[int]]:
        if batch_size <= 0:
            raise ValueError(f"CoLT generation batch_size must be positive, received {batch_size}.")

        prompt_ids = prompt_ids.unsqueeze(0).expand(batch_size, -1).contiguous()
        prompt_attention_mask = prompt_attention_mask.unsqueeze(0).expand(batch_size, -1).contiguous()
        if prompt_position_ids.ndim == 2 and prompt_position_ids.shape[0] == 4:
            prompt_position_ids = prompt_position_ids.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        else:
            prompt_position_ids = prompt_position_ids.unsqueeze(0).expand(batch_size, -1).contiguous()
        return self._generate_model_batch(
            prompt_ids=prompt_ids,
            prompt_attention_mask=prompt_attention_mask,
            prompt_position_ids=prompt_position_ids,
            batch_multi_modal_data=[multi_modal_data] * batch_size,
            meta_info=meta_info,
            eos_token_id=eos_token_id,
        ).tolist()

    def _generate_one(
        self,
        prompt_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        prompt_position_ids: torch.Tensor,
        multi_modal_data: Optional[dict[str, Any]],
        meta_info: dict[str, Any],
        eos_token_id: int,
    ) -> list[int]:
        return self._generate_batch(
            prompt_ids,
            prompt_attention_mask,
            prompt_position_ids,
            multi_modal_data,
            meta_info,
            eos_token_id,
            batch_size=1,
        )[0]

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        input_ids = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_ids = prompts.meta_info["eos_token_id"]
        generation_eos = eos_ids[0] if isinstance(eos_ids, (list, tuple)) else eos_ids
        batch_multi_modal_data = prompts.non_tensor_batch.get("multi_modal_data")

        response_ids = []
        repeated_multi_modal_data = []
        effective_n = None
        with self.update_sampling_params(**prompts.meta_info):
            n = int(self.sampling_params["n"])
            effective_n = n
            generation_batch_size = min(int(self.config.generation_batch_size), n)
            prompt_batch_size = min(int(self.config.generation_prompt_batch_size), input_ids.shape[0])
            for prompt_start in range(0, input_ids.shape[0], prompt_batch_size):
                prompt_end = min(prompt_start + prompt_batch_size, input_ids.shape[0])
                prompt_ids = input_ids[prompt_start:prompt_end]
                prompt_attention_mask = attention_mask[prompt_start:prompt_end]
                prompt_position_ids = position_ids[prompt_start:prompt_end]
                prompt_multi_modal_data = (
                    [None] * (prompt_end - prompt_start)
                    if batch_multi_modal_data is None
                    else list(batch_multi_modal_data[prompt_start:prompt_end])
                )
                for start in range(0, n, generation_batch_size):
                    current_batch_size = min(generation_batch_size, n - start)
                    batched_prompt_ids = prompt_ids.repeat_interleave(current_batch_size, dim=0)
                    batched_attention_mask = prompt_attention_mask.repeat_interleave(current_batch_size, dim=0)
                    batched_position_ids = prompt_position_ids.repeat_interleave(current_batch_size, dim=0)
                    batched_multi_modal_data = [
                        multi_modal_data
                        for multi_modal_data in prompt_multi_modal_data
                        for _ in range(current_batch_size)
                    ]
                    cached_latent_outputs = None
                    if prompt_end - prompt_start > 1:
                        prefix_position_ids = prompt_position_ids
                        if prefix_position_ids.ndim == 3:
                            prefix_position_ids = prefix_position_ids.transpose(0, 1)
                        prefix_model_inputs = self._prepare_batched_multimodal_inputs(
                            prompt_multi_modal_data, prompts.meta_info
                        )
                        cached_latent_outputs = self.inference_model._forward_latent_reasoning(
                            input_ids=prompt_ids.to(torch.cuda.current_device()),
                            attention_mask=prompt_attention_mask.to(torch.cuda.current_device()),
                            position_ids=prefix_position_ids.to(torch.cuda.current_device()),
                            num_hidden_generations=self.config.num_hidden_generations,
                            **prefix_model_inputs,
                        )
                        cached_latent_outputs = _repeat_cached_latent_outputs(
                            cached_latent_outputs, current_batch_size
                        )
                    generated_batch = self._generate_model_batch(
                        prompt_ids=batched_prompt_ids,
                        prompt_attention_mask=batched_attention_mask,
                        prompt_position_ids=batched_position_ids,
                        batch_multi_modal_data=batched_multi_modal_data,
                        meta_info=prompts.meta_info,
                        eos_token_id=generation_eos,
                        colt_cached_latent_outputs=cached_latent_outputs,
                    ).tolist()
                    response_ids.extend(generated_batch)
                    repeated_multi_modal_data.extend(batched_multi_modal_data)

        response_ids = VF.pad_2d_list_to_length(
            response_ids,
            self.pad_token_id,
            max_length=self.config.response_length,
        ).to(input_ids.device)
        n = effective_n
        if n > 1:
            input_ids = _repeat_interleave(input_ids, n)
            attention_mask = _repeat_interleave(attention_mask, n)
            position_ids = _repeat_interleave(position_ids, n)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.shape[1]
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(input_ids.shape[0], -1)
        if position_ids.ndim == 3:
            delta_position_id = delta_position_id.view(input_ids.shape[0], 1, -1).expand(
                input_ids.shape[0], position_ids.shape[1], -1
            )
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(response_ids, eos_token_id=eos_ids, dtype=attention_mask.dtype)
        attention_mask = torch.cat([attention_mask, response_mask], dim=-1)
        # CoLT teacher-forced scoring remains batch-1 because the actor parity
        # contract requires the same batch shape. Crop only right-padded
        # response positions; invalid positions are restored as zero log-probs.
        scoring_device = torch.cuda.current_device()
        rollout_log_prob_rows = []
        for row in range(sequence_ids.shape[0]):
            effective_response_length = max(int(response_mask[row].sum().item()), 1)
            row_model_inputs = (
                self._prepare_multimodal_inputs(repeated_multi_modal_data[row], prompts.meta_info)
                if repeated_multi_modal_data
                else {}
            )
            if position_ids.ndim == 3:
                row_position_ids = position_ids[row : row + 1, ..., : input_ids.shape[1] + effective_response_length]
            else:
                row_position_ids = position_ids[row : row + 1, : input_ids.shape[1] + effective_response_length]
            row_input_ids, row_attention_mask, row_position_ids = _trim_left_prompt_padding(
                input_ids=sequence_ids[row : row + 1, : input_ids.shape[1] + effective_response_length],
                attention_mask=attention_mask[row : row + 1, : input_ids.shape[1] + effective_response_length],
                position_ids=row_position_ids,
                response_length=effective_response_length,
            )
            if row_position_ids.ndim == 3:
                row_position_ids = row_position_ids.transpose(0, 1)
            scoring_output = self.inference_model(
                input_ids=row_input_ids.to(scoring_device),
                attention_mask=row_attention_mask.to(scoring_device),
                position_ids=row_position_ids.to(scoring_device),
                colt_rl_response_length=effective_response_length,
                num_hidden_generations=self.config.num_hidden_generations,
                **row_model_inputs,
            )
            policy_logits = scoring_output.logits.float() / max(float(self.sampling_params["temperature"]), 1e-5)
            row_log_probs = torch.log_softmax(policy_logits, dim=-1).gather(
                dim=-1,
                index=response_ids[row : row + 1, :effective_response_length].to(scoring_device).unsqueeze(-1),
            ).squeeze(-1)
            if effective_response_length < response_length:
                row_log_probs = torch.nn.functional.pad(
                    row_log_probs, (0, response_length - effective_response_length)
                )
            rollout_log_prob_rows.append(row_log_probs.to(input_ids.device))
        rollout_log_probs = torch.cat(rollout_log_prob_rows, dim=0)
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "rollout_log_probs": rollout_log_probs,
                "position_ids": position_ids,
            },
            batch_size=input_ids.shape[0],
        )
        non_tensor_batch = {}
        if batch_multi_modal_data is not None:
            non_tensor_batch["multi_modal_data"] = np.array(repeated_multi_modal_data, dtype=object)
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)
