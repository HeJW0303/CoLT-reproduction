from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from typing import Any

import torch

from .base import BaseModel


class Qwen3VLChat(BaseModel):
    """Minimal Transformers adapter for the CoLT Qwen3-VL implementation."""

    INTERLEAVE = True
    VIDEO_LLM = True
    VISION_PREPROCESS_PROFILES = {
        "legacy14_processor_resize": (14, True),
        "model_patch_processor_resize": (None, True),
        "model_patch_no_processor_resize": (None, False),
    }

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_new_tokens: int = 8192,
        do_sample: bool = False,
        temperature: float = 0.6,
        top_k: int = 20,
        use_custom_prompt: bool = False,
        system_prompt: str | None = None,
        post_process: bool = True,
        verbose: bool = False,
        use_vllm: bool = False,
        seed: int | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        if use_vllm:
            raise ValueError("CoLT latent generation is not supported by vLLM in this repository.")

        self.model_path = model_path
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self._use_custom_prompt = use_custom_prompt
        self.system_prompt = system_prompt
        self.post_process = post_process
        self.verbose = verbose
        self.seed = int(os.environ.get("COLT_EVAL_SEED", seed if seed is not None else 1234))
        self.reseed_per_sample = os.environ.get("COLT_RESEED_PER_SAMPLE", "0") == "1"
        self.vision_preprocess_profile = os.environ.get(
            "COLT_VISION_PREPROCESS_PROFILE",
            "legacy14_processor_resize",
        )
        if self.vision_preprocess_profile not in self.VISION_PREPROCESS_PROFILES:
            supported = ", ".join(sorted(self.VISION_PREPROCESS_PROFILES))
            raise ValueError(
                "Unsupported COLT_VISION_PREPROCESS_PROFILE="
                f"{self.vision_preprocess_profile!r}; expected one of: {supported}"
            )

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3-VL evaluation requires a CUDA GPU.")
        visible_gpus = torch.cuda.device_count()
        if visible_gpus != 1:
            raise RuntimeError(
                f"Each CoLT evaluation worker must see exactly one CUDA GPU, found {visible_gpus}."
            )

        # Each evaluation process sees exactly one physical GPU through
        # CUDA_VISIBLE_DEVICES. An explicit map avoids Accelerate's automatic
        # tied-parameter partitioner, which cannot analyze CoLT's nested models.
        self.device = torch.device("cuda:0")

        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self._processor_lock = threading.Lock()
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model.eval()
        parameter_devices = {parameter.device.type for parameter in self.model.parameters()}
        if parameter_devices != {"cuda"}:
            raise RuntimeError(f"Qwen3-VL model was not loaded entirely on CUDA: {sorted(parameter_devices)}")

        processor_patch_size = int(self.processor.image_processor.patch_size)
        processor_merge_size = int(self.processor.image_processor.merge_size)
        if processor_patch_size != 16 or processor_merge_size != 2:
            raise RuntimeError(
                "The loaded Qwen3-VL processor must use patch_size=16 and merge_size=2, "
                f"but reports patch_size={processor_patch_size}, merge_size={processor_merge_size}."
            )
        profile_patch_size, self.processor_do_resize = self.VISION_PREPROCESS_PROFILES[
            self.vision_preprocess_profile
        ]
        self.vision_patch_size = processor_patch_size if profile_patch_size is None else profile_patch_size
        if self.vision_preprocess_profile != "legacy14_processor_resize" and self.vision_patch_size != 16:
            raise RuntimeError(
                "The corrected Qwen3-VL profiles require image patch size 16, "
                f"but the loaded processor reports {processor_patch_size}."
            )

        adapter_name = "CoLT" if getattr(self.model, "latent_reasoning_mode", False) else "Qwen3-VL baseline"
        print(
            f"[{adapter_name} eval adapter] "
            f"model={model_path} seed={self.seed} device={self.device} "
            f"visible_gpus={visible_gpus} caller_do_sample={do_sample} "
            f"caller_max_new_tokens={max_new_tokens} "
            f"vision_profile={self.vision_preprocess_profile} "
            f"qwen_vl_utils_patch_size={self.vision_patch_size} "
            f"processor_patch_size={processor_patch_size} "
            f"processor_merge_size={processor_merge_size} "
            f"processor_do_resize={self.processor_do_resize} "
            f"reseed_per_sample={self.reseed_per_sample}"
        )
        if getattr(self.model, "latent_reasoning_mode", False):
            respect_generation_args = os.environ.get("COLT_RESPECT_GENERATION_ARGS", "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            print(
                "[CoLT generation config] "
                f"requested_do_sample={self.do_sample} "
                f"effective_do_sample={self.do_sample if respect_generation_args else True} "
                f"requested_max_new_tokens={self.max_new_tokens} "
                f"effective_max_new_tokens={self.max_new_tokens if respect_generation_args else 256} "
                f"respect_generation_args={respect_generation_args}"
            )
        else:
            print("[Qwen3-VL eval adapter] Native Hugging Face generation is active; latent reasoning is disabled.")

    def use_custom_prompt(self, dataset: str) -> bool:
        return self._use_custom_prompt

    def build_prompt(self, line, dataset: str):
        raise NotImplementedError("This adapter uses VLMEvalKit dataset prompts.")

    def _prepare_content(self, message: list[dict[str, Any]]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for item in message:
            kind = item["type"]
            value = item["value"]
            if kind == "text":
                content.append({"type": "text", "text": value})
            elif kind == "image":
                image = {"type": "image", "image": value}
                if self.min_pixels is not None:
                    image["min_pixels"] = self.min_pixels
                if self.max_pixels is not None:
                    image["max_pixels"] = self.max_pixels
                content.append(image)
            elif kind == "video":
                content.append({"type": "video", "video": value})
            else:
                raise ValueError(f"Unsupported message type: {kind}")
        return content

    def _prepare_model_inputs(self, messages):
        from qwen_vl_utils import process_vision_info

        with self._processor_lock:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        image_inputs, video_inputs = process_vision_info(
            messages,
            image_patch_size=self.vision_patch_size,
        )
        processor_kwargs: dict[str, Any] = {
            "text": [text],
            "images": image_inputs,
            "videos": video_inputs,
            "padding": True,
            "return_tensors": "pt",
        }
        if not self.processor_do_resize:
            processor_kwargs["images_kwargs"] = {"do_resize": False}
            if video_inputs is not None:
                processor_kwargs["videos_kwargs"] = {"do_resize": False}
        with self._processor_lock:
            return self.processor(**processor_kwargs)

    def _prepare_preprocessed_request(self, message, dataset=None):
        assert message is not None and self.check_content(message) == "listdict"
        for item in message:
            assert item["type"] in self.allowed_types

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self._prepare_content(message)})
        return {
            "dataset": dataset,
            "messages": messages,
            "inputs": self._prepare_model_inputs(messages),
        }

    def prepare_request(self, message, dataset=None):
        """Build CPU model inputs so the caller can overlap preprocessing with CUDA generation."""
        assert self.check_content(message) in ["str", "dict", "liststr", "listdict"]
        return self._prepare_preprocessed_request(self.preproc_content(message), dataset)

    def _reseed_sample(self, messages, dataset):
        if not self.reseed_per_sample:
            return
        payload = json.dumps(
            {"dataset": dataset, "messages": messages},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        sample_seed = (self.seed + int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")) % (2**63 - 1)
        torch.manual_seed(sample_seed)
        torch.cuda.manual_seed_all(sample_seed)

    @staticmethod
    def _extract_final_answer(response: str) -> str:
        matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", response, flags=re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[-1].strip()

        boxed = response.rfind("\\boxed{")
        if boxed >= 0:
            start = boxed + len("\\boxed{")
            depth = 1
            for pos in range(start, len(response)):
                if response[pos] == "{":
                    depth += 1
                elif response[pos] == "}":
                    depth -= 1
                    if depth == 0:
                        return response[start:pos].strip()
        return response.strip()

    @torch.inference_mode()
    def _generate_prepared_token_ids(self, prepared):
        messages = prepared["messages"]
        self._reseed_sample(messages, prepared["dataset"])
        inputs = prepared["inputs"].to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_k=self.top_k,
        )
        if os.environ.get("COLT_LOG_PREDICTED_K", "0") == "1":
            predicted_k = getattr(self.model, "last_oracle_k_prediction", None)
            used_k = getattr(self.model, "last_oracle_k_used", None)
            if predicted_k is not None:
                print(
                    f"[CoLT Oracle-K] predicted_k={predicted_k.reshape(-1).tolist()} used_k={used_k}",
                    flush=True,
                )
        return generated_ids[:, inputs.input_ids.shape[1] :]

    def _decode_generated_ids(self, generated_ids, *, skip_special_tokens):
        with self._processor_lock:
            return self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=False,
            )[0]

    @torch.inference_mode()
    def generate_prepared(self, prepared):
        generated_ids = self._generate_prepared_token_ids(prepared)
        response = self._decode_generated_ids(generated_ids, skip_special_tokens=True)
        if self.verbose:
            print(f"[CoLT raw response] {response}", flush=True)
        return self._extract_final_answer(response) if self.post_process else response

    @torch.inference_mode()
    def diagnose_prepared(self, prepared):
        """Generate once and retain token-level evidence for empty-response audits."""
        generated_ids = self._generate_prepared_token_ids(prepared)
        clean_response = self._decode_generated_ids(generated_ids, skip_special_tokens=True)
        raw_response = self._decode_generated_ids(generated_ids, skip_special_tokens=False)
        final_response = self._extract_final_answer(clean_response) if self.post_process else clean_response
        return {
            "token_ids": generated_ids[0].detach().cpu().tolist(),
            "raw_response": raw_response,
            "clean_response": clean_response,
            "final_response": final_response,
            "model_eos_token_id": getattr(self.model, "eos_token_id", None),
            "tokenizer_eos_token_id": self.processor.tokenizer.eos_token_id,
            "tokenizer_pad_token_id": self.processor.tokenizer.pad_token_id,
        }

    def generate_inner(self, message, dataset=None):
        return self.generate_prepared(self._prepare_preprocessed_request(message, dataset))
