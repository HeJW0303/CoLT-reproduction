from __future__ import annotations

import os
import re
from typing import Any

import torch

from .base import BaseModel


class Qwen3VLChat(BaseModel):
    """Minimal Transformers adapter for the CoLT Qwen3-VL implementation."""

    INTERLEAVE = True
    VIDEO_LLM = True

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

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        if not torch.cuda.is_available():
            raise RuntimeError("CoLT evaluation requires a CUDA GPU.")
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
            raise RuntimeError(f"CoLT model was not loaded entirely on CUDA: {sorted(parameter_devices)}")

        print(
            "[CoLT eval adapter] "
            f"model={model_path} seed={self.seed} device={self.device} "
            f"visible_gpus={visible_gpus} caller_do_sample={do_sample} "
            f"caller_max_new_tokens={max_new_tokens}"
        )
        if getattr(self.model, "latent_reasoning_mode", False):
            print(
                "[CoLT eval adapter] The official CoLT generate() currently forces "
                "do_sample=True and max_new_tokens=256."
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
    def generate_inner(self, message, dataset=None):
        from qwen_vl_utils import process_vision_info

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self._prepare_content(message)})

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_k=self.top_k,
        )
        generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        if self.verbose:
            print(f"[CoLT raw response] {response}", flush=True)
        return self._extract_final_answer(response) if self.post_process else response
