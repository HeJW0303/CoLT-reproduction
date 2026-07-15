from __future__ import annotations

import os

import torch

from .colt_qwen3_vl import Qwen3VLChat


class Qwen3VLBaseChat(Qwen3VLChat):
    """Qwen3-VL baseline adapter that cannot enter the CoLT latent path."""

    def __init__(self, *args, **kwargs) -> None:
        if os.environ.get("COLT_DISABLE_LATENT_REASONING", "0") != "1":
            raise RuntimeError(
                "Qwen3VLBaseChat requires COLT_DISABLE_LATENT_REASONING=1 before model construction."
            )
        super().__init__(*args, **kwargs)

        if getattr(self.model, "latent_reasoning_mode", None) is not False:
            raise RuntimeError("The base Qwen3-VL model did not disable latent reasoning.")
        forbidden_modules = (
            "decoder",
            "backward_decoder",
            "prj",
            "latent_predictor",
            "pj_in",
            "pj_back",
            "pj_out",
        )
        present = [name for name in forbidden_modules if hasattr(self.model, name)]
        if present:
            raise RuntimeError(f"The base Qwen3-VL model unexpectedly created CoLT modules: {present}")

        print(
            "[Qwen3-VL baseline] "
            f"mode=native-textual-cot do_sample={self.do_sample} "
            f"max_new_tokens={self.max_new_tokens} CoLT_modules=absent"
        )

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

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            generation_kwargs.update(temperature=self.temperature, top_k=self.top_k)
        generated_ids = self.model.generate(**inputs, **generation_kwargs)
        generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        if self.verbose:
            print(f"[Qwen3-VL baseline raw response] {response}", flush=True)
        return self._extract_final_answer(response) if self.post_process else response
