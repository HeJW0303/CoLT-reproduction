from __future__ import annotations

import os
from pathlib import Path

import torch

from .colt_qwen3_vl import Qwen3VLChat


class Qwen3VLBaseChat(Qwen3VLChat):
    """Qwen3-VL baseline adapter that cannot enter the CoLT latent path."""

    def __init__(self, *args, **kwargs) -> None:
        if os.environ.get("COLT_DISABLE_LATENT_REASONING", "0") != "1":
            raise RuntimeError(
                "Qwen3VLBaseChat requires COLT_DISABLE_LATENT_REASONING=1 before model construction."
            )
        expected_model_path = os.environ.get("QWEN3_VL_BASE_MODEL_PATH")
        requested_model_path = kwargs.get("model_path")
        if expected_model_path is None:
            raise RuntimeError("Qwen3VLBaseChat requires QWEN3_VL_BASE_MODEL_PATH to be set.")
        if requested_model_path is not None and Path(requested_model_path).resolve() != Path(expected_model_path).resolve():
            raise RuntimeError(
                "Qwen3VLBaseChat was asked to load a model other than QWEN3_VL_BASE_MODEL_PATH; "
                f"requested={requested_model_path!r}, expected={expected_model_path!r}."
            )
        super().__init__(*args, **kwargs)

        if Path(self.model_path).resolve() != Path(expected_model_path).resolve():
            raise RuntimeError(
                "Qwen3VLBaseChat must load exactly QWEN3_VL_BASE_MODEL_PATH; "
                f"model_path={self.model_path!r}, expected={expected_model_path!r}."
            )
        if getattr(self.model, "latent_reasoning_mode", None) is not False:
            raise RuntimeError("The base Qwen3-VL model did not disable latent reasoning.")
        forbidden_attributes = (
            "decoder",
            "backward_decoder",
            "prj",
            "latent_predictor",
            "pj_in",
            "pj_back",
            "pj_out",
            "alpha",
            "latent_to_decoder_scale",
            "num_latent",
            "cot_boundary_token_ids",
        )
        present = [name for name in forbidden_attributes if hasattr(self.model, name)]
        if present:
            raise RuntimeError(f"The base Qwen3-VL model unexpectedly created CoLT attributes: {present}")

        print(
            "[Qwen3-VL baseline] "
            f"mode=native-textual-cot do_sample={self.do_sample} "
            f"max_new_tokens={self.max_new_tokens} CoLT_modules=absent"
        )

    @torch.inference_mode()
    def generate_inner(self, message, dataset=None):
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self._prepare_content(message)})
        self._reseed_sample(messages, dataset)

        inputs = self._prepare_model_inputs(messages)

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
