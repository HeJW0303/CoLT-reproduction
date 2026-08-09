from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    _extend_colt_cached_attention_mask,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EASYR1_ROOT = REPO_ROOT / "EasyR1"


class FakeCache:
    def __init__(self, length: int):
        self.length = length

    def get_seq_length(self) -> int:
        return self.length

    def batch_repeat_interleave(self, repeats: int) -> None:
        del repeats


class FakeBackboneOutput:
    def __init__(self, hidden_states: torch.Tensor, cache: FakeCache):
        self.last_hidden_state = hidden_states
        self.past_key_values = cache
        self.hidden_states = (hidden_states,)
        self.rope_deltas = None

    def __getitem__(self, index: int) -> torch.Tensor:
        if index != 0:
            raise IndexError(index)
        return self.last_hidden_state


class FakeBackbone:
    def __init__(self, embedding: torch.nn.Embedding):
        self.embedding = embedding
        self.calls: list[dict[str, torch.Tensor | None]] = []

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.embedding

    def __call__(self, input_ids=None, inputs_embeds=None, past_key_values=None, **kwargs):
        attention_mask = kwargs.get("attention_mask")
        self.calls.append(
            {"attention_mask": None if attention_mask is None else attention_mask.detach().clone()}
        )
        del kwargs
        hidden_states = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        cache = past_key_values or FakeCache(0)
        cache.length += hidden_states.shape[1]
        return FakeBackboneOutput(hidden_states, cache)


class FakeCoLT:
    _forward_latent_response = Qwen3VLForConditionalGeneration._forward_latent_response
    latent_reasoning_generate = Qwen3VLForConditionalGeneration.latent_reasoning_generate

    def __init__(self):
        self.embedding = torch.nn.Embedding(5, 5)
        with torch.no_grad():
            self.embedding.weight.zero_()
            self.embedding.weight[3, 2] = 8.0
            self.embedding.weight[2, 1] = 7.0
            self.embedding.weight[1, 0] = 6.0
        self.model = FakeBackbone(self.embedding)
        self.lm_head = torch.nn.Identity()
        self.initial_latent = torch.tensor([[[0.0, 0.0, 0.0, 10.0, 0.0]]], requires_grad=True)
        self.eos_token_id = 4
        self.tokenizer = SimpleNamespace(all_special_ids=[], decode=lambda *args, **kwargs: "visible")
        self.latent_batch_sizes = []

    def _forward_latent_reasoning(self, input_ids, **kwargs):
        del kwargs
        self.latent_batch_sizes.append(input_ids.shape[0])
        return SimpleNamespace(
            hidden_states=(self.initial_latent,),
            past_key_values=FakeCache(input_ids.shape[1] + 3),
            k_logits=None,
            predicted_k=None,
        )


class FakeLatentPrefixCoLT:
    _forward_latent_reasoning = Qwen3VLForConditionalGeneration._forward_latent_reasoning
    _pool_question_hidden = Qwen3VLForConditionalGeneration._pool_question_hidden
    _resolve_oracle_k_inference_plan = Qwen3VLForConditionalGeneration._resolve_oracle_k_inference_plan

    def __init__(self):
        self.embedding = torch.nn.Embedding(8, 3)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(24, dtype=torch.float32).view(8, 3))
        self.model = FakeBackbone(self.embedding)
        self.config = SimpleNamespace(text_config=SimpleNamespace())
        self.prj = torch.nn.Identity()
        self.lm_head = torch.nn.Identity()
        self.inference_latent_transition = "official"
        self.oracle_k_enabled = False
        self.num_latent = 0
        self.last_oracle_k_prediction = None
        self.last_oracle_k_used = None
        self.last_oracle_k_conditioning = None

    def _predict_oracle_k(self, hidden_states, attention_mask):
        del hidden_states, attention_mask
        return None


def load_colt_reward_module():
    fake_reward = types.ModuleType("verl.reward_function.onethinker_reward")
    fake_reward.extract_answer = lambda text: text[8:-9] if text.startswith("<answer>") else None
    fake_reward.accuracy_reward = lambda response, ground_truth, *_: float(
        response == f"<answer>{ground_truth}</answer>"
    )
    fake_reward.answer_structure_bonus = lambda *_: 0.5
    previous = sys.modules.get("verl.reward_function.onethinker_reward")
    sys.modules["verl.reward_function.onethinker_reward"] = fake_reward
    try:
        path = EASYR1_ROOT / "verl/reward_function/colt_outcome.py"
        spec = importlib.util.spec_from_file_location("colt_outcome_test_module", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            sys.modules.pop("verl.reward_function.onethinker_reward", None)
        else:
            sys.modules["verl.reward_function.onethinker_reward"] = previous


class CoLTEasyR1ContractTests(unittest.TestCase):
    def test_latent_prefix_uses_last_valid_token_for_mixed_padding(self) -> None:
        model = FakeLatentPrefixCoLT()
        input_ids = torch.tensor([[1, 2, 0, 0], [0, 0, 3, 4]])
        attention_mask = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]])
        with patch(
            "transformers.models.qwen3_vl.modeling_qwen3_vl.DynamicCache",
            side_effect=lambda config: FakeCache(0),
        ):
            outputs = model._forward_latent_reasoning(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_hidden_generations=0,
            )
        expected = model.embedding(torch.tensor([2, 4])).unsqueeze(1)
        torch.testing.assert_close(outputs.hidden_states[0], expected)

    def test_cached_attention_mask_preserves_prompt_padding(self) -> None:
        prompt_mask = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
        actual = _extend_colt_cached_attention_mask(
            prompt_mask,
            past_seen_tokens=6,
            current_length=2,
            batch_size=1,
            device=prompt_mask.device,
            current_attention_mask=torch.tensor([[1, 0]], dtype=torch.long),
        )
        expected = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 0]], dtype=torch.long)
        torch.testing.assert_close(actual, expected)

    def test_cached_latent_paths_keep_left_padding_masked(self) -> None:
        model = FakeCoLT()
        prompt_ids = torch.tensor([[4, 4, 4, 4]])
        prompt_mask = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
        model.latent_reasoning_generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            num_hidden_generations=3,
            max_new_tokens=2,
            do_sample=False,
            eos_token_id=4,
        )
        self.assertEqual(
            [call["attention_mask"].tolist() for call in model.model.calls],
            [
                [[0, 0, 1, 1, 1, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1, 1, 1, 1]],
            ],
        )

        model.model.calls.clear()
        model._forward_latent_response(
            input_ids=torch.tensor([[4, 4, 4, 4, 3, 2]]),
            response_length=2,
            attention_mask=torch.tensor([[0, 0, 1, 1, 1, 0]], dtype=torch.long),
            num_hidden_generations=3,
        )
        self.assertEqual(model.model.calls[0]["attention_mask"].tolist(), [[0, 0, 1, 1, 1, 1, 1, 1, 0]])

    def test_teacher_forced_logits_match_latent_generation(self) -> None:
        model = FakeCoLT()
        prompt_ids = torch.tensor([[4, 4]])
        generated, rollout_log_probs = model.latent_reasoning_generate(
            input_ids=prompt_ids,
            num_hidden_generations=3,
            max_new_tokens=3,
            do_sample=False,
            eos_token_id=4,
            return_token_log_probs=True,
        )
        response_ids = generated[:, prompt_ids.shape[1] :]
        self.assertEqual(response_ids.tolist(), [[3, 2, 1]])

        scoring_output = model._forward_latent_response(
            input_ids=generated,
            response_length=response_ids.shape[1],
            num_hidden_generations=3,
        )
        self.assertEqual(torch.argmax(scoring_output.logits, dim=-1).tolist(), response_ids.tolist())
        recomputed_log_probs = torch.log_softmax(scoring_output.logits.float(), dim=-1).gather(
            -1, response_ids.unsqueeze(-1)
        ).squeeze(-1)
        torch.testing.assert_close(rollout_log_probs, recomputed_log_probs, rtol=0.0, atol=0.0)

    def test_teacher_forced_response_keeps_gradient_path(self) -> None:
        with torch.enable_grad():
            model = FakeCoLT()
            full_ids = torch.tensor([[4, 4, 3, 2, 1]])
            output = model._forward_latent_response(input_ids=full_ids, response_length=3)
            output.logits.sum().backward()
        self.assertIsNotNone(model.initial_latent.grad)
        self.assertGreater(float(model.initial_latent.grad.abs().sum()), 0.0)
        self.assertIsNotNone(model.embedding.weight.grad)

    def test_shared_prefix_scores_response_group_with_one_latent_prefill(self) -> None:
        with torch.enable_grad():
            model = FakeCoLT()
            full_ids = torch.tensor(
                [
                    [4, 4, 3, 2, 1],
                    [4, 4, 2, 1, 3],
                    [4, 4, 1, 3, 2],
                    [4, 4, 3, 1, 2],
                ]
            )
            output = model._forward_latent_response(
                input_ids=full_ids,
                response_length=3,
                shared_prompt_prefix=True,
            )
            output.logits.sum().backward()
        self.assertEqual(model.latent_batch_sizes, [1])
        self.assertEqual(output.logits.shape[:2], (4, 3))
        self.assertIsNotNone(model.initial_latent.grad)
        self.assertGreater(float(model.initial_latent.grad.abs().sum()), 0.0)
        self.assertIsNotNone(model.embedding.weight.grad)

    def test_hidden_reasoning_reward_accepts_tagged_and_bare_answers(self) -> None:
        reward = load_colt_reward_module()
        self.assertEqual(reward.parse_hidden_reasoning_answer("<answer>A</answer>"), ("A", 1.0))
        self.assertEqual(reward.parse_hidden_reasoning_answer("A"), ("A", 1.0))
        self.assertEqual(reward.parse_hidden_reasoning_answer("<think>x</think><answer>A</answer>"), ("A", 0.0))
        scores = reward.compute_score(
            [
                {
                    "response": "<answer>A</answer>",
                    "ground_truth": "A",
                    "data_type": "image",
                    "problem_type": "multiple choice",
                }
            ]
        )
        self.assertEqual(scores[0], {"overall": 1.5, "format": 1.0, "accuracy": 1.0, "structure_reward": 0.5})

    def test_latent_prompt_does_not_request_visible_thought(self) -> None:
        path = EASYR1_ROOT / "verl/utils/reasoning_prompt.py"
        spec = importlib.util.spec_from_file_location("colt_reasoning_prompt_test_module", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        prompt = module.format_reasoning_prompt("Question?", "Return A or B.", "latent")
        self.assertNotIn("<think>", prompt)
        self.assertIn("Reason internally", prompt)
        self.assertIn("<answer>", prompt)

    def test_runtime_parity_gate_is_present(self) -> None:
        trainer_source = (EASYR1_ROOT / "verl/trainer/ray_trainer.py").read_text(encoding="utf-8")
        self.assertIn('metrics["policy/rollout_logprob_max_abs_diff"]', trainer_source)
        self.assertIn("max_logprob_difference > 5e-2", trainer_source)

    def test_rollout_loads_in_actor_precision(self) -> None:
        rollout_source = (EASYR1_ROOT / "verl/workers/rollout/colt_transformers_rollout.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("torch_dtype=dtype", rollout_source)
        self.assertIn(".to(torch.cuda.current_device(), dtype=dtype)", rollout_source)
        self.assertNotIn("\n            dtype=dtype,", rollout_source)

    def test_overlong_image_filter_has_no_per_record_debug_output(self) -> None:
        dataset_source = (EASYR1_ROOT / "verl/utils/dataset.py").read_text(encoding="utf-8")
        self.assertNotIn('print(images, model_inputs["input_ids"].size(-1))', dataset_source)


if __name__ == "__main__":
    unittest.main()
