from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from transformers.models.qwen3_vl.modeling_colt_latent_heads import (
    LaReLatentRefocusingExtractor,
    pack_visual_tokens_by_sample,
)
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration


class _TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    eos_token = "<eos>"

    def encode(self, text, add_special_tokens=False):
        return [abs(hash(text)) % 30 + 2]

    def decode(self, token_ids, **kwargs):
        return " ".join(str(token_id) for token_id in token_ids)


class PackVisualTokensTests(unittest.TestCase):
    def test_variable_length_and_multi_image_sample_contract(self) -> None:
        flat = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)
        packed, mask = pack_visual_tokens_by_sample(
            flat,
            torch.tensor([2, 0, 4]),
            hidden_size=4,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        self.assertEqual(packed.shape, (3, 4, 4))
        self.assertEqual(mask.tolist(), [[True, True, False, False], [False] * 4, [True] * 4])
        torch.testing.assert_close(packed[0, :2], flat[:2])
        torch.testing.assert_close(packed[2], flat[2:])
        self.assertEqual(float(packed[1].abs().sum()), 0.0)

    def test_text_only_batch_returns_masked_dummy_token(self) -> None:
        packed, mask = pack_visual_tokens_by_sample(
            None,
            torch.tensor([0, 0]),
            hidden_size=8,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        self.assertEqual(packed.shape, (2, 1, 8))
        self.assertFalse(mask.any())

    def test_mismatched_cached_token_count_fails_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match prompt placeholders"):
            pack_visual_tokens_by_sample(
                torch.randn(3, 4),
                torch.tensor([1, 1]),
                hidden_size=4,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )


class LaReLatentRefocusingExtractorTests(unittest.TestCase):
    def _make_extractor(self, *, topk: int = 0, visual_dropout: float = 0.0):
        torch.manual_seed(42)
        extractor = LaReLatentRefocusingExtractor(
            model_hidden_size=16,
            extractor_hidden_size=12,
            num_layers=2,
            num_heads=3 if topk == 0 else 1,
            num_queries=3,
            max_steps=4,
            dropout=0.0,
            visual_dropout=visual_dropout,
            attention_topk=topk,
            reconstruction_steps=20,
        )
        extractor.reset_safe_output(initializer_range=0.02)
        return extractor

    def test_safe_initialization_is_exact_noop(self) -> None:
        extractor = self._make_extractor().eval()
        latent = torch.randn(2, 1, 16)
        visual = torch.randn(2, 5, 16)
        mask = torch.tensor([[True, True, True, False, False], [True] * 5])
        delta, attention, gate, reconstruction = extractor(
            latent,
            visual,
            mask,
            step_index=0,
            compute_reconstruction=False,
        )
        torch.testing.assert_close(delta, torch.zeros_like(delta), atol=0.0, rtol=0.0)
        self.assertEqual(attention.shape, (2, 3, 5))
        self.assertTrue(torch.isfinite(attention).all())
        torch.testing.assert_close(attention.sum(dim=-1), torch.ones(2, 3))
        torch.testing.assert_close(attention[0, :, 3:], torch.zeros(3, 2), atol=0.0, rtol=0.0)
        expected_gate = torch.full_like(gate, torch.sigmoid(torch.tensor(-2.0)))
        torch.testing.assert_close(gate, expected_gate)
        self.assertEqual(float(reconstruction), 0.0)

    def test_text_only_row_has_zero_delta_and_gate_without_nan(self) -> None:
        extractor = self._make_extractor().eval()
        latent = torch.randn(2, 1, 16)
        visual = torch.randn(2, 4, 16)
        mask = torch.tensor([[False] * 4, [True, True, False, False]])
        delta, attention, gate, _ = extractor(latent, visual, mask, step_index=1)
        self.assertTrue(torch.isfinite(attention).all())
        torch.testing.assert_close(delta[0], torch.zeros_like(delta[0]), atol=0.0, rtol=0.0)
        torch.testing.assert_close(gate[0], torch.zeros_like(gate[0]), atol=0.0, rtol=0.0)

    def test_sparse_topk_limits_attention_support(self) -> None:
        extractor = self._make_extractor(topk=2).eval()
        with torch.no_grad():
            torch.nn.init.normal_(extractor.output_projection.weight, std=0.02)
        latent = torch.randn(1, 1, 16)
        visual = torch.randn(1, 7, 16)
        mask = torch.ones(1, 7, dtype=torch.bool)
        _, attention, _, _ = extractor(latent, visual, mask, step_index=2)
        support = (attention > 0).sum(dim=-1)
        self.assertTrue(torch.all(support <= 2))

    def test_refocused_residual_depends_on_visual_tokens_after_warm_start(self) -> None:
        extractor = self._make_extractor().eval()
        with torch.no_grad():
            torch.nn.init.normal_(extractor.output_projection.weight, std=0.02)
        latent = torch.randn(2, 1, 16)
        mask = torch.ones(2, 5, dtype=torch.bool)
        visual_a = torch.randn(2, 5, 16)
        visual_b = torch.randn(2, 5, 16)
        delta_a, _, _, _ = extractor(latent, visual_a, mask, step_index=0)
        delta_b, _, _, _ = extractor(latent, visual_b, mask, step_index=0)
        self.assertFalse(torch.allclose(delta_a, delta_b))

    def test_denoising_loss_is_finite_and_bootstraps_gradients(self) -> None:
        extractor = self._make_extractor().train()
        latent = torch.randn(2, 1, 16)
        visual = torch.randn(2, 6, 16)
        mask = torch.tensor([[True] * 6, [True, True, True, False, False, False]])
        delta, _, _, reconstruction = extractor(
            latent,
            visual,
            mask,
            step_index=0,
            compute_reconstruction=True,
        )
        loss = delta.float().sum() + reconstruction
        loss.backward()
        self.assertTrue(torch.isfinite(reconstruction))
        self.assertIsNotNone(extractor.output_projection.weight.grad)
        self.assertGreater(float(extractor.output_projection.weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(extractor.latent_projection.weight.grad)
        self.assertGreater(float(extractor.latent_projection.weight.grad.abs().sum()), 0.0)
        self.assertIsNone(extractor.reconstruction_target_projection.weight.grad)

    def test_invalid_step_fails_instead_of_reusing_wrong_embedding(self) -> None:
        extractor = self._make_extractor().eval()
        with self.assertRaisesRegex(ValueError, "COLT_LARE_MAX_STEPS"):
            extractor(
                torch.randn(1, 1, 16),
                torch.randn(1, 2, 16),
                torch.ones(1, 2, dtype=torch.bool),
                step_index=4,
            )


class LaReQwenWiringTests(unittest.TestCase):
    def test_env_switch_constructs_self_describing_safe_module_on_cpu(self) -> None:
        config = Qwen3VLConfig(
            text_config={
                "vocab_size": 64,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "max_position_embeddings": 64,
                "rope_scaling": {
                    "mrope_interleaved": True,
                    "mrope_section": [2, 1, 1],
                    "rope_type": "default",
                },
            },
            vision_config={
                "depth": 1,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_heads": 2,
                "out_hidden_size": 16,
                "num_position_embeddings": 16,
                "deepstack_visual_indexes": [0],
            },
            image_token_id=60,
            video_token_id=61,
            vision_start_token_id=62,
            vision_end_token_id=63,
        )
        env = {
            "COLT_RL_MODE": "1",
            "COLT_RL_TOKENIZER_PATH": "tiny-tokenizer",
            "COLT_LARE_REFOCUS": "1",
            "COLT_LARE_DIM": "12",
            "COLT_LARE_LAYERS": "2",
            "COLT_LARE_HEADS": "3",
            "COLT_LARE_QUERIES": "3",
            "COLT_LARE_MAX_STEPS": "4",
            "COLT_LARE_RECON_STEPS": "20",
        }
        with patch.dict("os.environ", env, clear=False), patch(
            "transformers.models.qwen3_vl.modeling_qwen3_vl.AutoTokenizer.from_pretrained",
            return_value=_TinyTokenizer(),
        ):
            model = Qwen3VLForConditionalGeneration(config)

        self.assertTrue(model.lare_refocus_enabled)
        self.assertIsNotNone(model.lare_refocusing_extractor)
        self.assertTrue(model.model._colt_cache_visual_features)
        self.assertTrue(model.config.colt_lare_refocus)
        self.assertEqual(model.config.colt_lare_dim, 12)
        torch.testing.assert_close(
            model.lare_refocusing_extractor.output_projection.weight,
            torch.zeros_like(model.lare_refocusing_extractor.output_projection.weight),
            atol=0.0,
            rtol=0.0,
        )

        # Exercise the actual CoLT wiring without invoking the vision tower:
        # Qwen caches one flattened visual tensor, while prompt placeholders
        # define ownership for variable-length rows.
        model.model._colt_visual_embeds = torch.randn(3, 16)
        prompt_ids = torch.tensor([[60, 60, 2], [60, 2, 2]])
        reference_hidden = torch.randn(2, 1, 16)
        visual_tokens, visual_mask = model._prepare_lare_visual_inputs(prompt_ids, reference_hidden)
        self.assertEqual(visual_tokens.shape, (2, 2, 16))
        self.assertEqual(visual_mask.tolist(), [[True, True], [True, False]])
        base_next_latent = torch.randn(2, 1, 16)
        next_latent, attention, gate, reconstruction = model._apply_lare_refocusing(
            reference_hidden,
            base_next_latent,
            visual_tokens,
            visual_mask,
            step_index=0,
            compute_reconstruction=False,
        )
        torch.testing.assert_close(next_latent, base_next_latent, atol=0.0, rtol=0.0)
        self.assertEqual(attention.shape, (2, 3, 2))
        self.assertEqual(gate.shape, (2, 1, 1))
        self.assertEqual(float(reconstruction), 0.0)

        # A saved config must reconstruct the module even when the enabling
        # environment variable is absent, and loading must not zero trained
        # LaRe weights during missing-key-safe initialization.
        with torch.no_grad():
            model.lare_refocusing_extractor.output_projection.weight.fill_(0.125)
        trained_state = model.state_dict()
        auto_env = {
            "COLT_RL_MODE": "1",
            "COLT_RL_TOKENIZER_PATH": "tiny-tokenizer",
        }
        with patch.dict("os.environ", auto_env, clear=True), patch(
            "transformers.models.qwen3_vl.modeling_qwen3_vl.AutoTokenizer.from_pretrained",
            return_value=_TinyTokenizer(),
        ):
            restored_model = Qwen3VLForConditionalGeneration(config)
            restored_model.load_state_dict(trained_state)
        self.assertTrue(restored_model.lare_refocus_enabled)
        torch.testing.assert_close(
            restored_model.lare_refocusing_extractor.output_projection.weight,
            torch.full_like(restored_model.lare_refocusing_extractor.output_projection.weight, 0.125),
        )

        # A trained checkpoint is self-describing, but an explicit zero must
        # still restore the original architecture for controlled ablations.
        disabled_env = {
            "COLT_RL_MODE": "1",
            "COLT_RL_TOKENIZER_PATH": "tiny-tokenizer",
            "COLT_LARE_REFOCUS": "0",
        }
        with patch.dict("os.environ", disabled_env, clear=False), patch(
            "transformers.models.qwen3_vl.modeling_qwen3_vl.AutoTokenizer.from_pretrained",
            return_value=_TinyTokenizer(),
        ):
            disabled_model = Qwen3VLForConditionalGeneration(config)
        self.assertFalse(disabled_model.lare_refocus_enabled)
        self.assertIsNone(disabled_model.lare_refocusing_extractor)
        self.assertFalse(disabled_model.model._colt_cache_visual_features)


if __name__ == "__main__":
    unittest.main()
