from __future__ import annotations

from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest

import torch

from scripts.lkl_8gpu.lare.build_cot_attention_targets import (
    load_full_coverage_causal_audit,
    parse_teacher_heads,
    teacher_attention_metadata,
)
from scripts.lkl_8gpu.lare.validate_cot_attention_occlusion import (
    choose_occlusion_cells,
    mask_pixel_rows_with_mean,
    merged_cells_to_pixel_rows,
    pass_causal_gate,
)
from scripts.lkl_8gpu.lare.visualize_cot_attention_maps import load_translations
from llamafactory.data.collator import _has_nonempty_cot_attention_target


def _fake_teacher(num_layers: int = 36, num_heads: int = 32, head_dim: int = 8):
    layers = []
    for _ in range(num_layers):
        attention = SimpleNamespace(
            q_proj=SimpleNamespace(out_features=num_heads * head_dim),
            head_dim=head_dim,
        )
        layers.append(SimpleNamespace(self_attn=attention))
    language_model = SimpleNamespace(layers=layers)
    return SimpleNamespace(model=SimpleNamespace(language_model=language_model))


class LaReAttentionToolingTests(unittest.TestCase):
    def test_all_abstain_teacher_batch_is_safe(self) -> None:
        self.assertFalse(_has_nonempty_cot_attention_target([None, [[], [], []], []]))
        self.assertTrue(_has_nonempty_cot_attention_target([None, [[], [0.2, 0.8], []]]))

    def test_sparse_teacher_head_parser_preserves_order_and_validates_bounds(self) -> None:
        teacher = _fake_teacher()
        self.assertEqual(
            parse_teacher_heads("23:4,21:10,26:20", teacher, fallback_layer=18),
            [(23, 4), (21, 10), (26, 20)],
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            parse_teacher_heads("23:4,23:4", teacher, fallback_layer=18)
        with self.assertRaisesRegex(ValueError, "outside"):
            parse_teacher_heads("36:0", teacher, fallback_layer=18)
        with self.assertRaisesRegex(ValueError, "outside"):
            parse_teacher_heads("23:32", teacher, fallback_layer=18)

    def test_missing_sparse_head_spec_keeps_legacy_all_head_layer(self) -> None:
        heads = parse_teacher_heads(None, _fake_teacher(), fallback_layer=18)
        self.assertEqual(heads, [(18, head_index) for head_index in range(32)])

    def test_sparse_teacher_metadata_does_not_claim_l18_all_heads(self) -> None:
        metadata = teacher_attention_metadata(
            18,
            [(23, 4), (21, 10)],
            explicit_heads=True,
        )
        self.assertEqual(metadata["teacher_attention_mode"], "explicit_sparse_layer_head")
        self.assertIsNone(metadata["teacher_layer"])
        self.assertEqual(metadata["teacher_layer_fallback"], 18)
        self.assertEqual(metadata["teacher_head_pairs"], [[23, 4], [21, 10]])

    def test_bilingual_translation_sidecar_has_twelve_complete_rows(self) -> None:
        translations = load_translations(
            Path("scripts/lkl_8gpu/lare/translations/cot_attention_heldout12_zh.json")
        )
        self.assertEqual(len(translations), 12)
        self.assertTrue(all(entry["question_zh"] for entry in translations.values()))
        self.assertTrue(all(len(entry["cot_steps_zh"]) == 3 for entry in translations.values()))

    def test_occlusion_controls_have_equal_native_visual_cell_sizes(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(7)
        top, bottom, random_cells = choose_occlusion_cells(
            [0.01, 0.7, 0.02, 0.03, 0.04, 0.2], fraction=0.25, generator=generator
        )
        self.assertEqual(top.tolist(), [1, 5])
        self.assertEqual(set(bottom.tolist()), {0, 2})
        self.assertEqual(top.numel(), bottom.numel())
        self.assertEqual(top.numel(), random_cells.numel())

    def test_occlusion_masks_exact_merged_cells_and_requires_control_margin(self) -> None:
        # Qwen's merge size is two, so cell 1 owns contiguous raw rows 4..7.
        rows = merged_cells_to_pixel_rows(torch.tensor([1]), raw_pixel_rows=12, merge_size=2)
        self.assertEqual(rows.tolist(), [4, 5, 6, 7])
        pixels = torch.arange(24, dtype=torch.float32).reshape(12, 2)
        masked = mask_pixel_rows_with_mean(pixels, rows)
        self.assertTrue(torch.equal(masked[:4], pixels[:4]))
        self.assertTrue(torch.equal(masked[8:], pixels[8:]))
        self.assertTrue(torch.equal(masked[4], pixels.mean(dim=0)))
        passed, margin = pass_causal_gate(
            top_nll_delta=0.08,
            bottom_nll_delta=0.02,
            random_nll_deltas=[0.03],
            min_nll_increase=0.01,
            min_control_margin=0.02,
        )
        self.assertTrue(passed)
        self.assertAlmostEqual(margin, 0.05)
        self.assertFalse(
            pass_causal_gate(
                top_nll_delta=0.04,
                bottom_nll_delta=0.03,
                random_nll_deltas=[0.035],
                min_nll_increase=0.01,
                min_control_margin=0.02,
            )[0]
        )

    def test_full_coverage_audit_refuses_partial_or_mismatched_filter(self) -> None:
        # Special methods are looked up on the class, so use a tiny concrete
        # holder rather than relying on a dynamic SimpleNamespace __len__.
        class _Tokenized:
            _fingerprint = "frozen-tokenized"

            def __len__(self):
                return 2

        report = {
            "format": "colt_cot_attention_causal_occlusion_v1",
            "tokenized_fingerprint": "frozen-tokenized",
            "source_rows": 2,
            "teacher_model_path": "/model",
            "teacher_layer": 18,
            "teacher_heads": None,
            "query_pool": "mean",
            "num_steps": 3,
            "min_step_tokens": 8,
            "image_max_pixels": 802816,
            "image_min_pixels": 1024,
            "rows": [
                {"row_index": 0, "steps": [{"passed": True}, {"passed": False}, {"passed": True}]},
                {"row_index": 1, "steps": [{"passed": False}, {"passed": True}, {"passed": False}]},
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audit.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(
                load_full_coverage_causal_audit(
                    path,
                    tokenized=_Tokenized(),
                    teacher_model_path=Path("/model"),
                    teacher_layer=18,
                    layer_heads=None,
                    query_pool="mean",
                    num_steps=3,
                    min_step_tokens=8,
                    image_max_pixels=802816,
                    image_min_pixels=1024,
                ),
                {0: [True, False, True], 1: [False, True, False]},
            )
            report["rows"] = report["rows"][:1]
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cover every row"):
                load_full_coverage_causal_audit(
                    path,
                    tokenized=_Tokenized(),
                    teacher_model_path=Path("/model"),
                    teacher_layer=18,
                    layer_heads=None,
                    query_pool="mean",
                    num_steps=3,
                    min_step_tokens=8,
                    image_max_pixels=802816,
                    image_min_pixels=1024,
                )


if __name__ == "__main__":
    unittest.main()
