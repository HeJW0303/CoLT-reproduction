from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "LLaMA-Factory" / "src"))

from llamafactory.data.processor.supervised import SupervisedDatasetProcessor

from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    _compute_colt_backward_alignment_loss,
    _mask_colt_visual_cot_prompt_rows,
    _resolve_colt_loss_row_mask,
    _resolve_colt_visual_cot_rows,
)


class CoLTVisualControlTests(unittest.TestCase):
    def test_processor_preserves_visual_controls_and_masks_labels(self) -> None:
        processor = SupervisedDatasetProcessor.__new__(SupervisedDatasetProcessor)
        processor._encode_data_example = lambda **kwargs: ([1, 2, 3], [-100, 8, 9])
        examples = {
            "_prompt": [
                [{"role": "user", "content": "q"}],
                [{"role": "user", "content": "q"}],
            ],
            "_response": [
                [{"role": "assistant", "content": "a"}],
                [{"role": "assistant", "content": "a"}],
            ],
            "_system": ["", ""],
            "_tools": ["", ""],
            "_images": [None, None],
            "_videos": [None, None],
            "_audios": [None, None],
            "_step_bboxes": [[], [[[0.0, 0.0, 1.0, 1.0]]]],
            "_visual_only": [True, False],
            "_visual_cot": [False, True],
        }
        output = processor.preprocess_dataset(examples)
        self.assertEqual(output["labels"][0], [-100, -100, -100])
        self.assertEqual(output["labels"][1], [-100, 8, 9])
        self.assertEqual(output["visual_only"], [True, False])
        self.assertEqual(output["visual_cot"], [False, True])

    def test_all_ignore_labels_disable_custom_losses(self) -> None:
        labels = torch.tensor([[-100, -100, -100], [-100, 7, -100]])
        actual = _resolve_colt_loss_row_mask(
            labels,
            batch_size=2,
            device=labels.device,
        )
        torch.testing.assert_close(actual, torch.tensor([False, True]))

    def test_visual_only_metadata_overrides_nonempty_labels(self) -> None:
        labels = torch.tensor([[7, 8], [9, -100]])
        actual = _resolve_colt_loss_row_mask(
            labels,
            batch_size=2,
            device=labels.device,
            visual_only=torch.tensor([True, False]),
        )
        torch.testing.assert_close(actual, torch.tensor([False, True]))

    def test_explicit_visual_cot_mask_controls_rows(self) -> None:
        actual = _resolve_colt_visual_cot_rows(
            torch.tensor([False, True]),
            batch_size=2,
            device=torch.device("cpu"),
            bboxes=[[[0.0, 0.0, 1.0, 1.0]], []],
        )
        torch.testing.assert_close(actual, torch.tensor([False, True]))

    def test_visual_cot_fallback_uses_nonempty_bbox_rows(self) -> None:
        actual = _resolve_colt_visual_cot_rows(
            None,
            batch_size=2,
            device=torch.device("cpu"),
            bboxes=[[[0.0, 0.0, 1.0, 1.0]], []],
        )
        torch.testing.assert_close(actual, torch.tensor([True, False]))

    def test_mask_hides_question_kv_but_keeps_latent_and_answer_visible(self) -> None:
        attention_mask = torch.ones((2, 12), dtype=torch.long)
        actual = _mask_colt_visual_cot_prompt_rows(
            attention_mask,
            current_seq_len=7,
            rows_to_mask=torch.tensor([True, False]),
        )
        expected = torch.ones((2, 12), dtype=torch.long)
        expected[0, :7] = 0
        torch.testing.assert_close(actual, expected)

    def test_backward_alignment_ignores_visual_only_rows(self) -> None:
        cot_hidden = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
            ]
        )
        latent = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
        actual = _compute_colt_backward_alignment_loss(
            cot_hidden,
            latent,
            torch.nn.Identity(),
            probe_positions=torch.tensor([1, 1]),
            active_mask=torch.tensor([True, False]),
        )
        self.assertEqual(float(actual), 0.0)


if __name__ == "__main__":
    unittest.main()
