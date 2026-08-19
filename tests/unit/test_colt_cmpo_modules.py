from __future__ import annotations

import unittest

import torch

from transformers.models.qwen3_vl.modeling_colt_grounding import (
    bbox_to_token_indices,
    compute_grounding_contrastive_loss,
    compute_step_contrastive_loss,
    pool_roi_and_non_roi_features,
    pool_step_roi_and_nonroi_features,
)
from transformers.models.qwen3_vl.modeling_colt_latent_heads import (
    LatentStochasticHead,
    LowRankExplorationHead,
)


class BboxToTokenIndicesTests(unittest.TestCase):
    def test_xyxy_quarter_image(self) -> None:
        self.assertEqual(
            bbox_to_token_indices([0.0, 0.0, 0.5, 0.5], token_grid_h=4, token_grid_w=4),
            [0, 1, 4, 5],
        )

    def test_xywh(self) -> None:
        self.assertEqual(
            bbox_to_token_indices([0.0, 0.0, 0.5, 0.5], token_grid_h=4, token_grid_w=4, bbox_format="xywh"),
            [0, 1, 4, 5],
        )

    def test_degenerate_bbox_keeps_one_token(self) -> None:
        indices = bbox_to_token_indices([0.5, 0.5, 0.5, 0.5], token_grid_h=4, token_grid_w=4)
        self.assertEqual(len(indices), 1)
        self.assertEqual(indices[0], 2 * 4 + 2)

    def test_clamps_out_of_range_coordinates(self) -> None:
        indices = bbox_to_token_indices([0.512, 0.124, 0.81, 1.616], token_grid_h=4, token_grid_w=4)
        self.assertEqual(len(indices) > 0, True)
        self.assertTrue(max(indices) < 16)

    def test_rejects_invalid_format(self) -> None:
        with self.assertRaises(ValueError):
            bbox_to_token_indices([0.0, 0.0, 0.5, 0.5], token_grid_h=4, token_grid_w=4, bbox_format="bad")


class PoolRoiFeaturesTests(unittest.TestCase):
    def _make_single_image(self, grid_h: int = 8, grid_w: int = 8, hidden: int = 4) -> tuple:
        # merged grid is grid_h//2 x grid_w//2, so 4x4 -> 16 tokens
        tokens = (grid_h // 2) * (grid_w // 2)
        image_embeds = torch.arange(tokens * hidden, dtype=torch.float32).reshape(tokens, hidden)
        grid = torch.tensor([[1, grid_h, grid_w]])
        return image_embeds, grid

    def test_single_image_roi_and_non_roi(self) -> None:
        image_embeds, grid = self._make_single_image()
        roi, non_roi = pool_roi_and_non_roi_features(
            image_embeds, grid, [[[0.0, 0.0, 0.5, 0.5]]]
        )
        self.assertEqual(roi.shape, (1, 4))
        self.assertEqual(non_roi.shape, (1, 4))
        expected_roi = image_embeds[[0, 1, 4, 5]].mean(dim=0)
        self.assertTrue(torch.allclose(roi[0], expected_roi))
        # ROI and non-ROI are disjoint pools.
        self.assertFalse(torch.allclose(roi[0], non_roi[0]))

    def test_multiple_images_one_to_one(self) -> None:
        image_embeds, grid = self._make_single_image()
        # Two identical images, two samples.
        embeds = torch.cat([image_embeds, image_embeds], dim=0)
        grid2 = torch.tensor([[1, 8, 8], [1, 8, 8]])
        roi, non_roi = pool_roi_and_non_roi_features(
            embeds, grid2, [[[0.0, 0.0, 0.5, 0.5]], [[0.0, 0.0, 0.5, 0.5]]]
        )
        self.assertEqual(roi.shape, (2, 4))
        self.assertTrue(torch.allclose(roi[0], roi[1]))

    def test_whole_image_roi_falls_back(self) -> None:
        image_embeds, grid = self._make_single_image()
        roi, non_roi = pool_roi_and_non_roi_features(
            image_embeds, grid, [[[0.0, 0.0, 1.0, 1.0]]]
        )
        whole = image_embeds.mean(dim=0)
        self.assertTrue(torch.allclose(roi[0], whole))
        self.assertTrue(torch.allclose(non_roi[0], whole))

    def test_multiple_bboxes_union(self) -> None:
        image_embeds, grid = self._make_single_image()
        roi, non_roi = pool_roi_and_non_roi_features(
            image_embeds, grid, [[[0.0, 0.0, 0.5, 0.5], [0.5, 0.0, 1.0, 0.5]]]
        )
        expected = image_embeds[[0, 1, 2, 3, 4, 5, 6, 7]].mean(dim=0)
        self.assertTrue(torch.allclose(roi[0], expected))
        self.assertFalse(torch.allclose(roi[0], non_roi[0]))

    def test_requires_image_index_when_batch_mismatches(self) -> None:
        image_embeds, grid = self._make_single_image()
        with self.assertRaises(ValueError):
            pool_roi_and_non_roi_features(
                image_embeds, grid, [[[0.0, 0.0, 0.5, 0.5]], [[0.0, 0.0, 0.5, 0.5]]]
            )


class GroundingLossTests(unittest.TestCase):
    def test_loss_prefers_positive_roi(self) -> None:
        torch.manual_seed(0)
        z_H = torch.randn(4, 8)
        close_pos = z_H + 0.05 * torch.randn(4, 8)
        far_neg = torch.randn(4, 8)
        loss_good = compute_grounding_contrastive_loss(z_H, close_pos, far_neg)

        far_pos = torch.randn(4, 8)
        close_neg = z_H + 0.05 * torch.randn(4, 8)
        loss_bad = compute_grounding_contrastive_loss(z_H, far_pos, close_neg)
        self.assertLess(float(loss_good), float(loss_bad))

    def test_rejects_mismatched_shapes(self) -> None:
        with self.assertRaises(ValueError):
            compute_grounding_contrastive_loss(torch.randn(4, 8), torch.randn(4, 8), torch.randn(3, 8))


class StepGroundingTests(unittest.TestCase):
    def test_pool_step_features(self) -> None:
        # 单图，merged grid 4x4 = 16 tokens
        image_embeds = torch.arange(16 * 4, dtype=torch.float32).reshape(16, 4)
        grid = torch.tensor([[1, 8, 8]])
        step_bboxes = [
            [
                [[0.0, 0.0, 0.5, 0.5]],
                [[0.0, 0.0, 0.5, 0.5], [0.5, 0.0, 1.0, 0.5]],
                [[0.5, 0.0, 1.0, 0.5]],
            ]
        ]
        z_pos, z_neg = pool_step_roi_and_nonroi_features(image_embeds, grid, step_bboxes)
        self.assertEqual(z_pos.shape, (1, 3, 4))
        self.assertEqual(z_neg.shape, (1, 3, 4))
        # step1 只含左上 4 tokens
        expected_v1 = image_embeds[[0, 1, 4, 5]].mean(dim=0)
        self.assertTrue(torch.allclose(z_pos[0, 0], expected_v1))

    def test_step_contrastive_prefers_own_evidence(self) -> None:
        torch.manual_seed(0)
        B, K, D = 4, 3, 8
        z_H = torch.randn(B, K, D)
        z_pos = z_H + 0.05 * torch.randn(B, K, D)  # 正样本接近
        z_neg_same = torch.randn(B, K, D)
        loss_good = compute_step_contrastive_loss(z_H, z_pos, z_neg_same)
        far_pos = torch.randn(B, K, D)
        close_neg = z_H + 0.05 * torch.randn(B, K, D)
        loss_bad = compute_step_contrastive_loss(z_H, far_pos, close_neg)
        self.assertLess(float(loss_good), float(loss_bad))


class LowRankExplorationHeadTests(unittest.TestCase):
    def test_forward_shapes(self) -> None:
        torch.manual_seed(0)
        head = LowRankExplorationHead(hidden_size=32, rank=4)
        h = torch.randn(2, 1, 32)
        action, mean, std, delta = head(h, sample=True)
        self.assertEqual(action.shape, (2, 1, 4))
        self.assertEqual(mean.shape, (2, 1, 4))
        self.assertEqual(std.shape, (2, 1, 4))
        self.assertEqual(delta.shape, (2, 1, 32))

    def test_deterministic_mode(self) -> None:
        torch.manual_seed(0)
        head = LowRankExplorationHead(hidden_size=16, rank=4)
        h = torch.randn(3, 16)
        _, mean, _, _ = head(h, sample=False)
        action, mean2, _, _ = head(h, sample=False)
        self.assertTrue(torch.allclose(action, mean))
        self.assertTrue(torch.allclose(mean, mean2))

    def test_log_prob_shape_and_finite(self) -> None:
        torch.manual_seed(0)
        head = LowRankExplorationHead(hidden_size=16, rank=4)
        h = torch.randn(3, 1, 16)
        action, _, _, _ = head(h, sample=True)
        lp = head.log_prob(h, action)
        self.assertEqual(lp.shape, (3, 1))
        self.assertTrue(torch.isfinite(lp).all())


class LatentStochasticHeadMigrationTests(unittest.TestCase):
    def test_forward_matches_original_contract(self) -> None:
        torch.manual_seed(0)
        head = LatentStochasticHead(hidden_size=16)
        h = torch.randn(2, 1, 16)
        noise, mean, std = head(h, sample=True)
        self.assertEqual(noise.shape, (2, 1, 16))
        self.assertEqual(mean.shape, (2, 1, 16))
        self.assertEqual(std.shape, (2, 1, 16))
        zero_noise, _, _ = head(h, sample=False)
        self.assertTrue(torch.all(zero_noise == 0))


if __name__ == "__main__":
    unittest.main()
