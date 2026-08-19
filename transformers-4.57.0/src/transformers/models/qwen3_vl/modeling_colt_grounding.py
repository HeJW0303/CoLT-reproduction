# coding=utf-8
# Copyright 2026 CoLT-reproduction contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Visual-grounding utilities for CoLT's CMPO stage.

These helpers are deliberately stateless pure functions so that they can be
unit-tested without a model instance. They implement the visual side of the
CMPO grounding loss:

    L_ground = contrastive(z_H, z_V_pos, z_V_neg)

where ``z_H`` is the trajectory-level pooled latent, ``z_V_pos`` is the pooled
visual feature inside the annotated ROI, and ``z_V_neg`` is the pooled visual
feature outside the ROI (a same-image negative).
"""

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def bbox_to_token_indices(
    bbox: Sequence[float],
    token_grid_h: int,
    token_grid_w: int,
    bbox_format: str = "xyxy",
) -> List[int]:
    """Map a normalized bounding box to 1-D merged-visual-token indices.

    ``bbox`` is expected to be normalized to ``[0, 1]``. The returned indices are
    row-major positions inside a ``(token_grid_h, token_grid_w)`` feature map,
    matching the layout produced by Qwen3-VL's spatial merger.
    """
    if token_grid_h <= 0 or token_grid_w <= 0:
        raise ValueError(
            f"token grid dimensions must be positive, got ({token_grid_h}, {token_grid_w})"
        )
    if bbox_format == "xyxy":
        x1, y1, x2, y2 = bbox
    elif bbox_format == "xywh":
        x, y, w, h = bbox
        x1, y1, x2, y2 = x, y, x + w, y + h
    else:
        raise ValueError(f"bbox_format must be 'xyxy' or 'xywh', got {bbox_format!r}")

    # Clamp to the normalized [0, 1] range. LVR-format bboxes occasionally
    # exceed 1.0 (e.g. 1.002 or up to ~1.6) because of annotation slack; the
    # same clamp behavior is used by the LVR reference implementation.
    x1, y1 = max(0.0, float(x1)), max(0.0, float(y1))
    x2, y2 = min(1.0, float(x2)), min(1.0, float(y2))

    token_x1 = int(x1 * token_grid_w)
    token_y1 = int(y1 * token_grid_h)
    token_x2 = min(int(math.ceil(x2 * token_grid_w)), token_grid_w)
    token_y2 = min(int(math.ceil(y2 * token_grid_h)), token_grid_h)

    if token_x2 <= token_x1:
        token_x2 = min(token_x1 + 1, token_grid_w)
    if token_y2 <= token_y1:
        token_y2 = min(token_y1 + 1, token_grid_h)

    indices = [
        y * token_grid_w + x
        for y in range(token_y1, token_y2)
        for x in range(token_x1, token_x2)
    ]
    return indices


def _per_image_token_counts(
    image_grid_thw: torch.Tensor, spatial_merge_size: int
) -> torch.Tensor:
    """Return the number of merged visual tokens for each image."""
    if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
        raise ValueError(
            f"image_grid_thw must have shape (num_images, 3), got {tuple(image_grid_thw.shape)}"
        )
    frames, heights, widths = (
        image_grid_thw[:, 0],
        image_grid_thw[:, 1],
        image_grid_thw[:, 2],
    )
    merged_h = heights // spatial_merge_size
    merged_w = widths // spatial_merge_size
    if torch.any((merged_h <= 0) | (merged_w <= 0)):
        raise ValueError(
            "image_grid_thw contains an image smaller than the spatial merge size"
        )
    return frames * merged_h * merged_w


def _resolve_image_offsets(
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
    image_index: Optional[Sequence[int]],
    batch_size: int,
) -> Tuple[List[int], List[int]]:
    """Return per-image token offsets and per-sample image indices."""
    per_image = _per_image_token_counts(image_grid_thw, spatial_merge_size)
    offsets = torch.cumsum(
        torch.cat([torch.zeros(1, device=per_image.device, dtype=per_image.dtype), per_image]),
        dim=0,
    )[:-1].tolist()
    num_images = per_image.shape[0]

    if image_index is None:
        if num_images != batch_size:
            raise ValueError(
                "image_index is required when the number of images does not equal "
                f"the batch size (images={num_images}, batch={batch_size})"
            )
        sample_indices = list(range(batch_size))
    else:
        if len(image_index) != batch_size:
            raise ValueError(
                f"image_index length {len(image_index)} does not match batch size {batch_size}"
            )
        if any(i < 0 or i >= num_images for i in image_index):
            raise ValueError(f"image_index contains an out-of-range index (num_images={num_images})")
        sample_indices = [int(i) for i in image_index]

    return offsets, sample_indices


def pool_roi_and_non_roi_features(
    image_embeds: torch.Tensor,
    image_grid_thw: torch.Tensor,
    bboxes: Sequence[Sequence[Sequence[float]]],
    spatial_merge_size: int = 2,
    image_index: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pool visual features inside and outside each sample's ROI.

    ``bboxes`` is a per-sample list of normalized ``xyxy`` bboxes:
    ``[[x1,y1,x2,y2], ...]`` for every sample. When a sample has multiple bboxes
    their ROIs are unioned before pooling.

    Returns ``(roi_pool, non_roi_pool)`` each of shape ``(batch_size, hidden)``.
    """
    if image_embeds.ndim != 2:
        raise ValueError(
            f"image_embeds must have shape (tokens, hidden), got {tuple(image_embeds.shape)}"
        )
    batch_size = len(bboxes)
    if batch_size == 0:
        empty = image_embeds.new_zeros((0, image_embeds.shape[-1]))
        return empty, empty

    offsets, sample_indices = _resolve_image_offsets(
        image_grid_thw, spatial_merge_size, image_index, batch_size
    )
    per_image = _per_image_token_counts(image_grid_thw, spatial_merge_size).tolist()

    roi_pool = []
    non_roi_pool = []
    for sample_idx, image_id in enumerate(sample_indices):
        token_grid_h = int(image_grid_thw[image_id, 1].item()) // spatial_merge_size
        token_grid_w = int(image_grid_thw[image_id, 2].item()) // spatial_merge_size
        roi_local = []
        for sample_bbox in bboxes[sample_idx]:
            roi_local.extend(
                bbox_to_token_indices(sample_bbox, token_grid_h, token_grid_w, bbox_format="xyxy")
            )
        roi_local = sorted(set(roi_local))
        if not roi_local:
            raise ValueError(f"bbox {list(bboxes[sample_idx])} produced an empty ROI")

        offset = offsets[image_id]
        roi_global = torch.tensor(roi_local, device=image_embeds.device, dtype=torch.long) + offset
        roi_pool.append(image_embeds.index_select(0, roi_global).mean(dim=0))

        all_local = set(range(int(per_image[image_id])))
        non_roi_local = sorted(all_local - set(roi_local))
        if non_roi_local:
            non_roi_global = (
                torch.tensor(non_roi_local, device=image_embeds.device, dtype=torch.long) + offset
            )
            non_roi_pool.append(image_embeds.index_select(0, non_roi_global).mean(dim=0))
        else:
            all_global = torch.arange(
                int(per_image[image_id]), device=image_embeds.device, dtype=torch.long
            ) + offset
            non_roi_pool.append(image_embeds.index_select(0, all_global).mean(dim=0))

    return torch.stack(roi_pool, dim=0), torch.stack(non_roi_pool, dim=0)


def pool_step_roi_and_nonroi_features(
    image_embeds: torch.Tensor,
    image_grid_thw: torch.Tensor,
    step_bboxes: Sequence[Sequence[Sequence[Sequence[float]]]],
    spatial_merge_size: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pool per-step visual evidence (``z_pos``) and same-image negatives.

    ``step_bboxes`` has shape ``(batch, steps, n_bboxes, 4)`` where each
    ``n_bboxes`` group is the set of ROI boxes for that reasoning step.
    Returns ``(z_pos, z_neg_same)`` each of shape ``(batch, steps, hidden)``.
    """
    batch_size = len(step_bboxes)
    if batch_size == 0:
        return image_embeds.new_zeros((0, 0, image_embeds.shape[-1])), image_embeds.new_zeros(
            (0, 0, image_embeds.shape[-1])
        )
    steps = len(step_bboxes[0])
    hidden = image_embeds.shape[-1]
    z_pos = image_embeds.new_zeros((batch_size, steps, hidden))
    z_neg_same = image_embeds.new_zeros((batch_size, steps, hidden))
    for b in range(batch_size):
        for k in range(steps):
            roi, non_roi = pool_roi_and_non_roi_features(
                image_embeds,
                image_grid_thw,
                [step_bboxes[b][k]],
                spatial_merge_size=spatial_merge_size,
                image_index=[b],
            )
            z_pos[b, k] = roi[0]
            z_neg_same[b, k] = non_roi[0]
    return z_pos, z_neg_same


def compute_step_contrastive_loss(
    z_H: torch.Tensor,
    z_pos: torch.Tensor,
    z_neg_same: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Per-step InfoNCE grounding loss with cross-image and same-image negatives.

    ``z_H``, ``z_pos`` and ``z_neg_same`` all have shape ``(batch, steps, hidden)``.
    For every step ``k`` the query ``h_k`` is matched against its own ROI pool
    ``z_pos[:, k]`` while negatives include the same-image non-evidence pool
    and the other rows' positive pools (cross-image).
    """
    if z_H.shape != z_pos.shape or z_H.shape != z_neg_same.shape:
        raise ValueError(
            "z_H, z_pos and z_neg_same must share the same shape; "
            f"got {tuple(z_H.shape)}, {tuple(z_pos.shape)}, {tuple(z_neg_same.shape)}"
        )
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature!r}")

    batch, steps, _ = z_H.shape
    z_H = F.normalize(z_H, dim=-1)
    z_pos = F.normalize(z_pos, dim=-1)
    z_neg_same = F.normalize(z_neg_same, dim=-1)
    device = z_H.device
    total = z_H.new_zeros(())
    for k in range(steps):
        h = z_H[:, k]  # (B, D)
        pos = z_pos[:, k]  # (B, D)
        same_neg = z_neg_same[:, k]  # (B, D)
        pos_sim = (h * pos).sum(dim=-1) / temperature  # (B,)
        same_sim = (h * same_neg).sum(dim=-1) / temperature  # (B,)
        cross = (h @ pos.t()) / temperature  # (B, B)
        eye = torch.eye(batch, dtype=torch.bool, device=device)
        cross_neg = cross.masked_select(~eye).view(batch, -1)  # (B, B-1)
        logits = torch.cat([pos_sim.unsqueeze(1), same_sim.unsqueeze(1), cross_neg], dim=1)
        labels = torch.zeros(batch, dtype=torch.long, device=device)
        total = total + F.cross_entropy(logits, labels)
    return total / steps


def compute_grounding_contrastive_loss(
    z_H: torch.Tensor,
    z_V_pos: torch.Tensor,
    z_V_neg: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE-style visual grounding loss between latent and ROI evidence."""
    if z_H.shape != z_V_pos.shape or z_H.shape != z_V_neg.shape:
        raise ValueError(
            "z_H, z_V_pos and z_V_neg must share the same shape; "
            f"got {tuple(z_H.shape)}, {tuple(z_V_pos.shape)}, {tuple(z_V_neg.shape)}"
        )
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature!r}")

    z_H = F.normalize(z_H, dim=-1)
    z_V_pos = F.normalize(z_V_pos, dim=-1)
    z_V_neg = F.normalize(z_V_neg, dim=-1)

    pos_sim = (z_H * z_V_pos).sum(dim=-1) / temperature
    neg_sim = (z_H * z_V_neg).sum(dim=-1) / temperature
    logits = torch.stack([pos_sim, neg_sim], dim=1)
    targets = torch.zeros(z_H.shape[0], dtype=torch.long, device=z_H.device)
    return F.cross_entropy(logits, targets)
