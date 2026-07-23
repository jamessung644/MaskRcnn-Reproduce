"""RoIAlign + FPN 레벨 할당 (Mask R-CNN 논문 3절, FPN 논문 4.2절).

RoIAlign (Mask R-CNN 논문 3절 'RoIAlign'):
    RoIPool의 양자화를 제거하고, 각 bin 안의 규칙적인 샘플링 포인트에서
    bilinear interpolation으로 값을 계산해 평균한다. roi_align은 직접 구현한
    maskrcnn.ops.roi_align(aligned=True)을 쓴다 — torchvision.ops와 수치적으로
    일치하도록 검증한, 논문의 half-pixel 정렬 연산이다.

레벨 할당 (FPN 논문 식 (1)):
    k = floor(k0 + log2(sqrt(w*h) / 224)),  k0 = 4
    224^2 크기의 RoI는 P4에, 그보다 작으면 더 fine한 레벨에 할당된다.
    box/mask head는 P2~P5만 사용한다 (P6는 RPN 전용).
"""

import math
from typing import Dict, List

import torch
from torch import Tensor, nn
from ..ops import roi_align


class MultiScaleRoIAlign(nn.Module):
    def __init__(self, output_size: int, sampling_ratio: int,
                 canonical_scale: int = 224, canonical_level: int = 4,
                 min_level: int = 2, max_level: int = 5):
        super().__init__()
        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.canonical_scale = canonical_scale
        self.canonical_level = canonical_level
        self.min_level = min_level
        self.max_level = max_level

    def _assign_levels(self, boxes: Tensor) -> Tensor:
        """FPN 논문 식 (1)로 각 RoI의 피라미드 레벨을 정한다."""
        ws = boxes[:, 2] - boxes[:, 0]
        hs = boxes[:, 3] - boxes[:, 1]
        scale = torch.sqrt(ws * hs)
        levels = torch.floor(
            self.canonical_level + torch.log2(scale / self.canonical_scale + 1e-8)
        )
        return levels.clamp(min=self.min_level, max=self.max_level).to(torch.int64)

    def forward(self, features: Dict[str, Tensor],
                boxes_per_image: List[Tensor]) -> Tensor:
        """이미지별 박스 리스트를 받아 (sum(N_i), C, S, S) 피처를 반환한다.

        반환 순서는 입력 박스를 이미지 순서대로 이어붙인 순서와 같다.
        """
        # (batch_idx, x1, y1, x2, y2) 형식으로 병합
        rois = torch.cat([
            torch.cat([torch.full((b.shape[0], 1), i, dtype=b.dtype, device=b.device), b],
                      dim=1)
            for i, b in enumerate(boxes_per_image)
        ])

        levels = self._assign_levels(rois[:, 1:])

        num_channels = features["p2"].shape[1]
        output = torch.zeros(
            rois.shape[0], num_channels, self.output_size, self.output_size,
            dtype=features["p2"].dtype, device=rois.device,
        )

        for level in range(self.min_level, self.max_level + 1):
            idx = torch.where(levels == level)[0]
            if idx.numel() == 0:
                continue
            feat = features[f"p{level}"]
            spatial_scale = 1.0 / (2 ** level)
            output[idx] = roi_align(
                feat, rois[idx],
                output_size=self.output_size,
                spatial_scale=spatial_scale,
                sampling_ratio=self.sampling_ratio,
                aligned=True,
            )
        return output
